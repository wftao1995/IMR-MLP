"""Coordinate and gradient helpers for independent VCNEB saddle checks."""
import json
from pathlib import Path
import time

PLAN = {"tolerances": {"chart_cell_A": 1e-10, "direction_gradient_absolute": 3e-7, "direction_gradient_relative": 2e-6}, "gradient_differences": {"seed": 20261015, "steps": [1e-4, 1e-5, 1e-6]}}


def parse_summary(path):
    meta, scales, energies, rows = {}, {}, {}, []
    for line in Path(path).read_text().splitlines():
        bits = line.split()
        if not bits:
            continue
        if bits[0] == "#":
            if bits[1] == "cell_scale":
                scales[int(bits[2])] = float(bits[3])
            elif bits[1] == "image_energy":
                energies[int(bits[2])] = list(map(float, bits[3:]))
            elif len(bits) == 3:
                meta[bits[1]] = bits[2]
        else:
            rows.append(list(map(float, bits)))
    return meta, scales, energies, rows


def cholesky_and_derivatives(c):
    """Differentiate the scalar lower-triangular recurrence, not GPML pullback."""
    import numpy as np
    C = np.array([[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]])
    dC = np.zeros((6, 3, 3))
    for k, (a, b) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        dC[k, a, b] = dC[k, b, a] = 1
    L, dL = np.zeros((3, 3)), np.zeros((6, 3, 3))
    for i in range(3):
        for j in range(i + 1):
            value = C[i, j] - np.dot(L[i, :j], L[j, :j])
            deriv = dC[:, i, j] - (dL[:, i, :j] * L[j, :j] + L[i, :j] * dL[:, j, :j]).sum(axis=1)
            if i == j:
                if value <= 0:
                    raise ValueError("Cell metric left the positive-definite chart")
                L[i, j] = np.sqrt(value)
                dL[:, i, j] = deriv / (2 * L[i, j])
            else:
                L[i, j] = value / L[j, j]
                dL[:, i, j] = (deriv - L[i, j] * dL[:, j, j]) / L[j, j]
    return L, dL


class Oracle:
    def encode(self, atoms):
        import numpy as np
        A = atoms.cell.array
        D = self.Rinv @ A
        C = D @ D.T
        c = C[np.triu_indices(3)]
        L, _ = cholesky_and_derivatives(c)
        if np.max(np.abs(self.R @ L - A)) > PLAN["tolerances"]["chart_cell_A"]:
            raise ValueError("Saved cell is outside the supported lower-Cholesky gauge")
        fractional = atoms.get_scaled_positions(wrap=False)
        atom_q = fractional @ self.R if self.recast else fractional
        return np.r_[c * self.scales, atom_q.ravel()]

    def decode(self, y, template):
        import numpy as np
        c = y[:6] / self.scales
        L, _ = cholesky_and_derivatives(c)
        A = self.R @ L
        q = y[6:].reshape(-1, 3)
        frac = q @ self.Rinv if self.recast else q
        atoms = template.copy()
        atoms.set_cell(A, scale_atoms=False)
        atoms.set_positions(frac @ A)
        return atoms

    def evaluate(self, atoms, gradient=True):
        import numpy as np
        start = time.perf_counter()
        sample = atoms.copy()
        sample.calc = self.calc
        energy = float(sample.get_potential_energy())
        forces = sample.get_forces()
        stress = sample.get_stress(voigt=False)
        volume = float(sample.get_volume())
        H = energy + self.p * volume
        A = sample.cell.array
        output = {"energy": energy, "forces": forces, "stress": stress,
                  "virial": -volume * stress, "volume": volume, "enthalpy": H}
        if gradient:
            y = self.encode(sample)
            _, dL = cholesky_and_derivatives(y[:6] / self.scales)
            cell_gradient = np.array([volume * np.sum((stress + self.p * np.eye(3)) *
                                                     np.linalg.solve(A, self.R @ d)) for d in dL])
            atom_gradient = -forces @ ((self.Rinv @ A).T if self.recast else A.T)
            output["gradient"] = np.r_[cell_gradient / self.scales, atom_gradient.ravel()]
        self.times.append(time.perf_counter() - start)
        self.calls += 1
        if self.calls % 25 == 0:
            print(json.dumps({"oracle_calls": self.calls, "last_seconds": self.times[-1]}), flush=True)
        return output


def directional_checks(oracle, atoms, atomic_only=False):
    import numpy as np
    base = oracle.evaluate(atoms)
    y = oracle.encode(atoms)
    rng = np.random.default_rng(PLAN["gradient_differences"]["seed"])
    directions = []
    for j in ([] if atomic_only else range(6)):
        d = np.zeros(len(y)); d[j] = 1
        directions.append((f"cell_{j}", d))
    for j in range(3):
        d = np.zeros(len(y)); d[6 + j] = 1
        directions.append((f"atom_0_{j}", d))
    for j in range(2):
        d = rng.normal(size=len(y)); d /= np.linalg.norm(d)
        directions.append((f"mixed_{j}", d))
    rows = []
    for name, direction in directions:
        analytic = float(base["gradient"] @ direction)
        scan = []
        for h in PLAN["gradient_differences"]["steps"]:
            hp = oracle.evaluate(oracle.decode(y + h * direction, atoms), gradient=False)["enthalpy"]
            hm = oracle.evaluate(oracle.decode(y - h * direction, atoms), gradient=False)["enthalpy"]
            numeric = (hp - hm) / (2 * h)
            roundoff = 32 * np.finfo(float).eps * max(1, abs(hp), abs(hm)) / h
            scan.append({"step": h, "analytic": analytic, "finite_difference": numeric,
                         "enthalpy_plus": hp, "enthalpy_minus": hm,
                         "absolute_error": abs(analytic - numeric), "energy_roundoff_budget": roundoff})
        for j in (1, 2):
            truncation = abs(scan[j - 1]["finite_difference"] - scan[j]["finite_difference"]) / 99
            budget = (PLAN["tolerances"]["direction_gradient_absolute"] +
                      PLAN["tolerances"]["direction_gradient_relative"] * abs(analytic) +
                      scan[j]["energy_roundoff_budget"] + truncation)
            scan[j].update(truncation_budget=truncation, total_budget=budget,
                           passed=bool(scan[j]["absolute_error"] <= budget))
        rows.append({"direction": name, "pass": all(s["passed"] for s in scan[1:]), "scan": scan})
    return {"coordinate_mode": "recast" if oracle.recast else "fractional",
            "pass": all(row["pass"] for row in rows), "directions": rows}
