#!/usr/bin/env python3
"""Compare native LAMMPS sampling with the offline MaxVol command, without NumPy."""

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def frames(path):
    lines = path.read_text().splitlines()
    result = []
    i = 0
    while i < len(lines):
        count = int(lines[i])
        result.append((lines[i + 1], lines[i + 2:i + 2 + count]))
        i += count + 2
    return result


def field(header, key):
    return float(re.search(rf"\b{key}=([^\s]+)", header)[1])


def close(a, b, tolerance=2e-10):
    assert math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lmp", "plugin", "imrmlp", "model"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--imrmlp-libdir", required=True, type=Path)
    parser.add_argument("--work", type=Path, help="Retain regression inputs and outputs in this directory")
    args = parser.parse_args()
    for name in ("lmp", "plugin", "imrmlp", "model"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
        setattr(args, name, path)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMPI_MCA_rmaps_base_oversubscribe="1")
    env["LD_LIBRARY_PATH"] = os.pathsep.join([
        str(args.imrmlp_libdir.resolve()), str(args.imrmlp.parent), str(args.lmp.parent),
        env.get("LD_LIBRARY_PATH", "")])

    directory_context = (nullcontext(args.work.resolve()) if args.work else
                         tempfile.TemporaryDirectory(prefix="maxvol-lammps-"))
    with directory_context as directory:
        work = Path(directory)
        work.mkdir(parents=True, exist_ok=True)

        def run(command, expected_error=None):
            try:
                result = subprocess.run(command, cwd=work, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            except subprocess.TimeoutExpired as exception:
                raise RuntimeError(f"Timed out: {command}\n{exception.stdout!r}") from exception
            if expected_error:
                assert result.returncode != 0 and expected_error in result.stdout, result.stdout
            elif result.returncode:
                raise RuntimeError(f"{' '.join(map(str, command))}\n{result.stdout}")
            return result.stdout

        shutil.copyfile(args.model, work / "model.mtp")
        (work / "index.map").write_text("Al\nNi\n")
        # Invert the LAMMPS type map and use sparse IDs, nonzero box origin,
        # and only two atoms so four-rank runs exercise ranks with no atoms.
        for tilted in (False, True):
            tilt = "2 1 0.5 xy xz yz\n" if tilted else ""
            (work / f"atoms.{int(tilted)}").write_text(
                "MaxVol regression\n\n2 atoms\n2 atom types\n\n"
                "1 13 xlo xhi\n2 14 ylo yhi\n-3 9 zlo zhi\n" + tilt +
                "\nAtoms # atomic\n\n11 1 3 4 -1\n97 2 5.4 4.6 -0.5\n")
        (work / "pool.xyz").write_text("".join(
            '2\nLattice="12 0 0 0 12 0 0 0 12" pbc="T T T" '
            'Properties=species:S:1:pos:R:3\n'
            f'Ni 2 2 2\nAl {2 + distance} 2.6 2.5\n'
            for distance in (2.1, 2.4, 2.8, 3.1)))

        def lammps(ranks, name, *, tilted=False, periodic="p p p", options="",
                   body="run 4\nrun 0\nrun 2\n", expected_error=None, prefix="", mapping="1 0"):
            path = work / f"in.{name}"
            path.write_text(
                f'plugin load "{args.plugin}"\n'
                f'units metal\natom_style atomic\nboundary {periodic}\n'
                f'read_data atoms.{int(tilted)}\nmass * 1\n'
                'pair_style zero 6.0\npair_coeff * *\n'
                'velocity all set 0.02 0.01 0.0\nfix integrate all nve\n' + prefix +
                f'fix mv all mlip/maxvol 2 model.mtp model.mtp.sel index.map {mapping} '
                f'save {name}.xyz threshold-save 0 threshold-break none {options}\n'
                'thermo 1\nthermo_style custom step f_mv f_mv[1] f_mv[2]\n'
                'thermo_modify format float %.17g\n' + body)
            stdout = run([args.mpiexec, "-n", str(ranks), str(args.lmp),
                          "-log", "none", "-in", str(path)], expected_error)
            if expected_error:
                return stdout
            rows = []
            for line in stdout.splitlines():
                values = line.split()
                if len(values) == 4:
                    try:
                        rows.append([float(value) for value in values])
                    except ValueError:
                        pass
            assert rows and all(math.isfinite(row[1]) for row in rows), stdout
            return rows, frames(work / f"{name}.xyz")

        for dtype, mode, scaling, backend in (
                ("mtm", "nbh", "uniform", "full"),
                ("mtm", "cfg", "mlip", "online"),
                ("pot_efs", "nbh", "mlip", "stream"),
                ("pot_efs", "cfg", "uniform", "full")):
            settings = (work / "select.ini")
            settings.write_text(
                f"builder:dtype {dtype}\nbuilder:mode {mode}\nbuilder:row-scaling {scaling}\n"
                "builder:potfile model.mtp\nbuilder:w-energy 1\nbuilder:w-site 0.2\n"
                "builder:w-force 0.01\nbuilder:w-stress 0.001\n"
                f"selector:select-mode {backend}\nselector:init-mode data\n"
                "selector:weight 1.7\nselector:pow-scale 0.3\n")
            run([str(args.imrmlp), "maxvol_select", "--settings", "select.ini",
                 "--pool", "pool.xyz", "--index", "index.map"])
            state = (work / "model.mtp.sel").read_bytes()
            for tilted, periodic in ((False, "p p p"), (True, "p p p"),
                                     (True, "f f f"), (False, "p p f")):
                reference = None
                for ranks in (1, 2, 4):
                    name = f"{dtype}-{mode}-{int(tilted)}-{periodic.replace(' ', '')}-{ranks}"
                    rows, samples = lammps(ranks, name, tilted=tilted, periodic=periodic)
                    assert [field(h, "timestep") for h, _ in samples] == [2, 4, 6]
                    assert rows[-1][2:] == [3, 6], rows
                    assert all("energy=" not in h and "forces:" not in h for h, _ in samples)
                    assert all([r.split()[0] for r in atoms] == ["Ni", "Al"] for _, atoms in samples)
                    # Non-sampling thermo reports the previous grade/step without re-evaluation.
                    assert next(row for row in rows if row[0] == 3)[3] == 2
                    assert (work / "model.mtp.sel").read_bytes() == state
                    if reference is None:
                        reference = samples
                    else:
                        for (h1, a1), (h2, a2) in zip(reference, samples):
                            close(field(h1, "MV_grade"), field(h2, "MV_grade"))
                            assert a1 == a2
                    run([str(args.imrmlp), "maxvol_sample", "--settings", "select.ini",
                         "--state-file", "model.mtp.sel", "--input", name + ".xyz",
                         "--index", "index.map", "--output", "offline.xyz",
                         "--threshold-save", "-1", "--threshold-break", "1e100"])
                    offline = frames(work / "offline.xyz")
                    assert len(offline) == len(samples)
                    for (h1, _), (h2, _) in zip(samples, offline):
                        # The CLI intentionally formats MV_grade with six decimals.
                        close(field(h1, "MV_grade"), field(h2, "MV_grade"), 6e-7)
            print(f"PASS native/offline and MPI parity: {dtype}/{mode}/{backend}", flush=True)

        # Saving uses strict >; repeated jobs append, and setup/run 0 does not save.
        rows, samples = lammps(2, "quiet", options="threshold-save 1e100")
        assert not samples and rows[-1][2] == 0
        _, first = lammps(2, "append", body="run 2\n")
        _, second = lammps(2, "append", body="run 2\n")
        assert len(first) == 1 and len(second) == 2
        # Same timestep with changed geometry must refresh the grade.
        rows, samples = lammps(2, "refresh", body=(
            "run 0\ndisplace_atoms all random 0.2 0.2 0.2 12345 units box\nrun 0\n"))
        assert not samples and not math.isclose(rows[0][1], rows[-1][1], rel_tol=1e-8)
        # The triggering frame is flushed before a collective failure, even at setup.
        lammps(4, "stop", options="threshold-break 0", expected_error="threshold-break exceeded")
        assert len(frames(work / "stop.xyz")) == 1
        assert field(frames(work / "stop.xyz")[0][0], "timestep") == 0
        lammps(4, "bad-map", mapping="2 0", expected_error="type mapping is outside")
        lammps(4, "bad-output", options="save missing/frames.xyz", expected_error="cannot open sample output")
        lammps(2, "bad-threshold", options="threshold-save -1", expected_error="0 <= threshold-save")
        lammps(2, "bad-keyword", options="unknown 1", expected_error="Unknown fix mlip/maxvol keyword")
        (work / "model.mtp.sel").write_bytes(b"truncated state")
        lammps(4, "bad-state", expected_error="Cannot load MaxVol state")
        print("PASS cadence, append, refresh, thresholds and collective error handling", flush=True)


if __name__ == "__main__":
    main()
