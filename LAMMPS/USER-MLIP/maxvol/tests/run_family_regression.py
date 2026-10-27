#!/usr/bin/env python3
"""Check every native MTP-family loader against offline E/F/S MaxVol grades."""

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import tempfile

from run_regression import close, field, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lmp", "plugin", "imrmlp", "generator", "imrmlp-libdir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    for name in ("lmp", "plugin", "imrmlp", "generator", "imrmlp_libdir"):
        setattr(args, name, getattr(args, name).resolve())
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMPI_MCA_rmaps_base_oversubscribe="1")
    env["LD_LIBRARY_PATH"] = os.pathsep.join([
        str(args.imrmlp_libdir), str(args.imrmlp.parent), env.get("LD_LIBRARY_PATH", "")])
    directory_context = (nullcontext(args.work.resolve()) if args.work else
                         tempfile.TemporaryDirectory(prefix="maxvol-mtp-family-"))
    with directory_context as directory:
        work = Path(directory)
        work.mkdir(parents=True, exist_ok=True)

        def run(command, error=None):
            result = subprocess.run(command, cwd=work, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            if error:
                assert result.returncode != 0 and error in result.stdout, result.stdout
            elif result.returncode:
                raise RuntimeError(f"{command}\n{result.stdout}")
            return result.stdout

        run([str(args.generator), str(work)])
        (work / "index.map").write_text("Al\nNi\n")
        (work / "atoms.data").write_text(
            "MTP family regression\n\n3 atoms\n2 atom types\n\n"
            "1 13 xlo xhi\n2 14 ylo yhi\n-3 9 zlo zhi\n"
            "\nAtoms # charge\n\n11 1 0.1 3 4 -1\n97 2 0.15 5.4 4.6 -0.5\n"
            "201 1 0.05 8.1 4.8 -0.1\n")
        (work / "pool.xyz").write_text("".join(
            '3\nLattice="12 0 0 0 12 0 0 0 12" pbc="T T T" '
            'Properties=species:S:1:pos:R:3\n'
            f'Ni 2 2 2\nAl {2 + distance} 2.6 2.5\nNi 7.1 2.8 2.9\n'
            for distance in (2.1, 2.4, 2.8, 3.1)))

        def lammps(case, ranks, suffix="", options="", error=None, state=None, settings=None,
                   body="run 2\n"):
            name, _, model, config = case
            name = f"{name}-{ranks}{suffix}"
            output = work / f"{name}.xyz"
            output.unlink(missing_ok=True)
            deck = work / f"in.{name}"
            deck.write_text(
                f'plugin load "{args.plugin}"\nunits metal\natom_style charge\n'
                'read_data atoms.data\nmass * 1\npair_style zero 6\npair_coeff * *\n'
                'fix integrate all nve\n'
                f'fix mv all mlip/maxvol 1 "{model}" "{state or model + ".sel"}" '
                f'index.map 1 0 settings "{settings or config}" save "{output}" '
                f'threshold-save 0 threshold-break none {options}\n'
                'thermo_style custom step f_mv\n' + body)
            run([args.mpiexec, "-n", str(ranks), str(args.lmp), "-log", "none", "-in", str(deck)], error)
            return output

        def offline(case, native):
            _, _, model, config = case
            run([str(args.imrmlp), "maxvol_sample", "--settings", config,
                 "--state-file", model + ".sel", "--input", str(native),
                 "--index", "index.map", "--output", "offline.xyz",
                 "--threshold-save", "-1", "--threshold-break", "1e100"])
            actual, expected = frames(native), frames(work / "offline.xyz")
            assert len(actual) == len(expected) and actual
            for (a, _), (b, _) in zip(actual, expected):
                close(field(a, "MV_grade"), field(b, "MV_grade"), 6e-7)
                assert "energy=" not in a and "forces:" not in a and "charge:R" not in a
            return actual

        cases = [tuple(line.split("\t")) for line in (work / "models.tsv").read_text().splitlines()]
        for case in cases:
            name, pottype, model, config = case
            run([str(args.imrmlp), "maxvol_select", "--settings", config,
                 "--pool", "pool.xyz", "--index", "index.map"])
            state = Path(model + ".sel").read_bytes()
            reference = None
            for ranks in (1, 2, 4):
                actual = offline(case, lammps(case, ranks))
                assert [field(h, "timestep") for h, _ in actual] == [1, 2]
                if reference is not None:
                    for (a, _), (b, _) in zip(actual, reference):
                        close(field(a, "MV_grade"), field(b, "MV_grade"))
                reference = actual
            assert Path(model + ".sel").read_bytes() == state
            # Explicit canonical type selection must match metadata detection.
            offline(case, lammps(case, 2, "-explicit", options=f"pottype {pottype}"))
            print(f"PASS native/offline and 1/2/4-rank parity: {name}", flush=True)

        lammps(cases[0], 4, "-bad-type", options="pottype lj612", error="unsupported MTP")
        lammps(cases[0], 4, "-bad-size", state=cases[-1][2] + ".sel", error="coefficient count")
        print("PASS collective compatibility errors", flush=True)

        # Basis-only selection must load the derived MTP model consistently too.
        for case in (case for case in cases if case[0] in ("mtp_m_n2", "sog")):
            config = Path(case[3])
            config.write_text(config.read_text().replace("builder:dtype pot_efs", "builder:dtype mtm"))
            run([str(args.imrmlp), "maxvol_select", "--settings", str(config),
                 "--pool", "pool.xyz", "--index", "index.map"])
            offline(case, lammps(case, 4, "-mtm"))
            lammps(cases[0], 4, "-bad-mtm", state=case[2] + ".sel", error="pot_efs state")
        print("PASS derived-model basis states and composite-model rejection", flush=True)


if __name__ == "__main__":
    main()
