#!/usr/bin/env python3
"""Compare native LAMMPS E/F/global and site virials with the original IMR-MLP models."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
import subprocess
import tempfile


def run(command, environment, expected_error=None):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=environment, timeout=120)
    if expected_error is not None:
        if result.returncode == 0 or (expected_error and expected_error.lower() not in result.stdout.lower()):
            raise AssertionError(f"Expected failure containing {expected_error!r}:\n{result.stdout}")
    elif result.returncode:
        raise RuntimeError(f"Command failed: {' '.join(map(str, command))}\n{result.stdout}")
    return result.stdout


def close(a, b, label, per_atom=False):
    if len(a) != len(b):
        raise AssertionError(f"{label}: dimensions {len(a)} != {len(b)}")
    for i, (left, right) in enumerate(zip(a, b)):
        if isinstance(left, list):
            close(left, right, f"{label} row {i}", per_atom=True)
        elif not math.isfinite(left) or not math.isfinite(right) or abs(left-right) > ((1e-10+1e-9*max(abs(left), abs(right))) if per_atom and i >= 11 else (1e-8+2e-9*max(abs(left), abs(right)))):
            raise AssertionError(f"{label} column {i}: {left:.17g} != {right:.17g}")


def read_dump(path, nktv2p):
    lines = path.read_text().splitlines()
    header = [i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS")][-1]
    rows = []
    for line in lines[header+1:]:
        if line.startswith("ITEM:"):
            break
        values = list(map(float, line.split()))
        # id, fx/fy/fz, site energy, six LAMMPS stress*volume values, observables.
        for column in range(5, 11):
            values[column] /= -nktv2p
        rows.append(values)
    return sorted(rows, key=lambda row: row[0])


def setup(case):
    box = ("region box prism 0 12 0 12 0 12 1.1 -0.7 0.8 units box" if
           case["geometry"] == "triclinic" else "region box block 0 12 0 12 0 12 units box")
    positions = ["6 6 6", "8 6 6", "4 6 6"] if case["geometry"] == "tie" else ["2 2 2", "4.4 2.6 2.5", "7.1 2.8 2.9"]
    if case["geometry"] == "four_rank": positions = ["1.5 2 2", "4.5 2.6 2.5", "7.5 2.8 2.9"]
    extra_atom = "create_atoms 1 single 10.5 3.1 3.3 units box" if case["geometry"] == "four_rank" else ""
    return f"""units metal
atom_style charge
boundary {"f f f" if case["geometry"] == "cluster" else "p p p"}
newton on
processors * 1 1
{box}
create_box 2 box
create_atoms 2 single {positions[0]} units box
create_atoms 1 single {positions[1]} units box
create_atoms 2 single {positions[2]} units box
{extra_atom}
mass * 27
set atom 1 charge 0.2
set atom 2 charge -0.1
set atom 3 charge 0.2
"""


def observables(case, dump, metrics, nktv2p, steps=0):
    commands = ["compute energy all pe/atom", "compute stress all stress/atom NULL pair",
                "compute pressure all pressure NULL pair"]
    columns = ["id", "fx", "fy", "fz", "c_energy"] + [f"c_stress[{i}]" for i in range(1, 7)]
    if int(case["charges"]):
        commands.append("compute charge all mlip/native/charge")
        columns.append("c_charge")
    channels = int(case["channels"])
    if channels:
        commands.append("compute latent all mlip/native/latent")
        columns.extend(["c_latent"] if channels == 1 else [f"c_latent[{i}]" for i in range(1, channels+1)])
    commands += ["thermo 1", "thermo_style custom step pe " + " ".join(f"c_pressure[{i}]" for i in range(1, 7)), "thermo_modify format float %.17g",
                 f"dump result all custom 1 {dump} {' '.join(columns)}",
                 "dump_modify result sort id format float %.17g", f"run {steps}"]
    values = ["$(pe:%.17g)"] + [f"$(c_pressure[{i}]*vol/{nktv2p:.17g}:%.17g)" for i in range(1, 7)]
    commands.append(f'print "{" ".join(values)}" file {metrics} screen no')
    return "\n".join(commands)+"\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, help="Omit for styles built into lmp")
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda", "hip"], default="cpu")
    parser.add_argument("--gpm", action="store_true", help="Deploy self-contained GPM files after removing legacy model inputs")
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--models", nargs="+", help="Restrict model names; omitted means the entire family")
    parser.add_argument("--geometries", nargs="+", help="Restrict geometry/scenario names")
    parser.add_argument("--work", type=Path, help="Keep inputs and result files in this directory")
    parser.add_argument("--nktv2p", type=float, default=1.6021765e6,
                        help="LAMMPS metal pressure conversion (update.cpp)")
    args = parser.parse_args()
    environment = os.environ.copy()
    if args.gpm:
        environment["IMRMLP_NATIVE_GPM"] = "1"
    environment["OMP_NUM_THREADS"] = str(args.threads)
    if args.device != "cpu":
        environment["IMRMLP_REQUIRE_GPU"] = "1"
    temporary = tempfile.TemporaryDirectory(prefix="native-lammps-") if args.work is None else None
    work = args.work.resolve() if args.work else Path(temporary.name)
    work.mkdir(parents=True, exist_ok=True)
    run([str(args.generator.resolve()), str(work / "models")], environment)
    with (work / "models/cases.tsv").open() as handle:
        cases = list(csv.DictReader(handle, delimiter="\t"))
    if args.models:
        available = {case["name"] for case in cases}
        missing = set(args.models)-available
        if missing:
            parser.error(f"Unknown models: {sorted(missing)}")
        cases = [case for case in cases if case["name"] in args.models]
    if args.geometries:
        cases = [case for case in cases if case["geometry"] in args.geometries]
    if not cases:
        raise RuntimeError("No native models selected")
    plugin = f"plugin load {args.plugin.resolve()}\n" if args.plugin else ""
    base = [args.mpiexec, "-n", None, str(args.lmp.resolve()), "-log", "none", "-in"]
    records = []
    lifecycle_count = 0
    reload_count = 0
    dynamics_count = 0
    for case in cases:
        reference_lines = Path(case["reference"]).read_text().splitlines()
        expected_global = list(map(float, reference_lines[0].split()))
        expected_atoms = [list(map(float, line.split())) for line in reference_lines[1:]]
        pair = (f"pair_style mlip/native device {args.device} precision fp64 pottype {case['pottype']} " +
                (f"settings {case['settings']}" if case['settings'] else "") +
                (f" gpu_id {args.gpu_id}" if args.device != "cpu" else "") + "\n" +
                f"pair_coeff * * {case['model']} 0 1\n")
        for rank in args.ranks:
            label = f"{case['name']}-{case['geometry']}-{args.device}-{rank}"
            directory = work / label
            directory.mkdir(exist_ok=True)
            dump, metrics, restart = directory / "atoms.dump", directory / "global.txt", directory / "model.restart"
            script = directory / "in.static"
            script.write_text(plugin+setup(case)+pair+observables(case, dump, metrics, args.nktv2p)+
                              f"write_restart {restart}\n")
            command = base.copy(); command[2] = str(rank)
            output = run(command+[str(script)], environment)
            (directory / "log.txt").write_text(output)
            close(expected_global, list(map(float, metrics.read_text().split())), label+" global")
            close(expected_atoms, read_dump(dump, args.nktv2p), label+" atoms")
            statistics = [{"rank": int(r), "owned_atoms": int(n), "ghost_atoms": int(g), "gpu_id": int(d),
                           "neighbor_edges": int(e), "model_evaluations": int(v),
                           "force_kernel_launches": int(k), "model_device_bytes": int(m)}
                          for r, n, g, d, e, v, k, m in re.findall(
                              r"mlip/native rank=(\d+) owned_atoms=(\d+) ghost_atoms=(\d+) gpu_id=(-?\d+) "
                              r"neighbor_edges=(\d+) model_evaluations=(\d+) force_kernel_launches=(\d+) model_device_bytes=(\d+)", output)]
            if case["geometry"] == "four_rank" and rank == 4:
                if len(statistics) != 4 or any(item["owned_atoms"] != 1 for item in statistics):
                    raise AssertionError(f"{label}: expected exactly one owned atom on every rank")
            if args.device != "cpu":
                if len(statistics) != rank or any(item["owned_atoms"] and
                        (item["force_kernel_launches"] <= 0 or item["model_evaluations"] <= 0 or item["model_device_bytes"] <= 0)
                        for item in statistics):
                    raise AssertionError(f"{label}: missing owned-rank GPU kernel or memory evidence")
            records.append({"model": case["name"], "geometry": case["geometry"], "device": args.device,
                            "ranks": rank, "threads": args.threads, "statistics": statistics,
                            "gpu_id": args.gpu_id if args.device != "cpu" else None,
                            "placement": [{"rank": int(r), "owned_atoms": int(n), "ghost_atoms": int(g), "gpu_id": int(d)}
                                          for r, n, g, d in re.findall(
                                              r"mlip/native rank=(\d+) owned_atoms=(\d+) ghost_atoms=(\d+) gpu_id=(-?\d+)", output)]})
        # Restart with a different rank count; the model and settings references
        # are restored by the pair, without issuing another pair_coeff command.
        rank = args.ranks[0]
        directory = work / f"{case['name']}-{case['geometry']}-restart"
        directory.mkdir(exist_ok=True)
        dump, metrics = directory / "atoms.dump", directory / "global.txt"
        script = directory / "in.restart"
        script.write_text(plugin+f"read_restart {restart}\n"+observables(case, dump, metrics, args.nktv2p))
        command = base.copy(); command[2] = str(rank)
        run(command+[str(script)], environment)
        close(expected_global, list(map(float, metrics.read_text().split())), str(script)+" global")
        close(expected_atoms, read_dump(dump, args.nktv2p), str(script)+" atoms")
        # Rebuild the neighbor list after an owned-domain migration. Also
        # exercises a force/energy-only evaluation followed by full site output.
        directory = work / f"{case['name']}-{case['geometry']}-migration"
        directory.mkdir(exist_ok=True)
        dump, metrics = directory / "atoms.dump", directory / "global.txt"
        script = directory / "in.migration"
        cell_motion = ("" if case["geometry"] == "tie" else
                       "change_box all x scale 1.02 remap\nrun 0\nchange_box all x scale 0.9803921568627451 remap\n")
        migration_pair = pair.replace(" 0 1\n", " Al Ni\n") if case["name"] in {"mtp", "mtp_split"} else pair
        script.write_text(plugin+setup(case)+migration_pair+
                          "thermo_style custom step pe\nrun 0\n"+
                          cell_motion+
                          "displace_atoms all move 2 0 0 units box\n"+
                          observables(case, dump, metrics, args.nktv2p))
        command = base.copy(); command[2] = str(args.ranks[-1])
        run(command+[str(script)], environment)
        close(expected_global, list(map(float, metrics.read_text().split())), str(script)+" global")
        close(expected_atoms, read_dump(dump, args.nktv2p), str(script)+" atoms")
        lifecycle_count += 1
        # A short real integration for every family. The rank-one CPU trace is
        # independent of GPU selection and supplies the same fixed-seed reference.
        traces = []
        for trial, device, ranks in [("reference", "cpu", 1), ("deployment", args.device, args.ranks[-1])]:
            directory = work / f"{case['name']}-{case['geometry']}-dynamics-{trial}"
            directory.mkdir(exist_ok=True)
            dump, metrics, coordinates = directory / "atoms.dump", directory / "global.txt", directory / "coordinates.dump"
            script = directory / "in.dynamics"
            dynamic_pair = pair
            if device == "cpu" and args.device != "cpu":
                dynamic_pair = pair.replace(f"device {args.device}", "device cpu").replace(f" gpu_id {args.gpu_id}", "")
            script.write_text(plugin+setup(case)+dynamic_pair+
                              "velocity all create 1.0 246813 mom yes rot no dist gaussian\n"
                              "timestep 0.00001\nfix integrate all nve\n"+
                              observables(case, dump, metrics, args.nktv2p, steps=5)+
                              f"write_dump all custom {coordinates} id x y z vx vy vz modify sort id format float %.17g\n")
            command = base.copy(); command[2] = str(ranks)
            run(command+[str(script)], environment)
            lines = coordinates.read_text().splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
            traces.append((list(map(float, metrics.read_text().split())), read_dump(dump, args.nktv2p),
                           sorted([list(map(float, line.split())) for line in lines[start+1:]])))
        for index, quantity in enumerate(["global", "atoms", "trajectory"]):
            close(traces[0][index], traces[1][index], f"{case['name']} short dynamics {quantity}")
        dynamics_count += 1
        if case["name"] == "sog3":
            source = next((item for item in cases if item["name"] == "sog" and item["geometry"] == case["geometry"]), None)
            if source:
                directory = work / f"sog3-{case['geometry']}-reload"
                directory.mkdir(exist_ok=True)
                dump, metrics = directory / "atoms.dump", directory / "global.txt"
                script = directory / "in.reload"
                first = (f"pair_style mlip/native device {args.device} precision fp64 pottype {source['pottype']} " +
                         (f"settings {source['settings']}" if source['settings'] else "") +
                         (f" gpu_id {args.gpu_id}" if args.device != "cpu" else "") + "\n" +
                         f"pair_coeff * * {source['model']} 0 1\n")
                script.write_text(plugin+setup(case)+first+"compute old all mlip/native/latent\nrun 0\n"+
                                  "uncompute old\npair_style none\n"+pair+
                                  observables(case, dump, metrics, args.nktv2p))
                run(command+[str(script)], environment)
                close(expected_global, list(map(float, metrics.read_text().split())), str(script)+" global")
                close(expected_atoms, read_dump(dump, args.nktv2p), str(script)+" atoms")
                reload_count += 1
    # Invalid input must fail collectively instead of hanging one MPI rank.
    case = cases[0]
    corrupt = work / "corrupt.mtp"
    corrupt.write_text("MTP\nnot_a_model = invalid\n")
    errors = [
        ("missing", "pair_style mlip/native pottype mtp\npair_coeff * * /nonexistent/native-model 0 1\n", "cannot"),
        ("mapping", f"pair_style mlip/native pottype {case['pottype']} " + (f"settings {case['settings']}" if case['settings'] else "") + "\n" +
                    f"pair_coeff * * {case['model']} 0 Ni\n", "cannot mix"),
        ("precision", "pair_style mlip/native pottype mtp precision mixed\n", "fp64"),
        ("corrupt", f"pair_style mlip/native pottype mtp\npair_coeff * * {corrupt} 0 1\n", ""),
    ]
    sog = next((item for item in cases if item["name"] == "sog3"), None)
    if sog:
        for key, value in [("les_kernel", "coulomb"), ("sog_channels", "7"), ("lr_les", "false")]:
            settings = work / f"conflict-{key}.ini"
            settings.write_text(f"pot:{key} {value}\n")
            errors.append((key, f"pair_style mlip/native pottype mtp_les settings {settings}\n"
                                f"pair_coeff * * {sog['model']} 0 1\n", "configuration conflicts" if args.gpm else "conflicts with model metadata"))
    for label, tail, message in errors:
        script = work / f"in.error-{label}"
        script.write_text(plugin+setup(case)+tail)
        command = base.copy(); command[2] = str(args.ranks[-1])
        run(command+[str(script)], environment, message)
    (work / "results.json").write_text(json.dumps({"passed": records, "restart_cases": len(cases), "error_cases": len(errors), "migration_and_cell_cases": lifecycle_count, "reload_cases": reload_count, "short_dynamics_cases": dynamics_count}, indent=2)+"\n")
    print(f"Native LAMMPS family parity passed: {len(records)} static comparisons, {len(cases)} restarts, {lifecycle_count} migration/cell cases, {reload_count} reloads, {dynamics_count} short dynamics, {len(errors)} error cases")
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
