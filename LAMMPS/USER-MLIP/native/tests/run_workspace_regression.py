#!/usr/bin/env python3
"""Exercise native workspace reuse against a CPU list rebuilt at every step.

Prescribed motion keeps both paths on identical coordinates. No force result is
cached, and every recorded frame includes global and per-atom E/F/virial.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


def execute(command, environment, directory):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=environment, timeout=300)
    (directory / "output.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stdout}")
    return result.stdout


def frames(path):
    lines = path.read_text().splitlines()
    output = []
    for index, line in enumerate(lines):
        if line != "ITEM: TIMESTEP":
            continue
        step, count = int(lines[index+1]), int(lines[index+3])
        header = index+8
        assert lines[header].startswith("ITEM: ATOMS"), lines[header]
        rows = [list(map(float, row.split())) for row in lines[header+1:header+count+1]]
        output.append((step, sorted(rows, key=lambda row: row[0])))
    return output


def values(path):
    return [list(map(float, line.split())) for line in path.read_text().splitlines()
            if line and not line.startswith("#")]


def compare(reference, actual, label, absolute=1e-8, relative=2e-9):
    if isinstance(reference, (list, tuple)):
        if len(reference) != len(actual):
            raise AssertionError(f"{label}: different sizes {len(reference)} / {len(actual)}")
        for i, (a, b) in enumerate(zip(reference, actual)):
            compare(a, b, f"{label}[{i}]", absolute, relative)
    elif not math.isfinite(actual) or abs(reference-actual) > absolute+relative*max(abs(reference), abs(actual)):
        raise AssertionError(f"{label}: {reference:.17g} != {actual:.17g}")


def compare_frames(reference, actual, label):
    if len(reference) != len(actual):
        raise AssertionError(f"{label}: frame counts differ")
    for (step_a, atoms_a), (step_b, atoms_b) in zip(reference, actual):
        compare(step_a, step_b, label+" timestep")
        if len(atoms_a) != len(atoms_b):
            raise AssertionError(f"{label}: atom counts differ")
        for a, b in zip(atoms_a, atoms_b):
            # id/type/coordinates/forces/E/six virials precede q and latent.
            compare(a[:15], b[:15], f"{label} step {step_a} atom {a[0]}")
            compare(a[15:], b[15:], f"{label} charge/latent step {step_a} atom {a[0]}",
                    absolute=1e-10, relative=1e-9)


def source(case, scenario, plugin, device, directory, rebuilt, nktv2p):
    name = scenario["name"]
    length = 4 if name == "periodic_images_tie" else 16
    positions = [(2., 2., 2.), (7.1, 2., 2.), (12., 10., 10.)]
    if name == "feature_cutoff":
        positions[1] = (8.6, 2., 2.)
    if name == "periodic_images_tie":
        positions = [(1., 1., 1.), (3., 1., 1.), (1., 3., 1.)]
    if name == "type_change":
        # Keep interacting centers in the first domain at 1/2/4 ranks. LAMMPS
        # set changes owned types only; pre no cannot refresh remote ghost
        # chemistry. The separate type_change_mpi scene uses normal run setup;
        # this scene isolates a same-list type change.
        positions = [(1., 2., 2.), (2.2, 2., 2.), (3.4, 2.2, 2.)]
    if name == "request_switch":
        positions = [(2., 2., 2.), (4.5, 2.4, 2.), (7., 3., 2.)]
    if name in {"sorting_migration", "deforming_cell", "type_change_mpi"}:
        positions = [(1.5, 2., 2.), (5.5, 2.6, 2.5), (9.5, 2.8, 2.9), (13.5, 3.1, 3.3)]
    region = (f"region box prism 0 {length} 0 {length} 0 {length} 0 0 0" if name == "deforming_cell"
              else f"region box block 0 {length} 0 {length} 0 {length}")
    lines = ([f"plugin load {plugin}"] if plugin else []) + ["units metal", "atom_style charge", "boundary p p p",
             "newton on", "processors * 1 1", region,
             "create_box 2 box"]
    for index, position in enumerate(positions):
        lines.append(f"create_atoms {1+index%2} single {' '.join(map(str, position))} units box")
    lines += ["mass * 27", "atom_modify sort 1 1.0",
              f"pair_style mlip/native device {device} precision fp64 pottype {case['pottype']} settings {case['settings']}",
              f"pair_coeff * * {case['model']} 0 1", "neighbor 2.0 bin"]
    if rebuilt:
        lines.append("neigh_modify every 1 delay 0 check no")
    elif name == "sorting_migration":
        lines.append("neigh_modify every 1 delay 0 check yes")
    else:
        lines.append("neigh_modify every 100 delay 0 check no")
    lines += ["timestep 0.01", "compute e all pe", "compute ea all pe/atom",
              "compute wa all stress/atom NULL pair", "compute p all pressure NULL pair"]
    columns = ["id", "type", "xu", "yu", "zu", "fx", "fy", "fz", "c_ea"]
    # Store virials in eV so the tolerance matches the other regression suites.
    for k in range(1, 7):
        lines.append(f"variable wa{k} atom -c_wa[{k}]/{nktv2p:.17g}")
        lines.append(f"variable wg{k} equal c_p[{k}]*vol/{nktv2p:.17g}")
        columns.append(f"v_wa{k}")
    if int(case["charges"]):
        lines.append("compute q all mlip/native/charge")
        columns.append("c_q")
    channels = int(case["channels"])
    if channels:
        lines.append("compute z all mlip/native/latent")
        columns += ["c_z"] if channels == 1 else [f"c_z[{i}]" for i in range(1, channels+1)]
    lines += [f"dump atoms all custom 1 {directory/'atoms.dump'} {' '.join(columns)}",
              "dump_modify atoms sort id format float %.17g",
              f'fix totals all ave/time 1 1 1 c_e {" ".join(f"v_wg{k}" for k in range(1, 7))} file {directory/"global.txt"} format " %.17g"',
              "thermo 1", "thermo_style custom step atoms", "group moving id 2"]
    if name in {"cutoff_crossing", "feature_cutoff"}:
        lines += ["fix motion moving move wiggle 0.2 0 0 0.4 units box", "run 40"]
    elif name == "sorting_migration":
        lines += ["fix motion all move linear 10 0 0 units box", "run 40"]
    elif name == "deforming_cell":
        lines += ["fix deforming all deform 1 x erate 0.1 xy erate 0.05 remap x", "run 12"]
    elif name == "type_change":
        # pre no deliberately avoids the usual run-setup neighbor rebuild.
        lines += ["run 2", "set atom 2 type 1", "run 2 pre no", "set atom 1 type 2", "run 2 pre no"]
    elif name == "type_change_mpi":
        lines += ["run 2", "set atom 2 type 1", "run 2", "set atom 1 type 2", "run 2"]
    elif name == "request_switch":
        # Remove consumers for a force-only run, then restore the same complete
        # output contract. The last stage uses the same persistent pair object.
        lines += ["fix motion moving move linear 0.1 0 0 units box", "run 2", "undump atoms", "unfix totals", "run 2",
                  f"dump atoms all custom 1 {directory/'restored.dump'} {' '.join(columns)}",
                  "dump_modify atoms sort id format float %.17g",
                  "run 2"]
    else:
        # Multiple images of each physical atom have tied nearest distances.
        # A stationary run still calls every kernel on every step.
        lines.append("run 8")
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, help="Omit for styles built into lmp")
    parser.add_argument("--generator", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--device", choices=["cpu", "cuda", "hip"], default="cpu")
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--nktv2p", type=float, default=1.6021765e6)
    args = parser.parse_args()
    if not args.models_dir and not args.generator:
        parser.error("provide --models-dir or --generator")
    temporary = tempfile.TemporaryDirectory(prefix="native-workspace-") if not args.work else None
    work = args.work.resolve() if args.work else Path(temporary.name)
    work.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    if args.device != "cpu":
        environment["IMRMLP_REQUIRE_GPU"] = "1"
    models = args.models_dir.resolve() if args.models_dir else work/"models"
    if not args.models_dir:
        execute([str(args.generator.resolve()), str(models)], environment, work)
    with (models/"cases.tsv").open() as handle:
        cases = {case["name"]: case for case in csv.DictReader(handle, delimiter="\t")
                 if case["geometry"] == "orthogonal"}
    scenarios = [{"name": name, "model": model} for name, model in [
        ("cutoff_crossing", "mtp"), ("feature_cutoff", "scale_coordination"),
        ("periodic_images_tie", "scale_distance"), ("type_change", "mtp_split"),
        ("type_change_mpi", "mtp_split"),
        ("sorting_migration", "mtp_split"), ("deforming_cell", "mtp_split"),
        ("request_switch", "sog3")]]
    if args.scenarios:
        missing = set(args.scenarios)-{item["name"] for item in scenarios}
        if missing:
            parser.error(f"unknown scenarios: {sorted(missing)}")
        scenarios = [item for item in scenarios if item["name"] in args.scenarios]
    records = []
    for scenario in scenarios:
        reference = None
        for label, device, ranks, rebuilt in [("reference", "cpu", 1, True)] + [
                (f"{args.device}-r{rank}", args.device, rank, False) for rank in args.ranks]:
            directory = work/f"{scenario['name']}-{label}"
            directory.mkdir(exist_ok=True)
            script = directory/"input.lmp"
            script.write_text(source(cases[scenario["model"]], scenario, args.plugin.resolve() if args.plugin else None,
                                     device, directory, rebuilt, args.nktv2p))
            output = execute([args.mpiexec, "-n", str(ranks), str(args.lmp.resolve()),
                              "-log", "none", "-in", str(script)], environment, directory)
            atom_frames = frames(directory/"atoms.dump")
            global_frames = values(directory/"global.txt")
            restored = frames(directory/"restored.dump") if (directory/"restored.dump").exists() else []
            observed = (atom_frames, global_frames, restored)
            if reference is None:
                reference = observed
            else:
                compare_frames(reference[0], observed[0], f"{scenario['name']} {label}")
                compare(reference[1], observed[1], f"{scenario['name']} {label} global")
                compare_frames(reference[2], observed[2], f"{scenario['name']} {label} restored")
            builds = list(map(int, re.findall(r"Neighbor list builds = (\d+)", output)))
            if scenario["name"] in {"cutoff_crossing", "feature_cutoff"}:
                cutoff = 5.0 if scenario["name"] == "cutoff_crossing" else 6.5
                separations = [abs(rows[1][2]-rows[0][2]) for _, rows in atom_frames]
                if not min(separations) < cutoff < max(separations):
                    raise AssertionError("test motion did not cross the physical cutoff")
                if not rebuilt and builds != [0]:
                    raise AssertionError(f"cutoff crossing unexpectedly rebuilt its skin list: {builds}")
            if scenario["name"] == "sorting_migration" and not any(builds):
                raise AssertionError("migration scene never rebuilt a neighbor list")
            records.append({**scenario, "device": device, "ranks": ranks, "reference": rebuilt,
                            "frames": len(atom_frames)+len(restored), "neighbor_builds": builds,
                            "passed": True})
    summary = {"passed": True, "scenarios": len(scenarios), "runs": records}
    (work/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"passed": True, "scenarios": len(scenarios), "runs": len(records)}))


if __name__ == "__main__":
    main()
