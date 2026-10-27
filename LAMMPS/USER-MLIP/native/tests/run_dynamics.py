#!/usr/bin/env python3
"""10 ps FP64 NVE, timestep convergence and restart on bound MTP dimers."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def execute(command, env, log):
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=900)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout


def energies(output):
    rows = []
    reading = False
    for line in output.splitlines():
        fields = line.split()
        if fields == ["Step", "Time", "PotEng", "KinEng", "TotEng"]:
            reading = True
            continue
        if reading:
            try:
                row = list(map(float, fields))
            except ValueError:
                reading = False
                continue
            if len(row) != 5:
                reading = False
            else:
                rows.append(row)
    if not rows or not all(math.isfinite(x) for row in rows for x in row):
        raise AssertionError("Missing or nonfinite NVE energies")
    return rows


def positions(path):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    return sorted([list(map(float, line.split())) for line in lines[start+1:]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", required=True, type=Path)
    parser.add_argument("--plugin", type=Path, help="Omit for styles built into lmp")
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "hip"], default="cpu")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory(prefix="native-nve-") if not args.work else None
    work = args.work.resolve() if args.work else Path(temporary.name)
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    if args.device != "cpu":
        env["IMRMLP_REQUIRE_GPU"] = "1"
    execute([str(args.generator.resolve()), str(work/"models")], env, work/"generator.log")
    model = work/"models/stable-dimer.mtp"
    plugin = f"plugin load {args.plugin.resolve()}\n" if args.plugin else ""
    setup = "units metal\natom_style atomic\nboundary p p p\nnewton on\nprocessors * 1 1\n"
    setup += "region box block 0 48 0 12 0 12 units box\ncreate_box 1 box\n"
    for center in [2., 14., 26., 38.]:
        for x in [center, center+2.55]:
            setup += f"create_atoms 1 single {x:.17g} 2 2 units box\n"
    setup += "mass * 27\nvelocity all set 0 0 0\nneighbor 0.5 bin\nneigh_modify every 1 delay 0 check yes\n"
    monitor = "fix integrate all nve\nthermo 100\nthermo_style custom step time pe ke etotal\nthermo_modify format float %.17g\n"
    command = [args.mpiexec, "-n", str(args.ranks), str(args.lmp.resolve()), "-log", "none", "-in"]
    results = {}
    comparisons = {}
    devices = ["cpu"] if args.device == "cpu" else ["cpu", args.device]
    for device in devices:
        pair = f"pair_style mlip/native device {device} precision fp64 pottype mtp\npair_coeff * * {model} 0\n"
        results[device] = {}
        for dt in [.001, .0005]:
            steps = round(args.duration/dt)
            label = f"{device}-{dt}"
            dump = work/f"{label}.dump"
            script = work/f"in.{label}"
            final_dump = f"write_dump all custom {dump} id x y z vx vy vz modify sort id format float %.17g\n"
            script.write_text(plugin+setup+pair+monitor+f"timestep {dt}\nrun {steps}\n"+final_dump)
            output = execute(command+[str(script)], env, work/f"{label}.log")
            rows = energies(output)
            drift = max(abs(row[4]-rows[0][4]) for row in rows)/8.
            if drift > 1e-5:
                raise AssertionError(f"Unstable bound-dimer NVE: {label} max drift {drift} eV/atom")
            results[device][str(dt)] = {"duration_ps": args.duration, "steps": steps,
                                       "maximum_drift_eV_per_atom": drift}
            if dt == .001:
                restart = work/f"{device}.restart"
                first = work/f"in.{device}.first"
                first.write_text(plugin+setup+pair+monitor+f"timestep {dt}\nrun {steps//2}\nwrite_restart {restart}\n")
                execute(command+[str(first)], env, work/f"{device}.first.log")
                continued_dump = work/f"{device}.continued.dump"
                second = work/f"in.{device}.second"
                second.write_text(plugin+f"read_restart {restart}\n"+monitor+f"run {steps-steps//2}\n"+
                                  f"write_dump all custom {continued_dump} id x y z vx vy vz modify sort id format float %.17g\n")
                execute(command+[str(second)], env, work/f"{device}.second.log")
                for left, right in zip(positions(dump), positions(continued_dump)):
                    if any(abs(a-b)>1e-8+2e-9*max(abs(a), abs(b)) for a, b in zip(left, right)):
                        raise AssertionError(f"{device} NVE restart continuation differs")
        coarse = results[device]["0.001"]["maximum_drift_eV_per_atom"]
        fine = results[device]["0.0005"]["maximum_drift_eV_per_atom"]
        if fine > .4*coarse+1e-10:
            raise AssertionError(f"{device} timestep halving failed to reduce energy error: {coarse}, {fine}")
    if args.device != "cpu":
        for dt in ["0.001", "0.0005"]:
            increment = (results[args.device][dt]["maximum_drift_eV_per_atom"] -
                         results["cpu"][dt]["maximum_drift_eV_per_atom"])
            if increment > 1e-5:
                raise AssertionError("GPU drift increment exceeds 1e-5 eV/atom/10 ps")
            reference = positions(work/f"cpu-{dt}.dump")
            deployed = positions(work/f"{args.device}-{dt}.dump")
            if len(reference) != len(deployed):
                raise AssertionError("CPU/GPU final NVE atom count differs")
            position_error = velocity_error = 0.
            for left, right in zip(reference, deployed):
                if len(left) != 7 or len(right) != 7 or left[0] != right[0]:
                    raise AssertionError("CPU/GPU final NVE atom mapping differs")
                for column, (a, b) in enumerate(zip(left[1:], right[1:]), 1):
                    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>1e-8+2e-9*max(abs(a),abs(b)):
                        raise AssertionError(f"CPU/GPU final NVE trajectory differs at dt={dt}, atom={left[0]}, column={column}")
                    if column <= 3:
                        position_error = max(position_error, abs(a-b))
                    else:
                        velocity_error = max(velocity_error, abs(a-b))
            comparisons[dt] = {"drift_increment_eV_per_atom": increment,
                               "maximum_position_difference_angstrom": position_error,
                               "maximum_velocity_difference_angstrom_per_ps": velocity_error}
    (work/"dynamics.json").write_text(json.dumps({"ranks": args.ranks, "threads": args.threads,
                                                "models": results, "gpu_cpu_comparison": comparisons}, indent=2)+"\n")
    print(json.dumps(results, indent=2))
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
