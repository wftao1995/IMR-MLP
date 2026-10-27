#!/usr/bin/env python3
"""Five-head standalone LAMMPS regression with an independent analytic oracle.

No NumPy or IMR-MLP dependency. Keep --work to retain inputs, logs and JSON results.
MPI/GPU execution must be scheduled on resources allocated by the caller.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
from fixtures import CELL, FRACTIONAL, HEADS, TYPES, matvec, reference_energy, write_models

PRESSURE = 1602176.5  # LAMMPS 22Jul2025 src/update.cpp metal force->nktv2p
MODES = ("force", "energy", "virial", "site_energy", "site_virial", "all")


def close(actual, expected, label, atol=1e-9, rtol=1e-8):
    if len(actual) != len(expected): raise AssertionError(f"{label}: wrong length")
    error = max((abs(a-b) for a,b in zip(actual,expected)), default=0.)
    if any(not math.isfinite(a) or abs(a-b)>atol+rtol*abs(b) for a,b in zip(actual,expected)):
        raise AssertionError(f"{label}: maximum absolute error {error:.6g}")
    return error


def execute(args, script, ranks, log, expected_error=None):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    command = ([] if ranks == 1 and not args.always_mpi else shlex.split(args.mpiexec)+["-n",str(ranks)])
    command += [str(args.lmp.resolve()),"-log","none","-in",str(script)]
    for attempt in range(1,6):
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=args.timeout)
        log.write_text(result.stdout)
        # A scheduler RPC failure before the executable starts is independent
        # of the tested numerical result. Preserve it and retry only this exact
        # known launcher failure; application errors and timeouts never retry.
        not_started = (result.returncode != 0 and
                       'srun: error: Unable to create step for job ' in result.stdout and
                       'Unexpected message received' in result.stdout and
                       'LAMMPS (' not in result.stdout)
        if not not_started: break
        log.with_name(log.stem+f'.launcher-attempt{attempt}'+log.suffix).write_text(result.stdout)
        # A lost RPC reply must not create duplicate job steps. Retry only when
        # Slurm confirms that no ordinary step remains in this allocation.
        job = os.environ.get('SLURM_JOB_ID')
        if not job:
            match = re.search(r'Unable to create step for job (\d+)',result.stdout)
            job = match[1] if match else None
        if not job: break
        try:
            active = subprocess.run(['squeue','--noheader','--steps','--jobs',job,'--format=%i'],
                                    text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15)
        except (OSError,subprocess.TimeoutExpired): break
        log.with_name(log.stem+f'.launcher-attempt{attempt}.steps'+log.suffix).write_text(active.stdout)
        if active.returncode or any(line.strip() and not line.strip().endswith(('.batch','.extern'))
                                    for line in active.stdout.splitlines()): break
        if attempt < 5: time.sleep(2)
    if expected_error is not None:
        if result.returncode == 0 or expected_error.lower() not in result.stdout.lower():
            raise AssertionError(f"Expected rejection containing {expected_error!r}; see {log}\n{result.stdout[-5000:]}")
        return result.stdout
    if result.returncode: raise RuntimeError(f"{shlex.join(command)} failed; see {log}\n{result.stdout[-5000:]}")
    return result.stdout


def pair(args, model):
    style = args.style or ("mlip/opt" if args.device == "cpu" else "mlip/gpudirect")
    options = "" if args.device == "cpu" else f" device {args.device} precision fp64 gpu_id auto"
    if args.pair_options: options += " "+args.pair_options
    return f'pair_style {style}{options}\npair_coeff * * "{model}" Al Ni\n'


def setup(args):
    positions = [matvec(CELL,s) for s in FRACTIONAL]
    text = f'plugin load "{args.plugin.resolve()}"\n' if args.plugin else ""
    if args.threads > 1 or args.style == "mlip/opt/omp": text += f"package omp {args.threads}\n"
    text += "units metal\natom_style atomic\nboundary p p p\nnewton on\nprocessors * 1 1\n"
    text += "region box prism 0 12 0 12 0 12 1.1 -0.7 0.8 units box\ncreate_box 2 box\n"
    for xyz,t in zip(positions,TYPES): text += f"create_atoms {t+1} single "+" ".join(f"{x:.17g}" for x in xyz)+" units box\n"
    text += "mass 1 26.9815385\nmass 2 58.6934\nneighbor 0.5 bin\nneigh_modify every 1 delay 0 check yes\n"
    return text


def dump_rows(path):
    lines = path.read_text().splitlines()
    starts = [i for i,line in enumerate(lines) if line.startswith("ITEM: ATOMS")]
    if not starts: raise AssertionError(f"Missing dump: {path}")
    rows = []
    for line in lines[starts[-1]+1:]:
        if line.startswith("ITEM:"): break
        rows.append(list(map(float,line.split())))
    return sorted(rows,key=lambda x:x[0])


def thermo(output):
    header = None
    rows = []
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == "Step": header = fields; continue
        if header:
            try: row = list(map(float, fields))
            except ValueError: header = None; continue
            if len(row) == len(header): rows.append(dict(zip(header,row)))
            else: header = None
    if not rows: raise AssertionError("No thermo output")
    return rows


def snapshot(args, work, head, rank, mode, prefix=""):
    label = prefix+f"{head}-r{rank}-{mode}"
    script, dump = work/("in."+label), work/(label+".dump")
    energy = mode in ("energy","all")
    virial = mode in ("virial","all")
    site_energy = mode in ("site_energy","all")
    site_virial = mode in ("site_virial","all")
    fields, columns, computes = ["step"], ["id","fx","fy","fz"], ""
    if energy: fields.append("pe")
    if virial:
        computes += "compute wp all pressure NULL pair\n"
        fields += [f"c_wp[{k}]" for k in range(1,7)]
    if site_energy:
        computes += "compute se all pe/atom\n"; columns += ["c_se"]
    if site_virial:
        computes += "compute sw all stress/atom NULL pair\n"; columns += [f"c_sw[{k}]" for k in range(1,7)]
    text = setup(args)+pair(args,work/"models"/(head+".mtp"))+computes
    text += "thermo 1\nthermo_style custom "+" ".join(fields)+"\nthermo_modify format float %.17g\n"
    text += f'dump result all custom 1 "{dump}" '+" ".join(columns)+"\ndump_modify result sort id format float %.17g\nrun 0\n"
    script.write_text(text)
    output = execute(args,script,rank,work/(label+".log"))
    rows, global_values = dump_rows(dump), thermo(output)[-1]
    result = {"forces":[x for row in rows for x in row[1:4]]}
    if energy: result["energy"] = [global_values["PotEng"]]
    if virial: result["virial"] = [global_values[f"c_wp[{k}]"]*12**3/PRESSURE for k in range(1,7)]
    if site_energy: result["site_energy"] = [row[4] for row in rows]
    if site_virial:
        offset = 5 if site_energy else 4
        result["site_virial_sum"] = [-sum(row[offset+k] for row in rows)/PRESSURE for k in range(6)]
    return result


def oracle(head, positions=None, types=None):
    positions = positions if positions is not None else [matvec(CELL,s) for s in FRACTIONAL]
    types = types if types is not None else TYPES
    energy, sites = reference_energy(head,positions,types=types)
    force_sets, virial_sets = [], []
    for h in (1e-4,3e-5,1e-5):
        forces = []
        for i in range(len(positions)):
            for k in range(3):
                plus, minus = copy.deepcopy(positions), copy.deepcopy(positions)
                plus[i][k] += h; minus[i][k] -= h
                forces.append(-(reference_energy(head,plus,types=types)[0]-reference_energy(head,minus,types=types)[0])/(2*h))
        virial = []
        for a,b in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
            values = []
            for sign in (1,-1):
                transform = [[float(x==y) for y in range(3)] for x in range(3)]
                transform[a][b] += sign*h*(1 if a==b else .5)
                if a != b: transform[b][a] += sign*h*.5
                newcell = [[sum(transform[i][k]*CELL[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
                values.append(reference_energy(head,[matvec(transform,x) for x in positions],newcell,types)[0])
            virial.append(-(values[0]-values[1])/(2*h))
        force_sets.append(forces); virial_sets.append(virial)
    for values in force_sets[:-1]: close(values,force_sets[-1],head+" force FD convergence",1e-8,1e-6)
    for values in virial_sets[:-1]: close(values,virial_sets[-1],head+" strain FD convergence",1e-8,1e-6)
    return {"energy":[energy],"site_energy":sites,"forces":force_sets[-1],"virial":virial_sets[-1],"site_virial_sum":virial_sets[-1]}


def startup(args):
    text = f'plugin load "{args.plugin.resolve()}"\n' if args.plugin else ""
    if args.threads > 1 or args.style == "mlip/opt/omp": text += f"package omp {args.threads}\n"
    return text


def lifecycle_monitor(dump):
    return ("compute se all pe/atom\ncompute sw all stress/atom NULL pair\n"
            "compute wp all pressure NULL pair\nthermo 1\n"
            "thermo_style custom step pe c_wp[1] c_wp[2] c_wp[3] c_wp[4] c_wp[5] c_wp[6]\n"
            "thermo_modify format float %.17g\n"
            f'dump result all custom 1 "{dump}" id x y z type fx fy fz c_se c_sw[1] c_sw[2] c_sw[3] c_sw[4] c_sw[5] c_sw[6]\n'
            "dump_modify result sort id format float %.17g\n")


def lifecycle_result(output,dump,head,factor=1.0):
    rows = dump_rows(dump)
    if not rows: raise AssertionError("Lifecycle fixture unexpectedly lost every atom")
    expected = oracle(head,[r[1:4] for r in rows],[int(r[4])-1 for r in rows])
    values = thermo(output)[-1]
    actual = {"energy":[values["PotEng"]],"forces":[v for r in rows for v in r[5:8]],
              "site_energy":[r[8] for r in rows],
              "virial":[values[f"c_wp[{k}]"]*12**3/PRESSURE for k in range(1,7)],
              "site_virial_sum":[-sum(r[9+k] for r in rows)/PRESSURE for k in range(6)]}
    return {"atoms":len(rows),"maximum_errors":{
        key:close(v,[factor*x for x in expected[key]],"lifecycle "+head+" "+key,1e-8,1e-7)
        for key,v in actual.items()}}


def lifecycle(args,work,head,rank):
    results = {}
    model = work/"models"/(head+".mtp")
    base = setup(args)+pair(args,model)
    def scenario(name,change,expected_head=head,factor=1.0,before="",final_pre=True):
        label = f"lifecycle-{head}-r{rank}-{name}"
        script,dump = work/("in."+label),work/(label+".dump")
        script.write_text(base+before+lifecycle_monitor(dump)+"run 0\n"+change+
                          ("run 0\n" if final_pre else "run 0 pre no\n")+
                          # Regular dumps suppress repeated timesteps; reload,
                          # overlay and deletion can change state at the same step.
                          'undump result\n'+
                          f'write_dump all custom "{dump}" id x y z type fx fy fz c_se c_sw[1] c_sw[2] c_sw[3] c_sw[4] c_sw[5] c_sw[6] modify sort id format float %.17g\n')
        output=execute(args,script,rank,work/(label+".log"))
        results[name]=lifecycle_result(output,dump,expected_head,factor)
    motion=("velocity all set 0.03 -0.02 0.01\nfix move all nve\n"
            "timestep 0.001\nrun 3 pre no\nunfix move\n")
    scenario("sorting",motion,before="atom_modify sort 1 1.0\n")
    scenario("neighbors-every",motion,before="neigh_modify every 1 delay 0 check no\n")
    scenario("neighbors-adaptive",motion,before="neigh_modify every 1 delay 0 check yes\n")
    # Atom types are not part of LAMMPS' per-step coordinate forward exchange.
    # Normal run setup rebuilds borders so periodic/remote ghosts receive them.
    # Reusing topology with refreshed types is covered separately by core tests.
    scenario("type-change","set atom 1 type 2\nrun 1\n")
    scenario("migration","group moved id 1\ndisplace_atoms moved move 4.2 0.2 0.1 units box\nrun 1\n")
    scenario("grow-shrink","create_atoms 2 single 7.2 7.1 3.0 units box\nrun 1\n"
             "group doomed id 2\ndelete_atoms group doomed compress no\n")
    scenario("empty-ranks","group retained id 1 2\ngroup removed subtract all retained\n"
             "delete_atoms group removed compress no\n")
    replacement=HEADS[(HEADS.index(head)+1)%len(HEADS)]
    scenario("model-reload","pair_style zero 4.8\n"+pair(args,work/"models"/(replacement+".mtp")),expected_head=replacement)
    spec=pair(args,model).splitlines()[0].removeprefix("pair_style ")
    style=spec.split()[0]
    overlay=(f"pair_style hybrid/overlay {spec} {spec}\n"
             f'pair_coeff * * {style} 1 "{model}" Al Ni\n'
             f'pair_coeff * * {style} 2 "{model}" Al Ni\n')
    scenario("overlay-double",overlay,factor=2.0)
    def rejection(name,text,message):
        label=f"lifecycle-{head}-r{rank}-{name}"
        script=work/("in."+label); script.write_text(text)
        execute(args,script,rank,work/(label+".log"),expected_error=message)
        results[name]={"rejected":True,"diagnostic":message}
    rejection("centroid",base+"compute cs all centroid/stress/atom NULL pair\n"
              f'dump reject all custom 1 "{work/(head+"-centroid.dump")}" id c_cs[1]\nrun 0\n',
              "Pair style does not support compute centroid/stress/atom")
    rejection("neighbor-exclusion",base+"neigh_modify exclude type 1 2\nrun 0\n","neighbor exclusions")
    rejection("neighbor-include",base+"group included type 1\natom_modify first included\n"
              "neigh_modify include included\nrun 0\n","neigh_modify include")
    molecular=setup(args).replace("atom_style atomic","atom_style bond")
    rejection("special-bonds",molecular+pair(args,model)+"special_bonds lj/coul 0 0 0\nrun 0\n","special_bonds")

    # A private model copy lets us test restart's external-file verification
    # without mutating the shared head fixtures or any user-provided model.
    restart_model=work/"models"/f"restart-{head}-r{rank}.mtp"
    original=model.read_bytes(); restart_model.write_bytes(original)
    label=f"lifecycle-{head}-r{rank}-restart"
    restart=work/(label+".bin")
    writer=work/("in."+label+"-write")
    writer_dump=work/(label+"-write.dump")
    writer.write_text(setup(args)+pair(args,restart_model)+lifecycle_monitor(writer_dump)+
                      f'run 0\nwrite_restart "{restart}"\n')
    writer_output=execute(args,writer,rank,work/(label+"-write.log"))
    results["restart-write"]=lifecycle_result(writer_output,writer_dump,head)
    read_rank=args.restart_ranks or args.ranks[(args.ranks.index(rank)+1)%len(args.ranks)]
    reader=work/("in."+label+"-read")
    reader_dump=work/(label+"-read.dump")
    reader.write_text(startup(args)+f'read_restart "{restart}"\n'+lifecycle_monitor(reader_dump)+"run 0\n")
    read_output=execute(args,reader,read_rank,work/(label+"-read.log"))
    results["restart-read"]=lifecycle_result(read_output,reader_dump,head)
    results["restart-read"]["ranks"]={"written":rank,"read":read_rank}
    restart_model.write_bytes(original.replace(b"scaling = 1\n",b"scaling = 1.125\n",1))
    try:
        execute(args,reader,read_rank,work/(label+"-changed-model.log"),
                expected_error="External MTP model differs from the model saved in restart")
        results["restart-changed-model"]={"rejected":True}
    finally:
        restart_model.write_bytes(original)
    return results


def dynamics(args,work,head,rank):
    model = work/"models"/(head+".mtp")
    base = setup(args)+pair(args,model)+"velocity all create 5.0 12345 mom yes rot no dist gaussian\n"
    monitor = "fix integrator all nve\nthermo 100\nthermo_style custom step time pe ke etotal\nthermo_modify format float %.17g\n"
    results = {}
    for dt in (.001,.0005):
        steps = round(args.duration/dt)
        label = f"nve-{head}-r{rank}-{dt}"
        script, dump = work/("in."+label), work/(label+".dump")
        # Restart remaps wrapped coordinates; unwrapped positions preserve the trajectory.
        final = f'write_dump all custom "{dump}" id xu yu zu vx vy vz modify sort id format float %.17g\n'
        script.write_text(base+monitor+f"timestep {dt}\nrun {steps}\n"+final)
        output = execute(args,script,rank,work/(label+".log"))
        rows = thermo(output)
        drift = max(abs(row["TotEng"]-rows[0]["TotEng"]) for row in rows)/len(TYPES)
        if drift > args.drift_tolerance: raise AssertionError(f"{label}: drift {drift} exceeds {args.drift_tolerance} eV/atom")
        results[str(dt)] = {"steps":steps,"drift_eV_per_atom":drift}
        if dt == .001:
            restart = work/(label+".restart")
            first = work/("in."+label+".first")
            first.write_text(base+monitor+f'timestep {dt}\nrun {steps//2}\nwrite_restart "{restart}"\n')
            execute(args,first,rank,work/(label+".first.log"))
            second, resumed = work/("in."+label+".second"), work/(label+".resumed.dump")
            prefix = startup(args)
            # Pair settings and model validation must be restored by read_restart.
            second.write_text(prefix+f'read_restart "{restart}"\n'+monitor+f"run {steps-steps//2}\n"+
                              f'write_dump all custom "{resumed}" id xu yu zu vx vy vz modify sort id format float %.17g\n')
            execute(args,second,rank,work/(label+".second.log"))
            close([v for row in dump_rows(dump) for v in row],[v for row in dump_rows(resumed) for v in row],label+" automatic restart",1e-8,1e-8)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", required=True,type=Path)
    parser.add_argument("--plugin",type=Path,help="Omit for standard lmp with embedded styles or the mtp_lammps runner")
    parser.add_argument("--oracle-plugin",type=Path,help="Optional native IMR-MLP oracle plugin; never linked by the candidate")
    parser.add_argument("--oracle-lmp",type=Path,help="Oracle LAMMPS binary; defaults to --lmp (may contain static mlip/native)")
    parser.add_argument("--lifecycle",action="store_true",help="Sorting, neighbor caches, type/count changes, reload, overlay, rejection and automatic restart")
    parser.add_argument("--restart-ranks",type=int,help="Explicit rank count for restart reads; default rotates through allocated --ranks")
    parser.add_argument("--device",choices=("cpu","cuda","hip"),default="cpu")
    parser.add_argument("--style",default="")
    parser.add_argument("--pair-options",default="")
    parser.add_argument("--ranks",nargs="+",type=int,default=[1,2,4])
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--heads",nargs="+",choices=HEADS,default=list(HEADS))
    parser.add_argument("--modes",nargs="+",choices=MODES,default=list(MODES))
    parser.add_argument("--mpiexec",default="mpiexec")
    parser.add_argument("--always-mpi",action="store_true")
    parser.add_argument("--timeout",type=float,default=900)
    parser.add_argument("--nve",action="store_true")
    parser.add_argument("--duration",type=float,default=10.,help="NVE duration in ps for each of two timesteps")
    parser.add_argument("--drift-tolerance",type=float,default=1e-6)
    parser.add_argument("--work",type=Path)
    args = parser.parse_args()
    if min(args.ranks)<1 or args.threads<1 or args.duration<=0: parser.error("ranks, threads and duration must be positive")
    if args.restart_ranks is not None and args.restart_ranks < 1: parser.error("restart-ranks must be positive")
    temporary = tempfile.TemporaryDirectory(prefix="mtp-standalone-") if args.work is None else None
    work = args.work.resolve() if args.work else Path(temporary.name)
    work.mkdir(parents=True,exist_ok=True)
    write_models(work/"models")
    results = {"device":args.device,"threads":args.threads,"duration_ps":args.duration if args.nve else None,"restart_mode":"automatic","pressure_conversion":PRESSURE,"drift_tolerance":args.drift_tolerance,"results":{},"oracle":{},"lifecycle":{},"nve":{}}
    oracle_args = None
    if args.oracle_plugin or args.oracle_lmp:
        oracle_args = copy.copy(args)
        oracle_args.lmp = args.oracle_lmp or args.lmp
        oracle_args.plugin = args.oracle_plugin
        oracle_args.device = "cpu"
        oracle_args.style = "mlip/native"
        oracle_args.pair_options = "device cpu precision fp64 pottype mtp_m"
    def checkpoint(): (work/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    results["status"] = "running"
    checkpoint()
    try:
        for head in args.heads:
            expected = oracle(head)
            for rank in args.ranks:
                for mode in args.modes:
                    actual = snapshot(args,work,head,rank,mode)
                    label = f"{head}-r{rank}-{mode}"
                    results["results"][label] = {key:close(values,expected[key],label+" "+key,1e-8,1e-7) for key,values in actual.items()}
                    if oracle_args:
                        native = snapshot(oracle_args,work,head,rank,mode,"oracle-")
                        results["oracle"][label] = {
                            key:{"formula_max_error":close(values,expected[key],"native formula "+label+" "+key,1e-8,1e-7),
                                 "candidate_max_error":close(actual[key],values,"candidate/native "+label+" "+key,1e-9,1e-8)}
                            for key,values in native.items()}
                    checkpoint()
                if args.lifecycle:
                    results["lifecycle"][f"{head}-r{rank}"] = lifecycle(args,work,head,rank)
                    checkpoint()
                if args.nve:
                    results["nve"][f"{head}-r{rank}"] = dynamics(args,work,head,rank)
                    checkpoint()
        results["status"] = "passed"
    except Exception as error:
        results["status"] = "failed"
        results["error"] = str(error)
        raise
    finally:
        checkpoint()
    print(json.dumps({"passed":len(results["results"]),"nve":len(results["nve"]),"lifecycle":len(results["lifecycle"]),"native_oracle":len(results["oracle"]),"work":str(work)},indent=2))

if __name__ == "__main__": main()
