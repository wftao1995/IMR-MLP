#!/usr/bin/env python3
"""Alternate baseline/candidate LAMMPS runs and retain paired loop timings.

The caller provides scheduler allocation and frozen inputs. No remote submission,
GPU sharing decision, warmup auto-tuning, or hardware-dependent flags are hidden.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
import statistics
import subprocess


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): checksum.update(block)
    return checksum.hexdigest()


def resolve(value, root):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root/path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--work",required=True,type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root, work = args.config.resolve().parent, args.work.resolve()
    work.mkdir(parents=True,exist_ok=True)
    (work/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    repetitions = int(config.get("repetitions",5))
    if repetitions < 5: raise ValueError("Use at least five paired repetitions")
    result = {"repetitions":repetitions,"targets":{},"cases":{}}
    for name in ("baseline","candidate"):
        target = config[name]
        lmp = resolve(target["lmp"],root)
        result["targets"][name] = {"lmp":str(lmp),"sha256":digest(lmp),"env":target.get("env",{})}
        if target.get("plugin"):
            plugin = resolve(target["plugin"],root)
            result["targets"][name]["plugin"] = {"path":str(plugin),"sha256":digest(plugin)}
    for case in config["cases"]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+",case["name"]): raise ValueError("Unsafe case name")
        template = resolve(case["input"],root).read_text()
        if template.count("@PAIR@") != 1 or "@STEPS@" not in template or "@WARMUP@" not in template:
            raise ValueError("Input template requires one @PAIR@ and explicit @WARMUP@/@STEPS@")
        item = {"head":case.get("head"),"input_sha256":hashlib.sha256(template.encode()).hexdigest(),
                "files":{str(resolve(p,root)):digest(resolve(p,root)) for p in case.get("files",[])},
                "samples":[],"steps":int(case["steps"]),"warmup":int(case["warmup"])}
        if item["steps"]<=0 or item["warmup"]<0: raise ValueError("Invalid run length")
        for repeat in range(repetitions):
            sample = {}
            order = ("baseline","candidate") if repeat%2==0 else ("candidate","baseline")
            for name in order:
                target = config[name]
                # Equal-length input paths avoid role-dependent argv/allocation sizes.
                label = f'{case["name"]}-{repeat}-{name.ljust(9,"_")}'
                directory = work/label; directory.mkdir(exist_ok=True)
                plugin = f'plugin load "{resolve(target["plugin"],root)}"' if target.get("plugin") else ""
                text = template.replace("@PLUGIN@",plugin).replace("@PAIR@",target["pair"])
                text = text.replace("@STEPS@",str(item["steps"])).replace("@WARMUP@",str(item["warmup"]))
                for key,value in case.get("variables",{}).items(): text = text.replace("@"+key+"@",str(value))
                if re.search(r"@[A-Z_]+@",text): raise ValueError(f"Unresolved template variable in {label}")
                script = directory/"in.lammps"; script.write_text(text)
                command = list(target.get("launcher",[]))+[str(resolve(target["lmp"],root)),"-log","none","-in",str(script)]
                environment = os.environ.copy(); environment.update({str(k):str(v) for k,v in target.get("env",{}).items()})
                completed = subprocess.run(command,env=environment,cwd=resolve(case.get("cwd","."),root),text=True,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=float(config.get("timeout",3600)))
                (directory/"output.log").write_text(completed.stdout)
                if completed.returncode: raise RuntimeError(f"{label} failed: see {directory/'output.log'}")
                matches = re.findall(r"Loop time of ([0-9.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms",completed.stdout)
                if not matches: raise RuntimeError(f"No loop time in {label}")
                seconds, ranks, steps, atoms = matches[-1]
                if int(steps)!=item["steps"]: raise RuntimeError(f"Timed step count differs in {label}")
                seconds = float(seconds)
                if not math.isfinite(seconds) or seconds<=0: raise RuntimeError(f"Invalid timing in {label}")
                sample[name] = {"seconds":seconds,"seconds_per_step":seconds/int(steps),"ranks":int(ranks),"atoms":int(atoms),"command":command}
            if sample["baseline"]["atoms"]!=sample["candidate"]["atoms"]: raise RuntimeError("Atom counts differ")
            sample["speedup"] = sample["baseline"]["seconds"]/sample["candidate"]["seconds"]
            item["samples"].append(sample)
        ratios = [s["speedup"] for s in item["samples"]]
        item["paired_speedup"] = {"median":statistics.median(ratios),"min":min(ratios),"max":max(ratios)}
        # Resample complete pairs, never individual baseline/candidate timings.
        # A fixed seed makes the reported uncertainty reproducible from raw data.
        rng = random.Random(20261025)
        log_ratios = [math.log(r) for r in ratios]
        bootstrap = sorted(math.exp(statistics.mean(rng.choices(log_ratios,k=len(ratios)))) for _ in range(10000))
        item["paired_speedup"].update(geometric_mean=math.exp(statistics.mean(log_ratios)),
                                      bootstrap_95pct=[bootstrap[249],bootstrap[9749]])
        result["cases"][case["name"]] = item
        (work/"results.json").write_text(json.dumps(result,indent=2)+"\n")
        print(case["name"],json.dumps(item["paired_speedup"]),flush=True)
    print(json.dumps({name:value["paired_speedup"] for name,value in result["cases"].items()},indent=2))

if __name__ == "__main__": main()
