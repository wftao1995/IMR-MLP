#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.

"""Time whole pair-style steps on replicated, stationary configurations."""
import argparse
from pathlib import Path
import re
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--reference-lmp',required=True,type=Path)
p.add_argument('--lmp',required=True,type=Path)
p.add_argument('--model',required=True,type=Path)
p.add_argument('--data',required=True,type=Path)
p.add_argument('--work',required=True,type=Path)
p.add_argument('--ranks',type=int,nargs='+',default=[1,4])
p.add_argument('--repeat',type=int,default=3)
a=p.parse_args();a.work=a.work.resolve();a.work.mkdir(parents=True,exist_ok=True)
print('ranks,backend,repeat,atoms,steps,ms_per_step',flush=True)
for ranks in a.ranks:
    for backend in ('reference','direct','recurrence'):
        for repeat in range(a.repeat):
            tag=f'{backend}-np{ranks}-{repeat}'
            phase='' if backend=='reference' else 'phase '+backend
            source=f'''units metal
atom_style atomic
boundary p p p
newton on
read_data {a.data.resolve()}
replicate 2 2 2
mass * 1
pair_style mlip/opt-les {a.model.resolve()} 0 1 {phase}
pair_coeff * *
neighbor 0.3 bin
thermo 100
thermo_style custom step pe press
timer full
run 2
run 10
'''
            input_file=a.work/(tag+'.in');input_file.write_text(source)
            log=a.work/(tag+'.log')
            executable=a.reference_lmp if backend=='reference' else a.lmp
            subprocess.run(['mpirun','-np',str(ranks),str(executable.resolve()),'-in',str(input_file),'-log',str(log),'-screen','none'],cwd=a.work,check=True,timeout=120)
            match=re.findall(r'Loop time of ([\d.e+-]+) on .* for (\d+) steps with (\d+) atoms',log.read_text())[-1]
            seconds,steps,atoms=match
            print(f'{ranks},{backend},{repeat},{atoms},{steps},{float(seconds)*1000/int(steps)}',flush=True)
