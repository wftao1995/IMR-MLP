#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.

"""Compare short moving/changing-cell trajectories to a reference executable."""
import argparse
from pathlib import Path
import subprocess
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--reference-lmp',required=True,type=Path)
p.add_argument('--lmp',required=True,type=Path)
p.add_argument('--model',required=True,type=Path)
p.add_argument('--data',required=True,type=Path)
p.add_argument('--work',required=True,type=Path)
p.add_argument('--ranks',type=int,nargs='+',default=[1,4])
a=p.parse_args()
a.work=a.work.resolve();a.work.mkdir(parents=True,exist_ok=True)

def run(executable,tag,ranks,phase):
    dump=a.work/(tag+'.dump')
    source=f'''units metal
atom_style atomic
boundary p p p
newton on
read_data {a.data.resolve()}
mass * 1
pair_style mlip/opt-les {a.model.resolve()} 0 1 {phase}
pair_coeff * *
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
velocity all set 0.1 -0.03 0.02
fix integrate all nve
timestep 0.0001
compute ep all pe/atom
compute q all mlip/sog/latent
compute pp all pressure NULL pair
thermo 1
thermo_style custom step pe c_pp[1] c_pp[2] c_pp[3] c_pp[4] c_pp[5] c_pp[6] vol
thermo_modify format float %.17g
dump result all custom 1 {dump} id x y z fx fy fz c_ep c_q
dump_modify result sort id format float %.17g
run 8
change_box all xy delta 0.07 x scale 1.01 remap units box
displace_atoms all move 0.07 -0.04 0.02 units box
run 8
'''
    input_file=a.work/(tag+'.in');input_file.write_text(source)
    log=a.work/(tag+'.log')
    cmd=['mpirun','-np',str(ranks),str(executable.resolve()),'-in',str(input_file),'-log',str(log),'-screen','none']
    subprocess.run(cmd,cwd=a.work,check=True,timeout=120)
    lines=dump.read_text().splitlines()
    atoms=[]
    for i,line in enumerate(lines):
        if line.startswith('ITEM: ATOMS'):
            end=next((j for j in range(i+1,len(lines)) if lines[j].startswith('ITEM:')),len(lines))
            atoms.append(np.loadtxt(lines[i+1:end],ndmin=2))
    thermo=[];active=False
    for line in log.read_text().splitlines():
        if line.split()[:2]==['Step','PotEng']:active=True;continue
        if line.startswith('Loop time'):active=False
        if active:thermo.append(np.fromstring(line,sep=' '))
    return np.asarray(atoms),np.asarray(thermo)

reference=run(a.reference_lmp,'reference',1,'')
for ranks in a.ranks:
    for phase in ('direct','recurrence'):
        actual=run(a.lmp,f'{phase}-np{ranks}',ranks,'phase '+phase)
        errors=[]
        for ref,value in zip(reference,actual):
            if ref.shape!=value.shape:raise AssertionError((ref.shape,value.shape))
            np.testing.assert_allclose(value,ref,atol=2e-8,rtol=2e-10)
            errors.append(np.max(np.abs(value-ref)))
        print(ranks,phase,'max atom/thermo errors',errors,flush=True)
