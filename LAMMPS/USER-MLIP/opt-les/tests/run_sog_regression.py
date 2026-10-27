#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: GPL-2.0-only
# Adapted for the IMR-MLP public source layout, 2026-09-22.

"""Compare an opt-les test executable against generate_reference.cpp output."""
import argparse
import re
from pathlib import Path
import subprocess
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--lmp',required=True)
parser.add_argument('--cases',required=True,type=Path)
parser.add_argument('--ranks',type=int,nargs='+',default=[1,2,4])
parser.add_argument('--phase',choices=['direct','recurrence'],default='recurrence')
parser.add_argument('--skip-rejections',action='store_true')
parser.add_argument('--reload',action='store_true',
                    help='Recreate the pair style and switch kernels/channels before checking EFS')
args=parser.parse_args()
root=args.cases.resolve()
worst=np.zeros(5)
for name in (root/'cases.txt').read_text().split():
    ref=(root/(name+'.ref')).read_text().splitlines()
    energy=float(ref[0]);virial=np.fromstring(ref[1],sep=' ')
    peratom=np.loadtxt(ref[2:],ndmin=2)
    for ranks in args.ranks:
        tag=f'{name}-np{ranks}'
        compute='mlip/les/charge' if name.endswith('m4') else 'mlip/sog/latent'
        channels=peratom.shape[1]-5
        q_columns='c_q' if channels==1 else ' '.join(f'c_q[{c+1}]' for c in range(channels))
        source=f'''units metal
atom_style atomic
boundary p p p
newton on
read_data {root/name}.data
mass * 1
pair_style mlip/opt-les phase {args.phase}
pair_coeff * * {root/name}.mtp Al Ni
neighbor 0.3 bin
compute ep all pe/atom
compute st all stress/atom NULL pair
compute q all {compute}
compute pp all pressure NULL pair
compute es all reduce sum c_ep
compute ss all reduce sum c_st[1] c_st[2] c_st[3] c_st[4] c_st[5] c_st[6]
thermo 1
thermo_style custom step pe c_pp[1] c_pp[2] c_pp[3] c_pp[4] c_pp[5] c_pp[6] vol c_es c_ss[1] c_ss[2] c_ss[3] c_ss[4] c_ss[5] c_ss[6]
thermo_modify format float %.17g
dump result all custom 1 {root/tag}.dump id fx fy fz c_ep {q_columns}
dump_modify result sort id format float %.17g
run 0
'''
        if args.reload:
            # Metadata-backed styles require a new pair object when loading a
            # model again. Keep atom counts fixed and recreate dependent computes.
            alternate=name[:-1]+('3' if name.endswith('m4') else '4')
            source+=f'''undump result
uncompute q
pair_style none
pair_style mlip/opt-les {root/name}.mtp 0 1 phase {args.phase}
pair_coeff * *
run 0
pair_style none
pair_style mlip/opt-les {root/alternate}.mtp 0 1 phase {args.phase}
pair_coeff * *
run 0
pair_style none
pair_style mlip/opt-les {root/name}.mtp 0 1 phase {args.phase}
pair_coeff * *
compute q all {compute}
dump result all custom 1 {root/tag}.dump id fx fy fz c_ep {q_columns}
dump_modify result sort id format float %.17g
run 0
'''
        (root/(tag+'.in')).write_text(source)
        command=([str(Path(args.lmp).resolve())] if ranks==1 else ['mpirun','-np',str(ranks),str(Path(args.lmp).resolve())])
        command+=['-in',str(root/(tag+'.in')),'-log',str(root/(tag+'.log')),'-screen','none']
        subprocess.run(command,cwd=root,check=True)
        lines=(root/(tag+'.log')).read_text().splitlines()
        if name.startswith('g3') and ranks>1:
            ghosts=next(line for line in lines if line.startswith('Nghost:'))
            if float(ghosts.split()[1])!=0:
                raise AssertionError(f'{tag}: expected no ghost atoms in the empty-domain case')
        start=max(i for i,l in enumerate(lines) if l.split()[:2]==['Step','PotEng'])
        thermo=np.fromstring(lines[start+1],sep=' ')
        dump=(root/(tag+'.dump')).read_text().splitlines();start=next(i for i,l in enumerate(dump) if l.startswith('ITEM: ATOMS'))
        actual=np.loadtxt(dump[start+1:],ndmin=2)
        w=thermo[2:8]*thermo[8]/1.6021765e6
        atom_w=-thermo[10:16]/1.6021765e6
        errors=np.array([abs(thermo[1]-energy),np.max(abs(actual[:,1:4]-peratom[:,1:4])),np.max(abs(w-virial)),np.max(abs(actual[:,4]-peratom[:,4])),np.max(abs(actual[:,5:]-peratom[:,5:]))])
        worst=np.maximum(worst,errors)
        if np.max(errors)>2e-8 or np.max(abs(atom_w-virial))>2e-8 or abs(thermo[9]-energy)>2e-8:
            raise AssertionError(f'{tag}: E,F,W,atomic E,q errors={errors}; atom virial={atom_w}, reference={virial}, global={w}')
        print(tag,'passed',flush=True)
print('max E,F,W,atomic E,q errors:',worst)

if args.skip_rejections:raise SystemExit(0)

# Exercise the public loader and observable guard, including collective failure.
valid=(root/'g0m3.mtp').read_text()
failures={
    'duplicate':(valid+'\nsog_terms = 2\n','Duplicate SOG field'),
    'channels':(re.sub(r'sog_channels = \d+', 'sog_channels = 0', valid),'Unsupported SOG version or channel count'),
    'legacy':(valid+'\nremove_self_interaction = 1\n','rejects legacy'),
    'version':(re.sub(r'les_version = \d+', 'les_version = 2', valid),'Unsupported SOG version or channel count'),
    'missing':('\n'.join(line for line in valid.splitlines() if not line.startswith('sog_zero_mode')),'Missing SOG field'),
    'latent':('\n'.join('les_species_coeffs = {0.1}' if line.startswith('les_species_coeffs') else line for line in valid.splitlines()),'Malformed LES latent coefficient block'),
    'nonlinear':(valid+'\nmodel_coeffs = {1}\n','linear MTP short-range head only'),
    'charge':(valid,'Requested LES charge or SOG latent output is unavailable'),
}
if int(re.search(r'sog_channels = (\d+)', valid)[1])>1:
    failures['layout']=(valid,'Recreate SOG latent compute after changing channel count')
for name,(model,message) in failures.items():
    tag='reject-'+name
    (root/(tag+'.mtp')).write_text(model)
    source=(root/f'g0m3-np{args.ranks[0]}.in').read_text().replace(str(root/'g0m3.mtp'),str(root/(tag+'.mtp')))
    if name=='charge':
        source=source.replace('mlip/sog/latent','mlip/les/charge')
        source=re.sub(r'c_q\[\d+\]', 'c_q', source)
    if name=='layout':
        source+=f'undump result\npair_style none\npair_style mlip/opt-les {root}/g0m4.mtp 0 1\npair_coeff * *\nrun 0\n'
    (root/(tag+'.in')).write_text(source)
    command=['mpirun','-np','2',str(Path(args.lmp).resolve()),'-in',str(root/(tag+'.in')),'-log',str(root/(tag+'.log')),'-screen','none']
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=60)
    log=(root/(tag+'.log')).read_text()
    if result.returncode==0 or message not in log:
        raise AssertionError(f'{tag}: expected {message!r}, got {result.returncode}: {log[-1000:]}')
    print(tag,'passed',flush=True)
