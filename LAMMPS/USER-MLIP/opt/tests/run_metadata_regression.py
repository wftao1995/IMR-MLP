#!/usr/bin/env python3
# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT
"""Exercise both MTP declaration syntaxes and numeric/name mappings in real MPI LAMMPS."""
import argparse
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import tempfile

parser=argparse.ArgumentParser()
parser.add_argument('--lmp',required=True,type=Path)
parser.add_argument('--work',type=Path)
parser.add_argument('--mpirun',default='mpirun')
args=parser.parse_args()
work=args.work.resolve() if args.work else Path(tempfile.mkdtemp(prefix='mtp-metadata-lammps-'))
work.mkdir(parents=True,exist_ok=True)
model='''MTP
version = 1.1.0
potential_name = synthetic_metadata_test
species_count = 2
potential_tag = fixture-only
radial_basis_type = RBChebyshev
min_dist = 0
max_dist = 5
radial_basis_size = 2
radial_funcs_count = 1
radial_coeffs
0-0
{0.25,0.125}
0-1
{0.3,0.15}
1-0
{0.35,0.175}
1-1
{0.4,0.2}
alpha_moments_count = 1
alpha_index_basic_count = 1
alpha_index_basic = {{0,0,0,0}}
alpha_index_times_count = 0
alpha_index_times = {}
alpha_scalar_moments = 1
alpha_moment_mapping = {0}
species_coeffs = {0.125,0.25}
moment_coeffs = {0.75}
'''
metadata={'model_type':'mtp','elements':['Al','Ni'],'layout':'mtp_v1'}
(work/'legacy.mtp').write_text(model)
(work/'old-tagged.mtp').write_text(model.replace('potential_tag = fixture-only','potential_tag = fixture-only | imrmlp_meta_v1:'+json.dumps(metadata,separators=(',',':'))))
(work/'tagged.mtp').write_text(model.replace('potential_tag = fixture-only','potential_tag = '+json.dumps({'schema':'gpml.mtp.metadata','version':1,'type':'MTP','model':'linear','elements':['Al','Ni']},separators=(',',':'))))
(work/'structure.data').write_text('''Synthetic interface test, not a material model

3 atoms
2 atom types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 58.6934
2 26.9815385

Atoms # atomic

1 1 7 7 7
2 2 10.05 7.2 7.3
3 1 6 10 7.4
''')
env={**os.environ,'OMP_NUM_THREADS':'1'}
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
checks=[]
def run(style,declaration,ranks,label,valid=True,run_zero=True):
    script=('package omp 1\n' if style.endswith('/omp') else '')+'units metal\natom_style atomic\nboundary p p p\nnewton on\nread_data structure.data\n'+declaration+'\ncompute vir all pressure NULL pair\nthermo_style custom step pe c_vir[1] c_vir[2] c_vir[3] c_vir[4] c_vir[5] c_vir[6]\nthermo_modify format float %.17g\nrun 0\nwrite_dump all custom '+label+'.forces id type fx fy fz modify sort id format float %.17g\n'
    if not run_zero:
        script=script.split("\ncompute vir",1)[0]+"\n"
    path=work/(label+'.in');path.write_text(script)
    result=subprocess.run([args.mpirun,'-np',str(ranks),str(args.lmp.resolve()),'-in',path.name,'-log','none'],cwd=work,env=env,capture_output=True,text=True,timeout=30)
    (work/(label+'.log')).write_text(result.stdout+result.stderr)
    if not valid:
        assert result.returncode!=0,label+' unexpectedly accepted'
        checks.append(label+' rejected');return None
    assert result.returncode==0,result.stdout+result.stderr
    rows=[line.split() for line in result.stdout.splitlines() if re.match(r'^\s*0\s+[-+0-9.]',line) and len(line.split())==8]
    assert len(rows)==1,rows
    force_rows=(work/(label+'.forces')).read_text().split('ITEM: ATOMS')[1].splitlines()[1:]
    forces=[float(x) for row in force_rows if row.strip() for x in row.split()[2:]]
    checks.append(label+' passed')
    return [float(x) for x in rows[0][1:]]+forces

def near(left,right):
    assert len(left)==len(right)
    for a,b in zip(left,right):assert abs(a-b)<=1e-10+1e-10*max(abs(a),abs(b)),(a,b)

reference=None
for style in ['mlip','mlip/opt','mlip/opt/omp','mlip/transport','mlip/transport/omp']:
    for ranks in [1,2,4]:
        declarations=[f'pair_style {style} legacy.mtp 1 0\npair_coeff * *',f'pair_style {style}\npair_coeff * * legacy.mtp 1 0',f'pair_style {style} tagged.mtp Ni Al\npair_coeff * *',f'pair_style {style}\npair_coeff * * tagged.mtp Ni Al',f'pair_style {style}\npair_coeff * * old-tagged.mtp Ni Al']
        for i,declaration in enumerate(declarations):
            result=run(style,declaration,ranks,style.replace('/','-')+f'-{ranks}-{i}')
            if reference is None:reference=result
            else:near(reference,result)

for i,mapping in enumerate(['Ni Fe','0 Ni','Ni','2 0','0.5 1']):
    run('mlip/opt','pair_style mlip/opt\npair_coeff * * tagged.mtp '+mapping,2,f'invalid-{i}',False)
run('mlip/opt','pair_style mlip/opt\npair_coeff * * legacy.mtp Ni Al',2,'missing-elements',False)
# Validate parser state at command time, before run/init_style can hide a gap.
for style in ['mlip/opt','mlip/opt/omp']:
    label=style.replace('/','-')
    run(style,f'pair_style {style}\npair_coeff * * tagged.mtp Ni',2,
        label+'-incomplete-before-run',False,run_zero=False)
    run(style,f'pair_style {style} tagged.mtp Ni Al\npair_coeff * *\npair_style {style}',2,
        label+'-repeat-settings',False,run_zero=False)
    actual=run(style,f'pair_style {style} tagged.mtp Ni Al\npair_coeff * *\npair_style none\npair_style {style}\npair_coeff * * tagged.mtp Ni Al',2,
               label+'-recreate')
    near(reference,actual)
spline=Path(__file__).resolve().parents[2]/'transport/tests/simple_spline.mtp'
(work/'spline.mtp').write_text(spline.read_text())
run('mlip/spline','pair_style mlip/spline spline.mtp 0 0\npair_coeff * *\npair_style mlip/spline',2,
    'spline-repeat-settings',False,run_zero=False)
run('mlip/spline','pair_style mlip/spline\npair_coeff * * spline.mtp 0',2,
    'spline-incomplete-before-run',False,run_zero=False)
report={'checks':checks,'numeric_name_and_syntax_parity':True,'ranks':[1,2,4],'scientificResult':False}
(work/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'work':str(work),'scientificResult':False}))
