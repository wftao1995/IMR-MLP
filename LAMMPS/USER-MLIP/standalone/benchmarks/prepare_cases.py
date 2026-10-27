#!/usr/bin/env python3
"""Create portable, deterministic static-force benchmark inputs; no simulation.

The FCC-derived two-species structures are synthetic performance fixtures, not
relaxed InSe crystals. Optional real extxyz input contributes its first frame.
Model files are copied unchanged and hashed, with their provenance retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shlex
import shutil


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def write_data(path,box,atoms,title):
    lx,ly,lz,xy,xz,yz=box
    text=(f'{title}\n\n{len(atoms)} atoms\n2 atom types\n\n'
          f'0 {lx:.17g} xlo xhi\n0 {ly:.17g} ylo yhi\n0 {lz:.17g} zlo zhi\n'
          f'{xy:.17g} {xz:.17g} {yz:.17g} xy xz yz\n\nMasses\n\n'
          '1 114.818 # In\n2 78.971 # Se\n\nAtoms # atomic\n\n')
    for i,(kind,xyz) in enumerate(atoms,1):
        text+=f'{i} {kind} '+ ' '.join(f'{value:.17g}' for value in xyz)+'\n'
    path.write_text(text)


def synthetic(path,count,spacing,seed):
    cells=round((count/4)**(1/3))
    if 4*cells**3!=count: raise ValueError('Synthetic sizes must equal 4*n^3')
    rng=random.Random(seed+count); length=cells*spacing; atoms=[]
    basis=((0,0,0),(0,.5,.5),(.5,0,.5),(.5,.5,0))
    for i in range(cells):
        for j in range(cells):
            for k in range(cells):
                for b,s in enumerate(basis):
                    xyz=[((u+v+rng.uniform(-.02,.02))*spacing)%length for u,v in zip((i,j,k),s)]
                    atoms.append((1+b%2,xyz))
    write_data(path,(length,length,length,0,0,0),atoms,'Synthetic binary FCC-derived static performance fixture')
    return {'atoms':count,'kind':'synthetic FCC-derived, unrelaxed','spacing_A':spacing,
            'jitter_fraction':.02,'seed':seed+count,'sha256':digest(path)}


def real_frame(source,path):
    with source.open() as stream:
        first=stream.readline(); count=int(first)
        header=stream.readline(); lines=[stream.readline() for _ in range(count)]
    fields=dict(item.split('=',1) for item in shlex.split(header) if '=' in item)
    lattice=list(map(float,fields['Lattice'].split()))
    if len(lattice)!=9 or fields.get('pbc','T T T').split()!=['T','T','T']:
        raise ValueError('Real fixture requires a periodic 3D lattice')
    a,b,c=[lattice[i:i+3] for i in (0,3,6)]
    lx=math.sqrt(dot(a,a)); ex=[v/lx for v in a]
    xy=dot(b,ex); by=[b[i]-xy*ex[i] for i in range(3)]
    ly=math.sqrt(dot(by,by)); ey=[v/ly for v in by]; ez=cross(ex,ey)
    xz,yz,lz=dot(c,ex),dot(c,ey),dot(c,ez)
    if lz<=0: raise ValueError('Real lattice must have positive handedness')
    properties=fields['Properties'].split(':'); offsets={}; offset=0
    for i in range(0,len(properties),3):
        offsets[properties[i]]=offset; offset+=int(properties[i+2])
    atoms=[]
    for line in lines:
        values=line.split(); kind={'In':1,'Se':2}[values[offsets['species']]]
        original=list(map(float,values[offsets['pos']:offsets['pos']+3]))
        x,y,z=[dot(original,axis) for axis in (ex,ey,ez)]
        sz=z/lz; sy=(y-yz*sz)/ly; sx=(x-xy*sy-xz*sz)/lx
        sx,sy,sz=[value%1 for value in (sx,sy,sz)]
        atoms.append((kind,[lx*sx+xy*sy+xz*sz,ly*sy+yz*sz,lz*sz]))
    write_data(path,(lx,ly,lz,xy,xz,yz),atoms,'Real InSe pilot training frame 0, rigidly rotated to restricted triclinic')
    return {'atoms':count,'kind':'real InSe training frame','frame':0,'source':str(source.resolve()),
            'source_sha256':digest(source),'source_frame_sha256':hashlib.sha256((first+header+''.join(lines)).encode()).hexdigest(),
            'rotation_rows':[ex,ey,ez],'sha256':digest(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--models',required=True,type=Path)
    parser.add_argument('--real-xyz',type=Path)
    parser.add_argument('--sizes',type=int,nargs='+',default=[256,2048,16384])
    parser.add_argument('--spacing',type=float,default=4.8)
    parser.add_argument('--seed',type=int,default=20261023)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--steps',type=int,default=100)
    args=parser.parse_args(); out=args.output.resolve()
    if out.exists(): parser.error('Refusing to overwrite output directory')
    (out/'data').mkdir(parents=True); (out/'models').mkdir()
    metadata={'format':'mtp-static-benchmark-inputs-v1','generator_sha256':digest(Path(__file__)),
              'mapping':{'LAMMPS type 1':'model species 0 (In)','LAMMPS type 2':'model species 1 (Se)'},
              'structures':{},'models':{},'note':'Static force timings; five-step pilot models are not converged production potentials.'}
    for count in args.sizes:
        relative=f'data/{count}.data'
        metadata['structures'][relative]=synthetic(out/relative,count,args.spacing,args.seed)
    if args.real_xyz: metadata['structures']['data/real-inse.data']=real_frame(args.real_xyz,out/'data/real-inse.data')
    model_paths=sorted(args.models.glob('*.mtp'))
    if not model_paths: parser.error('No *.mtp models found')
    for source in model_paths:
        destination=out/'models'/source.name; shutil.copy2(source,destination)
        text=source.read_text(); tag=re.search(r'^#\s*mtp_model\s*=\s*(\S+)',text,re.M)
        dimensions={key:int(re.search(r'^'+key+r'\s*=\s*(\d+)',text,re.M)[1]) for key in
                    ('species_count','alpha_moments_count','alpha_index_basic_count','alpha_index_times_count','alpha_scalar_moments')}
        metadata['models']['models/'+source.name]={'sha256':digest(destination),'head':tag[1] if tag else 'unspecified',**dimensions}
    if (args.models/'manifest.json').is_file(): shutil.copy2(args.models/'manifest.json',out/'model-provenance.json')
    template='''@PLUGIN@
units metal
atom_style atomic
boundary p p p
newton on
read_data @DATA@
@PAIR@
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
# No time integration: positions stay fixed, so pilot stability does not affect timing.
thermo @STEPS@
thermo_style custom step
run @WARMUP@ post no
reset_timestep 0
run @STEPS@ post no
'''
    (out/'in.static').write_text(template)
    cases=[]
    for model,record in metadata['models'].items():
        for count in args.sizes:
            if record['alpha_scalar_moments']>1000 and count!=min(args.sizes): continue
            data=f'data/{count}.data'
            cases.append({'name':Path(model).stem+'-'+str(count),'head':record['head'],'input':'in.static',
                          'warmup':args.warmup,'steps':args.steps,'files':[model,data],
                          'variables':{'MODEL':model,'DATA':data}})
    for backend in ('cpu','cuda'):
        baseline=('pair_style mlip/native device cpu precision fp64 pottype mtp_m\npair_coeff * * "@MODEL@" 0 1' if backend=='cpu'
                  else 'pair_style mlip/gpudirect "@MODEL@" 0 1 device cuda precision fp64 gpu_id auto\npair_coeff * *')
        candidate=('pair_style mlip/opt "@MODEL@" 0 1\npair_coeff * *' if backend=='cpu' else baseline)
        targets={name:{'lmp':'/SET/PATH/TO/lmp','plugin':f'/SET/PATH/TO/{name}.so','pair':pair,
                       'env':{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}}
                 for name,pair in (('baseline',baseline),('candidate',candidate))}
        (out/f'config.{backend}.example.json').write_text(json.dumps({'repetitions':5,**targets,'cases':cases},indent=2)+'\n')
    metadata['template_sha256']=digest(out/'in.static')
    (out/'input-manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'output':str(out),'structures':len(metadata['structures']),'models':len(metadata['models']),'cases':len(cases)},indent=2))


if __name__=='__main__': main()
