#!/usr/bin/env python3
"""Compare standalone LAMMPS radial deployment with an independent native plugin.

IMR-MLP/gpmain are oracle/export tooling only, never candidate runtime dependencies.
Every model, input, dump, log and result is retained in --work.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from fixtures import HEADS, write_models
from run_regression import setup, execute, dump_rows, thermo, PRESSURE, MODES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gpmain','native-lmp','native-plugin','cpu-lmp','work'):
        parser.add_argument('--'+name,required=True,type=Path)
    for name in ('omp-lmp','gpu-lmp'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--gpu-device',choices=('cuda','hip'),default='cuda')
    parser.add_argument('--heads',nargs='+',choices=('classic',)+HEADS,default=('classic',)+HEADS)
    parser.add_argument('--modes',nargs='+',choices=MODES,default=('all',))
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=False)
    write_models(args.work/'fixtures')
    def block(name,size=2,extra='',lo=.5,hi=4.8):
        return f'radial_basis_type = {name}\nmin_dist = {lo}\nmax_dist = {hi}\nradial_basis_size = {size}\n'+extra
    variants={name:block(name,extra='rb_b = -0.7\n' if 'ChebyshevPoly' in name else '')
              for name in ('RBChebyshev','RBChebyCos','RBChebyshevPoly4','RBChebyshevPoly5','RBPolyB','RBPolyBCos')}
    variants['config-flip']=block('RBChebyshevConfigurable',extra=
        'basis_function_type = one_minus_chebyshev\ncutoff_type = smoothstep7\ncutoff_r1 = 1.2\ncutoff_r2 = 3.8\nshift_type = exponential_tanh\nshift_lambda = -0.7\nshift_scale_lambda = 0.8\n')
    variants['mixed-split']='radial_basis_split = 1\n'+''.join(f'pair {i//2} {i%2}\n'+
        block(name,lo=.2+.1*i,hi=4.2+.2*i,extra='rb_b = -0.7\n' if name.endswith('Poly4') else '')
        for i,name in enumerate(('RBPolyB','RBPolyBCos','RBChebyshev','RBChebyshevPoly4')))
    variants['single']=block('RBChebyshev',size=1)
    variants['PolyB8']=block('RBPolyB',size=8)
    targets=[('native',args.native_lmp,args.native_plugin,'mlip/native device cpu precision fp64 pottype mtp_m',1),
             ('cpu',args.cpu_lmp,None,'mlip/opt',1)]
    if args.omp_lmp:targets.append(('omp',args.omp_lmp,None,'mlip/opt/omp',2))
    if args.gpu_lmp:targets.append(('gpu',args.gpu_lmp,None,f'mlip/gpudirect device {args.gpu_device} precision fp64 gpu_id auto centers_per_chunk 3',1))
    results={'kind':'standalone radial vs native LAMMPS','cases':[]}
    def compare(actual,expected,label,atol=1e-9):
        if len(actual)!=len(expected):raise AssertionError(label+' size mismatch')
        error=max((abs(a-b) for a,b in zip(actual,expected)),default=0)
        if any(not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>atol+1e-9*abs(b) for a,b in zip(actual,expected)):
            raise AssertionError(f'{label}: max absolute error {error}')
        return error
    for head in args.heads:
        base=(args.work/'fixtures'/('li.mtp' if head=='classic' else head+'.mtp')).read_text()
        if head=='classic':
            base=re.sub(r'^potential_tag = .*$', 'potential_tag =',base,flags=re.M)
            base=re.sub(r'^species_coeffs = .*$', 'species_coeffs = {0.02,-0.01}',base,flags=re.M)
            base=re.sub(r'^moment_coeffs = .*$', 'moment_coeffs = {0.04,0.002}',base,flags=re.M)
            base=re.sub(r'^model_coeffs = .*\n?', '',base,flags=re.M)
            base=base.replace('alpha_scalar_moments = 3','alpha_scalar_moments = 2').replace('alpha_moment_mapping = {0, 1, 1}','alpha_moment_mapping = {0, 1}')
        for name,radial in variants.items():
            directory=args.work/head/name;directory.mkdir(parents=True)
            start,end=base.index('radial_basis_type'),base.index('radial_funcs_count')
            text=base[:start]+radial+base[end:]
            if name in ('single','PolyB8'):
                start,end=text.index('radial_coeffs'),text.index('alpha_moments_count')
                def coefficients(match):
                    values=match.group(1).split(',')
                    return '{'+(values[0] if name=='single' else ','.join(values+[str(.001*(-1)**j/(j+1)) for j in range(6)]))+'}'
                text=text[:start]+re.sub(r'\{([^{}]+)\}',coefficients,text[start:end])+text[end:]
            original=directory/'native.mtp';original.write_text(text)
            deployed=directory/'deploy.mtp'
            export=subprocess.run([str(args.gpmain.resolve()),'mtp-export-standalone','--input',str(original),'--output',str(deployed)],
                                  text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
            (directory/'export.log').write_text(export.stdout)
            if export.returncode:raise RuntimeError(str(directory)+' export failed: '+export.stdout)
            for mode in args.modes:
                snapshots={}
                for label,lmp,plugin,style,threads in targets:
                    options=SimpleNamespace(lmp=lmp,plugin=plugin,threads=threads,style='mlip/opt/omp' if threads>1 else '',
                                            always_mpi=False,timeout=120)
                    if label=='native' and head=='classic':style=style.replace('pottype mtp_m','pottype mtp')
                    script=directory/f'in.{label}.{mode}';dump=directory/f'{label}.{mode}.dump'
                    mapping="0 1" if head=="classic" else "Al Ni"
                    text=setup(options)+f'pair_style {style}\npair_coeff * * "{original if label=="native" else deployed}" {mapping}\n'
                    if mode in ('all','site_energy'):text+='compute se all pe/atom\n'
                    if mode in ('all','site_virial'):text+='compute sw all stress/atom NULL pair\n'
                    columns='id fx fy fz'+(' c_se' if mode in ('all','site_energy') else '')
                    if mode in ('all','site_virial'):columns+=' '+' '.join(f'c_sw[{i}]' for i in range(1,7))
                    text+=f'dump snap all custom 1 "{dump}" {columns}\ndump_modify snap sort id format float %.17g\nthermo 1\n'
                    fields='step'+(' pe' if mode in ('all','energy') else '')+(' pxx pyy pzz pxy pxz pyz vol' if mode in ('all','virial') else '')
                    text+=f'thermo_style custom {fields}\nthermo_modify norm no format float %.17g\nrun 0\n'
                    script.write_text(text)
                    output=execute(options,script,1,directory/f'{label}.{mode}.log')
                    snapshots[label]=(thermo(output)[-1],dump_rows(dump))
                reference=snapshots.pop('native')
                case={'head':head,'radial':name,'mode':mode,'model_sha256':hashlib.sha256(deployed.read_bytes()).hexdigest(),'targets':{}}
                for label,(th,rows) in snapshots.items():
                    ref_th,ref_rows=reference
                    errors={'forces':compare([x for row in rows for x in row[1:4]],[x for row in ref_rows for x in row[1:4]],label+' forces')}
                    if mode in ('all','energy'):errors['energy_per_atom']=compare([th['PotEng']/len(rows)],[ref_th['PotEng']/len(rows)],label+' E/N')
                    if mode in ('all','virial'):
                        keys=('Pxx','Pyy','Pzz','Pxy','Pxz','Pyz')
                        errors['global_virial']=compare([th[k]*th['Volume']/PRESSURE for k in keys],[ref_th[k]*ref_th['Volume']/PRESSURE for k in keys],label+' global W')
                    if mode in ('all','site_energy'):errors['site_energy']=compare([row[4] for row in rows],[row[4] for row in ref_rows],label+' site E')
                    if mode in ('all','site_virial'):
                        offset=5 if mode=='all' else 4
                        errors['site_virial']=compare([x/PRESSURE for row in rows for x in row[offset:]],[x/PRESSURE for row in ref_rows for x in row[offset:]],label+' site W')
                    case['targets'][label]=errors
                results['cases'].append(case)
                (args.work/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            print('PASS',head,name,flush=True)


if __name__=='__main__':main()
