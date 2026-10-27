#!/usr/bin/env python3
"""Opt-in multi-path NEB search from a complete, labelled, unfolded path.

Defaults: one baseline + 4 seeds x 3 smooth modes x 2 signs; each trial has an
independent 200-step / 2001-band-EFS / 3600-second budget. No restart recovery,
MIC, IDPP, atom matching or image reversal. See README.md for interpretation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
from ase.io import read,write
from search_support import dump,sha,settings,execute,physical_cores,runtime_inventory
from path_compare import mask,Tolerances,classify


def metadata(path):
    values={};scales={};energies={}
    for line in path.read_text().splitlines():
        fields=line.split()
        if len(fields)==3 and fields[0]=='#':values[fields[1]]=fields[2]
        if len(fields)==4 and fields[:2]==['#','cell_scale']:scales[int(fields[2])]=float(fields[3])
        if len(fields)>=5 and fields[:2]==['#','image_energy']:energies[int(fields[2])]=float(fields[4])
    return values,scales,energies


def load_source(source,setting_file,frames):
    config=settings(setting_file);meta,scales,_=metadata(source/'neb.dat')
    count=len(np.loadtxt(source/'neb.dat',ndmin=2)) if frames=='final' else int(config['nimgs'])
    suffix='_final' if frames=='final' else ''
    files=[source/f'nebimage_{i}{suffix}.vasp' for i in range(count)]
    images=[read(p,format='vasp') for p in files]
    if len(meta.get('pbc',''))!=3 or set(meta['pbc'])-set('01'):raise ValueError('Source summary must record three pbc bits')
    for frame in images:frame.pbc=[c=='1' for c in meta['pbc']]
    native={k[4:]:v for k,v in config.items() if k.startswith('neb:')}
    vc=meta.get('vcneb',native.get('vcneb','false'))=='true';native['vcneb']=str(vc).lower()
    if config.get('next_step','false')=='true':config.pop('next_step')
    if any(k.startswith('al:') and v.lower() not in ('false','0','') for k,v in config.items()):
        raise ValueError('Online active learning is unsupported for path search')
    if native.get('external_stress','') or native.get('extstress',''):raise ValueError('External stress is unsupported for path search')
    reference=images[0].cell.array.copy()
    comparison={'vcneb':vc}
    if vc:
        if meta.get('joint_metric')!='vcneb-frobenius-v1':raise ValueError('Missing or unsupported VC coordinate metric metadata')
        for key in ['symm_strain','recast_atomic_force','apply_scal','cell_weight','pressure_GPa']:
            if key not in meta:raise ValueError('Missing VC source metadata: '+key)
        if meta['symm_strain']!='true' or meta['pbc']!='111':raise ValueError('Only fully periodic symmetric-strain VC paths are supported')
        if set(scales)!=set(range(6)):raise ValueError('All six source cell metric scales are required')
        reference=np.loadtxt(source/'neb_reference_cell.data')
        if reference.shape!=(3,3) or not np.isfinite(reference).all() or abs(np.linalg.det(reference))<1e-12:
            raise ValueError('Invalid original VC reference cell')
        native.update(symm_strain=meta['symm_strain'],recast_atomic_force=meta['recast_atomic_force'],
            restore_F_cholesky=meta.get('restore_F_cholesky',native.get('restore_F_cholesky','true')),
            cellweight=meta['cell_weight'],pressure=meta['pressure_GPa'],apply_scal='true')
        # Reconstruct exact effective metric, including the symmetric shear factor.
        weights=[scales[0]**2,scales[3]**2,scales[5]**2,scales[4]**2/2,scales[2]**2/2,scales[1]**2/2]
        native['ciival']=','.join(format(v,'.17g') for v in weights)
        comparison.update(recast_atomic_force=meta['recast_atomic_force']=='true',cell_scales=[scales[j] for j in range(6)])
    if any(not np.array_equal(mask(frame),mask(images[0])) for frame in images):raise ValueError('Inconsistent fixed atom mappings')
    if any(not mask(frame).all() for frame in images):native['selective_dynamics']='true'
    config.setdefault('index','index.map')
    for key in ('index','pot:potfile','pot:typemap'):
        if key in config:config[key]=str((setting_file.parent/config[key]).resolve())
    return config,native,comparison,reference,images,files


def candidates(seeds,modes,amplitudes,strain):
    result=[dict(name='baseline',seed=None,mode=None,sign=0,amplitude=0.,strain=0.)]
    for number,(seed,mode,amplitude,sign) in enumerate(itertools.product(seeds,modes,amplitudes,[-1,1])):
        result.append(dict(name=f'trial-{number:04d}',seed=seed,mode=mode,sign=sign,amplitude=amplitude,strain=strain))
    return result


def result_record(folder,code,elapsed,gtol,count):
    result=dict(status='failed',returncode=code,wall_seconds=elapsed,force_tolerance=gtol,
        band_evaluations=None,image_evaluations=None,optimizer_iterations=None)
    if code=='timeout':return dict(result,status='timeout',reason='wall-time budget')
    if code not in (0,2) or not (folder/'neb.dat').exists():return dict(result,reason='calculator/process/output failure')
    meta,_,energies=metadata(folder/'neb.dat')
    table=np.loadtxt(folder/'neb.dat',ndmin=2)
    curve=[energies[i] for i in range(count)] if set(energies)==set(range(count)) else table[:,2].tolist()
    result.update(status=meta['status'],residual=float(meta['max_neb_force']),barrier_eV=max(curve)-curve[0],energy_curve_eV=curve,
        band_evaluations=int(meta['force_evaluations']),image_evaluations=int(meta['image_force_evaluations']),
        optimizer_iterations=int(meta['optimizer_iterations']))
    result['stop_reason']=result['status']
    if result['status']=='numerical_failure':result['status']='failed'
    if not np.isfinite([result['residual'],*curve]).all():raise ValueError('Nonfinite result')
    final=[read(folder/f'nebimage_{i}_final.vasp',format='vasp') for i in range(count)]
    initial=[read(folder/f'image-{i}.vasp',format='vasp') for i in range(count)]
    for frame in [*final,*initial]:frame.pbc=[c=='1' for c in meta['pbc']]
    endpoint=max(float(abs(final[i].positions-initial[i].positions).max()) for i in (0,count-1))
    endpoint=max(endpoint,max(float(abs(final[i].cell.array-initial[i].cell.array).max()) for i in (0,count-1)))
    fixed=0.
    # Fixed-cell selective coordinates are Cartesian; VC masks constrain the
    # native atomic chart and allow affine motion with the cell.
    vc=meta.get('vcneb')=='true'
    reference=np.loadtxt(folder/'reference.data') if vc and meta.get('recast_atomic_force')=='true' else np.eye(3)
    for a,b in zip(initial,final):
        delta=(b.get_scaled_positions(wrap=False)-a.get_scaled_positions(wrap=False))@reference if vc else b.positions-a.positions
        if (~mask(a)).any():fixed=max(fixed,float(abs(delta[~mask(a)]).max()))
    result.update(endpoint_error=endpoint,fixed_coordinate_error=fixed)
    if endpoint>1e-8 or fixed>1e-8:result.update(status='failed',reason='Endpoint or fixed-coordinate invariant failed')
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Existing complete NEB run directory, including neb.dat')
    p.add_argument('--settings',type=Path,help='Original input settings; default SOURCE/neb.ini')
    p.add_argument('--frames',choices=['initial','final'],default='final')
    p.add_argument('--output',type=Path,required=True,help='New directory; existing data are never overwritten')
    p.add_argument('--binary',type=Path,required=True,help='imrmlp, or imrmlp-mpi when ranks > 1')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--modes',nargs='+',choices=['sine1','sine2','mixed3'],default=['sine1','sine2','mixed3'])
    p.add_argument('--amplitudes',type=float,nargs='+',default=[.01],help='Additional atomic displacement bounds in Angstrom')
    p.add_argument('--strain',type=float,default=.001,help='Relative cell deformation Frobenius bound')
    p.add_argument('--atoms',type=int,nargs='+',help='Zero-based original atom labels; optional local support')
    p.add_argument('--center',type=int,help='Zero-based center for a frozen middle-image radial neighborhood')
    p.add_argument('--radius',type=float,default=0.)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--evaluations',type=int,default=2001)
    p.add_argument('--timeout',type=float,default=3600)
    p.add_argument('--ranks',type=int,default=1);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--omp-mode',choices=['images','atoms'],default='images')
    p.add_argument('--profile',action='store_true',help='Enable per-trial MPI+OMP timing; disabled by default')
    p.add_argument('--prepare-only',action='store_true',help='Generate/freeze all candidates without running EFS')
    p.add_argument('--rms-tolerance',type=float,default=.01);p.add_argument('--max-tolerance',type=float,default=.03)
    p.add_argument('--cell-tolerance',type=float,default=1e-4);p.add_argument('--energy-tolerance',type=float,default=1e-4)
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if min(args.steps,args.evaluations,args.ranks,args.threads)<=0 or not np.isfinite(args.timeout) or args.timeout<=0:p.error('Positive budgets and rank/thread counts required')
    if any(s<0 or s>=2**64 for s in args.seeds):p.error('Seeds must fit unsigned 64-bit integers')
    if any(not np.isfinite(v) or v<0 for v in [*args.amplitudes,args.strain,args.radius,args.rms_tolerance,args.max_tolerance,args.cell_tolerance,args.energy_tolerance]):p.error('Finite nonnegative amplitudes and tolerances required')
    args.source=args.source.resolve();args.output=args.output.resolve();setting_file=(args.settings or args.source/'neb.ini').resolve()
    config,native,comparison,reference,images,source_files=load_source(args.source,setting_file,args.frames)
    count=len(images);tasks=candidates(args.seeds,args.modes,args.amplitudes,args.strain)
    print(f'Path search: {len(tasks)} candidates (including baseline), each <= {args.steps} steps, {args.evaluations} band EFS, {args.timeout:g} seconds',flush=True)
    cores=physical_cores()[:args.ranks*args.threads]
    if len(cores)<args.ranks*args.threads:p.error('Insufficient physical cores in the current affinity mask')
    # Import only after argument validation, so --help works without the extension.
    from imrmlp import Atoms
    from imrmlp.neb import make_path_seed
    from imrmlp import _imrmlp
    source_atoms=[Atoms.from_ase(a,read_results=False) for a in images]
    for a,b in zip(source_atoms,images):a.selective_dynamics_mask=mask(b)
    # Validate even a baseline-only or zero-amplitude preparation without EFS.
    make_path_seed(source_atoms,settings=native,reference_cell=reference,amplitude=0,strain=0)
    args.output.mkdir(parents=True,exist_ok=False)
    binary=args.output/'binaries'/args.binary.name;binary.parent.mkdir()
    shutil.copy2(args.binary.resolve(),binary)
    for library in args.binary.resolve().parent.glob('libimrmlp*.so*'):shutil.copy2(library,binary.parent/library.name)
    frozen=args.output/'source';frozen.mkdir()
    inputs=[setting_file,args.source/'neb.dat',*source_files]
    if comparison['vcneb']:inputs.append(args.source/'neb_reference_cell.data')
    for path in inputs:shutil.copy2(path,frozen/path.name)
    assets={}
    for key in ('index','pot:typemap','pot:potfile'):
        if key in config:
            path=Path(config[key]);assets[key]=dict(path=str(path),sha256=sha(path))
            destination=frozen/(key.replace(':','-')+'-'+path.name);shutil.copy2(path,destination);config[key]=str(destination)
    runtime=runtime_inventory({'neb':binary})
    import ase,spglib
    manifest=dict(schema_version=1,host=platform.node(),platform=platform.platform(),source=str(args.source),
        source_hashes={str(p):sha(p) for p in inputs},assets=assets,runner_sha256=sha(__file__),
        comparator_sha256=sha(Path(__file__).with_name('path_compare.py')),extension_sha256=sha(_imrmlp.__file__),
        hardware=json.loads(subprocess.check_output(['lscpu','-J'],text=True)),python=sys.version,
        package_versions=dict(numpy=np.__version__,ase=ase.__version__,spglib=spglib.__version__),
        binary_sha256=sha(binary),libraries={p.name:sha(p) for p in binary.parent.glob('*.so*')},runtime=runtime,
        cpu_affinity=cores,settings=native,comparison_metric=comparison,reference_cell=reference.tolist(),
        tasks=tasks,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        limits='Finite local search; no saddle certification or global path minimality')
    dump(args.output/'manifest.json',manifest)
    rank_wrapper=args.output/'rank_exec.py'
    rank_wrapper.write_text("import os,sys\ncpus=list(map(int,sys.argv[1].split(',')));n=int(sys.argv[2]);r=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])\nos.sched_setaffinity(0,cpus[r*n:(r+1)*n]);os.execv(sys.argv[3],sys.argv[3:])\n")
    records={};paths={}
    for task in tasks:
        name=task['name'];folder=args.output/name;folder.mkdir()
        trial_start=time.perf_counter();process_started=False
        try:
            if name=='baseline':generated=images;info=dict(status='generated',subspace='unperturbed',atomic_displacement_A=0.,cell_deformation=0.)
            else:
                seeded,info=make_path_seed(source_atoms,settings=native,reference_cell=reference,
                    mode=task['mode'],seed=task['seed'],sign=task['sign'],amplitude=task['amplitude'],strain=task['strain'],
                    atoms=args.atoms,center=args.center,radius=args.radius)
                generated=[]
                for old,new in zip(images,seeded):
                    a=old.copy();a.positions=new.positions;a.cell=new.cell;generated.append(a)
            dump(folder/'seed.json',dict(task,**info))
            if info['status']=='skipped':records[name]=info;continue
            for i,a in enumerate(generated):
                target=folder/f'image-{i}.vasp'
                if name=='baseline' or i in (0,count-1):shutil.copy2(source_files[i],target)
                else:write(target,a,format='vasp',direct=False,sort=False)
            trial={k:v for k,v in config.items() if not k.startswith('image:') and k not in ('next_step','summary','mepout','input-format')}
            trial.update({'neb:'+k:v for k,v in native.items()})
            trial.update({'nimgs':count,'num_threads':args.threads,'maxiteration':args.steps,'max_total_iter':args.steps,
                'neb:max_force_evaluations':args.evaluations,'neb:solver_policy':'stable-v1','neb:omp_mode':args.omp_mode,
                'neb:keep_symm':'false','neb:path_pbc':''.join('1' if v else '0' for v in images[0].pbc),'neb:check_mic':'false','neb:init_method':'linear','neb:perturb_amplitude':0,
                'neb:variableimage':'false','neb:write_traj':'false'})
            for key in ['neb:symmetry_release','neb:symmetry_operations','neb:symmetry_diagnostics','neb:profile','neb:reference_cell']:
                trial.pop(key,None)
            if comparison['vcneb']:
                np.savetxt(folder/'reference.data',reference,fmt='%.17g');trial['neb:reference_cell']='reference.data'
            if args.profile:trial['neb:profile']='profile.json'
            for i in range(count):trial[f'image:{i}']=f'image-{i}.vasp'
            (folder/'neb.ini').write_text(''.join(f'{k} {v}\n' for k,v in trial.items()))
            if args.prepare_only:records[name]=dict(status='prepared',seed=info);continue
            env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(binary.parent)+':'+env.get('LD_LIBRARY_PATH',''),
                OMP_NUM_THREADS=str(args.threads),OMP_DYNAMIC='FALSE',OMP_PROC_BIND='close',OMP_PLACES='cores',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            command=[str(binary),'neb','neb.ini','--num_threads',str(args.threads)]
            if args.ranks>1:command=['mpiexec','--host',f'localhost:{len(cores)}','--bind-to','none','-n',str(args.ranks),sys.executable,str(rank_wrapper),','.join(map(str,cores)),str(args.threads),*command]
            command=['taskset','-c',','.join(map(str,cores)),*command]
            dump(folder/'command.json',dict(argv=command,environment={k:env[k] for k in ['LD_LIBRARY_PATH','OMP_NUM_THREADS','OMP_DYNAMIC','OMP_PROC_BIND','OMP_PLACES','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
            process_started=True
            code,elapsed=execute(command,folder,env,args.timeout)
            records[name]=result_record(folder,code,elapsed,float(trial.get('gtol',native.get('gtol',native.get('neb_ftol',.01)))),count)
            records[name]['seed']=info
            if records[name]['status']=='converged':
                paths[name]=[read(folder/f'nebimage_{i}_final.vasp',format='vasp') for i in range(count)]
                for a in paths[name]:a.pbc=images[0].pbc
        except Exception as error:
            records[name]=dict(status='failed',reason=str(error))
        finally:
            records[name]['total_wall_seconds']=time.perf_counter()-trial_start
            for key in ('band_evaluations','image_evaluations','optimizer_iterations'):
                records[name].setdefault(key,None if process_started else 0)
            # Unknown failed-process counters remain null, never fabricated as zero.
            dump(folder/'result.json',records[name]);dump(args.output/'results.json',records)
            print(name,records[name]['status'],flush=True)
    tolerance=Tolerances(args.rms_tolerance,args.max_tolerance,args.cell_tolerance,args.energy_tolerance)
    classification=classify(records,paths,reference=reference,settings=comparison,tolerances=tolerance)
    runtime_after=runtime_inventory({'neb':binary})
    unchanged=runtime['neb']['libraries']==runtime_after['neb']['libraries']
    source_unchanged=all(sha(path)==digest for path,digest in manifest['source_hashes'].items())
    dump(args.output/'report.json',dict(schema_version=1,energy_quantity='enthalpy' if comparison['vcneb'] else 'energy',
        candidate_count=len(tasks),status_counts=dict(Counter(v['status'] for v in records.values())),
        energy_unit='eV_total',tolerances=asdict(tolerance),runtime_unchanged=unchanged,source_unchanged=source_unchanged,
        runtime_after=runtime_after,**classification,limitation=manifest['limits']))
    return int(not unchanged or not source_unchanged or any(v['status']=='failed' for v in records.values()))


if __name__=='__main__':
    raise SystemExit(main())
