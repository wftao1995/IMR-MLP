"""Labelled path comparison and bounded multi-path CLI orchestration."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from ase import Atoms
from ase.io import read,write
from test_neb_parallel import binary,inputs,run

ROOT=Path(__file__).resolve().parents[2]
TOOLS=ROOT/'scripts/benchmark/neb'
sys.path.insert(0,str(TOOLS))
import path_compare as compare
import search_paths as search


def band(n=5,branch=0.):
    result=[]
    for s in np.linspace(0,1,n):
        result.append(Atoms('HHeLi',positions=[[1+s,1+branch*np.sin(np.pi*s),1],[3,2,1],[4,4,3]],cell=[9,10,11],pbc=True))
    return result


def test_comparison_is_sampling_origin_and_basis_invariant():
    a=band();b=band(13)
    u=np.array([[1,2,0],[0,1,0],[0,0,1]])
    for frame in b:frame.positions+=[.7,-.4,.3];frame.cell=u@frame.cell.array
    assert compare.compare_paths(a,b)['equivalent']
    assert np.array_equal(a[0].positions,band()[0].positions)


@pytest.mark.parametrize('change',['middle_reorder','global_reorder','winding','branch','reversal'])
def test_different_labelled_channels_are_not_duplicates(change):
    a=band();b=band()
    if change=='middle_reorder':b[2]=b[2][[1,0,2]]
    if change=='global_reorder':b=[frame[[1,0,2]] for frame in b]
    if change=='winding':
        for i,frame in enumerate(b):frame.positions[0]+=i*frame.cell.array[0]
    if change=='branch':b=band(branch=.2)
    if change=='reversal':b.reverse()
    assert not compare.compare_paths(a,b)['equivalent']


@pytest.mark.parametrize('global_permutation',[False,True])
def test_identical_species_do_not_enable_automatic_atom_matching(global_permutation):
    a=[Atoms('H2',positions=[[1+s,2,2],[5-s,2,2]],cell=[8,9,10],pbc=True) for s in np.linspace(0,1,5)]
    b=[frame.copy() for frame in a]
    for i in (range(5) if global_permutation else [2]):b[i]=b[i][[1,0]]
    assert not compare.compare_paths(a,b)['equivalent']


def test_equal_barriers_do_not_merge_different_paths_and_unconverged_paths_are_not_ranked():
    paths={'baseline':band(),'same':band(9),'branch':band(branch=.4)}
    records={k:dict(status='converged',barrier_eV=1.,residual=.001,force_tolerance=.01) for k in paths}
    records['unfinished']=dict(status='max_evaluations',barrier_eV=.1)
    result=compare.classify(records,paths)
    assert len(result['duplicate_groups'])==2
    assert 'unfinished' not in result['ranking']
    assert result['incomplete']==['unfinished']


def test_geometric_equivalence_records_one_global_operation_without_merging_labels():
    pytest.importorskip('spglib')
    a=[];b=[]
    for s in np.linspace(0,1,5):
        # Mirror-related bowed paths share the same symmetric endpoints.
        first=Atoms('H2',positions=[[1+s,2+.3*np.sin(np.pi*s),2],[5-s,2-.3*np.sin(np.pi*s),2]],cell=[6,7,8],pbc=True)
        second=first.copy();second.positions[:,1]=4-second.positions[:,1]
        a.append(first);b.append(second)
    assert not compare.compare_paths(a,b)['equivalent']
    relation=compare.geometry_relation(a,b)
    assert relation is not None and sorted(relation['permutation'])==[0,1]
    records={k:dict(status='converged',barrier_eV=1.,residual=.001,force_tolerance=.01) for k in ['a','b']}
    result=compare.classify(records,{'a':a,'b':b})
    assert len(result['duplicate_groups'])==2 and len(result['geometric_relations'])==1


@pytest.fixture
def source(tmp_path):
    work=tmp_path/'source';inputs(work,vc=True)
    result=run(work)
    assert result.returncode in (0,2),result.stdout+result.stderr
    return work


def arguments(source,output):
    return ['--source',str(source),'--output',str(output),'--binary',str(binary()),
            '--seeds','0','--modes','sine1','--steps','2','--evaluations','3']


def test_search_default_matrix():
    assert len(search.candidates([0,1,2,3],['sine1','sine2','mixed3'],[.01],.001))==25


def test_vc_comparison_keeps_a_constant_lattice_gauge_and_common_origin():
    a=band();b=[]
    for i,frame in enumerate(a):
        frame.set_cell(frame.cell.array*(1+.02*i),scale_atoms=True)
        other=frame.copy();other.positions+=np.array([[1,0,0],[0,-1,0],[0,0,1]])@other.cell.array+[.2,.3,-.7]
        b.append(other)
    metric=dict(vcneb=True,recast_atomic_force=True,cell_scales=[10]*6)
    assert compare.compare_paths(a,b,reference=a[0].cell.array,settings=metric)['equivalent']


def test_vc_fixed_coordinate_audit_uses_reference_cartesian_chart(tmp_path):
    from ase.constraints import FixScaled
    cell=np.array([[5.,0,0],[1.,6,0],[0,0,7.]])
    np.savetxt(tmp_path/'reference.data',cell)
    for i in range(3):
        a=Atoms('H',positions=[[1.,2.,3.]],cell=cell,pbc=True)
        a.set_constraint(FixScaled(0,mask=[True,False,False]))
        write(tmp_path/f'image-{i}.vasp',a,format='vasp')
        if i==1:a.positions[0,1]+=.2
        write(tmp_path/f'nebimage_{i}_final.vasp',a,format='vasp')
    (tmp_path/'neb.dat').write_text('# vcneb true\n# recast_atomic_force true\n# pbc 111\n# status converged\n# max_neb_force 0\n# force_evaluations 1\n# image_force_evaluations 3\n# optimizer_iterations 0\n0 0 0\n1 1 1\n2 2 0\n')
    result=search.result_record(tmp_path,0,.1,.01,3)
    assert result['status']=='converged' and result['fixed_coordinate_error']<1e-12


def test_prepare_and_run_preserve_sources_and_independent_budgets(source,tmp_path):
    from imrmlp.neb import make_path_seed
    original={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    output=tmp_path/'search'
    assert search.main(arguments(source,output))==0
    results=json.loads((output/'results.json').read_text())
    assert len(results)==3
    for row in results.values():
        assert row['status'] in ('converged','max_iterations','max_evaluations','stalled')
        assert row['band_evaluations']<=3
        assert row['endpoint_error']<1e-10
    assert all((source/name).read_bytes()==content for name,content in original.items())
    assert not (output/'baseline/profile.json').exists()
    assert not any((output/k/'release').exists() for k in results)
    with pytest.raises(FileExistsError):search.main(arguments(source,output))


def test_missing_vc_metadata_and_reference_are_rejected_before_output(source,tmp_path):
    output=tmp_path/'search';summary=source/'neb.dat'
    summary.write_text(summary.read_text().replace('# recast_atomic_force true\n',''))
    with pytest.raises(ValueError,match='metadata'):search.main(arguments(source,output))
    assert not output.exists()


def test_unknown_vc_metric_is_rejected_before_output(source,tmp_path):
    summary=source/'neb.dat';summary.write_text(summary.read_text().replace('vcneb-frobenius-v1','unknown'))
    output=tmp_path/'search'
    with pytest.raises(ValueError,match='metric'):search.main(arguments(source,output))
    assert not output.exists()


def test_one_failed_trial_does_not_discard_other_results(source,tmp_path,monkeypatch):
    execute=search.execute;calls=[]
    def failing(command,folder,env,timeout):
        calls.append(folder.name)
        if folder.name=='trial-0000':return 1,0.
        return execute(command,folder,env,timeout)
    monkeypatch.setattr(search,'execute',failing)
    output=tmp_path/'search';assert search.main(arguments(source,output))==1
    results=json.loads((output/'results.json').read_text())
    assert calls==['baseline','trial-0000','trial-0001']
    assert results['trial-0000']['status']=='failed'
    assert results['trial-0001']['status']!='failed'


@pytest.mark.parametrize('mode',['images','atoms'])
def test_search_mpi_matches_serial_candidate_geometry_and_counts(source,tmp_path,mode):
    mpi=binary(True)
    baseline=tmp_path/'serial';parallel=tmp_path/'mpi'
    assert search.main(arguments(source,baseline))==0
    args=arguments(source,parallel);args[args.index('--binary')+1]=str(mpi)
    assert search.main(args+['--ranks','3','--threads','2','--omp-mode',mode,'--profile'])==0
    a=json.loads((baseline/'results.json').read_text());b=json.loads((parallel/'results.json').read_text())
    for key in a:
        assert a[key]['status']==b[key]['status'];assert a[key]['band_evaluations']==b[key]['band_evaluations']
        assert a[key]['image_evaluations']==b[key]['image_evaluations']
        assert a[key]['barrier_eV']==pytest.approx(b[key]['barrier_eV'],abs=2e-6,rel=1e-8)
        assert (baseline/key/'image-1.vasp').read_bytes()==(parallel/key/'image-1.vasp').read_bytes()
        timing=json.loads((parallel/key/'profile.json').read_text());assert len(timing['ranks'])==3
        assert all(rank['symmetry_release']['image_evaluations']==0 for rank in timing['ranks'])
