"""Actual CLI opt-in to partial-periodic variable-cell coordinates."""
import re
import numpy as np
import pytest
from test_cli_error_paths import run_command


def prepare(work,rank):
    cell=np.zeros((3,3))
    if rank:cell[0]=[8,0,0]
    if rank==2:cell[1]=[.4,9,0]
    pbc=['T' if i<rank else 'F' for i in range(3)]
    (work/'index.map').write_text('H\n')
    for i,position in [(0,[1.3,0,.2]),(2,[1.2,.2,.4])]:
        (work/f'image{i}.xyz').write_text('2\nLattice="'+' '.join(map(str,cell.flat))+
            '" pbc="'+' '.join(pbc)+'" Properties=species:S:1:pos:R:3\nH 0 0 0\nH '+' '.join(map(str,position))+'\n')
    return cell,pbc


def potential_settings():
    return 'pot:pottype lj612\npot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 3\n'


@pytest.mark.parametrize('rank',[0,1,2])
@pytest.mark.parametrize('method',['linear','idpp'])
def test_cli_vcneb_periodic_subspace_keeps_open_rows_and_cartesian_images(tmp_path,rank,method):
    cell,pbc=prepare(tmp_path,rank)
    (tmp_path/'neb.ini').write_text(potential_settings()+
        'nimgs 3\nimage:0 image0.xyz\nimage:2 image2.xyz\ninput-format extxyz\n'
        'driver fire\nmax_total_iter 2\nmaxiteration 2\nmax_move .01\n'
        'neb:vcneb true\nneb:periodic_subspace true\nneb:iref 0\nneb:gtol 1e-8\n'
        f'neb:climb false\nneb:init_method {method}\nneb:idpp_max_steps 2\n')
    result=run_command('imrmlp',['neb','--settings','neb.ini','--num_threads','1'],tmp_path)
    assert result.returncode in (0,2),result.stdout+result.stderr
    lines=(tmp_path/'mep.xyz').read_text().splitlines()
    assert len(lines)==12
    for i in range(3):
        header=lines[4*i+1]
        actual=np.array(list(map(float,re.search(r'Lattice="([^"]+)"',header).group(1).split()))).reshape(3,3)
        np.testing.assert_array_equal(actual[rank:],cell[rank:])
        assert 'pbc="'+' '.join(pbc)+'"' in header
        assert np.isfinite(np.array([list(map(float,row.split()[1:4])) for row in lines[4*i+2:4*i+4]])).all()
        assert 'Cartesian' in (tmp_path/f'nebimage_{i}_final.vasp').read_text()
    if rank==0:assert not np.any(actual)


@pytest.mark.parametrize('rank',[0,1,2])
def test_cli_vcdimer_periodic_subspace_writes_explicit_restart_convention(tmp_path,rank):
    prepare(tmp_path,rank)
    (tmp_path/'dimer.ini').write_text(potential_settings()+
        'structure image0.xyz\ninput-format extxyz\ndimer:vcflag true\n'
        'dimer:periodic_subspace true\ndimer:maxiteration 0\n')
    result=run_command('imrmlp',['dimer','--settings','dimer.ini','--num_threads','1'],tmp_path)
    # A zero iteration budget produces a usable restart but reports nonconvergence.
    assert result.returncode==1,result.stdout+result.stderr
    assert 'reached maximum iterations without convergence' in result.stdout
    lines=(tmp_path/'dimer_modecar_final').read_text().splitlines()
    assert lines[0]==f'GPML_MODECAR 1 2 vc-periodic-gram-r{rank}-mixed-recast'
    values=[float(item) for line in lines[1:] for item in line.split()]
    assert len(values)==9+rank*(rank+1)//2+6
    assert np.isfinite(values).all()


def test_cli_periodic_subspace_rejects_3d_pressure(tmp_path):
    prepare(tmp_path,2)
    (tmp_path/'dimer.ini').write_text(potential_settings()+
        'structure image0.xyz\ninput-format extxyz\ndimer:vcflag true\n'
        'dimer:periodic_subspace true\ndimer:pressure 1\ndimer:maxiteration 0\n')
    result=run_command('imrmlp',['dimer','--settings','dimer.ini','--num_threads','1'],tmp_path)
    assert result.returncode!=0
    assert 'forbids three-dimensional pressure' in result.stdout+result.stderr
    assert not (tmp_path/'dimer_modecar_final').exists()
