"""Public Python paths preserve Cartesian Atoms while using explicit mixed DOFs."""
import numpy as np
import pytest
pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms,NEB,Dimer,Potential


def pair(rank):
    cell=np.zeros((3,3));pbc=[False]*3
    if rank:cell[0]=[8,0,0];pbc[0]=True
    if rank==2:cell[1]=[.4,9,0];pbc[1]=True
    a=Atoms(numbers=[1,1],positions=[[0,0,0],[1.3,0,.2]],cell=cell,pbc=pbc,elements=['H'])
    a.chemmap=[0,0]
    b=a.copy();b.positions=[[0,0,0],[1.2,.2,.4]]
    return a,b


@pytest.mark.parametrize('rank',[0,1,2])
@pytest.mark.parametrize('method',['linear','idpp'])
def test_public_neb_explicit_periodic_subspace_returns_cartesian_paths(rank,method):
    first,last=pair(rank)
    pot=Potential.from_settings({'pottype':'lj612','ntypes':1,'0-0':'1 1','cut:0-0':3})
    driver=NEB(pot,vcneb=True,n_images=3,max_iter=2,max_move=.01,init_method=method,periodic_subspace=True)
    result=driver.run(first,last)
    for image in result.images:
        assert image._impl.cartesian
        assert np.isfinite(image.positions).all() and np.isfinite(image.energy)
        np.testing.assert_array_equal(image.cell[rank:],first.cell[rank:])
        np.testing.assert_array_equal(image.pbc,first.pbc)
    np.testing.assert_allclose(result.images[0].positions,first.positions,rtol=0,atol=1e-12)
    np.testing.assert_allclose(result.images[-1].positions,last.positions,rtol=0,atol=1e-12)


@pytest.mark.parametrize('rank',[0,1,2])
def test_public_dimer_explicit_periodic_subspace_preserves_open_geometry(tmp_path,monkeypatch,rank):
    monkeypatch.chdir(tmp_path)
    first,_=pair(rank)
    pot=Potential.from_settings({'pottype':'lj612','ntypes':1,'0-0':'1 1','cut:0-0':3})
    driver=Dimer(pot,vc_flag=True,max_iter=0,periodic_subspace=True)
    result=driver.run(first)
    assert result.saddle_atoms._impl.cartesian
    np.testing.assert_allclose(result.saddle_atoms.positions,first.positions,atol=1e-12)
    np.testing.assert_array_equal(result.saddle_atoms.cell,first.cell)
    np.testing.assert_array_equal(result.saddle_atoms.pbc,first.pbc)
    assert np.isfinite(result.energy) and np.isfinite(result.curvature)
