"""The public seed API performs geometry operations without a calculator."""
import numpy as np
import pytest
from ase import Atoms as ASEAtoms
from imrmlp import Atoms
from imrmlp.neb import make_path_seed


def path():
    return [Atoms.from_ase(ASEAtoms('H2',positions=[[1+i*.05,1,1],[5-i*.05,1,1]],cell=[6,7,8],pbc=True)) for i in range(5)]


@pytest.mark.parametrize('mode',['sine1','sine2','mixed3'])
def test_public_seed_reproducibility_bounds_and_unchanged_source(mode):
    source=path();before=[a.positions.copy() for a in source]
    a,info=make_path_seed(source,mode=mode,seed=17)
    b,_=make_path_seed(source,mode=mode,seed=17)
    assert info['status']=='generated';assert .0099<info['atomic_displacement_A']<=.01000000001
    for i in range(5):
        np.testing.assert_array_equal(source[i].positions,before[i]);np.testing.assert_array_equal(a[i].positions,b[i].positions)
    for i in [0,4]:np.testing.assert_array_equal(a[i].positions,before[i])


def test_public_seed_rejects_missing_reference_and_bad_selection():
    with pytest.raises(ValueError,match='reference'):make_path_seed(path(),settings={'vcneb':True})
    with pytest.raises(ValueError,match='index'):make_path_seed(path(),atoms=[2])


def test_local_single_atom_and_fixed_mask():
    source=path()
    for a in source:a.pbc=[True,True,False];a.selective_dynamics_mask=np.array([[1,0,1],[0,0,0]])
    result,info=make_path_seed(source,settings={'selective_dynamics':True},atoms=[0])
    assert info['status']=='generated'
    for old,new in zip(source,result):
        np.testing.assert_array_equal(old.positions[1],new.positions[1]);assert old.positions[0,1]==new.positions[0,1]
