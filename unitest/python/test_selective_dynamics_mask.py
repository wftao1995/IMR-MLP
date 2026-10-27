"""Public atomic masks feed the existing fixed-cell selective-dynamics path."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Potential, Relaxation


def atoms():
    return Atoms(numbers=[1,1],elements=['H'],positions=[[2.,2.,2.],[3.5,2.3,2.4]],cell=np.eye(3)*12)


def test_mask_absence_binary_assignment_owned_getters_copy_and_clear():
    a=atoms()
    assert a.selective_dynamics_mask is None
    assert a._impl.selective_dynamics_mask is None
    source=np.array([[0,0,0],[1,0,1]],dtype=int)
    expected=source.astype(bool)
    a.selective_dynamics_mask=source
    source[:]=1
    np.testing.assert_array_equal(a.selective_dynamics_mask,expected)
    assert a.selective_dynamics_mask.dtype==np.bool_
    returned=a.selective_dynamics_mask; returned[:]=False
    native_returned=a._impl.selective_dynamics_mask; native_returned[:]=True
    np.testing.assert_array_equal(a.selective_dynamics_mask,expected)
    copied=a.copy(); copied.selective_dynamics_mask=np.ones((2,3),dtype=bool)
    np.testing.assert_array_equal(a.selective_dynamics_mask,expected)
    copied.selective_dynamics_mask=None
    assert copied.selective_dynamics_mask is None
    a.selective_dynamics_mask=None
    assert a.selective_dynamics_mask is None
    a.selective_dynamics_mask=[[True,False,True],[False,True,False]]
    np.testing.assert_array_equal(a.selective_dynamics_mask,[[1,0,1],[0,1,0]])


@pytest.mark.parametrize('shape',[(1,3),(3,3),(2,2),(2,4),(0,3),(3,),()])
def test_mask_rejects_wrong_shape_without_changing_old_mask(shape):
    a=atoms(); expected=np.array([[0,1,0],[1,0,1]],dtype=bool)
    a.selective_dynamics_mask=expected
    with pytest.raises(ValueError,match='shape'):
        a.selective_dynamics_mask=np.ones(shape)
    np.testing.assert_array_equal(a.selective_dynamics_mask,expected)


@pytest.mark.parametrize('invalid',[np.nan,np.inf,-1.,2.,.5])
@pytest.mark.parametrize('native',[False,True])
def test_mask_rejects_nonbinary_or_nonfinite_values_atomically(invalid,native):
    a=atoms(); expected=np.array([[0,1,0],[1,0,1]],dtype=bool)
    a.selective_dynamics_mask=expected
    values=expected.astype(float); values[1,2]=invalid
    with pytest.raises(ValueError,match='0.*1|binary'):
        if native: a._impl.selective_dynamics_mask=values
        else: a.selective_dynamics_mask=values
    np.testing.assert_array_equal(a.selective_dynamics_mask,expected)


def test_native_mask_shape_rejection_is_atomic_and_none_clears():
    a=atoms(); expected=np.array([[1,0,1],[0,1,0]],dtype=bool)
    a._impl.selective_dynamics_mask=expected
    for invalid in (np.zeros((1,3)),np.zeros((2,2)),np.zeros((0,3))):
        with pytest.raises(ValueError,match='shape'): a._impl.selective_dynamics_mask=invalid
        np.testing.assert_array_equal(a.selective_dynamics_mask,expected)
    a._impl.selective_dynamics_mask=None
    assert a.selective_dynamics_mask is None


@pytest.mark.parametrize('slanted',[False,True])
@pytest.mark.parametrize('mode',['fixed_atom','fixed_components','clear','disabled','absent'])
def test_real_fixed_cell_relaxation_honors_mask_in_existing_coordinates(slanted,mode):
    potential=Potential.from_settings({'pottype':'lj612','ntypes':1,'0-0':'1 1','cut:0-0':4})
    a=atoms()
    if slanted: a.cell=[[12.,0.,0.],[2.,13.,0.],[1.,3.,14.]]
    before=a.positions.copy(); cell=a.cell.copy()
    if mode in ('fixed_atom','clear','disabled'):
        a.selective_dynamics_mask=[[0,0,0],[1,1,1]]
    elif mode=='fixed_components':
        a.selective_dynamics_mask=[[0,0,0],[1,0,0]]
    if mode=='clear': a.selective_dynamics_mask=None
    result=Relaxation(potential,driver='lbfgspp',selective_dynamics=mode!='disabled',
        relax_cell=False,max_iter=300,gtol=1e-8,max_move=.05).run(a)
    assert result.converged,result.stop_reason
    np.testing.assert_array_equal(a.cell,cell)
    distance=np.linalg.norm(a.positions[1]-a.positions[0])
    assert distance==pytest.approx(2**(1/6),abs=2e-8)
    if mode in ('fixed_atom','fixed_components'):
        np.testing.assert_allclose(a.positions[0],before[0],atol=2e-13,rtol=0)
        assert np.linalg.norm(a.positions[1]-before[1])>.1
        if mode=='fixed_components':
            np.testing.assert_allclose(a.positions[1,1:],before[1,1:],atol=2e-13,rtol=0)
    else:
        assert np.linalg.norm(a.positions[0]-before[0])>.1
        assert np.linalg.norm(a.positions[1]-before[1])>.1
