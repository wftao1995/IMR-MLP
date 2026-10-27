"""Public NEB symmetry and perturbation controls use the shared C++ path codec."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, NEB, Potential


def endpoints():
    cell=np.diag([6., 7., 8.])
    a=Atoms(numbers=[1, 1], elements=['H'], positions=np.array([[.23,.17,.19],[.77,.17,.19]])@cell,
            cell=cell, pbc=True)
    b=a.copy();b.positions=np.array([[.28,.21,.19],[.72,.21,.19]])@cell
    return a,b


def potential():
    return Potential.from_settings({'pottype':'lj612','ntypes':1,'0-0':'1 1','cut:0-0':3})


def test_python_constraints_write_diagnostics_and_preserve_mirror(tmp_path):
    a,b=endpoints()
    report=tmp_path/'symmetry.dat'
    result=NEB(potential(),n_images=5,keep_symm=True,symmetry_diagnostics=str(report),
               solver_policy='stable-v1',max_iter=3,omp_mode='atoms').run(a,b)
    text=report.read_text()
    assert '# stage initial' in text and '# stage final' in text
    assert '# common_operations' in text and '# operation ' in text
    for image in result.images:
        assert image.positions[0,0]+image.positions[1,0]==pytest.approx(6.,abs=1e-10)
        np.testing.assert_allclose(image.positions[:,2],1.52,atol=1e-10)
    np.testing.assert_array_equal(result.images[0].positions,a.positions)
    np.testing.assert_array_equal(result.images[-1].positions,b.positions)


def test_python_seed_and_amplitude_reproduce_initial_perturbed_path(tmp_path):
    a,b=endpoints()
    # One band evaluation retains initialized coordinates under stable-v1.
    def path(seed, amplitude):
        return NEB(potential(),n_images=5,perturb_amplitude=amplitude,perturb_seed=seed,
                   solver_policy='stable-v1',max_force_evaluations=1,max_iter=2).run(a,b).images
    base,x,y,z=path(7,0),path(7,.03),path(7,.03),path(8,.03)
    displacement=[]
    for initial,first,second in zip(base,x,y):
        np.testing.assert_array_equal(first.positions,second.positions)
        displacement.extend(np.linalg.norm(first.positions-initial.positions,axis=1))
    assert max(displacement)==pytest.approx(.03,abs=1e-12)
    assert not np.allclose(x[2].positions,z[2].positions)


@pytest.mark.parametrize('settings', [dict(omp_mode='nested'),dict(keep_symm=True,perturb_amplitude=.1),
                                      dict(symprec=0),dict(perturb_seed=-1)])
def test_python_invalid_new_controls_fail(settings):
    a,b=endpoints()
    with pytest.raises((ValueError,RuntimeError)):
        NEB(potential(),max_iter=2,**settings).run(a,b)


def test_python_release_preserves_result_and_writes_separate_trials(tmp_path):
    a,b=endpoints()
    base=NEB(potential(),n_images=3,keep_symm=True,neb_ftol=100,solver_policy='stable-v1',max_iter=2).run(a,b)
    folder=tmp_path/'release'
    checked=NEB(potential(),n_images=3,keep_symm=True,neb_ftol=100,solver_policy='stable-v1',max_iter=2,
        symmetry_release=str(folder),symmetry_release_trials=1,symmetry_release_steps=2,
        symmetry_release_evaluations=2).run(a,b)
    import json
    report=json.loads((folder/'report.json').read_text())
    assert checked.symmetry_release_status==report['status']
    assert len(report['trials'])==2
    assert base.n_image_force_evals==checked.n_image_force_evals
    for x,y in zip(base.images,checked.images): np.testing.assert_array_equal(x.positions,y.positions)
    assert (folder/'trial-1/final/image-1.vasp').exists()


def test_python_release_default_and_output_failure_preserve_main(tmp_path):
    a,b=endpoints()
    options=dict(n_images=3,keep_symm=True,neb_ftol=100,solver_policy='stable-v1',max_iter=2)
    assert NEB(potential(),**options).run(a,b).symmetry_release_status=='disabled'
    result=NEB(potential(),**options,symmetry_release=str(tmp_path)).run(a,b)
    assert result.converged and result.symmetry_release_status=='failed'
