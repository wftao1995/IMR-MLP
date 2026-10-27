"""Animation keeps newly evaluated EFS but cannot reuse reference curvature labels."""
import numpy as np
import pytest
from test_cli_error_paths import run_command
pytest.importorskip('imrmlp._imrmlp')
import imrmlp

PAIR = 'MTP\nversion = 1.1.0\npotential_name = analytic_pair\nscaling = 1\nspecies_count = 1\npotential_tag =\nradial_basis_type = RBChebyshev\nmin_dist = 1\nmax_dist = 5\nradial_basis_size = 2\nradial_funcs_count = 1\nradial_coeffs\n0-0\n{1, 1}\nalpha_moments_count = 1\nalpha_index_basic_count = 1\nalpha_index_basic = {{0, 0, 0, 0}}\nalpha_index_times_count = 0\nalpha_index_times = {}\nalpha_scalar_moments = 1\nalpha_moment_mapping = {0}\nspecies_coeffs = {0}\nmoment_coeffs = {1}\n'

@pytest.mark.parametrize('include_hvp', [False, True])
@pytest.mark.parametrize('evaluate_efs', [False, True])
def test_animation_drops_old_geometry_labels_before_optional_efs(tmp_path, evaluate_efs, include_hvp):
    atom=imrmlp.Atoms(numbers=[1,1],elements=['H'],positions=[[2,2,2],[4.2,2.1,2.2]],cell=np.eye(3)*30)
    atom.energy=9.;atom.free_energy=8.;atom.forces=np.ones((2,3));atom.stress=np.eye(3)
    atom.set_hessian(np.eye(6)*7)
    if include_hvp: atom.set_hvp(np.ones((6,1)),np.ones((6,1))*2)
    imrmlp.write(tmp_path/'input.xyz',atom)
    (tmp_path/'pair.mtp').write_text(PAIR)
    (tmp_path/'index.map').write_text('H 0\n')
    (tmp_path/'settings.ini').write_text('relax:driver fire\n')
    np.savetxt(tmp_path/'hessian',np.diag(np.arange(1.,7.)),fmt='%.17g')
    result=run_command('imrmlp',['modefollowing','settings.ini','--potfile','pair.mtp',
        '--input','input.xyz','--input-format','extxyz','--nocell','--car-rep',
        '--hessian','hessian','--mode','1','--frames','2','--linetraj',
        '--amplitude','0.001','--request-efs',str(evaluate_efs).lower()],tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    final_reference=imrmlp.read_one(tmp_path/'POSCAR_SAD',format='poscar')
    np.testing.assert_allclose(final_reference.positions,atom.positions,atol=1e-10,rtol=0)
    np.testing.assert_allclose(final_reference.cell,atom.cell,atol=1e-10,rtol=0)
    frames=imrmlp.read(tmp_path/'vibmode_meta_0.xyz')
    assert len(frames)==2
    for i,frame in enumerate(frames):
        expected=atom.positions.copy();expected[0,0]+= -.001 if i==0 else .001
        np.testing.assert_allclose(frame.positions,expected,atol=1e-8,rtol=0)
        assert frame.hessian.shape==(0,0)
        assert frame.hvp_targets.size==0
        if evaluate_efs:
            distance=np.linalg.norm(frame.positions[1]-frame.positions[0])
            assert frame.energy==pytest.approx((distance-5)**2*(distance-1),abs=1e-7)
            # Existing request-efs writes fresh E+pV under fenergy; p=0 here.
            assert frame.free_energy==pytest.approx(frame.energy,abs=1e-7)
            assert frame.forces is not None
        else:
            assert frame.energy is None and frame.forces is None and frame.stress is None
            assert frame.free_energy is None
            assert frame.site_energies is None
