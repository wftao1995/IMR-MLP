"""Current descriptor and trajectory CLI contracts for angular limits and identity."""
import numpy as np
import pytest

from test_cli_error_paths import run_command


@pytest.mark.parametrize('sign', [-1., 1.])
def test_behler_zero_exponent_collinear_position_and_structure_jacobians(sign):
    from imrmlp import Atoms, Descriptor
    descriptor = Descriptor('behler', {
        'behler:dispcount': '2', 'behler:dispselect': '4 5',
        'behler:cutoff': '5', 'behler:eta': '.2',
        'behler:zeta': '0', 'behler:lambda': str(-sign),
    })
    atoms = Atoms(numbers=[1, 1, 1], elements=['H'],
                  positions=[[0., 0., 0.], [1., 0., 0.], [2 * sign, 0., 0.]],
                  cell=np.eye(3) * 20, pbc=False)
    local, indices, jac = descriptor.compute_with_gradient(atoms, 0)
    global_values, global_jac = descriptor.compute_averaged_with_gradient(atoms)
    assert np.isfinite(local).all() and np.isfinite(jac).all()
    assert np.isfinite(global_values).all() and np.isfinite(global_jac).all()
    for step in (1e-4, 1e-5):
        for i in range(len(atoms)):
            for a in range(3):
                plus, minus = atoms.copy(), atoms.copy()
                p, m = plus.positions.copy(), minus.positions.copy()
                p[i, a] += step
                m[i, a] -= step
                plus.positions, minus.positions = p, m
                local_fd = (descriptor.compute(plus)[0] - descriptor.compute(minus)[0]) / (2 * step)
                global_fd = (descriptor.compute_averaged(plus) - descriptor.compute_averaged(minus)) / (2 * step)
                np.testing.assert_allclose(jac[:, np.flatnonzero(indices == i)[0], a], local_fd, atol=1e-8)
                np.testing.assert_allclose(global_jac[:, i, a], global_fd, atol=1e-8)


def test_behler_current_python_requires_explicit_weighting():
    from imrmlp import Descriptor
    params = {'behler:dispcount': '2', 'behler:dispselect': '2 5', 'behler:cutoff': '5'}
    assert Descriptor('behler', params).feature_size == 2
    params['behler:weighted'] = 'false'
    assert Descriptor('behler', params).feature_size == 2
    params['behler:weighted'] = 'true'
    with pytest.raises(Exception, match='weighted=true requires weights or elements'):
        Descriptor('behler', params)


def test_behler_cli_requires_explicit_weighting(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.xyz').write_text(
        '1\npbc="F F F" Properties=species:S:1:pos:R:3\nH 0 0 0\n')
    (tmp_path / 'settings.ini').write_text(
        'behler:dispcount 2\nbehler:dispselect 2 5\nbehler:cutoff 5\nbehler:weighted true\n')
    result = run_command('imrmlp', [
        'calc_descriptors', '--input', 'input.xyz', '--input-format', 'extxyz',
        '--dtype', 'behler', '--settings', 'settings.ini', '--output', 'output.xyz',
        '--num_threads', '1'], tmp_path)
    assert result.returncode != 0
    assert 'weighted=true requires weights or elements' in result.stderr
    assert not (tmp_path / 'output.xyz').exists()


@pytest.mark.parametrize('change', ['species', 'species order', 'PBC'])
def test_fft_cli_rejects_changed_identity_before_output(tmp_path, change):
    (tmp_path / 'index.map').write_text('H 0\nHe 1\nLi 2\n')
    species = ('H', 'Li') if change == 'species' else ('He', 'H') if change == 'species order' else ('H', 'He')
    pbc = 'T F F' if change == 'PBC' else 'F F F'
    (tmp_path / 'input.xyz').write_text(
        '2\npbc="F F F" Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3\nH 1 0 0\nHe 3 0 0\n'
        f'2\npbc="{pbc}" Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3\n'
        f'{species[0]} 1.1 0 0\n{species[1]} 3.1 0 0\n')
    (tmp_path / 'output.xyz').write_text('previous result\n')
    result = run_command('imrmlp', [
        'smooth_traj', '--input', 'input.xyz', '--input-format', 'extxyz',
        '--output', 'output.xyz', '--output-format', 'extxyz', '--overwrite',
        '--num_threads', '1'], tmp_path)
    assert result.returncode != 0
    assert ('same PBC' if change == 'PBC' else 'same species') in result.stderr
    assert (tmp_path / 'output.xyz').read_text() == 'previous result\n'
