"""SO3 complex/real backends share the public value, gradient and CLI contract."""
import re

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Descriptor, write_one
from test_cli_error_paths import run_command

PARAMS = {'biso3_lmax': '2', 'biso3_rsize': '2', 'biso3_rmin': '0', 'biso3_rmax': '4'}


def geometry():
    return Atoms(numbers=[1, 1, 1], elements=['H'],
                 positions=[[.2, .3, .4], [1.5, .5, .8], [.7, 1.9, .9]],
                 cell=np.eye(3)*12, pbc=False)


def test_real_backend_preserves_values_gradients_and_rotations():
    atoms = geometry()
    reference = Descriptor('biso3', PARAMS)
    real = Descriptor('biso3', {**PARAMS, 'biso3_real_harmonics': 'true'})
    value, gradient = reference.compute_averaged_with_gradient(atoms)
    actual, actual_gradient = real.compute_averaged_with_gradient(atoms)
    np.testing.assert_allclose(actual, value, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(value, reference.compute(atoms).mean(axis=0), atol=1e-12)
    moved = atoms.copy()
    rotation, _ = np.linalg.qr(np.array([[.3, .5, .7], [.2, .8, -.4], [.9, -.1, .3]]))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    moved.positions = atoms.positions @ rotation
    np.testing.assert_allclose(real.compute(moved), reference.compute(atoms), atol=1e-11)
    step = 1e-6
    for atom in range(len(atoms)):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            displacement = np.zeros((len(atoms), 3)); displacement[atom, axis] = step
            plus.positions = atoms.positions + displacement
            minus.positions = atoms.positions - displacement
            fd = (real.compute_averaged(plus)-real.compute_averaged(minus))/(2*step)
            np.testing.assert_allclose(actual_gradient[:, atom, axis], fd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize('real', ['false', 'true'])
def test_so3_cli_matches_python(tmp_path, real):
    atoms = geometry()
    write_one(tmp_path/'input.xyz', atoms)
    (tmp_path/'index.map').write_text('H 0\n')
    (tmp_path/'settings.ini').write_text('')
    args = ['calc_descriptors', '--dtype', 'biso3', '--input', 'input.xyz',
            '--settings', 'settings.ini', '--output', 'output.xyz', '--global-name', 'global_so3',
            '--biso3-lmax', '2', '--biso3-rsize', '2', '--biso3-rmax', '4',
            '--biso3-real-harmonics', real, '--lossless', '1', '--num_threads', '1']
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 0, result.stdout+result.stderr
    header = (tmp_path/'output.xyz').read_text().splitlines()[1]
    match = re.search(r'\bglobal_so3="([^"]*)"', header)
    assert match, header
    expected = Descriptor('biso3', {**PARAMS, 'biso3_real_harmonics': real}).compute_averaged(atoms)
    np.testing.assert_allclose(np.fromstring(match.group(1), sep=' '), expected, rtol=1e-12, atol=1e-12)
