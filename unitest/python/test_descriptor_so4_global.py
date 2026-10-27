"""SO4 Python/global dispatch and the actual descriptor CLI share one contract."""
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from imrmlp import Atoms, Descriptor, write_one

ROOT = Path(imrmlp.__file__).resolve().parents[1]
BASE = {
    'biso4:twojmax': '2', 'biso4:cutoff': '2.7', 'biso4:radius': '4.2',
    'biso4:weights': '.7 1.3', 'biso4:selfweights': '.4 .8',
    'biso4:average': 'inner-density',
}


def geometry():
    # Short periodic rows give repeated images; the middle row is nonperiodic.
    return Atoms(numbers=[1, 2, 1], elements=['H', 'He'],
                 positions=[[.21, .32, .43], [1.17, .72, .64], [.49, 1.58, 1.19]],
                 cell=[[2.6, .2, .1], [.1, 4.7, .3], [.2, .1, 2.8]],
                 pbc=[True, False, True])


def _chart_point(reference, direction, step, include_cell):
    moved = reference.copy()
    if include_cell:
        fractional = reference.positions @ np.linalg.inv(reference.cell)
        moved.cell = reference.cell + step * direction[:9].reshape(3, 3)
        moved.positions = fractional @ moved.cell + step * direction[9:].reshape(-1, 3)
    else:
        moved.positions = reference.positions + step * direction.reshape(-1, 3)
    return moved


def _pullback_gradient(gradient, point, reference):
    """Express a newly returned r=f_current C+u gradient in the original chart."""
    pulled = gradient.copy()
    initial_f = reference.positions @ np.linalg.inv(reference.cell)
    current_f = point.positions @ np.linalg.inv(point.cell)
    pulled[:9] += ((initial_f-current_f).T @ gradient[9:].reshape(-1, 3)).ravel()
    return pulled


@pytest.mark.parametrize('convention,radial,normalized', [
    ('q1', 'linear', 'false'), ('q2', 'atan', 'true'),
    ('q3', 'linear', 'true'), ('q4', 'atan', 'false'),
])
def test_explicit_runtime_options_preserve_reference_snap_and_spline(convention, radial, normalized):
    params = {**BASE, 'biso4:quaternion_convention': convention,
              'biso4:radial_angle_map': radial, 'biso4:normalize_U': normalized}
    reference = Descriptor('biso4', {**params, 'biso4:contraction_backend': 'reference'})
    exact = Descriptor('biso4', {**params, 'biso4:contraction_backend': 'snap'})
    spline = Descriptor('biso4', {**params, 'biso4:contraction_backend': 'snap',
                                 'biso4:wigner_backend': 'spline'})
    atoms = geometry()
    expected = reference.compute(atoms)
    expected_global, expected_jac = reference.compute_averaged_with_gradient(atoms)
    assert expected.shape == (len(atoms), reference.feature_size)
    for backend, tolerance in [(exact, 2e-10), (spline, 2e-7)]:
        np.testing.assert_allclose(backend.compute(atoms), expected, rtol=tolerance, atol=tolerance)
        values, jacobian = backend.compute_averaged_with_gradient(atoms)
        np.testing.assert_allclose(values, expected_global, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(jacobian, expected_jac, rtol=tolerance, atol=tolerance)
    # The opt-in normalization and radial map must reach the implementation.
    alternate = Descriptor('biso4', {**params, 'biso4:normalize_U': 'true' if normalized == 'false' else 'false'})
    assert np.linalg.norm(alternate.compute_averaged(atoms)-expected_global) > 1e-4
    alternate = Descriptor('biso4', {**params, 'biso4:radial_angle_map': 'atan' if radial == 'linear' else 'linear'})
    assert np.linalg.norm(alternate.compute_averaged(atoms)-expected_global) > 1e-4


def test_global_atom_strain_and_raw_cell_jacobians_use_distinct_charts():
    descriptor = Descriptor('biso4', {**BASE, 'biso4:quaternion_convention': 'q2',
                                    'biso4:radial_angle_map': 'atan', 'biso4:normalize_U': 'true',
                                    'biso4:contraction_backend': 'snap'})
    atoms = geometry()
    values, atom_jac = descriptor.compute_averaged_with_gradient(atoms)
    strain_values, strain_jac = descriptor.compute_averaged_with_strain_gradient(atoms)
    cell_values, cell_jac = descriptor.compute_averaged_with_cell_gradient(atoms)
    d, n = descriptor.feature_size, len(atoms)
    assert values.shape == (d,) and atom_jac.shape == (d, n, 3)
    assert strain_jac.shape == cell_jac.shape == (d, 3, 3)
    np.testing.assert_allclose(values, strain_values, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(values, cell_values, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(atom_jac.sum(axis=1), 0., atol=2e-12)
    direction = np.linspace(-.09, .12, n*3).reshape(n, 3)
    strain = np.array([[.07, -.04, .02], [.01, -.06, .03], [-.02, .05, .08]])
    cell_direction = np.array([[.06, -.01, .02], [-.03, .07, .04], [.01, -.02, -.05]])
    for step in [2e-4, 1e-4]:
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions = atoms.positions + step*direction
        minus.positions = atoms.positions - step*direction
        fd = (descriptor.compute_averaged(plus)-descriptor.compute_averaged(minus))/(2*step)
        np.testing.assert_allclose(fd, np.einsum('fia,ia->f', atom_jac, direction), rtol=2e-6, atol=2e-7)
        plus, minus = atoms.copy(), atoms.copy()
        plus.cell, minus.cell = atoms.cell @ (np.eye(3)+step*strain).T, atoms.cell @ (np.eye(3)-step*strain).T
        plus.positions, minus.positions = atoms.positions @ (np.eye(3)+step*strain).T, atoms.positions @ (np.eye(3)-step*strain).T
        fd = (descriptor.compute_averaged(plus)-descriptor.compute_averaged(minus))/(2*step)
        np.testing.assert_allclose(fd, np.einsum('fab,ab->f', strain_jac, strain), rtol=2e-6, atol=2e-7)
        fractional = atoms.positions @ np.linalg.inv(atoms.cell)
        plus.cell, minus.cell = atoms.cell+step*cell_direction, atoms.cell-step*cell_direction
        plus.positions, minus.positions = fractional @ plus.cell, fractional @ minus.cell
        fd = (descriptor.compute_averaged(plus)-descriptor.compute_averaged(minus))/(2*step)
        np.testing.assert_allclose(fd, np.einsum('fab,ab->f', cell_jac, cell_direction), rtol=2e-6, atol=2e-7)
    fractional = atoms.copy()
    fractional.to_fractional()
    saved = fractional.positions.copy()
    for actual, expected in zip(descriptor.compute_averaged_with_cell_gradient(fractional), (values, cell_jac)):
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(fractional.positions, saved)
    assert not fractional._impl.cartesian


@pytest.mark.parametrize('include_cell', [False, True])
def test_weighted_global_curvature_has_full_cross_center_chain(include_cell):
    descriptor = Descriptor('biso4', {**BASE, 'biso4:quaternion_convention': 'q4',
                                    'biso4:radial_angle_map': 'atan', 'biso4:normalize_U': 'true',
                                    'biso4:contraction_backend': 'snap'})
    atoms = geometry()
    weights = np.linspace(-.04, .07, descriptor.feature_size)
    dof = 3*len(atoms)+(9 if include_cell else 0)
    direction = .08*np.sin(np.arange(dof)+1.)
    value, gradient, product = descriptor.compute_weighted_global_hvp(atoms, weights, direction, include_cell)
    dense_value, dense_gradient, hessian = descriptor.compute_weighted_global_hessian(atoms, weights, include_cell)
    assert gradient.shape == product.shape == (dof,)
    assert hessian.shape == (dof, dof)
    np.testing.assert_allclose(value, weights @ descriptor.compute_averaged(atoms), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(dense_value, value, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(dense_gradient, gradient, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(hessian @ direction, product, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(hessian, hessian.T, rtol=2e-10, atol=2e-10)
    _, atom_jac = descriptor.compute_averaged_with_gradient(atoms)
    expected_gradient = weights @ atom_jac.reshape(descriptor.feature_size, -1)
    if include_cell:
        _, cell_jac = descriptor.compute_averaged_with_cell_gradient(atoms)
        expected_gradient = np.r_[weights @ cell_jac.reshape(descriptor.feature_size, 9), expected_gradient]
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-10, atol=2e-10)
    for step in [2e-4, 1e-4]:
        plus, minus = (_chart_point(atoms, direction, sign*step, include_cell) for sign in [1., -1.])
        # The independent oracle differentiates the feature Jacobian, not a
        # second call to the Hessian implementation. Pull back its raw-cell
        # block because returned gradients use each geometry's own chart.
        def independent_gradient(point):
            _, jac = descriptor.compute_averaged_with_gradient(point)
            result = weights @ jac.reshape(descriptor.feature_size, -1)
            if include_cell:
                _, cell = descriptor.compute_averaged_with_cell_gradient(point)
                result = np.r_[weights @ cell.reshape(descriptor.feature_size, 9), result]
                result = _pullback_gradient(result, point, atoms)
            return result
        fd = (independent_gradient(plus)-independent_gradient(minus))/(2*step)
        np.testing.assert_allclose(fd, product, rtol=4e-6, atol=3e-7)
        fd_value = weights @ (descriptor.compute_averaged(plus)-descriptor.compute_averaged(minus))/(2*step)
        np.testing.assert_allclose(fd_value, gradient @ direction, rtol=3e-6, atol=2e-7)
    translation = np.zeros(dof)
    translation[-3*len(atoms):] = np.tile([.3, -.2, .1], len(atoms))
    _, _, zero = descriptor.compute_weighted_global_hvp(atoms, weights, translation, include_cell)
    np.testing.assert_allclose(zero, 0., atol=3e-11)


@pytest.mark.parametrize('average', ['outer', 'inner-density'])
def test_cli_serializes_true_local_and_global_so4_descriptors(tmp_path, average):
    build = Path(os.environ.get('IMRMLP_TEST_BUILD_DIR', ROOT/'build'/os.environ.get('BUILD_MODE', 'release'))).resolve()
    executable = build/'imrmlp'
    if not executable.exists():
        pytest.skip('Build the matching imrmlp before the SO4 CLI comparison')
    params = {**BASE, 'biso4:average': average, 'biso4:quaternion_convention': 'q3',
              'biso4:radial_angle_map': 'atan', 'biso4:normalize_U': 'true',
              'biso4:contraction_backend': 'snap'}
    atoms = geometry()
    write_one(tmp_path/'input.xyz', atoms)
    (tmp_path/'index.map').write_text('H 0\nHe 1\n')
    (tmp_path/'settings.ini').write_text(''.join(f'{key} {value}\n' for key, value in params.items()))
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(build)+':'+env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = '1'
    result = subprocess.run([str(executable), 'calc_descriptors', '--dtype', 'biso4',
        '--input', 'input.xyz', '--settings', 'settings.ini', '--output', 'output.xyz',
        '--local-name', 'local_so4', '--global-name', 'global_so4', '--lossless', '1',
        '--num_threads', '1'], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    lines = (tmp_path/'output.xyz').read_text().splitlines()
    match = re.search(r'\bglobal_so4="([^"]*)"', lines[1])
    assert match, lines[1]
    global_output = np.fromstring(match.group(1), sep=' ')
    properties = re.search(r'\bProperties=([^\s]+)', lines[1]).group(1).split(':')
    offset, span = 0, None
    for i in range(0, len(properties), 3):
        count = int(properties[i+2])
        if properties[i] == 'local_so4':
            span = slice(offset, offset+count)
        offset += count
    assert span is not None, lines[1]
    local_output = np.array([[float(token) for token in line.split()[span]] for line in lines[2:2+len(atoms)]])
    descriptor = Descriptor('biso4', params)
    assert local_output.shape == (len(atoms), descriptor.feature_size)
    assert global_output.shape == (descriptor.feature_size,)
    np.testing.assert_allclose(local_output, descriptor.compute(atoms), rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(global_output, descriptor.compute_averaged(atoms), rtol=2e-11, atol=2e-11)
    if average == 'outer':
        np.testing.assert_allclose(global_output, local_output.mean(axis=0), rtol=2e-11, atol=2e-11)
    else:
        assert np.linalg.norm(global_output-local_output.mean(axis=0)) > 1e-5


def test_global_api_rejects_wrong_mode_and_shapes():
    atoms = geometry()
    inner = Descriptor('biso4', BASE)
    weights = np.ones(inner.feature_size)
    for call in [lambda: inner.compute_weighted_global_hvp(atoms, weights[:-1], np.zeros(9)),
                 lambda: inner.compute_weighted_global_hvp(atoms, weights, np.zeros(8)),
                 lambda: inner.compute_weighted_global_hvp(atoms, weights*np.nan, np.zeros(9)),
                 lambda: Descriptor('biso4', {**BASE, 'biso4:average': 'outer'}).compute_weighted_global_hessian(atoms, weights),
                 lambda: Descriptor('biso4', {**BASE, 'biso4:wigner_backend': 'unknown'}),
                 lambda: Descriptor('biso4', {**BASE, 'biso4:quaternion_convention': 'q5'})]:
        with pytest.raises(ValueError):
            call()
