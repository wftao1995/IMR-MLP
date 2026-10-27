"""Public descriptor values and Cartesian/strain derivatives have owned results.

Self-contained regressions migrated from the external binding suite. Invariant
features have symmetric strain derivatives; the off-diagonal checks establish
normalization and sign, and do not distinguish a pure transpose.
"""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from imrmlp import Atoms, Descriptor


SOAP_PARAMS = {
    'soap:ntypes': '2', 'soap:nmax': '2', 'soap:lmax': '2',
    'soap:rcut': '3', 'soap:sigma': '.5', 'soap:rbtype': 'poly',
}


def test_behler_values_match_radial_reference():
    cutoff, eta = 4., .3
    descriptor = Descriptor('behler', {
        'behler:dispcount': '2', 'behler:dispselect': '1 2',
        'behler:cutoff': str(cutoff), 'behler:eta': str(eta),
    })
    atoms = Atoms(numbers=[1, 1, 1], elements=['H'],
                  positions=[[1., 1., 1.], [2., 1., 1.], [1., 2.5, 1.2]],
                  cell=np.eye(3) * 12, pbc=False)
    expected = []
    for i, position in enumerate(atoms.positions):
        distances = np.linalg.norm(np.delete(atoms.positions, i, axis=0) - position, axis=1)
        weights = .5 * (np.cos(np.pi * distances / cutoff) + 1.)
        expected.append([weights.sum(), (weights * np.exp(-eta * distances**2)).sum()])
    values = descriptor.compute(atoms)
    assert values.shape == (3, descriptor.feature_size)
    np.testing.assert_allclose(values, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('dtype,params', [
    ('behler', {'behler:dispcount': '2', 'behler:dispselect': '1 2',
                'behler:cutoff': '4', 'behler:eta': '.3'}),
    ('bondorder', {'bondorder:maxl': '2', 'bondorder:cutoff': '4'}),
])
def test_ordinary_global_values_are_means_with_fixed_rank(dtype, params):
    descriptor = Descriptor(dtype, params)
    atoms = Atoms(numbers=[1, 1, 1], elements=['H'],
                  positions=[[0., 0., 0.], [1.1, .2, .1], [.4, 2.8, .3]],
                  cell=np.eye(3) * 12, pbc=False)
    local = descriptor.compute(atoms)
    values = descriptor.compute_averaged(atoms)
    assert values.shape == (descriptor.feature_size,)
    assert values.dtype == np.float64
    np.testing.assert_allclose(values, local.mean(axis=0), rtol=1e-13, atol=1e-13)
    assert np.linalg.norm(values - local.sum(axis=0)) > 1e-6




def _check_center_gradient_fd(descriptor, atoms, center, *, atol=2e-6):
    """Perturb source atoms, including the center, rather than neighbor images."""
    values, indices, jacobian = descriptor.compute_with_gradient(atoms, center)
    assert values.shape == (descriptor.feature_size,)
    assert indices.ndim == 1 and indices.dtype == np.int64
    assert jacobian.shape == (descriptor.feature_size, len(indices), 3)
    assert values.dtype == jacobian.dtype == np.float64
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(jacobian))
    np.testing.assert_array_equal(indices, np.unique(indices))
    assert center in indices
    np.testing.assert_allclose(values, descriptor.compute(atoms)[center],
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jacobian.sum(axis=1), 0., rtol=0., atol=1e-12)
    positions = atoms.positions.copy()
    step = 1e-6
    for source in range(len(atoms)):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            shifted = positions.copy()
            shifted[source, axis] += step
            plus.positions = shifted
            shifted = positions.copy()
            shifted[source, axis] -= step
            minus.positions = shifted
            finite_difference = (descriptor.compute(plus)[center] -
                                 descriptor.compute(minus)[center]) / (2 * step)
            slot = np.flatnonzero(indices == source)
            expected = jacobian[:, slot[0], axis] if len(slot) else 0.
            np.testing.assert_allclose(expected, finite_difference,
                                       rtol=2e-5, atol=atol,
                                       err_msg=f'source={source}, axis={axis}')
    return values, indices, jacobian


@pytest.mark.parametrize('dtype,params,atol', [('behler', {'behler:dispcount': '3', 'behler:dispselect': '1 2 4',
                'behler:cutoff': '4', 'behler:eta': '.3'}, 2e-6), ('bondorder', {'bondorder:maxl': '2', 'bondorder:cutoff': '4'}, 2e-6)], ids=['behler', 'bondorder'])
def test_center_position_gradient_matches_source_atom_differences(dtype, params, atol):
    descriptor = Descriptor(dtype, params)
    atoms = Atoms(numbers=[1, 1, 1, 1], elements=['H'],
                  positions=[[0., 0., 0.], [1.1, .2, .1], [.4, 2.8, .3],
                             [10., 10., 10.]],
                  cell=np.eye(3) * 20, pbc=False)
    _, indices, jacobian = _check_center_gradient_fd(descriptor, atoms, 1, atol=atol)
    np.testing.assert_array_equal(indices, [0, 1, 2])
    assert np.linalg.norm(jacobian) > 1e-6


def _short_partial_pbc_atoms():
    return Atoms(numbers=[1, 2], elements=['H', 'He'],
                 positions=[[.2, 1., 1.], [1.15, 1.2, 1.4]],
                 cell=np.diag([2.6, 12., 12.]), pbc=[True, False, False])












@pytest.fixture
def tilted_self_image_strain_case():
    params = {'behler:dispcount': '1', 'behler:dispselect': '1', 'behler:cutoff': '4'}
    atoms = Atoms(numbers=[1], elements=['H'], positions=[[.2, 1., 1.]],
                  cell=[[2.6, .4, .2], [0., 12., 0.], [0., 0., 12.]],
                  pbc=[True, False, False])
    # Only +/-a lie inside rcut. G1 = 2 fc(|a|); differentiate the cosine
    # cutoff analytically, without using a product gradient as the oracle.
    q = np.pi * np.sqrt(174.) / 20
    value = np.array([1 + np.cos(q)])
    jacobian = (-np.pi * np.sin(q) / (20 * np.sqrt(174.)) *
                np.array([[169., 26., 13.], [26., 4., 2.], [13., 2., 1.]]))[None]
    return params, atoms, value, jacobian


def _affine_strain_copy(atoms, a, b, step):
    deformation = np.eye(3)
    deformation[a, b] += step
    strained = atoms.copy()
    # Column r'=(I+epsilon)r: stored row positions AND cell use the transpose.
    strained.positions = atoms.positions @ deformation.T
    strained.cell = atoms.cell @ deformation.T
    return strained


def test_self_image_strain_gradient_matches_exact_g1_and_preserves_sources(tilted_self_image_strain_case):
    params, atoms, expected_value, expected_jacobian = tilted_self_image_strain_case
    descriptor = Descriptor('behler', params)
    old_values, indices, source_jacobian = descriptor.compute_with_gradient(atoms, 0)
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_allclose(old_values, expected_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(source_jacobian, 0., rtol=0., atol=1e-12)
    assert np.linalg.norm(expected_jacobian) > 1

    fractional = atoms.copy()
    fractional.to_fractional()
    returned = []
    for source in (atoms, fractional):
        positions, cell, pbc = source.positions.copy(), source.cell.copy(), source.pbc.copy()
        species, cartesian = list(source.chemmap), source._impl.cartesian
        values, jacobian = descriptor.compute_with_strain_gradient(source, 0)
        assert values.shape == (1,)
        assert jacobian.shape == (1, 3, 3)
        assert values.dtype == jacobian.dtype == np.float64
        assert values.flags.owndata and jacobian.flags.owndata
        np.testing.assert_allclose(values, expected_value, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(source.positions, positions)
        np.testing.assert_array_equal(source.cell, cell)
        np.testing.assert_array_equal(source.pbc, pbc)
        assert source.chemmap == species
        assert source._impl.cartesian == cartesian
        returned.append((values, jacobian))

    for step in (2e-6, 1e-6):
        for a, b in ((0, 0), (0, 1)):
            plus = _affine_strain_copy(atoms, a, b, step)
            minus = _affine_strain_copy(atoms, a, b, -step)
            finite_difference = (descriptor.compute(plus)[0] - descriptor.compute(minus)[0]) / (2 * step)
            np.testing.assert_allclose(finite_difference, expected_jacobian[:, a, b],
                                       rtol=2e-7, atol=2e-9)
    for first, second in zip(*returned):
        assert not np.shares_memory(first, second)
        first[...] = -123
    next_values, next_jacobian = descriptor.compute_with_strain_gradient(atoms, 0)
    np.testing.assert_allclose(next_values, expected_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(next_jacobian, expected_jacobian, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(returned[1], (expected_value, expected_jacobian)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(IndexError, match='center'):
        descriptor.compute_with_strain_gradient(atoms, len(atoms))


def test_self_image_strain_reconstructs_linear_potential_virial(tilted_self_image_strain_case):
    params, atoms, expected_value, expected_jacobian = tilted_self_image_strain_case
    settings = {'pottype': 'descriptor_linear',
                'descriptor_linear:descriptor:dtype': 'behler',
                'descriptor_linear:species_count': '1'}
    settings.update({'descriptor_linear:descriptor:' + key: value for key, value in params.items()})
    potential = imrmlp.Potential.from_settings(settings)
    potential.coefficients = [.7]
    reference = atoms.copy()
    potential.calculate(reference)
    expected_energy = .7 * expected_value[0]
    expected_virial = -.7 * expected_jacobian[0]
    assert np.linalg.norm(expected_virial) > 1e-6
    np.testing.assert_allclose(reference.energy, expected_energy, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(reference.site_energies, [expected_energy], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(reference.forces, 0., rtol=0., atol=1e-12)
    np.testing.assert_allclose(reference.stress, expected_virial, rtol=1e-12, atol=1e-12)
    _, jacobian = Descriptor('behler', params).compute_with_strain_gradient(atoms, 0)
    np.testing.assert_allclose(-.7 * jacobian[0], reference.stress, rtol=1e-12, atol=1e-12)
    for step in (2e-6, 1e-6):
        for a, b in ((0, 0), (0, 1)):
            plus = _affine_strain_copy(atoms, a, b, step)
            minus = _affine_strain_copy(atoms, a, b, -step)
            derivative = (potential.calculate_energy(plus) - potential.calculate_energy(minus)) / (2 * step)
            np.testing.assert_allclose(-derivative, expected_virial[a, b], rtol=2e-7, atol=2e-9)




def _averaged_gradient_atoms():
    return Atoms(numbers=[1, 2, 1], elements=['H', 'He'],
                 positions=[[.2, .3, .4], [1.35, .5, .6], [.6, 2.7, .9]],
                 cell=np.eye(3) * 12, pbc=False)


def _check_averaged_gradient_fd(descriptor, atoms, *, atol=3e-6):
    positions = atoms.positions.copy()
    values, jacobian = descriptor.compute_averaged_with_gradient(atoms)
    assert values.shape == (descriptor.feature_size,)
    assert jacobian.shape == (descriptor.feature_size, len(atoms), 3)
    assert values.dtype == jacobian.dtype == np.float64
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(jacobian))
    np.testing.assert_allclose(values, descriptor.compute_averaged(atoms),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jacobian.sum(axis=1), 0., rtol=0., atol=1e-11)
    for step in [2e-5, 1e-5]:
        for source in range(len(atoms)):
            for axis in range(3):
                plus, minus = atoms.copy(), atoms.copy()
                shifted = positions.copy()
                shifted[source, axis] += step
                plus.positions = shifted
                shifted = positions.copy()
                shifted[source, axis] -= step
                minus.positions = shifted
                finite_difference = (descriptor.compute_averaged(plus) -
                                     descriptor.compute_averaged(minus)) / (2 * step)
                np.testing.assert_allclose(jacobian[:, source, axis], finite_difference,
                                           rtol=3e-5, atol=atol,
                                           err_msg=f'h={step}, source={source}, axis={axis}')
    np.testing.assert_array_equal(atoms.positions, positions)
    return values, jacobian


def _local_source_jacobian(descriptor, atoms, center):
    values, indices, gradient = descriptor.compute_with_gradient(atoms, center)
    full = np.zeros((descriptor.feature_size, len(atoms), 3))
    full[:, indices, :] = gradient
    return values, full


def test_averaged_gradient_generic_is_local_mean_and_matches_global_fd():
    descriptor = Descriptor('behler', {
        'behler:dispcount': '3', 'behler:dispselect': '1 2 4',
        'behler:cutoff': '4', 'behler:eta': '.3',
    })
    atoms = _averaged_gradient_atoms()
    values, jacobian = _check_averaged_gradient_fd(descriptor, atoms)
    local = [_local_source_jacobian(descriptor, atoms, center) for center in range(len(atoms))]
    np.testing.assert_allclose(values, np.mean([item[0] for item in local], axis=0),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jacobian, np.mean([item[1] for item in local], axis=0),
                               rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(jacobian) > 1e-6











def _check_averaged_strain_fd(descriptor, atoms, *, atol=3e-6):
    values, jacobian = descriptor.compute_averaged_with_strain_gradient(atoms)
    assert values.shape == (descriptor.feature_size,)
    assert jacobian.shape == (descriptor.feature_size, 3, 3)
    assert values.dtype == jacobian.dtype == np.float64
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(jacobian))
    np.testing.assert_allclose(values, descriptor.compute_averaged(atoms),
                               rtol=1e-12, atol=1e-12)
    for step in [2e-5, 1e-5]:
        for a in range(3):
            for b in range(3):
                plus = _affine_strain_copy(atoms, a, b, step)
                minus = _affine_strain_copy(atoms, a, b, -step)
                finite_difference = (descriptor.compute_averaged(plus) -
                                     descriptor.compute_averaged(minus)) / (2 * step)
                np.testing.assert_allclose(jacobian[:, a, b], finite_difference,
                                           rtol=4e-5, atol=atol,
                                           err_msg=f'h={step}, a={a}, b={b}')
    return values, jacobian


@pytest.mark.parametrize('dtype,params', [('behler', {'behler:dispcount': '3', 'behler:dispselect': '1 2 4',
                'behler:cutoff': '4', 'behler:eta': '.3'})], ids=['behler'])
def test_averaged_strain_is_local_mean_and_matches_global_affine_fd(dtype, params):
    descriptor = Descriptor(dtype, params)
    atoms = _averaged_gradient_atoms()
    values, jacobian = _check_averaged_strain_fd(descriptor, atoms)
    local = [descriptor.compute_with_strain_gradient(atoms, center)
             for center in range(len(atoms))]
    np.testing.assert_allclose(values, np.mean([item[0] for item in local], axis=0),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jacobian, np.mean([item[1] for item in local], axis=0),
                               rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(jacobian) > 1e-6






def test_averaged_strain_self_images_match_independent_g1_reference(tilted_self_image_strain_case):
    params, atoms, expected_value, expected_jacobian = tilted_self_image_strain_case
    values, jacobian = Descriptor('behler', params).compute_averaged_with_strain_gradient(atoms)
    np.testing.assert_allclose(values, expected_value, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-13, atol=1e-13)
