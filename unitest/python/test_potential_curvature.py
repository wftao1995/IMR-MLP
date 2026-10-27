"""Public Cartesian curvature queries, using independent pair and force oracles."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Potential


def inputs():
    potential = Potential.from_settings({
        'pottype': 'lj612', 'ntypes': 1, '0-0': '1 1', 'cut:0-0': 3,
    })
    atoms = Atoms(numbers=[1, 1], elements=['H'],
                  positions=[[2., 2., 2.], [3.3, 2., 2.]],
                  cell=np.eye(3) * 20, pbc=False)
    return potential, atoms


def pair_hessian(r):
    transverse = 24 * (r**-8 - 2*r**-14)
    radial = 24 * (26*r**-14 - 7*r**-8)
    k = np.diag([radial, transverse, transverse])
    return np.block([[k, -k], [-k, k]])


@pytest.mark.parametrize('method', ['auto', 'analytic', 'force_fd'])
def test_lj_queries_use_cartesian_curvature_and_independent_arrays(method):
    potential, atoms = inputs()
    exact = pair_hessian(1.3)
    direction = np.linspace(-.7, .9, 6)
    h = potential.hessian(atoms, method=method, step=1e-5)
    hv = potential.hessian_vector_product(atoms, direction, method=method, step=1e-5)
    assert h.shape == (6, 6) and hv.shape == (6,)
    np.testing.assert_allclose(h, exact, atol=2e-6, rtol=1e-7)
    np.testing.assert_allclose(hv, exact @ direction, atol=2e-6, rtol=1e-7)
    translation = np.tile([.2, -.3, .4], 2)
    np.testing.assert_allclose(potential.hessian_vector_product(atoms, translation, method=method),
                               0, atol=1e-10)
    assert potential.supports_analytical_hessian()
    assert potential.supports_analytical_hessian_vector_product()
    saved = h.copy(), hv.copy()
    potential.hessian(atoms)
    potential.hessian_vector_product(atoms, direction)
    del atoms, potential
    np.testing.assert_array_equal(h, saved[0])
    np.testing.assert_array_equal(hv, saved[1])


@pytest.mark.parametrize('fractional', [False, True])
def test_curvature_queries_preserve_source_geometry_results_and_labels(fractional):
    potential, atoms = inputs()
    potential.calculate(atoms)
    direction = np.linspace(-.7, .9, 6)
    atoms.set_hvp(direction[:, None], (pair_hessian(1.3) @ direction)[:, None])
    if fractional:
        atoms.to_fractional()
    saved = atoms.copy()
    for method in ('auto', 'force_fd'):
        np.testing.assert_allclose(potential.hessian(atoms, method=method), pair_hessian(1.3), atol=2e-4)
        np.testing.assert_allclose(potential.hessian_vector_product(atoms, direction, method=method),
                                   pair_hessian(1.3) @ direction, atol=2e-4)
    np.testing.assert_array_equal(atoms.positions, saved.positions)
    np.testing.assert_array_equal(atoms.cell, saved.cell)
    np.testing.assert_array_equal(atoms.pbc, saved.pbc)
    assert atoms._impl.cartesian == saved._impl.cartesian
    assert atoms.energy == saved.energy
    np.testing.assert_array_equal(atoms.forces, saved.forces)
    np.testing.assert_array_equal(atoms.stress, saved.stress)
    np.testing.assert_array_equal(atoms.hvp_directions, saved.hvp_directions)
    np.testing.assert_array_equal(atoms.hvp_targets, saved.hvp_targets)


def test_behler_descriptor_queries_use_native_curvature():
    _, atoms = inputs()
    potential = Potential.from_settings({
        'pottype': 'descriptor_linear', 'descriptor_linear:descriptor:dtype': 'behler',
        'descriptor_linear:descriptor:behler:dispcount': 1,
        'descriptor_linear:descriptor:behler:dispselect': '1',
        'descriptor_linear:descriptor:behler:cutoff': 3,
        'descriptor_linear:species_count': 1,
    })
    potential.coefficients = [1.]
    assert potential.supports_analytical_hessian()
    assert potential.supports_analytical_hessian_vector_product()
    direction = np.linspace(-.7, .9, 6)
    h = potential.hessian(atoms)
    hv = potential.hessian_vector_product(atoms, direction)
    # E(r)=2*fcut(r)=1+cos(pi*r/3), independent Behler G1 pair oracle.
    r = 1.3
    k = np.diag([-(np.pi/3)**2*np.cos(np.pi*r/3),
                 -np.pi/3*np.sin(np.pi*r/3)/r, -np.pi/3*np.sin(np.pi*r/3)/r])
    exact = np.block([[k, -k], [-k, k]])
    np.testing.assert_allclose(h, exact, atol=2e-7)
    np.testing.assert_allclose(hv, exact @ direction, atol=2e-7)
    delta = 1e-5 / np.linalg.norm(direction.reshape(-1, 3), axis=1).max()
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions = plus.positions + delta * direction.reshape(-1, 3)
    minus.positions = minus.positions - delta * direction.reshape(-1, 3)
    potential.calculate(plus)
    potential.calculate(minus)
    fd = -(plus.forces - minus.forces).reshape(-1)/(2*delta)
    np.testing.assert_allclose(potential.hessian_vector_product(atoms, direction, step=1e-5), fd, atol=1e-10)
    np.testing.assert_allclose(potential.hessian(atoms, method='analytic'), exact, atol=2e-7)
    np.testing.assert_allclose(potential.hessian_vector_product(atoms, direction, method='analytic'),
                               exact @ direction, atol=2e-7)
    np.testing.assert_array_equal(potential.hessian_vector_product(atoms, np.zeros(6)), np.zeros(6))


@pytest.mark.parametrize('step', [0, -1, np.inf, np.nan])
def test_query_step_validation(step):
    potential, atoms = inputs()
    for query in (lambda: potential.hessian(atoms, step=step),
                  lambda: potential.hessian_vector_product(atoms, np.ones(6), step=step)):
        with pytest.raises(ValueError, match='step'):
            query()


@pytest.mark.parametrize('direction', [np.ones(5), np.ones((6, 1)), np.full(6, np.nan)])
def test_query_direction_validation(direction):
    potential, atoms = inputs()
    with pytest.raises(Exception, match='direction'):
        potential.hessian_vector_product(atoms, direction)


def test_query_methods_species_and_precise_small_step():
    potential, atoms = inputs()
    with pytest.raises(ValueError, match='method'):
        potential.hessian(atoms, method='energy_fd')
    with pytest.raises(ValueError, match='method'):
        potential.hessian_vector_product(atoms, np.ones(6), method='hybrid')
    assert potential.hessian(atoms, method='analytic', step=1e-12).shape == (6, 6)
    atoms.chemmap = [0, 1]
    for query in (lambda: potential.hessian(atoms),
                  lambda: potential.hessian_vector_product(atoms, np.ones(6))):
        with pytest.raises(ValueError, match='species'):
            query()

@pytest.mark.parametrize('pbc', [False, (True, True, False), True])
@pytest.mark.parametrize('method', ['auto', 'analytic'])
def test_mtp_curvature_queries_accept_molecules_and_slabs(pbc, method):
    from pathlib import Path
    model = Path(__file__).resolve().parents[1] / 'test-data' / 'portable-mtp' / 'nonlinear.mtp'
    potential = Potential(str(model), 'mtp')
    atoms = Atoms(numbers=[13, 28, 13], elements=['Al', 'Ni'],
                  positions=[[.4, .6, .5], [2.7, .9, .7], [2.3, 3.4, 1.1]],
                  cell=[[12., .6, .2], [.1, 13., .3], [.2, .1, 14.]], pbc=pbc)
    potential.calculate(atoms)
    saved = atoms.copy()
    direction = np.linspace(-.4, .7, 9)
    result = potential.hessian(atoms, method=method)
    product = potential.hessian_vector_product(atoms, direction, method=method)
    np.testing.assert_allclose(product, result @ direction, atol=1e-8, rtol=1e-8)
    for step in (1e-4, 5e-5):
        oracle = np.empty((9, 9))
        for column in range(9):
            delta = np.zeros((3, 3))
            delta.flat[column] = step
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions = atoms.positions + delta
            minus.positions = atoms.positions - delta
            potential.calculate(plus)
            potential.calculate(minus)
            oracle[:, column] = -(plus.forces - minus.forces).reshape(-1) / (2 * step)
        np.testing.assert_allclose(result, oracle, atol=2e-5, rtol=2e-6)
    np.testing.assert_array_equal(atoms.positions, saved.positions)
    np.testing.assert_array_equal(atoms.cell, saved.cell)
    np.testing.assert_array_equal(atoms.pbc, saved.pbc)
    assert atoms.energy == saved.energy
    np.testing.assert_array_equal(atoms.forces, saved.forces)
    np.testing.assert_array_equal(atoms.stress, saved.stress)

@pytest.mark.parametrize('model_head', [False, True])
@pytest.mark.parametrize('step', [1e-3, 5e-4])
def test_les_native_curvature_and_requested_force_difference_step(tmp_path, model_head, step):
    from pathlib import Path
    import re
    fixture = Path(__file__).resolve().parents[1] / 'test-data' / 'portable-mtp' / 'les.mtp'
    original = fixture.read_text()
    count = int(re.search(r'alpha_scalar_moments\s*=\s*(\d+)', original).group(1))
    charge_moments = [.35, -.21, 0., .12] + [0.] * (count - 4)
    charges = ('les_species_coeffs = {.23,-.17}\nles_moment_coeffs = {'
               + ','.join(map(str, charge_moments)) + '}\n')
    # Match the versioned Coulomb fields written by the native LES Save APIs.
    metadata = ('les_version = 2\nles_kernel = coulomb\nles_sigma = .5\n'
                'les_dl = 2\nles_cutoff_mode = smooth\nles_taper_fraction = .1\n'
                'les_charge_treatment = uniform_background\n'
                'remove_self_interaction = 1\nles_response_parity = 1\n')
    if model_head:
        body = original.split('species_coeffs', 1)[0]
        body += 'species_coeffs = {0,0,0,0,0,0}\n'
        body += 'moment_coeffs = {' + ','.join(['.001'] * (2 * count)) + '}\n'
        body += 'model_coeffs = {.1,.2,.3,.4,.5}\n' + charges + metadata
    else:
        # Keep the classic fitted core, replacing its unversioned LES footer.
        body = re.split(r'(?m)^lr_sigma\s*=', original, maxsplit=1)[0] + charges + metadata
    model = tmp_path / 'les.mtp'
    model.write_text(body)
    settings = {'pottype': 'mtp_m' if model_head else 'mtp_les', 'potfile': model}
    if model_head:
        settings['extmodel'] = 'mtp_m_q2'
    potential = Potential.from_settings(settings)
    assert potential.supports_analytical_hessian()
    assert potential.supports_analytical_hessian_vector_product()
    atoms = Atoms(numbers=[13, 28, 13, 28], elements=['Al', 'Ni'],
                  positions=[[2., 2., 2.], [4.1, 2.2, 2.], [2.3, 4.2, 2.4], [5., 4.8, 3.1]],
                  cell=[[10., 0., 0.], [.8, 10.5, 0.], [-.3, .6, 11.]], pbc=True)
    potential.calculate(atoms)
    saved = atoms.copy()
    oracle = np.empty((12, 12))
    for column in range(12):
        delta = np.zeros((4, 3))
        delta.flat[column] = step
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions = atoms.positions + delta
        minus.positions = atoms.positions - delta
        potential.calculate(plus)
        potential.calculate(minus)
        oracle[:, column] = -(plus.forces - minus.forces).reshape(-1) / (2 * step)
    oracle = (oracle + oracle.T) / 2
    assert np.linalg.norm(oracle) > .01
    result = potential.hessian(atoms, method='force_fd', step=step)
    np.testing.assert_allclose(result, oracle, atol=2e-9, rtol=1e-10)
    native = potential.hessian(atoms, method='analytic')
    np.testing.assert_array_equal(potential.hessian(atoms), native)
    np.testing.assert_allclose(native, oracle, atol=2e-6, rtol=2e-5)
    direction = np.linspace(-.4, .7, 12)
    scale = step / np.linalg.norm(direction.reshape(-1, 3), axis=1).max()
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions = atoms.positions + scale * direction.reshape(-1, 3)
    minus.positions = atoms.positions - scale * direction.reshape(-1, 3)
    potential.calculate(plus)
    potential.calculate(minus)
    expected = -(plus.forces - minus.forces).reshape(-1) / (2 * scale)
    np.testing.assert_allclose(potential.hessian_vector_product(atoms, direction, method='force_fd', step=step),
                               expected, atol=2e-9, rtol=1e-10)
    analytic = potential.hessian_vector_product(atoms, direction, method='analytic')
    np.testing.assert_array_equal(potential.hessian_vector_product(atoms, direction), analytic)
    for fd_step in (step, step / 2):
        fd = potential.hessian_vector_product(atoms, direction, method='force_fd', step=fd_step)
        np.testing.assert_allclose(analytic, fd, atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(atoms.positions, saved.positions)
    np.testing.assert_array_equal(atoms.cell, saved.cell)
    np.testing.assert_array_equal(atoms.pbc, saved.pbc)
    assert atoms.energy == saved.energy
    np.testing.assert_array_equal(atoms.forces, saved.forces)
    np.testing.assert_array_equal(atoms.stress, saved.stress)
