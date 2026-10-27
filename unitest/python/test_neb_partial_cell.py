"""Fixed partial-periodic NEB needs periodic rows, not an artificial open row."""
import itertools
import numpy as np
import pytest
from imrmlp import Atoms, NEB, Potential


@pytest.mark.parametrize('open_axis', [1, 2])
@pytest.mark.parametrize('init_method', ['linear', 'idpp'])
def test_slab_neb_matches_normal_padding_and_preserves_normal_motion(open_axis, init_method):
    a = np.array([8., 0., 1.])
    b = np.array([.4, 9., .2])
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)
    cell = np.zeros((3, 3))
    cell[0], cell[2 if open_axis == 1 else 1] = a, b
    pbc = [True, True, True]
    pbc[open_axis] = False
    p0 = .95 * a + .3 * b + 2 * normal
    p1 = .05 * a + .3 * b + 2.2 * normal
    r0 = np.array([p0, p0 + 1.3 * normal])
    r1 = np.array([p1, p1 + 1.4 * (.8 * normal + .6 * a / np.linalg.norm(a))])
    first = Atoms(numbers=[1, 1], elements=['H'], positions=r0, cell=cell, pbc=pbc)
    last = Atoms(numbers=[1, 1], elements=['H'], positions=r1, cell=cell, pbc=pbc)
    first.chemmap = last.chemmap = [0, 0]
    source = first.copy(), last.copy()
    padded_first, padded_last = first.copy(), last.copy()
    padded = cell.copy()
    padded[open_axis] = 12 * normal
    padded_first.cell = padded_last.cell = padded
    potential = Potential.from_settings({'pottype': 'lj612', 'ntypes': 1,
                                         '0-0': '1 1', 'cut:0-0': 3})
    driver = NEB(potential, n_images=3, max_iter=3, max_move=.01, init_method=init_method)
    driver._impl.pot = potential.inner
    driver._impl.init([first._impl, Atoms()._impl, last._impl])
    middle = Atoms(_impl=driver._impl.images[1].copy(), elements=['H'])
    nearest = np.array([min((delta - i * a - j * b
                            for i, j in itertools.product(range(-2, 3), repeat=2)),
                           key=lambda item: item @ item) for delta in r1 - r0])
    if init_method == 'linear':
        np.testing.assert_allclose(middle.positions, r0 + .5 * nearest, atol=1e-11, rtol=0)
    assert (middle.positions[0] - r0[0]) @ normal > .01
    actual = driver.run(first, last)
    expected = driver.run(padded_first, padded_last)
    assert actual.stop_reason == expected.stop_reason
    assert actual.n_iter == expected.n_iter
    assert actual.max_neb_force == pytest.approx(expected.max_neb_force, abs=1e-8)
    for image, control in zip(actual.images, expected.images):
        np.testing.assert_array_equal(image.cell, cell)
        np.testing.assert_array_equal(image.pbc, pbc)
        np.testing.assert_allclose(image.positions, control.positions, atol=1e-8, rtol=0)
        assert np.isfinite(image.energy)
        recalculated = image.copy()
        potential.calculate(recalculated)
        assert image.energy == pytest.approx(recalculated.energy, abs=1e-10)
    for endpoint, original in zip((first, last), source):
        np.testing.assert_array_equal(endpoint.positions, original.positions)
        np.testing.assert_array_equal(endpoint.cell, original.cell)
        np.testing.assert_array_equal(endpoint.pbc, original.pbc)
