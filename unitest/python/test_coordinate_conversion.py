"""Public coordinate conversions respect the stored representation."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, read_one

CELL = np.array([[4., 0., 0.], [1., 5., 0.], [.2, .4, 6.]])
POSITIONS = np.array([[1., 2., 3.], [-.4, 5.6, 7.8]])
FRACTIONS = np.linalg.solve(CELL.T, POSITIONS.T).T


def test_cartesian_reader_output_is_safe_to_normalize(tmp_path):
    text = ('2\nLattice="4 0 0 1 5 0 .2 .4 6" pbc="T T F" '
            'Properties=species:S:1:pos:R:3\n'
            'H 1 2 3\nH -.4 5.6 7.8\n')
    path = tmp_path / 'coordinates.xyz'
    path.write_text(text)
    atoms = read_one(path)
    assert atoms._impl.cartesian
    atoms.to_cartesian()
    atoms.to_cartesian()
    np.testing.assert_array_equal(atoms.positions, POSITIONS)
    np.testing.assert_array_equal(atoms.cell, CELL)
    np.testing.assert_array_equal(atoms.pbc, [True, True, False])
    assert path.read_text() == text


@pytest.mark.parametrize('initial_fractional', [False, True])
def test_repeated_skew_coordinate_conversion_is_idempotent_and_roundtrips(initial_fractional):
    source = Atoms(numbers=[1, 1], positions=POSITIONS, cell=CELL, pbc=[True, True, False])
    source.energy = -2.
    source.forces = [[1., 2., 3.], [4., 5., 6.]]
    if initial_fractional:
        source.to_fractional()
    original_positions = source.positions.copy()
    atoms = source.copy()

    atoms.to_fractional()
    np.testing.assert_allclose(atoms.positions, FRACTIONS, rtol=1e-14, atol=1e-14)
    first_fractional = atoms.positions.copy()
    atoms.to_fractional()
    np.testing.assert_array_equal(atoms.positions, first_fractional)
    assert not atoms._impl.cartesian

    atoms.to_cartesian()
    np.testing.assert_allclose(atoms.positions, POSITIONS, rtol=1e-14, atol=1e-14)
    first_cartesian = atoms.positions.copy()
    atoms.to_cartesian()
    np.testing.assert_array_equal(atoms.positions, first_cartesian)
    assert atoms._impl.cartesian
    np.testing.assert_array_equal(atoms.cell, CELL)
    np.testing.assert_array_equal(atoms.forces, [[1., 2., 3.], [4., 5., 6.]])
    assert atoms.energy == -2.
    np.testing.assert_array_equal(source.positions, original_positions)
    assert source._impl.cartesian == (not initial_fractional)


def test_cartesian_molecule_without_cell_is_unchanged_by_normalization():
    atoms = Atoms(numbers=[1, 1], positions=POSITIONS, pbc=False)
    atoms.to_cartesian()
    np.testing.assert_array_equal(atoms.positions, POSITIONS)
    assert atoms._impl.cartesian
