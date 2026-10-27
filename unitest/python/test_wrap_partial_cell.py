"""Public wrapping of ASE partial cells without an artificial open lattice row."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
pytest.importorskip('ase')
from ase import Atoms as ASEAtoms
from imrmlp import Atoms


def test_wrap_ase_slab_with_zero_open_row():
    source = ASEAtoms('Al', positions=[[12., -1., 3.]],
                      cell=[10., 10., 0.], pbc=[True, True, False])
    atoms = Atoms.from_ase(source)
    atoms.wrap()
    np.testing.assert_allclose(atoms.positions, [[2., 9., 3.]])
    restored = atoms.to_ase()
    np.testing.assert_allclose(restored.positions, [[2., 9., 3.]])
    np.testing.assert_array_equal(restored.pbc, [True, True, False])
    np.testing.assert_array_equal(restored.cell.array, np.diag([10., 10., 0.]))
    np.testing.assert_array_equal(source.positions, [[12., -1., 3.]])
