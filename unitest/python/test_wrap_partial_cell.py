"""Public wrapping of ASE partial cells without an artificial open lattice row."""

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
pytest.importorskip('ase')
from ase import Atoms as ASEAtoms
from imrmlp import Atoms


def test_wrap_ase_slab_with_zero_open_row():
    source = ASEAtoms(
        'Al', positions=[[12.0, -1.0, 3.0]], cell=[10.0, 10.0, 0.0], pbc=[True, True, False]
    )
    atoms = Atoms.from_ase(source)
    atoms.wrap()
    np.testing.assert_allclose(atoms.positions, [[2.0, 9.0, 3.0]])
    restored = atoms.to_ase()
    np.testing.assert_allclose(restored.positions, [[2.0, 9.0, 3.0]])
    np.testing.assert_array_equal(restored.pbc, [True, True, False])
    np.testing.assert_array_equal(restored.cell.array, np.diag([10.0, 10.0, 0.0]))
    np.testing.assert_array_equal(source.positions, [[12.0, -1.0, 3.0]])
