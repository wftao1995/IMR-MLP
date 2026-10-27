import numpy as np
import pytest

from imrmlp import Atoms
from imrmlp.io import read, write


@pytest.mark.parametrize("format", ["extxyz", "cfg", "bin"])
def test_hvp_roundtrip(tmp_path, format):
    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [1.2, 0, 0]],
                  cell=np.eye(3) * 8, elements=["H"])
    directions = np.arange(12, dtype=float).reshape(6, 2) / 7
    targets = 1.3 * directions
    atoms.set_hvp(directions, targets)
    path = tmp_path / f"hvp.{format}"
    write(path, [atoms], format=format)
    restored = read(path, format=format)[0]
    np.testing.assert_array_equal(restored.hvp_directions, directions)
    np.testing.assert_array_equal(restored.hvp_targets, targets)


def test_hvp_validation_is_atomic():
    atoms = Atoms(numbers=[1], positions=[[0, 0, 0]], cell=np.eye(3) * 8)
    directions = np.ones((3, 1))
    atoms.set_hvp(directions, 2 * directions)
    for bad in [np.ones((2, 1)), np.full((3, 1), np.nan), np.zeros((3, 1))]:
        with pytest.raises(ValueError):
            atoms.set_hvp(bad, directions)
        np.testing.assert_array_equal(atoms.hvp_targets, 2 * directions)
    atoms.clear_hvp()
    assert atoms.hvp_directions.size == atoms.hvp_targets.size == 0
