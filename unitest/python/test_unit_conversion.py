"""Explicit numeric unit API; wrapping and import integration are tested separately."""
import numpy as np
import pytest
from imrmlp import _imrmlp
from imrmlp import Atoms


def test_typed_numeric_conversion():
    factor = _imrmlp.unit_conversion_factor
    assert factor("length", "Bohr", "A") == pytest.approx(.529177210903)
    assert factor("energy", "Hartree", "eV") == pytest.approx(27.211396641308)
    assert factor("time", "fs", "ns") == pytest.approx(1e-6)
    assert factor("pressure", "GPa", "bar") == pytest.approx(1e4)
    assert factor("length", "A", "nm") * factor("length", "nm", "A") == pytest.approx(1.)
    for quantity, source, target in [("length", "eV", "A"),
                                     ("virial", "GPa", "eV"),
                                     ("stress", "GPa", "eV")]:
        with pytest.raises(ValueError):
            factor(quantity, source, target)


def test_explicit_canonical_copy_preserves_source():
    atoms = Atoms(numbers=[1], positions=[[1., 2., 3.]], cell=np.eye(3))
    atoms.energy = 2.
    atoms.forces = [[1., 1., 1.]]
    atoms.velocities = [[1., 1., 1.]]
    atoms.stress = np.eye(3) * 3.
    atoms.set_hvp(np.ones((3, 1)), np.ones((3, 1)))
    converted = _imrmlp.canonicalize_atoms(atoms._impl, length="nm", energy="Hartree", time="fs")
    energy = _imrmlp.unit_conversion_factor("energy", "Hartree", "eV")
    np.testing.assert_array_equal(atoms.positions, [[1., 2., 3.]])
    np.testing.assert_allclose(converted.positions, [[10., 20., 30.]])
    np.testing.assert_allclose(converted.cell, np.eye(3) * 10.)
    assert converted.energy == 2. * energy
    np.testing.assert_allclose(converted.stress, np.eye(3) * 3. * energy)
    np.testing.assert_allclose(converted.forces, np.ones((1, 3)) * energy / 10.)
    np.testing.assert_allclose(converted.velocities, np.ones((1, 3)) * 1e4)
    np.testing.assert_array_equal(converted.hvp_directions, np.ones((3, 1)))
    np.testing.assert_allclose(converted.hvp_targets, np.ones((3, 1)) * energy / 100.)
    with pytest.raises(ValueError):
        _imrmlp.canonicalize_atoms(atoms._impl, mass="eV")
    np.testing.assert_array_equal(atoms.positions, [[1., 2., 3.]])


def test_fractional_and_missing_labels():
    atoms = Atoms(numbers=[1], positions=[[.2, .3, .4]], cell=np.eye(3))
    atoms._impl.to_fractional()
    converted = _imrmlp.canonicalize_atoms(atoms._impl, length="nm")
    np.testing.assert_allclose(converted.positions, [[.2, .3, .4]])
    assert not converted.has_energy
    assert not converted.has_stress
    assert not converted.has_forces


def test_pressure_virial_explicit_contract():
    factor = _imrmlp.pressure_to_virial_factor("GPa", 20., 1)
    assert factor == pytest.approx(20. * _imrmlp.unit_conversion_factor("pressure", "GPa", "eV/A^3"))
    assert _imrmlp.pressure_to_virial_factor("GPa", 20., -1) == -factor
    for volume, sign in [(0., 1), (float("nan"), 1), (20., 0)]:
        with pytest.raises(ValueError):
            _imrmlp.pressure_to_virial_factor("GPa", volume, sign)
    assert _imrmlp.source_unit_system("atomic")["length"] == "Bohr"
