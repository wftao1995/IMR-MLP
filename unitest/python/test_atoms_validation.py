# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Array validation and Python protocols for atomic structures."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, LinearTrainer
from imrmlp._imrmlp import _Atoms, _LinearTrainer


@pytest.mark.parametrize('field', ['positions', 'forces', 'velocities'])
@pytest.mark.parametrize('columns', [0, 1, 2, 4])
def test_vector_arrays_require_three_columns_before_mutation(field, columns):
    atom = Atoms(positions=[[1., 2., 3.], [4., 5., 6.]])
    original = np.arange(6., dtype=float).reshape(2, 3)
    setattr(atom._impl, field, original)
    with pytest.raises(ValueError, match='shape'):
        setattr(atom._impl, field, np.ones((2, columns)))
    np.testing.assert_array_equal(getattr(atom._impl, field), original)


@pytest.mark.parametrize('field', ['forces', 'velocities'])
def test_per_atom_vectors_require_matching_rows_and_allow_clearing(field):
    atom = Atoms(positions=np.ones((2, 3)))
    with pytest.raises(ValueError, match='shape'):
        setattr(atom._impl, field, np.ones((1, 3)))
    setattr(atom._impl, field, np.ones((2, 3)))
    assert getattr(atom._impl, field).shape == (2, 3)
    setattr(atom._impl, field, np.empty((0, 3)))
    assert getattr(atom._impl, field).shape == (0, 3)


def test_public_atoms_constructor_rejects_invalid_coordinates():
    with pytest.raises(ValueError, match='shape'):
        Atoms(positions=[[1., 2.]])


def test_public_velocities_assignment_copy_zeros_and_clear():
    atom = Atoms(numbers=[1, 8], positions=np.zeros((2, 3)))
    assert atom.velocities is None
    values = np.arange(6).reshape(2, 3)
    atom.velocities = values
    expected = values.copy()
    values[:] = -1
    np.testing.assert_array_equal(atom.velocities, expected)
    returned = atom.velocities
    returned[:] = -2
    np.testing.assert_array_equal(atom.velocities, expected)

    atom.velocities = [[0, 0, 0], [0, 0, 0]]
    np.testing.assert_array_equal(atom.velocities, np.zeros((2, 3)))
    assert atom._impl.has_velocities
    atom.velocities = None
    assert atom.velocities is None
    assert not atom._impl.has_velocities
    atom.velocities = expected
    atom.velocities = np.empty((0, 3))
    assert atom.velocities is None


@pytest.mark.parametrize('shape', [(1, 3), (3, 3), (2, 2), (2, 4), (0, 2)])
def test_public_velocities_reject_shape_before_mutation(shape):
    atom = Atoms(positions=np.zeros((2, 3)))
    original = np.arange(6.).reshape(2, 3)
    atom.velocities = original
    with pytest.raises(ValueError, match='shape'):
        atom.velocities = np.ones(shape)
    np.testing.assert_array_equal(atom.velocities, original)


def test_public_velocities_extxyz_and_ase_roundtrip(tmp_path):
    from imrmlp import read_one, write_one

    atom = Atoms(numbers=[1, 8], positions=[[0., 0., 0.], [1., 0., 0.]],
                 cell=np.eye(3) * 10)
    expected = np.array([[1.25, -2.5, 3.], [0., 4., -.75]])
    atom.velocities = expected
    path = tmp_path / 'velocities.extxyz'
    write_one(path, atom)
    np.testing.assert_allclose(read_one(path).velocities, expected)

    pytest.importorskip('ase')
    from ase.units import fs
    atom.masses = [2., 18.]
    converted = atom.to_ase()
    np.testing.assert_allclose(converted.get_velocities(), expected / (1000 * fs))
    restored = Atoms.from_ase(converted, read_results=False)
    np.testing.assert_allclose(restored.velocities, expected)
    np.testing.assert_allclose(restored.masses, atom.masses)


def test_atoms_length_and_representation():
    atom = Atoms(numbers=[1, 8], positions=[[0., 0., 0.], [1., 0., 0.]],
                 cell=np.eye(3) * 10)
    atom.energy = -1.25
    assert len(atom) == 2
    assert repr(atom) == "Atoms(natoms=2, species=['H', 'O'], vol=1000.0, E=-1.2500)"
    assert len(Atoms()) == 0
    assert repr(Atoms()) == 'Atoms(natoms=0, species=[])'


def test_uninitialized_linear_trainer_raises_instead_of_crashing(tmp_path):
    trainer = _LinearTrainer()
    for operation in (trainer.assemble, trainer.solve, trainer.train,
                      lambda: trainer.save(str(tmp_path / 'uninitialized.mtp'))):
        with pytest.raises(RuntimeError, match='initialized'):
            operation()
    with pytest.raises(ValueError, match='potential'):
        trainer.init([_Atoms()], None)
    with pytest.raises(RuntimeError, match='assembled'):
        LinearTrainer().solve()
    assert not (tmp_path / 'uninitialized.mtp').exists()


def test_initial_training_statistics_are_defined():
    trainer = _LinearTrainer()
    assert trainer.train_erms == 0
    assert trainer.train_frms == 0
    assert trainer.train_srms == 0
