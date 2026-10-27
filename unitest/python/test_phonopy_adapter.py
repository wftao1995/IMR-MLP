# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Phonopy owns mappings and fitting; the adapter preserves ASE force data."""
import numpy as np
import pytest

pytest.importorskip('ase')
pytest.importorskip('phonopy')
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from imrmlp.phonopy import calculate_forces


def make_phonon(calculator=None):
    cell = PhonopyAtoms(symbols=['He', 'H', 'He'],
                        cell=[[12, 0, 0], [1, 13, 0], [.5, .3, 14]],
                        scaled_positions=[[.1, .2, .3], [.2, .25, .3], [.3, .3, .4]])
    return Phonopy(cell, [[2, 1, 0], [0, 1, 0], [0, 0, 1]],
                   primitive_matrix=np.eye(3), calculator=calculator)


def test_preserves_displacement_order_species_cell_and_forces(tmp_path):
    phonon = make_phonon()
    phonon.generate_displacements(distance=.001)
    result = calculate_forces(phonon, LennardJones(rc=3))
    assert result.shape == (len(phonon.supercells_with_displacements), 6, 3)
    expected = []
    for cell in phonon.supercells_with_displacements:
        atoms = Atoms(symbols=cell.symbols, positions=cell.positions,
                      cell=cell.cell, pbc=True, calculator=LennardJones(rc=3))
        expected.append(atoms.get_forces())
    np.testing.assert_allclose(result, expected, atol=1e-11)
    np.testing.assert_array_equal(phonon.forces, result)
    phonon.produce_force_constants()
    assert np.isfinite(phonon.force_constants).all()
    # Save/load through Phonopy rather than a second YAML parser in the adapter.
    path = tmp_path / 'phonopy_params.yaml'
    phonon.save(path, settings={'force_constants': True})
    import phonopy
    restored = phonopy.load(path, is_compact_fc=False, symmetrize_fc=False)
    np.testing.assert_allclose(restored.force_constants, phonon.force_constants, atol=1e-12)


def test_requires_displacements_and_angstrom_ev_units():
    with pytest.raises(ValueError, match='displacements'):
        calculate_forces(make_phonon(), LennardJones())
    with pytest.raises(ValueError, match='Angstrom/eV'):
        calculate_forces(make_phonon('qe'), LennardJones())


@pytest.mark.parametrize('failure', ['exception', 'nan', 'shape'])
def test_calculation_failure_preserves_existing_forces(failure):
    phonon = make_phonon()
    phonon.generate_displacements(distance=.001)
    previous = np.ones((len(phonon.supercells_with_displacements), 6, 3))
    phonon.forces = previous

    class FailingCalculator(Calculator):
        implemented_properties = ['forces']
        calls = 0

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            if self.calls == 2 and failure == 'exception':
                raise RuntimeError('force calculation failed')
            shape = (1, 3) if self.calls == 2 and failure == 'shape' else (len(atoms), 3)
            self.results['forces'] = np.zeros(shape)
            if self.calls == 2 and failure == 'nan':
                self.results['forces'][0, 0] = np.nan

    with pytest.raises((ValueError, RuntimeError), match='forces|calculation failed'):
        calculate_forces(phonon, FailingCalculator())
    np.testing.assert_array_equal(phonon.forces, previous)


def test_native_calculator_fits_known_lj_hessian(tmp_path, monkeypatch):
    pytest.importorskip('imrmlp._imrmlp')
    from imrmlp.calculator import IMRMLPCalculator

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'index.map').write_text('H\n')
    potential = tmp_path / 'lj.ini'
    potential.write_text('pot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 4\n')
    r = 1.3
    cell = PhonopyAtoms(symbols=['H', 'H'], cell=np.eye(3)*20,
                        positions=[[2, 2, 2], [2+r, 2, 2]])
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix=np.eye(3),
                     calculator='vasp', is_symmetry=False)
    phonon.generate_displacements(distance=1e-5, is_plusminus=True)
    calculate_forces(phonon, IMRMLPCalculator(filename=potential, pottype='lj612', elements=['H']))
    phonon.produce_force_constants()
    radial = 624/r**14 - 168/r**8
    transverse = -48/r**14 + 24/r**8
    spring = np.diag([radial, transverse, transverse])
    expected = np.array([[spring, -spring], [-spring, spring]])
    np.testing.assert_allclose(phonon.force_constants, expected, rtol=2e-7, atol=1e-7)
