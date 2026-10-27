"""Fixed-cell relaxation of ordinary ASE molecules does not need padding."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
pytest.importorskip('ase')
from ase import Atoms as ASEAtoms
from imrmlp import Atoms, Potential, Relaxation


def lj_energy(radius):
    return 4 * ((1 / radius) ** 12 - (1 / radius) ** 6)


@pytest.mark.parametrize('optimizer', ['fire', 'lbfgs'])
@pytest.mark.parametrize('budget', [0, 3])
def test_ase_molecule_relaxation_matches_padding_and_keeps_physical_cell(optimizer, budget):
    potential = Potential.from_settings({'pottype': 'lj612', 'ntypes': 1,
                                         '0-0': '1 1', 'cut:0-0': 3})
    source = ASEAtoms('H2', positions=[[0., 0., 0.], [1.3, 0., 0.]], pbc=False)
    atoms = Atoms.from_ase(source)
    padded = atoms.copy(); padded.cell = np.eye(3) * 20
    for item in (atoms, padded):
        item.chemmap = [0, 0]
    relaxation = Relaxation(potential, driver=optimizer, max_iter=budget, gtol=1e-10, max_move=.02)
    result = relaxation.run(atoms)
    control = relaxation.run(padded)  # reuse the same native driver
    assert result.final_atoms is atoms
    assert result.init_energy == pytest.approx(lj_energy(1.3), abs=1e-12)
    assert np.isfinite(result.final_energy)
    assert result.final_energy == pytest.approx(control.final_energy, abs=1e-10)
    np.testing.assert_allclose(atoms.positions, padded.positions, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(atoms.cell, np.zeros((3, 3)))
    np.testing.assert_array_equal(atoms.pbc, [False] * 3)
    radius = np.linalg.norm(atoms.positions[1] - atoms.positions[0])
    assert atoms.energy == pytest.approx(lj_energy(radius), abs=1e-10)
    assert result.n_iter <= budget
    np.testing.assert_array_equal(source.positions, [[0., 0., 0.], [1.3, 0., 0.]])
    np.testing.assert_array_equal(source.cell.array, np.zeros((3, 3)))


def test_public_cartesian_atoms_without_cell_can_be_relaxed():
    potential = Potential.from_settings({'pottype': 'lj612', 'ntypes': 1,
                                         '0-0': '1 1', 'cut:0-0': 3})
    atoms = Atoms(numbers=[1, 1], positions=[[0., 0., 0.], [1.3, 0., 0.]], pbc=False)
    atoms.chemmap = [0, 0]
    result = Relaxation(potential, driver='lbfgs', max_iter=3, max_move=.02).run(atoms)
    assert np.isfinite(atoms.positions).all()
    assert result.final_energy < lj_energy(1.3)
    np.testing.assert_array_equal(atoms.cell, np.zeros((3, 3)))
