"""Ordinary Cartesian molecules can use the fixed-cell Dimer coordinate codec."""
import math
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
pytest.importorskip('ase')
from ase import Atoms as ASEAtoms
from imrmlp import Atoms, Dimer, Potential


def pair_energy(radius):
    return 4 * (radius ** -12 - radius ** -6)


@pytest.mark.parametrize('budget', [0, 2])
def test_public_ase_molecular_dimer_curvature_and_padding_equivalence(tmp_path, monkeypatch, budget):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'MODECAR').write_text('GPML_MODECAR 1 2 fixed-cartesian\n-1 0 0\n1 0 0\n')
    potential = Potential.from_settings({'pottype': 'lj612', 'ntypes': 1, '0-0': '1 1', 'cut:0-0': 3})
    source = ASEAtoms('H2', positions=[[0., 0., 0.], [1.3, 0., 0.]], pbc=False)
    atoms, = potential.calculate_many([source], elements=['H'])
    padded = atoms.copy(); padded.cell = np.eye(3) * 20
    initial = atoms.positions.copy()
    dimer = Dimer(potential, modecar='MODECAR', max_iter=budget, rot_max_iter=2, trn_max_iter=2)
    result = dimer.run(atoms)
    control = dimer.run(padded)
    assert np.isfinite(result.energy) and np.isfinite(result.curvature)
    assert result.energy == pytest.approx(control.energy, abs=1e-12)
    assert result.curvature == pytest.approx(control.curvature, abs=1e-10)
    np.testing.assert_allclose(result.saddle_atoms.positions, control.saddle_atoms.positions, atol=1e-12)
    radius = np.linalg.norm(result.saddle_atoms.positions[1] - result.saddle_atoms.positions[0])
    assert result.energy == pytest.approx(pair_energy(radius), abs=1e-12)
    if budget == 0:
        derivative = lambda r: 24 * (r ** -7 - 2 * r ** -13)
        shift = math.sqrt(2) * .005
        expected = math.sqrt(2) * (derivative(1.3 + shift) - derivative(1.3 - shift)) / (2 * .005)
        assert result.curvature == pytest.approx(expected, abs=1e-10)
    assert result.n_iter <= budget
    np.testing.assert_array_equal(result.saddle_atoms.cell, np.zeros((3, 3)))
    np.testing.assert_array_equal(result.saddle_atoms.pbc, [False] * 3)
    np.testing.assert_array_equal(atoms.positions, initial)
    np.testing.assert_array_equal(atoms.cell, np.zeros((3, 3)))
    np.testing.assert_array_equal(source.positions, initial)
    assert (tmp_path / 'dimer_modecar_final').read_text().splitlines()[0] == 'GPML_MODECAR 1 2 fixed-cartesian'


@pytest.mark.parametrize('boundary', ['variable_cell', 'periodic', 'fractional'])
def test_public_dimer_keeps_cell_required_boundaries(tmp_path, monkeypatch, boundary):
    monkeypatch.chdir(tmp_path)
    potential = Potential.from_settings({'pottype': 'lj612', 'ntypes': 1, '0-0': '1 1', 'cut:0-0': 3})
    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [1.3, 0, 0]], pbc=False, elements=['H'])
    if boundary == 'periodic':
        atoms.pbc = [True, False, False]
    if boundary == 'fractional':
        atoms._impl.cartesian = False
    dimer = Dimer(potential, max_iter=0, vc_flag=boundary == 'variable_cell')
    error = ValueError if boundary == 'periodic' else RuntimeError
    message = 'independent periodic cell vectors' if boundary == 'periodic' else 'nonsingular cell'
    with pytest.raises(error, match=message):
        dimer.run(atoms)
