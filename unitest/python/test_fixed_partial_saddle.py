"""Real ASE slab inputs retain their periodic rows and normal motion in fixed-cell solvers."""
import math
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
pytest.importorskip('ase')
from ase import Atoms as ASEAtoms
from imrmlp import Dimer, Potential, Relaxation


@pytest.mark.parametrize('kind', ['fire', 'lbfgs', 'dimer'])
@pytest.mark.parametrize('nonprefix', [False, True], ids=['TTF', 'TFT'])
@pytest.mark.parametrize('budget', [0, 2])
def test_fixed_partial_slab_matches_padding_preserves_source_and_normal_motion(tmp_path, monkeypatch, kind, nonprefix, budget):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'MODECAR').write_text('GPML_MODECAR 1 2 fixed-cartesian\n-1.2 0 -.5\n1.2 0 .5\n')
    potential = Potential.from_settings({'pottype': 'lj612', 'ntypes': 1, '0-0': '1 1', 'cut:0-0': 3})
    cell = np.zeros((3, 3)); cell[0] = [10, 0, 0]; cell[2 if nonprefix else 1] = [0, 10, 0]
    pbc = [True, not nonprefix, nonprefix]
    positions = np.array([[1., 1., 3.], [2.2, 1., 3.5]])
    source = ASEAtoms('H2', positions=positions, cell=cell, pbc=pbc)
    atoms, = potential.calculate_many([source], elements=['H'])
    padded = atoms.copy(); padded_cell = cell.copy(); padded_cell[1 if nonprefix else 2] = [0, 0, 20]; padded.cell = padded_cell
    if kind == 'dimer':
        driver = Dimer(potential, modecar='MODECAR', max_iter=budget, rot_max_iter=2, trn_max_iter=2)
        result, control = driver.run(atoms), driver.run(padded)
        final, expected = result.saddle_atoms, control.saddle_atoms
        energy, control_energy = result.energy, control.energy
        assert result.curvature == pytest.approx(control.curvature, abs=1e-9)
        if budget == 0:
            derivative = lambda r: 24 * (r ** -7 - 2 * r ** -13)
            shift = math.sqrt(2) * .005
            oracle = math.sqrt(2) * (derivative(1.3 + shift) - derivative(1.3 - shift)) / (2 * .005)
            assert result.curvature == pytest.approx(oracle, abs=1e-9)
        np.testing.assert_array_equal(atoms.positions, positions)
    else:
        driver = Relaxation(potential, driver=kind, max_iter=budget, max_move=.02, gtol=1e-10)
        result, control = driver.run(atoms), driver.run(padded)
        final, expected = result.final_atoms, control.final_atoms
        energy, control_energy = result.final_energy, control.final_energy
    assert np.isfinite(final.positions).all()
    np.testing.assert_allclose(final.positions, expected.positions, atol=1e-10)
    assert energy == pytest.approx(control_energy, abs=1e-10)
    radius = np.linalg.norm(final.positions[1] - final.positions[0])
    assert energy == pytest.approx(4 * (radius ** -12 - radius ** -6), abs=1e-10)
    if budget:
        assert abs(final.positions[0, 2] - 3) > 1e-8
    np.testing.assert_array_equal(final.cell, cell)
    np.testing.assert_array_equal(final.pbc, pbc)
    np.testing.assert_array_equal(final.numbers, [1, 1])
    np.testing.assert_array_equal(source.positions, positions)
    np.testing.assert_array_equal(source.cell.array, cell)
    np.testing.assert_array_equal(source.pbc, pbc)
