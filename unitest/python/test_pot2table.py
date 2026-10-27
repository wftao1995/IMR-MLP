# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Tabulation round trips, physical force columns and command failures."""
from pathlib import Path

import numpy as np
import pytest

from test_cli_error_paths import run_command


@pytest.fixture
def lj_input(tmp_path):
    (tmp_path / 'potential.lj').write_text('pot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 3\n')
    (tmp_path / 'index.map').write_text('H\n')
    return ['pot2table', '--potfile', 'potential.lj', '--pottype', 'lj612',
            '--rmin', '1.1']


def lj_energy_force(r):
    return 4 * (r**-12 - r**-6), 24 * (2 * r**-13 - r**-7)


@pytest.mark.extended
def test_pair_export_reload_and_lammps_forces(tmp_path, monkeypatch, lj_input):
    pytest.importorskip('imrmlp._imrmlp')
    from imrmlp import Atoms, Potential
    result = run_command('imrmlp', lj_input + [
        '--pair', '--pair-output', 'pair.table', '--nr', '1001', '--lammps'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / 'pair.table').read_text().splitlines()
    assert lines[1] == 'N 1001'
    grid = np.loadtxt(lines[2:])
    assert grid.shape == (1001, 3)
    np.testing.assert_allclose(np.diff(grid[:, 1]), 1.9 / 1000)
    np.testing.assert_allclose(grid[[0, -1], 1], [1.1, 3])

    lammps = np.loadtxt(tmp_path / 'lammps_pair_0_0', skiprows=5)
    # Exclude the cutoff itself, where this LJ implementation returns zero.
    energies, forces = lj_energy_force(lammps[:-1, 1])
    np.testing.assert_allclose(lammps[:-1, 2], energies, atol=1e-12)
    np.testing.assert_allclose(lammps[:-1, 3], forces, atol=1e-12)
    assert abs(lammps[0, 3]) > 1

    monkeypatch.chdir(tmp_path)
    potential = Potential('pair.table', pottype='pair_table')
    assert potential.cutoff == pytest.approx(3)
    for r in [1.2137, 1.6743, 2.3187]:
        atoms = Atoms(numbers=[1, 1], elements=['H'],
                      positions=[[0, 0, 0], [r, 0, 0]], cell=np.eye(3)*20, pbc=False)
        potential.calculate(atoms)
        energy, force = lj_energy_force(r)
        assert atoms.energy == pytest.approx(energy, abs=1e-6)
        np.testing.assert_allclose(atoms.forces, [[-force, 0, 0], [force, 0, 0]], atol=2e-4)
        expected_stress = np.zeros((3, 3))
        expected_stress[0, 0] = r * force
        np.testing.assert_allclose(atoms.stress, expected_stress, atol=3e-4)


@pytest.mark.parametrize('options', [
    ['--pair', '--nr', '0'], ['--pair', '--nr', '1'], ['--pair', '--nr', '2'],
    ['--pair', '--rmin', '3.1'], ['--pair', '--rmax', '1'],
    ['--pair', '--rmin', '0'], ['--pair', '--rmin', 'nan'], ['--pair', '--rmax', 'inf'],
    ['--threebody', '--nr', '1'], ['--threebody', '--ntheta', '1'],
])
def test_invalid_grid_fails_before_output(tmp_path, lj_input, options):
    # Replace the fixture's radius rather than passing duplicate options.
    args = lj_input[:-2] if '--rmin' in options else lj_input
    result = run_command('imrmlp', args + options, tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (tmp_path / 'pair_table.out').exists()
    assert not (tmp_path / 'threebody_table.out').exists()
    assert 'Done.' not in result.stdout


@pytest.mark.parametrize('kind', ['pair', 'threebody', 'lammps'])
@pytest.mark.parametrize('failure', ['missing_directory', 'full'])
def test_table_output_failures_return_error(tmp_path, lj_input, kind, failure):
    path = 'missing/output' if failure == 'missing_directory' else 'output'
    if failure == 'full':
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        (tmp_path / (path + '_0_0' if kind == 'lammps' else path)).symlink_to('/dev/full')
    if kind == 'lammps':
        options = ['--pair', '--lammps-prefix', path]
    else:
        options = ['--' + kind, '--' + kind + '-output', path]
    result = run_command('imrmlp', lj_input + ['--nr', '3', '--ntheta', '3'] + options, tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'Done.' not in result.stdout
    assert path in result.stderr


def test_minimum_pair_grid_can_be_loaded(tmp_path, monkeypatch, lj_input):
    pytest.importorskip('imrmlp._imrmlp')
    from imrmlp import Atoms, Potential
    result = run_command('imrmlp', lj_input + ['--pair', '--nr', '3'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    monkeypatch.chdir(tmp_path)
    potential = Potential('pair_table.out', pottype='pair_table')
    atoms = Atoms(numbers=[1, 1], elements=['H'], positions=[[0, 0, 0], [2.05, 0, 0]],
                  cell=np.eye(3)*20, pbc=False)
    potential.calculate(atoms)
    assert atoms.energy == pytest.approx(lj_energy_force(2.05)[0])
