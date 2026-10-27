# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Force-constant export and matrix-free modes remain independent of phonon postprocessing."""
import numpy as np
import pytest

from test_cli_error_paths import ROOT, run_command

MODEL = ROOT / 'LAMMPS/USER-MLIP/transport/tests/simple_analytic.mtp'


def prepare(tmp_path, mass='H 0.4 1.0\n'):
    (tmp_path / 'index.map').write_text(mass)
    (tmp_path / 'input.xyz').write_text(
        '3\nLattice="20 0 0 0 20 0 0 0 20" Properties=species:S:1:pos:R:3\n'
        'H 7 7 7\nH 9.5 7 7\nH 7.8 9 7.1\n')
    return ['calc_fc', '--potfile', MODEL, '--input', 'input.xyz',
            '--input-format', 'extxyz', '--num_threads', '1']


def read_full_fc(path):
    lines = path.read_text().splitlines()
    n, m = map(int, lines[0].split())
    assert n == m
    matrix = np.zeros((3*n, 3*n))
    assert len(lines) == 1 + 4*n*n
    for offset in range(1, len(lines), 4):
        i, j = [int(value)-1 for value in lines[offset].split()]
        matrix[3*i:3*i+3, 3*j:3*j+3] = np.loadtxt(lines[offset+1:offset+4])
    return matrix


@pytest.mark.parametrize('supercell', [[], ['--supercell', '2', '1', '1']])
def test_full_force_constants_match_force_finite_differences(tmp_path, supercell):
    args = prepare(tmp_path) + supercell + ['--reduced=false', '--check-symm']
    matrices = []
    for method in ('analytic', 'force_fd'):
        result = run_command('imrmlp', args + ['--hessian-method', method,
                                             '--hessian-nudge-pos', '0.00001'], tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'Acoustic sum rule maximum residual' in result.stdout
        matrices.append(read_full_fc(tmp_path / 'FORCE_CONSTANTS'))
    np.testing.assert_allclose(*matrices, rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(matrices[0].reshape(-1, 3, matrices[0].shape[0]//3, 3).sum(axis=2),
                               0, atol=1e-12)
    assert not (tmp_path / 'EIGENVALS').exists()
    assert (tmp_path / 'supercell_created.xsf').is_file()


def test_lowest_modes_use_mass_column_instead_of_radius(tmp_path):
    args = prepare(tmp_path)
    result = run_command('imrmlp', args + ['--reduced=false'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    hessian = read_full_fc(tmp_path / 'FORCE_CONSTANTS')
    (tmp_path / 'FORCE_CONSTANTS').unlink()
    # Equal unit masses allow an independent Cartesian translation projector.
    translations = np.tile(np.eye(3), (3, 1)) / np.sqrt(3)
    basis = np.linalg.qr(translations, mode='complete')[0][:, 3:]
    expected = np.linalg.eigvalsh(basis.T @ hessian @ basis)[:3]
    for mass, factor in [('H 0.4 1\n', 1), ('H 0.4 4\n', 4), ('H 9.0 4\n', 4),
                         ('H\n', 1.008), ('H 0.4\n', 1.008)]:
        (tmp_path / 'index.map').write_text(mass)
        result = run_command('imrmlp', args + ['--lowest-modes', '3'], tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        actual = np.loadtxt(tmp_path / 'EIGENVALS')[:, 0]
        np.testing.assert_allclose(actual, expected/factor, rtol=1e-7, atol=1e-10)
    assert not (tmp_path / 'FORCE_CONSTANTS').exists()


@pytest.mark.parametrize('target', ['directory', 'full'])
def test_force_constant_write_failures_are_reported(tmp_path, target):
    args = prepare(tmp_path)
    path = tmp_path / 'FORCE_CONSTANTS'
    if target == 'directory':
        path.mkdir()
    else:
        from pathlib import Path
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        path.symlink_to('/dev/full')
    result = run_command('imrmlp', args + ['--reduced=false'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'FORCE_CONSTANTS' in result.stderr


def test_analytic_export_uses_phonopy_supercell_atom_order(tmp_path, monkeypatch):
    pytest.importorskip('phonopy')
    pytest.importorskip('ase')
    pytest.importorskip('imrmlp._imrmlp')
    from ase import Atoms
    from ase.io import write
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from imrmlp.calculator import IMRMLPCalculator
    from imrmlp.phonopy import calculate_forces

    monkeypatch.chdir(tmp_path)
    args = prepare(tmp_path)
    unitcell = PhonopyAtoms(symbols=['H']*3, cell=np.eye(3)*20,
                            positions=[[7, 7, 7], [9.5, 7, 7], [7.8, 9, 7.1]])
    phonon = Phonopy(unitcell, [2, 1, 1], primitive_matrix=np.eye(3), is_symmetry=False)
    cell = phonon.supercell
    write('input.xyz', Atoms(symbols=cell.symbols, positions=cell.positions,
                            cell=cell.cell, pbc=True), format='extxyz')
    result = run_command('imrmlp', args + ['--reduced=false'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    analytic = read_full_fc(tmp_path / 'FORCE_CONSTANTS')
    phonon.generate_displacements(distance=1e-5, is_plusminus=True)
    calculate_forces(phonon, IMRMLPCalculator(filename=MODEL, elements=['H']))
    phonon.produce_force_constants()
    finite_difference = phonon.force_constants.transpose(0, 2, 1, 3).reshape(18, 18)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-7, atol=1e-8)
