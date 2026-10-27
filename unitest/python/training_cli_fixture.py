"""Nonlinear committee training through real CLI processes, without binding state."""
import hashlib
from pathlib import Path
import re

import numpy as np
import pytest

from test_cli_error_paths import run_command

MODEL = '''MTP
version = 1.1.0
potential_name = committee_cli_test
species_count = 1
potential_tag = test-only
radial_basis_type = RBChebyshev
min_dist = 0
max_dist = 5
radial_basis_size = 2
radial_funcs_count = 1
radial_coeffs
0-0
{0.25, 0.125}
alpha_moments_count = 1
alpha_index_basic_count = 1
alpha_index_basic = {{0,0,0,0}}
alpha_index_times_count = 0
alpha_index_times = {}
alpha_scalar_moments = 1
alpha_moment_mapping = {0}
species_coeffs = {0.125}
moment_coeffs = {0.75}
'''


def write_settings(path, extra=None, *, nonlinear=True):
    settings = {
        'potfile': 'base.mtp', 'trainset': 'train.cfg', 'format': 'mlpcfg',
        'index': 'index.map', 'members': '2', 'bootstrap': '1', 'mode': 'cfg',
    }
    if nonlinear:
        settings.update({
            'linear-only': 'false', 'output-directory': 'fitted',
            'train:bfgs_m': '2', 'train:bfgs_maxiteration': '50',
            'train:bfgs_grad_tol': '1e-12', 'train:bfgs_conv_tol': '0',
            'train:bfgs_step_tol': '0', 'train:penalty_weight': '0',
            'train:save_every': '100',
        })
    settings.update(extra or {})
    (path / 'settings.ini').write_text(''.join(f'{key} {value}\n' for key, value in settings.items()))


def evaluate(path, model, output):
    result = run_command('imrmlp', [
        'efs', 'input.xyz', output, '--input-format', 'extxyz',
        '--output-format', 'mlpcfg', '--potfile', model,
        '--index', 'index.map', '--num_threads', '1'], path)
    assert result.returncode == 0, result.stdout + result.stderr
    return read_labels(path / output)


def prepare_fixture(path):
    """Also used by the MPI acceptance run to generate identical small EFS data."""
    path.mkdir(parents=True, exist_ok=True)
    (path / 'base.mtp').write_text(MODEL)
    (path / 'teacher.mtp').write_text(MODEL.replace('{0.25, 0.125}', '{0.32, 0.075}')
                                         .replace('species_coeffs = {0.125}', 'species_coeffs = {0.18}'))
    (path / 'index.map').write_text('H 0\n')
    frames = []
    for i in range(6):
        positions = [[6, 6, 6], [7.8 + .15*i, 6.1, 6.2], [6.3, 8.9 - .1*i, 6.4]]
        frames.append('3\nLattice="20 0 0 0 20 0 0 0 20" pbc="F F F" '
                      'Properties=species:S:1:pos:R:3\n'
                      + ''.join('H ' + ' '.join(map(str, point)) + '\n' for point in positions))
    (path / 'input.xyz').write_text(''.join(frames))
    labels = evaluate(path, 'teacher.mtp', 'train.cfg')
    write_settings(path)
    return labels


def read_labels(path):
    labels = []
    for frame in path.read_text().split('BEGIN_CFG')[1:]:
        lines = frame.splitlines()
        size = int(lines[lines.index('Size') + 1])
        atom_line = next(i for i, line in enumerate(lines) if line.startswith('AtomData:'))
        keys = lines[atom_line].split()[1:]
        rows = [list(map(float, line.split())) for line in lines[atom_line + 1:atom_line + 1 + size]]
        forces = np.array([[row[keys.index(k)] for k in ('fx', 'fy', 'fz')] for row in rows])
        stress_line = next(i for i, line in enumerate(lines) if line.startswith('PlusStress:'))
        xx, yy, zz, yz, xz, xy = map(float, lines[stress_line + 1].split())
        labels.append((float(lines[lines.index('Energy') + 1]), forces,
                       np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])))
    return labels


def objective(predicted, targets, weights):
    # Match the normal EFS objective: E/N, F, and full 3x3 virial/N weights.
    loss = 0.
    for (energy, forces, stress), (ref_e, ref_f, ref_s) in zip(predicted, targets):
        loss += weights[0] * (energy - ref_e)**2 / len(forces)
        loss += weights[1] * np.sum((forces - ref_f)**2)
        loss += weights[2] * np.sum((stress - ref_s)**2) / len(forces)
    return loss


def radial_coefficients(path):
    values = path.read_text().split('radial_coeffs', 1)[1].split('alpha_moments_count', 1)[0]
    return np.array([float(value) for group in re.findall(r'\{([^}]*)\}', values)
                     for value in group.split(',')])


def run(path, *extra):
    return run_command('imrmlp', ['al_train', 'settings.ini', '--num_threads', '1', *extra], path)


def assert_failed_without_publication(result, path):
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Saved ' not in result.stdout
    assert not (path / 'fitted').exists()
    assert not list(path.glob('potential_fitted*'))
    assert not list(path.glob('*.iter-*.mtp'))


def force_difference(potential, atoms, directions, step=1e-4):
    result = np.empty_like(directions)
    for k, direction in enumerate(directions.T):
        vectors = direction.reshape(-1, 3)
        epsilon = step / np.linalg.norm(vectors, axis=1).max()
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions = atoms.positions + epsilon*vectors
        minus.positions = atoms.positions - epsilon*vectors
        potential.calculate(plus)
        potential.calculate(minus)
        result[:, k] = -(plus.forces-minus.forces).reshape(-1)/(2*epsilon)
    return result
