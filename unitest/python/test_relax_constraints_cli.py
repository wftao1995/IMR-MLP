"""Exercise versioned lattice constraints through the actual imrmlp executable."""
import gzip
import json
import math
import re

import numpy as np
import pytest

from test_cli_error_paths import run_command

CELL = np.array([[3.0, .2, .1], [.6, 3.5, .3], [.4, .5, 4.]])


def prepare(path, entry, extra=None, frames=None, atom_counts=None):
    frames = [CELL] if frames is None else frames
    atom_counts = [2] * len(frames) if atom_counts is None else atom_counts
    (path / 'index.map').write_text('H 0\n')
    (path / 'input.xyz').write_text(''.join(
        f'{count}\nLattice="' + ' '.join(map(str, cell.flat)) +
        '" pbc="T T T" Properties=species:S:1:pos:R:3\n'
        + ''.join(['H .7 .6 .5\n', 'H 1.9 1.3 1.1\n', 'H 2.4 2.5 2.6\n'][:count])
        for cell, count in zip(frames, atom_counts)))
    settings = {
        'input': 'input.xyz', 'input-format': 'extxyz',
        'output': 'output.xyz', 'output-format': 'extxyz',
        'pot:pottype': 'lj612', 'pot:ntypes': '1', 'pot:0-0': '0 1', 'pot:cut:0-0': '1',
        'isif': '6', 'file-constraints': 'constraints.json',
        'relax:cell_strain': 'False', 'relax:dprecond': 'False',
        'relax:driver': 'lbfgs', 'relax:gtol': '1e-7',
        'relax:lbfgspp:max_linesearch': '80',
        'relax:maxiteration': '600', 'relax:constraint_loops': '12',
        'relax:write_traj': 'False',
    }
    settings.update(extra or {})
    (path / 'settings.ini').write_text(''.join(f'{k} {v}\n' for k, v in settings.items()))
    (path / 'constraints.json').write_text(json.dumps({'version': 1, 'constraints': [entry]}))


def run(path, *extra):
    return run_command('imrmlp', ['relax', 'settings.ini', '--num_threads', '1', *extra], path)


def cells(path):
    return [np.array(list(map(float, x.split()))).reshape(3, 3)
            for x in re.findall(r'Lattice="([^"]+)"', (path / 'output.xyz').read_text())]


@pytest.mark.parametrize('entry', [
    {'kind': 'volume', 'target': 45},
    {'kind': 'lattice_length', 'axis': 'a', 'target': 3.3},
    {'kind': 'lattice_angle', 'axes': ['a', 'b'], 'target': 70},
    {'kind': 'lattice_ratio', 'axes': ['a', 'b'], 'target': .95},
])
@pytest.mark.parametrize('isif,recast', [('6', 'True'), ('3', 'True'), ('3', 'False')])
@pytest.mark.parametrize('driver', ['lbfgs', 'lbfgspp'])
def test_raw9_constraints_actual_command(tmp_path, entry, isif, recast, driver):
    prepare(tmp_path, dict(entry, residual_tolerance=1e-6),
            {'isif': isif, 'relax:atomic_force_recast': recast, 'relax:driver': driver})
    result = run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = cells(tmp_path)
    assert len(actual) == 1
    cell = actual[0]
    assert np.max(np.abs(cell - CELL)) > 1e-5
    target = entry['target']
    if entry['kind'] == 'volume':
        residual = np.linalg.det(cell) - target
    elif entry['kind'] == 'lattice_length':
        residual = np.dot(cell[0], cell[0]) - target**2
    elif entry['kind'] == 'lattice_angle':
        residual = np.dot(cell[0], cell[1]) / np.linalg.norm(cell[0]) / np.linalg.norm(cell[1]) - math.cos(math.radians(target))
    else:
        residual = np.dot(cell[0], cell[0]) / np.dot(cell[1], cell[1]) - target**2
    assert abs(residual) <= 1e-6
    assert 'Constraint 0: residual=' in result.stdout


@pytest.mark.parametrize('sequential,displaced', [
    pytest.param(False, False, id='batch'),
    pytest.param(False, True, id='batch-displaced'),
    pytest.param(True, True, id='seq-gzip-displaced'),
])
def test_implicit_targets_actual_command_are_per_frame(tmp_path, sequential, displaced):
    prepare(tmp_path, {'kind': 'volume', 'residual_tolerance': 1e-7},
            {'relax:apply_disp': str(displaced), 'relax:disp': '0.001'},
            frames=[CELL, CELL * 1.2], atom_counts=[2, 3])
    extra = []
    if sequential:
        (tmp_path / 'input.xyz.gz').write_bytes(gzip.compress((tmp_path / 'input.xyz').read_bytes()))
        extra = ['--seq-read', '--input', 'input.xyz.gz']
    result = run(tmp_path, *extra)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = cells(tmp_path)
    assert len(actual) == 2
    assert re.findall(r'(?m)^\d+$', (tmp_path / 'output.xyz').read_text()) == ['2', '3']
    for result_cell, original in zip(actual, [CELL, CELL * 1.2]):
        assert np.linalg.det(result_cell) == pytest.approx(np.linalg.det(original), abs=1e-7)
        if displaced:
            assert np.max(np.abs(result_cell - original)) > 1e-7
        else:
            assert result_cell == pytest.approx(original)


@pytest.mark.parametrize('failure', ['duplicate', 'strain', 'cartesian', 'pressure', 'mask'])
def test_invalid_constraint_request_actual_command_preserves_output(tmp_path, failure):
    extra = {'strain': {'relax:cell_strain': 'True'}, 'cartesian': {'relax:cartesian': 'True'},
             'pressure': {'relax:pressure': '1'},
             'mask': {'position-mask': '1 1 1'}}.get(failure)
    prepare(tmp_path, {'kind': 'volume', 'target': 45, 'residual_tolerance': 1e-6}, extra)
    if failure == 'duplicate':
        (tmp_path / 'constraints.json').write_text('{"version":1,"version":1,"constraints":[]}')
    (tmp_path / 'output.xyz').write_text('existing output')
    result = run(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / 'output.xyz').read_text() == 'existing output'


def test_constraint_budget_actual_command_has_nonconverged_exit(tmp_path):
    prepare(tmp_path, {'kind': 'volume', 'target': 80, 'residual_tolerance': 1e-12},
            {'relax:maxiteration': '0'})
    result = run(tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'Constraint 0: residual=' in result.stdout
    assert cells(tmp_path)[0] == pytest.approx(CELL)


def test_constraint_line_search_failure_is_reported(tmp_path):
    prepare(tmp_path, {'kind': 'volume', 'target': 45, 'residual_tolerance': 1e-6},
            {'relax:driver': 'lbfgspp', 'relax:lbfgspp:max_linesearch': '1'})
    result = run(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'exceed max linesearch' in result.stdout + result.stderr


@pytest.mark.parametrize('sequential', [False, True])
def test_later_frame_species_error_preserves_output(tmp_path, sequential):
    prepare(tmp_path, {'kind': 'volume', 'residual_tolerance': 1e-6},
            {'seq-write': 'True'}, frames=[CELL, CELL * 1.2])
    (tmp_path / 'index.map').write_text('H 0\nHe 1\n')
    contents = (tmp_path / 'input.xyz').read_text()
    boundary = contents.rfind('H 1.9')
    (tmp_path / 'input.xyz').write_text(contents[:boundary] + contents[boundary:].replace('H ', 'He ', 1))
    (tmp_path / 'output.xyz').write_text('existing output')
    result = run(tmp_path, *(['--seq-read'] if sequential else []))
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'frame 1 atom 1' in result.stdout + result.stderr
    assert 'species index' in result.stdout + result.stderr
    if sequential:
        actual = cells(tmp_path)
        assert len(actual) == 1
        assert actual[0] == pytest.approx(CELL)
    else:
        assert (tmp_path / 'output.xyz').read_text() == 'existing output'


def test_infeasible_lattice_equalities_do_not_report_success(tmp_path):
    prepare(tmp_path, {'kind': 'volume', 'target': 2, 'residual_tolerance': 1e-6},
            {'relax:maxiteration': '100'})
    # Hadamard's inequality bounds the volume by 1 when all three lengths are 1.
    constraints = [{'kind': 'volume', 'target': 2, 'residual_tolerance': 1e-6}]
    constraints.extend({'kind': 'lattice_length', 'axis': axis, 'target': 1,
                        'residual_tolerance': 1e-6} for axis in ['a', 'b', 'c'])
    (tmp_path / 'constraints.json').write_text(json.dumps({'version': 1, 'constraints': constraints}))
    result = run(tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert all(f'Constraint {i}: residual=' in result.stdout for i in range(4))
    assert 'did not converge' in result.stdout
