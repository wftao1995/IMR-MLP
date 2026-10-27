# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Validate phonopy displacement inputs and force-file failure reporting."""
from pathlib import Path

import pytest

from test_cli_error_paths import ROOT, run_command

SUPERCELL = '''supercell:
  lattice:
  - [ 20.0, 0.0, 0.0 ]
  - [ 0.0, 20.0, 0.0 ]
  - [ 0.0, 0.0, 20.0 ]
  points:
  - symbol: H
    coordinates: [ 0.05, 0.05, 0.05 ]
    mass: 1.0
    reduced_to: 1
  - symbol: H
    coordinates: [ 0.15, 0.05, 0.05 ]
    mass: 1.0
    reduced_to: 2

'''
DISPLACEMENT = '''displacements:
- atom: 1
  displacement:
    [ 0.01, 0.0, 0.0 ]
'''


def prepare(tmp_path, order=2):
    (tmp_path / 'index.map').write_text('H')  # Final newline is optional.
    text = SUPERCELL + DISPLACEMENT
    (tmp_path / 'input.yaml').write_text(text)
    return ['phonopyfc', ROOT / 'unitest/test-data/potential.mtp', 'input.yaml',
            '--num_threads', '1']


@pytest.mark.parametrize('order', [2])
def test_valid_displacements_produce_forces(tmp_path, order):
    result = run_command('imrmlp', prepare(tmp_path, order), tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = ['FORCE_SETS']
    for output in outputs:
        assert (tmp_path / output).stat().st_size > 0


@pytest.mark.parametrize('order', [2])
@pytest.mark.parametrize('failure', ['zero', 'negative', 'large', 'unknown_element', 'empty'])
def test_invalid_displacements_are_reported(tmp_path, order, failure):
    args = prepare(tmp_path, order)
    path = tmp_path / 'input.yaml'
    text = path.read_text()
    if failure in ['zero', 'negative', 'large']:
        number = {'zero': '0', 'negative': '-1', 'large': '1000000'}[failure]
        text = text.replace('- atom: 1', '- atom: ' + number)
        expected = 'displacement atom'
    elif failure == 'unknown_element':
        text = text.replace('symbol: H', 'symbol: Xe')
        expected = 'unknown element'
    else:
        text = SUPERCELL
        expected = 'phonopyfc'
    path.write_text(text)
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert expected in result.stderr


@pytest.mark.parametrize('order,output', [(2, 'FORCE_SETS')])
@pytest.mark.parametrize('failure', ['directory', 'full'])
def test_force_output_failures_are_reported(tmp_path, order, output, failure):
    args = prepare(tmp_path, order)
    path = tmp_path / output
    if failure == 'directory':
        path.mkdir()
    else:
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        path.symlink_to('/dev/full')
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert output in result.stderr
