# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Regression checks for selection, generation and committee command failures."""
from pathlib import Path

import pytest

from test_cli_error_paths import ROOT, run_command


@pytest.fixture
def inputs(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.xyz').write_text(
        '2\nLattice="12 0 0 0 12 0 0 0 12" '
        'Properties=species:S:1:pos:R:3\nH 1 1 1\nH 3.5 1 1\n')
    (tmp_path / 'committee_0.mtp').symlink_to(ROOT / 'unitest/test-data/potential.mtp')
    (tmp_path / 'settings.ini').write_text('bondorder:cutoff 5\ncluster:cell_size 15\n')
    return ['--num_threads', '1']


def command_args(command):
    if command == 'al_eval':
        return ['al_eval', '--input', 'input.xyz', '--members', '1']
    if command == 'gen_al_pool':
        return ['gen_al_pool', '--reference', 'input.xyz', '--gen-type', 'cluster',
                '--n-total', '12', '--settings', 'settings.ini']
    if command == 'unique':
        return ['unique', '--input', 'input.xyz', '--overwrite']
    return ['diverse_select', '--pool', 'input.xyz', '--n-select', '1',
            '--method', command, '--dtype', 'bondorder', '--settings', 'settings.ini']


@pytest.mark.parametrize('command', ['fps', 'cur', 'unique'])
def test_auxiliary_output_failure(tmp_path, inputs, command):
    args = command_args(command) + inputs
    if command == 'unique':
        (tmp_path / 'output-nodup.id').mkdir()
    else:
        args += ['--score-file', 'missing/scores.txt']
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize('command', ['maxvol_sample', 'maxvol_extend'])
def test_maxvol_candidate_read_failure(tmp_path, inputs, command):
    option = '--input' if command == 'maxvol_sample' else '--candidates'
    result = run_command('imrmlp', [command, option, 'missing.xyz', *inputs], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    # A successfully read empty candidate list is still a valid no-op.
    (tmp_path / 'empty.xyz').write_text('')
    result = run_command('imrmlp', [command, option, 'empty.xyz', *inputs], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('mode', ['extend', 'dump', 'extract'])
@pytest.mark.parametrize('state', ['missing', 'invalid', 'truncated'])
def test_maxvol_state_errors_do_not_overwrite_files(tmp_path, inputs, mode, state):
    state_file = tmp_path / 'broken.sel'
    original = b'invalid state file' if state == 'invalid' else b'VTCA\x0a\x00\x00\x00'
    if state != 'missing':
        state_file.write_bytes(original)
    args = ['maxvol_sample', '--state-file', state_file, *inputs]
    if mode == 'extend':
        args = ['maxvol_extend', '--state-file', state_file, '--candidates', 'input.xyz',
                '--dtype', 'bondorder', '--settings', 'settings.ini', *inputs]
    elif mode == 'dump':
        args += ['--dump-state', 'true']
    else:
        args += ['--extract-configs', 'extracted.xyz']
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Configs exported' not in result.stdout
    assert not (tmp_path / 'extracted.xyz').exists()
    if state == 'missing':
        assert not state_file.exists()
    else:
        assert state_file.read_bytes() == original
