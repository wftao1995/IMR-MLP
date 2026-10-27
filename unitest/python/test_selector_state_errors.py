# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""State I/O failures raise in Python and raw configuration exports are preserved."""
from pathlib import Path

import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp._imrmlp import _ActiveSelector, _GPIO

from test_cli_error_paths import ROOT, run_command


@pytest.fixture
def state(tmp_path):
    index = tmp_path / 'index.map'
    index.write_text('H 0\n')
    source = tmp_path / 'input.xyz'
    source.write_text('2\nLattice="12 0 0 0 12 0 0 0 12" '
                      'Properties=species:S:1:pos:R:3\nH 1 1 1\nH 3.5 1 1\n')
    io = _GPIO()
    io.read_map(str(index))
    atoms = io.read(str(source))
    selector = _ActiveSelector(str(ROOT / 'unitest/test-data/potential.mtp'))
    selector.add_candidates(atoms)
    selector.select()
    path = tmp_path / 'valid.sel'
    selector.save_state(str(path), str(index))
    return selector, path, index


@pytest.mark.parametrize('failure', ['missing_directory', 'full_device'])
def test_state_save_reports_output_errors(tmp_path, state, failure):
    selector, _, index = state
    output = tmp_path / 'missing/output.sel'
    if failure == 'full_device':
        output = Path('/dev/full')
        if not output.exists():
            pytest.skip('/dev/full is unavailable')
    with pytest.raises(RuntimeError, match=str(output)):
        selector.save_state(str(output), str(index))


@pytest.mark.parametrize('failure', ['missing', 'invalid', 'truncated'])
def test_state_load_errors_raise(tmp_path, state, failure):
    _, valid, index = state
    broken = tmp_path / 'broken.sel'
    if failure == 'invalid':
        broken.write_bytes(b'invalid state')
    elif failure == 'truncated':
        broken.write_bytes(valid.read_bytes()[:-1])
    with pytest.raises(RuntimeError, match='broken.sel'):
        _ActiveSelector.load_state(str(broken), str(index))


@pytest.mark.parametrize('selected_only', [False, True])
def test_export_preserves_raw_extxyz_and_reports_output_failure(tmp_path, state, selected_only):
    _, path, _ = state
    args = ['maxvol_sample', '--state-file', path, '--extract-configs', 'export.xyz', '--num_threads', '1']
    if selected_only:
        args += ['--selected-only', 'true']
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    exported = (tmp_path / 'export.xyz').read_bytes()
    assert exported and exported in path.read_bytes()
    assert b'H ' in exported
    args[4] = 'missing/export.xyz'
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Configs exported' not in result.stdout
