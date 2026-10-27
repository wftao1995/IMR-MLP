# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Descriptor and paired-error commands fail cleanly on invalid I/O or data."""
from pathlib import Path

import pytest

from test_cli_error_paths import ROOT, run_command


def xyz(energy=1, forces=True, species=('H', 'H')):
    properties = 'species:S:1:pos:R:3' + (':forces:R:3' if forces else '')
    header = f'{len(species)}\nLattice="12 0 0 0 12 0 0 0 12" Properties={properties}'
    if energy is not None:
        header += f' energy={energy}'
    rows = [f'{symbol} {1 + 2.5 * i} 1 1' + (' 0 0 0' if forces else '')
            for i, symbol in enumerate(species)]
    return header + '\n' + '\n'.join(rows) + '\n'


@pytest.fixture
def inputs(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\nHe 0\n')
    (tmp_path / 'input.xyz').write_text(xyz())
    (tmp_path / 'predicted.xyz').write_text(xyz(energy=0))
    (tmp_path / 'settings.ini').write_text('bondorder:cutoff 5\n')


@pytest.mark.parametrize('failure', ['missing_input', 'empty_input', 'missing_output_directory',
                                    'full_device', 'missing_potential', 'invalid_potential'])
def test_descriptor_failures_return_nonzero_without_aborting(tmp_path, inputs, failure):
    args = ['calc_descriptors', '--input', 'input.xyz', '--dtype', 'bondorder',
            '--settings', 'settings.ini', '--output', 'output.xyz', '--num_threads', '2']
    if failure == 'missing_input':
        (tmp_path / 'input.xyz').unlink()
    elif failure == 'empty_input':
        (tmp_path / 'input.xyz').write_text('')
    elif failure == 'missing_output_directory':
        args[args.index('--output') + 1] = 'missing/output.xyz'
    elif failure == 'full_device':
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        args[args.index('--output') + 1] = '/dev/full'
    else:
        args[args.index('--dtype') + 1] = 'mtp'
        args += ['--potfile', 'invalid.mtp']
        if failure == 'invalid_potential':
            (tmp_path / 'invalid.mtp').write_text('invalid potential\n')
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr


def test_mtp_descriptor_results_match_across_thread_counts(tmp_path, inputs):
    (tmp_path / 'input.xyz').write_text(xyz() * 3)
    args = ['calc_descriptors', '--input', 'input.xyz', '--potfile',
            ROOT / 'unitest/test-data/potential.mtp', '--output', 'desc.xyz']
    outputs = []
    for threads in (1, 2):
        result = run_command('imrmlp', args + ['--num_threads', str(threads)], tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append((tmp_path / 'desc.xyz').read_text())
    assert outputs[0] == outputs[1]
