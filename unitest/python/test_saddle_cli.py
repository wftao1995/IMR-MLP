# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Regression checks for validation, convergence and simulation output failures."""


from pathlib import Path


import pytest


from test_cli_error_paths import ROOT, run_command


@pytest.fixture
def simulation(tmp_path):
    (tmp_path / 'index.map').write_text('H\nHe\n')
    header = '2\nLattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3\n'
    (tmp_path / 'input.xyz').write_text(header + 'H 1 1 1\nH 3 1 1\n')
    (tmp_path / 'final.xyz').write_text(header + 'H 1.2 1 1\nH 3.2 1 1\n')
    # Fix the dimer direction so output-error checks are deterministic.
    (tmp_path / 'MODECAR').write_text('1 0 0\n-1 0 0\n')
    return f'pot:pottype mtp\npot:potfile {ROOT / "unitest/test-data/potential.mtp"}\n'


def prepare(tmp_path, simulation, command, extra=''):
    settings = {
        'neb': 'nimgs 3\nimage:0 input.xyz\nimage:2 final.xyz\ninput-format extxyz\n'
               'driver fire\nmax_total_iter 1\nneb:gtol 1e20\nneb:climb false\n',
        'dimer': 'structure input.xyz\ninput-format extxyz\ndimer:modecarf MODECAR\n'
                 'dimer:maxiteration 3\ndimer:rot_max_iter 1\ndimer:trn_max_iter 1\n'
                 'dimer:gtol 1e20\ndimer:min_curvature 1e20\n',
    }
    (tmp_path / 'input.ini').write_text(simulation + settings[command] + extra)
    return [command, '--settings', 'input.ini', '--num_threads', '1']


@pytest.mark.parametrize('command', ['neb', 'dimer'])
def test_calculations_write_outputs_on_success(tmp_path, simulation, command):
    extra = {'neb': 'neb:write_traj true\nneb:traj-format extxyz\nneb:traj path.xyz\n',
             'dimer': 'dimer:write_traj true\ndimer:trajfile path.xyz\n',
             'unused': ''}[command]
    result = run_command('imrmlp', prepare(tmp_path, simulation, command, extra), tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'path.xyz').stat().st_size > 0
    final = 'mep.xyz' if command == 'neb' else 'dimer_final.vasp'
    assert (tmp_path / final).stat().st_size > 0


@pytest.mark.parametrize('command,output', [
    ('neb', 'path'), ('neb', 'summary'), ('neb', 'trajectory'), ('neb', 'spline'),
    ('neb', 'initial_image'), ('neb', 'final_image'), ('neb', 'reference_cell'),
    ('dimer', 'final'), ('dimer', 'summary'), ('dimer', 'trajectory'),
    ('dimer', 'default_final'), ('dimer', 'mode'), ('dimer', 'endpoint'),
])
@pytest.mark.parametrize('failure', ['directory', 'full'])
def test_output_failures_are_not_success(tmp_path, simulation, command, output, failure):
    if failure == 'full' and not Path('/dev/full').exists():
        pytest.skip('/dev/full is unavailable')
    target = 'failed.xyz' if output in ('trajectory', 'path') else 'failed.vasp'
    fixed = {'spline': 'neb_hermite_spline.dat', 'initial_image': 'nebimage_0.vasp',
             'final_image': 'nebimage_0_final.vasp', 'reference_cell': 'neb_reference_cell.data',
             'default_final': 'dimer_final.vasp', 'mode': 'dimer_modecar_final',
             'endpoint': 'dimer_plus_final.vasp'}
    if output in fixed:
        target = fixed[output]
    path = tmp_path / target
    if failure == 'directory':
        path.mkdir()
    else:
        path.symlink_to('/dev/full')
    extra = ''
    options = []
    if command == 'neb':
        if output == 'path':
            options = ['--mepout', target, '--mepformat', 'extxyz']
        elif output == 'summary':
            options = ['--summary', target]
        elif output == 'trajectory':
            extra = f'neb:write_traj true\nneb:traj {target}\nneb:traj-format extxyz\n'
        elif output == 'reference_cell':
            extra = 'neb:vcneb true\n'
    else:
        if output == 'final':
            options = ['--output', target]
        elif output == 'summary':
            extra = f'dimer:summary {target}\n'
        elif output == 'trajectory':
            extra = f'dimer:write_traj true\ndimer:trajfile {target}\n'
    result = run_command('imrmlp', prepare(tmp_path, simulation, command, extra) + options, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert target in result.stderr


@pytest.mark.parametrize('failure', ['directory', 'full'])
def test_neb_example_write_failure_is_reported(tmp_path, failure):
    path = tmp_path / 'neb_simple_example.ini'
    if failure == 'directory':
        path.mkdir()
    else:
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        path.symlink_to('/dev/full')
    result = run_command('imrmlp', ['neb', '--create-input'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'neb_simple_example.ini' in result.stderr
    assert 'Sample input file created' not in result.stdout
