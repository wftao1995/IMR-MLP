# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Exercise standalone command exit codes and diagnostics in isolated directories."""
import json
import os
import re
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build' / os.environ.get('BUILD_MODE', 'release')


def run_command(name, args, cwd):
    executable = BUILD / name
    if not executable.exists():
        pytest.skip(f'Build {name} before running CLI tests')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(BUILD) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = '1'
    return subprocess.run([str(executable), *map(str, args)], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=30)


@pytest.fixture
def fitness_input(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.xyz').write_text(
        '1\nLattice="10 0 0 0 10 0 0 0 10" '
        'Properties=species:S:1:pos:R:3 energy=0\nH 0 0 0\n')
    return ['--input-format', 'extxyz', '--trainset', 'input.xyz',
            '--index', 'index.map', '--output', 'output.cfg']


@pytest.mark.parametrize('failure', ['missing_input', 'empty_input', 'missing_index',
                                    'empty_index', 'missing_directory', 'full_device'])
def test_fitness_reports_failures(tmp_path, fitness_input, failure):
    if failure == 'missing_input':
        (tmp_path / 'input.xyz').unlink()
    elif failure == 'empty_input':
        (tmp_path / 'input.xyz').write_text('')
    elif failure == 'missing_index':
        (tmp_path / 'index.map').unlink()
    elif failure == 'empty_index':
        (tmp_path / 'index.map').write_text('')
    elif failure == 'missing_directory':
        fitness_input[-1] = 'missing/output.cfg'
    else:
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        fitness_input[-1] = '/dev/full'
    result = run_command('calc_fitness', fitness_input, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Error:' in result.stderr
    assert 'Complete!' not in result.stdout


def test_fitness_success_and_empty_selection(tmp_path, fitness_input):
    result = run_command('calc_fitness', fitness_input, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'output.cfg').stat().st_size > 0
    result = run_command('calc_fitness', fitness_input + ['--threshold', '2'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('label,option', [('energy', 'energy'), ('forces', 'force'), ('stress', 'stress')])
def test_fitness_missing_labels_warn_and_use_neutral_contribution(tmp_path, fitness_input, label, option):
    path = tmp_path / 'input.xyz'
    path.write_text(path.read_text().replace(' energy=0', ''))
    args = fitness_input + ['--select_by_energy', 'false', f'--select_by_{option}', 'true']
    result = run_command('calc_fitness', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'has no {label}' in result.stdout + result.stderr
    output = (tmp_path / 'output.cfg').read_text()
    assert float(re.search(r'Feature\s+Fitness\s+(\S+)', output).group(1)) == pytest.approx(1)


def test_fitness_partial_frames_keep_available_contributions(tmp_path, fitness_input):
    (tmp_path / 'input.xyz').write_text(
        '1\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 energy=-40\nH 0 0 0\n'
        '1\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:forces:R:3\nH 0 0 0 1 0 0\n')
    result = run_command('calc_fitness', fitness_input + ['--select_by_force', 'true'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Configuration 1 has no forces' in result.stdout + result.stderr
    assert 'Configuration 2 has no energy' in result.stdout + result.stderr
    values = re.findall(r'Feature\s+Fitness\s+(\S+)', (tmp_path / 'output.cfg').read_text())
    assert [float(value) for value in values] == pytest.approx([0.5, 0.5])


def test_spline_export_reports_missing_directory(tmp_path):
    result = run_command('imrmlp', ['mtp2spline', ROOT / 'unitest/test-data/potential.mtp',
                                    'missing/output.mtp'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'missing/output.mtp' in result.stderr


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('rmin', [None, 2.0])
def test_spline_export_inner_radius_and_split_bases(tmp_path, split, rmin):
    # The fixture also contains binary training state after the potential text.
    data = (ROOT / 'unitest/test-data/potential.mtp').read_bytes()
    end = data.index(b'\n', data.index(b'moment_coeffs ='))
    potential = data[:end + 1].decode()
    if split:
        start = potential.index('radial_basis_type')
        end = potential.index('\tradial_funcs_count')
        basis = potential[start:end]
        potential = potential[:start] + 'radial_basis_split = 1\n' + ''.join(
            f'pair {i} {j}\n{basis}' for i in range(2) for j in range(2)) + potential[end:]
    (tmp_path / 'input.mtp').write_text(potential)
    extra = [] if rmin is None else ['--r-min', str(rmin)]
    result = run_command('imrmlp', ['mtp2spline', 'input.mtp', 'output.mtp',
                                   '--delta-r', '0.1', *extra], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    output = (tmp_path / 'output.mtp').read_text()
    assert 'radial_basis_split' not in output
    match = re.search(r'potential_tag = ([^\n]*)\n(radial_basis_type = RBSpline\n)', output)
    assert match
    metadata = json.loads(match[1])
    assert metadata['schema'] == 'gpml.mtp.metadata' and metadata['version'] == 1
    assert metadata['type'] == 'MTP' and metadata['model'] == 'linear'
    lines = output.split('radial_splines\n')[1].splitlines()
    nr, dr, radial_count = lines[0].split()
    nr, radial_count = int(nr), int(radial_count)
    radii = list(map(float, lines[1].split()))
    assert len(radii) == nr
    inner = 1.2 if rmin is None else rmin
    for line in lines[2:2 + 4 * radial_count]:
        values = list(map(float, line.split()[3:]))
        assert len(values) == nr
        assert values[0] != 0
        assert all(value == values[0] for r, value in zip(radii, values) if r < inner)
        assert values[-1] == pytest.approx(0, abs=1e-12)
    assert lines[2 + 4 * radial_count].startswith('alpha_moments_count')
