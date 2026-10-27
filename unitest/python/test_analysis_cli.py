# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Regression checks for free-energy estimates and trajectory smoothing."""
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_cli_error_paths import ROOT, run_command


@pytest.mark.parametrize('cell', ['10 0 0 0 12 0 0 0 14', '10 0 0 2 12 0 1 3 14'])
def test_potential_stress_diagnostic_matches_cell_finite_differences(tmp_path, cell):
    (tmp_path / 'index.map').write_text('H\n')
    (tmp_path / 'settings.ini').write_text(
        'pot:pottype lj612\npot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 4\n')
    (tmp_path / 'input.xyz').write_text(
        f'2\nLattice="{cell}" pbc="F F F" Properties=species:S:1:pos:R:3\n'
        'H 2 2 2\nH 3.2 2.7 2.3\n')
    result = run_command('imrmlp', ['testpot', '--settings', 'settings.ini',
                         '--input', 'input.xyz', '--format', 'extxyz', '--test-stress', '1',
                         '--nugedvalue', '1e-6', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    derivative_table = result.stdout.split('Stress dE/dh Comparison')[1]
    rows = re.findall(r'^[xyz]{2}\s+([-+\d.eE]+)\s+([-+\d.eE]+)',
                      derivative_table, re.MULTILINE)
    # A diagonal cell also prints the nine reconstructed virial components.
    assert len(rows) == (18 if cell.startswith('10 0 0 0') else 9)
    for numerical, analytical in rows:
        assert float(numerical) == pytest.approx(float(analytical), abs=1e-7)
    # Native virial W = -dE/dstrain gives compression-positive p = tr(W)/(3V).
    # For epsilon=sigma=1 the LJ pair virial trace is 48/r^12 - 24/r^6.
    r2 = 1.2**2 + 0.7**2 + 0.3**2
    trace = 48 / r2**6 - 24 / r2**3
    expected = trace / (3 * 10 * 12 * 14) * 160.21766208
    pressure = re.search(r'Pressure = ([-+\d.eE]+) GPa', result.stdout)
    assert pressure
    assert float(pressure[1]) == pytest.approx(expected, abs=5e-7)


def write_trajectory(path, energies):
    path.write_text(''.join(
        '1\nLattice="10 0 0 0 10 0 0 0 10" '
        f'Properties=species:S:1:pos:R:3 energy={energy}\nH 1 1 1\n'
        for energy in energies))


@pytest.mark.parametrize('energies', [[30], [-30], [1], [-30, 30], [30, 31]])
def test_free_energy_is_stable_for_large_energy_differences(tmp_path, energies):
    (tmp_path / 'index.map').write_text('H 0\n')
    write_trajectory(tmp_path / 'reference.xyz', energies)
    write_trajectory(tmp_path / 'mlp.cfg', [0] * len(energies))
    result = run_command('imrmlp', ['calc_ferr', '--input', 'reference.xyz',
                         '--input-format', 'extxyz'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = float(result.stdout.splitlines()[-1])
    kT = 300 * 8.61733326e-5
    minimum = min(energies)
    expected = minimum - kT * math.log(sum(math.exp(-(e - minimum) / kT)
                                         for e in energies) / len(energies))
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize('energy', ['nan', 'inf', '-inf'])
def test_free_energy_rejects_nonfinite_energy(tmp_path, energy):
    (tmp_path / 'index.map').write_text('H 0\n')
    write_trajectory(tmp_path / 'reference.xyz', [energy])
    write_trajectory(tmp_path / 'mlp.cfg', [0])
    result = run_command('imrmlp', ['calc_ferr', '--input', 'reference.xyz',
                         '--input-format', 'extxyz'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.fixture
def smooth_input(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    write_trajectory(tmp_path / 'input.xyz', [0, 2, -1, 3, 1])
    return ['smooth_traj', '--input', 'input.xyz', '--output', 'output.xyz',
            '--input-format', 'extxyz', '--output-format', 'extxyz',
            '--timestep', '1', '--wtype', '0', '--overwrite']


def smoothed_energies(tmp_path, args):
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    output = (tmp_path / 'output.xyz').read_text()
    return [float(value) for value in re.findall(r'\benergy=([^\s]+)', output)]


def test_smooth_cutoff_uses_inverse_fs_and_cli_overrides_settings(tmp_path, smooth_input):
    low = smoothed_energies(tmp_path, smooth_input + ['--freq-cutoff', '0.01'])
    high = smoothed_energies(tmp_path, smooth_input + ['--freq-cutoff', '1'])
    assert low == pytest.approx([1] * 5)
    assert len(high) == 5
    assert all(math.isfinite(value) for value in high)
    assert high != pytest.approx(low)
    assert smoothed_energies(tmp_path, smooth_input) == pytest.approx(high)
    settings = tmp_path / 'settings.ini'
    settings.write_text('freq-cutoff 1\n')
    assert smoothed_energies(tmp_path, smooth_input) == pytest.approx(high)
    assert smoothed_energies(tmp_path, smooth_input + ['--freq-cutoff', '0.01']) == pytest.approx(low)
    # Both settings spellings use the same inverse-femtosecond unit.
    settings.write_text('freq_cutoff 1\n')
    assert smoothed_energies(tmp_path, smooth_input) == pytest.approx(high)
    assert smoothed_energies(tmp_path, smooth_input + ['--freq-cutoff', '0.01']) == pytest.approx(low)
    settings.write_text('freq_cutoff 0.01\nfreq-cutoff 1\n')
    assert smoothed_energies(tmp_path, smooth_input) == pytest.approx(high)
    assert smoothed_energies(tmp_path, smooth_input + ['--freq-cutoff', '0.01']) == pytest.approx(low)


@pytest.mark.parametrize('rate', [1, 2, 3, 10])
def test_smooth_retains_every_selected_frame(tmp_path, smooth_input, rate):
    full = smoothed_energies(tmp_path, smooth_input + ['--freq-cutoff', '1'])
    actual = smoothed_energies(tmp_path, smooth_input + ['--freq-cutoff', '1',
                                                       '--output-rate', str(rate)])
    assert actual == pytest.approx(full[::rate])


@pytest.mark.parametrize('failure', ['empty', 'single', 'index', 'rate', 'cutoff',
                                    'timestep', 'missing_directory', 'full_device'])
def test_smooth_failures_return_nonzero(tmp_path, smooth_input, failure):
    if failure == 'empty':
        (tmp_path / 'input.xyz').write_text('')
    elif failure == 'single':
        write_trajectory(tmp_path / 'input.xyz', [0])
    elif failure == 'index':
        (tmp_path / 'index.map').write_text('')
    elif failure == 'rate':
        smooth_input += ['--output-rate', '0']
    elif failure == 'cutoff':
        smooth_input += ['--freq-cutoff', '-1']
    elif failure == 'timestep':
        smooth_input[smooth_input.index('--timestep') + 1] = '0'
    else:
        if failure == 'full_device' and not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        smooth_input[smooth_input.index('--output') + 1] = (
            '/dev/full' if failure == 'full_device' else 'missing/output.xyz')
    result = run_command('imrmlp', smooth_input, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr


def test_linear_training_help_uses_the_actual_settings_prefix(tmp_path):
    result = run_command('imrmlp', ['train_linear', '--help'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'fit:energy_weight' in result.stdout
    assert 'train:energy_weight' not in result.stdout


@pytest.fixture
def linear_training_input(tmp_path):
    (tmp_path / 'index.map').write_text('H\n')
    (tmp_path / 'settings.ini').write_text(
        'trainset input.xyz\ntrainset-format extxyz\noutput fitted.linear\n'
        'pot:pottype descriptor_linear\n'
        'pot:descriptor_linear:descriptor:dtype behler\n'
        'pot:descriptor_linear:descriptor:behler:cutoff 5\n'
        'pot:descriptor_linear:descriptor:behler:dispcount 2\n'
        'pot:descriptor_linear:descriptor:behler:dispselect 1 2\n'
        'pot:descriptor_linear:descriptor:behler:eta 0.3\n'
        'fit:force_weight 0\nfit:stress_weight 0\nfit:weight_L2 1e-12\n'
        'fit:save_to trainer.linear\n')
    frames = []
    for distance in (1.5, 2.0, 2.8, 3.7):
        cutoff = .5 * (math.cos(math.pi * distance / 5) + 1)
        energy = 2 * cutoff * (.7 - .2 * math.exp(-.3 * distance**2))
        frames.append(
            '2\nLattice="20 0 0 0 20 0 0 0 20" '
            f'Properties=species:S:1:pos:R:3 energy={energy}\n'
            f'H 2 2 2\nH {2 + distance} 2 2\n')
    (tmp_path / 'input.xyz').write_text(''.join(frames))
    return tmp_path / 'settings.ini'


@pytest.mark.parametrize('disable_debug', [False, True])
def test_linear_training_debug_uses_final_settings(tmp_path, linear_training_input, disable_debug):
    with linear_training_input.open('a') as settings:
        settings.write('debug true\n')
    flags = ['--debug=false'] if disable_debug else []
    result = run_command('imrmlp', ['train_linear', 'settings.ini',
                                  '--num_threads', '1', *flags], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Finished' in result.stdout
    assert (tmp_path / 'fitted.linear').stat().st_size > 100
    assert bool(list(tmp_path.glob('DEBUG*'))) == (not disable_debug)


def run_mpi_linear_training(tmp_path, args):
    executable = Path(os.environ.get('IMRMLP_MPI_EXECUTABLE', ROOT / 'build/mpi/imrmlp-mpi'))
    launcher = shutil.which('mpirun')
    if not launcher or not executable.is_file():
        pytest.skip('Build MPI imrmlp and provide mpirun to exercise MPI training')
    env = os.environ.copy()
    env.update({'OMPI_ALLOW_RUN_AS_ROOT': '1', 'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM': '1',
                'OMPI_MCA_rmaps_base_oversubscribe': '1', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1'})
    env['LD_LIBRARY_PATH'] = str(executable.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    return subprocess.run([launcher, '-n', '2', str(executable), 'train_linear', *args],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.extended
def test_linear_training_mpi_help_returns_on_all_ranks(tmp_path):
    result = run_mpi_linear_training(tmp_path, ['--help'])
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('fit:energy_weight') == 1
    assert 'Reading trajectory' not in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('save_failure', [None, 'trainer', 'final'])
@pytest.mark.extended
def test_linear_training_mpi_save_result_reaches_all_ranks(
        tmp_path, linear_training_input, save_failure):
    settings = linear_training_input.read_text()
    failed_path = None
    if save_failure == 'trainer':
        failed_path = 'missing/trainer.linear'
        settings = settings.replace('fit:save_to trainer.linear', f'fit:save_to {failed_path}')
    elif save_failure == 'final':
        failed_path = 'missing/final.linear'
        settings = settings.replace('output fitted.linear', f'output {failed_path}')
    linear_training_input.write_text(settings)
    result = run_mpi_linear_training(tmp_path, ['settings.ini', '--num_threads', '1'])
    output = result.stdout + result.stderr
    if failed_path is None:
        assert result.returncode == 0, output
        assert 'Finished' in result.stdout
        assert (tmp_path / 'fitted.linear').stat().st_size > 100
    else:
        assert result.returncode != 0, output
        assert failed_path in output
        assert output.count('Training failure on rank 0:') == 2, output
        assert 'Finished' not in result.stdout
        assert not (tmp_path / 'fitted.linear').exists()
        assert not (tmp_path / failed_path).exists()
    if save_failure == 'trainer':
        assert not (tmp_path / 'trainer.linear').exists()
    else:
        assert (tmp_path / 'trainer.linear').stat().st_size > 100


def test_smooth_worker_error_returns_failure_instead_of_aborting(tmp_path, smooth_input):
    (tmp_path / 'input.xyz').write_text(''.join(
        '1\nLattice="10 0 0 0 10 0 0 0 10" '
        f'Properties=species:S:1:pos:R:3:forces:R:3\nH 1 1 1 {force} 0 0\n'
        for force in ['0', 'nan']))
    result = run_command('imrmlp', smooth_input + ['--num_threads', '2'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert not (tmp_path / 'output.xyz').exists()


def test_smooth_rebuilds_velocities_for_position_only_frames(tmp_path, smooth_input):
    write_trajectory(tmp_path / 'input.xyz', [0, 1])
    (tmp_path / 'settings.ini').write_text('rebuild_velo true\n')
    result = run_command('imrmlp', smooth_input + ['--num_threads', '2'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    output = (tmp_path / 'output.xyz').read_text().lower()
    assert 'velo' in output
    assert 'nan' not in output


def extxyz_velocities(path):
    lines = path.read_text().splitlines()
    values = []
    for i in range(0, len(lines), 3):
        assert lines[i] == '1'
        properties = re.search(r'Properties=(\S+)', lines[i + 1])[1].split(':')
        offset = 0
        for name, _, width in zip(properties[::3], properties[1::3], properties[2::3]):
            if name == 'velocities':
                assert width == '3'
                values.append(list(map(float, lines[i + 2].split()[offset:offset + 3])))
                break
            offset += int(width)
        else:
            pytest.fail('Output has no velocities')
    return values


@pytest.mark.parametrize('format', ['extxyz'])
def test_smooth_rebuilt_velocities_are_written_in_angstrom_per_ps(tmp_path, format):
    count, dt, amplitude = 16, 2., .2
    omega = 2 * math.pi / (2 * count * dt)
    phases = [omega * dt * (i + .5) for i in range(count)]
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'settings.ini').write_text('rebuild_velo true\n')
    (tmp_path / 'input.xyz').write_text(''.join(
        '1\nLattice="10 0 0 0 10 0 0 0 10" pbc="T T T" '
        'Properties=species:S:1:pos:R:3\n'
        f'H {1 + amplitude * math.cos(phase):.17g} 1 1\n' for phase in phases))
    path = tmp_path / ('smoothed.nc' if format == 'netcdf' else 'smoothed.xyz')
    result = run_command('imrmlp', ['smooth_traj', '--input', 'input.xyz',
                         '--input-format', 'extxyz', '--output', path.name,
                         '--output-format', format, '--timestep', str(dt),
                         '--freq-cutoff', '1000000', '--wtype', '0'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    if format == 'netcdf':
        netcdf_file = pytest.importorskip('scipy.io').netcdf_file
        with netcdf_file(path, 'r', mmap=False) as nc:
            variable = nc.variables['velocities']
            assert variable.units == b'angstrom/picosecond'
            velocities = variable[:].copy()[:, 0, :].tolist()
    else:
        velocities = extxyz_velocities(path)
    assert len(velocities) == count
    for actual, phase in zip(velocities, phases):
        expected = [-1000 * amplitude * omega * math.sin(phase), 0., 0.]
        assert actual == pytest.approx(expected, abs=1e-10)


def test_smooth_keeps_existing_velocity_units_when_not_rebuilding(tmp_path, smooth_input):
    (tmp_path / 'input.xyz').write_text(''.join(
        '1\nLattice="10 0 0 0 10 0 0 0 10" '
        'Properties=species:S:1:pos:R:3:velocities:R:3\nH 1 1 1 .5 -.25 .1\n'
        for _ in range(4)))
    result = run_command('imrmlp', smooth_input, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    velocities = extxyz_velocities(tmp_path / 'output.xyz')
    assert len(velocities) == 4
    for actual in velocities:
        assert actual == pytest.approx([.5, -.25, .1], abs=1e-12)


def _write_rdf_cli_input(directory, *, mixed=False, unused_oxygen=False):
    directory.mkdir(parents=True, exist_ok=True)
    mapping = 'H 0\nO 1\n' if mixed or unused_oxygen else 'H 0\n'
    (directory / 'index.map').write_text(mapping)
    records = 'H 1 1 1\nO 2 1 1\nO 1 2 1\n' if mixed else 'H 1 1 1\nH 2 1 1\n'
    (directory / 'rdf.xyz').write_text(
        f'{3 if mixed else 2}\nLattice="20 0 0 0 20 0 0 0 20" pbc="T T T" '
        'Properties=species:S:1:pos:R:3\n' + records)
    (directory / 'rdf.ini').write_text('rdf:delta_r 0.1\n')
    return ['calc_prop', '--input', 'rdf.xyz', '--input-format', 'extxyz',
            '--index', 'index.map', '--settings', 'rdf.ini', '--rdf',
            '--ref-rlimit', '3', '--cutoff', '3', '--num_threads', '1']


def _rdf_rows(path):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    assert rows
    assert all(len(row) == 3 and all(math.isfinite(value) for value in row) for row in rows)
    return rows


def test_rdf_mixed_partial_coordination_uses_each_center_species(tmp_path):
    args = _write_rdf_cli_input(tmp_path, mixed=True)
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    total = _rdf_rows(tmp_path / 'gr_total.txt')
    curves = {pair: _rdf_rows(tmp_path / f'gr_partial_{pair}')
              for pair in ['H-H', 'H-O', 'O-H', 'O-O']}
    assert total[-1][2] == pytest.approx(2.)
    for pair, coordination in {'H-H': 0., 'H-O': 2., 'O-H': 1., 'O-O': 1.}.items():
        assert curves[pair][-1][2] == pytest.approx(coordination)
        assert len(curves[pair]) == len(total)
    shell_one = 4 * math.pi / 3 * (1.05**3 - .95**3)
    for pair in ['H-O', 'O-H']:
        row = next(row for row in curves[pair] if abs(row[0] - 1.) < 1e-12)
        assert row[1] == pytest.approx(8000 / shell_one, rel=6e-6)
    total_one = next(row for row in total if abs(row[0] - 1.) < 1e-12)
    assert total_one[1] == pytest.approx(4 * 8000 / (9 * shell_one), rel=6e-6)
    shell_oo = 4 * math.pi / 3 * (1.45**3 - 1.35**3)
    oxygen_pair = next(row for row in curves['O-O'] if abs(row[0] - 1.4) < 1e-12)
    assert oxygen_pair[1] == pytest.approx(4000 / shell_oo, rel=6e-6)


@pytest.mark.parametrize('coordinates,raw_coordination', [([-.1, .1], 1.), ([.1, 9.9], 0.)])
def test_rdf_no_pbc_uses_original_unwrapped_geometry(tmp_path, coordinates, raw_coordination):
    for mode, options, expected in [
            ('raw', ['--rdf-kd', '--rdf-no-pbc'], raw_coordination),
            ('periodic-kd', ['--rdf-kd'], 1.),
            ('periodic', [], 1.)]:
        directory = tmp_path / mode
        args = _write_rdf_cli_input(directory)
        args[args.index('--ref-rlimit') + 1] = '1'
        args[args.index('--cutoff') + 1] = '1'
        (directory / 'rdf.xyz').write_text(
            '2\nLattice="10 0 0 0 10 0 0 0 10" pbc="T T T" '
            'Properties=species:S:1:pos:R:3\n'
            + ''.join(f'H {x} 1 1\n' for x in coordinates))
        result = run_command('imrmlp', args + options, directory)
        assert result.returncode == 0, result.stdout + result.stderr
        rows = _rdf_rows(directory / 'gr_total.txt')
        assert rows[-1][2] == pytest.approx(expected)
        close_shell = next(row for row in rows if abs(row[0] - .2) < 1e-12)
        assert close_shell[2] == pytest.approx(expected)
        if expected:
            shell_volume = 4 * math.pi / 3 * (.25**3 - .15**3)
            assert close_shell[1] == pytest.approx(500. / shell_volume, rel=6e-6)
        else:
            assert all(row[1:] == [0., 0.] for row in rows)


def test_rdf_unused_species_keep_finite_files_without_changing_present_curves(tmp_path):
    outputs = {}
    for label, unused, rho_all in [('only-H', False, False), ('unused-O', True, False),
                                    ('unused-O-rho-all', True, True)]:
        directory = tmp_path / label
        args = _write_rdf_cli_input(directory, unused_oxygen=unused)
        if rho_all:
            args.append('--rdf-rho-all')
        result = run_command('imrmlp', args, directory)
        assert result.returncode == 0, result.stdout + result.stderr
        total = _rdf_rows(directory / 'gr_total.txt')
        hydrogen = _rdf_rows(directory / 'gr_partial_H-H')
        outputs[label] = total, hydrogen
        assert total[-1][2] == pytest.approx(1.)
        assert hydrogen[-1][2] == pytest.approx(1.)
        if unused:
            for pair in ['H-O', 'O-H', 'O-O']:
                missing = _rdf_rows(directory / f'gr_partial_{pair}')
                assert len(missing) == len(total)
                for absent_row, total_row in zip(missing, total):
                    assert absent_row[0] == total_row[0]
                    assert absent_row[1:] == [0., 0.]
    for label in ['unused-O', 'unused-O-rho-all']:
        for actual, expected in zip(outputs[label], outputs['only-H']):
            assert len(actual) == len(expected)
            for row, reference in zip(actual, expected):
                assert row == pytest.approx(reference, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize('blocked', ['gr_total.txt', 'gr_partial_H-H'])
def test_rdf_reports_blocked_total_and_partial_output_paths(tmp_path, blocked):
    args = _write_rdf_cli_input(tmp_path)
    (tmp_path / blocked).mkdir()
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert blocked in result.stdout + result.stderr
    assert 'RDF computation complete' not in result.stdout


def _write_vacf_cli_input(directory, frames, *, explicit_velocity, timestep, skip=1):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'index.map').write_text('H 1 1\n')
    properties = 'species:S:1:pos:R:3' + (':velocities:R:3' if explicit_velocity else '')
    records = []
    for frame in frames:
        velocity = ' .4 -.2 .1' if explicit_velocity else ''
        records.append('1\nLattice="50 0 0 0 50 0 0 0 50" pbc="F F F" '
                       f'Properties={properties}\n'
                       f'H {1 + .4 * frame:.17g} {2 - .2 * frame:.17g} {3 + .1 * frame:.17g}{velocity}\n')
    (directory / 'vacf.xyz').write_text(''.join(records))
    (directory / 'vacf.ini').write_text(
        f'vacf:timestep {timestep}\nvacf:algo 2\nvacf:total true\n'
        'vacf:types false\nvacf:track false\nvacf:groups false\n'
        'vacf:normalize false\nvacf:vdos true\nvacf:dfreq 1\nvacf:fmax 0.25\n')
    return ['calc_prop', '--settings', 'vacf.ini', '--input', 'vacf.xyz',
            '--input-format', 'extxyz', '--index', 'index.map', '--vacf',
            '--skip-every', str(skip), '--num_threads', '1']


def _vacf_numeric_rows(path):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    assert rows
    assert all(len(row) == 5 and all(math.isfinite(value) for value in row) for row in rows)
    return rows


@pytest.mark.parametrize('explicit_velocity', [True, False])
def test_vacf_sampling_uses_effective_timestep_without_rescaling_velocity(tmp_path, explicit_velocity):
    outputs = []
    for label, frames, timestep, skip in [('raw', range(9), 1, 2),
                                         ('sampled', range(0, 9, 2), 2, 1)]:
        directory = tmp_path / label
        args = _write_vacf_cli_input(directory, frames, explicit_velocity=explicit_velocity,
                                    timestep=timestep, skip=skip)
        result = run_command('imrmlp', args, directory)
        assert result.returncode == 0, result.stdout + result.stderr
        correlation = _vacf_numeric_rows(directory / 'VACF_total.dat')
        spectrum = _vacf_numeric_rows(directory / 'VACF_vdos_total.dat')
        outputs.append((correlation, spectrum))
        samples = 5 if explicit_velocity else 3  # Central differences omit both endpoint frames.
        assert len(correlation) == samples
        assert [row[0] for row in correlation] == pytest.approx([2 * i for i in range(samples)])
        # Velocities are .4, -.2, .1 Angstrom/ps, whether supplied or inferred
        # from sampled position increments .8, -.4, .2 over two ps.
        assert correlation[0][1:] == pytest.approx([.16, .04, .01, .07], abs=1e-12)
        # dfreq is deliberately coarse, so the transform length is samples and
        # spacing depends on the effective dt, rather than a padded target grid.
        spacing = 1 / (2 * 2 * (samples - 1))
        assert len(spectrum) == samples
        assert [row[0] for row in spectrum] == pytest.approx([i * spacing for i in range(samples)])
    for actual, expected in zip(outputs[0], outputs[1]):
        assert len(actual) == len(expected)
        for row, reference in zip(actual, expected):
            assert row == pytest.approx(reference, rel=1e-11, abs=1e-12)


@pytest.mark.parametrize('failed_output', ['VACF_total.dat', 'VACF_vdos_total.dat'])
def test_vacf_cli_reports_small_output_close_failures(tmp_path, failed_output):
    if not Path('/dev/full').exists():
        pytest.skip('/dev/full is unavailable')
    args = _write_vacf_cli_input(tmp_path, range(5), explicit_velocity=True, timestep=2)
    (tmp_path / failed_output).symlink_to('/dev/full')
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert failed_output in result.stdout + result.stderr


def _empty_analysis_input(directory, records):
    (directory / 'index.map').write_text('H 0\nO 1\n')
    (directory / 'analysis.xyz').write_text(
        f'{len(records)}\nLattice="20 0 0 0 20 0 0 0 20" pbc="F F F" '
        'Properties=species:S:1:pos:R:3\n' +
        ''.join(f'{symbol} {x} {y} {z}\n' for symbol, (x, y, z) in records))
    return ['calc_prop', '--input', 'analysis.xyz', '--input-format', 'extxyz',
            '--index', 'index.map', '--cutoff', '1.1', '--num_threads', '1']


def _angle_percentage_rows(path):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 181
    assert all(len(row) == 2 and all(math.isfinite(value) for value in row) for row in rows)
    assert [row[0] for row in rows] == pytest.approx(list(range(181)), abs=1e-10)
    return rows


def test_empty_ha_statistics_keep_existing_files_and_finite_zero_fractions(tmp_path):
    args = _empty_analysis_input(tmp_path, [('H', (1, 1, 1)), ('H', (6, 1, 1))])
    result = run_command('imrmlp', args + ['--ha-index'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in ['out_total.txt', 'Bond_H_H.txt', 'Bond_H_O.txt', 'Bond_O_O.txt']:
        path = tmp_path / name
        assert path.is_file()
        rows = []
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) == 3 and (fields[0].isdigit() or fields[0] == 'Total'):
                count, fraction = int(fields[1]), float(fields[2])
                assert count == 0
                assert math.isfinite(fraction) and fraction == 0.
                rows.append(fields)
        assert rows and rows[-1][0] == 'Total'


def test_bondangle_missing_chemical_combinations_are_zero_with_nonempty_right_angle(tmp_path):
    args = _empty_analysis_input(tmp_path, [
        ('H', (1, 1, 1)), ('O', (2, 1, 1)), ('O', (1, 2, 1)),
    ])
    result = run_command('imrmlp', args + ['--bondangle'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = list(tmp_path.glob('analyse_bondangles*.txt'))
    assert len(paths) == 6
    for path in paths:
        rows = _angle_percentage_rows(path)
        expected = [0.] * 181
        if path.name == 'analyse_bondanglesOHO.txt':
            expected[90] = 100.
        assert [row[1] for row in rows] == pytest.approx(expected, abs=1e-10)
    assert (tmp_path / 'analyse_bondanglesOHO.txt').is_file()
    assert (tmp_path / 'analyse_bondanglesHHH.txt').is_file()


def test_dihedral_missing_chemical_combinations_keep_nonempty_ninety_degree_percent(tmp_path):
    args = _empty_analysis_input(tmp_path, [
        ('H', (2.3, 3.7, 3)), ('H', (3, 3, 3)),
        ('H', (4, 3, 3)), ('H', (4.7, 3, 3.7)),
    ])
    result = run_command('imrmlp', args + ['--dihedral'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    paths = list(tmp_path.glob('analyse_dihedralangles*.txt'))
    assert len(paths) == 10
    for path in paths:
        rows = _angle_percentage_rows(path)
        expected = [0.] * 181
        if path.name == 'analyse_dihedralanglesHHHH.txt':
            expected[90] = 100.
        assert [row[1] for row in rows] == pytest.approx(expected, abs=1e-10)
    assert (tmp_path / 'analyse_dihedralanglesHHHH.txt').is_file()
    assert (tmp_path / 'analyse_dihedralanglesOOOO.txt').is_file()


@pytest.mark.parametrize('output', ['ha-directory', 'angle-full-device'])
def test_analysis_empty_output_failures_report_the_actual_path(tmp_path, output):
    args = _empty_analysis_input(tmp_path, [('H', (1, 1, 1)), ('H', (6, 1, 1))])
    if output == 'ha-directory':
        name = 'out_total.txt'
        (tmp_path / name).mkdir()
        flag = '--ha-index'
    else:
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        name = 'analyse_bondanglesHHH.txt'
        (tmp_path / name).symlink_to('/dev/full')
        flag = '--bondangle'
    result = run_command('imrmlp', args + [flag], tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert name in result.stdout + result.stderr


def test_bop_cli_keeps_nonzero_normalized_w2_in_annotated_output(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.xyz').write_text(
        '2\nLattice="10 0 0 0 10 0 0 0 10" pbc="F F F" '
        'Properties=species:S:1:pos:R:3\nH 2 2 2\nH 2 2 3\n')
    (tmp_path / 'bop.ini').write_text('bop:maxl 4\nbop:average true\nbop:corase true\n')
    result = run_command('imrmlp', [
        'calc_prop', '--input', 'input.xyz', '--input-format', 'extxyz',
        '--settings', 'bop.ini', '--cutoff', '1.1', '--bop',
        '--output-xyz', 'output.xyz', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / 'output.xyz').read_text().splitlines()
    assert lines[0] == '2' and len(lines) == 4
    properties = re.search(r'Properties=(\S+)', lines[1])[1].split(':')
    columns = {}
    offset = 0
    for name, _, width in zip(properties[::3], properties[1::3], properties[2::3]):
        columns[name] = (offset, int(width))
        offset += int(width)
    # One axial bond has only m=0. The independent 3j(2,2,2;0,0,0)
    # and 3j(4,4,4;0,0,0) constants give the normalized W values below.
    expected_w = [-math.sqrt(2 / 35), math.sqrt(18 / 1001)]
    for line in lines[2:]:
        fields = line.split()
        for key in ['bopW', 'bopW_A', 'bopW_C']:
            begin, width = columns[key]
            assert width == 2
            assert list(map(float, fields[begin:begin + width])) == pytest.approx(
                expected_w, abs=1e-12)
        for key in ['bopQ', 'bopQ_A', 'bopQ_C']:
            begin, width = columns[key]
            assert list(map(float, fields[begin:begin + width])) == pytest.approx([1, 1], abs=1e-12)

@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('cutoff', [0, 1000000])
def test_smooth_partial_fixed_cell_preserves_normal_motion_and_velocity_units(tmp_path, padded, cutoff):
    count, dt = 16, 2.
    omega = 2 * math.pi / (2 * count * dt)
    phases = [omega * dt * (i + .5) for i in range(count)]
    cell = '10 0 0 0 11 0 0 0 ' + ('12' if padded else '0')
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'settings.ini').write_text('rebuild_velo true\n')
    (tmp_path / 'input.xyz').write_text(''.join(
        f'1\nLattice="{cell}" pbc="T T F" Properties=species:S:1:pos:R:3\n'
        f'H {2 + .2 * math.cos(phase):.17g} 1 {3 + .3 * math.cos(phase):.17g}\n'
        for phase in phases))
    result = run_command('imrmlp', ['smooth_traj', '--input', 'input.xyz',
                         '--input-format', 'extxyz', '--output', 'output.xyz',
                         '--output-format', 'extxyz', '--timestep', str(dt),
                         '--freq-cutoff', str(cutoff), '--wtype', '0',
                         '--num_threads', '2'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    path = tmp_path / 'output.xyz'
    lines = path.read_text().splitlines()
    velocities = extxyz_velocities(path)
    assert len(velocities) == count
    for i, (velocity, phase) in enumerate(zip(velocities, phases)):
        expected = ([0., 0., 0.] if cutoff == 0 else
                    [-1000 * .2 * omega * math.sin(phase), 0.,
                     -1000 * .3 * omega * math.sin(phase)])
        assert velocity == pytest.approx(expected, abs=1e-10)
        position = list(map(float, lines[3 * i + 2].split()[1:4]))
        expected_position = ([2., 1., 3.] if cutoff == 0 else
                             [2 + .2 * math.cos(phase), 1., 3 + .3 * math.cos(phase)])
        assert position == pytest.approx(expected_position, abs=1e-11)
        assert 'pbc="T T F"' in lines[3 * i + 1]
        lattice = list(map(float, re.search(r'Lattice="([^"]+)"', lines[3 * i + 1])[1].split()))
        assert lattice == pytest.approx([10., 0., 0., 0., 11., 0., 0., 0., 12. if padded else 0.])
