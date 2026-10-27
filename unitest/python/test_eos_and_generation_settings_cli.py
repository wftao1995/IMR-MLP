"""EOS error paths and propagation of structure-generator settings."""
from pathlib import Path
import math
import re

import pytest

from test_cli_error_paths import ROOT, run_command


@pytest.fixture
def eos_curve(tmp_path):
    (tmp_path / 'index.map').write_text('H 1\n')
    (tmp_path / 'curve.dat').write_text('# Volume Energy Pressure Enthalpy\n' + ''.join(
        f'{v} {3 * (v ** (-2 / 3) - .2) ** 2 + 20 * (v ** (-2 / 3) - .2) ** 3} 0 0\n'
        for v in range(8, 17)))
    return ['calc_eos', '--fit-only', '--output', 'curve.dat', '--output-eos', 'fit.dat',
            '--out-veph', 'veph.dat', '--num_threads', '1']


@pytest.mark.parametrize('failure', ['missing', 'directory', 'empty', 'one_column',
                                    'text', 'suffix', 'nan', 'infinity', 'zero_volume',
                                    'negative_volume', 'too_few', 'repeated_volume'])
def test_eos_rejects_invalid_curve_data(tmp_path, eos_curve, failure):
    path = tmp_path / 'curve.dat'
    if failure == 'missing':
        path.unlink()
    elif failure == 'directory':
        path.unlink()
        path.mkdir()
    else:
        path.write_text({
            'empty': '# No samples\n', 'one_column': '10\n', 'text': 'bad 1\n',
            'suffix': '10oops 1\n', 'nan': '10 nan\n', 'infinity': 'inf 1\n',
            'zero_volume': '0 1\n', 'negative_volume': '-10 1\n',
            'too_few': '10 1\n11 2\n12 3\n', 'repeated_volume': '10 1\n' * 4,
        }[failure])
    result = run_command('imrmlp', eos_curve, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr
    assert not (tmp_path / 'fit.dat').exists()


@pytest.mark.parametrize('option', ['--output-eos', '--out-veph'])
@pytest.mark.parametrize('failure', ['missing_directory', 'full_device'])
def test_eos_reports_failed_fit_outputs(tmp_path, eos_curve, option, failure):
    if failure == 'full_device' and not Path('/dev/full').exists():
        pytest.skip('/dev/full is unavailable')
    eos_curve[eos_curve.index(option) + 1] = (
        '/dev/full' if failure == 'full_device' else 'missing/output.dat')
    result = run_command('imrmlp', eos_curve, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr


def test_eos_fits_valid_curve_and_writes_finite_results(tmp_path, eos_curve):
    result = run_command('imrmlp', eos_curve, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    fit = (tmp_path / 'fit.dat').read_text().lower()
    assert 'fitted parameters' in fit
    assert 'nan' not in fit and 'inf' not in fit
    rows = [line.split() for line in (tmp_path / 'veph.dat').read_text().splitlines()
            if not line.startswith('#')]
    assert len(rows) == 100
    assert all(math.isfinite(float(value)) for row in rows for value in row)


@pytest.mark.parametrize('failure', ['curve_directory', 'curve_full', 'trajectory_directory', 'trajectory_full'])
def test_eos_reports_failed_calculated_outputs(tmp_path, failure):
    # The shared potential has two species, even though this structure uses one.
    (tmp_path / 'index.map').write_text('H\nHe\n')
    (tmp_path / 'input.xyz').write_text(
        '2\nLattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3\n'
        'H 1 1 1\nH 3 1 1\n')
    (tmp_path / 'pot.ini').write_text(
        f'pot:pottype mtp\npot:potfile {ROOT / "unitest/test-data/potential.mtp"}\n')
    output = 'curve.dat'
    if failure == 'curve_directory':
        output = 'missing/curve.dat'
    elif failure == 'curve_full':
        output = '/dev/full'
    elif failure == 'trajectory_directory':
        (tmp_path / 'p-e.cfg').mkdir()
    else:
        (tmp_path / 'p-e.cfg').symlink_to('/dev/full')
    if failure.endswith('full') and not Path('/dev/full').exists():
        pytest.skip('/dev/full is unavailable')
    result = run_command('imrmlp', ['calc_eos', '--settings', 'pot.ini', '--curve', 'pv',
                                    '--input', 'input.xyz', '--skip-prelax', '--npos', '0',
                                    '--nneg', '0', '--output', output, '--num_threads', '1'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert ('write EOS' in result.stderr or 'open EOS' in result.stderr), result.stderr


def frames(path):
    lines = path.read_text().splitlines()
    result = []
    i = 0
    while i < len(lines):
        count = int(lines[i])
        header = lines[i + 1]
        positions = [[float(x) for x in line.split()[1:4]] for line in lines[i + 2:i + count + 2]]
        cell = [float(x) for x in re.search(r'Lattice="([^"]+)"', header).group(1).split()]
        result.append((count, cell, positions))
        i += count + 2
    return result














def eos_relaxation_input(tmp_path, gtol):
    (tmp_path / 'index.map').write_text('H\n')
    (tmp_path / 'input.xyz').write_text(
        '2\nLattice="20 0 0 0 20 0 0 0 20" '
        'Properties=species:S:1:pos:R:3 pbc="F F F"\n'
        'H 0 0 0\nH 1.5 0 0\n')
    (tmp_path / 'eos.ini').write_text(
        'pot:pottype lj612\npot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 3\n'
        'relax:driver fire\nrelax:maxiteration 1\nrelax:stepmax .001\n'
        f'relax:gtol {gtol}\nrelax:relax_cell false\nrelax:write_traj false\n')
    return ['calc_eos', '--settings', 'eos.ini', '--input', 'input.xyz',
            '--input-format', 'extxyz', '--output', 'curve.dat',
            '--output-eos', 'fit.dat', '--out-veph', 'veph.dat',
            '--npos', '1', '--nneg', '1', '--num_threads', '1']


@pytest.mark.parametrize('curve,skip_preliminary,stage', [
    ('pv', False, 'PV preliminary'),
    ('pv', True, 'PV pressure'),
    ('ev', True, 'EV expansion volume'),
])
def test_eos_stops_before_publishing_unconverged_relaxations(
        tmp_path, curve, skip_preliminary, stage):
    # LJ at r=1.5 has nonzero force; one small step cannot satisfy 1e-20.
    arguments = eos_relaxation_input(tmp_path, '1e-20') + ['--curve', curve]
    if skip_preliminary:
        arguments.append('--skip-prelax')
    trajectory = 'p-e.cfg' if curve == 'pv' else 'e-v.cfg'
    outputs = ['curve.dat', trajectory, 'fit.dat', 'veph.dat']
    previous = {}
    for filename in outputs:
        path = tmp_path / filename
        path.write_text(f'previous {filename}\n')
        previous[filename] = path.read_bytes()
    result = run_command('imrmlp', arguments, tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    assert f'EOS relaxation did not converge at {stage}' in result.stderr
    assert re.search(r'EOS relaxation did not converge at [^\n]+: \w+', result.stderr)
    for filename, original in previous.items():
        assert (tmp_path / filename).read_bytes() == original


def test_eos_converged_pressure_samples_keep_pressure_and_enthalpy(tmp_path):
    arguments = eos_relaxation_input(tmp_path, '1e20') + [
        '--curve', 'pv', '--skip-prelax', '--press-ub', '1', '--press-lb', '-1']
    result = run_command('imrmlp', arguments, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [[float(value) for value in line.split()]
            for line in (tmp_path / 'curve.dat').read_text().splitlines()
            if line and not line.startswith('#')]
    assert len(rows) == 3
    q = (1/1.5)**6
    energy_per_atom = 2*(q*q-q)
    derivative = 24/1.5*(q-2*q*q)
    conversion = 160.21766208
    reference_pressure = -1.5*derivative/(3*8000)*conversion
    for row, pressure in zip(rows, [-1, reference_pressure, 1]):
        assert row[0] == pytest.approx(pressure, abs=1e-6)
        assert row[1] == pytest.approx(4000., abs=1e-6)
        assert row[2] == pytest.approx(energy_per_atom, abs=1e-6)
        assert row[3] == pytest.approx(energy_per_atom + pressure/conversion*4000, abs=1e-6)
    assert (tmp_path / 'p-e.cfg').stat().st_size > 0
    assert 'did not converge' not in result.stderr


@pytest.mark.parametrize('source', ['settings', 'cli', 'settings-output'])
def test_eos_pressure_defaults_follow_the_effective_curve_setting(tmp_path, source):
    eos_relaxation_input(tmp_path, '1e20')
    options = 'npos 0\nnneg 0\nskip-prelax true\n'
    arguments = ['calc_eos', '--settings', 'eos.ini', '--input', 'input.xyz',
                 '--input-format', 'extxyz', '--num_threads', '1']
    if source == 'cli':
        arguments += ['--curve', 'pv']
    else:
        options += 'curve pv\n'
    output = 'custom.dat' if source == 'settings-output' else 'p-e.dat'
    if source == 'settings-output':
        options += 'output custom.dat\n'
    with (tmp_path / 'eos.ini').open('a') as settings:
        settings.write(options)
    result = run_command('imrmlp', arguments, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'p-e.cfg').stat().st_size > 0
    rows = [[float(value) for value in line.split()]
            for line in (tmp_path / output).read_text().splitlines()
            if line and not line.startswith('#')]
    assert len(rows) == 1
    assert rows[0][1] == pytest.approx(4000.)
    q = (1 / 1.5)**6
    assert rows[0][2] == pytest.approx(2*(q*q-q), abs=1e-6)
    assert not (tmp_path / 'e-v.dat').exists()
    assert not (tmp_path / 'e-v.cfg').exists()
    if source == 'settings-output':
        assert not (tmp_path / 'p-e.dat').exists()


def test_eos_unsorted_fit_interpolates_the_entire_sample_volume_domain(tmp_path):
    (tmp_path / 'index.map').write_text('H\n')
    volumes = [12, 8, 16, 10, 14]
    source = '# Volume Energy\n' + ''.join(
        f'{v} {(v**(-2/3)-.2)**2:.17g}\n' for v in volumes)
    (tmp_path / 'curve.dat').write_text(source)
    arguments = ['calc_eos', '--fit-only', '--output', 'curve.dat',
                 '--output-eos', 'fit.dat', '--out-veph', 'veph.dat',
                 '--ninterp', '3', '--num_threads', '1']
    result = run_command('imrmlp', arguments, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [[float(value) for value in line.split()]
            for line in (tmp_path / 'veph.dat').read_text().splitlines()
            if line and not line.startswith('#')]
    assert len(rows) == 3
    for row, volume in zip(rows, [8., 12., 16.]):
        x = volume**(-2/3)
        energy = (x-.2)**2
        pressure = 4/3*(x-.2)*volume**(-5/3)
        assert row[0] == pytest.approx(volume, abs=1e-10)
        assert row[1] == pytest.approx(energy, abs=1e-11)
        assert row[2] == pytest.approx(pressure*160.21766208, abs=1e-9)
        assert row[3] == pytest.approx(energy+pressure*volume, abs=1e-10)
    fitted = (tmp_path / 'fit.dat').read_text()
    equilibrium = float(re.search(r'^V0 =\s*(\S+)', fitted, re.MULTILINE).group(1))
    assert equilibrium == pytest.approx(.2**(-1.5), abs=1e-6)
    assert (tmp_path / 'curve.dat').read_text() == source
