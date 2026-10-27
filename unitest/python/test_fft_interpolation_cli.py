"""Fourier densification is opt-in and CLI sampling occurs afterwards."""
import math
import re

import numpy as np
import pytest

from test_cli_error_paths import run_command


def prepare(tmp_path, *, factor=3, rebuild=True):
    count, dt = 12, 2.
    (tmp_path/'index.map').write_text('H 0\n')
    (tmp_path/'settings.ini').write_text(f'interpolation_factor {factor}\n'+
        ('rebuild_force true\nrebuild_velo true\n' if rebuild else ''))
    (tmp_path/'input.xyz').write_text(''.join(
        '1\npbc="F F F" Lattice="0 0 0 0 0 0 0 0 0" '
        f'energy={3+.4*math.cos(2*math.pi*(j+.5)/(2*count)):.17g} config_type="frame-{j}" '
        'Properties=species:S:1:pos:R:3:masses:R:1:kind:I:1\n'
        f'H {2+.2*math.cos(2*math.pi*(j+.5)/(2*count)):.17g} 1 .5 4 {j}\n'
        for j in range(count)))
    return count, dt


def command(tmp_path, dt, *extra):
    return run_command('imrmlp', [
        'smooth_traj', '--input', 'input.xyz', '--output', 'output.xyz',
        '--input-format', 'extxyz', '--output-format', 'extxyz',
        '--timestep', str(dt), '--freq-cutoff', '1000000', '--wtype', '0',
        '--num_threads', '1', '--lossless', '1', *extra], tmp_path)


def headers(path):
    lines = path.read_text().splitlines()
    return [lines[i+1] for i in range(0, len(lines), 3)]


@pytest.mark.parametrize('factor,rate', [(2, 1), (3, 2)])
def test_interpolation_cli_values_derivatives_time_and_output_rate(tmp_path, factor, rate):
    import imrmlp
    count, dt = prepare(tmp_path, factor=factor)
    result = command(tmp_path, dt, '--output-rate', str(rate))
    assert result.returncode == 0, result.stdout+result.stderr
    frames = imrmlp.read(tmp_path/'output.xyz')
    indices = np.arange(0, factor*(count-1)+1, rate)
    assert len(frames) == len(indices)
    omega = 2*math.pi/(2*count*dt)
    phase = omega*dt*(indices/factor+.5)
    np.testing.assert_allclose([f.positions[0, 0] for f in frames], 2+.2*np.cos(phase), atol=2e-11)
    np.testing.assert_allclose([f.velocities[0, 0] for f in frames], -1000*.2*omega*np.sin(phase), atol=2e-9)
    np.testing.assert_allclose([f.forces[0, 0] for f in frames], -103.64269652680505*4*.2*omega**2*np.cos(phase), atol=2e-9)
    np.testing.assert_allclose([f.energy for f in frames], 3+.4*np.cos(phase), atol=2e-11)
    for header, index in zip(headers(tmp_path/'output.xyz'), indices):
        elapsed = re.search(r'\bfft_elapsed_fs=([^ ]+)', header)
        assert elapsed, header
        assert float(elapsed.group(1)) == pytest.approx(index*dt/factor)
        assert f'config_type="frame-{index//factor}"' in header, header
    rows = (tmp_path/'output.xyz').read_text().splitlines()[2::3]
    assert [int(row.split()[-1]) for row in rows] == list(indices//factor)


def test_interpolation_cli_override_and_default_factor_are_independent(tmp_path):
    count, dt = prepare(tmp_path, factor=3, rebuild=False)
    result = command(tmp_path, dt, '--interpolation-factor', '1')
    assert result.returncode == 0, result.stdout+result.stderr
    output = tmp_path/'output.xyz'
    assert len(headers(output)) == count
    assert 'fft_elapsed_fs' not in output.read_text()
    old = output.read_bytes()
    output.unlink()
    (tmp_path/'settings.ini').write_text('')
    result = command(tmp_path, dt)
    assert result.returncode == 0, result.stdout+result.stderr
    assert output.read_bytes() == old


@pytest.mark.parametrize('factor', ['0', '-1', '2.5', '2147483647'])
def test_bad_interpolation_factor_preserves_existing_output(tmp_path, factor):
    _, dt = prepare(tmp_path)
    (tmp_path/'output.xyz').write_text('previous result\n')
    result = command(tmp_path, dt, '--interpolation-factor', factor, '--overwrite')
    assert result.returncode != 0
    assert (tmp_path/'output.xyz').read_text() == 'previous result\n'


def test_default_cfg_output_retains_elapsed_time_after_decimation(tmp_path):
    count, dt = prepare(tmp_path, factor=3, rebuild=False)
    result = run_command('imrmlp', [
        'smooth_traj', '--input', 'input.xyz', '--output', 'output.cfg',
        '--input-format', 'extxyz', '--timestep', str(dt),
        '--freq-cutoff', '1000000', '--wtype', '0', '--output-rate', '2',
        '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout+result.stderr
    text = (tmp_path/'output.cfg').read_text()
    elapsed = [float(value) for value in re.findall(r'^Feature fft_elapsed_fs ([^\n]+)$', text, re.MULTILINE)]
    indices = np.arange(0, 3*(count-1)+1, 2)
    assert text.count('BEGIN_CFG') == len(indices)
    np.testing.assert_allclose(elapsed, indices*dt/3, rtol=1e-14, atol=1e-14)
