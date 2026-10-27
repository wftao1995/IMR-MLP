"""Independent harmonic-trajectory oracle for force reconstruction units."""
import math

import numpy as np
import pytest

from test_cli_error_paths import run_command


@pytest.mark.parametrize('velocity', [False, True])
def test_smooth_traj_reconstructs_mass_weighted_cartesian_force(tmp_path, velocity):
    import imrmlp
    count, dt, mass = 16, 2., 4.
    omega = 2 * math.pi / (2 * count * dt)
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'settings.ini').write_text(
        'rebuild_force true\n' + ('rebuild_velo true\n' if velocity else ''))
    (tmp_path / 'input.xyz').write_text(''.join(
        '1\npbc="F F F" Lattice="0 0 0 0 0 0 0 0 0" '
        'Properties=species:S:1:pos:R:3:masses:R:1\n'
        f'H {1 + .3 * math.cos(omega * dt * (j + .5)):.17g} 0 0 {mass}\n'
        for j in range(count)))
    result = run_command('imrmlp', [
        'smooth_traj', '--input', 'input.xyz', '--output', 'output.xyz',
        '--input-format', 'extxyz', '--output-format', 'extxyz',
        '--timestep', str(dt), '--freq-cutoff', '1000000', '--wtype', '0',
        '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'output.xyz')
    phase = omega * dt * (np.arange(count) + .5)
    force = -103.64269652680505 * mass * .3 * omega**2 * np.cos(phase)
    np.testing.assert_allclose([a.forces[0, 0] for a in frames], force, atol=1e-9)
    if velocity:
        expected = -1000 * .3 * omega * np.sin(phase)
        np.testing.assert_allclose([a.velocities[0, 0] for a in frames], expected, atol=1e-8)


def test_rebuilding_force_requires_explicit_mass_before_output(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'settings.ini').write_text('rebuild_force true\n')
    (tmp_path / 'input.xyz').write_text(
        '1\npbc="F F F" Properties=species:S:1:pos:R:3\nH 1 0 0\n' * 2)
    (tmp_path / 'output.xyz').write_text('previous result\n')
    result = run_command('imrmlp', [
        'smooth_traj', '--input', 'input.xyz', '--output', 'output.xyz',
        '--input-format', 'extxyz', '--output-format', 'extxyz', '--overwrite'], tmp_path)
    assert result.returncode != 0
    assert 'masses' in result.stderr
    assert (tmp_path / 'output.xyz').read_text() == 'previous result\n'
