"""FFT smoothing accepts ordinary CP2KXYZ molecular trajectories with no cell."""
import math
import re
import pytest

from test_cli_error_paths import run_command


@pytest.mark.parametrize('rebuild', [False, True])
def test_smooth_cp2kxyz_cellfree_molecule_and_velocity_units(tmp_path, rebuild):
    count, dt, amplitude = 16, 2., .2
    omega = 2 * math.pi / (2 * count * dt)
    phases = [omega * dt * (i + .5) for i in range(count)]
    (tmp_path / 'index.map').write_text('H 0\n')
    if rebuild:
        (tmp_path / 'settings.ini').write_text('rebuild_velo true\n')
    (tmp_path / 'input.xyz').write_text(''.join(
        f'1\n i = {i}, time = {i * dt}, E = 0\n'
        f'H {1 + amplitude * math.cos(phase):.17g} 2 3\n'
        for i, phase in enumerate(phases)))
    result = run_command('imrmlp', ['smooth_traj', '--input', 'input.xyz',
                         '--input-format', 'cp2kxyz', '--output', 'smooth.xyz',
                         '--output-format', 'extxyz', '--timestep', str(dt),
                         '--freq-cutoff', '1000000', '--wtype', '0'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / 'smooth.xyz').read_text().splitlines()
    assert len(lines) == 3 * count
    for i, phase in enumerate(phases):
        header, row = lines[3 * i + 1], lines[3 * i + 2].split()
        assert lines[3 * i] == '1'
        assert row[0] == 'H'
        assert 'pbc="F F F"' in header
        assert list(map(float, row[1:4])) == pytest.approx(
            [1 + amplitude * math.cos(phase), 2., 3.], abs=1e-11)
        properties = re.search(r'Properties=(\S+)', header).group(1)
        if rebuild:
            assert properties == 'species:S:1:pos:R:3:velocities:R:3'
            assert list(map(float, row[4:7])) == pytest.approx(
                [-1000 * amplitude * omega * math.sin(phase), 0., 0.], abs=1e-9)
        else:
            assert properties == 'species:S:1:pos:R:3'
