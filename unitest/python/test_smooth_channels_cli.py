"""Readable trajectories with holes in enabled FFT channels fail cleanly."""
import shlex
import numpy as np
import pytest
from test_cli_error_paths import run_command


def trajectory(hole=None):
    frames = []
    for f in range(4):
        present = lambda name: not (f == 2 and hole == name)
        header = ['Lattice="20 0 0 0 20 0 0 0 20"', 'pbc="F F F"']
        for name, value in [('energy', 6 + f), ('fenergy', 5.8 + f), ('temperature', 300 + 2 * f)]:
            if present(name): header.append(f'{name}={value}')
        if present('stress'): header.append('virial="' + ' '.join(map(str, (np.eye(3) * (1 + f)).ravel())) + '"')
        props = ['species:S:1', 'pos:R:3']
        row = ['H', str(1 + .1 * f), '2', '3']
        for name, values, width in [('forces', [.1 + f, .2, .3], 3), ('velocities', [1 + f, 2, 3], 3),
                                    ('site_en', [6 + f], 1), ('local', [3 + f], 1), ('vector', [5 + f, 6 + f], 2)]:
            if present(name):
                props.append(f'{name}:R:{width}')
                row.extend(map(str, values))
        header.append('Properties=' + ':'.join(props))
        frames.append('1\n' + ' '.join(header) + '\n' + ' '.join(row) + '\n')
    return ''.join(frames)


def run_smooth(tmp_path, text, cutoff, rebuild=False):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.xyz').write_text(text)
    (tmp_path / 'settings.ini').write_text('rebuild_velo ' + str(rebuild).lower() + '\n')
    return run_command('imrmlp', ['smooth_traj', '--input', 'input.xyz', '--input-format', 'extxyz',
                                  '--output', 'output.xyz', '--output-format', 'extxyz',
                                  '--timestep', '1', '--freq-cutoff', str(cutoff), '--wtype', '0'], tmp_path)


@pytest.mark.parametrize('hole', ['forces', 'velocities', 'site_en', 'energy', 'fenergy', 'stress', 'temperature', 'local', 'vector'])
def test_smooth_traj_missing_enabled_channels_reports_error_instead_of_crashing(tmp_path, hole):
    text = trajectory(hole)
    result = run_smooth(tmp_path, text, 1e9)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'incomplete' in result.stderr
    assert not (tmp_path / 'output.xyz').exists()
    assert (tmp_path / 'input.xyz').read_text() == text


def read_rows(path):
    lines = path.read_text().splitlines()
    result = []
    for i in range(0, len(lines), 3):
        fields = dict(item.split('=', 1) for item in shlex.split(lines[i + 1]))
        props = fields['Properties'].split(':')
        row, offset = lines[i + 2].split(), 0
        values = {}
        for j in range(0, len(props), 3):
            name, kind, width = props[j:j + 3]
            width = int(width)
            if kind == 'R': values[name] = np.array(list(map(float, row[offset:offset + width])))
            offset += width
        result.append((fields, values))
    return result


@pytest.mark.parametrize('cutoff', [0., 1e9])
def test_smooth_traj_complete_channels_have_independent_mean_and_signal_oracles(tmp_path, cutoff):
    text = trajectory()
    result = run_smooth(tmp_path, text, cutoff)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = read_rows(tmp_path / 'output.xyz')
    assert len(rows) == 4
    for i, (fields, values) in enumerate(rows):
        f = 1.5 if cutoff == 0 else i
        assert float(fields['energy']) == pytest.approx(6 + f, abs=1e-11)
        assert float(fields['fenergy']) == pytest.approx(5.8 + f, abs=1e-11)
        assert float(fields['temperature']) == pytest.approx(300 + 2 * f, abs=1e-11)
        np.testing.assert_allclose(list(map(float, fields['virial'].split())), (np.eye(3) * (1 + f)).ravel(), atol=1e-11, rtol=0)
        for name, expected in [('pos', [1 + .1 * f, 2, 3]), ('forces', [.1 + f, .2, .3]),
                               ('velocities', [1 + f, 2, 3]), ('site_en', [6 + f]),
                               ('local', [3 + f]), ('vector', [5 + f, 6 + f])]:
            np.testing.assert_allclose(values[name], expected, atol=1e-11, rtol=0)
    assert (tmp_path / 'input.xyz').read_text() == text


def test_smooth_traj_rebuild_velocity_accepts_a_source_velocity_hole(tmp_path):
    result = run_smooth(tmp_path, trajectory('velocities'), 1e9, rebuild=True)
    assert result.returncode == 0, result.stdout + result.stderr
    for fields, values in read_rows(tmp_path / 'output.xyz'):
        assert values['velocities'].shape == (3,)
        assert np.isfinite(values['velocities']).all()
