"""MSD retains open normal motion while unwrapping ordinary 2D EXTXYZ slabs."""
import numpy as np
import pytest
from test_cli_error_paths import run_command


def slab_text(padded, nonprefix):
    a, b = np.array([4., 0., 0.]), np.array([1., 3., 0.])
    starts = [np.array([4.4, 2.4, 1.]), np.array([1.1, .9, -2.])]
    velocities = [np.array([.6, .4, .7]), np.array([-.3, .2, -.5])]
    cell = np.array([a, b, [0., 0., 20. if padded else 0.]])
    pbc = 'T T F'
    if nonprefix:
        cell[[1, 2]] = cell[[2, 1]]
        pbc = 'T F T'
    result = ''
    for frame in range(8):
        positions = []
        for start, velocity in zip(starts, velocities):
            position = start + frame * velocity
            fb = position[1] / 3.
            fa = (position[0] - fb) / 4.
            positions.append(position - np.floor(fa) * a - np.floor(fb) * b)
        lattice = ' '.join(map(str, cell.ravel()))
        result += f'2\nLattice="{lattice}" pbc="{pbc}" Properties=species:S:1:pos:R:3\n'
        result += ''.join('H ' + ' '.join(map(str, position)) + '\n' for position in positions)
    return result


@pytest.mark.parametrize('fft', [False, True])
@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('nonprefix', [False, True])
def test_calc_prop_msd_slabs_follow_lag_squared_components(tmp_path, fft, padded, nonprefix):
    (tmp_path / 'index.map').write_text('H 0\n')
    text = slab_text(padded, nonprefix)
    (tmp_path / 'input.xyz').write_text(text)
    (tmp_path / 'settings.ini').write_text(
        f'msd:fft {str(fft).lower()}\nmsd:timestep 1\nmsd:total true\nmsd:projection true\n'
        'msd:drift_cm false\nmsd:der false\nmsd:do_lfit false\n')
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--input-format',
                                   'extxyz', '--msd', '--settings', 'settings.ini',
                                   '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = np.loadtxt(tmp_path / 'MSD_TOTAL.dat')
    t = np.arange(8, dtype=float)
    wanted = np.column_stack((t, .695 * t**2, .225 * t**2, .1 * t**2, .37 * t**2))
    assert np.isfinite(rows).all()
    np.testing.assert_allclose(rows, wanted, atol=1e-10, rtol=0)
    assert (tmp_path / 'input.xyz').read_text() == text
