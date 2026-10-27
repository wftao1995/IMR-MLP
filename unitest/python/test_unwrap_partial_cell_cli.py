"""Current trj_smooth unwraps ordinary slabs/wires without padding cell rows."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from test_cli_error_paths import run_command


def trajectory(kind, changing):
    raw, expected = [], []
    normal = np.array([-3., 1., 12.]) if kind == 'slab' else np.array([-1., 4., 7.])
    for i in range(3):
        cell = np.zeros((3, 3))
        if kind == 'slab':
            cell[0] = [4., 0., 1.]
            cell[2] = [1., 3., 0.]
            pbc = 'T F T'
            vector = cell[0] + cell[2]
        else:
            scale = i if changing else 0
            cell[2] = [4. + scale, 1. + .25 * scale, 0.]
            pbc = 'F F T'
            vector = cell[2]
        wrapped = .45 if i == 0 else -.55 + .1 * i
        positions = np.array([wrapped * vector + i * normal, -wrapped * vector - 2 * i * normal])
        unwrapped = .45 + .1 * i
        expected.append(np.array([unwrapped * vector + i * normal, -unwrapped * vector - 2 * i * normal]))
        lattice = ' '.join(map(str, cell.ravel()))
        rows = ''.join('H ' + ' '.join(map(str, pos)) + '\n' for pos in positions)
        raw.append(f'2\nLattice="{lattice}" pbc="{pbc}" Properties=species:S:1:pos:R:3\n' + rows)
    return ''.join(raw), np.array(expected)


@pytest.mark.parametrize('kind,changing', [('slab', False), ('wire', False), ('wire', True)])
@pytest.mark.parametrize('window', [1, 2])
def test_trj_smooth_partial_cell_has_independent_unwrapped_position_oracle(tmp_path, kind, changing, window):
    text, unwrapped = trajectory(kind, changing)
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.xyz').write_text(text)
    result = run_command('trj_smooth', ['--input', 'input.xyz', '--output', 'output.xyz',
                                    '--input-format', 'extxyz', '--output-format', 'extxyz',
                                    '--window', str(window)], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'output.xyz')
    assert len(frames) == 3
    for i, atom in enumerate(frames):
        wanted = unwrapped[i:min(3, i + window)].mean(axis=0)
        assert np.isfinite(atom.positions).all()
        np.testing.assert_allclose(atom.positions, wanted, atol=1e-12, rtol=0)


def test_trj_smooth_rejects_dependent_periodic_basis(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    frame = ('1\nLattice="4 0 1 0 0 0 8 0 2" pbc="T F T" '
             'Properties=species:S:1:pos:R:3\nH 1 2 3\n')
    (tmp_path / 'input.xyz').write_text(frame * 2)
    result = run_command('trj_smooth', ['--input', 'input.xyz', '--output', 'output.xyz',
                                    '--input-format', 'extxyz', '--output-format', 'extxyz',
                                    '--window', '1'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'independent periodic cell vectors' in result.stderr
    assert not (tmp_path / 'output.xyz').exists()
