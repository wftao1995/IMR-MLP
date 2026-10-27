"""Native CP2KXYZ streaming output reaches the existing ordinary geometry writer."""
import gzip
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from test_cli_error_paths import run_command

SOURCE = ('2\nProperties=species:S:1:pos:R:3\nH 1.25 -2.5 .375\nHe 4 5 6\n'
          '0\nProperties=species:S:1:pos:R:3\n'
          '1\nProperties=species:S:1:pos:R:3\nHe 1 -3 2\n')
EXPECTED_NUMBERS = [[1, 2], [], [2]]
EXPECTED_POSITIONS = [np.array([[1.25, -2.5, .375], [4., 5., 6.]]), np.empty((0, 3)), np.array([[1., -3., 2.]])]


def check_geometry(path):
    frames = list(imrmlp.iread(path, format='cp2kxyz'))
    assert [len(atom) for atom in frames] == [2, 0, 1]
    for i, atom in enumerate(frames):
        np.testing.assert_array_equal(atom.numbers, EXPECTED_NUMBERS[i])
        np.testing.assert_allclose(atom.positions, EXPECTED_POSITIONS[i], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(atom.pbc, [False, False, False])
        assert atom.energy is None
    return frames


@pytest.mark.parametrize('entry', ['convstr', 'imrmlp'])
@pytest.mark.parametrize('mode', [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize('compressed', [False, True])
def test_cp2kxyz_all_cli_modes_match_public_write_iter_geometry(tmp_path, entry, mode, compressed):
    (tmp_path / 'input.xyz').write_text(SOURCE)
    (tmp_path / 'index.map').write_text('H 0\nHe 1\n')
    output = tmp_path / ('output.cp2kxyz.gz' if compressed else 'output.cp2kxyz')
    args = ['--inputfiles', 'input.xyz', '--input-format', 'extxyz', '--outputfile', output,
            '--output-format', 'cp2kxyz', '--loadrate', str(mode[0]), '--outrate', str(mode[1]), '--index', 'index.map']
    if entry == 'imrmlp': args.insert(0, 'convstr')
    result = run_command(entry, args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    check_geometry(output)
    imrmlp.write_iter(tmp_path / 'python.cp2kxyz', iter(imrmlp.read(tmp_path / 'input.xyz')), format='cp2kxyz')
    check_geometry(tmp_path / 'python.cp2kxyz')
    text = gzip.decompress(output.read_bytes()).decode() if compressed else output.read_text()
    # Comments remain codec placeholders, with no new physical time protocol.
    assert text.count('time =') == 3
    assert (tmp_path / 'input.xyz').read_text() == SOURCE


@pytest.mark.parametrize('entry', ['convstr', 'imrmlp'])
def test_cp2kxyz_streaming_conversion_honors_existing_input_end_without_reading_bad_tail(tmp_path, entry):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.cp2kxyz').write_text('1\nfirst\nH 1 2 3\n1\nsecond\nH 2 3 4\n1\nbroken\n')
    args = ['--inputfiles', 'input.cp2kxyz', '--input-format', 'cp2kxyz',
            '--outputfile', 'prefix.cp2kxyz', '--output-format', 'cp2kxyz',
            '--loadrate', '1', '--outrate', '1', '--end', '1', '--index', 'index.map']
    if entry == 'imrmlp': args.insert(0, 'convstr')
    result = run_command(entry, args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'prefix.cp2kxyz', format='cp2kxyz')
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0].positions, [[1., 2., 3.]])
    np.testing.assert_array_equal(frames[1].positions, [[2., 3., 4.]])
