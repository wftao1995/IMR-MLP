"""Handwritten animated XSF geometry, forces and one-pass public trajectories."""
from contextlib import closing
import gzip
import random

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from test_cli_error_paths import run_command

VARIABLE = ('ANIMSTEPS 2\nSLAB\nPRIMVEC 1\n4 0 0\n.5 5 0\n0 0 8\n'
            'CONVVEC 1\n8 0 0\n0 10 0\n0 0 16\nPRIMCOORD 1\n2 1\n'
            'H .25 .5 .75 .1 -.2 .3\n8 1 2 3 -.4 .5 -.6\n'
            'PRIMVEC 2\n6 0 .2\n.4 7 0\n0 0 9\nPRIMCOORD 2\n2 1\n'
            '1 .5 .75 1.25 0 0 0\nO 2 3 4 .2 .3 .4\n')
PREFIX = ('ANIMSTEPS 2\nCRYSTAL\nPRIMVEC\n4 0 0\n0 5 0\n0 0 6\n'
          'PRIMCOORD 1\n1 1\nH .25 .5 .75 .1 -.2 .3\n')
SECOND = 'PRIMCOORD 2\n1 1\nH 1 2 3\n'
MOLECULE = ('ANIMSTEPS 2\nATOMS 1\nH 0 0 0 .1 0 0\nO 1 0 0 0 .2 0\n'
            'ATOMS 2\n1 0 0 .5\n8 1 0 .5\n')


def check_variable(frames):
    assert len(frames) == 2
    first, second = frames
    np.testing.assert_array_equal(first.numbers, [1, 8])
    np.testing.assert_array_equal(second.pbc, [True, True, False])
    np.testing.assert_allclose(first.cell, [[4, 0, 0], [.5, 5, 0], [0, 0, 8]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(second.cell, [[6, 0, .2], [.4, 7, 0], [0, 0, 9]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(second.positions, [[.5, .75, 1.25], [2, 3, 4]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(first.forces[0], [2.7211396641308, -5.4422793282616, 8.1634189923924], atol=1e-13)
    np.testing.assert_allclose(second.forces[1], [5.4422793282616, 8.1634189923924, 10.8845586565232], atol=1e-13)


@pytest.mark.parametrize('format', ['', 'xsf', 'axsf'])
def test_xsf_animation_bulk_and_read_one_use_declared_primitive_frames(tmp_path, format):
    path = tmp_path / 'variable.axsf'; path.write_text(VARIABLE)
    check_variable(imrmlp.read(path, format=format))
    first = imrmlp.read_one(path, format=format)
    np.testing.assert_allclose(first.positions[0], [.25, .5, .75], atol=1e-14)
    assert path.read_text() == VARIABLE


@pytest.mark.parametrize('suffix', ['.axsf', '.axsf.gz'])
def test_xsf_iread_compression_preserves_owned_frames_and_units(tmp_path, suffix):
    data = VARIABLE.encode()
    if suffix.endswith('.gz'): data = gzip.compress(data)
    if suffix.endswith('.zst'):
        import zstandard
        data = zstandard.ZstdCompressor().compress(data)
    path = tmp_path / ('variable' + suffix); path.write_bytes(data)
    with closing(imrmlp.iread(path)) as stream: frames = list(stream)
    check_variable(frames)
    frames[1].positions = [[9, 8, 7], [6, 5, 4]]
    np.testing.assert_allclose(frames[0].positions[0], [.25, .5, .75], atol=1e-14)


def test_xsf_fixed_and_molecular_frames_keep_pbc_and_independent_force_presence(tmp_path):
    path = tmp_path / 'fixed.xsf'; path.write_text(PREFIX + SECOND)
    first, second = list(imrmlp.iread(path))
    np.testing.assert_array_equal(first.cell, second.cell)
    np.testing.assert_array_equal(first.pbc, [True]*3)
    assert second.forces is None
    path.write_text(MOLECULE)
    first, second = list(imrmlp.iread(path))
    np.testing.assert_array_equal(first.pbc, [False]*3)
    np.testing.assert_array_equal(second.cell, np.zeros((3, 3)))
    assert second.forces is None
    np.testing.assert_allclose(first.forces[1], [0, 5.4422793282616, 0], atol=1e-13)
    np.testing.assert_allclose(second.positions, [[0, 0, .5], [1, 0, .5]], atol=1e-14)
    path.write_text('ANIMSTEPS 2\nATOMS 1\nATOMS 2\n')
    assert [len(a) for a in imrmlp.iread(path)] == [0, 0]


def test_xsf_iread_opens_lazily_and_early_close_leaves_bad_later_frame_unread(tmp_path):
    path = tmp_path / 'late.axsf'
    frames = imrmlp.iread(path) # No file yet.
    path.write_text(PREFIX + 'PRIMCOORD 2\n1 1\nH 1 2\n')
    first = next(frames)
    frames.close()
    np.testing.assert_allclose(first.positions[0], [.25, .5, .75], atol=1e-14)
    with pytest.raises(StopIteration): next(frames)
    frames = imrmlp.iread(path)
    next(frames)
    with pytest.raises(RuntimeError, match='XSF invalid or truncated frame 1'): next(frames)
    with pytest.raises(StopIteration): next(frames)


def test_xsf_complete_iteration_checks_compressed_trailer_but_early_close_does_not(tmp_path):
    # Large incompressible comment ensures the bad trailer is beyond input buffers.
    comment = '#' + random.Random(321).randbytes(262144).hex() + '\n'
    data = bytearray(gzip.compress((PREFIX + comment + SECOND).encode()))
    data[-8] ^= 1
    path = tmp_path / 'trailer.axsf.gz'; path.write_bytes(data)
    frames = imrmlp.iread(path)
    first = next(frames); frames.close()
    assert len(first) == 1
    with pytest.raises(RuntimeError): list(imrmlp.iread(path))


@pytest.mark.parametrize('text', [
    PREFIX, PREFIX + SECOND + 'PRIMCOORD 3\n1 1\nH 2 3 4\n',
    PREFIX + 'PRIMCOORD 3\n1 1\nH 1 2 3\n',
    'ANIMSTEPS 2\nATOMS 1\nH 0 0 0\nATOMS 2\nH 1 0 0\nO 2 0 0\n'])
def test_xsf_full_iteration_rejects_count_order_and_atom_count_mismatch(tmp_path, text):
    path = tmp_path / 'bad.axsf'; path.write_text(text)
    with pytest.raises(RuntimeError): list(imrmlp.iread(path))
    with pytest.raises(RuntimeError): imrmlp.read(path)


def test_xsf_cli_bounded_conversion_preserves_cells_forces_and_declared_count_errors(tmp_path):
    (tmp_path / 'input.axsf').write_text(VARIABLE)
    args = ['convstr', '--input-format=xsf', '--output-format=extxyz', '--loadrate=1', '--outrate=1',
            '--lossless=true', '--inputfiles', 'input.axsf', '--outputfile=output.xyz']
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    check_variable(imrmlp.read(tmp_path / 'output.xyz'))
    (tmp_path / 'input.axsf').write_text('ANIMSTEPS 2\nATOMS 1\nH 0 0 0\n')
    result = run_command('imrmlp', args + ['--overwrite'], tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    (tmp_path / 'input.axsf').write_text(PREFIX + 'BAD UNREAD TAIL\n')
    result = run_command('imrmlp', args + ['--overwrite', '--bstart=true', '--end=0'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(imrmlp.read(tmp_path / 'output.xyz')) == 1
