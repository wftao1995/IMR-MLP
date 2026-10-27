"""Bounded CP2KXYZ writing preserves the existing geometry-only codec."""
from pathlib import Path
import gzip

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp


def frames_for_cp2kxyz():
    elements = ['He', 'H']
    first = imrmlp.Atoms(numbers=[1, 2], positions=[[1, 2.5, 4.5], [4, 5, 6]],
                       cell=np.diag([8., 10., 12.]), pbc=True, elements=elements)
    first.energy = -12.5
    first.forces = [[.1, .2, .3], [-.1, -.2, -.3]]
    first.stress = np.eye(3)
    empty = imrmlp.Atoms(numbers=np.array([], dtype=int), positions=np.empty((0, 3)),
                       elements=elements)
    last = imrmlp.Atoms(numbers=[2], positions=[[1.25, -2.5, .375]],
                      cell=np.eye(3) * 20, elements=elements)
    return [first, empty, last]


def check_cp2kxyz_text(text, *, expected_counts=(2, 0, 1)):
    lines = iter(text.splitlines())
    parsed = []
    for count in expected_counts:
        assert int(next(lines)) == count
        # Native write_one comments are placeholders and never physical time.
        assert next(lines).strip() == 'i = 0, time = 0'
        parsed.append([next(lines).split() for _ in range(count)])
    assert list(lines) == []
    assert [row[0] for row in parsed[0]] == ['H', 'He']
    np.testing.assert_allclose([[float(x) for x in row[1:]] for row in parsed[0]],
                               [[1, 2.5, 4.5], [4, 5, 6]], rtol=1e-6, atol=1e-6)
    if len(parsed) == 3:
        assert parsed[1] == []
        assert parsed[2][0][0] == 'He'
        np.testing.assert_allclose([float(x) for x in parsed[2][0][1:]],
                                   [1.25, -2.5, .375], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('compressed,format', [(False, ''), (True, 'cp2kxyz')])
def test_cp2kxyz_write_iter_is_single_pass_and_uses_cartesian_geometry(tmp_path, compressed, format):
    frames = frames_for_cp2kxyz()
    first = frames[0]
    first.to_fractional()
    saved = first.copy()
    output = tmp_path / ('trajectory.cp2kxyz.gz' if compressed else 'trajectory.cp2kxyz')
    consumed = []

    def single_pass():
        for index, frame in enumerate(frames):
            if index:
                assert output.exists()  # The iterable is not first collected.
            consumed.append(index)
            yield frame

    assert imrmlp.write_iter(output, single_pass(), format=format) is None
    assert consumed == [0, 1, 2]
    text = gzip.decompress(output.read_bytes()).decode() if compressed else output.read_text()
    check_cp2kxyz_text(text)
    assert not first._impl.cartesian
    np.testing.assert_array_equal(first.positions, saved.positions)
    np.testing.assert_array_equal(first.cell, saved.cell)
    np.testing.assert_array_equal(first.pbc, saved.pbc)
    np.testing.assert_array_equal(first.forces, saved.forces)
    np.testing.assert_array_equal(first.stress, saved.stress)
    assert first.energy == saved.energy
    assert first._elements == saved._elements
    restored = list(imrmlp.iread(output, format='cp2kxyz'))
    assert [a.natoms for a in restored] == [2, 0, 1]
    np.testing.assert_array_equal(restored[0].numbers, [1, 2])
    np.testing.assert_allclose(restored[0].positions, [[1, 2.5, 4.5], [4, 5, 6]])
    for atom in restored:
        assert not np.any(atom.pbc)
        np.testing.assert_array_equal(atom.cell, np.zeros((3, 3)))
        assert atom.energy is None and atom.forces is None and atom.stress is None


@pytest.mark.parametrize('existing', [False, True])
def test_cp2kxyz_write_iter_empty_input_does_not_touch_target(tmp_path, existing):
    output = tmp_path / 'empty.cp2kxyz'
    if existing:
        output.write_text('sentinel')
    imrmlp.write_iter(output, iter(()))
    assert output.read_text() == 'sentinel' if existing else not output.exists()


@pytest.mark.parametrize('compressed', [False, True])
def test_cp2kxyz_write_iter_preserves_generator_error_and_complete_prefix(tmp_path, compressed):
    output = tmp_path / ('prefix.cp2kxyz.gz' if compressed else 'prefix.cp2kxyz')
    original = RuntimeError('generator stopped here')

    def interrupted():
        yield frames_for_cp2kxyz()[0]
        raise original

    with pytest.raises(RuntimeError, match='generator stopped here') as caught:
        imrmlp.write_iter(output, interrupted())
    assert caught.value is original
    text = gzip.decompress(output.read_bytes()).decode() if compressed else output.read_text()
    check_cp2kxyz_text(text, expected_counts=(2,))
    assert len(imrmlp.read(output, format='cp2kxyz')) == 1


def test_cp2kxyz_write_iter_rejects_hvp_labels(tmp_path):
    atom = imrmlp.Atoms(numbers=[1], positions=[[0, 0, 0]])
    directions = np.array([[1.], [0.], [0.]])
    atom.set_hvp(directions, 2 * directions)
    with pytest.raises(ValueError, match='HVP labels require'):
        imrmlp.write_iter(tmp_path / 'hvp.cp2kxyz', iter([atom]))
    np.testing.assert_array_equal(atom.hvp_targets, 2 * directions)


@pytest.mark.parametrize('compressed', [False, True])
def test_cp2kxyz_write_iter_checks_output_finalization(tmp_path, compressed):
    if not Path('/dev/full').exists():
        pytest.skip('/dev/full is unavailable')
    output = tmp_path / ('full.cp2kxyz.gz' if compressed else 'full.cp2kxyz')
    output.symlink_to('/dev/full')
    with pytest.raises(RuntimeError, match='write|clos'):
        imrmlp.write_iter(output, iter([frames_for_cp2kxyz()[0]]))
