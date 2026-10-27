"""Single-pass XDATCAR writers retain fixed or per-frame geometry headers."""
from contextlib import closing
import gzip

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp


def variable_frames():
    cells = [np.array([[4., 0., 0.], [.5, 6., 0.], [0., .2, 8.]]),
             np.array([[5., .2, 0.], [0., 7., .1], [.3, 0., 9.]])]
    fractions = [np.array([[.1, .2, .3], [.4, .5, .6], [.7, .2, .1]]),
                 np.array([[.2, .3, .4], [.5, .6, .7]])]
    numbers = [[1, 2, 1], [1, 2]]
    frames = [imrmlp.Atoms(numbers=species, elements=['He', 'H'],
                         positions=frac @ cell, cell=cell)
              for species, frac, cell in zip(numbers, fractions, cells)]
    frames[0].to_fractional()
    return frames, cells, fractions, [[1, 0, 2], [1, 0]]


@pytest.mark.parametrize('format,filename', [
    ('', 'trajectory.xdatcar6'),
    ('XDATCAR6', 'trajectory.data'),
    ('', 'trajectory.xdatcar6.gz'),
])
def test_write_iter_xdatcar6_has_independent_headers_and_geometry(tmp_path, format, filename):
    frames, cells, fractions, orders = variable_frames()
    snapshots = [frame.copy() for frame in frames]
    output = tmp_path / filename
    consumed = []

    def single_pass():
        for index, frame in enumerate(frames):
            if index:
                assert output.exists()
            consumed.append(index)
            yield frame

    assert imrmlp.write_iter(output, single_pass(), format=format) is None
    assert consumed == [0, 1]
    text = (gzip.decompress(output.read_bytes()).decode() if filename.endswith('.gz')
            else output.read_text())
    lines = text.splitlines()
    offset = 0
    # Inspect the text independently of native reading: headers, counts, and
    # fractional coordinates must describe each frame rather than the first.
    for index, (cell, frac, order) in enumerate(zip(cells, fractions, orders)):
        assert lines[offset] == 'XDATCAR by GPIO'
        assert float(lines[offset + 1]) == 1.
        np.testing.assert_allclose(
            [list(map(float, row.split())) for row in lines[offset + 2:offset + 5]],
            cell, atol=1e-10, rtol=0)
        assert lines[offset + 5].split() == ['He', 'H']
        assert list(map(int, lines[offset + 6].split())) == [1, 2 - index]
        assert lines[offset + 7].startswith('Direct configuration=')
        np.testing.assert_allclose(
            [list(map(float, row.split()))
             for row in lines[offset + 8:offset + 8 + len(order)]],
            frac[order], atol=1e-10, rtol=0)
        offset += 8 + len(order)
    assert offset == len(lines)

    with closing(imrmlp.iread(output, format='xdatcar6')) as stream:
        actual = list(stream)
    assert len(actual) == 2
    for frame, source, saved, cell, frac, order in zip(
            actual, frames, snapshots, cells, fractions, orders):
        np.testing.assert_allclose(frame.cell, cell, atol=1e-10, rtol=0)
        np.testing.assert_allclose(frame.positions, frac[order] @ cell, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(frame.numbers, saved.numbers[order])
        np.testing.assert_array_equal(source.positions, saved.positions)
        np.testing.assert_array_equal(source.cell, saved.cell)
        np.testing.assert_array_equal(source.numbers, saved.numbers)
        assert source.chemmap == saved.chemmap
        assert source._impl.cartesian == saved._impl.cartesian


@pytest.mark.parametrize('format', ['xdatcar', 'xdatcar6'])
@pytest.mark.parametrize('existing', [False, True])
def test_empty_xdatcar_iterator_leaves_output_untouched(tmp_path, existing, format):
    output = tmp_path / ('empty.' + format)
    if existing:
        output.write_bytes(b'keep existing data')
    imrmlp.write_iter(output, iter(()))
    assert output.exists() == existing
    if existing:
        assert output.read_bytes() == b'keep existing data'


@pytest.mark.parametrize('format', ['xdatcar', 'xdatcar6'])
def test_xdatcar_iterator_error_keeps_a_readable_complete_prefix(tmp_path, format):
    frames, cells, fractions, orders = variable_frames()
    output = tmp_path / ('prefix.' + format)
    error = ValueError('source generation failed')

    def interrupted():
        yield frames[0]
        raise error

    with pytest.raises(ValueError, match='source generation failed') as caught:
        imrmlp.write_iter(output, interrupted())
    assert caught.value is error
    with closing(imrmlp.iread(output)) as stream:
        actual = list(stream)
    assert len(actual) == 1
    np.testing.assert_allclose(actual[0].positions,
                               fractions[0][orders[0]] @ cells[0], atol=1e-9, rtol=0)


@pytest.mark.parametrize('format', ['xdatcar', 'xdatcar6'])
def test_xdatcar_write_iter_preserves_the_unsupported_hvp_contract(tmp_path, format):
    frame = variable_frames()[0][0]
    directions = np.ones((3 * len(frame), 1))
    frame.set_hvp(directions, 2 * directions)
    with pytest.raises(ValueError, match='HVP labels require'):
        imrmlp.write_iter(tmp_path / ('hvp.' + format), iter([frame]))
    np.testing.assert_array_equal(frame.hvp_directions, directions)
    np.testing.assert_array_equal(frame.hvp_targets, 2 * directions)


def fixed_frames():
    frames, cells, fractions, orders = variable_frames()
    cell = cells[0]
    second_frac = np.array([[.2, .3, .4], [.6, .5, .2], [.5, .6, .7]])
    second = imrmlp.Atoms(numbers=[1, 1, 2], elements=['He', 'H'],
                        positions=second_frac @ cell, cell=cell)
    return [frames[0], second], cell, [fractions[0], second_frac], [orders[0], [2, 0, 1]]


@pytest.mark.parametrize('format,filename', [
    ('', 'XDATCAR'), ('XDATCAR', 'fixed.data'), ('', 'fixed.xdatcar.gz'),
])
def test_fixed_xdatcar_write_iter_writes_one_header_and_remaps_each_frame(
        tmp_path, format, filename):
    frames, cell, fractions, orders = fixed_frames()
    snapshots = [frame.copy() for frame in frames]
    output = tmp_path / filename
    consumed = []

    def single_pass():
        for index, frame in enumerate(frames):
            if index:
                assert output.exists()
            consumed.append(index)
            yield frame

    assert imrmlp.write_iter(output, single_pass(), format=format) is None
    assert consumed == [0, 1]
    text = (gzip.decompress(output.read_bytes()).decode() if filename.endswith('.gz')
            else output.read_text())
    lines = text.splitlines()
    assert text.count('XDATCAR by GPIO') == 1
    assert text.count('Direct configuration=') == 2
    assert len(lines) == 15
    assert float(lines[1]) == 1.
    np.testing.assert_allclose([list(map(float, row.split())) for row in lines[2:5]],
                               cell, atol=1e-10, rtol=0)
    assert lines[5].split() == ['He', 'H']
    assert list(map(int, lines[6].split())) == [1, 2]
    for offset, frac, order in zip([7, 11], fractions, orders):
        assert lines[offset].startswith('Direct configuration=')
        np.testing.assert_allclose(
            [list(map(float, row.split())) for row in lines[offset + 1:offset + 4]],
            frac[order], atol=1e-10, rtol=0)
    with closing(imrmlp.iread(output, format='xdatcar')) as stream:
        actual = list(stream)
    assert len(actual) == 2
    for frame, source, saved, frac, order in zip(actual, frames, snapshots, fractions, orders):
        np.testing.assert_allclose(frame.cell, cell, atol=1e-10, rtol=0)
        np.testing.assert_allclose(frame.positions, frac[order] @ cell, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(frame.numbers, saved.numbers[order])
        np.testing.assert_array_equal(source.positions, saved.positions)
        np.testing.assert_array_equal(source.cell, saved.cell)
        np.testing.assert_array_equal(source.numbers, saved.numbers)
        assert source.chemmap == saved.chemmap
        assert source._impl.cartesian == saved._impl.cartesian


@pytest.mark.parametrize('change', ['cell', 'composition', 'shorter', 'longer'])
def test_fixed_xdatcar_write_iter_rejects_changes_before_starting_a_bad_frame(tmp_path, change):
    frames, cell, fractions, orders = fixed_frames()
    changed = frames[1].copy()
    if change == 'cell':
        changed.cell = cell * 1.2
    elif change == 'composition':
        changed.numbers = [2, 2, 1]
    else:
        count = 2 if change == 'shorter' else 4
        changed = imrmlp.Atoms(numbers=[1] * (count - 1) + [2], elements=['He', 'H'],
                             positions=np.arange(3 * count).reshape(count, 3) / 10,
                             cell=cell)
    output = tmp_path / 'fixed.xdatcar'
    with pytest.raises(RuntimeError, match='Failed to write streaming output'):
        imrmlp.write_iter(output, iter([frames[0], changed]))
    text = output.read_text()
    assert text.count('XDATCAR by GPIO') == 1
    assert text.count('Direct configuration=') == 1
    with closing(imrmlp.iread(output)) as stream:
        actual = list(stream)
    assert len(actual) == 1
    np.testing.assert_allclose(actual[0].positions,
                               fractions[0][orders[0]] @ cell, atol=1e-9, rtol=0)
