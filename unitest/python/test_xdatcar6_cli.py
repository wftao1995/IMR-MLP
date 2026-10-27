"""Exercise fixed and variable-cell XDATCAR conversion and error propagation."""
import gzip
import re

import pytest

from test_cli_error_paths import run_command


def frame(title, cell, positions, number):
    return (f'{title}\n1.0\n' +
            ''.join(' '.join(map(str, row)) + '\n' for row in cell) +
            'H He\n1 1\n' + f'Direct configuration= {number}\n' +
            ''.join(' '.join(map(str, row)) + '\n' for row in positions))


CELL1 = [[4, 0, 0], [0, 6, 0], [0, 0, 8]]
CELL2 = [[5, 1, 0], [0, 7, 1], [1, 0, 9]]
FRAME1 = frame('first cell', CELL1, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], 10)
FRAME2 = frame('second cell', CELL2, [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], 20)
FIXED_FRAME2 = 'Direct configuration= 20\n0.2 0.3 0.4\n0.5 0.6 0.7\n'


def convert(tmp_path, payload, loadrate, outrate, compressed=False, *,
            format='xdatcar6', entry='imrmlp', source_name=None):
    (tmp_path / 'index.map').write_text('H 0\nHe 1\n')
    source = tmp_path / (source_name or f'input.{format}' + ('.gz' if compressed else ''))
    if payload is not None:
        if compressed:
            source.write_bytes(gzip.compress(payload.encode()))
        else:
            source.write_text(payload)
    args = ['convstr'] if entry == 'imrmlp' else []
    if format:
        args += ['--input-format', format]
    return run_command(entry, args + [
        '--output-format', 'extxyz',
        '--index', 'index.map', '--loadrate', str(loadrate),
        '--outrate', str(outrate), source.name, 'output.xyz',
    ], tmp_path)


def read_xyz(path):
    lines = iter(path.read_text().splitlines())
    result = []
    for count in lines:
        header = next(lines)
        lattice = list(map(float, re.search(r'Lattice="([^"]+)"', header)[1].split()))
        rows = [next(lines).split() for _ in range(int(count))]
        result.append((lattice, [row[0] for row in rows],
                       [float(value) for row in rows for value in row[1:4]]))
    return result


@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.parametrize('loadrate', [0, 1])
@pytest.mark.parametrize('outrate', [0, 1])
def test_xdatcar6_conversion_keeps_first_frame_and_variable_cell(
        tmp_path, compressed, loadrate, outrate):
    result = convert(tmp_path, FRAME1 + FRAME2 + '\n', loadrate, outrate, compressed)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = read_xyz(tmp_path / 'output.xyz')
    assert len(frames) == 2
    for actual, cell, positions in zip(frames, [CELL1, CELL2], [
            [0.4, 1.2, 2.4, 1.6, 3.0, 4.8], [1.4, 2.3, 3.9, 3.2, 4.7, 6.9]]):
        assert actual[0] == pytest.approx([v for row in cell for v in row])
        assert actual[1] == ['H', 'He']
        assert actual[2] == pytest.approx(positions)


@pytest.mark.parametrize('loadrate', [0, 1])
@pytest.mark.parametrize('outrate', [0, 1])
@pytest.mark.parametrize('payload', [
    '',
    ' \n\t\n',
    'broken header\n1.0\n1 0\n',
    FRAME1 + FRAME2.rsplit('\n', 2)[0] + '\n',
    FRAME1 + FRAME2.replace('0.2 0.3 0.4', '0.2 invalid 0.4'),
    FRAME1 + FRAME2.replace('5 1 0', '5 invalid 0'),
])
def test_xdatcar6_conversion_reports_malformed_input(tmp_path, loadrate, outrate, payload):
    result = convert(tmp_path, payload, loadrate, outrate)
    assert result.returncode != 0, result.stdout + result.stderr
    message = result.stdout + result.stderr
    assert 'XDATCAR' in message or 'ERROR WHEN READING FILE' in message


@pytest.mark.parametrize('loadrate', [0, 1])
@pytest.mark.parametrize('outrate', [0, 1])
def test_fixed_xdatcar_conversion_preserves_both_frames(tmp_path, loadrate, outrate):
    result = convert(tmp_path, FRAME1 + FIXED_FRAME2.rstrip('\n'), loadrate, outrate,
                     format='xdatcar')
    assert result.returncode == 0, result.stdout + result.stderr
    frames = read_xyz(tmp_path / 'output.xyz')
    assert len(frames) == 2
    for actual, positions in zip(frames, [
            [0.4, 1.2, 2.4, 1.6, 3.0, 4.8],
            [0.8, 1.8, 3.2, 2.0, 3.6, 5.6]]):
        assert actual[0] == pytest.approx([v for row in CELL1 for v in row])
        assert actual[1] == ['H', 'He']
        assert actual[2] == pytest.approx(positions)


@pytest.mark.parametrize('loadrate', [0, 1])
@pytest.mark.parametrize('outrate', [0, 1])
def test_fixed_xdatcar_conversion_rejects_bad_tail(tmp_path, loadrate, outrate):
    payload = FRAME1 + FIXED_FRAME2.replace('0.5 0.6 0.7', '0.5 0.6')
    result = convert(tmp_path, payload, loadrate, outrate, format='xdatcar')
    assert result.returncode != 0, result.stdout + result.stderr
    message = result.stdout + result.stderr
    assert 'XDATCAR' in message or 'ERROR WHEN READING FILE' in message


@pytest.mark.parametrize('payload', ['broken header\n1.0\n1 0\n', None],
                         ids=['bad-header', 'missing-file'])
def test_fixed_xdatcar_conversion_reports_initial_failure(tmp_path, payload):
    result = convert(tmp_path, payload, 1, 1, format='xdatcar')
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'error:' in (result.stdout + result.stderr).lower()


def test_standalone_convstr_rejects_fixed_xdatcar_bad_tail(tmp_path):
    payload = FRAME1 + FIXED_FRAME2.replace('0.5 0.6 0.7', '0.5 0.6')
    result = convert(tmp_path, payload, 1, 1, format='xdatcar', entry='convstr')
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'XDATCAR' in result.stdout + result.stderr


@pytest.mark.parametrize('source_name', [
    'input.xdatcar', 'input.xdatcar6', 'xdatcar6', 'xdatcar6.gz',
])
def test_xdatcar_auto_format_reaches_sequential_reader(tmp_path, source_name):
    variable = 'xdatcar6' in source_name
    payload = FRAME1 + (FRAME2 if variable else FIXED_FRAME2)
    result = convert(tmp_path, payload, 1, 1, source_name.endswith('.gz'),
                     format='', source_name=source_name)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = read_xyz(tmp_path / 'output.xyz')
    assert len(frames) == 2
    cells = [CELL1, CELL2 if variable else CELL1]
    second_positions = ([1.4, 2.3, 3.9, 3.2, 4.7, 6.9] if variable else
                        [0.8, 1.8, 3.2, 2.0, 3.6, 5.6])
    for actual, cell, positions in zip(frames, cells, [
            [0.4, 1.2, 2.4, 1.6, 3.0, 4.8], second_positions]):
        assert actual[0] == pytest.approx([v for row in cell for v in row])
        assert actual[1] == ['H', 'He']
        assert actual[2] == pytest.approx(positions)


def test_xdatcar6_auto_format_reports_bad_initial_header(tmp_path):
    result = convert(tmp_path, 'broken header\n1.0\n1 0\n', 1, 1,
                     format='', source_name='input.xdatcar6')
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'XDATCAR6' in result.stdout + result.stderr


def xyz_frame(cell, species, positions):
    lattice = ' '.join(str(value) for row in cell for value in row)
    return (f'{len(species)}\nLattice="{lattice}" pbc="T T T" '
            'Properties=species:S:1:pos:R:3\n' + ''.join(
                symbol + ' ' + ' '.join(map(str, position)) + '\n'
                for symbol, position in zip(species, positions)))


def export_xdatcar(tmp_path, payload, loadrate, *, format='xdatcar', start=0):
    (tmp_path / 'index.map').write_text('H 0\nHe 1\n')
    (tmp_path / 'source.xyz').write_text(payload)
    return run_command('imrmlp', [
        'convstr', '--input-format', 'extxyz', '--output-format', format,
        '--index', 'index.map', '--loadrate', str(loadrate), '--outrate', '1',
        '--start', str(start), 'source.xyz', 'export.xdatcar',
    ], tmp_path)


@pytest.mark.parametrize('loadrate', [0, 1], ids=['bulk-input', 'sequential-input'])
def test_fixed_xdatcar_writer_keeps_species_positions_after_reordering(tmp_path, loadrate):
    # The skipped frame must not establish the fixed output header. Sequential
    # convstr also uses the same GPIO instance to read and write each frame.
    ignored = xyz_frame(CELL2, ['He'], [[1, 2, 3]])
    first = xyz_frame(CELL1, ['He', 'H', 'H'], [[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    second = xyz_frame(CELL1, ['H', 'He', 'H'], [[.5, 1, 1.5], [1.5, 2, 2.5], [2.5, 3, 3.5]])
    result = export_xdatcar(tmp_path, ignored + first + second, loadrate, start=1)
    assert result.returncode == 0, result.stdout + result.stderr
    result = convert(tmp_path, None, 1, 1, format='xdatcar', source_name='export.xdatcar')
    assert result.returncode == 0, result.stdout + result.stderr
    actual = read_xyz(tmp_path / 'output.xyz')
    assert len(actual) == 2
    for saved, expected in zip(actual, [
            [2, 3, 4, 3, 4, 5, 1, 2, 3],
            [.5, 1, 1.5, 2.5, 3, 3.5, 1.5, 2, 2.5]]):
        assert saved[0] == pytest.approx([v for row in CELL1 for v in row])
        assert saved[1] == ['H', 'H', 'He']
        assert saved[2] == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize('loadrate', [0, 1], ids=['bulk-input', 'sequential-input'])
@pytest.mark.parametrize('change', ['cell', 'composition', 'shorter', 'longer'])
def test_fixed_xdatcar_writer_rejects_incompatible_later_frames(tmp_path, loadrate, change):
    cell, species, positions = CELL1, ['H', 'He'], [[1, 2, 3], [2, 3, 4]]
    first = xyz_frame(cell, species, positions)
    if change == 'cell':
        cell = CELL2
    elif change == 'composition':
        species = ['H', 'H']
    elif change == 'shorter':
        species, positions = ['H'], positions[:1]
    else:
        species, positions = ['H', 'He', 'H'], positions + [[3, 4, 5]]
    result = export_xdatcar(tmp_path, first + xyz_frame(cell, species, positions), loadrate)
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'xdatcar6' in (result.stdout + result.stderr).lower()


def test_public_fixed_xdatcar_write_reorders_and_preserves_fractional_sources(tmp_path):
    pytest.importorskip('imrmlp._imrmlp')
    import imrmlp
    import numpy as np

    first = imrmlp.Atoms(numbers=[2, 1, 1], cell=CELL1,
                       positions=[[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    nearby_cell = np.array(CELL1, dtype=float)
    nearby_cell[0, 0] += 1e-12  # well below the ten-decimal cell output precision
    second = imrmlp.Atoms(numbers=[1, 2, 1], cell=nearby_cell,
                        positions=[[.5, 1, 1.5], [1.5, 2, 2.5], [2.5, 3, 3.5]])
    second.to_fractional()
    fractional = second.positions.copy()
    original_cell = second.cell.copy()
    path = tmp_path / 'public.xdatcar'
    imrmlp.write(path, [first, second], format='xdatcar')
    assert not second._impl.cartesian
    np.testing.assert_array_equal(second.positions, fractional)
    np.testing.assert_array_equal(second.cell, original_cell)
    np.testing.assert_array_equal(second.numbers, [1, 2, 1])
    saved = imrmlp.read(path, format='xdatcar')
    assert len(saved) == 2
    for actual, positions in zip(saved, [
            [[2, 3, 4], [3, 4, 5], [1, 2, 3]],
            [[.5, 1, 1.5], [2.5, 3, 3.5], [1.5, 2, 2.5]]]):
        np.testing.assert_array_equal(actual.numbers, [1, 1, 2])
        np.testing.assert_allclose(actual.positions, positions, atol=1e-8, rtol=0)


@pytest.mark.parametrize('change', ['cell', 'composition', 'shorter', 'longer'])
def test_public_fixed_xdatcar_write_rejects_incompatible_frames(tmp_path, capfd, change):
    pytest.importorskip('imrmlp._imrmlp')
    import imrmlp

    first = imrmlp.Atoms(numbers=[1, 2], cell=CELL1, positions=[[1, 2, 3], [2, 3, 4]])
    second = first.copy()
    if change == 'cell':
        second.cell = CELL2
    elif change == 'composition':
        second.numbers = [1, 1]
    else:
        second = imrmlp.Atoms(numbers=[1] if change == 'shorter' else [1, 2, 1],
                            cell=CELL1, positions=[[1, 2, 3]] if change == 'shorter'
                            else [[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    with pytest.raises(RuntimeError):
        imrmlp.write(tmp_path / 'rejected.xdatcar', [first, second], format='xdatcar')
    captured = capfd.readouterr()
    assert 'xdatcar6' in (captured.out + captured.err).lower()


def test_public_xdatcar6_write_retains_variable_cell_and_atom_count(tmp_path):
    pytest.importorskip('imrmlp._imrmlp')
    import imrmlp
    import numpy as np

    first = imrmlp.Atoms(numbers=[1, 2], cell=CELL1, positions=[[1, 2, 3], [2, 3, 4]])
    second = imrmlp.Atoms(numbers=[2], cell=CELL2, positions=[[1.5, 2, 2.5]])
    path = tmp_path / 'variable.xdatcar6'
    imrmlp.write(path, [first, second], format='xdatcar6')
    saved = imrmlp.read(path, format='xdatcar6')
    assert [len(atoms) for atoms in saved] == [2, 1]
    for actual, expected in zip(saved, [first, second]):
        np.testing.assert_array_equal(actual.numbers, expected.numbers)
        np.testing.assert_allclose(actual.cell, expected.cell, atol=1e-9, rtol=0)
        np.testing.assert_allclose(actual.positions, expected.positions, atol=1e-8, rtol=0)
