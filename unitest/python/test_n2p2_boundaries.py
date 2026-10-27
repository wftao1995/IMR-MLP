"""N2P2 complete-frame, unit, range and CLI failure contracts."""
import gzip

import numpy as np
import pytest

from test_cli_error_paths import run_command

pytest.importorskip('imrmlp._imrmlp')
from imrmlp._imrmlp import _GPIO
import imrmlp


LATTICE = 'lattice 6 0 0\nlattice .2 7 0\nlattice .1 .3 8\n'
ROWS = 'atom 1 2 3 H .25 -.5 .1 -.2 .3\natom 2 3 4 O 0 0 -.4 .5 -.6\n'
FRAME = 'begin\ncomment fixture\n' + LATTICE + ROWS + 'energy -.25\ncharge 0\nend\n'
EMPTY = 'begin\n' + LATTICE + 'end\n'
BOHR, HARTREE = .529177210903, 27.211396641308
SUFFIXES = ['', '.gz']
ENTRY_POINTS = ['convstr', 'imrmlp']


def write_input(path, text, suffix=''):
    payload = text.encode()
    if suffix == '.gz':
        payload = gzip.compress(payload)
    elif suffix == '.zst':
        payload = pytest.importorskip('compression.zstd').compress(payload)
    path.write_bytes(payload)
    return path.name


def convert(tmp_path, entry, inputs, *, sequential_output=False, fmt='n2p2', options=()):
    (tmp_path / 'index.map').write_text('H 0\nO 1\n')
    args = ['convstr'] if entry == 'imrmlp' else []
    if fmt is not None:
        args += ['--input-format', fmt]
    args += ['--output-format', 'mlpcfg', '--loadrate', '0',
             '--outrate', '1' if sequential_output else '0', *options,
             '--inputfiles', *inputs, '--outputfile', 'out.cfg']
    result = run_command(entry, args, tmp_path)
    if 'support not enabled' in result.stderr:
        pytest.skip(result.stderr)
    return result


def check_atoms(atom, numbers=(1, 8), tolerance=1e-9):
    assert atom.natoms == 2
    np.testing.assert_array_equal(atom.numbers, numbers)
    np.testing.assert_allclose(atom.cell, np.array([[6., 0., 0.], [.2, 7., 0.], [.1, .3, 8.]]) * BOHR,
                               atol=tolerance, rtol=0)
    np.testing.assert_allclose(atom.positions, np.array([[1., 2., 3.], [2., 3., 4.]]) * BOHR,
                               atol=tolerance, rtol=0)
    np.testing.assert_allclose(atom.forces, np.array([[.1, -.2, .3], [-.4, .5, -.6]]) * HARTREE / BOHR,
                               atol=tolerance, rtol=0)
    assert atom.energy == pytest.approx(-.25 * HARTREE, abs=tolerance)
    assert atom.stress is None


def test_n2p2_charge_and_site_energy_labels_survive_extxyz(tmp_path):
    source = tmp_path / 'labels.n2p2'
    source.write_text(FRAME.replace('charge 0\n', 'charge .8\n'))
    atoms = imrmlp.read_one(source, format='n2p2')
    np.testing.assert_allclose(atoms._impl.charges, [.25, 0], atol=1e-12)
    np.testing.assert_allclose(atoms.site_energies, [-.5 * HARTREE, 0], atol=1e-12)
    output = tmp_path / 'labels.xyz'
    imrmlp.write_one(output, atoms, format='extxyz')
    assert 'total_charge=0.8' in output.read_text().splitlines()[1]
    back = imrmlp.read_one(output, format='extxyz')
    np.testing.assert_allclose(back._impl.charges, [.25, 0], atol=1e-12)
    np.testing.assert_allclose(back.site_energies, atoms.site_energies, atol=1e-12)


@pytest.mark.parametrize('fmt', ['n2p2', 'n2p2dat', ''])
@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.parametrize('newline', [False, True])
def test_python_units_aliases_compression_and_complete_empty_frames(tmp_path, fmt, suffix, newline):
    path = tmp_path / f'input.n2p2{suffix}'
    text = FRAME + EMPTY + FRAME
    write_input(path, text if newline else text.rstrip('\n'), suffix)
    frames = imrmlp.read(path, format=fmt)
    assert [atom.natoms for atom in frames] == [2, 0, 2]
    check_atoms(frames[0])
    check_atoms(frames[2])
    check_atoms(imrmlp.read_one(path, format=fmt))
    assert frames[1].energy is None


def bad_inputs():
    cases = []

    def add(label, text):
        cases.append(pytest.param(text, id=label))

    for label, text in [
        ('missing-begin', FRAME.removeprefix('begin\n')),
        ('begin-fields', FRAME.replace('begin\n', 'begin extra\n')),
        ('missing-end', FRAME.removesuffix('end\n')),
        ('end-prefix', FRAME.replace('end\n', 'ending\n')),
        ('end-fields', FRAME.replace('end\n', 'end extra\n')),
        ('short-lattice', FRAME.replace('lattice 6 0 0', 'lattice 6 0')),
        ('missing-lattice', FRAME.replace('lattice 6 0 0\n', '')),
        ('extra-lattice', FRAME.replace(ROWS, LATTICE + ROWS)),
        ('short-atom', FRAME.replace('atom 1 2 3 H .25 -.5 .1 -.2 .3', 'atom 1 2 3 H')),
        ('wrong-atom-tag', FRAME.replace('atom 1', 'other 1')),
        ('unknown-element', FRAME.replace(' H ', ' Alien ')),
        ('missing-energy', FRAME.replace('energy -.25', 'energy')),
        ('extra-energy', FRAME.replace('energy -.25', 'energy -.25 extra')),
        ('duplicate-energy', FRAME.replace('energy -.25', 'energy -.25\nenergy 1')),
        ('missing-charge', FRAME.replace('charge 0', 'charge')),
        ('duplicate-charge', FRAME.replace('charge 0', 'charge 0\ncharge 1')),
        ('atom-after-energy', FRAME.replace('energy -.25', 'energy -.25\n' + ROWS)),
        ('atom-after-charge', FRAME.replace('charge 0', 'charge 0\n' + ROWS)),
        ('unknown-record', FRAME.replace('energy -.25', 'stress 0')),
        ('too-many-tokens', FRAME.replace('energy -.25', 'energy' + ' 0' * 250)),
        ('long-bad-number', FRAME.replace('energy -.25', 'energy ' + '1' * 12000)),
        ('overflow-energy', FRAME.replace('energy -.25', 'energy 1e308')),
        ('overflow-force', FRAME.replace('.1 -.2 .3', '1e308 -.2 .3')),
        ('overflow-ignored-site-energy', FRAME.replace('.25 -.5', '.25 1e308')),
    ]:
        add(label, text)
    for invalid in ['nan', 'inf', '-inf', 'junk', '1junk', '1e999', '++1']:
        add('position-' + invalid, FRAME.replace('atom 1 2', f'atom {invalid} 2'))
        add('force-' + invalid, FRAME.replace('.1 -.2 .3', f'.1 -.2 {invalid}'))
        add('lattice-' + invalid, FRAME.replace('lattice 6 0', f'lattice {invalid} 0'))
        add('charge-' + invalid, FRAME.replace('charge 0', f'charge {invalid}'))
        add('ignored-site-energy-' + invalid, FRAME.replace('.25 -.5', f'.25 {invalid}'))
        add('ignored-atomic-charge-' + invalid, FRAME.replace('.25 -.5', f'{invalid} -.5'))
    return cases


BAD_INPUTS = bad_inputs()


@pytest.mark.parametrize('payload', BAD_INPUTS)
@pytest.mark.parametrize('reader', [imrmlp.read, imrmlp.read_one])
def test_python_rejects_every_malformed_first_frame(tmp_path, payload, reader):
    path = tmp_path / 'input.n2p2'
    path.write_text(payload)
    with pytest.raises(RuntimeError):
        reader(path, format='n2p2')


@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.parametrize('payload', [FRAME.removesuffix('end\n'), 'begin\n', 'bogus\n'])
def test_python_bulk_rejects_bad_tail_while_read_one_remains_one_frame(tmp_path, suffix, payload):
    path = tmp_path / f'input.n2p2{suffix}'
    write_input(path, FRAME + payload, suffix)
    with pytest.raises(RuntimeError):
        imrmlp.read(path, format='n2p2')
    check_atoms(imrmlp.read_one(path, format='n2p2'))


@pytest.mark.parametrize('payload', ['', ' \n\t'])
def test_clean_eof_is_empty_bulk_but_no_single_frame(tmp_path, payload):
    path = tmp_path / 'input.n2p2'
    path.write_text(payload)
    assert imrmlp.read(path, format='n2p2') == []
    with pytest.raises(RuntimeError):
        imrmlp.read_one(path, format='n2p2')


def test_comments_long_lines_mapping_and_recovery(tmp_path):
    path = tmp_path / 'input.n2p2'
    path.write_text('\n' + FRAME.replace('comment fixture', 'comment ' + 'anything ' * 2000) + '\t')
    native = _GPIO()
    native.set_elements(['O'])
    for _ in range(2):
        with pytest.raises(RuntimeError):
            native.read_one(str(path), 'n2p2')
    native.set_elements(['O', 'H'])
    result = native.read_one(str(path), 'n2p2')
    np.testing.assert_array_equal(result.chemmap, [1, 0])
    native.set_elements(['Unused'] * 120 + ['H', 'O'])
    result = native.read_one(str(path), 'n2p2')
    np.testing.assert_array_equal(result.chemmap, [120, 121])


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('sequential_output', [False, True])
@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.parametrize('fmt', ['n2p2', None])
def test_cli_valid_multiple_files_keep_units_and_mapping(tmp_path, entry, sequential_output, suffix, fmt):
    names = [write_input(tmp_path / f'{name}.n2p2{suffix}', FRAME, suffix) for name in ['first', 'second']]
    result = convert(tmp_path, entry, names, sequential_output=sequential_output, fmt=fmt)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'out.cfg', format='mlpcfg')
    assert len(frames) == 2
    for atom in frames:
        check_atoms(atom, numbers=[0, 1], tolerance=1e-7)


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('sequential_output', [False, True])
@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.parametrize('prefix', ['', FRAME])
def test_cli_bad_frames_never_replace_an_existing_output(tmp_path, entry, sequential_output, suffix, prefix):
    name = write_input(tmp_path / f'bad.n2p2{suffix}', prefix + FRAME.removesuffix('end\n'), suffix)
    sentinel = b'preserve output on input failure\n'
    (tmp_path / 'out.cfg').write_bytes(sentinel)
    result = convert(tmp_path, entry, [name], sequential_output=sequential_output, options=['--overwrite'])
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / 'out.cfg').read_bytes() == sentinel


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('sequential_output', [False, True])
@pytest.mark.parametrize('failure', ['missing', 'directory', 'bad'])
@pytest.mark.parametrize('fmt', ['n2p2', None])
def test_cli_does_not_publish_good_prefix_after_failed_input(tmp_path, entry, sequential_output, failure, fmt):
    first = write_input(tmp_path / 'first.n2p2', FRAME)
    broken = tmp_path / 'broken.n2p2'
    if failure == 'bad':
        broken.write_text('begin\n')
    elif failure == 'directory':
        broken.mkdir()
    result = convert(tmp_path, entry, [first, broken.name], sequential_output=sequential_output, fmt=fmt)
    assert result.returncode == 1, result.stdout + result.stderr
    assert not (tmp_path / 'out.cfg').exists()


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('sequential_output', [False, True])
@pytest.mark.parametrize('fmt', ['n2p2', None])
def test_cli_sequential_input_retains_frames_and_labels(tmp_path, entry, sequential_output, fmt):
    name = write_input(tmp_path / 'input.n2p2', FRAME + EMPTY + FRAME)
    result = convert(tmp_path, entry, [name], sequential_output=sequential_output, fmt=fmt,
                     options=['--loadrate', '1', '--output-format', 'extxyz'])
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'out.cfg', format='extxyz')
    assert [atom.natoms for atom in frames] == [2, 0, 2]
    for atom in (frames[0], frames[2]):
        check_atoms(atom, tolerance=1e-7)
        np.testing.assert_allclose(atom._impl.charges, [.25, 0], atol=1e-12)
        np.testing.assert_allclose(atom.site_energies, [-.5 * HARTREE, 0], atol=1e-7)
    assert (tmp_path / 'out.cfg').read_text().count('total_charge=0') == 2


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('sequential_output', [False, True])
def test_cli_ranges_do_not_parse_bad_tail_beyond_a_skipped_end_frame(tmp_path, entry, sequential_output):
    first = write_input(tmp_path / 'first.n2p2', FRAME)
    second = write_input(tmp_path / 'second.n2p2', FRAME + 'BAD TAIL\n')
    result = convert(tmp_path, entry, [first, second], sequential_output=sequential_output,
                     options=['--loadrate', '2', '--end', '1'])
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'out.cfg', format='mlpcfg')
    assert len(frames) == 1
    check_atoms(frames[0], numbers=[0, 1], tolerance=1e-7)


@pytest.mark.parametrize('suffix', ['.gz'])
def test_cli_sequential_input_decompresses_on_demand(tmp_path, suffix):
    name = write_input(tmp_path / ('input.n2p2' + suffix), FRAME * 2, suffix)
    result = convert(tmp_path, 'convstr', [name], sequential_output=True, fmt=None,
                     options=['--loadrate', '1'])
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'out.cfg', format='mlpcfg')
    assert len(frames) == 2
    for atom in frames:
        check_atoms(atom, numbers=[0, 1], tolerance=1e-7)


@pytest.mark.parametrize('sequential_output', [False, True])
@pytest.mark.parametrize('limit', ['--end', '--select'])
def test_cli_sequential_range_stops_before_bad_tail_and_next_file(tmp_path, sequential_output, limit):
    first = write_input(tmp_path / 'first.n2p2', FRAME)
    second = write_input(tmp_path / 'second.n2p2', FRAME + 'BAD TAIL\n')
    outside = write_input(tmp_path / 'outside.n2p2', 'NOT A FRAME\n')
    result = convert(tmp_path, 'convstr', [first, second, outside], sequential_output=sequential_output,
                     options=['--loadrate', '1', limit, '1'])
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'out.cfg', format='mlpcfg')
    assert len(frames) == (2 if limit == '--end' else 1)
    for atom in frames:
        check_atoms(atom, numbers=[0, 1], tolerance=1e-7)


@pytest.mark.parametrize('sequential_output', [False, True])
def test_cli_sequential_input_reports_broken_tail(tmp_path, sequential_output):
    name = write_input(tmp_path / 'input.n2p2', FRAME + 'begin\n' + LATTICE + ROWS)
    result = convert(tmp_path, 'convstr', [name], sequential_output=sequential_output,
                     options=['--loadrate', '1'])
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'invalid or truncated frame 1' in result.stderr
