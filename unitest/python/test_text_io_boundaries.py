"""XSF, whitespace GRO and CP2KXYZ reject incomplete or unsafe structures."""
import gzip

import numpy as np
import pytest

from test_cli_error_paths import run_command

pytest.importorskip('imrmlp._imrmlp')
from imrmlp._imrmlp import _GPIO
import imrmlp


CELL = [[6., 0., 0.], [.2, 7., 0.], [.1, .3, 8.]]
POSITIONS = [[.125, .25, .375], [1.5, 2., 2.5]]
XSF_HEADER = ('# test crystal\nCRYSTAL\nPRIMVEC\n6 0 0\n.2 7 0\n.1 .3 8\n'
              'CONVVEC\n6 0 0\n.2 7 0\n.1 .3 8\nPRIMCOORD\n')
XSF = XSF_HEADER + '2 1\nH .125 .25 .375\nO 1.5 2 2.5\n'
GRO = ('two atoms\n2\n1MOL H 1 .0125 .025 .0375\n1MOL O 2 .15 .2 .25\n'
       '.6 .7 .8 0 0 .02 0 .01 .03\n')
CP2KXYZ = '2\nframe\nH .125 .25 .375\nO 1.5 2 2.5\n'
VALID = {'xsf': XSF, 'gro': GRO, 'cp2kxyz': CP2KXYZ}
SUFFIXES = ['', '.gz']
ENTRY_POINTS = ['convstr', 'imrmlp']
SUPPORTED = [('xsf', (0, 0)), ('xsf', (0, 1)), ('xsf', (1, 0)),
             ('gro', (0, 0)), ('gro', (0, 1)),
             ('cp2kxyz', (0, 0)), ('cp2kxyz', (0, 1))]


def write_input(path, text, suffix=''):
    payload = text.encode()
    if suffix == '.gz':
        payload = gzip.compress(payload)
    elif suffix == '.zst':
        payload = pytest.importorskip('compression.zstd').compress(payload)
    path.write_bytes(payload)
    return path.name


def convert(tmp_path, entry, fmt, mode, inputs, *, options=(), output='out.cfg'):
    (tmp_path / 'index.map').write_text('H 0\nO 1\n')
    args = ['convstr'] if entry == 'imrmlp' else []
    if fmt is not None:
        args += ['--input-format', fmt]
    args += ['--output-format', 'mlpcfg', '--loadrate', str(mode[0]),
             '--outrate', str(mode[1]), *options,
             '--inputfiles', *inputs, '--outputfile', output]
    result = run_command(entry, args, tmp_path)
    if 'support not enabled' in result.stderr:
        pytest.skip(result.stderr)
    return result


def check_atoms(atoms, *, cell=CELL, count=2, numbers=(1, 8)):
    assert atoms.natoms == count
    if count:
        np.testing.assert_array_equal(atoms.numbers, numbers)
        np.testing.assert_allclose(atoms.positions, POSITIONS, atol=1e-12, rtol=0)
    if cell is not None:
        np.testing.assert_allclose(atoms.cell, cell, atol=1e-12, rtol=0)
    assert atoms.energy is None and atoms.forces is None


@pytest.mark.parametrize('fmt', list(VALID))
@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.parametrize('reader', [imrmlp.read, imrmlp.read_one])
@pytest.mark.parametrize('trailing_newline', [False, True])
def test_valid_geometry_and_final_line_boundary(tmp_path, fmt, suffix, reader, trailing_newline):
    path = tmp_path / f'input.{fmt}{suffix}'
    write_input(path, VALID[fmt] if trailing_newline else VALID[fmt].rstrip('\n'), suffix)
    result = reader(path, format=fmt)
    atoms = result[0] if reader is imrmlp.read else result
    if reader is imrmlp.read:
        assert len(result) == 1
    check_atoms(atoms, cell=None if fmt == 'cp2kxyz' else CELL)


@pytest.mark.parametrize('fmt', ['xsf', 'gro', 'grodata'])
@pytest.mark.parametrize('explicit', [False, True])
def test_registered_guesses_and_python_gro_alias(tmp_path, fmt, explicit):
    native = 'gro' if fmt == 'grodata' else fmt
    path = tmp_path / f'input.{native}'
    write_input(path, VALID[native])
    for result in [imrmlp.read(path, format=fmt if explicit else '')[0],
                   imrmlp.read_one(path, format=fmt if explicit else '')]:
        check_atoms(result)


@pytest.mark.parametrize('reader', [imrmlp.read, imrmlp.read_one])
def test_xsf_numeric_species_comments_and_complete_zero_atom_structure(tmp_path, reader):
    path = tmp_path / 'input.xsf'
    numeric = XSF.replace('H .125', '1 .125').replace('O 1.5', '8 1.5')
    # Lexical whitespace/comments do not imply support for other XSF sections.
    numeric = '\n# leading comment\n' + numeric.replace('PRIMVEC\n', ' PRIMVEC \n# cell\n')
    path.write_text(numeric)
    result = reader(path, format='xsf')
    check_atoms(result[0] if reader is imrmlp.read else result)
    path.write_text(XSF_HEADER + '0 1')
    result = reader(path, format='xsf')
    check_atoms(result[0] if reader is imrmlp.read else result, count=0)


@pytest.mark.parametrize('reader', [imrmlp.read, imrmlp.read_one])
def test_xsf_distinct_conventional_cell_keeps_primitive_geometry(tmp_path, reader):
    path = tmp_path / 'input.xsf'
    path.write_text(XSF.replace('CONVVEC\n6 0 0', 'CONVVEC\n7 0 0'))
    result = reader(path, format='xsf')
    check_atoms(result[0] if reader is imrmlp.read else result)


@pytest.mark.parametrize('reader', [imrmlp.read, imrmlp.read_one])
def test_gro_orthogonal_box_multiple_residues_and_ignored_velocities(tmp_path, reader):
    path = tmp_path / 'input.gro'
    path.write_text(GRO.replace('1MOL O 2 .15 .2 .25', '2SOL O 2 .15 .2 .25 .1 .2 .3')
                   .replace('.6 .7 .8 0 0 .02 0 .01 .03', '.6 .7 .8'))
    result = reader(path, format='gro')
    check_atoms(result[0] if reader is imrmlp.read else result, cell=np.diag([6., 7., 8.]))


@pytest.mark.parametrize('suffix', SUFFIXES)
def test_cp2k_multiframe_keeps_complete_zero_atom_frames(tmp_path, suffix):
    path = tmp_path / f'input.cp2kxyz{suffix}'
    write_input(path, CP2KXYZ + '0\nzero frame\n' + CP2KXYZ.rstrip('\n'), suffix)
    result = imrmlp.read(path, format='cp2kxyz')
    assert [atoms.natoms for atoms in result] == [2, 0, 2]
    check_atoms(result[0], cell=None)
    check_atoms(result[2], cell=None)
    check_atoms(imrmlp.read_one(path, format='cp2kxyz'), cell=None)


def test_cp2k_clean_eof_is_distinct_from_missing_comment(tmp_path):
    path = tmp_path / 'input.cp2kxyz'
    path.write_text('')
    assert imrmlp.read(path, format='cp2kxyz') == []
    with pytest.raises(RuntimeError):
        imrmlp.read_one(path, format='cp2kxyz')
    for payload in ['0', '0\n', CP2KXYZ + '0\n']:
        path.write_text(payload)
        with pytest.raises(RuntimeError):
            imrmlp.read(path, format='cp2kxyz')


def bad_inputs():
    cases = []
    def add(fmt, label, payload):
        cases.append(pytest.param(fmt, payload, id=fmt + '-' + label))
    for species in ['-1', '119', '999999999999999999999', '8junk', 'NoSuchSpecies']:
        add('xsf', 'species-' + species, XSF.replace('H .125', species + ' .125'))
    for count in ['-1 1', '2147483648 1', '999999999999999999999 1', '2junk 1', '2', '2 2']:
        add('xsf', 'count-' + count, XSF.replace('2 1\n', count + '\n'))
    for label, payload in [
        ('empty', ''), ('bad-label', XSF.replace('CRYSTAL', 'MOLECULE')),
        ('missing-primvec', XSF.replace('PRIMVEC\n', '')),
        ('short-cell', XSF.replace('6 0 0\n', '6 0\n')),
        ('nonfinite-cell', XSF.replace('6 0 0\n', 'nan 0 0\n')),
        ('short-atom', XSF.replace('H .125 .25 .375', 'H .125 .25')),
        ('nonfinite-position', XSF.replace('H .125', 'H inf')),
        ('invalid-position', XSF.replace('H .125', 'H junk')),
        ('truncated', XSF.rsplit('O ', 1)[0]),
        ('huge-short', XSF_HEADER + '2147483647 1\n')]:
        add('xsf', label, payload)
    for count in ['', '-1', '0', '2147483648', '999999999999999999999', '2junk']:
        add('gro', 'count-' + count, GRO.replace('two atoms\n2\n', 'two atoms\n' + count + '\n'))
    for label, payload in [
        ('empty', ''), ('short-atom', GRO.replace('1MOL H 1 .0125 .025 .0375', '1MOL H')),
        ('unknown-species', GRO.replace('1MOL H', '1MOL NoSuchSpecies')),
        ('invalid-position', GRO.replace('H 1 .0125', 'H 1 junk')),
        ('nonfinite-position', GRO.replace('H 1 .0125', 'H 1 nan')),
        ('overflow-position', GRO.replace('H 1 .0125', 'H 1 1e308')),
        ('missing-box', GRO.rsplit('.6 ', 1)[0]),
        ('short-box', GRO.replace('.6 .7 .8 0 0 .02 0 .01 .03', '1 1')),
        ('four-box', GRO.replace('.6 .7 .8 0 0 .02 0 .01 .03', '1 1 1 1')),
        ('ten-box', GRO.replace('.6 .7 .8 0 0 .02 0 .01 .03', '1 1 1 0 0 0 0 0 0 0')),
        ('nonfinite-box', GRO.replace('.6 .7 .8', 'inf .7 .8')),
        ('overflow-box', GRO.replace('.6 .7 .8', '1e308 .7 .8')),
        ('huge-short', 'short\n2147483647\n')]:
        add('gro', label, payload)
    for count in [' ', '-1', '2147483648', '999999999999999999999', '2junk']:
        add('cp2kxyz', 'count-' + count, CP2KXYZ.replace('2\n', count + '\n', 1))
    for label, payload in [
        ('missing-comment', '2\n'), ('zero-no-comment', '0\n'),
        ('short-atom', CP2KXYZ.replace('H .125 .25 .375', 'H .125')),
        ('unknown-species', CP2KXYZ.replace('H .125', 'NoSuchSpecies .125')),
        ('invalid-position', CP2KXYZ.replace('H .125', 'H junk')),
        ('nonfinite-position', CP2KXYZ.replace('H .125', 'H -inf')),
        ('truncated', CP2KXYZ.rsplit('O ', 1)[0]),
        ('huge-short', '2147483647\nshort\n')]:
        add('cp2kxyz', label, payload)
    return cases


BAD_INPUTS = bad_inputs()


@pytest.mark.parametrize('fmt,payload', BAD_INPUTS)
@pytest.mark.parametrize('reader', [imrmlp.read, imrmlp.read_one])
def test_python_rejects_invalid_first_structure(tmp_path, fmt, payload, reader):
    path = tmp_path / f'bad.{fmt}'
    path.write_text(payload)
    with pytest.raises(RuntimeError):
        reader(path, format=fmt)


@pytest.mark.parametrize('fmt', list(VALID))
@pytest.mark.parametrize('reader', ['read', 'read_one'])
def test_native_python_mapping_is_explicit_and_reader_recovers(tmp_path, fmt, reader):
    native = _GPIO()
    native.set_elements(['O'])
    path = tmp_path / f'input.{fmt}'
    path.write_text(VALID[fmt])
    with pytest.raises(RuntimeError):
        getattr(native, reader)(str(path), fmt)
    native.set_elements(['H', 'O'])
    decoded = getattr(native, reader)(str(path), fmt)
    atom = decoded[0] if reader == 'read' else decoded
    np.testing.assert_array_equal(atom.chemmap, [0, 1])


@pytest.mark.parametrize('species', ['0', 'X', '118', 'Og'])
def test_xsf_dummy_and_upper_element_respect_custom_mapping(tmp_path, species):
    native = _GPIO()
    native.set_elements(['X', 'Og'])
    path = tmp_path / 'edge.xsf'
    path.write_text(XSF_HEADER + f'1 1\n{species} 0 0 0\n')
    atom = native.read_one(str(path), 'xsf')
    assert len(atom) == 1
    np.testing.assert_array_equal(atom.chemmap, [0 if species in ['0', 'X'] else 1])


@pytest.mark.parametrize('fmt', ['xsf', 'cp2kxyz', 'extxyz'])
@pytest.mark.parametrize('reader', [imrmlp.read, imrmlp.read_one])
def test_default_mapping_accepts_og_and_custom_mapping_can_exclude_it(tmp_path, fmt, reader):
    payload = {
        'xsf': XSF_HEADER + '1 1\nOg 0 0 0\n',
        'cp2kxyz': '1\nupper element\nOg 0 0 0\n',
        'extxyz': '1\nProperties=species:S:1:pos:R:3\nOg 0 0 0\n',
    }[fmt]
    path = tmp_path / f'upper.{fmt}'
    path.write_text(payload)
    result = reader(path, format=fmt)
    atom = result[0] if reader is imrmlp.read else result
    np.testing.assert_array_equal(atom.numbers, [118])
    native = _GPIO()
    native.set_elements(['H'])
    with pytest.raises(RuntimeError):
        native.read_one(str(path), fmt)
    native.set_elements([])
    np.testing.assert_array_equal(native.read_one(str(path), fmt).chemmap, [118])


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('fmt,mode', SUPPORTED)
@pytest.mark.parametrize('suffix', SUFFIXES)
def test_cli_supported_paths_keep_two_valid_inputs(tmp_path, entry, fmt, mode, suffix):
    sources = [write_input(tmp_path / f'{name}.{fmt}{suffix}', VALID[fmt], suffix)
               for name in ['first', 'second']]
    result = convert(tmp_path, entry, fmt, mode, sources)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'out.cfg', format='mlpcfg')
    assert len(frames) == 2
    for atoms in frames:
        check_atoms(atoms, cell=None if fmt == 'cp2kxyz' else CELL, numbers=[0, 1])


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('fmt,mode', SUPPORTED)
@pytest.mark.parametrize('failure', ['bad', 'missing', 'directory'])
def test_cli_never_publishes_success_after_skipping_a_failed_input(tmp_path, entry, fmt, mode, failure):
    first = write_input(tmp_path / f'first.{fmt}', VALID[fmt])
    broken = tmp_path / f'bad.{fmt}'
    if failure == 'bad':
        broken.write_text({'xsf': '# truncated\nCRYSTAL\n',
                           'gro': 'short\n1\n1MOL NoSuch 1 0 0 0\n1 1 1\n',
                           'cp2kxyz': '1\ntruncated\n'}[fmt])
    elif failure == 'directory':
        broken.mkdir()
    result = convert(tmp_path, entry, fmt, mode, [first, broken.name])
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('fmt,mode', SUPPORTED)
@pytest.mark.parametrize('suffix', SUFFIXES)
def test_cli_rejects_a_bad_first_frame_in_every_supported_path(tmp_path, entry, fmt, mode, suffix):
    payload = {'xsf': XSF.replace('H .125', '119 .125'),
               'gro': GRO.replace('H 1 .0125', 'H 1 nan'),
               'cp2kxyz': CP2KXYZ.replace('H .125', 'NoSuch .125')}[fmt]
    source = write_input(tmp_path / f'bad.{fmt}{suffix}', payload, suffix)
    result = convert(tmp_path, entry, fmt, mode, [source])
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('fmt,mode', [('xsf', (1, 1))])
def test_cli_xsf_sequential_reader_returns_the_single_frame(tmp_path, entry, fmt, mode):
    source = write_input(tmp_path / f'input.{fmt}', VALID[fmt])
    result = convert(tmp_path, entry, fmt, mode, [source])
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'out.cfg', format='mlpcfg')
    assert len(frames) == 1
    check_atoms(frames[0], numbers=[0, 1])


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('outrate', [0, 1])
def test_gro_cli_sequential_readers_return_both_frames(tmp_path, entry, outrate):
    source = write_input(tmp_path / 'input.gro',
                         GRO + GRO.replace('H 1 .0125', 'H 1 .0625'))
    result = convert(tmp_path, entry, None, (1, outrate), [source])
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'out.cfg', format='mlpcfg')
    assert len(frames) == 2
    check_atoms(frames[0], numbers=[0, 1])
    np.testing.assert_array_equal(frames[1].numbers, [0, 1])
    np.testing.assert_allclose(frames[1].positions,
                               [[.625, .25, .375], [1.5, 2., 2.5]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(frames[1].cell, CELL, rtol=0, atol=1e-12)


@pytest.mark.parametrize('outrate,bad_inside,suffix',
                         [(0, False, ''), (1, False, ''),
                          (2, True, ''), (1, False, '.gz')])
def test_gro_cli_sampling_checks_consumed_frames_only(tmp_path, outrate, bad_inside, suffix):
    prefix = GRO if bad_inside else GRO * 2
    source = write_input(tmp_path / f'input.gro{suffix}',
                         prefix + 'bad frame\n2\n1MOL H 1 .1 .2 .3\n', suffix)
    # Sequential input validates frame 1 even when output sampling skips it.
    # The legacy bulk GRO reader intentionally returns only the first frame.
    result = convert(tmp_path, 'convstr', None, (1, outrate), [source],
                     options=('--end', '1'))
    assert result.returncode == (1 if bad_inside else 0), result.stdout + result.stderr
    if not bad_inside:
        frames = imrmlp.read(tmp_path / 'out.cfg', format='mlpcfg')
        assert len(frames) == 2
        for frame in frames:
            check_atoms(frame, numbers=[0, 1])


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', [(1, 0), (1, 1)])
def test_cp2k_cli_sequential_readers_return_both_frames(tmp_path, entry, mode):
    payload = CP2KXYZ + CP2KXYZ.replace('H .125', 'H .625')
    source = write_input(tmp_path / 'input.cp2kxyz', payload)
    result = convert(tmp_path, entry, 'cp2kxyz', mode, [source])
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'out.cfg', format='mlpcfg')
    assert len(frames) == 2
    check_atoms(frames[0], cell=None, numbers=[0, 1])
    np.testing.assert_array_equal(frames[1].numbers, [0, 1])
    np.testing.assert_allclose(frames[1].positions,
                               [[.625, .25, .375], [1.5, 2., 2.5]], atol=1e-12, rtol=0)


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', [(0, 0), (0, 1)])
@pytest.mark.parametrize('suffix', SUFFIXES)
def test_cp2k_cli_bad_tail_does_not_become_a_successful_prefix(tmp_path, entry, mode, suffix):
    source = write_input(tmp_path / f'input.cp2kxyz{suffix}', CP2KXYZ + '0\n', suffix)
    result = convert(tmp_path, entry, 'cp2kxyz', mode, [source])
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'ERROR WHEN READING FILE' in result.stderr


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('fmt,mode', [item for item in SUPPORTED if item[0] != 'cp2kxyz'])
def test_cli_guessed_xsf_and_gro_formats_share_the_failure_contract(tmp_path, entry, fmt, mode):
    source = write_input(tmp_path / f'input.{fmt}', VALID[fmt])
    result = convert(tmp_path, entry, None, mode, [source])
    assert result.returncode == 0, result.stdout + result.stderr
    result = convert(tmp_path, entry, None, mode, [source, f'missing.{fmt}'], output='missing.cfg')
    assert result.returncode == 1, result.stdout + result.stderr
    assert not (tmp_path / 'missing.cfg').exists()


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('outrate', [0, 1])
@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.parametrize('split_files', [False, True], ids=['one-file', 'multiple-files'])
@pytest.mark.parametrize('bad_inside_range', [False, True], ids=['outside-range', 'inside-range'])
def test_cp2k_sampling_stops_at_the_consumed_inclusive_end(tmp_path, entry, outrate, suffix,
                                                          split_files, bad_inside_range):
    # Frame 1 is skipped by loadrate=2 but is still within end=1. It must be
    # parsed; frame 2 is outside the requested range and must not be parsed.
    prefix = CP2KXYZ if bad_inside_range else CP2KXYZ * 2
    if split_files:
        sources = [write_input(tmp_path / f'first.cp2kxyz{suffix}', prefix, suffix),
                   write_input(tmp_path / f'second.cp2kxyz{suffix}', '0\n', suffix)]
    else:
        sources = [write_input(tmp_path / f'input.cp2kxyz{suffix}', prefix + '0\n', suffix)]
    result = convert(tmp_path, entry, 'cp2kxyz', (2, outrate), sources, options=('--end', '1'))
    assert result.returncode == (1 if bad_inside_range else 0), result.stdout + result.stderr
    if not bad_inside_range:
        frames = imrmlp.read(tmp_path / 'out.cfg', format='mlpcfg')
        assert len(frames) == 1
        check_atoms(frames[0], cell=None, numbers=[0, 1])


CP2K_STREAM = ('2\n i = 1, time = 0.0, E = -1.5\n'
               'H .125 .25 .375\nO 1.5 2 2.5\n'
               '0\n\n'
               '1\n i = 3, time = 1.0, E = -2.0\nO -1 .5 3\n')


def _assert_cp2k_stream_frames(frames):
    assert [len(frame) for frame in frames] == [2, 0, 1]
    for frame, numbers in zip(frames, ([1, 8], [], [8])):
        np.testing.assert_array_equal(frame.numbers, numbers)
        assert frame.energy is None and frame.free_energy is None
        assert frame.forces is None
    # These are Cartesian Angstrom inputs, not Bohr values or comment energies.
    np.testing.assert_allclose(frames[0].positions, POSITIONS, rtol=0, atol=1e-12)
    np.testing.assert_allclose(frames[2].positions, [[-1., .5, 3.]], rtol=0, atol=1e-12)


@pytest.mark.parametrize('format', ['', 'cp2kxyz'])
def test_cp2k_iread_preserves_frame_order_units_and_owned_atoms(tmp_path, format):
    path = tmp_path / 'trajectory.cp2kxyz'
    path.write_text(CP2K_STREAM)
    reader = imrmlp.iread(path, format=format)
    first = next(reader)
    frames = [first, *reader]
    _assert_cp2k_stream_frames(frames)
    bulk = imrmlp.read(path, format='cp2kxyz')
    assert len(bulk) == len(frames)
    for actual, reference in zip(frames, bulk):
        np.testing.assert_array_equal(actual.numbers, reference.numbers)
        np.testing.assert_allclose(actual.positions, reference.positions, rtol=0, atol=1e-12)
    frames[-1].positions = [[9., 8., 7.]]
    np.testing.assert_allclose(first.positions, POSITIONS, rtol=0, atol=1e-12)
    empty = tmp_path / 'empty.cp2kxyz'
    empty.write_text('')
    assert list(imrmlp.iread(empty)) == []


def test_cp2k_iread_delays_bad_second_frame_and_allows_early_close(tmp_path):
    path = tmp_path / 'bad-tail.cp2kxyz'
    path.write_text(CP2KXYZ + '1\ncomment\n')
    reader = imrmlp.iread(path)
    first = next(reader)
    np.testing.assert_allclose(first.positions, POSITIONS, rtol=0, atol=1e-12)
    with pytest.raises(RuntimeError, match='CP2KXYZ.*frame 1'):
        next(reader)
    with pytest.raises(StopIteration):
        next(reader)
    np.testing.assert_allclose(first.positions, POSITIONS, rtol=0, atol=1e-12)

    early = imrmlp.iread(path)
    saved = next(early)
    early.close()
    early.close()
    with pytest.raises(StopIteration):
        next(early)
    np.testing.assert_allclose(saved.positions, POSITIONS, rtol=0, atol=1e-12)


def test_cp2k_iread_supports_normal_gzip_with_automatic_format(tmp_path):
    path = tmp_path / 'trajectory.cp2kxyz.gz'
    path.write_bytes(gzip.compress(CP2K_STREAM.encode()))
    _assert_cp2k_stream_frames(list(imrmlp.iread(path)))


def test_cp2k_geometry_only_read_supports_nonperiodic_descriptor(tmp_path):
    from imrmlp.descriptors import Descriptor

    path = tmp_path / 'molecule.cp2kxyz'
    path.write_text('2\nmolecule geometry only\nH 0 0 0\nH 1.5 0 0\n')
    atoms = imrmlp.read_one(path)
    np.testing.assert_array_equal(atoms.pbc, [False, False, False])
    np.testing.assert_array_equal(atoms.cell, np.zeros((3, 3)))
    np.testing.assert_array_equal(atoms.numbers, [1, 1])
    np.testing.assert_allclose(atoms.positions, [[0., 0., 0.], [1.5, 0., 0.]], rtol=0, atol=1e-12)
    assert atoms.energy is None and atoms.free_energy is None
    descriptor = Descriptor('behler', {
        'behler:dispcount': '1', 'behler:dispselect': '1', 'behler:cutoff': '4',
    })
    expected = .5 * (1. + np.cos(1.5 * np.pi / 4.))
    np.testing.assert_allclose(descriptor.compute(atoms), [[expected], [expected]],
                               rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(atoms.pbc, [False, False, False])
    np.testing.assert_array_equal(atoms.cell, np.zeros((3, 3)))


GRO_STREAM = ('first frame\n3\n'
              '1WAT H 1 .1 .2 .3 .01 .02 .03\n'
              '1WAT O 2 .4 .5 .6\n'
              '2MOL H 3 .7 .8 .9\n1 2 3\n'
              '\n2\n3MOL O 1 -.1 .25 .35\n3MOL H 2 .2 .3 .4\n2 3 4')


def _assert_gro_stream_frames(frames):
    assert [len(frame) for frame in frames] == [3, 2]
    for frame, numbers, groups in zip(frames, ([1, 8, 1], [8, 1]), ([2, 1], [2])):
        np.testing.assert_array_equal(frame.numbers, numbers)
        assert list(frame._impl.molsize) == groups
        assert frame.energy is None and frame.free_energy is None
        assert frame.forces is None and frame.velocities is None
    np.testing.assert_allclose(frames[0].positions,
                               [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(frames[1].positions,
                               [[-1., 2.5, 3.5], [2., 3., 4.]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(frames[0].cell, np.diag([10., 20., 30.]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(frames[1].cell, np.diag([20., 30., 40.]), rtol=0, atol=1e-12)
    assert list(frames[0]._impl.molid) == [0, 0, 1]
    assert list(frames[1]._impl.molid) == [0, 0]


@pytest.mark.parametrize('format', ['', 'grodata'])
def test_gro_iread_preserves_units_groups_and_owned_frames(tmp_path, format):
    path = tmp_path / 'trajectory.gro'
    path.write_text(GRO_STREAM)
    reader = imrmlp.iread(path, format=format)
    first = next(reader)
    frames = [first, *reader]
    _assert_gro_stream_frames(frames)
    # Bulk GRO intentionally retains its first-frame contract.
    bulk = imrmlp.read(path)
    assert len(bulk) == 1
    np.testing.assert_allclose(bulk[0].positions, first.positions, rtol=0, atol=1e-12)
    frames[1].positions = np.zeros((2, 3))
    np.testing.assert_allclose(first.positions,
                               [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], rtol=0, atol=1e-12)


def test_gro_iread_delays_bad_second_frame_and_allows_early_close(tmp_path):
    path = tmp_path / 'bad-tail.gro'
    path.write_text(GRO + 'bad frame\n2\n1MOL H 1 .1 .2 .3\n')
    reader = imrmlp.iread(path)
    first = next(reader)
    check_atoms(first)
    with pytest.raises(RuntimeError, match='GRO.*frame 1'):
        next(reader)
    with pytest.raises(StopIteration):
        next(reader)
    check_atoms(first)
    early = imrmlp.iread(path)
    saved = next(early)
    early.close()
    with pytest.raises(StopIteration):
        next(early)
    check_atoms(saved)


def test_gro_iread_supports_normal_gzip_with_automatic_format(tmp_path):
    path = tmp_path / 'trajectory.gro.gz'
    path.write_bytes(gzip.compress(GRO_STREAM.encode()))
    _assert_gro_stream_frames(list(imrmlp.iread(path)))


@pytest.mark.parametrize('pbc,label', [([False, False, False], 'ATOMS'),
                                      ([True, True, False], 'SLAB')])
def test_xsf_public_roundtrip_preserves_supported_boundaries(tmp_path, pbc, label):
    atoms = imrmlp.Atoms(numbers=[1, 8], positions=[[0., 0., 0.], [0., 0., 9.]],
                       cell=np.eye(3) * 10, pbc=pbc)
    positions, cell = atoms.positions.copy(), atoms.cell.copy()
    path = tmp_path / 'geometry.xsf'
    imrmlp.write(path, atoms, format='xsf')
    assert label in path.read_text().splitlines()
    restored = imrmlp.read_one(path)
    np.testing.assert_array_equal(restored.pbc, pbc)
    np.testing.assert_array_equal(restored.numbers, [1, 8])
    np.testing.assert_allclose(restored.positions, positions, rtol=0, atol=1e-12)
    np.testing.assert_allclose(restored.cell, cell if any(pbc) else np.zeros((3, 3)),
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(atoms.pbc, pbc)
    np.testing.assert_array_equal(atoms.positions, positions)
    np.testing.assert_array_equal(atoms.cell, cell)
    np.testing.assert_array_equal(atoms.numbers, [1, 8])
