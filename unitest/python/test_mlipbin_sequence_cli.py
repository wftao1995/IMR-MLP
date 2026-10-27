"""MLIPBIN conversion must distinguish a clean end from a damaged trajectory."""
import gzip
import re
import struct

import pytest

from test_cli_error_paths import run_command


ENTRY_POINTS = ['convstr', 'imrmlp']
IO_MODES = [(0, 0), (0, 1), (1, 0), (1, 1)]
SUFFIXES = ['', '.gz']


def frame(natoms=1, energy=-2., version=3):
    # Fixed binary fixtures are independent of the writer under test. The
    # format uses native byte order and fixed 32-bit integer fields.
    return (b'GPMLBIN2' + struct.pack('=Ii', version, natoms)
            + b'l' + struct.pack('=9d', 6., 0., 0., 0., 6., 0., 0., 0., 6.)
            + b'p' + struct.pack(f'={3 * natoms}d', *([.1, .2, .3] * natoms))
            + b't' + struct.pack(f'={natoms}i', *([0] * natoms))
            + b'e' + struct.pack('=d', energy) + b'#')


def write_input(path, payload, suffix=''):
    if suffix == '.gz':
        payload = gzip.compress(payload)
    elif suffix == '.zst':
        zstd = pytest.importorskip('compression.zstd')
        payload = zstd.compress(payload)
    path.write_bytes(payload)
    return path.name


def convert(tmp_path, entry, mode, inputs, *, options=(), format='bin', output='out.cfg'):
    (tmp_path / 'index.map').write_text('H 0\n')
    args = (['convstr'] if entry == 'imrmlp' else [])
    if format is not None:
        args += ['--input-format', format]
    args += ['--output-format', 'mlpcfg', '--loadrate', str(mode[0]),
             '--outrate', str(mode[1]), *options,
             '--inputfiles', *inputs, '--outputfile', output]
    result = run_command(entry, args, tmp_path)
    if 'support not enabled' in result.stderr:
        pytest.skip(result.stderr)
    return result


def energies(path):
    if not path.exists():
        return []
    return [float(x) for x in re.findall(r'(?m)^Energy\n([^\n]+)', path.read_text())]


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', IO_MODES)
@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.parametrize('prefix', [b'', frame(), frame(natoms=0)],
                         ids=['bad-first', 'bad-tail', 'zero-atom-tail'])
@pytest.mark.extended
def test_damaged_frame_is_never_a_successful_conversion(tmp_path, entry, mode, suffix, prefix):
    source = write_input(tmp_path / f'input.bin{suffix}', prefix + b'IMRMLP', suffix)
    result = convert(tmp_path, entry, mode, [source])
    assert result.returncode == 1, result.stdout + result.stderr
    if mode[0] == 1:
        assert f'MLIPBIN frame {int(bool(prefix))}:' in result.stderr
    else:
        assert 'ERROR WHEN READING FILE' in result.stderr
    # A streaming writer may leave complete preceding frames in its output.
    # The nonzero exit, rather than whole-file rollback, is the contract.
    assert len(energies(tmp_path / 'out.cfg')) <= int(bool(prefix))


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', IO_MODES)
@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.parametrize('version', [2, 3])
def test_valid_and_zero_atom_frames_are_preserved(tmp_path, entry, mode, suffix, version):
    payload = frame(energy=-2., version=version) + frame(natoms=0, energy=-3., version=version)
    payload += frame(energy=-4., version=version)
    source = write_input(tmp_path / f'input.bin{suffix}', payload, suffix)
    result = convert(tmp_path, entry, mode, [source])
    assert result.returncode == 0, result.stdout + result.stderr
    output = (tmp_path / 'out.cfg').read_text()
    assert energies(tmp_path / 'out.cfg') == [-2., -3., -4.]
    assert re.findall(r'(?m)^Size\n(\d+)', output) == ['1', '0', '1']


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', IO_MODES)
def test_empty_file_retains_each_conversion_mode_contract(tmp_path, entry, mode):
    source = write_input(tmp_path / 'empty.bin', b'')
    result = convert(tmp_path, entry, mode, [source])
    # Both-streaming conversion can represent an empty output; modes which
    # collect structures retain their existing "no structures" failure.
    assert result.returncode == (0 if mode == (1, 1) else 1), result.stdout + result.stderr
    assert 'MLIPBIN frame' not in result.stderr
    assert energies(tmp_path / 'out.cfg') == []


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', IO_MODES)
@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.parametrize('options', [('--end', '0'), ('--select', '0'),
                                    ('--bstart', 'false', '--start', '1', '--end', '1'),
                                    ('--bstart', 'false', '--select', '1')],
                         ids=['end-zero-based', 'select-zero-based',
                              'end-one-based', 'select-one-based'])
@pytest.mark.extended
def test_explicit_prefix_does_not_parse_the_next_binary_frame(tmp_path, entry, mode, suffix, options):
    source = write_input(tmp_path / f'input.bin{suffix}', frame() + b'IMRMLP', suffix)
    # Compression is intact: this checks the requested binary-frame boundary,
    # not whether compressed buffering can notice transport errors early.
    result = convert(tmp_path, entry, mode, [source], options=options)
    assert result.returncode == 0, result.stdout + result.stderr
    assert energies(tmp_path / 'out.cfg') == [-2.]


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', IO_MODES)
@pytest.mark.parametrize('options', [('--end', '1'), ('--select', '1')], ids=['end', 'select'])
def test_prefix_limit_is_global_across_input_files(tmp_path, entry, mode, options):
    sources = [write_input(tmp_path / 'first.bin', frame()),
               write_input(tmp_path / 'second.bin', frame(energy=-3.) + b'IMRMLP'),
               write_input(tmp_path / 'third.bin', b'IMRMLP')]
    result = convert(tmp_path, entry, mode, sources, options=options)
    assert result.returncode == 0, result.stdout + result.stderr
    assert energies(tmp_path / 'out.cfg') == ([-2., -3.] if options[0] == '--end' else [-3.])


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('outrate', [0, 1])
@pytest.mark.parametrize('split_files', [False, True], ids=['one-file', 'multiple-files'])
@pytest.mark.parametrize('bad_inside_range', [False, True], ids=['outside-range', 'inside-range'])
def test_sampling_still_observes_the_consumed_range_boundary(tmp_path, entry, outrate,
                                                           split_files, bad_inside_range):
    # Frame 1 is not selected by loadrate=2, but it is inside the requested
    # range and must be parsed. Frame 2 is outside it and must not be parsed.
    prefix = frame() if bad_inside_range else frame() + frame(energy=-3.)
    if split_files:
        sources = [write_input(tmp_path / 'first.bin', prefix),
                   write_input(tmp_path / 'second.bin', b'IMRMLP')]
    else:
        sources = [write_input(tmp_path / 'first.bin', prefix + b'IMRMLP')]
    result = convert(tmp_path, entry, (2, outrate), sources, options=('--end', '1'))
    assert result.returncode == (1 if bad_inside_range else 0), result.stdout + result.stderr
    if not bad_inside_range:
        assert energies(tmp_path / 'out.cfg') == [-2.]


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', IO_MODES)
@pytest.mark.parametrize('extension', ['bin', 'cfgbin', 'bincfg'])
@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.extended
def test_inferred_format_and_filename_aliases_share_failure_contract(tmp_path, entry, mode, extension, suffix):
    source = write_input(tmp_path / f'input.{extension}{suffix}', frame(), suffix)
    result = convert(tmp_path, entry, mode, [source], format=None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert energies(tmp_path / 'out.cfg') == [-2.]
    write_input(tmp_path / source, frame() + b'IMRMLP', suffix)
    result = convert(tmp_path, entry, mode, [source], format=None, output='bad.cfg')
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', IO_MODES)
@pytest.mark.parametrize('format', ['bin', None], ids=['explicit', 'inferred'])
def test_missing_input_is_not_skipped_after_a_valid_file(tmp_path, entry, mode, format):
    source = write_input(tmp_path / 'first.bin', frame())
    result = convert(tmp_path, entry, mode, [source, 'missing.bin'], format=format)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'cannot open MLIPBIN input file missing.bin' in result.stderr
    assert not (tmp_path / 'out.cfg').exists()


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', IO_MODES)
def test_existing_but_unreadable_input_is_not_skipped(tmp_path, entry, mode):
    source = write_input(tmp_path / 'first.bin', frame())
    (tmp_path / 'directory.bin').mkdir()
    result = convert(tmp_path, entry, mode, [source, 'directory.bin'])
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(energies(tmp_path / 'out.cfg')) <= 1


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('mode', IO_MODES)
@pytest.mark.parametrize('suffix', ['.gz'])
def test_compressed_transport_damage_is_not_a_clean_eof(tmp_path, entry, mode, suffix):
    source = tmp_path / f'input.bin{suffix}'
    write_input(source, frame() + frame(energy=-3.), suffix)
    source.write_bytes(source.read_bytes()[:-1])
    result = convert(tmp_path, entry, mode, [source.name])
    assert result.returncode == 1, result.stdout + result.stderr
