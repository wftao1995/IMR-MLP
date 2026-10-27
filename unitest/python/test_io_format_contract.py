"""Public names select native codecs, including their actual I/O directions."""
from pathlib import Path
import re
import shlex
import struct
import gzip

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from imrmlp import io
from test_cli_error_paths import run_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def single_thread():
    previous = imrmlp.get_num_threads()
    imrmlp.set_num_threads(1)
    yield
    imrmlp.set_num_threads(previous)


def native_formats():
    source = (ROOT / 'io/gpio.cpp').read_text()
    registry = source.split('GPIO::format_dict = {', 1)[1].split('\n};', 1)[0]
    registry = re.sub(r'//[^\n]*', '', registry)
    names = dict(re.findall(r'\{"([^"]+)",\s*GPIO::(\w+)\}', registry))
    reader = source.split('bool GPIO::read_atom_dispatch(', 1)[1].split('\nbool GPIO::read(', 1)[0]
    writer = source.split('bool GPIO::write_atom_dispatch(', 1)[1].split('\nbool GPIO::', 1)[0]
    readable = set(re.findall(r'case (\w+):', reader))
    writable = set(re.findall(r'case (\w+):', writer)) - {'UNKNOWN'}
    # These codecs are dispatched by filename, outside the stream switches.
    file_reader = source.split('bool GPIO::read(std::vector<Atoms> &atoms, std::string filename', 1)[1].split('\nbool GPIO::read(', 1)[0]
    # Stream codecs may also have filename-specific setup (for example, bounded
    # decompression for MLIPBIN); only codecs absent from dispatch are file-only.
    file_readable = set(re.findall(r'iformat == (\w+)', file_reader)) - readable
    assert file_readable == {'VASPRUN'}
    readable |= file_readable
    return names, readable, writable, file_readable


def test_all_public_constants_match_the_native_registry_and_dispatch():
    names, readable, writable, file_readable = native_formats()
    constants = {name: value for name, value in vars(io).items()
                 if not name.startswith('_') and name.isupper() and isinstance(value, str)}
    assert len(constants) == 21
    for name, value in constants.items():
        assert names[value] == name, (name, value)
        assert name in readable | writable
    assert io._READ_ONLY_FORMATS == {name for name, code in names.items() if code in readable - writable}
    assert io._WRITE_ONLY_FORMATS == {name for name, code in names.items() if code in writable - readable}
    # Filename-only does not mean whole-file-only: NetCDF also has bounded
    # filename read_one while remaining absent from the stream switch.
    one_reader = (ROOT / 'io/gpio.cpp').read_text().split(
        'bool GPIO::read_one(Atoms &atom, std::string filename',1)[1].split(
        'bool GPIO::read_one(Atoms &atom, std::istream',1)[0]
    single_file_readable = set(re.findall(r'iformat\s*==\s*(\w+)', one_reader)) & file_readable
    assert single_file_readable == set()
    assert io._READ_ALL_ONLY_FORMATS == {name for name, code in names.items()
                                       if code in file_readable-single_file_readable}


def labeled_atoms():
    atoms = imrmlp.Atoms(numbers=[1, 8], positions=[[.125, .25, .375], [1.5, 2., 2.5]],
                       cell=[[6., 0., 0.], [.2, 7., 0.], [.1, .3, 8.]])
    atoms.energy = -7.25
    atoms.forces = [[.1, -.2, .3], [-.4, .5, -.6]]
    atoms.stress = [[1., .2, .3], [.2, 2., .4], [.3, .4, 3.]]
    directions = np.arange(12, dtype=float).reshape(6, 2) / 11
    atoms.set_hvp(directions, 1.3 * directions)
    return atoms


def assert_geometry(actual, expected):
    np.testing.assert_array_equal(actual.numbers, expected.numbers)
    np.testing.assert_allclose(actual.positions, expected.positions, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual.cell, expected.cell, atol=1e-12, rtol=0)


def test_extxyz_text_metadata_survives_public_io_roundtrip(tmp_path):
    source = tmp_path / 'input.xyz'
    source.write_text(
        '1\nLattice="5 0 0 0 6 0 0 0 7" Properties=species:S:1:pos:R:3 '
        r'label="" config_type="liquid water" note="he said \"yes\"" path="C:\\tmp"'
        '\nH 1 2 3\n')
    first, second = tmp_path / 'first.xyz', tmp_path / 'second.xyz'
    imrmlp.write_one(first, imrmlp.read_one(source, format='extxyz'), format='extxyz')
    imrmlp.write_one(second, imrmlp.read_one(first, format='extxyz'), format='extxyz')
    for path in (first, second):
        fields = shlex.split(path.read_text().splitlines()[1])
        assert 'label=' in fields
        assert 'config_type=liquid water' in fields
        assert 'note=he said "yes"' in fields
        assert r'path=C:\tmp' in fields


@pytest.mark.parametrize('format', [io.MLIPBIN, 'mlpbin'])
@pytest.mark.parametrize('multiple', [False, True])
def test_binary_constant_and_legacy_alias_preserve_geometry_efs_hvp(tmp_path, format, multiple):
    atoms = labeled_atoms()
    path = tmp_path / 'roundtrip.bin'
    imrmlp.write(path, [atoms, atoms.copy()] if multiple else atoms, format=format)
    assert path.stat().st_size > 100
    assert io.guess_format(path) == io.MLIPBIN
    restored = imrmlp.read(path, format=format)
    assert len(restored) == (2 if multiple else 1)
    restored.append(imrmlp.read_one(path, format=format))
    # Native canonical spelling independently confirms the alias chose the codec.
    restored.append(imrmlp.read_one(path, format='bin'))
    for actual in restored:
        assert_geometry(actual, atoms)
        assert actual.energy == atoms.energy
        np.testing.assert_array_equal(actual.forces, atoms.forces)
        np.testing.assert_array_equal(actual.stress, atoms.stress)
        np.testing.assert_array_equal(actual.hvp_directions, atoms.hvp_directions)
        np.testing.assert_array_equal(actual.hvp_targets, atoms.hvp_targets)


YAML_FRAME = '''---
creator: LAMMPS
timestep: 0
natoms: 2
boundary: [p, p, p]
thermo:
  - keywords: [PotEng]
  - data: [-7.25]
box:
  - [0, 6]
  - [0, 7]
  - [0, 8]
keywords: [id, type, x, y, z, fx, fy, fz]
data:
  - [2, 2, 1.5, 2, 2.5, -0.4, 0.5, -0.6]
  - [1, 1, 0.125, 0.25, 0.375, 0.1, -0.2, 0.3]
...
'''


@pytest.mark.parametrize('format', [io.LMPYAML, 'lmpyaml', ''])
def test_yaml_constant_reads_geometry_energy_and_reorders_force_labels(tmp_path, format):
    path = tmp_path / 'trajectory.yaml'
    path.write_text(YAML_FRAME * 2)
    frames = imrmlp.read(path, format=format)
    assert len(frames) == 2
    frames.append(imrmlp.read_one(path, format=format))
    for atoms in frames:
        np.testing.assert_array_equal(atoms.chemmap, [0, 1])  # LAMMPS type IDs, not atomic numbers.
        np.testing.assert_allclose(atoms.positions, [[.125, .25, .375], [1.5, 2., 2.5]], atol=1e-12)
        np.testing.assert_array_equal(atoms.cell, np.diag([6., 7., 8.]))
        assert atoms.energy == -7.25
        np.testing.assert_allclose(atoms.forces, [[.1, -.2, .3], [-.4, .5, -.6]], atol=1e-12)
        assert atoms.stress is None


@pytest.mark.parametrize('format', [io.GRODATA, 'grodata', ''])
def test_gro_constant_converts_nm_geometry_to_angstrom(tmp_path, format):
    path = tmp_path / 'structure.gro'
    path.write_text('two atoms\n2\n    1MOL      H    1   0.125   0.250   0.375\n'
                    '    1MOL      O    2   0.150   0.200   0.250\n0.6 0.7 0.8\n')
    frames = imrmlp.read(path, format=format)
    assert len(frames) == 1
    frames.append(imrmlp.read_one(path, format=format))
    for atoms in frames:
        np.testing.assert_array_equal(atoms.numbers, [1, 8])
        np.testing.assert_allclose(atoms.positions, [[1.25, 2.5, 3.75], [1.5, 2., 2.5]], atol=1e-12)
        np.testing.assert_allclose(atoms.cell, np.diag([6., 7., 8.]), atol=1e-12)
        assert atoms.energy is None and atoms.forces is None and atoms.stress is None


@pytest.mark.parametrize('format', [io.N2P2DAT, 'n2p2dat', ''])
def test_n2p2_constant_reads_geometry_and_converts_energy_force_units(tmp_path, format):
    frame = ('begin\ncomment unit conversion\nlattice 6 0 0\nlattice 0 7 0\nlattice 0 0 8\n'
             'atom 1 2 3 H 0 0 0.1 -0.2 0.3\natom 2 3 4 O 0 0 -0.4 0.5 -0.6\n'
             'energy -0.25\ncharge 0\nend\n')
    path = tmp_path / 'training.n2p2'
    path.write_text(frame * 2)
    frames = imrmlp.read(path, format=format)
    assert len(frames) == 2
    frames.append(imrmlp.read_one(path, format=format))
    # GPIO's existing Hartree/Bohr convention: a name correction must not alter it.
    bohr, hartree = .529177210903, 27.211396641308
    for atoms in frames:
        np.testing.assert_array_equal(atoms.numbers, [1, 8])
        np.testing.assert_allclose(atoms.positions, np.array([[1, 2, 3], [2, 3, 4]]) * bohr, atol=1e-12)
        np.testing.assert_allclose(atoms.cell, np.diag([6., 7., 8.]) * bohr, atol=1e-12)
        assert atoms.energy == pytest.approx(-.25 * hartree, abs=1e-12)
        np.testing.assert_allclose(atoms.forces, np.array([[.1, -.2, .3], [-.4, .5, -.6]]) * hartree / bohr, atol=1e-9)
        assert atoms.stress is None


@pytest.mark.parametrize('format', [io.DLPOLY4, 'dlpoly4', ''])
def test_dlpoly_constant_writes_geometry_with_symbol_mapping(tmp_path, format):
    atoms = imrmlp.Atoms(numbers=[1, 8], elements=['H', 'O'],
                       positions=[[.125, .25, .375], [1.5, 2., 2.5]],
                       cell=[[6., 0., 0.], [.2, 7., 0.], [.1, .3, 8.]])
    path = tmp_path / 'structure.dlpoly'
    imrmlp.write_one(path, atoms, format=format)
    lines = path.read_text().splitlines()
    # Independent parsing of the CONFIG geometry, without the unavailable IMRMLP reader.
    assert [int(x) for x in lines[1].split()] == [0, 3, 2]
    np.testing.assert_allclose([[float(x) for x in line.split()] for line in lines[2:5]], atoms.cell, atol=1e-10)
    assert lines[5].split() == ['H', '1'] and lines[7].split() == ['O', '2']
    np.testing.assert_allclose([[float(x) for x in lines[i].split()] for i in [6, 8]], atoms.positions, atol=1e-10)
    assert len(lines) == 9


@pytest.mark.parametrize('format', sorted(io._READ_ONLY_FORMATS) + ['lmpyaml', 'grodata'])
@pytest.mark.parametrize('empty', [False, True])
def test_read_only_formats_reject_writing_before_touching_output(tmp_path, format, empty):
    path = tmp_path / 'keep-existing.data'
    path.write_bytes(b'original bytes\n')
    with pytest.raises(ValueError, match='does not support writing'):
        imrmlp.write(path, [] if empty else labeled_atoms(), format=format)
    assert path.read_bytes() == b'original bytes\n'


@pytest.mark.parametrize('format', sorted(io._WRITE_ONLY_FORMATS) + ['dlpoly4'])
@pytest.mark.parametrize('reader', [imrmlp.read, imrmlp.read_one])
def test_write_only_formats_reject_reading_even_an_empty_file(tmp_path, format, reader):
    path = tmp_path / 'empty.data'
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='does not support reading'):
        reader(path, format=format)


@pytest.mark.parametrize('format', sorted(io._READ_ALL_ONLY_FORMATS))
def test_file_only_readers_explain_the_supported_entrypoint(tmp_path, format):
    with pytest.raises(ValueError, match=r'use read\(\)'):
        imrmlp.read_one(tmp_path/'input.data', format=format)


@pytest.mark.parametrize('suffix', ['yaml', 'gro'])
def test_guessed_read_only_format_rejects_writing(tmp_path, suffix):
    path = tmp_path / ('uncreated.' + suffix)
    with pytest.raises(ValueError, match='does not support writing'):
        imrmlp.write(path, [])
    assert not path.exists()


@pytest.mark.parametrize('suffix', ['vtk', 'dlpoly'])
def test_guessed_write_only_format_rejects_reading(tmp_path, suffix):
    path = tmp_path / ('empty.' + suffix)
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='does not support reading'):
        imrmlp.read(path)




@pytest.mark.parametrize('compressed', [False, True])
def test_cp2kxyz_sequential_conversion_matches_batch_geometry(tmp_path, compressed):
    text = '2\nframe zero\nH 1 2 3\nO 4 5 6\n1\nframe one\nO 7 8 9'
    source = tmp_path / ('input.xyz.gz' if compressed else 'input.xyz')
    source.write_bytes(gzip.compress(text.encode()) if compressed else text.encode())
    (tmp_path / 'index.map').write_text('H 0\nO 1\n')
    result = run_command('convstr', ['--input-format', 'cp2kxyz', '--output-format', 'extxyz',
                                   '--loadrate', '1', '--outrate', '1',
                                   '--inputfiles', source.name, '--outputfile', 'out.xyz'], tmp_path)
    if 'support not enabled' in result.stderr:
        pytest.skip(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = imrmlp.read(source, format='cp2kxyz')
    actual = imrmlp.read(tmp_path / 'out.xyz', format='extxyz')
    assert [atom.natoms for atom in actual] == [2, 1]
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(after.numbers, before.numbers)
        np.testing.assert_allclose(after.positions, before.positions, atol=1e-12)


def test_cfg_scalar_magnetic_bin_conversion_reports_unsupported_instead_of_crashing(tmp_path):
    # This is a valid scalar-only magnetic BIN frame, also produced by NetCDF
    # input. Its magnetic representation has no CFG vector counterpart.
    payload = b'GPMLBIN2' + struct.pack('=Ii', 3, 1)
    payload += b'l' + struct.pack('=9d', 10, 0, 0, 0, 10, 0, 0, 0, 10)
    payload += b'p' + struct.pack('=3d', 1, 2, 3) + b't' + struct.pack('=i', 1)
    payload += b'M' + struct.pack('=d', 1.5) + b'#'
    source = tmp_path / 'scalar.bin'
    source.write_bytes(payload)
    atoms = imrmlp.read_one(source)
    control = tmp_path / 'copy.bin'
    imrmlp.write_one(control, atoms)
    assert len(imrmlp.read(control)) == 1
    (tmp_path / 'index.map').write_text('H 1\n')
    result = run_command('imrmlp', ['convstr', source.name, 'out.cfg',
                         '--input-format', 'bin', '--output-format', 'mlpcfg'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'scalar' in result.stderr.lower()
    assert not (tmp_path / 'out.cfg').exists() or (tmp_path / 'out.cfg').stat().st_size == 0


@pytest.fixture
def full_hessian_xyz(tmp_path):
    # Independent EXTXYZ input exercises dense labels across conversion codecs.
    hessian = np.eye(6)
    hessian[0, 3] = hessian[3, 0] = .25
    values = ' '.join(str(value) for value in hessian.ravel())
    source = tmp_path / 'hessian.xyz'
    source.write_text('2\nLattice="10 0 0 0 10 0 0 0 10" '
                      f'Properties=species:S:1:pos:R:3 hessian="{values}"\n'
                      'H 1 2 3\nH 4 5 6\n')
    return source, hessian


@pytest.mark.parametrize('format', ['bin', 'mlpcfg'])
def test_public_ml_label_write_retains_full_hessian(tmp_path, full_hessian_xyz, format):
    source, expected = full_hessian_xyz
    atoms = imrmlp.read_one(source, format='extxyz')
    output = tmp_path / 'retained.labels'
    imrmlp.write_one(output, atoms, format=format)
    np.testing.assert_array_equal(imrmlp.read_one(output,format=format).hessian,expected)
    np.testing.assert_array_equal(atoms.hessian,expected)


@pytest.mark.parametrize('format', ['bin', 'mlpcfg'])
def test_convstr_ml_label_output_retains_full_hessian(tmp_path, full_hessian_xyz, format):
    source, expected = full_hessian_xyz
    (tmp_path / 'index.map').write_text('H 0\n')
    output = tmp_path / 'retained.labels'
    result = run_command('imrmlp', ['convstr', source.name, output.name,
                         '--input-format', 'extxyz', '--output-format', format], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(imrmlp.read_one(output,format=format).hessian,expected)


def test_cfg_free_energy_survives_public_io_and_drives_conversion_sort(tmp_path):
    source = tmp_path / 'thermo.xyz'
    energies = [-1., -2., -3.]
    free_energies = [.4, -1.2, 0.]
    source.write_text(''.join(
        '1\nLattice="10 0 0 0 10 0 0 0 10" '
        f'Properties=species:S:1:pos:R:3 energy={energy} fenergy={free}\nH {i+1} 2 3\n'
        for i, (energy, free) in enumerate(zip(energies, free_energies))))
    cfg = tmp_path / 'thermo.cfg'
    # CFG stores model indices. Read with the same H -> 0 map used by CLI
    # conversion, while retaining the native free-energy label.
    from imrmlp._imrmlp import _GPIO
    reader = _GPIO()
    reader.set_elements(['H'])
    frames = [imrmlp.Atoms(_impl=raw, elements=['H'])
              for raw in reader.read(str(source), 'extxyz')]
    imrmlp.write(cfg, frames)
    assert cfg.read_text().count('Feature fenergy ') == 3
    (tmp_path / 'index.map').write_text('H 1\n')
    result = run_command('imrmlp', ['convstr', cfg.name, 'sorted.xyz',
                         '--input-format', 'mlpcfg', '--output-format', 'extxyz',
                         '--sort-by', 'fenergy'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = imrmlp.read(tmp_path / 'sorted.xyz')
    assert [a.energy for a in frames] == [-2., -3., -1.]
    assert [a.positions[0, 0] for a in frames] == [2., 3., 1.]
    headers = (tmp_path / 'sorted.xyz').read_text().splitlines()[1::3]
    values = [dict(token.split('=', 1) for token in shlex.split(header))
              for header in headers]
    assert [float(fields['fenergy']) for fields in values] == [-1.2, 0., .4]
