"""A file's parent directory must not select its native structure codec."""
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from imrmlp import io
from test_cli_error_paths import run_command


@pytest.mark.parametrize('directory', ['vasprun-analysis', 'binary-training', 'xsf-results', 'OUTCAR-cache'])
def test_unknown_basename_is_independent_of_relative_or_absolute_parent(tmp_path, monkeypatch, directory):
    monkeypatch.chdir(tmp_path)
    relative = Path(directory) / 'sample.unknown'
    for suffix in ['', '.gz']:
        path = Path(str(relative) + suffix)
        assert io.guess_format(path) == ''
        assert io.guess_format(path.resolve()) == ''
        assert io.guess_format('sample.unknown' + suffix) == ''


@pytest.mark.parametrize('basename,expected', [
    ('POSCAR_001', 'poscar'), ('CONTCAR', 'poscar'), ('XDATCAR6', 'xdatcar6'),
    ('XDATCAR', 'xdatcar'), ('OUTCAR', 'outcar'), ('vasprun.xml', 'vasprun'),
    ('vasprun', 'vasprun'), ('trajectory.cfgbin', 'bin'),
])
@pytest.mark.parametrize('suffix', ['', '.gz'])
def test_conventional_names_and_compression_suffixes_keep_their_codecs(tmp_path, basename, expected, suffix):
    assert io.guess_format(tmp_path / 'binary-vasprun-xsf-archive' / (basename + suffix)) == expected


@pytest.mark.parametrize('name,expected', [
    ('POSCAR_result.xyz', 'extxyz'), ('OUTCAR_result.cfg', 'mlpcfg'), ('vasprun_result.bin', 'bin'),
])
@pytest.mark.parametrize('suffix', ['', '.gz'])
def test_registered_suffixes_precede_directory_and_filename_hints(tmp_path, name, expected, suffix):
    assert io.guess_format(tmp_path / 'vasprun-binary-xsf' / (name + suffix)) == expected


def test_explicit_python_format_still_selects_the_codec_for_unknown_names(tmp_path):
    path = tmp_path / 'vasprun-binary-xsf' / 'sample.unknown'
    path.parent.mkdir()
    atom = imrmlp.Atoms(numbers=[1], positions=[[1., 2., 3.]], cell=np.eye(3) * 4.)
    assert io.guess_format(path) == ''
    imrmlp.write(path, atom, format='xyz')
    restored = imrmlp.read_one(path, format='extxyz')
    np.testing.assert_array_equal(restored.numbers, [1])
    np.testing.assert_allclose(restored.positions, [[1., 2., 3.]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(restored.cell, np.eye(3) * 4., atol=1e-12, rtol=0)


@pytest.mark.parametrize('absolute', [False, True])
@pytest.mark.parametrize('explicit', [False, True])
def test_real_convstr_uses_known_suffix_or_explicit_format_under_keyword_parent(tmp_path, absolute, explicit):
    directory = tmp_path / 'vasprun-binary-xsf'
    directory.mkdir()
    name = 'sample.unknown' if explicit else 'POSCAR_result.xyz'
    source = directory / name
    source.write_text('1\nLattice="4 0 0 0 4 0 0 0 4" Properties=species:S:1:pos:R:3\nH 1 2 3\n')
    argument = source if absolute else source.relative_to(tmp_path)
    target = directory / ('converted.unknown' if explicit else 'converted.xyz')
    args = ['--inputfiles', str(argument), '--outputfile', str(target)]
    if explicit:
        args += ['--input-format', 'extxyz', '--output-format', 'extxyz']
    result = run_command('convstr', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    restored = imrmlp.read_one(target, format='extxyz')
    np.testing.assert_array_equal(restored.numbers, [1])
    np.testing.assert_allclose(restored.positions, [[1., 2., 3.]], atol=1e-12, rtol=0)
