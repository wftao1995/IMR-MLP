"""findint's explicit input species map must preserve chemical identities."""
import shlex

import pytest
from test_cli_error_paths import run_command

HOST = ('1\nLattice="4 0 0 0 4 0 0 0 4" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\nSi 0 0 0\n')


def arguments(output='output.xyz'):
    return ['findint', '--input=host.xyz', '--input-format=extxyz',
            '--output=' + output, '--output-format=extxyz', '--insert=C', '--top=1']


def assert_cubic_host_and_carbon(path):
    lines = path.read_text().splitlines()
    assert int(lines[0]) == 2
    header = dict(field.split('=', 1) for field in shlex.split(lines[1]) if '=' in field)
    assert [float(x) for x in header['Lattice'].split()] == [4, 0, 0, 0, 4, 0, 0, 0, 4]
    rows = [line.split() for line in lines[2:]]
    assert [row[0] for row in rows] == ['Si', 'C']
    assert [float(x) for x in rows[0][1:4]] == [0, 0, 0]
    assert [float(x) for x in rows[1][1:4]] == pytest.approx([2, 2, 2], abs=1e-12)


@pytest.mark.parametrize('mapping', ['O .66\nSi 1.11\n', 'Si 1.11\nO .66\n'])
def test_findint_explicit_type_map_and_insertion_use_independent_chemical_symbols(tmp_path, mapping):
    source = tmp_path / 'host.xyz'; source.write_text(HOST)
    (tmp_path / 'species.map').write_text(mapping)
    result = run_command('imrmlp', arguments() + ['--index-file=species.map'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert_cubic_host_and_carbon(tmp_path / 'output.xyz')
    assert source.read_text() == HOST


@pytest.mark.parametrize('option', [[], ['--index-file', '']])
def test_findint_empty_map_option_retains_default_atomic_number_mapping(tmp_path, option):
    (tmp_path / 'host.xyz').write_text(HOST)
    result = run_command('imrmlp', arguments() + option, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert_cubic_host_and_carbon(tmp_path / 'output.xyz')


@pytest.mark.parametrize('contents', [None, '', '\n', '0\n1\n', 'Unknown\n', 'Si\nSi\n',
                                      'Si junk\n', 'Si nan\n', 'O\n\nSi\n'])
def test_findint_explicit_bad_maps_fail_before_touching_existing_output(tmp_path, contents):
    (tmp_path / 'host.xyz').write_text(HOST)
    if contents is not None:
        (tmp_path / 'bad.map').write_text(contents)
    output = tmp_path / 'output.xyz'; output.write_bytes(b'keep original output\n')
    result = run_command('imrmlp', arguments() + ['--index-file=bad.map'], tmp_path)
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'chemical species map' in result.stdout + result.stderr
    assert output.read_bytes() == b'keep original output\n'
    assert (tmp_path / 'host.xyz').read_text() == HOST


def test_findint_help_distinguishes_chemical_map_from_atom_selection(tmp_path):
    result = run_command('imrmlp', ['findint', '--help'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'chemical species map' in result.stdout + result.stderr
    assert '--near-idx' in result.stdout + result.stderr
    assert '0-based, one per line' not in result.stdout + result.stderr
