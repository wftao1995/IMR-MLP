"""Default molecular trajectory averaging removes COM drift without padding."""
import pytest
from test_cli_error_paths import run_command


@pytest.mark.parametrize('format', ['cp2kxyz', 'extxyz'])
@pytest.mark.parametrize('drift', [None, 'remove', 'keep'], ids=['default', 'remove', 'keep'])
def test_trjavg_cellfree_unequal_mass_molecule(tmp_path, format, drift):
    # Index-map columns are radius and mass; use exact unequal masses for the oracle.
    (tmp_path / 'index.map').write_text('H 1 1\nHe 1 3\n')
    if format == 'cp2kxyz':
        header = '2\n i = 0, time = 0, E = 0\n'
    else:
        header = ('2\nLattice="0 0 0 0 0 0 0 0 0" pbc="F F F" '
                  'Properties=species:S:1:pos:R:3\n')
    text = header + 'H 1 2 3\nHe 3 4 5\n' + header + 'H 1.2 2.4 3.6\nHe 3.8 3.8 5.2\n'
    (tmp_path / 'input.xyz').write_text(text)
    args = ['calc_trjavg', 'input.xyz', 'average.xyz', '--input-format', format,
            '--output-format', 'extxyz', '--num_threads', '1']
    if drift is not None:
        args += ['--drift', drift]
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / 'average.xyz').read_text().splitlines()
    assert len(lines) == 4
    assert lines[0] == '2'
    assert 'pbc="F F F"' in lines[1]
    assert 'Properties=species:S:1:pos:R:3' in lines[1]
    expected = [[1.1, 2.2, 3.3], [3.4, 3.9, 5.1]] if drift == 'keep' else [
        [.775, 2.225, 3.15], [3.075, 3.925, 4.95]]
    for line, symbol, position in zip(lines[2:], ['H', 'He'], expected):
        row = line.split()
        assert row[0] == symbol
        assert list(map(float, row[1:])) == pytest.approx(position, abs=1e-12)
    assert (tmp_path / 'input.xyz').read_text() == text
