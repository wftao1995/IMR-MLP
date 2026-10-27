"""Cartesian trajectory means need only the periodic slab lattice rows."""
import pytest
from test_cli_error_paths import run_command


@pytest.mark.parametrize('nonprefix', [False, True], ids=['TTF', 'TFT'])
@pytest.mark.parametrize('drift', [None, 'remove', 'keep'], ids=['default', 'remove', 'keep'])
def test_slab_average_crossing_preserves_normal_motion_and_matches_padding(tmp_path, nonprefix, drift):
    expected = [[10., 1.1, 3.2], [2.4, 2.3, 4.4]] if drift == 'keep' else [
        [9.7, .9, 2.9], [2.1, 2.1, 4.1]]
    outputs = []
    for padded in [False, True]:
        directory = tmp_path / ('padded' if padded else 'partial'); directory.mkdir()
        (directory / 'index.map').write_text('C 1 12\n')
        cell = ([10, 0, 0, 0, 0, 20 if padded else 0, 0, 10, 0] if nonprefix
                else [10, 0, 0, 0, 10, 0, 0, 0, 20 if padded else 0])
        pbc = 'T F T' if nonprefix else 'T T F'
        header = ('2\nLattice="' + ' '.join(map(str, cell)) + '" pbc="' + pbc +
                  '" Properties=species:S:1:pos:R:3\n')
        text = header + 'C 9.8 1 3\nC 2 2 4\n' + header + 'C .2 1.2 3.4\nC 2.8 2.6 4.8\n'
        (directory / 'input.xyz').write_text(text)
        args = ['calc_trjavg', 'input.xyz', 'average.xyz', '--input-format', 'extxyz',
                '--output-format', 'extxyz', '--wrap', 'false', '--num_threads', '1']
        if drift is not None:
            args += ['--drift', drift]
        result = run_command('imrmlp', args, directory)
        assert result.returncode == 0, result.stdout + result.stderr
        lines = (directory / 'average.xyz').read_text().splitlines()
        assert len(lines) == 4 and lines[0] == '2'
        assert f'pbc="{pbc}"' in lines[1]
        assert 'Lattice="' + ' '.join(map(str, cell)) + '"' in lines[1]
        actual = []
        for line, position in zip(lines[2:], expected):
            row = line.split(); assert row[0] == 'C'
            actual.append(list(map(float, row[1:])))
            assert actual[-1] == pytest.approx(position, abs=1e-11)
        outputs.append(actual)
        assert (directory / 'input.xyz').read_text() == text
    for partial, padded in zip(*outputs):
        assert partial == pytest.approx(padded, abs=1e-11)
