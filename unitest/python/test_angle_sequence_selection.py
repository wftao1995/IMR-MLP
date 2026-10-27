"""Selected chemical sequences retain existing per-combination angle percentages."""
import pytest

from test_cli_error_paths import run_command


def execute(tmp_path, name, frames, species, option, selection='', bounds='', threads=1):
    directory = tmp_path/name
    directory.mkdir()
    (directory/'index.map').write_text(''.join(f'{symbol} 1\n' for symbol in species))
    text = ''
    for rows in frames:
        text += f'{len(rows)}\nLattice="10 0 0 0 10 0 0 0 10" pbc="F F F" Properties=species:S:1:pos:R:3\n'
        text += ''.join(f'{symbol} {x} {y} {z}\n' for symbol, x, y, z in rows)
    (directory/'input.xyz').write_text(text)
    (directory/'angles.ini').write_text(selection+bounds)
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--input-format', 'extxyz',
        '--settings', 'angles.ini', '--cutoff', '1.1', '--'+option, '--num_threads', str(threads)], directory)
    assert result.returncode == 0, result.stdout+result.stderr
    return directory


def percentages(path):
    return [tuple(map(float, line.split())) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('threads', [1, 2])
def test_bond_sequence_selection_preserves_pooled_counts_and_empty_combinations(tmp_path, threads):
    first = [('H',3,3,3), ('O',4,3,3), ('O',3,4,3)]
    second = first + [('O',2,3,3)]
    baseline = execute(tmp_path, 'all', [first, second], ['H','O'], 'bondangle', threads=threads)
    selected = execute(tmp_path, 'selected', [first, second], ['H','O'], 'bondangle',
        'bondangle:triplets O H O; H H H\n', threads=threads)
    assert {p.name for p in selected.glob('analyse_bondangles*.txt')} == {
        'analyse_bondanglesOHO.txt', 'analyse_bondanglesHHH.txt'}
    for name in ('OHO', 'HHH'):
        filename = f'analyse_bondangles{name}.txt'
        assert (selected/filename).read_text() == (baseline/filename).read_text()
    rows = percentages(selected/'analyse_bondanglesOHO.txt')
    assert rows[90] == pytest.approx((90.,75.))
    assert rows[180] == pytest.approx((180.,25.))
    assert sum(value for _, value in percentages(selected/'analyse_bondanglesHHH.txt')) == 0
    bounded = execute(tmp_path, 'bounded', [first, second], ['H','O'], 'bondangle',
        'bondangle:triplets O H O\n', 'bondangle:high 1.5707963267948966\n', threads)
    assert percentages(bounded/'analyse_bondanglesOHO.txt')[90] == pytest.approx((90.,100.))


@pytest.mark.parametrize('reverse', [False, True])
def test_dihedral_sequence_reversal_keeps_the_existing_ninety_degree_histogram(tmp_path, reverse):
    chain = [('Be',2.3,3.7,3), ('He',3,3,3), ('Li',4,3,3), ('H',4.7,3,3.7)]
    baseline = execute(tmp_path, 'all', [chain], ['H','He','Li','Be'], 'dihedral')
    sequence = 'H Li He Be' if reverse else 'Be He Li H'
    selected = execute(tmp_path, 'selected', [chain], ['H','He','Li','Be'], 'dihedral',
        f'bondangle:dihedrals {sequence}; Be He Li H; H H H H\n')
    assert {p.name for p in selected.glob('analyse_dihedralangles*.txt')} == {
        'analyse_dihedralanglesBeHeLiH.txt', 'analyse_dihedralanglesHHHH.txt'}
    filename = 'analyse_dihedralanglesBeHeLiH.txt'
    assert (selected/filename).read_text() == (baseline/filename).read_text()
    assert percentages(selected/filename)[90] == pytest.approx((90.,100.))
    assert sum(value for _, value in percentages(selected/'analyse_dihedralanglesHHHH.txt')) == 0


@pytest.mark.parametrize('sequence', ['H H', 'H Xe H'])
def test_invalid_sequence_fails_before_replacing_existing_histograms(tmp_path, sequence):
    (tmp_path/'index.map').write_text('H 1\n')
    (tmp_path/'input.xyz').write_text(
        '1\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3\nH 1 1 1\n')
    (tmp_path/'angles.ini').write_text(f'bondangle:triplets {sequence}\n')
    output = tmp_path/'analyse_bondanglesHHH.txt'
    output.write_text('previous histogram\n')
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--input-format', 'extxyz',
        '--settings', 'angles.ini', '--bondangle', '--num_threads', '1'], tmp_path)
    assert result.returncode == 1
    assert 'mapped element symbols' in result.stdout+result.stderr
    assert output.read_text() == 'previous histogram\n'
