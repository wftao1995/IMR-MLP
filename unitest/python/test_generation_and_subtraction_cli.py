"""Generation constraints and standalone filtering/subtraction failures."""
from pathlib import Path

import pytest

from test_cli_error_paths import run_command


def frame(energy=10, forces=True, stress=True, species=('H', 'H')):
    header = f'{len(species)}\nLattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3'
    if forces:
        header += ':forces:R:3'
    if energy is not None:
        header += f' energy={energy}'
    if stress:
        header += ' virial="1 0 0 0 2 0 0 0 3"'
    rows = [f'{symbol} {1+2*i} 1 1' + (' 1 2 3' if forces else '')
            for i, symbol in enumerate(species)]
    return header + '\n' + '\n'.join(rows) + '\n'


@pytest.fixture
def inputs(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\nHe 0\n')
    (tmp_path / 'first.xyz').write_text(frame())
    (tmp_path / 'second.xyz').write_text(frame(energy=3))


@pytest.mark.parametrize('failure', ['missing_input', 'empty_input', 'bad_format',
                                    'missing_directory', 'full_device'])
def test_mindist_failures_return_nonzero(tmp_path, inputs, failure):
    args = ['--input', 'first.xyz', '--output', 'selected.xyz', '--mindist', '1']
    if failure == 'missing_input':
        (tmp_path / 'first.xyz').unlink()
    elif failure == 'empty_input':
        (tmp_path / 'first.xyz').write_text('')
    elif failure == 'bad_format':
        args += ['--input-format', 'not_a_format']
    elif failure == 'missing_directory':
        args[3] = 'missing/selected.xyz'
    else:
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        args[3] = '/dev/full'
        args += ['--output-format', 'extxyz']
    result = run_command('select_by_mindist', args, tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr


def test_mindist_success_including_empty_selection(tmp_path, inputs):
    args = ['--input', 'first.xyz', '--output', 'selected.xyz', '--mindist', '1']
    result = run_command('select_by_mindist', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'selected.xyz').read_text().startswith('2\n')
    args[3] = 'none.xyz'
    args[-1] = '-1'
    result = run_command('select_by_mindist', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('failure', ['energy', 'forces', 'stress', 'species', 'atoms', 'frames'])
def test_subtract_rejects_incompatible_labels_and_structures(tmp_path, inputs, failure):
    kwargs = {'energy': 3}
    if failure == 'energy': kwargs['energy'] = None
    elif failure == 'forces': kwargs['forces'] = False
    elif failure == 'stress': kwargs['stress'] = False
    elif failure == 'species': kwargs['species'] = ('H', 'He')
    elif failure == 'atoms': kwargs['species'] = ('H',)
    second = frame(**kwargs)
    if failure == 'frames': second += second
    (tmp_path / 'second.xyz').write_text(second)
    result = run_command('subtract_pot', ['--input1', 'first.xyz', '--input2', 'second.xyz',
                                         '--output', 'difference.xyz'], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert not (tmp_path / 'difference.xyz').exists()


@pytest.mark.parametrize('energy', [True, False])
def test_subtract_accepts_matching_available_labels(tmp_path, inputs, energy):
    (tmp_path / 'first.xyz').write_text(frame(energy=10 if energy else None))
    (tmp_path / 'second.xyz').write_text(frame(energy=3 if energy else None))
    result = run_command('subtract_pot', ['--input1', 'first.xyz', '--input2', 'second.xyz',
                                         '--output', 'difference.xyz'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    output = (tmp_path / 'difference.xyz').read_text()
    assert ('energy=7' in output) if energy else ('energy=' not in output)
    assert 'H 1 1 1 0 0 0' in output




@pytest.mark.parametrize('threshold,expected', [(5, [7, 8, 9]), (8, [9]), (-8, [7])])
def test_mindist_selection_uses_free_distance_and_strict_threshold(tmp_path, threshold, expected):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.xyz').write_text(''.join(
        '2\nLattice="10 0 0 0 10 0 0 0 10" pbc="F F F" '
        f'Properties=species:S:1:pos:R:3\nH 0 0 0\nH {distance} 0 0\n'
        for distance in (7, 8, 9)))
    result = run_command('select_by_mindist', [
        '--input', 'input.xyz', '--input-format', 'extxyz', '--output', 'selected.xyz',
        '--output-format', 'extxyz', '--mindist', str(threshold)], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = (tmp_path / 'selected.xyz').read_text().splitlines()
    assert len(rows) == 4 * len(expected)
    assert [int(rows[i]) for i in range(0, len(rows), 4)] == [2] * len(expected)
    assert [float(rows[i + 3].split()[1]) for i in range(0, len(rows), 4)] == expected
