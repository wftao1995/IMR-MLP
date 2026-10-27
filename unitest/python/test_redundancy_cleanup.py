"""Regression coverage for shared conversion and trajectory command paths."""
import numpy as np
import pytest

from test_cli_error_paths import run_command


def trajectory(positions):
    return ''.join(
        '1\nLattice="100 0 0 0 100 0 0 0 100" '
        'Properties=species:S:1:pos:R:3 pbc="F F F"\n'
        f'H {x} 0 0\n' for x in positions
    )


def smooth(tmp_path, positions, *options):
    (tmp_path / 'index.map').write_text('H 1\n')
    (tmp_path / 'input.xyz').write_text(trajectory(positions))
    return run_command('trj_smooth', ['--input', 'input.xyz', '--output', 'output.xyz',
                       '--input-format', 'extxyz', '--output-format', 'extxyz',
                       '--window', '2', *options], tmp_path)


@pytest.mark.parametrize('stride, expected', [(1, [1, 1, 1]), (2, [1, 1]), (5, [1])])
def test_smoothing_preserves_tail_and_partial_output_stride(tmp_path, stride, expected):
    result = smooth(tmp_path, [1, 1, 1], '--output-rate', str(stride))
    assert result.returncode == 0, result.stdout + result.stderr
    rows = (tmp_path / 'output.xyz').read_text().splitlines()
    positions = [float(row.split()[1]) for row in rows if row.startswith('H ')]
    assert positions == expected


@pytest.mark.parametrize('options', [('--window', '0'), ('--window', '-1'), ('--output-rate', '0')])
def test_smoothing_rejects_invalid_window_and_stride(tmp_path, options):
    result = smooth(tmp_path, [1, 1, 1], *options)
    assert result.returncode == 1, result.stdout + result.stderr
    assert not (tmp_path / 'output.xyz').exists()


def test_smoothing_rejects_empty_input_and_write_failure(tmp_path):
    result = smooth(tmp_path, [])
    assert result.returncode == 1, result.stdout + result.stderr
    result = smooth(tmp_path, [1, 2, 3], '--output', 'missing/output.xyz')
    assert result.returncode == 1, result.stdout + result.stderr


def test_shared_atoms_conversion_preserves_copy_policy():
    imrmlp = pytest.importorskip('imrmlp')
    from imrmlp.utils import _coerce_atoms
    original = imrmlp.Atoms(numbers=[1], positions=[[1, 2, 3]])
    assert _coerce_atoms(original) is original
    copied = _coerce_atoms(original, copy_atoms=True)
    assert copied is not original
    copied.positions = [[4, 5, 6]]
    np.testing.assert_array_equal(original.positions, [[1, 2, 3]])


def test_calculate_many_keeps_input_while_omp_wrapper_uses_common_path():
    imrmlp = pytest.importorskip('imrmlp')
    from test_cli_error_paths import ROOT
    potential = imrmlp.Potential(ROOT / 'unitest/test-data/potential.mtp')
    original = imrmlp.Atoms(numbers=[1, 1], positions=[[0, 0, 0], [2.5, 0, 0]],
                          cell=np.eye(3) * 20, pbc=False)
    calculated = potential.calculate_many([original])[0]
    assert calculated is not original
    assert original.forces is None
    potential.calculate(original, use_omp=True)
    np.testing.assert_allclose(original.forces, calculated.forces, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('fast', ['true', 'false'])
def test_pair_comparison_and_batch_dedup_use_same_equivalence(tmp_path, fast):
    (tmp_path / 'index.map').write_text('H 1\nHe 1\n')
    header = ('3\nLattice="8 0 0 0 9 0 0 0 10" '
              'Properties=species:S:1:pos:R:3 pbc="T T T"\n')
    first = header + 'H 1 1 1\nHe 3 2 1\nH 2 4 5\n'
    translated = header + 'H 4 7 9\nH 3 4 5\nHe 5 5 5\n'
    different = header + 'H 1 1 1\nHe 3 2.5 1\nH 2 4 5\n'
    (tmp_path / 'first.xyz').write_text(first)
    (tmp_path / 'second.xyz').write_text(translated)
    pair = run_command('imrmlp', ['sameq', '--input1', 'first.xyz', '--input2', 'second.xyz'], tmp_path)
    assert pair.returncode == 0, pair.stdout + pair.stderr
    assert 'Equivalent' in pair.stdout
    (tmp_path / 'pool.xyz').write_text(first + translated + different)
    batch = run_command('imrmlp', ['unique', '--input', 'pool.xyz', '--output', 'unique.xyz',
                                 '--output-format', 'extxyz', '--fastalgo', fast], tmp_path)
    assert batch.returncode == 0, batch.stdout + batch.stderr
    assert (tmp_path / 'output-nodup.id').read_text().split() == ['0', '2']


def test_unique_overwrite_replaces_existing_valid_frames(tmp_path):
    imrmlp = pytest.importorskip('imrmlp')
    (tmp_path / 'index.map').write_text('H 1\nHe 1\n')
    header = ('3\nLattice="8 0 0 0 9 0 0 0 10" '
              'Properties=species:S:1:pos:R:3 pbc="T T T" ')
    first = header + 'energy=11\nH 1 1 1\nHe 3 2 1\nH 2 4 5\n'
    translated = header + 'energy=22\nH 4 7 9\nH 3 4 5\nHe 5 5 5\n'
    different = header + 'energy=33\nH 1 1 1\nHe 3 2.5 1\nH 2 4 5\n'
    source = first + translated + different
    (tmp_path / 'pool.xyz').write_text(source)
    output = tmp_path / 'unique.xyz'
    sentinel = ('1\nLattice="8 0 0 0 9 0 0 0 10" '
                'Properties=species:S:1:pos:R:3 pbc="T T T" energy=999\nHe 6 6 6\n')
    output.write_text(sentinel)
    assert imrmlp.read_one(output).energy == 999
    original = output.read_bytes()
    arguments = ['unique', '--input', 'pool.xyz', '--output', 'unique.xyz',
                 '--input-format', 'extxyz', '--output-format', 'extxyz',
                 '--fastalgo', 'true', '--preserve', 'true']
    rejected = run_command('imrmlp', arguments, tmp_path)
    assert rejected.returncode != 0, rejected.stdout + rejected.stderr
    assert '--overwrite' in rejected.stderr
    assert output.read_bytes() == original
    assert not (tmp_path / 'output-nodup.id').exists()

    for _ in range(2):
        result = run_command('imrmlp', arguments + ['--overwrite'], tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        restored = imrmlp.read(output)
        assert len(restored) == 2
        assert [atoms.energy for atoms in restored] == [11, 33]
        np.testing.assert_array_equal(restored[0].positions,
                                      [[1, 1, 1], [3, 2, 1], [2, 4, 5]])
        np.testing.assert_array_equal(restored[1].positions,
                                      [[1, 1, 1], [3, 2.5, 1], [2, 4, 5]])
        assert (tmp_path / 'output-nodup.id').read_text().split() == ['0', '2']
        assert (tmp_path / 'pool.xyz').read_text() == source




@pytest.mark.parametrize('pbc', ['F F F', 'T F F', 'T T F', 'T T T'])
@pytest.mark.parametrize('fast', ['true', 'false'])
@pytest.mark.parametrize('mirrors', ['true', 'false'])
def test_cell_aware_pair_and_batch_comparison(tmp_path, pbc, fast, mirrors):
    (tmp_path / 'index.map').write_text('H 1\nHe 1\nLi 1\nBe 1\n')
    header = ('4\nLattice="12 0 0 0 12 0 0 0 12" '
              f'Properties=species:S:1:pos:R:3 pbc="{pbc}"\n')
    first = header + 'H 1 1 1\nHe 2.2 1.4 1.8\nLi 3.4 3.1 2.5\nBe 1.7 4.3 3.8\n'
    reflected = header + 'H 1 1 -1\nHe 2.2 1.4 -1.8\nLi 3.4 3.1 -2.5\nBe 1.7 4.3 -3.8\n'
    (tmp_path / 'first.xyz').write_text(first)
    (tmp_path / 'second.xyz').write_text(reflected)
    pair = run_command('imrmlp', ['sameq', '--input1', 'first.xyz', '--input2', 'second.xyz',
                                 '--allow-reflections', mirrors], tmp_path)
    assert pair.returncode == 0, pair.stdout + pair.stderr
    assert ('Not equivalent' if mirrors == 'false' else 'Equivalent') in pair.stdout
    (tmp_path / 'pool.xyz').write_text(first + reflected)
    batch = run_command('imrmlp', ['unique', '--input', 'pool.xyz', '--output', 'unique.xyz',
                                 '--output-format', 'extxyz', '--fastalgo', fast,
                                 '--allow-reflections', mirrors], tmp_path)
    assert batch.returncode == 0, batch.stdout + batch.stderr
    expected = ['0', '1'] if mirrors == 'false' else ['0']
    assert (tmp_path / 'output-nodup.id').read_text().split() == expected
