# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Compare CLI entry points that share geometry kernels."""
import numpy as np
import pytest

from test_cli_error_paths import run_command


def write_xyz(path, positions, cell=(12, 13, 14), energy=None):
    lattice = np.diag(cell).ravel()
    path.write_text(
        f'{len(positions)}\nLattice="' + ' '.join(map(str, lattice)) +
        '" pbc="T T T" Properties=species:S:1:pos:R:3' +
        (f' energy={energy}' if energy is not None else '') + '\n' +
        ''.join('H ' + ' '.join(map(str, row)) + '\n' for row in positions))


def succeed(result):
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('fast', ['true', 'false'])
def test_sameq_and_dedup_agree_on_translations_and_distortions(tmp_path, fast):
    (tmp_path / 'index.map').write_text('H\n')
    original = np.array([[1, 2, 3], [2.4, 2.5, 3.2], [3.8, 4.2, 2.1]])
    shifted = original[::-1] + [3.1, -2, 1.2]
    distorted = original.copy()
    distorted[2] += [.7, -.4, .5]
    frames = []
    for i, positions in enumerate([original, shifted, distorted]):
        path = tmp_path / f'{i}.xyz'
        write_xyz(path, positions, energy=i)
        frames.append(path.read_text())
    for name, expected in [('1.xyz', 'Equivalent'), ('2.xyz', 'Not equivalent')]:
        result = run_command('imrmlp', ['sameq', '--input1', '0.xyz', '--input2', name,
                                       '--tol', '0.01', '--num_threads', '1'], tmp_path)
        succeed(result)
        assert result.stdout.splitlines()[-1] == expected
    (tmp_path / 'pool.xyz').write_text(''.join(frames))
    result = run_command('imrmlp', ['remove_duplicate_structures', '--input', 'pool.xyz',
                                   '--output', 'unique.xyz', '--output-format', 'extxyz',
                                   '--pos-tol', '0.01', '--fastalgo', fast,
                                   '--num_threads', '1'], tmp_path)
    succeed(result)
    assert (tmp_path / 'output-nodup.id').read_text().split() == ['0', '2']
    assert 'energy=0' in (tmp_path / 'unique.xyz').read_text()


def test_average_entry_points_keep_their_explicit_drift_rules(tmp_path):
    (tmp_path / 'index.map').write_text('H 0.4 2\n')
    first = np.array([[1, 2, 3], [3, 2, 3]])
    write_xyz(tmp_path / 'first.xyz', first)
    write_xyz(tmp_path / 'second.xyz', first + [.5, 0, 0])
    (tmp_path / 'trajectory.xyz').write_text(
        (tmp_path / 'first.xyz').read_text() + (tmp_path / 'second.xyz').read_text())
    succeed(run_command('imrmlp', ['calc_prop', '--input', 'trajectory.xyz', '--avgpos',
                                   '--num_threads', '1'], tmp_path))
    succeed(run_command('imrmlp', ['calc_trjavg', '--input-file', 'trajectory.xyz',
                                   '--output-file', 'centroid.xyz', '--output-format', 'extxyz',
                                   '--wrap=false', '--num_threads', '1'], tmp_path))
    mean = np.loadtxt(tmp_path / 'COMPUTE_POSAVG_RESULT.xyz', skiprows=2, usecols=(1, 2, 3))
    centered = np.loadtxt(tmp_path / 'centroid.xyz', skiprows=2, usecols=(1, 2, 3))
    np.testing.assert_allclose(mean, first + [.25, 0, 0])
    np.testing.assert_allclose(centered, first)


def test_batch_conversion_preserves_native_input_ownership(tmp_path):
    pytest.importorskip('imrmlp._imrmlp')
    from imrmlp import Atoms, Potential
    potential_file = tmp_path / 'lj.ini'
    potential_file.write_text('pot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 4\n')
    potential = Potential(potential_file, 'lj612')
    source = Atoms(numbers=[1, 1], elements=['H'], positions=[[1, 2, 3], [2.3, 2, 3]],
                   cell=np.eye(3)*20, pbc=False)
    result = potential.calculate_many([source])[0]
    assert result is not source
    assert source.energy is None
    assert np.isfinite(result.energy)
    np.testing.assert_allclose(source.positions, [[1, 2, 3], [2.3, 2, 3]])


def trajectory(positions):
    return ''.join(
        '1\nLattice="100 0 0 0 100 0 0 0 100" '
        'Properties=species:S:1:pos:R:3 pbc="F F F"\n'
        f'H {x} 0 0\n' for x in positions
    )


@pytest.mark.parametrize('coordinates, expected', [('cartesian', 2.0), ('fractional', 1.925)])
def test_trajectory_average_exposes_coordinate_choice(tmp_path, coordinates, expected):
    (tmp_path / 'index.map').write_text('H 1\n')
    first = trajectory([1]).replace('100 0 0 0 100 0 0 0 100', '10 0 0 0 10 0 0 0 10')
    second = trajectory([3]).replace('100 0 0 0 100 0 0 0 100', '12 0 0 0 12 0 0 0 12')
    (tmp_path / 'input.xyz').write_text(first + second)
    args = ['calc_trjavg', 'input.xyz', 'average.xyz', '--output-format', 'extxyz',
            '--average-coordinates', coordinates, '--drift', 'keep', '--wrap', 'false']
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    row = next(row for row in (tmp_path / 'average.xyz').read_text().splitlines() if row.startswith('H '))
    assert float(row.split()[1]) == pytest.approx(expected)


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
    batch = run_command('imrmlp', ['remove_duplicate_structures', '--input', 'pool.xyz', '--output', 'unique.xyz',
                                 '--output-format', 'extxyz', '--fastalgo', fast,
                                 '--allow-reflections', mirrors], tmp_path)
    assert batch.returncode == 0, batch.stdout + batch.stderr
    expected = ['0', '1'] if mirrors == 'false' else ['0']
    assert (tmp_path / 'output-nodup.id').read_text().split() == expected
