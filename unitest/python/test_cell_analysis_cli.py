# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Cell command failures and trajectory analysis output regressions."""
from pathlib import Path
import re

import numpy as np
import pytest

from test_cli_error_paths import run_command


CELL_COMMANDS = ['mk_orth', 'mk_orth_search', 'trans_reduce', 'primi_reduce',
                 'mk_lower', 'mk_super', 'swap_axis', 'find_mindist']


@pytest.fixture
def cell_input(tmp_path):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.xyz').write_text(
        '2\nLattice="10 0 0 0 10 0 0 0 10" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        'H 1 1 1\nH 3 1 1\n')
    return ['--input', 'input.xyz', '--input-format', 'extxyz', '--num_threads', '1']


def cell_args(command, inputs):
    args = [command, *inputs, '--output', 'output.xyz', '--output-format', 'extxyz']
    if command == 'mk_super':
        args += ['--supercell', '1', '1', '1']
    elif command == 'swap_axis':
        args += ['--order', 'yzx']
    elif command in ('mk_orth', 'mk_orth_search'):
        args += ['--max-expand', '2']
    return args


@pytest.mark.parametrize('avgpos', [False, True])
@pytest.mark.parametrize('full_device', [False, True])
def test_calc_prop_reports_failed_structure_outputs(tmp_path, cell_input, avgpos, full_device):
    target = tmp_path / ('COMPUTE_POSAVG_RESULT.xyz' if avgpos else 'output.xyz')
    if full_device:
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        target.symlink_to('/dev/full')
    else:
        target.mkdir()
    args = ['--avgpos'] if avgpos else ['--output-xyz', str(target)]
    result = run_command('imrmlp', ['calc_prop', *cell_input, *args], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Complete!' not in result.stdout
    assert result.stderr


def test_calc_prop_average_positions_preserve_cell(tmp_path, cell_input):
    result = run_command('imrmlp', ['calc_prop', *cell_input, '--avgpos',
                                   '--output-xyz', 'copy.xyz'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Complete!' in result.stdout
    output = (tmp_path / 'COMPUTE_POSAVG_RESULT.xyz').read_text()
    cell = re.search(r'Lattice="([^"]+)"', output).group(1)
    assert [float(value) for value in cell.split()] == [10, 0, 0, 0, 10, 0, 0, 0, 10]
    assert (tmp_path / 'copy.xyz').stat().st_size > 0


@pytest.mark.parametrize('periodic', [False, True])
def test_calc_prop_average_positions_across_cell_boundary(tmp_path, periodic):
    (tmp_path / 'index.map').write_text('H 1\n')
    header = ('1\nLattice="10 0 0 0 10 0 0 0 10" '
              f'pbc="{"T" if periodic else "F"} F F" Properties=species:S:1:pos:R:3\n')
    (tmp_path / 'input.xyz').write_text(header + 'H 9.8 0 0\n' + header + 'H 0.2 0 0\n')
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--avgpos',
                                   '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    row = (tmp_path / 'COMPUTE_POSAVG_RESULT.xyz').read_text().splitlines()[2].split()
    assert float(row[1]) == pytest.approx(10 if periodic else 5)


@pytest.mark.parametrize('tetrahedron', [False, True])
def test_calc_prop_cna_handles_common_neighbor_graphs(tmp_path, tetrahedron):
    (tmp_path / 'index.map').write_text('H 1\n')
    rows = ['H 2 2 2', 'H 3 2 2', 'H 2.5 2.866025403784 2']
    if tetrahedron:
        rows.append('H 2.5 2.288675134595 2.816496580928')
    (tmp_path / 'input.xyz').write_text(
        f'{len(rows)}\nLattice="20 0 0 0 20 0 0 0 20" pbc="F F F" '
        'Properties=species:S:1:pos:R:3\n' + '\n'.join(rows) + '\n')
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--c-cna',
                                   '--cutoff', '1.1', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Complete!' in result.stdout


def read_numeric_rows(path):
    return [[float(value) for value in line.split()]
            for line in path.read_text().splitlines() if line and not line.startswith('#')]


@pytest.mark.parametrize('reverse', [False, True])
def test_dihedral_chemistry_follows_chain_reversal(tmp_path, reverse):
    (tmp_path / 'index.map').write_text('H 1\nHe 1\nLi 1\nBe 1\n')
    rows = ['H 2.3 3.7 3', 'Li 3 3 3', 'He 4 3 3', 'Be 4.7 3.7 3']
    if reverse:
        rows.reverse()
    (tmp_path / 'input.xyz').write_text(
        '4\nLattice="10 0 0 0 10 0 0 0 10" pbc="F F F" '
        'Properties=species:S:1:pos:R:3\n' + '\n'.join(rows) + '\n')
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--dihedral',
                                   '--cutoff', '1.1', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    histogram = read_numeric_rows(tmp_path / 'analyse_dihedralanglesBeHeLiH.txt')
    assert sum(row[1] for row in histogram) == pytest.approx(100)


@pytest.mark.parametrize('mode', ['rad', 'dir'])
@pytest.mark.parametrize('ha', [False, True])
def test_voronoi_neighbors_keep_opposite_images_and_single_ownership(tmp_path, mode, ha):
    (tmp_path / 'index.map').write_text('H 1\n')
    (tmp_path / 'input.ini').write_text(f'voro_index:mode {mode}\n')
    (tmp_path / 'input.xyz').write_text(
        '8\nLattice="4 0 0 0 4 0 0 0 4" Properties=species:S:1:pos:R:3\n' +
        ''.join(f'H {x} {y} {z}\n' for x in (1, 3) for y in (1, 3) for z in (1, 3)))
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--voro-neighbor',
                                   '--settings', 'input.ini', '--bondangle', '--num_threads', '1',
                                   *(['--ha-index'] if ha else [])], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    histogram = read_numeric_rows(tmp_path / 'analyse_bondanglesHHH.txt')
    assert histogram[90][1] == pytest.approx(80)
    assert histogram[180][1] == pytest.approx(20)
    voro = (tmp_path / 'voro_out_total.txt').read_text()
    counts = [int(line.split()[1]) for line in voro.splitlines() if line.startswith('0-')]
    assert sum(counts) == 8


def test_sro_empty_frame_is_reported_without_losing_following_frame(tmp_path):
    (tmp_path / 'index.map').write_text('H 1\n')
    header = '2\nLattice="10 0 0 0 10 0 0 0 10" pbc="F F F" Properties=species:S:1:pos:R:3\n'
    (tmp_path / 'input.xyz').write_text(header + 'H 1 1 1\nH 5 1 1\n' +
                                       header + 'H 1 1 1\nH 1.5 1 1\n')
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--sro',
                                   '--cutoff', '1', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'No SRO result for configuration 1' in result.stdout + result.stderr
    output = (tmp_path / 'output_sro.txt').read_text()
    assert 'Configuration 1  unavailable' in output
    assert 'Configuration 2   1nn:' in output


def test_sro_shells_do_not_depend_on_the_first_atom(tmp_path):
    (tmp_path / 'index.map').write_text('H 1\nHe 1\n')
    rows = ['H 1 1 1', 'H 2 1 1', 'He 6 1 1', 'He 8 1 1']
    outputs = []
    for order in (rows, rows[2:] + rows[:2]):
        (tmp_path / 'input.xyz').write_text(
            '4\nLattice="20 0 0 0 20 0 0 0 20" pbc="F F F" Properties=species:S:1:pos:R:3\n' +
            '\n'.join(order) + '\n')
        result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--sro',
                                       '--cutoff', '2.1', '--num_threads', '1'], tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(((tmp_path / 'output_sro.txt').read_text(),
                        (tmp_path / 'output_neighbor.txt').read_text()))
    assert outputs[0] == outputs[1]
    assert '1nn:' in outputs[0][0] and '2nn:' in outputs[0][0]


@pytest.mark.parametrize('mode', ['dir', 'rad'])
def test_voronoi_only_options_compute_the_same_statistics(tmp_path, mode):
    (tmp_path / 'index.map').write_text('H 1\n')
    (tmp_path / 'input.ini').write_text(f'voro_index:mode {mode}\n')
    (tmp_path / 'input.xyz').write_text(
        '8\nLattice="4 0 0 0 4 0 0 0 4" Properties=species:S:1:pos:R:3\n' +
        ''.join(f'H {x} {y} {z}\n' for x in (1, 3) for y in (1, 3) for z in (1, 3)))
    outputs = []
    for extra in ([], ['--voro-neighbor']):
        result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--settings', 'input.ini',
                                       '--voro-index', '--num_threads', '1', *extra], tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append((tmp_path / 'voro_out_total.txt').read_text())
    assert outputs[0] == outputs[1]
    assert 'nan' not in outputs[1].lower()
    assert sum(int(line.split()[1]) for line in outputs[1].splitlines() if line.startswith('0-')) == 8


def test_ha_pair_cutoff_matches_short_neighbor_graph(tmp_path):
    (tmp_path / 'index.map').write_text('H 1\n')
    (tmp_path / 'input.ini').write_text('pair_cutoff:H-H 1.1\n')
    (tmp_path / 'input.xyz').write_text(
        '4\nLattice="10 0 0 0 10 0 0 0 10" pbc="F F F" '
        'Properties=species:S:1:pos:R:3\nH 2 2 2\nH 3 2 2\nH 3 3 2\nH 2 3 2\n')
    outputs = []
    for options in (['--cutoff', '1.1'], ['--cutoff', '1.5', '--pair-cutoff']):
        result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--ha-index',
                                       '--settings', 'input.ini', '--num_threads', '1',
                                       '--output-all-indices', *options], tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append((tmp_path / 'out_total.txt').read_text())
    assert outputs[0] == outputs[1]
    counts = {line.split()[0]: int(line.split()[1]) for line in outputs[1].splitlines()
              if line.split() and line.split()[0].isdigit()}
    assert counts['1001'] == 4


@pytest.mark.parametrize('kind,positions', [
    ('bond', 'H 2 2 2\nH 3 2 2\nH 4 2 2\n'),
    ('dihedral', 'H 2.3 3.7 3\nH 3 3 3\nH 4 3 3\nH 4.7 3.7 3\n'),
])
def test_angle_histogram_defaults_include_180_degrees(tmp_path, kind, positions):
    (tmp_path / 'index.map').write_text('H\n')
    natoms = len(positions.splitlines())
    (tmp_path / 'input.xyz').write_text(
        f'{natoms}\nLattice="10 0 0 0 10 0 0 0 10" pbc="F F F" '
        'Properties=species:S:1:pos:R:3\n' + positions)
    option = '--bondangle' if kind == 'bond' else '--dihedral'
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', option,
                                   '--cutoff', '1.1', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = read_numeric_rows(tmp_path / f'analyse_{kind}angles{"H" * natoms}.txt')
    assert [row[0] for row in rows] == pytest.approx(list(range(181)))
    assert sum(row[1] for row in rows) == pytest.approx(100)
    assert rows[-1][1] == pytest.approx(100)


def test_sro_thread_counts_preserve_frame_order(tmp_path):
    (tmp_path / 'index.map').write_text('H\nHe\n')
    header = ('Lattice="20 0 0 0 20 0 0 0 20" pbc="F F F" '
              'Properties=species:S:1:pos:R:3\n')
    # The larger first frame finishes later. Its neighbors are unlike species;
    # the second frame contains two pairs of like species.
    large = [f'{"H" if (x+y+z) % 2 == 0 else "He"} {2+x} {2+y} {2+z}\n'
             for x in range(12) for y in range(12) for z in range(12)]
    small = 'H 2 2 2\nH 3 2 2\nHe 6 2 2\nHe 7 2 2\n'
    (tmp_path / 'input.xyz').write_text(
        f'{len(large)}\n' + header + ''.join(large) + '4\n' + header + small)
    outputs = []
    for threads in (1, 2, 4):
        result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--sro',
                                       '--cutoff', '1.1', '--num_threads', str(threads)], tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        sro = (tmp_path / 'output_sro.txt').read_text()
        neighbors = (tmp_path / 'output_neighbor.txt').read_text()
        rows = [list(map(float, line.split('1nn:')[1].split()))
                for line in sro.splitlines() if line.startswith('Configuration')]
        assert rows == [[1, -1, 1], [-1, 1, -1]]
        outputs.append((sro, neighbors))
    assert outputs[0] == outputs[1] == outputs[2]


@pytest.mark.parametrize('species', [('H',) * 4, ('H', 'He', 'Li', 'Be')])
@pytest.mark.parametrize('threads', [1, 2])
def test_calc_prop_dihedral_without_bondangle(tmp_path, species, threads):
    (tmp_path / 'index.map').write_text('\n'.join(dict.fromkeys(species)) + '\n')
    positions = ('2 2 2', '3 2 2', '3 3 2', '3 3 3')
    (tmp_path / 'input.xyz').write_text(
        '4\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3\n' +
        ''.join(f'{symbol} {position}\n' for symbol, position in zip(species, positions)))
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz',
                         '--input-format', 'extxyz', '--dihedral', '--cutoff', '1.1',
                         '--num_threads', str(threads)], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = read_numeric_rows(tmp_path / f'analyse_dihedralangles{"".join(species)}.txt')
    assert sum(row[1] for row in rows) == pytest.approx(100)
    peak = max(rows, key=lambda row: row[1])
    assert peak[1] == pytest.approx(100)
    assert abs(peak[0] - 90) <= rows[1][0] - rows[0][0]


@pytest.mark.parametrize('index, masses', [
    ('H\nHe\n', (1.008, 4.002602)),
    ('H 1.0\nHe 1.5\n', (1.008, 4.002602)),
    ('H 1.0 2.0\nHe 1.5 3.0\n', (2.0, 3.0)),
])
def test_calc_prop_mass_fallback_and_override(tmp_path, index, masses):
    (tmp_path / 'index.map').write_text(index)
    # Only H moves: the drift-corrected MSD depends on the mass ratio.
    (tmp_path / 'input.xyz').write_text(''.join(
        '2\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3\n'
        f'H {x} 1 1\nHe 5 1 1\n' for x in (1, 2)))
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz',
                         '--input-format', 'extxyz', '--msd', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = read_numeric_rows(tmp_path / 'MSD_TOTAL.dat')
    assert rows[0][1] == pytest.approx(0, abs=1e-12)
    expected = sum(m * m for m in masses) / (2 * sum(masses) ** 2)
    assert rows[1][1] == pytest.approx(expected)


def test_calc_prop_rdf_average_density_applies_to_total_and_partial(tmp_path, cell_input):
    path = tmp_path / 'input.xyz'
    frame = path.read_text()
    path.write_text(frame + frame.replace('10 0 0 0 10 0 0 0 10', '20 0 0 0 20 0 0 0 20'))
    curves = []
    for extra in ([], ['--rdf-rho-all']):
        result = run_command('imrmlp', ['calc_prop', *cell_input, '--rdf',
                             '--ref-rlimit', '4', *extra], tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        total = read_numeric_rows(tmp_path / 'gr_total.txt')
        partial = read_numeric_rows(tmp_path / 'gr_partial_H-H')
        assert [row[1] for row in total] == pytest.approx([row[1] for row in partial])
        curves.append(total)
    occupied = [i for i, row in enumerate(curves[0]) if row[1] > 0]
    assert occupied
    for i in occupied:
        # rho_first / rho_all = (2/1000) / (4/9000) = 4.5.
        assert curves[1][i][1] == pytest.approx(curves[0][i][1] * 4.5, rel=1e-5)
        assert curves[1][i][2] == curves[0][i][2]


@pytest.mark.parametrize('cell,pbc,positions,expected', [
    ('100 0 0 0 100 0 0 0 100', 'T T T',
     'H 20 50 50\nH 80 50 50\n', 40),
    ('10 0 0 0 10 0 0 0 10', 'F F F',
     'H 1 1 1\nH 1 1 1\nH 3 1 1\n', 0),
    ('4 0 0 3 2 0 0 0 10', 'T T T', 'H 0 0 0\n', np.sqrt(5)),
    ('4 0 0 3 2 0 0 0 10', 'T T T',
     'H 0 0 0\nH 3.43 .98 0\n', np.sqrt(1.2253)),
    ('10 0 0 0 10 0 0 0 10', 'T F F',
     'H 0 1 0\nH 0 9 0\n', 8),
    ('10 0 0 0 10 0 0 0 10', 'T T T',
     'H 0 0 0\nH 10 0 0\n', 0),
])
def test_find_mindist_reports_physical_distances(tmp_path, cell, pbc, positions, expected):
    (tmp_path / 'index.map').write_text('H 0\n')
    (tmp_path / 'input.xyz').write_text(
        f'{len(positions.splitlines())}\nLattice="{cell}" pbc="{pbc}" '
        'Properties=species:S:1:pos:R:3\n' + positions)
    result = run_command('imrmlp', ['find_mindist', '--input', 'input.xyz',
                         '--input-format', 'extxyz', '--output', 'distances.txt',
                         '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    values = [float(line) for line in (tmp_path / 'distances.txt').read_text().splitlines()]
    assert values == pytest.approx([expected, expected])
