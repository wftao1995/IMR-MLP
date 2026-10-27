# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Bond-order settings, supplied Voronoi neighborhoods, and missing neighbors."""
import math
import re

import pytest

from test_cli_error_paths import run_command


def compute_bop(tmp_path, positions, settings='', options=(), periodic=False, box=20):
    (tmp_path / 'index.map').write_text('H 1\n')
    (tmp_path / 'input.ini').write_text(settings)
    (tmp_path / 'input.xyz').write_text(
        f'{len(positions)}\nLattice="{box} 0 0 0 {box} 0 0 0 {box}" '
        f'pbc="{"T T T" if periodic else "F F F"}" Properties=species:S:1:pos:R:3\n' +
        ''.join(f'H {x} {y} {z}\n' for x, y, z in positions))
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz', '--settings', 'input.ini',
                                   '--bop', '--output-xyz', 'output.xyz', '--num_threads', '1',
                                   *options], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (tmp_path / 'output.xyz').read_text().splitlines()
    schema = re.search(r'Properties=(\S+)', lines[1]).group(1).split(':')
    fields = {}
    offset = 0
    for name, kind, count in zip(schema[::3], schema[1::3], schema[2::3]):
        width = int(count)
        if name.startswith('bop'):
            fields[name] = [[float(value) for value in line.split()[offset:offset + width]]
                            for line in lines[2:]]
            assert len(fields[name]) == len(positions)
            assert all(len(row) == width for row in fields[name])
            assert all(math.isfinite(value) for row in fields[name] for value in row)
        offset += width
    return fields, result.stdout + result.stderr


@pytest.mark.parametrize('settings', ['', 'bop:nnn 1\n',
                                      'bop:corase true\n',
                                      'bop:corase true\nbop:average false\n'])
def test_bop_pair_preserves_unit_order_for_all_averages(tmp_path, settings):
    fields, _ = compute_bop(tmp_path, [(2, 2, 2), (3, 2, 2)], settings)
    for key, rows in fields.items():
        if key.startswith('bopQ'):
            for row in rows:
                assert row == pytest.approx([1, 1])


@pytest.mark.parametrize('average', [False, True])
def test_bop_coarse_handles_more_atoms_than_l_channels(tmp_path, average):
    fields, _ = compute_bop(tmp_path, [(2, 2, 2), (3, 2, 2), (3, 3, 2), (2, 3, 2)],
                            f'bop:corase true\nbop:average {str(average).lower()}\n')
    for key in ('bopQ', 'bopW'):
        for actual, expected in zip(fields[key + '_C'], fields[key]):
            assert actual == pytest.approx(expected)


@pytest.mark.parametrize('override', [False, True])
def test_bop_uses_global_cutoff_unless_explicitly_overridden(tmp_path, override):
    fields, output = compute_bop(tmp_path, [(2, 2, 2), (3, 2, 2), (3, 4, 2)],
                                 'bop:cutoff 5\n' if override else '', ['--cutoff', '1.1'])
    assert fields['bopQ'][0] == pytest.approx([math.sqrt(0.4)] * 2 if override else [1, 1])
    if not override:
        assert fields['bopQ'][2] == [0, 0]
        assert '1 atoms have no neighbors' in output


def test_bop_isolated_atom_warns_and_outputs_finite_zeros(tmp_path):
    fields, output = compute_bop(tmp_path, [(2, 2, 2)], 'bop:corase true\n')
    assert 'have no neighbors' in output
    assert 'set to zero' in output
    for rows in fields.values():
        assert rows == [[0, 0]]


@pytest.mark.parametrize('mode', ['dir', 'rad'])
def test_bop_voronoi_selection_takes_precedence_over_radius_settings(tmp_path, mode):
    fields, output = compute_bop(tmp_path, [(1, 1, 1)],
                                 f'voro_index:mode {mode}\nbop:cutoff 0.25\n',
                                 ['--voro-neighbor', '--cutoff', '0.5'], periodic=True, box=2)
    assert fields['bopQ'][0] == pytest.approx([0, math.sqrt(7 / 12)], abs=1e-12)
    assert fields['bopQ_A'][0] == pytest.approx(fields['bopQ'][0], abs=1e-12)
    assert 'have no neighbors' not in output
