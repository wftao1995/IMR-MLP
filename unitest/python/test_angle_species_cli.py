"""Real calc_prop processes preserve selected-key histograms on an image-rich graph.

CLI files expose per-key percentages, not absolute combination counts. Native
angle tests cover raw counts; these tests compare every observable bin exactly.
"""
import math

import pytest

from test_cli_error_paths import run_command


def execute(directory, kind, selection='', pair_cutoff=False):
    directory.mkdir()
    (directory/'index.map').write_text('H 1\nO 1\nHe 1\n')
    # Short skew periodic x/y vectors give several image slots per source ID;
    # nonperiodic z and unequal heights also exercise nonplanar dihedrals.
    frames = []
    for shift in (0., .06):
        frames.append(
            '3\nLattice="1.6 0 0 0.35 1.7 0 0.2 0.1 4.8" pbc="T T F" '
            'Properties=species:S:1:pos:R:3\n'
            'H 0.1 0.1 0.2\n'
            f'O {0.8+shift:.8f} 0.6 0.8\n'
            'He 1.1 0.9 0.3\n')
    (directory/'input.xyz').write_text(''.join(frames))
    # H-H equality at |a|=1.6 must be retained by the pair-cutoff path.
    cutoffs = ('pair_cutoff:H-H 1.6\npair_cutoff:H-O 1.65\n'
               'pair_cutoff:H-He 1.5\npair_cutoff:O-O 1.8\n'
               'pair_cutoff:O-He 1.7\npair_cutoff:He-He 1.9\n')
    (directory/'angles.ini').write_text(selection + cutoffs)
    options = ['calc_prop', '--input', 'input.xyz', '--input-format', 'extxyz',
               '--settings', 'angles.ini', '--cutoff', '2.1', '--'+kind,
               '--num_threads', '1']
    if pair_cutoff:
        options.append('--pair-cutoff')
    result = run_command('imrmlp', options, directory)
    assert result.returncode == 0, result.stdout + result.stderr
    return directory


def assert_histograms(direct, selected, prefix, names, empty=()):
    expected = {prefix+name+'.txt' for name in names}
    assert {path.name for path in selected.glob(prefix+'*.txt')} == expected
    for filename in expected:
        # Compare all grid points and percentage bins, including empty bins.
        actual = (selected/filename).read_text()
        assert actual == (direct/filename).read_text()
        rows = [tuple(map(float, line.split())) for line in actual.splitlines()]
        assert len(rows) == 181
        assert [angle for angle, _ in rows] == list(range(181))
        assert all(math.isfinite(value) and value >= 0 for _, value in rows)
        if filename in {prefix+name+'.txt' for name in empty}:
            assert all(value == 0 for _, value in rows)
        else:
            assert sum(value > 0 for _, value in rows) > 1
            # Writer precision is six significant digits, so the sum has rounding.
            assert sum(value for _, value in rows) == pytest.approx(100., abs=.001)


@pytest.mark.parametrize('pair_cutoff', [False, True])
def test_bond_species_cli_matches_direct_skew_partial_pbc_bins(tmp_path, pair_cutoff):
    direct = execute(tmp_path/'direct', 'bondangle', pair_cutoff=pair_cutoff)
    selected = execute(tmp_path/'selected', 'bondangle',
                       'bondangle:triplets O H O; O H H; H H O; H O H\n', pair_cutoff)
    # Reversed mixed endpoints select one canonical H-H-O key exactly once.
    assert_histograms(direct, selected, 'analyse_bondangles', ['OHO', 'HHO', 'HOH'])


@pytest.mark.parametrize('pair_cutoff', [False, True])
def test_dihedral_species_cli_matches_direct_skew_partial_pbc_bins(tmp_path, pair_cutoff):
    direct = execute(tmp_path/'direct', 'dihedral', pair_cutoff=pair_cutoff)
    selected = execute(tmp_path/'selected', 'dihedral',
                       'bondangle:dihedrals O H O H; H O H O; O H H H; H H H O; H O O O\n',
                       pair_cutoff)
    # Whole-chain reverse duplicates and equal-center reversed endpoints must
    # preserve the direct image ownership and translated-endpoint exclusions.
    # With the H-H pair radius at |a|, its only H endpoints are collinear;
    # that selected combination has a finite all-zero histogram.
    assert_histograms(direct, selected, 'analyse_dihedralangles',
                      ['OHOH', 'HHHO', 'HOOO'], empty=['HHHO'] if pair_cutoff else [])
