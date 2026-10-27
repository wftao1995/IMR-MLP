"""Average-position output is geometry, and the annotated trajectory keeps labels."""
import re
import numpy as np
import pytest

from test_cli_error_paths import run_command


@pytest.mark.parametrize('extra', [[], ['--avgpos-fractional']], ids=['cartesian', 'fractional'])
def test_avgpos_writes_only_mean_geometry_and_preserves_trajectory_labels(tmp_path, extra):
    (tmp_path / 'index.map').write_text('H 0\n')
    header = ('Lattice="10 0 0 0 10 0 0 0 10" pbc="T T T" '
              'Properties=species:S:1:pos:R:3:forces:R:3:velocities:R:3:site_en:R:1 '
              'energy={energy} fenergy={free_energy} virial="1 0 0 0 2 0 0 0 3"')
    text = ('1\n' + header.format(energy=-1, free_energy=-2) +
            '\nH 1 2 3 .1 .2 .3 4 5 6 -1\n1\n' +
            header.format(energy=-3, free_energy=-4) +
            '\nH 3 4 5 .4 .5 .6 7 8 9 -3\n')
    (tmp_path / 'input.xyz').write_text(text)
    result = run_command('imrmlp', ['calc_prop', '--input', 'input.xyz',
                                   '--input-format', 'extxyz', '--avgpos', *extra,
                                   '--output-xyz', 'copy.xyz', '--num_threads', '1'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    average = (tmp_path / 'COMPUTE_POSAVG_RESULT.xyz').read_text().splitlines()
    assert average[0] == '1'
    properties = re.search(r'Properties=(\S+)', average[1]).group(1)
    assert properties == 'species:S:1:pos:R:3'
    for label in ('energy', 'fenergy', 'virial', 'stress'):
        assert not re.search(rf'\b{label}=', average[1])
    np.testing.assert_allclose(list(map(float, average[2].split()[1:])), [2., 3., 4.])
    assert 'Lattice="10 0 0 0 10 0 0 0 10"' in average[1]
    assert 'pbc="T T T"' in average[1]

    copied = (tmp_path / 'copy.xyz').read_text().splitlines()
    assert len(copied) == 6
    for offset, energy, free_energy, expected in [
        (0, -1, -2, [1, 2, 3, .1, .2, .3, 4, 5, 6, -1]),
        (3, -3, -4, [3, 4, 5, .4, .5, .6, 7, 8, 9, -3]),
    ]:
        assert copied[offset] == '1'
        for label, value in [('energy', energy), ('fenergy', free_energy)]:
            actual = float(re.search(rf'\b{label}=(\S+)', copied[offset + 1]).group(1))
            assert actual == value
        assert 'virial="1 0 0 0 2 0 0 0 3"' in copied[offset + 1]
        assert 'Properties=species:S:1:pos:R:3:forces:R:3:velocities:R:3:site_en:R:1' in copied[offset + 1]
        np.testing.assert_allclose(list(map(float, copied[offset + 2].split()[1:])), expected)
    assert (tmp_path / 'input.xyz').read_text() == text
