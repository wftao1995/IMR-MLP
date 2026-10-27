"""Automatic variable-cell scaling through the real V2/LJ command."""
import numpy as np
import pytest
from test_cli_error_paths import run_command
from test_static_relax_metrics import prepare_metric


@pytest.mark.parametrize('rank', [0, 1, 2, 3])
@pytest.mark.parametrize('recast', [False, True])
def test_cli_rotated_and_singular_cell_scaling(tmp_path, rank, recast):
    prepare_metric(tmp_path, {'isif': 3, 'relax:driver': 'lbfgs',
                             'relax:cartesian': 'false',
                             'relax:atomic_force_recast': str(recast),
                             'relax:periodic_subspace': str(rank < 3),
                             'relax:maxiteration': 2})
    cell = np.array([[4., -4., 0.], [4., 4., 0.], [0., 0., 5.]])
    cell[rank:] = 0.
    pbc = ' '.join('T' if i < rank else 'F' for i in range(3))
    (tmp_path / 'input.xyz').write_text(
        '2\nLattice="' + ' '.join(map(str, cell.flat)) +
        '" pbc="' + pbc + '" Properties=species:S:1:pos:R:3\n'
        'H 1 1 1\nH 2.4 1 1\n')
    result = run_command('imrmlp', ['relax', 'settings.ini', '--num_threads', '1'], tmp_path)
    assert result.returncode in (0, 2), result.stdout + result.stderr
    from imrmlp import read
    output = read(str(tmp_path / 'output.xyz'), format='extxyz')[0]
    positions = output.positions
    assert np.isfinite(positions).all()
    assert np.linalg.norm(positions - [[1, 1, 1], [2.4, 1, 1]]) > 1e-7
    r = np.linalg.norm(positions[1] - positions[0])
    assert 4*(r**-12-r**-6) < 4*(1.4**-12-1.4**-6)
    np.testing.assert_array_equal(output.pbc, [i < rank for i in range(3)])
    np.testing.assert_array_equal(output.cell[rank:], cell[rank:])
