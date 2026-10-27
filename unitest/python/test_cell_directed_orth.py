"""Specified lattice directions create bounded integer cells without strain."""
import re
import numpy as np
import pytest
from test_cli_error_paths import run_command


def fixture(tmp_path, cell):
    (tmp_path/'index.map').write_text('H 1\n')
    (tmp_path/'input.xyz').write_text(
        f'1\nLattice="{" ".join(map(str, cell.flatten()))}" pbc="T T T" '
        'Properties=species:S:1:pos:R:3\nH .2 .3 .4\n')


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
@pytest.mark.parametrize('sign', [1, -1])
def test_directed_integer_completion_rotates_to_axis_without_straining(tmp_path, axis, sign):
    cell = np.array([[2., 0, 0], [-1, np.sqrt(3), 0], [0, 0, 3.]])
    fixture(tmp_path, cell)
    result = run_command('imrmlp', ['mk_orth', '--input=input.xyz', '--input-format=extxyz',
        '--output=output.xyz', '--output-format=extxyz', '--axis-choice', str(sign), str(sign), '0',
        f'--axis-to={axis}', '--max-expand=2', '--tol=1e-11', '--num_threads=1'], tmp_path)
    assert result.returncode == 0, result.stdout+result.stderr
    match = re.search(r'Directed supercell matrix[^\n]*\n([^\n]+)\n([^\n]+)\n([^\n]+)', result.stdout)
    assert match, result.stdout
    integer = np.array([list(map(int, row.split())) for row in match.groups()])
    rows = (tmp_path/'output.xyz').read_text().splitlines()
    assert int(rows[0]) == 2
    output = np.fromstring(re.search(r'Lattice="([^"]+)"', rows[1]).group(1), sep=' ').reshape(3, 3)
    selected = 'xyz'.index(axis)
    np.testing.assert_array_equal(integer[selected], [sign, sign, 0])
    np.testing.assert_allclose(output[selected], np.eye(3)[selected]*2, atol=1e-10)
    unrotated = integer@cell
    np.testing.assert_allclose(output@output.T, unrotated@unrotated.T, atol=1e-10)
    assert np.linalg.det(output) == pytest.approx(2*np.linalg.det(cell))
    rotation = np.linalg.solve(unrotated, output)
    np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-10)
    assert np.linalg.det(rotation) == pytest.approx(1.)


def test_directed_search_failure_is_explicit_and_search_only_does_not_write(tmp_path):
    fixture(tmp_path, np.array([[1., 0, 0], [np.sqrt(2), 1, 0], [np.sqrt(3), np.sqrt(5), 1]]))
    args = ['mk_orth', '--input=input.xyz', '--input-format=extxyz', '--output=output.xyz',
            '--axis-choice', '1', '0', '0', '--tol=1e-12', '--max-expand=2', '--num_threads=1']
    result = run_command('imrmlp', args, tmp_path)
    assert result.returncode == 1, result.stdout+result.stderr
    assert 'completion' in result.stderr.lower()
    assert not (tmp_path/'output.xyz').exists()
    fixture(tmp_path, np.eye(3)*2)
    result = run_command('imrmlp', [*args, '--search-only'], tmp_path)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'Directed supercell matrix' in result.stdout
    assert not (tmp_path/'output.xyz').exists()


def test_find_mindist_help_drops_the_unused_rmin_option(tmp_path):
    result = run_command('imrmlp', ['find_mindist', '--help'], tmp_path)
    assert 'rmin' not in result.stdout+result.stderr
