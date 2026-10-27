"""XSF Cartesian force columns use Hartree/Angstrom on disk and eV/Angstrom in IMRMLP."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp


HA_TO_EV = 27.211396641308
CELL = np.array([[3., .2, .1], [.3, 4., .4], [.5, .6, 5.]])
ROWS = 'H 1 2 3 .01 -.02 .03\n8 4 5 6 -.04 .05 0\n'
EXPECTED_FORCES = np.array([[.27211396641308, -.54422793282616, .81634189923924],
                            [-1.08845586565232, 1.3605698320654, 0.]])


def xsf_text(rows=ROWS, periodic=True):
    if not periodic:
        return 'ATOMS\n' + rows
    vectors = ''.join(' '.join(map(str, row)) + '\n' for row in CELL)
    return ('CRYSTAL\nPRIMVEC\n' + vectors + 'CONVVEC\n' + vectors +
            'PRIMCOORD\n2 1\n' + rows)


@pytest.mark.parametrize('periodic', [False, True])
def test_xsf_public_read_has_independently_converted_cartesian_forces(tmp_path, periodic):
    path = tmp_path / 'forces.xsf'
    path.write_text(xsf_text(periodic=periodic))
    for atom in [imrmlp.read_one(path), imrmlp.read(path)[0]]:
        np.testing.assert_allclose(atom.forces, EXPECTED_FORCES, atol=1e-13, rtol=1e-13)
        np.testing.assert_array_equal(atom.positions, [[1., 2., 3.], [4., 5., 6.]])
        np.testing.assert_array_equal(atom.numbers, [1, 8])
        np.testing.assert_array_equal(atom.pbc, [periodic] * 3)
        assert atom.energy is None and atom.stress is None


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('fractional', [False, True])
def test_xsf_public_write_emits_hartree_forces_and_preserves_source(tmp_path, periodic, fractional):
    atom = imrmlp.Atoms(numbers=[1, 8], elements=['O', 'H'],
                      positions=[[1., 2., 3.], [4., 5., 6.]],
                      cell=CELL, pbc=periodic)
    force_ha = np.array([[.0123456789123, -.0234567891234, .0345678912345],
                         [-.0456789123456, .0567891234567, 0.]])
    atom.forces = force_ha * HA_TO_EV
    atom.stress = np.diag([1., 2., 3.])
    atom.energy = -7.5
    if fractional:
        atom.to_fractional()
    before = atom.copy()
    path = tmp_path / 'written.xsf'
    imrmlp.write_one(path, atom)
    lines = [line for line in path.read_text().splitlines() if not line.startswith('#')]
    start = lines.index('PRIMCOORD') + 2 if periodic else lines.index('ATOMS') + 1
    rows = [line.split() for line in lines[start:]]
    assert len(rows) == 2 and all(len(row) == 7 for row in rows)
    np.testing.assert_allclose([[float(value) for value in row[4:]] for row in rows],
                               force_ha, atol=1e-15, rtol=1e-13)
    restored = imrmlp.read_one(path)
    np.testing.assert_allclose(restored.forces, before.forces, atol=1e-13, rtol=1e-13)
    np.testing.assert_array_equal(atom.forces, before.forces)
    np.testing.assert_array_equal(atom.stress, before.stress)
    np.testing.assert_array_equal(atom.positions, before.positions)
    np.testing.assert_array_equal(atom.cell, before.cell)
    np.testing.assert_array_equal(atom.pbc, before.pbc)
    assert atom.energy == before.energy
    assert atom.chemmap == before.chemmap and atom._impl.cartesian == before._impl.cartesian


@pytest.mark.parametrize('periodic', [False, True])
def test_xsf_geometry_only_keeps_force_absence(tmp_path, periodic):
    source = tmp_path / 'geometry.xsf'
    source.write_text(xsf_text('H 1 2 3\nO 4 5 6\n', periodic))
    atom = imrmlp.read_one(source)
    assert atom.forces is None
    target = tmp_path / 'geometry-copy.xsf'
    imrmlp.write_one(target, atom)
    lines = [line for line in target.read_text().splitlines() if not line.startswith('#')]
    start = lines.index('PRIMCOORD') + 2 if periodic else lines.index('ATOMS') + 1
    assert all(len(line.split()) == 4 for line in lines[start:])
    assert imrmlp.read_one(target).forces is None


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('rows', [
    'H 1 2 3 .01 .02\nO 4 5 6 .01 .02 .03\n',
    'H 1 2 3 .01 .02 .03 .04\nO 4 5 6 .01 .02 .03\n',
    'H 1 2 3 .01 .02 .03\nO 4 5 6\n',
    'H 1 2 3\nO 4 5 6 .01 .02 .03\n',
    'H 1 2 3 .01 nan .03\nO 4 5 6 .01 .02 .03\n',
])
def test_xsf_public_read_rejects_incomplete_or_mixed_force_columns(tmp_path, periodic, rows):
    path = tmp_path / 'bad-forces.xsf'
    path.write_text(xsf_text(rows, periodic))
    with pytest.raises(RuntimeError, match='failed'):
        imrmlp.read_one(path)
    with pytest.raises(RuntimeError, match='failed'):
        imrmlp.read(path)


def test_convstr_xsf_force_columns_reach_extxyz_in_electron_volts(tmp_path):
    from test_cli_error_paths import run_command

    source = tmp_path / 'input.xsf'
    source.write_text(xsf_text())
    result = run_command('imrmlp', ['convstr', '--input-format', 'xsf',
                                  '--output-format', 'extxyz', source.name, 'output.xyz'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    converted = imrmlp.read_one(tmp_path / 'output.xyz')
    np.testing.assert_allclose(converted.forces, EXPECTED_FORCES, atol=1e-6, rtol=1e-6)
