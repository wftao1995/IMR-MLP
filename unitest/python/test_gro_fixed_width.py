"""Standard fixed-width GRO geometry reaches public readers and convstr."""
from contextlib import closing
import gzip

import numpy as np
import pytest

from test_cli_error_paths import run_command
pytest.importorskip('imrmlp._imrmlp')
import imrmlp


GRO_FIXED = ('fixed columns\n3\n'
             '    1WAT      H12345-123.456-234.567 345.678\n'
             '    1WAT      O12346   0.125  -0.250   0.375  1.0000  2.0000  3.0000\n'
             '    2WAT      H12347   0.500   0.625   0.750\n'
             '1 2 3 .1 .2 .3 .4 .5 .6\n')
GRO_FIXED_SECOND = ('second columns\n1\n'
                    '    3MOL      O    0  -0.125   0.250  -0.375\n4 5 6')


def check_first(atom):
    np.testing.assert_array_equal(atom.numbers, [1, 8, 1])
    np.testing.assert_allclose(atom.positions, [[-1234.56, -2345.67, 3456.78],
                                               [1.25, -2.5, 3.75], [5, 6.25, 7.5]])
    np.testing.assert_allclose(atom.cell, [[10, 1, 2], [3, 20, 4], [5, 6, 30]])
    assert list(atom._impl.molid) == [0, 0, 1]
    assert list(atom._impl.molsize) == [2, 1]
    assert atom.velocities is None


@pytest.mark.parametrize('compressed', [False, True])
def test_fixed_width_gro_iread_preserves_physical_values_and_owned_frames(tmp_path, compressed):
    path = tmp_path / ('structure.gro.gz' if compressed else 'structure.gro')
    text = GRO_FIXED + GRO_FIXED_SECOND
    path.write_bytes(gzip.compress(text.encode()) if compressed else text.encode())
    with closing(imrmlp.iread(path)) as frames:
        first = next(frames)
        second = next(frames)
        with pytest.raises(StopIteration):
            next(frames)
    check_first(first)
    np.testing.assert_allclose(second.positions, [[-1.25, 2.5, -3.75]])
    np.testing.assert_allclose(second.cell, np.diag([40., 50., 60.]))
    check_first(imrmlp.read_one(path))


def test_fixed_width_gro_conversion_preserves_geometry(tmp_path):
    (tmp_path / 'input.gro').write_text(GRO_FIXED)
    (tmp_path / 'index.map').write_text('H\nO\n')
    result = run_command('imrmlp', [
        'convstr', '--input-format', 'gro', '--output-format', 'extxyz',
        '--index', 'index.map', '--loadrate', '1', '--outrate', '1',
        '--lossless', 'true', '--inputfiles', 'input.gro', '--outputfile', 'output.xyz',
    ], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    atom = imrmlp.read_one(tmp_path / 'output.xyz')
    np.testing.assert_array_equal(atom.numbers, [1, 8, 1])
    np.testing.assert_allclose(atom.positions, [[-1234.56, -2345.67, 3456.78],
                                               [1.25, -2.5, 3.75], [5, 6.25, 7.5]])
    np.testing.assert_allclose(atom.cell, [[10, 1, 2], [3, 20, 4], [5, 6, 30]])


def test_fixed_width_gro_unknown_atomname_is_not_inferred(tmp_path):
    path = tmp_path / 'unknown.gro'
    path.write_text('water\n1\n    1WAT     OW    1   0.125   0.250   0.375\n1 1 1\n')
    with pytest.raises(RuntimeError):
        imrmlp.read_one(path)



FIXED_ROWS = ['    1WAT      H    1   0.100   0.200   0.300',
              '    1WAT      O    2   0.400   0.500   0.600']
GRO_VELOCITIES = np.array([[.125, -.25, 1.5], [-1., 0., 2.3456]])
ANGSTROM_VELOCITIES = np.array([[1.25, -2.5, 15.], [-10., 0., 23.456]])


def standard_velocity_frame(velocities):
    rows = [row + (''.join(f'{value:8.4f}' for value in velocities[i])
                   if velocities is not None else '') + '\n'
            for i, row in enumerate(FIXED_ROWS)]
    return 'standard velocities\n2\n' + ''.join(rows) + '1 2 3\n'


@pytest.mark.parametrize('compressed', [False, True])
def test_standard_gro_velocity_units_zero_absence_and_extxyz_bridge(tmp_path, compressed):
    path = tmp_path / ('velocity.gro.gz' if compressed else 'velocity.gro')
    text = (standard_velocity_frame(GRO_VELOCITIES) +
            standard_velocity_frame(np.zeros((2, 3))) + standard_velocity_frame(None))
    path.write_bytes(gzip.compress(text.encode()) if compressed else text.encode())
    with closing(imrmlp.iread(path)) as stream:
        frames = list(stream)
    assert len(frames) == 3
    np.testing.assert_allclose(frames[0].velocities, ANGSTROM_VELOCITIES, atol=1e-13, rtol=0)
    np.testing.assert_array_equal(frames[1].velocities, np.zeros((2, 3)))
    assert frames[2].velocities is None
    np.testing.assert_allclose(imrmlp.read_one(path).velocities, ANGSTROM_VELOCITIES,
                               atol=1e-13, rtol=0)
    for atom in frames:
        np.testing.assert_allclose(atom.positions, [[1., 2., 3.], [4., 5., 6.]], atol=1e-13, rtol=0)
    snapshots = [atom.copy() for atom in frames]
    output = tmp_path / 'velocity.extxyz'
    imrmlp.write_iter(output, iter(frames))
    restored = imrmlp.read(output)
    np.testing.assert_allclose(restored[0].velocities, ANGSTROM_VELOCITIES, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(restored[1].velocities, np.zeros((2, 3)))
    assert restored[2].velocities is None
    for source, saved in zip(frames, snapshots):
        np.testing.assert_array_equal(source.positions, saved.positions)
        np.testing.assert_array_equal(source.cell, saved.cell)
        if saved.velocities is not None:
            np.testing.assert_array_equal(source.velocities, saved.velocities)
    frames[1].velocities = np.ones((2, 3))
    np.testing.assert_allclose(frames[0].velocities, ANGSTROM_VELOCITIES, atol=1e-13, rtol=0)


def test_standard_gro_velocity_units_reach_ase_momenta_without_changing_source(tmp_path):
    pytest.importorskip('ase')
    from ase.units import fs

    path = tmp_path / 'velocity.gro'
    path.write_text(standard_velocity_frame(GRO_VELOCITIES))
    with closing(imrmlp.iread(path)) as stream:
        atom = next(stream)
    saved = atom.copy()
    converted = atom.to_ase()
    np.testing.assert_allclose(converted.get_velocities(), ANGSTROM_VELOCITIES / (1000 * fs),
                               atol=1e-14, rtol=1e-13)
    restored = imrmlp.Atoms.from_ase(converted, read_results=False)
    np.testing.assert_allclose(restored.velocities, ANGSTROM_VELOCITIES, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(atom.positions, saved.positions)
    np.testing.assert_array_equal(atom.velocities, saved.velocities)


@pytest.mark.parametrize('suffix', [
    '', '  0.1250 -0.2500', '     bad  0.0000  0.0000',
    '     nan  0.0000  0.0000', '  0.1250 -0.2500  1.5000 ignored',
])
def test_partial_or_nonstandard_gro_velocities_keep_geometry_only(tmp_path, suffix):
    rows = (FIXED_ROWS[0] + '  0.1250 -0.2500  1.5000\n' + FIXED_ROWS[1] + suffix + '\n')
    path = tmp_path / 'partial-velocity.gro'
    path.write_text('partial velocities\n2\n' + rows + '1 2 3\n')
    atom = imrmlp.read_one(path)
    assert atom.velocities is None
    np.testing.assert_allclose(atom.positions, [[1., 2., 3.], [4., 5., 6.]], atol=1e-13, rtol=0)


def test_whitespace_gro_velocity_suffix_keeps_legacy_geometry_contract(tmp_path):
    path = tmp_path / 'whitespace.gro'
    path.write_text('legacy\n2\n1WAT H 1 .1 .2 .3 .125 -.25 1.5\n'
                    '1WAT O 2 .4 .5 .6 -1 0 2.3456\n1 2 3\n')
    assert imrmlp.read_one(path).velocities is None
