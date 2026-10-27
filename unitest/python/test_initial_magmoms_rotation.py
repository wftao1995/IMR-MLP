"""Known initial Cartesian magnetic vectors follow the geometry's changed axes."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
ase = pytest.importorskip('ase')
from ase.calculators.singlepoint import SinglePointCalculator

import imrmlp
from test_cli_error_paths import run_command


@pytest.mark.parametrize('scalar', [False, True])
def test_ase_extxyz_mk_lower_rotates_initial_and_computed_moments_independently(tmp_path, scalar):
    source = ase.Atoms(numbers=[1, 8], positions=[[1., 2., 3.], [2., 3., 4.]],
                       cell=[[0., 4., 0.], [-5., 0., 0.], [0., 0., 6.]], pbc=True)
    initial = np.array([.25, -.75]) if scalar else np.array([[1., 0., 0.], [0., 2., 1.]])
    computed = np.array([[0., 3., 0.], [0., 0., 4.]])
    source.set_initial_magnetic_moments(initial)
    source.calc = SinglePointCalculator(source, magmoms=computed)
    native = imrmlp.Atoms.from_ase(source)
    saved = native.copy()
    imrmlp.write_one(tmp_path / 'input.extxyz', native)
    result = run_command('imrmlp', [
        'mk_lower', '--input', 'input.extxyz', '--input-format', 'extxyz',
        '--output', 'output.extxyz', '--output-format', 'extxyz',
    ], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    transformed = imrmlp.read_one(tmp_path / 'output.extxyz')
    expected_initial = initial if scalar else np.array([[0., -1., 0.], [2., 0., 1.]])
    expected_computed = np.array([[3., 0., 0.], [0., 0., 4.]])
    np.testing.assert_allclose(transformed.cell, np.diag([4., 5., 6.]), atol=1e-12, rtol=0)
    np.testing.assert_allclose(transformed.positions, [[2., -1., 3.], [3., -2., 4.]],
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(transformed.initial_magmoms, expected_initial, atol=1e-12, rtol=0)
    np.testing.assert_allclose(transformed.magmoms, expected_computed, atol=1e-12, rtol=0)
    restored = transformed.to_ase()
    np.testing.assert_allclose(restored.get_initial_magnetic_moments(), expected_initial,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(restored.get_magnetic_moments(), expected_computed, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(native.positions, saved.positions)
    np.testing.assert_array_equal(native.initial_magmoms, initial)
    np.testing.assert_array_equal(native.magmoms, computed)
    np.testing.assert_array_equal(source.positions, [[1., 2., 3.], [2., 3., 4.]])
    np.testing.assert_array_equal(source.get_initial_magnetic_moments(), initial)
    np.testing.assert_array_equal(source.calc.results['magmoms'], computed)
