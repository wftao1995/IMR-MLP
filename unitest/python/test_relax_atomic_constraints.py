"""Native AUL geometry constraints through the public Python relaxation API."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Potential, Relaxation


@pytest.fixture
def zero_potential(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'index.map').write_text('H\n')
    return Potential.from_settings({
        'pottype': 'lj612', 'ntypes': 1, '0-0': '0 1', 'cut:0-0': 3,
    })


def triangle():
    return Atoms(numbers=[1, 1, 1], elements=['H'],
                 positions=[[1., 1., 1.], [2.6, 1.2, 1.1], [2.9, 2.6, 1.5]],
                 cell=np.eye(3) * 12, pbc=False)


def angle(positions):
    u, v = positions[0] - positions[1], positions[2] - positions[1]
    return np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)


@pytest.mark.parametrize('driver', ['lbfgspp', 'fire'])
@pytest.mark.parametrize('variable_cell', [False, True])
def test_distance_and_angle_reach_noninitial_targets(zero_potential, driver, variable_cell):
    atoms = triangle()
    initial = atoms.positions.copy()
    result = Relaxation(
        zero_potential, driver=driver, downhill_check=driver == 'fire',
        relax_cell=variable_cell,
        max_iter=1500, gtol=1e-7, max_move=.1, constraint_loops=12, max_linesearch=80,
        constraints=[
            {'kind': 'distance', 'atoms': [0, 1], 'target': 1.4,
             'residual_tolerance': 1e-6},
            {'kind': 'angle', 'atoms': [0, 1, 2], 'target': 109.5,
             'residual_tolerance': 1e-6},
        ],
    ).run(atoms)
    assert result.converged, result.stop_reason
    assert result.final_atoms is atoms
    assert result.n_iter <= 1500
    assert not np.allclose(initial, atoms.positions)
    d = atoms.positions[1] - atoms.positions[0]
    assert abs(d @ d - 1.4**2) <= 1e-6
    assert abs(angle(atoms.positions) - np.cos(np.deg2rad(109.5))) <= 1e-6
    assert np.linalg.det(atoms.cell) > 0
    if not variable_cell:
        np.testing.assert_array_equal(atoms.cell, np.eye(3) * 12)


@pytest.mark.parametrize('fractional', [False, True])
def test_partial_pbc_distance_uses_initial_nearest_image(zero_potential, fractional):
    atoms = Atoms(numbers=[1, 1], elements=['H'],
                  positions=[[7.8, 1., 1.], [.3, 1.4, 1.2]],
                  cell=np.diag([8., 9., 12.]), pbc=[True, False, True])
    if fractional:
        atoms.to_fractional()
    result = Relaxation(zero_potential, driver='lbfgspp', gtol=1e-7,
                        max_iter=500, constraints=[
                            {'kind': 'distance', 'atoms': [0, 1], 'target': .8,
                             'residual_tolerance': 1e-6},
                        ]).run(atoms)
    assert result.converged, result.stop_reason
    d = atoms.positions[1] - atoms.positions[0] + atoms.cell[0]
    assert abs(d @ d - .8**2) <= 1e-6
    np.testing.assert_array_equal(atoms.pbc, [True, False, True])


def test_omitted_targets_rebind_when_driver_is_reused(zero_potential):
    constraints = [{'kind': 'distance', 'atoms': [0, 1], 'residual_tolerance': 1e-8}]
    driver = Relaxation(zero_potential, constraints=constraints, max_iter=0)
    # The Python wrapper snapshots declarations, not a frame's bound targets.
    constraints[0]['target'] = 6.
    for scale in (1., 1.7):
        atoms = triangle()
        atoms.positions = atoms.positions * scale
        initial = atoms.positions.copy()
        result = driver.run(atoms)
        assert result.converged
        assert result.n_iter == 0
        np.testing.assert_allclose(atoms.positions, initial, atol=1e-14)


def test_unmet_constraint_with_zero_budget_is_not_converged(zero_potential):
    atoms = triangle()
    initial = atoms.positions.copy()
    result = Relaxation(zero_potential, driver='fire', downhill_check=True,
                        max_iter=0, constraints=[
                            {'kind': 'distance', 'atoms': [0, 1], 'target': 1.4,
                             'residual_tolerance': 1e-8},
                        ]).run(atoms)
    assert not result.converged
    assert result.stop_reason == 'iteration_limit'
    assert result.n_iter == 0
    np.testing.assert_allclose(atoms.positions, initial, atol=1e-14)


def test_bad_atom_index_fails_before_mutating_input(zero_potential):
    atoms = triangle()
    initial = atoms.positions.copy()
    driver = Relaxation(zero_potential, constraints=[
        {'kind': 'angle', 'atoms': [0, 1, 3], 'target': 100.,
         'residual_tolerance': 1e-6},
    ])
    with pytest.raises(ValueError, match='index|bounds'):
        driver.run(atoms)
    np.testing.assert_array_equal(atoms.positions, initial)


def test_failed_constraint_rebinding_clears_previous_success(zero_potential):
    driver = Relaxation(zero_potential, constraints=[
        {'kind': 'distance', 'atoms': [0, 1], 'residual_tolerance': 1e-6},
    ])
    assert driver.run(triangle()).converged
    short = Atoms(numbers=[1], elements=['H'], positions=[[1., 1., 1.]],
                  cell=np.eye(3) * 12, pbc=False)
    with pytest.raises(ValueError, match='index|bounds'):
        driver.run(short)
    assert not driver.converged
    assert driver.stop_reason == 'not_started'
    with pytest.raises(RuntimeError, match='init'):
        driver._impl.run()


def test_fire_constrained_real_potential_can_raise_physical_energy(zero_potential):
    potential = Potential.from_settings({
        'pottype': 'lj612', 'ntypes': 1, '0-0': '1 1', 'cut:0-0': 3,
    })
    atoms = Atoms(numbers=[1, 1], elements=['H'],
                  positions=[[1., 1., 1.], [2.4, 1., 1.]],
                  cell=np.eye(3) * 12, pbc=False)
    potential.calculate(atoms)
    initial_energy = atoms.energy
    result = Relaxation(potential, driver='fire', downhill_check=True,
                        gtol=1e-6, max_iter=2000, constraint_loops=15,
                        constraints=[
                            {'kind': 'distance', 'atoms': [0, 1], 'target': 1.8,
                             'residual_tolerance': 1e-6},
                        ]).run(atoms)
    assert result.converged, result.stop_reason
    d = atoms.positions[1] - atoms.positions[0]
    assert abs(d @ d - 1.8**2) <= 1e-6
    assert atoms.energy > initial_energy
    assert np.linalg.norm(atoms.forces) > 1e-3
    predicted = atoms.copy()
    potential.calculate(predicted)
    assert atoms.energy == pytest.approx(predicted.energy, abs=1e-12)
    np.testing.assert_allclose(atoms.forces, predicted.forces, atol=1e-12)
