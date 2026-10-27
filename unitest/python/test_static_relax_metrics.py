"""Static optimization metrics/charts using actual LJ forces and CLI files."""
import numpy as np
import pytest

from test_cli_error_paths import run_command
from test_static_relax_constraints import prepare, output_positions, atoms, lj


def cli(path):
    return run_command('imrmlp', ['relax', 'settings.ini', '--num_threads', '1'], path)


def prepare_metric(path, extra):
    prepare(path, [], dict({'file-constraints': '', 'relax:driver': 'gd',
                           'relax:maxiteration': 3, 'relax:gtol': '1e-12'}, **extra))
    # An omitted constraints file is intentional; an empty V2 document is invalid.
    settings = (path / 'settings.ini').read_text()
    settings = '\n'.join(line for line in settings.splitlines()
                         if not line.startswith('file-constraints '))+'\n'
    (path / 'settings.ini').write_text(settings)


@pytest.mark.parametrize('matrix', [
    '5\n'+ ' '.join(['1']*25),  # wrong full coordinate count
    '6\n1 0 0',                # truncated
    '6\n'+ ' '.join(['nan']*36),
    '6\n'+ ' '.join(map(str, np.triu(np.ones((6, 6))).flat)),  # asymmetric
])
def test_cli_invalid_full_hessian_preserves_existing_output(tmp_path, matrix):
    prepare_metric(tmp_path, {'relax:hessian_precond_file': 'metric.txt'})
    (tmp_path / 'metric.txt').write_text(matrix)
    (tmp_path / 'output.xyz').write_text('previous successful output')
    result = cli(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / 'output.xyz').read_text() == 'previous successful output'
    assert not (tmp_path / 'reltrj.xyz').exists()


@pytest.mark.parametrize('driver', ['gd', 'fire'])
def test_cli_neighbor_graph_metric_moves_lj_pair_downhill(tmp_path, driver):
    prepare_metric(tmp_path, {'relax:driver': driver, 'relax:sparse_precond': 'True',
                             'relax:sparse_shift': '.2',
                             'relax:fire:downhill_check': 'True'})
    result = cli(tmp_path)
    assert result.returncode in (0, 2), result.stdout + result.stderr
    pos = output_positions(tmp_path)
    initial_r = 1.4
    r = np.linalg.norm(pos[1]-pos[0])
    energy = lambda d: 4*(d**-12-d**-6)
    assert r != pytest.approx(initial_r, abs=1e-9)
    assert energy(r) < energy(initial_r)
    np.testing.assert_allclose(pos.mean(axis=0), [1.7, 1., 1.], atol=1e-10)


@pytest.mark.parametrize('driver', ['gd', 'fire'])
def test_python_neighbor_graph_metric_preserves_physical_outputs(lj, driver):
    from imrmlp import Relaxation
    a = atoms()
    lj.calculate(a)
    energy0 = a.energy
    relax = Relaxation(lj, driver=driver, max_iter=4, gtol=1e-12,
                       sparse_precond=True, sparse_shift=.2,
                       downhill_check=driver == 'fire')
    result = relax.run(a)
    assert result.n_iter <= 4
    assert a.energy < energy0
    predicted = a.copy()
    lj.calculate(predicted)
    assert a.energy == pytest.approx(predicted.energy, abs=1e-12)
    np.testing.assert_allclose(a.forces, predicted.forces, atol=1e-12)


@pytest.mark.parametrize('driver', ['gd', 'fire', 'lbfgs'])
def test_python_rigid_group_keeps_internal_distance_under_real_force(lj, driver):
    from imrmlp import Atoms, Relaxation
    a = Atoms(numbers=[1, 1, 1], elements=['H'],
              positions=[[1., 1., 1.], [2.4, 1., 1.], [3.8, 1.5, 1.2]],
              cell=12*np.eye(3), pbc=False)
    before = a.positions.copy()
    result = Relaxation(lj, driver=driver, max_iter=5, gtol=1e-12,
                        rigid_groups=[[0, 1]], max_move=.03).run(a)
    assert result.n_iter <= 5
    assert np.linalg.norm(a.positions[1]-a.positions[0]) == pytest.approx(1.4, abs=2e-10)
    assert np.linalg.norm(a.positions-before) > 1e-7
    np.testing.assert_array_equal(a.cell, 12*np.eye(3))


def test_native_callback_chart_keeps_nonlinear_manifold(lj):
    from imrmlp._imrmlp import _Relaxation
    a = atoms()
    # One active bond separation; the second atom follows a parabola.
    # Initial q=1.4 exactly reconstructs the supplied frame.
    def full(q):
        s = q[0]-1.4
        return np.array([1., 1., 1., 1.+q[0], 1.+.2*s*s, 1.])
    def reduced(x):
        return np.array([x[3]-1.])
    def jac(q):
        return np.array([[0.], [0.], [0.], [1.], [.4*(q[0]-1.4)], [0.]])
    def curvature(q, gradient):
        return np.array([[.4*gradient[4]]])
    native = _Relaxation()
    native.driver = 'gd'
    native.max_iter = 5
    native.gtol = 1e-12
    native.write_traj = False
    native.set_elements(['H'])
    native.set_coordinate_map(reduced, full, jac, curvature)
    # Verify the curvature contraction independently from optimization at a
    # point with a nonzero transverse force and nonzero chart curvature.
    point = np.array([1.2])
    probe = atoms()
    probe.positions = full(point).reshape(2, 3)
    lj.calculate(probe)
    full_gradient = -probe.forces.reshape(-1)
    hessian = np.asarray(lj.hessian(probe, method='analytic')).reshape(6, 6)
    pulled = native.mapped_hessian(point, full_gradient, hessian)[0, 0]
    for step in (1e-5, 5e-6):
        gradients = []
        for value in (point[0]+step, point[0]-step):
            probe.positions = full([value]).reshape(2, 3)
            lj.calculate(probe)
            gradients.append((jac([value]).T @ (-probe.forces.reshape(-1))).item())
        assert pulled == pytest.approx((gradients[0]-gradients[1])/(2*step), abs=2e-5)
    native.init(lj.inner, a._impl)
    assert native.run() in (0, 1, -1)
    pos = np.asarray(native.atom.positions)
    q = pos[1, 0]-1.
    assert q != pytest.approx(1.4, abs=1e-8)
    np.testing.assert_allclose(pos.reshape(-1), full([q]), atol=2e-11)
    np.testing.assert_array_equal(a.positions, [[1., 1., 1.], [2.4, 1., 1.]])


@pytest.mark.parametrize('failure', ['roundtrip', 'jacobian', 'callback'])
def test_callback_initialization_failure_preserves_source(lj, failure):
    from imrmlp._imrmlp import _Relaxation
    a = atoms()
    before = a.positions.copy()
    def full(q):
        if failure == 'callback':
            raise ValueError('intentional chart error')
        return np.asarray(q) + (1. if failure == 'roundtrip' else 0.)
    native = _Relaxation()
    native.driver = 'gd'
    native.set_elements(['H'])
    native.set_coordinate_map(lambda x: x, full,
                              lambda q: np.eye(5 if failure == 'jacobian' else 6))
    with pytest.raises((ValueError, RuntimeError)):
        native.init(lj.inner, a._impl)
    assert not native.converged
    np.testing.assert_array_equal(a.positions, before)


@pytest.mark.parametrize('budget', [0, 4])
def test_native_online_fc_uses_accepted_step_schedule(lj, budget):
    from imrmlp._imrmlp import _Relaxation
    a = atoms()
    native = _Relaxation()
    native.driver = 'gd'
    native.max_iter = budget
    native.gtol = 1e-12
    native.write_traj = False
    native.fc_update_interval = 2
    native.set_elements(['H'])
    native.init(lj.inner, a._impl)
    # Initial curvature initializes the metric; refreshes require accepted moves.
    initial_count = native.fc_refresh_count
    assert native.accepted_steps == 0
    native.run()
    assert native.accepted_steps <= budget
    assert native.fc_refresh_count-initial_count == native.accepted_steps//2
    if budget == 0:
        assert native.fc_refresh_count == initial_count
        np.testing.assert_allclose(native.atom.positions, a.positions, atol=1e-14)


def test_periodic_subspace_preserves_missing_open_row(lj):
    from imrmlp import Atoms, Relaxation
    a = Atoms(numbers=[1, 1], elements=['H'],
              positions=[[.3, .4, 1.], [1.7, .4, 1.]],
              cell=np.diag([5., 6., 0.]), pbc=[True, True, False])
    result = Relaxation(lj, driver='gd', max_iter=2, gtol=1e-12,
                        relax_cell=True, periodic_subspace=True,
                        thin_cell_scaling=True, max_move=.01).run(a)
    assert result.n_iter <= 2
    np.testing.assert_array_equal(a.cell[2], [0., 0., 0.])
    np.testing.assert_array_equal(a.pbc, [True, True, False])
    assert np.linalg.norm(np.cross(a.cell[0], a.cell[1])) > 0
    assert np.isfinite(a.positions).all()


def test_aul_auto_equality_is_distinct_from_sqp(lj):
    from imrmlp import Relaxation
    a = atoms()
    result = Relaxation(lj, driver='lbfgs', aul_auto=True,
                        max_iter=1200, gtol=1e-6, constraint_loops=20,
                        constraints=[{'kind': 'distance', 'atoms': [0, 1],
                                      'target': 1.6, 'residual_tolerance': 1e-6}]).run(a)
    assert result.converged, result.stop_reason
    assert np.linalg.norm(a.positions[1]-a.positions[0]) == pytest.approx(1.6, abs=1e-6)
    assert np.linalg.norm(a.forces) > .1


def test_high_level_callback_map_dictionary(lj):
    from imrmlp import Relaxation
    a = atoms()
    mapping = {
        'to_reduced': lambda x: np.array([x[3]]),
        'to_full': lambda q: np.array([1., 1., 1., q[0], 1., 1.]),
        'jacobian': lambda q: np.array([[0.], [0.], [0.], [1.], [0.], [0.]]),
        'curvature': lambda q, gradient: np.zeros((1, 1)),
    }
    result = Relaxation(lj, driver='gd', max_iter=3, gtol=1e-12,
                        coordinate_map=mapping).run(a)
    assert result.n_iter <= 3
    assert a.positions[1, 0] != pytest.approx(2.4, abs=1e-8)
    np.testing.assert_allclose(a.positions[0], [1., 1., 1.], atol=1e-13)
    np.testing.assert_allclose(a.positions[1, 1:], [1., 1.], atol=1e-13)


def test_cli_full_indefinite_hessian_metric_preserves_physical_objective(tmp_path):
    prepare_metric(tmp_path, {'relax:hessian_precond_file': 'metric.txt',
                             'relax:precond_floor': '.1'})
    # A negative eigenvalue is valid input: the optimizer metric uses |lambda|.
    hessian = np.diag([4., -2., 3., 4., 2., 3.])
    hessian[0, 3] = hessian[3, 0] = .7
    (tmp_path / 'metric.txt').write_text('6\n'+' '.join(map(str, hessian.flat))+'\n')
    result = cli(tmp_path)
    assert result.returncode in (0, 2), result.stdout + result.stderr
    pos = output_positions(tmp_path)
    r = np.linalg.norm(pos[1]-pos[0])
    assert 4*(r**-12-r**-6) < 4*(1.4**-12-1.4**-6)


def test_python_map_initialization_binds_reference_before_reduction(lj):
    from imrmlp import Relaxation
    a = atoms()
    reference = {}
    calls = []
    def initialize(full):
        calls.append('initialize')
        reference['origin'] = np.array(full, copy=True)
    def reduced(full):
        assert 'origin' in reference
        return np.asarray(full)-reference['origin']
    def full(q):
        return reference['origin']+np.asarray(q)
    mapping = dict(initialize=initialize, to_reduced=reduced, to_full=full,
                   jacobian=lambda q: np.eye(len(q)))
    Relaxation(lj, max_iter=0, coordinate_map=mapping).run(a)
    assert calls == ['initialize']
    np.testing.assert_allclose(reference['origin'].reshape(2, 3), a.positions, atol=1e-14)
