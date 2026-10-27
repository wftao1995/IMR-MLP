"""Version-2 static constraints through real LJ CLI and Python consumers."""
import json
import re

import numpy as np
import pytest

from test_cli_error_paths import run_command


POSITIONS = np.array([[1., 1., 1.], [2.4, 1., 1.]])


def prepare(path, constraints, extra=None):
    (path / 'index.map').write_text('H 0\n')
    (path / 'input.xyz').write_text(
        '2\nLattice="12 0 0 0 12 0 0 0 12" pbc="F F F" '
        'Properties=species:S:1:pos:R:3\nH 1 1 1\nH 2.4 1 1\n')
    settings = {
        'input': 'input.xyz', 'input-format': 'extxyz',
        'output': 'output.xyz', 'output-format': 'extxyz',
        'pot:pottype': 'lj612', 'pot:ntypes': 1, 'pot:0-0': '1 1',
        'pot:cut:0-0': 3, 'isif': 2, 'file-constraints': 'constraints.json',
        'relax:driver': 'sqp', 'relax:gtol': '1e-6',
        'relax:maxiteration': 600, 'relax:stepmax': '.1',
        'relax:write_traj': 'False',
    }
    settings.update(extra or {})
    (path / 'settings.ini').write_text(''.join(f'{k} {v}\n' for k, v in settings.items()))
    (path / 'constraints.json').write_text(json.dumps({'version': 2, 'constraints': constraints}))


def run(path):
    return run_command('imrmlp', ['relax', 'settings.ini', '--num_threads', '1'], path)


def output_positions(path):
    lines = (path / 'output.xyz').read_text().splitlines()
    assert int(lines[0]) == 2
    assert re.search(r'pbc="F F F"', lines[1])
    return np.array([list(map(float, line.split()[1:4])) for line in lines[2:4]])


def distance_constraint(**bounds):
    return dict(kind='distance', atoms=[0, 1], residual_tolerance=1e-7, **bounds)


def test_cli_explicit_sqp_keeps_physical_lj_force_at_equality(tmp_path):
    # Setting the legacy AUL flag must not override an explicit SQP backend.
    prepare(tmp_path, [distance_constraint(target=1.8)],
            {'relax:with_constraint_aul': 'True', 'relax:constraint_loops': 1})
    result = run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    pos = output_positions(tmp_path)
    assert np.linalg.norm(pos[1]-pos[0]) == pytest.approx(1.8, abs=1e-7)
    # At 1.8 the analytic LJ radial force is nonzero: constrained convergence
    # must use stationarity with multipliers, not a zero physical force test.
    force = abs(24*(2/1.8**13-1/1.8**7))
    assert force > .1
    assert 'Constraint 0: residual=' in result.stdout


def test_cli_two_sided_wall_and_formula_have_actual_geometric_effect(tmp_path):
    prepare(tmp_path, [
        distance_constraint(lower=1.6, upper=1.8),
        dict(kind='position', atoms=[0], component='x', target=1., residual_tolerance=1e-7),
        dict(kind='wall', atoms=[1], normal=[0, 2, 0], lower=.8, upper=1.2,
             residual_tolerance=1e-7),
        dict(kind='formula', expression='z(0)+z(1)', target=2., residual_tolerance=1e-7),
    ])
    result = run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    pos = output_positions(tmp_path)
    assert np.linalg.norm(pos[1]-pos[0]) == pytest.approx(1.6, abs=3e-6)
    assert pos[0, 0] == pytest.approx(1., abs=1e-7)
    assert .8-1e-7 <= pos[1, 1] <= 1.2+1e-7
    assert pos[:, 2].sum() == pytest.approx(2., abs=1e-7)


@pytest.mark.parametrize('expression', [
    '__import__(0)', 'x(99)', 'x(0);x(1)', 'sqrt(-1)', '1/(x(0)-x(0))',
])
def test_cli_invalid_or_undefined_formula_preserves_existing_output(tmp_path, expression):
    prepare(tmp_path, [dict(kind='formula', expression=expression, target=0.,
                            residual_tolerance=1e-7)])
    (tmp_path / 'output.xyz').write_text('previous successful output')
    result = run(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / 'output.xyz').read_text() == 'previous successful output'
    assert not (tmp_path / 'reltrj.xyz').exists()


@pytest.fixture
def lj(tmp_path, monkeypatch):
    pytest.importorskip('imrmlp._imrmlp')
    from imrmlp import Potential
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'index.map').write_text('H 0\n')
    return Potential.from_settings({'pottype': 'lj612', 'ntypes': 1,
                                    '0-0': '1 1', 'cut:0-0': 3})


def atoms():
    from imrmlp import Atoms
    return Atoms(numbers=[1, 1], elements=['H'], positions=POSITIONS.copy(),
                 cell=12*np.eye(3), pbc=False)


@pytest.mark.parametrize('driver', ['sqp', 'nlc'])
def test_python_sqp_converges_with_nonzero_physical_forces(lj, driver):
    from imrmlp import Relaxation
    a = atoms()
    relax = Relaxation(lj, driver=driver, constraint_version=2,
                       constraints=[distance_constraint(target=1.8)],
                       max_iter=600, gtol=1e-6)
    result = relax.run(a)
    assert result.converged, result.stop_reason
    assert np.linalg.norm(a.positions[1]-a.positions[0]) == pytest.approx(1.8, abs=1e-7)
    assert np.linalg.norm(a.forces) > .1
    assert relax._impl.kkt_residual <= 1e-6
    predicted = a.copy()
    lj.calculate(predicted)
    assert a.energy == pytest.approx(predicted.energy, abs=1e-12)
    np.testing.assert_allclose(a.forces, predicted.forces, atol=1e-12)


def test_native_relaxation_accepts_v2_box_wall_and_formula(lj):
    from imrmlp._imrmlp import _Relaxation
    a = atoms()
    native = _Relaxation()
    native.set_elements(['H'])
    native.driver = 'sqp'
    native.gtol = 1e-6
    native.max_iter = 600
    native.write_traj = False
    entries = [distance_constraint(lower=1.6, upper=1.8),
               dict(kind='position', atoms=[0], component='x', lower=1., upper=1.,
                    residual_tolerance=1e-7),
               dict(kind='wall', atoms=[1], normal=[0, 1, 0], lower=.8, upper=1.2,
                    residual_tolerance=1e-7),
               dict(kind='formula', expression='z(0)+z(1)', target=2.,
                    residual_tolerance=1e-7)]
    native.init(lj.inner, a._impl, json.dumps({'version': 2, 'constraints': entries}))
    assert native.run() == 0
    pos = np.asarray(native.atom.positions)
    assert np.linalg.norm(pos[1]-pos[0]) == pytest.approx(1.6, abs=3e-6)
    assert native.kkt_residual <= 1e-6
    np.testing.assert_array_equal(a.positions, POSITIONS)


def test_python_v2_symmetry_declaration_selects_initial_operations_and_sites(lj):
    from imrmlp import Atoms, Relaxation
    a = Atoms(numbers=[1, 1], elements=['H'],
              positions=[[0., 0., 0.], [2., 2., 2.]], cell=4*np.eye(3), pbc=True)
    relax = Relaxation(lj, max_iter=0, constraint_version=2,
                       constraints=[dict(kind='symmetry', operations=[0], sites=[0])])
    result = relax.run(a)
    assert result.converged
    assert relax._impl.keep_symm
    assert relax._impl.symmetry_operations == '0'
    assert relax._impl.symmetry_sites == '0'
    np.testing.assert_allclose(a.positions, [[0, 0, 0], [2, 2, 2]], atol=1e-12)
