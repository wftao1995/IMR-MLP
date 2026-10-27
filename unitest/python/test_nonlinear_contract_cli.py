"""Generic training settings and CG failure boundaries through public entry points."""
import hashlib
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from test_cli_error_paths import BUILD, run_command
from training_cli_fixture import (
    MODEL, evaluate, objective, prepare_fixture, radial_coefficients,
)


UNSUPPORTED = [
    ('weight_rad_w0', '1e-8', '0.5'), ('weight_rad_w1', '0', '0.5'),
    ('weight_rad_w2', '0', '0.5'), ('radial_reg_type', '0', '1'),
    ('rand_low', '-0.1', '-0.5'), ('rand_up', '0.1', '0.5'),
    ('lfit_fastalgo', 'true', 'false'), ('popsize', '20', '30'),
]


def write_settings(path, extra=None):
    settings = {
        'pot:pottype': 'mtp', 'pot:potfile': 'base.mtp',
        'trainset': 'train.cfg', 'trainset-format': 'mlpcfg',
        'index': 'index.map', 'output': 'final.mtp',
        'fit:energy_weight': '1', 'fit:force_weight': '0.5',
        'fit:stress_weight': '0.1', 'fit:bfgs_m': '2',
        'fit:bfgs_maxiteration': '60', 'fit:bfgs_grad_tol': '1e-10',
        'fit:bfgs_conv_tol': '0', 'fit:bfgs_step_tol': '0',
        'fit:save_every': '100', 'fit:save_to': 'fitted.mtp',
    }
    settings.update(extra or {})
    (path / 'settings.ini').write_text(''.join(f'{key} {value}\n' for key, value in settings.items()))


def run(path, *flags):
    return run_command('imrmlp', ['trainnlpot', 'settings.ini', '--num_threads', '1', *flags], path)


def fingerprints(path, names):
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in names}


@pytest.mark.parametrize('key,default,other', UNSUPPORTED)
@pytest.mark.parametrize('use_default', [True, False])
def test_unsupported_generic_settings_fail_before_cli_input_and_output(tmp_path, key, default, other, use_default):
    # Invalid input makes ordering observable: the specific option must win.
    names = ('base.mtp', 'train.cfg', 'fitted.mtp', 'final.mtp', 'fitted.mtp.iter-1.mtp')
    for name in names:
        (tmp_path / name).write_text(f'preserve {name}\n')
    before = fingerprints(tmp_path, names)
    write_settings(tmp_path, {f'fit:{key}': default if use_default else other})
    result = run(tmp_path, '--do-lbfgs', '--debug')
    assert result.returncode == 1, result.stdout + result.stderr
    assert key in result.stderr
    assert 'Construct potential loader' not in result.stdout
    assert 'Loading training set' not in result.stdout
    assert fingerprints(tmp_path, names) == before
    assert not list(tmp_path.glob('DEBUG*'))
    assert not (tmp_path / 'output_grad_info').exists()


@pytest.mark.parametrize('key,default,other', UNSUPPORTED)
@pytest.mark.parametrize('use_default', [True, False])
def test_python_passthrough_rejects_unused_settings_without_mutation(tmp_path, monkeypatch,
                                                                  key, default, other, use_default):
    pytest.importorskip('imrmlp._imrmlp')
    import imrmlp
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'base.mtp').write_text(MODEL)
    potential = imrmlp.Potential(tmp_path / 'base.mtp', 'mtp')
    atoms = imrmlp.Atoms(numbers=[1, 1, 1], elements=['H'],
                       positions=[[6, 6, 6], [8, 6.1, 6.2], [6.3, 8.9, 6.4]],
                       cell=np.eye(3) * 20, pbc=False)
    potential.calculate(atoms)
    before = potential.coefficients.copy()
    (tmp_path / 'fitted.mtp').write_text('previous successful output')
    settings = {key: default if use_default else other}
    trainer = imrmlp.NonlinearTrainer(max_iterations=2, bfgs_m=2,
                                    save_to='fitted.mtp', **settings)
    with pytest.raises(RuntimeError, match=key):
        trainer.train(potential, [atoms])
    np.testing.assert_array_equal(potential.coefficients, before)
    assert (tmp_path / 'base.mtp').read_text() == MODEL
    assert (tmp_path / 'fitted.mtp').read_text() == 'previous successful output'
    assert not list(tmp_path.glob('*.iter-*'))


def test_omitted_unsupported_settings_keep_existing_bfgs_objective_gradient_and_steps(tmp_path):
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {'fit:bfgs_maxiteration': '2', 'fit:bfgs_grad_tol': '0'})
    result = run(tmp_path, '--calc-grad')
    assert result.returncode == 0, result.stdout + result.stderr
    # Captured before this change with default model radial regularization intact.
    assert float((tmp_path / 'output_loss_info').read_text()) == pytest.approx(0.011305826700427296, rel=1e-12)
    np.testing.assert_allclose(np.loadtxt(tmp_path / 'output_grad_info'),
                              [-0.035075060859336246, 0.0019493211813363066,
                               -0.36467772266323634, -0.011366415824972696], rtol=1e-12, atol=1e-14)
    result = run(tmp_path, '--do-lbfgs')
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_allclose(radial_coefficients(tmp_path / 'final.mtp'),
                              [0.318302761060558104, 0.0960433787491914298], rtol=1e-12)
    content = (tmp_path / 'final.mtp').read_text()
    for key, expected in [('species_coeffs', .180091423508989629), ('moment_coeffs', .768389579499717423)]:
        assert float(re.search(rf'{key} = \{{([^}}]+)\}}', content).group(1)) == pytest.approx(expected, rel=1e-12)
    assert (tmp_path / 'final.mtp').read_bytes() == (tmp_path / 'fitted.mtp').read_bytes()


@pytest.mark.parametrize('weights', [(1., 0., 0.), (1., 1., .1), (0., 1., 0.), (0., 0., 1.)])
def test_cg_true_efs_optimization_changes_radial_coefficients(tmp_path, weights):
    target = prepare_fixture(tmp_path)
    before = evaluate(tmp_path, 'base.mtp', 'before.cfg')
    write_settings(tmp_path, {'fit:penalty_weight': '0', 'fit:weight_L2': '0', **{
        f'fit:{key}_weight': str(value) for key, value in zip(('energy', 'force', 'stress'), weights)}})
    original = fingerprints(tmp_path, ('base.mtp', 'train.cfg'))
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 0, result.stdout + result.stderr
    after = evaluate(tmp_path, 'final.mtp', 'after.cfg')
    assert objective(after, target, weights) < objective(before, target, weights) * .01
    assert np.linalg.norm(radial_coefficients(tmp_path / 'final.mtp') - radial_coefficients(tmp_path / 'base.mtp')) > 1e-5
    assert (tmp_path / 'final.mtp').read_bytes() == (tmp_path / 'fitted.mtp').read_bytes()
    assert fingerprints(tmp_path, ('base.mtp', 'train.cfg')) == original
    assert 'CG report:' in result.stdout


def test_cg_finite_budget_is_reported_without_false_convergence(tmp_path):
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {'fit:bfgs_maxiteration': '1', 'fit:bfgs_grad_tol': '0'})
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CG report: iterations=1' in result.stdout
    assert 'termination_type=5' in result.stdout
    assert 'not converged' in result.stdout + result.stderr
    assert 'CG accepted iteration 1:' in result.stdout
    assert (tmp_path / 'final.mtp').is_file()


def test_cg_initially_converged_model_is_saved_without_an_accepted_step(tmp_path):
    prepare_fixture(tmp_path)
    evaluate(tmp_path, 'base.mtp', 'train.cfg')
    write_settings(tmp_path, {'fit:penalty_weight': '0', 'fit:weight_L2': '0',
                             'fit:bfgs_grad_tol': '1e-12', 'fit:save_every': '1'})
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CG report: iterations=0' in result.stdout
    assert 'termination_type=4' in result.stdout
    assert 'CG initial point:' in result.stdout
    assert 'CG accepted iteration' not in result.stdout
    assert 'not converged' not in result.stdout + result.stderr
    assert not list(tmp_path.glob('*.iter-*'))
    np.testing.assert_array_equal(radial_coefficients(tmp_path / 'final.mtp'),
                                  radial_coefficients(tmp_path / 'base.mtp'))
    assert (tmp_path / 'final.mtp').read_bytes() == (tmp_path / 'fitted.mtp').read_bytes()


def test_bfgs_then_cg_resets_accepted_iteration_state(tmp_path):
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {'fit:bfgs_maxiteration': '2', 'fit:bfgs_grad_tol': '0',
                             'validset': 'train.cfg', 'validset-format': 'mlpcfg',
                             'fit:validation_interval': '1'})
    result = run(tmp_path, '--do-lbfgs', '--do-cg')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'LBFGS report: iterations=2' in result.stdout
    assert re.search(r'(?m)^CG report: iterations=2', result.stdout)
    assert 'CG accepted iteration 1:' in result.stdout
    assert 'CG initial point:' in result.stdout
    assert (tmp_path / 'final.mtp').read_bytes() == (tmp_path / 'fitted.mtp').read_bytes()


def test_cg_trainable_groups_freeze_radial_coefficients(tmp_path):
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {'fit:trainable_groups': 'species moment'})
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(radial_coefficients(tmp_path / 'final.mtp'),
                                  radial_coefficients(tmp_path / 'base.mtp'))
    final = (tmp_path / 'final.mtp').read_text()
    assert float(re.search(r'species_coeffs = \{([^}]+)\}', final).group(1)) != pytest.approx(.125)


@pytest.mark.parametrize('extra,diagnostic', [
    ({'fit:bfgs_maxiteration': '0'}, 'bfgs_maxiteration'),
    ({'fit:bfgs_maxiteration': '-1'}, 'bfgs_maxiteration'),
    ({'fit:bfgs_conv_tol': '-1'}, 'tol'), ({'fit:bfgs_step_tol': 'nan'}, 'tol'),
    ({'fit:bfgs_grad_tol': '-1'}, 'tol'), ({'fit:save_every': '0'}, 'save_every'),
    ({'fit:validation_interval': '0'}, 'validation_interval'),
    ({'fit:energy_weight': '-1'}, 'weight'), ({'fit:force_weight': 'nan'}, 'weight'),
    ({'fit:energy_weight': '0', 'fit:force_weight': '0', 'fit:stress_weight': '0'}, 'weight'),
    ({'fit:device': 'unavailable_gpu'}, 'unavailable GPU backend'), ({'fit:training_schedule': 'staged'}, 'Unknown fit:training_schedule'),
    ({'fit:parameter_scaling': 'jacobian'}, 'scal'),
    ({'fit:parameter_scaling': 'group_auto', 'fit:scale_floor': '-1'}, 'scal'),
    ({'fit:scaled_step_max': '-1'}, 'scal'), ({'fit:charge_weight': '-1'}, 'charge'),
    ({'fit:hvp_weight': '-1'}, 'HVP'), ({'fit:linear_refit': 'sr', 'fit:linear_refit_solver': 'invalid'}, 'linear refit solver'),
])
def test_cg_invalid_settings_fail_before_input_and_preserve_outputs(tmp_path, extra, diagnostic):
    write_settings(tmp_path, extra)
    (tmp_path / 'fitted.mtp').write_text('previous optimizer output')
    (tmp_path / 'final.mtp').write_text('previous CLI output')
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 1, result.stdout + result.stderr
    assert diagnostic.lower() in result.stderr.lower()
    assert 'Loading training set' not in result.stdout
    assert (tmp_path / 'fitted.mtp').read_text() == 'previous optimizer output'
    assert (tmp_path / 'final.mtp').read_text() == 'previous CLI output'
    assert not list(tmp_path.glob('*.iter-*'))


@pytest.mark.parametrize('optimizer', ['--do-sgd', '--do-varpro', '--linear-refit-only',
                                       '--calc-grad', '--check-grad', '--check-grad-all', '--test-curv'])
def test_cg_disallowed_optimizer_combinations_fail_before_input(tmp_path, optimizer):
    write_settings(tmp_path)
    result = run(tmp_path, '--do-cg', optimizer)
    assert result.returncode == 1, result.stdout + result.stderr
    assert '--do-cg' in result.stderr
    assert 'Loading training set' not in result.stdout


@pytest.mark.parametrize('phase', ['optimizer_final', 'cli_final', 'checkpoint'])
def test_cg_write_failures_do_not_replace_previous_success(tmp_path, phase):
    prepare_fixture(tmp_path)
    extra = {'fit:bfgs_maxiteration': '2', 'fit:bfgs_grad_tol': '0'}
    if phase == 'optimizer_final':
        extra['fit:save_to'] = 'missing/fitted.mtp'
    elif phase == 'cli_final':
        extra['output'] = 'missing/final.mtp'
    else:
        extra['fit:save_to'] = 'missing/fitted.mtp'
        extra['fit:save_every'] = '1'
    write_settings(tmp_path, extra)
    (tmp_path / 'fitted.mtp').write_text('previous optimizer output')
    (tmp_path / 'final.mtp').write_text('previous CLI output')
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Finished' not in result.stdout
    assert not (tmp_path / 'missing').exists()
    assert (tmp_path / 'final.mtp').read_text() == 'previous CLI output'
    if phase != 'cli_final':
        assert (tmp_path / 'fitted.mtp').read_text() == 'previous optimizer output'
    assert not list(tmp_path.glob('.*tmp*'))


@pytest.mark.parametrize('phase', ['optimizer_final', 'cli_final', 'checkpoint'])
def test_cg_rejects_output_links_without_overwriting_targets(tmp_path, phase):
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {'fit:bfgs_maxiteration': '2', 'fit:bfgs_grad_tol': '0',
                             'fit:save_every': '1' if phase == 'checkpoint' else '100'})
    (tmp_path / 'previous.mtp').write_text('previous successful output')
    for name in ('fitted.mtp', 'final.mtp'):
        (tmp_path / name).write_text(f'previous {name}')
    target = {'optimizer_final': 'fitted.mtp', 'cli_final': 'final.mtp',
              'checkpoint': 'fitted.mtp.iter-000001.mtp'}[phase]
    (tmp_path / target).unlink(missing_ok=True)
    (tmp_path / target).symlink_to('previous.mtp')
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 1, result.stdout + result.stderr
    assert (tmp_path / target).is_symlink()
    assert (tmp_path / 'previous.mtp').read_text() == 'previous successful output'
    if phase != 'cli_final':
        assert (tmp_path / 'final.mtp').read_text() == 'previous final.mtp'
    if phase == 'checkpoint':
        assert (tmp_path / 'fitted.mtp').read_text() == 'previous fitted.mtp'
    assert 'Finished' not in result.stdout


@pytest.mark.parametrize('failure', ['missing_labels', 'nonfinite_energy'])
def test_cg_bad_labels_do_not_publish_models(tmp_path, failure):
    prepare_fixture(tmp_path)
    extra = {}
    if failure == 'missing_labels':
        extra.update({'trainset': 'input.xyz', 'trainset-format': 'extxyz'})
    else:
        file = tmp_path / 'train.cfg'
        file.write_text(re.sub(r'(?m)^Energy\n[^\n]+', 'Energy\nnan', file.read_text()))
    write_settings(tmp_path, extra)
    for name in ('fitted.mtp', 'final.mtp'):
        (tmp_path / name).write_text(f'previous {name}')
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 1, result.stdout + result.stderr
    for name in ('fitted.mtp', 'final.mtp'):
        assert (tmp_path / name).read_text() == f'previous {name}'
    assert not list(tmp_path.glob('*.iter-*'))


def test_cg_real_serializer_failure_preserves_existing_models(tmp_path):
    resource = pytest.importorskip('resource')
    import signal
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {'fit:bfgs_maxiteration': '1', 'fit:bfgs_grad_tol': '0'})
    for name in ('fitted.mtp', 'final.mtp'):
        (tmp_path / name).write_text(f'previous {name}')

    def limit_model_writes():
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (100, 100))

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(BUILD) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = '1'
    result = subprocess.run([str(BUILD / 'imrmlp'), 'trainnlpot', 'settings.ini',
                             '--num_threads', '1', '--do-cg'], cwd=tmp_path, env=env,
                            preexec_fn=limit_model_writes, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Finished' not in result.stdout
    for name in ('fitted.mtp', 'final.mtp'):
        assert (tmp_path / name).read_text() == f'previous {name}'
    assert not list(tmp_path.glob('.imrmlp-cg-save-*'))


def test_cg_hvp_only_public_cli_reduces_nonzero_teacher_projection_error(tmp_path):
    pytest.importorskip('imrmlp._imrmlp')
    import imrmlp
    from imrmlp.io import write
    from training_cli_fixture import force_difference

    prepare_fixture(tmp_path)
    teacher = imrmlp.Potential(tmp_path / 'teacher.mtp', 'mtp')
    initial = imrmlp.Potential(tmp_path / 'base.mtp', 'mtp')
    data = []
    for i, distance in enumerate((1.5, 2.1, 2.8)):
        atoms = imrmlp.Atoms(numbers=[1, 1], elements=['H'], pbc=False,
                           positions=[[2., 2., 2.], [2.+distance, 2.2, 2.4]],
                           cell=np.eye(3)*20)
        directions = np.sin(np.arange(1., 13.).reshape(6, 2)*(.2+.03*i))
        atoms.set_hvp(directions, force_difference(teacher, atoms, directions))
        data.append(atoms)
    assert any(np.linalg.norm(a.hvp_targets) > 1e-6 for a in data)
    before = sum(np.linalg.norm(force_difference(initial, a, a.hvp_directions)-a.hvp_targets)**2
                 for a in data)
    assert before > 1e-8
    write(tmp_path / 'train.xyz', data, format='extxyz')
    write_settings(tmp_path, {
        'trainset': 'train.xyz', 'trainset-format': 'extxyz',
        'fit:energy_weight': '0', 'fit:force_weight': '0', 'fit:stress_weight': '0',
        'fit:hvp_weight': '.7', 'fit:penalty_weight': '0', 'fit:weight_L2': '0',
        'fit:bfgs_maxiteration': '100', 'fit:bfgs_grad_tol': '1e-8',
    })
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 0, result.stdout + result.stderr
    fitted = imrmlp.Potential(tmp_path / 'final.mtp', 'mtp')
    after = sum(np.linalg.norm(force_difference(fitted, a, a.hvp_directions)-a.hvp_targets)**2
                for a in data)
    assert after < before*.01
    assert 'HVP monitor:' in result.stdout


def test_cg_full_hessian_only_cli_reduces_nonzero_teacher_matrix_error(tmp_path):
    pytest.importorskip('imrmlp._imrmlp')
    import imrmlp
    from training_cli_fixture import force_difference

    prepare_fixture(tmp_path)
    teacher = imrmlp.Potential(tmp_path / 'teacher.mtp', 'mtp')
    initial = imrmlp.Potential(tmp_path / 'base.mtp', 'mtp')
    data, targets, text = [], [], []
    for distance in (1.5, 2.1, 2.8):
        atoms = imrmlp.Atoms(numbers=[1, 1], elements=['H'], pbc=False,
                           positions=[[2., 2., 2.], [2.+distance, 2.2, 2.4]],
                           cell=np.eye(3)*20)
        target = force_difference(teacher, atoms, np.eye(6))
        targets.append(target); data.append(atoms)
        flat = ' '.join(format(x, '.17g') for x in target.flat)
        text += ['2', 'Lattice="20 0 0 0 20 0 0 0 20" Properties=species:S:1:pos:R:3 '
                 'pbc="F F F" hessian="'+flat+'"']
        text += ['H '+' '.join(format(x, '.17g') for x in position) for position in atoms.positions]
    assert np.linalg.norm(targets[0]-np.diag(np.diag(targets[0]))) > 1e-6
    before = sum(np.linalg.norm(force_difference(initial, a, np.eye(6))-target)**2
                 for a, target in zip(data, targets))
    assert before > 1e-8
    (tmp_path/'train.xyz').write_text('\n'.join(text)+'\n')
    write_settings(tmp_path, {
        'trainset': 'train.xyz', 'trainset-format': 'extxyz',
        'fit:energy_weight': '0', 'fit:force_weight': '0', 'fit:stress_weight': '0',
        'fit:hessian_weight': '.7', 'fit:penalty_weight': '0', 'fit:weight_L2': '0',
        'fit:bfgs_maxiteration': '100', 'fit:bfgs_grad_tol': '1e-8',
    })
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 0, result.stdout + result.stderr
    fitted = imrmlp.Potential(tmp_path / 'final.mtp', 'mtp')
    after = sum(np.linalg.norm(force_difference(fitted, a, np.eye(6))-target)**2
                for a, target in zip(data, targets))
    assert after < .01*before
    assert 'Full Hessian FD monitor:' in result.stdout
