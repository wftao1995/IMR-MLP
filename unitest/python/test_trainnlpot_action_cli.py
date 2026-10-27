"""Generic nonlinear action selection and gradient-output failures via real CLIs."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from test_cli_error_paths import ROOT, run_command
from training_cli_fixture import prepare_fixture, radial_coefficients
from test_nonlinear_contract_cli import run, write_settings


ACTIONS = ['do-sgd', 'do-lbfgs', 'do-cg', 'do-varpro',
           'calc-grad', 'check-grad', 'check-grad-all', 'test-curv']
INVALID_ACTIONS = [
    ({}, [], 'select a training or diagnostic action'),
    ({'fit:linear_refit': 'off'}, [], 'select a training or diagnostic action'),
    ({'fit:random_init': 'true'}, [], 'select a training or diagnostic action'),
    ({'fit:force_random_init': 'true'}, [], 'select a training or diagnostic action'),
    ({'fit:apply_disp': 'true'}, [], 'select a training or diagnostic action'),
    ({'fit:training_schedule': 'staged'}, [], 'select a training or diagnostic action'),
    ({}, ['--balance-atomic-fitness'], 'select a training or diagnostic action'),
    ({}, ['--linear-refit-only'], '--linear-refit-only requires fit:linear_refit'),
    *[({name: 'true'}, [f'--{name}=false'], 'select a training or diagnostic action')
      for name in ACTIONS],
    ({'linear-refit-only': 'true'}, ['--linear-refit-only=false'],
     'select a training or diagnostic action'),
]


def snapshot(path):
    return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}


@pytest.mark.parametrize('extra,flags,message', INVALID_ACTIONS)
@pytest.mark.parametrize('existing_outputs', [False, True])
def test_empty_actions_precede_inputs_and_preserve_every_output(tmp_path, extra, flags,
                                                              message, existing_outputs):
    # Invalid data/model text makes the ordering of the action guard observable.
    for name in ('base.mtp', 'train.cfg', 'index.map'):
        (tmp_path / name).write_text(f'invalid input sentinel {name}\n')
    if existing_outputs:
        for name in ('final.mtp', 'fitted.mtp', 'fitted.mtp.pre-refit',
                     'DEBUG', 'DEBUG_0', 'objective.json', 'linear-refit.json',
                     'output_loss_info', 'output_grad_info'):
            (tmp_path / name).write_text(f'previous output {name}\n')
    write_settings(tmp_path, {'debug': 'true', 'fit:debug': 'true',
                             'fit:objective-report': 'objective.json', **extra})
    before = snapshot(tmp_path)
    result = run(tmp_path, *flags)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stderr
    assert 'Loading training set' not in result.stdout
    assert 'Construct potential loader' not in result.stdout
    assert 'Finished' not in result.stdout
    assert snapshot(tmp_path) == before


def test_help_needs_neither_settings_nor_action(tmp_path):
    result = run_command('imrmlp', ['trainnlpot', '--help'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--linear-refit-only' in result.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('explicit', [False, True])
def test_valid_linear_refit_without_optimizer_keeps_radial_parameters(tmp_path, explicit):
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {'fit:linear_refit': 'sr',
                             'fit:penalty_weight': '0', 'fit:weight_L2': '0'})
    before = snapshot(tmp_path)
    flags = ['--linear-refit-only'] if explicit else []
    result = run(tmp_path, *flags)
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(radial_coefficients(tmp_path / 'final.mtp'),
                                  radial_coefficients(tmp_path / 'base.mtp'))
    assert (tmp_path / 'final.mtp').read_bytes() == (tmp_path / 'fitted.mtp').read_bytes()
    assert (tmp_path / 'fitted.mtp.pre-refit').is_file()
    report = json.loads((tmp_path / 'linear-refit.json').read_text())
    assert report['after'] < report['before']
    for name, data in before.items():
        assert (tmp_path / name).read_bytes() == data


@pytest.mark.parametrize('mode', ['invalid', 'OFF'])
def test_unknown_linear_refit_block_is_rejected_before_save(tmp_path, mode):
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {'fit:linear_refit': mode})
    (tmp_path / 'final.mtp').write_text('preserve final model\n')
    result = run(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f'Unsupported conditional linear block: {mode}' in result.stderr
    assert (tmp_path / 'final.mtp').read_text() == 'preserve final model\n'
    assert 'Finished' not in result.stdout


@pytest.mark.parametrize('action', ['calc-grad', 'check-grad', 'check-grad-all', 'test-curv'])
@pytest.mark.parametrize('settings_action', [False, True])
def test_diagnostics_are_effective_actions_and_do_not_publish_models(tmp_path, action, settings_action):
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {action: 'true' if settings_action else 'false'})
    for name in ('final.mtp', 'fitted.mtp', 'fitted.mtp.pre-refit'):
        (tmp_path / name).write_text(f'previous {name}\n')
    before = snapshot(tmp_path)
    result = run(tmp_path, *([] if settings_action else [f'--{action}']))
    assert result.returncode == 0, result.stdout + result.stderr
    for name, data in before.items():
        assert (tmp_path / name).read_bytes() == data
    if action == 'calc-grad':
        assert float((tmp_path / 'output_loss_info').read_text()) == pytest.approx(0.011305826700427296, rel=1e-12)
        np.testing.assert_allclose(np.loadtxt(tmp_path / 'output_grad_info'),
                                  [-0.035075060859336246, 0.0019493211813363066,
                                   -0.36467772266323634, -0.011366415824972696],
                                  rtol=1e-12, atol=1e-14)
    assert 'Finished' in result.stdout


@pytest.mark.parametrize('target', ['output_loss_info', 'output_grad_info'])
@pytest.mark.parametrize('failure', ['directory', 'missing_parent', 'full_device'])
def test_calc_grad_output_failure_is_reported_without_model_publication(tmp_path, target, failure):
    prepare_fixture(tmp_path)
    write_settings(tmp_path)
    protected = ('base.mtp', 'train.cfg', 'final.mtp', 'fitted.mtp')
    for name in protected[2:]:
        (tmp_path / name).write_text(f'previous {name}\n')
    before = {name: (tmp_path / name).read_bytes() for name in protected}
    if failure == 'directory':
        (tmp_path / target).mkdir()
    elif failure == 'missing_parent':
        (tmp_path / target).symlink_to('missing/output')
    else:
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full unavailable')
        (tmp_path / target).symlink_to('/dev/full')
    result = run(tmp_path, '--calc-grad')
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'calc-grad output' in result.stderr
    assert target in result.stderr
    assert 'Finished' not in result.stdout
    assert {name: (tmp_path / name).read_bytes() for name in protected} == before


def mpi_run(path, ranks, *flags):
    executable = Path(os.environ.get('IMRMLP_MPI_EXECUTABLE', ROOT / 'build/mpi/imrmlp-mpi'))
    launcher = shutil.which('mpirun')
    if not launcher or not executable.is_file():
        pytest.skip('Build MPI imrmlp and provide mpirun for collective CLI tests')
    env = os.environ.copy()
    env.update({'OMPI_ALLOW_RUN_AS_ROOT': '1', 'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM': '1',
                'OMPI_MCA_rmaps_base_oversubscribe': '1', 'OMP_NUM_THREADS': '1'})
    env['LD_LIBRARY_PATH'] = str(executable.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    return subprocess.run([launcher, '-n', str(ranks), str(executable), 'trainnlpot',
                           'settings.ini', '--num_threads', '1', *flags], cwd=path,
                          env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('ranks', [1, 2])
@pytest.mark.parametrize('flags,success', [(['--help'], True), ([], False),
                                         (['--linear-refit-only'], False),
                                         (['--calc-grad=false'], False)])
@pytest.mark.extended
def test_mpi_early_exit_preserves_outputs_on_all_ranks(tmp_path, ranks, flags, success):
    write_settings(tmp_path, {'debug': 'true', 'calc-grad': 'false'})
    (tmp_path / 'final.mtp').write_text('previous MPI output\n')
    before = snapshot(tmp_path)
    result = mpi_run(tmp_path, ranks, *flags)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    if not success:
        assert ('select a training or diagnostic action' in result.stderr
                or '--linear-refit-only requires fit:linear_refit' in result.stderr)
        assert 'Loading training set' not in result.stdout
        assert 'Finished' not in result.stdout
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('ranks', [1, 2])
@pytest.mark.parametrize('action', ['implicit_refit', 'explicit_refit', 'calc-grad'])
@pytest.mark.extended
def test_mpi_valid_refit_and_diagnostic_return_on_all_ranks(tmp_path, ranks, action):
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {'fit:linear_refit': 'off' if action == 'calc-grad' else 'sr',
                             'fit:penalty_weight': '0', 'fit:weight_L2': '0'})
    flags = [] if action == 'implicit_refit' else [
        '--linear-refit-only' if action == 'explicit_refit' else '--calc-grad']
    result = mpi_run(tmp_path, ranks, *flags)
    assert result.returncode == 0, result.stdout + result.stderr
    if action == 'calc-grad':
        assert np.isfinite(float((tmp_path / 'output_loss_info').read_text()))
        assert np.isfinite(np.loadtxt(tmp_path / 'output_grad_info')).all()
        assert not (tmp_path / 'final.mtp').exists()
    else:
        np.testing.assert_array_equal(radial_coefficients(tmp_path / 'final.mtp'),
                                      radial_coefficients(tmp_path / 'base.mtp'))
        report = json.loads((tmp_path / 'linear-refit.json').read_text())
        assert report['after'] < report['before']


@pytest.mark.parametrize('ranks', [1, 2])
@pytest.mark.parametrize('target', ['output_loss_info', 'output_grad_info'])
@pytest.mark.parametrize('failure', ['directory', 'full_device'])
@pytest.mark.extended
def test_mpi_calc_grad_root_output_failure_reaches_all_ranks(tmp_path, ranks, target, failure):
    prepare_fixture(tmp_path)
    write_settings(tmp_path)
    (tmp_path / 'final.mtp').write_text('previous MPI output\n')
    if failure == 'directory':
        (tmp_path / target).mkdir()
    else:
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full unavailable')
        (tmp_path / target).symlink_to('/dev/full')
    result = mpi_run(tmp_path, ranks, '--calc-grad')
    assert result.returncode != 0, result.stdout + result.stderr
    assert result.stderr.count('Training failure on rank 0:') == ranks
    assert result.stderr.count(f'calc-grad output: {target}') == ranks
    assert 'Finished' not in result.stdout
    assert (tmp_path / 'final.mtp').read_text() == 'previous MPI output\n'
