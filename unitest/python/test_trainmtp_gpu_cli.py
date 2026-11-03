"""Dedicated GPU trainmtp commands through actual processes and native reloads."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from test_cli_error_paths import ROOT
from test_trainmtp_action_cli import write_settings
from training_cli_fixture import (
    prepare_fixture, evaluate, objective, radial_coefficients,
)

DEVICE = os.environ.get('IMRMLP_TRAINMTP_DEVICE')
EXECUTABLE = Path(os.environ.get('IMRMLP_GPU_EXECUTABLE',
    ROOT / 'build/cuda/imrmlp-cu'))
pytestmark = pytest.mark.skipif(not DEVICE, reason='Set IMRMLP_TRAINMTP_DEVICE for real GPU tests')


def invoke(path, *actions):
    assert EXECUTABLE.is_file(), f'Build GPU executable: {EXECUTABLE}'
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(EXECUTABLE.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = '1'
    return subprocess.run([str(EXECUTABLE), 'trainmtp', 'settings.ini',
                           '--num_threads', '1', *actions], cwd=path, env=env,
                          capture_output=True, text=True, timeout=60)


def prepare(path, extra=None):
    labels = prepare_fixture(path)
    write_settings(path, {
        'fit:device': DEVICE, 'fit:gpu_id': '0',
        'fit:gpu_batch_atoms': '6', 'fit:gpu_batch_structures': '2',
        'fit:gpu_memory_mib': '8', 'fit:gpu_workspace_mib': '1',
        'fit:bfgs_maxiteration': '3', 'fit:epoch_limit': '2',
        'fit:linear_fit_bfgs': 'true', 'fit:bfgs_rescale': 'true',
        'fit:skip_preinit': 'false', 'fit:pre_init_iter': '1',
        **(extra or {}),
    })
    return labels


@pytest.mark.parametrize('action', ['--do-lbfgs', '--do-cg', '--do-sgd'])
@pytest.mark.parametrize('fast', ['false', 'true'])
def test_gpu_training_updates_reloads_and_resumes(tmp_path, action, fast):
    targets = prepare(tmp_path, {'fit:lfit_fastalgo': fast})
    before = evaluate(tmp_path, 'base.mtp', 'before.cfg')
    result = invoke(tmp_path, action)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = 'trainmtp GPU nonlinear report '
    report = json.loads(next(line.split(marker, 1)[1] for line in result.stdout.splitlines() if marker in line))
    assert report['gradient_evaluations'] > 0
    assert report['batches'] > 1
    assert report['peak_device_bytes'] <= 9 * 1024**2
    after = evaluate(tmp_path, '_Trained_Potential_.mtp', 'after.cfg')
    assert objective(after, targets, (1., .5, .1)) < objective(before, targets, (1., .5, .1))
    final_radial = radial_coefficients(tmp_path / '_Trained_Potential_.mtp')
    assert np.linalg.norm(final_radial - radial_coefficients(tmp_path / 'base.mtp')) > 1e-10
    np.testing.assert_array_equal(final_radial, radial_coefficients(tmp_path / 'fitted.mtp'))
    (tmp_path / 'restart.mtp').write_bytes((tmp_path / '_Trained_Potential_.mtp').read_bytes())
    result = invoke(tmp_path, action, '--potfile', 'restart.mtp')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('device_loss', ['false', 'true'])
def test_gpu_diagnostics_preserve_final_output_and_frozen_radial(tmp_path, device_loss):
    prepare(tmp_path, {'fit:gpu_device_loss': device_loss, 'fit:fix_radial_coeffs': 'true'})
    sentinel = tmp_path / '_Trained_Potential_.mtp'
    sentinel.write_text('keep existing model\n')
    result = invoke(tmp_path, '--calc-grad')
    assert result.returncode == 0, result.stdout + result.stderr
    assert np.isfinite(np.loadtxt(tmp_path / 'output_grad_info')).all()
    assert sentinel.read_text() == 'keep existing model\n'
    result = invoke(tmp_path, '--do-lbfgs')
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(radial_coefficients(sentinel), radial_coefficients(tmp_path / 'base.mtp'))


@pytest.mark.parametrize('fast', ['false', 'true'])
def test_gpu_linear_and_invalid_actions_preserve_outputs(tmp_path, fast):
    prepare(tmp_path, {'fit:lfit_fastalgo': fast})
    result = invoke(tmp_path, '--linear-fit')
    assert result.returncode == 0, result.stdout + result.stderr
    marker = 'trainmtp GPU linear report '
    report = json.loads(next(line.split(marker, 1)[1] for line in result.stdout.splitlines() if marker in line))
    assert report['verified']
    previous = (tmp_path / '_Trained_Potential_.mtp').read_bytes()
    result = invoke(tmp_path, '--linear-fit', '--do-cg')
    assert result.returncode != 0
    assert 'Unsupported GPU trainmtp action' in result.stderr
    assert (tmp_path / '_Trained_Potential_.mtp').read_bytes() == previous


def test_gpu_budget_failure_preserves_existing_model(tmp_path):
    prepare(tmp_path, {'fit:gpu_batch_atoms': '1'})
    destination = tmp_path / '_Trained_Potential_.mtp'
    destination.write_text('preserve final model\n')
    result = invoke(tmp_path, '--do-lbfgs')
    assert result.returncode != 0
    assert 'gpu_batch_atoms' in result.stderr
    assert destination.read_text() == 'preserve final model\n'
