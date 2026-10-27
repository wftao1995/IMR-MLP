"""Explicit trainmtp actions and diagnostic-only output through real processes."""
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from test_cli_error_paths import ROOT, run_command
from training_cli_fixture import prepare_fixture


def write_settings(path, extra=None):
    settings = {
        'potfile': 'base.mtp', 'trainset': 'train.cfg',
        'trainset-format': 'mlpcfg', 'index': 'index.map',
        'fit:save_to': 'fitted.mtp', 'fit:skip_preinit': 'true',
        'fit:bfgs_maxiteration': '1', 'fit:bfgs_m': '2',
        'fit:bfgs_grad_tol': '0', 'fit:bfgs_conv_tol': '0',
        'fit:bfgs_step_tol': '0', 'fit:save_every': '100',
        'fit:rescale_mode': 'fixed', 'fit:epoch_limit': '1',
        'fit:batch_size': '6', 'fit:learning_rate': '0.001',
        'fit:energy_weight': '1', 'fit:force_weight': '0.5',
        'fit:stress_weight': '0.1', 'fit:penalty_weight': '0',
    }
    settings.update(extra or {})
    (path / 'settings.ini').write_text(''.join(f'{key} {value}\n' for key, value in settings.items()))


def run(path, *flags, command='trainmtp'):
    return run_command('imrmlp', [command, 'settings.ini', '--num_threads', '1', *flags], path)


INVALID_ACTIONS = [
    ({}, [], 'select a training or diagnostic action'),
    ({}, ['--balance-at'], 'select a training or diagnostic action'),
    ({'fit:force_randomlinear': 'true'}, [], 'select a training or diagnostic action'),
    ({'linear-fit': 'true'}, ['--linear-fit=false'], 'select a training or diagnostic action'),
    ({'do-lbfgs': 'true'}, ['--do-lbfgs=false'], 'select a training or diagnostic action'),
    ({'do-matrix-ata': 'true'}, ['--do-matrix-ata=false'], 'select a training or diagnostic action'),
    ({}, ['--do-rescale'], '--do-rescale requires --linear-fit'),
    ({'do-rescale': 'true'}, [], '--do-rescale requires --linear-fit'),
    ({}, ['--do-lbfgs', '--do-rescale'], '--do-rescale requires --linear-fit'),
    ({}, ['--do-matrix-ata', '--do-rescale'], '--do-rescale requires --linear-fit'),
    ({'linear-fit': 'true', 'do-rescale': 'true'}, ['--linear-fit=false'],
     '--do-rescale requires --linear-fit'),
]


@pytest.mark.parametrize('existing_outputs', [False, True])
@pytest.mark.parametrize('extra,flags,message', INVALID_ACTIONS)
def test_invalid_actions_precede_input_and_preserve_outputs(tmp_path, existing_outputs,
                                                          extra, flags, message):
    # These inputs deliberately cannot be parsed; action validation must win.
    for name in ('base.mtp', 'train.cfg', 'index.map'):
        (tmp_path / name).write_text(f'invalid input sentinel {name}\n')
    if existing_outputs:
        for name in ('_Trained_Potential_.mtp', 'fitted.mtp', 'DEBUG', 'DEBUG_0',
                     'MATATA.info', 'output_grad_info', '_RANDOM_IT_0.mtp'):
            (tmp_path / name).write_text(f'keep previous output {name}\n')
    write_settings(tmp_path, {'debug': 'true', 'fit:debug': 'true', **extra})
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = run(tmp_path, *flags)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stderr
    assert 'Loading training set' not in result.stdout
    assert 'Construct potential loader' not in result.stdout
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_help_needs_no_action_or_input(tmp_path):
    result = run_command('imrmlp', ['trainmtp', '--help'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '--linear-fit' in result.stdout
    assert not list(tmp_path.iterdir())


DIAGNOSTICS = ['--do-matrix-a', '--do-matrix-ata', '--calc-grad',
               '--check-grad', '--check-grad-all', '--test-curv']


@pytest.mark.parametrize('action', DIAGNOSTICS)
def test_diagnostic_actions_preserve_models(tmp_path, action):
    prepare_fixture(tmp_path)
    write_settings(tmp_path)
    (tmp_path / '_Trained_Potential_.mtp').write_text('previous trained potential\n')
    (tmp_path / 'fitted.mtp').write_text('previous checkpoint\n')
    names = ('base.mtp', 'train.cfg', '_Trained_Potential_.mtp', 'fitted.mtp')
    before = {name: (tmp_path / name).read_bytes() for name in names}
    result = run(tmp_path, action)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Finished' in result.stdout
    assert {name: (tmp_path / name).read_bytes() for name in names} == before
    assert not list(tmp_path.glob('*.iter-*'))
    if action == '--do-matrix-ata':
        assert 'Training run' not in result.stdout
        assert (tmp_path / 'MATATA.info').is_file()
        for channel in ('E', 'F', 'S'):
            data = (tmp_path / f'MATATA_{channel}').read_bytes()
            assert np.frombuffer(data, dtype=np.int32, count=2).tolist() == [2, 2]
            assert np.isfinite(np.frombuffer(data, dtype=np.float64, offset=9)).all()
            assert len(data) == 9 + 4 * 8
            assert (tmp_path / f'VECATA_{channel}').stat().st_size == 4 + 2 * 8
    elif action == '--do-matrix-a':
        for channel in ('Energy', 'Force', 'Stress'):
            assert (tmp_path / f'MatrixA_{channel}_CMPT_ON_CORE0').stat().st_size > 0
    elif action == '--calc-grad':
        gradient = np.loadtxt(tmp_path / 'output_grad_info')
        assert gradient.shape == (4,)
        assert np.isfinite(gradient).all()
    else:
        assert 'On coeff 0' in result.stdout
        assert (' HE = ' if action == '--test-curv' else ' N_CURV = ') in result.stdout


@pytest.mark.parametrize('action', ['--linear-fit', '--do-lbfgs', '--do-cg',
                                    '--do-sgd', '--do-df'])
def test_training_actions_really_publish_models(tmp_path, action):
    prepare_fixture(tmp_path)
    write_settings(tmp_path)
    original = (tmp_path / 'base.mtp').read_bytes()
    result = run(tmp_path, action)
    assert result.returncode == 0, result.stdout + result.stderr
    trained = (tmp_path / '_Trained_Potential_.mtp').read_bytes()
    assert trained.startswith(b'MTP\n')
    assert trained != original
    assert (tmp_path / 'base.mtp').read_bytes() == original
    assert 'Finished' in result.stdout


def test_random_search_remains_a_real_action_with_its_own_output(tmp_path):
    prepare_fixture(tmp_path)
    write_settings(tmp_path)
    (tmp_path / '_Trained_Potential_.mtp').write_text('keep ordinary final output\n')
    result = run(tmp_path, '--do-random')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Iteration: 0' in result.stdout
    assert (tmp_path / '_RANDOM_IT_0.mtp').read_text().startswith('MTP\n')
    assert (tmp_path / '_Trained_Potential_.mtp').read_text() == 'keep ordinary final output\n'


@pytest.mark.parametrize('settings_action', [False, True])
def test_linear_rescale_and_training_plus_matrix_export(tmp_path, settings_action):
    prepare_fixture(tmp_path)
    extra = {'linear-fit': 'true', 'do-rescale': 'true', 'do-matrix-ata': 'true'}
    write_settings(tmp_path, extra if settings_action else {})
    flags = [] if settings_action else ['-lf', '-rs', '-ata']
    result = run(tmp_path, *flags, command='trainmtp')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Rescale: fixed radial scaling; one linear fit' in result.stdout
    assert (tmp_path / '_Trained_Potential_.mtp').read_text().startswith('MTP\n')
    assert (tmp_path / 'MATATA.info').is_file()


def test_existing_optimizer_sequence_is_preserved(tmp_path):
    prepare_fixture(tmp_path)
    write_settings(tmp_path)
    result = run(tmp_path, '--do-lbfgs', '--do-cg', '--do-matrix-ata')
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'fitted.mtp').is_file()
    assert (tmp_path / '_Trained_Potential_.mtp').read_bytes() == (tmp_path / 'fitted.mtp').read_bytes()
    assert (tmp_path / 'MATATA.info').is_file()


def test_cli_action_overrides_false_settings_and_rescale_can_be_disabled(tmp_path):
    prepare_fixture(tmp_path)
    write_settings(tmp_path, {'do-matrix-ata': 'false', 'do-rescale': 'true'})
    result = run(tmp_path, '--do-matrix-ata', '--do-rescale=false')
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'MATATA.info').is_file()
    assert not (tmp_path / '_Trained_Potential_.mtp').exists()


@pytest.mark.parametrize('ranks', [1, 2])
@pytest.mark.parametrize('flags,success', [(['--help'], True), ([], False),
                                         (['--do-rescale'], False),
                                         (['--do-lbfgs=false'], False)])
@pytest.mark.extended
def test_mpi_early_exit_is_collective_and_preserves_outputs(tmp_path, ranks, flags, success):
    executable = Path(os.environ.get('IMRMLP_MPI_EXECUTABLE', ROOT / 'build/mpi/imrmlp-mpi'))
    launcher = shutil.which('mpirun')
    if not launcher or not executable.is_file():
        pytest.skip('Build MPI imrmlp and provide mpirun to exercise collective CLI exits')
    write_settings(tmp_path, {'debug': 'true'})
    (tmp_path / '_Trained_Potential_.mtp').write_text('preserve MPI output\n')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    env = os.environ.copy()
    env.update({'OMPI_ALLOW_RUN_AS_ROOT': '1', 'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM': '1',
                'OMPI_MCA_rmaps_base_oversubscribe': '1', 'OMP_NUM_THREADS': '1'})
    env['LD_LIBRARY_PATH'] = str(executable.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    result = subprocess.run([launcher, '-n', str(ranks), str(executable), 'trainmtp',
                             'settings.ini', '--num_threads', '1', *flags], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    if not success:
        assert 'requires --linear-fit' in result.stderr or 'select a training or diagnostic action' in result.stderr
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
