"""Restore native training and external .sel workflows with one mode switch."""

import struct

import pytest

from test_cli_error_paths import run_command
from test_gpm_workflow_cli import prepare_fixture, run, sample, select, training_settings

CLI = 'imrmlp'


@pytest.mark.parametrize('command', ['trainmtp', 'trainnlpot', 'train_linear'])
@pytest.mark.parametrize('from_settings', [False, True])
def test_legacy_training_defaults(tmp_path, command, from_settings):
    prepare_fixture(tmp_path)
    action = training_settings(tmp_path, command, {'legacy': 'true'} if from_settings else {})
    run(tmp_path, command, '--settings=fit.ini', *action, *([] if from_settings else ['--legacy']))
    output = tmp_path / (
        '_Trained_Potential_.mtp' if command == 'trainmtp' else '_Trained_Potential_'
    )
    assert output.is_file()
    assert not output.read_bytes().startswith(b'GPMLMODL')
    assert not list(tmp_path.glob('*.gpm'))
    assert not list(tmp_path.glob('*.sel'))


@pytest.mark.parametrize('command', ['trainmtp', 'trainnlpot'])
def test_legacy_nonlinear_checkpoint_names(tmp_path, command):
    prepare_fixture(tmp_path)
    training_settings(
        tmp_path,
        command,
        {
            'fit:linear_refit': 'off',
            'fit:skip_preinit': 'true',
            'fit:bfgs_maxiteration': '2',
            'fit:bfgs_m': '2',
            'fit:bfgs_grad_tol': '0',
            'fit:bfgs_conv_tol': '0',
            'fit:bfgs_step_tol': '0',
            'fit:save_every': '1',
            'fit:penalty_weight': '0',
        },
    )
    run(tmp_path, command, '--settings=fit.ini', '--legacy', '--do-lbfgs')
    pattern = (
        '_Current_potential_save_*.mtp' if command == 'trainmtp' else 'potential_fitted.iter-*.mtp'
    )
    snapshots = list(tmp_path.glob(pattern))
    assert snapshots
    assert all(not p.read_bytes().startswith(b'GPMLMODL') for p in snapshots)
    assert not list(tmp_path.glob('*.gpm'))


def test_cli_can_disable_settings_legacy_mode(tmp_path):
    prepare_fixture(tmp_path)
    action = training_settings(tmp_path, 'trainmtp', {'legacy': 'true'})
    run(tmp_path, 'trainmtp', '--settings=fit.ini', '--legacy=false', *action)
    assert (tmp_path / '_Trained_Potential_.gpm').read_bytes().startswith(b'GPMLMODL')
    assert not (tmp_path / '_Trained_Potential_.mtp').exists()


@pytest.mark.parametrize('command', ['trainmtp', 'trainnlpot', 'train_linear'])
def test_legacy_training_honors_explicit_native_paths(tmp_path, command):
    prepare_fixture(tmp_path)
    action = training_settings(
        tmp_path, command, {'output': 'final.native', 'fit:save_to': 'checkpoint.native'}
    )
    run(tmp_path, command, '--settings=fit.ini', '--legacy', *action)
    assert (tmp_path / 'final.native').is_file()
    assert not list(tmp_path.glob('_Trained_Potential_*'))
    assert not list(tmp_path.glob('*.gpm'))


@pytest.mark.parametrize('conflict', ['output', 'checkpoint', 'embedding'])
def test_legacy_training_rejects_gpm_requests_before_writing(tmp_path, conflict):
    prepare_fixture(tmp_path)
    action = training_settings(
        tmp_path, 'trainmtp', {'fit:save_to': 'checkpoint.gpm'} if conflict == 'checkpoint' else {}
    )
    extra = (
        ['--output=final.gpm']
        if conflict == 'output'
        else ['--save-maxvol-state']
        if conflict == 'embedding'
        else []
    )
    result = run_command(
        CLI, ['trainmtp', '--settings=fit.ini', '--legacy', *action, *extra], tmp_path
    )
    assert result.returncode != 0
    assert '--legacy' in result.stderr
    assert not list(tmp_path.glob('*.gpm'))
    assert not list(tmp_path.glob('_Trained_Potential_*'))


def test_legacy_select_matches_existing_sel_writer(tmp_path):
    prepare_fixture(tmp_path)
    select(tmp_path, '--state-output=explicit.sel')
    select(tmp_path, '--legacy')
    legacy = tmp_path / 'base.mtp.sel'
    assert struct.unpack('<ii', legacy.read_bytes()[:8]) == (0x41435456, 10)
    assert legacy.read_bytes() == (tmp_path / 'explicit.sel').read_bytes()
    assert not (tmp_path / 'base.mtp.maxvol.gpm').exists()


def test_legacy_sample_uses_sel_even_when_gpm_sidecar_exists(tmp_path):
    prepare_fixture(tmp_path)
    select(tmp_path, '--legacy')
    state = (tmp_path / 'base.mtp.sel').read_bytes()
    sample(tmp_path, '--potfile=base.mtp', '--state-file=base.mtp.sel', '--dtype=pot_efs')
    expected = (tmp_path / 'sampled.xyz').read_bytes()
    (tmp_path / 'base.mtp.maxvol.gpm').write_bytes(b'invalid GPM must not be opened')
    (tmp_path / 'sample.ini').write_text('legacy true\n')
    sample(tmp_path, '--potfile=base.mtp', '--dtype=pot_efs')
    assert (tmp_path / 'sampled.xyz').read_bytes() == expected
    assert (tmp_path / 'base.mtp.sel').read_bytes() == state


def test_legacy_extends_sel_in_place_and_to_explicit_path(tmp_path):
    prepare_fixture(tmp_path)
    select(tmp_path, '--legacy')
    path = tmp_path / 'base.mtp.sel'
    original = path.read_bytes()
    args = [
        '--legacy',
        '--potfile=base.mtp',
        '--state-file=base.mtp.sel',
        '--dtype=pot_efs',
        '--candidates=train.cfg',
        '--input-format=mlpcfg',
    ]
    run(tmp_path, 'maxvol_extend', *args, '--state-output=continued.sel')
    assert path.read_bytes() == original
    assert (tmp_path / 'continued.sel').read_bytes() != original
    run(tmp_path, 'maxvol_extend', *args)
    assert path.read_bytes() != original
    assert not list(tmp_path.glob('*.gpm'))
    run(tmp_path, 'maxvol_sample', '--legacy', '--state-file=base.mtp.sel', '--dump-state=true')


def test_legacy_empty_extension_preserves_structure_output(tmp_path):
    prepare_fixture(tmp_path)
    select(tmp_path, '--legacy', '--threshold=1e100')
    state = tmp_path / 'base.mtp.sel'
    original = state.read_bytes()
    output = tmp_path / 'new_selected.xyz'
    output.write_bytes(b'keep previous structures')
    run(
        tmp_path,
        'maxvol_extend',
        '--legacy',
        '--potfile=base.mtp',
        '--state-file=base.mtp.sel',
        '--dtype=pot_efs',
        '--candidates=train.cfg',
        '--input-format=mlpcfg',
    )
    assert output.read_bytes() == b'keep previous structures'
    assert state.read_bytes() != original


def test_legacy_three_stage_extension_does_not_create_state(tmp_path):
    prepare_fixture(tmp_path)
    args = [
        '--legacy',
        '--potfile=base.mtp',
        '--dtype=pot_efs',
        '--train=train.cfg',
        '--candidates=train.cfg',
        '--input-format=mlpcfg',
    ]
    run(tmp_path, 'maxvol_extend', *args)
    assert not list(tmp_path.glob('*.sel'))
    assert not list(tmp_path.glob('*.gpm'))
    run(tmp_path, 'maxvol_extend', *args, '--state-output=explicit.sel')
    assert (tmp_path / 'explicit.sel').is_file()


def test_legacy_extension_does_not_resume_embedded_gpm_implicitly(tmp_path):
    prepare_fixture(tmp_path)
    select(tmp_path, '--state-output=session.gpm')
    original = (tmp_path / 'session.gpm').read_bytes()
    result = run_command(
        CLI,
        [
            'maxvol_extend',
            '--legacy',
            '--potfile=session.gpm',
            '--dtype=pot_efs',
            '--candidates=train.cfg',
            '--input-format=mlpcfg',
        ],
        tmp_path,
    )
    assert result.returncode != 0
    assert '--train' in result.stderr
    assert (tmp_path / 'session.gpm').read_bytes() == original


@pytest.mark.parametrize(
    'command,options',
    [
        ('maxvol_select', ['--pool=train.cfg', '--state-output=session.gpm']),
        ('maxvol_sample', ['--input=train.cfg', '--state-file=session.gpm']),
        ('maxvol_extend', ['--candidates=train.cfg', '--state-file=session.gpm']),
    ],
)
def test_legacy_rejects_gpm_state_paths(tmp_path, command, options):
    prepare_fixture(tmp_path)
    select(tmp_path, '--state-output=session.gpm')
    original = (tmp_path / 'session.gpm').read_bytes()
    result = run_command(
        CLI,
        [
            command,
            '--legacy',
            '--potfile=base.mtp',
            '--dtype=pot_efs',
            '--input-format=mlpcfg',
            *options,
        ],
        tmp_path,
    )
    assert result.returncode != 0
    assert 'legacy .sel' in result.stderr
    assert (tmp_path / 'session.gpm').read_bytes() == original
