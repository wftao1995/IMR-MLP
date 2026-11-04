"""Single-file model/state workflows and legacy CLI compatibility."""

import json
import struct

import pytest

from test_cli_error_paths import run_command

MODEL = '''MTP
version = 1.1.0
potential_name = committee_cli_test
species_count = 1
potential_tag = test-only
radial_basis_type = RBChebyshev
min_dist = 0
max_dist = 5
radial_basis_size = 2
radial_funcs_count = 1
radial_coeffs
0-0
{0.25, 0.125}
alpha_moments_count = 1
alpha_index_basic_count = 1
alpha_index_basic = {{0,0,0,0}}
alpha_index_times_count = 0
alpha_index_times = {}
alpha_scalar_moments = 1
alpha_moment_mapping = {0}
species_coeffs = {0.125}
moment_coeffs = {0.75}
'''


def prepare_fixture(path):
    """Also used by the MPI acceptance run to generate identical small EFS data."""
    path.mkdir(parents=True, exist_ok=True)
    (path / 'base.mtp').write_text(MODEL)
    (path / 'teacher.mtp').write_text(
        MODEL.replace('{0.25, 0.125}', '{0.32, 0.075}').replace(
            'species_coeffs = {0.125}', 'species_coeffs = {0.18}'
        )
    )
    (path / 'index.map').write_text('H 0\n')
    frames = []
    for i in range(6):
        positions = [[6, 6, 6], [7.8 + 0.15 * i, 6.1, 6.2], [6.3, 8.9 - 0.1 * i, 6.4]]
        frames.append(
            '3\nLattice="20 0 0 0 20 0 0 0 20" pbc="F F F" '
            'Properties=species:S:1:pos:R:3\n'
            + ''.join('H ' + ' '.join(map(str, point)) + '\n' for point in positions)
        )
    (path / 'input.xyz').write_text(''.join(frames))
    run(
        path,
        'efs',
        'input.xyz',
        'train.cfg',
        '--input-format=extxyz',
        '--output-format=mlpcfg',
        '--potfile=teacher.mtp',
        '--index=index.map',
    )


def run(path, command, *args):
    result = run_command('imrmlp', [command, *args, '--num_threads=1'], path)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def manifest(path):
    raw = path.read_bytes()
    assert raw[:8] == b'GPMLMODL'
    size = struct.unpack_from('<Q', raw, 16)[0]
    return json.loads(raw[64 : 64 + size])


def history_count(path):
    data = manifest(path)
    block = data['active_learning']['maxvol']['state']['cfg_row_start_']['block']
    return int(next(item for item in data['blocks'] if item['id'] == block)['shape'][0]) - 1


def select(path, *extra):
    return run(
        path,
        'maxvol_select',
        '--potfile=base.mtp',
        '--pool=train.cfg',
        '--input-format=mlpcfg',
        '--output-format=extxyz',
        '--output=selected.xyz',
        '--dtype=pot_efs',
        *extra,
    )


def sample(path, *source, output='sampled.xyz'):
    return run(
        path,
        'maxvol_sample',
        *source,
        '--input=train.cfg',
        '--input-format=mlpcfg',
        '--output-format=extxyz',
        f'--output={output}',
        '--threshold-save=0',
        '--threshold-break=1e100',
    )


def test_single_gpm_entrypoints_need_no_external_model_or_index(tmp_path):
    prepare_fixture(tmp_path)
    select(tmp_path, '--state-output=session.gpm')
    for name in ('base.mtp', 'teacher.mtp', 'index.map'):
        (tmp_path / name).unlink()
    before = (tmp_path / 'session.gpm').read_bytes()
    sample(tmp_path, '--potfile=session.gpm', output='by-model.xyz')
    sample(tmp_path, '--state-file=session.gpm', output='by-state.xyz')
    assert (tmp_path / 'by-model.xyz').read_bytes() == (tmp_path / 'by-state.xyz').read_bytes()
    assert (tmp_path / 'session.gpm').read_bytes() == before
    run(tmp_path, 'maxvol_sample', '--state-file=session.gpm', '--dump-state=true')
    run(tmp_path, 'maxvol_sample', '--state-file=session.gpm', '--extract-configs=history.xyz')
    assert (tmp_path / 'history.xyz').is_file()
    run(
        tmp_path,
        'maxvol_extend',
        '--state-file=session.gpm',
        '--candidates=train.cfg',
        '--input-format=mlpcfg',
        '--state-output=continued.gpm',
    )
    assert (tmp_path / 'session.gpm').read_bytes() == before
    assert history_count(tmp_path / 'continued.gpm') == 12
    run(
        tmp_path,
        'maxvol_extend',
        '--potfile=continued.gpm',
        '--candidates=train.cfg',
        '--input-format=mlpcfg',
    )
    assert history_count(tmp_path / 'continued.gpm') == 18
    run(tmp_path, 'model_validate', '--input=continued.gpm')


def test_legacy_sources_and_new_default_state_output(tmp_path):
    prepare_fixture(tmp_path)
    original = (tmp_path / 'base.mtp').read_bytes()
    select(tmp_path, '--state-output=base.mtp.sel')
    assert (tmp_path / 'base.mtp.sel').read_bytes()[:8] != b'GPMLMODL'
    sample(tmp_path, '--potfile=base.mtp', '--state-file=base.mtp.sel', '--dtype=pot_efs')
    legacy = (tmp_path / 'sampled.xyz').read_bytes()
    sample(tmp_path, '--potfile=base.mtp', '--dtype=pot_efs')
    assert (tmp_path / 'sampled.xyz').read_bytes() == legacy
    select(tmp_path)
    assert manifest(tmp_path / 'base.mtp.maxvol.gpm')['active_learning']['method'] == 'maxvol'
    sample(tmp_path, '--potfile=base.mtp')
    assert (tmp_path / 'sampled.xyz').read_bytes() == legacy
    assert (tmp_path / 'base.mtp').read_bytes() == original
    run(
        tmp_path,
        'maxvol_extend',
        '--potfile=base.mtp',
        '--state-file=base.mtp.sel',
        '--candidates=train.cfg',
        '--input-format=mlpcfg',
        '--dtype=pot_efs',
    )


def test_gpm_model_only_select_and_conflicting_overrides(tmp_path):
    prepare_fixture(tmp_path)
    run(
        tmp_path,
        'model_convert',
        '--input=base.mtp',
        '--type=mtp',
        '--index=index.map',
        '--output=model.gpm',
    )
    select(tmp_path, '--potfile=model.gpm', '--state-output=session.gpm')
    original = (tmp_path / 'session.gpm').read_bytes()
    for args, message in [
        (['--state-file=session.gpm', '--potfile=teacher.mtp'], 'differs'),
        (['--state-file=session.gpm', '--dtype=mtm'], 'conflicts'),
    ]:
        result = run_command(
            'imrmlp',
            [
                'maxvol_sample',
                *args,
                '--input=train.cfg',
                '--input-format=mlpcfg',
                '--output=failed.xyz',
                '--num_threads=1',
            ],
            tmp_path,
        )
        assert result.returncode != 0
        assert message in result.stderr
        assert not (tmp_path / 'failed.xyz').exists()
        assert (tmp_path / 'session.gpm').read_bytes() == original


def training_settings(path, command, extra=None):
    values = {
        'trainset': 'train.cfg',
        'trainset-format': 'mlpcfg',
        'fit:energy_weight': '1',
        'fit:force_weight': '.05',
        'fit:stress_weight': '.001',
    }
    action = []
    if command == 'trainmtp':
        values.update(
            {
                'potfile': 'base.mtp',
                'fit:skip_preinit': 'true',
                'fit:rescale_mode': 'fixed',
                'fit:lfit_fastalgo': 'true',
            }
        )
        action = ['--linear-fit']
    elif command == 'trainnlpot':
        values.update(
            {
                'pot:pottype': 'mtp',
                'pot:potfile': 'base.mtp',
                'fit:linear_refit': 'sr',
                'fit:penalty_weight': '0',
            }
        )
        action = ['--linear-refit-only']
    else:
        values.update(
            {
                'pot:pottype': 'descriptor_linear',
                'pot:descriptor_linear:descriptor:dtype': 'behler',
                'pot:descriptor_linear:descriptor:behler:cutoff': '5',
                'pot:descriptor_linear:descriptor:behler:dispcount': '2',
                'pot:descriptor_linear:descriptor:behler:dispselect': '1 2',
                'pot:descriptor_linear:descriptor:behler:eta': '.3',
                'fit:weight_L2': '.001',
            }
        )
    values.update(extra or {})
    (path / 'fit.ini').write_text(''.join(f'{key} {value}\n' for key, value in values.items()))
    return action


@pytest.mark.parametrize('command', ['trainmtp', 'trainnlpot', 'train_linear'])
@pytest.mark.parametrize('with_state', [False, True])
def test_training_default_gpm_and_optional_maxvol(tmp_path, command, with_state):
    prepare_fixture(tmp_path)
    action = training_settings(tmp_path, command)
    run(
        tmp_path,
        command,
        '--settings=fit.ini',
        *action,
        *(['--save-maxvol-state'] if with_state else []),
    )
    output = tmp_path / '_Trained_Potential_.gpm'
    data = manifest(output)
    assert ('active_learning' in data) == with_state
    run(tmp_path, 'model_validate', '--input=' + str(output))
    if with_state:
        assert history_count(output) == 6
        (tmp_path / 'index.map').unlink()
        sample(tmp_path, '--potfile=' + str(output))
    else:
        checkpoints = list(tmp_path.glob('potential*.gpm'))
        for checkpoint in checkpoints:
            assert 'active_learning' not in manifest(checkpoint)


def test_training_rebuild_failure_keeps_previous_output(tmp_path):
    prepare_fixture(tmp_path)
    action = training_settings(tmp_path, 'trainmtp', {'builder:dtype': 'unknown'})
    output = tmp_path / '_Trained_Potential_.gpm'
    output.write_bytes(b'keep previous model')
    result = run_command(
        'imrmlp',
        ['trainmtp', '--settings=fit.ini', *action, '--save-maxvol-state', '--num_threads=1'],
        tmp_path,
    )
    assert result.returncode != 0
    assert output.read_bytes() == b'keep previous model'
    assert not list(tmp_path.glob('.gpml-training-*'))


def test_python_native_selector_restores_gpm_magic_without_suffix(tmp_path):
    from imrmlp._imrmlp import _ActiveSelector
    from imrmlp import Atoms
    import numpy as np

    prepare_fixture(tmp_path)
    select(tmp_path, '--state-output=session.gpm')
    (tmp_path / 'session.gpm').rename(tmp_path / 'checkpoint.bin')
    for name in ('base.mtp', 'teacher.mtp', 'index.map'):
        (tmp_path / name).unlink()
    selector = _ActiveSelector.load_state(str(tmp_path / 'checkpoint.bin'), '')
    atom = Atoms(
        positions=[[0, 0, 0], [2.1, 0.1, 0.2]], cell=np.eye(3) * 20, numbers=[0, 0], pbc=[False] * 3
    )
    assert np.isfinite(selector.grade(atom._impl))


@pytest.mark.parametrize('command', ['trainmtp', 'trainnlpot'])
def test_default_nonlinear_checkpoints_are_model_only_gpm(tmp_path, command):
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
    run(tmp_path, command, '--settings=fit.ini', '--do-lbfgs')
    snapshots = list(tmp_path.glob('potential*.iter-*.gpm'))
    assert snapshots
    for snapshot in snapshots:
        assert 'active_learning' not in manifest(snapshot)
        run(tmp_path, 'model_validate', '--input=' + str(snapshot))


def test_training_explicit_native_output_and_checkpoint_remain_supported(tmp_path):
    prepare_fixture(tmp_path)
    action = training_settings(tmp_path, 'trainmtp', {'fit:save_to': 'native.mtp'})
    run(tmp_path, 'trainmtp', '--settings=fit.ini', *action, '--output=native-final.mtp')
    assert (tmp_path / 'native-final.mtp').read_bytes().startswith(b'MTP\n')
    assert not (tmp_path / '_Trained_Potential_.gpm').exists()
    rejected = run_command(
        'imrmlp',
        [
            'trainmtp',
            '--settings=fit.ini',
            *action,
            '--output=native-final.mtp',
            '--save-maxvol-state',
            '--num_threads=1',
        ],
        tmp_path,
    )
    assert rejected.returncode != 0
    assert 'requires a .gpm output' in rejected.stderr


@pytest.mark.parametrize('with_state', [False, True])
def test_retraining_a_bound_gpm_rebuilds_or_drops_its_old_state(tmp_path, with_state):
    prepare_fixture(tmp_path)
    select(tmp_path, '--state-output=input.gpm')
    run(
        tmp_path,
        'maxvol_extend',
        '--potfile=input.gpm',
        '--candidates=train.cfg',
        '--input-format=mlpcfg',
    )
    assert history_count(tmp_path / 'input.gpm') == 12
    before = (tmp_path / 'input.gpm').read_bytes()
    action = training_settings(tmp_path, 'trainmtp', {'potfile': 'input.gpm'})
    run(
        tmp_path,
        'trainmtp',
        '--settings=fit.ini',
        *action,
        *(['--save-maxvol-state'] if with_state else []),
    )
    output = tmp_path / '_Trained_Potential_.gpm'
    assert ('active_learning' in manifest(output)) == with_state
    if with_state:
        assert history_count(output) == 6
    assert (tmp_path / 'input.gpm').read_bytes() == before
    run(tmp_path, 'model_validate', '--input=' + str(output))


@pytest.mark.parametrize('with_state', [False, True])
def test_fixed_core_gpm_contains_the_complete_hybrid(tmp_path, with_state):
    prepare_fixture(tmp_path)
    action = training_settings(tmp_path, 'trainnlpot')
    run(
        tmp_path,
        'trainnlpot',
        '--settings=fit.ini',
        *action,
        '--fixed-core=teacher.mtp',
        '--fixed-core-type=mtp',
        *(['--save-maxvol-state'] if with_state else []),
    )
    output = tmp_path / '_Trained_Potential_.gpm'
    data = manifest(output)
    assert data['models'][data['root_model']]['type'] == 'hybrid'
    assert ('active_learning' in data) == with_state
    for name in ('base.mtp', 'teacher.mtp'):
        (tmp_path / name).unlink()
    run(tmp_path, 'model_validate', '--input=' + str(output))
    if with_state:
        sample(tmp_path, '--potfile=' + str(output))


def test_independent_descriptors_keep_legacy_default_and_require_model_for_gpm(tmp_path):
    prepare_fixture(tmp_path)
    (tmp_path / 'descriptor.ini').write_text(
        'behler:cutoff 5\nbehler:dispcount 2\nbehler:dispselect 1 2\nbehler:eta .3\n'
    )
    args = [
        '--settings=descriptor.ini',
        '--dtype=behler',
        '--pool=train.cfg',
        '--input-format=mlpcfg',
        '--output-format=extxyz',
        '--output=selected.xyz',
    ]
    run(tmp_path, 'maxvol_select', *args)
    assert (tmp_path / 'potential.mtp.sel').exists()
    rejected = run_command(
        'imrmlp',
        ['maxvol_select', *args, '--state-output=descriptor.gpm', '--num_threads=1'],
        tmp_path,
    )
    assert rejected.returncode != 0
    assert 'requires a target potential' in rejected.stderr
    assert not (tmp_path / 'descriptor.gpm').exists()


@pytest.mark.extended
@pytest.mark.parametrize('ranks', [1, 2])
@pytest.mark.parametrize('command', ['trainmtp', 'trainnlpot', 'train_linear'])
def test_mpi_training_and_sampling_restore_the_complete_pool(tmp_path, ranks, command):
    import os
    from pathlib import Path
    import shutil
    import subprocess

    executable = os.environ.get('IMRMLP_GPM_MPI_EXECUTABLE')
    if not executable:
        pytest.skip('Set IMRMLP_GPM_MPI_EXECUTABLE to a matching MPI build')
    launcher = shutil.which('mpiexec')
    if not launcher:
        pytest.skip('MPI launcher is unavailable')
    executable = Path(executable).resolve()
    prepare_fixture(tmp_path)
    action = training_settings(tmp_path, command)
    env = dict(
        os.environ,
        OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',
        LD_LIBRARY_PATH=str(executable.parent) + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
    )

    def mpi_run(*args):
        return subprocess.run(
            [launcher, '-n', str(ranks), str(executable), *args, '--num_threads=1'],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    trained = mpi_run(command, '--settings=fit.ini', *action, '--save-maxvol-state')
    assert trained.returncode == 0, trained.stdout + trained.stderr
    output = tmp_path / '_Trained_Potential_.gpm'
    assert history_count(output) == 6
    before = output.read_bytes()
    sampled = mpi_run(
        'maxvol_sample',
        '--state-file=' + str(output),
        '--input=train.cfg',
        '--input-format=mlpcfg',
        '--threshold-save=0',
        '--threshold-break=1e100',
    )
    assert sampled.returncode == 0, sampled.stdout + sampled.stderr
    assert output.read_bytes() == before
    if ranks == 2:
        output.unlink()
        output.mkdir()
        (output / 'keep').write_text('previous result')
        rejected = mpi_run(command, '--settings=fit.ini', *action, '--save-maxvol-state')
        assert rejected.returncode != 0
        assert (output / 'keep').read_text() == 'previous result'
        assert not list(tmp_path.glob('.gpml-training-*'))
