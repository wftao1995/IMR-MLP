"""Template initialization, reuse and safe species migration."""

import pytest
from test_gpm_workflow_cli import prepare_fixture, run


@pytest.mark.parametrize('head', ['linear', 'li', 'q2', 'fs', 'n1', 'n2'])
def test_initialize_template_without_training_data(tmp_path, head):
    prepare_fixture(tmp_path)
    text = (tmp_path / 'base.mtp').read_text()
    first = text.index('radial_coeffs')
    last = text.index('alpha_moments_count')
    text = text[:first] + text[last:]
    text = text[: text.index('species_coeffs')]
    (tmp_path / 'template.mtp').write_text(text)
    options = []
    if head != 'linear':
        (tmp_path / 'new.ini').write_text(
            f'pot:pottype mtp_m\npot:extmodel mtp_m_{head}\npot:n_input 2\npot:n_neuron 3\npot:n_neuron2 2\n'
        )
        options = ['--settings=new.ini']
    run(
        tmp_path,
        'initmtpm',
        '--input=template.mtp',
        '--output=initial.gpm',
        '--elements',
        'H',
        '--randomize=false',
        *options,
    )
    run(tmp_path, 'model_validate', '--input=initial.gpm')
    run(tmp_path, 'initmtpm', '--input=initial.gpm', '--output=reused.gpm')
    run(tmp_path, 'model_validate', '--input=reused.gpm')
    run(
        tmp_path,
        'adjustmtp',
        '--input=reused.gpm',
        '--output=expanded.gpm',
        '--append-elements',
        'He',
        '--new-species-init=clone',
        '--clone-from',
        '0',
    )
    run(
        tmp_path,
        'adjustmtp',
        '--input=expanded.gpm',
        '--output=reordered.gpm',
        '--neworder',
        '1',
        '0',
        '--basis-order=level',
    )
    run(tmp_path, 'model_validate', '--input=reordered.gpm')


def test_dataset_statistics_do_not_change_reused_parameters(tmp_path):
    prepare_fixture(tmp_path)
    run(tmp_path, 'initmtpm', '--input=base.mtp', '--output=initial.gpm')
    run(
        tmp_path,
        'initmtpm',
        '--input=initial.gpm',
        '--output=stats.gpm',
        '--dataset=train.cfg',
        '--format=mlpcfg',
    )
    assert (tmp_path / 'initial.gpm').read_bytes() == (tmp_path / 'stats.gpm').read_bytes()
    run(
        tmp_path,
        'initmtpm',
        '--input=initial.gpm',
        '--output=fitted.gpm',
        '--dataset=train.cfg',
        '--format=mlpcfg',
        '--fit-species-offsets',
    )
    run(tmp_path, 'model_validate', '--input=fitted.gpm')


@pytest.mark.parametrize(
    'options',
    [
        ['--type=unknown'],
        ['--type=mtp_m'],
        ['--fit-species-offsets'],
        ['--normalize-descriptors'],
        ['--index=missing.map'],
        ['--offset-target=charge'],
    ],
)
def test_invalid_initialization_keeps_existing_output(tmp_path, options):
    from test_cli_error_paths import run_command

    prepare_fixture(tmp_path)
    (tmp_path / 'old.gpm').write_bytes(b'keep old output')
    result = run_command(
        'imrmlp', ['initmtpm', '--input=base.mtp', '--output=old.gpm', *options], tmp_path
    )
    assert result.returncode != 0
    assert (tmp_path / 'old.gpm').read_bytes() == b'keep old output'
    assert not list(tmp_path.glob('.mtp-workflow-*'))


def test_template_can_define_new_element_layout(tmp_path):
    prepare_fixture(tmp_path)
    text = (tmp_path / 'base.mtp').read_text()
    text = text[: text.index('radial_coeffs')] + text[text.index('alpha_moments_count') :]
    text = text[: text.index('species_coeffs')]
    (tmp_path / 'template.mtp').write_text(text)
    run(tmp_path, 'initmtpm', '--input=template.mtp', '--output=two.gpm', '--elements', 'H', 'He')
    run(tmp_path, 'model_validate', '--input=two.gpm')


def test_legacy_initializer_marks_complete_model_initialized(tmp_path):
    prepare_fixture(tmp_path)
    (tmp_path / 'old.ini').write_text(
        'pot:pottype mtp_m\npot:extmodel mtp_m_q2\npot:potfile base.mtp\npotfile legacy.gpm\ndataset train.cfg\nformat mlpcfg\nindex index.map\nrandomize false\n'
    )
    # An ordinary initialized energy head is not an MTP-M template.
    text = (tmp_path / 'base.mtp').read_text()
    text = text[: text.index('species_coeffs')]
    (tmp_path / 'head.mtp').write_text(text)
    (tmp_path / 'old.ini').write_text(
        (tmp_path / 'old.ini').read_text().replace('pot:potfile base.mtp', 'pot:potfile head.mtp')
    )
    run(tmp_path, 'initmtpm', '--settings=old.ini')
    run(tmp_path, 'model_validate', '--input=legacy.gpm')


@pytest.mark.parametrize('ranks', [1, 2])
def test_mpi_initialization_and_failure_propagation(tmp_path, ranks):
    import os
    from pathlib import Path
    import subprocess

    executable = os.environ.get('MTP_WORKFLOW_MPI_EXECUTABLE')
    if not executable:
        pytest.skip('Set MTP_WORKFLOW_MPI_EXECUTABLE to an MPI CLI')
    prepare_fixture(tmp_path)
    text = (tmp_path / 'base.mtp').read_text()
    text = text[: text.index('radial_coeffs')] + text[text.index('alpha_moments_count') :]
    text = text[: text.index('species_coeffs')]
    (tmp_path / 'template.mtp').write_text(text)
    executable = Path(executable).resolve()
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(executable.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = '1'

    def launch(*args):
        return subprocess.run(
            ['mpiexec', '-n', str(ranks), str(executable), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    result = launch('initmtpm', '--input=template.mtp', '--elements', 'H', '--output=mpi.gpm')
    assert result.returncode == 0, result.stdout + result.stderr
    before = (tmp_path / 'mpi.gpm').read_bytes()
    result = launch('initmtpm', '--input=template.mtp', '--output=mpi.gpm', '--fit-species-offsets')
    assert result.returncode != 0
    assert (tmp_path / 'mpi.gpm').read_bytes() == before
    result = launch(
        'adjustmtp',
        '--input=mpi.gpm',
        '--output=mpi-expanded.gpm',
        '--append-elements',
        'He',
        '--new-species-init=clone',
        '--clone-from',
        '0',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run(tmp_path, 'model_validate', '--input=mpi-expanded.gpm')
