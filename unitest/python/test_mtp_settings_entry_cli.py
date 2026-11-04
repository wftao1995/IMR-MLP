"""Settings-only MTP initialization and adjustment, with CLI precedence."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
from test_cli_error_paths import ROOT, run_command
from test_gpm_workflow_cli import manifest, run

CLI = 'imrmlp'
HEAD = 'pot:extmodel mtp_m_n2\npot:n_input 2\npot:n_neuron 3\npot:n_neuron2 2\n'


@pytest.fixture
def template(tmp_path):
    shutil.copyfile(
        ROOT / 'unitest/test-data/mtp-basis-corpus/shapeev-06.mtp', tmp_path / 'template.mtp'
    )
    return tmp_path


def elements(path):
    return [entry['symbol'] for entry in manifest(path)['species']]


def write(path, name, text):
    (path / name).write_text(text)


@pytest.mark.parametrize('kind', ['mtp', 'mtp_m'])
@pytest.mark.parametrize('custom_output', [False, True])
def test_initialize_entirely_from_settings_without_dataset(template, kind, custom_output):
    path = template
    output = 'initial.gpm' if custom_output else '_Initialized_Potential_.gpm'
    # Exercise both the top-level type option and the old pot:pottype spelling.
    config = 'input template.mtp\nelements H He\nrandomize false\n'
    config += 'type mtp\n' if kind == 'mtp' else 'pot:pottype mtp_m\n' + HEAD
    if custom_output:
        config += 'output initial.gpm\n'
    write(path, 'init.ini', config)
    run(path, 'initmtpm', '--settings=init.ini')
    run(path, 'model_validate', '--input=' + output)
    assert elements(path / output) == ['H', 'He']
    data = manifest(path / output)
    node = data['models'][data['root_model']]
    assert node['type'] == kind
    if kind == 'mtp_m':
        assert node['configuration']['metadata']['nn']['hidden_layers'] == [3, 2]
    write(path, 'head.ini', HEAD if kind == 'mtp_m' else '')
    run(
        path,
        'initmtpm',
        '--input=template.mtp',
        '--type=' + kind,
        '--elements',
        'H',
        'He',
        '--output=cli.gpm',
        '--randomize=false',
        '--settings=head.ini',
    )
    assert (path / output).read_bytes() == (path / 'cli.gpm').read_bytes()
    assert not (path / 'train.cfg').exists()
    assert not (path / 'index.map').exists()


def test_initialization_cli_options_override_settings(template):
    path = template
    write(
        path,
        'init.ini',
        'input missing.mtp\noutput keep.gpm\ntype unknown\nelements Li\nrandomize false\n',
    )
    (path / 'keep.gpm').write_bytes(b'keep existing result')
    run(
        path,
        'initmtpm',
        '--settings=init.ini',
        '--input=template.mtp',
        '--output=chosen.gpm',
        '--type=mtp',
        '--elements',
        'H',
        'He',
    )
    run(path, 'model_validate', '--input=chosen.gpm')
    assert elements(path / 'chosen.gpm') == ['H', 'He']
    assert (path / 'keep.gpm').read_bytes() == b'keep existing result'


@pytest.fixture
def initialized(template):
    run(
        template,
        'initmtpm',
        '--input=template.mtp',
        '--type=mtp',
        '--elements',
        'H',
        'He',
        '--output=initial.gpm',
        '--randomize=false',
    )
    return template


@pytest.mark.parametrize('mode', ['custom', 'default', 'inplace'])
def test_adjustment_settings_apply_paths_and_species_permutation(initialized, mode):
    path = initialized
    before = (path / 'initial.gpm').read_bytes()
    output = {
        'custom': 'chosen.gpm',
        'default': '_Adjusted_Potential_.gpm',
        'inplace': 'initial.gpm',
    }[mode]
    config = 'input initial.gpm\nneworder 1 0\nbasis-order level-group\n'
    if mode != 'default':
        config += 'output ' + output + '\n'
    write(path, 'adjust.ini', config)
    run(path, 'adjustmtp', '--settings=adjust.ini')
    run(path, 'model_validate', '--input=' + output)
    assert elements(path / output) == ['He', 'H']
    if mode != 'inplace':
        assert (path / 'initial.gpm').read_bytes() == before
    assert not (path / 'output-adj.mtp').exists()
    assert not (path / 'potential.mtp').exists()
    if mode == 'custom':
        assert not (path / '_Adjusted_Potential_.gpm').exists()


def test_adjustment_settings_support_extension_and_cli_overrides(initialized):
    path = initialized
    config = (
        'input missing.gpm\noutput keep.gpm\nappend-elements Li\n'
        'new-species-init clone\nclone-from 1\nneworder 2 0 1\nbasis-order level-group\n'
    )
    write(path, 'adjust.ini', config)
    (path / 'keep.gpm').write_bytes(b'keep existing result')
    run(path, 'adjustmtp', '--settings=adjust.ini', '--input=initial.gpm', '--output=chosen.gpm')
    run(path, 'model_validate', '--input=chosen.gpm')
    assert elements(path / 'chosen.gpm') == ['Li', 'H', 'He']
    assert (path / 'keep.gpm').read_bytes() == b'keep existing result'
    # Explicit vector options also override the deck's permutation.
    run(
        path,
        'adjustmtp',
        '--settings=adjust.ini',
        '--input=initial.gpm',
        '--output=override.gpm',
        '--neworder',
        '0',
        '1',
        '2',
    )
    assert elements(path / 'override.gpm') == ['H', 'He', 'Li']


@pytest.mark.parametrize(
    'command,config',
    [
        ('initmtpm', 'input missing.mtp\noutput old.gpm\ntype mtp\n'),
        ('initmtpm', 'input template.mtp\noutput old.gpm\ntype unknown\n'),
        ('adjustmtp', 'input initial.gpm\noutput old.gpm\nneworder 0 0\n'),
    ],
)
def test_settings_failures_preserve_existing_output(initialized, command, config):
    path = initialized
    (path / 'old.gpm').write_bytes(b'keep existing result')
    write(path, 'bad.ini', config)
    result = run_command(CLI, [command, '--settings=bad.ini'], path)
    assert result.returncode != 0
    assert (path / 'old.gpm').read_bytes() == b'keep existing result'
    assert not list(path.glob('.mtp-workflow-*'))


def test_implicit_adjustment_output_cannot_overwrite_input(initialized):
    path = initialized
    target = path / '_Adjusted_Potential_.gpm'
    target.write_bytes((path / 'initial.gpm').read_bytes())
    before = target.read_bytes()
    write(path, 'adjust.ini', 'input _Adjusted_Potential_.gpm\nneworder 1 0\n')
    result = run_command(CLI, ['adjustmtp', '--settings=adjust.ini'], path)
    assert result.returncode != 0
    assert 'Explicit output is required' in result.stderr
    assert target.read_bytes() == before


@pytest.mark.parametrize('command', ['initmtpm', 'adjustmtp'])
def test_missing_settings_file_fails_without_outputs(tmp_path, command):
    result = run_command(CLI, [command, '--settings=missing.ini'], tmp_path)
    assert result.returncode != 0
    assert not list(tmp_path.glob('*.gpm'))


def test_command_catalog_describes_broader_initialization(tmp_path):
    result = run_command(CLI, ['--help'], tmp_path)
    assert result.returncode == 0
    assert 'Initialize MTP-based models' in result.stdout


@pytest.mark.parametrize('ranks', [1, 2])
def test_mpi_settings_entry_and_collective_error(template, ranks):
    executable = os.environ.get('MTP_WORKFLOW_MPI_EXECUTABLE')
    if not executable:
        pytest.skip('Set MTP_WORKFLOW_MPI_EXECUTABLE to a matching MPI CLI')
    executable = Path(executable).resolve()
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(executable.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = '1'

    def launch(command, settings):
        return subprocess.run(
            [
                'mpiexec',
                '-n',
                str(ranks),
                str(executable),
                command,
                '--settings=' + settings,
                '--num_threads=1',
            ],
            cwd=template,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )

    write(
        template,
        'init.ini',
        'input template.mtp\noutput initial.gpm\nelements H He\n'
        'pot:pottype mtp_m\nrandomize false\n' + HEAD,
    )
    result = launch('initmtpm', 'init.ini')
    assert result.returncode == 0, result.stdout + result.stderr
    write(template, 'adjust.ini', 'input initial.gpm\noutput reordered.gpm\nneworder 1 0\n')
    result = launch('adjustmtp', 'adjust.ini')
    assert result.returncode == 0, result.stdout + result.stderr
    assert elements(template / 'reordered.gpm') == ['He', 'H']
    before = (template / 'reordered.gpm').read_bytes()
    write(template, 'bad.ini', 'input initial.gpm\noutput reordered.gpm\nneworder 0 0\n')
    result = launch('adjustmtp', 'bad.ini')
    assert result.returncode != 0
    assert (template / 'reordered.gpm').read_bytes() == before
