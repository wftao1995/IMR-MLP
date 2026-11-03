"""Bidirectional model conversion and portable legacy MaxVol exports."""
import struct

import pytest

from test_cli_error_paths import run_command
from test_gpm_workflow_cli import prepare_fixture, run, sample, select


def test_convert_model_without_state_and_inference(tmp_path):
    prepare_fixture(tmp_path)
    run(tmp_path, 'model_convert', '--input=base.mtp', '--type=mtp', '--index=index.map', '--output=model.gpm')
    run(tmp_path, 'model_convert', '--input=model.gpm', '--legacy', '--output=restored.mtp')
    assert (tmp_path / 'restored.mtp').read_text().startswith('MTP\n')
    assert 'H' in (tmp_path / 'restored.mtp.index.map').read_text()
    assert not (tmp_path / 'restored.mtp.sel').exists()
    run(tmp_path, 'model_convert', '--input=restored.mtp', '--type=mtp', '--index=restored.mtp.index.map', '--output=again.gpm')
    run(tmp_path, 'model_validate', '--input=again.gpm')


@pytest.mark.parametrize('mode', ['full', 'stream', 'online'])
def test_native_state_survives_removing_gpm_and_sources(tmp_path, mode):
    prepare_fixture(tmp_path)
    select(tmp_path, f'--select-mode={mode}', '--state-output=session.gpm')
    sample(tmp_path, '--potfile=session.gpm', output='before.xyz')
    run(tmp_path, 'model_convert', '--input=session.gpm', '--legacy', '--output=restored.mtp')
    assert struct.unpack('<ii', (tmp_path / 'restored.mtp.sel').read_bytes()[:8]) == (0x41435456, 10)
    for name in ['session.gpm', 'base.mtp', 'index.map']:
        (tmp_path / name).unlink()
    sample(tmp_path, '--potfile=restored.mtp', '--pottype=mtp', '--state-file=restored.mtp.sel',
           '--index=restored.mtp.index.map', '--dtype=pot_efs', output='after.xyz')
    assert (tmp_path / 'before.xyz').read_bytes() == (tmp_path / 'after.xyz').read_bytes()
    sample(tmp_path, '--settings=restored.mtp.ini', output='from-settings.xyz')
    assert (tmp_path / 'before.xyz').read_bytes() == (tmp_path / 'from-settings.xyz').read_bytes()
    run(tmp_path, 'maxvol_extend', '--settings=restored.mtp.ini', '--candidates=train.cfg', '--input-format=mlpcfg',
        '--output-format=extxyz', '--output=extended.xyz', '--state-output=continued.sel')
    assert (tmp_path / 'continued.sel').is_file()


def test_native_export_failure_keeps_model_and_sidecars(tmp_path):
    prepare_fixture(tmp_path)
    select(tmp_path, '--state-output=session.gpm')
    for name in ['restored.mtp', 'restored.mtp.index.map']:
        (tmp_path / name).write_bytes(b'old output')
    (tmp_path / 'restored.mtp.sel').mkdir()
    result = run_command('imrmlp', ['model_convert', '--input=session.gpm', '--legacy', '--output=restored.mtp'], tmp_path)
    assert result.returncode != 0
    assert 'directory' in result.stderr
    for name in ['restored.mtp', 'restored.mtp.index.map']:
        assert (tmp_path / name).read_bytes() == b'old output'
    assert not list(tmp_path.glob('.native-export-*'))


def test_inference_removes_old_state_and_resume_requires_state(tmp_path):
    prepare_fixture(tmp_path)
    select(tmp_path, '--state-output=session.gpm')
    run(tmp_path, 'model_convert', '--input=session.gpm', '--legacy', '--output=restored.mtp')
    assert (tmp_path / 'restored.mtp.sel').exists()
    run(tmp_path, 'model_convert', '--input=session.gpm', '--legacy', '--output=restored.mtp', '--profile=inference')
    assert not (tmp_path / 'restored.mtp.sel').exists()
    run(tmp_path, 'model_convert', '--input=base.mtp', '--output=model.gpm')
    result = run_command('imrmlp', ['model_convert', '--input=model.gpm', '--legacy', '--output=restored.mtp', '--profile=resume'], tmp_path)
    assert result.returncode != 0
    assert 'requires AL state' in result.stderr


def test_legacy_rejects_gpm_destination(tmp_path):
    prepare_fixture(tmp_path)
    run(tmp_path, 'model_convert', '--input=base.mtp', '--output=model.gpm')
    before = (tmp_path / 'model.gpm').read_bytes()
    result = run_command('imrmlp', ['model_convert', '--input=model.gpm', '--legacy', '--output=model.gpm'], tmp_path)
    assert result.returncode != 0
    assert (tmp_path / 'model.gpm').read_bytes() == before


@pytest.mark.parametrize('family', ['hybrid', 'hybrid_scale'])
def test_complete_hybrid_export_is_portable(tmp_path, family):
    import numpy as np
    def read_labels(path):
        labels=[]
        for frame in path.read_text().split('BEGIN_CFG')[1:]:
            lines=frame.splitlines();size=int(lines[lines.index('Size')+1])
            atom_line=next(i for i,line in enumerate(lines) if line.startswith('AtomData:'))
            keys=lines[atom_line].split()[1:]
            rows=[list(map(float,line.split())) for line in lines[atom_line+1:atom_line+1+size]]
            forces=np.array([[row[keys.index(k)] for k in ('fx','fy','fz')] for row in rows])
            stress_line=next(i for i,line in enumerate(lines) if line.startswith('PlusStress:'))
            labels.append((float(lines[lines.index('Energy')+1]),forces,np.fromstring(lines[stress_line+1],sep=' ')))
        return labels

    prepare_fixture(tmp_path)
    children = ('pot:pot1:pottype mtp\npot:pot1:potfile base.mtp\npot:pot1:train false\n'
                'pot:pot2:pottype zbl\npot:pot2:z 1\npot:pot2:r0 1\npot:pot2:r1 3\n')
    if family == 'hybrid_scale':
        children = ('pot:feature min_distance\npot:boundary1 1 3\n'
                    'pot:pot1:pottype mtp\npot:pot1:potfile base.mtp\n'
                    'pot:pot2:pottype mtp\npot:pot2:potfile base.mtp\n')
    (tmp_path / 'hybrid.ini').write_text(f'pot:pottype {family}\npot:npot 2\npot:typemap index.map\n' + children)
    run(tmp_path, 'model_convert', '--settings=hybrid.ini', '--output=hybrid.gpm')
    run(tmp_path, 'efs', 'input.xyz', 'before.cfg', '--potfile=hybrid.gpm', f'--pottype={family}',
        '--input-format=extxyz', '--output-format=mlpcfg')
    run(tmp_path, 'model_convert', '--input=hybrid.gpm', '--legacy', '--output=exported.model')
    assert (tmp_path / 'exported.model.pot1').is_file()
    assert (tmp_path / 'exported.model.pot2').is_file()
    if family == 'hybrid':
        assert 'pot1:train false' in (tmp_path / 'exported.model').read_text()
    moved = tmp_path / 'moved'
    moved.mkdir()
    for path in tmp_path.glob('exported.model*'):
        path.rename(moved / path.name)
    for name in ['base.mtp', 'hybrid.gpm', 'hybrid.ini', 'index.map']:
        (tmp_path / name).unlink()
    run(moved, 'efs', '../input.xyz', 'after.cfg', '--settings=exported.model.ini',
        '--input-format=extxyz', '--output-format=mlpcfg')
    before, after = read_labels(tmp_path / 'before.cfg'), read_labels(moved / 'after.cfg')
    assert len(before) == len(after) > 0
    for expected, actual in zip(before, after):
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)


def test_truncated_gpm_does_not_replace_existing_output(tmp_path):
    prepare_fixture(tmp_path)
    run(tmp_path, 'model_convert', '--input=base.mtp', '--output=model.gpm')
    raw = (tmp_path / 'model.gpm').read_bytes()
    (tmp_path / 'model.gpm').write_bytes(raw[:len(raw) // 2])
    for name in ['old.mtp', 'old.mtp.sel', 'old.mtp.index.map', 'old.mtp.ini']:
        (tmp_path / name).write_bytes(b'keep')
    result = run_command('imrmlp', ['model_convert', '--input=model.gpm', '--legacy', '--output=old.mtp'], tmp_path)
    assert result.returncode != 0
    for name in ['old.mtp', 'old.mtp.sel', 'old.mtp.index.map', 'old.mtp.ini']:
        assert (tmp_path / name).read_bytes() == b'keep'
    assert not list(tmp_path.glob('.native-export-*'))


@pytest.mark.parametrize('ranks', [1, 2])
def test_mpi_native_publication_and_collective_failure(tmp_path, ranks):
    import os
    from pathlib import Path
    import shutil
    import subprocess

    executable = os.environ.get('MODEL_CONVERT_MPI_EXECUTABLE')
    if not executable or not shutil.which('mpiexec'):
        pytest.skip('Set MODEL_CONVERT_MPI_EXECUTABLE to a matching MPI build')
    executable = Path(executable).resolve()
    prepare_fixture(tmp_path)
    select(tmp_path, '--state-output=session.gpm')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(executable.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = '1'
    for output, success in [('restored.mtp', True), ('session.gpm', False)]:
        result = subprocess.run(['mpiexec', '-n', str(ranks), str(executable), 'model_convert',
                                 '--input=session.gpm', '--legacy', '--output=' + output],
                                cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        if success:
            assert result.stdout.count('Saved restored.mtp') == 1
        else:
            assert '--legacy requires' in result.stderr
    sample(tmp_path, '--settings=restored.mtp.ini')
    assert not list(tmp_path.glob('.native-export-*'))
