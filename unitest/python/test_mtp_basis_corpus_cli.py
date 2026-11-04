"""Optional MPI/CUDA gates on the real contraction-graph corpus."""
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
from test_gpm_workflow_cli import run

ROOT = Path(__file__).resolve().parents[2]
CASES = ['ps1-7b-517.mtp', 'ps1-5b-5323.mtp', 'ps2-7b-4464.mtp',
         'shapeev/30/potential.mtp_0']


def initialize(path, name):
    if '/' in name:
        external = os.environ.get('MTP_BASIS_CORPUS_ROOT')
        if not external:
            pytest.skip('Set MTP_BASIS_CORPUS_ROOT for the largest external graph')
        source = Path(external) / name
    else:
        source = ROOT / 'unitest/test-data/mtp-basis-corpus' / name
    text = source.read_text()
    assert 'scaling =' not in text
    text = text.replace('potential_name = MTP1m', 'potential_name = MTP1m\nscaling = 1')
    size = int(re.search(r'alpha_scalar_moments\s*=\s*(\d+)', text)[1])
    (path / 'template.mtp').write_text(text)
    (path / 'empty.ini').write_text('')
    (path / 'atoms.xyz').write_text(
        '4\nLattice="6.4 0 0 0 6.4 0 0 0 6.4" pbc="T T T" '
        'Properties=species:S:1:pos:R:3\n'
        'Al 2 2 2\nNi 4.2 2.3 2.1\nAl 2.2 4.3 2.4\nNi 2.4 2.2 4.6\n')
    run(path, 'initmtpm', '--input=template.mtp', '--type=mtp', '--elements', 'Al', 'Ni',
        '--output=model.gpm', '--rand-lb=.04', '--rand-ub=.09', f'--rand-scale={1 / size:.17g}')
    (path / 'template.mtp').unlink()
    return (path / 'model.gpm').read_bytes()


def values(path):
    lines = path.read_text().splitlines()
    count = int(lines[lines.index('Size') + 1])
    first = next(i for i, line in enumerate(lines) if line.startswith('AtomData:'))
    columns = lines[first].split()[1:]
    forces = [float(row.split()[columns.index(k)])
              for row in lines[first + 1:first + count + 1] for k in ['fx', 'fy', 'fz']]
    stress = next(i for i, line in enumerate(lines) if line.startswith('PlusStress:'))
    result = np.array([float(lines[lines.index('Energy') + 1]), *forces,
                       *map(float, lines[stress + 1].split())])
    assert np.isfinite(result).all()
    assert np.max(np.abs(forces)) > 1e-12
    return result


def labels(path, model, output):
    run(path, 'efs', 'atoms.xyz', output, '--potfile=' + model, '--pottype=auto',
        '--settings=empty.ini', '--index=absent.map',
        '--input-format=extxyz', '--output-format=mlpcfg')
    return values(path / output)


def execute(executable, path, args, mpi=False):
    executable = Path(executable).resolve()
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(executable.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
    command = (['mpiexec', '-n', '2'] if mpi else []) + [str(executable), *args, '--num_threads=1']
    result = subprocess.run(command, cwd=path, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.extended
@pytest.mark.parametrize('name', CASES)
def test_mpi_real_corpus_reorder_and_extend(tmp_path, name):
    executable = os.environ.get('MTP_WORKFLOW_MPI_EXECUTABLE')
    if not executable:
        pytest.skip('Set MTP_WORKFLOW_MPI_EXECUTABLE to an MPI CLI')
    before = initialize(tmp_path, name)
    expected = labels(tmp_path, 'model.gpm', 'before.cfg')
    execute(executable, tmp_path, ['adjustmtp', '--input=model.gpm', '--output=after.gpm',
            '--basis-order=level-group', '--append-elements', 'Cu', '--new-species-init=clone',
            '--clone-from', '1', '--neworder', '2', '0', '1'], mpi=True)
    assert (tmp_path / 'model.gpm').read_bytes() == before
    np.testing.assert_allclose(labels(tmp_path, 'after.gpm', 'after.cfg'), expected,
                               atol=1e-10, rtol=2e-9)


@pytest.mark.extended
@pytest.mark.parametrize('name', CASES)
def test_cuda_real_corpus_reorder_and_efs(tmp_path, name):
    executable = os.environ.get('MTP_WORKFLOW_GPU_EXECUTABLE')
    if not executable:
        pytest.skip('Set MTP_WORKFLOW_GPU_EXECUTABLE to a CUDA CLI')
    before = initialize(tmp_path, name)
    expected = labels(tmp_path, 'model.gpm', 'before.cfg')
    execute(executable, tmp_path, ['adjustmtp', '--input=model.gpm', '--output=after.gpm',
            '--basis-order=level-group', '--append-elements', 'Cu', '--new-species-init=clone',
            '--clone-from', '1', '--neworder', '2', '0', '1'])
    assert (tmp_path / 'model.gpm').read_bytes() == before
    np.testing.assert_allclose(labels(tmp_path, 'after.gpm', 'after-cpu.cfg'), expected,
                               atol=1e-10, rtol=2e-9)
    execute(executable, tmp_path, ['efs', 'atoms.xyz', 'gpu.cfg', '--potfile=after.gpm',
            '--pottype=auto', '--settings=empty.ini', '--index=absent.map', '--device=cuda',
            '--input-format=extxyz', '--output-format=mlpcfg'])
    np.testing.assert_allclose(values(tmp_path / 'gpu.cfg'), expected, atol=1e-9, rtol=1e-7)
