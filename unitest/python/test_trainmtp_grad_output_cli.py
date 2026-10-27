"""Real dedicated trainmtp gradient publication and collective output failures.

MPI acceptance after building the matching executable:
  python unitest/python/test_trainmtp_grad_output_cli.py \
    --mpi-executable /absolute/build/mpi/imrmlp-mpi --work-root /tmp/trainmtp-grad-mpi
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import pytest

from test_cli_error_paths import run_command
from test_mtp_sgd_cli import FINAL, prepare_sgd_fixture


GRADIENT = 'output_grad_info'


def prepare_grad_fixture(path, failure=None):
    path = prepare_sgd_fixture(path, extra={
        'pot:pottype': 'mtp', 'pot:potfile': 'base.mtp',
    })
    if failure == 'directory':
        (path / GRADIENT).mkdir()
    elif failure == 'full_device':
        if not Path('/dev/full').exists():
            raise RuntimeError('/dev/full is required for this output failure check')
        (path / GRADIENT).symlink_to('/dev/full')
    return path


def gradient_values(path):
    rows = (path / GRADIENT).read_text().splitlines()
    assert len(rows) == 127
    assert all(len(row.split()) == 1 for row in rows)
    values = [float(row) for row in rows]
    assert all(math.isfinite(value) for value in values)
    return values


def check_no_training_output(path, original):
    assert (path / 'base.mtp').read_bytes() == original
    for name in ('output_loss_info', FINAL, 'fitted.mtp', 'fitted.mtp0'):
        assert not (path / name).exists(), name


def compare_reference(path, reference):
    values = gradient_values(path)
    expected = gradient_values(reference)
    assert values == pytest.approx(expected, rel=5e-14, abs=2e-14)
    # E-only loss is (E+100)^2/N for this four-atom, two-species fixture.
    # Each species energy coefficient occurs twice, so its derivative is E+100.
    loss = float((reference / 'output_loss_info').read_text())
    assert values[96:98] == pytest.approx([2 * math.sqrt(loss)] * 2, rel=5e-14)
    assert max(abs(value) for value in values) > 1.
    return values


def test_trainmtp_calc_grad_writes_127_precise_reference_values_without_training(tmp_path):
    path = prepare_grad_fixture(tmp_path / 'dedicated')
    reference = prepare_grad_fixture(tmp_path / 'reference')
    original = (path / 'base.mtp').read_bytes()
    result = run_command('imrmlp', ['trainmtp', 'settings.ini', '--calc-grad', '--num_threads', '1'], path)
    assert result.returncode == 0, result.stdout + result.stderr
    baseline = run_command('imrmlp', ['trainnlpot', 'settings.ini', '--calc-grad', '--num_threads', '1'], reference)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    compare_reference(path, reference)
    check_no_training_output(path, original)
    assert (reference / 'base.mtp').read_bytes() == original


@pytest.mark.parametrize('failure', ['directory', 'full_device'])
def test_trainmtp_calc_grad_output_failures_return_nonzero_and_preserve_source(tmp_path, failure):
    if failure == 'full_device' and not Path('/dev/full').exists():
        pytest.skip('/dev/full unavailable')
    path = prepare_grad_fixture(tmp_path, failure)
    original = (path / 'base.mtp').read_bytes()
    result = run_command('imrmlp', ['trainmtp', 'settings.ini', '--calc-grad', '--num_threads', '1'], path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert GRADIENT in result.stdout + result.stderr
    assert 'Finished' not in result.stdout
    check_no_training_output(path, original)
    if failure == 'directory':
        assert (path / GRADIENT).is_dir()
        assert not list((path / GRADIENT).iterdir())
    else:
        assert (path / GRADIENT).is_symlink()
        assert (path / GRADIENT).readlink() == Path('/dev/full')


def run_mpi_acceptance(executable, work_root, ranks=(2, 3)):
    """Run the same true CLI contract with a one-frame, mostly empty partition."""
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise RuntimeError(f'MPI executable does not exist: {executable}')
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(executable.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '1'
    results = []

    def run(path, rank_count, command):
        cmd = ['mpiexec', '--bind-to', 'none', '-n', str(rank_count), str(executable),
               command, 'settings.ini', '--calc-grad', '--num_threads', '1']
        with (work_root / 'commands.txt').open('a') as output:
            output.write(f'cwd={path}\n' + ' '.join(cmd) + '\n')
        process = subprocess.run(cmd, cwd=path, env=env, capture_output=True, text=True, timeout=45)
        (path / 'stdout.log').write_text(process.stdout)
        (path / 'stderr.log').write_text(process.stderr)
        results.append({'ranks': rank_count, 'command': command,
                        'directory': str(path), 'returncode': process.returncode})
        (work_root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        return process

    reference = prepare_grad_fixture(work_root / 'reference-n1')
    baseline = run(reference, 1, 'trainnlpot')
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    for rank_count in ranks:
        for failure in (None, 'directory', 'full_device'):
            path = prepare_grad_fixture(work_root / f'n{rank_count}-{failure or "success"}', failure)
            original = (path / 'base.mtp').read_bytes()
            result = run(path, rank_count, 'trainmtp')
            if failure is None:
                assert result.returncode == 0, result.stdout + result.stderr
                compare_reference(path, reference)
            else:
                assert result.returncode == 1, result.stdout + result.stderr
                assert GRADIENT in result.stdout + result.stderr
                assert 'Finished' not in result.stdout
                # imrmlp reports the same root-output error on every participating rank.
                assert (result.stdout + result.stderr).count('Training failure on rank 0:') == rank_count
            check_no_training_output(path, original)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpi-executable', required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--ranks', type=int, nargs='+', default=[2, 3])
    arguments = parser.parse_args()
    print(json.dumps(run_mpi_acceptance(arguments.mpi_executable, arguments.work_root, arguments.ranks), indent=2))
