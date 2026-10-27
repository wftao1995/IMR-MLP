"""Small real trainmtp SGD runs; no Python extension or external dataset needed."""
from pathlib import Path
import math
import re
import shutil

import pytest

from test_cli_error_paths import ROOT, run_command


MODEL = ROOT / 'unitest' / 'test-data' / 'portable-mtp' / 'plain.mtp'
FINAL = '_Trained_Potential_.mtp'


def prepare_sgd_fixture(path, scheduler='none', epochs=2, extra=None):
    """Also usable to prepare a one-frame empty-shard MPI CLI smoke run."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MODEL, path / 'base.mtp')
    (path / 'index.map').write_text('H 0\nHe 1\n')
    (path / 'train.cfg').write_text(
        'BEGIN_CFG\nSize\n4\nSupercell\n12 0 0\n0 12 0\n0 0 12\n'
        'AtomData: id type cartes_x cartes_y cartes_z fx fy fz\n'
        '1 0 2 2 2 0 0 0\n2 1 4.1 2.2 2 0 0 0\n'
        '3 0 2.3 4.2 2.4 0 0 0\n4 1 5 4.8 3.1 0 0 0\n'
        'Energy\n-100\nEND_CFG\n')
    settings = {
        'potfile': 'base.mtp', 'trainset': 'train.cfg',
        'trainset-format': 'mlpcfg', 'index': 'index.map',
        'fit:energy_weight': '1', 'fit:force_weight': '0',
        'fit:stress_weight': '0', 'fit:penalty_weight': '0',
        'fit:weight_L2': '0', 'fit:bfgs_rescale': 'false',
        'fit:do_orthogonalization': 'false', 'fit:epoch_limit': str(epochs),
        'fit:batch_size': '64', 'fit:learning_rate': '1e-7',
        'fit:momentum': '0', 'fit:save_every': '100',
        'fit:save_to': 'fitted.mtp', 'fit:lr_scheduler': scheduler,
        'fit:lr_decay_steps': '1', 'fit:lr_decay_rate': '.5',
        'fit:lr_final': '0',
    }
    settings.update(extra or {})
    (path / 'settings.ini').write_text(''.join(f'{key} {value}\n' for key, value in settings.items()))
    return path


def run_sgd(path):
    return run_command('imrmlp', ['trainmtp', 'settings.ini', '--do-sgd', '--num_threads', '1'], path)


def coefficients(path):
    text = Path(path).read_text()
    radial = text.split('radial_coeffs', 1)[1].split('alpha_moments_count', 1)[0]
    groups = re.findall(r'\{([^{}]+)\}', radial)
    groups.extend(re.search(rf'{name}\s*=\s*\{{([^}}]+)\}}', text).group(1)
                  for name in ('species_coeffs', 'moment_coeffs'))
    return [float(value) for group in groups for value in group.split(',')]


@pytest.mark.parametrize('scheduler', ['none', 'step', 'cosine', 'exponential'])
def test_trainmtp_sgd_valid_schedulers_save_and_reload_a_changed_model(tmp_path, scheduler):
    prepare_sgd_fixture(tmp_path, scheduler)
    original = (tmp_path / 'base.mtp').read_bytes()
    before = coefficients(tmp_path / 'base.mtp')
    result = run_sgd(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Epoch: 1' in result.stdout
    after = coefficients(tmp_path / FINAL)
    assert len(after) == len(before)
    assert all(math.isfinite(value) for value in after)
    assert max(abs(a-b) for a, b in zip(after, before)) > 1e-10
    assert coefficients(tmp_path / 'fitted.mtp') == after
    assert (tmp_path / 'base.mtp').read_bytes() == original
    # Verify the actual published model can be loaded by another CLI command.
    reload = run_command('imrmlp', [
        'efs', 'train.cfg', 'reloaded.cfg', '--input-format', 'mlpcfg',
        '--output-format', 'mlpcfg', '--pottype', 'mtp',
        '--potfile', FINAL, '--index', 'index.map', '--num_threads', '1',
    ], tmp_path)
    assert reload.returncode == 0, reload.stdout + reload.stderr
    text = (tmp_path / 'reloaded.cfg').read_text()
    assert 'BEGIN_CFG' in text
    assert math.isfinite(float(re.search(r'\bEnergy\s*\n\s*(\S+)', text).group(1)))


@pytest.mark.parametrize('scheduler,extra,message', [
    ('cosnie', {}, 'Unknown SGD learning rate scheduler'),
    ('step', {'fit:lr_decay_steps': '0'}, 'positive decay steps'),
])
def test_trainmtp_sgd_invalid_scheduler_exits_cleanly_without_publishing(tmp_path, scheduler, extra, message):
    prepare_sgd_fixture(tmp_path, scheduler, extra=extra)
    original = (tmp_path / 'base.mtp').read_bytes()
    result = run_sgd(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stdout + result.stderr
    assert (tmp_path / 'base.mtp').read_bytes() == original
    for name in (FINAL, 'fitted.mtp', 'fitted.mtp0'):
        assert not (tmp_path / name).exists()


def test_trainmtp_sgd_none_ignores_unused_scheduler_controls(tmp_path):
    plain = prepare_sgd_fixture(tmp_path / 'plain')
    unused = prepare_sgd_fixture(tmp_path / 'unused', extra={
        'fit:lr_decay_steps': '0', 'fit:lr_decay_rate': '-1',
        'fit:lr_final': '-1', 'fit:lr_warmup_steps': '-1',
        'fit:lr_warmup_init': '-1',
    })
    for path in (plain, unused):
        result = run_sgd(path)
        assert result.returncode == 0, result.stdout + result.stderr
    assert coefficients(plain / FINAL) == coefficients(unused / FINAL)


def test_trainmtp_sgd_zero_epochs_preserves_coefficients_and_skips_core_checkpoints(tmp_path):
    prepare_sgd_fixture(tmp_path, 'cosine', epochs=0)
    before = coefficients(tmp_path / 'base.mtp')
    result = run_sgd(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Epoch:' not in result.stdout
    # trainmtp retains its outer publication, while the SGD core is an empty operation.
    assert coefficients(tmp_path / FINAL) == before
    assert not (tmp_path / 'fitted.mtp').exists()
    assert not (tmp_path / 'fitted.mtp0').exists()
