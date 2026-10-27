"""Ordinary MTP CG plus one terminal refit through independent CLI processes."""
import hashlib
import json
import re

import numpy as np
import pytest
from test_cli_error_paths import run_command
from training_cli_fixture import evaluate, prepare_fixture, radial_coefficients
from test_nonlinear_contract_cli import write_settings


OPTIONS = {
    'fit:bfgs_maxiteration': '1', 'fit:bfgs_grad_tol': '0',
    'fit:bfgs_conv_tol': '0', 'fit:bfgs_step_tol': '0',
    'fit:parameter_scaling': 'none', 'fit:save_every': '1',
    'fit:penalty_weight': '.00001', 'fit:weight_L1': '0', 'fit:weight_L2': '.0002',
    'fit:linear_refit_report': 'refit.json',
}


def run(path, *flags):
    return run_command('imrmlp', ['trainnlpot', 'settings.ini', '--num_threads', '1', *flags], path)


def hashes(path):
    return {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
            for name in ('base.mtp', 'train.cfg')}


def linear_coefficients(path):
    text = path.read_text()
    return np.array([float(value) for key in ('species_coeffs', 'moment_coeffs')
                     for value in re.search(rf'{key} = \{{([^}}]+)\}}', text).group(1).split(',')])


def cg_trace(result):
    return re.findall(r'(?m)^CG (?:accepted iteration .*|report:.*)$', result.stdout)


def assert_same_efs(first, second):
    assert len(first) == len(second)
    for left, right in zip(first, second):
        for a, b in zip(left, right):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)


def test_cg_terminal_refit_matches_manual_cli_chain_and_preserves_pure_cg(tmp_path):
    combined, manual = tmp_path/'combined', tmp_path/'manual'
    prepare_fixture(combined)
    prepare_fixture(manual)
    before = hashes(combined)
    write_settings(combined, {**OPTIONS, 'fit:linear_refit': 'sr'})
    result = run(combined, '--do-cg')
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('Linear refit accepted:') == 1
    assert 'CG report: iterations=1' in result.stdout
    assert 'termination_type=5' in result.stdout
    assert 'not converged' in result.stdout + result.stderr
    assert (combined/'fitted.mtp.cg.snapshot').read_bytes() == (combined/'fitted.mtp.pre-refit').read_bytes()
    assert (combined/'fitted.mtp.cg.snapshot.iter-000001.mtp').is_file()
    assert not (combined/'fitted.mtp.iter-000001.mtp').exists()
    report = json.loads((combined/'refit.json').read_text())
    assert report['status'] == 'accepted'
    assert report['mode'] == 'sr'
    assert report['after'] < report['before']
    assert report['after'] == pytest.approx(report['predicted'], rel=1e-9, abs=1e-12)
    assert report['parameter_indices']  # A real affine solve, with nonzero native L2.
    assert max(report['l2']) > 0

    # The legacy pure-CG request uses the omitted/default off setting. Its trace
    # and endpoint must match the combined run's pre-refit snapshot exactly.
    write_settings(manual, OPTIONS)
    pure = run(manual, '--do-cg')
    assert pure.returncode == 0, pure.stdout + pure.stderr
    assert pure.stdout.count('Linear refit accepted:') == 0
    assert cg_trace(pure) == cg_trace(result)
    assert (manual/'fitted.mtp').read_bytes() == (manual/'final.mtp').read_bytes()
    assert (manual/'final.mtp').read_bytes() == (combined/'fitted.mtp.cg.snapshot').read_bytes()
    assert (manual/'fitted.mtp.iter-000001.mtp').is_file()
    assert not (manual/'fitted.mtp.cg.snapshot').exists()
    assert not (manual/'fitted.mtp.pre-refit').exists()
    assert not (manual/'refit.json').exists()
    np.testing.assert_array_equal(radial_coefficients(combined/'final.mtp'),
                                  radial_coefficients(combined/'fitted.mtp.cg.snapshot'))
    assert np.linalg.norm(linear_coefficients(combined/'final.mtp') -
                          linear_coefficients(combined/'fitted.mtp.cg.snapshot')) > 1e-9

    # A separate public invocation performs only the linear solve at that exact
    # saved CG endpoint, with the same data/weights/regularization.
    write_settings(manual, {**OPTIONS, 'fit:linear_refit': 'sr',
                           'pot:potfile': 'final.mtp', 'output': 'manual-final.mtp',
                           'fit:save_to': 'refitted.mtp'})
    refit = run(manual, '--linear-refit-only')
    assert refit.returncode == 0, refit.stdout + refit.stderr
    assert refit.stdout.count('Linear refit accepted:') == 1
    assert not cg_trace(refit)
    np.testing.assert_array_equal(radial_coefficients(combined/'final.mtp'),
                                  radial_coefficients(manual/'manual-final.mtp'))
    np.testing.assert_allclose(linear_coefficients(combined/'final.mtp'),
                               linear_coefficients(manual/'manual-final.mtp'), rtol=1e-11, atol=1e-12)
    final = evaluate(combined, 'final.mtp', 'final-efs.cfg')
    assert_same_efs(final, evaluate(combined, 'fitted.mtp', 'fitted-efs.cfg'))
    assert_same_efs(final, evaluate(manual, 'manual-final.mtp', 'manual-efs.cfg'))
    # Recompute the full final objective via another independent process; it
    # must describe the conditional result rather than the old CG checkpoint.
    write_settings(combined, {**OPTIONS, 'pot:potfile': 'final.mtp', 'fit:linear_refit': 'off'})
    check = run(combined, '--calc-grad')
    assert check.returncode == 0, check.stdout + check.stderr
    loss = float((combined/'output_loss_info').read_text())
    assert loss == pytest.approx(report['after'], rel=1e-9, abs=1e-12)
    assert hashes(combined) == before


@pytest.mark.parametrize('failure', ['report', 'fit_publication', 'cli_publication'])
def test_cg_terminal_refit_cli_failures_preserve_prior_final_and_checkpoint(tmp_path, failure):
    prepare_fixture(tmp_path)
    before = hashes(tmp_path)
    options = {**OPTIONS, 'fit:linear_refit': 'sr'}
    previous = 'previous successful CLI model\n'
    (tmp_path/'final.mtp').write_text(previous)
    if failure == 'report':
        options['fit:linear_refit_report'] = 'missing/report.json'
        (tmp_path/'fitted.mtp').write_text('previous optimizer model\n')
    elif failure == 'fit_publication':
        (tmp_path/'fitted.mtp').mkdir()
        (tmp_path/'fitted.mtp'/'marker').write_text('preserved directory')
    else:
        (tmp_path/'previous.mtp').write_text(previous)
        (tmp_path/'final.mtp').unlink()
        (tmp_path/'final.mtp').symlink_to('previous.mtp')
    write_settings(tmp_path, options)
    result = run(tmp_path, '--do-cg')
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'Finished' not in result.stdout
    assert 'CG report: iterations=1' in result.stdout
    assert (tmp_path/'final.mtp').read_text() == previous
    assert (tmp_path/'fitted.mtp.cg.snapshot').read_bytes() == (tmp_path/'fitted.mtp.pre-refit').read_bytes()
    # The accepted pure-CG checkpoint remains a genuine reloadable model.
    predictions = evaluate(tmp_path, 'fitted.mtp.cg.snapshot', 'checkpoint-efs.cfg')
    assert all(np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()
               for energy, forces, stress in predictions)
    if failure == 'report':
        assert 'linear refit report' in (result.stdout + result.stderr).lower()
        assert result.stdout.count('Linear refit accepted:') == 0
        assert (tmp_path/'fitted.mtp').read_text() == 'previous optimizer model\n'
        assert not (tmp_path/'missing').exists()
        assert not (tmp_path/'refit.json').exists()
    else:
        assert result.stdout.count('Linear refit accepted:') == 1
        assert json.loads((tmp_path/'refit.json').read_text())['status'] == 'accepted'
        if failure == 'fit_publication':
            assert (tmp_path/'fitted.mtp').is_dir()
            assert (tmp_path/'fitted.mtp'/'marker').read_text() == 'preserved directory'
        else:
            assert (tmp_path/'final.mtp').is_symlink()
            assert (tmp_path/'previous.mtp').read_text() == previous
            accepted = evaluate(tmp_path, 'fitted.mtp', 'accepted-efs.cfg')
            assert all(np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()
                       for energy, forces, stress in accepted)
            np.testing.assert_array_equal(radial_coefficients(tmp_path/'fitted.mtp'),
                                          radial_coefficients(tmp_path/'fitted.mtp.cg.snapshot'))
            assert np.linalg.norm(linear_coefficients(tmp_path/'fitted.mtp') -
                                  linear_coefficients(tmp_path/'fitted.mtp.cg.snapshot')) > 1e-9
    assert hashes(tmp_path) == before
    assert not list(tmp_path.glob('.imrmlp-*'))
