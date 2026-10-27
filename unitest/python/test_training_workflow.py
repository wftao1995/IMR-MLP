"""Three-set fitting and validation-only L2 selection through real entry points."""
import re

import numpy as np
import pytest

from test_cli_error_paths import run_command
from training_cli_fixture import prepare_fixture
from test_nonlinear_contract_cli import write_settings


def configure(path, **extra):
    prepare_fixture(path)
    (path / 'valid.cfg').write_bytes((path / 'train.cfg').read_bytes())
    (path / 'test.cfg').write_bytes((path / 'train.cfg').read_bytes())
    settings = {
        'validset': 'valid.cfg', 'validset-format': 'mlpcfg',
        'testset': 'test.cfg', 'testset-format': 'mlpcfg',
        'fit:bfgs_maxiteration': '2', 'fit:penalty_weight': '0',
        'fit:bfgs_grad_tol': '0', 'fit:weight_L2': '.001',
        **extra,
    }
    write_settings(path, settings)


def train(path):
    result = run_command('imrmlp', ['trainnlpot', 'settings.ini', '--do-lbfgs', '--num_threads', '1'], path)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def coefficients(path):
    text = (path / 'final.mtp').read_text()
    radial = re.search(r'radial_coeffs\s+0-0\s*\{([^}]+)\}', text).group(1)
    return np.array([*map(float, radial.split(',')),
                     float(re.search(r'species_coeffs = \{([^}]+)\}', text).group(1)),
                     float(re.search(r'moment_coeffs = \{([^}]+)\}', text).group(1))])


@pytest.mark.parametrize('refit', [False, True])
@pytest.mark.parametrize('search', [False, True])
def test_heldout_labels_never_change_fitted_coefficients(tmp_path, refit, search):
    paths = [tmp_path / 'original', tmp_path / 'changed']
    results = []
    for i, path in enumerate(paths):
        configure(path, **{'fit:refit_train_valid': str(refit).lower(),
                           'fit:regularization_search': str(search).lower()})
        if i:
            text = (path / 'test.cfg').read_text()
            text = re.sub(r'(\nEnergy\s*\n)\s*[^\n]+', r'\g<1>100000', text)
            (path / 'test.cfg').write_text(text)
        results.append(train(path))
        assert 'Final test energy_RMSE_eV_atom=' in results[-1].stdout
        assert (path / 'fitted.mtp').exists()
        if search:
            assert 'Selected L2 multiplier:' in results[-1].stdout
    np.testing.assert_array_equal(coefficients(paths[0]), coefficients(paths[1]))
    first = float(re.search(r'Final test energy_RMSE_eV_atom=([^\s]+)', results[0].stdout).group(1))
    second = float(re.search(r'Final test energy_RMSE_eV_atom=([^\s]+)', results[1].stdout).group(1))
    assert second > first + 1000


def test_l2_candidates_match_independent_runs(tmp_path):
    search = tmp_path / 'search'
    configure(search, **{'fit:regularization_search': 'true'})
    train(search)
    for index, multiplier in enumerate((.1, 1., 10.)):
        path = tmp_path / f'independent-{index}'
        configure(path, **{'fit:weight_L2': str(.001 * multiplier)})
        train(path)
        candidates = list(search.glob(f'fitted.mtp.workflow-*.candidate-{index}'))
        assert len(candidates) == 1
        assert candidates[0].read_bytes() == (path / 'final.mtp').read_bytes()


def test_refit_removes_validation_callbacks_and_keeps_budget(tmp_path):
    configure(tmp_path, **{'fit:refit_train_valid': 'true'})
    result = train(tmp_path)
    assert 'training_configurations\t12' in result.stdout
    # First stage may validate; the final combined-data run must have no heldout callback.
    final = result.stdout.split('# training_configurations\t12', 1)[1]
    assert 'Validation at accepted iteration' not in final


def test_python_heldout_metrics_and_refit(tmp_path, monkeypatch):
    pytest.importorskip('imrmlp._imrmlp')
    import imrmlp
    from training_cli_fixture import MODEL
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'base.mtp').write_text(MODEL)
    potential = imrmlp.Potential(tmp_path / 'base.mtp', 'mtp')
    atom = imrmlp.Atoms(numbers=[1, 1, 1], elements=['H'],
                      positions=[[6, 6, 6], [8, 6.1, 6.2], [6.3, 8.9, 6.4]],
                      cell=np.eye(3) * 20, pbc=False)
    potential.calculate(atom)
    test = atom.copy()
    trainer = imrmlp.NonlinearTrainer(max_iterations=2, bfgs_m=2, penalty_weight=0,
                                    refit_train_valid='true', save_to='fitted.mtp')
    result = trainer.train(potential, [atom], [atom], test_set=[test])
    assert result.valid_erms is None
    assert result.test_erms == pytest.approx(0, abs=1e-12)
    assert result.test_frms == pytest.approx(0, abs=1e-12)
    assert result.test_stress_gpa == pytest.approx(0, abs=1e-12)
    assert result.selected_l2_multiplier is None
    assert (tmp_path / 'fitted.mtp').exists()
