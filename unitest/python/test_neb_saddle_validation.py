"""Exercise saddle certification on known spectra, independently of native EFS."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('ase')
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('neb_saddle_check', ROOT/'scripts/validation/check_neb_saddle.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class OneAtom:
    def __len__(self):
        return 1


class QuadraticEnthalpy:
    def __init__(self, negative_modes):
        self.diagonal = np.r_[np.arange(1., 7.), np.zeros(3)]
        self.diagonal[:negative_modes] *= -1

    def encode(self, saddle):
        return np.zeros(9)

    def decode(self, point, saddle):
        return point

    def evaluate(self, point):
        return {'gradient': self.diagonal*point}


@pytest.mark.parametrize('negative_modes,expected', [(0, 'FAIL'), (1, 'PASS'), (2, 'FAIL')])
def test_low_modes_distinguish_minima_and_saddle_indices(tmp_path, negative_modes, expected):
    result = validation.low_modes(QuadraticEnthalpy(negative_modes), OneAtom(), tmp_path, 3000)
    assert result['status'] == expected
    assert result['removed_translation_modes'] == 3
    for record in result['steps']:
        assert record['negative_low_modes'] == negative_modes
        assert record['uncertain_low_modes'] == 0
    saved = np.load(tmp_path/'low_modes.npz')
    assert np.linalg.norm(saved['translations'].T@saved['modes_h']) < 1e-12


def test_hessian_product_budget_cannot_certify_a_saddle(tmp_path):
    result = validation.low_modes(QuadraticEnthalpy(1), OneAtom(), tmp_path, 1)
    assert result['status'] == 'INDETERMINATE'
    assert 'budget' in result['reason']
