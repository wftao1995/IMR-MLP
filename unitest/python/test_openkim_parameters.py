"""Installed parameterized KIM model verifies public factory settings end to end."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from test_cli_error_paths import BUILD


@pytest.fixture
def parameters(tmp_path):
    flags = BUILD / '.flags'
    if flags.exists() and '-DUSE_KIM' not in flags.read_text().split():
        pytest.skip('This IMRMLP build has USE_KIM=0')
    mapping = tmp_path / 'index.map'
    mapping.write_text('Ar\n')
    return {'pottype': 'openkim', 'kim_model': 'ex_model_Ar_P_LJ',
            'typemap': mapping, 'kim_parameter:epsilon': '.02',
            'kim_parameter:sigma': '3', 'kim_parameter:cutoff': '6'}


def pair(distance):
    return imrmlp.Atoms(numbers=[18, 18], elements=['Ar'],
                      positions=[[0, 0, 0], [distance, 0, 0]], pbc=False)


@pytest.mark.extended
def test_openkim_factory_applies_parameter_arrays_before_evaluation(parameters):
    pot = imrmlp.Potential.from_settings(parameters)
    a, b = pair(4), pair(5)
    pot.calculate(a); pot.calculate(b)
    def lj(r):
        x = (3 / r) ** 6
        return 4 * .02 * (x*x - x)
    assert a.energy - b.energy == pytest.approx(lj(4) - lj(5), abs=1e-12)
    x = (3 / 4) ** 6
    derivative = 24 * .02 * (x - 2*x*x) / 4
    np.testing.assert_allclose(a.forces, [[derivative, 0, 0], [-derivative, 0, 0]], atol=1e-12)
    outside = pair(6.5)
    pot.calculate(outside)
    assert outside.energy == 0
    np.testing.assert_array_equal(outside.forces, np.zeros((2, 3)))
    parameters['kim_parameter:cutoff'] = '7'
    imrmlp.Potential.from_settings(parameters).calculate(outside)
    assert abs(outside.energy) > 1e-8


@pytest.mark.parametrize('name,value', [('epsilon', '.02 .03'), ('epsilon', 'nan'),
                                       ('epsilon', ''), ('not_a_parameter', '1')])
@pytest.mark.extended
def test_openkim_factory_rejects_bad_parameters(parameters, name, value):
    parameters['kim_parameter:' + name] = value
    with pytest.raises(RuntimeError, match='(parameter|expected)'):
        imrmlp.Potential.from_settings(parameters)
