"""ASE consumes only the current potential's computed atomic charges."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
pytest.importorskip('ase')
from ase import Atoms as ASEAtoms
from ase.calculators.calculator import PropertyNotImplementedError

from imrmlp import Potential
from imrmlp.calculator import IMRMLPCalculator

def test_lj_missing_computed_charges_is_not_initial_charge_fallback():
    potential = Potential.from_settings({'pottype': 'lj612', 'ntypes': 1,
                                         '0-0': '1 1', 'cut:0-0': 3})
    source = ASEAtoms('H2', positions=[[0., 0., 0.], [1.3, 0., 0.]], pbc=False)
    source.set_initial_charges([.4, -.2])
    source.calc = IMRMLPCalculator(potential=potential, elements=['H'])
    with pytest.raises(PropertyNotImplementedError):
        source.get_charges()
    assert 'charges' not in source.calc.results
    np.testing.assert_array_equal(source.get_initial_charges(), [.4, -.2])
    assert np.isfinite(source.get_potential_energy())


class ChargeOutput:
    def __init__(self, values):
        self.values = values

    def calculate(self, atoms, properties=('energy', 'forces', 'stress')):
        assert atoms.charges is None
        if self.values is not None:
            atoms._impl.charges = self.values


@pytest.mark.parametrize('missing', [None, [.1], [.1, float('nan')]])
def test_same_geometry_fresh_missing_or_unsuitable_output_clears_charges(missing):
    potential = ChargeOutput([.3, -.1])
    source = ASEAtoms('H2', positions=[[0., 0., 0.], [1.3, 0., 0.]])
    calculator = IMRMLPCalculator(potential=potential, elements=['H'])
    source.calc = calculator
    np.testing.assert_allclose(source.get_charges(), [.3, -.1])
    potential.values = missing
    calculator.calculate(source, properties=['charges'], system_changes=[])
    assert 'charges' not in calculator.results
    with pytest.raises(PropertyNotImplementedError):
        source.get_charges()
