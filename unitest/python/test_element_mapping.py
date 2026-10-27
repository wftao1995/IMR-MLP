"""Explicit model element orders must not relabel missing atomic numbers."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Descriptor, Potential
from imrmlp.utils import chemmap_from_numbers


def test_explicit_elements_constructor_rejects_atomic_number_species_index_collision():
    # H has Z=1, which is also a valid species index for Ni in this order.
    with pytest.raises(ValueError, match='Atomic number 1 is not present in elements'):
        Atoms(numbers=[1], elements=['Al', 'Ni'], positions=[[0., 0., 0.]])


def test_explicit_elements_numbers_assignment_keeps_prior_state_on_missing_element():
    atoms = Atoms(numbers=[13, 28], elements=['Ni', 'Al'],
                  positions=[[0., 0., 0.], [1., 0., 0.]])
    positions = atoms.positions.copy()
    with pytest.raises(ValueError, match='Atomic number 1 is not present in elements'):
        atoms.numbers = [13, 1]
    assert atoms.chemmap == [1, 0]
    np.testing.assert_array_equal(atoms.numbers, [13, 28])
    np.testing.assert_array_equal(atoms.positions, positions)


@pytest.mark.parametrize('elements', [['Ni', 'Al'], ['28', '13']])
def test_explicit_elements_keep_nonidentity_order_and_subset_indices(elements):
    original_elements = list(elements)
    for numbers, expected in [([13, 28, 13], [1, 0, 1]), ([13, 13], [1, 1])]:
        atoms = Atoms(numbers=numbers, elements=elements,
                      positions=np.zeros((len(numbers), 3)))
        assert atoms.chemmap == expected
        np.testing.assert_array_equal(atoms.numbers, numbers)
    assert elements == original_elements


def test_no_elements_keeps_existing_atomic_number_and_utility_contracts():
    atoms = Atoms(numbers=[28, 13], positions=np.zeros((2, 3)))
    assert atoms.chemmap == [28, 13]
    np.testing.assert_array_equal(atoms.numbers, [28, 13])
    # The utility's existing no-elements behavior is independently auto-indexed.
    assert chemmap_from_numbers([28, 13]) == [1, 0]


def test_ase_calculator_rejects_missing_element_before_potential_evaluation():
    ase = pytest.importorskip('ase')
    from imrmlp.calculator import IMRMLPCalculator

    calls = []

    class CalculationSpy:
        def calculate(self, *args, **kwargs):
            calls.append(True)
            raise AssertionError('A missing element must not reach potential evaluation')

    source = ase.Atoms('H', positions=[[1., 2., 3.]], cell=np.eye(3) * 12, pbc=False)
    positions = source.positions.copy()
    source.calc = IMRMLPCalculator(potential=CalculationSpy(), elements=['Al', 'Ni'])
    with pytest.raises(ValueError, match='Atomic number 1 is not present in elements'):
        source.get_potential_energy()
    assert calls == []
    np.testing.assert_array_equal(source.numbers, [1])
    np.testing.assert_array_equal(source.positions, positions)
    assert source.calc.results == {}


@pytest.mark.parametrize('numbers', [[13, 28, 13], [13, 13]], ids=['mixed', 'subset'])
def test_ase_calculator_uses_fixed_model_order_for_real_multispecies_evaluation(numbers):
    ase = pytest.importorskip('ase')
    from imrmlp.calculator import IMRMLPCalculator

    params = {'behler:dispcount': '2', 'behler:dispselect': '1 2',
              'behler:cutoff': '4', 'behler:eta': '.3',
              'behler:weighted': 'true', 'behler:weights': '1 2'}
    settings = {'pottype': 'descriptor_linear', 'descriptor_linear:species_count': '2',
                'descriptor_linear:descriptor:dtype': 'behler'}
    settings.update({'descriptor_linear:descriptor:' + key: value
                     for key, value in params.items()})
    potential = Potential.from_settings(settings)
    potential.coefficients = np.linspace(-.17, .31, Descriptor('behler', params).feature_size)
    positions = np.array([[1., 1., 1.], [2.5, 1.2, 1.1], [1.2, 3.1, 1.4]])[:len(numbers)]
    cell = np.eye(3) * 12
    elements = ['Ni', 'Al']
    reference = Atoms(numbers=numbers, elements=elements, positions=positions,
                      cell=cell, pbc=False)
    potential.calculate(reference)
    source = ase.Atoms(numbers=numbers, positions=positions, cell=cell, pbc=False)
    calculator = IMRMLPCalculator(potential=potential, elements=elements)
    source.calc = calculator
    assert source.get_potential_energy() == pytest.approx(reference.energy, abs=1e-12)
    np.testing.assert_allclose(source.get_forces(), reference.forces, atol=1e-12, rtol=1e-12)
    assert calculator._imrmlp_atoms.chemmap == [1 if number == 13 else 0 for number in numbers]
    np.testing.assert_array_equal(source.numbers, numbers)
    np.testing.assert_array_equal(source.positions, positions)
    np.testing.assert_array_equal(source.cell.array, cell)
    assert elements == ['Ni', 'Al']
