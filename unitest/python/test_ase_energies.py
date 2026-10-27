"""ASE exposes only complete per-atom energies from the current evaluation."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
pytest.importorskip('ase')
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator

from imrmlp import Atoms, Potential
from imrmlp.calculator import IMRMLPCalculator

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def atoms():
    return Atoms(numbers=[13, 13, 13], elements=['Al'],
                 positions=[[1., 1., 1.], [3.2, 1., 1.], [5.7, 1.3, 1.]],
                 cell=np.eye(3) * 20, pbc=False)


def test_mtp_energies_match_native_order_sum_and_changed_geometry(atoms, tmp_path,
                                                               monkeypatch):
    monkeypatch.chdir(tmp_path)
    potential = Potential(str(ROOT / 'unitest/test-data/potential.mtp'))
    potential.calculate(atoms)
    expected = atoms.site_energies.copy()
    ase_atoms = atoms.to_ase(attach_calculator=False)
    ase_atoms.calc = IMRMLPCalculator(potential=potential, elements=['Al'])

    np.testing.assert_allclose(ase_atoms.get_potential_energies(), expected)
    assert ase_atoms.get_potential_energies().sum() == pytest.approx(
        ase_atoms.get_potential_energy())
    np.testing.assert_allclose(ase_atoms.get_forces(), atoms.forces)
    np.testing.assert_allclose(ase_atoms.get_stress(), atoms.to_ase().get_stress())

    permutation = [2, 0, 1]
    permuted = ase_atoms[permutation]
    permuted.calc = IMRMLPCalculator(potential=potential, elements=['Al'])
    np.testing.assert_allclose(permuted.get_potential_energies(), expected[permutation])

    ase_atoms.positions[1, 0] += 0.3
    reference = Atoms.from_ase(ase_atoms, read_results=False)
    reference.chemmap = atoms.chemmap
    potential.calculate(reference)
    updated = ase_atoms.get_potential_energies()
    assert not np.allclose(updated, expected)
    np.testing.assert_allclose(updated, reference.site_energies)
    assert updated.sum() == pytest.approx(ase_atoms.get_potential_energy())
    updated[:] = 0
    np.testing.assert_allclose(ase_atoms.get_potential_energies(), reference.site_energies)


class OutputPotential:
    """Exercise missing, malformed and changing native output contracts."""

    def __init__(self, values, total=6.):
        self.values = values
        self.total = total
        self.fail = False

    def calculate(self, atoms, properties=('energy', 'forces', 'stress')):
        self.properties = tuple(properties)
        assert atoms.site_energies is None
        assert atoms.energy is None
        if self.fail:
            raise RuntimeError('evaluation failed')
        if self.total is not None:
            atoms.energy = self.total
        if self.values is not None:
            atoms._impl.site_energies = self.values


@pytest.mark.parametrize('values,total', [
    (None, 6.),
    ([], 6.),
    ([1., 2.], 6.),
    ([0., 0., 0.], 6.),
    ([1., 2., 9.], 6.),
    ([1., np.nan, 3.], 6.),
    ([1., np.inf, 3.], 6.),
    ([1., 2., 3.], None),
    ([1., 2., 3.], np.nan),
])
def test_calculator_rejects_missing_or_incomplete_energies(atoms, values, total):
    ase_atoms = atoms.to_ase(attach_calculator=False)
    ase_atoms.calc = IMRMLPCalculator(potential=OutputPotential(values, total),
                                   elements=['Al'])
    with pytest.raises(PropertyNotImplementedError, match='per-atom energies'):
        ase_atoms.get_potential_energies()
    assert 'energies' not in ase_atoms.calc.results


def test_missing_energies_do_not_block_total_energy_or_default_calculate(atoms):
    ase_atoms = atoms.to_ase(attach_calculator=False)
    ase_atoms.calc = IMRMLPCalculator(potential=OutputPotential(None), elements=['Al'])
    assert ase_atoms.get_potential_energy() == pytest.approx(6.)
    assert ase_atoms.calc._pot.properties == ('energy',)
    ase_atoms.calc.calculate(ase_atoms)
    assert ase_atoms.calc._pot.properties == ('energy', 'forces', 'stress')
    assert ase_atoms.calc.results['energy'] == pytest.approx(6.)
    assert 'energies' not in ase_atoms.calc.results


def test_missing_total_energy_is_not_synthesized_as_zero(atoms):
    ase_atoms = atoms.to_ase(attach_calculator=False)
    ase_atoms.calc = IMRMLPCalculator(potential=OutputPotential(None, total=None),
                                   elements=['Al'])
    with pytest.raises(PropertyNotImplementedError):
        ase_atoms.get_potential_energy()
    assert 'energy' not in ase_atoms.calc.results


def test_zero_site_and_total_energy_are_valid(atoms):
    ase_atoms = atoms.to_ase(attach_calculator=False)
    ase_atoms.calc = IMRMLPCalculator(potential=OutputPotential([0., 0., 0.], total=0.),
                                   elements=['Al'])
    np.testing.assert_array_equal(ase_atoms.get_potential_energies(), [0., 0., 0.])
    assert ase_atoms.get_potential_energy() == 0.


def test_same_geometry_missing_output_and_failed_evaluation_clear_cache(atoms):
    potential = OutputPotential([1., 2., 3.])
    ase_atoms = atoms.to_ase(attach_calculator=False)
    calculator = IMRMLPCalculator(potential=potential, elements=['Al'])
    ase_atoms.calc = calculator
    np.testing.assert_array_equal(ase_atoms.get_potential_energies(), [1., 2., 3.])

    potential.values = None
    with pytest.raises(PropertyNotImplementedError):
        calculator.calculate(ase_atoms, properties=['energies'], system_changes=[])
    assert 'energies' not in calculator.results
    assert calculator._imrmlp_atoms.site_energies is None

    potential.values = [2., 1., 3.]
    calculator.calculate(properties=['energies'], system_changes=[])
    np.testing.assert_array_equal(calculator.results['energies'], [2., 1., 3.])
    potential.fail = True
    with pytest.raises(RuntimeError, match='evaluation failed'):
        calculator.calculate(ase_atoms, properties=['energies'], system_changes=[])
    assert calculator.results == {}








@pytest.mark.parametrize('attach_calculator', [False, True])
def test_to_ase_transfers_complete_site_energies_without_aliasing(atoms,
                                                               attach_calculator):
    atoms.energy = 6.
    atoms._impl.site_energies = [1., 2., 3.]
    converted = atoms.to_ase(attach_calculator=attach_calculator)
    atoms._impl.site_energies = [3., 2., 1.]
    if attach_calculator:
        assert 'energies' not in converted.arrays
        np.testing.assert_array_equal(converted.get_potential_energies(), [1., 2., 3.])
        returned = converted.get_potential_energies()
        returned[:] = 0
        np.testing.assert_array_equal(converted.get_potential_energies(), [1., 2., 3.])
    else:
        assert converted.calc is None
        np.testing.assert_array_equal(converted.arrays['energies'], [1., 2., 3.])


@pytest.mark.parametrize('values', [[], [1., 2.], [1., 2., 9.], [1., np.nan, 3.]])
def test_to_ase_omits_incomplete_site_energies(atoms, values):
    atoms.energy = 6.
    atoms._impl.site_energies = values
    converted = atoms.to_ase()
    assert 'energies' not in converted.arrays
    assert converted.get_potential_energy() == pytest.approx(6.)
    with pytest.raises(PropertyNotImplementedError):
        converted.get_potential_energies()


@pytest.mark.parametrize('attach_calculator', [False, True])
def test_ase_roundtrip_preserves_site_energies_without_aliasing(atoms,
                                                             attach_calculator):
    atoms.energy = 6.
    atoms._impl.site_energies = [3., 1., 2.]
    source = atoms.to_ase(attach_calculator=attach_calculator)
    converted = Atoms.from_ase(source)
    np.testing.assert_array_equal(converted.site_energies, [3., 1., 2.])
    assert converted.energy == 6.

    if source.calc is not None:
        source.calc.results['energies'][:] = 0
    else:
        source.arrays['energies'][:] = 0
    np.testing.assert_array_equal(converted.site_energies, [3., 1., 2.])


@pytest.mark.parametrize('total', [None, 42.])
def test_from_ase_uses_cached_site_energies_without_total_energy_constraint(atoms,
                                                                          total):
    source = atoms.to_ase(attach_calculator=False)
    source.arrays['energies'] = np.array([9., 8., 7.])
    source.calc = SinglePointCalculator(source, energies=[1., 2., 3.])
    if total is not None:
        source.calc.results['energy'] = total
    converted = Atoms.from_ase(source)
    np.testing.assert_array_equal(converted.site_energies, [1., 2., 3.])
    assert converted.energy == total


def test_from_ase_falls_back_to_array_when_calculator_lacks_site_energies(atoms):
    source = atoms.to_ase(attach_calculator=False)
    source.arrays['energies'] = np.array([3., 2., 1.])
    source.calc = SinglePointCalculator(source, energy=6.)
    converted = Atoms.from_ase(source)
    np.testing.assert_array_equal(converted.site_energies, [3., 2., 1.])


def test_from_ase_preserves_missing_and_skipped_results(atoms):
    source = atoms.to_ase(attach_calculator=False)
    assert Atoms.from_ase(source).site_energies is None
    source.arrays['energies'] = np.array([1., 2., 3.])
    source.calc = SinglePointCalculator(source, energy=6., energies=[1., 2., 3.])
    converted = Atoms.from_ase(source, read_results=False)
    assert converted.energy is None
    assert converted.site_energies is None


def test_from_ase_rejects_wrong_site_energy_count(atoms):
    source = atoms.to_ase(attach_calculator=False)
    source.calc = SinglePointCalculator(source, energies=[1., 2.])
    with pytest.raises(ValueError, match=r'per-atom energies.*shape \(N,\)'):
        Atoms.from_ase(source)


class RecordingPotential:
    """Expose the explicit charge actually received by the native input."""

    def __init__(self):
        self.seen = []

    def calculate(self, atoms, properties=('energy', 'forces', 'stress')):
        charge = atoms.total_charge
        self.seen.append(charge)
        atoms.energy = 17. if charge is None else 10. + charge


def test_calculator_forwards_explicit_charge_and_invalidates_ase_energy_cache(atoms):
    source = atoms.to_ase(attach_calculator=False)
    source.set_initial_charges([.2, .3, .4])
    positions = source.positions.copy()
    initial_charges = source.get_initial_charges().copy()
    potential = RecordingPotential()
    source.calc = IMRMLPCalculator(potential=potential, elements=['Al'])

    assert source.get_potential_energy() == pytest.approx(17.)
    assert source.get_potential_energy() == pytest.approx(17.)
    assert potential.seen == [None]
    assert 'total_charge' not in source.info

    source.info['total_charge'] = .8
    assert source.get_potential_energy() == pytest.approx(10.8)
    assert potential.seen == [None, .8]
    source.info['total_charge'] = .8
    source.info['label'] = 'same geometry and explicit charge'
    assert source.get_potential_energy() == pytest.approx(10.8)
    assert potential.seen == [None, .8]
    assert source.info['total_charge'] == .8

    source.info['total_charge'] = 0.
    assert source.get_potential_energy() == pytest.approx(10.)
    assert potential.seen == [None, .8, 0.]
    source.info['total_charge'] = None
    assert source.get_potential_energy() == pytest.approx(17.)
    assert potential.seen == [None, .8, 0., None]
    assert source.info['total_charge'] is None
    del source.info['total_charge']
    assert source.get_potential_energy() == pytest.approx(17.)
    assert potential.seen == [None, .8, 0., None]

    np.testing.assert_array_equal(source.positions, positions)
    np.testing.assert_array_equal(source.get_initial_charges(), initial_charges)
    assert source.info == {'label': 'same geometry and explicit charge'}


@pytest.mark.parametrize('attach_calculator', [True, False])
@pytest.mark.parametrize('cell', [None, np.diag([8., 9., 0.])], ids=['no-cell', 'two-vectors'])
def test_rankless_ase_export_preserves_internal_virial(cell, attach_calculator):
    native = Atoms(numbers=[1, 1], elements=['H'],
                   positions=[[0., 0., 0.], [1.1, .2, .3]], cell=cell, pbc=False)
    native.energy = -1.3
    native.forces = [[.1, .2, .3], [-.1, -.2, -.3]]
    native.stress = [[1., .2, .3], [.2, 2., .4], [.3, .4, 3.]]
    positions, lattice, virial = native.positions.copy(), native.cell.copy(), native.stress.copy()
    exported = native.to_ase(attach_calculator=attach_calculator)
    assert exported.cell.rank < 3
    assert 'stress' in exported.info
    if exported.calc is not None:
        assert 'stress' not in exported.calc.results
    converted = Atoms.from_ase(exported)
    np.testing.assert_array_equal(converted.positions, positions)
    np.testing.assert_array_equal(converted.cell, lattice)
    np.testing.assert_array_equal(converted.pbc, [False, False, False])
    np.testing.assert_array_equal(converted.stress, virial)
    np.testing.assert_array_equal(converted.forces, native.forces)
    assert converted.energy == native.energy
    np.testing.assert_array_equal(native.positions, positions)
    np.testing.assert_array_equal(native.cell, lattice)
    np.testing.assert_array_equal(native.stress, virial)


@pytest.mark.parametrize('free_energy,site_energies', [
    (-1.2, [1., 2., 3.]), (0., [0., 0., 0.]), (None, None),
], ids=['independent-labels', 'zeros', 'missing'])
def test_native_extxyz_ase_roundtrip_preserves_free_and_site_energies(
        atoms, tmp_path, free_energy, site_energies):
    from ase.io import read as ase_read
    from imrmlp import write_one

    atoms.energy = 6.
    atoms.free_energy = free_energy
    if site_energies is not None:
        atoms._impl.site_energies = site_energies
    positions = atoms.positions.copy()
    path = tmp_path / 'native.extxyz'
    write_one(path, atoms)
    source = ase_read(path)
    # These are the native writer's historical spellings, not ASE results.
    if free_energy is not None:
        assert source.info['fenergy'] == free_energy
    if site_energies is not None:
        np.testing.assert_array_equal(source.arrays['site_en'], site_energies)
    converted = Atoms.from_ase(source)
    assert converted.energy == 6.
    assert converted.free_energy == free_energy
    if site_energies is None:
        assert converted.site_energies is None
    else:
        np.testing.assert_array_equal(converted.site_energies, site_energies)
    skipped = Atoms.from_ase(source, read_results=False)
    assert skipped.energy is None and skipped.free_energy is None
    assert skipped.site_energies is None
    np.testing.assert_array_equal(atoms.positions, positions)
    np.testing.assert_array_equal(source.positions, positions)
    if free_energy is not None:
        assert source.info['fenergy'] == free_energy
    if site_energies is not None:
        np.testing.assert_array_equal(source.arrays['site_en'], site_energies)
        np.testing.assert_array_equal(atoms.site_energies, site_energies)


def test_from_ase_standard_results_and_fields_take_priority_over_native_aliases(atoms):
    source = atoms.to_ase(attach_calculator=False)
    source.info['fenergy'] = -1.2
    source.arrays['site_en'] = np.array([1., 2., 3.])
    alias_only = Atoms.from_ase(source)
    assert alias_only.free_energy == -1.2
    np.testing.assert_array_equal(alias_only.site_energies, [1., 2., 3.])

    source.info['free_energy'] = -2.4
    source.arrays['energies'] = np.array([3., 2., 1.])
    standard_fields = Atoms.from_ase(source)
    assert standard_fields.free_energy == -2.4
    np.testing.assert_array_equal(standard_fields.site_energies, [3., 2., 1.])

    source.calc = SinglePointCalculator(source, free_energy=-3.6, energies=[2., 3., 1.])
    cached = Atoms.from_ase(source)
    assert cached.free_energy == -3.6
    np.testing.assert_array_equal(cached.site_energies, [2., 3., 1.])
    assert source.info['fenergy'] == -1.2 and source.info['free_energy'] == -2.4
    np.testing.assert_array_equal(source.arrays['site_en'], [1., 2., 3.])
    np.testing.assert_array_equal(source.arrays['energies'], [3., 2., 1.])


def test_from_ase_ignores_stale_cache_without_recalculation_or_source_reset(atoms):
    class NoRecalculation(SinglePointCalculator):
        calls = 0

        def calculate(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError('conversion must not calculate')

    source = atoms.to_ase(attach_calculator=False)
    source.info['total_charge'] = .8
    source.set_velocities([[.1, .2, .3], [.2, .3, .4], [.3, .4, .5]])
    expected_velocities = Atoms.from_ase(source, read_results=False).velocities
    source.calc = NoRecalculation(
        source, energy=6., free_energy=-1.2, energies=[1., 2., 3.],
        forces=np.arange(9, dtype=float).reshape(3, 3),
        stress=np.arange(6, dtype=float), charges=[.1, .2, .3], magmoms=[1., 2., 3.])
    cached = {key: value.copy() if isinstance(value, np.ndarray) else value
              for key, value in source.calc.results.items()}
    cached_positions = source.calc.atoms.positions.copy()
    valid = Atoms.from_ase(source)
    assert valid.energy == 6. and valid.free_energy == -1.2
    np.testing.assert_array_equal(valid.site_energies, cached['energies'])
    np.testing.assert_array_equal(valid.forces, cached['forces'])

    source.positions[0, 0] += .15
    assert source.calc.check_state(source)
    converted = Atoms.from_ase(source)
    assert converted.energy is None and converted.free_energy is None
    assert converted.site_energies is None and converted.forces is None
    assert converted.stress is None and converted.charges is None and converted.magmoms is None
    np.testing.assert_array_equal(converted.positions, source.positions)
    assert converted.total_charge == .8
    np.testing.assert_array_equal(converted.velocities, expected_velocities)
    skipped = Atoms.from_ase(source, read_results=False)
    assert skipped.energy is None and skipped.total_charge == .8
    np.testing.assert_array_equal(skipped.velocities, expected_velocities)
    assert source.calc.calls == 0
    assert set(source.calc.results) == set(cached)
    for key, value in cached.items():
        np.testing.assert_array_equal(source.calc.results[key], value)
    np.testing.assert_array_equal(source.calc.atoms.positions, cached_positions)

    # Explicit legacy labels retain their existing meaning despite stale cache.
    source.info['energy'] = 9.
    source.info['fenergy'] = -4.
    source.arrays['site_en'] = np.array([4., 3., 2.])
    legacy = Atoms.from_ase(source)
    assert legacy.energy == 9. and legacy.free_energy == -4.
    np.testing.assert_array_equal(legacy.site_energies, [4., 3., 2.])
    assert legacy.forces is None and source.calc.calls == 0
