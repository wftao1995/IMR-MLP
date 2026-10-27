"""Physical quantities, element mappings and driver output regression tests."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Potential, NEB, Dimer, Relaxation, read, read_one, write, write_one

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def system(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    potential = Potential(str(ROOT / 'unitest/test-data/potential.mtp'))
    atoms = Atoms(numbers=[13, 13], positions=[[1., 1., 1.], [3., 1., 1.]],
                  cell=np.eye(3) * 30, elements=['Al'])
    return potential, atoms


def test_ase_stress_matches_energy_derivative(system):
    pytest.importorskip('ase')
    from imrmlp.calculator import IMRMLPCalculator
    potential, atoms = system
    ase_atoms = atoms.to_ase()
    ase_atoms.calc = IMRMLPCalculator(potential=potential, elements=['Al'])
    h = 1e-6
    energies = []
    for strain in [-h, h]:
        strained = ase_atoms.copy()
        strained.calc = IMRMLPCalculator(potential=potential, elements=['Al'])
        cell = strained.cell.array.copy()
        cell[:, 0] *= 1 + strain
        strained.set_cell(cell, scale_atoms=True)
        energies.append(strained.get_potential_energy())
    expected = (energies[1] - energies[0]) / (2 * h * ase_atoms.get_volume())
    assert ase_atoms.get_stress()[0] == pytest.approx(expected, rel=1e-6)
    potential.calculate(atoms)
    assert atoms.to_ase().get_stress()[0] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize('matrix', [False, True])
def test_ase_import_uses_negative_extensive_virial(system, matrix):
    pytest.importorskip('ase')
    from ase.calculators.singlepoint import SinglePointCalculator
    _, atoms = system
    source = atoms.to_ase()
    stress = np.array([[1., 6., 5.], [6., 2., 4.], [5., 4., 3.]])
    source.calc = SinglePointCalculator(source, stress=stress if matrix else [1, 2, 3, 4, 5, 6])
    converted = Atoms.from_ase(source)
    np.testing.assert_allclose(converted.stress, -stress * source.get_volume())
    np.testing.assert_allclose(converted.to_ase().get_stress(voigt=False), stress)


@pytest.fixture
def exportable_atoms():
    atoms = Atoms(numbers=[13, 8], positions=[[.1, .2, .3], [1., 2., 3.]],
                  cell=np.diag([4., 5., 20.]), pbc=[True, True, False])
    atoms.energy = 3.
    atoms._impl.site_energies = [1., 2.]
    atoms.forces = [[.1, .2, .3], [-.1, -.2, -.3]]
    atoms.stress = [[1., .2, .3], [.2, 2., .4], [.3, .4, 3.]]
    return atoms


@pytest.fixture(params=['scalar', 'vector'])
def magnetic_values(request):
    if request.param == 'scalar':
        return np.array([.7, -.2])
    return np.array([[.1, .2, .3], [-.4, .5, -.6]])


def test_ase_default_export_writes_and_reads_extxyz(exportable_atoms, tmp_path):
    pytest.importorskip('ase')
    from ase.io import read as ase_read, write as ase_write
    atoms = exportable_atoms
    exported = atoms.to_ase()
    assert not ({'energy', 'stress'} & exported.info.keys())
    assert not ({'forces', 'energies'} & exported.arrays.keys())
    path = tmp_path / 'ase-results.xyz'
    ase_write(path, exported, format='extxyz')
    back = Atoms.from_ase(ase_read(path, format='extxyz'))
    np.testing.assert_array_equal(back.numbers, atoms.numbers)
    np.testing.assert_array_equal(back.pbc, atoms.pbc)
    np.testing.assert_allclose(back.positions, atoms.positions)
    np.testing.assert_allclose(back.cell, atoms.cell)
    assert back.energy == pytest.approx(atoms.energy)
    np.testing.assert_allclose(back.site_energies, atoms.site_energies)
    np.testing.assert_allclose(back.forces, atoms.forces)
    np.testing.assert_allclose(back.stress, atoms.stress)


def test_ase_default_export_with_only_forces_is_writable(tmp_path):
    pytest.importorskip('ase')
    from ase.io import read as ase_read, write as ase_write
    atoms = Atoms(numbers=[13], positions=[[1., 2., 3.]], cell=np.eye(3) * 5)
    assert atoms.to_ase().calc is None
    atoms.forces = [[.1, .2, .3]]
    exported = atoms.to_ase()
    assert set(exported.calc.results) == {'forces'}
    assert 'forces' not in exported.arrays
    path = tmp_path / 'ase-forces.xyz'
    ase_write(path, exported, format='extxyz')
    np.testing.assert_allclose(ase_read(path, format='extxyz').get_forces(), atoms.forces)


def test_ase_export_without_calculator_preserves_legacy_fields(exportable_atoms):
    pytest.importorskip('ase')
    atoms = exportable_atoms
    exported = atoms.to_ase(attach_calculator=False)
    assert exported.calc is None
    assert exported.info['energy'] == atoms.energy
    np.testing.assert_array_equal(exported.arrays['energies'], atoms.site_energies)
    np.testing.assert_array_equal(exported.arrays['forces'], atoms.forces)
    np.testing.assert_array_equal(exported.info['stress'], [1., 2., 3., .4, .3, .2])
    np.testing.assert_array_equal(Atoms.from_ase(exported).stress, atoms.stress)


def test_ase_extxyz_native_reader_recognizes_site_energies(exportable_atoms, tmp_path):
    pytest.importorskip('ase')
    from ase.io import write as ase_write
    path = tmp_path / 'ase-native.xyz'
    ase_write(path, exportable_atoms.to_ase(), format='extxyz')
    assert ':energies:R:1' in path.read_text().splitlines()[1]
    back = read_one(path, format='extxyz')
    np.testing.assert_allclose(back.site_energies, exportable_atoms.site_energies)
    assert back.energy == pytest.approx(exportable_atoms.energy)
    np.testing.assert_allclose(back.forces, exportable_atoms.forces)
    np.testing.assert_allclose(back.stress, exportable_atoms.stress)
    np.testing.assert_array_equal(back.pbc, exportable_atoms.pbc)


def test_public_charges_are_optional_read_only_copies(exportable_atoms):
    atoms = exportable_atoms
    assert atoms.charges is None
    atoms._impl.charges = [.25, -.1]
    values = atoms.charges
    assert values.shape == (2,)
    np.testing.assert_array_equal(values, [.25, -.1])
    values[:] = 0
    np.testing.assert_array_equal(atoms.charges, [.25, -.1])
    with pytest.raises(AttributeError):
        atoms.charges = [.5, -.2]


def test_magmoms_preserve_shape_and_own_their_values(exportable_atoms, magnetic_values):
    atoms = exportable_atoms
    assert atoms.magmoms is None
    supplied = magnetic_values.copy()
    atoms.magmoms = supplied
    supplied[:] = 99.
    np.testing.assert_array_equal(atoms.magmoms, magnetic_values)
    result = atoms.magmoms
    result[:] = 88.
    np.testing.assert_array_equal(atoms.magmoms, magnetic_values)
    copied = atoms.copy()
    atoms.magmoms = np.zeros_like(magnetic_values)
    assert atoms.magmoms.shape == magnetic_values.shape
    np.testing.assert_array_equal(atoms.magmoms, np.zeros_like(magnetic_values))
    np.testing.assert_array_equal(copied.magmoms, magnetic_values)
    alternate = np.zeros((2, 3)) if magnetic_values.ndim == 1 else np.zeros(2)
    atoms.magmoms = alternate
    np.testing.assert_array_equal(atoms.magmoms, alternate)
    atoms.magmoms = None
    assert atoms.magmoms is None
    assert atoms.energy == 3.


@pytest.mark.parametrize('attach_calculator', [False, True])
def test_ase_computed_magmoms_roundtrip_preserves_representation(
        exportable_atoms, magnetic_values, attach_calculator):
    pytest.importorskip('ase')
    atoms = exportable_atoms
    atoms.magmoms = magnetic_values
    original = atoms.copy()
    exported = atoms.to_ase(attach_calculator=attach_calculator)
    assert 'initial_magmoms' not in exported.arrays
    if attach_calculator:
        assert 'magmoms' not in exported.arrays
        assert 'magmom' not in exported.calc.results
        np.testing.assert_array_equal(exported.get_magnetic_moments(), magnetic_values)
        stored = exported.calc.results['magmoms']
    else:
        assert exported.calc is None
        stored = exported.arrays['magmoms']
    np.testing.assert_array_equal(stored, magnetic_values)
    converted = Atoms.from_ase(exported)
    np.testing.assert_array_equal(converted.magmoms, magnetic_values)
    assert Atoms.from_ase(exported, read_results=False).magmoms is None
    stored[:] = 99.
    np.testing.assert_array_equal(converted.magmoms, magnetic_values)
    np.testing.assert_array_equal(atoms.magmoms, original.magmoms)
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.cell, original.cell)
    np.testing.assert_array_equal(atoms.pbc, original.pbc)


def test_ase_magmoms_prefer_cache_and_never_infer_initial_results(exportable_atoms):
    pytest.importorskip('ase')
    from ase.calculators.singlepoint import SinglePointCalculator
    source = exportable_atoms.to_ase(attach_calculator=False)
    source.set_initial_magnetic_moments([9., 8.])
    source.arrays['magmoms'] = np.array([.6, -.3])
    source.calc = SinglePointCalculator(source, magmoms=[.7, -.2], magmom=5.)
    imported = Atoms.from_ase(source)
    np.testing.assert_array_equal(imported.magmoms, [.7, -.2])
    assert Atoms.from_ase(source, read_results=False).magmoms is None
    np.testing.assert_array_equal(source.arrays['magmoms'], [.6, -.3])
    np.testing.assert_array_equal(source.get_initial_magnetic_moments(), [9., 8.])
    source.calc = None
    np.testing.assert_array_equal(Atoms.from_ase(source).magmoms, [.6, -.3])
    del source.arrays['magmoms']
    source.calc = SinglePointCalculator(source, magmom=5.)
    assert Atoms.from_ase(source).magmoms is None


def test_computed_magmoms_survive_both_real_extxyz_directions(tmp_path, magnetic_values):
    pytest.importorskip('ase')
    from ase.io import read as ase_read, write as ase_write
    width = 1 if magnetic_values.ndim == 1 else 3
    rows = []
    for symbol, position, moments in zip(('Al', 'O'), ((1, 2, 3), (2, 3, 4)), magnetic_values):
        values = (*position, *np.atleast_1d(moments))
        rows.append(symbol + ' ' + ' '.join(map(str, values)))
    source = tmp_path / 'native-magmoms.xyz'
    source.write_text('2\nLattice="10 0 0 0 10 0 0 0 10" pbc="T T T" '
                      f'Properties=species:S:1:pos:R:3:magmoms:R:{width}\n' +
                      '\n'.join(rows) + '\n')
    atoms = read_one(source)
    np.testing.assert_array_equal(atoms.magmoms, magnetic_values)
    original = atoms.copy()
    exported = atoms.to_ase()
    np.testing.assert_array_equal(exported.get_magnetic_moments(), magnetic_values)
    ase_path = tmp_path / 'ase-magmoms.xyz'
    ase_write(ase_path, exported, format='extxyz')
    from_file = read_one(ase_path)
    np.testing.assert_array_equal(from_file.magmoms, magnetic_values)
    imported = Atoms.from_ase(ase_read(ase_path))
    np.testing.assert_array_equal(imported.magmoms, magnetic_values)
    native_path = tmp_path / 'rewritten-magmoms.xyz'
    write_one(native_path, imported)
    np.testing.assert_array_equal(ase_read(native_path).get_magnetic_moments(), magnetic_values)
    np.testing.assert_array_equal(atoms.magmoms, original.magmoms)
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.pbc, original.pbc)


def test_magmoms_reject_dual_native_representation_without_dropping_data(tmp_path):
    pytest.importorskip('ase')
    import struct
    payload = b'GPMLBIN2' + struct.pack('=Ii', 3, 1)
    payload += b'l' + struct.pack('=9d', 10, 0, 0, 0, 10, 0, 0, 0, 10)
    payload += b'p' + struct.pack('=3d', 1, 2, 3) + b't' + struct.pack('=i', 1)
    payload += b'M' + struct.pack('=d', 1.5)
    payload += b'm' + struct.pack('=3d', .1, .2, .3) + b'#'
    source = tmp_path / 'dual-magmoms.bin'
    source.write_bytes(payload)
    atoms = read_one(source)
    with pytest.raises(ValueError, match='Simultaneous scalar and vector'):
        _ = atoms.magmoms
    with pytest.raises(ValueError, match='Simultaneous scalar and vector'):
        atoms.to_ase()
    copied = tmp_path / 'dual-copy.bin'
    write_one(copied, atoms)
    with pytest.raises(ValueError, match='Simultaneous scalar and vector'):
        _ = read_one(copied).magmoms
    atoms.magmoms = None
    assert atoms.magmoms is None


def test_magmoms_reject_wrong_shape_or_nonfinite_values_without_replacing_state(exportable_atoms):
    atoms = exportable_atoms
    atoms.magmoms = [.7, -.2]
    for invalid in (np.zeros(1), np.zeros((2, 2)), np.array([np.nan, 0.])):
        with pytest.raises(ValueError, match='Magnetic moments.*shape|Magnetic moments.*finite'):
            atoms.magmoms = invalid
        np.testing.assert_array_equal(atoms.magmoms, [.7, -.2])


def test_free_energy_is_optional_and_independent_of_energy(exportable_atoms, tmp_path):
    atoms = exportable_atoms
    assert atoms.free_energy is None
    atoms.free_energy = -1.2
    copied = atoms.copy()
    path = tmp_path / 'free-energy.xyz'
    write_one(path, atoms)
    restored = read_one(path)
    assert restored.free_energy == -1.2
    assert restored.energy == atoms.energy == 3.
    atoms.free_energy = 0
    assert atoms.free_energy == 0.
    assert atoms.energy == 3.
    assert copied.free_energy == -1.2
    atoms.free_energy = None
    assert atoms.free_energy is None
    write_one(path, atoms)
    assert 'fenergy=' not in path.read_text().splitlines()[1]
    assert read_one(path).energy == 3.


@pytest.mark.parametrize('attach_calculator', [False, True])
def test_ase_free_energy_remains_independent_with_either_export_mode(
        exportable_atoms, attach_calculator):
    pytest.importorskip('ase')
    atoms = exportable_atoms
    atoms.free_energy = -1.2
    original = atoms.copy()
    exported = atoms.to_ase(attach_calculator=attach_calculator)
    if attach_calculator:
        assert 'free_energy' not in exported.info
        assert exported.get_potential_energy() == 3.
        assert exported.get_potential_energy(force_consistent=True) == -1.2
        assert exported.calc.results['free_energy'] == -1.2
    else:
        assert exported.calc is None
        assert exported.info['energy'] == 3.
        assert exported.info['free_energy'] == -1.2
    converted = Atoms.from_ase(exported)
    assert converted.energy == 3.
    assert converted.free_energy == -1.2
    assert Atoms.from_ase(exported, read_results=False).free_energy is None
    exported.info['free_energy'] = 99.
    assert atoms.free_energy == original.free_energy == -1.2
    assert converted.free_energy == -1.2
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.forces, original.forces)
    np.testing.assert_array_equal(atoms.stress, original.stress)


def test_ase_free_energy_uses_cached_result_before_legacy_info(exportable_atoms):
    pytest.importorskip('ase')
    from ase.calculators.singlepoint import SinglePointCalculator
    source = exportable_atoms.to_ase(attach_calculator=False)
    source.info['free_energy'] = 99.
    source.info['total_charge'] = .8
    source.calc = SinglePointCalculator(source, energy=-1., free_energy=-1.2)
    imported = Atoms.from_ase(source)
    assert imported.energy == -1.
    assert imported.free_energy == -1.2
    skipped = Atoms.from_ase(source, read_results=False)
    assert skipped.free_energy is None
    assert skipped.energy is None
    assert skipped.total_charge == .8
    assert source.info['free_energy'] == 99.
    assert source.calc.results['free_energy'] == -1.2
    source.calc = None
    assert Atoms.from_ase(source).free_energy == 99.


def test_ase_free_energy_zero_and_missing_remain_distinct(exportable_atoms):
    pytest.importorskip('ase')
    from ase.calculators.calculator import PropertyNotImplementedError
    atoms = exportable_atoms
    missing = atoms.to_ase()
    with pytest.raises(PropertyNotImplementedError):
        missing.get_potential_energy(force_consistent=True)
    assert Atoms.from_ase(missing).free_energy is None
    atoms.free_energy = 0.
    zero = atoms.to_ase()
    assert zero.get_potential_energy(force_consistent=True) == 0.
    assert zero.get_potential_energy() == atoms.energy
    assert Atoms.from_ase(zero).free_energy == 0.


def test_ase_free_energy_survives_real_extxyz_and_native_alias_reader(tmp_path):
    pytest.importorskip('ase')
    from ase.io import read as ase_read, write as ase_write
    source = tmp_path / 'native-free-energy.xyz'
    source.write_text('1\nLattice="10 0 0 0 10 0 0 0 10" pbc="T T T" '
                      'energy=-1 fenergy=-1.2 total_charge=.8 '
                      'Properties=species:S:1:pos:R:3\nAl 1 2 3\n')
    atoms = read_one(source)
    assert atoms.free_energy == -1.2
    exported = atoms.to_ase()
    path = tmp_path / 'ase-free-energy.xyz'
    ase_write(path, exported, format='extxyz')
    assert 'free_energy=' in path.read_text().splitlines()[1]
    native = read_one(path)
    assert native.energy == -1.
    assert native.free_energy == -1.2
    assert native.total_charge == .8
    imported = Atoms.from_ase(ase_read(path))
    assert imported.energy == -1.
    assert imported.free_energy == -1.2
    np.testing.assert_array_equal(native.numbers, atoms.numbers)
    np.testing.assert_allclose(native.positions, atoms.positions)
    np.testing.assert_array_equal(native.pbc, atoms.pbc)
    assert exported.get_potential_energy(force_consistent=True) == -1.2
    assert atoms.free_energy == -1.2


def test_total_charge_is_optional_independent_native_state(exportable_atoms, tmp_path):
    atoms = exportable_atoms
    atoms._impl.charges = [.1, .05]
    assert atoms.total_charge is None
    atoms.total_charge = .8
    copied = atoms.copy()
    assert copied.total_charge == .8
    path = tmp_path / 'explicit-charge.xyz'
    write_one(path, atoms)
    header = path.read_text().splitlines()[1]
    assert 'total_charge=0.8' in header
    assert 'total_charge="' not in header
    assert read_one(path).total_charge == .8
    atoms.total_charge = 0
    assert atoms.total_charge == 0.
    assert copied.total_charge == .8
    atoms.total_charge = None
    assert atoms.total_charge is None
    write_one(path, atoms)
    assert 'total_charge=' not in path.read_text().splitlines()[1]
    np.testing.assert_array_equal(atoms.charges, [.1, .05])


@pytest.mark.parametrize('charge', [None, 0., .8])
@pytest.mark.parametrize('attach_calculator', [False, True])
def test_ase_total_charge_is_input_metadata_independent_of_computed_charges(
        exportable_atoms, charge, attach_calculator):
    pytest.importorskip('ase')
    atoms = exportable_atoms
    atoms._impl.charges = [.1, .05]
    atoms.total_charge = charge
    original = atoms.copy()
    exported = atoms.to_ase(attach_calculator=attach_calculator)
    if charge is None:
        assert 'total_charge' not in exported.info
    else:
        assert exported.info['total_charge'] == charge
    assert 'initial_charges' not in exported.arrays
    if exported.calc is not None:
        assert 'total_charge' not in exported.calc.results

    for read_results in (True, False):
        converted = Atoms.from_ase(exported, read_results=read_results)
        assert converted.total_charge == charge
        if read_results:
            np.testing.assert_array_equal(converted.charges, [.1, .05])
        else:
            assert converted.charges is None
            assert converted.energy is None
            assert converted.forces is None
        np.testing.assert_array_equal(converted.numbers, atoms.numbers)
        np.testing.assert_array_equal(converted.pbc, atoms.pbc)
        np.testing.assert_allclose(converted.positions, atoms.positions)
    exported.info['total_charge'] = 3.
    assert atoms.total_charge == charge
    assert converted.total_charge == charge
    np.testing.assert_array_equal(atoms.positions, original.positions)
    np.testing.assert_array_equal(atoms.cell, original.cell)
    np.testing.assert_array_equal(atoms.pbc, original.pbc)
    np.testing.assert_array_equal(atoms.charges, original.charges)
    assert atoms.energy == original.energy
    np.testing.assert_array_equal(atoms.forces, original.forces)
    np.testing.assert_array_equal(atoms.stress, original.stress)


def test_ase_explicit_total_charge_import_does_not_use_initial_or_atomic_charges():
    pytest.importorskip('ase')
    from ase import Atoms as ASEAtoms
    from ase.calculators.singlepoint import SinglePointCalculator
    source = ASEAtoms(numbers=[13, 8], positions=[[1., 2., 3.], [2., 3., 4.]],
                      cell=np.eye(3) * 10, pbc=True)
    source.info['total_charge'] = .8
    source.set_initial_charges([9., 8.])
    source.calc = SinglePointCalculator(source, charges=[.1, .05])
    for read_results in (True, False):
        converted = Atoms.from_ase(source, read_results=read_results)
        assert converted.total_charge == .8
        if read_results:
            np.testing.assert_array_equal(converted.charges, [.1, .05])
        else:
            assert converted.charges is None
    assert source.info == {'total_charge': .8}
    np.testing.assert_array_equal(source.get_initial_charges(), [9., 8.])
    np.testing.assert_array_equal(source.calc.results['charges'], [.1, .05])
    del source.info['total_charge']
    assert Atoms.from_ase(source).total_charge is None


def test_ase_total_charge_survives_real_extxyz_write_and_read(tmp_path):
    pytest.importorskip('ase')
    from ase.io import read as ase_read, write as ase_write
    source = tmp_path / 'native-charge.xyz'
    source.write_text('2\nLattice="10 0 0 0 10 0 0 0 10" pbc="T T T" '
                      'energy=-1 total_charge=.8 '
                      'Properties=species:S:1:pos:R:3:charges:R:1\n'
                      'Al 1 2 3 .1\nO 2 3 4 .05\n')
    atoms = read_one(source)
    assert atoms.total_charge == .8
    exported = atoms.to_ase()
    path = tmp_path / 'ase-charge.xyz'
    ase_write(path, exported, format='extxyz')
    native = read_one(path)
    assert native.total_charge == .8
    assert native.energy == -1.
    np.testing.assert_allclose(native.charges, [.1, .05])
    converted = Atoms.from_ase(ase_read(path), read_results=False)
    assert converted.total_charge == .8
    assert converted.charges is None
    np.testing.assert_array_equal(native.numbers, atoms.numbers)
    np.testing.assert_allclose(native.positions, atoms.positions)
    np.testing.assert_array_equal(native.pbc, atoms.pbc)
    assert atoms.total_charge == .8
    np.testing.assert_array_equal(atoms.charges, [.1, .05])


@pytest.mark.parametrize('attach_calculator', [False, True])
def test_ase_computed_charges_roundtrip_has_independent_results(exportable_atoms,
                                                              attach_calculator):
    pytest.importorskip('ase')
    atoms = exportable_atoms
    atoms._impl.charges = [.25, -.1]
    source = atoms.to_ase(attach_calculator=attach_calculator)
    assert 'initial_charges' not in source.arrays
    if attach_calculator:
        assert 'charges' not in source.arrays
        np.testing.assert_array_equal(source.get_charges(), atoms.charges)
        stored = source.calc.results['charges']
    else:
        assert source.calc is None
        stored = source.arrays['charges']
    np.testing.assert_array_equal(stored, atoms.charges)
    converted = Atoms.from_ase(source)
    np.testing.assert_array_equal(converted.charges, [.25, -.1])
    stored[:] = 0
    np.testing.assert_array_equal(atoms.charges, [.25, -.1])
    np.testing.assert_array_equal(converted.charges, [.25, -.1])
    atoms._impl.charges = [.6, -.3]
    np.testing.assert_array_equal(converted.charges, [.25, -.1])


def test_ase_computed_charges_prefer_cache_and_skip_initial_guesses(exportable_atoms):
    pytest.importorskip('ase')
    from ase.calculators.singlepoint import SinglePointCalculator
    source = exportable_atoms.to_ase(attach_calculator=False)
    source.set_initial_charges([9., 8.])
    source.arrays['charges'] = np.array([.6, -.2])
    source.calc = SinglePointCalculator(source, charges=[.25, -.1])
    np.testing.assert_array_equal(Atoms.from_ase(source).charges, [.25, -.1])
    assert Atoms.from_ase(source, read_results=False).charges is None
    source.calc = None
    np.testing.assert_array_equal(Atoms.from_ase(source).charges, [.6, -.2])
    del source.arrays['charges']
    imported = Atoms.from_ase(source)
    assert imported.charges is None
    assert 'charges' not in imported.to_ase().calc.results


def test_ase_charge_only_results_survive_real_extxyz_roundtrips(tmp_path):
    pytest.importorskip('ase')
    from ase.io import read as ase_read, write as ase_write
    atoms = Atoms(numbers=[1, 8], positions=[[.1, .2, .3], [1., 2., 3.]],
                  cell=np.diag([4., 5., 20.]), pbc=[True, True, False])
    atoms._impl.charges = [.25, -.1]
    exported = atoms.to_ase()
    assert set(exported.calc.results) == {'charges'}
    assert 'charges' not in exported.arrays
    ase_path = tmp_path / 'ase-charges.xyz'
    ase_write(ase_path, exported, format='extxyz')
    assert ':charge:R:1' in ase_path.read_text().splitlines()[1]
    native = read_one(ase_path, format='extxyz')
    np.testing.assert_array_equal(native.charges, atoms.charges)
    np.testing.assert_array_equal(native.numbers, atoms.numbers)
    np.testing.assert_allclose(native.positions, atoms.positions)
    np.testing.assert_array_equal(native.pbc, atoms.pbc)
    assert native.energy is None
    np.testing.assert_array_equal(Atoms.from_ase(ase_read(ase_path)).charges, atoms.charges)

    native_path = tmp_path / 'native-charges.xyz'
    write_one(native_path, native, format='extxyz')
    assert ':charges:R:1' in native_path.read_text().splitlines()[1]
    np.testing.assert_allclose(ase_read(native_path).get_charges(), atoms.charges, atol=1e-12)


@pytest.mark.parametrize('values', [[.25], [np.nan, -.1]])
def test_ase_charges_require_complete_finite_arrays(exportable_atoms, values):
    pytest.importorskip('ase')
    from ase.calculators.singlepoint import SinglePointCalculator
    source = exportable_atoms.to_ase()
    source.calc = SinglePointCalculator(source, charges=values)
    with pytest.raises(ValueError, match=r'charges.*shape \(N,\).*finite'):
        Atoms.from_ase(source)
    exportable_atoms._impl.charges = values
    with pytest.raises(ValueError, match=r'charges.*shape \(N,\).*finite'):
        exportable_atoms.to_ase()


@pytest.mark.parametrize('attach_calculator', [False, True])
def test_ase_velocities_have_physical_units_and_independent_state(exportable_atoms,
                                                                attach_calculator):
    pytest.importorskip('ase')
    atoms = exportable_atoms
    expected = np.array([[10., -2., 0.], [0., 3., 4.]])
    atoms._impl.velocities = expected
    atoms._impl.charges = [.25, -.1]
    source = atoms.to_ase(attach_calculator=attach_calculator)
    assert source.has('momenta')
    assert 'velocities' not in source.arrays
    if attach_calculator:
        assert not ({'momenta', 'velocities'} & source.calc.results.keys())
    else:
        assert source.calc is None

    # SI reference is independent of the conversion's ASE time-unit factor.
    # 1 Angstrom/ps = 100 m/s; ASE may use a different CODATA release.
    kinetic_ev = .5 * np.sum(source.get_masses()[:, None] * 1.66053906660e-27
                             * (expected * 100) ** 2) / 1.602176634e-19
    assert source.get_kinetic_energy() == pytest.approx(kinetic_ev, rel=1e-7)
    converted = Atoms.from_ase(source)
    np.testing.assert_allclose(converted.velocities, expected, rtol=1e-14, atol=1e-14)
    geometry_and_state = Atoms.from_ase(source, read_results=False)
    np.testing.assert_allclose(geometry_and_state.velocities, expected, rtol=1e-14, atol=1e-14)
    assert geometry_and_state.energy is None
    assert geometry_and_state.forces is None and geometry_and_state.stress is None
    assert geometry_and_state.site_energies is None and geometry_and_state.charges is None

    source.arrays['momenta'][:] = 0
    np.testing.assert_array_equal(atoms.velocities, expected)
    np.testing.assert_allclose(converted.velocities, expected, rtol=1e-14, atol=1e-14)
    values = converted.velocities
    values[:] = 0
    np.testing.assert_allclose(converted.velocities, expected, rtol=1e-14, atol=1e-14)
    atoms._impl.velocities = np.zeros((2, 3))
    np.testing.assert_allclose(converted.velocities, expected, rtol=1e-14, atol=1e-14)


def test_ase_momenta_import_uses_source_masses_and_si_velocity(exportable_atoms):
    pytest.importorskip('ase')
    source = exportable_atoms.to_ase(attach_calculator=False)
    masses = np.array([4., 20.])
    source.set_masses(masses)
    source.set_momenta([[np.sqrt(2 * masses[0]), 0., 0.],
                        [0., -np.sqrt(4 * masses[1]), 0.]])
    assert source.get_kinetic_energy() == pytest.approx(3.)
    expected = np.zeros((2, 3))
    expected[0, 0] = np.sqrt(2 * 1.602176634e-19 / (masses[0] * 1.66053906660e-27)) / 100
    expected[1, 1] = -np.sqrt(4 * 1.602176634e-19 / (masses[1] * 1.66053906660e-27)) / 100
    converted = Atoms.from_ase(source)
    np.testing.assert_allclose(converted.velocities, expected, rtol=1e-7, atol=0)
    source.arrays['momenta'][:] = 0
    np.testing.assert_allclose(converted.velocities, expected, rtol=1e-7, atol=0)


def test_ase_explicit_isotope_masses_preserve_momenta_and_energy(tmp_path):
    ase = pytest.importorskip('ase')
    from ase.units import fs
    source = ase.Atoms('H2', positions=[[0., 0., 0.], [1., 0., 0.]], pbc=False,
                       masses=[2., 3.], momenta=[[2., 0., 0.], [0., 6., 0.]])
    atoms = Atoms.from_ase(source, read_results=False)
    np.testing.assert_array_equal(atoms.masses, [2., 3.])
    np.testing.assert_allclose(atoms.velocities, np.array([[1., 0., 0.], [0., 2., 0.]]) * (1000 * fs))
    for attach in [False, True]:
        exported = atoms.to_ase(attach_calculator=attach)
        assert exported.has('masses')
        np.testing.assert_array_equal(exported.get_masses(), [2., 3.])
        np.testing.assert_allclose(exported.get_momenta(), [[2., 0., 0.], [0., 6., 0.]])
        assert exported.get_kinetic_energy() == pytest.approx(7.)
    # Native generic EXTXYZ stores explicit masses and velocities together.
    path = tmp_path / 'isotopes.xyz'
    write_one(path, atoms)
    assert 'masses:R:1' in path.read_text().splitlines()[1]
    restored = read_one(path).to_ase()
    np.testing.assert_array_equal(restored.get_masses(), [2., 3.])
    np.testing.assert_allclose(restored.get_momenta(), [[2., 0., 0.], [0., 6., 0.]], atol=1e-9)
    assert restored.get_kinetic_energy() == pytest.approx(7.)


def test_ase_explicit_masses_are_owned_optional_metadata():
    ase = pytest.importorskip('ase')
    source = ase.Atoms('H2', masses=[2., 3.])
    atoms = Atoms.from_ase(source)
    source.arrays['masses'][:] = 9.
    values = atoms.masses
    values[:] = 8.
    exported = atoms.to_ase()
    exported.arrays['masses'][:] = 7.
    copied = atoms.copy()
    copied.masses = [4., 5.]
    np.testing.assert_array_equal(atoms.masses, [2., 3.])
    atoms.masses = None
    assert atoms.masses is None
    assert not atoms.to_ase().has('masses')
    plain = Atoms.from_ase(ase.Atoms('H2'))
    assert plain.masses is None
    assert not plain.to_ase().has('masses')


def test_ase_velocity_absence_and_real_zeros_remain_distinct(exportable_atoms):
    pytest.importorskip('ase')
    assert exportable_atoms.velocities is None
    source = exportable_atoms.to_ase()
    assert not source.has('momenta')
    assert Atoms.from_ase(source).velocities is None
    # A generic array is not ASE's standard momentum/velocity state.
    source.arrays['velocities'] = np.ones((2, 3))
    assert Atoms.from_ase(source, read_results=False).velocities is None
    source.set_velocities(np.zeros((2, 3)))
    converted = Atoms.from_ase(source, read_results=False)
    np.testing.assert_array_equal(converted.velocities, np.zeros((2, 3)))
    exported = converted.to_ase()
    assert exported.calc is None
    assert exported.has('momenta')
    np.testing.assert_array_equal(exported.get_velocities(), np.zeros((2, 3)))


def test_ase_velocities_require_complete_finite_arrays(exportable_atoms):
    pytest.importorskip('ase')
    source = exportable_atoms.to_ase()
    source.arrays['momenta'] = np.zeros((2, 2))
    with pytest.raises(ValueError, match=r'Velocities.*shape \(N, 3\).*finite'):
        Atoms.from_ase(source)
    exportable_atoms._impl.velocities = [[np.nan, 0., 0.], [0., 0., 0.]]
    with pytest.raises(ValueError, match=r'Velocities.*shape \(N, 3\).*finite'):
        exportable_atoms.to_ase()


@pytest.mark.parametrize('fractional', [False, True])
@pytest.mark.parametrize('target', ['ase', 'pymatgen'])
def test_coordinate_export_preserves_geometry_and_source(system, fractional, target):
    pytest.importorskip(target)
    _, atoms = system
    expected = atoms.positions.copy()
    if fractional:
        atoms.to_fractional()
    original = atoms.positions.copy()
    converted = atoms.to_ase() if target == 'ase' else atoms.to_pymatgen()
    positions = converted.positions if target == 'ase' else converted.cart_coords
    np.testing.assert_allclose(positions, expected)
    np.testing.assert_array_equal(atoms.positions, original)
    assert atoms._impl.cartesian == (not fractional)


def test_write_preserves_elements_and_native_species_indices(system, tmp_path):
    _, atoms = system
    write(tmp_path / 'single.xyz', atoms)
    np.testing.assert_array_equal(read_one(tmp_path / 'single.xyz').numbers, [13, 13])
    write(tmp_path / 'frames.xyz', [atoms, atoms.copy()])
    assert [frame.numbers.tolist() for frame in read(tmp_path / 'frames.xyz')] == [[13, 13]] * 2
    write(tmp_path / 'native.cfg', atoms)
    assert read_one(tmp_path / 'native.cfg').chemmap == [0, 0]
    assert atoms.chemmap == [0, 0]


def test_element_mapping_snapshots_survive_caller_list_reordering(tmp_path):
    elements = ['H', 'He']
    atoms = Atoms(numbers=[1, 2], positions=[[1., 2., 3.], [2., 3., 4.]],
                  cell=np.eye(3) * 10, pbc=False, elements=elements)
    copied = atoms.copy()
    elements.reverse()
    for frame in (atoms, copied):
        np.testing.assert_array_equal(frame.numbers, [1, 2])
        assert frame.chemmap == [0, 1]
    path = tmp_path / 'owned-elements.xyz'
    write(path, [atoms, copied], format='extxyz')
    lines = path.read_text().splitlines()
    assert [line.split()[0] for line in lines[2:4]] == ['H', 'He']
    assert [line.split()[0] for line in lines[6:8]] == ['H', 'He']
    restored = read(path, format='extxyz')
    assert len(restored) == 2
    for frame in restored:
        np.testing.assert_array_equal(frame.numbers, [1, 2])
    copied.numbers = [2, 1]
    np.testing.assert_array_equal(copied.numbers, [2, 1])
    assert copied.chemmap == [1, 0]
    np.testing.assert_array_equal(atoms.numbers, [1, 2])
    assert atoms.chemmap == [0, 1]


def test_ase_calculator_element_snapshot_controls_real_species_energy():
    pytest.importorskip('ase')
    from ase import Atoms as ASEAtoms
    from imrmlp.calculator import IMRMLPCalculator

    potential = Potential.from_settings({
        'pottype': 'lj612', 'ntypes': 2, '0-0': '1 1', '1-1': '4 1',
        'cut:0-0': 3, 'cut:0-1': 3, 'cut:1-1': 3,
    })
    elements = ['H', 'He']
    calculator = IMRMLPCalculator(potential=potential, elements=elements)
    elements.reverse()
    atoms = ASEAtoms('H2', positions=[[1., 1., 1.], [2.5, 1., 1.]],
                     cell=np.eye(3) * 20, pbc=False)
    atoms.calc = calculator
    # First actual calculation occurs after the caller changed its list.
    # Species0 has epsilon1; a live alias would choose species1/epsilon4.
    expected = 4 * ((1 / 1.5) ** 12 - (1 / 1.5) ** 6)
    assert atoms.get_potential_energy() == pytest.approx(expected, abs=1e-12)
    assert calculator._imrmlp_atoms.chemmap == [0, 0]


@pytest.mark.parametrize('format,multiple', [
    ('extxyz', False), ('bin', True), ('mlpcfg', True),
])
def test_write_fractional_coordinates_preserves_geometry_and_source(
        exportable_atoms, tmp_path, format, multiple):
    cartesian = exportable_atoms
    cartesian.cell = [[4., .3, .1], [.2, 5., .4], [.1, .2, 20.]]
    # MLPCFG does not store PBC; retain its existing fully periodic contract.
    if format == 'mlpcfg':
        cartesian.pbc = True
    expected_positions = cartesian.positions.copy()
    fractional = cartesian.copy()
    fractional.to_fractional()
    source_positions = fractional.positions.copy()
    source_chemmap = fractional.chemmap
    frames = [cartesian, fractional] if multiple else [fractional]
    path = tmp_path / f'coordinates.{format}'
    if multiple:
        write(path, frames, format=format)
    else:
        write_one(path, fractional, format=format)
    restored = read(path, format=format)
    assert len(restored) == len(frames)
    for frame in restored:
        assert frame._impl.cartesian
        np.testing.assert_allclose(frame.positions, expected_positions, atol=1e-12)
        np.testing.assert_allclose(frame.cell, cartesian.cell, atol=1e-12)
        np.testing.assert_array_equal(frame.numbers, cartesian.numbers)
        np.testing.assert_array_equal(frame.pbc, cartesian.pbc)
        assert frame.energy == pytest.approx(cartesian.energy)
        np.testing.assert_allclose(frame.site_energies, cartesian.site_energies)
        np.testing.assert_allclose(frame.forces, cartesian.forces)
        np.testing.assert_allclose(frame.stress, cartesian.stress)
    assert cartesian._impl.cartesian
    np.testing.assert_array_equal(cartesian.positions, expected_positions)
    assert not fractional._impl.cartesian
    np.testing.assert_array_equal(fractional.positions, source_positions)
    assert fractional.chemmap == source_chemmap
    np.testing.assert_array_equal(fractional.cell, cartesian.cell)
    np.testing.assert_array_equal(fractional.pbc, cartesian.pbc)
    assert fractional.energy == cartesian.energy
    np.testing.assert_array_equal(fractional.site_energies, cartesian.site_energies)
    np.testing.assert_array_equal(fractional.forces, cartesian.forces)
    np.testing.assert_array_equal(fractional.stress, cartesian.stress)


def test_mismatched_output_and_neb_maps_are_rejected(system, tmp_path):
    potential, atoms = system
    other = atoms.copy()
    other._elements = ['O']
    with pytest.raises(ValueError, match='element mapping'):
        write(tmp_path / 'mixed.xyz', [atoms, other])
    assert not (tmp_path / 'mixed.xyz').exists()
    with pytest.raises(ValueError, match='element mapping'):
        NEB(potential, n_images=3).run(atoms, other)


@pytest.mark.parametrize('charge', [2.5, 0., None])
def test_neb_preloads_common_explicit_charge_before_native_initialization(system, charge):
    potential, initial = system
    initial.total_charge = charge
    initial.energy = 17.
    initial._impl.charges = [.4, -.4]
    final = initial.copy()
    final.positions = final.positions + [.2, 0, 0]
    final.energy = 19.
    originals = [frame.copy() for frame in (initial, final)]
    result = NEB(potential, n_images=3, max_iter=1, neb_ftol=1e20,
                 climb=False).run(initial, final)
    assert len(result.images) == 3
    assert [image.total_charge for image in result.images] == [charge] * 3
    # Computed atomic charges from the endpoints must not be templated inward.
    assert result.images[1].charges is None
    for source, original in zip((initial, final), originals):
        assert source.total_charge == original.total_charge
        assert source.energy == original.energy
        np.testing.assert_array_equal(source.charges, original.charges)
        np.testing.assert_array_equal(source.positions, original.positions)
        np.testing.assert_array_equal(source.cell, original.cell)
        np.testing.assert_array_equal(source.pbc, original.pbc)
        np.testing.assert_array_equal(source.numbers, original.numbers)
        assert source._impl.cartesian == original._impl.cartesian


def run_driver(kind, potential, atoms, trajectory=True):
    if kind == 'neb':
        driver = NEB(potential, n_images=3, max_iter=1, neb_ftol=1e20,
                     climb=False, write_traj=trajectory)
        final = atoms.copy()
        final.positions = final.positions + [0.2, 0, 0]
        run = lambda: driver.run(atoms, final)
        filename = 'nebtraj.cfg'
    elif kind == 'dimer':
        driver = Dimer(potential, max_iter=1, rot_max_iter=1, trn_max_iter=1,
                       write_traj=trajectory)
        run = lambda: driver.run(atoms)
        filename = 'dimer.trj'
    else:
        driver = Relaxation(potential, max_iter=1, gtol=1e-20,
                            trajfile='relax.xyz' if trajectory else '')
        run = lambda: driver.run(atoms.copy())
        filename = 'relax.xyz'
    return driver, run, filename


@pytest.mark.parametrize('kind', ['neb', 'dimer', 'relax'])
def test_driver_trajectories_and_mappings_survive_repeated_runs(system, kind):
    potential, atoms = system
    driver, run, filename = run_driver(kind, potential, atoms)
    for _ in range(2):
        result = run()
        frames = read(filename, format='extxyz' if kind == 'dimer' else '')
        assert frames
        if kind == 'neb':
            outputs = result.images
        else:
            outputs = [result.saddle_atoms if kind == 'dimer' else result.final_atoms]
            assert all(frame.numbers.tolist() == [13, 13] for frame in frames)
        assert all(output.numbers.tolist() == [13, 13] for output in outputs)


def test_dimer_final_energy_and_labels_match_returned_geometry(system):
    potential, atoms = system
    potential.calculate(atoms)
    _, run, _ = run_driver('dimer', potential, atoms, trajectory=False)
    result = run()
    final = result.saddle_atoms
    assert not np.allclose(final.positions, atoms.positions)
    stored_energy, stored_forces, stored_stress = final.energy, final.forces, final.stress
    potential.calculate(final)
    assert result.energy == pytest.approx(final.energy)
    assert stored_energy == pytest.approx(final.energy)
    np.testing.assert_allclose(stored_forces, final.forces)
    np.testing.assert_allclose(stored_stress, final.stress)


def test_relaxation_fractional_input_has_the_same_initial_physics(system):
    potential, atoms = system
    atoms.cell = [[30., .3, .1], [.2, 31., .4], [.1, .2, 32.]]
    positions = atoms.positions.copy()
    reference = atoms.copy()
    potential.calculate(reference)
    assert np.linalg.norm(reference.forces) > 1e-6
    cartesian = atoms.copy()
    fractional = atoms.copy()
    fractional.to_fractional()

    # A large tolerance isolates initialization; this is not a convergence test.
    expected = Relaxation(potential, max_iter=1, gtol=1e20).run(cartesian)
    actual = Relaxation(potential, max_iter=1, gtol=1e20).run(fractional)
    assert actual.init_energy == pytest.approx(reference.energy, abs=1e-10)
    assert actual.init_energy == pytest.approx(expected.init_energy, abs=1e-10)
    assert actual.final_energy == pytest.approx(expected.final_energy, abs=1e-10)
    assert actual.final_atoms is fractional  # Existing run() result replaces its input.
    for final in [cartesian, fractional]:
        assert final._impl.cartesian
        np.testing.assert_allclose(final.positions, positions, atol=1e-12)
        np.testing.assert_allclose(final.cell, atoms.cell, atol=1e-12)
        np.testing.assert_allclose(final.forces, reference.forces, atol=1e-10)
        np.testing.assert_allclose(final.stress, reference.stress, atol=1e-10)
        np.testing.assert_array_equal(final.numbers, atoms.numbers)
    assert atoms._impl.cartesian
    np.testing.assert_array_equal(atoms.positions, positions)


def test_neb_mixed_coordinate_endpoints_fill_the_same_cartesian_path(system):
    potential, initial = system
    initial.cell = [[30., .3, .1], [.2, 31., .4], [.1, .2, 32.]]
    final = initial.copy()
    final.positions = final.positions + [.2, 0, 0]
    reference = initial.copy()
    potential.calculate(reference)
    assert np.linalg.norm(reference.forces) > 1e-6
    fractional = initial.copy()
    fractional.to_fractional()
    original_fractional = fractional.positions.copy()
    original_final = final.positions.copy()

    # Two empty interior images are filled natively before this initial evaluation.
    expected = NEB(potential, n_images=4, max_iter=1, neb_ftol=1e20).run(initial, final)
    actual = NEB(potential, n_images=4, max_iter=1, neb_ftol=1e20).run(fractional, final)
    assert len(actual.images) == 4
    assert actual.images[0].energy == pytest.approx(reference.energy, abs=1e-10)
    for index, (got, want) in enumerate(zip(actual.images, expected.images)):
        assert got._impl.cartesian
        np.testing.assert_allclose(got.positions, want.positions, atol=1e-12)
        np.testing.assert_allclose(got.positions, initial.positions + [index * .2 / 3, 0, 0],
                                   atol=1e-12)
        np.testing.assert_allclose(got.cell, initial.cell, atol=1e-12)
        assert got.energy == pytest.approx(want.energy, abs=1e-10)
        np.testing.assert_allclose(got.forces, want.forces, atol=1e-10)
        np.testing.assert_array_equal(got.numbers, initial.numbers)
    assert not fractional._impl.cartesian
    np.testing.assert_array_equal(fractional.positions, original_fractional)
    assert final._impl.cartesian
    np.testing.assert_array_equal(final.positions, original_final)


@pytest.mark.parametrize('kind', ['neb', 'dimer', 'relax'])
@pytest.mark.parametrize('failure', ['directory', 'full'])
def test_python_drivers_report_trajectory_output_failures(system, kind, failure):
    potential, atoms = system
    driver, run, filename = run_driver(kind, potential, atoms)
    path = Path(filename)
    if path.exists():
        path.unlink()
    if failure == 'directory':
        path.mkdir()
    else:
        if not Path('/dev/full').exists():
            pytest.skip('/dev/full is unavailable')
        path.symlink_to('/dev/full')
    with pytest.raises(RuntimeError, match='trajectory'):
        run()


@pytest.mark.parametrize('method', ['getE', 'getEFS', 'getEFSOMP'])
@pytest.mark.parametrize('species', [[], [0], [0, 0, 0], [-1, 0], [1000000, 0]])
def test_invalid_species_raise_without_evaluating(system, method, species):
    potential, atoms = system
    atoms.energy = 123
    atoms.chemmap = species
    with pytest.raises(ValueError, match='chemmap'):
        getattr(potential.inner, method)(atoms._impl)
    assert atoms.energy == 123


@pytest.mark.parametrize('method', ['getE', 'getEFS', 'getEFSOMP'])
def test_species_agnostic_mindist_accepts_nonnegative_species(tmp_path, monkeypatch, method):
    monkeypatch.chdir(tmp_path)
    potential = Potential('', pottype='mindist')
    atoms = Atoms(positions=[[1, 1, 1], [2, 1, 1]], cell=np.eye(3) * 20, pbc=False)
    atoms.chemmap = [0, 5]
    getattr(potential.inner, method)(atoms._impl)
    assert atoms.energy == pytest.approx((1 - potential.cutoff)**2)
    assert atoms.forces[0, 0] < 0
    for invalid in [[-1, 0], [0]]:
        atoms.chemmap = invalid
        with pytest.raises(ValueError, match='chemmap'):
            getattr(potential.inner, method)(atoms._impl)


@pytest.mark.parametrize('read_results', [False, True])
@pytest.mark.parametrize('attach_calculator', [False, True])
def test_ase_explicit_input_arrays_stay_separate_from_results(read_results, attach_calculator):
    ase = pytest.importorskip('ase')
    from ase.calculators.singlepoint import SinglePointCalculator
    source = ase.Atoms('HHe', positions=[[0., 0., 0.], [1.5, 0., 0.]], pbc=False)
    source.set_tags([2, 0])
    source.set_initial_charges([.8, -.2])
    source.set_initial_magnetic_moments([1.5, -.5])
    # Cache the results after setting the input state, so it remains valid.
    source.calc = SinglePointCalculator(source, energy=3., charges=[.25, -.1], magmoms=[.7, -.2])
    atoms = Atoms.from_ase(source, read_results=read_results)
    np.testing.assert_array_equal(atoms.tags, [2, 0])
    np.testing.assert_array_equal(atoms.initial_charges, [.8, -.2])
    np.testing.assert_array_equal(atoms.initial_magmoms, [1.5, -.5])
    assert atoms.total_charge is None  # Neither set of atomic charges supplies Q.
    if read_results:
        np.testing.assert_array_equal(atoms.charges, [.25, -.1])
        np.testing.assert_array_equal(atoms.magmoms, [.7, -.2])
        assert atoms.energy == 3.
    else:
        assert atoms.charges is None and atoms.magmoms is None and atoms.energy is None
    exported = atoms.to_ase(attach_calculator=attach_calculator)
    np.testing.assert_array_equal(exported.get_tags(), [2, 0])
    np.testing.assert_array_equal(exported.get_initial_charges(), [.8, -.2])
    np.testing.assert_array_equal(exported.get_initial_magnetic_moments(), [1.5, -.5])
    assert all(exported.has(name) for name in ('tags', 'initial_charges', 'initial_magmoms'))
    assert 'total_charge' not in exported.info
    if read_results and attach_calculator:
        np.testing.assert_array_equal(exported.get_charges(), [.25, -.1])
        np.testing.assert_array_equal(exported.get_magnetic_moments(), [.7, -.2])
        assert not ({'tags', 'initial_charges', 'initial_magmoms'} & exported.calc.results.keys())
    elif read_results:
        assert exported.calc is None
        np.testing.assert_array_equal(exported.arrays['charges'], [.25, -.1])
        np.testing.assert_array_equal(exported.arrays['magmoms'], [.7, -.2])
    else:
        assert exported.calc is None
        assert 'charges' not in exported.arrays and 'magmoms' not in exported.arrays


def test_ase_input_arrays_are_owned_optional_and_preserve_vector_moments():
    ase = pytest.importorskip('ase')
    from ase.calculators.singlepoint import SinglePointCalculator
    source = ase.Atoms('HHe')
    source.set_tags([4, -1])
    source.set_initial_charges([.4, -.3])
    vectors = np.array([[1., 2., 3.], [-1., -2., -3.]])
    source.set_initial_magnetic_moments(vectors)
    source.calc = SinglePointCalculator(source, charges=[.25, -.1], magmoms=[.7, -.2])
    atoms = Atoms.from_ase(source)
    expected = {'tags': np.array([4, -1]), 'initial_charges': np.array([.4, -.3]),
                'initial_magmoms': vectors.copy()}
    for name in expected:
        source.arrays[name][:] = 9
        returned = getattr(atoms, name)
        returned[:] = 8
    exported = atoms.to_ase(attach_calculator=False)
    copied = atoms.copy()
    for name, values in expected.items():
        exported.arrays[name][:] = 7
        np.testing.assert_array_equal(getattr(atoms, name), values)
        np.testing.assert_array_equal(getattr(copied, name), values)
    copied.tags = [9, 8]
    copied.initial_charges = None
    copied.initial_magmoms = [0., 0.]
    for name, values in expected.items():
        np.testing.assert_array_equal(getattr(atoms, name), values)
    atoms.initial_magmoms = [.1, .2]
    np.testing.assert_array_equal(atoms.initial_magmoms, [.1, .2])
    atoms.initial_magmoms = vectors
    np.testing.assert_array_equal(atoms.initial_magmoms, vectors)
    atoms.initial_charges = None
    assert atoms.initial_charges is None
    np.testing.assert_array_equal(atoms.tags, [4, -1])
    np.testing.assert_array_equal(atoms.initial_magmoms, vectors)
    atoms.initial_magmoms = None
    atoms.tags = None
    cleared = atoms.to_ase()
    assert not any(cleared.has(name) for name in expected)
    np.testing.assert_array_equal(atoms.charges, [.25, -.1])
    np.testing.assert_array_equal(atoms.magmoms, [.7, -.2])

    plain = Atoms.from_ase(ase.Atoms('HHe'), read_results=False)
    assert all(getattr(plain, name) is None for name in expected)
    assert not any(plain.to_ase().has(name) for name in expected)
    plain.tags = np.zeros(2, dtype=np.int64)
    plain.initial_charges = np.zeros(2)
    plain.initial_magmoms = np.zeros(2)
    zeros = plain.to_ase()
    assert all(zeros.has(name) for name in expected)
    assert plain.charges is None and plain.magmoms is None and plain.total_charge is None
    # Check the original Python integers before narrowing to native int.
    limits = np.array([-(2**31), 2**31 - 1], dtype=np.int64)
    plain.tags = limits
    np.testing.assert_array_equal(plain.tags, limits)
    with pytest.raises(ValueError, match='native integer range'):
        plain.tags = np.array([2**32 + 2, 0], dtype=np.int64)
    np.testing.assert_array_equal(plain.tags, limits)
    with pytest.raises(ValueError, match='integers'):
        plain.tags = [1.5, 0.]


def test_ase_input_arrays_survive_independent_extxyz_fields(tmp_path):
    pytest.importorskip('ase')
    path = tmp_path / 'input-state.xyz'
    path.write_text('2\nLattice="10 0 0 0 10 0 0 0 10" pbc="F F F" '
                    'Properties=species:S:1:pos:R:3:tags:I:1:initial_charges:R:1:'
                    'initial_magmoms:R:3:charges:R:1:magmoms:R:1\n'
                    'H 0 0 0 2 .8 1 2 3 .25 .7\n'
                    'He 1.5 0 0 0 -.2 -1 -2 -3 -.1 -.2\n')
    atoms = read_one(path)
    np.testing.assert_array_equal(atoms.tags, [2, 0])
    np.testing.assert_array_equal(atoms.initial_charges, [.8, -.2])
    np.testing.assert_array_equal(atoms.initial_magmoms, [[1., 2., 3.], [-1., -2., -3.]])
    exported = atoms.to_ase()
    np.testing.assert_array_equal(exported.get_initial_charges(), [.8, -.2])
    np.testing.assert_array_equal(exported.get_initial_magnetic_moments(), [[1., 2., 3.], [-1., -2., -3.]])
    np.testing.assert_array_equal(exported.get_charges(), [.25, -.1])
    np.testing.assert_array_equal(exported.get_magnetic_moments(), [.7, -.2])
    assert atoms.total_charge is None and 'total_charge' not in exported.info
    rewritten = tmp_path / 'rewritten-state.xyz'
    write_one(rewritten, atoms)
    header = rewritten.read_text().splitlines()[1]
    assert 'tags:I:1' in header and 'initial_charges:R:1' in header and 'initial_magmoms:R:3' in header
    restored = read_one(rewritten)
    np.testing.assert_array_equal(restored.tags, [2, 0])
    np.testing.assert_array_equal(restored.initial_charges, [.8, -.2])
    np.testing.assert_array_equal(restored.initial_magmoms, [[1., 2., 3.], [-1., -2., -3.]])
    np.testing.assert_array_equal(restored.charges, [.25, -.1])
    np.testing.assert_array_equal(restored.magmoms, [.7, -.2])
