"""Explicit species clones preserve old predictions and survive native saves."""
import json
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from training_cli_fixture import MODEL


def structure(z, elements):
    return imrmlp.Atoms(numbers=[z, z], elements=elements, pbc=False,
                      positions=[[1., 1., 1.], [2.2, 1.3, 1.1]], cell=np.eye(3)*12)


def load_model(tmp_path, tagged=True):
    text = MODEL
    if tagged:
        metadata = dict(schema='gpml.mtp.metadata', version=1,
                        type='MTP', model='linear', elements=['H'])
        text = text.replace('potential_tag = test-only',
                            'potential_tag = ' + json.dumps(metadata))
    path = tmp_path/'original.mtp'
    path.write_text(text)
    return imrmlp.Potential(path)


def test_append_species_clones_old_ids_and_saves_metadata(tmp_path):
    potential = load_model(tmp_path)
    assert potential.supports_species_append()
    a = structure(1, ['H'])
    potential.calculate(a)
    old_coefficients = potential.coefficients.copy()
    mapping = potential.append_species([0, 0], ['He', 'Li'])
    assert potential.species_count == 3
    np.testing.assert_array_equal(potential.coefficients[np.asarray(mapping)], old_coefficients)
    for number in (1, 2, 3):
        clone = structure(number, ['H', 'He', 'Li'])
        potential.calculate(clone)
        assert clone.energy == pytest.approx(a.energy, abs=1e-12)
        np.testing.assert_allclose(clone.forces, a.forces, atol=1e-12)
    path = tmp_path/'expanded.mtp'
    potential.save(path)
    saved = path.read_text()
    assert all(f'"{name}"' in saved for name in ['H', 'He', 'Li'])
    restored = imrmlp.Potential(path)
    assert restored.species_count == 3
    for number in (1, 2, 3):
        clone = structure(number, ['H', 'He', 'Li'])
        restored.calculate(clone)
        assert clone.energy == pytest.approx(a.energy, abs=1e-12)
        np.testing.assert_allclose(clone.forces, a.forces, atol=1e-12)


def test_append_species_errors_preserve_parameters(tmp_path):
    potential = load_model(tmp_path)
    before = potential.coefficients.copy()
    for sources, names in [([1], ['He']), ([-1], ['He']), ([0], None),
                           ([0], ['H']), ([0], ['invalid']), ([0, 0], ['He'])]:
        with pytest.raises(RuntimeError):
            potential.append_species(sources, names)
        assert potential.species_count == 1
        np.testing.assert_array_equal(potential.coefficients, before)
    assert potential.append_species([]) == list(range(len(before)))
    core = imrmlp.Potential.from_settings({'pottype': 'zbl', 'z': '1', 'r0': '.5', 'r1': '4'})
    assert not core.supports_species_append()
    with pytest.raises(RuntimeError):
        core.append_species([0])


def test_untagged_append_uses_explicit_internal_clone_ids(tmp_path):
    potential = load_model(tmp_path, tagged=False)
    old = potential.coefficients.copy()
    mapping = potential.append_species([0])
    assert potential.species_count == 2
    np.testing.assert_array_equal(potential.coefficients[np.asarray(mapping)], old)
    potential.save(tmp_path/'indices.mtp')
    assert imrmlp.Potential(tmp_path/'indices.mtp').species_count == 2
