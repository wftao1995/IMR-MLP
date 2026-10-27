"""Format detection and descriptor fitting must work through the public API."""
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from imrmlp.io import guess_format


@pytest.mark.parametrize('filename,expected', [
    ('structure.xyz', 'extxyz'), ('trajectory.extxyz.gz', 'extxyz'),
    ('structure.cfg.gz', 'mlpcfg'), ('POSCAR', 'poscar'),
    ('XDATCAR6', 'xdatcar6'), ('vasprun.xml', 'vasprun'),
    ('dump.lammpstrj', 'lmptrj'), ('structure.lmpdat', 'lmpdat'),
    ('training.n2p2', 'n2p2'), ('structure.gro', 'gro'),
    ('something.unrecognized', ''),
])
def test_guess_format_returns_a_string(filename, expected):
    value = guess_format(Path(filename))
    assert isinstance(value, str)
    assert value == expected


def test_guessed_format_can_be_used_for_extxyz_io(tmp_path):
    path = tmp_path / 'atoms.extxyz'
    atoms = imrmlp.Atoms(numbers=[13, 13], positions=[[1., 1., 1.], [3., 1., 1.]],
                       cell=np.eye(3) * 15, elements=['Al'])
    atoms.energy = 1.25
    imrmlp.write(path, atoms, format=guess_format(path))
    restored = imrmlp.read_one(path, format=guess_format(path))
    np.testing.assert_array_equal(restored.numbers, atoms.numbers)
    np.testing.assert_allclose(restored.positions, atoms.positions)
    assert restored.energy == atoms.energy


def test_linear_training_saves_a_reloadable_descriptor_model(tmp_path):
    potential = imrmlp.Potential.from_settings({
        'pottype': 'descriptor_linear',
        'descriptor_linear:descriptor:dtype': 'behler',
        'descriptor_linear:descriptor:behler:cutoff': '5',
        'descriptor_linear:descriptor:behler:dispcount': '2',
        'descriptor_linear:descriptor:behler:dispselect': '1 2',
        'descriptor_linear:descriptor:behler:eta': '0.3',
    })
    assert potential.supports_linear_fit()
    training = []
    for distance in (1.5, 2.0, 2.8, 3.7):
        atoms = imrmlp.Atoms(numbers=[13, 13], elements=['Al'],
                           positions=[[2., 2., 2.], [2. + distance, 2., 2.]],
                           cell=np.eye(3) * 20)
        cutoff = .5 * (np.cos(np.pi * distance / 5) + 1)
        atoms.energy = 2 * cutoff * (.7 - .2 * np.exp(-.3 * distance**2))
        training.append(atoms)
    path = tmp_path / 'fitted.linear'
    trainer = imrmlp.LinearTrainer(force_weight=0, stress_weight=0, reg_L2=1e-12,
                                save_to=str(path))
    result = trainer.train(potential, training)
    assert result.train_erms < 1e-8
    assert path.stat().st_size > 100
    loaded = imrmlp.Potential(str(path), 'descriptor_linear')
    for atoms in training:
        before, after = atoms.copy(), atoms.copy()
        potential.calculate(before)
        loaded.calculate(after)
        assert after.energy == pytest.approx(before.energy, abs=1e-12)
        np.testing.assert_allclose(after.forces, before.forces, atol=1e-12)
        np.testing.assert_allclose(after.stress, before.stress, atol=1e-12)


def test_loading_empty_descriptor_file_fails_instead_of_creating_a_new_model(tmp_path):
    path = tmp_path / 'empty.linear'
    path.write_text('')
    with pytest.raises(RuntimeError, match='model format'):
        imrmlp.Potential(str(path), 'descriptor_linear')
