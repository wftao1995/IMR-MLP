"""Fixed physical core + fitted residual, through the public Python interface."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
import imrmlp
from training_cli_fixture import MODEL


def test_fixed_core_copies_labels_and_saves_complete_hybrid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'base.mtp').write_text(MODEL)
    potential = imrmlp.Potential('base.mtp')
    core = imrmlp.Potential.from_settings({'pottype': 'zbl', 'z': '1', 'r0': '.5', 'r1': '4'})
    core_before = []
    data = []
    for distance in (1.1, 1.4, 1.9):
        a = imrmlp.Atoms(numbers=[1, 1], elements=['H'], pbc=False,
                       positions=[[1., 1., 1.], [1.+distance, 1.2, 1.1]], cell=np.eye(3)*12)
        potential.calculate(a)
        b = a.copy()
        core.calculate(b)
        a.energy = a.energy + b.energy
        a.forces = a.forces + b.forces
        a.stress = a.stress + b.stress
        data.append(a)
        core_before.append((b.energy, b.forces.copy()))
    labels = [(a.energy, a.forces.copy(), a.positions.copy()) for a in data]
    x = potential.coefficients
    begin, end = potential.parameter_groups['species']
    x[begin:end] += .2
    potential.coefficients = x
    trainer = imrmlp.NonlinearTrainer(max_iterations=2, trainable_groups=['species'],
                                    parameter_scaling='none', bfgs_m=2, penalty_weight=0,
                                    save_to='residual.mtp')
    result = trainer.train(potential, data, fixed_core=core, hybrid_output='complete')
    assert result.hybrid_file == 'complete'
    combined = imrmlp.Potential('complete', 'hybrid')
    for a, (energy, forces, positions), (core_energy, core_forces) in zip(data, labels, core_before):
        assert a.energy == energy
        np.testing.assert_array_equal(a.forces, forces)
        np.testing.assert_array_equal(a.positions, positions)
        residual_prediction, core_prediction, full = a.copy(), a.copy(), a.copy()
        potential.calculate(residual_prediction)
        core.calculate(core_prediction)
        combined.calculate(full)
        assert core_prediction.energy == core_energy
        np.testing.assert_array_equal(core_prediction.forces, core_forces)
        assert full.energy == pytest.approx(residual_prediction.energy + core_energy, abs=1e-11)
        np.testing.assert_allclose(full.forces, residual_prediction.forces + core_forces, atol=1e-11)


def test_fixed_core_rejects_alias_before_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'base.mtp').write_text(MODEL)
    potential = imrmlp.Potential('base.mtp')
    a = imrmlp.Atoms(numbers=[1, 1], elements=['H'], pbc=False,
                   positions=[[1., 1., 1.], [2.3, 1.2, 1.1]], cell=np.eye(3)*12)
    potential.calculate(a)
    before = potential.coefficients.copy()
    trainer = imrmlp.NonlinearTrainer(max_iterations=1, save_to='residual.mtp')
    with pytest.raises(RuntimeError, match='separate'):
        trainer.train(potential, [a], fixed_core=potential)
    np.testing.assert_array_equal(potential.coefficients, before)
    assert not (tmp_path/'residual.mtp').exists()


def test_low_level_fixed_core_uses_default_save_destination(tmp_path, monkeypatch):
    from imrmlp._imrmlp import _fit_nonlinear
    monkeypatch.chdir(tmp_path)
    (tmp_path/'base.mtp').write_text(MODEL)
    potential = imrmlp.Potential('base.mtp')
    core = imrmlp.Potential.from_settings({'pottype': 'zbl', 'z': '1', 'r0': '.5', 'r1': '4'})
    a = imrmlp.Atoms(numbers=[1, 1], elements=['H'], pbc=False,
                   positions=[[1., 1., 1.], [2.3, 1.2, 1.1]], cell=np.eye(3)*12)
    potential.calculate(a)
    b = a.copy()
    core.calculate(b)
    a.energy += b.energy
    a.forces = a.forces + b.forces
    a.stress = a.stress + b.stress
    # save_to intentionally omitted: the native trainer supplies its default.
    result = _fit_nonlinear(potential.inner, [a._impl], [],
                            {'bfgs_maxiteration': '1', 'bfgs_m': '2', 'parameter_scaling': 'none',
                             'penalty_weight': '0', 'trainable_groups': 'species'},
                            [], core.inner)
    assert (tmp_path/result['hybrid_file']).is_file()
    assert result['hybrid_file'].endswith('.hybrid')
