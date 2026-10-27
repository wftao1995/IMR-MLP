# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""Package imports, failure semantics and ownership of saddle/path results."""
import gc
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Dimer, NEB, Potential

ROOT = Path(__file__).resolve().parents[2]


def atoms():
    a = Atoms(positions=[[1., 1., 1.], [3., 1., 1.]], cell=np.eye(3)*30,
              elements=['H', 'He'])
    a.chemmap = [0, 0]
    return a


def potential():
    return Potential(str(ROOT / 'unitest/test-data/potential.mtp'))


def test_dimer_explicit_mode_errors_are_exceptions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = Dimer(potential(), modecar='missing', write_traj=True)
    with pytest.raises(RuntimeError, match='MODECAR'):
        d.run(atoms())
    assert not (tmp_path / 'dimer.trj').exists()


def test_dimer_repeated_run_keeps_potential_and_copies_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'MODECAR').write_text('1 0 0\n-1 0 0\n')
    d = Dimer(potential(), max_iter=0, write_traj=True, modecar='MODECAR')
    gc.collect()
    initial = atoms()
    first = d.run(initial)
    assert not first.converged and first.exit_code == 1
    assert first.stop_reason == 'max_iterations'
    assert first.n_iter == 0
    assert np.isfinite(first.energy) and np.isfinite(first.curvature)
    assert first.max_force > 0
    saved = first.saddle_atoms.positions.copy()
    initial.positions = initial.positions + 0.1
    second = d.run(initial)
    np.testing.assert_allclose(first.saddle_atoms.positions, saved)
    assert not np.array_equal(first.saddle_atoms.positions, second.saddle_atoms.positions)
    del d, initial
    gc.collect()
    assert np.isfinite(second.saddle_atoms.energy)
    assert (tmp_path / 'dimer.trj').stat().st_size > 0
    assert '# status: max_iterations' in (tmp_path / 'dimer.dat').read_text()


@pytest.mark.parametrize('vcneb', [False, True])
def test_neb_package_trajectory_and_result_lifetime(tmp_path, monkeypatch, vcneb):
    monkeypatch.chdir(tmp_path)
    initial, final = atoms(), atoms()
    final.positions = final.positions + 0.1
    if vcneb:
        final.cell = final.cell * 1.01
    n = NEB(potential(), n_images=3, max_iter=1, climb=False,
            write_traj=True, vcneb=vcneb, pressure=0.1 if vcneb else 0,
            neb_ftol=1e20)
    result = n.run(initial, final)
    assert result.converged
    assert len(result.images) == 3
    saved = result.images[1].positions.copy()
    del n
    gc.collect()
    np.testing.assert_allclose(result.images[1].positions, saved)
    assert np.isfinite(result.images[1].energy)
    assert any(p.stat().st_size > 0 for p in tmp_path.glob('*'))


@pytest.mark.parametrize('vcneb', [False, True])
def test_neb_evaluates_endpoint_properties_without_input_labels(tmp_path, monkeypatch, vcneb):
    monkeypatch.chdir(tmp_path)
    pot = potential()
    initial, final = atoms(), atoms()
    final.positions = final.positions + 0.1
    assert initial.energy is None and final.energy is None
    pressure = 0.1 if vcneb else 0.0
    result = NEB(pot, n_images=3, max_iter=1, neb_ftol=1e20,
                 vcneb=vcneb, pressure=pressure).run(initial, final)
    for supplied, returned in zip((initial, final), (result.images[0], result.images[-1])):
        reference = supplied.copy()
        pot.calculate(reference)
        enthalpy = reference.energy + pressure * abs(np.linalg.det(reference.cell)) / 160.21766208
        assert returned.energy == pytest.approx(enthalpy, abs=1e-8)
        np.testing.assert_allclose(returned.forces, reference.forces, atol=1e-10)
        np.testing.assert_allclose(returned.positions, supplied.positions, atol=1e-12)
    assert initial.energy is None and final.energy is None
