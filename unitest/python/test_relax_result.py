# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp._imrmlp import _Relaxation
import imrmlp
from imrmlp import Atoms, Potential


def test_saddle_search_exports():
    for name in ("Dimer", "DimerResult", "_Dimer"):
        assert hasattr(imrmlp, name)
    assert imrmlp._Dimer is imrmlp._imrmlp._Dimer
    assert imrmlp.NEB is not None
    assert imrmlp.Relaxation is not None


def test_status_and_modified_iteration_limit():
    root = Path(imrmlp.__file__).resolve().parents[1]
    potential = Potential(str(Path(os.environ.get('IMRMLP_TEST_DATA_ROOT', root / 'unitest/test-data')) / 'potential.mtp'))
    atoms = Atoms(positions=[[1., 1., 1.], [3.5, 1., 1.]], cell=np.eye(3)*12)
    atoms.chemmap = [0, 0]
    driver = _Relaxation()
    driver.driver = 'lbfgspp'
    driver.init(potential.inner, atoms._impl)
    driver.max_iter = 1
    driver.gtol = 1e-12
    driver.run()
    assert driver.n_iter <= 1
    assert np.isfinite(driver.energy)
    assert driver.converged == (driver.stop_reason == 'gradient_tolerance')
    for name in ('converged', 'stop_reason', 'n_iter'):
        with pytest.raises(AttributeError):
            setattr(driver, name, getattr(driver, name))


def test_high_level_relaxation_returns_actual_final_structure():
    from imrmlp import Relaxation
    root = Path(imrmlp.__file__).resolve().parents[1]
    potential = Potential(str(Path(os.environ.get('IMRMLP_TEST_DATA_ROOT', root / 'unitest/test-data')) / 'potential.mtp'))
    atoms = Atoms(positions=[[1., 1., 1.], [3.5, 1., 1.]], cell=np.eye(3)*12)
    atoms.chemmap = [0, 0]
    initial = atoms.positions.copy()
    relax = Relaxation(potential, max_iter=1, gtol=1e-12)
    result = relax.run(atoms)
    assert result.final_atoms is atoms
    assert not np.allclose(atoms.positions, initial, atol=1e-10, rtol=0)
    assert np.isfinite(result.init_energy)
    assert np.isfinite(result.final_energy)
    assert result.n_iter <= 1
    assert result.converged == relax.converged
