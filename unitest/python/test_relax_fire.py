"""FIRE step limits and downhill mode through the public Python wrapper."""
import numpy as np
import pytest

pytest.importorskip('imrmlp._imrmlp')
from imrmlp import Atoms, Potential, Relaxation


@pytest.fixture
def fire_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'index.map').write_text('H\n')
    potential = Potential.from_settings({
        'pottype': 'lj612', 'ntypes': 1, '0-0': '1 1', 'cut:0-0': 3,
    })
    # The unit cell makes the default fractional optimizer coordinates equal
    # to Cartesian coordinates, independently of the native preconditioner.
    atoms = Atoms(numbers=[1, 1], elements=['H'],
                  positions=[[.1, .5, .5], [.9, .5, .5]],
                  cell=np.eye(3), pbc=False)
    return potential, atoms


def test_automatic_first_step_tracks_python_max_move_on_reuse(fire_input):
    potential, initial = fire_input
    relaxation = Relaxation(potential, driver='fire', max_iter=1,
                            max_move=.001, gtol=1e-12)
    for maximum in (.001, .00002):
        relaxation._impl.max_move = maximum
        atoms = initial.copy()
        start = atoms.positions.copy()
        result = relaxation.run(atoms)
        assert result.n_iter == 1
        assert not result.converged
        assert np.linalg.norm(atoms.positions - start) == pytest.approx(.1 * maximum)


@pytest.mark.parametrize('downhill', [False, True])
def test_downhill_option_runs_with_the_requested_mode(fire_input, downhill):
    potential, atoms = fire_input
    potential.calculate(atoms)
    initial_energy = atoms.energy
    relaxation = Relaxation(potential, driver='fire', downhill_check=downhill,
                            max_iter=3, max_move=.01, gtol=1e-12)
    assert relaxation._impl.downhill_check is downhill
    result = relaxation.run(atoms)
    assert 0 < result.n_iter <= 3
    if downhill:
        assert result.final_energy <= initial_energy
