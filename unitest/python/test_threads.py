"""Changing the public thread request must preserve every reported MTP output."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("imrmlp._imrmlp")
import imrmlp

MODEL = Path(__file__).resolve().parents[1] / "test-data" / "pot4b.mtp"


@pytest.fixture(autouse=True)
def restore_threads():
    before = imrmlp.get_num_threads()
    yield
    imrmlp.set_num_threads(before)


def geometry(count=8):
    positions = np.array([[2.6*i + .023*j, 2.6*j + .017*k, 2.6*k + .019*i]
                          for i in range(2) for j in range(2) for k in range(2)])[:count]
    atoms = imrmlp.Atoms(positions=positions, cell=np.eye(3)*20, pbc=False)
    atoms.chemmap = [0]*count
    return atoms


def check_values(potential, count):
    expected, actual = geometry(count), geometry(count)
    potential.calculate(expected, use_omp=False)
    potential.calculate(actual, use_omp=True)
    for name in ("energy", "forces", "stress", "site_energies"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=1e-11, atol=1e-11, err_msg=name)
    assert np.asarray(actual.stress).shape == (3, 3)
    assert actual.energy == pytest.approx(np.sum(actual.site_energies), rel=1e-11, abs=1e-11)


@pytest.mark.parametrize("requested", [1, 2, 4])
def test_thread_setting_survives_model_construction(requested):
    imrmlp.set_num_threads(requested)
    assert imrmlp.get_num_threads() == requested
    first = imrmlp.Potential(MODEL)
    assert imrmlp.get_num_threads() == requested
    second = imrmlp.Potential(MODEL)
    assert imrmlp.get_num_threads() == requested
    for potential in (first, second):
        check_values(potential, 8)


@pytest.mark.parametrize("warm", [False, True])
def test_existing_models_follow_thread_changes_without_stale_results(warm):
    imrmlp.set_num_threads(4)
    models = [imrmlp.Potential(MODEL), imrmlp.Potential(MODEL)]
    if warm:
        for potential in models:
            check_values(potential, 8)
    # Cold reduction previously crashed. Warm reduction reused inactive buffers.
    for requested in (1, 2, 4, 1, 4):
        imrmlp.set_num_threads(requested)
        for potential in models:
            for count in (8, 2):
                check_values(potential, count)
                assert imrmlp.get_num_threads() == requested


@pytest.mark.parametrize("invalid", [0, -1, -10])
def test_invalid_thread_counts_do_not_change_the_request(invalid):
    imrmlp.set_num_threads(2)
    with pytest.raises(ValueError, match="positive"):
        imrmlp.set_num_threads(invalid)
    assert imrmlp.get_num_threads() == 2
    check_values(imrmlp.Potential(MODEL), 8)
