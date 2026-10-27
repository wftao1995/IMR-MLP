# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

"""MTP uses canonical potential type identifiers."""
from pathlib import Path
import os

import pytest

pytest.importorskip("imrmlp._imrmlp")
from imrmlp.potential import Potential


def test_mtp_type_loads_potential():
    filename = Path(os.environ.get("IMRMLP_TEST_DATA_ROOT", Path(__file__).resolve().parents[2] / "unitest/test-data")) / "potential.mtp"
    assert Potential.MTP == "mtp"
    potential = Potential(str(filename))
    assert potential.pottype == "mtp"
    assert potential.cutoff > 0
    assert potential.species_count > 0
    assert Potential(str(filename), "mtp").pottype == "mtp"
