# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

pytest.importorskip("imrmlp._imrmlp")
from imrmlp._imrmlp import _NEB
from imrmlp import NEB, Potential


def test_neb_tangent_parameter():
    root = Path(__file__).resolve().parents[2]
    potential = Potential(str(root / "unitest/test-data/potential.mtp"))
    neb = NEB(potential, tangent_type=2)
    assert neb._impl.tangent_type == 2
    with pytest.raises(TypeError, match="tagent_type"):
        NEB(potential, tagent_type=2)


def test_neb_binding_uses_tangent_parameter():
    neb = _NEB()
    neb.tangent_type = 3
    assert neb.tangent_type == 3
    with pytest.raises(AttributeError):
        _ = neb.tagent_type
