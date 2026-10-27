# Copyright (c) 2026 IMR-MLP contributors
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

pytest.importorskip("imrmlp._imrmlp")
import imrmlp
from imrmlp._imrmlp import _ALPotential


def test_online_al_rejects_removed_mode():
    assert _ALPotential.create({}) is None
    assert _ALPotential.create({"mode": "none"}) is None
    with pytest.raises(ValueError, match="Unsupported active-learning mode"):
        _ALPotential.create({"mode": "committee"})


def test_maxvol_and_cur_remain_usable():
    matrix = np.array([[1., 0.], [0., 1.], [1., 1.], [2., -1.]])
    selector = imrmlp.MaxVol()
    rows = selector.select_rows(matrix, 2)
    assert len(set(rows)) == 2
    assert np.linalg.matrix_rank(matrix[rows]) == 2
    scores = selector.compute_scores(matrix, rows)
    assert np.all(np.isfinite(scores))
    result = imrmlp.cur_select(matrix, k=2, deterministic=True)
    assert len(set(result.selected_indices)) == 2
    assert np.all(np.isfinite(result.scores))


@pytest.mark.parametrize("dtype", ["soap"])
def test_descriptor_rejects_removed_type(dtype):
    with pytest.raises(RuntimeError, match=f"Unknown descriptor type: {dtype}"):
        imrmlp.Descriptor(dtype)


def test_bondorder_descriptor_remains_usable():
    descriptor = imrmlp.Descriptor("bondorder", {"bondorder:maxl": "4"})
    assert descriptor.dtype == "bondorder"
    assert descriptor.feature_size > 0
