"""CUR boundary contracts must be safe through the public Python interface."""
import numpy as np
import pytest

pytest.importorskip("imrmlp._imrmlp")
import imrmlp


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0)])
@pytest.mark.parametrize("deterministic", [False, True])
def test_empty_cur_input_returns_an_empty_selection(shape, deterministic):
    result = imrmlp.cur_select(np.empty(shape), 1, deterministic=deterministic,
                             compute_error=True)
    assert result.selected_indices == []
    assert result.scores.size == 0
    assert result.reconstruction_error == 0


def test_zero_and_negative_cur_row_requests():
    matrix = np.eye(3)
    result = imrmlp.cur_select(matrix, 0, compute_error=True)
    assert result.selected_indices == []
    assert result.scores.size == 0
    assert result.reconstruction_error == 1
    assert imrmlp.cur_select(matrix, 0).reconstruction_error == -1
    with pytest.raises(ValueError, match="nonnegative"):
        imrmlp.cur_select(matrix, -1)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("count", [0, 1, 5])
def test_cur_rejects_nonfinite_values_even_for_zero_row_requests(invalid, count):
    matrix = np.eye(3)
    matrix[1, 2] = invalid
    with pytest.raises(ValueError, match="finite"):
        imrmlp.cur_select(matrix, count)


def test_cur_row_oversampling_and_analytic_error():
    matrix = np.array([[3., 0., 0.], [0., 2., 0.], [0., 0., 1.],
                       [3., 0., 0.], [0., 0., 0.]])
    result = imrmlp.cur_select(matrix, 2, deterministic=True, compute_error=True)
    np.testing.assert_allclose(result.scores, [.25, .5, 0., .25, 0.], atol=2e-13)
    assert result.selected_indices[0] == 1
    assert result.selected_indices[1] in (0, 3)
    assert result.reconstruction_error == pytest.approx(1 / np.sqrt(23), abs=2e-13)
    for requested in (4, 8):
        selected = imrmlp.cur_select(matrix, requested, deterministic=True, compute_error=True)
        assert len(set(selected.selected_indices)) == min(requested, len(matrix))
        assert all(0 <= i < len(matrix) for i in selected.selected_indices)
        assert selected.reconstruction_error < 1e-12
