// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Core>
#include <vector>

/// Result of CUR row selection.
struct CURResult {
    /// Indices of selected rows in the original matrix.
    std::vector<int> selected_indices;
    /// Leverage score for every row (m entries, sum ≈ 1), or empty for a
    /// zero-row request or a matrix with an empty dimension.
    std::vector<double> scores;
    /// Relative reconstruction error ||A - A·C⁺·C||_F / ||A||_F, where
    /// C = A[selected_indices, :]. Defined as zero when A is zero/empty.
    /// For k=0 and nonzero A the empty projection has error one.
    /// Negative if compute_error was false.
    double reconstruction_error = -1.0;
};

/// CUR row selection via SVD leverage scores.
///
/// Given a descriptor matrix A (m structures × n features), selects min(k,m)
/// distinct rows using leverage scores from the first min(k,m,n) left singular
/// vectors. k counts rows, so k>n is allowed. Zero singular values are not
/// truncated; a degenerate singular subspace can have nonunique scores.
///
/// Two selection strategies:
///   - Probabilistic (default): weighted random sampling without replacement.
///   - Deterministic (deterministic=true): top-k rows by leverage score,
///     breaking equal-score ties by increasing original row index.
///
/// k=0 or either matrix dimension being zero returns empty indices and scores,
/// without an SVD. Negative k and non-finite matrix values are rejected with
/// std::invalid_argument, including for k=0. Oversized dimensions throw
/// std::length_error; numerical decomposition failures throw std::runtime_error.
///
/// @param A              Descriptor matrix (m × n). Row-major preferred for
///                       compatibility with DescriptorBuilder output, but
///                       any layout is accepted.
/// @param k              Nonnegative row count; capped at m, not n.
/// @param deterministic  If true, select top-k by score.
/// @param seed           Random seed for probabilistic sampling.
/// @param compute_error  If true, also compute relative reconstruction error
///                       (adds O(m·k·n) cost).
CURResult cur_select_rows(const Eigen::MatrixXd &A, int k,
                          bool deterministic = false, int seed = 42,
                          bool compute_error = false);
