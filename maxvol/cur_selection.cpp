// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "maxvol/cur_selection.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

// ── Wheel-based weighted sampling without replacement ────────────────────
// Adapted from the CUR implementation in apps/fastfit.cpp.
// Given scores (non-negative weights), selects k distinct indices with
// probability proportional to score.
static std::vector<int> wheel_sample(const std::vector<double> &scores, int k,
                                     std::mt19937 &rng)
{
    const int m = static_cast<int>(scores.size());
    if (k >= m) {
        std::vector<int> all(m);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    double total = std::accumulate(scores.begin(), scores.end(), 0.0);
    if (total <= 0) {
        // All scores zero — fall back to uniform random
        std::vector<int> idx(m);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        idx.resize(k);
        return idx;
    }

    // Working copies: we'll remove selected items
    std::vector<int>    available(m);
    std::vector<double> remaining_scores(m);
    std::iota(available.begin(), available.end(), 0);
    for (int i = 0; i < m; ++i) remaining_scores[i] = scores[i] / total;

    // Build cumulative wheel
    std::vector<double> wheel(m);
    double cum = 0;
    for (int i = 0; i < m; ++i) {
        cum += remaining_scores[i];
        wheel[i] = cum;
    }
    wheel.back() = 1.0; // guard against rounding

    std::uniform_real_distribution<double> dice(0.0, 1.0);
    std::vector<int> selected;
    selected.reserve(k);

    for (int sel = 0; sel < k; ++sel) {
        double val = dice(rng);
        int picked = -1;
        for (int j = 0; j < static_cast<int>(available.size()); ++j) {
            if (val <= wheel[j]) {
                picked = j;
                break;
            }
        }
        if (picked < 0) picked = static_cast<int>(available.size()) - 1;

        selected.push_back(available[picked]);

        // Remove picked from available and rebuild wheel
        available.erase(available.begin() + picked);
        remaining_scores.erase(remaining_scores.begin() + picked);
        wheel.erase(wheel.begin() + picked);

        if (sel == k - 1) break; // done

        double new_total = 0;
        for (double s : remaining_scores) new_total += s;
        if (new_total <= 0) {
            // Remaining scores are zero — fill with uniform random
            std::shuffle(available.begin(), available.end(), rng);
            for (int t = 0; sel + 1 + t < k; ++t)
                selected.push_back(available[t]);
            break;
        }
        cum = 0;
        for (int i = 0; i < static_cast<int>(remaining_scores.size()); ++i) {
            remaining_scores[i] /= new_total;
            cum += remaining_scores[i];
            wheel[i] = cum;
        }
        wheel.back() = 1.0;
    }

    return selected;
}

// ═══════════════════════════════════════════════════════════════════════════
CURResult cur_select_rows(const Eigen::MatrixXd &A, int k,
                          bool deterministic, int seed, bool compute_error)
{
    CURResult result;
    if (k < 0)
        throw std::invalid_argument("CUR row count k must be nonnegative");
    if (!A.allFinite())
        throw std::invalid_argument("CUR descriptor matrix must contain only finite values");
    if (A.rows() > std::numeric_limits<int>::max() || A.cols() > std::numeric_limits<int>::max())
        throw std::length_error("CUR matrix dimensions exceed the supported index range");
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());

    // Empty dimensions and an explicit zero-row request have an empty row
    // space. Do not send a zero-column matrix to Eigen's SVD.
    if (m == 0 || n == 0 || k == 0) {
        if (compute_error)
            result.reconstruction_error = (A.size() == 0 || A.isZero(0.0)) ? 0.0 : 1.0;
        return result;
    }
    if (k > m) k = m;

    // ── 1. Thin SVD ──────────────────────────────────────────────────────
    // Use rank r = min(k, n) for leverage scores.
    int r = std::min(k, n);

    Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(A);
    if (svd.info() != Eigen::Success || !svd.matrixU().allFinite())
        throw std::runtime_error("CUR singular value decomposition failed");
    const auto &U = svd.matrixU(); // m × min(m, n)

    // Use at most r columns of U
    int rank = std::min(r, static_cast<int>(U.cols()));

    // ── 2. Leverage scores: p_i = ||U(i, 1:rank)||² / rank ──────────────
    result.scores.resize(m);
    for (int i = 0; i < m; ++i) {
        double sq = 0;
        for (int j = 0; j < rank; ++j)
            sq += U(i, j) * U(i, j);
        result.scores[i] = sq / rank;
    }

    // ── 3. Select rows ───────────────────────────────────────────────────
    if (deterministic) {
        // Top-k by leverage score
        std::vector<int> indices(m);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&](int a, int b) {
                              if (result.scores[a] == result.scores[b]) return a < b;
                              return result.scores[a] > result.scores[b];
                          });
        result.selected_indices.assign(indices.begin(), indices.begin() + k);
    } else {
        std::mt19937 rng(seed);
        result.selected_indices = wheel_sample(result.scores, k, rng);
    }

    // ── 4. Optional reconstruction error ──────────────────────────────────
    if (compute_error && !result.selected_indices.empty()) {
        int ks = static_cast<int>(result.selected_indices.size());

        // Work at unit scale: the row space and relative error are invariant
        // under a global rescaling, while norms of finite A can overflow or
        // underflow when computed by summing squares at the original scale.
        const double scale = A.cwiseAbs().maxCoeff();
        if (scale == 0.0) {
            result.reconstruction_error = 0.0;
            return result;
        }
        const Eigen::MatrixXd scaled_A = A / scale;
        // Build C = scaled_A[selected, :]
        Eigen::MatrixXd C(ks, n);
        for (int i = 0; i < ks; ++i)
            C.row(i) = scaled_A.row(result.selected_indices[i]);

        // C⁺ = pseudoinverse(C), then project each row onto the selected span.
        Eigen::MatrixXd C_pinv = C.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::MatrixXd A_recon = scaled_A * C_pinv * C;
        result.reconstruction_error = (scaled_A - A_recon).stableNorm() / scaled_A.stableNorm();
        if (!std::isfinite(result.reconstruction_error))
            throw std::runtime_error("CUR row-space reconstruction failed");
    }

    return result;
}
