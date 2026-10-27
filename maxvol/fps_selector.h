// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "maxvol/b_matrix_builder.h"
#include "base/structures.h"
#include <Eigen/Core>
#include <vector>
#include <random>

/// Farthest Point Sampling (FPS) selector for structure diversity.
///
/// Iteratively selects structures from a pool by maximizing the minimum
/// descriptor-space distance to already-selected structures.  Complements
/// MaxVol (which maximizes submatrix determinant) with a purely geometric
/// diversity criterion.
///
/// Typical workflow:
///   1. add_candidates(pool)              — build descriptor matrix
///   2. select(n) or select_by_threshold(t) — run FPS
///   3. get_selected()                    — retrieve results
class FPSSelector
{
public:
    enum DistanceMetric { Euclidean, Cosine };

    DistanceMetric metric = Euclidean;
    bool   verbose = false;
    int    random_seed = 42;

    FPSSelector() = default;
    explicit FPSSelector(BMatrixBuilder *builder);
    ~FPSSelector();
    FPSSelector(const FPSSelector &) = delete;
    FPSSelector &operator=(const FPSSelector &) = delete;

    // ── Candidate management ──────────────────────────────────────────
    void add_candidates(std::vector<Atoms> &cfgs);
    void clear_candidates();
    int  n_candidates() const;
    int  feat_dim() const;

    // ── Selection ─────────────────────────────────────────────────────
    /// Select exactly n structures.
    /// @param n         Number of structures to select.
    /// @param first_idx Initial point index (-1 = random).
    /// @return          Number of structures actually selected.
    int  select(int n, int first_idx = -1);

    /// Select until the min-distance to any selected drops below threshold.
    /// @param dist_thr  Stop when min-distance < dist_thr.
    /// @param max_n     Hard cap on selection count (0 = unlimited).
    /// @param first_idx Initial point index (-1 = random).
    /// @return          Number of structures selected.
    int  select_by_threshold(double dist_thr, int max_n = 0,
                             int first_idx = -1);

    // ── Results ───────────────────────────────────────────────────────
    const std::vector<Atoms> &get_selected() const;
    const std::vector<int>    &selected_indices() const;
    const std::vector<double> &selection_scores() const;
    const std::vector<int>    &selection_order() const;

private:
    BMatrixBuilder *builder_ = nullptr;
    bool own_builder_ = false;
    int  ncols_ = 0;

    // Descriptor matrix: n_cfgs x ncols, one row per structure
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> D_;

    std::vector<Atoms> cfgs_;

    // Results
    std::vector<Atoms>  selected_cfgs_;
    std::vector<int>    selected_indices_;
    std::vector<double> selection_scores_;
    std::vector<int>    selection_order_;

    // Internal
    std::mt19937 rng_;
    bool rng_seeded_ = false;

    void ensure_rng();
    double compute_distance(int i, int j) const;
    double compute_distance_vec(const Eigen::RowVectorXd &a,
                                const Eigen::RowVectorXd &b) const;
    int  compute_fps_order(int first_idx, int max_k, double dist_thr,
                           bool use_threshold);
};
