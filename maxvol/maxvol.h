// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <cstdint>
#include <string>
#include <vector>

/// Maximum Volume (MaxVol) row selection algorithm.
///
/// Given a tall matrix A (m rows x n cols, m >> n), greedily selects k rows
/// whose submatrix has (approximately) maximum absolute determinant.
///
/// Uses Sherman-Morrison rank-1 updates to avoid O(k³) full inverse
/// recomputation at each swap iteration.  A numerically-stable fallback
/// (full recompute) triggers when the Sherman-Morrison denominator is near
/// zero or every N iterations.
///
/// Ref: Goreinov, Tyrtyshnikov, Zamarashkin, Linear Algebra Appl. 261 (1997).
class MaxVol
{
public:
    using RowMatrixXd = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

    double tolerance = 1e-6;      // converge when max |B[l,i]| <= 1+tol
    int    max_iterations = 100;  // iteration safety cap
    int    max_swaps = 0;         // swap count limit (0 = unlimited)
    bool   verbose = false;

    /// How often to force a full recompute (for numerical stability).
    /// 0 = disabled (recompute only when SM denominator is unsafe).
    int    sm_recompute_every = 20;

    /// A single swap event recorded during MaximizeVol.
    struct SwapStep
    {
        int    ind_in;   // row index (in the full matrix A) that entered
        int    ind_out;  // row index that was replaced
        double gain;     // |B[l,i]| — the volume multiplier

        SwapStep() = default;
        SwapStep(int _in, int _out, double _gain)
            : ind_in(_in), ind_out(_out), gain(_gain) {}
    };

    MaxVol() = default;

    // ── Selection API ──────────────────────────────────────────────────

    /// Select k rows maximizing submatrix volume.
    /// @param A  Design matrix (m x n, m >> n). Accepts ColMajor / RowMajor.
    /// @param k  Rows to select; values above m are clamped to m.
    /// @return   Indices of selected rows in [0, m).
    std::vector<int> select_rows(const Eigen::Ref<const Eigen::MatrixXd> &A,
                                 int k);
    std::vector<int> select_rows(const Eigen::Ref<const RowMatrixXd> &A,
                                 int k);
    template <typename Derived>
    std::vector<int> select_rows(const Eigen::MatrixBase<Derived> &A, int k)
    {
        if constexpr (Derived::IsRowMajor) {
            const Eigen::Ref<const RowMatrixXd> ref(A.derived());
            return select_rows(ref, k);
        } else {
            const Eigen::Ref<const Eigen::MatrixXd> ref(A.derived());
            return select_rows(ref, k);
        }
    }

    /// MaxVol "gain" score for every row.
    Eigen::VectorXd compute_scores(
        const Eigen::Ref<const Eigen::MatrixXd> &A,
        const std::vector<int> &selected);
    Eigen::VectorXd compute_scores(
        const Eigen::Ref<const RowMatrixXd> &A,
        const std::vector<int> &selected);
    template <typename Derived>
    Eigen::VectorXd compute_scores(const Eigen::MatrixBase<Derived> &A,
                                   const std::vector<int> &selected)
    {
        if constexpr (Derived::IsRowMajor) {
            const Eigen::Ref<const RowMatrixXd> ref(A.derived());
            return compute_scores(ref, selected);
        } else {
            const Eigen::Ref<const Eigen::MatrixXd> ref(A.derived());
            return compute_scores(ref, selected);
        }
    }

    /// Score candidate rows only, using the cached pseudo-inverse.
    /// @param B_candidate  Candidate row vectors (nrows x n_cols).
    /// @param selected     Row indices of the active set (same as for compute_scores).
    /// @return             nrows x 1 vector — per-row MaxVol scores, or an
    ///                     empty vector if warm_cache() has not prepared this
    ///                     selected set / column count.
    Eigen::VectorXd score_rows(
        const Eigen::Ref<const Eigen::MatrixXd> &B_candidate,
        const std::vector<int> &selected);
    Eigen::VectorXd score_rows(
        const Eigen::Ref<const RowMatrixXd> &B_candidate,
        const std::vector<int> &selected);
    template <typename Derived>
    Eigen::VectorXd score_rows(const Eigen::MatrixBase<Derived> &B_candidate,
                               const std::vector<int> &selected)
    {
        if constexpr (Derived::IsRowMajor) {
            const Eigen::Ref<const RowMatrixXd> ref(B_candidate.derived());
            return score_rows(ref, selected);
        } else {
            const Eigen::Ref<const Eigen::MatrixXd> ref(B_candidate.derived());
            return score_rows(ref, selected);
        }
    }

    /// Per-subset max gain scores (analogous to MLIP-3 rows_ranges).
    /// @param A            Design matrix (m rows x n columns).
    /// @param selected     Indices of currently selected rows.
    /// @param row_ranges   Start row of each subset ([0, r1, r2, ..., m]).
    /// @param out_grades   Output: max |B[l,i]| within each subset.
    void compute_subset_scores(
        const Eigen::Ref<const Eigen::MatrixXd> &A,
        const std::vector<int> &selected,
        const std::vector<int> &row_ranges,
        std::vector<double> &out_grades);
    void compute_subset_scores(
        const Eigen::Ref<const RowMatrixXd> &A,
        const std::vector<int> &selected,
        const std::vector<int> &row_ranges,
        std::vector<double> &out_grades);
    template <typename Derived>
    void compute_subset_scores(const Eigen::MatrixBase<Derived> &A,
                               const std::vector<int> &selected,
                               const std::vector<int> &row_ranges,
                               std::vector<double> &out_grades)
    {
        if constexpr (Derived::IsRowMajor) {
            const Eigen::Ref<const RowMatrixXd> ref(A.derived());
            compute_subset_scores(ref, selected, row_ranges, out_grades);
        } else {
            const Eigen::Ref<const Eigen::MatrixXd> ref(A.derived());
            compute_subset_scores(ref, selected, row_ranges, out_grades);
        }
    }

    /// Incremental: add rows to an existing selection.
    std::vector<int> add_rows(const Eigen::Ref<const Eigen::MatrixXd> &A,
                              const std::vector<int> &current,
                              int add_k);
    std::vector<int> add_rows(const Eigen::Ref<const RowMatrixXd> &A,
                              const std::vector<int> &current,
                              int add_k);
    template <typename Derived>
    std::vector<int> add_rows(const Eigen::MatrixBase<Derived> &A,
                              const std::vector<int> &current, int add_k)
    {
        if constexpr (Derived::IsRowMajor) {
            const Eigen::Ref<const RowMatrixXd> ref(A.derived());
            return add_rows(ref, current, add_k);
        } else {
            const Eigen::Ref<const Eigen::MatrixXd> ref(A.derived());
            return add_rows(ref, current, add_k);
        }
    }

    /// Pre-warm the inverse / pseudo-inverse cache without computing full scores.
    /// Call before score_rows() in multi-candidate scoring to avoid O(m·k·n)
    /// repeated matmul work.
    void warm_cache(const std::vector<int> &selected,
                    const Eigen::Ref<const Eigen::MatrixXd> &A);
    void warm_cache(const std::vector<int> &selected,
                    const Eigen::Ref<const RowMatrixXd> &A);
    template <typename Derived>
    void warm_cache(const std::vector<int> &selected,
                    const Eigen::MatrixBase<Derived> &A)
    {
        if constexpr (Derived::IsRowMajor) {
            const Eigen::Ref<const RowMatrixXd> ref(A.derived());
            warm_cache(selected, ref);
        } else {
            const Eigen::Ref<const Eigen::MatrixXd> ref(A.derived());
            warm_cache(selected, ref);
        }
    }

    // ── Diagnostics ────────────────────────────────────────────────────

    /// Swap history from the most recent select_rows() call.
    const std::vector<SwapStep> &swap_history() const { return swap_tracker_; }

    /// Log generalized row volume of C = A[selected, :].
    /// Only valid after select_rows() has been called.
    double log_determinant() const { return log_det_; }

    // ── Serialization (checkpoint / resume) ────────────────────────────

    /// Save MaxVol state to a binary file.
    void save(const std::string &filename) const;
    /// Save MaxVol state to a binary stream (for embedding in single-file state).
    void save(std::ostream &os) const;

    /// Load MaxVol checkpoint metadata and return the saved row selection.
    /// Runtime diagnostics and factorization caches are not persisted.
    /// Only version-3 checkpoints are accepted.
    std::vector<int> load(const std::string &filename);
    /// Load MaxVol checkpoint metadata from a binary stream.
    std::vector<int> load(std::istream &is);

private:
    template <typename MatrixType>
    std::vector<int> select_rows_impl(const MatrixType &A, int k);

    template <typename MatrixType>
    Eigen::VectorXd compute_scores_impl(const MatrixType &A,
                                        const std::vector<int> &selected);

    template <typename MatrixType>
    Eigen::VectorXd score_rows_impl(const MatrixType &B_candidate,
                                    const std::vector<int> &selected);

    template <typename MatrixType>
    void compute_subset_scores_impl(const MatrixType &A,
                                    const std::vector<int> &selected,
                                    const std::vector<int> &row_ranges,
                                    std::vector<double> &out_grades);

    template <typename MatrixType>
    std::vector<int> add_rows_impl(const MatrixType &A,
                                   const std::vector<int> &current,
                                   int add_k);

    template <typename MatrixType>
    void warm_cache_impl(const std::vector<int> &selected,
                         const MatrixType &A);

    /// QR column-pivoting on A^T for a well-conditioned initial subset.
    template <typename MatrixType>
    std::vector<int> qr_init(const MatrixType &A, int k);

    /// Compute truncated-SVD pseudo-inverse of C = A[selected, :] and cache it.
    /// Returns true if the cache was already valid (no recomputation needed).
    template <typename MatrixType>
    bool ensure_cache(const std::vector<int> &selected,
                      const MatrixType &A);

    /// One iteration with Sherman-Morrison update.
    template <typename MatrixType>
    bool maxvol_iteration(const MatrixType &A,
                          std::vector<int> &selected,
                          Eigen::MatrixXd &B,
                          Eigen::MatrixXd &C,
                          Eigen::MatrixXd &C_inv,
                          int &sm_steps_since_recompute,
                          bool &sm_updates_enabled);

    // Per-select_rows state (exposed via diagnostics)
    std::vector<SwapStep> swap_tracker_;
    double log_det_ = 0.0;
    int    cold_start_k_ = 0;   // k at cold start (for serialization)

    // Cache for compute_scores — C_inv reused across grade_all calls
    mutable std::vector<int> cached_selected_;
    mutable int cached_ncols_ = -1;
    mutable Eigen::MatrixXd cached_pinv_;
    mutable Eigen::MatrixXd cached_active_;
    mutable Eigen::VectorXd cached_gram_inv_diag_;
    mutable Eigen::VectorXd cached_active_leverage_;
    mutable bool cached_wide_ = false;
    mutable std::uint64_t cached_signature_ = 0;
};
