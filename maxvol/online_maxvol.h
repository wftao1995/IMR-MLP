// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "maxvol/b_matrix_builder.h"
#include <Eigen/Core>
#include <iosfwd>
#include <vector>

using OnlineRowMatrix = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

struct OnlineSwapStep {
    BRowRef entering;
    BRowRef outgoing;
    double gain = 0.0;
};

/// Persistent active matrix for approximate incremental MaxVol updates.
///
/// Candidate batches are not retained after maximize_with_candidates() returns.
/// Consequently, a later exchange may make a previously rejected historical
/// row extrapolative again.  Use full MaxVol or exact streaming when dominance
/// over the complete cumulative pool is required.
class OnlineMaxVolState
{
public:
    /// Scratch storage for repeated scoring calls. Matrix intermediates are
    /// bounded by score_block_rows(); scores grow to fit the requested rows.
    /// OpenMP callers need one workspace per worker.
    struct ScoreWorkspace {
        Eigen::MatrixXd z;
        Eigen::MatrixXd coeff;
        Eigen::VectorXd scores;
        int rows = 0;
    };

    using ScoreView = Eigen::VectorBlock<Eigen::VectorXd>;

    /// Row block size keeping the two possible matrix intermediates within
    /// 4 MiB per worker (or one row each for exceptionally wide active sets).
    int score_block_rows() const;

    void clear();
    void reset_identity(int ncols, double init_value = 1.0);
    bool initialize_from_rows(
        const Eigen::Ref<const OnlineRowMatrix> &C,
        const std::vector<BRowRef> &refs = {});

    Eigen::VectorXd score_rows(
        const Eigen::Ref<const OnlineRowMatrix> &candidates) const;
    /// Score rows without reallocating intermediates on every call.
    ScoreView score_rows(
        const Eigen::Ref<const OnlineRowMatrix> &candidates,
        ScoreWorkspace &workspace) const;

    int maximize_with_candidates(
        const Eigen::Ref<const OnlineRowMatrix> &candidates,
        const std::vector<BRowRef> &candidate_refs,
        double threshold, int swap_limit = 0);

    void save(std::ostream &os) const;
    bool load(std::istream &is);

    bool empty() const { return C_.rows() == 0; }
    bool valid() const { return valid_; }
    int active_size() const { return static_cast<int>(C_.rows()); }
    int coeff_count() const { return static_cast<int>(C_.cols()); }
    double log_determinant() const { return log_det_; }
    const OnlineRowMatrix &active_rows() const { return C_; }
    const std::vector<BRowRef> &selected_refs() const { return selected_refs_; }
    const std::vector<OnlineSwapStep> &last_swaps() const { return last_swaps_; }

private:
    enum class FactorMode {
        None,
        SquareInverse,
        GramInverse,
        RegularizedGramInverse,
    };

    bool refactorize();
    bool refactorize_square(const Eigen::MatrixXd &dense);
    bool refactorize_gram(const Eigen::MatrixXd &dense);
    int append_candidates(const Eigen::Ref<const OnlineRowMatrix> &candidates,
                          const std::vector<BRowRef> &candidate_refs,
                          int append_limit);
    bool update_square(const Eigen::RowVectorXd &coeff,
                       int slot,
                       const Eigen::RowVectorXd &replacement);
    bool update_gram(int slot,
                     const Eigen::RowVectorXd &replacement);
    double factorization_error() const;

    OnlineRowMatrix C_;
    // C^-1 for a full-rank square state, otherwise
    // (C C^T + regularization I)^-1 for a wide state.
    Eigen::MatrixXd inverse_;
    std::vector<BRowRef> selected_refs_;
    double log_det_ = 0.0;
    double regularization_ = 0.0;
    FactorMode factor_mode_ = FactorMode::None;
    bool valid_ = false;
    int updates_since_factorization_ = 0;
    std::vector<OnlineSwapStep> last_swaps_;
};
