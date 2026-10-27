// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_BASIS_ANALYSIS_DECORRELATOR_H
#define IMRMLP_BASIS_ANALYSIS_DECORRELATOR_H

#include "CorrelationAnalyzer.h"
#include <Eigen/Dense>
#include <set>
#include <string>

namespace BasisAnalysis {

// Shared VIF pruning: use Schur-complement inverse updates in the positive
// definite phase; rebuild for rank deficiency or after 32 updates.
void prune_vif(const Eigen::MatrixXd &correlation, std::set<int> &selected,
               std::set<int> &removed, double target, int max_features = 0);

/// Abstract interface for de-correlation strategies.
///
/// Each concrete strategy selects a subset of scalar-moment indices
/// that minimizes redundancy according to a specific criterion.
class Decorrelator {
public:
    virtual ~Decorrelator() = default;

    /// Human-readable strategy name (used in diagnostic reports).
    virtual std::string name() const = 0;

    /// Select a subset of scalar moments.
    ///
    /// @param  corr         Pre-computed correlation analysis result
    /// @param  nscalar      Total number of scalar moments
    /// @return (selected_indices, removed_indices) — indices in [0, nscalar)
    virtual std::pair<std::set<int>, std::set<int>>
    select(const CorrelationResult &corr,
           int nscalar) = 0;
};

} // namespace BasisAnalysis

#endif // IMRMLP_BASIS_ANALYSIS_DECORRELATOR_H
