// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_BASIS_ANALYSIS_STRATEGIES_THRESHOLD_PRUNING_H
#define IMRMLP_BASIS_ANALYSIS_STRATEGIES_THRESHOLD_PRUNING_H

#include "../Decorrelator.h"

namespace BasisAnalysis {

/// Prune features based on Pearson correlation or VIF thresholds.
///
/// Algorithm:
///   1. Sort features by descending max_abs_corr (to any other feature).
///   2. Iteratively remove features that are correlated above the
///      pearson threshold with any kept feature.
///   3. Then iteratively remove the feature with the highest VIF
///      until all VIFs are below the vif threshold.
///
/// Pearson pruning is O(n^2); VIF pruning uses spectral factorizations and
/// inverse updates in the positive definite phase. Selection is deterministic.
class ThresholdPruning : public Decorrelator {
public:
    /// @param pearson_threshold  Remove if |pearson| > this (default 0.95)
    /// @param vif_threshold      Remove until all VIF <= this (default 10.0)
    ThresholdPruning(double pearson_threshold = 0.95,
                     double vif_threshold = 10.0);

    std::string name() const override;
    std::pair<std::set<int>, std::set<int>>
    select(const CorrelationResult &corr,
           int nscalar) override;

private:
    double pearson_threshold_;
    double vif_threshold_;
};

} // namespace BasisAnalysis

#endif
