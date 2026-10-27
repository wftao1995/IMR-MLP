// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_BASIS_ANALYSIS_STRATEGIES_ITERATIVE_GREEDY_VIF_H
#define IMRMLP_BASIS_ANALYSIS_STRATEGIES_ITERATIVE_GREEDY_VIF_H

#include "../Decorrelator.h"

namespace BasisAnalysis {

/// Iteratively remove the feature with the highest VIF until all VIFs are
/// below the target threshold.
///
/// At each iteration:
///   1. Build the sub-correlation matrix from the currently selected features.
///   2. Compute VIF for each.
///   3. If max(VIF) > vif_target, remove the feature with max VIF and repeat.
///
/// Optionally also enforces an upper bound on the number of features.
class IterativeGreedyVIF : public Decorrelator {
public:
    /// @param vif_target    Stop when max(VIF) <= this (default 5.0)
    /// @param max_features  Maximum features to retain, in addition to the VIF target (0 = no limit)
    IterativeGreedyVIF(double vif_target = 5.0, int max_features = 0);

    std::string name() const override;
    std::pair<std::set<int>, std::set<int>>
    select(const CorrelationResult &corr,
           int nscalar) override;

private:
    double vif_target_;
    int max_features_;  // 0 = unlimited
};

} // namespace BasisAnalysis

#endif
