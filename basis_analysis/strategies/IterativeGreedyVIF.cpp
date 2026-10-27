// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "IterativeGreedyVIF.h"
#include <cmath>
#include <vector>

namespace BasisAnalysis {

IterativeGreedyVIF::IterativeGreedyVIF(double vif_target, int max_features)
    : vif_target_(vif_target), max_features_(max_features) {}

std::string IterativeGreedyVIF::name() const {
    std::string s = "IterativeGreedyVIF(target=" + std::to_string(vif_target_);
    if (max_features_ > 0) s += ", max_features=" + std::to_string(max_features_);
    s += ")";
    return s;
}

std::pair<std::set<int>, std::set<int>>
IterativeGreedyVIF::select(const CorrelationResult &corr,
                            int nscalar) {
    std::set<int> selected;
    for (int i = 0; i < nscalar; ++i) selected.insert(i);
    std::set<int> removed;

    prune_vif(corr.pearson, selected, removed, vif_target_, max_features_);

    return {selected, removed};
}

} // namespace BasisAnalysis
