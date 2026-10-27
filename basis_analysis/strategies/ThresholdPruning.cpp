// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "ThresholdPruning.h"
#include <algorithm>
#include <cmath>

namespace BasisAnalysis {

ThresholdPruning::ThresholdPruning(double pearson_threshold, double vif_threshold)
    : pearson_threshold_(pearson_threshold), vif_threshold_(vif_threshold) {}

std::string ThresholdPruning::name() const {
    return "ThresholdPruning(pearson>" + std::to_string(pearson_threshold_) +
           ", vif>" + std::to_string(vif_threshold_) + ")";
}

std::pair<std::set<int>, std::set<int>>
ThresholdPruning::select(const CorrelationResult &corr,
                          int nscalar) {
    // Start with all features selected
    std::set<int> selected;
    for (int i = 0; i < nscalar; ++i) {
        selected.insert(i);
    }

    // ── Phase 1: Pearson correlation pruning ──
    // Compute max_abs_corr for each feature
    std::vector<std::pair<double, int>> features_by_corr;
    for (int i = 0; i < nscalar; ++i) {
        double max_corr = 0.0;
        for (int j = 0; j < nscalar; ++j) {
            if (i == j) continue;
            max_corr = std::max(max_corr, std::abs(corr.pearson(i, j)));
        }
        features_by_corr.emplace_back(max_corr, i);
    }
    // Sort by descending max correlation (most correlated first to remove);
    // use stable_sort for deterministic behavior with equal scores
    std::stable_sort(features_by_corr.begin(), features_by_corr.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    std::set<int> removed;
    for (const auto &[max_corr, idx] : features_by_corr) {
        if (removed.count(idx)) continue;
        // Check if this feature is highly correlated with any kept feature
        bool should_remove = false;
        for (int kept : selected) {
            if (kept == idx) continue;
            if (std::abs(corr.pearson(idx, kept)) > pearson_threshold_) {
                should_remove = true;
                break;
            }
        }
        if (should_remove) {
            selected.erase(idx);
            removed.insert(idx);
        }
    }

    prune_vif(corr.pearson, selected, removed, vif_threshold_);

    return {selected, removed};
}

} // namespace BasisAnalysis
