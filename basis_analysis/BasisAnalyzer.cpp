// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "BasisAnalyzer.h"
#include "potentials/potential_mtp.h"
#include <stdexcept>

namespace BasisAnalysis {

BasisAnalyzer::BasisAnalyzer(Potentials::PotMTP &pot, size_t max_samples)
    : pot_(pot), collector_(pot, max_samples) {}

std::unique_ptr<CorrelationResult>
BasisAnalyzer::analyze(const std::vector<Atoms> &atoms) {
    // Phase 1: Collect raw descriptors
    Eigen::MatrixXd desc = collector_.collect(atoms);
    // Phase 2: Correlation analysis
    CorrelationAnalyzer ca(desc);
    auto result = std::make_unique<CorrelationResult>(ca.compute_all());

    // Feed to report
    report_.set_analysis(*result);

    last_result_ = std::make_unique<CorrelationResult>(*result);
    return result;
}

std::pair<std::set<int>, std::set<int>>
BasisAnalyzer::apply_strategy(const CorrelationResult &corr,
                               std::unique_ptr<Decorrelator> strategy,
                               bool add_back_path) {
    int nscalar = pot_.alpha_scalar_moments;
    if (!strategy || corr.nfeatures != nscalar || corr.nsamples < 0 ||
        corr.pearson.rows() != nscalar || corr.pearson.cols() != nscalar)
        throw std::invalid_argument("Invalid basis analysis result or strategy");
    if (corr.nsamples == 0) {
        // No observations justify removing any basis functions.
        std::set<int> all;
        for (int i = 0; i < nscalar; ++i) all.insert(i);
        report_.add_strategy_result(strategy->name(), all, {});
        return {all, {}};
    }

    auto [selected, removed] = strategy->select(corr, nscalar);

    // Optional DAG preservation
    if (add_back_path && !removed.empty()) {
        pot_.add_back_path_scalars_to_remove_id(removed);
        selected.clear();
        for (int i = 0; i < nscalar; ++i) {
            if (removed.count(i) == 0) selected.insert(i);
        }
    }

    report_.add_strategy_result(strategy->name(), selected, removed);
    return {selected, removed};
}

std::pair<std::set<int>, std::set<int>>
BasisAnalyzer::run(const std::vector<Atoms> &atoms,
                   std::unique_ptr<Decorrelator> strategy,
                   bool add_back_path) {
    auto corr = analyze(atoms);
    return apply_strategy(*corr, std::move(strategy), add_back_path);
}

} // namespace BasisAnalysis
