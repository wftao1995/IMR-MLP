// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_BASIS_ANALYSIS_DIAGNOSTIC_REPORT_H
#define IMRMLP_BASIS_ANALYSIS_DIAGNOSTIC_REPORT_H

#include "CorrelationAnalyzer.h"
#include <string>
#include <set>
#include <vector>

namespace BasisAnalysis {

/// Generates structured diagnostic report files from a correlation analysis
/// and de-correlation results.
///
/// Output files (prefix = "correlation" by default):
///   prefix_summary.txt      — human-readable summary
///   prefix_matrix.dat       — full Pearson correlation matrix
///   prefix_vif.dat          — VIF per feature
///   prefix_eigenvalues.dat  — eigenvalue spectrum
///   prefix_pairs.dat        — highly correlated pairs
///   prefix_selection.dat    — selection outcome
class DiagnosticReport {
public:
    DiagnosticReport() = default;

    /// Feed the correlation analysis result.
    void set_analysis(const CorrelationResult &result);

    /// Add the outcome of a de-correlation strategy run.
    void add_strategy_result(const std::string &strategy_name,
                             const std::set<int> &selected,
                             const std::set<int> &removed);

    /// Write all report files.
    /// @param prefix  Path prefix for output files
    /// @param addback_used  Whether add_back_path was applied
    void write_all(const std::string &prefix = "correlation",
                   bool addback_used = false);

private:
    void write_summary(const std::string &filename, bool addback_used);
    void write_correlation_matrix(const std::string &filename);
    void write_vif(const std::string &filename);
    void write_eigenvalues(const std::string &filename);
    void write_pairs(const std::string &filename);
    void write_selection(const std::string &filename);

    CorrelationResult corr_;
    struct StrategyOutcome {
        std::string name;
        std::set<int> selected;
        std::set<int> removed;
    };
    std::vector<StrategyOutcome> outcomes_;
};

} // namespace BasisAnalysis

#endif
