// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "DiagnosticReport.h"
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace BasisAnalysis {

namespace {
std::ofstream open_report(const std::string &filename) {
    std::ofstream ofs(filename);
    if (!ofs) throw std::runtime_error("Failed to open diagnostic report: " + filename);
    return ofs;
}

void finish_report(std::ofstream &ofs, const std::string &filename) {
    ofs.close();
    if (!ofs) throw std::runtime_error("Failed to write diagnostic report: " + filename);
}
}

void DiagnosticReport::set_analysis(const CorrelationResult &result) {
    corr_ = result;
    outcomes_.clear();
}

void DiagnosticReport::add_strategy_result(const std::string &strategy_name,
                                            const std::set<int> &selected,
                                            const std::set<int> &removed) {
    outcomes_.push_back({strategy_name, selected, removed});
}

void DiagnosticReport::write_all(const std::string &prefix, bool addback_used) {
    write_summary(prefix + "_summary.txt", addback_used);
    write_correlation_matrix(prefix + "_matrix.dat");
    write_vif(prefix + "_vif.dat");
    write_eigenvalues(prefix + "_eigenvalues.dat");
    write_pairs(prefix + "_pairs.dat");
    write_selection(prefix + "_selection.dat");
}

void DiagnosticReport::write_summary(const std::string &filename, bool addback_used) {
    auto ofs = open_report(filename);

    ofs << "# ============================================================\n";
    ofs << "#  MTP Basis Correlation Analysis — Diagnostic Report\n";
    ofs << "# ============================================================\n\n";

    ofs << "## Data Summary\n";
    ofs << "  Scalar moments (features): " << corr_.nfeatures << "\n";
    ofs << "  Sampled atoms:             " << corr_.nsamples << "\n";
    ofs << "  Numerical rank:            " << corr_.rank() << "\n\n";

    ofs << "  Constant/unsampled features: " << (corr_.pearson.diagonal().array() == 0.).count() << "\n\n";
    ofs << "## Correlation Overview\n";
    ofs << "  Condition number:          " << std::scientific
        << corr_.cond_number << "\n";

    // Count highly correlated
    auto pairs = corr_.highly_correlated_pairs(0.95);
    ofs << "  Pairs with |r| > 0.95:    " << pairs.size() << "\n";

    // Mean and max |correlation|
    double sum_abs_corr = 0.0, max_abs_corr = 0.0;
    int npairs = 0;
    for (int i = 0; i < corr_.nfeatures; ++i) {
        for (int j = i + 1; j < corr_.nfeatures; ++j) {
            double val = std::abs(corr_.pearson(i, j));
            sum_abs_corr += val;
            if (val > max_abs_corr) max_abs_corr = val;
            ++npairs;
        }
    }
    double mean_abs_corr = (npairs > 0) ? (sum_abs_corr / npairs) : 0.0;
    ofs << "  Mean |pearson|:            " << std::fixed
        << std::setprecision(6) << mean_abs_corr << "\n";
    ofs << "  Max  |pearson|:            " << max_abs_corr << "\n\n";

    // VIF summary
    int vif_gt_10 = 0, vif_gt_5 = 0;
    double max_vif = 0.0;
    for (int i = 0; i < corr_.vif.size(); ++i) {
        double v = corr_.vif[i];
        if (v > max_vif) max_vif = v;
        if (v > 10.0) ++vif_gt_10;
        if (v > 5.0) ++vif_gt_5;
    }
    ofs << "## VIF Summary\n";
    ofs << "  Max VIF:                   " << std::fixed
        << std::setprecision(2) << max_vif << "\n";
    ofs << "  Features with VIF > 10:    " << vif_gt_10 << "\n";
    ofs << "  Features with VIF > 5:     " << vif_gt_5 << "\n\n";

    // Eigenvalue spectrum summary
    if (corr_.eigenvalues.size() > 0) {
        double total = corr_.eigenvalues.sum();
        double cum = 0.0;
        ofs << "## Eigenvalue Spectrum (top 10)\n";
        int top_n = std::min(10, static_cast<int>(corr_.eigenvalues.size()));
        for (int i = 0; i < top_n; ++i) {
            cum += corr_.eigenvalues[i];
            ofs << "  " << std::setw(3) << i << ": " << std::scientific
                << corr_.eigenvalues[i]
                << "  (" << std::fixed << std::setprecision(1)
                << (total > 0 ? 100.0 * cum / total : 0.) << "% cumulative)\n";
        }
        ofs << "\n";
    }

    // Strategy results
    for (const auto &out : outcomes_) {
        ofs << "## Strategy: " << out.name << "\n";
        ofs << "  Selected features: " << out.selected.size()
            << " / " << corr_.nfeatures << "\n";
        ofs << "  Removed features:  " << out.removed.size() << "\n";
        if (addback_used) {
            ofs << "  (add-back-path was applied to preserve DAG integrity)\n";
        }
        ofs << "\n";
    }

    finish_report(ofs, filename);
}

void DiagnosticReport::write_correlation_matrix(const std::string &filename) {
    auto ofs = open_report(filename);
    ofs << "# Pearson correlation matrix (" << corr_.nfeatures
        << "x" << corr_.nfeatures << ")\n";
    ofs << "# Each row: feature_i  feature_j  pearson(i,j)\n";
    ofs << "# Only upper triangle (j > i) + diagonal\n";
    for (int i = 0; i < corr_.nfeatures; ++i) {
        ofs << i << " " << i << " " << corr_.pearson(i, i) << "\n";
        for (int j = i + 1; j < corr_.nfeatures; ++j) {
            ofs << i << " " << j << " " << corr_.pearson(i, j) << "\n";
        }
    }
    finish_report(ofs, filename);
}

void DiagnosticReport::write_vif(const std::string &filename) {
    auto ofs = open_report(filename);
    ofs << "# VIF per scalar moment\n";
    ofs << "# feature_index  VIF\n";
    for (int i = 0; i < corr_.vif.size(); ++i) {
        double v = corr_.vif[i];
        ofs << i << " " << (std::isfinite(v) ? v : -1.0) << "\n";
    }
    finish_report(ofs, filename);
}

void DiagnosticReport::write_eigenvalues(const std::string &filename) {
    auto ofs = open_report(filename);
    ofs << "# Eigenvalue spectrum of correlation matrix\n";
    ofs << "# index  eigenvalue  cumulative_fraction\n";
    double total = corr_.eigenvalues.sum();
    double cum = 0.0;
    for (int i = 0; i < corr_.eigenvalues.size(); ++i) {
        cum += corr_.eigenvalues[i];
        ofs << i << " " << corr_.eigenvalues[i] << " "
            << (total > 0 ? cum / total : 0.0) << "\n";
    }
    finish_report(ofs, filename);
}

void DiagnosticReport::write_pairs(const std::string &filename) {
    auto ofs = open_report(filename);
    ofs << "# Highly correlated feature pairs\n";
    ofs << "# feature_i  feature_j  |pearson|\n";
    // Report pairs with |r| > 0.8 by default
    auto pairs = corr_.highly_correlated_pairs(0.80);
    for (const auto &[i, j] : pairs) {
        ofs << i << " " << j << " " << std::abs(corr_.pearson(i, j)) << "\n";
    }
    finish_report(ofs, filename);
}

void DiagnosticReport::write_selection(const std::string &filename) {
    auto ofs = open_report(filename);
    for (const auto &out : outcomes_) {
        ofs << "# Strategy: " << out.name << "\n";
        ofs << "# Selected (" << out.selected.size() << "):";
        for (int id : out.selected) ofs << " " << id;
        ofs << "\n";
        ofs << "# Removed (" << out.removed.size() << "):";
        for (int id : out.removed) ofs << " " << id;
        ofs << "\n\n";
    }
    finish_report(ofs, filename);
}

} // namespace BasisAnalysis
