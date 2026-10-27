// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_BASIS_ANALYSIS_BASIS_ANALYZER_H
#define IMRMLP_BASIS_ANALYSIS_BASIS_ANALYZER_H

#include "DescriptorCollector.h"
#include "CorrelationAnalyzer.h"
#include "Decorrelator.h"
#include "DiagnosticReport.h"
#include <memory>
#include <set>
#include <vector>

namespace Potentials {
class PotMTP;
}

namespace BasisAnalysis {

/// Top-level orchestrator for MTP basis correlation analysis and
/// de-correlation.
///
/// Usage:
///   BasisAnalyzer analyzer(pot, max_samples);
///   auto [selected, removed] = analyzer.run(atoms, strategy, addback);
///   analyzer.report().write_all("correlation");
class BasisAnalyzer {
public:
    /// @param pot          Loaded PotMTP potential (used for descriptors)
    /// @param max_samples  Maximum atoms to sample for correlation matrix
    BasisAnalyzer(Potentials::PotMTP &pot, size_t max_samples = 50000);

    /// Run the full pipeline: collect descriptors → analyze →
    /// apply de-correlation strategy → optionally preserve DAG.
    ///
    /// @param atoms          Training structures
    /// @param strategy       De-correlation strategy to apply
    /// @param add_back_path  If true, call add_back_path_scalars_to_remove_id
    /// @return (selected_scalar_ids, removed_scalar_ids) — indices in [0, nscalar)
    std::pair<std::set<int>, std::set<int>>
    run(const std::vector<Atoms> &atoms,
        std::unique_ptr<Decorrelator> strategy,
        bool add_back_path = true);

    /// Phase 1 only: collect and analyze, without removing anything.
    /// Populates the internal DiagnosticReport.
    std::unique_ptr<CorrelationResult>
    analyze(const std::vector<Atoms> &atoms);

    /// Phase 2 only: apply a strategy to an already-computed analysis.
    std::pair<std::set<int>, std::set<int>>
    apply_strategy(const CorrelationResult &corr,
                   std::unique_ptr<Decorrelator> strategy,
                   bool add_back_path = true);

    /// Access internal components.
    DescriptorCollector &collector() { return collector_; }
    DiagnosticReport &report() { return report_; }
    const CorrelationResult *last_result() const { return last_result_.get(); }
    Potentials::PotMTP &potential() { return pot_; }

private:
    Potentials::PotMTP &pot_;
    DescriptorCollector collector_;
    DiagnosticReport report_;
    std::unique_ptr<CorrelationResult> last_result_;
};

} // namespace BasisAnalysis

#endif
