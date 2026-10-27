// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_BASIS_ANALYSIS_DESCRIPTOR_COLLECTOR_H
#define IMRMLP_BASIS_ANALYSIS_DESCRIPTOR_COLLECTOR_H

#include <Eigen/Dense>
#include <vector>
#include "base/structures.h"

namespace Potentials {
class PotMTP;
}

namespace BasisAnalysis {

/// Collects raw scalar-moment descriptor vectors from training structures
/// into a (nfeatures x nsamples) matrix. Each column is one atom's
/// descriptor vector B[0..nscalar-1].
///
/// Supports random subsampling when total atoms exceed max_samples,
/// to keep the correlation analysis tractable.
class DescriptorCollector {
public:
    /// @param pot      Reference to a loaded PotMTP potential (used for
    ///                  descriptor computation via compute_descriptors).
    /// @param max_samples  Maximum number of atom sites to sample.
    /// @param seed     Seed for reproducible random subsampling.
    DescriptorCollector(Potentials::PotMTP &pot, size_t max_samples = 50000, unsigned seed = 42);

    /// Collect descriptors from all structures.
    /// Returns a matrix of shape (nscalar x nsampled).
    /// The matrix is stored column-major: each column = one atom.
    Eigen::MatrixXd collect(const std::vector<Atoms> &atoms);

    size_t max_samples() const { return max_samples_; }
    size_t nsites_collected() const { return nsites_; }

private:
    Potentials::PotMTP &pot_;
    size_t max_samples_;
    unsigned seed_;
    size_t nsites_ = 0;

    /// Build a selection mask: which atoms to sample when total > max_samples.
    std::vector<int> build_selection(int total_sites);
};

} // namespace BasisAnalysis

#endif // IMRMLP_BASIS_ANALYSIS_DESCRIPTOR_COLLECTOR_H
