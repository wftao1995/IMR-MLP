// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "descriptors/descriptors.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace Potentials { class PotMTP; }

/// @brief MTM descriptor wrapping the MTP scalar basis.
/// @ingroup descriptors
///
/// Computes per-atom scalar moment values B_alpha using the Moment Tensor
/// Basis (MTM) contraction tree.  Provides the MTP descriptor as a standard
/// Descriptors subclass so it can be used generically (e.g. with
/// DescriptorBuilder for MaxVol active learning).
///
/// The underlying PotMTP provides the basis specification (radial basis,
/// alpha_index_basic/times, moment mapping, radial coefficients, etc.) and
/// must outlive this descriptor.
class DescriptorMTM : public Descriptors
{
public:
    /// Construct from an existing, fully-loaded PotMTP.
    /// @param pot  Must outlive this descriptor.  The descriptor reads
    ///             radial_coeffs, basis parameters, and radial basis from it.
    explicit DescriptorMTM(Potentials::PotMTP &pot);
    ~DescriptorMTM() override;

    /// Compute scalar moments for a single atomic neighborhood.
    /// @param nbh  Neighbor list (distances must be <= cutoff_).
    /// @param out  Output descriptor vector, resized to feature_size().
    void compute(Neighborhood &nbh, Eigen::VectorXd &out) override;

    /// Compute scalar moments with optional position gradients.
    /// When opts.compute_derivative is true, opts.gradient->grad is resized
    /// to (feature_size, n_neighbors, 3) and filled with d(desc)/d(pos).
    void compute(Neighborhood &nbh, Eigen::VectorXd &out,
                 const ComputeOptions &opts) override;

    /// Number of scalar moments = alpha_scalar_moments.
    int feature_size() const override { return ncols_; }
    bool supports_weighted_site_hvp() const override { return true; }
    bool supports_weighted_site_hessian() const override { return true; }
    void compute_weighted_site_hvp(Neighborhood &, const Eigen::VectorXd &,
        const DescriptorSiteMatrix &, double &, DescriptorSiteMatrix &, DescriptorSiteMatrix &) override;
    void compute_weighted_site_hessian(Neighborhood &, const Eigen::VectorXd &,
        double &, DescriptorSiteMatrix &, Eigen::MatrixXd &) override;

    /// Access the underlying MTP potential (read-only configuration).
    Potentials::PotMTP &pot() const { return *pot_; }

    std::unique_ptr<Descriptors> clone() const override;

private:
    struct CurvatureWorkspace;
    std::unique_ptr<CurvatureWorkspace> curvature_;
    /// Compute d(basis_scalar_moment) / d(neighbor_position) for all neighbors.
    /// \p jac is pre-sized to (feature_size, nbhcount, 3) and zeroed.
    void compute_moment_jacobian(
        Neighborhood &nbh,
        Eigen::Tensor<double, 3, Eigen::RowMajor> &jac);

    Potentials::PotMTP *pot_;
    int ncols_;              // alpha_scalar_moments
    int alpha_moments_count_;
    int alpha_index_basic_count_;
    int alpha_index_times_count_;
    int rb_size_;
    int max_mu_;
    int max_alpha_index_basic_;

    // ── Per-instance compute buffers (thread-safe, unlike PotMTP globals) ──
    std::vector<double> moment_vals_;
    std::vector<double> rb_vals_;
    std::vector<double> rb_ders_;
    std::vector<double> val_mu_;
    std::vector<double> der_mu_;
    std::vector<double> dist_powers_;
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> coords_powers_;
};
