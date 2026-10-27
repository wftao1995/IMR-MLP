// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential.h"
#include "descriptors/descriptor_wrapper.h"

namespace Potentials {

/// \brief Linear descriptor potential: E = sum_f w_f * desc_f per atom.
///
/// Energy per atom:  E_i = w^T * desc(neighborhood_i)
/// Force on atom j:  F_j = -sum_f w_f * d(desc_{center})/d(pos_j)
///                     (summed over all atoms whose neighborhood contains j)
///
/// The descriptor type and its parameters are specified through the
/// \c descriptor sub-setting namespace:
/// \code
///   pot:pottype descriptor_linear
///   pot:descriptor_linear:cutoff     5.0
///   pot:descriptor_linear:descriptor:dtype  behler
///   pot:descriptor_linear:descriptor:behler:cutoff  5.0
///   pot:descriptor_linear:descriptor:behler:dispcount  4
///   pot:descriptor_linear:descriptor:behler:dispselect  1,2,3,4
/// \endcode
/// The neighbor radius is taken from the descriptor, including SOAP padding.
/// An independently configured cutoff must match that radius.
/// MTM uses descriptor:mtm:potfile as a read-only template dependency. Saved
/// models retain its absolute path; the template must remain available on reload.
class PotentialDescriptorLinear : public AnyPotential
{
    friend struct GPMCodecAccess;
public:
    PotentialDescriptorLinear(GPSettings settings, std::unique_ptr<Descriptors> descriptor = {});
    explicit PotentialDescriptorLinear(const std::string &filename);

    using AnyPotential::Save;
    void Save(std::ostream &os) override;

    using AnyPotential::getE;
    using AnyPotential::getEOMP;
    void getE(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void getEOMP(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override;
    bool supports_site_efs() const override { return true; }
    bool supports_site_parameter_gradients() const override { return true; }
    void CalcSiteEFS(Neighborhood &, double &, SiteCoordinateMatrix &) override;
    bool supports_site_hvp() const override;
    bool supports_site_hessian() const override;
    void apply_site_hvp(Neighborhood &, const SiteCoordinateMatrix &, double &,
                        SiteCoordinateMatrix &, SiteCoordinateMatrix &) override;
    void evaluate_site_hessian(Neighborhood &, double &, SiteCoordinateMatrix &, HessianMatrix &) override;
    bool supports_analytical_hessian() const override { return supports_site_hessian(); }
    bool supports_analytical_hessian_vector_product() const override { return supports_site_hvp(); }
    bool supports_analytical_cell_hessian() const override { return supports_site_hessian(); }
    using AnyPotential::applyHessianVectorProduct;
    void applyHessianVectorProduct(Atoms &, AnyNeighborhoods &, const Eigen::VectorXd &, Eigen::VectorXd &) override;
    void getEFSHessian(Atoms &, HessianMatrix &) override;
    void getEFSHessian(Atoms &, AnyNeighborhoods &, HessianMatrix &) override;
    void getEFSHessianOMP(Atoms &, AnyNeighborhoods &, HessianMatrix &) override;
    void getEFSHessianSymm(Atoms &, HessianMatrix &, double = 1e-4) override;
    void getEFSHessianCell(Atoms &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &) override;
    void getEFSHessianCell(Atoms &, AnyNeighborhoods &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &) override;
    void getEFSHessianCellOMP(Atoms &, AnyNeighborhoods &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &) override;
    void getEFSHessianCellSymm(Atoms &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &, double = 1e-4) override;
    void getEFSHessianCellSymmOMP(Atoms &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &, double = 1e-4) override;

    // ── Training interface ────────────────────────────────────────────────
    int get_NL_coeffs_count() const override { return n_features_; }
    std::vector<ConditionalLinearBlock> conditional_linear_blocks(double) const override {
        if (typeid(*this) != typeid(PotentialDescriptorLinear)) return {};
        ConditionalLinearBlock block; block.name = "linear";
        for (int i = 0; i < n_features_; ++i) block.append(i);
        return {std::move(block)};
    }
    void nl_get_coeffs(double *x) const override;
    void nl_set_coeffs(const double *x) override;
    void random_init() override;

    // ── Linear fitting interface ───────────────────────────────────────────
    bool supports_linear_fit() const override { return true; }
    void linear_compute_descriptor(Neighborhood &nbh, double *desc_out) const override;
    void linear_compute_descriptor_grad(Neighborhood &nbh, double *desc_out, double *grad_out) const override;

    // Per-atom parameter gradient: dE_i/dw_f (and optionally dF/dw_f)
    void AccumulateFPGard(Neighborhood &nbh,
                          double *out_gard_accumulator,
                          double se_weight,
                          Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight) override;

    // Structure-level gradient: sum_i dE_i/dw_f (+ forces if dLdF given)
    void AccumulateEFSCombinationGard(Atoms &atom,
                                      AnyNeighborhoods &nbhs,
                                      double dLdE,
                                      double *out_gard_accumulator) override;
    void AccumulateEFSCombinationGard(Atoms &atom,
                                      AnyNeighborhoods &nbhs,
                                      double dLdE,
                                      Eigen::Matrix<double, -1, 3> &dLdF,
                                      Eigen::Matrix3d &dLdS,
                                      Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &wgt_buf,
                                      double *out_gard_accumulator) override;

private:
    void assemble_curvature(Atoms &, AnyNeighborhoods &, HessianMatrix &,
                            Eigen::MatrixXd *, Eigen::MatrixXd *);
    static GPSettings LoadSettings(const std::string &filename);
    GPSettings descriptor_settings_;
    // ── Internal helpers ──────────────────────────────────────────────────
    /// Compute descriptor (and gradient if force_output is non-null).
    /// Returns the descriptor vector. If force_output is requested and the
    /// descriptor supports gradients, fills the per-atom gradient contribution.
    void eval_descriptor(Neighborhood &nbh, Eigen::VectorXd &desc,
                         DescriptorGradient *grad_output);

    // ── Data ──────────────────────────────────────────────────────────────
    std::unique_ptr<descriptor_wrapper> desc_wrapper_;
    Eigen::VectorXd coeffs_;
    int n_features_ = 0;

    // Per-atom compute buffers (reused across calls to avoid reallocation)
    Eigen::VectorXd desc_buf_;
    DescriptorGradient grad_buf_;
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> tmp_se_ders_wgts_;
};

} // namespace Potentials
