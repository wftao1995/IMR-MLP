// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential.h"
#include <memory>
#include <cmath>

namespace Potentials {

class PotentialHybridScale : public AnyPotential {
    friend struct GPMCodecAccess;
public:
    void set_element_map(const std::map<int, std::string> &mapping) override {
        for (auto &pot : ppots_) pot->set_element_map(mapping);
    }
    enum class FeatureKind {
        MinDistance,
        SmoothCoordination
    };

    enum class TransitionKind {
        Smoothstep
    };

    struct Boundary {
        double left = 0.0;
        double right = 0.0;
    };

    int npot = 0;
    std::string feature = "smooth_coordination";
    std::string transition = "smoothstep";
    double feature_r0 = 0.0;
    double feature_r1 = -1.0;

    explicit PotentialHybridScale(GPSettings settings, std::vector<std::unique_ptr<AnyPotential>> children = {});
    ~PotentialHybridScale() override = default;

    std::vector<const AnyPotential *> children() const {
        std::vector<const AnyPotential *> result;
        for (const auto &child : ppots_) result.push_back(child.get());
        return result;
    }
    const std::vector<double> &child_cutoffs() const { return pot_cutoffs_; }
    const std::vector<std::string> &child_types() const { return pot_types_; }
    const std::vector<Boundary> &boundaries() const { return boundaries_; }
    FeatureKind feature_kind() const { return feature_kind_; }
    TransitionKind transition_kind() const { return transition_kind_; }
    bool supports_site_efs() const override;
    void CalcSiteEFS(Neighborhood &, double &, SiteCoordinateMatrix &) override;

    void getEFS(Atoms &atom) override;
    void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void getEFSOMP(Atoms &atom) override;
    void getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void getEFSOMPthread(Atoms &atom) override;
    void getEFSOMPthread(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void getE(Atoms &atom) override;
    void getE(Atoms &atom, AnyNeighborhoods &nbhs) override;

    bool supports_site_hvp() const override;
    bool supports_site_hessian() const override;

    void apply_site_hvp(Neighborhood &, const SiteCoordinateMatrix &, double &,
                        SiteCoordinateMatrix &, SiteCoordinateMatrix &) override;
    void evaluate_site_hessian(Neighborhood &, double &, SiteCoordinateMatrix &,
                               HessianMatrix &) override;

    bool supports_analytical_hessian_vector_product() const override { return supports_site_hvp(); }
    bool supports_analytical_hessian() const override { return supports_site_hessian(); }
    bool supports_analytical_cell_hessian() const override { return supports_site_hessian(); }
    bool supports_analytical_hessian_for(const Atoms &atom) const override {
        return supports_site_hessian() && feature_regular_for(atom);
    }
    bool supports_analytical_hessian_vector_product_for(const Atoms &atom) const override {
        return supports_site_hvp() && feature_regular_for(atom);
    }
    bool supports_analytical_cell_hessian_for(const Atoms &atom) const override {
        return atom.cell_.allFinite() && std::abs(atom.cell_.determinant()) > 1e-14 &&
               supports_analytical_hessian_for(atom);
    }
    using AnyPotential::applyHessianVectorProduct;
    void applyHessianVectorProduct(Atoms &, AnyNeighborhoods &, const Eigen::VectorXd &,
                                   Eigen::VectorXd &) override;
    void getEFSHessian(Atoms &, HessianMatrix &) override;
    void getEFSHessian(Atoms &, AnyNeighborhoods &, HessianMatrix &) override;
    void getEFSHessianOMP(Atoms &, AnyNeighborhoods &, HessianMatrix &) override;
    void getEFSHessianSymm(Atoms &, HessianMatrix &, double = 1e-4) override;
    void getEFSHessianCell(Atoms &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &) override;
    void getEFSHessianCell(Atoms &, AnyNeighborhoods &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &) override;
    void getEFSHessianCellOMP(Atoms &, AnyNeighborhoods &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &) override;
    void getEFSHessianCellSymm(Atoms &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &, double = 1e-4) override;
    void getEFSHessianCellSymmOMP(Atoms &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &, double = 1e-4) override;


    int get_NL_coeffs_count() const override;
    void nl_get_coeffs(double *x) const override;
    void nl_set_coeffs(const double *x) override;
    std::vector<NLParameterGroup> nl_parameter_groups() const override;
    std::vector<ConditionalLinearBlock> conditional_linear_blocks(double penalty_weight) const override;
    void conditional_linear_set_coeffs(const std::vector<int>& indices, const Eigen::VectorXd& values) override;
    void random_init() override;
    void init_coeffs() override;
    void train_init() override;
    void set_inited() override;
    void set_empty() override;
    bool inited() override;
    void AddPenaltyGard(double weight, double &loss, double *loss_gard) override;
    void Save(const std::string &path) override;

    void AccumulateEFSCombinationGard(Atoms &atom, AnyNeighborhoods &nbhs,
                                      double dLdE,
                                      Eigen::Matrix<double, -1, 3> &dLdF,
                                      Eigen::Matrix3d &dLdS,
                                      Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &wgt_d2UdRdx,
                                      double *out_gard_accumulator) override;
    void AccumulateEFSCombinationGard(Atoms &atom, AnyNeighborhoods &nbhs,
                                      double dLdE,
                                      double *out_gard_accumulator) override;

private:
    friend bool supports_frame_local_efs(const AnyPotential &pot);
    bool feature_regular_for(const Atoms &) const;
    std::vector<std::unique_ptr<AnyPotential>> ppots_;
    std::vector<double> pot_cutoffs_;
    std::vector<std::string> pot_types_;
    std::vector<int> parameter_offsets_;
    std::vector<Boundary> boundaries_;
    FeatureKind feature_kind_ = FeatureKind::SmoothCoordination;
    TransitionKind transition_kind_ = TransitionKind::Smoothstep;

    void InitSettings();
    static GPSettings load_manifest_settings(const GPSettings &settings);
    void load_potentials(const GPSettings &settings, std::vector<std::unique_ptr<AnyPotential>> children = {});
    void load_boundaries(const GPSettings &settings);
    std::unique_ptr<AnyPotential> make_subpotential(const GPSettings &settings) const;

    static void make_site_neighborhood(const Neighborhood &nbh, double rcut,
                                       Neighborhood &sub_nbh,
                                       std::vector<int> &source_rows);
    void calc_feature(const Neighborhood &nbh, double &q,
                      Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &q_ders) const;
    void calc_weights(double q, std::vector<double> &weights,
                      std::vector<double> &weight_ders) const;
    static void smoothstep(double x0, double x1, double x,
                           double &value, double &derivative);
    static Boundary parse_boundary(const std::string &text, const std::string &key);
    void site_curvature(Neighborhood &, const SiteCoordinateMatrix *, double &,
                        SiteCoordinateMatrix &, SiteCoordinateMatrix *, HessianMatrix *);
    void assemble_curvature(Atoms &, AnyNeighborhoods &, HessianMatrix &,
                            Eigen::MatrixXd *, Eigen::MatrixXd *);
};

} // namespace Potentials
