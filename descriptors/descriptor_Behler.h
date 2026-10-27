// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_DESCRIPTOR_BEHLER_H
#define IMRMLP_DESCRIPTOR_BEHLER_H

#include "descriptors.h"
#include "utils/gputils.h"

/** ACSF G1--G5 and chemically weighted ACSF.
 * Unweighted angular features retain the historical unordered-pair convention.
 * Weighted angular features use ordered pairs and shifted Gaussian distances.
 * Species are explicit internal type indices into weights (or elements -> Z).
 */
class DescriptorBehler : public Descriptors, public ConnectSettings {
public:
    using FeatureParameters = Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>;
    std::vector<std::vector<double>> descriptors_;
private:
    int dispcount_ = 0;
    std::vector<int> dispselect_;
    double configured_cutoff_ = 0.;
    double eta_ = 0., rshift_ = 0., kppa_ = 0., zeta_ = 1., lambda_ = 0.;
    bool weighted_ = false;
    std::vector<double> rcs_, etas_, rshifts_, kppas_, zetas_, lambdas_;
    std::vector<double> weights_, effective_weights_;
    std::vector<std::string> elements_;
    FeatureParameters featureparams_;

    double species_weight(int type) const { return weighted_ ? effective_weights_[type] : 1.; }
    std::string effective_parameter_setting(int column,const std::vector<double>& configured) const {
        if(featureparams_.rows()!=dispcount_ || dispcount_<=0) return Parameters::format_value(configured);
        std::vector<double> effective(dispcount_);
        for(int f=0;f<dispcount_;++f) effective[f]=featureparams_(f,column);
        return Parameters::format_value(effective);
    }
    void validate_neighborhood(const Neighborhood &) const;
    void compute_feature(Neighborhood &, int, double &,
        Eigen::Tensor<double, 3, Eigen::RowMajor> *);
    void weighted_curvature(Neighborhood &, const Eigen::VectorXd &,
        const DescriptorSiteMatrix *, double &, DescriptorSiteMatrix &,
        DescriptorSiteMatrix *, Eigen::MatrixXd *);
public:
    DescriptorBehler() { cutoff_ = 0.; InitSettings(); }
    DescriptorBehler(Neighborhoods &, double *, int, int *);
    explicit DescriptorBehler(const GPSettings &);
    std::unique_ptr<Descriptors> clone() const override {
        auto copy = std::make_unique<DescriptorBehler>(*this);
        copy->InitSettings();
        return copy;
    }
    void InitSettings() {
        Connect(dispcount_, "dispcount", "number of symmetry functions");
        Connect(dispselect_, "dispselect", "G1--G5 in output order", -1);
        auto cutoff_binding=Connect(configured_cutoff_, "cutoff", "default cutoff radius; effective maximum after initialization");
        cutoff_binding->format=[this] { return Parameters::format_value(cutoff_); };
        const auto convert_cutoff=cutoff_binding->convert;
        cutoff_binding->convert=[this,convert_cutoff](const Parameters::Occurrences& values,Parameters::Source source) {
            auto commit=convert_cutoff(values,source);
            return [this,commit=std::move(commit)] { commit(); cutoff_=configured_cutoff_; };
        };
        Connect(eta_, "eta", "default Gaussian parameter");
        Connect(rshift_, "rshift", "default Gaussian distance shift");
        Connect(kppa_, "kppa", "default G3 wave number");
        Connect(zeta_, "zeta", "default angular exponent");
        Connect(lambda_, "lambda", "default angular sign parameter");
        Connect(weighted_, "weighted", "chemical weights; ordered angular pairs and shifted Gaussians");
        auto rcs_binding=Connect(rcs_, "rcs", "per-feature cutoff radii; empty inherits cutoff", -1);
        rcs_binding->format=[this] { return effective_parameter_setting(0,rcs_); };
        auto etas_binding=Connect(etas_, "etas", "per-feature eta; empty inherits eta", -1);
        etas_binding->format=[this] { return effective_parameter_setting(1,etas_); };
        auto rshifts_binding=Connect(rshifts_, "rshifts", "per-feature Gaussian shifts; empty inherits rshift", -1);
        rshifts_binding->format=[this] { return effective_parameter_setting(2,rshifts_); };
        auto kppas_binding=Connect(kppas_, "kppas", "per-feature G3 wave numbers; empty inherits kppa", -1);
        kppas_binding->format=[this] { return effective_parameter_setting(3,kppas_); };
        auto zetas_binding=Connect(zetas_, "zetas", "per-feature angular exponents; empty inherits zeta", -1);
        zetas_binding->format=[this] { return effective_parameter_setting(4,zetas_); };
        auto lambdas_binding=Connect(lambdas_, "lambdas", "per-feature angular parameters; empty inherits lambda", -1);
        lambdas_binding->format=[this] { return effective_parameter_setting(5,lambdas_); };
        auto weights_binding=Connect(weights_, "weights", "explicit internal-type chemical weights", -1);
        weights_binding->format=[this] { return Parameters::format_value(effective_weights_); };
        Connect(elements_, "elements", "internal-type element symbols, providing atomic-number weights", -1);
    }
    bool weighted() const { return weighted_; }
    int species_count() const { return weighted_ ? static_cast<int>(effective_weights_.size()) : 0; }
    const std::vector<double> &species_weights() const { return effective_weights_; }
    void validate_species(int) const;
    void validate_species(const std::vector<int> &) const;
    const FeatureParameters &feature_parameters() const { return featureparams_; }
    void compute_descriptors(Neighborhoods &);
    void compute(Neighborhood &, Eigen::VectorXd &) override;
    void compute(Neighborhood &, Eigen::VectorXd &, const ComputeOptions &) override;
    int feature_size() const override { return dispcount_; }
    bool supports_weighted_site_hvp() const override { return true; }
    bool supports_weighted_site_hessian() const override { return true; }
    void compute_weighted_site_hvp(Neighborhood &, const Eigen::VectorXd &,
        const DescriptorSiteMatrix &, double &, DescriptorSiteMatrix &, DescriptorSiteMatrix &) override;
    void compute_weighted_site_hessian(Neighborhood &, const Eigen::VectorXd &,
        double &, DescriptorSiteMatrix &, Eigen::MatrixXd &) override;
    // Parameters use F x 6 rows: rc, eta, rshift, kappa, zeta, lambda.
    void set_parameters(int, int *, double *);
    void set_d_parameters(int, int *, double **);
    void set_d_parameters(const Eigen::MatrixXd &);
    void init();
    friend std::ostream &operator<<(std::ostream &, const DescriptorBehler &);
};
#endif
