// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "descriptor_Behler.h"
#include "tools/chem.h"
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace {
void cutoff(double r, double rc, double &value, double &derivative) {
    if (r >= rc) { value = derivative = 0.; return; }
    const double angle = M_PI * r / rc;
    value = .5 * (std::cos(angle) + 1.);
    derivative = -.5 * M_PI / rc * std::sin(angle);
}
}

void DescriptorBehler::validate_species(int type) const {
    if (type < 0 || (weighted_ && static_cast<size_t>(type) >= effective_weights_.size()))
        GPERROR("Behler species index is outside the configured weights range");
}
void DescriptorBehler::validate_species(const std::vector<int> &types) const {
    for (int type : types) validate_species(type);
}
void DescriptorBehler::validate_neighborhood(const Neighborhood &nbh) const {
    if (featureparams_.rows() != dispcount_ || dispcount_ <= 0)
        GPERROR("Behler must be initialized before computation");
    if (nbh.nbhcount_ < 0 || nbh.pos_.size() != static_cast<size_t>(nbh.nbhcount_) ||
        nbh.distances_.size() != static_cast<size_t>(nbh.nbhcount_) ||
        (weighted_ && nbh.chemmap_.size() != static_cast<size_t>(nbh.nbhcount_)))
        GPERROR("Invalid Behler neighborhood arrays");
    if (weighted_) {
        validate_species(nbh.centertype);
        validate_species(nbh.chemmap_);
    }
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        const double radius = nbh.pos_[j].norm();
        if (!nbh.pos_[j].allFinite() || !std::isfinite(radius) || radius <= 0. ||
            !std::isfinite(nbh.distances_[j]) || nbh.distances_[j] <= 0.)
            GPERROR("Behler requires finite nonzero neighbor distances");
        if (std::abs(radius - nbh.distances_[j]) > 1e-10 * std::max(1., radius))
            GPERROR("Behler neighbor distances must match relative coordinates");
    }
}

void DescriptorBehler::init() {
    if (dispcount_ <= 0 || dispselect_.size() != static_cast<size_t>(dispcount_))
        GPERROR("Behler dispcount must be positive and match the dispselect length");
    for (int gi : dispselect_)
        if (gi < 1 || gi > 5) GPERROR("Behler dispselect entries must be between 1 and 5");
    // Configured arrays remain empty when they inherit scalars. Resolve only
    // into the immutable effective feature table, preserving reinitialization.
    const std::vector<double>* configured[]={&rcs_,&etas_,&rshifts_,&kppas_,&zetas_,&lambdas_};
    const double defaults[]={configured_cutoff_,eta_,rshift_,kppa_,zeta_,lambda_};
    FeatureParameters effective(dispcount_,6);
    for(int p=0;p<6;++p) {
        const auto& values=*configured[p];
        if(!values.empty() && values.size()!=static_cast<size_t>(dispcount_))
            GPERROR("Behler per-feature parameter array length must equal dispcount");
        for(int f=0;f<dispcount_;++f) {
            const double v=values.empty() ? defaults[p] : values[f];
            if(!std::isfinite(v)) GPERROR("Behler parameters must be finite");
            effective(f,p)=v;
        }
    }
    for(int f=0;f<dispcount_;++f)
        if(effective(f,0)<=0. || effective(f,1)<0. || effective(f,4)<0. || std::abs(effective(f,5))>1.)
            GPERROR("Behler requires rc>0, eta>=0, zeta>=0 and abs(lambda)<=1");
    featureparams_=std::move(effective);
    cutoff_=featureparams_.col(0).maxCoeff();
    if (!weights_.empty() && !elements_.empty() && weights_.size() != elements_.size())
        GPERROR("Behler weights and elements must have the same length");
    std::vector<double> atomic_weights;
    for (const auto &element : elements_) {
        const auto entry = Chemicals::atomic_numbers.find(element);
        if (entry == Chemicals::atomic_numbers.end() || entry->second <= 0)
            GPERROR("Unknown Behler element: " + element);
        atomic_weights.push_back(entry->second);
    }
    effective_weights_ = weights_.empty() ? std::move(atomic_weights) : weights_;
    for (double weight : effective_weights_)
        if (!std::isfinite(weight)) GPERROR("Behler species weights must be finite");
    if (weighted_ && effective_weights_.empty()) GPERROR("Behler weighted=true requires weights or elements");
}
DescriptorBehler::DescriptorBehler(const GPSettings &settings) : DescriptorBehler() {
    ApplySettings(settings);
    if (dispselect_.empty()) {
        if (dispcount_ < 1 || dispcount_ > 5)
            GPERROR("Behler requires 1 <= dispcount <= 5 when dispselect is omitted");
        for (int f = 0; f < dispcount_; ++f) dispselect_.push_back(f + 1);
    }
    init();
}
void DescriptorBehler::set_parameters(int count, int *selected, double *parameters) {
    if (count <= 0 || !selected || !parameters)
        GPERROR("Behler requires a positive dispcount and parameter/selection arrays");
    dispcount_ = count;
    dispselect_.assign(selected, selected + count);
    cutoff_ = configured_cutoff_ = parameters[0]; eta_ = parameters[1]; rshift_ = parameters[2];
    kppa_ = parameters[3]; zeta_ = parameters[4]; lambda_ = parameters[5];
    rcs_.clear(); etas_.clear(); rshifts_.clear(); kppas_.clear(); zetas_.clear(); lambdas_.clear();
    // Preserve the historical explicit init() lifecycle, while rejecting invalid selections immediately.
    for (int gi : dispselect_)
        if (gi < 1 || gi > 5) GPERROR("Behler dispselect entries must be between 1 and 5");
}
void DescriptorBehler::set_d_parameters(int count, int *selected, double **parameters) {
    if (count <= 0 || !selected || !parameters)
        GPERROR("Behler requires a positive dispcount and parameter/selection arrays");
    Eigen::MatrixXd rows(count, 6);
    for (int f = 0; f < count; ++f) {
        if (!parameters[f]) GPERROR("Behler feature parameter row is null");
        for (int p = 0; p < 6; ++p) rows(f, p) = parameters[f][p];
    }
    dispcount_ = count; dispselect_.assign(selected, selected + count);
    set_d_parameters(rows);
}
void DescriptorBehler::set_d_parameters(const Eigen::MatrixXd &rows) {
    if (rows.rows() != dispcount_ || rows.cols() != 6)
        GPERROR("Behler feature parameters must have shape (dispcount, 6)");
    rcs_.resize(dispcount_); etas_.resize(dispcount_); rshifts_.resize(dispcount_);
    kppas_.resize(dispcount_); zetas_.resize(dispcount_); lambdas_.resize(dispcount_);
    for (int f = 0; f < dispcount_; ++f) {
        rcs_[f] = rows(f, 0); etas_[f] = rows(f, 1); rshifts_[f] = rows(f, 2);
        kppas_[f] = rows(f, 3); zetas_[f] = rows(f, 4); lambdas_[f] = rows(f, 5);
    }
    init();
}
DescriptorBehler::DescriptorBehler(Neighborhoods &nbhs, double *parameters, int count, int *selected)
    : DescriptorBehler() {
    set_parameters(count, selected, parameters); init(); compute_descriptors(nbhs);
}

// A feature is evaluated with immutable resolved parameters. Analytic first
// derivatives avoid constructing second-order jets on the ordinary value/Jacobian path.
void DescriptorBehler::compute_feature(Neighborhood &nbh, int f, double &value,
    Eigen::Tensor<double, 3, Eigen::RowMajor> *gradient) {
    const int gi = dispselect_[f];
    const double rc = featureparams_(f, 0), eta = featureparams_(f, 1);
    const double shift = featureparams_(f, 2), kappa = featureparams_(f, 3);
    const double zeta = featureparams_(f, 4), lambda = featureparams_(f, 5);
    const double angular_shift = weighted_ ? shift : 0.;
    const double pre = (weighted_ ? 2. : 1.) * std::pow(2., 1. - zeta);
    value = 0.;
    auto add = [&](int j, const Eigen::RowVector3d &g) {
        if (gradient) for (int a = 0; a < 3; ++a) (*gradient)(f, j, a) += g[a];
    };
    for (int i = 0; i < nbh.nbhcount_; ++i) {
        const double ri = nbh.distances_[i];
        if (ri >= rc) continue;
        const auto &pi = nbh.pos_[i];
        double fi, dfi; cutoff(ri, rc, fi, dfi);
        const double wi = weighted_ ? species_weight(nbh.chemmap_[i]) : 1.;
        if (gi <= 3) {
            double term = fi, derivative = dfi;
            if (gi == 2) {
                const double dr = ri - shift, e = std::exp(-eta * dr * dr);
                term = e * fi; derivative = e * (dfi - 2. * eta * dr * fi);
            } else if (gi == 3) {
                term = std::cos(kappa * ri) * fi;
                derivative = std::cos(kappa * ri) * dfi - kappa * std::sin(kappa * ri) * fi;
            }
            value += wi * term;
            add(i, (wi * derivative / ri) * pi);
            continue;
        }
        for (int j = i + 1; j < nbh.nbhcount_; ++j) {
            const double rj = nbh.distances_[j];
            if (rj >= rc) continue;
            const auto &pj = nbh.pos_[j];
            double fj, dfj; cutoff(rj, rc, fj, dfj);
            const Eigen::RowVector3d delta = pi - pj;
            const double rij = delta.norm();
            double fij = 1., dfij = 0.;
            if (gi == 4) {
                if (rij <= 0. && eta * angular_shift != 0.)
                    GPERROR("Behler shifted G4 has a cusp at coincident neighbor rows");
                if (rij >= rc) continue;
                cutoff(rij, rc, fij, dfij);
            }
            const double cosine = std::clamp(pi.dot(pj) / (ri * rj), -1., 1.);
            const double base = std::max(0., 1. + lambda * cosine);
            if (base == 0. && zeta > 0. && zeta < 1.)
                GPERROR("Behler angular curvature is undefined for this exponent at a collinear zero");
            const double angular = std::pow(base, zeta);
            const double da = zeta == 0. ? 0. : (base == 0. && zeta > 1. ? 0. :
                zeta * lambda * std::pow(base, zeta - 1.));
            const double di = ri - angular_shift, dj = rj - angular_shift;
            const double dij = rij - angular_shift;
            const double gaussian = std::exp(-eta * (di * di + dj * dj + (gi == 4 ? dij * dij : 0.)));
            const double cut = fi * fj * fij;
            const double scale = pre * wi * (weighted_ ? species_weight(nbh.chemmap_[j]) : 1.);
            value += scale * angular * gaussian * cut;
            if (!gradient) continue;
            const Eigen::RowVector3d dc_i = pj / (ri * rj) - cosine * pi / (ri * ri);
            const Eigen::RowVector3d dc_j = pi / (ri * rj) - cosine * pj / (rj * rj);
            Eigen::RowVector3d de_i = di / ri * pi, de_j = dj / rj * pj;
            Eigen::RowVector3d df_i = (dfi / ri * fj * fij) * pi;
            Eigen::RowVector3d df_j = (dfj / rj * fi * fij) * pj;
            if (gi == 4 && rij > 0.) {
                de_i += dij / rij * delta; de_j -= dij / rij * delta;
                df_i += (fi * fj * dfij / rij) * delta;
                df_j -= (fi * fj * dfij / rij) * delta;
            }
            add(i, scale * gaussian * (da * cut * dc_i + angular * (df_i - 2. * eta * cut * de_i)));
            add(j, scale * gaussian * (da * cut * dc_j + angular * (df_j - 2. * eta * cut * de_j)));
        }
    }
}
void DescriptorBehler::compute(Neighborhood &nbh, Eigen::VectorXd &out) {
    compute(nbh, out, ComputeOptions{});
}
void DescriptorBehler::compute(Neighborhood &nbh, Eigen::VectorXd &out, const ComputeOptions &options) {
    validate_neighborhood(nbh);
    Eigen::VectorXd values(dispcount_);
    Eigen::Tensor<double, 3, Eigen::RowMajor> local_gradient;
    const bool derivative = options.compute_derivative && options.gradient;
    if (derivative) { local_gradient.resize(dispcount_, nbh.nbhcount_, 3); local_gradient.setZero(); }
    for (int f = 0; f < dispcount_; ++f) compute_feature(nbh, f, values[f], derivative ? &local_gradient : nullptr);
    if (!values.allFinite()) GPERROR("Non-finite Behler descriptor value");
    if (derivative) for (Eigen::Index k = 0; k < local_gradient.size(); ++k)
        if (!std::isfinite(local_gradient.data()[k])) GPERROR("Non-finite Behler descriptor gradient");
    out = std::move(values);
    if (derivative) options.gradient->grad = std::move(local_gradient);
}
void DescriptorBehler::compute_descriptors(Neighborhoods &nbhs) {
    descriptors_.resize(nbhs.size());
    for (int i = 0; i < nbhs.size(); ++i) {
        Eigen::VectorXd values; compute(nbhs[i], values);
        descriptors_[i].assign(values.data(), values.data() + values.size());
    }
}
std::ostream &operator<<(std::ostream &os, const DescriptorBehler &descriptor) {
    os << "Dispcount: " << descriptor.dispcount_ << "\nCutoff: " << descriptor.cutoff_
       << "\nFeature parameters (rc eta rshift kappa zeta lambda):\n" << descriptor.featureparams_ << '\n';
    for (size_t i = 0; i < descriptor.descriptors_.size(); ++i) {
        os << " config: " << i + 1 << " descriptor:";
        for (double value : descriptor.descriptors_[i]) os << ' ' << value;
        os << '\n';
    }
    return os;
}
