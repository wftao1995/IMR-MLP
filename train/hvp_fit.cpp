// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "hvp_fit.h"
#include <cmath>

namespace Training {
void HVPFitSettings::validate() const {
    if (!std::isfinite(weight) || weight < 0 || !std::isfinite(power) ||
        !std::isfinite(step) || step <= 0)
        GPERROR("HVP requires finite hvp_weight >= 0, hvp_pow and hvp_fd_step > 0");
}

double HVPFitSettings::observation_weight(const Atoms &atom, double normalization) const {
    validate();
    atom.validate_hvp();
    if (weight == 0 || !atom.has_hvp()) return 0;
    const double result = weight * atom.fitness * normalization /
        std::pow(atom.size(), power) / atom.hvp_directions_.cols();
    if (!std::isfinite(result) || result < 0) GPERROR("Invalid HVP observation weight");
    return result;
}

namespace {
Atoms geometry_copy(const Atoms &a) {
    Atoms out;
    out.cell_ = a.cell_;
    out.pos_ = a.pos_;
    out.chemmap_ = a.chemmap_;
    out.pbc_ = a.pbc_;
    out.cartesian = a.cartesian;
    out.id = 0; // Scratch geometries do not require a dataset index.
    out.charges_ = a.charges_;
    out.collinear_ = a.collinear_;
    out.magmom_scalar_ = a.magmom_scalar_;
    out.magmoms_ = a.magmoms_;
    out.global_attribs_d_ = a.global_attribs_d_;
    out.global_attribs_i_ = a.global_attribs_i_;
    out.global_attribs_v_ = a.global_attribs_v_;
    out.global_attribs_vi_ = a.global_attribs_vi_;
    out.features_ = a.features_;
    return out;
}
double displace_probe(const Atoms &reference, int k, double step, Atoms &plus, Atoms &minus) {
    const auto direction = reference.hvp_directions_.col(k);
    double max_norm = 0;
    for (int i = 0; i < reference.size(); ++i)
        max_norm = std::max(max_norm, direction.segment<3>(3 * i).stableNorm());
    const double epsilon = step / max_norm;
    if (!std::isfinite(epsilon) || epsilon <= 0) GPERROR("Invalid HVP displacement scale");
    for (int i = 0; i < reference.size(); ++i) {
        const Eigen::RowVector3d delta = epsilon * direction.segment<3>(3 * i).transpose();
        plus.pos_[i] += delta;
        minus.pos_[i] -= delta;
    }
    return epsilon;
}
void check_forces(const Atoms &a) {
    if (a.forces_.size() != size_t(a.size())) GPERROR("HVP prediction did not return all forces");
    for (const auto &f : a.forces_)
        if (!f.allFinite()) GPERROR("Non-finite force in HVP prediction");
}
}

HVPFitResult finite_difference_hvp_loss(Potentials::AnyPotential &pot,
    const Atoms &reference, const HVPFitSettings &settings,
    double normalization, const ForcePullback &pullback) {
    HVPFitResult result;
    const double weight = settings.observation_weight(reference, normalization);
    if (weight == 0) return result;
    if (!reference.cartesian) GPERROR("HVP fitting requires Cartesian positions");
    for (int k = 0; k < reference.hvp_directions_.cols(); ++k) {
        Atoms plus = geometry_copy(reference), minus = geometry_copy(reference);
        const double epsilon = displace_probe(reference, k, settings.step, plus, minus);
        Neighborhoods plus_nbhs(plus, pot.cutoff), minus_nbhs(minus, pot.cutoff);
        minus.resetEFS();
        pot.getEFS(minus, minus_nbhs);
        check_forces(minus);
        plus.resetEFS();
        pot.getEFS(plus, plus_nbhs);
        check_forces(plus);
        Eigen::Matrix<double, -1, 3> residual(reference.size(), 3);
        for (int i = 0; i < reference.size(); ++i)
            residual.row(i) = (-(plus.forces_[i] - minus.forces_[i]) / (2 * epsilon)
                - reference.hvp_targets_.col(k).segment<3>(3 * i)).transpose();
        const double error = residual.squaredNorm();
        if (!std::isfinite(error)) GPERROR("Non-finite HVP residual");
        result.squared_error += error;
        result.loss += weight * error;
        result.observations += 3 * reference.size();
        if (pullback) {
            Eigen::Matrix<double, -1, 3> force_weight = -weight / epsilon * residual;
            pullback(plus, plus_nbhs, force_weight);
            // Refresh caches for the other geometry before its force pullback.
            minus.resetEFS();
            pot.getEFS(minus, minus_nbhs);
            check_forces(minus);
            force_weight = weight / epsilon * residual;
            pullback(minus, minus_nbhs, force_weight);
        }
    }
    return result;
}
Eigen::MatrixXd finite_difference_hvp_observations(const Atoms &reference,
    double step, double cutoff, const ForceObservation &forces) {
    HVPFitSettings{1.,0.,step}.validate();
    reference.validate_hvp();
    if (!reference.cartesian || !reference.has_hvp() || !forces)
        GPERROR("HVP observations require Cartesian probes and a force evaluator");
    const int ndof = 3 * reference.size();
    Eigen::MatrixXd result;
    for (int k = 0; k < reference.hvp_directions_.cols(); ++k) {
        Atoms plus = geometry_copy(reference), minus = geometry_copy(reference);
        const double epsilon = displace_probe(reference,k,step,plus,minus);
        Neighborhoods plus_nbhs(plus,cutoff), minus_nbhs(minus,cutoff);
        const auto fp = forces(plus,plus_nbhs), fm = forces(minus,minus_nbhs);
        if (fp.rows() != ndof || fm.rows() != ndof || fp.cols() <= 0
            || fm.cols() != fp.cols() || !fp.allFinite() || !fm.allFinite())
            GPERROR("HVP force observations require finite matching 3N by columns matrices");
        if (k == 0) result.resize(ndof*reference.hvp_directions_.cols(),fp.cols());
        if (fp.cols() != result.cols()) GPERROR("HVP force observation column count changed");
        result.middleRows(k*ndof,ndof) = -(fp-fm)/(2.*epsilon);
    }
    return result;
}

void finite_difference_hvp_vjp(Potentials::AnyPotential &pot,
    const Atoms &reference, double step, const Eigen::MatrixXd &cotangents,
    const ForcePullback &pullback) {
    HVPFitSettings{1.,0.,step}.validate();
    reference.validate_hvp();
    if (!reference.cartesian || !reference.has_hvp()
        || cotangents.rows() != 3 * reference.size()
        || cotangents.cols() != reference.hvp_directions_.cols()
        || !cotangents.allFinite() || !pullback)
        GPERROR("HVP VJP requires Cartesian probes and finite matching output cotangents");
    for (int k = 0; k < cotangents.cols(); ++k) {
        Atoms plus = geometry_copy(reference), minus = geometry_copy(reference);
        const double epsilon = displace_probe(reference, k, step, plus, minus);
        Neighborhoods plus_nbhs(plus, pot.cutoff), minus_nbhs(minus, pot.cutoff);
        Eigen::Matrix<double,-1,3> force_weights(reference.size(),3);
        for (int i = 0; i < reference.size(); ++i)
            force_weights.row(i) = -cotangents.col(k).segment<3>(3*i).transpose() / (2.*epsilon);
        plus.resetEFS(); pot.getEFS(plus,plus_nbhs); check_forces(plus);
        pullback(plus,plus_nbhs,force_weights);
        force_weights *= -1.;
        minus.resetEFS(); pot.getEFS(minus,minus_nbhs); check_forces(minus);
        pullback(minus,minus_nbhs,force_weights);
    }
}
} // namespace Training
