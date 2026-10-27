// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "descriptors/descriptor_MTM.h"
#include "potentials/potential_mtp.h"
#include <memory>

struct DescriptorMTM::CurvatureWorkspace {
    Potentials::PotMTP::ompbuffers buffer;
    explicit CurvatureWorkspace(const Potentials::PotMTP &pot)
        : buffer(pot.max_alpha_index_basic, pot.alpha_moments_count, pot.rb_size, pot.max_mu) {}
};

DescriptorMTM::~DescriptorMTM() = default;

namespace {
struct WeightedMomentHead {
    const Eigen::VectorXd &weights;
    void propagate(const std::vector<double> &basis, double &value,
        Eigen::VectorXd &gradient, Eigen::MatrixXd &hessian, int &, int &) const {
        value = 0.;
        gradient.setZero(basis.size());
        for (int i = 0; i < weights.size(); ++i) {
            value += weights[i] * basis[i + 1];
            gradient[i + 1] = weights[i];
        }
        hessian.resize(0, 0);
    }
};
}

void DescriptorMTM::compute_weighted_site_hvp(Neighborhood &nbh, const Eigen::VectorXd &weights,
    const DescriptorSiteMatrix &direction, double &value, DescriptorSiteMatrix &gradient,
    DescriptorSiteMatrix &product) {
    if (weights.size() != ncols_ || !weights.allFinite() ||
        direction.rows() != nbh.nbhcount_ || !direction.allFinite())
        GPERROR("Invalid MTM weighted site HVP weights or direction");
    if (!curvature_) curvature_ = std::make_unique<CurvatureWorkspace>(*pot_);
    auto &buffer = curvature_->buffer;
    WeightedMomentHead model{weights};
    pot_->CalcSiteHessImpl<true, WeightedMomentHead, 1>(nbh, buffer, buffer.buff_site_energy,
        buffer.buff_site_energy_ders, buffer.buff_site_hessian, model, 0, &direction,
        &buffer.buff_site_hvp);
    value = buffer.buff_site_energy;
    gradient = buffer.buff_site_energy_ders;
    product = buffer.buff_site_hvp;
}

void DescriptorMTM::compute_weighted_site_hessian(Neighborhood &nbh, const Eigen::VectorXd &weights,
    double &value, DescriptorSiteMatrix &gradient, Eigen::MatrixXd &hessian) {
    if (weights.size() != ncols_ || !weights.allFinite())
        GPERROR("Invalid MTM weighted site Hessian weights");
    if (!curvature_) curvature_ = std::make_unique<CurvatureWorkspace>(*pot_);
    auto &buffer = curvature_->buffer;
    WeightedMomentHead model{weights};
    pot_->CalcSiteHessImpl<true, WeightedMomentHead, 1>(nbh, buffer, buffer.buff_site_energy,
        buffer.buff_site_energy_ders, buffer.buff_site_hessian, model, 0);
    Eigen::MatrixXd local(3 * nbh.nbhcount_, 3 * nbh.nbhcount_);
    for (int i = 0; i < nbh.nbhcount_; ++i)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            local.block<3, 3>(3 * i, 3 * j) = buffer.buff_site_hessian(i, j);
    value = buffer.buff_site_energy;
    gradient = buffer.buff_site_energy_ders;
    hessian = std::move(local);
}

DescriptorMTM::DescriptorMTM(Potentials::PotMTP &pot)
    : pot_(&pot),
      ncols_(pot.get_alpha_scalar_moments()),
      alpha_moments_count_(pot.alpha_moments_count),
      alpha_index_basic_count_(pot.alpha_index_basic_count),
      alpha_index_times_count_(pot.alpha_index_times_count),
      rb_size_(pot.rb_size),
      max_mu_(pot.max_mu),
      max_alpha_index_basic_(pot.max_alpha_index_basic)
{
    cutoff_ = pot.get_cutoff();

    // Allocate per-instance buffers (avoid racing on PotMTP globals)
    moment_vals_.resize(alpha_moments_count_);
    rb_vals_.resize(rb_size_);
    rb_ders_.resize(rb_size_);
    val_mu_.resize(max_mu_);
    der_mu_.resize(max_mu_);
    dist_powers_.resize(max_alpha_index_basic_);
    coords_powers_.resize(max_alpha_index_basic_ + 1, 3);
    coords_powers_(0, 0) = coords_powers_(0, 1) = coords_powers_(0, 2) = 0;
    coords_powers_(1, 0) = coords_powers_(1, 1) = coords_powers_(1, 2) = 1;
}

void DescriptorMTM::compute(Neighborhood &nbh, Eigen::VectorXd &out)
{
    out.resize(ncols_);
    std::ranges::fill(moment_vals_, 0.0);
    std::ranges::fill(rb_vals_, 0.0);
    std::ranges::fill(val_mu_, 0.0);

    const int itype = nbh.centertype;
    const int species_count = pot_->species_count;
    const int radial_func_count = pot_->radial_func_count;
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double rij = nbh.distances_[jj];
        const int jtype = nbh.chemmap_[jj];
        if (rij > pot_->cutoffs(itype, jtype))
            continue;

        if (pot_->radial_basis_split) {
            pot_->p_radical_basis_pair[itype * species_count + jtype]->
                calc_val(rij, rb_vals_);
        } else {
            pot_->p_radical_basis->calc_val(rij, rb_vals_);
        }

        const double rijinv = 1.0 / rij;
        dist_powers_[0] = 1.0;
        for (int k = 1; k < max_alpha_index_basic_; ++k) {
            dist_powers_[k] = dist_powers_[k - 1] * rijinv;
            for (int a = 0; a < 3; ++a)
                coords_powers_(k + 1, a) =
                    coords_powers_(k, a) * nbh.pos_[jj][a];
        }

        for (int mu = 0; mu < max_mu_; ++mu) {
            val_mu_[mu] = 0.0;
            const int offset =
                (itype * species_count + jtype) * radial_func_count * rb_size_
                + mu * rb_size_;
            for (int xi = 0; xi < rb_size_; ++xi)
                val_mu_[mu] += pot_->radial_coeffs[offset + xi] * rb_vals_[xi];
        }

        for (int i = 0; i < alpha_index_basic_count_; ++i) {
            const int mu = pot_->alpha_index_basic(i, 0);
            const int k = pot_->nu_arr_list[i];
            const double radial = val_mu_[mu] * dist_powers_[k];
            const double px =
                coords_powers_(pot_->alpha_index_basic(i, 1) + 1, 0);
            const double py =
                coords_powers_(pot_->alpha_index_basic(i, 2) + 1, 1);
            const double pz =
                coords_powers_(pot_->alpha_index_basic(i, 3) + 1, 2);
            moment_vals_[i] += radial * px * py * pz;
        }
    }

    for (int i = 0; i < alpha_index_times_count_; ++i) {
        const int lhs = pot_->alpha_index_times(i, 0);
        const int rhs = pot_->alpha_index_times(i, 1);
        const int factor = pot_->alpha_index_times(i, 2);
        const int result = pot_->alpha_index_times(i, 3);
        moment_vals_[result] +=
            factor * moment_vals_[lhs] * moment_vals_[rhs];
    }

    for (int i = 0; i < ncols_; ++i)
        out[i] = moment_vals_[pot_->alpha_moment_mapping[i]];
}

void DescriptorMTM::compute(Neighborhood &nbh, Eigen::VectorXd &out,
                             const ComputeOptions &opts)
{
    // Always compute values first.
    compute(nbh, out);

    // Compute position gradient if requested.
    if (opts.compute_derivative && opts.gradient != nullptr) {
        auto &grad = opts.gradient->grad;
        grad.resize(ncols_, nbh.nbhcount_, 3);
        grad.setZero();
        compute_moment_jacobian(nbh, grad);
    }
}

void DescriptorMTM::compute_moment_jacobian(
    Neighborhood &nbh,
    Eigen::Tensor<double, 3, Eigen::RowMajor> &jac)
{
    const int nbhcount = nbh.nbhcount_;
    const int itype = nbh.centertype;
    const int species_count = pot_->species_count;
    const int radial_func_count = pot_->radial_func_count;
    const auto &cutoffs = pot_->cutoffs;
    const auto &alpha_index_basic = pot_->alpha_index_basic;
    const auto &alpha_index_times = pot_->alpha_index_times;
    const auto &alpha_moment_mapping = pot_->alpha_moment_mapping;
    const auto &nu_arr_list = pot_->nu_arr_list;
    const double *radial_coeffs = pot_->radial_coeffs;

    // Full moment Jacobian sized for all moments (basic + non-basic).
    // Intermediate storage: we later extract the scalar-moment rows.
    Eigen::Tensor<double, 3, Eigen::RowMajor> moment_jac(
        alpha_moments_count_, nbhcount, 3);
    moment_jac.setZero();

    std::ranges::fill(rb_ders_, 0);
    std::ranges::fill(der_mu_, 0);

    for (int jj = 0; jj < nbhcount; ++jj) {
        const double rij = nbh.distances_[jj];
        const int jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;

        // Evaluate radial basis values AND derivatives.
        if (pot_->radial_basis_split)
            pot_->p_radical_basis_pair[itype * species_count + jtype]->calc_der(rij, rb_vals_, rb_ders_);
        else
            pot_->p_radical_basis->calc_der(rij, rb_vals_, rb_ders_);

        // Build distance powers r^{-k} and coordinate powers x^k / y^k / z^k.
        dist_powers_[0] = 1;
        for (int k = 1; k < max_alpha_index_basic_; ++k) {
            dist_powers_[k] = dist_powers_[k - 1] * rijinv;
            for (int a = 0; a < 3; a++)
                coords_powers_(k + 1, a) = coords_powers_(k, a) * nbh.pos_[jj][a];
        }

        // val_mu[mu] and der_mu[mu] (radial part and its radial derivative).
        for (int mu = 0; mu < max_mu_; ++mu) {
            val_mu_[mu] = 0;
            der_mu_[mu] = 0;
            int idxoffset = (itype * species_count + jtype) * radial_func_count * rb_size_ + mu * rb_size_;
            for (int xi = 0; xi < rb_size_; ++xi) {
                val_mu_[mu] += radial_coeffs[idxoffset + xi] * rb_vals_[xi];
                der_mu_[mu] += radial_coeffs[idxoffset + xi] * rb_ders_[xi];
            }
        }

        // Basic moment Jacobian: d(b_i) / d(pos_{j,a})
        //   b_i = val_mu * r^{-nu} * x^{beta1} * y^{beta2} * z^{beta3}
        //
        // Two contributions per coordinate:
        //   1. Radial: d(val)/dr * (x_a / r) * angular_part
        //   2. Angular: val * d(angular_part) / dx_a
        for (int i = 0; i < alpha_index_basic_count_; ++i) {
            int mu = alpha_index_basic(i, 0);
            double val = val_mu_[mu];
            double der = der_mu_[mu];
            int k = nu_arr_list[i];
            double &powk = dist_powers_[k];

            val *= powk;                                      // val_mu * r^{-nu}
            der = der * powk - k * val / rij;                 // d(val * r^{-nu}) / dr

            double pow0 = coords_powers_(alpha_index_basic(i, 1) + 1, 0); // x^{beta1}
            double pow1 = coords_powers_(alpha_index_basic(i, 2) + 1, 1); // y^{beta2}
            double pow2 = coords_powers_(alpha_index_basic(i, 3) + 1, 2); // z^{beta3}
            double mult0 = pow0 * pow1 * pow2;                // angular part

            // ── Radial contribution: (der / r) * mult0 * pos[a] ──
            mult0 *= der / rij;
            moment_jac(i, jj, 0) += mult0 * nbh.pos_[jj][0];
            moment_jac(i, jj, 1) += mult0 * nbh.pos_[jj][1];
            moment_jac(i, jj, 2) += mult0 * nbh.pos_[jj][2];

            // ── Angular contributions ──
            // d/dx: beta1 * val * x^{beta1-1} * y^{beta2} * z^{beta3}
            moment_jac(i, jj, 0) +=
                alpha_index_basic(i, 1) * val *
                coords_powers_(alpha_index_basic(i, 1), 0) *
                pow1 * pow2;
            // d/dy: beta2 * val * x^{beta1} * y^{beta2-1} * z^{beta3}
            moment_jac(i, jj, 1) +=
                alpha_index_basic(i, 2) * val *
                coords_powers_(alpha_index_basic(i, 2), 1) *
                pow0 * pow2;
            // d/dz: beta3 * val * x^{beta1} * y^{beta2} * z^{beta3-1}
            moment_jac(i, jj, 2) +=
                alpha_index_basic(i, 3) * val *
                coords_powers_(alpha_index_basic(i, 3), 2) *
                pow0 * pow1;
        }
    }

    // Non-basic moments via product rule:
    //   b_k = c * b_i * b_j   =>   db_k/dr = c * (db_i/dr * b_j + b_i * db_j/dr)
    for (int i = 0; i < alpha_index_times_count_; ++i) {
        int idx0 = alpha_index_times(i, 0);
        int idx1 = alpha_index_times(i, 1);
        double val2 = alpha_index_times(i, 2);
        double v0 = moment_vals_[idx0];
        double v1 = moment_vals_[idx1];
        int idx_out = alpha_index_times(i, 3);
        for (int j = 0; j < nbhcount; ++j) {
            for (int a = 0; a < 3; ++a) {
                moment_jac(idx_out, j, a) +=
                    val2 * (moment_jac(idx0, j, a) * v1 + v0 * moment_jac(idx1, j, a));
            }
        }
    }

    // Extract scalar-moment Jacobians via alpha_moment_mapping.
    for (int i = 0; i < ncols_; ++i) {
        int src = alpha_moment_mapping[i];
        for (int j = 0; j < nbhcount; ++j)
            for (int a = 0; a < 3; ++a)
                jac(i, j, a) = moment_jac(src, j, a);
    }
}

std::unique_ptr<Descriptors> DescriptorMTM::clone() const
{
    return std::make_unique<DescriptorMTM>(*pot_);
}
