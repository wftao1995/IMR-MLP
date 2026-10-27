// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_descriptor_linear.h"
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <filesystem>

using namespace Potentials;

// ═════════════════════════════════════════════════════════════════════════════
// Constructor
// ═════════════════════════════════════════════════════════════════════════════

PotentialDescriptorLinear::PotentialDescriptorLinear(GPSettings settings, std::unique_ptr<Descriptors> descriptor)
{
    // Extract descriptor sub-settings (key = "descriptor:" prefix)
    auto desc_settings = settings.ExtractSubSettings("descriptor");

    const std::string configured_cutoff = !settings["cutoff"].empty()
        ? settings["cutoff"] : desc_settings["cutoff"];
    // These descriptors have no built-in radius default. Preserve the
    // legacy cutoff as their fallback when no type-specific radius is given.
    const std::string dtype = desc_settings["dtype"];
    if (dtype == "mtm" && !desc_settings["mtm:potfile"].empty())
        desc_settings.Modify("mtm:potfile", std::filesystem::absolute(
            desc_settings["mtm:potfile"]).lexically_normal().string());
    if ((dtype == "behler" || dtype == "bondorder") && desc_settings[dtype + ":cutoff"].empty())
        desc_settings.Modify(dtype + ":cutoff", configured_cutoff.empty() ? "5.0" : configured_cutoff);

    // Create the descriptor wrapper.
    if (descriptor) {
        desc_wrapper_ = std::make_unique<descriptor_wrapper>();
        desc_wrapper_->InitSettings();desc_wrapper_->ApplySettings(desc_settings);
        desc_wrapper_->desc = std::move(descriptor);
    } else desc_wrapper_ = std::make_unique<descriptor_wrapper>(desc_settings);
    n_features_   = desc_wrapper_->feature_size();
    if (n_features_ <= 0) GPERROR("Linear descriptor potential requires at least one feature");
    // Store effective defaults as well as explicit settings so a saved model
    // does not silently change if descriptor defaults change in a later build.
    descriptor_settings_ = desc_wrapper_->CurrentSettings();
    if (dtype == "mtm") descriptor_settings_.Modify("mtm:potfile", desc_settings["mtm:potfile"]);
    if (auto *configured = dynamic_cast<ConnectSettings *>(desc_wrapper_->desc.get())) {
        for (const auto &[key, value] : configured->CurrentSettings().settings)
            descriptor_settings_.Modify(dtype + ":" + key, value);
    }
    cutoff = desc_wrapper_->desc->cutoff_;
    if (!std::isfinite(cutoff) || cutoff <= 0)
        GPERROR("Linear descriptor potential requires a finite positive descriptor cutoff");
    if (!configured_cutoff.empty()) {
        const double requested = std::stod(configured_cutoff);
        if (!std::isfinite(requested) || requested <= 0 ||
            std::abs(requested - cutoff) > 1e-12 * cutoff)
            GPERROR("Linear descriptor potential cutoff must match the descriptor neighborhood radius (including SOAP padding)");
    }
    cutoffsq = cutoff * cutoff;

    // Initialize coefficients to zero.
    coeffs_.setZero(n_features_);
    if (!settings["species_count"].empty()) {
        species_count = Parameters::parse_value<int>(settings["species_count"]);
        if (species_count < 0) GPERROR("Linear descriptor species_count must be nonnegative");
    }
    if (settings.count("descriptor_linear_version")) {
        if (Parameters::parse_value<int>(settings["feature_count"]) != n_features_)
            GPERROR("Linear descriptor feature count does not match saved coefficients");
        std::istringstream values(settings["coefficients"]);
        for (int i = 0; i < n_features_; ++i) {
            if (!(values >> coeffs_[i]) || !std::isfinite(coeffs_[i]))
                GPERROR("Invalid or missing linear descriptor coefficient");
        }
        std::string extra;
        if (values >> extra) GPERROR("Extra linear descriptor coefficient data");
    }

    // Pre-allocate compute buffers.
    desc_buf_.resize(n_features_);

    nl_inited = true;
}

GPSettings PotentialDescriptorLinear::LoadSettings(const std::string &filename)
{
    GPSettings settings(filename);
    if (settings["pottype"] != "descriptor_linear" ||
        settings["descriptor_linear_version"] != "1")
        GPERROR("Invalid or unsupported descriptor_linear model format");
    for (const auto &[key, value] : settings.settings) {
        if (settings.count(key) != 1)
            GPERROR("Duplicate linear descriptor model setting: " + key);
    }
    for (const auto &key : {"descriptor:dtype", "cutoff", "species_count", "feature_count", "coefficients"}) {
        if (settings[key].empty()) GPERROR("Missing linear descriptor model setting: " + std::string(key));
    }
    return settings;
}

PotentialDescriptorLinear::PotentialDescriptorLinear(const std::string &filename)
    : PotentialDescriptorLinear(LoadSettings(filename)) {}

void PotentialDescriptorLinear::Save(std::ostream &os)
{
    if (!coeffs_.allFinite()) GPERROR("Cannot save nonfinite linear descriptor coefficients");
    os << std::setprecision(std::numeric_limits<double>::max_digits10)
       << "pottype descriptor_linear\n"
       << "descriptor_linear_version 1\n"
       << "cutoff " << cutoff << '\n'
       << "species_count " << species_count << '\n'
       << "feature_count " << n_features_ << '\n';
    for (const auto &[key, value] : descriptor_settings_.settings)
        os << "descriptor:" << key << ' ' << value << '\n';
    os << "coefficients";
    for (double coefficient : coeffs_) os << ' ' << coefficient;
    os << '\n';
    if (!os) GPERROR("Failed to write linear descriptor model");
}

// ═════════════════════════════════════════════════════════════════════════════
// Energy / Force evaluation
// ═════════════════════════════════════════════════════════════════════════════

void PotentialDescriptorLinear::getE(Atoms &atom, AnyNeighborhoods &nbhs)
{
    if (!add_efs) atom.resetE();

    for (int i = 0; i < nbhs.size(); ++i) {
        desc_wrapper_->compute(nbhs[i], desc_buf_);
        const double site_energy = coeffs_.dot(desc_buf_);
        atom.energy_ += site_energy;
        atom.site_energies_[i] += site_energy;
    }
    atom.have_energy = true;
}

void PotentialDescriptorLinear::getEOMP(Atoms &atom, AnyNeighborhoods &nbhs)
{
    getE(atom, nbhs);
}

void PotentialDescriptorLinear::getEFS(Atoms &atom, AnyNeighborhoods &nbhs)
{
    if (!add_efs) atom.resetEFS();

    const int natoms = nbhs.size();
    bool need_forces = atom.has_forces();

    for (int i = 0; i < natoms; ++i) {
        Neighborhood &nbh = nbhs[i];

        // Evaluate descriptor (with gradient if forces are needed).
        DescriptorGradient *grad_ptr = need_forces ? &grad_buf_ : nullptr;

        if (grad_ptr) {
            ComputeOptions opts;
            opts.compute_derivative = true;
            opts.gradient = grad_ptr;
            desc_wrapper_->compute(nbh, desc_buf_, opts);
        } else {
            desc_wrapper_->compute(nbh, desc_buf_);
        }

        // Site energy = w^T * desc
        double site_energy = coeffs_.dot(desc_buf_);
        atom.energy_ += site_energy;
        atom.site_energies_[i] += site_energy;

        // Force contribution (if needed and gradient available)
        if (need_forces && grad_ptr) {
            auto &grad = grad_buf_.grad;
            const int nbhcount = nbh.nbhcount_;

            // For each descriptor feature f: force += -w_f * d(desc_f)/d(pos)
            // F_j (on atom j) = -sum_f w_f * grad(f, j, :)
            // Center atom gets +sum_j F_j (Newton's 3rd law)
            for (int j = 0; j < nbhcount; ++j) {
                int global_j = nbh.indices_[j];
                Eigen::Vector3d fcont;
                fcont.setZero();
                for (int f = 0; f < n_features_; ++f) {
                    fcont(0) += coeffs_[f] * grad(f, j, 0);
                    fcont(1) += coeffs_[f] * grad(f, j, 1);
                    fcont(2) += coeffs_[f] * grad(f, j, 2);
                }
                // fcont = sum_f w_f * d(desc_f)/d(pos_j)
                // force_on_j = -fcont
                atom.forces_[i] += fcont;       // center: +sum_j dE_j/dx_i
                atom.forces_[global_j] -= fcont; // Newton: F_ji = -F_ij

                // Virial stress: S_ab -= F_j,a * pos_j,b
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        atom.stress_(a, b) -= fcont(a) * nbh.pos_[j](b);
            }
        }
    }

    atom.setEFSflag();
}

// ═════════════════════════════════════════════════════════════════════════════
// Training interface
// ═════════════════════════════════════════════════════════════════════════════

void PotentialDescriptorLinear::nl_get_coeffs(double *x) const
{
    std::memcpy(x, coeffs_.data(), n_features_ * sizeof(double));
}

void PotentialDescriptorLinear::nl_set_coeffs(const double *x)
{
    std::memcpy(coeffs_.data(), x, n_features_ * sizeof(double));
}

void PotentialDescriptorLinear::random_init()
{
    coeffs_.setRandom();
    // Scale to a reasonable range for potentials (~0.1 eV).
    coeffs_ *= 0.1;
}

// ═════════════════════════════════════════════════════════════════════════════
// Per-atom parameter gradient accumulation
// ═════════════════════════════════════════════════════════════════════════════

void PotentialDescriptorLinear::AccumulateFPGard(
    Neighborhood &nbh,
    double *out_gard_accumulator,
    double se_weight,
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight)
{
    if (f_weight != nullptr) {
        // Need both descriptor values AND position gradients for force training.
        ComputeOptions opts;
        opts.compute_derivative = true;
        opts.gradient = &grad_buf_;
        desc_wrapper_->compute(nbh, desc_buf_, opts);

        // Energy contribution: dE_i/dw_f = desc_f
        for (int f = 0; f < n_features_; ++f)
            out_gard_accumulator[f] += se_weight * desc_buf_[f];

        // f_weight already combines center/neighbor force weights and the
        // negative stress contraction, so it multiplies d(desc)/d(pos) directly.
        auto &grad = grad_buf_.grad;
        for (int f = 0; f < n_features_; ++f) {
            double fcontrib = 0.0;
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                fcontrib += (*f_weight)(j, 0) * grad(f, j, 0)
                          + (*f_weight)(j, 1) * grad(f, j, 1)
                          + (*f_weight)(j, 2) * grad(f, j, 2);
            }
            out_gard_accumulator[f] += fcontrib;
        }
    } else {
        // Energy-only: just descriptor values, no gradient needed.
        desc_wrapper_->compute(nbh, desc_buf_);
        for (int f = 0; f < n_features_; ++f)
            out_gard_accumulator[f] += se_weight * desc_buf_[f];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Structure-level gradient accumulation (energy only)
// ═════════════════════════════════════════════════════════════════════════════

void PotentialDescriptorLinear::AccumulateEFSCombinationGard(
    Atoms &atom, AnyNeighborhoods &nbhs,
    double dLdE,
    double *out_gard_accumulator)
{
    const int natoms = nbhs.size();
    for (int i = 0; i < natoms; ++i) {
        desc_wrapper_->compute(nbhs[i], desc_buf_);
        for (int f = 0; f < n_features_; ++f)
            out_gard_accumulator[f] += dLdE * desc_buf_[f];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Structure-level gradient accumulation (energy + forces + stress)
// ═════════════════════════════════════════════════════════════════════════════

void PotentialDescriptorLinear::AccumulateEFSCombinationGard(
    Atoms &atom, AnyNeighborhoods &nbhs,
    double dLdE,
    Eigen::Matrix<double, -1, 3> &dLdF,
    Eigen::Matrix3d &dLdS,
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &wgt_buf,
    double *out_gard_accumulator)
{
    const int natoms = nbhs.size();
    for (int i = 0; i < natoms; ++i) {
        Neighborhood &nbh = nbhs[i];
        const int nbhcount = nbh.nbhcount_;

        // Build force weights: w_j = dLdF_i - dLdF_j - dLdS * pos_j
        // (following the same pattern as PotMTP).
        tmp_se_ders_wgts_.resize(nbhcount, 3);
        for (int j = 0; j < nbhcount; ++j) {
            for (int a = 0; a < 3; ++a) {
                tmp_se_ders_wgts_(j, a) = dLdF(i, a) - dLdF(nbh.indices_[j], a);
            }
        }
        for (int j = 0; j < nbhcount; ++j) {
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    tmp_se_ders_wgts_(j, a) -= dLdS(a, b) * nbh.pos_[j][b];
        }

        // Delegate to the per-atom accumulator.
        AccumulateFPGard(nbh, out_gard_accumulator, dLdE, &tmp_se_ders_wgts_);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Linear fitting interface
// ═════════════════════════════════════════════════════════════════════════════

void PotentialDescriptorLinear::linear_compute_descriptor(
    Neighborhood &nbh, double *desc_out) const
{
    Eigen::VectorXd buf;
    desc_wrapper_->compute(nbh, buf);
    std::memcpy(desc_out, buf.data(), n_features_ * sizeof(double));
}

void PotentialDescriptorLinear::linear_compute_descriptor_grad(
    Neighborhood &nbh, double *desc_out, double *grad_out) const
{
    Eigen::VectorXd buf;
    DescriptorGradient grad_buf;
    ComputeOptions opts;
    opts.compute_derivative = true;
    opts.gradient = &grad_buf;
    desc_wrapper_->compute(nbh, buf, opts);
    std::memcpy(desc_out, buf.data(), n_features_ * sizeof(double));

    auto &grad = grad_buf.grad;
    const int nnb = nbh.nbhcount_;
    for (int f = 0; f < n_features_; ++f)
        for (int j = 0; j < nnb; ++j)
            for (int a = 0; a < 3; ++a)
                grad_out[f * nnb * 3 + j * 3 + a] = grad(f, j, a);
}
