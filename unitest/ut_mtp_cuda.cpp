// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "runtime/mtp/gpu-direct/gpu_backend.h"

#ifdef IMRMLP_HIP
#define IMRMLP_GPU_TEST_TAG "[hip]"
#else
#define IMRMLP_GPU_TEST_TAG "[cuda]"
#endif

#ifdef IMRMLP_GPU

#include "runtime/mtp/gpu-direct/mtp_gpu_direct.h"
#include "potentials/mtp_gpu_adapter.h"
#include "potentials/potential_mtp_m.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace {

using MLIPCore::GpuDirect::MTPGpuComputeRequest;
using MLIPCore::GpuDirect::MTPGpuBatchRequest;
using MLIPCore::GpuDirect::MTPGpuGradientRequest;
using MLIPCore::GpuDirect::MTPGpuLossRequest;
using MLIPCore::GpuDirect::MTPGpuDirect;
using MLIPCore::GpuDirect::MTPGpuInitializationOptions;
using MLIPCore::GpuDirect::MTPGpuForceKernel;
using MLIPCore::GpuDirect::MTPGpuModel;
using MLIPCore::GpuDirect::MTPGpuPotentialView;
using MLIPCore::GpuDirect::MTPGpuPrecision;
using MLIPCore::GpuDirect::RadialBasis;
using Potentials::MTPGpuBatch;
using Potentials::MTPGpuEvaluator;
using Potentials::PotMTP;

struct SmallSystem {
    static constexpr int n = 4;
    std::array<std::array<double, 3>, n> x{{
        {{0.0, 0.0, 0.0}},
        {{1.1, 0.2, -0.1}},
        {{-0.4, 1.3, 0.3}},
        {{0.2, -0.6, 1.4}},
    }};
    std::array<std::array<double, 3>, n> force{};
    std::array<std::array<double, 6>, n> virial{};
    std::array<double, n> site_energy{};
    std::array<int, n> type{{1, 1, 1, 1}};
    std::array<int, n> centers{{0, 1, 2, 3}};
    std::array<int, n> numneigh{{3, 3, 3, 3}};
    std::array<std::array<int, 3>, n> neighbors{};
    std::array<double *, n> xptr{};
    std::array<double *, n> fptr{};
    std::array<double *, n> vptr{};
    std::array<int *, n> nptr{};

    SmallSystem()
    {
        for (int i = 0; i < n; ++i) {
            int k = 0;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    // Exercise the LAMMPS-style high-bit neighbor mask too.
                    neighbors[i][k++] = j | 0x40000000;
                }
            }
            xptr[i] = x[i].data();
            fptr[i] = force[i].data();
            vptr[i] = virial[i].data();
            nptr[i] = neighbors[i].data();
        }
    }
};

double radial_value(double r, RadialBasis basis,
                    double coeff0 = 0.7, double coeff1 = -0.2)
{
    constexpr double rmax = 5.0;
    const double ksi = (2.0 * r - rmax) / rmax;
    double rb0 = 0.0;
    if (basis == RadialBasis::ChebyCos) {
        rb0 = 0.5 * (1.0 + std::cos(std::acos(-1.0) * r / rmax));
    } else if (basis == RadialBasis::ChebyshevPoly5) {
        const double q = 1.0 - r / rmax;
        rb0 = 6.0 * std::pow(q, 5) - 15.0 * std::pow(q, 4) +
              10.0 * std::pow(q, 3);
    } else {
        const double rm = r - rmax;
        rb0 = rm * rm;
    }
    const double rb1 = ksi * rb0;
    return coeff0 * rb0 + coeff1 * rb1;
}

double reference_energy(const SmallSystem &system, RadialBasis basis,
                        double radial0 = 0.7, double radial1 = -0.2,
                        double species = 0.3, double moment0 = 1.2,
                        double moment1 = -0.4)
{
    double total = 0.0;
    for (int i = 0; i < SmallSystem::n; ++i) {
        double b0 = 0.0;
        double b1 = 0.0;
        for (int j = 0; j < SmallSystem::n; ++j) {
            if (i == j) continue;
            const double rx = system.x[j][0] - system.x[i][0];
            const double ry = system.x[j][1] - system.x[i][1];
            const double rz = system.x[j][2] - system.x[i][2];
            const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
            const double radial = radial_value(r, basis, radial0, radial1);
            b0 += radial;
            b1 += radial * rx / r;
        }
        total += species + moment0 * b0 + moment1 * b1 * b1;
    }
    return total;
}

MTPGpuPotentialView small_potential_view(RadialBasis basis = RadialBasis::Chebyshev)
{
    static const int spmap[] = {-1, 0};
    static const int basic[] = {
        0, 0, 0, 0,
        0, 1, 0, 0,
    };
    static const int times[] = {1, 1, 1, 2};
    static const int mapping[] = {0, 2};
    static const int nu[] = {0, 1};
    static const double radial[] = {0.7, -0.2};
    static const double species[] = {0.3};
    static const double moments[] = {1.2, -0.4};

    MTPGpuPotentialView view;
    view.species_count = 1;
    view.max_type = 2;
    view.spmap = spmap;
    view.alpha_moments_count = 3;
    view.alpha_index_basic_count = 2;
    view.alpha_index_times_count = 1;
    view.alpha_scalar_moments = 2;
    view.radial_func_count = 1;
    view.rb_size = 2;
    view.max_mu = 1;
    view.max_alpha_index_basic = 2;
    view.min_dist = 0.0;
    view.max_dist = 5.0;
    view.cutoff = 5.0;
    view.scaling = 1.0;
    view.radial_basis = basis;
    view.radial_coeffs = radial;
    view.radial_coeffs_count = 2;
    view.species_coeffs = species;
    view.moment_coeffs = moments;
    view.alpha_index_basic = basic;
    view.alpha_index_times = times;
    view.alpha_moment_mapping = mapping;
    view.nu_arr_list = nu;
    return view;
}

MTPGpuPotentialView sparse_slot_potential_view()
{
    static const int times[] = {1, 1, 1, 700};
    static const int mapping[] = {0, 700};
    MTPGpuPotentialView view = small_potential_view();
    view.alpha_moments_count = 1024;
    view.alpha_index_times = times;
    view.alpha_moment_mapping = mapping;
    return view;
}

MTPGpuPotentialView long_propagation_group_potential_view()
{
    constexpr int term_count = 1024;
    static const std::vector<int> times = [] {
        std::vector<int> entries(4 * term_count);
        for (int term = 0; term < term_count; ++term) {
            entries[4 * term + 0] = 1;
            entries[4 * term + 1] = 1;
            entries[4 * term + 2] = 1;
            entries[4 * term + 3] = 2;
        }
        return entries;
    }();
    // Keep the composite contribution on the same scale as the small view.
    static const double moments[] = {1.2, -0.4 / term_count};

    MTPGpuPotentialView view = small_potential_view();
    view.alpha_index_times_count = term_count;
    view.alpha_index_times = times.data();
    view.moment_coeffs = moments;
    return view;
}

MTPGpuPotentialView two_species_potential_view()
{
    static const int spmap[] = {-1, 0, 1};
    static const int basic[] = {
        0, 0, 0, 0,
        0, 1, 0, 0,
    };
    static const int times[] = {1, 1, 1, 2};
    static const int mapping[] = {0, 2};
    static const int nu[] = {0, 1};
    static const double radial[] = {
         0.70, -0.20,  0.45,  0.10,
        -0.35,  0.25,  0.80, -0.15,
    };
    static const double species[] = {0.3, -0.1};
    static const double moments[] = {1.2, -0.4};

    MTPGpuPotentialView view;
    view.species_count = 2;
    view.max_type = 3;
    view.spmap = spmap;
    view.alpha_moments_count = 3;
    view.alpha_index_basic_count = 2;
    view.alpha_index_times_count = 1;
    view.alpha_scalar_moments = 2;
    view.radial_func_count = 1;
    view.rb_size = 2;
    view.max_mu = 1;
    view.max_alpha_index_basic = 2;
    view.min_dist = 0.0;
    view.max_dist = 5.0;
    view.cutoff = 5.0;
    view.scaling = 1.0;
    view.radial_basis = RadialBasis::Chebyshev;
    view.radial_coeffs = radial;
    view.radial_coeffs_count = 8;
    view.species_coeffs = species;
    view.moment_coeffs = moments;
    view.alpha_index_basic = basic;
    view.alpha_index_times = times;
    view.alpha_moment_mapping = mapping;
    view.nu_arr_list = nu;
    return view;
}

std::string split_radial_potential_text()
{
    return
        "MTP\n"
        "version = 1.1.0\n"
        "potential_name = MTP1m\n"
        "scaling = 1\n"
        "species_count = 2\n"
        "potential_tag =\n"
        "radial_basis_split = 1\n"
        "pair 0 0\n"
        "radial_basis_type = RBChebyshev\n"
        "  min_dist = 0.5\n"
        "  max_dist = 4.0\n"
        "  radial_basis_size = 2\n"
        "pair 0 1\n"
        "radial_basis_type = RBChebyCos\n"
        "  min_dist = 0.0\n"
        "  max_dist = 5.0\n"
        "  radial_basis_size = 2\n"
        "pair 1 0\n"
        "radial_basis_type = RBChebyshevPoly5\n"
        "  min_dist = 0.4\n"
        "  max_dist = 4.5\n"
        "  radial_basis_size = 2\n"
        "  rb_b = -0.7\n"
        "pair 1 1\n"
        "radial_basis_type = RBChebyshev\n"
        "  min_dist = 0.6\n"
        "  max_dist = 3.2\n"
        "  radial_basis_size = 2\n"
        "radial_funcs_count = 1\n"
        "radial_coeffs\n"
        "  0-0\n"
        "    {0.70, -0.20}\n"
        "  0-1\n"
        "    {0.45, 0.10}\n"
        "  1-0\n"
        "    {-0.35, 0.25}\n"
        "  1-1\n"
        "    {0.80, -0.15}\n"
        "alpha_moments_count = 5\n"
        "alpha_index_basic_count = 4\n"
        "alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}\n"
        "alpha_index_times_count = 3\n"
        "alpha_index_times = {{1, 1, 1, 4}, {2, 2, 1, 4}, {3, 3, 1, 4}}\n"
        "alpha_scalar_moments = 2\n"
        "alpha_moment_mapping = {0, 4}\n"
        "species_coeffs = {0.3, -0.1}\n"
        "moment_coeffs = {1.2, -0.4}\n";
}

std::string single_radial_potential_text(const std::string &basis)
{
    std::ostringstream output;
    output << "MTP\n"
           << "version = 1.1.0\n"
           << "potential_name = MTP1m\n"
           << "scaling = 1\n"
           << "species_count = 1\n"
           << "potential_tag =\n"
           << "radial_basis_type = " << basis << '\n'
           << "  min_dist = 0.5\n"
           << "  max_dist = 5.0\n"
           << "  radial_basis_size = 4\n";
    if (basis == "RBChebyshevPoly5") output << "  rb_b = -0.7\n";
    output << "radial_funcs_count = 1\n"
           << "radial_coeffs\n"
           << "  0-0\n"
           << "    {0.70, -0.20, 0.12, -0.05}\n"
           << "alpha_moments_count = 5\n"
           << "alpha_index_basic_count = 4\n"
           << "alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}\n"
           << "alpha_index_times_count = 3\n"
           << "alpha_index_times = {{1, 1, 1, 4}, {2, 2, 1, 4}, {3, 3, 1, 4}}\n"
           << "alpha_scalar_moments = 2\n"
           << "alpha_moment_mapping = {0, 4}\n"
           << "species_coeffs = {0.3}\n"
           << "moment_coeffs = {1.2, -0.4}\n";
    return output.str();
}

std::string configurable_radial_potential_text(
    const std::string &cutoff, const std::string &shift,
    double cutoff_b = -0.4, double shift_lambda = 1.2)
{
    std::ostringstream output;
    output << "MTP\n"
           << "version = 1.1.0\n"
           << "potential_name = MTP1m\n"
           << "scaling = 1\n"
           << "species_count = 1\n"
           << "potential_tag =\n"
           << "radial_basis_type = RBChebyshevConfigurable\n"
           << "  min_dist = 0.5\n"
           << "  max_dist = 5.0\n"
           << "  radial_basis_size = 4\n"
           << "  cutoff_type = " << cutoff << '\n';
    if (cutoff == "poly5") output << "  cutoff_b = " << cutoff_b << '\n';
    output << "  shift_type = " << shift << '\n';
    if (shift == "exponential")
        output << "  shift_lambda = " << shift_lambda << '\n';
    output << "radial_funcs_count = 1\n"
           << "radial_coeffs\n"
           << "  0-0\n"
           << "    {0.70, -0.20, 0.12, -0.05}\n"
           << "alpha_moments_count = 5\n"
           << "alpha_index_basic_count = 4\n"
           << "alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}\n"
           << "alpha_index_times_count = 3\n"
           << "alpha_index_times = {{1, 1, 1, 4}, {2, 2, 1, 4}, {3, 3, 1, 4}}\n"
           << "alpha_scalar_moments = 2\n"
           << "alpha_moment_mapping = {0, 4}\n"
           << "species_coeffs = {0.3}\n"
           << "moment_coeffs = {1.2, -0.4}\n";
    return output.str();
}

Atoms split_radial_atoms()
{
    Atoms atoms;
    atoms.cell_ = 20.0 * Eigen::Matrix3d::Identity();
    atoms.pos_ = {{1.0, 1.0, 1.0},
                  {2.1, 1.2, 1.0},
                  {1.3, 2.6, 1.4},
                  {5.6, 1.0, 1.2},
                  {2.35, 1.2, 1.0}};
    atoms.chemmap_ = {0, 1, 0, 1, 0};
    atoms.resetEFS();
    return atoms;
}

Atoms configurable_radial_atoms(double pair_distance = -1.0)
{
    Atoms atoms;
    atoms.cell_ = 20.0 * Eigen::Matrix3d::Identity();
    if (pair_distance >= 0.0) {
        atoms.pos_ = {{1.0, 1.0, 1.0},
                      {1.0 + pair_distance, 1.0, 1.0}};
    } else {
        // Includes one pair below min_dist and several pairs close to the
        // upper interval, while retaining non-collinear force/stress terms.
        atoms.pos_ = {{1.0, 1.0, 1.0},
                      {2.1, 1.2, 1.0},
                      {1.3, 2.6, 1.4},
                      {5.6, 1.0, 1.2},
                      {2.35, 1.2, 1.0}};
    }
    atoms.chemmap_.assign(atoms.pos_.size(), 0);
    atoms.resetEFS();
    return atoms;
}

double atom_efs_objective(const Atoms &atoms, double denergy,
                          const Eigen::Matrix<double, -1, 3> &dforce,
                          const Eigen::Matrix3d &dstress)
{
    double value = denergy * atoms.energy_;
    for (int atom = 0; atom < atoms.size(); ++atom)
        for (int axis = 0; axis < 3; ++axis)
            value += dforce(atom, axis) * atoms.forces_[atom][axis];
    value += (dstress.array() * atoms.stress_.array()).sum();
    return value;
}

template <class Potential>
void validate_reduced_precision_inference(
    Potential &potential, const Atoms &source,
    const Neighborhoods &neighborhoods, const Atoms &reference,
    MTPGpuPrecision precision, double energy_per_atom_limit = 1.0e-5)
{
    INFO(MLIPCore::GpuDirect::precision_name(precision));
    std::string error;
    MTPGpuEvaluator evaluator;
    REQUIRE(evaluator.initialize(potential, 0, precision, &error));
    INFO(error);

    MTPGpuBatch batch;
    REQUIRE(batch.build(source, neighborhoods, &error));
    Atoms candidate = source;
    REQUIRE(batch.evaluate(evaluator, candidate, false, &error));
    INFO(error);

    REQUIRE(std::isfinite(candidate.energy_));
    const double energy_per_atom_error =
        std::abs(candidate.energy_ - reference.energy_) /
        static_cast<double>(source.size());
    INFO(energy_per_atom_error);
    REQUIRE(energy_per_atom_error <= energy_per_atom_limit);
    double force_delta_sq = 0.0;
    double force_reference_sq = 0.0;
    double stress_delta_sq = 0.0;
    double stress_reference_sq = 0.0;
    for (int atom = 0; atom < source.size(); ++atom) {
        REQUIRE(std::isfinite(candidate.site_energies_[atom]));
        for (int axis = 0; axis < 3; ++axis) {
            const double delta = candidate.forces_[atom][axis] -
                reference.forces_[atom][axis];
            REQUIRE(std::isfinite(candidate.forces_[atom][axis]));
            REQUIRE(std::abs(delta) <= 1.0e-3);
            force_delta_sq += delta * delta;
            force_reference_sq += reference.forces_[atom][axis] *
                reference.forces_[atom][axis];
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            const double delta = candidate.stress_(row, column) -
                reference.stress_(row, column);
            REQUIRE(std::isfinite(candidate.stress_(row, column)));
            REQUIRE(std::abs(delta) <= 1.0e-3);
            stress_delta_sq += delta * delta;
            stress_reference_sq += reference.stress_(row, column) *
                reference.stress_(row, column);
        }
    }
    REQUIRE(std::sqrt(force_delta_sq /
                      std::max(force_reference_sq, 1.0e-24)) <= 1.0e-4);
    const double stress_limit = precision == MTPGpuPrecision::Mixed
        ? 1.0e-4 : 5.0e-4;
    REQUIRE(std::sqrt(stress_delta_sq /
                      std::max(stress_reference_sq, 1.0e-20)) <=
            stress_limit);

    const auto stats = evaluator.context().potential_stats();
    REQUIRE(stats.moment_element_bytes == static_cast<int>(sizeof(float)));
    REQUIRE(stats.adjoint_element_bytes == static_cast<int>(
        precision == MTPGpuPrecision::Mixed ? sizeof(double) : sizeof(float)));

    batch.configure_gradient_reduction(true, 128);
    Atoms ordered_first = source;
    Atoms ordered_second = source;
    REQUIRE(batch.evaluate(evaluator, ordered_first, false, &error));
    REQUIRE(batch.evaluate(evaluator, ordered_second, true, &error));
    REQUIRE(std::memcmp(&ordered_first.energy_, &ordered_second.energy_,
                        sizeof(double)) == 0);
    for (int atom = 0; atom < source.size(); ++atom) {
        REQUIRE(ordered_first.site_energies_[atom] ==
                ordered_second.site_energies_[atom]);
        for (int axis = 0; axis < 3; ++axis) {
            REQUIRE(ordered_first.forces_[atom][axis] ==
                    ordered_second.forces_[atom][axis]);
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            REQUIRE(ordered_first.stress_(row, column) ==
                    ordered_second.stress_(row, column));
        }
    }

    Atoms energy_only = source;
    REQUIRE(batch.evaluate_energy(evaluator, energy_only, true, &error));
    REQUIRE(energy_only.energy_ == Catch::Approx(candidate.energy_)
        .margin(5.0e-5));
}

std::string analytic_mtpm_potential_text(bool quadratic)
{
    std::string text = configurable_radial_potential_text(
        "cosine", "exponential", -0.4, 0.7);
    text.resize(text.find("species_coeffs"));
    text += "species_coeffs = {0.13, -0.04, 0.07}\n";
    text += "moment_coeffs = {0.20, -0.10, -0.15, 0.08}\n";
    text += quadratic
        ? "model_coeffs = {0.8, -0.3, 0.12, -0.07, 0.09}\n"
        : "model_coeffs = {0.65}\n";
    return text;
}

std::string neural_mtpm_potential_text(int descriptor_count,
                                       int model_coefficient_count,
                                       int scalar_count = 2)
{
    std::string text = configurable_radial_potential_text(
        "cosine", "exponential", -0.4, 0.7);
    text.resize(text.find("species_coeffs"));
    if (scalar_count != 2) {
        text.resize(text.find("alpha_scalar_moments"));
        std::ostringstream mapping;
        mapping << "alpha_scalar_moments = " << scalar_count
                << "\nalpha_moment_mapping = {";
        for (int scalar = 0; scalar < scalar_count; ++scalar) {
            if (scalar != 0) mapping << ", ";
            mapping << (scalar % 2 == 0 ? 0 : 4);
        }
        text += mapping.str() + "}\n";
    }
    std::ostringstream output;
    output << text << "species_coeffs = {0.13";
    for (int descriptor = 0; descriptor < descriptor_count; ++descriptor) {
        output << ", " << 0.035 * static_cast<double>(descriptor + 1) *
            (descriptor % 2 == 0 ? 1.0 : -1.0);
    }
    output << "}\nmoment_coeffs = {";
    for (int coefficient = 0; coefficient < scalar_count * descriptor_count;
         ++coefficient) {
        if (coefficient != 0) output << ", ";
        output << 0.09 * std::sin(0.53 * static_cast<double>(coefficient + 1)) +
            0.025 * std::cos(0.31 * static_cast<double>(coefficient + 2));
    }
    output << "}\nmodel_coeffs = {";
    for (int coefficient = 0; coefficient < model_coefficient_count;
         ++coefficient) {
        if (coefficient != 0) output << ", ";
        output << 0.12 * std::sin(0.37 * static_cast<double>(coefficient + 1)) +
            0.04 * std::cos(0.19 * static_cast<double>(coefficient + 2));
    }
    output << "}\n";
    return output.str();
}

template <class Potential>
void validate_mtpm_cuda(Potential &potential,
                        int expected_neural_cache_stride = 0)
{
    std::string error;
    MTPGpuEvaluator evaluator;
    REQUIRE(evaluator.initialize(potential, 0, &error));
    INFO(error);
    REQUIRE(evaluator.context().potential_stats()
                .neural_cache_doubles_per_center ==
            expected_neural_cache_stride);

    Atoms source = configurable_radial_atoms();
    Neighborhoods neighborhoods(source, potential.cutoff);
    Atoms cpu = source;
    potential.getEFS(cpu, neighborhoods);
    MTPGpuBatch batch;
    REQUIRE(batch.build(source, neighborhoods, &error));
    Atoms gpu = source;
    REQUIRE(batch.evaluate(evaluator, gpu, false, &error));
    REQUIRE(gpu.energy_ ==
        Catch::Approx(cpu.energy_).epsilon(3e-12).margin(3e-11));
    for (int atom = 0; atom < source.size(); ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            REQUIRE(gpu.forces_[atom][axis] ==
                Catch::Approx(cpu.forces_[atom][axis])
                    .epsilon(8e-11).margin(8e-10));
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            REQUIRE(gpu.stress_(row, column) ==
                Catch::Approx(cpu.stress_(row, column))
                    .epsilon(8e-11).margin(8e-10));
        }
    }

    constexpr double denergy = 0.61;
    Eigen::Matrix<double, -1, 3> dforce(source.size(), 3);
    for (int atom = 0; atom < source.size(); ++atom)
        for (int axis = 0; axis < 3; ++axis)
            dforce(atom, axis) =
                0.013 * static_cast<double>((atom + 2) * (axis + 1) - 4);
    Eigen::Matrix3d dstress;
    dstress << 0.08, -0.03, 0.02,
               0.05, -0.06, 0.01,
              -0.04, 0.07, 0.03;

    potential.train_init();
    const int coefficient_count = potential.get_NL_coeffs_count();
    std::vector<double> cpu_gradient(coefficient_count, 0.0);
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> force_parameter_buffer;
    potential.AccumulateEFSCombinationGard(
        source, neighborhoods, denergy, dforce, dstress,
        force_parameter_buffer, cpu_gradient.data());
    std::vector<double> force_direction(3 * source.size());
    std::vector<double> stress_direction(9);
    for (int atom = 0; atom < source.size(); ++atom)
        for (int axis = 0; axis < 3; ++axis)
            force_direction[3 * atom + axis] = dforce(atom, axis);
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            stress_direction[3 * row + column] = dstress(row, column);
    std::vector<double> gpu_gradient(coefficient_count, 0.0);
    REQUIRE(batch.accumulate_gradient(
        evaluator, {denergy}, force_direction, stress_direction,
        gpu_gradient.data(), coefficient_count, false, &error));
    INFO(error);
    for (int parameter = 0; parameter < coefficient_count; ++parameter) {
        INFO(parameter);
        REQUIRE(gpu_gradient[parameter] ==
            Catch::Approx(cpu_gradient[parameter])
                .epsilon(2e-9).margin(6e-8));
    }

    // Exercise both sub-warp and cooperative descriptor reductions, including
    // a second gradient direction on the same retained forward workspace.
    if (expected_neural_cache_stride > 0) {
        for (int threads : {1, 16, 32, 64, 128, 256, 512}) {
            auto request = batch.request();
            double energy = 0.0;
            request.total_energy = &energy;
            request.propagate_threads = threads;
            REQUIRE(evaluator.context().compute_batch(request, &error));
            REQUIRE(energy == Catch::Approx(cpu.energy_)
                .epsilon(3e-12).margin(3e-11));
            for (int atom = 0; atom < source.size(); ++atom) {
                for (int axis = 0; axis < 3; ++axis) {
                    REQUIRE(request.force[3 * atom + axis] ==
                        Catch::Approx(cpu.forces_[atom][axis])
                            .epsilon(8e-11).margin(8e-10));
                }
            }
            MTPGpuGradientRequest gradient_request;
            const double energy_direction = denergy;
            std::vector<int> center_structure(source.size(), 0);
            gradient_request.structure_count = 1;
            gradient_request.center_structure = center_structure.data();
            gradient_request.dloss_denergy = &energy_direction;
            gradient_request.dloss_dforce = force_direction.data();
            gradient_request.dloss_dstress = stress_direction.data();
            gradient_request.gradient = gpu_gradient.data();
            gradient_request.gradient_count = coefficient_count;
            gradient_request.propagate_threads = threads;
            REQUIRE(evaluator.context().accumulate_gradient(gradient_request, &error));
            for (int parameter = 0; parameter < coefficient_count; ++parameter) {
                REQUIRE(gpu_gradient[parameter] ==
                    Catch::Approx(cpu_gradient[parameter])
                        .epsilon(2e-9).margin(6e-8));
            }
            std::vector<double> zero_force(force_direction.size(), 0.0);
            std::vector<double> zero_stress(9, 0.0);
            const double zero_energy = 0.0;
            gradient_request.dloss_denergy = &zero_energy;
            gradient_request.dloss_dforce = zero_force.data();
            gradient_request.dloss_dstress = zero_stress.data();
            REQUIRE(evaluator.context().accumulate_gradient(gradient_request, &error));
            for (double value : gpu_gradient) REQUIRE(value == Catch::Approx(0.0).margin(1e-12));

            request.compute_forces = false;
            request.site_virial = nullptr;
            REQUIRE(evaluator.context().compute_batch(request, &error));
            REQUIRE(energy == Catch::Approx(cpu.energy_)
                .epsilon(3e-12).margin(3e-11));
        }
        batch.configure_gradient_reduction(true, 128);
        Atoms ordered = source;
        REQUIRE(batch.evaluate(evaluator, ordered, false, &error));
        REQUIRE(ordered.energy_ == Catch::Approx(cpu.energy_)
            .epsilon(3e-12).margin(3e-11));
        REQUIRE(batch.accumulate_gradient(
            evaluator, {denergy}, force_direction, stress_direction,
            gpu_gradient.data(), coefficient_count, false, &error));
        const auto ordered_gradient = gpu_gradient;
        REQUIRE(batch.accumulate_gradient(
            evaluator, {denergy}, force_direction, stress_direction,
            gpu_gradient.data(), coefficient_count, false, &error));
        REQUIRE(std::memcmp(ordered_gradient.data(), gpu_gradient.data(),
                            coefficient_count * sizeof(double)) == 0);
        for (int parameter = 0; parameter < coefficient_count; ++parameter) {
            REQUIRE(gpu_gradient[parameter] == Catch::Approx(cpu_gradient[parameter])
                .epsilon(2e-9).margin(6e-8));
        }
    }

    std::vector<double> coefficients(coefficient_count);
    potential.nl_get_coeffs(coefficients.data());
    constexpr double step = 2e-6;
    for (int parameter = 0; parameter < coefficient_count; ++parameter) {
        const double original = coefficients[parameter];
        coefficients[parameter] = original + step;
        potential.nl_set_coeffs(coefficients.data());
        Atoms plus_atoms = source;
        potential.getEFS(plus_atoms, neighborhoods);
        const double plus = atom_efs_objective(
            plus_atoms, denergy, dforce, dstress);
        coefficients[parameter] = original - step;
        potential.nl_set_coeffs(coefficients.data());
        Atoms minus_atoms = source;
        potential.getEFS(minus_atoms, neighborhoods);
        const double minus = atom_efs_objective(
            minus_atoms, denergy, dforce, dstress);
        coefficients[parameter] = original;
        potential.nl_set_coeffs(coefficients.data());
        const double finite_difference = (plus - minus) / (2.0 * step);
        INFO(parameter);
        REQUIRE(cpu_gradient[parameter] ==
            Catch::Approx(finite_difference)
                .epsilon(8e-6).margin(8e-5));
    }
}

double evaluate(MTPGpuDirect &gpu, SmallSystem &system, bool outputs,
                bool compute_global_energy = true,
                MTPGpuForceKernel force_kernel = MTPGpuForceKernel::Auto)
{
    for (auto &f : system.force) f.fill(0.0);
    for (auto &v : system.virial) v.fill(0.0);
    system.site_energy.fill(0.0);

    double energy = 0.0;
    MTPGpuComputeRequest req;
    req.ilist = system.centers.data();
    req.numneigh = system.numneigh.data();
    req.firstneigh = system.nptr.data();
    req.inum = SmallSystem::n;
    req.nlocal = SmallSystem::n;
    req.ntotal = SmallSystem::n;
    req.type = system.type.data();
    req.x = system.xptr.data();
    req.force = system.fptr.data();
    req.total_energy = compute_global_energy ? &energy : nullptr;
    req.site_energy = outputs ? system.site_energy.data() : nullptr;
    req.site_virial = outputs ? system.vptr.data() : nullptr;
    req.accumulate_force = false;
    req.compute_global_energy = compute_global_energy;
    req.compute_site_energy = outputs;
    req.force_kernel = force_kernel;
    req.neighbor_index_mask = 0x3fffffff;
    req.neighbor_lanes = 8; // 3 neighbors deliberately leave a partial tile.

    std::string error;
    const bool ok = gpu.compute_lammps(req, &error);
    INFO(error);
    REQUIRE(ok);
    return energy;
}

void evaluate_forces_only(MTPGpuDirect &gpu, SmallSystem &system)
{
    for (auto &f : system.force) f.fill(0.0);

    MTPGpuComputeRequest req;
    req.ilist = system.centers.data();
    req.numneigh = system.numneigh.data();
    req.firstneigh = system.nptr.data();
    req.inum = SmallSystem::n;
    req.nlocal = SmallSystem::n;
    req.ntotal = SmallSystem::n;
    req.type = system.type.data();
    req.x = system.xptr.data();
    req.force = system.fptr.data();
    req.total_energy = nullptr;
    req.site_energy = nullptr;
    req.site_virial = nullptr;
    req.accumulate_force = false;
    req.compute_global_energy = false;
    req.compute_site_energy = false;
    req.neighbor_index_mask = 0x3fffffff;
    req.neighbor_lanes = 8;

    std::string error;
    const bool ok = gpu.compute_lammps(req, &error);
    INFO(error);
    REQUIRE(ok);
}

double evaluate_batch(MTPGpuDirect &gpu, SmallSystem &system,
                      bool reuse_topology, bool deterministic = false,
                      bool compute_forces = true, int propagate_threads = 256)
{
    std::array<int, SmallSystem::n + 1> offsets{};
    std::array<int, SmallSystem::n * 3> neighbors{};
    std::array<double, SmallSystem::n * 3> positions{};
    std::array<double, SmallSystem::n * 3> force{};
    std::array<double, SmallSystem::n * 6> virial{};
    std::array<double, SmallSystem::n> site_energy{};
    std::array<double, SmallSystem::n * 3 * 3> vectors{};
    for (int i = 0; i < SmallSystem::n; ++i) {
        offsets[i + 1] = offsets[i] + 3;
        for (int a = 0; a < 3; ++a) positions[3 * i + a] = system.x[i][a];
        for (int jj = 0; jj < 3; ++jj) {
            const int edge = 3 * i + jj;
            const int atj = system.neighbors[i][jj] & 0x3fffffff;
            neighbors[edge] = atj;
            for (int a = 0; a < 3; ++a) {
                vectors[3 * edge + a] = system.x[atj][a] - system.x[i][a];
            }
        }
    }

    double energy = 0.0;
    MTPGpuBatchRequest req;
    req.centers = system.centers.data();
    req.offsets = offsets.data();
    req.neighbors = neighbors.data();
    req.neighbor_vectors = vectors.data();
    req.center_count = SmallSystem::n;
    req.nlocal = SmallSystem::n;
    req.ntotal = SmallSystem::n;
    req.type = system.type.data();
    req.x = positions.data();
    req.force = compute_forces ? force.data() : nullptr;
    req.total_energy = &energy;
    req.site_energy = site_energy.data();
    req.site_virial = compute_forces ? virial.data() : nullptr;
    req.accumulate_force = false;
    req.reuse_topology = reuse_topology;
    req.compute_forces = compute_forces;
    req.deterministic_reduction = deterministic;

    req.propagate_threads = propagate_threads;
    std::string error;
    const bool ok = gpu.compute_batch(req, &error);
    INFO(error);
    REQUIRE(ok);
    for (int i = 0; i < SmallSystem::n; ++i) {
        if (compute_forces) {
            for (int a = 0; a < 3; ++a) system.force[i][a] = force[3 * i + a];
            for (int a = 0; a < 6; ++a) system.virial[i][a] = virial[6 * i + a];
        }
        system.site_energy[i] = site_energy[i];
    }
    return energy;
}

double training_objective(double energy, const SmallSystem &system,
                          double denergy,
                          const std::array<double, SmallSystem::n * 3> &dforce,
                          const std::array<double, 9> &dstress)
{
    double value = denergy * energy;
    for (int i = 0; i < SmallSystem::n; ++i) {
        for (int a = 0; a < 3; ++a) {
            value += dforce[3 * i + a] * system.force[i][a];
        }
        value += dstress[0] * system.virial[i][0];
        value += dstress[4] * system.virial[i][1];
        value += dstress[8] * system.virial[i][2];
        value += (dstress[1] + dstress[3]) * system.virial[i][3];
        value += (dstress[2] + dstress[6]) * system.virial[i][4];
        value += (dstress[5] + dstress[7]) * system.virial[i][5];
    }
    return value;
}

} // namespace

TEST_CASE("GPU MTP precision policy parsing is explicit", "[mtp][gpu][precision]" IMRMLP_GPU_TEST_TAG)
{
    MTPGpuInitializationOptions defaults;
    REQUIRE(defaults.precision == MTPGpuPrecision::FP64);
    REQUIRE(std::string(MLIPCore::GpuDirect::precision_name(
                defaults.precision)) == "fp64");

    MTPGpuPrecision parsed = MTPGpuPrecision::FP64;
    REQUIRE(MLIPCore::GpuDirect::parse_precision("mixed", &parsed));
    REQUIRE(parsed == MTPGpuPrecision::Mixed);
    REQUIRE(MLIPCore::GpuDirect::parse_precision(
        "mixed-aggressive", &parsed));
    REQUIRE(parsed == MTPGpuPrecision::MixedAggressive);
    REQUIRE(std::string(MLIPCore::GpuDirect::precision_name(parsed)) ==
            "mixed-aggressive");
    REQUIRE(MLIPCore::GpuDirect::parse_precision("fp64", &parsed));
    REQUIRE(parsed == MTPGpuPrecision::FP64);
    REQUIRE_FALSE(MLIPCore::GpuDirect::parse_precision("fp32", &parsed));
    REQUIRE_FALSE(MLIPCore::GpuDirect::parse_precision("MIXED", &parsed));
    REQUIRE_FALSE(MLIPCore::GpuDirect::parse_precision("mixed", nullptr));

    MTPGpuForceKernel force_kernel = MTPGpuForceKernel::Auto;
    REQUIRE(MLIPCore::GpuDirect::parse_force_kernel("basic-tile", &force_kernel));
    REQUIRE(force_kernel == MTPGpuForceKernel::BasicTile);
    REQUIRE(MLIPCore::GpuDirect::parse_force_kernel(
        "center-neighbors", &force_kernel));
    REQUIRE(force_kernel == MTPGpuForceKernel::CenterNeighbors);
    REQUIRE(std::string(MLIPCore::GpuDirect::force_kernel_name(force_kernel)) ==
            "center-neighbors");
    REQUIRE_FALSE(MLIPCore::GpuDirect::parse_force_kernel("center", &force_kernel));
    REQUIRE_FALSE(MLIPCore::GpuDirect::parse_force_kernel("auto", nullptr));
}

TEST_CASE("GPU MTP force kernel variants preserve E F and site virial",
          "[mtp][gpu][force-kernel]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    MTPGpuDirect gpu;
    REQUIRE(gpu.set_device(0, &error));
    REQUIRE(gpu.initialize(small_potential_view(), &error));
    SmallSystem basic_system;
    SmallSystem center_system;
    const double basic_energy = evaluate(
        gpu, basic_system, true, true, MTPGpuForceKernel::BasicTile);
    const double center_energy = evaluate(
        gpu, center_system, true, true, MTPGpuForceKernel::CenterNeighbors);
    SmallSystem auto_system;
    const double auto_energy = evaluate(
        gpu, auto_system, true, true, MTPGpuForceKernel::Auto);
    REQUIRE(center_energy == Catch::Approx(basic_energy).margin(1e-12));
    REQUIRE(auto_energy == Catch::Approx(basic_energy).margin(1e-12));
    for (int i = 0; i < SmallSystem::n; ++i) {
        REQUIRE(center_system.site_energy[i] ==
                Catch::Approx(basic_system.site_energy[i]).margin(1e-12));
        REQUIRE(auto_system.site_energy[i] ==
                Catch::Approx(basic_system.site_energy[i]).margin(1e-12));
        for (int component = 0; component < 3; ++component) {
            REQUIRE(center_system.force[i][component] ==
                    Catch::Approx(basic_system.force[i][component]).margin(1e-12));
            REQUIRE(auto_system.force[i][component] ==
                    Catch::Approx(basic_system.force[i][component]).margin(1e-12));
        }
        for (int component = 0; component < 6; ++component) {
            REQUIRE(center_system.virial[i][component] ==
                    Catch::Approx(basic_system.virial[i][component]).margin(1e-12));
            REQUIRE(auto_system.virial[i][component] ==
                    Catch::Approx(basic_system.virial[i][component]).margin(1e-12));
        }
    }
    const auto stats = gpu.runtime_stats();
    // The small fixture is below Auto's four-SM launch threshold and also has
    // fewer basic moments than the default tile.
    REQUIRE(stats.basic_tile_force_launches == 2);
    REQUIRE(stats.center_neighbors_force_launches == 1);
    REQUIRE(stats.topology_uploads == 3);
    REQUIRE(stats.type_uploads == 3);
    REQUIRE(stats.coordinate_uploads == 3);
    REQUIRE(stats.topology_upload_bytes > 0);
    REQUIRE(stats.type_upload_bytes > 0);
    REQUIRE(stats.coordinate_upload_bytes > 0);
}

TEST_CASE("GPU MTP mixed storage inference preserves FP64 interfaces",
          "[mtp][gpu][precision]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    MTPGpuDirect fp64;
    MTPGpuDirect mixed;
    MTPGpuDirect aggressive;
    REQUIRE(fp64.set_device(0, &error));
    REQUIRE(mixed.set_device(0, &error));
    REQUIRE(aggressive.set_device(0, &error));
    REQUIRE(fp64.initialize(small_potential_view(), &error));
    MTPGpuInitializationOptions mixed_options;
    mixed_options.precision = MTPGpuPrecision::Mixed;
    REQUIRE(mixed.initialize(small_potential_view(), mixed_options, &error));
    MTPGpuInitializationOptions aggressive_options;
    aggressive_options.precision = MTPGpuPrecision::MixedAggressive;
    REQUIRE(aggressive.initialize(
        small_potential_view(), aggressive_options, &error));
    INFO(error);

    REQUIRE(fp64.precision() == MTPGpuPrecision::FP64);
    REQUIRE(mixed.precision() == MTPGpuPrecision::Mixed);
    REQUIRE(aggressive.precision() == MTPGpuPrecision::MixedAggressive);
    REQUIRE(fp64.potential_stats().moment_element_bytes ==
            static_cast<int>(sizeof(double)));
    REQUIRE(mixed.potential_stats().moment_element_bytes ==
            static_cast<int>(sizeof(float)));
    REQUIRE(fp64.potential_stats().adjoint_element_bytes ==
            static_cast<int>(sizeof(double)));
    REQUIRE(mixed.potential_stats().adjoint_element_bytes ==
            static_cast<int>(sizeof(double)));
    REQUIRE(aggressive.potential_stats().moment_element_bytes ==
            static_cast<int>(sizeof(float)));
    REQUIRE(aggressive.potential_stats().adjoint_element_bytes ==
            static_cast<int>(sizeof(float)));
    REQUIRE(fp64.potential_stats().potential_bytes ==
            mixed.potential_stats().potential_bytes);

    SmallSystem fp64_system;
    SmallSystem mixed_system;
    SmallSystem aggressive_system;
    const double fp64_energy = evaluate_batch(fp64, fp64_system, false);
    const double mixed_energy = evaluate_batch(mixed, mixed_system, false);
    const double aggressive_energy =
        evaluate_batch(aggressive, aggressive_system, false);
    REQUIRE(std::isfinite(mixed_energy));
    REQUIRE(std::abs(mixed_energy - fp64_energy) <=
            1e-5 * SmallSystem::n);
    double force_delta_sq = 0.0;
    double force_reference_sq = 0.0;
    double virial_delta_sq = 0.0;
    double virial_reference_sq = 0.0;
    for (int atom = 0; atom < SmallSystem::n; ++atom) {
        REQUIRE(std::isfinite(mixed_system.site_energy[atom]));
        REQUIRE(std::abs(mixed_system.site_energy[atom] -
                         fp64_system.site_energy[atom]) <= 1e-5);
        for (int component = 0; component < 3; ++component) {
            const double delta = mixed_system.force[atom][component] -
                fp64_system.force[atom][component];
            REQUIRE(std::isfinite(mixed_system.force[atom][component]));
            REQUIRE(std::abs(delta) <= 1e-3);
            force_delta_sq += delta * delta;
            force_reference_sq += fp64_system.force[atom][component] *
                fp64_system.force[atom][component];
        }
        for (int component = 0; component < 6; ++component) {
            const double delta = mixed_system.virial[atom][component] -
                fp64_system.virial[atom][component];
            REQUIRE(std::isfinite(mixed_system.virial[atom][component]));
            REQUIRE(std::abs(delta) <= 1e-3);
            virial_delta_sq += delta * delta;
            virial_reference_sq += fp64_system.virial[atom][component] *
                fp64_system.virial[atom][component];
        }
    }
    REQUIRE(std::sqrt(force_delta_sq /
                      std::max(force_reference_sq, 1e-24)) <= 1e-4);
    REQUIRE(std::sqrt(virial_delta_sq /
                      std::max(virial_reference_sq, 1e-20)) <= 1e-4);

    REQUIRE(std::isfinite(aggressive_energy));
    REQUIRE(std::abs(aggressive_energy - fp64_energy) <=
            1e-5 * SmallSystem::n);
    force_delta_sq = 0.0;
    virial_delta_sq = 0.0;
    for (int atom = 0; atom < SmallSystem::n; ++atom) {
        REQUIRE(std::isfinite(aggressive_system.site_energy[atom]));
        REQUIRE(std::abs(aggressive_system.site_energy[atom] -
                         fp64_system.site_energy[atom]) <= 1e-5);
        for (int component = 0; component < 3; ++component) {
            const double delta = aggressive_system.force[atom][component] -
                fp64_system.force[atom][component];
            REQUIRE(std::isfinite(
                aggressive_system.force[atom][component]));
            REQUIRE(std::abs(delta) <= 1e-3);
            force_delta_sq += delta * delta;
        }
        for (int component = 0; component < 6; ++component) {
            const double delta = aggressive_system.virial[atom][component] -
                fp64_system.virial[atom][component];
            REQUIRE(std::isfinite(
                aggressive_system.virial[atom][component]));
            REQUIRE(std::abs(delta) <= 1e-3);
            virial_delta_sq += delta * delta;
        }
    }
    REQUIRE(std::sqrt(force_delta_sq /
                      std::max(force_reference_sq, 1e-24)) <= 1e-4);
    REQUIRE(std::sqrt(virial_delta_sq /
                      std::max(virial_reference_sq, 1e-20)) <= 5e-4);

    SmallSystem fp64_energy_only;
    SmallSystem mixed_energy_only;
    REQUIRE(evaluate_batch(mixed, mixed_energy_only, true, false, false) ==
            Catch::Approx(evaluate_batch(
                fp64, fp64_energy_only, true, false, false)).margin(4e-5));
    SmallSystem aggressive_energy_only;
    REQUIRE(evaluate_batch(
                aggressive, aggressive_energy_only, true, false, false) ==
            Catch::Approx(evaluate_batch(
                fp64, fp64_energy_only, true, false, false)).margin(4e-5));

    SmallSystem ordered_first;
    SmallSystem ordered_second;
    const double ordered_energy_first =
        evaluate_batch(mixed, ordered_first, true, true);
    const double ordered_energy_second =
        evaluate_batch(mixed, ordered_second, true, true);
    REQUIRE(std::memcmp(&ordered_energy_first, &ordered_energy_second,
                        sizeof(double)) == 0);
    REQUIRE(std::memcmp(ordered_first.force.data(),
                        ordered_second.force.data(),
                        sizeof(ordered_first.force)) == 0);
    REQUIRE(std::memcmp(ordered_first.virial.data(),
                        ordered_second.virial.data(),
                        sizeof(ordered_first.virial)) == 0);

    SmallSystem aggressive_ordered_first;
    SmallSystem aggressive_ordered_second;
    const double aggressive_ordered_energy_first = evaluate_batch(
        aggressive, aggressive_ordered_first, true, true);
    const double aggressive_ordered_energy_second = evaluate_batch(
        aggressive, aggressive_ordered_second, true, true);
    REQUIRE(std::memcmp(&aggressive_ordered_energy_first,
                        &aggressive_ordered_energy_second,
                        sizeof(double)) == 0);
    REQUIRE(std::memcmp(aggressive_ordered_first.force.data(),
                        aggressive_ordered_second.force.data(),
                        sizeof(aggressive_ordered_first.force)) == 0);
    REQUIRE(std::memcmp(aggressive_ordered_first.virial.data(),
                        aggressive_ordered_second.virial.data(),
                        sizeof(aggressive_ordered_first.virial)) == 0);

    const auto memory = mixed.memory_stats();
    const std::size_t moment_slots =
        SmallSystem::n * mixed.potential_stats().active_moment_slots;
    REQUIRE(memory.precision == MTPGpuPrecision::Mixed);
    REQUIRE(memory.moment_workspace_bytes == moment_slots * sizeof(float));
    REQUIRE(memory.adjoint_workspace_bytes == moment_slots * sizeof(double));
    REQUIRE(memory.conversion_buffer_bytes == 0);
    REQUIRE(memory.tracked_device_bytes >=
            memory.potential_bytes + memory.moment_workspace_bytes +
                memory.adjoint_workspace_bytes);

    const auto aggressive_memory = aggressive.memory_stats();
    REQUIRE(aggressive_memory.precision ==
            MTPGpuPrecision::MixedAggressive);
    REQUIRE(aggressive_memory.moment_workspace_bytes ==
            moment_slots * sizeof(float));
    REQUIRE(aggressive_memory.adjoint_workspace_bytes ==
            moment_slots * sizeof(float));
    REQUIRE(aggressive_memory.conversion_buffer_bytes == 0);
    REQUIRE(aggressive_memory.tracked_device_bytes <
            memory.tracked_device_bytes);
    REQUIRE(memory.tracked_device_bytes -
                aggressive_memory.tracked_device_bytes ==
            moment_slots * (sizeof(double) - sizeof(float)));

    MTPGpuGradientRequest gradient_request;
    REQUIRE_FALSE(mixed.accumulate_gradient(gradient_request, &error));
    REQUIRE(error.find("mixed precision GPU fitting is not implemented") !=
            std::string::npos);
    REQUIRE_FALSE(aggressive.accumulate_gradient(gradient_request, &error));
    REQUIRE(error.find("mixed precision GPU fitting is not implemented") !=
            std::string::npos);

    MTPGpuPotentialView nonlinear = small_potential_view();
    nonlinear.model = MTPGpuModel::MTPMQuadratic;
    MTPGpuDirect unsupported_model;
    REQUIRE(unsupported_model.set_device(0, &error));
    REQUIRE_FALSE(unsupported_model.initialize(
        nonlinear, mixed_options, &error));
    REQUIRE(error.find("classic linear PotMTP inference only") !=
            std::string::npos);

    for (const RadialBasis basis : {RadialBasis::ChebyCos,
                                    RadialBasis::ChebyshevPoly5}) {
        INFO(static_cast<int>(basis));
        MTPGpuPotentialView alternate_radial = small_potential_view(basis);
        REQUIRE(unsupported_model.initialize(
            alternate_radial, mixed_options, &error));
        SmallSystem alternate_system;
        const double alternate_energy = evaluate_batch(
            unsupported_model, alternate_system, false);
        REQUIRE(alternate_energy == Catch::Approx(
            reference_energy(alternate_system, basis)).margin(2e-4));
    }

    // Changing precision on an existing evaluator must invalidate and release
    // the typed batch workspaces before they can be reused.
    REQUIRE(mixed.initialize(small_potential_view(), &error));
    REQUIRE(mixed.precision() == MTPGpuPrecision::FP64);
    REQUIRE(mixed.memory_stats().moment_workspace_bytes == 0);
    REQUIRE(mixed.memory_stats().adjoint_workspace_bytes == 0);

    REQUIRE(aggressive.initialize(
        small_potential_view(), mixed_options, &error));
    REQUIRE(aggressive.precision() == MTPGpuPrecision::Mixed);
    REQUIRE(aggressive.memory_stats().moment_workspace_bytes == 0);
    REQUIRE(aggressive.memory_stats().adjoint_workspace_bytes == 0);
}

TEST_CASE("direct GPU MTP matches energy and force derivatives", "[mtp][gpu]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) || device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    for (const RadialBasis basis : {RadialBasis::Chebyshev,
                                    RadialBasis::ChebyCos,
                                    RadialBasis::ChebyshevPoly5}) {
        MTPGpuDirect gpu;
        REQUIRE(gpu.set_device(0, &error));
        REQUIRE(gpu.initialize(small_potential_view(basis), &error));
        INFO(error);

        SmallSystem system;
        const double energy = evaluate(gpu, system, true);
        REQUIRE(energy == Catch::Approx(reference_energy(system, basis)).epsilon(1e-11));
        double site_energy_sum = 0.0;
        for (const double value : system.site_energy) site_energy_sum += value;
        REQUIRE(site_energy_sum == Catch::Approx(energy).epsilon(1e-12));

        std::array<std::array<double, 3>, SmallSystem::n> analytic_force = system.force;
        const auto analytic_virial = system.virial;
        SmallSystem site_only_system;
        REQUIRE(evaluate(gpu, site_only_system, true, false) == 0.0);
        double site_only_sum = 0.0;
        for (double value : site_only_system.site_energy) {
            site_only_sum += value;
        }
        REQUIRE(site_only_sum == Catch::Approx(energy).epsilon(1e-12));
        SmallSystem forces_only_system;
        evaluate_forces_only(gpu, forces_only_system);
        for (int i = 0; i < SmallSystem::n; ++i) {
            for (int a = 0; a < 3; ++a) {
                REQUIRE(forces_only_system.force[i][a] ==
                    Catch::Approx(analytic_force[i][a]).margin(1e-12));
            }
        }
        const auto original_x = system.x;
        constexpr double eps = 1e-6;
        for (int i = 0; i < SmallSystem::n; ++i) {
            for (int a = 0; a < 3; ++a) {
                system.x[i][a] += eps;
                const double ep = evaluate(gpu, system, false);
                system.x[i][a] -= 2.0 * eps;
                const double em = evaluate(gpu, system, false);
                system.x[i][a] += eps;
                const double finite_force = -(ep - em) / (2.0 * eps);
                REQUIRE(analytic_force[i][a] == Catch::Approx(finite_force).margin(2e-7));
            }
        }

        for (int a = 0; a < 3; ++a) {
            double net_force = 0.0;
            for (const auto &f : analytic_force) net_force += f[a];
            REQUIRE(net_force == Catch::Approx(0.0).margin(1e-11));
        }

        system.x = original_x;
        for (int i = 0; i < SmallSystem::n; ++i) {
            system.x[i][0] = original_x[i][0] * (1.0 + eps);
        }
        const double exx_plus = evaluate(gpu, system, false);
        for (int i = 0; i < SmallSystem::n; ++i) {
            system.x[i][0] = original_x[i][0] * (1.0 - eps);
        }
        const double exx_minus = evaluate(gpu, system, false);
        const double finite_vxx = -(exx_plus - exx_minus) / (2.0 * eps);
        double analytic_vxx = 0.0;
        for (const auto &v : analytic_virial) analytic_vxx += v[0];
        REQUIRE(analytic_vxx == Catch::Approx(finite_vxx).margin(5e-7));

        // Symmetric xy strain: x' = x + eps*y/2, y' = y + eps*x/2.
        for (int i = 0; i < SmallSystem::n; ++i) {
            system.x[i][0] = original_x[i][0] + 0.5 * eps * original_x[i][1];
            system.x[i][1] = original_x[i][1] + 0.5 * eps * original_x[i][0];
            system.x[i][2] = original_x[i][2];
        }
        const double exy_plus = evaluate(gpu, system, false);
        for (int i = 0; i < SmallSystem::n; ++i) {
            system.x[i][0] = original_x[i][0] - 0.5 * eps * original_x[i][1];
            system.x[i][1] = original_x[i][1] - 0.5 * eps * original_x[i][0];
        }
        const double exy_minus = evaluate(gpu, system, false);
        const double finite_vxy = -(exy_plus - exy_minus) / (2.0 * eps);
        double analytic_vxy = 0.0;
        for (const auto &v : analytic_virial) analytic_vxy += v[3];
        REQUIRE(analytic_vxy == Catch::Approx(finite_vxy).margin(5e-7));
    }
}

TEST_CASE("GPU mixed inference supports alternate single radial bases",
          "[mtp][gpu][precision][radial]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    for (const std::string basis :
         {"RBChebyCos", "RBChebyshevPoly5"}) {
        INFO(basis);
        std::istringstream input(single_radial_potential_text(basis));
        PotMTP potential;
        REQUIRE_NOTHROW(potential.Load(input));
        REQUIRE_FALSE(potential.radial_basis_split);
        REQUIRE(potential.p_radical_basis->name == basis);

        Atoms source = configurable_radial_atoms();
        Neighborhoods neighborhoods(source, potential.cutoff);
        Atoms cpu = source;
        potential.getEFS(cpu, neighborhoods);
        MTPGpuEvaluator fp64;
        REQUIRE(fp64.initialize(potential, 0, &error));
        MTPGpuBatch batch;
        REQUIRE(batch.build(source, neighborhoods, &error));
        Atoms gpu = source;
        REQUIRE(batch.evaluate(fp64, gpu, false, &error));
        REQUIRE(gpu.energy_ ==
            Catch::Approx(cpu.energy_).epsilon(3e-12).margin(3e-11));
        for (int atom = 0; atom < source.size(); ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                REQUIRE(gpu.forces_[atom][axis] ==
                    Catch::Approx(cpu.forces_[atom][axis])
                        .epsilon(8e-11).margin(8e-10));
            }
        }
        validate_reduced_precision_inference(
            potential, source, neighborhoods, cpu, MTPGpuPrecision::Mixed);
        validate_reduced_precision_inference(
            potential, source, neighborhoods, cpu,
            MTPGpuPrecision::MixedAggressive);
    }
}

TEST_CASE("GPU configurable radial bases match CPU boundaries EFS and gradients",
          "[mtp][gpu][configurable-radial][gradient]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    struct RadialCase {
        const char *cutoff;
        const char *shift;
        double cutoff_b;
        double shift_lambda;
    };
    const std::array<RadialCase, 7> cases{{
        {"quadratic", "linear", -0.4, 0.0},
        {"quadratic", "exponential", -0.4, 1.2},
        {"quadratic", "exponential", -0.4, 0.0},
        {"cosine", "linear", -0.4, 0.0},
        {"cosine", "exponential", -0.4, -0.8},
        {"poly5", "linear", -0.7, 0.0},
        {"poly5", "exponential", -0.7, 1.2},
    }};

    for (const auto &radial_case : cases) {
        INFO("cutoff=" << radial_case.cutoff);
        INFO("shift=" << radial_case.shift);
        std::istringstream input(configurable_radial_potential_text(
            radial_case.cutoff, radial_case.shift, radial_case.cutoff_b,
            radial_case.shift_lambda));
        PotMTP potential;
        REQUIRE_NOTHROW(potential.Load(input));
        REQUIRE_FALSE(potential.radial_basis_split);
        REQUIRE(potential.p_radical_basis->name ==
                "RBChebyshevConfigurable");

        MTPGpuEvaluator evaluator;
        REQUIRE(evaluator.initialize(potential, 0, &error));
        INFO(error);

        // Gate the piecewise interval before using the basis in full E/F/S.
        for (const double distance :
             {0.5 - 1e-6, 0.5, 0.5 + 1e-6, 2.4,
              5.0 - 1e-6, 5.0, 5.0 + 1e-6}) {
            INFO("distance=" << distance);
            Atoms source = configurable_radial_atoms(distance);
            Neighborhoods neighborhoods(source, potential.cutoff);
            Atoms cpu = source;
            potential.getEFS(cpu, neighborhoods);

            MTPGpuBatch batch;
            REQUIRE(batch.build(source, neighborhoods, &error));
            Atoms gpu = source;
            REQUIRE(batch.evaluate(evaluator, gpu, false, &error));
            REQUIRE(gpu.energy_ ==
                Catch::Approx(cpu.energy_).epsilon(2e-12).margin(2e-11));
            for (int atom = 0; atom < source.size(); ++atom) {
                for (int axis = 0; axis < 3; ++axis) {
                    REQUIRE(gpu.forces_[atom][axis] ==
                        Catch::Approx(cpu.forces_[atom][axis])
                            .epsilon(4e-11).margin(5e-10));
                }
            }
            validate_reduced_precision_inference(
                potential, source, neighborhoods, cpu,
                MTPGpuPrecision::Mixed, 1.005e-5);
        }

        Atoms source = configurable_radial_atoms();
        Neighborhoods neighborhoods(source, potential.cutoff);
        Atoms cpu = source;
        potential.getEFS(cpu, neighborhoods);
        MTPGpuBatch batch;
        REQUIRE(batch.build(source, neighborhoods, &error));
        Atoms gpu = source;
        REQUIRE(batch.evaluate(evaluator, gpu, false, &error));
        REQUIRE(gpu.energy_ ==
            Catch::Approx(cpu.energy_).epsilon(2e-12).margin(2e-11));
        for (int atom = 0; atom < source.size(); ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                INFO(atom);
                INFO(axis);
                REQUIRE(gpu.forces_[atom][axis] ==
                    Catch::Approx(cpu.forces_[atom][axis])
                        .epsilon(4e-11).margin(5e-10));
            }
        }
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                INFO(row);
                INFO(column);
                REQUIRE(gpu.stress_(row, column) ==
                    Catch::Approx(cpu.stress_(row, column))
                        .epsilon(4e-11).margin(5e-10));
            }
        }
        validate_reduced_precision_inference(
            potential, source, neighborhoods, cpu,
            MTPGpuPrecision::Mixed);
        validate_reduced_precision_inference(
            potential, source, neighborhoods, cpu,
            MTPGpuPrecision::MixedAggressive);

        constexpr double denergy = 0.7;
        Eigen::Matrix<double, -1, 3> dforce(source.size(), 3);
        for (int atom = 0; atom < source.size(); ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                dforce(atom, axis) =
                    0.01 * static_cast<double>((atom + 1) * (axis + 2) - 5);
            }
        }
        Eigen::Matrix3d dstress;
        dstress << 0.11, 0.03, -0.02,
                   0.07, -0.07, 0.04,
                  -0.01, 0.06, 0.05;

        potential.train_init();
        std::vector<double> cpu_gradient(potential.reg_coeff_size, 0.0);
        potential.AccumulateEFSCombinationGard(
            source, neighborhoods, denergy, dforce, dstress,
            cpu_gradient.data());
        std::vector<double> force_direction(3 * source.size());
        std::vector<double> stress_direction(9);
        for (int atom = 0; atom < source.size(); ++atom)
            for (int axis = 0; axis < 3; ++axis)
                force_direction[3 * atom + axis] = dforce(atom, axis);
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                stress_direction[3 * row + column] = dstress(row, column);
        std::vector<double> gpu_gradient(potential.reg_coeff_size, 0.0);
        REQUIRE(batch.accumulate_gradient(
            evaluator, {denergy}, force_direction, stress_direction,
            gpu_gradient.data(), static_cast<int>(gpu_gradient.size()),
            false, &error));
        INFO(error);
        for (int parameter = 0; parameter < potential.reg_coeff_size;
             ++parameter) {
            INFO(parameter);
            REQUIRE(gpu_gradient[parameter] ==
                Catch::Approx(cpu_gradient[parameter])
                    .epsilon(5e-10).margin(3e-8));
        }

        constexpr double step = 2e-6;
        for (int parameter = 0; parameter < potential.reg_coeff_size;
             ++parameter) {
            const double original = potential.regression_coeffs[parameter];
            potential.regression_coeffs[parameter] = original + step;
            Atoms plus_atoms = source;
            potential.getEFS(plus_atoms, neighborhoods);
            const double plus = atom_efs_objective(
                plus_atoms, denergy, dforce, dstress);
            potential.regression_coeffs[parameter] = original - step;
            Atoms minus_atoms = source;
            potential.getEFS(minus_atoms, neighborhoods);
            const double minus = atom_efs_objective(
                minus_atoms, denergy, dforce, dstress);
            potential.regression_coeffs[parameter] = original;
            const double finite_difference = (plus - minus) / (2.0 * step);
            INFO(parameter);
            REQUIRE(cpu_gradient[parameter] ==
                Catch::Approx(finite_difference)
                    .epsilon(4e-6).margin(5e-5));
        }

        std::ostringstream saved;
        potential.Save(saved);
        std::istringstream reload_input(saved.str());
        PotMTP reloaded;
        REQUIRE_NOTHROW(reloaded.Load(reload_input));
        MTPGpuEvaluator reloaded_evaluator;
        REQUIRE(reloaded_evaluator.initialize(reloaded, 0, &error));
    }
}

TEST_CASE("GPU analytic MTP-M models match CPU EFS and all parameter gradients",
          "[mtp][gpu][mtp-m][gradient]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    SECTION("quadratic external model") {
        std::istringstream input(analytic_mtpm_potential_text(true));
        Potentials::PotMTP_M<Potentials::m_model_mtpq2> potential;
        REQUIRE_NOTHROW(potential.Load(input));
        REQUIRE(potential.get_NL_coeffs_count() == 16);
        validate_mtpm_cuda(potential);
    }

    SECTION("ArcSinh external model") {
        std::istringstream input(analytic_mtpm_potential_text(false));
        Potentials::PotMTP_M<Potentials::m_model_mtpfs> potential;
        REQUIRE_NOTHROW(potential.Load(input));
        REQUIRE(potential.get_NL_coeffs_count() == 12);
        validate_mtpm_cuda(potential);
    }
}

TEST_CASE("GPU neural MTP-M models match CPU EFS and all parameter gradients",
          "[mtp][gpu][mtp-m][nn][gradient]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    SECTION("one hidden layer with supported activations") {
        constexpr int descriptor_count = 3;
        constexpr int hidden_count = 4;
        constexpr int model_coefficient_count =
            descriptor_count * hidden_count + 2 * hidden_count;
        GPSettings settings;
        settings.Modify("n_input", std::to_string(descriptor_count));
        settings.Modify("n_neuron", std::to_string(hidden_count));
        settings.Modify("activation_function", GENERATE("TANH", "RELU", "SIGMOID", "LINEAR"));
        Potentials::PotMTP_M<Potentials::m_model_nn1> potential;
        potential.model.init(settings);
        std::istringstream input(neural_mtpm_potential_text(
            descriptor_count, model_coefficient_count));
        REQUIRE_NOTHROW(potential.Load(input));
        REQUIRE(potential.model_coeff_size == model_coefficient_count);
        REQUIRE(potential.get_NL_coeffs_count() == 34);
        validate_mtpm_cuda(
            potential, 4 * descriptor_count + 4 * hidden_count);
    }

    SECTION("two hidden layers with supported activations") {
        constexpr int descriptor_count = 3;
        constexpr int hidden1_count = 4;
        constexpr int hidden2_count = 3;
        constexpr int model_coefficient_count =
            descriptor_count * hidden1_count + hidden1_count +
            hidden1_count * hidden2_count + 2 * hidden2_count;
        GPSettings settings;
        settings.Modify("n_input", std::to_string(descriptor_count));
        settings.Modify("n_neuron", std::to_string(hidden1_count));
        settings.Modify("n_neuron2", std::to_string(hidden2_count));
        settings.Modify("activation_function", GENERATE("SIGMOID", "TANH", "RELU"));
        Potentials::PotMTP_M<Potentials::m_model_nn2> potential;
        potential.model.init(settings);
        std::istringstream input(neural_mtpm_potential_text(
            descriptor_count, model_coefficient_count));
        REQUIRE_NOTHROW(potential.Load(input));
        REQUIRE(potential.model_coeff_size == model_coefficient_count);
        REQUIRE(potential.get_NL_coeffs_count() == 48);
        validate_mtpm_cuda(
            potential, 4 * descriptor_count + 6 * hidden1_count +
                4 * hidden2_count);
    }

    SECTION("descriptor and hidden dimensions cross warp boundaries") {
        constexpr int descriptors = 9;
        constexpr int hidden1 = 35;
        constexpr int hidden2 = 17;
        constexpr int coefficients = descriptors * hidden1 + hidden1 +
            hidden1 * hidden2 + 2 * hidden2;
        GPSettings settings;
        settings.Modify("n_input", std::to_string(descriptors));
        settings.Modify("n_neuron", std::to_string(hidden1));
        settings.Modify("n_neuron2", std::to_string(hidden2));
        settings.Modify("activation_function", "TANH");
        Potentials::PotMTP_M<Potentials::m_model_nn2> potential;
        potential.model.init(settings);
        std::istringstream input(neural_mtpm_potential_text(descriptors, coefficients));
        REQUIRE_NOTHROW(potential.Load(input));
        validate_mtpm_cuda(potential, 4 * descriptors + 6 * hidden1 + 4 * hidden2);
    }

    SECTION("repeated scalar mappings cross a descriptor reduction warp") {
        constexpr int descriptors = 3;
        constexpr int hidden1 = 4;
        constexpr int hidden2 = 3;
        constexpr int coefficients = descriptors * hidden1 + hidden1 +
            hidden1 * hidden2 + 2 * hidden2;
        GPSettings settings;
        settings.Modify("n_input", std::to_string(descriptors));
        settings.Modify("n_neuron", std::to_string(hidden1));
        settings.Modify("n_neuron2", std::to_string(hidden2));
        settings.Modify("activation_function", "TANH");
        Potentials::PotMTP_M<Potentials::m_model_nn2> potential;
        potential.model.init(settings);
        std::istringstream input(neural_mtpm_potential_text(descriptors, coefficients, 35));
        REQUIRE_NOTHROW(potential.Load(input));
        validate_mtpm_cuda(potential, 4 * descriptors + 6 * hidden1 + 4 * hidden2);
    }
}

TEST_CASE("GPU MTP split radial bases match CPU EFS and full gradients",
          "[mtp][gpu][split-radial][gradient]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    std::istringstream input(split_radial_potential_text());
    PotMTP potential;
    REQUIRE_NOTHROW(potential.Load(input));
    REQUIRE(potential.radial_basis_split);
    REQUIRE(potential.p_radical_basis_pair.size() == 4);
    REQUIRE(potential.p_radical_basis_pair[0]->name == "RBChebyshev");
    REQUIRE(potential.p_radical_basis_pair[1]->name == "RBChebyCos");
    REQUIRE(potential.p_radical_basis_pair[2]->name == "RBChebyshevPoly5");

    Atoms source = split_radial_atoms();
    REQUIRE((source.pos_[1] - source.pos_[4]).norm() <
        potential.p_radical_basis_pair[2]->min_val);
    REQUIRE((source.pos_[1] - source.pos_[3]).norm() >
        potential.p_radical_basis_pair[3]->max_val);
    REQUIRE((source.pos_[1] - source.pos_[3]).norm() < potential.cutoff);
    Neighborhoods neighborhoods(source, potential.cutoff);
    Atoms cpu = source;
    potential.getEFS(cpu, neighborhoods);

    MTPGpuEvaluator evaluator;
    REQUIRE(evaluator.initialize(potential, 0, &error));
    INFO(error);
    MTPGpuBatch batch;
    REQUIRE(batch.build(source, neighborhoods, &error));
    Atoms gpu = source;
    REQUIRE(batch.evaluate(evaluator, gpu, false, &error));
    INFO(error);

    REQUIRE(gpu.energy_ == Catch::Approx(cpu.energy_).epsilon(2e-12).margin(2e-11));
    for (int atom = 0; atom < source.size(); ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            INFO(atom);
            INFO(axis);
            REQUIRE(gpu.forces_[atom][axis] ==
                Catch::Approx(cpu.forces_[atom][axis]).epsilon(3e-11).margin(3e-10));
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            INFO(row);
            INFO(column);
            REQUIRE(gpu.stress_(row, column) ==
                Catch::Approx(cpu.stress_(row, column)).epsilon(3e-11).margin(3e-10));
        }
    }
    validate_reduced_precision_inference(
        potential, source, neighborhoods, cpu, MTPGpuPrecision::Mixed);
    validate_reduced_precision_inference(
        potential, source, neighborhoods, cpu,
        MTPGpuPrecision::MixedAggressive);

    constexpr double denergy = 0.7;
    Eigen::Matrix<double, -1, 3> dforce(source.size(), 3);
    dforce << -0.03, 0.02, 0.01,
               0.04, -0.01, 0.03,
              -0.02, 0.05, -0.04,
               0.01, -0.03, 0.02,
               0.02, 0.01, -0.02;
    Eigen::Matrix3d dstress;
    dstress << 0.11, 0.03, -0.02,
               0.07, -0.07, 0.04,
              -0.01, 0.06, 0.05;

    potential.train_init();
    std::vector<double> cpu_gradient(potential.reg_coeff_size, 0.0);
    potential.AccumulateEFSCombinationGard(
        source, neighborhoods, denergy, dforce, dstress, cpu_gradient.data());

    std::vector<double> force_direction(3 * source.size());
    for (int atom = 0; atom < source.size(); ++atom)
        for (int axis = 0; axis < 3; ++axis)
            force_direction[3 * atom + axis] = dforce(atom, axis);
    std::array<double, 9> stress_direction{};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            stress_direction[3 * row + column] = dstress(row, column);
    std::vector<double> gpu_gradient(potential.reg_coeff_size, 0.0);
    REQUIRE(batch.accumulate_gradient(
        evaluator, {denergy}, force_direction,
        std::vector<double>(stress_direction.begin(), stress_direction.end()),
        gpu_gradient.data(), static_cast<int>(gpu_gradient.size()), false,
        &error));
    INFO(error);

    for (int parameter = 0; parameter < potential.reg_coeff_size; ++parameter) {
        INFO(parameter);
        REQUIRE(gpu_gradient[parameter] == Catch::Approx(
            cpu_gradient[parameter]).epsilon(4e-10).margin(2e-8));
    }

    std::vector<double> coefficients(
        potential.regression_coeffs,
        potential.regression_coeffs + potential.reg_coeff_size);
    constexpr double step = 2e-6;
    for (int parameter = 0; parameter < potential.reg_coeff_size; ++parameter) {
        const double original = coefficients[parameter];
        potential.regression_coeffs[parameter] = original + step;
        Atoms plus_atoms = source;
        potential.getEFS(plus_atoms, neighborhoods);
        const double plus = atom_efs_objective(
            plus_atoms, denergy, dforce, dstress);

        potential.regression_coeffs[parameter] = original - step;
        Atoms minus_atoms = source;
        potential.getEFS(minus_atoms, neighborhoods);
        const double minus = atom_efs_objective(
            minus_atoms, denergy, dforce, dstress);
        potential.regression_coeffs[parameter] = original;

        const double finite_difference = (plus - minus) / (2.0 * step);
        INFO(parameter);
        INFO(cpu_gradient[parameter]);
        INFO(finite_difference);
        REQUIRE(cpu_gradient[parameter] == Catch::Approx(
            finite_difference).epsilon(3e-6).margin(3e-5));
    }

    std::ostringstream saved;
    potential.Save(saved);
    std::istringstream reload_input(saved.str());
    PotMTP reloaded;
    REQUIRE_NOTHROW(reloaded.Load(reload_input));
    REQUIRE(reloaded.radial_basis_split);
    MTPGpuEvaluator reloaded_evaluator;
    REQUIRE(reloaded_evaluator.initialize(reloaded, 0, &error));
    Atoms reloaded_gpu = source;
    REQUIRE(batch.evaluate(reloaded_evaluator, reloaded_gpu, false, &error));
    REQUIRE(reloaded_gpu.energy_ == Catch::Approx(gpu.energy_).margin(2e-11));
}

TEST_CASE("GPU MTP contiguous batch API matches the LAMMPS API", "[mtp][gpu]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) || device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    MTPGpuDirect gpu;
    REQUIRE(gpu.set_device(0, &error));
    REQUIRE(gpu.initialize(small_potential_view(), &error));

    SmallSystem lammps_system;
    const double lammps_energy = evaluate(gpu, lammps_system, true);
    const auto lammps_force = lammps_system.force;
    const auto lammps_virial = lammps_system.virial;

    SmallSystem batch_system;
    const double batch_energy = evaluate_batch(gpu, batch_system, false);
    const std::uint64_t uploaded_generation = gpu.topology_generation();
    REQUIRE(batch_energy == Catch::Approx(lammps_energy).epsilon(1e-12));
    for (int i = 0; i < SmallSystem::n; ++i) {
        for (int a = 0; a < 3; ++a) {
            REQUIRE(batch_system.force[i][a] == Catch::Approx(lammps_force[i][a]).margin(1e-12));
        }
        for (int a = 0; a < 6; ++a) {
            REQUIRE(batch_system.virial[i][a] == Catch::Approx(lammps_virial[i][a]).margin(1e-12));
        }
    }

    // Exercise cached CSR indices while displacements and coordinates change.
    batch_system.x[1][0] += 0.01;
    const double reused_energy = evaluate_batch(gpu, batch_system, true);
    REQUIRE(gpu.topology_generation() == uploaded_generation);
    REQUIRE(reused_energy == Catch::Approx(reference_energy(
        batch_system, RadialBasis::Chebyshev)).epsilon(1e-11));

    const double energy_only = evaluate_batch(
        gpu, batch_system, true, false, false);
    REQUIRE(gpu.topology_generation() == uploaded_generation);
    REQUIRE(energy_only == Catch::Approx(reference_energy(
        batch_system, RadialBasis::Chebyshev)).epsilon(1e-11));
    double site_energy_sum = 0.0;
    for (double value : batch_system.site_energy) site_energy_sum += value;
    REQUIRE(site_energy_sum == Catch::Approx(energy_only).epsilon(1e-12));

    const double radial[] = {0.65, -0.1};
    const double species[] = {0.2};
    const double moments[] = {1.1, -0.25};
    REQUIRE(gpu.update_coefficients(radial, species, moments, &error));
    const double updated_energy = evaluate_batch(gpu, batch_system, true);
    REQUIRE(updated_energy == Catch::Approx(reference_energy(
        batch_system, RadialBasis::Chebyshev,
        radial[0], radial[1], species[0], moments[0], moments[1])).epsilon(1e-11));
}

TEST_CASE("GPU MTP compacts sparse declared moment slots", "[mtp][gpu][compact]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    MTPGpuDirect dense;
    MTPGpuDirect sparse;
    REQUIRE(dense.set_device(0, &error));
    REQUIRE(sparse.set_device(0, &error));
    REQUIRE(dense.initialize(small_potential_view(), &error));
    REQUIRE(sparse.initialize(sparse_slot_potential_view(), &error));

    const auto dense_stats = dense.potential_stats();
    const auto sparse_stats = sparse.potential_stats();
    REQUIRE(dense_stats.declared_moment_slots == 3);
    REQUIRE(dense_stats.active_moment_slots == 3);
    REQUIRE(sparse_stats.declared_moment_slots == 1024);
    REQUIRE(sparse_stats.active_moment_slots == 3);

    SmallSystem dense_system;
    SmallSystem sparse_system;
    const double dense_energy = evaluate_batch(dense, dense_system, false);
    const double sparse_energy = evaluate_batch(sparse, sparse_system, false);
    REQUIRE(sparse_energy == Catch::Approx(dense_energy).epsilon(1e-13));
    for (int atom = 0; atom < SmallSystem::n; ++atom) {
        for (int component = 0; component < 3; ++component) {
            REQUIRE(sparse_system.force[atom][component] == Catch::Approx(
                dense_system.force[atom][component]).margin(1e-13));
        }
        for (int component = 0; component < 6; ++component) {
            REQUIRE(sparse_system.virial[atom][component] == Catch::Approx(
                dense_system.virial[atom][component]).margin(1e-13));
        }
    }

    constexpr double denergy = 0.7;
    const std::array<double, SmallSystem::n * 3> dforce{{
        -0.03, 0.02, 0.01, 0.04, -0.01, 0.03,
        -0.02, 0.05, -0.04, 0.01, -0.03, 0.02,
    }};
    const std::array<double, 9> dstress{{
        0.11, 0.03, -0.02, 0.07, -0.07, 0.04, -0.01, 0.06, 0.05,
    }};
    std::array<int, SmallSystem::n> center_structure{};
    std::array<double, 5> dense_gradient{};
    std::array<double, 5> sparse_gradient{};
    MTPGpuGradientRequest request;
    request.structure_count = 1;
    request.center_structure = center_structure.data();
    request.dloss_denergy = &denergy;
    request.dloss_dforce = dforce.data();
    request.dloss_dstress = dstress.data();
    request.gradient_count = static_cast<int>(dense_gradient.size());

    request.gradient = dense_gradient.data();
    REQUIRE(dense.accumulate_gradient(request, &error));
    request.gradient = sparse_gradient.data();
    REQUIRE(sparse.accumulate_gradient(request, &error));
    for (size_t parameter = 0; parameter < dense_gradient.size(); ++parameter) {
        INFO(parameter);
        REQUIRE(sparse_gradient[parameter] ==
            Catch::Approx(dense_gradient[parameter]).margin(1e-12));
    }

    const double sparse_energy_only = evaluate_batch(
        sparse, sparse_system, true, true, false);
    REQUIRE(sparse_energy_only == Catch::Approx(dense_energy).epsilon(1e-13));
}

TEST_CASE("GPU MTP block-cooperates on extreme propagation groups",
          "[mtp][gpu][propagation][block]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    MTPGpuDirect fast;
    MTPGpuDirect ordered;
    REQUIRE(fast.set_device(0, &error));
    REQUIRE(ordered.set_device(0, &error));
    REQUIRE(fast.initialize(long_propagation_group_potential_view(), &error));
    REQUIRE(ordered.initialize(long_propagation_group_potential_view(), &error));
    const auto stats = fast.potential_stats();
    REQUIRE(stats.forward_block_groups == 1);
    REQUIRE(stats.reverse_block_groups == 1);

    SmallSystem fast_system;
    SmallSystem ordered_system;
    const double fast_energy = evaluate_batch(fast, fast_system, false, false);
    const double ordered_energy =
        evaluate_batch(ordered, ordered_system, false, true);
    REQUIRE(fast_energy == Catch::Approx(ordered_energy).epsilon(2e-12));
    for (int atom = 0; atom < SmallSystem::n; ++atom) {
        for (int component = 0; component < 3; ++component) {
            REQUIRE(fast_system.force[atom][component] == Catch::Approx(
                ordered_system.force[atom][component]).epsilon(2e-11).margin(1e-10));
        }
        for (int component = 0; component < 6; ++component) {
            REQUIRE(fast_system.virial[atom][component] == Catch::Approx(
                ordered_system.virial[atom][component]).epsilon(2e-11).margin(1e-10));
        }
    }

    constexpr double denergy = 0.7;
    const std::array<double, SmallSystem::n * 3> dforce{{
        -0.03, 0.02, 0.01, 0.04, -0.01, 0.03,
        -0.02, 0.05, -0.04, 0.01, -0.03, 0.02,
    }};
    const std::array<double, 9> dstress{{
        0.11, 0.03, -0.02, 0.07, -0.07, 0.04, -0.01, 0.06, 0.05,
    }};
    std::array<int, SmallSystem::n> center_structure{};
    std::array<double, 5> fast_gradient{};
    std::array<double, 5> ordered_gradient{};
    MTPGpuGradientRequest request;
    request.structure_count = 1;
    request.center_structure = center_structure.data();
    request.dloss_denergy = &denergy;
    request.dloss_dforce = dforce.data();
    request.dloss_dstress = dstress.data();
    request.gradient_count = static_cast<int>(fast_gradient.size());
    request.gradient = fast_gradient.data();
    REQUIRE(fast.accumulate_gradient(request, &error));
    request.deterministic_reduction = true;
    request.gradient = ordered_gradient.data();
    REQUIRE(ordered.accumulate_gradient(request, &error));
    for (size_t parameter = 0; parameter < fast_gradient.size(); ++parameter) {
        INFO(parameter);
        REQUIRE(fast_gradient[parameter] == Catch::Approx(
            ordered_gradient[parameter]).epsilon(3e-11).margin(1e-9));
    }

    const double fast_energy_only =
        evaluate_batch(fast, fast_system, true, false, false);
    const double ordered_energy_only =
        evaluate_batch(ordered, ordered_system, true, true, false);
    REQUIRE(fast_energy_only ==
        Catch::Approx(ordered_energy_only).epsilon(2e-12));
}

TEST_CASE("GPU MTP isolates logical subgroups across propagation boundaries",
          "[mtp][gpu][propagation][subgroup]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    const int term_count = GENERATE(31, 32, 33, 63, 64, 65, 511, 512, 513);
    const int threads = GENERATE(16, 32, 64, 128, 256);
    CAPTURE(term_count, threads);
    // Adjacent groups have different operands and coefficients. A wave64
    // shuffle accidentally using width=64 contaminates the first group.
    std::vector<int> times;
    for (int group = 0; group < 2; ++group) {
        for (int term = 0; term < term_count; ++term) {
            times.insert(times.end(), {group, 1, group + 1, group + 2});
        }
    }
    const int mapping[] = {2, 3};
    const double moments[] = {0.7 / term_count, -0.2 / term_count};
    auto view = small_potential_view();
    view.alpha_moments_count = 4;
    view.alpha_index_times_count = term_count * 2;
    view.alpha_index_times = times.data();
    view.alpha_moment_mapping = mapping;
    view.moment_coeffs = moments;
    MTPGpuDirect fast;
    MTPGpuDirect ordered;
    REQUIRE(fast.set_device(0, &error));
    REQUIRE(ordered.set_device(0, &error));
    REQUIRE(fast.initialize(view, &error));
    REQUIRE(ordered.initialize(view, &error));
    const auto stats = fast.potential_stats();
    REQUIRE(stats.forward_block_groups == (term_count > 512 ? 2 : 0));

    SmallSystem fast_system;
    SmallSystem ordered_system;
    const double fast_energy = evaluate_batch(fast, fast_system, false, false, true, threads);
    const double ordered_energy =
        evaluate_batch(ordered, ordered_system, false, true, true, threads);
    REQUIRE(fast_energy == Catch::Approx(ordered_energy).epsilon(2e-12));
    for (int atom = 0; atom < SmallSystem::n; ++atom) {
        for (int component = 0; component < 3; ++component) {
            REQUIRE(fast_system.force[atom][component] == Catch::Approx(
                ordered_system.force[atom][component]).epsilon(2e-11).margin(1e-10));
        }
        for (int component = 0; component < 6; ++component) {
            REQUIRE(fast_system.virial[atom][component] == Catch::Approx(
                ordered_system.virial[atom][component]).epsilon(2e-11).margin(1e-10));
        }
    }

    constexpr double denergy = 0.7;
    const std::array<double, SmallSystem::n * 3> dforce{{
        -0.03, 0.02, 0.01, 0.04, -0.01, 0.03,
        -0.02, 0.05, -0.04, 0.01, -0.03, 0.02,
    }};
    const std::array<double, 9> dstress{{
        0.11, 0.03, -0.02, 0.07, -0.07, 0.04, -0.01, 0.06, 0.05,
    }};
    std::array<int, SmallSystem::n> center_structure{};
    std::array<double, 5> fast_gradient{};
    std::array<double, 5> ordered_gradient{};
    MTPGpuGradientRequest request;
    request.propagate_threads = threads;
    request.structure_count = 1;
    request.center_structure = center_structure.data();
    request.dloss_denergy = &denergy;
    request.dloss_dforce = dforce.data();
    request.dloss_dstress = dstress.data();
    request.gradient_count = static_cast<int>(fast_gradient.size());
    request.gradient = fast_gradient.data();
    REQUIRE(fast.accumulate_gradient(request, &error));
    request.deterministic_reduction = true;
    request.gradient = ordered_gradient.data();
    REQUIRE(ordered.accumulate_gradient(request, &error));
    for (size_t parameter = 0; parameter < fast_gradient.size(); ++parameter) {
        INFO(parameter);
        REQUIRE(fast_gradient[parameter] == Catch::Approx(
            ordered_gradient[parameter]).epsilon(3e-11).margin(1e-9));
    }

    const double fast_energy_only =
        evaluate_batch(fast, fast_system, true, false, false, threads);
    const double ordered_energy_only =
        evaluate_batch(ordered, ordered_system, true, true, false, threads);
    REQUIRE(fast_energy_only ==
        Catch::Approx(ordered_energy_only).epsilon(2e-12));
}

TEST_CASE("GPU MTP handles interleaved propagation tiers and dependent levels",
          "[mtp][gpu][propagation][tiers]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) ||
        device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr)
            FAIL("Required GPU is unavailable: " << error);
        SKIP("No usable GPU device: " << error);
    }

    const int threads = GENERATE(1, 16, 32, 64, 128, 256);
    CAPTURE(threads);
    // Adjacent destinations alternate execution tiers, with a partial batch
    // of long groups and a second level reading outputs from different tiers.
    const std::array<int, 7> counts{{33, 1, 513, 32, 512, 31, 511}};
    std::vector<int> times, mapping;
    std::vector<double> coefficients;
    for (int group = 0; group < static_cast<int>(counts.size()); ++group) {
        for (int term = 0; term < counts[group]; ++term)
            times.insert(times.end(), {group % 2, 1, 1, group + 2});
        mapping.push_back(group + 2);
        coefficients.push_back((group % 2 ? -0.2 : 0.3) / counts[group]);
    }
    for (int term = 0; term < 33; ++term)
        times.insert(times.end(), {2, 4, 1, 9});
    mapping.push_back(9);
    coefficients.push_back(1e-6);
    auto view = small_potential_view();
    view.alpha_moments_count = 10;
    view.alpha_index_times_count = static_cast<int>(times.size() / 4);
    view.alpha_index_times = times.data();
    view.alpha_scalar_moments = static_cast<int>(mapping.size());
    view.alpha_moment_mapping = mapping.data();
    view.moment_coeffs = coefficients.data();
    MTPGpuDirect fast, ordered;
    REQUIRE(fast.set_device(0, &error));
    REQUIRE(ordered.set_device(0, &error));
    REQUIRE(fast.initialize(view, &error));
    REQUIRE(ordered.initialize(view, &error));
    SmallSystem actual, reference;
    const double energy = evaluate_batch(fast, actual, false, false, true, threads);
    const double expected = evaluate_batch(ordered, reference, false, true, true, threads);
    REQUIRE(energy == Catch::Approx(expected).epsilon(2e-12));
    for (int atom = 0; atom < SmallSystem::n; ++atom) {
        for (int axis = 0; axis < 3; ++axis)
            REQUIRE(actual.force[atom][axis] == Catch::Approx(reference.force[atom][axis])
                .epsilon(2e-11).margin(1e-10));
        for (int axis = 0; axis < 6; ++axis)
            REQUIRE(actual.virial[atom][axis] == Catch::Approx(reference.virial[atom][axis])
                .epsilon(2e-11).margin(1e-10));
    }
    REQUIRE(evaluate_batch(fast, actual, true, false, false, threads) ==
            Catch::Approx(expected).epsilon(2e-12));
}

TEST_CASE("GPU MTP training gradient matches coefficient finite differences",
          "[mtp][gpu][gradient]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) || device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    MTPGpuDirect gpu;
    REQUIRE(gpu.set_device(0, &error));
    REQUIRE(gpu.initialize(small_potential_view(), &error));
    SmallSystem system;

    constexpr double denergy = 0.7;
    const std::array<double, SmallSystem::n * 3> dforce{{
        -0.03, 0.02, 0.01,
         0.04, -0.01, 0.03,
        -0.02, 0.05, -0.04,
         0.01, -0.03, 0.02,
    }};
    const std::array<double, 9> dstress{{
        0.11, 0.03, -0.02,
        0.07, -0.07, 0.04,
        -0.01, 0.06, 0.05,
    }};
    const double base_energy = evaluate_batch(gpu, system, false);

    std::array<int, SmallSystem::n> center_structure{};
    std::array<double, 5> analytic{};
    MTPGpuGradientRequest grad_req;
    grad_req.structure_count = 1;
    grad_req.center_structure = center_structure.data();
    grad_req.dloss_denergy = &denergy;
    grad_req.dloss_dforce = dforce.data();
    grad_req.dloss_dstress = dstress.data();
    grad_req.gradient = analytic.data();
    grad_req.gradient_count = static_cast<int>(analytic.size());
    REQUIRE(gpu.accumulate_gradient(grad_req, &error));
    INFO(error);

    std::array<double, 2> radial{{0.7, -0.2}};
    std::array<double, 1> species{{0.3}};
    std::array<double, 2> moments{{1.2, -0.4}};
    constexpr double eps = 2e-6;
    for (int parameter = 0; parameter < 5; ++parameter) {
        double *value = parameter < 2 ? &radial[parameter]
            : parameter == 2 ? &species[0] : &moments[parameter - 3];
        const double original = *value;
        *value = original + eps;
        REQUIRE(gpu.update_coefficients(radial.data(), species.data(), moments.data(), &error));
        const double plus_energy = evaluate_batch(gpu, system, true);
        const double plus = training_objective(
            plus_energy, system, denergy, dforce, dstress);
        *value = original - eps;
        REQUIRE(gpu.update_coefficients(radial.data(), species.data(), moments.data(), &error));
        const double minus_energy = evaluate_batch(gpu, system, true);
        const double minus = training_objective(
            minus_energy, system, denergy, dforce, dstress);
        *value = original;
        const double finite = (plus - minus) / (2.0 * eps);
        INFO(parameter);
        INFO(analytic[parameter]);
        INFO(finite);
        REQUIRE(analytic[parameter] == Catch::Approx(finite).margin(3e-5));
    }

    REQUIRE(gpu.update_coefficients(radial.data(), species.data(), moments.data(), &error));
    REQUIRE(evaluate_batch(gpu, system, true) == Catch::Approx(base_energy).epsilon(1e-12));
    std::array<double, 5> accumulated;
    accumulated.fill(0.25);
    grad_req.gradient = accumulated.data();
    grad_req.accumulate_gradient = true;
    REQUIRE(gpu.accumulate_gradient(grad_req, &error));
    for (int i = 0; i < 5; ++i) {
        REQUIRE(accumulated[i] == Catch::Approx(analytic[i] + 0.25).margin(1e-11));
    }
}

TEST_CASE("GPU MTP radial gradient maps compact two-species partial rows",
          "[mtp][gpu][gradient]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) || device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    MTPGpuDirect gpu;
    REQUIRE(gpu.set_device(0, &error));
    REQUIRE(gpu.initialize(two_species_potential_view(), &error));
    SmallSystem system;
    system.type = {{1, 2, 1, 2}};

    constexpr double denergy = 0.7;
    const std::array<double, SmallSystem::n * 3> dforce{{
        -0.03, 0.02, 0.01,
         0.04, -0.01, 0.03,
        -0.02, 0.05, -0.04,
         0.01, -0.03, 0.02,
    }};
    const std::array<double, 9> dstress{{
        0.11, 0.03, -0.02,
        0.07, -0.07, 0.04,
        -0.01, 0.06, 0.05,
    }};
    std::array<int, SmallSystem::n> center_structure{};
    std::array<double, 12> tiled_gradient{};

    evaluate_batch(gpu, system, false);
    MTPGpuGradientRequest request;
    request.structure_count = 1;
    request.center_structure = center_structure.data();
    request.dloss_denergy = &denergy;
    request.dloss_dforce = dforce.data();
    request.dloss_dstress = dstress.data();
    request.gradient = tiled_gradient.data();
    request.gradient_count = static_cast<int>(tiled_gradient.size());
    request.radial_parameter_tile = 2;
    REQUIRE(gpu.accumulate_gradient(request, &error));
    INFO(error);

    evaluate_batch(gpu, system, true);
    std::array<double, 12> single_tile_gradient{};
    request.gradient = single_tile_gradient.data();
    request.radial_parameter_tile = 64;
    REQUIRE(gpu.accumulate_gradient(request, &error));
    for (size_t parameter = 0; parameter < tiled_gradient.size(); ++parameter) {
        INFO(parameter);
        REQUIRE(single_tile_gradient[parameter] ==
            Catch::Approx(tiled_gradient[parameter]).margin(1e-10));
    }

    std::array<double, 8> radial{{
         0.70, -0.20,  0.45,  0.10,
        -0.35,  0.25,  0.80, -0.15,
    }};
    const std::array<double, 2> species{{0.3, -0.1}};
    const std::array<double, 2> moments{{1.2, -0.4}};
    constexpr double eps = 2e-6;
    for (int parameter = 0; parameter < static_cast<int>(radial.size()); ++parameter) {
        const double original = radial[parameter];
        radial[parameter] = original + eps;
        REQUIRE(gpu.update_coefficients(
            radial.data(), species.data(), moments.data(), &error));
        const double plus_energy = evaluate_batch(gpu, system, true);
        const double plus = training_objective(
            plus_energy, system, denergy, dforce, dstress);
        radial[parameter] = original - eps;
        REQUIRE(gpu.update_coefficients(
            radial.data(), species.data(), moments.data(), &error));
        const double minus_energy = evaluate_batch(gpu, system, true);
        const double minus = training_objective(
            minus_energy, system, denergy, dforce, dstress);
        radial[parameter] = original;
        const double finite = (plus - minus) / (2.0 * eps);
        INFO(parameter);
        INFO(tiled_gradient[parameter]);
        INFO(finite);
        REQUIRE(tiled_gradient[parameter] == Catch::Approx(finite).margin(3e-5));
    }
}

TEST_CASE("GPU MTP constructs supervised loss and chunk gradient on device",
          "[mtp][gpu][gradient][loss]" IMRMLP_GPU_TEST_TAG)
{
    int device_count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&device_count, &error) || device_count == 0) {
        if (std::getenv("IMRMLP_REQUIRE_GPU") != nullptr) {
            FAIL("Required GPU is unavailable: " << error);
        }
        SKIP("No usable GPU device: " << error);
    }

    MTPGpuDirect gpu;
    REQUIRE(gpu.set_device(0, &error));
    REQUIRE(gpu.initialize(small_potential_view(), &error));
    SmallSystem system;
    const double energy = evaluate_batch(gpu, system, false);

    std::array<int, 2> structure_offsets{{0, SmallSystem::n}};
    std::array<int, SmallSystem::n> center_structure{};
    const std::array<double, 1> target_energy{{energy - 0.25}};
    const std::array<double, 1> energy_weight{{1.3}};
    std::array<double, SmallSystem::n * 3> target_force{};
    const std::array<double, SmallSystem::n> force_weight{{0.2, 0.3, 0.4, 0.5}};
    const std::array<double, SmallSystem::n * 3> force_error{{
        -0.03, 0.02, 0.01,
         0.04, -0.01, 0.03,
        -0.02, 0.05, -0.04,
         0.01, -0.03, 0.02,
    }};
    std::array<double, 9> prediction_stress{};
    for (int atom = 0; atom < SmallSystem::n; ++atom) {
        prediction_stress[0] += system.virial[atom][0];
        prediction_stress[4] += system.virial[atom][1];
        prediction_stress[8] += system.virial[atom][2];
        prediction_stress[1] += system.virial[atom][3];
        prediction_stress[2] += system.virial[atom][4];
        prediction_stress[5] += system.virial[atom][5];
    }
    prediction_stress[3] = prediction_stress[1];
    prediction_stress[6] = prediction_stress[2];
    prediction_stress[7] = prediction_stress[5];
    const std::array<double, 9> stress_error{{
        0.11, 0.03, -0.02,
        0.07, -0.07, 0.04,
        -0.01, 0.06, 0.05,
    }};
    std::array<double, 9> target_stress{};
    for (int atom = 0; atom < SmallSystem::n; ++atom) {
        for (int a = 0; a < 3; ++a) {
            const int component = 3 * atom + a;
            target_force[component] =
                system.force[atom][a] - force_error[component];
        }
    }
    for (int component = 0; component < 9; ++component) {
        target_stress[component] =
            prediction_stress[component] - stress_error[component];
    }
    const std::array<double, 1> stress_weight{{0.7}};

    std::array<double, MTPGpuLossRequest::metric_count> metrics{};
    MTPGpuLossRequest loss_request;
    loss_request.structure_count = 1;
    loss_request.structure_offsets = structure_offsets.data();
    loss_request.center_structure = center_structure.data();
    loss_request.target_energy = target_energy.data();
    loss_request.energy_weight = energy_weight.data();
    loss_request.target_force = target_force.data();
    loss_request.force_weight = force_weight.data();
    loss_request.target_stress = target_stress.data();
    loss_request.stress_weight = stress_weight.data();
    loss_request.metrics = metrics.data();
    REQUIRE(gpu.prepare_loss(loss_request, &error));
    INFO(error);

    double expected_force_loss = 0.0;
    double expected_force_squared = 0.0;
    std::array<double, SmallSystem::n * 3> force_adjoint{};
    for (int atom = 0; atom < SmallSystem::n; ++atom) {
        double atom_squared = 0.0;
        for (int a = 0; a < 3; ++a) {
            const int component = 3 * atom + a;
            atom_squared += force_error[component] * force_error[component];
            force_adjoint[component] =
                2.0 * force_weight[atom] * force_error[component];
        }
        expected_force_loss += force_weight[atom] * atom_squared;
        expected_force_squared += atom_squared;
    }
    double expected_stress_squared = 0.0;
    std::array<double, 9> stress_adjoint{};
    for (int component = 0; component < 9; ++component) {
        expected_stress_squared += stress_error[component] * stress_error[component];
        stress_adjoint[component] =
            2.0 * stress_weight[0] * stress_error[component];
    }
    const double expected_energy_squared = 0.25 * 0.25;
    const double expected_energy_loss = energy_weight[0] * expected_energy_squared;
    const double expected_stress_loss = stress_weight[0] * expected_stress_squared;
    REQUIRE(metrics[0] == Catch::Approx(
        expected_energy_loss + expected_force_loss + expected_stress_loss).margin(1e-13));
    REQUIRE(metrics[1] == Catch::Approx(expected_energy_loss).margin(1e-13));
    REQUIRE(metrics[2] == Catch::Approx(expected_force_loss).margin(1e-13));
    REQUIRE(metrics[3] == Catch::Approx(expected_stress_loss).margin(1e-13));
    REQUIRE(metrics[4] == Catch::Approx(
        expected_energy_squared / (SmallSystem::n * SmallSystem::n)).margin(1e-13));
    REQUIRE(metrics[5] == Catch::Approx(expected_force_squared).margin(1e-13));
    REQUIRE(metrics[6] == Catch::Approx(expected_stress_squared).margin(1e-13));

    std::array<double, 5> device_gradient{};
    MTPGpuGradientRequest device_request;
    device_request.structure_count = 1;
    device_request.gradient = device_gradient.data();
    device_request.gradient_count = static_cast<int>(device_gradient.size());
    device_request.accumulate_gradient_on_device = true;
    device_request.copy_gradient_to_host = false;
    device_request.adjoints_on_device = true;
    REQUIRE(gpu.accumulate_gradient(device_request, &error));

    // Re-evaluate the same chunk and retain the first chunk's gradient on the
    // device. Only this final call copies the two-chunk sum to the host.
    REQUIRE(evaluate_batch(gpu, system, true) ==
            Catch::Approx(energy).epsilon(1e-12));
    REQUIRE(gpu.prepare_loss(loss_request, &error));
    device_request.accumulate_gradient = true;
    device_request.copy_gradient_to_host = true;
    REQUIRE(gpu.accumulate_gradient(device_request, &error));

    const std::array<double, 1> energy_adjoint{{
        2.0 * energy_weight[0] * (energy - target_energy[0])}};
    std::array<double, 5> host_gradient{};
    MTPGpuGradientRequest host_request;
    host_request.structure_count = 1;
    host_request.center_structure = center_structure.data();
    host_request.dloss_denergy = energy_adjoint.data();
    host_request.dloss_dforce = force_adjoint.data();
    host_request.dloss_dstress = stress_adjoint.data();
    host_request.gradient = host_gradient.data();
    host_request.gradient_count = static_cast<int>(host_gradient.size());
    REQUIRE(gpu.accumulate_gradient(host_request, &error));
    for (size_t parameter = 0; parameter < host_gradient.size(); ++parameter) {
        INFO(parameter);
        REQUIRE(device_gradient[parameter] ==
            Catch::Approx(2.0 * host_gradient[parameter]).margin(1e-11));
    }

    REQUIRE(evaluate_batch(gpu, system, true, true) ==
            Catch::Approx(energy).epsilon(1e-12));
    loss_request.deterministic_reduction = true;
    REQUIRE(gpu.prepare_loss(loss_request, &error));
    std::array<double, 5> ordered_gradient{};
    MTPGpuGradientRequest ordered_request;
    ordered_request.structure_count = 1;
    ordered_request.gradient = ordered_gradient.data();
    ordered_request.gradient_count = static_cast<int>(ordered_gradient.size());
    ordered_request.adjoints_on_device = true;
    ordered_request.deterministic_reduction = true;
    REQUIRE(gpu.accumulate_gradient(ordered_request, &error));
    for (size_t parameter = 0; parameter < host_gradient.size(); ++parameter) {
        INFO(parameter);
        REQUIRE(ordered_gradient[parameter] ==
            Catch::Approx(host_gradient[parameter]).margin(1e-11));
    }

    const auto first_ordered_metrics = metrics;
    REQUIRE(evaluate_batch(gpu, system, true, true) ==
            Catch::Approx(energy).epsilon(1e-12));
    std::array<double, MTPGpuLossRequest::metric_count> repeated_metrics{};
    loss_request.metrics = repeated_metrics.data();
    REQUIRE(gpu.prepare_loss(loss_request, &error));
    std::array<double, 5> repeated_gradient{};
    ordered_request.gradient = repeated_gradient.data();
    REQUIRE(gpu.accumulate_gradient(ordered_request, &error));
    REQUIRE(repeated_metrics == first_ordered_metrics);
    REQUIRE(repeated_gradient == ordered_gradient);
}

#endif
