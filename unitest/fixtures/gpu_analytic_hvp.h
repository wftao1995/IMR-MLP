// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_hessian.h"
#include <array>

// G4c1 fixed inputs and gates. This fixture performs no device work itself.
namespace GpuAnalyticHvpFixture
{
namespace F = GpuHessianFixture;
inline constexpr const char *version = "gpu-analytic-hvp-v1";
inline constexpr double cpu_tolerance = 2e-10;
inline constexpr double gpu_atol = 2e-8, gpu_rtol = 2e-8;
inline constexpr std::array<double, 4> steps{{1e-3, 5e-4, 2e-4, 1e-4}};
inline constexpr std::array<const char *, 7> radial_names{
    {"RBChebyshev", "RBChebyCos", "RBChebyshevPoly4", "RBChebyshevPoly5", "RBPolyB", "RBPolyBCos",
     "RBChebyshevConfigurable"}};

inline GPSettings settings(const std::string &device = "cpu", double step = 1e-4, int atoms = 64,
                           int memory = 64, int host = 16)
{
    auto result = F::settings(device, step, atoms, 4, memory, host);
    result.Modify("method", "analytic");
    return result;
}

inline Eigen::VectorXd apply(Potentials::PotMTP &model, const Atoms &input,
                             const Eigen::VectorXd &direction)
{
    ComputeHessian query(settings());
    Eigen::VectorXd result;
    query.compute_hvp(input, direction, result, &model);
    return result;
}

inline std::string radial_block(const std::string &name, const std::string &cut = "cosine_left",
                                const std::string &shift = "exponential", bool flipped = false)
{
    std::ostringstream out;
    out << "radial_basis_type = " << name
        << "\nmin_dist = 0.5\nmax_dist = 5\nradial_basis_size = 4\n";
    if (name == "RBChebyshevPoly4" || name == "RBChebyshevPoly5")
        out << "rb_b = -0.7\n";
    if (name == "RBChebyshevConfigurable")
    {
        out << "basis_function_type = " << (flipped ? "one_minus_chebyshev" : "chebyshev")
            << "\ncutoff_type = " << cut << "\n";
        if (cut == "poly3" || cut == "poly4" || cut == "poly5")
            out << "cutoff_b = -0.7\n";
        if (cut == "smoothstep7")
            out << "cutoff_r1 = 1.4\ncutoff_r2 = 3.7\n";
        out << "shift_type = " << shift << "\n";
        if (shift != "linear" && shift != "i2")
            out << "shift_lambda = 0.7\n";
        if (shift == "exponential_tanh")
            out << "shift_scale_lambda = 0.8\n";
    }
    return out.str();
}

inline void radial_model(Potentials::PotMTP &model, const std::string &block)
{
    std::ostringstream out;
    out << "MTP\nversion = 1.1.0\npotential_name = analytic-hvp-fixture\nscaling = 0.65\nspecies_count = 1\npotential_tag =\n"
        << block << "radial_funcs_count = 1\nradial_coeffs\n0-0\n{0.7,-0.2,0.12,-0.05}\n"
        << "alpha_moments_count = 5\nalpha_index_basic_count = 4\n"
        << "alpha_index_basic = {{0,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}\n"
        << "alpha_index_times_count = 3\nalpha_index_times = {{1,1,1,4},{2,2,1,4},{3,3,1,4}}\n"
        << "alpha_scalar_moments = 2\nalpha_moment_mapping = {0,4}\n"
        << "species_coeffs = {0.3}\nmoment_coeffs = {0.12,-0.04}\n";
    std::istringstream input(out.str());
    model.Load(input);
    model.omp_init();
    model.set_inited();
}

inline Atoms single_species(int count = 3)
{
    auto input = F::cluster(count);
    std::fill(input.chemmap_.begin(), input.chemmap_.end(), 0);
    return input;
}
} // namespace GpuAnalyticHvpFixture
