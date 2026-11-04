// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potentials/potential_mtp.h"
#include "base/neighborhood.h"
#include <cmath>
#include <sstream>
#include <stdexcept>

// Versioned, synthetic inputs shared by the CPU oracle, device tests and
// benchmark. No external model or private trajectory is required.
namespace MTPLinearEnergyFixture
{
inline constexpr const char *version = "mtp-linear-energy-v1";
inline constexpr double absolute_tolerance = 2e-9;
inline constexpr double relative_tolerance = 2e-10;

inline bool close(double actual, double expected)
{
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(actual - expected) <=
               absolute_tolerance + relative_tolerance * std::abs(expected);
}

inline void load(Potentials::PotMTP &potential, int species = 2, bool sparse = false)
{
    std::ostringstream text;
    text << "MTP\nversion = 1.1.0\npotential_name = MTP1m\nscaling = 1\nspecies_count = " << species
         << "\npotential_tag =\n";
    if (species == 2)
        text << "radial_basis_split = 1\n";
    for (int a = 0; a < species; ++a)
        for (int b = 0; b < species; ++b)
        {
            if (species == 2)
                text << "pair " << a << ' ' << b << '\n';
            text << "radial_basis_type = " << (a != b ? "RBChebyCos" : "RBChebyshev")
                 << "\n  min_dist = 0.5\n  max_dist = " << (a == b ? 4.0 : 5.0)
                 << "\n  radial_basis_size = 2\n";
        }
    text << "radial_funcs_count = 1\nradial_coeffs\n";
    for (int a = 0; a < species; ++a)
        for (int b = 0; b < species; ++b)
            text << "  " << a << '-' << b << "\n    {" << .7 - .15 * a - .1 * b << ", -0.20}\n";
    const int product = sparse ? 8 : 4;
    text << "alpha_moments_count = " << product + 1 << "\nalpha_index_basic_count = 4\n"
         << "alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}\n"
         << "alpha_index_times_count = 3\nalpha_index_times = {{1, 1, 1, " << product
         << "}, {2, 2, 1, " << product << "}, {3, 3, 1, " << product << "}}\n"
         << "alpha_scalar_moments = " << (sparse ? 3 : 2)
         << "\nalpha_moment_mapping = " << (sparse ? "{8, 0, 8}" : "{0, 4}")
         << "\nspecies_coeffs = " << (species == 2 ? "{0.3, -0.1}" : "{0.3}")
         << "\nmoment_coeffs = " << (sparse ? "{1.2, -0.4, 0.2}" : "{1.2, -0.4}") << '\n';
    std::istringstream input(text.str());
    potential.Load(input);
}

inline Atoms frame(int index, int species = 2, int side = 2)
{
    Atoms atoms;
    const double length = 2.6 * side;
    atoms.cell_ << length, 0., 0., .17, length, 0., -.13, .21, length;
    for (int x = 0; x < side; ++x)
        for (int y = 0; y < side; ++y)
            for (int z = 0; z < side; ++z)
            {
                atoms.pos_.push_back({.11 + 2.6 * x + .03 * std::sin(index + x),
                                      .19 + 2.6 * y + .04 * std::cos(index + z), .07 + 2.6 * z});
                atoms.chemmap_.push_back((x + y + z + index) % species);
                atoms.atomic_fitness.push_back(.2 + (x + y + z) % 3);
            }
    atoms.fitness = .3 + .1 * (index % 7);
    return atoms;
}
} // namespace MTPLinearEnergyFixture
