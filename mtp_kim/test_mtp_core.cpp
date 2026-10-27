// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mtp_kim_core.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string test_model()
{
    // Pair sections are intentionally out of lexicographic order. Atom 2 is
    // species 1, which detects accidentally using its atom index as a species
    // index in computeEFS.
    return
        "MTP\n"
        "version = 1.1.0\n"
        "potential_name = mtp_kim_test\n"
        "scaling = 0.2\n"
        "species_count = 2\n"
        "potential_tag =\n"
        "radial_basis_split = 1\n"
        "pair 1 1\n"
        "radial_basis_type = RBChebyshev\n"
        "min_dist = 0.4\nmax_dist = 4.0\nradial_basis_size = 2\n"
        "pair 1 0\n"
        "radial_basis_type = RBChebyCos\n"
        "min_dist = 0.4\nmax_dist = 3.8\nradial_basis_size = 2\n"
        "pair 0 1\n"
        "radial_basis_type = RBChebyshevPoly5\n"
        "min_dist = 0.5\nmax_dist = 4.2\nradial_basis_size = 2\n"
        "rb_b = -0.3\n"
        "pair 0 0\n"
        "radial_basis_type = RBChebyshevConfigurable\n"
        "min_dist = 0.5\nmax_dist = 3.6\nradial_basis_size = 2\n"
        "cutoff_type = cosine\nshift_type = linear\n"
        "radial_funcs_count = 1\n"
        "radial_coeffs\n"
        "1-0\n{1.1, -0.2}\n"
        "0-1\n{0.7, 0.4}\n"
        "0-0\n{1.3, -0.1}\n"
        "1-1\n{0.5, 0.9}\n"
        "alpha_moments_count = 1\n"
        "alpha_index_basic_count = 1\n"
        "alpha_index_basic = {{0, 0, 0, 0}}\n"
        "alpha_index_times_count = 0\n"
        "alpha_index_times = {}\n"
        "alpha_scalar_moments = 1\n"
        "alpha_moment_mapping = {0}\n"
        "species_coeffs = {0.25, -0.4}\n"
        "moment_coeffs = {1.7}\n";
}

bool close(double actual, double expected, double tolerance = 1e-10)
{
    return std::abs(actual - expected) <= tolerance *
        std::max({1.0, std::abs(actual), std::abs(expected)});
}

double reference_energy(const mtp_kim::MTPCore &core,
                        const std::vector<int> &species,
                        const std::vector<double> &coordinates)
{
    double energy = 0.0;
    std::vector<double> values(core.basisSize());
    std::vector<double> derivatives(core.basisSize());
    for (int i = 0; i < static_cast<int>(species.size()); ++i) {
        energy += core.species_coeffs[species[i]];
        for (int j = 0; j < static_cast<int>(species.size()); ++j) {
            if (i == j) continue;
            const double dx = coordinates[3*j] - coordinates[3*i];
            const double dy = coordinates[3*j+1] - coordinates[3*i+1];
            const double dz = coordinates[3*j+2] - coordinates[3*i+2];
            const double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            const int pair = species[i] * core.speciesCount() + species[j];
            if (distance <= core.r_mins[pair] || distance >= core.cutoffs[pair])
                continue;
            core.evalRadial(species[i], species[j], distance,
                            values.data(), derivatives.data());
            const int offset = pair * core.radialFuncCount() * core.basisSize();
            double radial = 0.0;
            for (int xi = 0; xi < core.basisSize(); ++xi)
                radial += core.radial_coeffs[offset + xi] * values[xi];
            energy += core.moment_coeffs[0] * radial;
        }
    }
    return energy;
}

int fail(const std::string &message)
{
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main()
{
    mtp_kim::MTPCore core;
    std::istringstream input(test_model());
    if (!core.load(input)) return fail("could not load valid split-basis model");
    if (core.speciesCount() != 2 || core.basisSize() != 2 ||
        core.radialFuncCount() != 1 || !close(core.maxCutoff(), 4.2))
        return fail("loaded metadata is incorrect");

    const std::vector<int> species = {0, 0, 1};
    const std::vector<int> contributing = {1, 1, 1};
    std::vector<double> coordinates = {
        0.0, 0.0, 0.0,
        1.2, 0.1, 0.0,
        2.4, 0.3, 0.2,
    };
    double energy = 0.0;
    double forces[9] = {};
    double virial[6] = {};
    core.computeEFS(static_cast<int>(species.size()), species.data(),
                    contributing.data(), coordinates.data(), energy,
                    forces, virial);

    const double expected = reference_energy(core, species, coordinates);
    if (!close(energy, expected))
        return fail("energy does not use the requested species-pair bases");

    for (int component = 0; component < 9; ++component) {
        constexpr double step = 1e-6;
        auto lower = coordinates;
        auto upper = coordinates;
        lower[component] -= step;
        upper[component] += step;
        double lower_energy = 0.0;
        double upper_energy = 0.0;
        core.computeEFS(3, species.data(), contributing.data(), lower.data(),
                        lower_energy, nullptr, nullptr);
        core.computeEFS(3, species.data(), contributing.data(), upper.data(),
                        upper_energy, nullptr, nullptr);
        const double finite_difference_force =
            -(upper_energy - lower_energy) / (2.0 * step);
        if (!close(forces[component], finite_difference_force, 2e-7))
            return fail("force does not match the energy finite difference");
    }

    for (int axis = 0; axis < 3; ++axis) {
        if (!close(forces[axis] + forces[3 + axis] + forces[6 + axis], 0.0))
            return fail("total force is not zero");
    }

    const double expected_virial[6] = {
        -(coordinates[0]*forces[0] + coordinates[3]*forces[3] + coordinates[6]*forces[6]),
        -(coordinates[1]*forces[1] + coordinates[4]*forces[4] + coordinates[7]*forces[7]),
        -(coordinates[2]*forces[2] + coordinates[5]*forces[5] + coordinates[8]*forces[8]),
        -(coordinates[2]*forces[1] + coordinates[5]*forces[4] + coordinates[8]*forces[7]),
        -(coordinates[0]*forces[2] + coordinates[3]*forces[5] + coordinates[6]*forces[8]),
        -(coordinates[1]*forces[0] + coordinates[4]*forces[3] + coordinates[7]*forces[6]),
    };
    double virial_only[6] = {};
    double ignored_energy = 0.0;
    core.computeEFS(3, species.data(), contributing.data(), coordinates.data(),
                    ignored_energy, nullptr, virial_only);
    for (int component = 0; component < 6; ++component) {
        if (!close(virial[component], expected_virial[component], 2e-10) ||
            !close(virial_only[component], expected_virial[component], 2e-10))
            return fail("virial value or KIM component order is incorrect");
    }

    std::string duplicate_text = test_model();
    const auto position = duplicate_text.find("pair 0 0");
    duplicate_text.replace(position, std::string("pair 0 0").size(), "pair 0 1");
    std::istringstream duplicate_input(duplicate_text);
    mtp_kim::MTPCore invalid;
    if (invalid.load(duplicate_input))
        return fail("duplicate split-basis pair was accepted");

    std::cout << "MTP core tests passed\n";
    return 0;
}
