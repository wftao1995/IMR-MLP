// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include <catch2/catch_all.hpp>

#include "LAMMPS/USER-MLIP/opt/radial_basis_opt.h"
#include "basis/basis.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

TEST_CASE("LAMMPS optimized radial bases match the IMRMLP basis implementation",
          "[lammps][radial]")
{
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"RBChebyshev", ""},
        {"RBChebyCos", ""},
        {"RBChebyshevPoly4", "rb_b = -0.7\n"},
        {"RBChebyshevPoly5", "rb_b = -0.7\n"},
        {"RBChebyshevConfigurable",
         "cutoff_type = poly5\ncutoff_b = -0.7\n"
         "shift_type = exponential\nshift_lambda = 1.2\n"},
    };

    for (const auto &radial_case : cases) {
        INFO(radial_case.first);
        const std::string body =
            "min_dist = 0.5\nmax_dist = 4.5\nradial_basis_size = 6\n" +
            radial_case.second;
        std::istringstream reference_input(body);
        std::unique_ptr<AnyBasis> reference(
            create_basis_by_name(radial_case.first, reference_input, 1.3));
        std::istringstream optimized_input(
            "= " + radial_case.first + "\n" + body + "radial_funcs_count");
        const auto parsed = MLIPOpt::read_radial_basis(optimized_input, 1.3);
        REQUIRE(parsed.next_token == "radial_funcs_count");

        for (const double distance : {0.5, 0.9, 2.1, 4.1, 4.5}) {
            std::vector<double> values(6), derivatives(6);
            std::vector<double> reference_values(6), reference_derivatives(6);
            parsed.basis.evaluate(
                distance, values.data(), derivatives.data());
            reference->calc_der(
                distance, reference_values, reference_derivatives);
            for (int i = 0; i < 6; ++i) {
                REQUIRE(values[i] == Catch::Approx(reference_values[i])
                    .margin(2e-12 * std::max(1.0, std::abs(reference_values[i]))));
                REQUIRE(derivatives[i] == Catch::Approx(reference_derivatives[i])
                    .margin(2e-11 * std::max(1.0, std::abs(reference_derivatives[i]))));
            }
        }
    }
}

TEST_CASE("LAMMPS optimized radial parser supports ordered split bases",
          "[lammps][radial][split]")
{
    std::istringstream input(
        "= 1\n"
        "pair 1 1 radial_basis_type = RBChebyshev\n"
        "min_dist = 0.5 max_dist = 3 radial_basis_size = 2\n"
        "pair 0 1 radial_basis_type = RBChebyCos\n"
        "min_dist = 0 max_dist = 4 radial_basis_size = 2\n"
        "pair 1 0 radial_basis_type = RBChebyshevPoly5\n"
        "min_dist = 0.4 max_dist = 4.5 radial_basis_size = 2 rb_b = -0.7\n"
        "pair 0 0 radial_basis_type = RBChebyshevConfigurable\n"
        "min_dist = 0.2 max_dist = 5 radial_basis_size = 2\n"
        "cutoff_type = cosine shift_type = linear\n"
        "radial_funcs_count");
    std::string token = "radial_basis_split";
    bool split = false;
    int size = 0;
    double cutoff = 0.0;
    const auto bases = MLIPOpt::read_radial_bases(
        input, token, 2, 1.0, split, size, cutoff);

    REQUIRE(split);
    REQUIRE(size == 2);
    REQUIRE(cutoff == Catch::Approx(5.0));
    REQUIRE(token == "radial_funcs_count");
    REQUIRE(bases[0].name == "RBChebyshevConfigurable");
    REQUIRE(bases[1].name == "RBChebyCos");
    REQUIRE(bases[2].name == "RBChebyshevPoly5");
    REQUIRE(bases[3].name == "RBChebyshev");
}

TEST_CASE("LAMMPS optimized radial parser rejects Poly5 without rb_b",
          "[lammps][radial][validation]")
{
    std::istringstream input(
        "= RBChebyshevPoly5 min_dist = 0.5 max_dist = 4.5 "
        "radial_basis_size = 2 radial_funcs_count");
    REQUIRE_THROWS_WITH(
        MLIPOpt::read_radial_basis(input, 1.0),
        "Invalid MTP radial basis: RBChebyshevPoly5 is missing rb_b");
}
