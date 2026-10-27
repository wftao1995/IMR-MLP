// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "potentials/potential_mtp.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using Catch::Approx;

std::string split_potential(bool with_coefficients = true) {
    std::ostringstream out;
    out << "MTP\n"
        << "version = 1.1.0\n"
        << "potential_name = MTP1m\n"
        << "scaling = 1\n"
        << "species_count = 2\n"
        << "potential_tag =\n"
        << "radial_basis_split = 1\n"
        << "pair 0 0\n"
        << "radial_basis_type = RBChebyshev\n"
        << "  min_dist = 0.5\n"
        << "  max_dist = 4.0\n"
        << "  radial_basis_size = 2\n"
        << "pair 0 1\n"
        << "radial_basis_type = RBChebyCos\n"
        << "  min_dist = 0.0\n"
        << "  max_dist = 5.0\n"
        << "  radial_basis_size = 2\n"
        << "pair 1 0\n"
        << "radial_basis_type = RBChebyshevPoly5\n"
        << "  min_dist = 0.4\n"
        << "  max_dist = 4.5\n"
        << "  radial_basis_size = 2\n"
        << "  rb_b = -0.7\n"
        << "pair 1 1\n"
        << "radial_basis_type = RBChebyshev\n"
        << "  min_dist = 0.6\n"
        << "  max_dist = 3.2\n"
        << "  radial_basis_size = 2\n"
        << "radial_funcs_count = 1\n";
    if (with_coefficients) {
        out << "radial_coeffs\n"
            << "  0-0\n    {0.70, -0.20}\n"
            << "  0-1\n    {0.45, 0.10}\n"
            << "  1-0\n    {-0.35, 0.25}\n"
            << "  1-1\n    {0.80, -0.15}\n";
    }
    out << "alpha_moments_count = 5\n"
        << "alpha_index_basic_count = 4\n"
        << "alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, "
           "{0, 0, 1, 0}, {0, 0, 0, 1}}\n"
        << "alpha_index_times_count = 3\n"
        << "alpha_index_times = {{1, 1, 1, 4}, {2, 2, 1, 4}, {3, 3, 1, 4}}\n"
        << "alpha_scalar_moments = 2\n"
        << "alpha_moment_mapping = {0, 4}\n";
    if (with_coefficients) {
        out << "species_coeffs = {0.3, -0.1}\n"
            << "moment_coeffs = {1.2, -0.4}\n";
    }
    return out.str();
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

} // namespace

TEST_CASE("Spline export preserves pair bases and clamps the inner radius",
          "[mtp][spline]") {
    const bool split = GENERATE(false, true);
    const double rmin = GENERATE(-1., 0.2, 2.);
    Potentials::PotMTP pot;
    if (split) {
        std::stringstream input(split_potential());
        pot.Load(input);
    } else {
        pot.Load("unitest/test-data/potential.mtp");
    }
    const auto path = std::filesystem::current_path() / "mtp-spline.mtp";
    pot.SaveSpline(path.string(), rmin, 0.1);
    std::istringstream output(read_file(path));

    // The LAMMPS readers require a single RBSpline header followed by
    // species_count^2 * radial_func_count rows carrying explicit pair indices.
    std::string line, token;
    while (std::getline(output, line) && !line.starts_with("potential_tag")) {}
    std::getline(output, line);
    REQUIRE(line == "radial_basis_type = RBSpline");
    double minimum, maximum;
    output >> token;
    REQUIRE(token == "min_dist");
    output >> token >> minimum >> token;
    REQUIRE(minimum == 0);
    REQUIRE(token == "max_dist");
    output >> token >> maximum >> token;
    REQUIRE(maximum == Approx(pot.cutoff));
    REQUIRE(token == "radial_splines");
    int nr, radial_count;
    double dr;
    output >> nr >> dr >> radial_count;
    REQUIRE(nr > 2);
    REQUIRE(radial_count == pot.radial_func_count);
    std::vector<double> radii(nr), values(nr);
    for (double &r : radii) output >> r;
    REQUIRE(radii.front() == 0);
    REQUIRE(radii.back() == Approx(pot.cutoff));
    std::vector<bool> seen(pot.species_count * pot.species_count * radial_count, false);
    for (int row = 0; row < (int)seen.size(); ++row) {
        int mu, i, j;
        output >> mu >> i >> j;
        REQUIRE(output.good());
        REQUIRE((mu >= 0 && mu < radial_count));
        REQUIRE((i >= 0 && i < pot.species_count));
        REQUIRE((j >= 0 && j < pot.species_count));
        const int pair = i * pot.species_count + j;
        const int index = pair * radial_count + mu;
        REQUIRE_FALSE(seen[index]);
        seen[index] = true;
        for (double &v : values) output >> v;
        auto *basis = split ? pot.p_radical_basis_pair[pair] : pot.p_radical_basis;
        const double inner = rmin < 0 ? basis->min_val : rmin;
        std::vector<double> basis_values(pot.rb_size);
        for (int k = 0; k < nr; ++k) {
            const double r = std::max(inner, radii[k]);
            double expected = 0;
            if (r <= basis->max_val) {
                basis->calc_val(r, basis_values);
                for (int xi = 0; xi < pot.rb_size; ++xi)
                    expected += pot.radial_coeffs[index * pot.rb_size + xi] * basis_values[xi];
            }
            REQUIRE(values[k] == Approx(expected).margin(1e-12));
            if (radii[k] <= inner) REQUIRE(values[k] == values.front());
        }
    }
    output >> token;
    REQUIRE(token == "alpha_moments_count");
    REQUIRE(pot.radial_basis_split == split);
    std::filesystem::remove(path);
}

TEST_CASE("adjust_species validates and permutes every species parameter",
          "[adjustmtp][species]") {
    std::stringstream input(split_potential());
    Potentials::PotMTP pot;
    pot.Load(input);

    REQUIRE_THROWS_AS(pot.adjust_species({0}), GPException);
    REQUIRE_THROWS_AS(pot.adjust_species({0, 0}), GPException);
    REQUIRE_THROWS_AS(pot.adjust_species({0, 2}), GPException);
    REQUIRE_THROWS_AS(pot.adjust_species({-1, 0}), GPException);

    pot.adjust_species({1, 0});
    REQUIRE(pot.species_count == 2);
    REQUIRE(pot.species_coeffs[0] == Approx(-0.1));
    REQUIRE(pot.species_coeffs[1] == Approx(0.3));
    REQUIRE(pot.moment_coeffs[0] == Approx(1.2));
    REQUIRE(pot.moment_coeffs[1] == Approx(-0.4));
    REQUIRE(pot.radial_coeffs[0] == Approx(0.80));
    REQUIRE(pot.radial_coeffs[1] == Approx(-0.15));
    REQUIRE(pot.radial_coeffs[2] == Approx(-0.35));
    REQUIRE(pot.radial_coeffs[3] == Approx(0.25));
    REQUIRE(pot.radial_coeffs[4] == Approx(0.45));
    REQUIRE(pot.radial_coeffs[5] == Approx(0.10));
    REQUIRE(pot.radial_coeffs[6] == Approx(0.70));
    REQUIRE(pot.radial_coeffs[7] == Approx(-0.20));
    REQUIRE(pot.p_radical_basis_pair[0]->min_val == Approx(0.6));
    REQUIRE(pot.p_radical_basis_pair[1]->name == "RBChebyshevPoly5");
    REQUIRE(pot.p_radical_basis_pair[2]->name == "RBChebyCos");
    REQUIRE(pot.p_radical_basis_pair[3]->min_val == Approx(0.5));
}

TEST_CASE("adjust_species expands split radial potentials with cloned default bases",
          "[adjustmtp][species][split]") {
    std::stringstream input(split_potential());
    Potentials::PotMTP pot;
    pot.Load(input);
    pot.random_lower = 0.2;
    pot.random_upper = 0.3;
    pot.radial_mode = true;

    pot.adjust_species({2, 1, 0});
    REQUIRE(pot.species_count == 3);
    REQUIRE(pot.radial_coeff_size == 18);
    REQUIRE(pot.lin_coeff_size == 5);
    REQUIRE(pot.reg_coeff_size == 23);
    REQUIRE(pot.species_coeffs[0] >= 0.2);
    REQUIRE(pot.species_coeffs[0] <= 0.3);
    REQUIRE(pot.species_coeffs[1] == Approx(-0.1));
    REQUIRE(pot.species_coeffs[2] == Approx(0.3));
    REQUIRE(pot.moment_coeffs[0] == Approx(1.2));
    REQUIRE(pot.moment_coeffs[1] == Approx(-0.4));
    REQUIRE(pot.radial_coeffs[(1 * 3 + 2) * 2] == Approx(-0.35));
    REQUIRE(pot.radial_coeffs[(1 * 3 + 2) * 2 + 1] == Approx(0.25));
    REQUIRE(pot.p_radical_basis_pair.size() == 9);
    REQUIRE(pot.p_radical_basis_pair[0]->name == "RBChebyshev");
    REQUIRE(pot.p_radical_basis_pair[0]->min_val == Approx(0.5));
    REQUIRE(pot.p_radical_basis_pair[1 * 3 + 2]->name == "RBChebyshevPoly5");
    REQUIRE(pot.cutoff == Approx(5.0));
}

TEST_CASE("SaveOrdered writes loadable parameterized and empty ladders",
          "[adjustmtp][reorder][ladder]") {
    const auto parameterized_path =
        std::filesystem::current_path() / "adjustmtp-parameterized.mtp";
    const auto empty_path =
        std::filesystem::current_path() / "adjustmtp-empty.mtp";

    {
        std::stringstream input(split_potential());
        Potentials::PotMTP pot;
        pot.Load(input);
        pot.ReorderByLevel();
        pot.SaveOrdered(parameterized_path.string());
    }
    Potentials::PotMTP parameterized_ladder(
        parameterized_path.string() + ".ladder2");
    REQUIRE(parameterized_ladder.inited());
    REQUIRE(parameterized_ladder.alpha_scalar_moments == 1);
    REQUIRE(parameterized_ladder.species_coeffs[0] == Approx(0.3));
    REQUIRE(parameterized_ladder.species_coeffs[1] == Approx(-0.1));
    REQUIRE(parameterized_ladder.moment_coeffs[0] == Approx(1.2));

    {
        std::stringstream input(split_potential(false));
        Potentials::PotMTP pot;
        pot.Load(input);
        pot.ReorderByLevel();
        pot.SaveOrdered(empty_path.string());
    }
    const std::string empty_output = read_file(empty_path);
    REQUIRE(empty_output.find("radial_coeffs") == std::string::npos);
    REQUIRE(empty_output.find("species_coeffs") == std::string::npos);
    REQUIRE(empty_output.find("moment_coeffs") == std::string::npos);
    Potentials::PotMTP empty_main(empty_path.string());
    Potentials::PotMTP empty_ladder(empty_path.string() + ".ladder2");
    REQUIRE_FALSE(empty_main.inited());
    REQUIRE_FALSE(empty_ladder.inited());

    std::filesystem::remove(parameterized_path);
    std::filesystem::remove(parameterized_path.string() + ".ladder2");
    std::filesystem::remove(empty_path);
    std::filesystem::remove(empty_path.string() + ".ladder2");
}
