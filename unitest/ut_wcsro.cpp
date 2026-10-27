// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_wcsro.h"
#include "base/neighborhood.h"

using Catch::Approx;

TEST_CASE("WCSRO normalizes like and unlike pairs without consuming counts", "[wcsro]")
{
    const bool square = GENERATE(false, true);
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = square
        ? std::vector<Eigen::RowVector3d>{{2, 2, 2}, {3, 2, 2}, {3, 3, 2}, {2, 3, 2}}
        : std::vector<Eigen::RowVector3d>{{2, 2, 2}, {3, 2, 2}, {2.5, 2.866025403784, 2},
                                        {2.5, 2.288675134595, 2.816496580928}};
    atoms.chemmap_ = {0, 0, 1, 1};
    Neighborhoods nbhs(atoms, 1.1);
    ComputeWCSRO calc(GPSettings{});
    std::vector<size_t> counter(4), types(2);
    size_t total_atoms = 0, total_pairs = 0;
    // Repeated output must not change subsequent cumulative statistics.
    for (int frames = 1; frames <= 2; ++frames)
    {
        calc.compute(nbhs, counter, types, total_atoms, total_pairs);
        const auto original = counter;
        REQUIRE(total_atoms == 4 * frames);
        REQUIRE(total_pairs == (square ? 8 : 12) * frames);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            std::vector<std::vector<double>> result;
            calc.postprocess(counter, types, total_atoms, total_pairs, result);
            REQUIRE(counter == original);
            REQUIRE(result[0][0] == Approx(square ? 0.0 : 1.0 / 3.0).margin(1e-14));
            REQUIRE(result[1][1] == Approx(result[0][0]).margin(1e-14));
            REQUIRE(result[0][1] == Approx(square ? 0.0 : -1.0 / 3.0).margin(1e-14));
            REQUIRE(result[1][0] == Approx(result[0][1]).margin(1e-14));
        }
    }
}

TEST_CASE("WCSRO uses conditional neighbor probabilities at unequal coordination", "[wcsro]")
{
    const std::string definition = GENERATE("WC", "PM");
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    // A center with three B leaves. The A atom has degree 3, each B degree 1.
    atoms.pos_ = {{5,5,5}, {4,5,5}, {6,5,5}, {5,6,5}};
    atoms.chemmap_ = {0,1,1,1};
    Neighborhoods nbhs(atoms, 1.1);
    GPSettings settings;
    settings.Modify("sro_type", definition);
    ComputeWCSRO calc(settings);
    std::vector<size_t> counter(3), types(2);
    size_t total_atoms = 0, total_pairs = 0;
    calc.compute(nbhs, counter, types, total_atoms, total_pairs);
    REQUIRE(counter == std::vector<size_t>{0,6,0});
    REQUIRE(types == std::vector<size_t>{1,3});
    REQUIRE(total_pairs == 6);
    std::vector<std::vector<double>> result;
    calc.postprocess(counter, types, total_atoms, total_pairs, result);
    CHECK(result[0][1] == Approx(-1.0/3.0).margin(1e-14));
    CHECK(result[1][0] == Approx(-3.0).margin(1e-14));
    CHECK(result[0][0] == Approx(definition == "WC" ? 1.0 : -1.0/3.0).margin(1e-14));
    CHECK(result[1][1] == Approx(definition == "WC" ? 1.0 : -3.0).margin(1e-14));
}

TEST_CASE("WCSRO honors explicit radial shells and cutoff fallbacks", "[wcsro]")
{
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{5,5,5}, {6,5,5}, {8,5,5}};
    atoms.chemmap_ = {0,1,1};
    Neighborhoods nbhs(atoms, 4.0);
    GPSettings settings;
    settings.Modify("r_inn", "1");
    settings.Modify("r_out", "3");
    ComputeWCSRO calc(settings);
    std::vector<size_t> counter(3), types(2);
    size_t total_atoms = 0, total_pairs = 0;
    calc.compute(nbhs, counter, types, total_atoms, total_pairs);
    // The two boundary distances 1 and 3 are excluded; only the B-B pair at 2 remains.
    CHECK(counter == std::vector<size_t>{0,0,2});
    CHECK(total_pairs == 2);
    CHECK(types == std::vector<size_t>{1,2});
    CHECK(calc.inner_radius(0.2) == Approx(1.0));
    CHECK(calc.outer_radius(5.0) == Approx(3.0));
    ComputeWCSRO defaults(GPSettings{});
    CHECK(defaults.inner_radius(0.2) == Approx(0.2));
    CHECK(defaults.outer_radius(5.0) == Approx(5.0));
    REQUIRE_THROWS_AS(defaults.inner_radius(-0.2), std::invalid_argument);
    REQUIRE_THROWS_AS(defaults.outer_radius(0.0), std::invalid_argument);
    GPSettings inner_only;
    inner_only.Modify("r_inn", "2");
    REQUIRE_THROWS_AS(ComputeWCSRO(inner_only).outer_radius(1), std::invalid_argument);
    GPSettings outer_only;
    outer_only.Modify("r_out", "2");
    REQUIRE_THROWS_AS(ComputeWCSRO(outer_only).inner_radius(3), std::invalid_argument);
}

TEST_CASE("WCSRO rejects invalid definitions and radial intervals", "[wcsro]")
{
    const auto entry = GENERATE(Catch::Generators::values<std::pair<std::string, std::string>>({
        {"sro_type", "GM"}, {"sro_type", "WCMD"}, {"sro_type", "invalid"}, {"r_inn", "-2"},
        {"r_out", "-2"}, {"r_out", "0"}, {"r_inn", "nan"}, {"r_out", "inf"}}));
    GPSettings settings;
    settings.Modify(entry.first, entry.second);
    REQUIRE_THROWS_AS(ComputeWCSRO(settings), std::invalid_argument);
    settings = GPSettings{};
    settings.Modify("r_inn", "3");
    settings.Modify("r_out", "2");
    REQUIRE_THROWS_AS(ComputeWCSRO(settings), std::invalid_argument);
}

TEST_CASE("WCSRO validates type IDs and count storage before accumulation", "[wcsro]")
{
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{5,5,5}, {6,5,5}};
    atoms.chemmap_ = {0,1};
    ComputeWCSRO calc(GPSettings{});
    std::vector<size_t> counter(3), types(2);
    size_t total_atoms = 0, total_pairs = 0;
    const int invalid = GENERATE(0,1,2);
    if (invalid == 0) counter.resize(2);
    if (invalid == 1) types.resize(1);
    if (invalid == 2) atoms.chemmap_[1] = 2;
    Neighborhoods nbhs(atoms, 1.1);
    REQUIRE_THROWS_AS(calc.compute(nbhs, counter, types, total_atoms, total_pairs), std::invalid_argument);
    CHECK(total_atoms == 0);
    CHECK(total_pairs == 0);
    for (size_t count : counter) CHECK(count == 0);
    for (size_t count : types) CHECK(count == 0);
}

TEST_CASE("WCSRO rejects undefined or inconsistent postprocessing domains", "[wcsro]")
{
    ComputeWCSRO calc(GPSettings{});
    std::vector<size_t> counter(3), types{1,1};
    size_t total_atoms = 2, total_pairs = 0;
    const int invalid = GENERATE(0,1,2,3,4,5,6);
    if (invalid == 0) { types = {0,0}; total_atoms = 0; }
    if (invalid == 1) { types = {2,0}; }
    // invalid==2: both types exist but there are no neighbor pairs.
    if (invalid == 3) { counter = {2,0,0}; total_pairs = 2; }
    if (invalid == 4) { types = {1,2}; }
    if (invalid == 5) { counter = {0,1,0}; total_pairs = 1; }
    if (invalid == 6) { counter.resize(2); }
    std::vector<std::vector<double>> result;
    REQUIRE_THROWS_AS(calc.postprocess(counter, types, total_atoms, total_pairs, result), std::invalid_argument);
}
