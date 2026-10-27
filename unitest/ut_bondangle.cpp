// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_bondangle.h"
#include "base/neighborhood.h"
#include <cmath>
#include <numeric>

using Catch::Approx;

TEST_CASE("Dihedral chemistry is unchanged when a chain is reversed", "[angles]")
{
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            for (int c = 0; c < 4; ++c)
                for (int d = 0; d < 4; ++d)
                {
                    const dihedral_type_index forward(a, b, c, d), reverse(d, c, b, a);
                    REQUIRE_FALSE(forward < reverse);
                    REQUIRE_FALSE(reverse < forward);
                }
    const dihedral_type_index key(0, 2, 1, 3);
    REQUIRE(std::vector<int>(key.chem, key.chem + 4) == std::vector<int>{3, 1, 2, 0});
}

TEST_CASE("Angle grids use radians and initialization preserves the spacing", "[angles]")
{
    ComputeAngles defaults;
    REQUIRE(defaults.angles.size() == 181);
    REQUIRE(defaults.angles[1] == Approx(M_PI / 180));
    REQUIRE(defaults.angles.back() == Approx(M_PI));
    const auto grid = defaults.angles;
    defaults.init();
    REQUIRE(defaults.angles == grid);

    GPSettings settings;
    settings.Modify("sep", "0.5");
    settings.Modify("low", "0.5");
    settings.Modify("high", "3.14");
    ComputeAngles configured(settings);
    REQUIRE(configured.angles == std::vector<double>{0.5, 1, 1.5, 2, 2.5, 3});
}

TEST_CASE("Bond angle histograms retain upper endpoint observations", "[angles]")
{
    const bool pair_cutoff = GENERATE(false, true);
    const double degree = GENERATE(90., 179.75, 180.);
    const double step = GENERATE(M_PI / 180, 0.5);
    const double angle = degree * M_PI / 180;
    Atoms atoms;
    atoms.cell_ = 10 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{4, 3, 3}, {3, 3, 3}, {3 + std::cos(angle), 3 + std::sin(angle), 3}};
    atoms.chemmap_ = {0, 0, 0};
    Neighborhoods nbhs(atoms, 1.1);
    ComputeAngles compute(step, 0, M_PI, 1);
    bondangle_count count;
    double cutoff = 1.1;
    double *cutoffs[] = {&cutoff};
    const auto result = pair_cutoff ? compute.compute(nbhs, count, cutoffs)
                                    : compute.compute(nbhs, count);
    const bondangle_type_index key(0, 0, 0);
    const auto &bins = result.at(key);
    REQUIRE(count.at(key) == 1);
    REQUIRE(std::accumulate(bins.begin(), bins.end(), 0) == 1);
    if (degree > 179) REQUIRE(bins.back() == 1);
    else REQUIRE(bins.at(step == 0.5 ? 3 : 90) == 1);
}

TEST_CASE("Dihedral histograms retain planar and nearly planar observations", "[angles]")
{
    const bool pair_cutoff = GENERATE(false, true);
    const double degree = GENERATE(0., 90., 179.75, 180.);
    const double angle = degree * M_PI / 180;
    Atoms atoms;
    atoms.cell_ = 10 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{2.3, 3.7, 3}, {3, 3, 3}, {4, 3, 3},
                  {4.7, 3 - 0.7 * std::cos(angle), 3 + 0.7 * std::sin(angle)}};
    atoms.chemmap_ = {0, 0, 0, 0};
    Neighborhoods nbhs(atoms, 1.1);
    ComputeAngles compute(M_PI / 180, 0, M_PI, 1);
    dihedral_count count;
    double cutoff = 1.1;
    double *cutoffs[] = {&cutoff};
    const auto result = pair_cutoff ? compute.compute(nbhs, count, cutoffs)
                                    : compute.compute(nbhs, count);
    const dihedral_type_index key(0, 0, 0, 0);
    const auto &bins = result.at(key);
    REQUIRE(count.at(key) == 1);
    REQUIRE(std::accumulate(bins.begin(), bins.end(), 0) == 1);
    REQUIRE(bins.at(static_cast<int>(std::round(degree))) == 1);
}

TEST_CASE("Periodic dihedral images retain the simple cubic torsion counts", "[angles][dihedral][periodic]") {
    const int repeat = GENERATE(1, 3);
    CAPTURE(repeat);
    Atoms atoms;
    atoms.cell_ = repeat * Eigen::Matrix3d::Identity();
    for (int x = 0; x < repeat; ++x)
        for (int y = 0; y < repeat; ++y)
            for (int z = 0; z < repeat; ++z)
                atoms.pos_.push_back({double(x), double(y), double(z)});
    atoms.chemmap_.assign(atoms.size(), 0);
    const auto source = atoms;
    Neighborhoods short_neighbors(atoms, 1.1), long_neighbors(atoms, 1.45);
    ComputeAngles compute(M_PI / 180, 0, M_PI, 1);
    dihedral_count count;
    const auto result = compute.compute(short_neighbors, count);
    const dihedral_type_index key(0, 0, 0, 0);
    const auto &bins = result.at(key);
    // Three undirected central bonds per site. Each has four non-collinear
    // left and right edges: four at 0 degrees, eight at 90, four at 180.
    REQUIRE(count.at(key) == 48 * atoms.size());
    REQUIRE(bins.size() == 181);
    for (int bin = 0; bin < bins.size(); ++bin) {
        const int expected = bin == 0 || bin == 180 ? 12 * atoms.size()
                           : bin == 90 ? 24 * atoms.size() : 0;
        CAPTURE(bin);
        REQUIRE(bins[bin] == expected);
    }
    REQUIRE(std::accumulate(bins.begin(), bins.end(), 0) == 48 * atoms.size());
    double cutoff = 1.1;
    double *cutoffs[] = {&cutoff};
    dihedral_count filtered_count;
    const auto filtered = compute.compute(long_neighbors, filtered_count, cutoffs);
    REQUIRE(filtered_count.at(key) == 48 * atoms.size());
    REQUIRE(filtered.at(key) == bins);
    REQUIRE(atoms.pos_ == source.pos_);
    REQUIRE(atoms.cell_ == source.cell_);
    REQUIRE(atoms.pbc_ == source.pbc_);
    REQUIRE(atoms.cartesian == source.cartesian);
}
