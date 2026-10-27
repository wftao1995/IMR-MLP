// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_sro.h"
#include "base/neighborhood.h"
#include "base/neighborhood_cluster_simple.h"
#include <array>
#include <numeric>

using Catch::Approx;

namespace
{
class CompactNeighbors : public AnyNeighborhoodsSimple
{
public:
    explicit CompactNeighbors(const AnyNeighborhoods &source)
    {
        nbhs.resize(source.size());
        for (int i = 0; i < source.size(); ++i)
        {
            nbhs[i].indices_ = source[i].indices_;
            nbhs[i].distances_ = source[i].distances_;
        }
    }
};
}

TEST_CASE("SRO shell statistics are independent of atom order for all neighbor formats", "[sro]")
{
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    const std::vector<Eigen::RowVector3d> positions{{1, 1, 1}, {2, 1, 1}, {6, 1, 1}, {8, 1, 1}};
    const std::array<int, 4> types{0, 0, 1, 1};
    std::array<int, 4> order{0, 1, 2, 3};
    ComputeSRO compute(GPSettings{});
    do
    {
        atoms.pos_.clear();
        atoms.chemmap_.clear();
        for (int i : order)
        {
            atoms.pos_.push_back(positions[i]);
            atoms.chemmap_.push_back(types[i]);
        }
        Neighborhoods full(atoms, 2.1);
        CompactNeighbors compact(full);
        NeighborhoodsClusterSimple half(atoms, 2.1);
        for (const auto &result : {compute.compute(atoms, full),
                                   compute.compute(atoms, compact, nullptr, false),
                                   compute.compute(atoms, half)})
        {
            REQUIRE(result.size() == 6);
            for (int i = 0; i < result.size(); ++i)
            {
                REQUIRE(result[i].order == i / 3);
                REQUIRE(result[i].sro == Approx(i == 0 || i == 5 ? -3. : 1.));
            }
        }
    } while (std::next_permutation(order.begin(), order.end()));
}

TEST_CASE("SRO finds shells beyond an isolated first atom", "[sro]")
{
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{10, 10, 10}, {1, 1, 1}, {2, 1, 1}};
    atoms.chemmap_ = {0, 0, 0};
    Neighborhoods nbhs(atoms, 1.1);
    ComputeSRO compute(GPSettings{});
    const auto result = compute.compute(atoms, nbhs);
    REQUIRE(result.size() == 1);
    REQUIRE(result[0].sro == Approx(0));
}

TEST_CASE("SRO distance clustering is independent of neighbor traversal order", "[sro]")
{
    AnyNeighborhoods nbhs;
    nbhs.resize(2);
    nbhs[0].distances_ = {1.04, 1.00, 2.0};
    nbhs[1].distances_ = {1.08, 2.03};
    std::vector<double> expected, actual;
    int count;
    get_orders(nbhs, expected, count, 0.05);
    std::swap(nbhs[0], nbhs[1]);
    for (int i = 0; i < nbhs.size(); ++i)
        std::reverse(nbhs[i].distances_.begin(), nbhs[i].distances_.end());
    get_orders(nbhs, actual, count, 0.05);
    REQUIRE(actual == expected);
    REQUIRE(count == expected.size());
}

TEST_CASE("SRO assigns overlapping distance windows to disjoint shells", "[sro]")
{
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{1, 1, 1}, {2, 1, 1}, {5, 1, 1}, {6.04, 1, 1}, {10, 1, 1}, {11.08, 1, 1}};
    atoms.chemmap_ = {0, 0, 0, 1, 1, 1};
    if (GENERATE(false, true))
    {
        std::reverse(atoms.pos_.begin(), atoms.pos_.end());
        std::reverse(atoms.chemmap_.begin(), atoms.chemmap_.end());
    }
    Neighborhoods full(atoms, 1.1);
    CompactNeighbors compact(full);
    NeighborhoodsClusterSimple half(atoms, 1.1);
    ComputeSRO calc(GPSettings{});
    std::map<sro_counter, size_t> counts;
    std::vector<pair_srotype> result;
    SECTION("full") { result = calc.compute(atoms, full, &counts); }
    SECTION("compact") { result = calc.compute(atoms, compact, &counts, false); }
    SECTION("half") { result = calc.compute(atoms, half, &counts); }
    size_t total = 0;
    for (const auto &[key, count] : counts)
        total += count;
    REQUIRE(total == 6);
    REQUIRE(counts[sro_counter{0, 0, 1}] == 1);
    REQUIRE(counts[sro_counter{1, 0, 1}] == 0);
    const std::array<double, 6> expected{-1, 0, 1, 1, 1, -3};
    REQUIRE(result.size() == expected.size());
    for (int i = 0; i < result.size(); ++i)
        REQUIRE(result[i].sro == Approx(expected[i]).margin(1e-14));
}

TEST_CASE("SRO rejects unsupported definitions instead of returning zero", "[sro][errors]")
{
    GPSettings settings;
    for (const auto *type : {"GM", "WCMD", "invalid"})
    {
        settings.Modify("sro_type", type);
        auto message = Catch::Matchers::ContainsSubstring("SRO");
        REQUIRE_THROWS_WITH(ComputeSRO{settings}, message);
    }
    for (const auto *type : {"WC", "PM"})
    {
        settings.Modify("sro_type", type);
        REQUIRE_NOTHROW(ComputeSRO{settings});
    }
    REQUIRE(compute_sro_value(WC, 1, 4, .5, .5, true) == Approx(0.0));
    REQUIRE(compute_sro_value(PM, 1, 4, .5, .5, true) == Approx(0.0));
}
