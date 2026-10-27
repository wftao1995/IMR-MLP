// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_cna.h"
#include "base/neighborhood.h"
#include <array>

TEST_CASE("CNA handles isolated atoms and small complete neighbor graphs", "[cna]")
{
    const int count = GENERATE(1, 2, 3, 4);
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{2, 2, 2}, {3, 2, 2}, {2.5, 2.866025403784, 2},
                  {2.5, 2.288675134595, 2.816496580928}};
    atoms.pos_.resize(count);
    atoms.chemmap_.assign(count, 0);
    Neighborhoods nbhs(atoms, 1.1);
    ComputeCNA calc(GPSettings{});
    const auto result = calc.compute(nbhs);
    const std::array<int, 4> indices{0, 1000, 1100, 1211};
    const cna_atom expected = count == 1 ? cna_atom{} : cna_atom{{indices[count - 1], count - 1}};
    REQUIRE(result.size() == count);
    for (const auto &site : result)
        REQUIRE(site == expected);
}

TEST_CASE("CNA counts both ends and preserves the central species after reordering", "[cna]")
{
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    const std::vector<Eigen::RowVector3d> positions{{2, 2, 2}, {3, 2, 2},
        {2.5, 2.866025403784, 2}, {2.5, 2.288675134595, 2.816496580928}};
    std::array<int, 4> order{0, 1, 2, 3};
    const cna_atom expected{{1211, 3}};
    ComputeCNA calc(GPSettings{});
    do
    {
        atoms.pos_.clear();
        atoms.chemmap_.clear();
        for (int i : order)
        {
            atoms.pos_.push_back(positions[i]);
            atoms.chemmap_.push_back(i % 2);
        }
        Neighborhoods nbhs(atoms, 1.1);
        const auto plain = calc.compute(nbhs);
        const auto species = calc.compute_sp(nbhs);
        for (int i = 0; i < atoms.size(); ++i)
        {
            REQUIRE(plain[i] == expected);
            int same = 0, different = 0;
            for (const auto &[index, count] : species[i])
            {
                REQUIRE(index.chem[0] == atoms.chemmap_[i]);
                (index.chem[0] == index.chem[1] ? same : different) += count;
            }
            REQUIRE(same == 1);
            REQUIRE(different == 2);
        }
    } while (std::next_permutation(order.begin(), order.end()));
}

TEST_CASE("CNA counts unbonded pairs at both ends without self pairs", "[cna]")
{
    Atoms atoms;
    atoms.cell_ = 10 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{1, 1, 1}, {2, 1, 1}, {3, 1, 1}};
    atoms.chemmap_ = {0, 1, 0};
    GPSettings settings;
    settings.Modify("unbond", "true");
    ComputeCNA calc(settings);
    Neighborhoods nbhs(atoms, 1.1);
    const auto result = calc.compute(nbhs);
    const cna_atom end{{1000, 1}, {2100, 1}}, middle{{1000, 2}};
    REQUIRE(result[0] == end);
    REQUIRE(result[1] == middle);
    REQUIRE(result[2] == end);
    for (const auto &site : calc.compute_sp(nbhs))
    {
        int count = 0;
        for (const auto &[index, value] : site)
            count += value;
        REQUIRE(count == 2);
    }
}

TEST_CASE("CNA distinguishes disjoint common bonds and closed bond chains", "[cna]")
{
    const bool bcc = GENERATE(false, true);
    const std::vector<Eigen::RowVector3d> basis = bcc
        ? std::vector<Eigen::RowVector3d>{{0, 0, 0}, {.5, .5, .5}}
        : std::vector<Eigen::RowVector3d>{{0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0}};
    Atoms atoms;
    // Keep distinct images of a site out of the same common-neighbor graph.
    atoms.cell_ = 4 * Eigen::Matrix3d::Identity();
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 4; ++z)
                for (const auto &p : basis)
                {
                    atoms.pos_.push_back(p + Eigen::RowVector3d(x, y, z));
                    atoms.chemmap_.push_back(0);
                }
    Neighborhoods nbhs(atoms, bcc ? 1.01 : .75);
    ComputeCNA calc(GPSettings{});
    const cna_atom expected = bcc ? cna_atom{{1666, 8}, {1444, 6}} : cna_atom{{1421, 12}};
    for (const auto &site : calc.compute(nbhs))
    {
        CAPTURE(bcc);
        REQUIRE(site.size() == expected.size());
        for (const auto &[index, count] : site)
        {
            CAPTURE(index, count);
            REQUIRE(expected.contains(index));
            REQUIRE(count == expected.at(index));
        }
    }
}

TEST_CASE("CNA primitive and conventional FCC cells retain image-resolved signatures", "[cna][periodic]")
{
    const bool primitive = GENERATE(false, true);
    Atoms atoms;
    if (primitive) {
        atoms.cell_ << 0, .5, .5, .5, 0, .5, .5, .5, 0;
        atoms.pos_ = {{0, 0, 0}};
    } else {
        atoms.cell_ = Eigen::Matrix3d::Identity();
        atoms.pos_ = {{0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0}};
    }
    atoms.chemmap_.assign(atoms.size(), 3);
    Neighborhoods nbhs(atoms, .75);
    for (int i = 0; i < nbhs.size(); ++i) REQUIRE(nbhs[i].nbhcount_ == 12);
    ComputeCNA calc(GPSettings{});
    const auto plain = calc.compute(nbhs);
    const auto species = calc.compute_sp(nbhs);
    REQUIRE(plain.size() == atoms.size()); REQUIRE(species.size() == atoms.size());
    // An FCC nearest-neighbor pair has four shared images, two disjoint
    // shared bonds, and longest chain one, independently of cell replication.
    const cna_atom expected{{1421, 12}};
    for (int i = 0; i < atoms.size(); ++i) {
        CAPTURE(primitive, i);
        REQUIRE(plain[i] == expected);
        REQUIRE(species[i].size() == 1);
        const auto &[index, count] = *species[i].begin();
        REQUIRE(index.chem[0] == 3); REQUIRE(index.chem[1] == 3);
        REQUIRE(index.index == 1421); REQUIRE(index.index_type == 420);
        REQUIRE(count == 12);
    }
}

TEST_CASE("CNA periodic self-image pairs include distinct unbonded two-hop images", "[cna][periodic]")
{
    Atoms atoms;
    atoms.cell_ = Eigen::Vector3d(1, 10, 10).asDiagonal();
    atoms.pbc_ = {true, false, false};
    atoms.pos_ = {{.2, .3, .4}}; atoms.chemmap_ = {7};
    Neighborhoods nbhs(atoms, 1.1);
    REQUIRE(nbhs[0].nbhcount_ == 2);
    GPSettings settings; settings.Modify("unbond", "true");
    ComputeCNA calc(settings);
    const auto plain = calc.compute(nbhs);
    REQUIRE(plain.size() == 1);
    // The +/-1 bonded pairs have no common neighbor; +/-2 unbonded pairs
    // each share the intervening image. Both ends belong to the same site.
    const cna_atom expected{{1000, 2}, {2100, 2}};
    REQUIRE(plain[0] == expected);
    const auto species = calc.compute_sp(nbhs);
    REQUIRE(species.size() == 1); REQUIRE(species[0].size() == 2);
    for (const auto &[index, count] : species[0]) {
        REQUIRE(index.chem[0] == 7); REQUIRE(index.chem[1] == 7);
        REQUIRE(count == 2);
        if (index.index == 1000) REQUIRE(index.index_type == 0);
        else { REQUIRE(index.index == 2100); REQUIRE(index.index_type == 100); }
    }
}
