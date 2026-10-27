// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_boo.h"
#include "base/neighborhood.h"
#include "base/neighborhood_voro_direct.h"

using Catch::Approx;

namespace
{
Atoms cluster(std::vector<Eigen::RowVector3d> positions)
{
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = std::move(positions);
    atoms.chemmap_.resize(atoms.size(), 0);
    return atoms;
}
}

TEST_CASE("Averaging identical bond environments preserves Q", "[bop]")
{
    Atoms atoms = cluster({{2, 2, 2}, {3, 2, 2}});
    ComputeBondOrder compute(GPSettings{});
    compute.compute(atoms);
    REQUIRE(atoms.local_attribs_v_.at("bopQ").isOnes(1e-12));
    REQUIRE(atoms.local_attribs_v_.at("bopQ_A").isOnes(1e-12));
}

TEST_CASE("Dimer normalized W2 and W4 retain their analytic values under rotation", "[bop]")
{
    const double angle = GENERATE(0., 0.7);
    Atoms atoms = cluster({{2, 2, 2}, {2, 2, 3}});
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(angle, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    atoms.cell_ *= rotation;
    for (auto &position : atoms.pos_) position *= rotation;
    const auto positions = atoms.pos_;
    const auto cell = atoms.cell_;
    const auto species = atoms.chemmap_;

    GPSettings settings;
    settings.Modify("maxl", "4");
    settings.Modify("cutoff", "1.1");
    settings.Modify("average", "true");
    settings.Modify("corase", "true");
    ComputeBondOrder compute(settings);
    compute.compute(atoms);

    // An axial single bond has only m=0 at even l, so normalized W_l is
    // (l l l; 0 0 0). Its factorial formula gives these independent constants.
    const double w2 = -std::sqrt(2. / 35);
    const double w4 = std::sqrt(18. / 1001);
    for (const std::string suffix : {"", "_A", "_C"})
    {
        INFO(suffix);
        const auto &q = atoms.local_attribs_v_.at("bopQ" + suffix);
        const auto &w = atoms.local_attribs_v_.at("bopW" + suffix);
        REQUIRE(q.rows() == 2);
        REQUIRE(q.cols() == 2);
        REQUIRE(q.isOnes(1e-12));
        REQUIRE(w.rows() == 2);
        REQUIRE(w.cols() == 2);
        for (int i = 0; i < 2; ++i)
        {
            REQUIRE(w(0, i) == Approx(w2).margin(1e-12));
            REQUIRE(w(1, i) == Approx(w4).margin(1e-12));
        }
    }
    REQUIRE(atoms.cell_.isApprox(cell, 0));
    REQUIRE(atoms.chemmap_ == species);
    for (int i = 0; i < 2; ++i)
        REQUIRE(atoms.pos_[i].isApprox(positions[i], 0));
}

TEST_CASE("Coarse BOP works independently of spherical-harmonic averaging", "[bop]")
{
    const bool average = GENERATE(false, true);
    Atoms atoms = cluster({{2, 2, 2}, {3, 2, 2}, {3, 3, 2}, {2, 3, 2}});
    GPSettings settings;
    settings.Modify("average", average ? "true" : "false");
    settings.Modify("corase", "true");
    ComputeBondOrder compute(settings);
    for (int repeat = 0; repeat < 2; ++repeat)
    {
        compute.compute(atoms);
        for (const std::string key : {"bopQ", "bopW"})
        {
            const auto &base = atoms.local_attribs_v_.at(key);
            const auto &coarse = atoms.local_attribs_v_.at(key + "_C");
            REQUIRE(coarse.rows() == 2);
            REQUIRE(coarse.cols() == 4);
            REQUIRE(coarse.allFinite());
            // All four corners have the same scalar Q/W values.
            REQUIRE(coarse.isApprox(base, 1e-12));
        }
    }
}

TEST_CASE("BOP nearest-neighbor settings retain the requested count", "[bop]")
{
    const std::string option = GENERATE("nnn", "minnn", "maxnn");
    Atoms atoms = cluster({{2, 2, 2}, {3, 2, 2}, {3, 4, 2}});
    GPSettings settings;
    settings.Modify(option, "1");
    if (option == "minnn") settings.Modify("cutoff", "0.2");
    ComputeBondOrder compute(settings);
    compute.compute(atoms);
    REQUIRE(atoms.local_attribs_v_.at("bopQ").isOnes(1e-12));
}

TEST_CASE("BOP uses supplied neighborhoods without rebuilding them", "[bop]")
{
    Atoms atoms = cluster({{2, 2, 2}, {3, 2, 2}, {3, 4, 2}});
    Neighborhoods nbhs(atoms, 1.1);
    ComputeBondOrder compute(GPSettings{});
    compute.compute(atoms, nbhs);
    const auto &q = atoms.local_attribs_v_.at("bopQ");
    REQUIRE(q.leftCols(2).isOnes(1e-12));
    REQUIRE(q.col(2).isZero());
}

TEST_CASE("Isolated atoms have finite zero BOP values in every output", "[bop]")
{
    const bool average = GENERATE(false, true);
    Atoms atoms = cluster({{2, 2, 2}});
    GPSettings settings;
    settings.Modify("average", average ? "true" : "false");
    settings.Modify("corase", "true");
    ComputeBondOrder compute(settings);
    compute.compute(atoms);
    for (const auto &[key, values] : atoms.local_attribs_v_)
    {
        INFO(key);
        REQUIRE(values.allFinite());
        REQUIRE(values.isZero());
    }
}

TEST_CASE("Voronoi BOP retains cubic order under rotation", "[bop]")
{
    const double angle = GENERATE(0., 0.7);
    Atoms atoms = cluster({{1, 1, 1}});
    atoms.cell_ = 2 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {true, true, true};
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(angle, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    atoms.cell_ *= rotation;
    atoms.pos_[0] *= rotation;
    NeighborhoodsVoroDir nbhs(atoms, 10, 0, 0, 0);
    GPSettings settings;
    settings.Modify("corase", "true");
    ComputeBondOrder compute(settings);
    compute.compute(atoms, nbhs);
    for (const std::string key : {"bopQ", "bopQ_A", "bopQ_C"})
    {
        const auto &q = atoms.local_attribs_v_.at(key);
        REQUIRE(q(0, 0) == Approx(0).margin(1e-12));
        REQUIRE(q(1, 0) == Approx(std::sqrt(7. / 12)));
    }
}
