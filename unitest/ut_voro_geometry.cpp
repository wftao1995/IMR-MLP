// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "base/neighborhood_voro_direct.h"
#include "base/neighborhood_voro_radi.h"
#include "base/voro_geometry.h"
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>

using Catch::Approx;

namespace
{
std::unique_ptr<NeighborhoodsVoro> neighbors(Atoms &atoms, bool radical, double tolerance, int minimum = 0)
{
    std::map<int, double> radii{{0, 0.8}, {1, 1.2}};
    if (radical)
        return std::make_unique<NeighborhoodsVoroRad>(atoms, radii, 12, tolerance, 0, minimum);
    return std::make_unique<NeighborhoodsVoroDir>(atoms, 12, tolerance, 0, minimum);
}

Atoms cube(int repeat)
{
    Atoms atoms;
    atoms.cell_ = 2 * repeat * Eigen::Matrix3d::Identity();
    for (int x = 0; x < repeat; ++x)
        for (int y = 0; y < repeat; ++y)
            for (int z = 0; z < repeat; ++z)
            {
                atoms.pos_.push_back({1. + 2 * x, 1. + 2 * y, 1. + 2 * z});
                atoms.chemmap_.push_back(0);
            }
    return atoms;
}

void check_neighbors(const Atoms &atoms, int center, const Neighborhood &nbh)
{
    REQUIRE(nbh.centertype == atoms.chemmap_[center]);
    REQUIRE(nbh.indices_.size() == nbh.nbhcount_);
    REQUIRE(nbh.chemmap_.size() == nbh.nbhcount_);
    REQUIRE(nbh.pos_.size() == nbh.nbhcount_);
    REQUIRE(nbh.distances_.size() == nbh.nbhcount_);
    for (int j = 0; j < nbh.nbhcount_; ++j)
    {
        CAPTURE(center, j);
        REQUIRE(nbh.indices_[j] >= 0);
        REQUIRE(nbh.indices_[j] < atoms.size());
        REQUIRE(nbh.chemmap_[j] == atoms.chemmap_[nbh.indices_[j]]);
        REQUIRE(nbh.pos_[j].allFinite());
        REQUIRE(nbh.pos_[j].norm() > 0);
        REQUIRE(std::isfinite(nbh.distances_[j]));
        REQUIRE(nbh.distances_[j] == Approx(nbh.pos_[j].norm()).margin(1e-12));
    }
}

void check_unique_matches(const Neighborhood &a, const Neighborhood &b,
                          const Eigen::Matrix3d &rotation)
{
    for (int j = 0; j < a.nbhcount_; ++j)
    {
        int matches = 0;
        for (int k = 0; k < b.nbhcount_; ++k)
            if (a.indices_[j] == b.indices_[k] &&
                (a.pos_[j] * rotation - b.pos_[k]).norm() < 1e-9)
                ++matches;
        CAPTURE(j, a.indices_[j]);
        REQUIRE(matches == 1);
    }
}

void check_rotation(Atoms &atoms, bool radical, double tolerance)
{
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    Atoms rotated(atoms);
    rotated.cell_ *= rotation;
    for (auto &position : rotated.pos_) position *= rotation;
    const auto original = neighbors(atoms, radical, tolerance);
    const auto transformed = neighbors(rotated, radical, tolerance);
    for (int i = 0; i < atoms.size(); ++i)
    {
        REQUIRE(transformed->voro_volume_each_[i] == Approx(original->voro_volume_each_[i]));
        const auto &a = (*original)[i];
        const auto &b = (*transformed)[i];
        check_neighbors(atoms, i, a);
        check_neighbors(rotated, i, b);
        REQUIRE(b.nbhcount_ == a.nbhcount_);
        check_unique_matches(a, b, rotation);
        check_unique_matches(b, a, rotation.transpose());
    }
}
}

TEST_CASE("Voronoi neighbors keep distinct periodic images", "[voro]")
{
    const bool radical = GENERATE(false, true);
    const int repeat = GENERATE(1, 2);
    const double tolerance = GENERATE(0., -0.001, 0.001);
    Atoms atoms = cube(repeat);
    const auto nbhs = neighbors(atoms, radical, tolerance);
    REQUIRE(std::accumulate(nbhs->voro_volume_each_.begin(), nbhs->voro_volume_each_.end(), 0.)
            == Approx(atoms.get_volume()));
    for (int i = 0; i < atoms.size(); ++i)
    {
        const auto &nbh = (*nbhs)[i];
        REQUIRE(nbh.nbhcount_ == 6);
        Eigen::RowVector3d sum = Eigen::RowVector3d::Zero();
        for (int j = 0; j < nbh.nbhcount_; ++j)
        {
            REQUIRE(nbh.indices_[j] >= 0);
            REQUIRE(nbh.indices_[j] < atoms.size());
            REQUIRE(nbh.distances_[j] == Approx(2));
            sum += nbh.pos_[j];
        }
        REQUIRE(sum.norm() < 1e-10);
    }
    check_rotation(atoms, radical, tolerance);
}

TEST_CASE("Radical Voronoi refinement preserves unequal-radius planes", "[voro]")
{
    Atoms atoms = cube(2);
    for (int i = 0; i < atoms.size(); ++i) atoms.chemmap_[i] = i % 2;
    const auto original = neighbors(atoms, true, 0);
    const auto refined = neighbors(atoms, true, -0.001);
    REQUIRE(original->voro_volume_each_[0] != Approx(original->voro_volume_each_[1]));
    REQUIRE(std::accumulate(refined->voro_volume_each_.begin(), refined->voro_volume_each_.end(), 0.)
            == Approx(atoms.get_volume()));
    for (int i = 0; i < atoms.size(); ++i)
        REQUIRE(refined->voro_volume_each_[i] == Approx(original->voro_volume_each_[i]));
    check_rotation(atoms, true, -0.001);
}

TEST_CASE("Small radical faces can be removed without moving the remaining planes", "[voro]")
{
    Atoms atoms = cube(2);
    Eigen::Matrix3d shear = Eigen::Matrix3d::Identity();
    shear(1, 0) = 0.01;
    atoms.cell_ *= shear;
    for (int i = 0; i < atoms.size(); ++i)
    {
        atoms.pos_[i] *= shear;
        atoms.chemmap_[i] = i % 2;
    }
    const auto original = neighbors(atoms, true, 0);
    const auto refined = neighbors(atoms, true, -0.01);
    for (int i = 0; i < atoms.size(); ++i)
    {
        REQUIRE((*refined)[i].nbhcount_ < (*original)[i].nbhcount_);
        // Dropping the tiny diagonal faces leaves a parallelepiped whose
        // height is fixed by the unequal-radius planes along z.
        const double height = atoms.chemmap_[i] == 0 ? 1.6 : 2.4;
        REQUIRE(refined->voro_volume_each_[i] == Approx(4 * 1.0001 * height));
    }
    check_rotation(atoms, true, -0.01);
}

TEST_CASE("Radical neighbors are resolved when a face lies behind its generating atom", "[voro]")
{
    Atoms atoms;
    atoms.cell_ = 4 * Eigen::Matrix3d::Identity();
    atoms.pos_ = {{0.5, 2, 2}, {1, 2, 2}};
    atoms.chemmap_ = {0, 1};
    const auto nbhs = neighbors(atoms, true, 0);
    REQUIRE(std::accumulate(nbhs->voro_volume_each_.begin(), nbhs->voro_volume_each_.end(), 0.)
            == Approx(atoms.get_volume()));
    REQUIRE((*nbhs)[0].nbhcount_ > 0);
    REQUIRE((*nbhs)[1].nbhcount_ > 0);
    // The actual 0.5-Angstrom pair must survive both radical geometries:
    // center 0 has plane offset -0.55; center 1 has offset +1.05, so its
    // generating atom lies closer than the face (the legitimate small root).
    for (int center : {0, 1})
    {
        const auto &nbh = (*nbhs)[center];
        const int other = 1 - center;
        int matches = 0;
        for (int j = 0; j < nbh.nbhcount_; ++j)
            if (nbh.indices_[j] == other &&
                (nbh.pos_[j] - (atoms.pos_[other] - atoms.pos_[center])).norm() < 1e-9)
                ++matches;
        REQUIRE(matches == 1);
    }
    check_rotation(atoms, true, -0.001);
}

TEST_CASE("Voronoi filtering removes small faces and handles excessive filtering", "[voro]")
{
    const bool radical = GENERATE(false, true);
    Atoms atoms = cube(1);
    atoms.cell_(1, 0) = 0.02;
    const auto original = neighbors(atoms, radical, 0);
    const auto refined = neighbors(atoms, radical, -0.01);
    REQUIRE((*refined)[0].nbhcount_ < (*original)[0].nbhcount_);
    REQUIRE(refined->voro_volume_each_[0] >= original->voro_volume_each_[0]);
    check_rotation(atoms, radical, -0.01);
    for (int minimum : {0, 100})
    {
        const auto excessive = neighbors(atoms, radical, -1, minimum);
        REQUIRE(excessive->voro_volume_each_[0] == Approx(original->voro_volume_each_[0]));
        REQUIRE((*excessive)[0].nbhcount_ == (*original)[0].nbhcount_);
    }
}

TEST_CASE("Triclinic self-image neighbors keep the center atom ID", "[voro]")
{
    const bool radical = GENERATE(false, true);
    const double tolerance = GENERATE(0., -0.001);
    Atoms atoms;
    atoms.cell_ << 2, 0, 0, 0.3, 2, 0, 0.2, 0.4, 10;
    atoms.pos_ = {{0.5, 0.5, 1}, {0.5, 0.5, 6}};
    atoms.chemmap_ = {0, 1};
    check_rotation(atoms, radical, tolerance);
    const auto nbhs = neighbors(atoms, radical, 0);
    REQUIRE(std::accumulate(nbhs->voro_volume_each_.begin(), nbhs->voro_volume_each_.end(), 0.)
            == Approx(atoms.get_volume()));
    for (int i = 0; i < atoms.size(); ++i)
        for (int j = 0; j < (*nbhs)[i].nbhcount_; ++j)
            if (std::abs((*nbhs)[i].pos_[j][2]) < 1e-10)
                REQUIRE((*nbhs)[i].indices_[j] == i);
}

TEST_CASE("Radical face recovery never substitutes a zero self image", "[voro][voro-recovery]")
{
    // This slab has physical power-bisector faces at z = +/-2.58 for the
    // atom at z = -5 and its periodic image at +5. The transverse faces
    // belong to the center's nonzero +/-2 lattice images. A few ULPs of
    // geometric roundoff must not change those analytic identities.
    const int ulps = GENERATE(Catch::Generators::range(-8, 9));
    double height = 2.58;
    for (int k = 0; k < std::abs(ulps); ++k)
        height = std::nextafter(height, ulps < 0 ? 0. : std::numeric_limits<double>::infinity());
    CAPTURE(ulps, height);
    Atoms atoms;
    atoms.cell_ = Eigen::Vector3d(2, 2, 10).asDiagonal();
    atoms.pos_ = {{0, 0, -5}, {0, 0, 0}};
    atoms.chemmap_ = {0, 1};
    std::map<int, double> radii{{0, 0.8}, {1, 1.2}};
    voro::voronoicell_neighbor cell;
    cell.init(-1, 1, -1, 1, -4, 4);
    REQUIRE(cell.nplane(0, 0, -1, 2 * height, 0));
    REQUIRE(cell.nplane(0, 0, 1, 2 * height, 0));
    const auto faces = voro_geometry::faces(cell, atoms, 1, &radii);
    REQUIRE(faces.size() == 6);
    int axial_faces = 0;
    for (const auto &face : faces)
    {
        REQUIRE(face.area > 0);
        REQUIRE(face.displacement.allFinite());
        CHECK(face.displacement.norm() > 0);
        const bool axial = std::abs(face.normal[2]) > 0.5;
        const Eigen::RowVector3d expected = (axial ? 5. : 2.) * face.normal;
        CHECK(face.neighbor == (axial ? 0 : 1));
        CHECK((face.displacement - expected).norm() < 1e-9);
        if (axial)
        {
            ++axial_faces;
            REQUIRE(face.area == Approx(4));
            REQUIRE(face.offset == Approx(2.58).margin(1e-13));
        }
        // Independent forward power-distance equality at a point on the
        // face, rather than reproducing the inverse quadratic solver.
        const Eigen::RowVector3d point = face.offset * face.normal;
        const double center_power = point.squaredNorm() - 1.2 * 1.2;
        const double other_radius = axial ? 0.8 : 1.2;
        CHECK(center_power == Approx((point - expected).squaredNorm() -
                                     other_radius * other_radius).margin(1e-12));
    }
    REQUIRE(axial_faces == 2);
}

TEST_CASE("Radical face recovery rejects a plane without a physical generating image", "[voro][voro-recovery]")
{
    Atoms atoms;
    atoms.cell_ = Eigen::Vector3d(2, 2, 10).asDiagonal();
    atoms.pos_ = {{0, 0, -0.5}, {0, 0, 0}};
    atoms.chemmap_ = {0, 1};
    std::map<int, double> radii{{0, 0.8}, {1, 1.2}};
    voro::voronoicell_neighbor cell;
    cell.init(-1, 1, -1, 1, -1, 5);
    REQUIRE(cell.nplane(0, 0, -1, 1, 0));
    // A genuine geometric face at z = -0.5 is deliberately mislabeled id0.
    // The generating atom's correct radical plane is z = -1.05, and no
    // periodic image generates this plane. Clamping a negative quadratic
    // discriminant to zero would incorrectly accept the 0.5 displacement.
    REQUIRE_THROWS_AS(voro_geometry::faces(cell, atoms, 1, &radii), std::runtime_error);
}
