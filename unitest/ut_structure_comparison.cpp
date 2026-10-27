// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "symmetry/structure_comparison.h"
#include <Eigen/Geometry>
#include <limits>

using namespace Symmetry3D;

namespace {
Atoms crystal() {
    Atoms atom;
    atom.cell_ << 8, 0, 0, 0, 9, 0, 0, 0, 10;
    atom.pos_ = {{1, 1, 1}, {3, 2, 1}, {2, 4, 5}};
    atom.chemmap_ = {0, 1, 0};
    return atom;
}
}

TEST_CASE("Shared structure comparison handles translations permutations and supercells", "[equivalence]") {
    const bool descriptors = GENERATE(false, true);
    StructureComparisonOptions options;
    options.use_descriptors = descriptors;
    StructureComparator comparator(options);
    const auto original = crystal();
    auto other = original;
    for (auto &position : other.pos_) position += Eigen::RowVector3d(2.3, -4.1, 11.2);
    other.wrap();
    std::reverse(other.pos_.begin(), other.pos_.end());
    std::reverse(other.chemmap_.begin(), other.chemmap_.end());
    other.swap_axis(231);
    other.mksuper(Eigen::Vector3i(2, 1, 1));
    const auto first = comparator.prepare(original);
    const auto second = comparator.prepare(other);
    REQUIRE(comparator.equivalent(first, second));
    REQUIRE(comparator.equivalent(second, first));
    REQUIRE(first.atoms.pos_ == comparator.prepare(original).atoms.pos_);
}

TEST_CASE("Shared structure comparison checks chemistry and bijective positions", "[equivalence]") {
    StructureComparisonOptions options;
    options.use_descriptors = false;
    StructureComparator comparator(options);
    const auto first = comparator.prepare(crystal());
    auto other = crystal();
    std::swap(other.chemmap_[0], other.chemmap_[1]);
    REQUIRE_FALSE(comparator.equivalent(first, comparator.prepare(other)));
    other = crystal();
    other.pos_[2] = other.pos_[0] + Eigen::RowVector3d(.01, 0, 0);
    REQUIRE_FALSE(comparator.equivalent(first, comparator.prepare(other)));
}

TEST_CASE("Shared structure comparison scales volume only when requested", "[equivalence]") {
    auto original = crystal();
    const int dimensions = GENERATE(0, 1, 2, 3);
    for (int i = 0; i < 3; ++i) original.pbc_[i] = i < dimensions;
    auto other = original;
    other.scale_to_volume(8 * other.get_volume());
    StructureComparator absolute;
    REQUIRE_FALSE(absolute.equivalent(absolute.prepare(original), absolute.prepare(other)));
    StructureComparisonOptions options;
    options.scale_volume = true;
    StructureComparator scaled(options);
    REQUIRE(scaled.equivalent(scaled.prepare(original), scaled.prepare(other)));
}

TEST_CASE("Structure comparison handles arbitrary cell rotations without idealization", "[equivalence]") {
    StructureComparisonOptions options;
    options.use_descriptors = GENERATE(false, true);
    StructureComparator comparator(options);
    auto original = crystal();
    original.cell_(0, 1) = 0.00003;
    original.local_attribs_v_["forces"] = Eigen::MatrixXd::Ones(3, original.size());
    const auto rotation = Eigen::AngleAxisd(.37, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    auto other = original;
    other.cell_ *= rotation;
    for (auto &position : other.pos_) position = (position + Eigen::RowVector3d(3, -4, 7)) * rotation;
    const auto first = comparator.prepare(original);
    REQUIRE(first.atoms.local_attribs_v_.empty());
    REQUIRE(first.atoms.cartesian);
    REQUIRE(original.local_attribs_v_.count("forces") == 1);
    REQUIRE(comparator.equivalent(first, comparator.prepare(other)));
    REQUIRE(comparator.equivalent(comparator.prepare(other), first));
    // The small shear is retained rather than idealized to orthogonal axes.
    REQUIRE(std::abs(std::abs(first.atoms.get_volume()) - std::abs(original.get_volume())) < 1e-8);
    const Eigen::Matrix3d metric = first.atoms.cell_ * first.atoms.cell_.transpose();
    REQUIRE(std::max({std::abs(metric(0, 1)), std::abs(metric(0, 2)), std::abs(metric(1, 2))}) > 1e-6);
}

TEST_CASE("Cell-aware comparison respects partial and absent periodicity", "[equivalence]") {
    const int dimensions = GENERATE(0, 1, 2);
    StructureComparisonOptions options;
    options.use_descriptors = GENERATE(false, true);
    StructureComparator comparator(options);
    auto original = crystal();
    for (int i = 0; i < 3; ++i) original.pbc_[i] = i < dimensions;
    const auto first = comparator.prepare(original);
    auto fractional = original;
    fractional.to_fractional();
    REQUIRE(comparator.equivalent(first, comparator.prepare(fractional)));
    auto other = original;
    for (auto &position : other.pos_) position += Eigen::RowVector3d(10, -11, 14);
    if (dimensions) other.pos_[0] += original.cell_.row(0);
    other.wrap();
    const auto rotation = Eigen::AngleAxisd(.61, Eigen::Vector3d(2, 3, 1).normalized()).toRotationMatrix();
    other.cell_ *= rotation;
    for (auto &position : other.pos_) position *= rotation;
    other.swap_axis(231);
    std::reverse(other.pos_.begin(), other.pos_.end());
    std::reverse(other.chemmap_.begin(), other.chemmap_.end());
    const auto second = comparator.prepare(other);
    REQUIRE(comparator.equivalent(first, second));
    REQUIRE(comparator.equivalent(second, first));
    other = original;
    // A single atom shifted by a nonperiodic cell vector is not an image.
    other.pos_[0] += original.cell_.row(2);
    REQUIRE_FALSE(comparator.equivalent(first, comparator.prepare(other)));
    other = original;
    other.cell_.row(2) *= 2;
    REQUIRE_FALSE(comparator.equivalent(first, comparator.prepare(other)));
    other = original;
    other.pbc_[2] = true;
    REQUIRE_FALSE(comparator.equivalent(first, comparator.prepare(other)));
}

TEST_CASE("Lattice symmetry matching has an explicit mirror policy", "[equivalence]") {
    const bool periodic = GENERATE(false, true);
    const bool descriptors = GENERATE(false, true);
    Atoms original;
    original.cell_ = 12 * Eigen::Matrix3d::Identity();
    original.pbc_ = {periodic, periodic, periodic};
    original.pos_ = {{1, 1, 1}, {2.2, 1.4, 1.8}, {3.4, 3.1, 2.5}, {1.7, 4.3, 3.8}};
    original.chemmap_ = {0, 1, 2, 3};
    auto other = original;
    // Rotate atoms within the same cubic cell, exercising point group matching.
    for (auto &position : other.pos_) {
        const auto old = position.eval();
        position << -old[1], old[0], old[2];
    }
    StructureComparisonOptions options;
    options.use_descriptors = descriptors;
    options.allow_reflections = false;
    StructureComparator proper(options);
    REQUIRE(proper.equivalent(proper.prepare(original), proper.prepare(other)));
    for (auto &position : other.pos_) position[2] *= -1;
    REQUIRE_FALSE(proper.equivalent(proper.prepare(original), proper.prepare(other)));
    options.allow_reflections = true;
    StructureComparator mirrors(options);
    REQUIRE(mirrors.equivalent(mirrors.prepare(original), mirrors.prepare(other)));
}

TEST_CASE("Comparison validates geometry before reduction and strips per-atom results", "[equivalence]") {
    StructureComparator comparator;
    auto invalid = crystal();
    invalid.chemmap_.pop_back();
    REQUIRE_THROWS_AS(comparator.prepare(invalid), std::invalid_argument);
    invalid = crystal();
    invalid.pos_[0][0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(comparator.prepare(invalid), std::invalid_argument);
    invalid = crystal();
    invalid.cell_.row(2).setZero();
    REQUIRE_THROWS_AS(comparator.prepare(invalid), std::invalid_argument);
    REQUIRE_THROWS_AS(comparator.prepare(Atoms{}), std::invalid_argument);
    auto original = crystal();
    original.mksuper(Eigen::Vector3i(2, 1, 1));
    original.local_attribs_v_["forces"] = Eigen::MatrixXd::Ones(3, original.size());
    const auto prepared = comparator.prepare(original);
    REQUIRE(prepared.atoms.size() == 3);
    REQUIRE(prepared.atoms.local_attribs_v_.empty());
    REQUIRE(original.size() == 6);
    REQUIRE(original.local_attribs_v_.at("forces").cols() == 6);
}

TEST_CASE("Position matching reassigns overlapping tolerance spheres bijectively", "[equivalence]") {
    StructureComparisonOptions options;
    options.use_descriptors = false;
    StructureComparator comparator(options);
    Atoms original;
    original.pbc_ = {false, false, false};
    original.cell_ = Eigen::Vector3d(20, 21, 22).asDiagonal();
    original.pos_ = {{1, 2, 3}, {1.18, 2, 3}, {4.7, 5.4, 6.1}};
    original.chemmap_ = {0, 0, 1};
    auto other = original;
    other.pos_[0][0] = 1.09;
    other.pos_[1][0] = 1.01;
    REQUIRE(comparator.equivalent(comparator.prepare(original), comparator.prepare(other)));
    other.pos_[0][0] = 1.02;
    other.pos_[1][0] = 1.04;
    REQUIRE_FALSE(comparator.equivalent(comparator.prepare(original), comparator.prepare(other)));
}

TEST_CASE("Lattice operations cannot exchange periodic and open directions", "[equivalence]") {
    StructureComparisonOptions options;
    options.use_descriptors = GENERATE(false, true);
    StructureComparator comparator(options);
    Atoms original;
    original.cell_ = 12 * Eigen::Matrix3d::Identity();
    original.pbc_ = {true, true, false};
    original.pos_ = {{1, 1, 1}, {2.2, 1.4, 1.8}, {3.4, 3.1, 2.5}, {1.7, 4.3, 3.8}};
    original.chemmap_ = {0, 1, 2, 3};
    auto other = original;
    for (auto &position : other.pos_) {
        const auto old = position.eval();
        position << old[2], old[1], -old[0];
    }
    REQUIRE_FALSE(comparator.equivalent(comparator.prepare(original), comparator.prepare(other)));
    REQUIRE_FALSE(comparator.equivalent(comparator.prepare(other), comparator.prepare(original)));
}
