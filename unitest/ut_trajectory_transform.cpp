// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "analyse/trajectory_transform.h"

using Catch::Approx;

namespace {

Atoms make_frame(double first, double second)
{
    Atoms atoms;
    atoms.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atoms.pos_ = {Eigen::RowVector3d(first, 1.0, 1.0),
                  Eigen::RowVector3d(second, 1.0, 1.0)};
    atoms.chemmap_ = {0, 0};
    return atoms;
}

} // namespace

TEST_CASE("Fixed-cell centroid removes mass-weighted trajectory drift",
          "[trajectory][centroid]")
{
    std::vector<Atoms> frames{make_frame(1.0, 4.0), make_frame(1.5, 4.35)};
    const std::vector<double> masses{1.0, 2.0};
    const Atoms centroid = fixed_cell_centroid(frames, masses);
    REQUIRE(centroid.pos_[0](0) == Approx(1.05).margin(1e-12));
    REQUIRE(centroid.pos_[1](0) == Approx(3.975).margin(1e-12));
    REQUIRE(!centroid.has_forces());
}

TEST_CASE("Minimum-image displacement removes center-of-mass translation",
          "[trajectory][centroid]")
{
    const Atoms reference = make_frame(1.0, 4.0);
    const Atoms frame = make_frame(1.5, 4.35);
    const Eigen::VectorXd displacement = minimum_image_displacement(
        reference, frame, {1.0, 2.0});
    REQUIRE(displacement(0) == Approx(0.1).margin(1e-12));
    REQUIRE(displacement(3) == Approx(-0.05).margin(1e-12));
    REQUIRE(displacement.segment<3>(0).tail<2>().norm() == Approx(0.0));
    REQUIRE(displacement.segment<3>(3).tail<2>().norm() == Approx(0.0));
}

TEST_CASE("Fixed-cell trajectory validation rejects cell fluctuations",
          "[trajectory][centroid]")
{
    std::vector<Atoms> frames{make_frame(1.0, 4.0), make_frame(1.1, 4.1)};
    frames[1].cell_(0, 0) += 1e-3;
    REQUIRE_THROWS_AS(validate_fixed_cell_trajectory(frames), std::invalid_argument);
}

TEST_CASE("Crystal displacements are invariant to translations across half a cell",
          "[trajectory][centroid]")
{
    Atoms reference = make_frame(1.0, 4.0);
    reference.cell_ << 10.0, 0.0, 0.0, 2.0, 9.0, 0.0, 1.0, 0.5, 8.0;
    const std::vector<double> masses{1.0, 2.0};
    for (const Eigen::RowVector3d shift : {Eigen::RowVector3d(5.0, 0.0, 0.0),
                                          Eigen::RowVector3d(15.0, -8.0, 4.0)}) {
        Atoms frame = reference;
        frame.pos_[0] += shift + Eigen::RowVector3d(0.05, 0.02, -0.03);
        frame.pos_[1] += shift + Eigen::RowVector3d(-0.025, -0.01, 0.015);
        frame.wrap();
        const auto displacement = minimum_image_displacement(reference, frame, masses);
        REQUIRE(displacement.segment<3>(0).isApprox(Eigen::Vector3d(0.05, 0.02, -0.03), 1e-11));
        REQUIRE(displacement.segment<3>(3).isApprox(Eigen::Vector3d(-0.025, -0.01, 0.015), 1e-11));
    }
    const auto translated = minimum_image_displacement(reference, make_frame(1.5, 4.35), masses, false);
    REQUIRE(translated(0) == Approx(0.5));
    REQUIRE(translated(3) == Approx(0.35));
}

TEST_CASE("Fixed-cell centroid unwraps drift across half-cell image boundaries",
          "[trajectory][centroid]")
{
    const std::vector<double> masses{1.0, 2.0};
    std::vector<Atoms> frames;
    for (int i = 0; i < 5; ++i) {
        // Adjacent atoms cross the half-cell image boundary on opposite sides.
        Atoms frame = make_frame(1.0 + 5.05 * i, 4.0 + 4.975 * i);
        frame.wrap();
        frames.push_back(std::move(frame));
    }
    const auto centroid = fixed_cell_centroid(frames, masses);
    REQUIRE(centroid.pos_[0](0) == Approx(1.1).margin(1e-12));
    REQUIRE(centroid.pos_[1](0) == Approx(3.95).margin(1e-12));
    for (int i = 0; i < 5; ++i) {
        const auto displacement = minimum_image_displacement(centroid, frames[i], masses);
        REQUIRE(displacement(0) == Approx(0.05 * (i - 2)).margin(1e-12));
        REQUIRE(displacement(3) == Approx(-0.025 * (i - 2)).margin(1e-12));
    }
}

TEST_CASE("Trajectory unwrapping only shifts periodic lattice directions", "[trajectory][unwrap]") {
    const bool with_reference = GENERATE(false, true);
    const bool skew_cell = GENERATE(false, true);
    const bool periodic = GENERATE(false, true);
    Atoms first = make_frame(1, 2);
    if (skew_cell) first.cell_ << 10, 0, 0, 2, 9, 0, 1, 0.5, 8;
    first.pbc_ = {periodic, false, periodic};
    const auto position = [&](double x, double y, double z) -> Eigen::RowVector3d {
        return Eigen::RowVector3d(x, y, z) * first.cell_;
    };
    first.pos_ = {position(.9, .1, .9), position(.8, .2, .8)};
    Atoms second = first, third = first;
    second.pos_ = {position(.1, 1.6, .1), position(.0, 1.7, .0)};
    third.pos_ = {position(.3, 3.1, .3), position(.2, 3.2, .2)};
    std::vector<Atoms> frames = with_reference ? std::vector<Atoms>{second, third}
                                             : std::vector<Atoms>{first, second, third};
    if (with_reference) unwarp(first, frames);
    else unwarp(frames);
    const int offset = with_reference ? 0 : 1;
    const double image = periodic ? 1.0 : 0.0;
    REQUIRE(frames[offset].pos_[0].isApprox(position(.1 + image, 1.6, .1 + image), 1e-12));
    REQUIRE(frames[offset].pos_[1].isApprox(position(image, 1.7, image), 1e-12));
    REQUIRE(frames[offset + 1].pos_[0].isApprox(position(.3 + image, 3.1, .3 + image), 1e-12));
    REQUIRE(frames[offset + 1].pos_[1].isApprox(position(.2 + image, 3.2, .2 + image), 1e-12));
}
