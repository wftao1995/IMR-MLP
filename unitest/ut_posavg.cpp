// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_posavg.h"

using Catch::Approx;

TEST_CASE("Average positions preserve a single frame's cell", "[posavg]")
{
    Atoms atom;
    atom.cell_ << 10, 0, 0, 2, 11, 0, 1, 3, 12;
    atom.pbc_ = {true, true, false};
    atom.pos_ = {{1, 2, 3}};
    atom.chemmap_ = {0};
    ComputePosavg avg;
    avg.init(atom);
    avg.accumulate(atom);
    avg.compute();
    REQUIRE(avg.atomacc.cell_.isApprox(atom.cell_));
    REQUIRE(avg.atomacc.pos_[0].isApprox(atom.pos_[0]));
    REQUIRE(avg.atomacc.pbc_ == atom.pbc_);
}

TEST_CASE("Average positions average changing cells and support continued accumulation", "[posavg]")
{
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 10.0;
    atom.pos_ = {{1, 0, 0}};
    atom.chemmap_ = {0};
    atom.site_energies_ = {2};
    ComputePosavg avg;
    REQUIRE_THROWS_AS(avg.compute(), std::invalid_argument);
    avg.init(atom);
    REQUIRE_THROWS_AS(avg.compute(), std::invalid_argument);
    avg.accumulate(atom);
    atom.cell_ *= 1.2;
    atom.pos_[0][0] = 3;
    atom.site_energies_[0] = 4;
    avg.accumulate(atom);
    avg.compute();
    REQUIRE(avg.atomacc.cell_.isApprox(Eigen::Matrix3d::Identity() * 11.0));
    REQUIRE(avg.atomacc.pos_[0][0] == Approx(2));
    REQUIRE(avg.atomacc.site_energies_[0] == Approx(3));
    atom.cell_ = Eigen::Matrix3d::Identity() * 14.0;
    atom.pos_[0][0] = 5;
    atom.site_energies_[0] = 6;
    avg.accumulate(atom);
    avg.compute();
    REQUIRE(avg.atomacc.cell_.isApprox(Eigen::Matrix3d::Identity() * 12.0));
    REQUIRE(avg.atomacc.pos_[0][0] == Approx(3));
    REQUIRE(avg.atomacc.site_energies_[0] == Approx(4));
}

TEST_CASE("Average positions follow repeated periodic crossings", "[posavg]")
{
    const bool periodic = GENERATE(false, true);
    const bool wrapped = GENERATE(false, true);
    Atoms atom;
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {periodic, false, false};
    atom.pos_ = {{9.8, 0, 0}};
    atom.chemmap_ = {0};
    const std::vector<double> positions = wrapped
        ? std::vector<double>{9.8, .2, 4.9, 9.6, .2}
        : std::vector<double>{9.8, 10.2, 14.9, 19.6, 20.2};
    ComputePosavg avg;
    avg.init(atom);
    for (double x : positions)
    {
        atom.pos_[0][0] = x;
        avg.accumulate(atom);
    }
    avg.compute();
    REQUIRE(avg.atomacc.pos_[0][0] == Approx(periodic || !wrapped ? 14.94 : 4.94));
}

TEST_CASE("Average positions unwrap only periodic lattice directions in changing skew cells", "[posavg]")
{
    Atoms atom;
    atom.cell_ << 10, 0, 0, 2, 8, 0, 1, 3, 12;
    const Eigen::Matrix3d cell = atom.cell_;
    atom.pbc_ = {true, false, true};
    atom.cartesian = GENERATE(false, true);
    atom.chemmap_ = {0};
    auto position = [&](const Eigen::RowVector3d &fractional, double scale) -> Eigen::RowVector3d {
        if (atom.cartesian)
            return fractional * (scale * cell);
        return fractional;
    };
    const auto first = position({.98, .98, .02}, 1);
    atom.pos_ = {first};
    ComputePosavg avg;
    avg.init(atom);
    avg.accumulate(atom);
    atom.cell_ = 1.2 * cell;
    atom.pos_[0] = position({.02, .02, .98}, 1.2);
    avg.accumulate(atom);
    avg.compute();
    const auto second = position({1.02, .02, -.02}, 1.2);
    REQUIRE(avg.atomacc.pos_[0].isApprox((first + second) / 2));
    REQUIRE(avg.atomacc.cell_.isApprox(1.1 * cell));

    atom.cell_ = 1.4 * cell;
    atom.pos_[0] = position({.08, .04, .92}, 1.4);
    avg.accumulate(atom);
    avg.compute();
    const auto third = position({1.08, .04, -.08}, 1.4);
    REQUIRE(avg.atomacc.pos_[0].isApprox((first + second + third) / 3));
    REQUIRE(avg.atomacc.cell_.isApprox(1.2 * cell));

    avg.init(atom);
    avg.accumulate(atom);
    avg.compute();
    REQUIRE(avg.atomacc.pos_[0].isApprox(atom.pos_[0]));
}

TEST_CASE("Position average explicitly selects Cartesian or fractional averaging", "[posavg]") {
    Atoms first;
    first.cell_ = 10 * Eigen::Matrix3d::Identity();
    first.pos_ = {{1, 0, 0}};
    first.chemmap_ = {0};
    auto second = first;
    second.cell_ *= 1.2;
    second.pos_[0][0] = 3;
    const bool fractional = GENERATE(false, true);
    PositionAverageOptions options;
    options.coordinates = fractional ? AverageCoordinates::Fractional : AverageCoordinates::Cartesian;
    ComputePosavg average(options);
    average.init(first);
    average.accumulate(first);
    average.accumulate(second);
    average.compute();
    REQUIRE(average.atomacc.pos_[0][0] == Approx(fractional ? 1.925 : 2.0));
    average.compute();
    REQUIRE(average.atomacc.pos_[0][0] == Approx(fractional ? 1.925 : 2.0));
}

TEST_CASE("Position average drift correction uses masses and rejects variable cells", "[posavg]") {
    Atoms first;
    first.cell_ = 10 * Eigen::Matrix3d::Identity();
    first.pos_ = {{1, 0, 0}, {3, 0, 0}};
    first.chemmap_ = {0, 1};
    auto second = first;
    second.pos_[0][0] += .2;
    second.pos_[1][0] += .8;
    const bool remove_drift = GENERATE(false, true);
    PositionAverageOptions options;
    options.remove_drift = remove_drift;
    options.masses = {1, 3};
    ComputePosavg average(options);
    average.init(first);
    average.accumulate(first);
    average.accumulate(second);
    average.compute();
    REQUIRE(average.atomacc.pos_[0][0] == Approx(remove_drift ? .775 : 1.1));
    REQUIRE(average.atomacc.pos_[1][0] == Approx(remove_drift ? 3.075 : 3.4));
    if (remove_drift) {
        second.cell_ *= 1.1;
        REQUIRE_THROWS_AS(average.accumulate(second), std::invalid_argument);
    }
}

TEST_CASE("Wrapped average output does not lose winding on further accumulation", "[posavg]") {
    Atoms atom;
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{9.8, 0, 0}};
    atom.chemmap_ = {0};
    PositionAverageOptions options;
    options.wrap_result = true;
    ComputePosavg average(options);
    average.init(atom);
    average.accumulate(atom);
    atom.pos_[0][0] = .2;
    average.accumulate(atom);
    average.compute();
    atom.pos_[0][0] = .6;
    average.accumulate(atom);
    average.compute();
    REQUIRE(average.atomacc.pos_[0][0] == Approx(.2));
}

TEST_CASE("Position averaging drops incomplete site-energy labels", "[posavg]") {
    Atoms first;
    first.cell_ = 10 * Eigen::Matrix3d::Identity();
    first.pos_ = {{1, 2, 3}};
    first.chemmap_ = {0};
    first.site_energies_ = {2};
    auto second = first;
    second.pos_[0][0] = 3;
    second.site_energies_.clear();
    ComputePosavg average;
    average.init(first);
    average.accumulate(first);
    average.accumulate(second);
    average.compute();
    REQUIRE(average.atomacc.site_energies_.empty());
    REQUIRE(average.atomacc.pos_[0][0] == Catch::Approx(2));
}

TEST_CASE("Position averaging accepts mixed input coordinate representations", "[posavg]") {
    Atoms first;
    first.cell_ = 10 * Eigen::Matrix3d::Identity();
    first.pos_ = {{1, 2, 3}};
    first.chemmap_ = {0};
    auto second = first;
    second.pos_[0][0] = 3;
    second.to_fractional();
    PositionAverageOptions options;
    options.coordinates = AverageCoordinates::Cartesian;
    ComputePosavg average(options);
    average.init(first);
    average.accumulate(first);
    average.accumulate(second);
    average.compute();
    REQUIRE(average.atomacc.cartesian);
    REQUIRE(average.atomacc.pos_[0].isApprox(Eigen::RowVector3d(2, 2, 3)));
    REQUIRE_FALSE(average.atomacc.has_energy());
}
