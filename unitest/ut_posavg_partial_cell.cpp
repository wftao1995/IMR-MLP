#include <catch2/catch_all.hpp>
#include "analyse/compute_posavg.h"
#include "analyse/trajectory_transform.h"

namespace {
std::vector<Atoms> slab_frames(bool nonprefix, bool padded) {
    Atoms first; first.cell_.setZero(); first.cell_.row(0) << 10, 0, 0;
    const int y_axis = nonprefix ? 2 : 1, open_axis = nonprefix ? 1 : 2;
    first.cell_.row(y_axis) << 0, 10, 0;
    if (padded) first.cell_.row(open_axis) << 0, 0, 20;
    first.pbc_ = {false, false, false}; first.pbc_[0] = first.pbc_[y_axis] = true;
    first.pos_ = {{9.8, 1, 3}, {2, 2, 4}}; first.chemmap_ = {0, 0};
    auto second = first; second.pos_ = {{.2, 1.2, 3.4}, {2.8, 2.6, 4.8}};
    return {first, second};
}
}

TEST_CASE("Cartesian slab averaging keeps open-direction motion across periodic crossings", "[posavg][partial_cell]")
{
    const bool nonprefix = GENERATE(false, true);
    const bool padded = GENERATE(false, true);
    const bool remove_drift = GENERATE(false, true);
    const auto frames = slab_frames(nonprefix, padded);
    PositionAverageOptions options; options.remove_drift = remove_drift; options.masses = {12, 12};
    ComputePosavg average(options); average.init(frames[0]);
    for (const auto &frame : frames) average.accumulate(frame);
    average.compute();
    const std::vector<Eigen::RowVector3d> expected = remove_drift
        ? std::vector<Eigen::RowVector3d>{{9.7, .9, 2.9}, {2.1, 2.1, 4.1}}
        : std::vector<Eigen::RowVector3d>{{10, 1.1, 3.2}, {2.4, 2.3, 4.4}};
    for (int i = 0; i < 2; ++i) REQUIRE(average.atomacc.pos_[i].isApprox(expected[i], 1e-12));
    REQUIRE(average.atomacc.cartesian);
    REQUIRE(average.atomacc.cell_.isApprox(frames[0].cell_));
    REQUIRE(average.atomacc.chemmap_ == frames[0].chemmap_);
    REQUIRE(average.atomacc.pbc_ == frames[0].pbc_);
    REQUIRE(frames[0].pos_[0].isApprox(Eigen::RowVector3d(9.8, 1, 3)));
    REQUIRE(frames[1].pos_[0].isApprox(Eigen::RowVector3d(.2, 1.2, 3.4)));
}

TEST_CASE("Wrapped Cartesian slab average retains winding on continued accumulation", "[posavg][partial_cell]")
{
    auto frames = slab_frames(false, false); auto atom = frames[0];
    atom.pos_.resize(1); atom.chemmap_ = {0};
    PositionAverageOptions options; options.wrap_result = true;
    ComputePosavg average(options); average.init(atom); average.accumulate(atom);
    atom.pos_[0] << .4, 1, 4; average.accumulate(atom); average.compute();
    REQUIRE(average.atomacc.pos_[0].isApprox(Eigen::RowVector3d(.1, 1, 3.5), 1e-11));
    atom.pos_[0] << 1, 1, 5; average.accumulate(atom); average.compute();
    REQUIRE(average.atomacc.pos_[0].isApprox(Eigen::RowVector3d(.4, 1, 4), 1e-11));
}

TEST_CASE("Shared periodic coordinate map retains the full-cell skew convention", "[trajectory][partial_cell]")
{
    Atoms reference; reference.cell_ << 4, 0, 0, 1, 3, 0, 2, 1, 5;
    reference.pbc_ = {true, true, false}; reference.pos_ = {{0, 0, 0}}; reference.chemmap_ = {0};
    auto current = reference; current.pos_[0] << 5.7, -.5, 8;
    REQUIRE((reference.cell_ * trajectory_periodic_coordinate_map(reference)).isApprox(Eigen::Matrix3d::Identity()));
    const auto step = minimum_image_displacement(reference, current, {1}, false);
    REQUIRE(step.isApprox(Eigen::Vector3d(2.7, 2.5, 8), 1e-12));
}

TEST_CASE("Partial-cell averaging keeps fractional and FC boundaries and rejects dependent rows", "[posavg][partial_cell]")
{
    auto frames = slab_frames(false, false);
    SECTION("fractional averaging still needs a full cell") {
        PositionAverageOptions options; options.coordinates = AverageCoordinates::Fractional;
        ComputePosavg average(options); average.init(frames[0]);
        REQUIRE_THROWS_WITH(average.accumulate(frames[0]), Catch::Matchers::ContainsSubstring("nonsingular cell"));
    }
    SECTION("stored fractional input still needs a full cell") {
        frames[0].cartesian = false; ComputePosavg average; average.init(frames[0]);
        REQUIRE_THROWS_WITH(average.accumulate(frames[0]), Catch::Matchers::ContainsSubstring("nonsingular cell"));
    }
    SECTION("dependent periodic rows remain invalid") {
        frames[0].cell_.row(1) = 2 * frames[0].cell_.row(0);
        ComputePosavg average; average.init(frames[0]);
        REQUIRE_THROWS_WITH(average.accumulate(frames[0]), Catch::Matchers::ContainsSubstring("independent periodic"));
    }
    SECTION("FC fixed-cell validation is unchanged") {
        REQUIRE_THROWS_WITH(validate_fixed_cell_trajectory(frames), Catch::Matchers::ContainsSubstring("singular cell"));
    }
}
