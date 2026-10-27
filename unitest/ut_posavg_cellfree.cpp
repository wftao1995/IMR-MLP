#include <catch2/catch_all.hpp>
#include "analyse/compute_posavg.h"
#include "analyse/trajectory_transform.h"

namespace {
std::vector<Atoms> molecular_frames(int cell_kind) {
    Atoms first; first.pbc_ = {false, false, false}; first.cell_.setZero();
    if (cell_kind) first.cell_ << 10, 0, 0, 2, 11, 0, 1, 3, 12;
    first.pos_ = {{1, 2, 3}, {3, 4, 5}}; first.chemmap_ = {0, 1};
    auto second = first;
    second.pos_ = {{1.2, 2.4, 3.6}, {3.8, 3.8, 5.2}};
    if (cell_kind == 2) { first.to_fractional(); second.to_fractional(); }
    return {first, second};
}
}

TEST_CASE("NPBC molecular averaging removes unequal-mass drift without a cell", "[posavg][cellfree]")
{
    const int cell_kind = GENERATE(0, 1, 2); // zero Cartesian, full Cartesian, full fractional
    const bool remove_drift = GENERATE(false, true);
    const auto frames = molecular_frames(cell_kind);
    const auto original = frames;
    PositionAverageOptions options; options.remove_drift = remove_drift;
    options.masses = {1, 3}; options.geometry_only = true;
    ComputePosavg average(options); average.init(frames[0]);
    for (const auto &frame : frames) average.accumulate(frame);
    average.compute();
    auto result = average.atomacc;
    if (!result.cartesian) result.to_cartesian();
    // COM step=(.65,-.05,.3), distinct internal steps remain after removing it.
    const Eigen::RowVector3d expected_first = remove_drift
        ? Eigen::RowVector3d(.775, 2.225, 3.15) : Eigen::RowVector3d(1.1, 2.2, 3.3);
    const Eigen::RowVector3d expected_second = remove_drift
        ? Eigen::RowVector3d(3.075, 3.925, 4.95) : Eigen::RowVector3d(3.4, 3.9, 5.1);
    REQUIRE(result.pos_[0].isApprox(expected_first, 1e-12));
    REQUIRE(result.pos_[1].isApprox(expected_second, 1e-12));
    REQUIRE(result.cell_.isApprox(frames[0].cell_));
    REQUIRE(result.is_nonperiodic());
    for (int f = 0; f < 2; ++f) {
        REQUIRE(frames[f].cartesian == original[f].cartesian);
        for (int i = 0; i < 2; ++i) REQUIRE(frames[f].pos_[i].isApprox(original[f].pos_[i]));
    }
}

TEST_CASE("NPBC minimum-image displacement retains internal motion and COM semantics", "[trajectory][cellfree]")
{
    const bool remove_drift = GENERATE(false, true);
    const auto frames = molecular_frames(0);
    const auto displacement = minimum_image_displacement(frames[0], frames[1], {1, 3}, remove_drift);
    Eigen::VectorXd expected(6);
    if (remove_drift) expected << -.45, .45, .3, .15, -.15, -.1;
    else expected << .2, .4, .6, .8, -.2, .2;
    REQUIRE(displacement.allFinite());
    REQUIRE(displacement.isApprox(expected, 1e-12));
    // The stricter finite-temperature FC path still requires a true fixed cell.
    REQUIRE_THROWS_WITH(validate_fixed_cell_trajectory(frames), Catch::Matchers::ContainsSubstring("singular cell"));
}
