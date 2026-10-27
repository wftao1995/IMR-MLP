#include <catch2/catch_all.hpp>
#include "analyse/compute_msd.h"

TEST_CASE("Cell-free NPBC MSD follows the independent ballistic lag squared law", "[msd][cellfree]")
{
    const bool fft = GENERATE(false, true);
    const bool time_average = GENERATE(false, true);
    const int cells = GENERATE(0, 1, 2, 3); // zero Cartesian, padded, fractional, mixed zero/fractional
    constexpr double dt = 2;
    std::vector<Atoms> frames(8);
    for (int f = 0; f < 8; ++f) {
        auto &frame = frames[f]; frame.pbc_ = {false, false, false};
        frame.cell_.setZero();
        if (cells == 1 || cells == 2 || (cells == 3 && f % 2))
            frame.cell_ << 40, 0, 0, 2, 50, 0, 3, 4, 60;
        frame.pos_ = {{1 + .2 * dt * f, 2 + .3 * dt * f, 3 + .4 * dt * f}};
        frame.chemmap_ = {0};
        if (cells == 2 || (cells == 3 && f % 2)) frame.to_fractional();
    }
    const auto original = frames;
    GPSettings settings;
    settings.Modify("fft", fft ? "true" : "false");
    settings.Modify("tacavg", time_average ? "true" : "false");
    settings.Modify("timestep", "2"); settings.Modify("drift_cm", "false");
    settings.Modify("der", "false"); settings.Modify("do_lfit", "false");
    settings.Modify("total", "true"); settings.Modify("projection", "true");
    ComputeMSD msd(settings); msd.compute(frames);
    REQUIRE(msd.msd_total.size() == 8);
    for (int lag = 0; lag < 8; ++lag) {
        const double time2 = lag * dt * lag * dt;
        REQUIRE(msd.msd_total[lag] == Catch::Approx(.29 * time2).margin(1e-10));
        REQUIRE(msd.msd_total_x[lag] == Catch::Approx(.04 * time2).margin(1e-10));
        REQUIRE(msd.msd_total_y[lag] == Catch::Approx(.09 * time2).margin(1e-10));
        REQUIRE(msd.msd_total_z[lag] == Catch::Approx(.16 * time2).margin(1e-10));
    }
    for (int f = 0; f < 8; ++f) {
        REQUIRE(frames[f].pos_[0].isApprox(original[f].pos_[0]));
        REQUIRE(frames[f].cell_.isApprox(original[f].cell_));
        REQUIRE(frames[f].cartesian == original[f].cartesian);
        REQUIRE(frames[f].pbc_ == original[f].pbc_);
    }
}

TEST_CASE("MSD rejects singular fractional input and missing periodic vectors", "[msd][cellfree]")
{
    const bool fractional = GENERATE(false, true);
    std::vector<Atoms> frames(2);
    for (auto &frame : frames) {
        frame.cell_.setZero(); frame.pbc_ = {false, false, false};
        frame.pos_ = {{1, 2, 3}}; frame.chemmap_ = {0};
    }
    if (fractional) frames[0].cartesian = false;
    else frames[1].pbc_ = {true, false, false};
    ComputeMSD msd;
    REQUIRE_THROWS_WITH(msd.compute(frames), Catch::Matchers::ContainsSubstring(
        fractional ? "non-singular cell" : "independent periodic cell vectors"));
}
