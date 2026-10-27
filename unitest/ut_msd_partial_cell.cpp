#include <catch2/catch_all.hpp>
#include "analyse/compute_msd.h"
#include <cmath>

namespace {
std::vector<Atoms> slab_trajectory(int cells, bool nonprefix, bool changing = false) {
    const Eigen::RowVector3d a0(4, 0, 0), b0(1, 3, 0);
    const Eigen::RowVector3d starts[] = {.9 * a0 + .8 * b0 + Eigen::RowVector3d(0, 0, 1),
                                       .2 * a0 + .3 * b0 + Eigen::RowVector3d(0, 0, -2)};
    const Eigen::RowVector3d velocities[] = {{.6, .4, .7}, {-.3, .2, -.5}};
    std::vector<Atoms> frames;
    for (int f = 0; f < 8; ++f) {
        const double scale = changing ? 1 + .02 * f : 1.;
        const Eigen::RowVector3d a = scale * a0, b = scale * b0;
        Atoms atom;
        atom.cell_.row(0) = a;
        atom.cell_.row(1) = b;
        atom.cell_.row(2) << 0, 0, cells == 0 ? 0. : 20.;
        atom.pbc_ = {true, true, false};
        if (nonprefix) {
            atom.cell_.row(1).swap(atom.cell_.row(2));
            atom.pbc_ = {true, false, true};
        }
        for (int i = 0; i < 2; ++i) {
            Eigen::RowVector3d position = starts[i] + f * velocities[i];
            const double fb = position[1] / (3 * scale);
            const double fa = (position[0] - fb * scale) / (4 * scale);
            position -= std::floor(fa) * a + std::floor(fb) * b;
            atom.pos_.push_back(position);
        }
        atom.chemmap_ = {0, 0};
        atom.features_["keep"] = "source";
        if (cells == 2) atom.to_fractional();
        frames.push_back(atom);
    }
    return frames;
}
GPSettings msd_settings(bool fft, bool tacavg) {
    GPSettings settings;
    settings.Modify("fft", fft ? "true" : "false");
    settings.Modify("tacavg", tacavg ? "true" : "false");
    settings.Modify("timestep", "1");
    settings.Modify("drift_cm", "false");
    settings.Modify("der", "false");
    settings.Modify("do_lfit", "false");
    settings.Modify("total", "true");
    settings.Modify("projection", "true");
    return settings;
}
void check_ballistic(ComputeMSD &msd) {
    REQUIRE(msd.msd_total.size() == 8);
    for (int lag = 0; lag < 8; ++lag) {
        const double t2 = lag * lag;
        REQUIRE(msd.msd_total[lag] == Catch::Approx(.695 * t2).margin(1e-10));
        REQUIRE(msd.msd_total_x[lag] == Catch::Approx(.225 * t2).margin(1e-10));
        REQUIRE(msd.msd_total_y[lag] == Catch::Approx(.1 * t2).margin(1e-10));
        REQUIRE(msd.msd_total_z[lag] == Catch::Approx(.37 * t2).margin(1e-10));
    }
}
}

TEST_CASE("MSD slab zero open rows retain independent ballistic motion and winding", "[msd][partialcell]") {
    const auto fft = GENERATE(false, true);
    const auto tacavg = GENERATE(false, true);
    const auto nonprefix = GENERATE(false, true);
    const auto cells = GENERATE(0, 1, 2); // Cartesian zero-open, Cartesian padding, full-rank fractional.
    const auto changing = GENERATE(false, true);
    const auto frames = slab_trajectory(cells, nonprefix, changing);
    const auto before = frames;
    ComputeMSD msd(msd_settings(fft, tacavg));
    msd.compute(frames);
    check_ballistic(msd);
    for (int f = 0; f < frames.size(); ++f) {
        REQUIRE(frames[f].pos_ == before[f].pos_);
        REQUIRE(frames[f].cell_ == before[f].cell_);
        REQUIRE(frames[f].pbc_ == before[f].pbc_);
        REQUIRE(frames[f].cartesian == before[f].cartesian);
        REQUIRE(frames[f].chemmap_ == before[f].chemmap_);
        REQUIRE(frames[f].features_ == before[f].features_);
    }
}

TEST_CASE("MSD slab support retains periodic-basis fractional and FSC restrictions", "[msd][partialcell]") {
    auto frames = slab_trajectory(0, false);
    auto settings = msd_settings(false, true);
    SECTION("dependent periodic basis") {
        for (auto &frame : frames) frame.cell_.row(1) = 2 * frame.cell_.row(0);
    }
    SECTION("singular fractional cell") { frames[0].cartesian = false; }
    SECTION("finite-size correction remains fully periodic") {
        settings.Modify("fsc_mode", "1");
        settings.Modify("do_lfit", "true");
    }
    ComputeMSD msd(settings);
    REQUIRE_THROWS_AS(msd.compute(frames), std::invalid_argument);
}

TEST_CASE("MSD mixed full-rank NPBC frames retain their original fractional winding reference", "[msd][partialcell]") {
    std::vector<Atoms> frames(3);
    const double x[] = {3.6, .4, 1.2};
    for (int f = 0; f < 3; ++f) {
        frames[f].cell_ = 4. * Eigen::Matrix3d::Identity();
        frames[f].pos_ = {{x[f], 1, 2}};
        frames[f].chemmap_ = {0};
        frames[f].pbc_ = {f != 0, false, false};
    }
    ComputeMSD msd(msd_settings(false, true));
    msd.compute(frames);
    REQUIRE(msd.msd_total.size() == 3);
    for (int lag = 0; lag < 3; ++lag)
        REQUIRE(msd.msd_total[lag] == Catch::Approx(.64 * lag * lag).margin(1e-12));
}
