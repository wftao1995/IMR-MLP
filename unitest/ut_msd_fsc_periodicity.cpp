#include <catch2/catch_all.hpp>
#include "analyse/compute_msd.h"
#include <cmath>

TEST_CASE("MSD periodic finite-size correction preserves the full-PBC physical formula", "[msd][fsc][pbc]")
{
    const int mode = GENERATE(0, 1, 2);
    const int boundary = GENERATE(0, 1, 2, 3); // TTT, FFF, TTF, later-frame TTF
    std::vector<Atoms> frames(4);
    for (int f = 0; f < 4; ++f) {
        frames[f].cell_ = 10 * Eigen::Matrix3d::Identity();
        frames[f].pos_ = {{std::sqrt(6. * f), 0, 0}}; frames[f].chemmap_ = {0};
        frames[f].pbc_ = {true, true, true};
        if (boundary == 1) frames[f].pbc_ = {false, false, false};
        if (boundary == 2 || (boundary == 3 && f == 3)) frames[f].pbc_[2] = false;
    }
    GPSettings settings;
    settings.Modify("fft", "false"); settings.Modify("tacavg", "false");
    settings.Modify("drift_cm", "false"); settings.Modify("do_lfit", "true");
    settings.Modify("timestep", "1"); settings.Modify("fit_start", "1");
    settings.Modify("fit_end", "3"); settings.Modify("fsc_mode", std::to_string(mode));
    settings.Modify("temperature", "300"); settings.Modify("viscosity", "1");
    ComputeMSD msd(settings);
    if (mode > 0 && boundary > 0) {
        REQUIRE_THROWS_WITH(msd.compute(frames), Catch::Matchers::ContainsSubstring("fully periodic boundaries in every frame"));
        return;
    }
    msd.compute(frames);
    const double correction = mode == 0 ? 0 :
        8.617333262e-5 * 300 * 2.837298 /
        (6 * std::acos(-1.) * 6.241509074e-3 * 10);
    REQUIRE(msd.k_total == Catch::Approx(6).margin(1e-10));
    REQUIRE(msd.diffusion_total.D_PBC == Catch::Approx(1).margin(1e-10));
    REQUIRE(msd.diffusion_total.D_correction == Catch::Approx(correction).margin(1e-12));
    REQUIRE(msd.diffusion_total.D_infinite == Catch::Approx(1 + correction).margin(1e-10));
}
