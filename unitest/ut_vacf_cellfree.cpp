#include <catch2/catch_all.hpp>
#include "analyse/compute_vacf.h"
#include <cmath>

TEST_CASE("Derived VACF needs only independent periodic cell vectors", "[vacf][cellfree]")
{
    const int algorithm = GENERATE(1, 2);
    const int cell_kind = GENERATE(0, 1, 2); // NPBC zero, TTF zero open row, padded TTF
    std::vector<Atoms> frames(8);
    for (int f = 0; f < 8; ++f) {
        auto &frame = frames[f];
        frame.cell_.setZero(); frame.pbc_ = {false, false, false};
        Eigen::RowVector3d position(3.9 + .2 * f, 2.9 + .3 * f, 1 + .4 * f);
        if (cell_kind > 0) {
            frame.pbc_ = {true, true, false};
            frame.cell_.row(0) << 4, 0, 0; frame.cell_.row(1) << 1, 3, 0;
            if (cell_kind == 2) frame.cell_.row(2) << 0, 0, 20;
            // Known periodic translations, independent of the MIC implementation.
            const double y_image = std::floor(position[1] / 3);
            position -= y_image * Eigen::RowVector3d(1, 3, 0);
            position[0] -= 4 * std::floor(position[0] / 4);
        }
        frame.pos_ = {position}; frame.chemmap_ = {0};
    }
    const auto source = frames;
    GPSettings settings;
    settings.Modify("algo", std::to_string(algorithm)); settings.Modify("timestep", "1");
    settings.Modify("vdos", "false"); settings.Modify("normalize", "false");
    settings.Modify("total", "true"); settings.Modify("types", "false");
    settings.Modify("ndim", "3");
    ComputeVACF vacf(settings); vacf.compute(frames);
    REQUIRE(vacf.vacf_isotropic.total.size() == 6);
    for (const auto &row : vacf.vacf_isotropic.total) {
        REQUIRE(row[0] == Catch::Approx(.04).margin(1e-12));
        REQUIRE(row[1] == Catch::Approx(.09).margin(1e-12));
        REQUIRE(row[2] == Catch::Approx(.16).margin(1e-12));
        REQUIRE(row[3] == Catch::Approx(.29 / 3).margin(1e-12));
    }
    for (int f = 0; f < 8; ++f) {
        REQUIRE(frames[f].pos_[0].isApprox(source[f].pos_[0]));
        REQUIRE(frames[f].cell_.isApprox(source[f].cell_));
        REQUIRE(frames[f].pbc_ == source[f].pbc_);
        REQUIRE(frames[f].velocities_.empty());
    }
}

TEST_CASE("Derived VACF still rejects dependent periodic cell vectors", "[vacf][cellfree]")
{
    std::vector<Atoms> frames(3);
    for (int f = 0; f < 3; ++f) {
        frames[f].cell_.setZero(); frames[f].cell_.row(0) << 2, 0, 0;
        frames[f].cell_.row(1) << 4, 0, 0; frames[f].pbc_ = {true, true, false};
        frames[f].pos_ = {{.1 * f, 0, 1}}; frames[f].chemmap_ = {0};
    }
    ComputeVACF vacf;
    REQUIRE_THROWS_WITH(vacf.compute(frames), Catch::Matchers::ContainsSubstring("independent finite periodic"));
}
