#include <catch2/catch_all.hpp>
#include "analyse/compute_fftfilter.h"
#include "analyse/compute_sro.h"
#include "apps/relax_masks.h"
#include <sstream>

namespace {
Atoms mask_atoms(int count) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 10;
    atom.pos_.assign(count, Eigen::RowVector3d(1, 2, 3));
    atom.chemmap_.assign(count, 0);
    atom.sel_dyncell_.setZero();
    atom.sel_dynpos_.assign(count, Eigen::RowVector3d::Zero());
    return atom;
}

void require_masks_unchanged(const std::vector<Atoms> &actual,
                             const std::vector<Atoms> &expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i].sel_dyncell_.isApprox(expected[i].sel_dyncell_));
        REQUIRE(actual[i].sel_dynpos_.size() == expected[i].sel_dynpos_.size());
        for (size_t j = 0; j < actual[i].sel_dynpos_.size(); ++j)
            REQUIRE(actual[i].sel_dynpos_[j].isApprox(expected[i].sel_dynpos_[j]));
    }
}
}

TEST_CASE("Seldyn masks preserve sparse atoms and map all three position components", "[entrypoints][relax][seldyn]") {
    std::vector<Atoms> atoms{mask_atoms(2), mask_atoms(1)};
    std::istringstream input(
        "# two configurations in reversed order\n2 ! count\n"
        "2 1\n1 0 0\n0 1 0\n0 0 1\n0 0 1 0\n"
        "1 2\n1 1 1\n1 1 1\n1 1 1\n1 1 0 1\n");
    AppsRelax::read_masks(input, atoms);
    REQUIRE(atoms[0].sel_dyncell_.isApprox(Eigen::Matrix3d::Ones()));
    REQUIRE(atoms[1].sel_dyncell_.isApprox(Eigen::Matrix3d::Identity()));
    REQUIRE(atoms[0].sel_dynpos_[0].isApprox(Eigen::RowVector3d::Ones()));
    REQUIRE(atoms[0].sel_dynpos_[1].isApprox(Eigen::RowVector3d(1, 0, 1)));
    REQUIRE(atoms[1].sel_dynpos_[0].isApprox(Eigen::RowVector3d(0, 1, 0)));
    REQUIRE(atoms[0].pos_[0].isApprox(Eigen::RowVector3d(1, 2, 3)));
}

TEST_CASE("Seldyn permits a configuration without explicit atomic masks", "[entrypoints][relax][seldyn]") {
    std::vector<Atoms> atoms{mask_atoms(2)};
    std::istringstream input("1\n1 2\n1 0 0\n0 1 0\n0 0 1\n");
    AppsRelax::read_masks(input, atoms);
    for (const auto &mask : atoms[0].sel_dynpos_)
        REQUIRE(mask.isApprox(Eigen::RowVector3d::Ones()));
}

TEST_CASE("Seldyn rejects malformed indices dimensions and values without partial updates", "[entrypoints][relax][seldyn][errors]") {
    const std::string text = GENERATE(std::string(""),
        "-1\n", "2\n", "1x\n", "999999999999999999999\n", "1 2\n",
        "1\n0 2\n", "1\n2 2\n", "1\n1 -1\n", "1\n1 1\n", "1\n1 2x\n",
        "1\n1 2\n1 0 0\n0 1 0\n",
        "1\n1 2\n1 0 0 0\n0 1 0\n0 0 1\n",
        "1\n1 2\n1 0 0\n0 nan 0\n0 0 1\n",
        "1\n1 2\n1 0 0\n0 1 0\n0 0 1\n1 0 0\n",
        "1\n1 2\n1 0 0\n0 1 0\n0 0 1\n-1 1 0 1\n",
        "1\n1 2\n1 0 0\n0 1 0\n0 0 1\n2 1 0 1\n",
        "1\n1 2\n1 0 0\n0 1 0\n0 0 1\n0 1 0 inf\n",
        "1\n1 2\n1 0 0\n0 1 0\n0 0 1\n0 1 0 1x\n",
        "1\n1 2\n1 0 0\n0 1 0\n0 0 1\n0 1 0 1\n0 0 1 0\n",
        "1\n1 2\n1 0 0\n0 1 0\n0 0 1\n1 2\n",
        "1\n1 2\n1 0 0\n0 1 0\n0 0 1\nextra\n");
    INFO(text);
    std::vector<Atoms> atoms{mask_atoms(2)};
    const auto original = atoms;
    std::istringstream input(text);
    REQUIRE_THROWS_AS(AppsRelax::read_masks(input, atoms), std::invalid_argument);
    require_masks_unchanged(atoms, original);
}

TEST_CASE("Seldyn rejects missing or duplicate configurations after a valid record", "[entrypoints][relax][seldyn][errors]") {
    std::vector<Atoms> atoms{mask_atoms(2), mask_atoms(2)};
    const auto original = atoms;
    const std::string first = "2\n1 2\n1 0 0\n0 1 0\n0 0 1\n0 1 0 1\n";
    const std::string second = GENERATE(std::string(""), "1 2\n1 0 0\n0 1 0\n0 0 1\n");
    std::istringstream input(first + second);
    REQUIRE_THROWS_AS(AppsRelax::read_masks(input, atoms), std::invalid_argument);
    require_masks_unchanged(atoms, original);
}

TEST_CASE("SRO and TSRO reject unsupported definitions instead of returning zero", "[entrypoints][sro][errors]") {
    const std::string definition = GENERATE("GM", "WCMD", "invalid", "wc");
    GPSettings settings;
    settings.Modify("sro_type", definition);
    REQUIRE_THROWS_AS(ComputeSRO(settings), std::invalid_argument);
    REQUIRE_THROWS_AS(parse_sro_type(definition), std::invalid_argument);
}

TEST_CASE("SRO value calculation rejects unsupported enum values", "[entrypoints][sro][errors]") {
    const auto definition = static_cast<decltype(WC)>(99);
    REQUIRE_THROWS_AS(compute_sro_value(definition, 1, 4, .5, .5, false), std::invalid_argument);
}

TEST_CASE("Implemented SRO definitions retain their formulas and default constructors", "[entrypoints][sro]") {
    REQUIRE(parse_sro_type("WC") == WC);
    REQUIRE(parse_sro_type("PM") == PM);
    REQUIRE(compute_sro_value(WC, 0, 4, .5, .5, true) == Catch::Approx(1));
    REQUIRE(compute_sro_value(PM, 0, 4, .5, .5, true) == Catch::Approx(-1));
    REQUIRE_NOTHROW(ComputeSRO(GPSettings{}));
}



TEST_CASE("Legacy FFT rejects velocity reconstruction without changing the trajectory", "[entrypoints][fft][errors]") {
    std::vector<Atoms> atoms{mask_atoms(1), mask_atoms(1)};
    atoms[1].pos_[0](0) = 11;
    atoms[0].velocities_ = {Eigen::RowVector3d(1, 2, 3)};
    atoms[1].velocities_ = {Eigen::RowVector3d(4, 5, 6)};
    const auto original = atoms;
    ComputeSmoothFFT smoother;
    smoother.rebuild_velo = true;
    REQUIRE_THROWS_WITH(smoother.compute(atoms, 1, ComputeSmoothFFT::BLOCK),
                        Catch::Matchers::ContainsSubstring("use compute2"));
    for (size_t i = 0; i < atoms.size(); ++i) {
        REQUIRE(atoms[i].pos_[0].isApprox(original[i].pos_[0]));
        REQUIRE(atoms[i].cell_.isApprox(original[i].cell_));
        REQUIRE(atoms[i].velocities_[0].isApprox(original[i].velocities_[0]));
    }
}
