#include <catch2/catch_all.hpp>
#include "mindriver/dimer_new.h"
#include "potentials/potential_lj.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
struct Workspace {
    std::filesystem::path previous = std::filesystem::current_path(), path;
    Workspace() {
        char name[] = "/tmp/imrmlp_dimer_cellfree_XXXXXX";
        const char *directory = mkdtemp(name);
        if (!directory) throw std::runtime_error("Cannot create Dimer test workspace");
        path = directory; std::filesystem::current_path(path);
        std::ofstream("modecar") << "GPML_MODECAR 1 2 fixed-cartesian\n-1 0 0\n1 0 0\n";
    }
    ~Workspace() { std::filesystem::current_path(previous); std::filesystem::remove_all(path); }
};
Atoms molecule(bool padding) {
    Atoms atom; atom.cell_.setZero(); atom.pbc_ = {false, false, false};
    if (padding) atom.cell_ = 20 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{0, 0, 0}, {1.3, 0, 0}}; atom.chemmap_ = {0, 0}; return atom;
}
GPSettings potential_settings() {
    GPSettings settings; settings.Modify("ntypes", "1"); settings.Modify("0-0", "1 1");
    settings.Modify("cut:0-0", "3"); return settings;
}
void configure(DimerDriver &driver, Potentials::AnyPotential &potential, int budget) {
    driver.pot = &potential; driver.max_iter = budget; driver.modecarf = "modecar";
    driver.rot_max_iter = driver.trn_max_iter = 2; driver.summary.clear();
}
double energy(double r) { return 4 * (std::pow(r, -12) - std::pow(r, -6)); }
double derivative(double r) { return 24 * (std::pow(r, -7) - 2 * std::pow(r, -13)); }
}

TEST_CASE("Fixed Cartesian Dimer preserves molecular curvature without a cell", "[dimer][cellfree]")
{
    Workspace workspace; const int budget = GENERATE(0, 2);
    auto settings = potential_settings(); Potentials::LJ612 potential(settings);
    auto input = molecule(false), padded = molecule(true);
    DimerDriver driver, control; configure(driver, potential, budget); configure(control, potential, budget);
    driver.init(input); control.init(padded);
    REQUIRE(driver.xx_center.allFinite()); REQUIRE(driver.ivrefcell.isZero());
    REQUIRE(driver.offset == 0); REQUIRE(driver.nlen == 6);
    driver.run(); control.run();
    REQUIRE(std::isfinite(driver.energy)); REQUIRE(std::isfinite(driver.curvature));
    REQUIRE(driver.energy == Catch::Approx(control.energy).margin(1e-12));
    REQUIRE(driver.curvature == Catch::Approx(control.curvature).margin(1e-10));
    const double radius = (driver.atom_center.pos_[1] - driver.atom_center.pos_[0]).norm();
    REQUIRE(driver.energy == Catch::Approx(energy(radius)).margin(1e-12));
    if (budget == 0) {
        const double shift = std::sqrt(2.) * driver.ddr;
        const double expected = std::sqrt(2.) * (derivative(1.3 + shift) - derivative(1.3 - shift)) / (2 * driver.ddr);
        REQUIRE(driver.curvature == Catch::Approx(expected).margin(1e-10));
    }
    REQUIRE(driver.atom_center.cell_.isZero()); REQUIRE(driver.atom_center.is_nonperiodic());
    for (int i = 0; i < 2; ++i) {
        REQUIRE(driver.atom_center.pos_[i].isApprox(control.atom_center.pos_[i], 1e-12));
        REQUIRE(driver.g_plus.segment<3>(3*i).allFinite());
    }
    REQUIRE(input.cell_.isZero()); REQUIRE(input.pos_[0].isZero()); REQUIRE(input.pos_[1][0] == 1.3);
    std::ifstream output("dimer_modecar_final"); std::string line; std::getline(output, line);
    REQUIRE(line == "GPML_MODECAR 1 2 fixed-cartesian");
}

TEST_CASE("Dimer reinitialization restores the genuine inverse for padded and VC modes", "[dimer][cellfree]")
{
    Workspace workspace; auto settings = potential_settings(); Potentials::LJ612 potential(settings);
    auto input = molecule(false), padded = molecule(true); DimerDriver driver; configure(driver, potential, 0);
    driver.init(input); driver.refresh_state(); REQUIRE(driver.ivrefcell.isZero());
    driver.init(padded); driver.refresh_state();
    REQUIRE(driver.ivrefcell.isApprox(.05 * Eigen::Matrix3d::Identity()));
    REQUIRE(driver.energy == Catch::Approx(energy(1.3)).margin(1e-12));
    driver.vc_flag = true; driver.modecarf.clear(); driver.init(padded); driver.refresh_state();
    REQUIRE(driver.ivrefcell.isApprox(.05 * Eigen::Matrix3d::Identity()));
    REQUIRE(driver.offset == 6); REQUIRE(driver.xx_center.size() == 12);
    REQUIRE(driver.xx_center.head<6>().isZero(1e-12)); REQUIRE(driver.g_plus.allFinite());
    REQUIRE(std::isfinite(driver.curvature));
}

TEST_CASE("Dimer rejects missing periodic rows and singular VC or fractional input", "[dimer][cellfree]")
{
    const int boundary = GENERATE(0, 1, 2); auto input = molecule(false); DimerDriver driver;
    if (boundary == 0) driver.vc_flag = true;
    if (boundary == 1) input.pbc_ = {true, false, false};
    if (boundary == 2) input.cartesian = false;
    REQUIRE_THROWS_WITH(driver.init(input), Catch::Matchers::ContainsSubstring(
        boundary == 1 ? "independent periodic cell vectors" : "nonsingular cell"));
}
