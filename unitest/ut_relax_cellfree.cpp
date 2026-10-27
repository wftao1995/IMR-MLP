#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include "potentials/potential_lj.h"
#include <cmath>

namespace {
Atoms molecule(bool padded) {
    Atoms atom; atom.pbc_ = {false, false, false}; atom.cell_.setZero();
    if (padded) atom.cell_ = 20 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{0, 0, 0}, {1.3, 0, 0}}; atom.chemmap_ = {0, 0};
    return atom;
}
GPSettings lj_settings() {
    GPSettings settings; settings.Modify("ntypes", "1"); settings.Modify("0-0", "1 1");
    settings.Modify("cut:0-0", "3"); return settings;
}
void configure(RelaxationDrvV2 &driver, const std::string &optimizer, int budget) {
    driver.control.driver = optimizer; driver.control.max_iter = budget;
    driver.control.max_move = .02; driver.control.indexfile.clear();
    driver.tol.gtol = 1e-10;
}
double lj_energy(double radius) {
    return 4 * (std::pow(1 / radius, 12) - std::pow(1 / radius, 6));
}
}

TEST_CASE("Fixed-cell recast molecular relaxation agrees with padding without changing flags", "[relax][cellfree]")
{
    const std::string optimizer = GENERATE("fire", "lbfgs");
    const int budget = GENERATE(0, 3);
    auto settings = lj_settings(); Potentials::LJ612 potential(settings);
    auto input = molecule(false), padded_input = molecule(true);
    RelaxationDrvV2 driver, control; configure(driver, optimizer, budget); configure(control, optimizer, budget);
    driver.init(&potential, input); control.init(&potential, padded_input);
    REQUIRE(driver.x.allFinite()); REQUIRE(driver.ivrefcell.allFinite());
    driver.run(); control.run();
    REQUIRE(driver.energy0 == Catch::Approx(lj_energy(1.3)).margin(1e-12));
    REQUIRE(driver.x.allFinite()); REQUIRE(driver.g.allFinite());
    REQUIRE(driver.atom.cell_.isZero()); REQUIRE(driver.atom.is_nonperiodic());
    REQUIRE_FALSE(driver.flags.cartesian); REQUIRE(driver.flags.atomic_force_recast);
    REQUIRE(driver.atom.energy_ == Catch::Approx(control.atom.energy_).margin(1e-10));
    const double radius = (driver.atom.pos_[1] - driver.atom.pos_[0]).norm();
    REQUIRE(driver.atom.energy_ == Catch::Approx(lj_energy(radius)).margin(1e-10));
    for (int i = 0; i < 2; ++i) {
        REQUIRE(driver.atom.pos_[i].allFinite());
        REQUIRE(driver.atom.pos_[i].isApprox(control.atom.pos_[i], 1e-10));
    }
    REQUIRE(input.pos_[0].isZero()); REQUIRE(input.pos_[1][0] == 1.3);
    // No persistent coordinate flag or inverse from the previous initialization.
    driver.init(&potential, padded_input); driver.run();
    REQUIRE(driver.ivrefcell.isApprox(.05 * Eigen::Matrix3d::Identity()));
    REQUIRE_FALSE(driver.flags.cartesian);
    REQUIRE(driver.atom.energy_ == Catch::Approx(control.atom.energy_).margin(1e-10));
}

TEST_CASE("Cell-free recast distance constraint retains the existing full-objective derivative", "[relax][cellfree][atomic_constraints]")
{
    auto settings = lj_settings(); Potentials::LJ612 potential(settings); auto input = molecule(false);
    RelaxationDrvV2 driver; configure(driver, "lbfgs", 0); driver.flags.with_constraint_aul = true;
    regulator distance; distance.regulator_t = regulator::DISTANCE; distance.integer_specs = {0, 1};
    distance.type = 10; distance.weight_init = 2.3; distance.lambda_init = .37;
    distance.to_target_g0 = true; distance.double_target = 1.25;
    std::vector<regulator> constraints{distance}; driver.init(&potential, input, &constraints);
    auto point = driver.x; point[0] += .01;
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(point.size()); driver(point, gradient);
    REQUIRE(gradient.allFinite());
    for (int i = 0; i < point.size(); ++i) {
        auto plus = point, minus = point; plus[i] += 1e-6; minus[i] -= 1e-6;
        Eigen::VectorXd scratch = Eigen::VectorXd::Zero(point.size());
        const double fp = driver(plus, scratch), fm = driver(minus, scratch);
        REQUIRE(gradient[i] == Catch::Approx((fp - fm) / 2e-6).margin(3e-6));
    }
}

TEST_CASE("Cell-free relaxation keeps Cartesian mask coordinates and lattice-required boundaries", "[relax][cellfree]")
{
    auto settings = lj_settings(); Potentials::LJ612 potential(settings); auto input = molecule(false);
    SECTION("atomic mask still fixes the first atom") {
        RelaxationDrvV2 driver; configure(driver, "lbfgs", 3); driver.flags.selective_dynamics = true;
        driver.control.seldyn_file.clear(); input.sel_dynpos_ = {Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Ones()};
        driver.init(&potential, input); driver.run();
        REQUIRE(driver.atom.pos_[0].isZero()); REQUIRE(driver.atom.pos_[1].allFinite());
        REQUIRE(driver.atom.pos_[1][0] < 1.3);
    }
    SECTION("symmetry still needs a lattice") {
        const bool keep = GENERATE(false, true);
        RelaxationDrvV2 driver; driver.flags.keep_symm = keep; driver.flags.symm_force = !keep;
        REQUIRE_THROWS_WITH(driver.init(&potential, input), Catch::Matchers::ContainsSubstring("nonsingular cell"));
    }
    SECTION("fixed volume still needs a lattice") {
        RelaxationDrvV2 driver; driver.flags.fix_vol = true;
        REQUIRE_THROWS_WITH(driver.init(&potential, input), Catch::Matchers::ContainsSubstring("nonsingular cell"));
    }
    SECTION("fractional positions still need a lattice") {
        RelaxationDrvV2 driver; input.cartesian = false;
        REQUIRE_THROWS_WITH(driver.init(&potential, input), Catch::Matchers::ContainsSubstring("Cartesian input"));
    }
}
