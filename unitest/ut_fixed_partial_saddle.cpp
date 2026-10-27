#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
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
        char name[] = "/tmp/imrmlp_partial_saddle_XXXXXX"; const char *directory = mkdtemp(name);
        if (!directory) throw std::runtime_error("Cannot create partial-cell saddle workspace");
        path = directory; std::filesystem::current_path(path);
        std::ofstream("MODECAR") << "GPML_MODECAR 1 2 fixed-cartesian\n-1.2 0 -.5\n1.2 0 .5\n";
    }
    ~Workspace() { std::filesystem::current_path(previous); std::filesystem::remove_all(path); }
};
GPSettings lj_settings() {
    GPSettings settings; settings.Modify("ntypes", "1"); settings.Modify("0-0", "1 1");
    settings.Modify("cut:0-0", "3"); return settings;
}
Atoms slab(bool nonprefix, bool padding) {
    Atoms atom; atom.cell_.setZero(); atom.cell_.row(0) << 10, 0, 0;
    atom.cell_.row(nonprefix ? 2 : 1) << 0, 10, 0;
    if (padding) atom.cell_.row(nonprefix ? 1 : 2) << 0, 0, 20;
    atom.pbc_ = {true, !nonprefix, nonprefix};
    atom.pos_ = {{1, 1, 3}, {2.2, 1, 3.5}}; atom.chemmap_ = {0, 0}; return atom;
}
void configure(RelaxationDrvV2 &driver, const std::string &optimizer, int budget) {
    driver.control.driver = optimizer; driver.control.max_iter = budget;
    driver.control.max_move = .02; driver.control.indexfile.clear(); driver.control.seldyn_file.clear();
    driver.tol.gtol = 1e-10;
}
void configure(DimerDriver &driver, Potentials::AnyPotential &potential, int budget) {
    driver.pot = &potential; driver.modecarf = "MODECAR"; driver.max_iter = budget;
    driver.rot_max_iter = driver.trn_max_iter = 2; driver.summary.clear();
}
double energy(double r) { return 4 * (std::pow(r, -12) - std::pow(r, -6)); }
double derivative(double r) { return 24 * (std::pow(r, -7) - 2 * std::pow(r, -13)); }
}

TEST_CASE("Fixed recast slab relaxation matches padding and retains normal motion", "[relax][partial_cell]")
{
    const bool nonprefix = GENERATE(false, true);
    const int budget = GENERATE(0, 2);
    const std::string optimizer = GENERATE("fire", "lbfgs");
    auto settings = lj_settings(); Potentials::LJ612 potential(settings);
    auto input = slab(nonprefix, false), padded = slab(nonprefix, true);
    RelaxationDrvV2 driver, control; configure(driver, optimizer, budget); configure(control, optimizer, budget);
    driver.init(&potential, input); control.init(&potential, padded); driver.run(); control.run();
    REQUIRE(driver.x.allFinite()); REQUIRE(driver.g.allFinite()); REQUIRE(driver.ivrefcell.isZero());
    REQUIRE(driver.atom.energy_ == Catch::Approx(control.atom.energy_).margin(1e-10));
    REQUIRE(driver.atom.cell_.isApprox(input.cell_)); REQUIRE(driver.atom.pbc_ == input.pbc_);
    REQUIRE(driver.atom.chemmap_ == input.chemmap_);
    for (int i = 0; i < 2; ++i) REQUIRE(driver.atom.pos_[i].isApprox(control.atom.pos_[i], 1e-10));
    if (budget) REQUIRE(std::abs(driver.atom.pos_[0][2] - 3) > 1e-8);
    const double radius = (driver.atom.pos_[1] - driver.atom.pos_[0]).norm();
    REQUIRE(driver.atom.energy_ == Catch::Approx(energy(radius)).margin(1e-10));
    REQUIRE(input.pos_[0].isApprox(Eigen::RowVector3d(1, 1, 3)));
    REQUIRE(input.pos_[1].isApprox(Eigen::RowVector3d(2.2, 1, 3.5)));
    driver.init(&potential, padded); driver.run();
    REQUIRE((driver.ivrefcell * padded.cell_).isApprox(Eigen::Matrix3d::Identity()));
    REQUIRE_FALSE(driver.flags.cartesian); REQUIRE(driver.flags.atomic_force_recast);
}

TEST_CASE("Fixed Cartesian slab Dimer keeps its radial curvature and normal motion", "[dimer][partial_cell]")
{
    Workspace workspace; const bool nonprefix = GENERATE(false, true);
    const int budget = GENERATE(0, 2);
    auto settings = lj_settings(); Potentials::LJ612 potential(settings);
    auto input = slab(nonprefix, false), padded = slab(nonprefix, true);
    DimerDriver driver, control; configure(driver, potential, budget); configure(control, potential, budget);
    driver.init(input); control.init(padded); driver.run(); control.run();
    REQUIRE(driver.energy == Catch::Approx(control.energy).margin(1e-10));
    REQUIRE(driver.curvature == Catch::Approx(control.curvature).margin(1e-9));
    REQUIRE(driver.atom_center.cell_.isApprox(input.cell_)); REQUIRE(driver.atom_center.pbc_ == input.pbc_);
    for (int i = 0; i < 2; ++i) REQUIRE(driver.atom_center.pos_[i].isApprox(control.atom_center.pos_[i], 1e-10));
    if (budget) REQUIRE(std::abs(driver.atom_center.pos_[0][2] - 3) > 1e-8);
    else {
        const double shift = std::sqrt(2.) * driver.ddr;
        const double oracle = std::sqrt(2.) * (derivative(1.3 + shift) - derivative(1.3 - shift)) / (2 * driver.ddr);
        REQUIRE(driver.curvature == Catch::Approx(oracle).margin(1e-9));
    }
    REQUIRE(input.pos_[0].isApprox(Eigen::RowVector3d(1, 1, 3)));
    driver.init(padded); driver.refresh_state();
    REQUIRE((driver.ivrefcell * padded.cell_).isApprox(Eigen::Matrix3d::Identity()));
}

TEST_CASE("Partial recast atomic constraints and masks retain Cartesian coordinates", "[relax][partial_cell]")
{
    auto settings = lj_settings(); Potentials::LJ612 potential(settings); auto input = slab(false, false);
    SECTION("distance objective derivative") {
        RelaxationDrvV2 driver; configure(driver, "lbfgs", 0); driver.flags.with_constraint_aul = true;
        regulator distance; distance.regulator_t = regulator::DISTANCE; distance.integer_specs = {0, 1};
        distance.type = 10; distance.weight_init = 2.3; distance.lambda_init = .37;
        distance.to_target_g0 = true; distance.double_target = 1.25;
        std::vector<regulator> constraints{distance}; driver.init(&potential, input, &constraints);
        const auto point = driver.x; Eigen::VectorXd gradient = Eigen::VectorXd::Zero(point.size()); driver(point, gradient);
        for (int i = 0; i < point.size(); ++i) {
            auto plus = point, minus = point; plus[i] += 1e-6; minus[i] -= 1e-6;
            Eigen::VectorXd scratch = Eigen::VectorXd::Zero(point.size());
            const double fp = driver(plus, scratch), fm = driver(minus, scratch);
            REQUIRE(gradient[i] == Catch::Approx((fp - fm) / 2e-6).margin(4e-6));
        }
    }
    SECTION("first atom remains masked") {
        RelaxationDrvV2 driver; configure(driver, "lbfgs", 2); driver.flags.selective_dynamics = true;
        input.sel_dynpos_ = {Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Ones()};
        driver.init(&potential, input); driver.run();
        REQUIRE(driver.atom.pos_[0].isApprox(input.pos_[0]));
        REQUIRE(std::abs(driver.atom.pos_[1][2] - input.pos_[1][2]) > 1e-8);
    }
}

TEST_CASE("Partial-cell optimization keeps genuine-lattice requirements", "[relax][dimer][partial_cell]")
{
    auto settings = lj_settings(); Potentials::LJ612 potential(settings); auto input = slab(false, false);
    RelaxationDrvV2 relax; DimerDriver dimer; configure(dimer, potential, 0);
    SECTION("dependent periodic basis") {
        input.cell_.row(1) = 2 * input.cell_.row(0);
        REQUIRE_THROWS_WITH(relax.init(&potential, input), Catch::Matchers::ContainsSubstring("independent periodic"));
        REQUIRE_THROWS_WITH(dimer.init(input), Catch::Matchers::ContainsSubstring("independent periodic"));
    }
    SECTION("variable cell") { relax.flags.relax_cell = true; dimer.vc_flag = true; REQUIRE_THROWS(relax.init(&potential, input)); REQUIRE_THROWS(dimer.init(input)); }
    SECTION("fractional input") { input.cartesian = false; REQUIRE_THROWS(relax.init(&potential, input)); REQUIRE_THROWS(dimer.init(input)); }
    SECTION("fully periodic input") { input.pbc_ = {true, true, true}; REQUIRE_THROWS(relax.init(&potential, input)); REQUIRE_THROWS(dimer.init(input)); }
    SECTION("symmetry and fixed volume remain lattice-dependent") {
        const bool symmetry = GENERATE(false, true);
        relax.flags.keep_symm = symmetry; relax.flags.fix_vol = !symmetry;
        REQUIRE_THROWS_WITH(relax.init(&potential, input), Catch::Matchers::ContainsSubstring("nonsingular cell"));
    }
}
