#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include "mindriver/relax_constraints_io.h"

using Catch::Approx;
namespace {
struct SQPHarmonic : Potentials::AnyPotential {
    std::vector<Eigen::RowVector3d> targets;
    bool fail = false;
    void getEFS(Atoms &a, AnyNeighborhoods &) override { getEFSOMP(a); }
    void getEFSOMP(Atoms &a) override {
        if (fail) throw std::runtime_error("trial failure");
        a.energy_ = 0; a.stress_.setZero(); a.forces_.resize(a.size());
        for (int i = 0; i < a.size(); ++i) {
            auto delta = (a.pos_[i] - targets[i]).eval();
            a.energy_ += .5 * delta.squaredNorm(); a.forces_[i] = -delta;
        }
    }
};
Atoms sqp_atoms() {
    Atoms a; a.cell_ = Eigen::Matrix3d::Identity() * 10;
    a.pbc_ = {false, false, false}; a.cartesian = true;
    a.pos_ = {{.2,.2,0}, {2.8,-.2,0}}; a.chemmap_ = {0,0};
    return a;
}
}

TEST_CASE("V2 modern SQP converges at nonzero raw force with distance equality", "[sqp]") {
    const std::string driver = GENERATE("sqp", "nlc", "legacy");
    auto a = sqp_atoms(); SQPHarmonic p; p.targets = {{0,0,0},{3,0,0}};
    RelaxationDrvV2 d; d.flags.cartesian = true;
    d.control.driver = driver == "legacy" ? "lbfgs" : driver;
    d.flags.with_constraint_nlc = driver == "legacy";
    d.control.max_move = .3; d.tol.gtol = 1e-6;
    regulator r; r.regulator_t = regulator::DISTANCE; r.integer_specs = {0,1};
    r.to_target_g0 = true; r.double_target = 1; r.tol = 1e-8; r.type = 10;
    std::vector<regulator> regs{r}; d.init(&p, a, &regs);
    REQUIRE(d.run() == 0);
    REQUIRE((d.atom.pos_[1] - d.atom.pos_[0]).norm() == Approx(1).margin(1e-8));
    REQUIRE(d.atom.pos_[0][0] == Approx(1).margin(2e-6));
    REQUIRE(d.atom.pos_[1][0] == Approx(2).margin(2e-6));
    REQUIRE(d.g.cwiseAbs().maxCoeff() > .9);
    REQUIRE(d.kkt_residual < 1e-6);
    REQUIRE(d.f == Approx(d.atom.energy_).margin(1e-13));
}

TEST_CASE("V2 SQP trials preserve accepted checkpoints on evaluation failure", "[sqp]") {
    auto a = sqp_atoms(); SQPHarmonic p; p.targets = {{0,0,0},{3,0,0}};
    RelaxationDrvV2 d; d.control.driver = "sqp"; d.flags.cartesian = true;
    d.init(&p, a); d.apply_modifiers(d.modifiers_list); d.eval(); d.accept_checkpoint(false);
    const auto saved = d.atom.pos_; p.fail = true;
    REQUIRE_THROWS(d.run_alg());
    d.restore_checkpoint(); REQUIRE(d.atom.pos_ == saved);
}

TEST_CASE("V2 SQP analytic constraint Jacobian follows a coordinate mask", "[sqp][fd]") {
    auto a = sqp_atoms(); SQPHarmonic p; p.targets = a.pos_;
    a.sel_dynpos_ = {{0,0,0},{1,0,1}};
    RelaxationDrvV2 d; d.control.driver = "sqp"; d.flags.cartesian = true;
    d.flags.selective_dynamics = true; d.control.seldyn_file.clear();
    regulator r; r.regulator_t = regulator::DISTANCE; r.integer_specs = {0,1}; r.type = 10;
    std::vector<regulator> regs{r}; d.init(&p, a, &regs); d.apply_modifiers(d.modifiers_list);
    const auto x = d.x; d.eval(); const auto jac = d.constraint_jacobian;
    for (double h : {1e-5, 1e-6}) for (int j = 0; j < x.size(); ++j) {
        d.x = x; d.x[j] += h; d.eval(); const double plus = d.constraints[0].f;
        d.x = x; d.x[j] -= h; d.eval();
        REQUIRE(jac(0,j) == Approx((plus - d.constraints[0].f)/(2*h)).margin(1e-8));
    }
}
