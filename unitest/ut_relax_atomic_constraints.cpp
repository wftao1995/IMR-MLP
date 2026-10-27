#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include "potentials/potential_lj.h"
#include <limits>

using Catch::Approx;
namespace {
Atoms atomic_geometry() {
    Atoms a;
    a.cell_ << 4.7, 0.3, 0.1, 0.2, 5.1, 0.4, 0.1, 0.3, 5.6;
    a.pos_ = {Eigen::RowVector3d(.90,.22,.30)*a.cell_,
              Eigen::RowVector3d(.08,.35,.43)*a.cell_,
              Eigen::RowVector3d(.18,.92,.65)*a.cell_};
    a.chemmap_ = {0,0,0}; a.sel_dyncell_.setOnes();
    return a;
}
struct AtomicZeroPotential : Potentials::AnyPotential {
    int calls = 0;
    void getEFS(Atoms &a, AnyNeighborhoods &) override { getEFSOMP(a); }
    void getEFSOMP(Atoms &a) override {
        ++calls; a.energy_ = 0; a.stress_.setZero();
        a.forces_.assign(a.size(), Eigen::RowVector3d::Zero());
    }
};
regulator atomic_reg(int kind, bool target = true) {
    regulator r;
    r.regulator_t = static_cast<decltype(r.regulator_t)>(kind);
    r.integer_specs = kind == regulator::DISTANCE ? std::vector<int>{0,1} : std::vector<int>{0,1,2};
    r.type = 10; r.weight_init = 2.3; r.lambda_init = .37;
    r.to_target_g0 = target; r.double_target = kind == regulator::DISTANCE ? 1.6 : 113;
    return r;
}
void objective_fd(RelaxationDrvV2 &d, const Eigen::VectorXd &point) {
    Eigen::VectorXd analytic = Eigen::VectorXd::Zero(point.size());
    d(point, analytic);
    for (int i = 0; i < point.size(); ++i) {
        auto plus = point, minus = point; plus[i] += 1e-6; minus[i] -= 1e-6;
        Eigen::VectorXd tmp = Eigen::VectorXd::Zero(point.size());
        const double fp = d(plus, tmp), fm = d(minus, tmp);
        INFO("optimizer component " << i);
        REQUIRE(analytic[i] == Approx((fp-fm)/2e-6).margin(3e-6).epsilon(3e-6));
    }
}
}

TEST_CASE("Atomic geometry AUL maps every full and masked optimizer derivative", "[atomic_constraints][fd]") {
    const int kind = GENERATE(regulator::DISTANCE, regulator::ANGLE);
    const int mode = GENERATE(0,1,2,3); // fixed, raw9, strain9, exact Cholesky6
    const int coordinates = GENERATE(0,1,2); // Cartesian, fractional, reference Cartesian
    const bool mask = GENERATE(false,true);
    const bool move_atoms = GENERATE(false,true);
    if (mode == 0 && !move_atoms) return;
    CAPTURE(kind, mode, coordinates, mask, move_atoms);
    auto a = atomic_geometry(); AtomicZeroPotential p; RelaxationDrvV2 d;
    d.flags.cartesian = coordinates == 0;
    d.flags.atomic_force_recast = coordinates == 2;
    d.flags.relax_cell = mode != 0; d.flags.relax_atom = move_atoms;
    d.flags.relax_cell_use_strain = mode >= 2; d.flags.symm_strain = mode == 3;
    d.flags.dprecond = false; d.flags.with_constraint_aul = true;
    d.control.cell_factor = .35; d.control.seldyn_file = "";
    d.flags.selective_dynamics = mask;
    if (mask) {
        a.sel_dynpos_.assign(3, Eigen::RowVector3d::Ones());
        a.sel_dynpos_[0].setZero(); a.sel_dynpos_[1][1] = 0;
        if (mode == 3) d.configure_cell_mask("1 0 1 1 0 1");
        else if (mode) d.configure_cell_mask("1 1 0 1 1 1 0 1 1");
    }
    std::vector<regulator> regs{atomic_reg(kind)}; d.init(&p, a, &regs);
    REQUIRE_FALSE(d.constraints[0].atomic_images[0].isZero());
    // Match run_impl: initialize the reduced optimizer coordinates before direct eval.
    d.apply_modifiers(d.modifiers_list);
    auto point = d.x;
    for (int i = 0; i < point.size(); ++i) point[i] += .012 * ((i % 3)-1);
    objective_fd(d, point);
}

TEST_CASE("Atomic geometry core finite differences and fixed image continuity", "[atomic_constraints][fd]") {
    const int kind = GENERATE(regulator::DISTANCE, regulator::ANGLE);
    auto a = atomic_geometry(); auto r = atomic_reg(kind, false);
    Constraints::atomic_geometry_setup(r, a);
    auto value = Constraints::evaluate_atomic_geometry(r, a);
    REQUIRE(value.residual == Approx(0).margin(2e-15));
    const auto image = r.atomic_images;
    const double target = r.double_target;
    a.pos_[0] += Eigen::RowVector3d(.21,.05,-.09);
    Constraints::atomic_geometry_setup(r, a);
    REQUIRE(r.double_target == target);
    REQUIRE(r.atomic_images[0] == image[0]); REQUIRE(r.atomic_images[1] == image[1]);
    value = Constraints::evaluate_atomic_geometry(r, a);
    Eigen::RowVector3d sum = Eigen::RowVector3d::Zero();
    for (size_t i = 0; i < r.integer_specs.size(); ++i) {
        sum += value.cartesian_gradients[i];
        for (int c = 0; c < 3; ++c) {
            auto plus = a, minus = a;
            plus.pos_[r.integer_specs[i]][c] += 1e-6;
            minus.pos_[r.integer_specs[i]][c] -= 1e-6;
            REQUIRE(value.cartesian_gradients[i][c] == Approx((Constraints::evaluate_atomic_geometry(r, plus).residual-
                Constraints::evaluate_atomic_geometry(r, minus).residual)/2e-6).margin(1e-7));
        }
    }
    REQUIRE(sum.norm() < 1e-13);
    auto translated = a; for (auto &p : translated.pos_) p += Eigen::RowVector3d(5,-3,7);
    REQUIRE(Constraints::evaluate_atomic_geometry(r, translated).residual == Approx(value.residual).margin(1e-13));
    auto reversed = r;
    if (kind == regulator::DISTANCE) { std::swap(reversed.integer_specs[0],reversed.integer_specs[1]); reversed.atomic_images[0] *= -1; }
    else { std::swap(reversed.integer_specs[0],reversed.integer_specs[2]); std::swap(reversed.atomic_images[0],reversed.atomic_images[1]); }
    REQUIRE(Constraints::evaluate_atomic_geometry(reversed, a).residual == Approx(value.residual).margin(1e-13));
}

TEST_CASE("Atomic target binding respects partial periodicity and skew nearest image", "[atomic_constraints]") {
    auto a = atomic_geometry();
    a.pbc_ = {true,false,false};
    auto r = atomic_reg(regulator::ANGLE,false); Constraints::atomic_geometry_setup(r,a);
    for (const auto &image : r.atomic_images) { REQUIRE(image[1] == 0); REQUIRE(image[2] == 0); }
    a.pbc_ = {false,false,false}; r = atomic_reg(regulator::DISTANCE,false);
    Constraints::atomic_geometry_setup(r,a); REQUIRE(r.atomic_images[0].isZero());
    REQUIRE(r.double_target == Approx((a.pos_[1]-a.pos_[0]).norm()));
    a.cell_ << 1,0,0, 2.1,.1,0, 0,0,4;
    a.pos_[0].setZero(); a.pos_[1] = Eigen::RowVector3d(.49,.49,0)*a.cell_;
    a.pbc_ = {true,true,false}; r = atomic_reg(regulator::DISTANCE,false);
    Constraints::atomic_geometry_setup(r,a);
    REQUIRE(r.double_target == Approx(a.get_distance(0,1)));
    REQUIRE(Constraints::evaluate_atomic_geometry(r,a).residual == Approx(0).margin(1e-14));
}

TEST_CASE("Atomic geometry rejects invalid indices targets and degenerate arms", "[atomic_constraints][errors]") {
    const int bad = GENERATE(0,1,2,3,4,5,6,7,8);
    auto a = atomic_geometry(); auto r = atomic_reg(regulator::ANGLE);
    if (bad == 0) r.integer_specs = {0,1};
    if (bad == 1) r.integer_specs[2] = 3;
    if (bad == 2) r.integer_specs[2] = 0;
    if (bad == 3) r.double_target = 180;
    if (bad == 4) r.double_target = 0;
    if (bad == 5) r.double_target = std::numeric_limits<double>::infinity();
    if (bad == 6) a.pos_[0] = a.pos_[1];
    if (bad == 7) { a.pbc_ = {false,false,false}; a.pos_[0] = a.pos_[1] + Eigen::RowVector3d(1,0,0); a.pos_[2] = a.pos_[1] - Eigen::RowVector3d(1,0,0); }
    if (bad == 8) a.pos_[0][0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(Constraints::atomic_geometry_setup(r,a));
}

TEST_CASE("Atomic constraints with a frozen endpoint converge and all frozen unmet remains unmet", "[atomic_constraints]") {
    const bool all_frozen = GENERATE(false,true);
    const int kind = GENERATE(regulator::DISTANCE, regulator::ANGLE);
    auto a = atomic_geometry(); AtomicZeroPotential p; RelaxationDrvV2 d;
    d.flags.cartesian = true; d.flags.with_constraint_aul = true; d.flags.selective_dynamics = true;
    d.control.seldyn_file = ""; d.control.max_iter = 600; d.control.constraint_iter_limit = 15;
    d.tol.gtol = 1e-8; d.tol.gtol0 = 1e-3;
    a.sel_dynpos_.assign(a.size(),Eigen::RowVector3d::Zero());
    if (!all_frozen) a.sel_dynpos_[kind == regulator::DISTANCE ? 1 : 2].setOnes();
    std::vector<regulator> regs{atomic_reg(kind)};
    regs[0].lambda_init = 0; regs[0].tol = 1e-7;
    d.init(&p,a,&regs); d.run();
    REQUIRE(d.atom.pos_[0].isApprox(a.pos_[0],1e-14));
    if (all_frozen) { REQUIRE_FALSE(d.state_vars.converged); REQUIRE(std::abs(d.constraints[0].f) > regs[0].tol); }
    else { REQUIRE(d.state_vars.converged); REQUIRE(std::abs(d.constraints[0].f) <= regs[0].tol); }
}

TEST_CASE("Real Lennard Jones plus atomic AUL full objective finite differences", "[atomic_constraints][fd]") {
    const int kind = GENERATE(regulator::DISTANCE, regulator::ANGLE);
    const int mode = GENERATE(0,1,2,3);
    const bool recast = GENERATE(false,true);
    auto a = atomic_geometry();
    GPSettings potential_settings;
    potential_settings.Modify("ntypes", "1"); potential_settings.Modify("0-0", "1 1");
    potential_settings.Modify("cut:0-0", "2.4");
    Potentials::LJ612 p(potential_settings); RelaxationDrvV2 d;
    d.flags.relax_cell = mode != 0; d.flags.relax_cell_use_strain = mode >= 2;
    d.flags.symm_strain = mode == 3;
    d.flags.atomic_force_recast = recast; d.flags.dprecond = false;
    d.flags.with_constraint_aul = true;
    std::vector<regulator> regs{atomic_reg(kind)}; d.init(&p,a,&regs);
    auto point = d.x;
    for (int i = 0; i < point.size(); ++i) point[i] += .006 * ((i%3)-1);
    objective_fd(d,point);
    REQUIRE(std::abs(d.atom.energy_) > 1e-6);
}

TEST_CASE("All frozen atomic geometry at its initial target is satisfied", "[atomic_constraints]") {
    auto a = atomic_geometry(); AtomicZeroPotential p; RelaxationDrvV2 d;
    d.flags.cartesian = true; d.flags.with_constraint_aul = true; d.flags.selective_dynamics = true;
    d.control.seldyn_file = "";
    a.sel_dynpos_.assign(a.size(),Eigen::RowVector3d::Zero());
    std::vector<regulator> regs{atomic_reg(regulator::DISTANCE,false),atomic_reg(regulator::ANGLE,false)};
    d.init(&p,a,&regs); d.run(); REQUIRE(d.state_vars.converged);
    for (const auto &r : d.constraints) REQUIRE(std::abs(r.f) <= r.tol);
}

TEST_CASE("Cartesian cell-only atomic constraints update cell and preserve positions", "[atomic_constraints][fd]") {
    const int mode = GENERATE(1,2,3);
    auto a = atomic_geometry(); AtomicZeroPotential p; RelaxationDrvV2 d;
    d.flags.cartesian = true; d.flags.relax_atom = false; d.flags.relax_cell = true;
    d.flags.relax_cell_use_strain = mode >= 2; d.flags.symm_strain = mode == 3;
    d.flags.dprecond = false; d.flags.with_constraint_aul = true;
    std::vector<regulator> regs{atomic_reg(regulator::DISTANCE)}; d.init(&p,a,&regs);
    auto point = d.x; point[0] += .08; point[1] -= .03;
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(point.size()); d(point,gradient);
    REQUIRE_FALSE(d.atom.cell_.isApprox(a.cell_,1e-10));
    for (int i = 0; i < a.size(); ++i) REQUIRE(d.atom.pos_[i].isApprox(a.pos_[i],1e-14));
    objective_fd(d,point);
}
