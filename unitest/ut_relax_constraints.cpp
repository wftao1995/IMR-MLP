#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include "potentials/potential_lj.h"
#include <limits>

using Catch::Approx;
namespace {
Atoms geometry() {
    Atoms a;
    a.cell_ << 4.7, 0.3, 0.1, 0.2, 5.1, 0.4, 0.1, 0.3, 5.6;
    a.pos_ = {{0.8, 1.0, 1.2}, {1.4, 1.7, 1.9}};
    a.chemmap_ = {0, 0}; a.sel_dyncell_.setOnes();
    return a;
}
struct SmoothPotential : Potentials::AnyPotential {
    int calls = 0, throw_after = -1;
    double cell_weight = 0, pair_weight = 0, atom_weight = 0, center = 1.05;
    void evaluate(Atoms &a) {
        if (throw_after >= 0 && calls >= throw_after) throw std::runtime_error("injected potential failure");
        ++calls;
        a.energy_ = 0.5 * cell_weight * a.cell_.squaredNorm();
        a.stress_ = -cell_weight * a.cell_.transpose() * a.cell_;
        a.forces_.assign(a.size(), Eigen::RowVector3d::Zero());
        if (a.size() >= 2) {
            const Eigen::RowVector3d delta = a.pos_[1] - a.pos_[0];
            a.energy_ += 0.5 * pair_weight * delta.squaredNorm();
            a.forces_[0] += pair_weight * delta;
            a.forces_[1] -= pair_weight * delta;
            a.stress_ -= pair_weight * delta.transpose() * delta;
        }
        const double dx = a.pos_[0][0] - center;
        a.energy_ += 0.5 * atom_weight * dx * dx;
        a.forces_[0][0] -= atom_weight * dx;
    }
    void getEFSOMP(Atoms &a) override { evaluate(a); }
    void getEFS(Atoms &a, AnyNeighborhoods &) override { evaluate(a); }
};
GPSettings raw_settings(bool atoms = false) {
    GPSettings s;
    s.Modify("relax_cell", "true"); s.Modify("cell_strain", "false");
    s.Modify("dprecond", "false"); s.Modify("relax_atom", atoms ? "true" : "false");
    s.Modify("with_constraint_aul", "true"); s.Modify("write_traj", "false");
    s.Modify("gtol", "1e-7"); s.Modify("maxiteration", "500");
    return s;
}
regulator lattice(int kind) {
    regulator r; r.regulator_t = static_cast<decltype(r.regulator_t)>(kind);
    r.type = 10; r.weight_init = 0.7; r.lambda_init = 0.17;
    if (kind == regulator::CONSTANT_VECTOR) r.integer_specs = {1};
    if (kind == regulator::CONSTANT_ANGLE || kind == regulator::COUPLE_VECTOR) r.integer_specs = {0, 2};
    return r;
}
regulator scalar(double target = 1.0) {
    regulator r; r.type = 10; r.g0 = target; r.tol = 1e-7;
    r.inputs = {0}; r.outputs = {0}; r.x.resize(1); r.g.resize(1);
    r.compute = [](double *x, int, double *g, int, double *f, double target, Atoms *, int *, void *) {
        *f = x[0] - target; g[0] = 1;
    };
    return r;
}
void scalar_driver(RelaxationDrvV2 &d) {
    d.flags.cartesian = true; d.flags.with_constraint_aul = true;
    d.tol.gtol = 1e-7; d.tol.gtol0 = 0.1;
    d.control.constraint_iter_limit = 10; d.control.max_iter = 500;
}
void fd(RelaxationDrvV2 &d, const Eigen::VectorXd &x) {
    Eigen::VectorXd analytic;
    analytic.setZero(x.size()); d(x, analytic);
    for (int i = 0; i < x.size(); ++i) {
        auto xp = x, xm = x; xp[i] += 1e-6; xm[i] -= 1e-6;
        Eigen::VectorXd tmp = Eigen::VectorXd::Zero(x.size());
        const double fp = d(xp, tmp), fm = d(xm, tmp);
        INFO("coordinate " << i);
        REQUIRE(analytic[i] == Approx((fp - fm) / 2e-6).margin(1e-5));
    }
}
}

TEST_CASE("Raw9 lattice AUL full objective matches finite differences", "[relax_constraints][fd]") {
    const int kind = GENERATE(regulator::CONSTANT_VOLUME, regulator::CONSTANT_VECTOR, regulator::CONSTANT_ANGLE, regulator::COUPLE_VECTOR);
    const bool atoms = GENERATE(false, true);
    const bool recast = GENERATE(false, true);
    const double factor = GENERATE(0.25, 3.0);
    CAPTURE(kind, atoms, recast, factor);
    auto a = geometry(); SmoothPotential p; p.cell_weight = 0.03; p.pair_weight = 0.2;
    auto settings = raw_settings(atoms); settings.Modify("atomic_force_recast", recast ? "true" : "false");
    RelaxationDrvV2 d(settings); d.control.cell_factor = factor;
    std::vector<regulator> regs{lattice(kind)}; d.init(&p, a, &regs);
    auto point = d.x; point[1] += 0.08; point[3] -= 0.04; point[8] += 0.03;
    if (atoms) point[9] += 0.025;
    fd(d, point);
}

TEST_CASE("Lattice callbacks reject dimensions axes and singular vectors", "[relax_constraints][errors]") {
    auto a = geometry(); regulator r; double x[9]{}, g[9]{}, f;
    REQUIRE_THROWS(Constraints::constant_vector_setup(r, a, -1));
    REQUIRE_THROWS(Constraints::constant_vector_setup(r, a, 3));
    REQUIRE_THROWS(Constraints::constant_angle_setup(r, a, 0, 0));
    REQUIRE_THROWS(Constraints::couple_vector_setup(r, a, 1, 4));
    REQUIRE_THROWS(Constraints::constant_volume(x, 6, g, 9, &f, 1));
    REQUIRE_THROWS(Constraints::constant_vector(x, 2, g, 3, &f, 1));
    REQUIRE_THROWS(Constraints::constant_angle(x, 6, g, 6, &f, 0));
    REQUIRE_THROWS(Constraints::couple_vector(x, 6, g, 6, &f, 1));
    a.cell_.row(0).setZero(); REQUIRE_THROWS(Constraints::constant_angle_setup(r, a, 0, 1));
}

TEST_CASE("Constraint initialization validates before geometry mutation or evaluation", "[relax_constraints][errors]") {
    const int bad = GENERATE(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19);
    auto a = geometry(); const auto original = a.cell_; SmoothPotential p;
    RelaxationDrvV2 d(raw_settings()); std::vector<regulator> regs{lattice(regulator::CONSTANT_VECTOR)};
    switch (bad) {
    case 0: regs[0].integer_specs.clear(); break;
    case 1: regs[0].integer_specs = {3}; break;
    case 2: regs[0].weight_init = -1; break;
    case 3: regs[0].tol = 0; break;
    case 4: regs[0].lambda_init = std::numeric_limits<double>::infinity(); break;
    case 5: regs[0].regulator_t = static_cast<decltype(regs[0].regulator_t)>(99); break;
    case 6: regs[0].type = 9; break;
    case 7: d.flags.relax_cell_use_strain = true; break;
    case 8: d.flags.relax_cell_use_strain = true; d.flags.symm_strain = false; break;
    case 9: d.flags.relax_cell = false; break;
    case 10: d.flags.fix_shape = true; break;
    case 11: d.flags.fix_vol = true; break;
    case 12: d.flags.dprecond = true; break;
    case 13: a.pbc_[1] = false; break;
    case 14: a.cell_.row(0) *= -1; break;
    case 15: regs[0].to_target_g0 = true; regs[0].g0 = 0; break;
    case 16: regs[0].weight_multply = std::numeric_limits<double>::quiet_NaN(); break;
    case 17: d.flags.cartesian = true; break;
    case 18: d.control.pressure = 0.1; break;
    case 19: d.external_stress(0,0) = 1; break;
    }
    REQUIRE_THROWS(d.init(&p, a, &regs)); REQUIRE(p.calls == 0); REQUIRE_FALSE(d.initialized);
    REQUIRE(d.constraints.empty());
    REQUIRE_THROWS(d.run()); REQUIRE(p.calls == 0);
    d.flags.relax_cell = true; d.flags.relax_cell_use_strain = false; d.flags.fix_shape = d.flags.fix_vol = d.flags.dprecond = d.flags.cartesian = false;
    d.control.pressure = 0; d.external_stress.setZero();
    a = geometry(); regs = {lattice(regulator::CONSTANT_VECTOR)};
    REQUIRE_NOTHROW(d.init(&p, a, &regs)); REQUIRE(d.initialized); REQUIRE(a.cell_.isApprox(original));
}

TEST_CASE("Custom constraint layout and callback output are checked", "[relax_constraints][errors]") {
    const int bad = GENERATE(0,1,2,3,4,5);
    auto a = geometry(); SmoothPotential p; RelaxationDrvV2 d; scalar_driver(d);
    std::vector<regulator> regs{scalar()};
    if (bad == 0) regs[0].compute = nullptr;
    if (bad == 1) regs[0].inputs = {-1};
    if (bad == 2) regs[0].outputs = {6};
    if (bad == 3) regs[0].g.clear();
    if (bad == 4) regs[0].x.clear();
    if (bad == 5) regs[0].compute = [](double *, int, double *, int, double *, double, Atoms *, int *, void *) {};
    if (bad < 5) { REQUIRE_THROWS(d.init(&p, a, &regs)); REQUIRE(p.calls == 0); }
    else { d.init(&p, a, &regs); REQUIRE_THROWS(d.run()); REQUIRE_FALSE(d.state_vars.converged); }
}

TEST_CASE("Constraint reuse clears null and empty requests and resets multiplier state", "[relax_constraints]") {
    auto a = geometry(); SmoothPotential p; RelaxationDrvV2 d(raw_settings());
    std::vector<regulator> regs{lattice(regulator::CONSTANT_VECTOR)};
    d.init(&p, a, &regs); d.constraints[0].lambda = 123;
    d.init(&p, a); REQUIRE(d.constraints.empty());
    d.init(&p, a, &regs); REQUIRE(d.constraints[0].lambda == Approx(0.17));
    std::vector<regulator> none; d.init(&p, a, &none); REQUIRE(d.constraints.empty());
    a.cell_ *= 1.1; d.init(&p, a, &regs);
    REQUIRE(d.constraints[0].g0 == Approx(a.cell_.row(1).squaredNorm()));
    REQUIRE(d.constraints[0].weight == Approx(0.7));
}

TEST_CASE("AUL loops and numerical controls reject invalid requests before evaluation", "[relax_constraints][errors]") {
    const int bad = GENERATE(0,1,2,3,4,5,6,7);
    auto a = geometry(); SmoothPotential p; RelaxationDrvV2 d; scalar_driver(d);
    switch (bad) {
    case 0: d.control.constraint_iter_limit = 0; break;
    case 1: d.control.constraint_iter_limit = -1; break;
    case 2: d.tol.gtol0 = 1e-9; break;
    case 3: d.tol.gtol0 = std::numeric_limits<double>::quiet_NaN(); break;
    case 4: d.tol.gtol = std::numeric_limits<double>::infinity(); break;
    case 5: d.tol.eps = 0; break;
    case 6: d.tol.ftol = -1; break;
    case 7: d.tol.xtol = std::numeric_limits<double>::quiet_NaN(); break;
    }
    std::vector<regulator> regs{scalar()}; REQUIRE_THROWS(d.init(&p, a, &regs)); REQUIRE(p.calls == 0);
}

TEST_CASE("Single AUL loop uses final tolerance and converges a pure equality", "[relax_constraints]") {
    auto a = geometry(); a.pos_[0][0] = 1.02; SmoothPotential p;
    RelaxationDrvV2 d; scalar_driver(d); d.control.constraint_iter_limit = 1;
    std::vector<regulator> regs{scalar()}; d.init(&p, a, &regs);
    const int result = d.run();
    CAPTURE(result, d.state_vars.curr_iter, d.state_vars.curr_constraint_iter, d.constraints[0].f, d.constraints[0].lambda, d.constraints[0].weight, d.g.norm());
    REQUIRE(result == 0); REQUIRE(d.state_vars.curr_iter > 0);
    REQUIRE(d.state_vars.curr_constraint_iter == 1); REQUIRE(d.tol.gtol == Approx(1e-7));
    REQUIRE(std::abs(d.constraints[0].f) <= regs[0].tol);
}

TEST_CASE("AUL tightens a coarse stationary point and reoptimizes updated multipliers", "[relax_constraints]") {
    const std::string driver = GENERATE("lbfgs", "cg", "gd");
    auto a = geometry(); a.pos_[0][0] = 1.0; SmoothPotential p; p.atom_weight = 1;
    RelaxationDrvV2 d; scalar_driver(d); d.control.driver = driver;
    d.control.constraint_iter_limit = 12; d.control.min_move = 1e-12;
    std::vector<regulator> regs{scalar()}; regs[0].weight_multply = 2;
    d.init(&p, a, &regs);
    const int result = d.run();
    CAPTURE(result, d.state_vars.curr_iter, d.state_vars.curr_constraint_iter, d.constraints[0].f, d.constraints[0].lambda, d.constraints[0].weight, d.g.norm());
    REQUIRE(result == 0); REQUIRE(d.state_vars.curr_iter > 0);
    REQUIRE(d.state_vars.curr_constraint_iter > 1);
    REQUIRE(d.constraints[0].lambda == Approx(0.05).margin(2e-6));
    REQUIRE(std::abs(d.constraints[0].f) <= regs[0].tol);
    REQUIRE(d.g.cwiseAbs().maxCoeff() <= d.tol.gtol);
}

TEST_CASE("AUL accepted checkpoint restores rejected GD residuals", "[relax_constraints]") {
    auto a = geometry(); a.pos_[0][0] = 1.001; SmoothPotential p;
    RelaxationDrvV2 d; scalar_driver(d); d.control.driver = "gd";
    d.control.constraint_iter_limit = 1; d.control.min_move = 0.1; d.control.max_move = 0.1;
    std::vector<regulator> regs{scalar()}; d.init(&p, a, &regs);
    REQUIRE(d.run() != 0); REQUIRE(d.state_vars.curr_iter == 0);
    REQUIRE(d.x[0] == Approx(1.001)); REQUIRE(d.constraints[0].f == Approx(0.001));
    REQUIRE(d.constraints[0].f == Approx(d.x[0] - d.constraints[0].g0));
}

TEST_CASE("AUL exceptions restore tolerance accepted geometry and regulator state", "[relax_constraints][errors]") {
    auto a = geometry(); a.pos_[0][0] = 1.5; SmoothPotential p; p.throw_after = 1;
    RelaxationDrvV2 d; scalar_driver(d); const double final_tol = d.tol.gtol;
    std::vector<regulator> regs{scalar()}; d.init(&p, a, &regs);
    REQUIRE_THROWS(d.run()); REQUIRE(d.stop_reason == "error"); REQUIRE_FALSE(d.state_vars.converged);
    REQUIRE(d.tol.gtol == final_tol); REQUIRE(d.x[0] == Approx(1.5)); REQUIRE(d.constraints[0].f == Approx(0.5));
    p.throw_after = -1; REQUIRE_NOTHROW(d.init(&p, a, &regs)); REQUIRE(d.run() == 0);
}

TEST_CASE("AUL checks nonfinite residuals and updates atomically", "[relax_constraints][errors]") {
    auto a = geometry(); SmoothPotential p; RelaxationDrvV2 d; scalar_driver(d);
    std::vector<regulator> regs{scalar(), scalar()}; d.init(&p, a, &regs);
    d.constraints[0].f = 1; d.constraints[1].f = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(d.check_cons_convergence()); REQUIRE_THROWS(d.update_regs());
    REQUIRE(d.constraints[0].lambda == 0); REQUIRE(d.constraints[0].weight == 1);
    d.constraints[1].f = 1; d.constraints[1].weight_multply = std::numeric_limits<double>::max();
    d.constraints[1].weight = 2;
    REQUIRE_THROWS(d.update_regs()); REQUIRE(d.constraints[0].lambda == 0);
}

TEST_CASE("AUL global accepted step budget and infeasible equalities never report success", "[relax_constraints]") {
    auto a = geometry(); SmoothPotential p; RelaxationDrvV2 d; scalar_driver(d);
    d.control.max_iter = 2; std::vector<regulator> regs{scalar(1), scalar(2)};
    d.init(&p, a, &regs); REQUIRE(d.run() != 0); REQUIRE_FALSE(d.state_vars.converged);
    REQUIRE(d.state_vars.curr_iter <= 2); REQUIRE_FALSE(d.check_cons_convergence());
}

TEST_CASE("Four raw9 lattice equalities converge to noninitial targets", "[relax_constraints]") {
    const int kind = GENERATE(regulator::CONSTANT_VOLUME, regulator::CONSTANT_VECTOR, regulator::CONSTANT_ANGLE, regulator::COUPLE_VECTOR);
    auto a = geometry(); SmoothPotential p; RelaxationDrvV2 d(raw_settings());
    std::vector<regulator> regs{lattice(kind)};
    regs = d.prepare_constraints(a, regs);
    regs[0].to_target_g0 = true; regs[0].g0 *= 1.02; regs[0].lambda_init = 0;
    regs[0].tol = 1e-7;
    d.control.constraint_iter_limit = 12;
    const auto initial = a.cell_; d.init(&p, a, &regs);
    const int result = d.run();
    CAPTURE(kind, result, d.state_vars.curr_iter, d.state_vars.curr_constraint_iter, d.constraints[0].f, d.constraints[0].lambda, d.constraints[0].weight, d.g.norm());
    REQUIRE(result == 0); REQUIRE(std::abs(d.constraints[0].f) <= regs[0].tol);
    REQUIRE_FALSE(d.atom.cell_.isApprox(initial, 1e-6)); REQUIRE(d.state_vars.curr_iter > 0);
}
TEST_CASE("Direct AUL invocation requires evaluated accepted state", "[relax_constraints][errors]") {
    RelaxationDrvV2 d; REQUIRE_THROWS(d.run_aul());
    auto a = geometry(); SmoothPotential p; scalar_driver(d);
    std::vector<regulator> regs{scalar()}; d.init(&p, a, &regs);
    REQUIRE_THROWS(d.run_aul()); REQUIRE(p.calls == 0);
}

TEST_CASE("One sided penalty convergence uses the callback sign convention", "[relax_constraints]") {
    RelaxationDrvV2 d;
    regulator r; r.type = 1; r.f = 2; d.constraints = {r};
    REQUIRE(d.check_cons_convergence());
    d.constraints[0].f = -2; REQUIRE_FALSE(d.check_cons_convergence());
    d.constraints[0].type = -1; REQUIRE(d.check_cons_convergence());
    d.constraints[0].f = 2; REQUIRE_FALSE(d.check_cons_convergence());
}
TEST_CASE("Zero relaxation logging interval rejects before evaluation", "[relax_constraints][errors]") {
    RelaxationDrvV2 d; d.control.nblock = 0; auto a = geometry(); SmoothPotential p;
    REQUIRE_THROWS(d.init(&p, a)); REQUIRE(p.calls == 0);
}

TEST_CASE("Real Lennard Jones raw9 AUL objective matches finite differences", "[relax_constraints][fd]") {
    const int kind = GENERATE(regulator::CONSTANT_VOLUME, regulator::CONSTANT_VECTOR, regulator::CONSTANT_ANGLE, regulator::COUPLE_VECTOR);
    const bool atoms = GENERATE(false, true);
    const bool recast = GENERATE(false, true);
    CAPTURE(kind, atoms, recast);
    GPSettings potential_settings;
    potential_settings.Modify("ntypes", "1"); potential_settings.Modify("0-0", "1 1");
    potential_settings.Modify("cut:0-0", "2");
    Potentials::LJ612 p(potential_settings);
    auto a = geometry(); auto settings = raw_settings(atoms);
    settings.Modify("atomic_force_recast", recast ? "true" : "false");
    RelaxationDrvV2 d(settings);
    std::vector<regulator> regs{lattice(kind)}; d.init(&p, a, &regs);
    auto point = d.x; point[1] += 0.08; point[3] -= 0.04; point[8] += 0.03;
    if (atoms) point[9] += 0.025;
    fd(d, point);
    REQUIRE(std::abs(d.atom.energy_) > 0.1);
    REQUIRE(d.atom.forces_[0].norm() > 0.01);
}

TEST_CASE("Constrained joint relaxation minimizes a real Lennard Jones pair", "[relax_constraints]") {
    const std::string optimizer = GENERATE("lbfgs", "lbfgspp");
    GPSettings potential_settings;
    potential_settings.Modify("ntypes", "1"); potential_settings.Modify("0-0", "1 1");
    potential_settings.Modify("cut:0-0", "2");
    Potentials::LJ612 p(potential_settings);
    auto a = geometry(); const double initial_distance = (a.pos_[1] - a.pos_[0]).norm();
    auto settings = raw_settings(true);
    settings.Modify("driver", optimizer); settings.Modify("constraint_loops", "20");
    settings.Modify("lbfgspp:max_linesearch", "80");
    settings.Modify("maxiteration", "600"); settings.Modify("gtol", "1e-6");
    RelaxationDrvV2 d(settings);
    auto r = lattice(regulator::CONSTANT_VOLUME);
    r.to_target_g0 = true; r.g0 = a.get_volume() * 1.005; r.lambda_init = 0; r.tol = 1e-6;
    std::vector<regulator> regs{r}; d.init(&p, a, &regs);
    const int result = d.run();
    CAPTURE(result, d.state_vars.curr_iter, d.state_vars.curr_constraint_iter, d.constraints[0].f, d.g.norm());
    REQUIRE(result == 0);
    REQUIRE(d.atom.get_volume() == Approx(r.g0).margin(r.tol));
    REQUIRE(d.atom.energy_ == Approx(-1).margin(1e-8));
    const double distance = (d.atom.pos_[1] - d.atom.pos_[0]).norm();
    REQUIRE(distance == Approx(std::pow(2.0, 1.0 / 6.0)).margin(1e-6));
    REQUIRE(std::abs(distance - initial_distance) > 0.01);
    REQUIRE(d.g.cwiseAbs().maxCoeff() <= d.tol.gtol);
}

TEST_CASE("AUL initial tolerance is configurable through settings", "[relax_constraints]") {
    GPSettings settings;
    settings.Modify("with_constraint_aul", "true");
    settings.Modify("gtol", "0.2"); settings.Modify("gtol0", "0.5");
    RelaxationDrvV2 d(settings);
    REQUIRE(d.tol.gtol0 == Approx(0.5)); REQUIRE(d.tol.gtol == Approx(0.2));
    REQUIRE_NOTHROW(d.validate_options());
}

TEST_CASE("AUL LBFGSpp line search budget failure restores accepted state", "[relax_constraints][errors]") {
    auto a = geometry(); a.pos_[0][0] = 1.00000001; SmoothPotential p;
    RelaxationDrvV2 d; scalar_driver(d); d.control.driver = "lbfgspp";
    d.control.constraint_iter_limit = 1; d.lbfgspp_param.max_linesearch = 1;
    auto r = scalar(); r.weight_init = 10000; std::vector<regulator> regs{r};
    d.init(&p, a, &regs);
    REQUIRE_THROWS(d.run()); REQUIRE(d.stop_reason == "error"); REQUIRE_FALSE(d.state_vars.converged);
    REQUIRE(d.state_vars.curr_iter == 0); REQUIRE(d.x[0] == a.pos_[0][0]);
    REQUIRE(d.constraints[0].f == Approx(a.pos_[0][0] - r.g0).margin(1e-15));
    REQUIRE(d.constraints[0].lambda == 0); REQUIRE(d.constraints[0].weight == 10000);
    REQUIRE(d.tol.gtol == 1e-7);
    d.lbfgspp_param.max_linesearch = 80;
    REQUIRE(d.run() == 0); REQUIRE(d.state_vars.curr_iter > 0);
    REQUIRE(d.g.cwiseAbs().maxCoeff() <= d.tol.gtol);
}
