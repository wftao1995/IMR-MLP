#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include <limits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <memory>

using Catch::Approx;
namespace {
Atoms fire_atoms() {
    Atoms a;
    a.cell_ = 10 * Eigen::Matrix3d::Identity();
    a.pbc_ = {false, false, false};
    a.pos_ = {{1, 0, 0}};
    a.chemmap_ = {0};
    return a;
}
struct FirePotential : Potentials::AnyPotential {
    int calls = 0, throw_after = -1;
    double stiffness = 10;
    bool cell = false, vector = false;
    double energy_offset = 0;
    Eigen::RowVector3d curvature{1, 4, 2};
    std::vector<double> samples;
    void evaluate(Atoms &a) {
        if (calls == throw_after) throw std::runtime_error("FIRE injected potential error");
        ++calls;
        const double q = cell ? a.cell_(0, 0) / 10 - 0.8 : a.pos_[0][0];
        samples.push_back(q);
        a.energy_ = energy_offset + 0.5 * stiffness * q * q;
        a.forces_.assign(a.size(), Eigen::RowVector3d::Zero());
        a.stress_.setZero();
        if (cell) {
            a.stress_.diagonal().setConstant(-stiffness * q * (q + 0.8) / 3);
            if (vector) {
                a.energy_ += 0.5 * a.pos_[0].cwiseProduct(curvature).dot(a.pos_[0]);
                a.forces_[0] = -a.pos_[0].cwiseProduct(curvature);
            }
        }
        else if (vector) {
            a.energy_ = energy_offset + 0.5 * a.pos_[0].cwiseProduct(curvature).dot(a.pos_[0]);
            a.forces_[0] = -a.pos_[0].cwiseProduct(curvature);
        } else a.forces_[0][0] = -stiffness * q;
    }
    void getEFSOMP(Atoms &a) override { evaluate(a); }
    void getEFS(Atoms &a, AnyNeighborhoods &) override { evaluate(a); }
};
std::unique_ptr<RelaxationDrvV2> fire_driver(FirePotential &p, int budget, bool downhill = true,
                                            bool cell = false, std::vector<regulator> regs = {}) {
    GPSettings s;
    s.Modify("driver", "fire");
    s.Modify("write_traj", "false");
    s.Modify("cartesian", "true");
    s.Modify("atomic_force_recast", "false");
    s.Modify("dprecond", "false");
    auto d = std::make_unique<RelaxationDrvV2>(s);
    d->flags.fix_shape = cell;
    d->flags.relax_atom = !cell;
    d->flags.with_constraint_aul = !regs.empty();
    d->control.max_iter = budget;
    d->control.max_move = 100;
    d->first_step_delay = 0;
    d->tol.gtol = 1e-10;
    d->tol.ftol = d->tol.xtol = 0;
    d->buffer_fire.downhill_check = downhill;
    auto a = fire_atoms();
    d->init(&p, a, regs.empty() ? nullptr : &regs);
    d->LatPos2X();
    d->eval();
    d->accept_checkpoint(false);
    return d;
}
regulator fire_constraint() {
    regulator r;
    r.type = 10; r.g0 = 2; r.weight_init = 100; r.lambda_init = 3;
    r.inputs = {0}; r.outputs = {0}; r.x.resize(1); r.g.resize(1);
    r.compute = [](double *x, int, double *g, int, double *f, double target, Atoms *, int *, void *) {
        *f = x[0] - target; g[0] = 1;
    };
    return r;
}
}

TEST_CASE("V2 FIRE downhill restores rejected trials and retries with one timestep reduction", "[relax_fire]") {
    FirePotential p;
    auto d = fire_driver(p, 4);
    d->run_fire();
    REQUIRE(d->state_vars.curr_iter == 4);
    REQUIRE(p.calls == 5);
    // Zero velocity and one kick per retry give dx=-dt^2*g.
    REQUIRE(p.samples[1] == Approx(-39));
    REQUIRE(p.samples[2] == Approx(-9));
    REQUIRE(p.samples[3] == Approx(-1.5));
    REQUIRE(p.samples[4] == Approx(0.375));
    REQUIRE(d->x[0] == Approx(0.375));
    REQUIRE(d->f < 5);
    REQUIRE(d->g[0] == Approx(10 * d->x[0]));
    REQUIRE(d->atom.forces_[0][0] == Approx(-d->g[0]));
    REQUIRE(d->checkpoint_g.isApprox(d->g));
    REQUIRE(d->x_prev[0] == Approx(1));
    REQUIRE(d->f_prev == Approx(5));
}

TEST_CASE("V2 FIRE trial budgets include rejections and false mode accepts uphill", "[relax_fire]") {
    const int budget = GENERATE(0, 1, 2);
    const bool downhill = GENERATE(false, true);
    FirePotential p;
    auto d = fire_driver(p, budget, downhill);
    d->run_fire();
    REQUIRE(d->state_vars.curr_iter == budget);
    REQUIRE(p.calls == budget + 1 + (!downhill && budget == 2 ? 1 : 0));
    if (!downhill && budget == 2) REQUIRE(p.samples[2] == Approx(-29));
    if (downhill || budget == 0) {
        REQUIRE(d->x[0] == Approx(1));
        REQUIRE(d->f == Approx(5));
        REQUIRE(d->g[0] == Approx(10));
        REQUIRE(d->atom.pos_[0][0] == Approx(1));
    } else REQUIRE(d->f > 5);
}

TEST_CASE("V2 FIRE tries the minimum timestep before reporting failure", "[relax_fire]") {
    const int budget = GENERATE(1, 10);
    FirePotential p;
    auto d = fire_driver(p, budget);
    d->buffer_fire.ts_min = 1;
    d->tol.xtol = d->tol.ftol = 1e9;
    d->run_fire();
    REQUIRE(d->state_vars.curr_iter == std::min(budget, 2));
    REQUIRE(p.calls == std::min(budget, 2) + 1);
    REQUIRE(d->x[0] == Approx(1));
    REQUIRE_FALSE(d->state_vars.converged);
    REQUIRE(d->state_vars.reach_ftol == 0);
    REQUIRE(d->state_vars.reach_xtol == 0);
    REQUIRE((d->stop_reason == "minimum_step") == (budget >= 2));
}

TEST_CASE("V2 FIRE keeps the unpreconditioned gradient for stopping and checkpoints", "[relax_fire]") {
    const bool downhill = GENERATE(false, true);
    FirePotential p;
    auto d = fire_driver(p, 1, downhill);
    d->prec_scaling << 1e12, 1, 1e6;
    d->tol.gtol = 1e-4;
    d->run_fire();
    REQUIRE(d->state_vars.curr_iter == 1);
    REQUIRE_FALSE(d->state_vars.converged);
    REQUIRE(d->g[0] > 9);
    REQUIRE(d->g[0] == Approx(10 * d->x[0]));
    REQUIRE(d->checkpoint_g.isApprox(d->g));
}

TEST_CASE("V2 FIRE accepts equal objectives and gates the complete AUL objective", "[relax_fire]") {
    FirePotential p;
    SECTION("equal objective is accepted") {
        p.stiffness = 2;
        auto d = fire_driver(p, 1);
        // Zero initial velocity and one kick give exactly dx=-2 at dt=1.
        d->buffer_fire.alpha_start = 0;
        d->buffer_fire.ts_init = 1;
        d->run_fire();
        REQUIRE(d->x[0] == Approx(-1));
        REQUIRE(d->checkpoint_x[0] == Approx(-1));
    }
    SECTION("physical energy can rise while the penalized objective falls") {
        auto d = fire_driver(p, 1, true, false, {fire_constraint()});
        const double baseline = d->f;
        d->buffer_fire.ts_init = 0.02;
        d->run_fire();
        REQUIRE(d->atom.energy_ > 5);
        REQUIRE(d->f < baseline);
        REQUIRE(d->x[0] > 1);
        REQUIRE(d->constraints[0].f == Approx(d->x[0] - 2));
    }
    SECTION("rejection restores regulator buffers and multiplier state") {
        auto d = fire_driver(p, 1, true, false, {fire_constraint()});
        const auto before = d->constraints[0];
        d->run_fire();
        const auto &after = d->constraints[0];
        REQUIRE(d->x[0] == Approx(1));
        REQUIRE(after.f == Approx(before.f));
        REQUIRE(after.x == before.x);
        REQUIRE(after.g == before.g);
        REQUIRE(after.lambda == before.lambda);
        REQUIRE(after.weight == before.weight);
        REQUIRE(after.update == before.update);
    }
}

TEST_CASE("V2 FIRE restores fixed shape scaling at rejected and exceptional exits", "[relax_fire]") {
    const bool failure = GENERATE(false, true);
    FirePotential p; p.cell = true;
    auto d = fire_driver(p, 1, true, true);
    const auto cell = d->atom.cell_;
    if (failure) {
        p.throw_after = p.calls + 1; // run() evaluates its accepted baseline first.
        REQUIRE_THROWS_AS(d->run(), std::runtime_error);
        REQUIRE(d->stop_reason == "error");
    } else d->run_fire();
    REQUIRE(d->scaling == Approx(1));
    REQUIRE(d->checkpoint_scaling == Approx(1));
    REQUIRE(d->x[0] == Approx(1));
    REQUIRE(d->atom.cell_.isApprox(cell));
    REQUIRE(d->f == Approx(0.2));
}

TEST_CASE("V2 FIRE rejects invalid dynamics and preconditioning at direct entry", "[relax_fire]") {
    FirePotential p;
    auto d = fire_driver(p, 0);
    SECTION("initial timestep exceeds maximum") { d->buffer_fire.ts_init = 11; }
    SECTION("nonpositive minimum timestep") { d->buffer_fire.ts_min = 0; }
    SECTION("invalid timestep reduction") { d->buffer_fire.ts_reduce = 1; }
    SECTION("invalid mixing") { d->buffer_fire.alpha_start = 1.1; }
    SECTION("unsupported integration") { d->buffer_fire.mdint = 1; }
    SECTION("nonfinite timestep") { d->buffer_fire.ts_init = std::numeric_limits<double>::infinity(); }
    SECTION("negative delay") { d->buffer_fire.ndelay = -1; }
    SECTION("zero preconditioning") { d->prec_scaling[0] = 0; }
    SECTION("nonfinite preconditioning") { d->prec_scaling[0] = std::numeric_limits<double>::quiet_NaN(); }
    SECTION("wrong preconditioner dimensions") { d->prec_scaling.resize(0); }
    REQUIRE_THROWS_AS(d->run_fire(), std::invalid_argument);
    REQUIRE(p.calls == 1);
    REQUIRE(d->state_vars.curr_iter == 0);
}

TEST_CASE("V2 FIRE trajectory output contains accepted trials only", "[relax_fire]") {
    const int budget = GENERATE(1, 4);
    FirePotential p;
    auto d = fire_driver(p, budget);
    char name[] = "/tmp/imrmlp_fire_trajectory_XXXXXX";
    const int fd = mkstemp(name);
    REQUIRE(fd >= 0);
    close(fd);
    struct Cleanup { const char *path; ~Cleanup() { std::filesystem::remove(path); } } cleanup{name};
    d->flags.write_traj = true;
    d->control.nblock = 1;
    d->ofs.open(name);
    REQUIRE(d->iobj.seq_write_init(d->ofs, "mlpcfg", name));
    d->run_fire();
    d->ofs.close();
    std::ifstream in(name);
    const std::string output{std::istreambuf_iterator<char>(in), {}};
    if (budget == 1) REQUIRE(output.empty());
    else {
        REQUIRE(output.find("BEGIN_CFG") != std::string::npos);
        REQUIRE(output.find("BEGIN_CFG", output.find("BEGIN_CFG") + 1) == std::string::npos);
    }
}

TEST_CASE("V2 FIRE AUL inner loops share the trial budget and refresh dual baselines", "[relax_fire]") {
    FirePotential p;
    auto d = fire_driver(p, 4, true, false, {fire_constraint()});
    d->buffer_fire.ts_init = 0.02;
    d->control.constraint_iter_limit = 3;
    d->tol.gtol0 = 1e3; // First inner solve exits; multiplier update changes f at the same x.
    d->run_aul();
    REQUIRE(d->state_vars.curr_iter == 4);
    REQUIRE(d->state_vars.curr_constraint_iter >= 2);
    REQUIRE(d->constraints[0].lambda != Approx(3));
    REQUIRE(d->x[0] > 1.1);
    REQUIRE(d->checkpoint_f < 152.5); // Baseline with lambda=-97 and weight=101 at x=1.
    REQUIRE(d->g.isApprox(d->checkpoint_g));
    REQUIRE(d->f == d->checkpoint_f);
}

TEST_CASE("V2 FIRE retries correctly after restoring reduced position coordinates", "[relax_fire]") {
    FirePotential p;
    auto d = fire_driver(p, 4);
    auto a = fire_atoms();
    a.pos_[0][1] = 2;
    a.sel_dynpos_ = {{1, 0, 0}};
    d->flags.selective_dynamics = true;
    d->init(&p, a);
    d->apply_modifiers(d->modifiers_list);
    d->eval();
    d->accept_checkpoint(false);
    d->run_fire();
    REQUIRE(d->x.size() == 1);
    REQUIRE(d->x[0] == Approx(0.375));
    REQUIRE(d->atom.pos_[0][0] == Approx(0.375));
    REQUIRE(d->atom.pos_[0][1] == Approx(2));
    REQUIRE(d->atom.pos_[0][2] == Approx(0));
    REQUIRE(d->g[0] == Approx(3.75));
}

TEST_CASE("V2 FIRE public run reports minimum-step and iteration exits at the accepted state", "[relax_fire]") {
    const int budget = GENERATE(0, 1, 10);
    const bool stationary = GENERATE(false, true);
    FirePotential p;
    auto d = fire_driver(p, budget);
    if (stationary) p.stiffness = 0;
    p.calls = 0;
    d->buffer_fire.ts_min = 1;
    const int result = d->run();
    REQUIRE(d->x[0] == Approx(1));
    REQUIRE(d->atom.pos_[0][0] == Approx(1));
    REQUIRE(d->scaling == Approx(d->checkpoint_scaling));
    if (stationary) {
        REQUIRE(result == 0);
        REQUIRE(p.calls == 1);
        REQUIRE(d->state_vars.curr_iter == 0);
        REQUIRE(d->stop_reason == "gradient_tolerance");
    } else {
        REQUIRE(result == -1);
        REQUIRE(p.calls == std::min(budget, 2) + 1);
        REQUIRE(d->state_vars.curr_iter == std::min(budget, 2));
        REQUIRE(d->stop_reason == (budget >= 2 ? "minimum_step" : "iteration_limit"));
        REQUIRE(d->g[0] == Approx(10));
    }
}

TEST_CASE("V2 FIRE permits explicit small timesteps below the default minimum", "[relax_fire]") {
    const bool downhill = GENERATE(false, true);
    const bool rejection = GENERATE(false, true);
    FirePotential p;
    auto d = fire_driver(p, rejection && !downhill ? 2 : 1, downhill);
    d->buffer_fire.ts_init = 1e-5;
    d->buffer_fire.ts_max = 5e-4;
    REQUIRE(d->buffer_fire.ts_min == Approx(0.001));
    REQUIRE_NOTHROW(d->validate_options());
    if (rejection) d->prec_scaling[0] = 1e-12;
    d->run_fire();
    // Neither a rejection nor P<=0 may raise the effective floor above ts_init.
    REQUIRE(d->buffer_fire.ts == Approx(1e-5));
    REQUIRE(d->buffer_fire.ts <= d->buffer_fire.ts_max);
    if (rejection && downhill) {
        REQUIRE(d->stop_reason == "minimum_step");
        REQUIRE(d->state_vars.curr_iter == 1);
        REQUIRE(d->x[0] == Approx(1));
        REQUIRE(d->g[0] == Approx(10));
    } else {
        REQUIRE(d->state_vars.curr_iter == (rejection ? 2 : 1));
        if (!rejection) REQUIRE(d->f < 5);
    }
}

TEST_CASE("V2 FIRE mixes the same two-dimensional velocity before one Euler kick", "[relax_fire]") {
    FirePotential p; p.vector = true;
    auto d = fire_driver(p, 1);
    auto a = fire_atoms(); a.pos_[0] = {1, 0.5, 0};
    d->init(&p, a);
    d->buffer_fire.ts_init = 0.1;
    d->run();
    REQUIRE(d->atom.pos_[0][0] == Approx(0.99));
    REQUIRE(d->atom.pos_[0][1] == Approx(0.48));
    REQUIRE(d->buffer_fire.ts == Approx(0.1));
    // Two steps from the same initial point: v1=(-.1,-.2), then mix v1
    // with the new force (-.99,-1.92) and apply just one dt*F kick.
    d->init(&p, a); d->control.max_iter = 2;
    d->run();
    REQUIRE(d->atom.pos_[0][0] == Approx(0.9700380858620663));
    REQUIRE(d->atom.pos_[0][1] == Approx(0.4408314392476437));
}

TEST_CASE("V2 FIRE bounds a three-dimensional displacement without enlarging it", "[relax_fire]") {
    FirePotential p; p.vector = true; p.curvature.setOnes();
    auto d = fire_driver(p, 1);
    auto a = fire_atoms(); a.pos_[0].setOnes();
    d->init(&p, a); d->control.max_move = 1;
    d->buffer_fire.ts_init = 1;
    d->run();
    REQUIRE((d->atom.pos_[0] - a.pos_[0]).norm() == Approx(1));
    REQUIRE(d->atom.pos_[0][0] == Approx(1 - 1 / std::sqrt(3.0)));
    REQUIRE(d->atom.pos_[0][1] == Approx(d->atom.pos_[0][0]));
    REQUIRE(d->atom.pos_[0][2] == Approx(d->atom.pos_[0][0]));
    d->init(&p, a); d->buffer_fire.ts_init = 0.5;
    d->run();
    REQUIRE(d->atom.pos_[0].isApprox(Eigen::RowVector3d::Constant(0.75)));
}

TEST_CASE("V2 FIRE alpha one restarts with a full acceleration after rejection", "[relax_fire]") {
    FirePotential p;
    auto d = fire_driver(p, 4);
    d->buffer_fire.alpha_start = 1;
    d->run_fire();
    REQUIRE(d->state_vars.curr_iter == 4);
    REQUIRE(d->x[0] == Approx(0.375));
    REQUIRE(d->f < 5);
    REQUIRE(d->g[0] == Approx(3.75));
}

TEST_CASE("V2 FIRE derives automatic initial limits for each run and caps explicit limits", "[relax_fire]") {
    FirePotential p;
    auto d = fire_driver(p, 1);
    d->first_step_delay = 1;
    REQUIRE(d->first_step < 0);
    d->control.max_move = 0.001;
    auto start = d->atom.pos_[0];
    d->run();
    REQUIRE((d->atom.pos_[0] - start).norm() == Approx(0.0001));
    REQUIRE(d->first_step < 0);
    start = d->atom.pos_[0];
    d->control.max_move = 0.00002;
    d->run();
    REQUIRE((d->atom.pos_[0] - start).norm() == Approx(0.000002));
    REQUIRE(d->first_step < 0);
    start = d->atom.pos_[0];
    d->first_step = 1;
    d->run();
    REQUIRE((d->atom.pos_[0] - start).norm() == Approx(d->control.max_move));
}

TEST_CASE("V2 FIRE clears uphill velocity even when position backtracking is disabled", "[relax_fire]") {
    FirePotential p;
    auto d = fire_driver(p, 2, false);
    d->buffer_fire.halvstepback = false;
    d->buffer_fire.ts_init = 0.4;
    d->run_fire();
    REQUIRE(p.calls == 3);
    REQUIRE(p.samples[1] == Approx(-0.6));
    REQUIRE(p.samples[2] == Approx(-0.36));
    REQUIRE(d->buffer_fire.ts == Approx(0.2));
    REQUIRE(d->f < 0.5 * p.stiffness * 0.6 * 0.6);
}

TEST_CASE("V2 FIRE leaves a frozen scale and the first reduced position intact", "[relax_fire]") {
    FirePotential p; p.cell = true; p.vector = true;
    auto d = fire_driver(p, 1, true, true);
    char name[] = "/tmp/imrmlp_fire_scale_mask_XXXXXX";
    const int fd = mkstemp(name); REQUIRE(fd >= 0); close(fd);
    struct Cleanup { const char *path; ~Cleanup() { std::filesystem::remove(path); } } cleanup{name};
    std::ofstream(name) << "4 0 1 0 0\n";
    d->flags.relax_atom = true;
    d->flags.selective_dynamics = true;
    d->control.seldyn_file = name;
    d->buffer_fire.ts_init = 0.1;
    auto a = fire_atoms(); a.pos_[0][0] = 2;
    d->init(&p, a);
    d->run();
    REQUIRE(d->x.size() == 1);
    REQUIRE(d->x[0] == Approx(1.98));
    REQUIRE(d->atom.pos_[0][0] == Approx(1.98));
    REQUIRE(d->scaling == Approx(1));
    REQUIRE(d->atom.cell_.isApprox(a.cell_));
    REQUIRE(d->checkpoint_scaling == Approx(1));
}

TEST_CASE("V2 FIRE AUL iterations are unchanged by a constant physical energy shift", "[relax_fire]") {
    FirePotential p;
    auto d = fire_driver(p, 8, true, false, {fire_constraint()});
    d->buffer_fire.ts_init = 0.001;
    d->control.constraint_iter_limit = 3;
    d->tol.gtol0 = 1e3;
    d->run();
    const auto point = d->x, gradient = d->g;
    const double objective = d->f;
    REQUIRE(d->state_vars.curr_iter == 8);
    REQUIRE(d->f > 10 * std::abs(d->energy0));
    FirePotential shifted; shifted.energy_offset = 10000;
    auto translated = fire_driver(shifted, 8, true, false, {fire_constraint()});
    translated->buffer_fire.ts_init = d->buffer_fire.ts_init;
    translated->control.constraint_iter_limit = 3;
    translated->tol.gtol0 = 1e3;
    translated->run();
    REQUIRE(translated->state_vars.curr_iter == 8);
    REQUIRE(translated->x.isApprox(point));
    REQUIRE(translated->g.isApprox(gradient));
    REQUIRE(translated->f == Approx(objective + shifted.energy_offset));
}
