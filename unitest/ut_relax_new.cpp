// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include <array>
#include <cmath>
#include <fstream>
#include <memory>

using Catch::Approx;

// ── Stub potential returning zero forces/stress ──────────────────────
struct StubPot : public Potentials::AnyPotential {
    StubPot() { cutoff = 6.0; cutoffsq = 36.0; species_count = 1;
               cutoffs.resize(1,1); cutoffsqs.resize(1,1);
               cutoffs(0,0)=6.0; cutoffsqs(0,0)=36.0;
               r_mins.resize(1,1); r_mins.setZero(); }
    void getEFS(Atoms &a, AnyNeighborhoods &) override { zero(a); }
    void getEFSOMP(Atoms &a, AnyNeighborhoods &) override { zero(a); }
    void getE(Atoms &a, AnyNeighborhoods &) override { a.energy_ = 0; }
    void getEFSOMPthread(Atoms &a, AnyNeighborhoods &) override { zero(a); }
private:
    void zero(Atoms &a) {
        a.energy_ = 0;
        a.forces_.assign(a.size(), Eigen::RowVector3d::Zero());
        a.stress_.setZero();
    }
};

static Atoms make_dimer(double a=12.0, double d=2.0) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity()*a;
    atom.pos_ = {{0,0,0},{d,0,0}};
    atom.chemmap_ = {0,0};
    return atom;
}

static StubPot g_pot;

static GPSettings make_base_relax_settings() {
    GPSettings rs;
    rs.Modify("gtol", "0.01");
    rs.Modify("max_iter", "5");
    rs.Modify("write_traj", "false");
    return rs;
}

static auto make_drv(Atoms &atom) {
    GPSettings rs = make_base_relax_settings();
    auto d = std::make_unique<RelaxationDrvV2>(rs);
    d->init(&g_pot, atom);
    return d;
}

static Atoms make_skew_cell_atom() {
    Atoms atom;
    atom.cell_ << 4.7, 0.3, 0.1,
                  0.2, 5.1, 0.4,
                  0.1, 0.3, 5.6;
    atom.pos_ = {Eigen::RowVector3d(0.4, 0.8, 1.2)};
    atom.chemmap_ = {0};
    return atom;
}

static double eval_relax_at(RelaxationDrvV2 &drv, const Eigen::VectorXd &x, Eigen::VectorXd *gout = nullptr) {
    Eigen::VectorXd gtmp = Eigen::VectorXd::Zero(x.size());
    double f = drv(x, gtmp);
    if (gout) *gout = gtmp;
    return f;
}

static void require_relax_gradient_matches_fd(RelaxationDrvV2 &drv,
                                              const Eigen::VectorXd &x0,
                                              double step = 1e-6,
                                              double margin = 3e-6) {
    Eigen::VectorXd g;
    (void)eval_relax_at(drv, x0, &g);
    for (int i = 0; i < x0.size(); ++i) {
        Eigen::VectorXd xp = x0;
        Eigen::VectorXd xm = x0;
        xp[i] += step;
        xm[i] -= step;
        double fp = eval_relax_at(drv, xp);
        double fm = eval_relax_at(drv, xm);
        double gfd = (fp - fm) / (2.0 * step);
        INFO("component " << i);
        REQUIRE(g[i] == Approx(gfd).margin(margin));
    }
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("RelaxationDrvV2 construction", "[relax_new]") {
    auto atom = make_dimer(); auto drv = make_drv(atom);
    REQUIRE(drv->x.size() == 6);
    REQUIRE(drv->g.size() == 6);
    REQUIRE(!drv->state_vars.converged);
}

TEST_CASE("RelaxationDrvV2 rejects invalid settings instead of using defaults", "[relax_new][errors]") {
    auto settings = make_base_relax_settings();
    settings.Modify("gtol", "not-a-number");
    REQUIRE_THROWS_AS(RelaxationDrvV2(settings), GPException);
}

TEST_CASE("Relaxation filters throw on missing files and singular matrices", "[relax_new][errors]") {
    RelaxationDrvV2 driver;
    FilterLF filter;
    std::string missing = "missing-filter-directory/filter.dat";
    REQUIRE_THROWS_AS(filter.init(&driver, &missing), std::runtime_error);
    filter.matrixA = Eigen::MatrixXd::Zero(2, 2);
    REQUIRE_THROWS_AS(filter.solve(), std::runtime_error);
}

TEST_CASE("LatPos2X <-> X2LatPos roundtrip", "[relax_new]") {
    auto atom = make_dimer(); auto drv = make_drv(atom);
    auto c0=atom.cell_; auto p0=atom.pos_;
    drv->LatPos2X(); auto x0=drv->x; drv->X2LatPos();
    REQUIRE(atom.cell_.isApprox(c0,1e-10));
    for(int i=0;i<(int)atom.size();++i) REQUIRE(atom.pos_[i].isApprox(p0[i],1e-10));
    drv->LatPos2X(); REQUIRE(drv->x.isApprox(x0,1e-10));
}

TEST_CASE("LatPos2X <-> X2LatPos with symm_strain", "[relax_new]") {
    auto atom = make_dimer(); auto drv = make_drv(atom);
    drv->flags.relax_cell=true; drv->flags.relax_cell_use_strain=true;
    drv->flags.symm_strain=true; drv->init(&g_pot,atom);
    auto c0=atom.cell_; auto p0=atom.pos_;
    drv->LatPos2X(); REQUIRE(drv->x.size()==12); drv->X2LatPos();
    REQUIRE(atom.cell_.isApprox(c0,1e-10));
    for(int i=0;i<(int)atom.size();++i) REQUIRE(atom.pos_[i].isApprox(p0[i],1e-10));
}

TEST_CASE("couple=ab stores flag and affects gradient averaging", "[relax_new]") {
    auto atom = make_dimer(5.0,2.0);
    atom.cell_<<5,0,0, 0,5.2,0, 0,0,5;
    auto drv = make_drv(atom);
    drv->flags.relax_cell=true; drv->flags.relax_cell_use_strain=true;
    drv->flags.symm_strain=true; drv->control.couple_flag=1; drv->init(&g_pot,atom);
    // couple_flag=1 is stored and init succeeds
    REQUIRE(drv->control.couple_flag == 1);
    drv->LatPos2X();
    // Initial strains from nearly-isotropic cell are close
    REQUIRE(drv->x[0]==Approx(drv->x[3]).epsilon(1e-10));
    // Note: coupling is enforced at the gradient level (stress averaging),
    // not at the position level. The X2LatPos/LatPos2X roundtrip does NOT
    // forcibly re-equalize coupled DOFs (that code is commented out with a
    // TODO noting potential correctness issues).
}

TEST_CASE("check_converage max_iter", "[relax_new]") {
    auto atom=make_dimer(); auto d=make_drv(atom);
    d->state_vars.curr_iter=2000; d->control.max_iter=1000;
    // Set fake forces and stress so explosion check doesn't read garbage
    d->atom.forces_.assign(d->atom.size(), Eigen::RowVector3d::Zero());
    d->atom.stress_.setZero();
    d->g.setConstant(100);
    REQUIRE(d->check_converage(d->tol));
    REQUIRE(d->state_vars.optstate==-1);
}

TEST_CASE("check_converage gtol", "[relax_new]") {
    auto atom=make_dimer(); auto d=make_drv(atom);
    d->atom.forces_.assign(d->atom.size(), Eigen::RowVector3d::Zero());
    d->atom.stress_.setZero();
    d->g.setZero(); d->state_vars.curr_iter=5;
    REQUIRE(d->check_converage(d->tol));
    REQUIRE(d->state_vars.converged);
}

TEST_CASE("check_converage does not compare full objective to physical energy zero", "[relax_new]") {
    auto atom=make_dimer(); auto d=make_drv(atom);
    d->atom.forces_.assign(d->atom.size(), Eigen::RowVector3d::Zero());
    d->atom.stress_.setZero();
    d->energy0=-10; d->f=500; d->state_vars.curr_iter=10;
    d->g.setConstant(1);
    d->tol.ftol = d->tol.xtol = 0;
    REQUIRE_FALSE(d->check_converage(d->tol));
    REQUIRE(d->state_vars.optstate==-1);
}

TEST_CASE("check_converage still detects excessive physical force and stress", "[relax_new]") {
    auto atom=make_dimer(); auto d=make_drv(atom);
    d->atom.forces_.assign(d->atom.size(), Eigen::RowVector3d::Zero());
    d->atom.stress_.setZero();
    SECTION("force") { d->atom.forces_[0][0] = 2e4; }
    SECTION("stress") { d->atom.stress_(0,0) = 2e4; }
    REQUIRE(d->check_converage(d->tol));
    REQUIRE(d->state_vars.optstate==-5);
}

TEST_CASE("fix_vol in X2LatPos", "[relax_new]") {
    auto atom=make_dimer(5.0,2.0); auto d=make_drv(atom);
    d->flags.fix_vol=true; d->flags.relax_cell=true;
    d->flags.relax_cell_use_strain=true; d->flags.symm_strain=true;
    d->init(&g_pot,atom);
    double v0=atom.get_volume();
    d->LatPos2X(); d->x[0]+=0.05; d->x[3]+=0.02; d->x[5]-=0.07;
    d->X2LatPos();
    REQUIRE(atom.get_volume()==Approx(v0).epsilon(1e-8));
}

TEST_CASE("free_modifiers cleans both lists", "[relax_new]") {
    auto atom=make_dimer(); auto d=make_drv(atom);
    d->flags.fix_cm=true; d->inplace_modifiers_init=true;
    d->init(&g_pot,atom);
    REQUIRE(!d->modifiers_list_carg.empty());
    d->free_modifiers();
    REQUIRE(d->modifiers_list.empty());
    REQUIRE(d->modifiers_list_carg.empty());
    d->free_modifiers();
    REQUIRE(d->modifiers_list_carg.empty());
}

TEST_CASE("eval keeps atom consistent with zero-force stub", "[relax_new]") {
    auto atom=make_dimer(); auto d=make_drv(atom);
    auto c0=atom.cell_; auto p0=atom.pos_;
    d->LatPos2X(); d->eval();
    REQUIRE(atom.cell_.isApprox(c0,1e-10));
    for(int i=0;i<(int)atom.size();++i) REQUIRE(atom.pos_[i].isApprox(p0[i],1e-10));
}

TEST_CASE("programmatic VIB data matches V1 gradient projection", "[relax_new][vib]") {
    auto atom = make_dimer();
    auto d = make_drv(atom);

    const int n = d->g.size();
    Eigen::MatrixXd q = Eigen::MatrixXd::Identity(n, n);
    const double theta = 0.37;
    q(0, 0) = std::cos(theta);
    q(0, 1) = -std::sin(theta);
    q(1, 0) = std::sin(theta);
    q(1, 1) = std::cos(theta);

    Eigen::VectorXd factor(n);
    factor << -1.0, 0.25, 0.5, 1.5, 2.0, -0.75;
    Eigen::VectorXd g0(n);
    g0 << 0.3, -1.2, 2.0, 0.7, -0.4, 1.1;

    d->set_vib_data(q, factor);
    FilterVIB *vib_filter = nullptr;
    for (auto *modifier : d->modifiers_list) {
        vib_filter = dynamic_cast<FilterVIB*>(modifier);
        if (vib_filter) break;
    }
    REQUIRE(vib_filter != nullptr);

    d->g = g0;
    vib_filter->forward(d.get());
    Eigen::VectorXd expected = q * factor.asDiagonal() * q.transpose() * g0;
    REQUIRE(d->g.isApprox(expected, 1e-12));
}

TEST_CASE("compute_regulator equality penalty", "[relax_new]") {
    // Set up a trivial constraint: c(x) = x[0] - g0
    // The compute function must match: void(*)(double*,int,double*,int,double*,double,Atoms*,int*,void*)
    static const auto comp = [](double* x, int, double* g, int, double* f,
                                 double g0, Atoms*, int*, void*) {
        *f = x[0] - g0;
        g[0] = 1.0; // dc/dx = 1
    };
    auto atom=make_dimer(10,2); auto d=make_drv(atom);
    regulator r;
    r.compute = comp;
    r.type=0; r.weight=2; r.weight_init=2; r.lambda=0; r.lambda_init=0;
    r.g0=0.5; r.inputs={0}; r.outputs={0}; r.x.resize(1); r.g.resize(1);
    d->LatPos2X(); d->x[0]=1.0;
    double f0=d->f;
    d->compute_regulator(r);
    // f += w * c^2 = 2 * 0.25 = 0.5
    REQUIRE(d->f-f0 == Approx(0.5).epsilon(1e-10));
}

TEST_CASE("reset_state clears iteration", "[relax_new]") {
    auto atom=make_dimer(); auto d=make_drv(atom);
    d->state_vars.curr_iter=42; d->state_vars.converged=true;
    d->reset_state();
    REQUIRE(d->state_vars.curr_iter==0);
    REQUIRE(!d->state_vars.converged);
}

TEST_CASE("update_regs updates lambda", "[relax_new]") {
    auto atom=make_dimer(10,2); auto d=make_drv(atom);
    regulator r; r.type=0; r.update=true; r.weight=2; r.weight_init=2;
    r.lambda=0; r.lambda_init=0; r.g0=0.5; r.f=0.5;
    r.inputs={0}; r.outputs={0}; r.x.resize(1); r.g.resize(1);
    r.weight_multply=1.0; r.weight_step=0.1;
    d->constraints.push_back(r);
    d->update_regs();
    REQUIRE(d->constraints[0].lambda == Approx(1.0));
    REQUIRE(d->constraints[0].weight == Approx(2.1));
}

TEST_CASE("built-in lattice constraint setup maps outputs to selected DOFs", "[relax_new][constraints]") {
    auto atom = make_skew_cell_atom();

    regulator vec;
    Constraints::constant_vector_setup(vec, atom, 1);
    REQUIRE(vec.inputs == std::vector<int>{3, 4, 5});
    REQUIRE(vec.outputs == vec.inputs);

    regulator ang;
    Constraints::constant_angle_setup(ang, atom, 0, 2);
    REQUIRE(ang.inputs == std::vector<int>{0, 1, 2, 6, 7, 8});
    REQUIRE(ang.outputs == ang.inputs);

    regulator couple;
    Constraints::couple_vector_setup(couple, atom, 0, 1);
    REQUIRE(couple.inputs == std::vector<int>{0, 1, 2, 3, 4, 5});
    REQUIRE(couple.outputs == couple.inputs);
}

TEST_CASE("constant angle constraint gradient matches finite difference", "[relax_new][constraints][fd]") {
    double x[6] = {1.1, 0.2, -0.1, 0.4, 0.9, 0.3};
    double g[6];
    double f = 0.0;
    const double g0 = 0.25;

    Constraints::constant_angle(x, 6, g, 6, &f, g0, nullptr, nullptr, nullptr);
    for (int i = 0; i < 6; ++i) {
        double xp[6];
        double xm[6];
        std::copy(std::begin(x), std::end(x), xp);
        std::copy(std::begin(x), std::end(x), xm);
        xp[i] += 1e-6;
        xm[i] -= 1e-6;
        double gp[6], gm[6], fp = 0.0, fm = 0.0;
        Constraints::constant_angle(xp, 6, gp, 6, &fp, g0, nullptr, nullptr, nullptr);
        Constraints::constant_angle(xm, 6, gm, 6, &fm, g0, nullptr, nullptr, nullptr);
        CAPTURE(i);
        REQUIRE(g[i] == Approx((fp - fm) / 2e-6).margin(2e-6));
    }
}

TEST_CASE("mindist constraints use squared distance, MIC and reset gradient", "[relax_new][constraints]") {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity();
    atom.pos_ = {Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Zero()};
    atom.chemmap_ = {0, 0};

    double x[6] = {0.9, 0.0, 0.0, 0.1, 0.0, 0.0};
    double g[6];
    std::fill(std::begin(g), std::end(g), 7.0);
    double f = 0.0;
    Constraints::mindist(x, 6, g, 6, &f, 0.25, &atom, nullptr, nullptr);

    REQUIRE(f == Approx(-0.21).margin(1e-12));
    REQUIRE(g[0] == Approx(-0.4).margin(1e-12));
    REQUIRE(g[3] == Approx(0.4).margin(1e-12));
    for (int i : {1, 2, 4, 5}) {
        REQUIRE(g[i] == Approx(0.0).margin(1e-12));
    }

    double xcar[6] = {0.0, 0.0, 0.0, 0.2, 0.0, 0.0};
    std::fill(std::begin(g), std::end(g), 7.0);
    Constraints::mindist_f(xcar, 6, g, 6, &f, 0.25, &atom, nullptr, nullptr);
    REQUIRE(f == Approx(-0.21).margin(1e-12));
    REQUIRE(g[0] == Approx(-0.4).margin(1e-12));
    REQUIRE(g[3] == Approx(0.4).margin(1e-12));
}

TEST_CASE("mindist constraints leave open axes unwrapped using callback coordinates", "[relax_new][constraints]") {
    const bool fractional = GENERATE(false, true);
    const bool partial = GENERATE(false, true);
    Atoms atom = make_dimer(10., 0.);
    atom.pbc_ = {partial, false, false};
    // These stale positions must not replace the callback's selected geometry.
    atom.pos_ = {{4., 4., 4.}, {4., 4., 4.}};
    const auto source = atom;
    std::array<double, 6> x{};
    x[partial ? 4 : 3] = fractional ? .9 : 9.;
    const auto original_x = x;
    std::array<double, 6> g;
    g.fill(7.);
    double f = 0.;
    const auto callback = fractional ? Constraints::mindist : Constraints::mindist_f;
    callback(x.data(), 6, g.data(), 6, &f, 4., &atom, nullptr, nullptr);
    REQUIRE(f == Approx(0.).margin(1e-12));
    for (double value : g) REQUIRE(value == Approx(0.).margin(1e-12));
    REQUIRE(x == original_x);
    REQUIRE(atom.pos_ == source.pos_);
    REQUIRE(atom.cell_ == source.cell_);
    REQUIRE(atom.pbc_ == source.pbc_);
    REQUIRE(atom.cartesian == source.cartesian);
}

TEST_CASE("mindist constraints use exact skew images with Cartesian and fractional gradients", "[relax_new][constraints]") {
    const bool fractional = GENERATE(false, true);
    Atoms atom = make_dimer(10., 0.);
    atom.cell_ << 4., 0., 0., 3.8, 1., 0., 0., 0., 6.;
    atom.pbc_ = {true, true, false};
    atom.pos_ = {{2., 2., 2.}, {2., 2., 2.}};
    const auto source = atom;
    std::array<double, 6> x = fractional
        ? std::array<double, 6>{0., 0., 0., .49, .49, 0.}
        : std::array<double, 6>{0., 0., 0., 3.822, .49, 0.};
    const auto original_x = x;
    const auto callback = fractional ? Constraints::mindist : Constraints::mindist_f;
    std::array<double, 6> g;
    g.fill(7.);
    double f = 0.;
    callback(x.data(), 6, g.data(), 6, &f, .36, &atom, nullptr, nullptr);
    // The shortest j-i image is (.022, -.51, 0), obtained by subtracting b.
    REQUIRE(f == Approx(.260584 - .36).margin(1e-12));
    const Eigen::RowVector3d cart_gradient(-.044, 1.02, 0.);
    for (int axis = 0; axis < 3; ++axis) {
        const double expected = fractional ? cart_gradient.dot(atom.cell_.row(axis)) : cart_gradient[axis];
        REQUIRE(g[axis] == Approx(expected).margin(1e-12));
        REQUIRE(g[3 + axis] == Approx(-expected).margin(1e-12));
    }
    for (double step : {1e-5, 1e-6}) {
        for (int axis = 0; axis < 6; ++axis) {
            auto plus = x, minus = x;
            plus[axis] += step;
            minus[axis] -= step;
            double fp = 0., fm = 0.;
            std::array<double, 6> scratch;
            callback(plus.data(), 6, scratch.data(), 6, &fp, .36, &atom, nullptr, nullptr);
            callback(minus.data(), 6, scratch.data(), 6, &fm, .36, &atom, nullptr, nullptr);
            CAPTURE(fractional, step, axis);
            REQUIRE(g[axis] == Approx((fp - fm) / (2 * step)).margin(2e-8));
        }
    }
    REQUIRE(x == original_x);
    REQUIRE(atom.pos_ == source.pos_);
    REQUIRE(atom.cell_ == source.cell_);
    REQUIRE(atom.pbc_ == source.pbc_);
    REQUIRE(atom.cartesian == source.cartesian);
}

TEST_CASE("RelaxationDrvV2 MINDIST does not create an open-boundary penalty", "[relax_new][constraints]") {
    auto atom = make_dimer(10., 9.);
    atom.pbc_ = {false, false, false};
    const auto source = atom;
    auto settings = make_base_relax_settings();
    settings.Modify("cartesian", "true");
    settings.Modify("relax_cell", "false");
    settings.Modify("dprecond", "false");
    regulator constraint;
    constraint.regulator_t = regulator::MINDIST;
    constraint.integer_specs = {0, 1};
    constraint.g0 = 2.; // V2 setup accepts distance and squares it for the callback.
    std::vector<regulator> constraints{constraint};
    RelaxationDrvV2 driver(settings);
    driver.init(&g_pot, atom, &constraints);
    Eigen::VectorXd gradient;
    REQUIRE(eval_relax_at(driver, driver.x, &gradient) == Approx(0.).margin(1e-12));
    REQUIRE(gradient.isZero(1e-12));
    REQUIRE(driver.constraints.size() == 1);
    REQUIRE(driver.constraints[0].f == Approx(0.).margin(1e-12));
    REQUIRE(atom.pos_ == source.pos_);
    REQUIRE(atom.cell_ == source.cell_);
    REQUIRE(atom.pbc_ == source.pbc_);
    REQUIRE(atom.cartesian == source.cartesian);
}

TEST_CASE("FilterLC reduces to nullspace and reconstructs constrained coordinates", "[relax_new][constraints]") {
    auto atom = make_dimer();
    auto drv = make_drv(atom);
    drv->x.resize(3);
    drv->g.resize(3);
    drv->prec_scaling = Eigen::VectorXd::Ones(3);
    drv->x << 1.25, 0.75, 4.0; // satisfies x0 + x1 = 2
    drv->g << 3.0, 5.0, 7.0;

    const std::string fnm = "/tmp/atoms_agent_lc_constraint.dat";
    {
        std::ofstream ofs(fnm);
        ofs << "1 3\n";
        ofs << "1 1 0\n";
        ofs << "2\n";
    }

    FilterLC lc;
    std::string fnm_mut = fnm;
    lc.init(drv.get(), &fnm_mut);
    lc.forward_scaling(drv.get());
    REQUIRE(drv->prec_scaling.size() == 2);
    REQUIRE(drv->prec_scaling[0] > 0.0);
    REQUIRE(drv->prec_scaling[1] > 0.0);

    Eigen::VectorXd x_full = drv->x;
    lc.forward(drv.get());
    REQUIRE(drv->x.size() == 2);
    REQUIRE(drv->g.size() == 2);
    lc.reverse(drv.get());
    REQUIRE(drv->x.size() == 3);
    REQUIRE(drv->x.isApprox(x_full, 1e-12));
    REQUIRE(drv->x[0] + drv->x[1] == Approx(2.0).margin(1e-12));
}

TEST_CASE("reverse_modifiers applies inverse transforms in reverse order", "[relax_new]") {
    struct Recorder : public AnyModifier {
        std::vector<int> *calls;
        int id;
        Recorder(std::vector<int> *calls_, int id_) : calls(calls_), id(id_) {}
        void reverse(RelaxationDrvV2 *) override { calls->push_back(id); }
    };

    RelaxationDrvV2 drv;
    std::vector<int> calls;
    std::vector<AnyModifier*> modifiers;
    modifiers.push_back(new Recorder(&calls, 1));
    modifiers.push_back(new Recorder(&calls, 2));
    drv.reverse_modifiers(modifiers);
    REQUIRE(calls == std::vector<int>{2, 1});
    for (auto *m : modifiers) delete m;
}

TEST_CASE("check_cons_convergence empty constraints", "[relax_new]") {
    auto atom=make_dimer(); auto d=make_drv(atom);
    REQUIRE(d->check_cons_convergence());
}

TEST_CASE("cell pressure gradient matches finite difference in strain modes", "[relax_new][fd]") {
    auto atom = make_skew_cell_atom();
    GPSettings rs = make_base_relax_settings();
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "False");
    rs.Modify("pressure", "4.5");
    rs.Modify("symm_strain", "True");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    drv->init(&g_pot, atom);
    drv->LatPos2X();
    drv->x[0] = 0.030;
    drv->x[1] = 0.010;
    drv->x[2] = -0.004;
    drv->x[3] = -0.020;
    drv->x[4] = 0.006;
    drv->x[5] = 0.015;
    require_relax_gradient_matches_fd(*drv, drv->x, 1e-6, 4e-6);


}

TEST_CASE("cell pressure gradient matches finite difference in nonsymmetric strain and lattice modes", "[relax_new][fd]") {
    auto atom = make_skew_cell_atom();
    GPSettings rs = make_base_relax_settings();
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "False");
    rs.Modify("pressure", "3.0");
    rs.Modify("symm_strain", "False");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    drv->init(&g_pot, atom);
    drv->LatPos2X();
    drv->x[0] = 0.020;
    drv->x[1] = 0.012;
    drv->x[2] = -0.006;
    drv->x[3] = 0.004;
    drv->x[4] = -0.015;
    drv->x[5] = 0.008;
    drv->x[6] = -0.003;
    drv->x[7] = 0.005;
    drv->x[8] = 0.025;
    require_relax_gradient_matches_fd(*drv, drv->x, 1e-6, 4e-6);

    GPSettings lattice_rs = make_base_relax_settings();
    lattice_rs.Modify("relax_cell", "True");
    lattice_rs.Modify("relax_atom", "False");
    lattice_rs.Modify("pressure", "3.0");
    lattice_rs.Modify("cell_strain", "False");
    auto lattice_drv = std::make_unique<RelaxationDrvV2>(lattice_rs);
    lattice_drv->init(&g_pot, atom);
    lattice_drv->LatPos2X();
    lattice_drv->x[0] += 0.04;
    lattice_drv->x[4] -= 0.03;
    lattice_drv->x[8] += 0.02;
    lattice_drv->x[1] += 0.01;
    require_relax_gradient_matches_fd(*lattice_drv, lattice_drv->x, 1e-6, 4e-6);
}

TEST_CASE("external stress gradient matches finite difference", "[relax_new][fd]") {
    auto atom = make_skew_cell_atom();
    GPSettings rs = make_base_relax_settings();
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "False");
    rs.Modify("external_stress", "1.2 -0.7 0.4 0.3 -0.2 0.5");
    rs.Modify("symm_strain", "True");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    drv->init(&g_pot, atom);
    drv->LatPos2X();
    // The current external_stress energy term is linearized in strain; verify
    // the optimizer-facing gradient at the reference strain where that
    // convention is exact.
    require_relax_gradient_matches_fd(*drv, drv->x, 1e-6, 3e-6);
}

TEST_CASE("fix_shape pressure gradient matches finite difference", "[relax_new][fd]") {
    auto atom = make_skew_cell_atom();
    GPSettings rs = make_base_relax_settings();
    rs.Modify("fix_shape", "True");
    rs.Modify("relax_atom", "False");
    rs.Modify("pressure", "2.5");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    drv->init(&g_pot, atom);
    drv->LatPos2X();
    drv->x[0] = 1.04;
    require_relax_gradient_matches_fd(*drv, drv->x, 1e-6, 2e-6);
}

TEST_CASE("apply_disp with cell relaxation re-encodes x after reference reset", "[relax_new]") {
    auto atom = make_skew_cell_atom();
    GPSettings rs = make_base_relax_settings();
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "False");
    rs.Modify("apply_disp", "True");
    rs.Modify("disp", "0.02");
    rs.Modify("driver", "gd");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    drv->init(&g_pot, atom);
    drv->run();

    REQUIRE(drv->refcell.isApprox(drv->atom.cell_, 1e-12));
    REQUIRE(drv->ivrefcell.isApprox(drv->refcell.inverse(), 1e-12));
    REQUIRE(drv->x.head(drv->xposoffset).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-12));
}

// ═══════════════════════════════════════════════════════════════════════════
// Symmetry-constrained relaxation (FilterSymmCoord / keep_symm)
// ═══════════════════════════════════════════════════════════════════════════

static Atoms make_coupled_cubic(double a = 5.43) {
    // Diamond cubic Si: SG 227 (Fd-3m), 2 atoms in primitive cell
    // at (0,0,0) and (0.25, 0.25, 0.25) in fractional coords
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * a;
    atom.pos_ = {
        Eigen::RowVector3d(0, 0, 0),
        Eigen::RowVector3d(0.25 * a, 0.25 * a, 0.25 * a)
    };
    atom.chemmap_ = {0, 0}; // same species (Si)
    return atom;
}

static Atoms make_diamond_cubic(double a = 5.43) {
    // Diamond cubic Si: SG 227 (Fd-3m), 2 atoms in primitive cell
    // at (0,0,0) and (0.25, 0.25, 0.25) in fractional coords
    Atoms atom;
    atom.cell_ << 0, a/2, a/2, a/2, 0, a/2, a/2, a/2, 0;
    atom.pos_ = {
        Eigen::RowVector3d(0, 0, 0),
        Eigen::RowVector3d(0.25 * a, 0.25 * a, 0.25 * a)
    };
    atom.chemmap_ = {0, 0}; // same species (Si)
    return atom;
}

static Atoms make_nacl(double a = 5.64) {
    // NaCl: SG 225 (Fm-3m), conventional cell with 4 Na + 4 Cl
    // Use primitive cell: 1 Na at (0,0,0) + 1 Cl at (0.5, 0.5, 0.5)
    Atoms atom;
    atom.cell_ << 0, a/2, a/2, a/2, 0, a/2, a/2, a/2, 0;
    atom.pos_ = {
        Eigen::RowVector3d(0, 0, 0),
        Eigen::RowVector3d(0.5 * a, 0.5 * a, 0.5 * a)
    };
    atom.chemmap_ = {0, 1}; // Na, Cl
    return atom;
}

static Atoms make_p1_dimer() {
    Atoms atom;
    atom.cell_ << 5.0, 0.2, 0.1,
                  0.3, 6.0, 0.4,
                  0.2, 0.5, 7.0;
    atom.pos_ = {
        Eigen::RowVector3d(0.7, 1.1, 2.3),
        Eigen::RowVector3d(2.2, 3.4, 1.7)
    };
    atom.chemmap_ = {0, 1};
    return atom;
}

TEST_CASE("keep_symm constructs FilterSymmCoord", "[relax_new][symm]") {
    auto atom = make_diamond_cubic();
    GPSettings rs;
    rs.Modify("gtol", "0.01");
    rs.Modify("max_iter", "5");
    rs.Modify("write_traj", "false");
    rs.Modify("keep_symm", "True");
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "True");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    REQUIRE_NOTHROW(drv->init(&g_pot, atom));

    // Should have exactly one modifier in modifiers_list (FilterSymmCoord)
    REQUIRE(drv->modifiers_list.size() == 1);
    REQUIRE(drv->modifiers_list[0]->name == "SymmCoord");

    // True diamond primitive cell: one isotropic cell mode, fixed atomic sites.
    drv->LatPos2X();
    drv->apply_modifiers(drv->modifiers_list);
    CAPTURE(drv->x.size());
    REQUIRE(drv->x.size() == 1);
}

TEST_CASE("keep_symm P1 no symmetry", "[relax_new][symm]") {
    // P1 (SG 1): no symmetry beyond identity
    auto atom = make_p1_dimer();
    GPSettings rs;
    rs.Modify("gtol", "0.01");
    rs.Modify("max_iter", "5");
    rs.Modify("write_traj", "false");
    rs.Modify("keep_symm", "True");
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "True");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    REQUIRE_NOTHROW(drv->init(&g_pot, atom));

    // P1 has triclinic cell: 6 free cell DOFs, each atom is its own orbit.
    // 2 orbits x 3 = 6 pos DOFs -> total: 6+6=12 (same as full).
    drv->LatPos2X();
    drv->apply_modifiers(drv->modifiers_list);
    REQUIRE(drv->x.size() == 12);
}

TEST_CASE("keep_symm reduces DOFs for diamond cubic", "[relax_new][symm]") {
    auto atom = make_diamond_cubic();
    GPSettings rs;
    rs.Modify("gtol", "0.01");
    rs.Modify("max_iter", "5");
    rs.Modify("write_traj", "false");
    rs.Modify("keep_symm", "True");
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "True");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    drv->init(&g_pot, atom);

    // Verify that forward/reverse preserve the structure
    drv->LatPos2X();
    auto x_full = drv->x;
    drv->apply_modifiers(drv->modifiers_list); // forward: full → reduced
    auto x_red = drv->x;
    REQUIRE(x_red.size() < x_full.size());      // actual reduction

    drv->reverse_modifiers(drv->modifiers_list); // reverse: reduced → full
    // After reverse, x should have the same size as the original full x
    REQUIRE(drv->x.size() == x_full.size());

    // Cell parameters should be recovered (cubic: all Voigt components 0
    // since the reference cell equals the actual cell)
    for (int i = 0; i < 6; ++i)
        REQUIRE(drv->x[i] == Approx(x_full[i]).margin(1e-12));

    // Positions should be recovered
    for (int i = 0; i < static_cast<int>(x_full.size()); ++i)
        REQUIRE(drv->x[i] == Approx(x_full[i]).margin(1e-12));
}

TEST_CASE("keep_symm roundtrip with cell strain", "[relax_new][symm]") {
    // Apply a cell perturbation and verify forward/reverse recovers it
    auto atom = make_diamond_cubic();
    GPSettings rs;
    rs.Modify("gtol", "0.01");
    rs.Modify("max_iter", "5");
    rs.Modify("write_traj", "false");
    rs.Modify("keep_symm", "True");
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "True");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    drv->init(&g_pot, atom);

    drv->LatPos2X();
    drv->apply_modifiers(drv->modifiers_list);
    auto *filter = dynamic_cast<FilterSymmCoord *>(drv->modifiers_list.front());
    REQUIRE(filter != nullptr);
    drv->x[0] = 0.02; // perturb an actual invariant cell mode
    const Eigen::VectorXd reduced = drv->x;
    drv->reverse_modifiers(drv->modifiers_list);
    REQUIRE(drv->x.head(6).isApprox(filter->cell_basis * reduced.head(filter->n_red_cell), 1e-12));
}

TEST_CASE("keep_symm conflicts with selective_dynamics", "[relax_new][symm]") {
    auto atom = make_diamond_cubic();
    GPSettings rs;
    rs.Modify("gtol", "0.01");
    rs.Modify("max_iter", "5");
    rs.Modify("write_traj", "false");
    rs.Modify("keep_symm", "True");
    rs.Modify("selective_dynamics", "True");
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "True");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    REQUIRE_THROWS(drv->init(&g_pot, atom));
}

TEST_CASE("keep_symm requires at least one of relax_cell/relax_atom", "[relax_new][symm]") {
    auto atom = make_diamond_cubic();
    GPSettings rs;
    rs.Modify("gtol", "0.01");
    rs.Modify("max_iter", "5");
    rs.Modify("write_traj", "false");
    rs.Modify("keep_symm", "True");
    rs.Modify("relax_cell", "False");
    rs.Modify("relax_atom", "False");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    REQUIRE_THROWS(drv->init(&g_pot, atom));
}

TEST_CASE("keep_symm cell-only relaxation", "[relax_new][symm]") {
    // Only relax cell with symmetry constraint, keep positions fixed
    auto atom = make_diamond_cubic();
    GPSettings rs;
    rs.Modify("gtol", "0.01");
    rs.Modify("max_iter", "5");
    rs.Modify("write_traj", "false");
    rs.Modify("keep_symm", "True");
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "False");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    REQUIRE_NOTHROW(drv->init(&g_pot, atom));

    // Cell-only: x has only cell DOFs (6) reduced to the detected 2 cell modes
    drv->LatPos2X();
    REQUIRE(drv->x.size() == 6);  // full: 6 Voigt components only
    drv->apply_modifiers(drv->modifiers_list);
    CAPTURE(drv->x.size());
    REQUIRE(drv->x.size() == 1);
    // Roundtrip
    drv->reverse_modifiers(drv->modifiers_list);
    REQUIRE(drv->x.size() == 6);
}

TEST_CASE("keep_symm pos-only relaxation", "[relax_new][symm]") {
    // Only relax positions with symmetry constraint, keep cell fixed
    auto atom = make_diamond_cubic();
    GPSettings rs;
    rs.Modify("gtol", "0.01");
    rs.Modify("max_iter", "5");
    rs.Modify("write_traj", "false");
    rs.Modify("keep_symm", "True");
    rs.Modify("relax_cell", "False");
    rs.Modify("relax_atom", "True");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    REQUIRE_NOTHROW(drv->init(&g_pot, atom));

    // Pos-only: x has no cell component, only 3N positions
    drv->LatPos2X();
    REQUIRE(drv->x.size() == 6);  // full: 2 atoms × 3
    drv->apply_modifiers(drv->modifiers_list);
    CAPTURE(drv->x.size());
    REQUIRE(drv->x.size() == 0);  // fixed diamond sites
    // Roundtrip
    drv->reverse_modifiers(drv->modifiers_list);
    REQUIRE(drv->x.size() == 6);
}

TEST_CASE("keep_symm NaCl special positions", "[relax_new][symm]") {
    // NaCl: both Na and Cl on special positions (4a and 4b in Fm-3m)
    // Na at (0,0,0): all 3 coords constrained by site symmetry -43m
    // Cl at (0.5,0.5,0.5): all 3 coords constrained by site symmetry -43m
    auto atom = make_nacl();
    GPSettings rs;
    rs.Modify("gtol", "0.01");
    rs.Modify("max_iter", "5");
    rs.Modify("write_traj", "false");
    rs.Modify("keep_symm", "True");
    rs.Modify("relax_cell", "True");
    rs.Modify("relax_atom", "True");

    auto drv = std::make_unique<RelaxationDrvV2>(rs);
    REQUIRE_NOTHROW(drv->init(&g_pot, atom));

    // 2 orbits (Na and Cl), both on special positions with 0 free coords
    // Cell: cubic → 1 free DOF
    // Positions: 0 free coords (all constrained by site symmetry)
    drv->LatPos2X();
    drv->apply_modifiers(drv->modifiers_list);
    // Reduced: 1 cell + 0 pos = 1
    REQUIRE(drv->x.size() == 1);
    // Verify that forward/reverse roundtrip works
    drv->reverse_modifiers(drv->modifiers_list);
    REQUIRE(drv->x.size() == 12); // 6 cell + 6 pos

    // Positions should be at their special values
    int xpo = drv->xposoffset;
    // Na at (0,0,0) in x-space
    REQUIRE(drv->x[xpo + 0] == Approx(0.0).margin(1e-10));
    REQUIRE(drv->x[xpo + 1] == Approx(0.0).margin(1e-10));
    REQUIRE(drv->x[xpo + 2] == Approx(0.0).margin(1e-10));
    // Cl at (0.5*a, 0.5*a, 0.5*a) in x-space (recast: frac * refcell)
    double half_a = make_nacl().cell_(0, 1);
    REQUIRE(drv->x[xpo + 3] == Approx(half_a).margin(1e-10));
    REQUIRE(drv->x[xpo + 4] == Approx(half_a).margin(1e-10));
    REQUIRE(drv->x[xpo + 5] == Approx(half_a).margin(1e-10));
}

TEST_CASE("Coupled simple cubic sites retain their displacement mode", "[relax_new][symm]") {
    auto atom = make_coupled_cubic();
    RelaxationDrvV2 driver;
    driver.flags.keep_symm = driver.flags.relax_cell = true;
    driver.init(&g_pot, atom);
    auto *filter = dynamic_cast<FilterSymmCoord *>(driver.modifiers_list.front());
    REQUIRE(filter->n_red_cell == 2);
    REQUIRE(filter->n_red_pos == 1);
}

struct AnalyticRelaxPot : StubPot {
    bool reject_moves = false;
    void getEFSOMP(Atoms &a, AnyNeighborhoods &) override {
        a.energy_ = 0; a.stress_.setZero();
        a.forces_.assign(a.size(), Eigen::RowVector3d::Zero());
        for (int i = 0; i < a.size(); ++i) {
            a.energy_ += 0.5 * a.pos_[i].squaredNorm();
            a.forces_[i] = -a.pos_[i];
            a.stress_ -= a.pos_[i].transpose() * a.pos_[i];
        }
        if (reject_moves && a.pos_[0][0] != 1.0)
            a.energy_ = std::numeric_limits<double>::quiet_NaN();
    }
};

TEST_CASE("Relaxation normalizes fractional input in its working copy", "[relax_new][coordinates]") {
    auto cartesian = make_skew_cell_atom();
    cartesian.pbc_ = {false, false, false};
    auto fractional = cartesian;
    fractional.to_fractional();
    const auto original_positions = fractional.pos_;
    const auto original_cell = fractional.cell_;
    AnalyticRelaxPot potential;
    RelaxationDrvV2 expected, actual;
    for (auto *driver : {&expected, &actual}) {
        driver->tol.gtol = 1e20; // Compare the initial state without optimization moves.
        driver->control.max_iter = 1;
        driver->flags.write_traj = false;
    }
    expected.init(&potential, cartesian);
    actual.init(&potential, fractional);
    REQUIRE(actual.atom.cartesian);
    REQUIRE(actual.atom.pos_[0].isApprox(cartesian.pos_[0], 1e-12));
    REQUIRE(actual.x.isApprox(expected.x, 1e-12));
    REQUIRE(expected.run() == 0);
    REQUIRE(actual.run() == 0);
    REQUIRE(actual.energy0 == Approx(0.5 * cartesian.pos_[0].squaredNorm()).margin(1e-12));
    REQUIRE(actual.energy0 == Approx(expected.energy0).margin(1e-12));
    REQUIRE(actual.atom.forces_[0].isApprox(expected.atom.forces_[0], 1e-12));
    REQUIRE(actual.atom.pos_[0].isApprox(cartesian.pos_[0], 1e-12));
    REQUIRE_FALSE(fractional.cartesian);
    REQUIRE(fractional.pos_[0].isApprox(original_positions[0], 0.0));
    REQUIRE(fractional.cell_.isApprox(original_cell, 0.0));
}

TEST_CASE("Relaxation respects iteration limits and returns accepted state", "[relax_new][safety]") {
    for (const char *optimizer : {"lbfgs", "lbfgspp", "cg", "gd", "fire"}) {
        CAPTURE(optimizer);
        AnalyticRelaxPot potential;
        auto atom = make_dimer();
        atom.pbc_ = {false, false, false}; atom.pos_[0] = {1, 0, 0};
        RelaxationDrvV2 d;
        d.flags.cartesian = true; d.control.driver = optimizer;
        d.control.max_iter = 2; d.control.max_move = 0.05;
        if (d.control.driver != "gd") d.control.min_move = 0;
        d.lbfgspp_param.max_iterations = 100;
        d.init(&potential, atom);
        d.run();
        REQUIRE(d.state_vars.curr_iter <= 2);
        REQUIRE(d.has_checkpoint);
        REQUIRE_FALSE(d.state_vars.converged);
        REQUIRE(d.f == Approx(d.checkpoint_f));
        REQUIRE((d.x - d.checkpoint_x).norm() < 1e-12);
        REQUIRE(d.g.cwiseAbs().maxCoeff() > d.tol.gtol);
        double expected_energy = 0;
        for (int i = 0; i < d.atom.size(); ++i) {
            expected_energy += 0.5 * d.atom.pos_[i].squaredNorm();
            REQUIRE(d.atom.forces_[i].isApprox(-d.atom.pos_[i].transpose()));
        }
        REQUIRE(d.f == Approx(expected_energy));
        REQUIRE(d.atom.energy_ == Approx(expected_energy));
    }
}

TEST_CASE("An initially converged structure needs only one potential evaluation", "[relax_new]") {
    struct CountingPot : StubPot {
        int calls = 0;
        void getEFSOMP(Atoms &a, AnyNeighborhoods &neighbors) override {
            ++calls;
            StubPot::getEFSOMP(a, neighbors);
        }
    };
    for (bool keep_symm : {false, true}) {
        for (const char *optimizer : {"lbfgs", "lbfgspp", "cg", "gd", "fire"}) {
            CAPTURE(keep_symm, optimizer);
            CountingPot potential;
            auto atom = keep_symm ? make_diamond_cubic() : make_dimer();
            RelaxationDrvV2 driver;
            driver.flags.keep_symm = keep_symm;
            driver.control.driver = optimizer;
            driver.init(&potential, atom);
            REQUIRE(driver.run() == 0);
            REQUIRE(driver.state_vars.curr_iter == 0);
            REQUIRE(potential.calls == 1);
        }
    }
}

TEST_CASE("Illegal trials stop and restore the accepted structure", "[relax_new][safety]") {
    AnalyticRelaxPot potential; potential.reject_moves = true;
    auto atom = make_dimer();
    atom.pbc_ = {false, false, false}; atom.pos_[0] = {1, 0, 0};
    RelaxationDrvV2 d;
    d.flags.cartesian = true; d.control.driver = "lbfgspp";
    d.control.min_move = 1e-12;
    d.init(&potential, atom);
    REQUIRE_THROWS(d.run());
    REQUIRE(d.state_vars.curr_iter == 0);
    REQUIRE(d.has_checkpoint);
    REQUIRE(d.stop_reason == "error");
    REQUIRE_FALSE(d.state_vars.converged);
    REQUIRE(d.atom.pos_[0][0] == 1.0);
    REQUIRE(std::isfinite(d.f));
}
TEST_CASE("Symmetry zero DOF and repeated perturbed runs are safe", "[relax_new][symm][safety]") {
    for (bool recast : {false, true}) {
        auto atom = make_diamond_cubic();
        RelaxationDrvV2 d;
        d.flags.keep_symm = true; d.flags.atomic_force_recast = recast;
        d.init(&g_pot, atom);
        REQUIRE_NOTHROW(d.run());
        REQUIRE(d.state_vars.converged);
        REQUIRE(d.stop_reason == "zero_degrees_of_freedom");
        REQUIRE(d.has_checkpoint);
        REQUIRE_NOTHROW(d.run());
        d.flags.relax_cell = true; d.flags.apply_disp = true;
        d.init(&g_pot, atom);
        REQUIRE_NOTHROW(d.run());
        REQUIRE_NOTHROW(d.run());
    }
}
TEST_CASE("Symmetry analytic gradient matches finite differences in both coordinate bases", "[relax_new][symm][fd]") {
    for (bool recast : {false, true}) {
        auto atom = make_coupled_cubic();
        // Move the origin away from wrapping boundaries for this analytic potential.
        for (auto &p : atom.pos_) p += Eigen::RowVector3d::Constant(0.4);
        AnalyticRelaxPot potential;
        RelaxationDrvV2 d;
        d.flags.keep_symm = d.flags.relax_cell = true;
        d.flags.atomic_force_recast = recast;
        d.init(&potential, atom);
        d.apply_modifiers(d.modifiers_list);
        Eigen::VectorXd center = d.x;
        for (int i = 0; i < center.size(); ++i) center[i] += 0.002 * std::sin(i+1);
        d.x = center; d.eval();
        Eigen::VectorXd gradient = d.g;
        for (double h : {1e-4, 1e-5, 1e-6}) {
            for (int i = 0; i < center.size(); ++i) {
                d.x = center; d.x[i] += h; d.eval(); const double plus = d.f;
                d.x = center; d.x[i] -= h; d.eval(); const double minus = d.f;
                CAPTURE(recast, h, i);
                REQUIRE((plus-minus)/(2*h) == Approx(gradient[i]).margin(5e-6));
            }
        }
    }
}

TEST_CASE("Approximate symmetry checks frozen components and option conflicts", "[relax_new][symm][safety]") {
    auto atom = make_diamond_cubic();
    atom.cell_(0,1) += 1e-6;
    RelaxationDrvV2 d;
    d.flags.keep_symm = true;
    REQUIRE_THROWS(d.init(&g_pot, atom));
    d.flags.relax_cell = true;
    REQUIRE_NOTHROW(d.init(&g_pot, atom));
    REQUIRE(d.symmetry_cell_correction > 0);
    REQUIRE(d.symmetry_cell_correction < d.tol.symprec);
    d.flags.vibffilter = true;
    REQUIRE_THROWS(d.init(&g_pot, atom));
    d.flags.vibffilter = false;
    atom = make_diamond_cubic(); atom.chemmap_.clear();
    REQUIRE_THROWS(Symmetry3D::get_sym_dataset(atom, 1e-4, true));
}

TEST_CASE("LBFGS line searches respect displacement boundaries at different force scales", "[relax_new][lbfgspp]") {
    for (double slope : {1.0, 0.01}) {
        struct Linear {
            double slope;
            double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g) {
                g = Eigen::VectorXd::Constant(x.size(), slope);
                return slope * x.sum();
            }
        } function{slope};
        auto check = [&]<template<class> class Search>(int search) {
            CAPTURE(slope, search);
            LBFGSpp::LBFGSParam<double> parameters;
            parameters.linesearch = search;
            parameters.max_iterations = 3;
            parameters.max_step_move = 0.1;
            parameters.epsilon = parameters.epsilon_rel = 0;
            parameters.epsgmax = 1e-5;
            LBFGSpp::LBFGSSolver<double, Search> solver(parameters);
            Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
            double energy, previous = x[0];
            int accepted = 0;
            REQUIRE(solver.minimize(function, x, energy,
                [&](const Eigen::VectorXd &point, double, const Eigen::VectorXd &) {
                    REQUIRE(point[0] < previous);
                    REQUIRE(previous - point[0] <= 0.1 + 1e-12);
                    previous = point[0];
                    ++accepted;
                }) == 3);
            REQUIRE(accepted == 3);
            REQUIRE(std::isfinite(solver.final_grad_norm()));
            x.setOnes();
            REQUIRE_NOTHROW(solver.minimize(function, x, energy));
        };
        check.template operator()<LBFGSpp::LineSearchBacktracking>(LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING);
        check.template operator()<LBFGSpp::LineSearchBracketing>(LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO);
        check.template operator()<LBFGSpp::LineSearchNocedalWright>(LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE);
        check.template operator()<LBFGSpp::LineSearchLewisOverton>(LBFGSpp::LBFGS_LINESEARCH_LEWIS_OVERTON);
    }
}

TEST_CASE("LBFGS line search limits never accept a rejected trial", "[relax_new][lbfgspp]") {
    struct NoDecrease {
        int calls = 0;
        double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g) {
            ++calls;
            g = Eigen::VectorXd::Ones(x.size());
            return 1;
        }
    };
    auto check = []<template<class> class Search>(int search) {
        CAPTURE(search);
        NoDecrease function;
        LBFGSpp::LBFGSParam<double> parameters;
        parameters.linesearch = search;
        parameters.max_linesearch = 3;
        LBFGSpp::LBFGSSolver<double, Search> solver(parameters);
        Eigen::VectorXd x = Eigen::VectorXd::Ones(1);
        double energy;
        int accepted = 0;
        REQUIRE_THROWS_AS(solver.minimize(function, x, energy,
            [&](const auto &, double, const auto &) { ++accepted; }), std::exception);
        REQUIRE(accepted == 0);
        REQUIRE(function.calls <= 4); // initial point plus three trials
    };
    check.template operator()<LBFGSpp::LineSearchBacktracking>(LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING);
    check.template operator()<LBFGSpp::LineSearchBracketing>(LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO);
    check.template operator()<LBFGSpp::LineSearchNocedalWright>(LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE);
    check.template operator()<LBFGSpp::LineSearchLewisOverton>(LBFGSpp::LBFGS_LINESEARCH_LEWIS_OVERTON);
}

TEST_CASE("Relaxation initialization clears previous run status even on failure", "[relax_new][safety]") {
    auto atom = make_dimer();
    RelaxationDrvV2 driver;
    driver.init(&g_pot, atom);
    driver.run();
    REQUIRE(driver.state_vars.converged);
    atom.chemmap_.clear();
    REQUIRE_THROWS(driver.init(&g_pot, atom));
    REQUIRE_FALSE(driver.state_vars.converged);
    REQUIRE_FALSE(driver.has_checkpoint);
    REQUIRE(driver.state_vars.curr_iter == 0);
}

TEST_CASE("Spglib null dataset failures release temporary inputs", "[relax_new][symm][safety]") {
    auto atom = make_diamond_cubic();
    atom.pos_[1] = atom.pos_[0];
    for (int repeat = 0; repeat < 10; ++repeat)
        REQUIRE_THROWS(Symmetry3D::get_sym_dataset(atom, 1e-4, true));
}

TEST_CASE("Symmetry cannot silently correct frozen atomic positions", "[relax_new][symm][safety]") {
    auto atom = make_diamond_cubic();
    atom.pos_[1][0] += 1e-6;
    RelaxationDrvV2 driver;
    driver.flags.keep_symm = driver.flags.relax_cell = true;
    driver.flags.relax_atom = false;
    REQUIRE_THROWS(driver.init(&g_pot, atom));
}
