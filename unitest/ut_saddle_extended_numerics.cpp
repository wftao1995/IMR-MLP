// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/nroptimize.h"
#include "mindriver/neb.h"
#include "mindriver/cg.h"
#include "mindriver/sgd.h"
#include "mindriver/dimer_new.h"
#include <numbers>

namespace {
class PairQuadratic : public Potentials::AnyPotential {
public:
    PairQuadratic() { cutoff = 6; cutoffsq = 36; species_count = 1; }
    void getEFS(Atoms &a, AnyNeighborhoods &) override {
        a.resetEFS();
        const Eigen::RowVector3d r = a.pos_[1] - a.pos_[0];
        a.energy_ = 0.5 * r.squaredNorm();
        a.forces_[0] = r;
        a.forces_[1] = -r;
        a.stress_ = -r.transpose() * r;
    }
};

Atoms pair_structure() {
    Atoms a;
    a.cell_ << 8, 0.4, 0.2, 0.3, 9, 0.5, 0.1, 0.2, 10;
    a.pos_ = {{2, 2, 2}, {3.7, 2.6, 2.3}};
    a.chemmap_ = {0, 0};
    return a;
}

class Barrier : public Potentials::AnyPotential {
public:
    double y_stiffness = 2;
    double y_linear = 0;
    Barrier() { cutoff = 3; cutoffsq = 9; species_count = 1; }
    void getEFS(Atoms &a, AnyNeighborhoods &) override {
        a.resetEFS();
        const double x = a.pos_[0](0)-10, y = a.pos_[0](1)-10, z = a.pos_[0](2)-10;
        a.energy_ = (x*x-1)*(x*x-1) + 0.5*y_stiffness*y*y + y_linear*y + 1.5*z*z;
        a.forces_[0] << -4*x*(x*x-1), -y_stiffness*y-y_linear, -3*z;
    }
};
}

TEST_CASE("VC NEB optimizer coordinates round trip without moving atoms", "[neb][numerics]") {
    PairQuadratic pot;
    auto a = pair_structure();
    std::vector<Atoms> images(3, a);
    images[1].pos_[1](0) += 0.1;
    images[2].pos_[1](0) += 0.2;
    NEBDriver d;
    d.pot = &pot; d.vcneb = true; d.check_mic = false;
    d.symm_strain = GENERATE(false, true);
    d.restore_F_cholesky = GENERATE(false, true);
    d.recast_atomic_force = GENERATE(false, true);
    d.init(images);
    const auto before = d.images[1];
    Eigen::VectorXd x;
    d.LatPos2X(x);
    d.X2LatPos_interf(x);
    CAPTURE(d.symm_strain, d.restore_F_cholesky, d.recast_atomic_force);
    CHECK(d.images[1].cell_.isApprox(before.cell_, 1e-12));
    for (int i=0; i<a.size(); ++i)
        CHECK(d.images[1].pos_[i].isApprox(before.pos_[i], 1e-12));
}

TEST_CASE("VC NEB coupled forces match enthalpy finite differences", "[neb][numerics]") {
    PairQuadratic pot;
    std::vector<Atoms> images(3, pair_structure());
    NEBDriver d;
    d.pot = &pot; d.vcneb = true; d.check_mic = false; d.pressure = 0.7;
    d.symm_strain = GENERATE(false, true);
    d.restore_F_cholesky = GENERATE(false, true);
    d.recast_atomic_force = GENERATE(false, true);
    d.init(images);
    Eigen::MatrixXd x = d.x;
    x(1,0) += 0.12; x(1,1) += 0.04; x(1,2) -= 0.03;
    d.X2LatPos(x); d.eval();
    const Eigen::RowVectorXd force = d.g.row(1);
    for (int j=0; j<x.cols(); ++j) {
        auto xp=x, xm=x; xp(1,j)+=1e-5; xm(1,j)-=1e-5;
        d.X2LatPos(xp); d.eval(); const double ep=d.images[1].energy_;
        d.X2LatPos(xm); d.eval(); const double em=d.images[1].energy_;
        CAPTURE(j, d.symm_strain, d.restore_F_cholesky, d.recast_atomic_force);
        CHECK(force(j) == Catch::Approx(-(ep-em)/2e-5).margin(2e-8));
    }
}

TEST_CASE("NEB optimizers recover a multi-image double well path", "[neb][numerics]") {
    Barrier pot;
    Atoms a;
    a.cell_ = 20*Eigen::Matrix3d::Identity();
    a.pos_ = {{9,10,10}}; a.chemmap_ = {0};
    std::vector<Atoms> images(7, a);
    for (int i=0; i<7; ++i) {
        const double t = i/6.;
        const double bend = std::sin(std::numbers::pi*t);
        // Unequal spacing and an off-saddle central image exercise both the
        // springs and the reversal of the climbing image's parallel force.
        images[i].pos_[0] << 9+2*t+0.04*bend, 10+0.12*bend, 10+0.07*bend;
    }
    NEBDriver d; d.pot = &pot; d.check_mic = false;
    d.climb = GENERATE(false, true); d.neb_ftol = 1e-6;
    const int tangent_type = GENERATE(1, 2);
    d.tangent_type = tangent_type;
    d.sproj_type = GENERATE(1, 2);
    const std::string optimizer = GENERATE("fire", "sgd", "cg", "lbfgs");
    d.init(images);
    CAPTURE(optimizer, d.climb, d.tangent_type, d.sproj_type);
    REQUIRE(d.run(optimizer, 1500, 0.03) == 0);
    CHECK(d.optimizer_iterations <= 1500);
    CHECK(d.tangent_type == tangent_type);
    CHECK(d.max_neb_force < d.neb_ftol);
    CHECK(d.images.front().pos_[0].isApprox(images.front().pos_[0], 1e-14));
    CHECK(d.images.back().pos_[0].isApprox(images.back().pos_[0], 1e-14));
    CHECK(d.images[3].energy_ - d.images[0].energy_ == Catch::Approx(1).margin(1e-8));
    for (int i=1; i<6; ++i) {
        CHECK(d.images[i].pos_[0](1) == Catch::Approx(10).margin(1e-6));
        CHECK(d.images[i].pos_[0](2) == Catch::Approx(10).margin(1e-6));
        CHECK(d.images[i].pos_[0](0) > d.images[i-1].pos_[0](0));
    }
}

TEST_CASE("Force-only CG descends a positive quadratic", "[optimizer][numerics]") {
    Eigen::Matrix2d h;
    const bool coupled = GENERATE(false, true);
    if (coupled) h << 3, 1, 1, 2;
    else h = 2*Eigen::Matrix2d::Identity();
    auto objective = [&h](const Eigen::VectorXd &x, Eigen::VectorXd &g) {
        g = h*x;
        return 0.5*x.dot(g);
    };
    GPSettings settings;
    SimpleCGF opt(settings); opt.max_iter = 100; opt.gtol = 1e-8;
    Eigen::VectorXd x(2); x << 0.2, -0.3;
    double energy;
    opt.minimize(objective, x, energy);
    CHECK(x.norm() < 1e-8);
    CHECK(energy < 1e-16);
}

TEST_CASE("Original NEB tangent has a nonconservative transverse Jacobian", "[neb][numerics]") {
    Barrier pot;
    Atoms a;
    a.cell_ = 20*Eigen::Matrix3d::Identity();
    a.pos_ = {{9,10,10}}; a.chemmap_ = {0};
    std::vector<Atoms> images(7, a);
    for (int i=0; i<7; ++i) images[i].pos_[0](0) = 9+i/3.;
    NEBDriver d; d.pot = &pot; d.check_mic = false; d.tangent_type = 2;
    d.init(images);
    Eigen::VectorXd x, gp(15), gm(15);
    d.LatPos2X(x);
    Eigen::MatrixXd jacobian(5,5);
    for (int j=0; j<5; ++j) {
        auto plus=x, minus=x;
        plus(3*j+1)+=1e-6; minus(3*j+1)-=1e-6;
        d.X2LatPos_interf(plus); d.LatPos2X(plus, gp);
        d.X2LatPos_interf(minus); d.LatPos2X(minus, gm);
        for (int i=0; i<5; ++i) jacobian(i,j)=(gp(3*i+1)-gm(3*i+1))/2e-6;
    }
    Eigen::EigenSolver<Eigen::MatrixXd> solver(jacobian);
    REQUIRE(solver.info() == Eigen::Success);
    CHECK((jacobian-jacobian.transpose()).norm() > 1);
    CHECK(solver.eigenvalues().imag().cwiseAbs().maxCoeff()
          == Catch::Approx(8*std::sqrt(5.)/9).margin(1e-7));
    for (const auto &eigenvalue : solver.eigenvalues())
        CHECK(eigenvalue.real() == Catch::Approx(2).margin(1e-7));
}

TEST_CASE("NEB force safeguard rolls back rejected paths within its budget", "[neb][numerics]") {
    Barrier pot;
    const bool stable = GENERATE(true,false);
    if (!stable) { pot.y_stiffness = 0; pot.y_linear = 0.2; }
    Atoms a;
    a.cell_ = 20*Eigen::Matrix3d::Identity();
    a.pos_ = {{9,10,10}}; a.chemmap_ = {0};
    std::vector<Atoms> images(3, a);
    images[1].pos_[0] << 10,10.1,10;
    images[2].pos_[0] << 11,10,10;
    NEBDriver d; d.pot = &pot; d.check_mic = false; d.tangent_type = 2;
    d.init(images);
    Eigen::VectorXd x;
    d.LatPos2X(x);
    const Eigen::VectorXd initial = x;
    const int budget = GENERATE(1,9);
    double objective = 0;
    int calls = 0;
    auto uphill_trial = [&](int limit) {
        ++calls;
        x(1) += 0.1; // Deliberately propose a path with twice the residual.
        return limit;
    };
    CHECK(d.minimize_guarded(uphill_trial, budget, 0.005, x, objective) == budget);
    CHECK(calls == 1);
    CHECK(d.images[1].pos_[0].transpose().isApprox(x, 1e-14));
    CHECK_FALSE(d.check_convergence());
    if (budget == 1 || !stable) {
        CHECK(x.isApprox(initial, 1e-14));
        CHECK(d.max_neb_force == Catch::Approx(0.2));
    } else {
        CHECK((x-initial).norm() <= 0.005+1e-14);
        CHECK(d.max_neb_force < 0.2);
    }
    if (!stable && budget > 1)
        CHECK(d.stop_reason == NEBDriver::StopReason::numerical_failure);
    CHECK(d.images[1].energy_ == Catch::Approx(1+0.5*pot.y_stiffness*std::pow(x(1)-10,2)+pot.y_linear*(x(1)-10)));
}

TEST_CASE("Original NEB tangent handles zero force during delayed climbing", "[neb][numerics]") {
    Barrier pot;
    Atoms a;
    a.cell_ = 20*Eigen::Matrix3d::Identity();
    a.pos_ = {{9,10,10}}; a.chemmap_ = {0};
    std::vector<Atoms> images(3, a);
    images[1].pos_[0](0) = 10;
    images[2].pos_[0](0) = 11;
    NEBDriver d; d.pot = &pot; d.check_mic = false; d.tangent_type = 2;
    d.climb = true; d.climb_delay = 10;
    const std::string optimizer = GENERATE("fire", "cg", "lbfgs", "sgd");
    d.init(images);
    REQUIRE(d.run(optimizer, 40, 0.03) == 0);
    CHECK(d.climbing_active);
    CHECK(d.optimizer_iterations <= 40);
    CHECK(d.images[1].pos_[0].isApprox(images[1].pos_[0], 1e-14));
}

TEST_CASE("Original NEB tangent and projected forces match analytic vectors", "[neb][numerics]") {
    Barrier pot;
    Atoms a;
    a.cell_ = 20*Eigen::Matrix3d::Identity();
    a.pos_ = {{9,10,10}}; a.chemmap_ = {0};
    std::vector<Atoms> images(3, a);
    images[1].pos_[0] << 10.1,10.3,10.2;
    images[2].pos_[0] << 11,10,10;
    NEBDriver d; d.pot = &pot; d.check_mic = false;
    d.tangent_type = 2; d.sproj_type = GENERATE(1,2);
    d.climb = GENERATE(false,true);
    d.init(images);
    Eigen::VectorXd x, g(3);
    d.LatPos2X(x); d.LatPos2X(x, g);
    const Eigen::Vector3d backward = images[1].pos_[0]-images[0].pos_[0];
    const Eigen::Vector3d forward = images[2].pos_[0]-images[1].pos_[0];
    const Eigen::Vector3d tangent = (backward.normalized()+forward.normalized()).normalized();
    const Eigen::Vector3d force(0.396,-0.6,-0.6);
    const double spring = d.sproj_type == 1
        ? d.spring*(forward.norm()-backward.norm())
        : d.spring*(forward-backward).dot(tangent);
    const Eigen::Vector3d expected = d.climb
        ? (force-2*force.dot(tangent)*tangent).eval()
        : (force+(spring-force.dot(tangent))*tangent).eval();
    CHECK((g+expected).norm() < 1e-12);
    CHECK(d.max_neb_force == Catch::Approx(expected.norm()).margin(1e-12));
}

TEST_CASE("NR pseudoinverse preserves unstable modes and removes translation modes", "[nr][numerics]") {
    Eigen::MatrixXd h(4,4);
    h << -2, 1, 0, 0, 1, 3, 0, 0, 0, 0, 2, -2, 0, 0, -2, 2;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(h);
    Eigen::MatrixXd inverse;
    NROptimizeAPP::inverse_hessian(h, solver.eigenvalues(), solver.eigenvectors(), inverse, {}, 1e-8);
    CHECK((h*inverse*h).isApprox(h, 1e-12));
    CHECK((inverse*h*inverse).isApprox(inverse, 1e-12));
    Eigen::VectorXd translation(4); translation << 0,0,1,1;
    CHECK((inverse*translation).norm() < 1e-12);
    const Eigen::VectorXd unstable = solver.eigenvectors().col(0);
    CHECK(unstable.dot(inverse*unstable) == Catch::Approx(1/solver.eigenvalues()(0)));
    NROptimizeAPP::inverse_hessian(h, solver.eigenvalues(), solver.eigenvectors(), inverse, {0,1,2,3}, 1e-8);
    CHECK(inverse.isZero());
}

TEST_CASE("Dimer curvature approaches the quartic Hessian quadratically", "[dimer][numerics]") {
    Barrier pot;
    Atoms a;
    a.cell_ = 20*Eigen::Matrix3d::Identity();
    a.pos_ = {{10.2,10.1,10.05}}; a.chemmap_ = {0};
    DimerDriver d; d.pot = &pot; d.summary = ""; d.init(a);
    Eigen::VectorXd mode(3); mode << 0.8,0.6,0;
    const double exact = (12*0.2*0.2-4)*0.8*0.8 + 2*0.6*0.6;
    double previous_error = 0;
    for (const double separation : {0.02,0.01,0.005}) {
        d.ddr = separation;
        Eigen::VectorXd gradient;
        const double curvature = d.operator_rot(mode, gradient);
        const double error = curvature-exact;
        CAPTURE(separation);
        CHECK(error == Catch::Approx(4*separation*separation*std::pow(0.8,4)).margin(1e-11));
        if (previous_error != 0) CHECK(previous_error/error == Catch::Approx(4).margin(1e-6));
        CHECK(std::abs(gradient.dot(mode)) < 1e-12);
        previous_error = error;
    }
}

TEST_CASE("NR periodic triclinic Hessian matches force derivatives and translations", "[nr][numerics]") {
    using namespace NROptimizeAPP;
    Potentials::PotMTP pot("unitest/test-data/potential.mtp");
    auto atom = pair_structure();
    atom.cell_ *= 0.55;
    atom.pos_.push_back({1.4, 3.8, 3.1}); atom.chemmap_.push_back(1);
    const Eigen::Matrix3d ref = atom.cell_;
    NROptions opts; opts.nocell = GENERATE(false, true); opts.offset = opts.nocell ? 0 : 6;
    Eigen::VectorXd x, g, vals; Eigen::MatrixXd h, vecs;
    LatPos2X(atom, ref.inverse(), x, opts);
    if (!opts.nocell) { x(0)=0.04; x(3)=0.025; x(4)=-0.02; x(5)=0.03; }
    X2LatPos(x, ref, atom, opts);
    constexpr double pressure = 0.013;
    compute_hessian(atom, ref, pressure, pot, 1e-5, vals, vecs, h, opts);
    CHECK(h.isApprox(h.transpose(), 1e-12));
    CHECK((h*vecs-vecs*vals.asDiagonal()).norm() < 1e-9*h.norm());
    for (int axis=0; axis<3; ++axis) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(x.size());
        for (int i=0; i<atom.size(); ++i) translation(opts.offset+3*i+axis)=1;
        CHECK((h*translation).norm() < 1e-9*h.norm());
    }
    for (const double delta : {1e-4, 1e-5, 1e-6}) {
        Eigen::MatrixXd fd(x.size(), x.size());
        for (int j=0; j<x.size(); ++j) {
            auto xp=x, xm=x; xp(j)+=delta; xm(j)-=delta;
            auto ap=atom, am=atom;
            X2LatPos(xp, ref, ap, opts); pot.getEFSOMP(ap);
            X2LatPos(xm, ref, am, opts); pot.getEFSOMP(am);
            Eigen::VectorXd gp, gm;
            LatPos2G(ap, pressure, ref, Eigen::Matrix3d::Zero(), gp, opts);
            LatPos2G(am, pressure, ref, Eigen::Matrix3d::Zero(), gm, opts);
            fd.col(j)=(gp-gm)/(2*delta);
        }
        CAPTURE(opts.nocell, delta);
        CHECK((h-fd).norm()/h.norm() < 2e-5);
        CHECK((fd-fd.transpose()).norm()/h.norm() < 2e-5);
    }
}

TEST_CASE("Original tangent corrects a residual that increases along force flow", "[neb][numerics]") {
    Barrier pot;
    Atoms a; a.cell_=20*Eigen::Matrix3d::Identity(); a.pos_={{9,10,10}}; a.chemmap_={0};
    std::vector<Atoms> path(7,a);
    const std::array<double,7> xs{-1,-.95,-.6,.1,.2,.7,1};
    // The symmetric part of this original-tangent transverse Jacobian has
    // a negative eigenvalue, despite the positive physical y curvature (2).
    const std::array<double,5> bend{-.25214688,.92720791,.18920099,.11940662,.1632559};
    for (int i=0; i<7; ++i) {
        path[i].pos_[0](0)=10+xs[i];
        if (i>0 && i<6) path[i].pos_[0](1)+=1e-3*bend[i-1];
    }
    NEBDriver d; d.pot=&pot; d.check_mic=false; d.tangent_type=2; d.spring=0; d.neb_ftol=1e-10;
    d.init(path);
    Eigen::VectorXd x,g; d.LatPos2X(x); g.resize(x.size()); double objective=d(x,g);
    const Eigen::VectorXd initial=x, initial_g=g;
    Eigen::VectorXd trial_g=g;
    d(x-1e-4*g,trial_g);
    REQUIRE(trial_g.squaredNorm() > initial_g.squaredNorm());
    auto rejected_step = [&](int budget) {
        for (int i=0; i<5; ++i) x(3*i+1)+=1e-3*bend[i];
        return budget;
    };
    CHECK(d.minimize_guarded(rejected_step,9,.02,x,objective) == 9);
    d(x,g);
    CHECK(g.squaredNorm() < .01*initial_g.squaredNorm());
    CHECK((x-initial).norm() <= .02);
    CHECK(d.stop_reason != NEBDriver::StopReason::numerical_failure);
}

TEST_CASE("Force-only optimizer counts include the first accepted move", "[optimizer][numerics]") {
    const bool initial_convergence = GENERATE(false, true);
    const bool use_cg = GENERATE(false, true);
    int evaluations=0;
    auto objective = [&](const Eigen::VectorXd &x, Eigen::VectorXd &g) {
        ++evaluations; g=x; return .5*x.squaredNorm();
    };
    auto stop = [&]() { return evaluations >= (initial_convergence ? 1 : 2); };
    Eigen::VectorXd x=Eigen::VectorXd::Ones(1); double energy=0;
    GPSettings settings;
    int steps=0;
    if (use_cg) {
        SimpleCGF opt(settings); opt.converged_callback=stop; opt.max_iter=5; opt.gtol=0;
        steps=opt.minimize(objective,x,energy);
    } else {
        SimpleSGD opt(settings); opt.converged_callback=stop; opt.max_iter=5; opt.gtol=0;
        steps=opt.minimize(objective,x,energy);
    }
    CHECK(steps == (initial_convergence ? 0 : 1));
    CHECK(evaluations == steps+1);
}
