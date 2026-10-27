// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
#include <catch2/catch_all.hpp>
#include "apps/nroptimize.h"
#include "mindriver/dimer_new.h"
#include "tools/constants.h"
#include <filesystem>

namespace {
// One unstable direction, two stable directions; exact saddle at the origin.
class QuadraticSaddle : public Potentials::AnyPotential {
public:
    QuadraticSaddle() { cutoff = 3; cutoffsq = 9; species_count = 1; }
    void getEFS(Atoms &a, AnyNeighborhoods &) override {
        a.resetEFS();
        for (int i = 0; i < a.size(); ++i) {
            Eigen::Vector3d gradient(-a.pos_[i](0), 2*a.pos_[i](1), 3*a.pos_[i](2));
            a.energy_ += a.pos_[i].dot(gradient) / 2;
            a.forces_[i] = -gradient;
            a.stress_ -= a.pos_[i].transpose() * gradient.transpose();
        }
    }
};
Atoms structure() {
    Atoms atom;
    atom.cell_ << 20, 1, 2, 0, 25, 3, 0, 0, 30;
    atom.pos_ = {{1, 1, 1}, {3, 1.3, 1.1}};
    atom.chemmap_ = {0, 0};
    return atom;
}
}

TEST_CASE("Dimer gradients have the scale of the underlying potential", "[dimer][numerics]") {
    QuadraticSaddle pot;
    auto atom = structure();
    atom.pos_ = {{0.2, 0.3, -0.4}}; atom.chemmap_ = {0};
    DimerDriver d;
    d.init(atom); d.pot = &pot;
    d.modedir << 1, 0, 0;
    Eigen::VectorXd g;
    d.operator_trn(d.xx_center, g);
    REQUIRE(d.curvature == Catch::Approx(-1));
    INFO("translation gradient = " << g.transpose());
    CHECK(g.isApprox(Eigen::Vector3d(0.2, 0.6, -1.2), 1e-10));
    Eigen::VectorXd mode(3); mode << 0.8, 0.6, 0;
    d.operator_rot(mode, g);
    for (int i = 0; i < 3; ++i) {
        auto plus = mode, minus = mode;
        plus(i) += 1e-5; minus(i) -= 1e-5;
        Eigen::VectorXd unused;
        const double fd = (d.operator_rot(plus, unused)-d.operator_rot(minus, unused))/2e-5;
        INFO("rotation component " << i);
        CHECK(g(i) == Catch::Approx(fd).margin(1e-8));
    }
}

TEST_CASE("Dimer reaches a known first order saddle", "[dimer][numerics]") {
    QuadraticSaddle pot;
    auto atom = structure(); atom.pos_ = {{0.1, 0.1, 0.1}}; atom.chemmap_ = {0};
    DimerDriver d; d.summary = ""; d.debug_level = -1;
    d.trn_driver = GENERATE("fire", "cg", "lbfgs", "sgd");
    d.rot_driver = d.trn_driver;
    CAPTURE(d.trn_driver);
    d.dimer_gtol = d.trn_gtol = 1e-6;
    d.rot_force_min = d.rot_gtol = 1e-6;
    d.max_iter = 20; d.pot = &pot; d.init(atom);
    d.modedir << 0.8, 0.6, 0;
    REQUIRE(d.run() == 0);
    REQUIRE(d.converged);
    REQUIRE(d.atom_center.pos_[0].norm() < 2e-6);
    REQUIRE(d.curvature == Catch::Approx(-1).margin(1e-6));
    REQUIRE(std::abs(d.modedir(0)) > 0.999999);
    std::filesystem::remove("dimer_modecar_final");
}

TEST_CASE("VC Dimer enthalpy gradient matches coordinate derivatives", "[dimer][numerics]") {
    QuadraticSaddle pot;
    auto atom = structure(); atom.pos_ = {{0.2, 0.3, -0.4}}; atom.chemmap_ = {0};
    DimerDriver d; d.vc_flag = true; d.pressure = 0.1;
    d.symm_strain = GENERATE(false, true);
    d.restore_F_cholesky = GENERATE(false, true);
    d.recast_atomic_force = GENERATE(false, true);
    d.init(atom);
    Eigen::VectorXd x = d.xx_center, g; x(0) += 0.03; x(1) += 0.01;
    d.X2LatPos(x, atom); pot.getEFSOMP(atom);
    double energy; d.LatPos2G(atom, g, energy);
    CHECK(energy == Catch::Approx(atom.energy_ + d.pressure*GPa_to_eVA3*atom.get_volume()));
    for (int i=0; i<x.size(); ++i) {
        auto plus = x, minus = x; plus(i) += 1e-5; minus(i) -= 1e-5;
        double ep, em; Eigen::VectorXd unused;
        d.X2LatPos(plus, atom); pot.getEFSOMP(atom); d.LatPos2G(atom, unused, ep);
        d.X2LatPos(minus, atom); pot.getEFSOMP(atom); d.LatPos2G(atom, unused, em);
        INFO("VC-Dimer component " << i << " symmetric " << d.symm_strain
             << " Cholesky " << d.restore_F_cholesky << " recast " << d.recast_atomic_force);
        CHECK(g(i) == Catch::Approx((ep-em)/2e-5).margin(1e-7));
    }
}

TEST_CASE("NR energy gradient and Hessian match fixed and variable cell coordinates", "[nr][numerics]") {
    using namespace NROptimizeAPP;
    omp_num_threads = 1; global_omp_init();
    Potentials::PotMTP pot("unitest/test-data/potential.mtp");
    auto atom = structure();
    const Eigen::Matrix3d ref = atom.cell_;
    NROptions opts; opts.nocell = GENERATE(false, true); opts.offset = opts.nocell ? 0 : 6;
    double pressure = 0.01;
    Eigen::VectorXd x, g, vals; Eigen::MatrixXd h, vecs;
    LatPos2X(atom, ref.inverse(), x, opts);
    if (!opts.nocell) x(0) += 0.02;
    X2LatPos(x, ref, atom, opts); pot.getEFSOMP(atom);
    g.resize(x.size()); LatPos2G(atom, pressure, ref, Eigen::Matrix3d::Zero(), g, opts);
    compute_hessian(atom, ref, pressure, pot, 1e-5, vals, vecs, h, opts);
    Eigen::MatrixXd symmetric = h.selfadjointView<Eigen::Lower>();
    constexpr double delta = 1e-6;
    for (int i=0; i<x.size(); ++i) {
        auto xp=x, xm=x; xp(i)+=delta; xm(i)-=delta;
        auto ap=atom, am=atom;
        X2LatPos(xp, ref, ap, opts); pot.getEFSOMP(ap);
        X2LatPos(xm, ref, am, opts); pot.getEFSOMP(am);
        double ep=ap.energy_+pressure*ap.get_volume(), em=am.energy_+pressure*am.get_volume();
        INFO("NR gradient component " << i << " fixed cell " << opts.nocell);
        CHECK(g(i) == Catch::Approx((ep-em)/(2*delta)).margin(2e-5));
        Eigen::VectorXd gp=g, gm=g;
        LatPos2G(ap, pressure, ref, Eigen::Matrix3d::Zero(), gp, opts);
        LatPos2G(am, pressure, ref, Eigen::Matrix3d::Zero(), gm, opts);
        INFO("column " << i << " fixed cell " << opts.nocell);
        INFO("Hessian difference " << (symmetric.col(i)-(gp-gm)/(2*delta)).transpose());
        CHECK((symmetric.col(i)-(gp-gm)/(2*delta)).cwiseAbs().maxCoeff() < 2e-3);
    }
}

TEST_CASE("NR inverts the free subspace before applying constraints", "[nr][numerics]") {
    Eigen::MatrixXd h(2,2); h << 2,1,1,2;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(h);
    Eigen::MatrixXd inverse;
    NROptimizeAPP::inverse_hessian(h, solver.eigenvalues(), solver.eigenvectors(), inverse, {0}, 1e-8);
    REQUIRE(inverse(1,1) == Catch::Approx(0.5));
    REQUIRE(inverse.row(0).isZero()); REQUIRE(inverse.col(0).isZero());
}

TEST_CASE("Dimer validates settings and reports the final allowed state", "[dimer][state]") {
    QuadraticSaddle pot;
    auto atom = structure(); atom.pos_ = {{0.05, 0, 0}}; atom.chemmap_ = {0};
    DimerDriver d; d.pot = &pot; d.summary = "";
    SECTION("invalid geometry and output interval") {
        d.ddr = 0; REQUIRE_THROWS(d.init(atom));
        d.ddr = 0.005; d.nblock = 0; REQUIRE_THROWS(d.init(atom));
    }
    SECTION("seed is reproducible") {
        d.random_seed = 42; d.init(atom); const auto first = d.modedir;
        d.init(atom); REQUIRE(d.modedir == first);
        d.random_seed = 43; d.init(atom); REQUIRE_FALSE(d.modedir.isApprox(first));
    }
    SECTION("initial and last permitted translations are checked") {
        d.init(atom); d.modedir << 1,0,0;
        d.max_iter = 0; REQUIRE(d.run() == 1); REQUIRE_FALSE(d.converged);
        d.max_iter = 1; d.trn_max_iter = 200; d.trn_gtol = 1e-6;
        REQUIRE(d.run() == 0); REQUIRE(d.converged);
        REQUIRE(d.iter == 1);
        REQUIRE(d.max_center_gradient < d.dimer_gtol);
        REQUIRE(d.max_trn_force == Catch::Approx(std::abs(d.atom_center.pos_[0](0))).margin(1e-10));
        d.max_iter = 0; REQUIRE(d.run() == 0);
        REQUIRE(d.iter == 0);
        std::filesystem::remove("dimer_modecar_final");
    }
}
