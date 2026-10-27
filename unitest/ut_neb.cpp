// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "mindriver/neb.h"
#include "tools/constants.h"
#include "potentials/potential_lj.h"

using Catch::Approx;

struct NEBStubPot : public Potentials::AnyPotential {
    NEBStubPot() {
        cutoff = 6.0;
        cutoffsq = 36.0;
        species_count = 1;
        cutoffs.resize(1, 1);
        cutoffsqs.resize(1, 1);
        cutoffs(0, 0) = 6.0;
        cutoffsqs(0, 0) = 36.0;
        r_mins.resize(1, 1);
        r_mins.setZero();
    }
    void getEFS(Atoms &a, AnyNeighborhoods &) override { zero(a); }
    void getEFSOMP(Atoms &a, AnyNeighborhoods &) override { zero(a); }
    void getE(Atoms &a, AnyNeighborhoods &) override { a.energy_ = 0.0; }
    void getEFSOMPthread(Atoms &a, AnyNeighborhoods &) override { zero(a); }
private:
    void zero(Atoms &a) {
        a.energy_ = 0.0;
        a.forces_.assign(a.size(), Eigen::RowVector3d::Zero());
        a.stress_.setZero();
    }
};

struct NEBProfilePot : public NEBStubPot {
    void getEFS(Atoms &a, AnyNeighborhoods &) override { profile(a); }
    void getEFSOMP(Atoms &a, AnyNeighborhoods &) override { profile(a); }
    void getE(Atoms &a, AnyNeighborhoods &) override { a.energy_ = a.pos_[0](0); }
    void getEFSOMPthread(Atoms &a) override { profile(a); }
private:
    void profile(Atoms &a) {
        a.energy_ = a.pos_[0](0);
        a.forces_.assign(a.size(), Eigen::Vector3d::Zero());
        a.stress_.setZero();
    }
};

static Atoms make_neb_atom()
{
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity();
    atom.pos_ = {Eigen::RowVector3d(0.1, 0.2, 0.3)};
    atom.chemmap_ = {0};
    return atom;
}

static NEBDriver make_vcneb_driver()
{
    NEBDriver drv;
    drv.vcneb = true;
    drv.symm_strain = true;
    drv.recast_atomic_force = false;
    drv.refcell = Eigen::Matrix3d::Identity();
    drv.ivrefcell = Eigen::Matrix3d::Identity();
    drv.vcoffset = 6;
    drv.npx = 3;
    drv.images = {make_neb_atom(), make_neb_atom(), make_neb_atom()};
    drv.x.resize(3, 9);
    drv.g.resize(3, 9);
    drv.x.setZero();
    drv.g.setZero();
    return drv;
}

TEST_CASE("NEB normalizes supplied fractional images before filling its path", "[neb][coordinates]") {
    auto first = make_neb_atom();
    first.cell_ << 10, 0.3, 0.1, 0.2, 11, 0.4, 0.1, 0.2, 12;
    first.pbc_ = {false, false, false};
    first.pos_[0] << 1, 2, 3;
    auto supplied = first, last = first;
    supplied.pos_[0][0] = 1.2;
    last.pos_[0][0] = 1.6;
    std::vector<Atoms> cartesian{first, supplied, Atoms{}, last};
    auto mixed = cartesian;
    mixed[0].to_fractional();
    mixed[1].to_fractional();
    const auto original = mixed;
    NEBProfilePot potential;
    NEBDriver expected, actual;
    expected.pot = actual.pot = &potential;
    expected.init(cartesian);
    actual.init(mixed);
    expected.eval();
    actual.eval();
    REQUIRE(actual.x.isApprox(expected.x, 1e-12));
    REQUIRE(actual.images.size() == cartesian.size());
    for (size_t i = 0; i < actual.images.size(); ++i) {
        REQUIRE(actual.images[i].cartesian);
        REQUIRE(actual.images[i].pos_[0].isApprox(expected.images[i].pos_[0], 1e-12));
        REQUIRE(actual.images[i].energy_ == Approx(expected.images[i].energy_).margin(1e-12));
        REQUIRE(actual.images[i].energy_ == Approx(1.0 + 0.2 * i).margin(1e-12));
        REQUIRE(mixed[i].cartesian == original[i].cartesian);
        REQUIRE(mixed[i].cell_.isApprox(original[i].cell_, 0.0));
        REQUIRE(mixed[i].size() == original[i].size());
        if (mixed[i].size() != 0)
            REQUIRE(mixed[i].pos_[0].isApprox(original[i].pos_[0], 0.0));
    }
    REQUIRE(mixed[2].size() == 0);
}

TEST_CASE("VCNEB cell forces are derivatives of enthalpy", "[neb][numerics]")
{
    NEBStubPot pot;
    auto d = make_vcneb_driver();
    d.pot = &pot;
    d.pressure = 1.0;
    const bool non_unit_reference = GENERATE(false, true);
    if (non_unit_reference) {
        d.refcell = Eigen::Vector3d(2.0, 3.0, 4.0).asDiagonal();
        d.ivrefcell = d.refcell.inverse();
    }
    d.restore_F_cholesky = GENERATE(false, true);
    Eigen::MatrixXd x = d.x;
    for (int i = 0; i < 3; ++i) {
        x(i,0)=1.2; x(i,3)=1.1; x(i,5)=0.9;
        x(i,1)=0.03; x(i,2)=0.02; x(i,4)=0.01;
    }
    d.X2LatPos(x); d.eval();
    const Eigen::VectorXd force = d.g.row(1);
    REQUIRE(d.images[1].energy_ == Approx(d.pressure*GPa_to_eVA3*d.images[1].get_volume()));
    for (int j=0; j<6; ++j) {
        auto xp=x, xm=x; xp(1,j)+=1e-5; xm(1,j)-=1e-5;
        d.X2LatPos(xp); d.eval(); const double ep=d.images[1].energy_;
        d.X2LatPos(xm); d.eval(); const double em=d.images[1].energy_;
        INFO("cell component " << j << " Cholesky " << d.restore_F_cholesky);
        CHECK(force(j) == Approx(-(ep-em)/2e-5).margin(1e-10));
    }
}

struct DoubleWellPot : public NEBStubPot {
    void profile(Atoms &a) {
        const double x = a.pos_[0](0)-10, y = a.pos_[0](1)-10, z = a.pos_[0](2)-10;
        a.energy_ = (x*x-1)*(x*x-1)+y*y+z*z;
        a.forces_ = {Eigen::Vector3d(-4*x*(x*x-1), -2*y, -2*z)};
        a.stress_.setZero();
    }
    void getEFS(Atoms &a, AnyNeighborhoods &) override { profile(a); }
    void getEFSOMP(Atoms &a, AnyNeighborhoods &) override { profile(a); }
    void getEFSOMPthread(Atoms &a) override { profile(a); }
    void getEFSOMPthread(Atoms &a, AnyNeighborhoods &) override { profile(a); }
};

TEST_CASE("CI NEB recovers a known double well barrier", "[neb][numerics]") {
    DoubleWellPot pot;
    Atoms a = make_neb_atom();
    a.cell_ = 20*Eigen::Matrix3d::Identity();
    a.pos_[0] << 9,10,10;
    Atoms middle=a, final=a;
    middle.pos_[0] << 10,10.05,10.03;
    final.pos_[0] << 11,10,10;
    NEBDriver d; d.solver_policy=GENERATE("legacy","stable-v1"); d.pot=&pot; d.climb=true; d.climb_delay=0;
    d.check_mic=false; d.neb_ftol=1e-6;
    std::vector<Atoms> path{a,middle,final};
    d.init(path);
    REQUIRE(d.run("fire", 300, 0.05) == 0);
    REQUIRE(d.converged);
    REQUIRE(d.climbing_image == 1);
    REQUIRE(d.images[1].energy_ - d.images[0].energy_ == Approx(1).margin(1e-8));
    REQUIRE(d.images[1].pos_[0](1) == Approx(10).margin(1e-6));
    REQUIRE(d.images[1].pos_[0](2) == Approx(10).margin(1e-6));
}

TEST_CASE("NEB X2LatPos uses xin argument and symm Voigt order", "[neb]")
{
    auto drv = make_vcneb_driver();
    Eigen::MatrixXd xin = Eigen::MatrixXd::Zero(3, 9);

    Eigen::Matrix3d ctarget;
    ctarget << 1.20, 0.11, 0.07,
               0.11, 1.35, 0.05,
               0.07, 0.05, 1.50;
    xin(1, 0) = ctarget(0, 0);
    xin(1, 1) = ctarget(0, 1);
    xin(1, 2) = ctarget(0, 2);
    xin(1, 3) = ctarget(1, 1);
    xin(1, 4) = ctarget(1, 2);
    xin(1, 5) = ctarget(2, 2);
    xin(1, 6) = 0.25;
    xin(1, 7) = 0.50;
    xin(1, 8) = 0.75;

    drv.x(1, 0) = 9.0;
    drv.x(1, 6) = 9.0;
    drv.X2LatPos(xin);

    Eigen::Matrix3d cell = drv.images[1].cell_;
    Eigen::Matrix3d cround = drv.get_strain(cell);
    REQUIRE(cround.isApprox(ctarget, 1e-12));

    Eigen::RowVector3d frac = drv.images[1].pos_[0] * drv.images[1].cell_.inverse();
    REQUIRE(frac(0) == Approx(0.25).margin(1e-12));
    REQUIRE(frac(1) == Approx(0.50).margin(1e-12));
    REQUIRE(frac(2) == Approx(0.75).margin(1e-12));
}

TEST_CASE("NEB LatPos2X stores symm strain as xx xy xz yy yz zz", "[neb]")
{
    auto drv = make_vcneb_driver();
    Eigen::Matrix3d ctarget;
    ctarget << 1.10, 0.12, 0.13,
               0.12, 1.30, 0.23,
               0.13, 0.23, 1.60;
    drv.images[1].cell_ = drv.get_cell(ctarget);

    Eigen::VectorXd xflat;
    drv.LatPos2X(xflat);

    REQUIRE(drv.x(1, 0) == Approx(ctarget(0, 0)).margin(1e-12));
    REQUIRE(drv.x(1, 1) == Approx(ctarget(0, 1)).margin(1e-12));
    REQUIRE(drv.x(1, 2) == Approx(ctarget(0, 2)).margin(1e-12));
    REQUIRE(drv.x(1, 3) == Approx(ctarget(1, 1)).margin(1e-12));
    REQUIRE(drv.x(1, 4) == Approx(ctarget(1, 2)).margin(1e-12));
    REQUIRE(drv.x(1, 5) == Approx(ctarget(2, 2)).margin(1e-12));
}

TEST_CASE("NEB rebuild_path inserts VC symm strain image with correct Voigt order", "[neb]")
{
    auto drv = make_vcneb_driver();
    drv.iclimb_ = 1;
    drv.spconsts = {drv.spring, drv.spring};

    Eigen::Matrix3d c1;
    c1 << 1.10, 0.20, 0.30,
          0.20, 1.40, 0.50,
          0.30, 0.50, 1.70;
    Eigen::Matrix3d c2;
    c2 << 1.30, 0.40, 0.60,
          0.40, 1.80, 0.70,
          0.60, 0.70, 2.10;

    drv.x.row(1).setZero();
    drv.x.row(2).setZero();
    drv.x(1, 0) = c1(0, 0);
    drv.x(1, 1) = c1(0, 1);
    drv.x(1, 2) = c1(0, 2);
    drv.x(1, 3) = c1(1, 1);
    drv.x(1, 4) = c1(1, 2);
    drv.x(1, 5) = c1(2, 2);
    drv.x(2, 0) = c2(0, 0);
    drv.x(2, 1) = c2(0, 1);
    drv.x(2, 2) = c2(0, 2);
    drv.x(2, 3) = c2(1, 1);
    drv.x(2, 4) = c2(1, 2);
    drv.x(2, 5) = c2(2, 2);
    drv.x(1, 6) = 0.2;
    drv.x(1, 7) = 0.3;
    drv.x(1, 8) = 0.4;
    drv.x(2, 6) = 0.4;
    drv.x(2, 7) = 0.5;
    drv.x(2, 8) = 0.6;

    drv.rebuild_path();

    REQUIRE(drv.images.size() == 4);
    Eigen::Matrix3d cmid = 0.5 * (c1 + c2);
    Eigen::Matrix3d inserted = drv.get_strain(drv.images[2].cell_);
    REQUIRE(inserted.isApprox(cmid, 1e-12));
    REQUIRE(drv.x(2, 0) == Approx(cmid(0, 0)).margin(1e-12));
    REQUIRE(drv.x(2, 1) == Approx(cmid(0, 1)).margin(1e-12));
    REQUIRE(drv.x(2, 2) == Approx(cmid(0, 2)).margin(1e-12));
    REQUIRE(drv.x(2, 3) == Approx(cmid(1, 1)).margin(1e-12));
    REQUIRE(drv.x(2, 4) == Approx(cmid(1, 2)).margin(1e-12));
    REQUIRE(drv.x(2, 5) == Approx(cmid(2, 2)).margin(1e-12));
}

TEST_CASE("NEB VC insertion decodes recast coordinates with the reference cell", "[neb]")
{
    auto drv = make_vcneb_driver();
    drv.recast_atomic_force = true;
    drv.refcell = 2.0 * Eigen::Matrix3d::Identity();
    drv.ivrefcell = 0.5 * Eigen::Matrix3d::Identity();
    drv.iclimb_ = 1;
    drv.spconsts = {drv.spring, drv.spring};
    for (int i = 0; i < 3; ++i) {
        drv.x.row(i).setZero();
        drv.x(i, 0) = drv.x(i, 3) = drv.x(i, 5) = 1.0;
    }
    drv.x(1, 6) = 0.2;
    drv.x(2, 6) = 0.4;

    drv.rebuild_path();

    REQUIRE(drv.images[2].cell_.isApprox(2.0 * Eigen::Matrix3d::Identity(), 1e-12));
    REQUIRE(drv.images[2].pos_[0](0) == Approx(0.3).margin(1e-12));
}

TEST_CASE("NEB VC scaling keeps cell and atom scaling in separate offsets", "[neb]")
{
    auto drv = make_vcneb_driver();
    drv.apply_scal = true;
    drv.ciistr = "10 20 30 40 50 60";
    drv.fill_path();

    REQUIRE(drv.scalval.size() == 9);
    REQUIRE(drv.scalval[0] == Approx(std::sqrt(10.0)));
    REQUIRE(drv.scalval[1] == Approx(std::sqrt(120.0)));
    REQUIRE(drv.scalval[2] == Approx(std::sqrt(100.0)));
    REQUIRE(drv.scalval[3] == Approx(std::sqrt(20.0)));
    REQUIRE(drv.scalval[4] == Approx(std::sqrt(80.0)));
    REQUIRE(drv.scalval[5] == Approx(std::sqrt(30.0)));
    REQUIRE(drv.scalval[6] == Approx(1.0));
    REQUIRE(drv.scalval[7] == Approx(1.0));
    REQUIRE(drv.scalval[8] == Approx(1.0));
}

TEST_CASE("NEB operator tolerates null trajectory writer on first iteration", "[neb]")
{
    std::vector<Atoms> atoms = {make_neb_atom(), make_neb_atom(), make_neb_atom()};
    atoms[1].pos_[0] = Eigen::RowVector3d(0.2, 0.2, 0.3);
    atoms[2].pos_[0] = Eigen::RowVector3d(0.3, 0.2, 0.3);

    NEBStubPot pot;
    NEBDriver drv;
    drv.pot = &pot;
    drv.iobj = nullptr;
    drv.write_traj = false;
    drv.init(atoms);

    Eigen::VectorXd xflat;
    drv.LatPos2X(xflat);
    Eigen::VectorXd gflat = Eigen::VectorXd::Zero(xflat.size());
    REQUIRE_NOTHROW(drv(xflat, gflat));
}

TEST_CASE("NEB variable image signals rebuild from operator", "[neb]")
{
    std::vector<Atoms> atoms = {make_neb_atom(), make_neb_atom(), make_neb_atom()};
    atoms[1].pos_[0] = Eigen::RowVector3d(0.5, 0.2, 0.3);
    atoms[2].pos_[0] = Eigen::RowVector3d(0.9, 0.2, 0.3);

    NEBStubPot pot;
    NEBDriver drv;
    drv.pot = &pot;
    drv.iobj = nullptr;
    drv.variable_image = true;
    drv.tangent_type = GENERATE(1, 2);
    drv.image_adjust_interval = 1;
    drv.image_insert_thresh = 0.5;
    drv.max_image_count = 5;
    drv.init(atoms);

    Eigen::VectorXd xflat;
    drv.LatPos2X(xflat);
    Eigen::VectorXd gflat = Eigen::VectorXd::Zero(xflat.size());
    REQUIRE_NOTHROW(drv(xflat, gflat));
    REQUIRE(drv.needs_image_adjust);
    drv.rebuild_path();
    REQUIRE(drv.images.size() == 4);
    REQUIRE_FALSE(drv.needs_image_adjust);
    REQUIRE(drv.run("fire", 500, 0.02) == 0);
    REQUIRE(drv.images.size() == 5);
    REQUIRE(drv.optimizer_iterations <= 500);
}

TEST_CASE("NEB convergence includes tangential spring force", "[neb]")
{
    std::vector<Atoms> atoms = {make_neb_atom(), make_neb_atom(), make_neb_atom()};
    atoms[0].cell_ = atoms[1].cell_ = atoms[2].cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atoms[0].pos_[0] = Eigen::RowVector3d(0.0, 0.2, 0.3);
    atoms[1].pos_[0] = Eigen::RowVector3d(0.1, 0.2, 0.3);
    atoms[2].pos_[0] = Eigen::RowVector3d(0.4, 0.2, 0.3);

    NEBStubPot pot;
    NEBDriver drv;
    drv.pot = &pot;
    drv.spring = 5.0;
    drv.init(atoms);
    Eigen::VectorXd xflat;
    drv.LatPos2X(xflat);
    Eigen::VectorXd gflat = Eigen::VectorXd::Zero(xflat.size());
    drv(xflat, gflat);

    REQUIRE(drv.max_perp_force == Approx(0.0).margin(1e-12));
    REQUIRE(drv.max_spring_force == Approx(1.0).margin(1e-12));
    REQUIRE(drv.max_neb_force == Approx(1.0).margin(1e-12));
    REQUIRE_FALSE(drv.check_convergence());
}

TEST_CASE("NEB validates endpoints and image count", "[neb]")
{
    NEBStubPot pot;
    NEBDriver drv;
    drv.pot = &pot;
    std::vector<Atoms> too_short = {make_neb_atom(), make_neb_atom()};
    REQUIRE_THROWS_WITH(drv.init(too_short), Catch::Matchers::ContainsSubstring("at least 3 images"));

    std::vector<Atoms> mismatched = {make_neb_atom(), make_neb_atom(), make_neb_atom()};
    mismatched[1].resize(2);
    REQUIRE_THROWS_WITH(drv.init(mismatched), Catch::Matchers::ContainsSubstring("same atom count"));
}

TEST_CASE("NEB run performs optimization and reports status", "[neb]")
{
    std::vector<Atoms> atoms = {make_neb_atom(), make_neb_atom(), make_neb_atom()};
    atoms[0].cell_ = atoms[1].cell_ = atoms[2].cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atoms[0].pos_[0](0) = 0.0;
    atoms[1].pos_[0](0) = 0.1;
    atoms[2].pos_[0](0) = 0.4;

    NEBStubPot pot;
    NEBDriver drv; drv.solver_policy=GENERATE("legacy","stable-v1");
    drv.pot = &pot;
    drv.spring = 1.0;
    drv.neb_ftol = 1e-4;
    drv.init(atoms);
    const int rc = drv.run("fire", 500, 0.02);

    REQUIRE(rc == 0);
    REQUIRE(drv.converged);
    REQUIRE(drv.stop_reason_string() == "converged");
    REQUIRE(drv.force_evaluations > 0);
    REQUIRE(drv.max_neb_force < drv.neb_ftol);
}

TEST_CASE("NEB climbing image selection does not retain stale indices", "[neb]")
{
    std::vector<Atoms> atoms(5, make_neb_atom());
    for (auto &a : atoms) a.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    const std::array<double, 5> first_profile{0.0, 0.2, 2.0, 0.4, 1.0};
    for (int i = 0; i < 5; ++i) atoms[i].pos_[0](0) = first_profile[i];

    NEBProfilePot pot;
    NEBDriver drv;
    drv.pot = &pot;
    drv.climb = true;
    drv.init(atoms);
    Eigen::VectorXd xflat;
    drv.LatPos2X(xflat);
    Eigen::VectorXd gflat = Eigen::VectorXd::Zero(xflat.size());
    drv(xflat, gflat);
    REQUIRE(drv.iclimbs == std::set<int>{2});

    // Move the maximum from image 2 to image 1. The old index must disappear.
    xflat[0] = 2.0;
    xflat[3] = 0.2;
    drv(xflat, gflat);
    REQUIRE(drv.iclimbs == std::set<int>{1});
    REQUIRE(drv.climbing_image == 1);
}

TEST_CASE("NEB reports requested CI path without an internal barrier", "[neb]")
{
    std::vector<Atoms> atoms = {make_neb_atom(), make_neb_atom(), make_neb_atom()};
    atoms[0].pos_[0](0) = 0.0;
    atoms[1].pos_[0](0) = 0.2;
    atoms[2].pos_[0](0) = 0.4;
    for (auto &a : atoms) a.cell_ = 10.0 * Eigen::Matrix3d::Identity();

    NEBStubPot pot;
    NEBDriver drv; drv.solver_policy=GENERATE("legacy","stable-v1");
    drv.pot = &pot;
    drv.climb = true;
    drv.init(atoms);
    const int rc = drv.run("fire", 5, 0.02);

    REQUIRE(rc == 2);
    REQUIRE_FALSE(drv.converged);
    REQUIRE(drv.stop_reason_string() == "no_internal_barrier");
}

TEST_CASE("NEB variable springs use the higher endpoint as reference", "[neb]")
{
    NEBDriver drv;
    drv.variable_spring = true;
    drv.spring_min = 1.0;
    drv.spring_max = 15.0;
    drv.images = {make_neb_atom(), make_neb_atom(), make_neb_atom(), make_neb_atom()};
    drv.images[0].energy_ = 0.0;
    drv.images[1].energy_ = 0.5;
    drv.images[2].energy_ = 3.0;
    drv.images[3].energy_ = 1.0;
    drv.spconsts.assign(3, drv.spring);

    drv.update_spring_constants();

    REQUIRE(drv.spconsts[0] == Approx(1.0));
    REQUIRE(drv.spconsts[1] == Approx(15.0));
    REQUIRE(drv.spconsts[2] == Approx(15.0));
}

TEST_CASE("VCNEB joint distance equals scaled cell and recast atom displacement", "[neb][numerics]")
{
    auto drv = make_vcneb_driver();
    drv.recast_atomic_force = true;
    drv.cell_weight = 7.0;
    drv.apply_scal = GENERATE(false, true);
    drv.ciistr = "10 20 30 40 50 60";
    drv.fill_path();
    Eigen::Matrix3d deformation;
    deformation << 1.1, 0, 0, .07, .9, 0, .04, .03, 1.2;
    auto a = drv.images[0], b = drv.images[1];
    b.set_cell(deformation);
    b.pos_[0] = Eigen::RowVector3d(.2, .3, .4) * deformation;
    const auto ca = drv.get_strain(a.cell_), cb = drv.get_strain(b.cell_);
    Eigen::VectorXd delta(9);
    delta << cb(0,0)-ca(0,0), cb(0,1)-ca(0,1), cb(0,2)-ca(0,2),
             cb(1,1)-ca(1,1), cb(1,2)-ca(1,2), cb(2,2)-ca(2,2), .1, .1, .1;
    const double expected = delta.cwiseProduct(drv.scalval).norm();
    REQUIRE(drv.distance(a, b, 0) == Approx(expected).margin(1e-12));
    // A single shear must carry both equal off-diagonal tensor entries.
    REQUIRE(drv.scalval[1] == Approx(std::sqrt(2*(drv.apply_scal ? 60.0 : 7.0))));
}

TEST_CASE("NEB MIC respects nonperiodic axes and disabled wrapping", "[neb][numerics]") {
    const bool periodic = GENERATE(false, true);
    const bool mic = GENERATE(false, true);
    Atoms a = make_neb_atom(), b = a;
    a.cell_ = b.cell_ = 10 * Eigen::Matrix3d::Identity();
    a.pbc_ = b.pbc_ = {periodic, false, false};
    a.pos_[0] << 1, 1, 1;
    b.pos_[0] << 9, 9, 1;
    NEBStubPot pot;
    NEBDriver d; d.pot = &pot; d.check_mic = mic;
    std::vector<Atoms> path{a, Atoms(), b};
    d.init(path);
    CHECK(d.images[1].pos_[0](0) == Approx(periodic && mic ? 0 : 5).margin(1e-12));
    CHECK(d.images[1].pos_[0](1) == Approx(5));
    CHECK(d.images[1].pbc_ == a.pbc_);
    CHECK(d.distance(d.images.front(), d.images.back(), 0) ==
          Approx(std::sqrt((periodic && mic ? 4. : 64.) + 64.)));
}

TEST_CASE("NEB skew paths find images beyond the nearest fractional window", "[neb][numerics][mic]") {
    const int mode = GENERATE(0, 1, 2); // full PBC, only A periodic, MIC disabled
    CAPTURE(mode);
    Atoms first = make_neb_atom(), last = first;
    first.cell_ << 1, 0, 0, 10, 1, 0, 0, 0, 10;
    last.cell_ = first.cell_;
    first.pbc_ = last.pbc_ = {true, mode != 1, mode != 1};
    first.pos_[0].setZero();
    last.pos_[0] << 5.39, .49, 0; // fractional (.49,.49,0)
    if (mode == 1) last.to_fractional();
    std::vector<Atoms> path{first, Atoms{}, last};
    const auto source = path;
    NEBProfilePot potential;
    NEBDriver driver;
    driver.pot = &potential;
    driver.check_mic = mode != 2;
    driver.init(path);
    // Subtracting five A vectors gives (.39,.49,0), whereas the old +/-1
    // fractional search chose (-3.61,-.51,0). These minima are not tied.
    const Eigen::RowVector3d expected(mode == 2 ? 5.39 : .39, .49, 0);
    REQUIRE(driver.images.back().pos_[0].isApprox(expected, 1e-12));
    REQUIRE(driver.images[1].pos_[0].isApprox(.5 * expected, 1e-12));
    REQUIRE(driver.distance(driver.images.front(), driver.images.back(), 0) ==
            Approx(std::sqrt(mode == 2 ? 29.2922 : .3922)).margin(1e-12));
    // distance() also selects the image directly from the original long path.
    auto original_first = first, original_last = source.back();
    if (!original_last.cartesian) original_last.to_cartesian();
    REQUIRE(driver.distance(original_first, original_last, 0) ==
            Approx(expected.norm()).margin(1e-12));
    for (int i : {0, 2}) {
        REQUIRE(path[i].pos_ == source[i].pos_);
        REQUIRE(path[i].cell_ == source[i].cell_);
        REQUIRE(path[i].pbc_ == source[i].pbc_);
        REQUIRE(path[i].cartesian == source[i].cartesian);
    }
    REQUIRE(path[1].size() == 0);
}

TEST_CASE("VCNEB MIC retains the reference metric and coordinate distance", "[neb][numerics][mic]") {
    const bool recast = GENERATE(false, true);
    CAPTURE(recast);
    auto first = make_neb_atom(), last = first;
    first.pos_[0].setZero();
    last.pos_[0] << .49, .49, 0;
    // Current cells are both I. The chosen reference cell, not I, defines
    // image selection, while recasting controls the final distance units.
    NEBDriver driver;
    driver.vcneb = true;
    driver.symm_strain = false;
    driver.recast_atomic_force = recast;
    driver.refcell << 1, 0, 0, 10, 1, 0, 0, 0, 10;
    driver.ivrefcell = driver.refcell.inverse();
    const auto original_first = first, original_last = last;
    // The reference-metric minimum has fractional coordinates(-4.51,.49,0).
    REQUIRE(driver.distance(first, last, 0) ==
            Approx(std::sqrt(recast ? .3922 : 20.5802)).margin(1e-12));
    REQUIRE(first.pos_ == original_first.pos_);
    REQUIRE(last.pos_ == original_last.pos_);
    REQUIRE(first.cell_ == original_first.cell_);
    REQUIRE(last.cell_ == original_last.cell_);
    REQUIRE(first.cartesian == original_first.cartesian);
    REQUIRE(last.cartesian == original_last.cartesian);
}

TEST_CASE("IDPP preserves fixed atomic coordinates", "[neb][numerics]") {
    const bool vc = GENERATE(false, true);
    Atoms a = make_neb_atom();
    a.cell_ = 10 * Eigen::Matrix3d::Identity();
    a.pos_ = {Eigen::RowVector3d(2,2,2), Eigen::RowVector3d(3,2,2)};
    a.chemmap_ = {0,0};
    a.sel_dynpos_ = {Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Ones()};
    Atoms b = a; b.pos_[1] << 2,3,2;
    NEBStubPot pot;
    NEBDriver d; d.pot = &pot; d.sel_dyn = true; d.vcneb = vc;
    d.recast_atomic_force = GENERATE(false, true);
    d.check_mic = false; d.init_method = "idpp"; d.idpp_fmax = 1e-6;
    d.idpp_max_steps = 500;
    std::vector<Atoms> path{a, Atoms(), b};
    d.init(path);
    CHECK(d.images[1].pos_[0].isApprox(a.pos_[0], 1e-14));
    CHECK((d.images[1].pos_[1]-d.images[1].pos_[0]).norm() == Approx(1).margin(2e-5));
    CHECK(d.images.front().pos_[1].isApprox(a.pos_[1], 1e-14));
    CHECK(d.images.back().pos_[1].isApprox(b.pos_[1], 1e-14));
}

TEST_CASE("IDPP rejects a coincident pair instead of reporting zero gradient convergence", "[neb][numerics]") {
    Atoms a = make_neb_atom();
    a.cell_ = 10 * Eigen::Matrix3d::Identity();
    a.pos_ = {Eigen::RowVector3d(2,2,2), Eigen::RowVector3d(4,2,2)};
    a.chemmap_ = {0,0};
    Atoms b = a; std::swap(b.pos_[0], b.pos_[1]);
    NEBStubPot pot;
    NEBDriver d; d.pot = &pot; d.init_method = "idpp"; d.check_mic = false;
    std::vector<Atoms> path{a, Atoms(), b};
    CHECK_THROWS_WITH(d.init(path), Catch::Matchers::ContainsSubstring("coincident atoms"));
}

TEST_CASE("NEB selects multiple climbing and valley images and honors pinning", "[neb]") {
    const bool pin = GENERATE(false, true);
    const std::string mode = GENERATE("highest", "all_local");
    std::vector<Atoms> path(5, make_neb_atom());
    const std::array<double, 5> profile{0, 3, 1, 2, 0};
    for (int i=0; i<5; ++i) {
        path[i].cell_ = 20 * Eigen::Matrix3d::Identity();
        path[i].pos_[0] << profile[i], i, 0;
    }
    NEBProfilePot pot;
    NEBDriver d; d.pot = &pot; d.climb = true; d.vally = true;
    d.climb_mode = mode; d.pin = pin; d.check_mic = false;
    d.init(path);
    Eigen::VectorXd x, g; d.LatPos2X(x); g.resize(x.size()); d(x,g);
    CHECK(d.iclimbs == (mode == "highest" ? std::set<int>{1} : std::set<int>{1,3}));
    CHECK(d.ivallys == std::set<int>{2});
    // Valley refinement returns the full physical force, with no spring.
    CHECK(g.segment<3>(3).norm() == Approx(0));
    x(0)=1; x(3)=4; x(6)=2; d(x,g);
    if (pin) {
        CHECK(d.iclimbs == (mode == "highest" ? std::set<int>{1} : std::set<int>{1,3}));
        CHECK(d.ivallys == std::set<int>{2});
    } else {
        CHECK(d.iclimbs == std::set<int>{2});
        CHECK(d.ivallys.empty());
    }
}

TEST_CASE("Variable springs reset after a barrier disappears", "[neb]") {
    NEBDriver d; d.variable_spring = true; d.spring_min = 0.5; d.spring_max = 5;
    d.images.assign(3, make_neb_atom()); d.spconsts.assign(2, 5);
    for (auto &a : d.images) a.energy_ = 0;
    d.update_spring_constants();
    CHECK(d.spconsts == std::vector<double>{0.5,0.5});
}

TEST_CASE("NEB removes a crowded image while retaining endpoints and saddle", "[neb]") {
    NEBStubPot pot;
    NEBDriver d; d.pot = &pot; d.variable_image = true; d.min_image_count = 3;
    d.image_insert_thresh = 10; d.image_remove_thresh = .2; d.check_mic = false;
    std::vector<Atoms> path(6, make_neb_atom());
    const std::array<double,6> positions{0,.001,.5,1,1.5,2};
    for (int i=0; i<6; ++i) {
        path[i].cell_ = 20 * Eigen::Matrix3d::Identity();
        path[i].pos_[0](0) = positions[i];
        path[i].energy_ = 0; // Explicit synthetic profile before any interior EFS.
    }
    d.init(path);
    d.images[3].energy_ = 10;
    d.check_image_spacing();
    REQUIRE(d.needs_image_adjust);
    d.rebuild_path();
    REQUIRE(d.images.size() == 5);
    CHECK(d.images.front().pos_[0](0) == Approx(0));
    CHECK(d.images.back().pos_[0](0) == Approx(2));
    CHECK(d.images[2].pos_[0](0) == Approx(1));
    CHECK(d.spconsts.size() == 4);
    CHECK(d.run("fire", 500, .05) == 0);
}

TEST_CASE("EB spring forces match the full elastic-band expression", "[neb][numerics]") {
    NEBStubPot pot;
    NEBDriver d; d.pot=&pot; d.check_mic=false; d.tangent_type=3; d.sproj_type=3; d.spring=2;
    std::vector<Atoms> path(3, make_neb_atom());
    for (int i=0; i<3; ++i) {
        path[i].cell_=20*Eigen::Matrix3d::Identity();
        path[i].pos_[0] << i, (i==1 ? .3 : 0), 0;
    }
    d.init(path);
    Eigen::VectorXd x,g; d.LatPos2X(x); g.resize(x.size()); d(x,g);
    const double length=std::sqrt(1.09);
    CHECK(g(0) == Approx(0).margin(1e-14));
    CHECK(g(1) == Approx(4*.3*(length-1)/length));
    CHECK(d.run("fire",1000,.05) == 0);
}

struct CurvedBarrierPot : public NEBStubPot {
    void profile(Atoms &a) {
        const double x=a.pos_[0](0)-10, y=a.pos_[0](1)-10, z=a.pos_[0](2)-10;
        const double bend = y - .8*x*x;
        a.energy_ = (x*x-1)*(x*x-1) + 10*bend*bend + z*z;
        a.forces_ = {Eigen::RowVector3d(-4*x*(x*x-1)+32*x*bend, -20*bend, -2*z)};
        a.stress_.setZero();
    }
    void getEFS(Atoms &a, AnyNeighborhoods &) override { profile(a); }
    void getEFSOMPthread(Atoms &a) override { profile(a); }
};

TEST_CASE("Original tangent converges on a stiff curved minimum energy path", "[neb][numerics]") {
    const std::string optimizer=GENERATE("fire", "lbfgs", "cg", "sgd");
    CurvedBarrierPot pot;
    std::vector<Atoms> path(7, make_neb_atom());
    for (int i=0; i<7; ++i) {
        const double x = -1 + i/3.;
        path[i].cell_ = 20*Eigen::Matrix3d::Identity();
        path[i].pos_[0] << 10+x, 10+.8*x*x, 10;
    }
    NEBDriver d; d.solver_policy=GENERATE("legacy","stable-v1"); d.pot=&pot; d.check_mic=false; d.climb=true;
    d.tangent_type=2; d.spring=1; d.neb_ftol=1e-5;
    d.init(path);
    CAPTURE(optimizer);
    REQUIRE(d.run(optimizer,2000,.08) == 0);
    CHECK(d.max_neb_force < 1e-5);
    REQUIRE(d.climbing_image >= 0);
    CHECK(d.images[d.climbing_image].energy_ == Approx(1).margin(1e-8));
    CHECK(d.images[d.climbing_image].pos_[0].isApprox(Eigen::RowVector3d(10,10,10),1e-6));
}

TEST_CASE("NEB minimum-distance penalty uses consistent endpoint energies", "[neb][numerics]") {
    NEBStubPot pot;
    Atoms a; a.resize(2); a.cell_=20*Eigen::Matrix3d::Identity(); a.chemmap_={0,0};
    a.pos_={Eigen::RowVector3d(5,5,5), Eigen::RowVector3d(6,5,5)};
    a.resetEFS();
    std::vector<Atoms> path(3,a);
    for (int i=0; i<3; ++i) for (auto &p:path[i].pos_) p(1)+=i;
    NEBDriver d; d.pot=&pot; d.check_mic=false; d.mindist=true;
    d.pot_mindist.cutoff=2; d.pot_mindist.factor=3; d.pot_mindist.power=2;
    d.init(path); d.eval();
    for (const auto &image:d.images) CHECK(image.energy_ == Approx(3));
    CHECK(d.images[1].forces_[0](0) == Approx(-6));
    CHECK(d.images[1].forces_[1](0) == Approx(6));
}
