// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_hessian.h"
#include "unitest/many_body_mtp_fixture.h"
#include "potentials/potential_mtp_m.h"
#include <filesystem>
#include <fstream>
#include "tools/hessian_utils.h"
#include "potentials/potential_lj.h"
#include "base/neighborhood.h"
#include <cmath>
#include <algorithm>

using Catch::Approx;
using namespace Potentials;

class QuadraticPotential : public AnyPotential {
public:
    double stiffness = 3.25;

    QuadraticPotential() { cutoff = 8.0; }

    void getEFS(Atoms &atom, AnyNeighborhoods &) override {
        atom.energy_ = 0.0;
        atom.forces_.assign(atom.size(), Eigen::Vector3d::Zero());
        for (int i = 0; i < atom.size(); ++i) {
            atom.energy_ += 0.5 * stiffness * atom.pos_[i].squaredNorm();
            atom.forces_[i] = -stiffness * atom.pos_[i].transpose();
        }
    }

    bool supports_analytical_hessian() const override { return true; }

    void getEFSHessianSymmOMP(Atoms &atom, HessianMatrix &hessian,
                              double) override {
        hessian.resize(atom.size());
        for (int i = 0; i < atom.size(); ++i)
            hessian(i, i) = stiffness * Eigen::Matrix3d::Identity();
    }
};

// ── Test helpers ───────────────────────────────────────────────────────
static LJ612 make_pot(double eps = 1.0, double sig = 2.5, double cutoff = 6.0) {
    GPSettings s;
    s.Modify("ntypes", "1");
    s.Modify("0-0", fmt::format("{} {}", eps, sig));
    s.Modify("cut:0-0", std::to_string(cutoff));
    return LJ612(s);
}

static Atoms make_dimer(double a = 12.0, double d = 2.0) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * a;
    atom.pos_ = {{0, 0, 0}, {d, 0, 0}};
    atom.chemmap_ = {0, 0};
    return atom;
}

static Atoms make_trimer_chain(double a = 15.0, double d = 2.0) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * a;
    atom.pos_ = {{0, 0, 0}, {d, 0, 0}, {2*d, 0, 0}};
    atom.chemmap_ = {0, 0, 0};
    return atom;
}

static Atoms make_triangle(double a = 15.0) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * a;
    atom.pos_ = {{0.0, 0.0, 0.0}, {2.3, 0.0, 0.0}, {0.7, 1.8, 0.0}};
    atom.chemmap_ = {0, 0, 0};
    return atom;
}

static ComputeHessian make_hess(double nudge = 0.005, double cutoff = 1e-6) {
    GPSettings s;
    s.Modify("nudge_pos", std::to_string(nudge));
    s.Modify("nudge_cell", std::to_string(nudge));
    s.Modify("cutoff", std::to_string(cutoff));
    s.Modify("req_cell", "false");
    s.Modify("req_fc", "true");
    return ComputeHessian(s);
}

// ═══════════════════════════════════════════════════════════════════════
// Energy-based force constants
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Hessian: energy-based FC is symmetric", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_dimer(15.0, 2.3);
    auto hess = make_hess();
    hess.set_method("energy_fd");

    Eigen::MatrixXd H;
    hess.compute_fc(atom, H, &pot);
    REQUIRE(H.rows() == 6);
    REQUIRE(H.cols() == 6);

    for (int i = 0; i < H.rows(); ++i)
        for (int j = i + 1; j < H.cols(); ++j)
            REQUIRE(H(i, j) == Approx(H(j, i)).epsilon(1e-8));
}

TEST_CASE("Hessian: energy-based FC near equilibrium is positive definite", "[hessian]") {
    double sig = 2.5;
    double req = std::pow(2.0, 1.0 / 6.0) * sig;
    auto pot = make_pot(1.0, sig, 8.0);
    auto atom = make_dimer(15.0, req);
    auto hess = make_hess();
    hess.set_method("energy_fd");

    Eigen::MatrixXd H;
    hess.compute_fc(atom, H, &pot);
    for (int i = 0; i < 6; ++i)
        REQUIRE(H(i, i) > 0);
}

TEST_CASE("Hessian: energy-based FC all elements finite", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_dimer(15.0, 2.3);
    auto hess = make_hess();
    hess.set_method("energy_fd");

    Eigen::MatrixXd H;
    hess.compute_fc(atom, H, &pot);
    for (int i = 0; i < H.rows(); ++i)
        for (int j = 0; j < H.cols(); ++j)
            REQUIRE(std::isfinite(H(i, j)));
}

// ═══════════════════════════════════════════════════════════════════════
// Trimer chain — multi-atom FC
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Hessian: trimer chain FC symmetry (energy)", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_trimer_chain(15.0, 2.5);
    auto hess = make_hess();
    hess.set_method("energy_fd");

    Eigen::MatrixXd H;
    hess.compute_fc(atom, H, &pot);
    REQUIRE(H.rows() == 9);

    for (int i = 0; i < 9; ++i)
        for (int j = i + 1; j < 9; ++j)
            REQUIRE(H(i, j) == Approx(H(j, i)).epsilon(1e-8));
}

// ═══════════════════════════════════════════════════════════════════════
// Full Hessian compute() with cell terms
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Hessian: full compute with FC only", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_dimer(15.0, 2.3);
    auto hess = make_hess();
    hess.set_method("energy_fd");
    hess.request_fc = true;
    hess.request_cell = false;
    hess.request_response_pos_cell = false;

    Eigen::MatrixXd H;
    hess.compute(atom, H, &pot);
    REQUIRE(H.rows() == 6);
    REQUIRE(H.cols() == 6);
    REQUIRE(std::abs(H(0, 0)) > 1e-10);
}

// cell Hessian and cell_strain tests omitted: cell perturbation in LJ dimer
// is too slow for unit testing. Cell Hessian is tested via integration tests.

// Reduced FC tests omitted: spglib symmetry detection on LJ dimer is slow.
// Reduced FC with symmetry is tested via integration tests.

// ═══════════════════════════════════════════════════════════════════════
// Edge cases
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Hessian: empty atoms returns empty FC", "[hessian]") {
    auto pot = make_pot();
    Atoms atom;
    auto hess = make_hess();
    hess.set_method("energy_fd");

    Eigen::MatrixXd H;
    hess.compute_fc(atom, H, &pot);
    REQUIRE(H.rows() == 0);
    REQUIRE(H.cols() == 0);
}

TEST_CASE("Hessian: nudgepos affects numerical result", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_dimer(15.0, 2.3);

    GPSettings s1, s2;
    s1.Modify("nudge_pos", "0.001");
    s1.Modify("cutoff", "1e-6");
    s1.Modify("req_cell", "false");
    s1.Modify("req_fc", "true");
    s2.Modify("nudge_pos", "0.01");
    s2.Modify("cutoff", "1e-6");
    s2.Modify("req_cell", "false");
    s2.Modify("req_fc", "true");

    auto hess1 = ComputeHessian(s1);
    auto hess2 = ComputeHessian(s2);
    hess1.set_method("energy_fd");
    hess2.set_method("energy_fd");

    Eigen::MatrixXd H1, H2;
    hess1.compute_fc(atom, H1, &pot);
    hess2.compute_fc(atom, H2, &pot);

    // Different nudgepos gives slightly different FC due to truncation error
    REQUIRE(H1(0, 0) != Approx(H2(0, 0)).epsilon(1e-4));
}

// ═══════════════════════════════════════════════════════════════════════
// FC consistency with finite-difference forces
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Hessian: FC matches force finite-difference", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_dimer(15.0, 2.3);
    auto hess = make_hess(0.001); // small nudge for accuracy
    hess.set_method("energy_fd");

    Eigen::MatrixXd H;
    hess.compute_fc(atom, H, &pot);

    // Compute numerical Hessian from forces: H_ij = -(f_i(+d_j) - f_i(-d_j)) / (2d)
    double dx = 0.001;
    double inv2d = 1.0 / (2.0 * dx);
    auto atnew = atom;

    // Rebuild nbhs at each step for correct force
    // Diagonal check via force FD on atom 0, x-direction
    atnew.pos_[0][0] += dx;
    { Neighborhoods nbh(atnew, 8.0); pot.getEFS(atnew, nbh); }
    double fp = atnew.forces_[0][0];
    atnew.pos_[0][0] -= 2 * dx;
    { Neighborhoods nbh(atnew, 8.0); pot.getEFS(atnew, nbh); }
    double fm = atnew.forces_[0][0];
    atnew.pos_[0][0] = atom.pos_[0][0];
    double H00_fd = -(fp - fm) * inv2d;

    INFO("FC(0,0) from energy: " << H(0, 0) << " from force FD: " << H00_fd);
    REQUIRE(H(0, 0) == Approx(H00_fd).epsilon(0.05));
}

// ═══════════════════════════════════════════════════════════════════════
// Regression coverage for the generalized implementation
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Hessian: explicit force FD is complete and agrees with energy FD", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_triangle();

    auto force_hess = make_hess(0.001);
    force_hess.set_method("force_fd");
    auto energy_hess = make_hess(0.001);
    energy_hess.set_method("energy_fd");

    Eigen::MatrixXd hf, he;
    force_hess.compute_fc(atom, hf, &pot);
    energy_hess.compute_fc(atom, he, &pot);
    REQUIRE(hf.rows() == 9);
    REQUIRE(hf.isApprox(hf.transpose(), 1e-12));
    REQUIRE(hf.isApprox(he, 5e-3));
    // A non-collinear geometry exercises same-atom cross-coordinate terms,
    // which the old neighbor-pair shortcut never filled.
    REQUIRE(std::abs(hf(0, 1)) > 1e-8);
}

TEST_CASE("Hessian: auto selects an analytical atomic Hessian when available", "[hessian]") {
    QuadraticPotential pot;
    auto atom = make_dimer(15.0, 2.3);
    auto hess = make_hess();
    hess.use_symmetry = false;

    Eigen::MatrixXd H;
    hess.compute_fc(atom, H, &pot);
    REQUIRE(H.rows() == 6);
    REQUIRE(H.isApprox(3.25 * Eigen::MatrixXd::Identity(6, 6), 1e-12));

    hess.set_method("energy_fd");
    Eigen::MatrixXd Hfd;
    hess.compute_fc(atom, Hfd, &pot);
    REQUIRE(Hfd.isApprox(H, 1e-8));
}

TEST_CASE("Hessian: block API shares analytical and finite-difference backends", "[hessian]") {
    QuadraticPotential pot;
    auto atom = make_dimer(15.0, 2.3);
    auto hess = make_hess(0.001);
    for (const std::string method : {"analytic", "auto", "force_fd",
                                     "energy_fd", "hybrid"}) {
        DYNAMIC_SECTION(method) {
            hess.set_method(method);
            Eigen::MatrixXd dense;
            Potentials::HessianMatrix blocks;
            hess.compute_fc(atom, dense, &pot);
            hess.compute_fc_blocks(atom, blocks, &pot);
            REQUIRE(HessianUtils::to_dense(blocks).isApprox(dense, 1e-12));
        }
    }
}

TEST_CASE("Hessian: compute layout honors independent requests", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_dimer(15.0, 2.3);
    auto hess = make_hess(0.002);
    hess.set_method("energy_fd");

    Eigen::MatrixXd hfc;
    hess.request_fc = true;
    hess.request_cell = false;
    hess.request_response_pos_cell = true;
    hess.compute(atom, hfc, &pot);
    REQUIRE(hfc.rows() == 6);
    REQUIRE(hfc.cols() == 6);

    hess.request_fc = false;
    hess.request_cell = true;
    hess.request_response_pos_cell = true;
    Eigen::MatrixXd hcell;
    hess.compute(atom, hcell, &pot);
    REQUIRE(hcell.rows() == 9);
    REQUIRE(hcell.cols() == 9);
    REQUIRE(hcell.isApprox(hcell.transpose(), 1e-10));

    hess.request_fc = true;
    hess.request_response_pos_cell = false;
    Eigen::MatrixXd hblock;
    hess.compute(atom, hblock, &pot);
    REQUIRE(hblock.rows() == 15);
    REQUIRE(hblock.block(0, 9, 9, 6).isZero(1e-12));
    REQUIRE(hblock.block(9, 0, 6, 9).isZero(1e-12));
}

TEST_CASE("Hessian: strain and fractional coordinates use the requested dimensions", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_dimer(15.0, 2.3);
    auto hess = make_hess(0.002);
    hess.set_method("energy_fd");
    hess.request_fc = false;
    hess.request_cell = true;
    hess.cell_strain = true;
    hess.symm_strain = false;

    Eigen::MatrixXd hs;
    hess.compute(atom, hs, &pot);
    REQUIRE(hs.rows() == 6);
    REQUIRE(hs.cols() == 6);
    REQUIRE(hs.isApprox(hs.transpose(), 1e-10));

    hess.request_fc = true;
    hess.symm_strain = true;
    hess.request_response_pos_cell = true;
    Eigen::MatrixXd hstrain_fc;
    hess.compute(atom, hstrain_fc, &pot);
    REQUIRE(hstrain_fc.rows() == 12);
    REQUIRE(hstrain_fc.allFinite());
    REQUIRE(hstrain_fc.isApprox(hstrain_fc.transpose(), 1e-10));

    const auto original_pos = atom.pos_;
    hess.request_cell = false;
    hess.request_fc = true;
    hess.fractional_pos = true;
    Eigen::MatrixXd hf;
    hess.compute_fc(atom, hf, &pot);
    REQUIRE(hf.rows() == 6);
    REQUIRE(hf.allFinite());
    REQUIRE(atom.pos_ == original_pos);
}

TEST_CASE("Hessian: enthalpy objective contributes the pV curvature", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_dimer(15.0, 2.3);

    auto energy = make_hess(0.002);
    energy.set_method("energy_fd");
    energy.request_fc = false;
    energy.request_cell = true;

    auto enthalpy = energy;
    enthalpy.set_objective("enthalpy");
    enthalpy.pressure = 0.01;

    Eigen::MatrixXd he, hh;
    energy.compute(atom, he, &pot);
    enthalpy.compute(atom, hh, &pot);
    REQUIRE(hh.rows() == 9);
    REQUIRE(hh(0, 4) != Approx(he(0, 4)).margin(1e-8));
}

TEST_CASE("Hessian: hybrid keeps force diagonal and energy off-diagonal", "[hessian]") {
    auto pot = make_pot(1.0, 2.5, 8.0);
    auto atom = make_triangle();
    auto force = make_hess(0.001);
    auto energy = make_hess(0.001);
    auto hybrid = make_hess(0.001);
    force.set_method("force_fd");
    energy.set_method("energy_fd");
    hybrid.set_method("hybrid");

    Eigen::MatrixXd hf, he, hh;
    force.compute_fc(atom, hf, &pot);
    energy.compute_fc(atom, he, &pot);
    hybrid.compute_fc(atom, hh, &pot);
    for (int i = 0; i < hh.rows(); ++i) {
        REQUIRE(hh(i, i) == Approx(hf(i, i)).epsilon(1e-8));
        for (int j = i + 1; j < hh.cols(); ++j)
            REQUIRE(hh(i, j) == Approx(he(i, j)).epsilon(1e-8));
    }
}

TEST_CASE("Hessian: LJ auto and analytic reach exact pair curvature", "[hessian][lj]") {
    auto pot = make_pot(1, 1, 3);
    auto atom = make_dimer(20, 2);
    atom.pbc_ = {false, false, false};
    const auto source = atom;
    Eigen::Matrix3d pair = Eigen::Matrix3d::Zero();
    pair.diagonal() << -633.0 / 1024, 93.0 / 1024, 93.0 / 1024;
    Eigen::MatrixXd expected(6, 6);
    expected << pair, -pair, -pair, pair;
    // A deliberately large FD displacement makes accidental Auto->FD routing visible.
    auto hess = make_hess(0.1, 0);
    for (auto method : {ComputeHessian::HessianMethod::Auto,
                        ComputeHessian::HessianMethod::Analytic}) {
        hess.method = method;
        Eigen::MatrixXd result;
        hess.compute_fc(atom, result, &pot);
        REQUIRE(result.isApprox(expected, 1e-13));
        HessianMatrix blocks;
        hess.compute_fc_blocks(atom, blocks, &pot);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                REQUIRE(blocks(i, j).isApprox(expected.block<3, 3>(3 * i, 3 * j), 1e-13));
    }
    for (double step : {2e-5, 1e-5}) {
        auto force = make_hess(step, 0);
        force.method = ComputeHessian::HessianMethod::ForceFD;
        Eigen::MatrixXd result;
        force.compute_fc(atom, result, &pot);
        REQUIRE((result - expected).cwiseAbs().maxCoeff() < 2e-8);
    }
    REQUIRE(atom.cell_.isApprox(source.cell_, 0));
    REQUIRE(atom.pbc_ == source.pbc_);
    REQUIRE(atom.cartesian == source.cartesian);
    REQUIRE(atom.has_energy() == source.has_energy());
    REQUIRE(atom.has_forces() == source.has_forces());
    REQUIRE(atom.has_stresses() == source.has_stresses());
    for (int i = 0; i < atom.size(); ++i)
        REQUIRE(atom.pos_[i].isApprox(source.pos_[i], 0));
}

TEST_CASE("Directional HVP fallback uses two fresh force evaluations and preserves inputs", "[hessian][hvp][query]") {
    struct AdditiveQuadratic : AnyPotential {
        int calls = 0;
        double maximum_displacement = 0.;
        Atoms center;
        void getEFS(Atoms &atom, AnyNeighborhoods &) override {
            ++calls;
            REQUIRE(atom.energy_ == 0.);
            REQUIRE(atom.stress_.isZero());
            for (int i = 0; i < atom.size(); ++i) {
                REQUIRE(atom.forces_[i].isZero());
                maximum_displacement = std::max(maximum_displacement,
                    (atom.pos_[i] - center.pos_[i]).norm());
                atom.forces_[i] += -3.25 * atom.pos_[i];
            }
        }
    } pot;
    auto atom = make_dimer(20., 1.3);
    atom.pbc_ = {false, false, false};
    pot.center = atom;
    pot.add_efs = true;
    atom.resetEFS();
    atom.energy_ = 19.;
    atom.stress_.setConstant(23.);
    atom.forces_.assign(atom.size(), Eigen::RowVector3d::Constant(29.));
    const auto source = atom;
    const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(6, -.7, .9);
    GPSettings settings;
    settings.Modify("nudge_pos", "0.00001");
    ComputeHessian compute(settings);
    Eigen::VectorXd product;
    compute.compute_hvp(atom, direction, product, &pot);
    REQUIRE(product.isApprox(3.25 * direction, 1e-9));
    REQUIRE(pot.calls == 2);
    REQUIRE(pot.maximum_displacement == Approx(1e-5).margin(1e-12));
    REQUIRE(atom.pos_ == source.pos_);
    REQUIRE(atom.energy_ == source.energy_);
    REQUIRE(atom.forces_ == source.forces_);
    REQUIRE(atom.stress_ == source.stress_);
    compute.compute_hvp(atom, Eigen::VectorXd::Zero(6), product, &pot);
    REQUIRE(product.isZero());
    REQUIRE(pot.calls == 2);
    compute.method = ComputeHessian::HessianMethod::Analytic;
    REQUIRE_THROWS_WITH(compute.compute_hvp(atom, direction, product, &pot),
                        Catch::Matchers::ContainsSubstring("unavailable"));
    REQUIRE(pot.calls == 2);
}

TEST_CASE("LJ directional HVP queries match independent pair curvature", "[hessian][hvp][query][lj]") {
    auto pot = make_pot(1., 1., 3.);
    auto atom = make_dimer(20., 1.3);
    atom.pbc_ = {false, false, false};
    const double r = 1.3;
    const double transverse = 24. * (std::pow(r, -8) - 2. * std::pow(r, -14));
    const double radial = 24. * (26. * std::pow(r, -14) - 7. * std::pow(r, -8));
    const Eigen::Matrix3d k = Eigen::Vector3d(radial, transverse, transverse).asDiagonal();
    Eigen::MatrixXd exact(6, 6);
    exact << k, -k, -k, k;
    const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(6, -.7, .9);
    GPSettings settings;
    settings.Modify("nudge_pos", "0.00001");
    ComputeHessian compute(settings);
    for (auto method : {ComputeHessian::HessianMethod::Auto,
                        ComputeHessian::HessianMethod::Analytic,
                        ComputeHessian::HessianMethod::ForceFD}) {
        compute.method = method;
        Eigen::VectorXd product;
        compute.compute_hvp(atom, direction, product, &pot);
        REQUIRE((product - exact * direction).norm() < 2e-6);
        auto alias = direction;
        compute.compute_hvp(atom, alias, alias, &pot);
        REQUIRE((alias - product).norm() < 1e-12);
    }
}

namespace {
struct CellHessianThreadScope {
    int previous = omp_num_threads;
    CellHessianThreadScope() { omp_num_threads = 1; global_omp_init(); }
    ~CellHessianThreadScope() { omp_num_threads = previous; global_omp_init(); }
};

// Independent energy oracle: perturb the actual requested cell and atomic
// coordinates, without using the analytical backend's coordinate transform.
Eigen::MatrixXd mtp_cell_energy_fd(AnyPotential &pot, const Atoms &reference,
                                 double step, bool strain, bool fractional,
                                 double pressure = 0.0, double cutoff = 0.0) {
    const int cell_dof = strain ? 6 : 9;
    const int n = cell_dof + 3 * reference.size();
    auto q = Eigen::VectorXd::Zero(n).eval();
    const Eigen::Matrix3d invcell = reference.cell_.inverse();
    auto energy = [&](const Eigen::VectorXd &displacement) {
        Atoms work = reference;
        if (strain) {
            Eigen::Matrix3d deformation = Eigen::Matrix3d::Identity();
            deformation.diagonal() += displacement.head<3>();
            // Engineering shear: gamma_yz, gamma_xz, gamma_xy.
            deformation(1, 2) = deformation(2, 1) = displacement(3) / 2;
            deformation(0, 2) = deformation(2, 0) = displacement(4) / 2;
            deformation(0, 1) = deformation(1, 0) = displacement(5) / 2;
            work.cell_ = reference.cell_ * deformation;
        } else {
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    work.cell_(a, b) += displacement(3 * a + b);
        }
        for (int i = 0; i < reference.size(); ++i) {
            const Eigen::RowVector3d delta =
                displacement.segment<3>(cell_dof + 3 * i).transpose();
            if (fractional)
                work.pos_[i] = (reference.pos_[i] * invcell + delta) * work.cell_;
            else
                work.pos_[i] = reference.pos_[i] + delta;
        }
        if (cutoff > 1e-5) {
            Neighborhoods nbhs(work, cutoff);
            pot.getEFS(work, nbhs);
        } else {
            pot.getEFS(work);
        }
        return work.energy_ + pressure * work.get_volume();
    };
    const double center = energy(q);
    Eigen::MatrixXd result(n, n);
    for (int i = 0; i < n; ++i) {
        q(i) = step; const double plus = energy(q);
        q(i) = -step; const double minus = energy(q);
        q(i) = 0;
        result(i, i) = (plus - 2 * center + minus) / (step * step);
        for (int j = 0; j < i; ++j) {
            q(i) = step; q(j) = step; const double pp = energy(q);
            q(j) = -step; const double pm = energy(q);
            q(i) = -step; const double mm = energy(q);
            q(j) = step; const double mp = energy(q);
            q(i) = q(j) = 0;
            result(i, j) = result(j, i) = (pp - pm + mm - mp) / (4 * step * step);
        }
    }
    return result;
}
}

TEST_CASE("Hessian: native MTP cell derivatives honor fixed Cartesian coordinates",
          "[hessian][mtp][cell-coordinates]") {
    CellHessianThreadScope threads;
    PotMTP pot;
    load_many_body_mtp_fixture(pot, 2);
    pot.omp_init();
    const bool images = GENERATE(false, true);
    const bool strain = GENERATE(false, true);
    const double step = GENERATE(.001, .0005);
    CAPTURE(images, strain, step);
    Atoms atom;
    if (images) atom.cell_ << 5.2, .6, .2, .1, 5.6, .3, .2, .1, 5.8;
    else atom.cell_ << 12, 1.2, .4, .1, 13, .7, .2, .3, 14;
    atom.pbc_ = {true, true, true};
    atom.pos_ = {{.4, .6, .5}, {1.9, .9, .7}, {2.3, 3.4, 1.1}};
    atom.chemmap_ = {0, 1, 0};
    pot.getEFS(atom);
    const Atoms source = atom;
    const int cell_dof = strain ? 6 : 9;
    const auto oracle = mtp_cell_energy_fd(pot, atom, step, strain, false);
    if (images) {
        REQUIRE(oracle.topLeftCorner(cell_dof, cell_dof).norm() > .01);
        REQUIRE(oracle.block(0, cell_dof, cell_dof, 9).norm() > .01);
    }
    GPSettings settings;
    settings.Modify("nudge_cell", std::to_string(step));
    settings.Modify("nudge_pos", std::to_string(step));
    settings.Modify("nudge_frac", std::to_string(step));
    ComputeHessian calculator(settings);
    calculator.request_cell = true;
    calculator.cell_strain = strain;
    calculator.engineering_shear = true;
    calculator.use_symmetry = false;
    for (auto method : {ComputeHessian::HessianMethod::Auto,
                        ComputeHessian::HessianMethod::Analytic}) {
        calculator.method = method;
        calculator.request_fc = true;
        calculator.request_response_pos_cell = true;
        Eigen::MatrixXd result;
        calculator.compute(atom, result, &pot);
        REQUIRE((result.topLeftCorner(cell_dof, cell_dof)
                 - oracle.topLeftCorner(cell_dof, cell_dof)).norm() < 2e-6);
        REQUIRE((result.block(0, cell_dof, cell_dof, 9)
                 - oracle.block(0, cell_dof, cell_dof, 9)).norm() < 2e-6);
        REQUIRE((result - oracle).norm() < 3e-6);
        calculator.request_response_pos_cell = false;
        calculator.compute(atom, result, &pot);
        REQUIRE(result.block(0, cell_dof, cell_dof, 9).isZero(1e-14));
        REQUIRE((result.topLeftCorner(cell_dof, cell_dof)
                 - oracle.topLeftCorner(cell_dof, cell_dof)).norm() < 2e-6);
        calculator.request_fc = false;
        calculator.compute(atom, result, &pot);
        REQUIRE(result.rows() == cell_dof);
        REQUIRE((result - oracle.topLeftCorner(cell_dof, cell_dof)).norm() < 2e-6);
    }
    calculator.request_fc = true;
    calculator.request_response_pos_cell = true;
    calculator.fractional_pos = true;
    calculator.method = ComputeHessian::HessianMethod::Auto;
    const auto fractional_oracle = mtp_cell_energy_fd(pot, atom, step, strain, true);
    Eigen::MatrixXd fractional_result;
    calculator.compute(atom, fractional_result, &pot);
    REQUIRE((fractional_result - fractional_oracle).norm() < 3e-6);
    calculator.method = ComputeHessian::HessianMethod::Analytic;
    REQUIRE_THROWS_WITH(calculator.compute(atom, fractional_result, &pot),
        Catch::Matchers::ContainsSubstring("unavailable"));
    calculator.fractional_pos = false;
    calculator.objective = ComputeHessian::HessianObjective::Enthalpy;
    calculator.pressure = .002;
    REQUIRE_THROWS_WITH(calculator.compute(atom, fractional_result, &pot),
        Catch::Matchers::ContainsSubstring("unavailable"));
    calculator.method = ComputeHessian::HessianMethod::Auto;
    calculator.compute(atom, fractional_result, &pot);
    const auto enthalpy_oracle = mtp_cell_energy_fd(pot, atom, step, strain, false, .002);
    REQUIRE((fractional_result - enthalpy_oracle).norm() < 3e-6);
    REQUIRE(atom.pos_ == source.pos_);
    REQUIRE(atom.cell_ == source.cell_);
    REQUIRE(atom.pbc_ == source.pbc_);
    REQUIRE(atom.chemmap_ == source.chemmap_);
    REQUIRE(atom.cartesian == source.cartesian);
    REQUIRE(atom.energy_ == source.energy_);
    REQUIRE(atom.forces_ == source.forces_);
    REQUIRE(atom.stress_ == source.stress_);
    REQUIRE(atom.site_energies_ == source.site_energies_);
    REQUIRE(atom.have_energy == source.have_energy);
    REQUIRE(atom.have_stress == source.have_stress);
    REQUIRE_FALSE(pot.hc_correction);
    REQUIRE_FALSE(pot.rR_correction);
}

TEST_CASE("Hessian: analytical methods respect an explicit neighborhood cutoff",
          "[hessian][mtp][cutoff]") {
    CellHessianThreadScope threads;
    PotMTP pot;
    load_many_body_mtp_fixture(pot, 2);
    pot.omp_init();
    const double cutoff = GENERATE(0.0, 1e-6, 1.0, 2.0, 4.8);
    const double step = GENERATE(.001, .0005);
    const bool strain = GENERATE(false, true);
    CAPTURE(cutoff, step, strain);
    Atoms atom;
    atom.cell_ << 5.2, .6, .2, .1, 5.6, .3, .2, .1, 5.8;
    atom.pbc_ = {true, true, true};
    atom.pos_ = {{.4, .6, .5}, {1.9, .9, .7}, {2.3, 3.4, 1.1}};
    atom.chemmap_ = {0, 1, 0};
    const int cell_dof = strain ? 6 : 9;
    const Eigen::MatrixXd oracle = mtp_cell_energy_fd(
        pot, atom, step, strain, false, 0.0, cutoff);
    const Eigen::MatrixXd atomic = oracle.bottomRightCorner(9, 9);
    if (cutoff == 1.0) REQUIRE(oracle.norm() < 1e-8);
    else REQUIRE(atomic.norm() > .01);
    for (auto method : {ComputeHessian::HessianMethod::Auto,
                        ComputeHessian::HessianMethod::Analytic}) {
        ComputeHessian calculator;
        calculator.set_cutoff(cutoff);
        calculator.use_symmetry = false;
        calculator.method = method;
        calculator.request_cell = true;
        calculator.cell_strain = strain;
        calculator.engineering_shear = true;
        Eigen::MatrixXd result;
        calculator.compute(atom, result, &pot);
        REQUIRE((result - oracle).norm() < 3e-6);
        calculator.request_fc = false;
        calculator.compute(atom, result, &pot);
        REQUIRE((result - oracle.topLeftCorner(cell_dof, cell_dof)).norm() < 3e-6);
        calculator.compute_fc(atom, result, &pot);
        REQUIRE((result - atomic).norm() < 3e-6);
        HessianMatrix blocks;
        calculator.compute_fc_blocks(atom, blocks, &pot);
        REQUIRE((HessianUtils::to_dense(blocks) - atomic).norm() < 3e-6);
        const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(9, -.4, .7);
        Eigen::VectorXd product;
        calculator.compute_hvp(atom, direction, product, &pot);
        REQUIRE((product - atomic * direction).norm() < 3e-6);
    }
}

TEST_CASE("Hessian: serial analytical providers accept a cutoff override",
          "[hessian][lj][cutoff]") {
    auto pot = make_pot(1, 1, 3);
    auto atom = make_dimer(20, 2);
    for (auto method : {ComputeHessian::HessianMethod::Auto,
                        ComputeHessian::HessianMethod::Analytic}) {
        ComputeHessian calculator;
        calculator.set_cutoff(1.0);
        calculator.method = method;
        Eigen::MatrixXd dense;
        calculator.compute_fc(atom, dense, &pot);
        REQUIRE(dense.isZero(1e-14));
        HessianMatrix blocks;
        calculator.compute_fc_blocks(atom, blocks, &pot);
        REQUIRE(HessianUtils::to_dense(blocks).isZero(1e-14));
    }
}

namespace {
struct MCellFixtureFiles {
    std::filesystem::path directory;
    MCellFixtureFiles() {
        char pattern[] = "/tmp/imrmlp_m_cell_XXXXXX";
        const char *path = mkdtemp(pattern);
        if (!path) throw std::runtime_error("Cannot create M cell fixture directory");
        directory = path;
    }
    ~MCellFixtureFiles() { std::filesystem::remove_all(directory); }
    template<class Model>
    std::unique_ptr<AnyPotential> make(bool linear) {
        PotMTP original;
        load_many_body_mtp_fixture(original, 2);
        std::ostringstream serialized;
        original.Save(serialized);
        std::istringstream input(serialized.str());
        const auto path = directory / "model.mtp";
        std::ofstream out(path);
        std::string line;
        while (std::getline(input, line)) {
            if (line.starts_with("species_coeffs")) break;
            out << line << '\n';
        }
        out << "species_coeffs = {0,0,0,0" << (linear ? "" : ",0,0") << "}\n"
            << "moment_coeffs = {-0.15,0.004,0.005"
            << (linear ? "" : ",0.075,-0.002,-0.0025") << "}\n"
            << "model_coeffs = {" << (linear ? "" : "1.0,0.3,0.2,-0.1,0.15") << "}\n";
        out.close();
        GPSettings settings;
        settings.Modify("potfile", path.string());
        auto result = std::make_unique<PotMTP_M<Model>>(settings);
        result->model.coeffs = result->model_coeffs;
        result->omp_init();
        return result;
    }
};
}

TEST_CASE("Hessian: MTP-M OMP APIs retain the nonlinear model and additive EFS",
          "[hessian][mtp_m][model-dispatch]") {
    CellHessianThreadScope threads;
    MCellFixtureFiles files;
    const bool linear = GENERATE(false, true);
    const double cutoff = GENERATE(0.0, 2.0, 3.5);
    const double step = GENERATE(.001, .0005);
    CAPTURE(linear, cutoff, step);
    std::unique_ptr<AnyPotential> owner;
    if (linear) owner = files.make<m_model_mtpli>(true);
    else owner = files.make<m_model_mtpq2>(false);
    AnyPotential &api = *owner;
    Atoms atom;
    atom.cell_ << 5.2, .6, .2, .1, 5.6, .3, .2, .1, 5.8;
    atom.pbc_ = {true, true, true};
    atom.pos_ = {{.4, .6, .5}, {1.9, .9, .7}, {2.3, 3.4, 1.1}};
    atom.chemmap_ = {0, 1, 0};
    api.getEFS(atom);
    const auto source = atom;
    const auto oracle = mtp_cell_energy_fd(api, atom, step, false, false, 0.0, cutoff);
    REQUIRE(oracle.bottomRightCorner(9, 9).norm() > .01);
    if (cutoff != 2.0) {
        REQUIRE(oracle.topLeftCorner(9, 9).norm() > .001);
        REQUIRE(oracle.block(0, 9, 9, 9).norm() > .001);
    }
    std::vector<double> coefficients(api.get_NL_coeffs_count());
    api.nl_get_coeffs(coefficients.data());
    ComputeHessian calculator;
    calculator.set_cutoff(cutoff);
    calculator.request_cell = true;
    calculator.use_symmetry = false;
    for (auto method : {ComputeHessian::HessianMethod::Auto,
                        ComputeHessian::HessianMethod::Analytic}) {
        calculator.method = method;
        Eigen::MatrixXd result;
        calculator.compute(atom, result, &api);
        REQUIRE((result - oracle).norm() < 3e-6);
        calculator.request_fc = false;
        calculator.compute(atom, result, &api);
        REQUIRE((result - oracle.topLeftCorner(9, 9)).norm() < 3e-6);
        calculator.request_fc = true;
        calculator.request_response_pos_cell = false;
        calculator.compute(atom, result, &api);
        REQUIRE(result.block(0, 9, 9, 9).isZero(1e-14));
        REQUIRE((result.topLeftCorner(9, 9) - oracle.topLeftCorner(9, 9)).norm() < 3e-6);
        calculator.request_response_pos_cell = true;
        calculator.compute_fc(atom, result, &api);
        REQUIRE((result - oracle.bottomRightCorner(9, 9)).norm() < 3e-6);
        const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(9, -.4, .7);
        Eigen::VectorXd product;
        calculator.compute_hvp(atom, direction, product, &api);
        REQUIRE((product - result * direction).norm() < 1e-9);
    }
    // Direct public OMP dispatch, including seeded EFS and overwritten outputs.
    const double rcut = cutoff > 1e-5 ? cutoff : api.cutoff;
    Neighborhoods nbhs(atom, rcut);
    auto fresh = atom;
    api.getEFS(fresh, nbhs);
    for (int entry = 0; entry < 4; ++entry) {
        // Symmetry APIs use the complete native neighborhood.
        if ((entry == 1 || entry == 3) && cutoff > 0.0) continue;
        CAPTURE(entry);
        auto seeded = atom;
        seeded.energy_ = .7;
        for (int i = 0; i < seeded.size(); ++i) {
            seeded.site_energies_[i] = .1 * (i + 1);
            seeded.forces_[i].setConstant(.2);
        }
        seeded.stress_.setConstant(.3);
        const auto seed = seeded;
        HessianMatrix blocks;
        Eigen::MatrixXd cell, mixed;
        auto evaluate = [&] {
            blocks.resize(1); blocks(0, 0).setConstant(77.0);
            if (entry == 0) api.getEFSHessianOMP(seeded, nbhs, blocks);
            if (entry == 1) api.getEFSHessianSymmOMP(seeded, blocks);
            if (entry == 2) api.getEFSHessianCellOMP(seeded, nbhs, blocks, cell, mixed);
            if (entry == 3) api.getEFSHessianCellSymmOMP(seeded, blocks, cell, mixed);
        };
        auto check = [&](double seed_scale, double count) {
            REQUIRE(seeded.energy_ == Approx(seed_scale * seed.energy_ + count * fresh.energy_).margin(1e-10));
            REQUIRE((seeded.stress_ - seed_scale * seed.stress_ - count * fresh.stress_).norm() < 1e-10);
            for (int i = 0; i < seeded.size(); ++i) {
                REQUIRE(seeded.site_energies_[i] == Approx(seed_scale * seed.site_energies_[i] + count * fresh.site_energies_[i]).margin(1e-10));
                REQUIRE((seeded.forces_[i] - seed_scale * seed.forces_[i] - count * fresh.forces_[i]).norm() < 1e-10);
            }
            REQUIRE((HessianUtils::to_dense(blocks) - oracle.bottomRightCorner(9, 9)).norm() < 3e-6);
            REQUIRE(seeded.pos_ == seed.pos_);
            REQUIRE(seeded.cell_ == seed.cell_);
        };
        api.add_efs = false;
        evaluate(); check(0.0, 1.0);
        seeded = seed;
        api.add_efs = true;
        evaluate(); check(1.0, 1.0);
        evaluate(); check(1.0, 2.0);
    }
    api.add_efs = false;
    auto *mtp = dynamic_cast<PotMTP *>(&api);
    REQUIRE(mtp != nullptr);
    if (linear) dynamic_cast<PotMTP_M<m_model_mtpli> &>(api).lr_les = true;
    else dynamic_cast<PotMTP_M<m_model_mtpq2> &>(api).lr_les = true;
    HessianMatrix rejected;
    Eigen::MatrixXd cell, mixed;
    REQUIRE_THROWS_AS(api.getEFSHessianOMP(atom, nbhs, rejected), GPException);
    REQUIRE_THROWS_AS(api.getEFSHessianSymmOMP(atom, rejected), GPException);
    REQUIRE_THROWS_AS(api.getEFSHessianCellOMP(atom, nbhs, rejected, cell, mixed), GPException);
    REQUIRE_THROWS_AS(api.getEFSHessianCellSymmOMP(atom, rejected, cell, mixed), GPException);
    std::vector<double> after(coefficients.size());
    api.nl_get_coeffs(after.data());
    REQUIRE(after == coefficients);
    REQUIRE(atom.pos_ == source.pos_); REQUIRE(atom.cell_ == source.cell_);
    REQUIRE(atom.energy_ == source.energy_); REQUIRE(atom.forces_ == source.forces_);
    REQUIRE(atom.stress_ == source.stress_); REQUIRE(atom.site_energies_ == source.site_energies_);
}

TEST_CASE("Hessian: native analytical queries support molecules and slabs",
          "[hessian][partial-pbc]") {
    CellHessianThreadScope threads;
    MCellFixtureFiles files;
    const bool nonlinear = GENERATE(false, true);
    const int periodic = GENERATE(0, 2, 3);
    const bool symmetry = GENERATE(false, true);
    const double step = GENERATE(.001, .0005);
    CAPTURE(nonlinear, periodic, symmetry, step);
    std::unique_ptr<AnyPotential> owner;
    if (nonlinear) owner = files.make<m_model_mtpq2>(false);
    else {
        auto mtp = std::make_unique<PotMTP>();
        load_many_body_mtp_fixture(*mtp, 2);
        mtp->omp_init();
        owner = std::move(mtp);
    }
    AnyPotential &api = *owner;
    Atoms atom;
    atom.cell_ << 5.2, .6, .2, .1, 5.6, .3, .2, .1, 5.8;
    atom.pbc_ = {periodic > 0, periodic > 1, periodic > 2};
    atom.pos_ = {{.4, .6, .5}, {1.9, .9, .7}, {2.3, 3.4, 1.1}};
    atom.chemmap_ = {0, 1, 0};
    api.getEFS(atom);
    const auto source = atom;
    const Eigen::MatrixXd oracle = mtp_cell_energy_fd(api, atom, step, false, false);
    Eigen::MatrixXd force_fd(9, 9);
    for (int column = 0; column < 9; ++column) {
        auto plus = atom, minus = atom;
        plus.pos_[column / 3](column % 3) += step;
        minus.pos_[column / 3](column % 3) -= step;
        api.getEFS(plus);
        api.getEFS(minus);
        for (int i = 0; i < 3; ++i)
            force_fd.block<3, 1>(3 * i, column) =
                -(plus.forces_[i] - minus.forces_[i]).transpose() / (2 * step);
    }
    REQUIRE(force_fd.norm() > .01);
    REQUIRE((force_fd - oracle.bottomRightCorner(9, 9)).norm() < 3e-6);
    ComputeHessian calculator;
    calculator.use_symmetry = symmetry;
    for (auto method : {ComputeHessian::HessianMethod::Auto,
                        ComputeHessian::HessianMethod::Analytic}) {
        calculator.method = method;
        Eigen::MatrixXd dense;
        calculator.compute_fc(atom, dense, &api);
        REQUIRE((dense - force_fd).norm() < 3e-6);
        HessianMatrix blocks;
        calculator.compute_fc_blocks(atom, blocks, &api);
        REQUIRE((HessianUtils::to_dense(blocks) - force_fd).norm() < 3e-6);
        const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(9, -.4, .7);
        Eigen::VectorXd product;
        calculator.compute_hvp(atom, direction, product, &api);
        REQUIRE((product - dense * direction).norm() < 1e-9);
        calculator.request_cell = true;
        calculator.compute(atom, dense, &api);
        REQUIRE((dense - oracle).norm() < 3e-6);
        calculator.request_fc = false;
        calculator.compute(atom, dense, &api);
        REQUIRE((dense - oracle.topLeftCorner(9, 9)).norm() < 3e-6);
        calculator.request_fc = true;
        calculator.request_response_pos_cell = false;
        calculator.compute(atom, dense, &api);
        REQUIRE(dense.block(0, 9, 9, 9).isZero(1e-14));
        REQUIRE((dense.topLeftCorner(9, 9) - oracle.topLeftCorner(9, 9)).norm() < 3e-6);
        calculator.request_response_pos_cell = true;
        calculator.request_cell = false;
    }
    REQUIRE(atom.pos_ == source.pos_); REQUIRE(atom.cell_ == source.cell_);
    REQUIRE(atom.pbc_ == source.pbc_); REQUIRE(atom.cartesian == source.cartesian);
    REQUIRE(atom.energy_ == source.energy_); REQUIRE(atom.forces_ == source.forces_);
    REQUIRE(atom.stress_ == source.stress_); REQUIRE(atom.site_energies_ == source.site_energies_);
    REQUIRE(atom.have_energy == source.have_energy); REQUIRE(atom.have_stress == source.have_stress);
}
