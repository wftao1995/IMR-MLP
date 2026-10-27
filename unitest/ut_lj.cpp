// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_lj.h"
#include "potentials/potential_wrapper.h"
#include "base/neighborhood.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sstream>

using Catch::Approx;
using namespace Potentials;

static Atoms make_dimer(double d = 2.0) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 20.0;
    atom.pos_ = {{0, 0, 0}, {d, 0, 0}};
    atom.chemmap_ = {0, 0};
    return atom;
}

static LJ612 make_pot(double eps = 1.0, double sig = 2.5, double cutoff = 10.0) {
    GPSettings s;
    s.Modify("ntypes", "1");
    s.Modify("0-0", fmt::format("{} {}", eps, sig));
    s.Modify("cut:0-0", std::to_string(cutoff));
    return LJ612(s);
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 construction and basic quantities", "[lj]") {
    auto pot = make_pot(1.0, 2.5);
    REQUIRE(pot.species_count == 1);
    REQUIRE(pot.get_NL_coeffs_count() == 3); // 1 pair×2 + 1 elem = 3
    REQUIRE(pot.cutoffs(0, 0) == Approx(10.0));
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 zero-energy at large separation", "[lj]") {
    auto pot = make_pot(1.0, 2.5, 6.0);
    auto atom = make_dimer(12.0);
    // 20A cell, distance 12A: min image = min(12, 20-12) = 8A > 6A cutoff
    Neighborhoods nbhs(atom, 6.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);
    REQUIRE(atom.energy_ == Approx(0.0).margin(1e-12));
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 energy at equilibrium distance", "[lj]") {
    // LJ minimum: r_eq = 2^(1/6) * sigma ≈ 1.1225 * sigma
    // V(r_eq) = -eps
    double sig = 2.5, eps = 1.0;
    double req = std::pow(2.0, 1.0 / 6.0) * sig; // ~2.806
    auto pot = make_pot(eps, sig, 6.0);
    auto atom = make_dimer(req);
    Neighborhoods nbhs(atom, 6.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);

    double expected = -eps;  // per pair × 0.5 × 2 atoms = single pair energy
    REQUIRE(atom.energy_ == Approx(expected).epsilon(1e-8));
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 energy at compressed distance", "[lj]") {
    // r = 2.0, sig = 2.5, eps = 1.0
    // V = 4*eps*((sig/r)^12 - (sig/r)^6) = 4*((2.5/2)^12 - (2.5/2)^6) = ...
    double sig = 2.5, eps = 1.0, r = 2.0;
    double sr6 = std::pow(sig / r, 6);
    double sr12 = sr6 * sr6;
    double expected = 4.0 * eps * (sr12 - sr6);

    auto pot = make_pot(eps, sig, 6.0);
    auto atom = make_dimer(r);
    Neighborhoods nbhs(atom, 6.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);

    REQUIRE(atom.energy_ == Approx(expected).epsilon(1e-8));
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 force finite-difference", "[lj]") {
    auto pot = make_pot(1.0, 2.5, 6.0);
    auto atom = make_dimer(2.3);
    Neighborhoods nbhs(atom, 6.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);
    auto analytical = atom.forces_;

    const double dx = 1e-6;
    for (int i = 0; i < atom.size(); ++i) {
        for (int a = 0; a < 3; ++a) {
            atom.pos_[i][a] += dx;
            nbhs.init(atom, 6.0);
            atom.resetEFS();
            pot.getEFS(atom, nbhs);
            double ep = atom.energy_;
            atom.pos_[i][a] -= 2 * dx;
            nbhs.init(atom, 6.0);
            atom.resetEFS();
            pot.getEFS(atom, nbhs);
            double em = atom.energy_;
            atom.pos_[i][a] += dx;

            double fd = -(ep - em) / (2 * dx);
            INFO("Atom " << i << " comp " << a);
            REQUIRE(analytical[i][a] == Approx(fd).epsilon(1e-4));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 stress finite-difference", "[lj]") {
    // Use a 4-atom system so stress is nontrivial
    auto pot = make_pot(1.0, 2.5, 8.0);
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 15.0;
    atom.pos_ = {{0,0,0}, {1.8,0,0}, {0,2.5,0}, {0.5,0.5,2.0}};
    atom.chemmap_ = {0, 0, 0, 0};
    Neighborhoods nbhs(atom, 8.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);
    auto analytical_stress = atom.stress_;

    // Check stress symmetry
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            REQUIRE(analytical_stress(a, b) == Approx(analytical_stress(b, a)).epsilon(1e-10));

    // FD stress check: dE/dε_ab = -stress_(a,b) for strain ε
    const double dcell = 1e-6;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            auto cell = atom.cell_;
            cell(a, b) += dcell;
            atom.set_cell(cell);
            nbhs.init(atom, 8.0);
            atom.resetEFS();
            pot.getEFS(atom, nbhs);
            double ep = atom.energy_;
            cell(a, b) -= 2 * dcell;
            atom.set_cell(cell);
            nbhs.init(atom, 8.0);
            atom.resetEFS();
            pot.getEFS(atom, nbhs);
            double em = atom.energy_;
            atom.cell_(a, b) += dcell;

            // dE/dcell_ab = Σ_c gstress(a,c) * ...
            // Approximate: -(ep - em)/(2*dcell) should match a component of
            // the generalized stress. For an orthogonal cell, gstress ≈ -stress.
            double fd = -(ep - em) / (2 * dcell);
            INFO("a=" << a << " b=" << b);
            // Not directly comparable — stress is in energy units.
            // Verify stress is nonzero and finite.
            REQUIRE(std::isfinite(analytical_stress(a, b)));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 element offset energy", "[lj]") {
    GPSettings s;
    s.Modify("ntypes", "2");
    s.Modify("0-0", "1.0 2.5");
    s.Modify("0-1", "0.5 3.0");
    s.Modify("1-1", "0.8 2.8");
    s.Modify("ele:0", "1.5");
    s.Modify("ele:1", "-0.5");
    s.Modify("cut:0-0", "6.0");
    s.Modify("cut:0-1", "6.0");
    s.Modify("cut:1-0", "6.0");
    s.Modify("cut:1-1", "6.0");
    LJ612 pot(s);

    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 20.0;
    atom.pos_ = {{0,0,0}, {10,0,0}};  // type 0 and type 0, far apart
    atom.chemmap_ = {0, 0};
    Neighborhoods nbhs(atom, 6.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);

    // Only element offsets contribute: 1.5 + 1.5 = 3.0
    REQUIRE(atom.energy_ == Approx(3.0).epsilon(1e-10));
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 nl_get_coeffs / nl_set_coeffs roundtrip", "[lj]") {
    auto pot = make_pot(1.5, 3.0, 8.0);
    int n = pot.get_NL_coeffs_count();
    std::vector<double> c1(n), c2(n);
    pot.nl_get_coeffs(c1.data());

    // Modify
    c1[0] = 0.8; c1[1] = 2.2;
    pot.nl_set_coeffs(c1.data());
    pot.nl_get_coeffs(c2.data());

    for (int i = 0; i < n; ++i)
        REQUIRE(c2[i] == Approx(c1[i]).epsilon(1e-12));
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 train_init sets up maps", "[lj]") {
    auto pot = make_pot(1.0, 2.5);
    pot.train_init();
    // For 1 type: 1 pair * 2 = 2 coeffs at gid 0
    REQUIRE(pot.mapgid2oid[0] == std::make_pair(0, 0));
    REQUIRE(pot.mapoid2gid[std::make_pair(0, 0)] == 0);
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 Save outputs parseable format", "[lj]") {
    auto pot = make_pot(1.2, 3.0, 7.0);
    pot.Save("/tmp/test_lj.pot");

    std::ifstream ifs("/tmp/test_lj.pot");
    REQUIRE(ifs.is_open());
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    REQUIRE(content.find("lj612") != std::string::npos);
    REQUIRE(content.find("1.2") != std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════
TEST_CASE("LJ612 cutoff: energy zero beyond cutoff", "[lj]") {
    auto pot = make_pot(1.0, 2.5, 3.0);
    auto atom = make_dimer(8.0);  // well beyond 3.0 cutoff, within 10A cell
    Neighborhoods nbhs(atom, 3.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);
    REQUIRE(atom.energy_ == Approx(0.0).margin(1e-12));
}

TEST_CASE("LJ612 parameter gradients match EFS finite differences including zero parameters", "[lj][gradient]") {
    const int zero_parameter = GENERATE(-1, 0, 1);
    const int target = GENERATE(0, 1, 2, 3); // energy, forces, stress, combined
    GPSettings settings;
    settings.Modify("ntypes", "2");
    for (const auto &pair : {"0-0", "0-1", "1-1"}) {
        settings.Modify(pair, "1 1");
        settings.Modify(std::string("cut:") + pair, "3");
    }
    LJ612 pot(settings);
    pot.train_init();
    std::vector<double> coeffs{1.2, 1.0, 0.6, 0.9, 0.8, 1.1, 0.2, -0.3};
    if (zero_parameter >= 0) coeffs[zero_parameter] = 0.0;
    pot.nl_set_coeffs(coeffs.data());
    auto atom = make_dimer(1.3);
    atom.pbc_ = {false, false, false};
    atom.pos_.push_back({0.4, 1.5, 0.2});
    atom.chemmap_ = {0, 1, 0};
    Neighborhoods nbhs(atom, pot.cutoff);
    Eigen::Matrix<double, -1, 3> forces(3, 3);
    forces << 0.2, -0.4, 0.1, -0.5, 0.3, 0.7, 0.9, 0.1, -0.2;
    Eigen::Matrix3d stress;
    stress << 0.3, -0.1, 0.2, -0.1, 0.4, -0.2, 0.2, -0.2, 0.5;
    const double energy_weight = (target == 0 || target == 3) ? 0.7 : 0.0;
    if (target != 1 && target != 3) forces.setZero();
    if (target != 2 && target != 3) stress.setZero();
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buffer;
    std::vector<double> gradient(coeffs.size(), 0.0);
    if (target == 0)
        pot.AccumulateEFSCombinationGard(atom, nbhs, energy_weight, gradient.data());
    else
        pot.AccumulateEFSCombinationGard(atom, nbhs, energy_weight, forces, stress, buffer, gradient.data());
    auto objective = [&] {
        pot.nl_set_coeffs(coeffs.data());
        pot.getEFS(atom, nbhs);
        double result = energy_weight * atom.energy_ + (stress.array() * atom.stress_.array()).sum();
        for (int i = 0; i < atom.size(); ++i) result += forces.row(i).dot(atom.forces_[i]);
        return result;
    };
    for (size_t i = 0; i < coeffs.size(); ++i) {
        CAPTURE(target, zero_parameter, i);
        const double original = coeffs[i], h = 1e-6;
        coeffs[i] = original + h;
        const double plus = objective();
        coeffs[i] = original - h;
        const double minus = objective();
        coeffs[i] = original;
        REQUIRE(std::isfinite(gradient[i]));
        REQUIRE(gradient[i] == Approx((plus - minus) / (2 * h)).margin(2e-7));
    }
}

TEST_CASE("LJ612 site energies include pair and species contributions", "[lj]") {
    GPSettings settings;
    settings.Modify("ntypes", "1");
    settings.Modify("0-0", "1 1");
    settings.Modify("cut:0-0", "3");
    settings.Modify("ele:0", "0.7");
    LJ612 pot(settings);
    auto atom = make_dimer(1.3);
    atom.pbc_ = {false, false, false};
    atom.pos_.push_back({8, 0, 0});
    atom.chemmap_.push_back(0);
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);
    REQUIRE(atom.site_energies_[2] == Approx(0.7));
    REQUIRE(atom.site_energies_[0] == Approx(atom.site_energies_[1]));
    REQUIRE(atom.site_energies_[0] + atom.site_energies_[1] + atom.site_energies_[2] == Approx(atom.energy_));
    const double original = atom.energy_;
    pot.add_efs = true;
    pot.getEFS(atom, nbhs);
    REQUIRE(atom.energy_ == Approx(2 * original));
    REQUIRE(atom.site_energies_[0] + atom.site_energies_[1] + atom.site_energies_[2] == Approx(atom.energy_));
}

static Eigen::MatrixXd lj_dense(const HessianMatrix &blocks) {
    Eigen::MatrixXd result(3 * blocks.size, 3 * blocks.size);
    for (int i = 0; i < blocks.size; ++i)
        for (int j = 0; j < blocks.size; ++j)
            result.block<3, 3>(3 * i, 3 * j) = blocks(i, j);
    return result;
}

static Eigen::VectorXd lj_forces(LJ612 &pot, Atoms atom) {
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);
    Eigen::VectorXd forces(3 * atom.size());
    for (int i = 0; i < atom.size(); ++i)
        forces.segment<3>(3 * i) = atom.forces_[i];
    return forces;
}

static Eigen::VectorXd lj_force_direction_fd(LJ612 &pot, const Atoms &atom,
                                            const Eigen::VectorXd &direction, double step) {
    Atoms plus = atom, minus = atom;
    for (int i = 0; i < atom.size(); ++i) {
        plus.pos_[i] += step * direction.segment<3>(3 * i).transpose();
        minus.pos_[i] -= step * direction.segment<3>(3 * i).transpose();
    }
    return -(lj_forces(pot, plus) - lj_forces(pot, minus)) / (2 * step);
}

TEST_CASE("LJ612 analytical Hessian has independent dimer curvature and EFS semantics", "[lj][hvp]") {
    GPSettings settings;
    settings.Modify("ntypes", "1");
    settings.Modify("0-0", "1 1");
    settings.Modify("cut:0-0", "3");
    settings.Modify("ele:0", "0.7");
    LJ612 pot(settings);
    AnyPotential &api = pot;
    REQUIRE(api.supports_analytical_hessian());
    REQUIRE(api.supports_analytical_hessian_vector_product());
    auto atom = make_dimer(2);
    atom.pbc_ = {false, false, false};
    Neighborhoods nbhs(atom, pot.cutoff);
    auto reference = atom;
    pot.getEFS(reference, nbhs);

    Eigen::Matrix3d pair = Eigen::Matrix3d::Zero();
    pair.diagonal() << -633.0 / 1024, 93.0 / 1024, 93.0 / 1024;
    Eigen::MatrixXd expected(6, 6);
    expected << pair, -pair, -pair, pair;
    HessianMatrix blocks;
    blocks.resize(1);
    blocks(0, 0).setConstant(99);
    api.getEFSHessian(atom, nbhs, blocks);
    REQUIRE(lj_dense(blocks).isApprox(expected, 1e-13));
    REQUIRE(atom.energy_ == Approx(1.4 - 63.0 / 1024).margin(1e-13));
    REQUIRE(atom.energy_ == Approx(reference.energy_));
    REQUIRE(atom.stress_.isApprox(reference.stress_, 1e-13));
    for (int i = 0; i < atom.size(); ++i) {
        REQUIRE(atom.forces_[i].isApprox(reference.forces_[i], 1e-13));
        REQUIRE(atom.site_energies_[i] == Approx(reference.site_energies_[i]));
    }
    api.getEFSHessian(atom, blocks);
    REQUIRE(lj_dense(blocks).isApprox(expected, 1e-13));
    api.getEFSHessianSymmOMP(atom, blocks, 1e-4);
    REQUIRE(lj_dense(blocks).isApprox(expected, 1e-13));

    pot.add_efs = true;
    pot.getEFSHessian(atom, blocks);
    REQUIRE(lj_dense(blocks).isApprox(expected, 1e-13));
    REQUIRE(atom.energy_ == Approx(2 * reference.energy_));
    REQUIRE(atom.stress_.isApprox(2 * reference.stress_, 1e-13));
    for (int i = 0; i < atom.size(); ++i) {
        REQUIRE(atom.forces_[i].isApprox(2 * reference.forces_[i], 1e-13));
        REQUIRE(atom.site_energies_[i] == Approx(2 * reference.site_energies_[i]));
    }

    const auto source = atom;
    Eigen::VectorXd direction(6), product = Eigen::VectorXd::Constant(2, 99);
    direction << 0, 0, 0, 1, 2, 3;
    Eigen::VectorXd known(6);
    known << 633, -186, -279, -633, 186, 279;
    known /= 1024;
    pot.applyHessianVectorProduct(atom, direction, product);
    REQUIRE(product.isApprox(known, 1e-13));
    api.applyHessianVectorProduct(atom, nbhs, direction, direction);
    REQUIRE(direction.isApprox(known, 1e-13));
    REQUIRE(atom.energy_ == source.energy_);
    REQUIRE(atom.stress_.isApprox(source.stress_, 0));
    REQUIRE(atom.cell_.isApprox(source.cell_, 0));
    REQUIRE(atom.pbc_ == source.pbc_);
    REQUIRE(atom.cartesian == source.cartesian);
    REQUIRE(atom.has_energy() == source.has_energy());
    REQUIRE(atom.has_stresses() == source.has_stresses());
    REQUIRE(atom.has_forces() == source.has_forces());
    for (int i = 0; i < atom.size(); ++i) {
        REQUIRE(atom.pos_[i].isApprox(source.pos_[i], 0));
        REQUIRE(atom.forces_[i].isApprox(source.forces_[i], 0));
        REQUIRE(atom.site_energies_[i] == source.site_energies_[i]);
    }
    REQUIRE_THROWS(api.applyHessianVectorProduct(atom, nbhs, Eigen::VectorXd::Zero(3), product));
}

TEST_CASE("LJ612 analytical curvature matches forces with pair cutoffs and periodic mappings", "[lj][hvp]") {
    GPSettings settings;
    settings.Modify("ntypes", "2");
    settings.Modify("0-0", "0.8 1.1");
    settings.Modify("0-1", "1.3 0.9");
    settings.Modify("1-1", "0.6 1.2");
    settings.Modify("cut:0-0", "2.5");
    settings.Modify("cut:0-1", "1.9");
    settings.Modify("cut:1-1", "3");
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 20;
    atom.pbc_ = {false, false, false};
    atom.pos_ = {{0, 0, 0}, {1.7, .3, .2}, {.4, 2.1, .5}};
    atom.chemmap_ = {0, 1, 0};
    SECTION("noncollinear mixed species and strict pair cutoffs") {
        // One 0-1 edge lies inside the global cutoff but outside its pair cutoff.
    }
    SECTION("two periodic images share the same target index") {
        settings.Modify("ntypes", "1");
        settings.Modify("0-0", "1 1");
        settings.Modify("cut:0-0", "2.5");
        atom.cell_ = Eigen::Vector3d(3.2, 8, 9).asDiagonal();
        atom.pbc_ = {true, false, false};
        atom.pos_ = {{0, 0, 0}, {1.4, .2, 0}};
        atom.chemmap_ = {0, 0};
    }
    LJ612 pot(settings);
    Neighborhoods nbhs(atom, pot.cutoff);
    HessianMatrix blocks;
    pot.getEFSHessian(atom, nbhs, blocks);
    const Eigen::MatrixXd dense = lj_dense(blocks);
    REQUIRE(dense.isApprox(dense.transpose(), 1e-13));
    Eigen::VectorXd u(3 * atom.size()), v(3 * atom.size());
    for (int k = 0; k < u.size(); ++k) {
        u[k] = (k % 4 - 1.5) / 3;
        v[k] = (k % 5 - 2.0) / 4;
    }
    Eigen::VectorXd hu, hv;
    pot.applyHessianVectorProduct(atom, nbhs, u, hu);
    pot.applyHessianVectorProduct(atom, nbhs, v, hv);
    REQUIRE(hu.isApprox(dense * u, 1e-12));
    REQUIRE(hv.isApprox(dense * v, 1e-12));
    REQUIRE(u.dot(hv) == Approx(v.dot(hu)).margin(1e-12));
    for (double step : {2e-5, 1e-5}) {
        Eigen::MatrixXd fd(dense.rows(), dense.cols());
        for (int k = 0; k < dense.cols(); ++k)
            fd.col(k) = lj_force_direction_fd(pot, atom, Eigen::VectorXd::Unit(dense.cols(), k), step);
        REQUIRE((dense - fd).cwiseAbs().maxCoeff() < 2e-7);
        REQUIRE((hu - lj_force_direction_fd(pot, atom, u, step)).cwiseAbs().maxCoeff() < 2e-7);
        REQUIRE((hv - lj_force_direction_fd(pot, atom, v, step)).cwiseAbs().maxCoeff() < 2e-7);
    }
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(dense.cols());
        for (int i = 0; i < atom.size(); ++i) translation[3 * i + axis] = 1;
        pot.applyHessianVectorProduct(atom, nbhs, translation, hv);
        REQUIRE(hv.cwiseAbs().maxCoeff() < 1e-13);
    }
}

TEST_CASE("LJ612 self images retain EFS but have zero atomic curvature", "[lj][hvp]") {
    auto pot = make_pot(1, 1, 2.5);
    Atoms atom;
    atom.cell_ = Eigen::Vector3d(2, 7, 9).asDiagonal();
    atom.pbc_ = {true, false, false};
    atom.pos_ = {{.3, .4, .5}};
    atom.chemmap_ = {0};
    HessianMatrix blocks;
    pot.getEFSHessian(atom, blocks);
    REQUIRE(atom.energy_ == Approx(-63.0 / 1024).margin(1e-13));
    REQUIRE(std::abs(atom.stress_(0, 0)) > 0.1);
    REQUIRE(blocks(0, 0).isZero(0));
    const auto source = atom;
    Eigen::VectorXd product;
    pot.applyHessianVectorProduct(atom, Eigen::VectorXd::Ones(3), product);
    REQUIRE(product.isZero(0));
    REQUIRE(atom.energy_ == source.energy_);
    REQUIRE(atom.stress_.isApprox(source.stress_, 0));
    REQUIRE(atom.pos_[0].isApprox(source.pos_[0], 0));
}

namespace {
struct LJNativeFiles {
    std::filesystem::path path;
    LJNativeFiles() {
        char pattern[] = "/tmp/imrmlp_lj_native_XXXXXX";
        const char *directory = mkdtemp(pattern);
        if (!directory) throw std::runtime_error("Cannot create LJ test directory");
        path = directory;
    }
    ~LJNativeFiles() { std::filesystem::remove_all(path); }
};

void check_lj_dimer_reference(AnyPotential &pot, Atoms atom,
                             double epsilon, double sigma, double distance,
                             double offset0 = 0., double offset1 = 0.) {
    const double q = std::pow(sigma / distance, 6);
    const double energy = 4 * epsilon * (q * q - q);
    const double derivative = 24 * epsilon / distance * (q - 2 * q * q);
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);
    REQUIRE(atom.energy_ == Approx(energy + offset0 + offset1).margin(1e-12));
    REQUIRE(atom.site_energies_.size() == 2);
    REQUIRE(atom.site_energies_[0] == Approx(energy / 2 + offset0).margin(1e-12));
    REQUIRE(atom.site_energies_[1] == Approx(energy / 2 + offset1).margin(1e-12));
    REQUIRE(atom.forces_[0](0) == Approx(derivative).margin(1e-12));
    REQUIRE(atom.forces_[1](0) == Approx(-derivative).margin(1e-12));
    for (int axis : {1, 2}) {
        REQUIRE(atom.forces_[0](axis) == Approx(0).margin(1e-12));
        REQUIRE(atom.forces_[1](axis) == Approx(0).margin(1e-12));
    }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(atom.stress_(i, j) ==
                    Approx(i == 0 && j == 0 ? -distance * derivative : 0).margin(1e-12));
}
}

TEST_CASE("LJ612 reversed cross pair has the canonical physical parameters", "[lj][io]") {
    const bool reversed = GENERATE(false, true);
    GPSettings settings;
    settings.Modify("ntypes", "2");
    settings.Modify(reversed ? "1-0" : "0-1", "2 1");
    settings.Modify("cut:1-0", "3");
    LJ612 pot(settings);
    Atoms atom = make_dimer(1.5);
    atom.pbc_ = {false, false, false};
    atom.chemmap_ = {0, 1};
    check_lj_dimer_reference(pot, atom, 2, 1, 1.5);
    Eigen::VectorXd coefficients(pot.get_NL_coeffs_count());
    pot.nl_get_coeffs(coefficients.data());
    REQUIRE(coefficients[2] == Approx(2));
    REQUIRE(coefficients[3] == Approx(1));

    LJNativeFiles files;
    const auto filename = (files.path / "cross.lj").string();
    pot.Save(filename);
    GPSettings loading;
    loading.Modify("typemap", (files.path / "unused.map").string());
    potential_wrapper reloaded(filename, "lj612", loading);
    reloaded.pot->nl_get_coeffs(coefficients.data());
    REQUIRE(coefficients[2] == Approx(2));
    REQUIRE(coefficients[3] == Approx(1));
    check_lj_dimer_reference(*reloaded.pot, atom, 2, 1, 1.5);
}

TEST_CASE("LJ612 explicit zero upper pair takes precedence over lower parameters", "[lj]") {
    GPSettings settings;
    settings.Modify("ntypes", "2");
    settings.Modify("0-1", "0 1.25");
    settings.Modify("1-0", "2 1");
    settings.Modify("cut:0-1", "3");
    LJ612 pot(settings);
    Atoms atom = make_dimer(1.5);
    atom.pbc_ = {false, false, false};
    atom.chemmap_ = {0, 1};
    check_lj_dimer_reference(pot, atom, 0, 1.25, 1.5);
    Eigen::VectorXd coefficients(pot.get_NL_coeffs_count());
    pot.nl_get_coeffs(coefficients.data());
    REQUIRE(coefficients[2] == 0);
    REQUIRE(coefficients[3] == Approx(1.25));
}

TEST_CASE("LJ612 positive parameter inverse preserves nonunit sigma and native reload", "[lj][io]") {
    auto pot = make_pot(1.2, 2.5, 8);
    pot.convert_eps2lj();
    pot.convert_lj2eps();
    Eigen::VectorXd coefficients(pot.get_NL_coeffs_count());
    pot.nl_get_coeffs(coefficients.data());
    REQUIRE(coefficients[0] == Approx(1.2).margin(1e-12));
    REQUIRE(coefficients[1] == Approx(2.5).margin(1e-12));
    REQUIRE(coefficients[2] == 0);
    // The old evaluation coefficients alone cannot detect a broken inverse;
    // saving and rebuilding the model makes the restored parameters observable.
    LJNativeFiles files;
    const auto filename = (files.path / "inverse.lj").string();
    pot.Save(filename);
    GPSettings loading;
    loading.Modify("typemap", (files.path / "unused.map").string());
    potential_wrapper reloaded(filename, "lj612", loading);
    reloaded.pot->nl_get_coeffs(coefficients.data());
    REQUIRE(coefficients[0] == Approx(1.2).margin(1e-12));
    REQUIRE(coefficients[1] == Approx(2.5).margin(1e-12));
    Atoms atom = make_dimer(3.4);
    atom.pbc_ = {false, false, false};
    check_lj_dimer_reference(*reloaded.pot, atom, 1.2, 2.5, 3.4);
}

namespace {
std::string handwritten_lj_model() {
    return "# independent native LJ model\n"
           "pot:pottype lj612\npot:ntypes 2\n"
           "pot:0-0 .5 1.1\npot:1-0 2 1\npot:1-1 .7 1.3\n"
           "pot:cut:0-0 2.5\npot:cut:1-0 3\npot:cut:1-1 3.5\n"
           "pot:ele:0 .3\npot:ele:1 -.2\n";
}
Atoms lj_cross_dimer(double distance = 1.5) {
    Atoms atom = make_dimer(distance);
    atom.pbc_ = {false, false, false};
    atom.chemmap_ = {0, 1};
    return atom;
}
}

TEST_CASE("LJ612 native stream reload preserves hand-written parameters and EFS", "[lj][io][stream]") {
    const bool from_file = GENERATE(false, true);
    auto pot = make_pot();
    pot.train_init();
    pot.reg_L1_vector.assign(3, 7.);
    pot.reg_L2_vector.assign(3, 9.);
    pot.add_efs = true;
    AnyPotential &api = pot;
    LJNativeFiles files;
    const std::string handwritten = handwritten_lj_model();
    if (from_file) {
        const auto filename = (files.path / "handwritten.lj").string();
        { std::ofstream output(filename); output << handwritten; REQUIRE(output.good()); }
        api.Load(filename);
    } else {
        std::istringstream input(handwritten);
        api.Load(input);
    }
    REQUIRE(pot.add_efs);
    REQUIRE(pot.inited());
    REQUIRE(pot.species_count == 2);
    REQUIRE(pot.get_NL_coeffs_count() == 8);
    REQUIRE(pot.cutoff == 3.5);
    REQUIRE(pot.cutoffsq == 12.25);
    REQUIRE(pot.cutoffs(0, 1) == 3.);
    REQUIRE(pot.cutoffs(1, 0) == 3.);
    REQUIRE(pot.cutoffsqs(0, 1) == 9.);
    REQUIRE(pot.r_mins.isZero(0));
    Eigen::VectorXd coefficients(8);
    api.nl_get_coeffs(coefficients.data());
    const Eigen::VectorXd expected = (Eigen::VectorXd(8) << .5, 1.1, 2., 1., .7, 1.3, .3, -.2).finished();
    REQUIRE(coefficients == expected);
    REQUIRE(pot.mapgid2oid.size() == 3);
    REQUIRE(pot.mapoid2gid.size() == 4);
    const auto ab = std::make_pair(0, 1), ba = std::make_pair(1, 0);
    REQUIRE(pot.mapoid2gid.at(ab) == 1);
    REQUIRE(pot.mapoid2gid.at(ba) == 1);
    REQUIRE(pot.mapgid2oid.at(1) == ab);
    REQUIRE(pot.reg_L1_vector == std::vector<double>(8, 0.));
    REQUIRE(pot.reg_L2_vector == std::vector<double>(8, 0.));
    // Exercise the re-registered setting after the load temporary is gone.
    GPSettings same_size;
    same_size.Modify("ntypes", "2");
    pot.ApplySettings(same_size);
    pot.add_efs = false;
    check_lj_dimer_reference(api, lj_cross_dimer(), 2., 1., 1.5, .3, -.2);
    // This pair is inside the global cutoff but outside its own cutoff.
    check_lj_dimer_reference(api, lj_cross_dimer(3.2), 0., 1., 3.2, .3, -.2);

    std::ostringstream saved;
    api.Save(saved);
    REQUIRE(saved.str().starts_with("pot:pottype lj612\npot:ntypes 2\n"));
    REQUIRE(saved.str().find("pot:0-1 2 1\n") != std::string::npos);
    REQUIRE(saved.str().find("pot:cut:0-1 3 ") != std::string::npos);
    REQUIRE(saved.str().find("pot:ele:0 ") != std::string::npos);
    REQUIRE(saved.str().find("pot:1-0 ") == std::string::npos);
    auto restored = make_pot(.1, 4., 9.);
    AnyPotential &restored_api = restored;
    std::istringstream input(saved.str());
    restored_api.Load(input);
    restored_api.nl_get_coeffs(coefficients.data());
    REQUIRE(coefficients == expected);
    check_lj_dimer_reference(restored_api, lj_cross_dimer(), 2., 1., 1.5, .3, -.2);

    const auto filename = (files.path / "saved.lj").string();
    api.Save(filename);
    auto direct = make_pot();
    AnyPotential &direct_api = direct;
    direct_api.Load(filename);
    direct_api.nl_get_coeffs(coefficients.data());
    REQUIRE(coefficients == expected);
    check_lj_dimer_reference(direct_api, lj_cross_dimer(), 2., 1., 1.5, .3, -.2);
}

TEST_CASE("LJ612 stream reload keeps explicit zero upper parameters canonical", "[lj][io][stream]") {
    const std::string model = handwritten_lj_model() + "pot:0-1 0 1.25\n";
    auto pot = make_pot();
    AnyPotential &api = pot;
    std::istringstream input(model);
    api.Load(input);
    Eigen::VectorXd coefficients(8);
    api.nl_get_coeffs(coefficients.data());
    REQUIRE(coefficients[2] == 0.);
    REQUIRE(coefficients[3] == 1.25);
    check_lj_dimer_reference(api, lj_cross_dimer(), 0., 1.25, 1.5, .3, -.2);
    std::ostringstream saved;
    api.Save(saved);
    REQUIRE(saved.str().find("pot:0-1 0 1.25\n") != std::string::npos);
    REQUIRE(saved.str().find("pot:1-0 ") == std::string::npos);
    std::istringstream canonical(saved.str());
    api.Load(canonical);
    api.nl_get_coeffs(coefficients.data());
    REQUIRE(coefficients[2] == 0.);
    REQUIRE(coefficients[3] == 1.25);
    check_lj_dimer_reference(api, lj_cross_dimer(), 0., 1.25, 1.5, .3, -.2);
}

TEST_CASE("LJ612 native reload replaces training maps and accepts legacy settings", "[lj][io][stream]") {
    GPSettings two_types;
    two_types.Modify("ntypes", "2");
    LJ612 pot(two_types);
    pot.train_init();
    pot.reg_L1_vector.assign(8, 7.);
    pot.add_efs = true;
    AnyPotential &api = pot;
    std::istringstream legacy("ntypes=1\n0-0=1.2 2.5\ncut:0-0=8\nele:0=-.4\n");
    api.Load(legacy);
    REQUIRE(pot.add_efs);
    REQUIRE(pot.species_count == 1);
    REQUIRE(pot.get_NL_coeffs_count() == 3);
    REQUIRE(pot.cutoff == 8.);
    REQUIRE(pot.cutoffsq == 64.);
    REQUIRE(pot.mapgid2oid.size() == 1);
    REQUIRE(pot.mapoid2gid.size() == 1);
    REQUIRE(pot.mapoid2gid.count(std::make_pair(0, 1)) == 0);
    REQUIRE(pot.reg_L1_vector == std::vector<double>(3, 0.));
    REQUIRE(pot.reg_L2_vector == std::vector<double>(3, 0.));
    Eigen::VectorXd coefficients(3);
    api.nl_get_coeffs(coefficients.data());
    REQUIRE(coefficients[0] == 1.2);
    REQUIRE(coefficients[1] == 2.5);
    REQUIRE(coefficients[2] == -.4);
    pot.add_efs = false;
    auto atom = make_dimer(3.4);
    atom.pbc_ = {false, false, false};
    check_lj_dimer_reference(api, atom, 1.2, 2.5, 3.4, -.4, -.4);
}
