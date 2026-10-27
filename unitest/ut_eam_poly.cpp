// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_eam_poly.h"
#include "potentials/potential_eamfs_spline.h"
#include <filesystem>
#include "base/neighborhood.h"
#include "analyse/compute_hessian.h"
#include "apps/apps.h"
#include "io/gpio.h"
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>

using Catch::Approx;
using namespace Potentials;

// Helper: create a simple 1-element, 2-atom test structure
static Atoms make_test_atom(double a_lat) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * a_lat;
    // 2 atoms along x-axis
    atom.pos_ = {
        {0.0, 0.0, 0.0},
        {a_lat / 2.0, 0.0, 0.0}
    };
    atom.chemmap_ = {0, 0};
    return atom;
}

// Helper: build a simple 1-element EAM potential with known params
static PotentialEAMPoly make_test_pot() {
    // npoly_rho=4, npoly_f=3, npoly_pair=4, nelems=1, basis=2
    Eigen::MatrixXd rmax_p = Eigen::MatrixXd::Constant(1, 1, 6.0);
    Eigen::MatrixXd rmax_r = Eigen::MatrixXd::Constant(1, 1, 5.0);
    PotentialEAMPoly pot(4, 3, 4, 1, rmax_p, rmax_r, 2);
    return pot;
}

TEST_CASE("PotentialEAMPoly construction", "[eam_poly]") {
    auto pot = make_test_pot();

    // Verify coefficient layout
    REQUIRE(pot.nelems == 1);
    REQUIRE(pot.npoly_rho == 4);
    REQUIRE(pot.npoly_f == 3);
    REQUIRE(pot.npoly_pair == 4);

    // ncoeffs: density(4) + embedding(3) + pair(4) + species(1) = 12
    REQUIRE(pot.get_NL_coeffs_count() == 12);

    // Verify basis objects were created
    REQUIRE(pot.basis_rho.size() == 1);
    REQUIRE(pot.basis_pair.size() == 1);
    REQUIRE(pot.basis_rho[0] != nullptr);
    REQUIRE(pot.basis_pair[0] != nullptr);

    // Verify cutoffs
    REQUIRE(pot.cutoffs(0, 0) == Approx(6.0));  // max(pair=6, rho=5)
    REQUIRE(pot.species_count == 1);
}

TEST_CASE("PotentialEAMPoly zero coefficients gives zero energy", "[eam_poly]") {
    auto pot = make_test_pot();
    auto atom = make_test_atom(4.0);
    Neighborhoods nbhs(atom, 6.0);

    // All coeffs are zero (default)
    atom.resetEFS();
    pot.getEFS(atom, nbhs);

    REQUIRE(atom.energy_ == Approx(0.0).epsilon(1e-12));
}

TEST_CASE("PotentialEAMPoly species energy", "[eam_poly]") {
    auto pot = make_test_pot();
    // Set only the species coefficient (last coeff in layout)
    pot.regression_coeffs[pot.n_coeffs - 1] = 1.5;  // species_coeffs[0]
    pot.nl_set_coeffs(pot.regression_coeffs.data());

    auto atom = make_test_atom(4.0);
    Neighborhoods nbhs(atom, 6.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);

    // 2 atoms, each contributes 1.5 eV from species energy
    REQUIRE(atom.energy_ == Approx(3.0).epsilon(1e-10));
}

TEST_CASE("PotentialEAMPoly pair function manual", "[eam_poly]") {
    // 1-element, 4 basis functions for pair
    Eigen::MatrixXd rmax_p = Eigen::MatrixXd::Constant(1, 1, 6.0);
    Eigen::MatrixXd rmax_r = Eigen::MatrixXd::Constant(1, 1, 5.0);
    PotentialEAMPoly pot(4, 3, 4, 1, rmax_p, rmax_r, 2);

    // Set pair_coeff[0] = 2.0, all others = 0
    // pair_coeffs start at offset: n_density_coeffs(4) + n_embedding_coeffs(3) = 7
    pot.regression_coeffs[7] = 2.0;
    pot.nl_set_coeffs(pot.regression_coeffs.data());

    // Create 2 atoms at distance r = 2.0
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 10.0;
    atom.pos_ = {{0, 0, 0}, {2.0, 0, 0}};
    atom.chemmap_ = {0, 0};
    Neighborhoods nbhs(atom, 6.0);

    atom.resetEFS();
    pot.getEFS(atom, nbhs);

    // Manual: normalized-quadratic configurable basis_0(r):
    // x = r / rmax = 2.0 / 6.0 = 1/3
    // (1-x)^2 = (2/3)^2 = 4/9
    // ksi = 2*x - 1 = 2/3 - 1 = -1/3
    // Q_0(r) = T_0(ksi) * (1-x)^2 = 1 * 4/9
    // Q_1(r) = T_1(ksi) * (1-x)^2 = (-1/3) * 4/9
    // Q_2(r) = T_2(ksi) * (1-x)^2 = (2*(-1/3)^2 - 1) * 4/9 = (2/9 - 1) * 4/9 = (-7/9) * 4/9
    // Q_3(r) = T_3(ksi) * (1-x)^2 = (4*(-1/3)^3 - 3*(-1/3)) * 4/9 = (-4/27 + 1) * 4/9 = (23/27) * 4/9

    // Only coeff[0] = 2.0, so V(r) = 2.0 * Q_0 = 2.0 * 4/9 = 8/9
    // U = 0.5 * V(r) per pair, 2 atoms: each sees 1 neighbor
    // total U = 2 * 0.5 * 8/9 = 8/9 ≈ 0.888...
    double expected = 8.0 / 9.0;
    REQUIRE(atom.energy_ == Approx(expected).epsilon(1e-10));
}

TEST_CASE("PotentialEAMPoly force finite difference", "[eam_poly]") {
    // Full nontrivial coefficients
    Eigen::MatrixXd rmax_p = Eigen::MatrixXd::Constant(1, 1, 6.0);
    Eigen::MatrixXd rmax_r = Eigen::MatrixXd::Constant(1, 1, 5.0);
    PotentialEAMPoly pot(4, 3, 4, 1, rmax_p, rmax_r, 2);

    // Set all coefficients to nonzero values
    for (int i = 0; i < pot.n_coeffs; ++i) {
        pot.regression_coeffs[i] = 0.1 * (i + 1);
    }
    pot.nl_set_coeffs(pot.regression_coeffs.data());

    // 3-atom setup for nontrivial forces
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 12.0;
    atom.pos_ = {
        {0.0, 0.0, 0.0},
        {1.8, 0.0, 0.0},
        {0.5, 2.0, 0.0}
    };
    atom.chemmap_ = {0, 0, 0};
    Neighborhoods nbhs(atom, 6.0);

    // Compute analytical forces
    atom.resetEFS();
    pot.getEFS(atom, nbhs);
    auto analytical_forces = atom.forces_;

    // Finite difference
    const double dx = 1e-6;
    for (int i = 0; i < atom.size(); ++i) {
        for (int a = 0; a < 3; ++a) {
            // +dx
            atom.pos_[i][a] += dx;
            nbhs.init(atom, 6.0);
            atom.resetEFS();
            pot.getEFS(atom, nbhs);
            double e_plus = atom.energy_;
            // -dx
            atom.pos_[i][a] -= 2 * dx;
            nbhs.init(atom, 6.0);
            atom.resetEFS();
            pot.getEFS(atom, nbhs);
            double e_minus = atom.energy_;
            // restore
            atom.pos_[i][a] += dx;

            double fd_force = -(e_plus - e_minus) / (2 * dx);
            INFO("Atom " << i << " component " << a);
            REQUIRE(analytical_forces[i][a] == Approx(fd_force).epsilon(1e-4).margin(1e-8));
        }
    }
}

TEST_CASE("PotentialEAMPoly save load roundtrip", "[eam_poly]") {
    auto pot = make_test_pot();
    for (int i = 0; i < pot.n_coeffs; ++i) {
        pot.regression_coeffs[i] = 0.5 * (i - 3) * (i - 3);
    }
    pot.nl_set_coeffs(pot.regression_coeffs.data());

    // Save
    pot.Save("/tmp/test_eam_poly.pot");

    // Load into new potential (provide dummy constructor params — they get overwritten)
    PotentialEAMPoly pot2(2, 2, 2, 1,
        Eigen::MatrixXd::Constant(1, 1, 1.0),
        Eigen::MatrixXd::Constant(1, 1, 1.0), 2);
    pot2.Load("/tmp/test_eam_poly.pot");

    // Verify coefficients match
    REQUIRE(pot2.get_NL_coeffs_count() == pot.get_NL_coeffs_count());
    REQUIRE(pot2.npoly_rho == pot.npoly_rho);
    REQUIRE(pot2.npoly_pair == pot.npoly_pair);
    REQUIRE(pot2.nelems == pot.nelems);
    REQUIRE(pot2.basis_poly == pot.basis_poly);
    std::vector<double> c1(pot.n_coeffs), c2(pot2.n_coeffs);
    pot.nl_get_coeffs(c1.data());
    pot2.nl_get_coeffs(c2.data());
    for (int i = 0; i < pot.n_coeffs; ++i) {
        REQUIRE(c2[i] == Approx(c1[i]).epsilon(1e-12));
    }

    // Verify energy matches
    auto atom = make_test_atom(4.0);
    Neighborhoods nbhs1(atom, 6.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs1);
    double e1 = atom.energy_;

    Neighborhoods nbhs2(atom, 6.0);
    atom.resetEFS();
    pot2.getEFS(atom, nbhs2);
    double e2 = atom.energy_;

    REQUIRE(e2 == Approx(e1).epsilon(1e-12));
}

TEST_CASE("PotentialEAMPoly load template without coeffs", "[eam_poly]") {
    // Write a template file without coeffs tag
    {
        std::ofstream ofs("/tmp/test_eam_template.pot");
        ofs << "# Template EAM potential\n";
        ofs << "nelems       1\n";
        ofs << "npoly_rho    3\n";
        ofs << "npoly_f      3\n";
        ofs << "npoly_pair   3\n";
        ofs << "basis_poly   2\n";
        ofs << "b            -1.3333333\n";
        ofs << "rmax_pair    5.5\n";
        ofs << "rmax_rho     4.5\n";
        ofs.close();
    }

    PotentialEAMPoly pot(2, 2, 2, 1,
        Eigen::MatrixXd::Constant(1, 1, 1.0),
        Eigen::MatrixXd::Constant(1, 1, 1.0), 2);
    pot.Load("/tmp/test_eam_template.pot");

    // Verify parameters were loaded
    REQUIRE(pot.nelems == 1);
    REQUIRE(pot.npoly_rho == 3);
    REQUIRE(pot.npoly_f == 3);
    REQUIRE(pot.npoly_pair == 3);
    REQUIRE(pot.basis_poly == 2);
    REQUIRE(pot.rmax_pair(0, 0) == Approx(5.5));
    REQUIRE(pot.rmax_rho(0, 0) == Approx(4.5));

    // Coefficients should be zero-initialized (template mode)
    std::vector<double> coeffs(pot.n_coeffs);
    pot.nl_get_coeffs(coeffs.data());
    for (double c : coeffs) {
        REQUIRE(c == Approx(0.0).margin(1e-15));
    }

    // Should produce zero energy
    auto atom = make_test_atom(4.0);
    Neighborhoods nbhs(atom, 6.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);
    REQUIRE(atom.energy_ == Approx(0.0).margin(1e-15));
}

TEST_CASE("PotentialEAMPoly AccumulateFPGard gradient", "[eam_poly]") {
    // Simple case: npoly_f=1 (linear embedding) for easy manual verification
    // Coeff layout: density[2], embedding[1], pair[2], species[1] = 6 coeffs
    Eigen::MatrixXd rmax_p = Eigen::MatrixXd::Constant(1, 1, 6.0);
    Eigen::MatrixXd rmax_r = Eigen::MatrixXd::Constant(1, 1, 5.0);
    PotentialEAMPoly pot(2, 1, 2, 1, rmax_p, rmax_r, 2);
    REQUIRE(pot.n_coeffs == 6);

    // Set nontrivial coefficients
    pot.regression_coeffs = {0.5, -0.3, 0.2, 1.0, -0.5, 0.1};
    pot.nl_set_coeffs(pot.regression_coeffs.data());
    pot.train_init();

    // 2-atom setup: atom 0 at origin, atom 1 at distance 1.5
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 10.0;
    atom.pos_ = {{0, 0, 0}, {1.5, 0, 0}};
    atom.chemmap_ = {0, 0};
    Neighborhoods nbhs(atom, 6.0);

    // Compute analytical gradient (energy-only mode, sum over ALL sites)
    // AccumulateFPGard computes per-site gradients; FD computes total energy gradient.
    std::vector<double> grad(pot.n_coeffs, 0.0);
    for (int ii = 0; ii < nbhs.size(); ++ii) {
        pot.AccumulateFPGard(nbhs[ii], grad.data(), 1.0, nullptr);
    }

    // Finite difference verification (total energy)
    const double dc = 1e-6;
    for (int k = 0; k < pot.n_coeffs; ++k) {
        std::vector<double> c(pot.n_coeffs);
        pot.nl_get_coeffs(c.data());

        c[k] += dc;
        pot.nl_set_coeffs(c.data());
        atom.resetEFS();
        pot.getEFS(atom, nbhs);
        double ep = atom.energy_;

        c[k] -= 2 * dc;
        pot.nl_set_coeffs(c.data());
        atom.resetEFS();
        pot.getEFS(atom, nbhs);
        double em = atom.energy_;

        // Restore original coefficients
        pot.nl_set_coeffs(pot.regression_coeffs.data());

        double fd = (ep - em) / (2 * dc);
        INFO("Coeff " << k << ": analytical=" << grad[k] << " fd=" << fd);
        REQUIRE(grad[k] == Approx(fd).margin(1e-5));
    }
}

TEST_CASE("PotentialEAMPoly OutputLAMMPSset", "[eam_poly]") {
    auto pot = make_test_pot();
    // Set nontrivial coefficients
    pot.regression_coeffs = {0.5, -0.3, 0.1, -0.1, 0.2, 0.2, -0.1, 1.0, -0.5, 0.3, -0.2, 0.1};
    pot.nl_set_coeffs(pot.regression_coeffs.data());

    // Write LAMMPS setfl HE file with explicit rho range including negative
    pot.OutputLAMMPSset("/tmp/test_eam_poly.eam", 100, 50, -2.0, 8.0);

    // Verify file exists and has reasonable content
    std::ifstream ifs("/tmp/test_eam_poly.eam");
    REQUIRE(ifs.is_open());

    std::string line;
    std::getline(ifs, line); // comment 1
    std::getline(ifs, line); // comment 2
    std::getline(ifs, line); // comment 3

    // 4th line: nelems element_symbols
    std::getline(ifs, line);
    REQUIRE(line.find("1") != std::string::npos);

    // HE format 5th line: nrho drho nr dr cutoff rhomax
    std::getline(ifs, line);
    std::istringstream iss(line);
    int nrho, nr;
    double drho, dr, rcut, rhomax_val;
    iss >> nrho >> drho >> nr >> dr >> rcut >> rhomax_val;
    REQUIRE(nrho == 50);
    REQUIRE(nr == 100);
    REQUIRE(rhomax_val == Approx(8.0));

    // HE format: drho = (rhomax - rhomin) / (nrho - 1)
    double drho_expected = (8.0 - (-2.0)) / 49.0;
    REQUIRE(drho == Approx(drho_expected).epsilon(1e-4));

    // Skip element header line (atomic number, mass, lattice constant, lattice type)
    std::getline(ifs, line);

    // Read all numerical values
    std::vector<double> values;
    double v;
    while (ifs >> v) values.push_back(v);
    REQUIRE(values.size() > 0);

    // Check that at least some values are non-zero
    double max_abs = 0;
    for (double x : values) {
        if (std::abs(x) > max_abs) max_abs = std::abs(x);
    }
    REQUIRE(max_abs > 0.0);

    // Verify embedding values at negative rho are computed correctly
    // First nrho values = F(rho) for rho from -2.0 to 8.0
    double rho_start = -2.0;
    double rho_step = drho_expected;
    double f0 = pot.eval_embed(0, rho_start) + 0.1; // species energy is part of exported F
    REQUIRE(values[0] == Approx(f0).epsilon(1e-10));
}

TEST_CASE("PotentialEAMPoly construct from GPSettings", "[eam_poly]") {
    GPSettings settings;
    settings.Modify("nelems",      "1");
    settings.Modify("npoly_rho",   "3");
    settings.Modify("npoly_f",     "2");
    settings.Modify("npoly_pair",  "3");
    settings.Modify("basis_poly",  "2");
    settings.Modify("b",           "-1.3333333");
    settings.Modify("rmax_pair_0_0", "5.5");
    settings.Modify("rmax_rho_0_0",  "4.5");
    settings.Modify("coeffs",      "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9");

    PotentialEAMPoly pot(settings);

    REQUIRE(pot.nelems == 1);
    REQUIRE(pot.npoly_rho == 3);
    REQUIRE(pot.npoly_f == 2);
    REQUIRE(pot.npoly_pair == 3);
    REQUIRE(pot.rmax_pair(0, 0) == Approx(5.5));
    REQUIRE(pot.rmax_rho(0, 0) == Approx(4.5));
    REQUIRE(pot.regression_coeffs[0] == Approx(0.1));

    // Verify functional
    auto atom = make_test_atom(4.0);
    Neighborhoods nbhs(atom, 6.0);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);
    // Should produce nontrivial energy
    REQUIRE(std::abs(atom.energy_) > 1e-10);
}

TEST_CASE("PotentialEAMPoly construct from file path", "[eam_poly]") {
    // Save a potential first
    auto pot = make_test_pot();
    pot.regression_coeffs = {0.5, -0.3, 0.1, -0.1, 0.2, 0.2, -0.1, 1.0, -0.5, 0.3, -0.2, 0.1};
    pot.nl_set_coeffs(pot.regression_coeffs.data());
    pot.Save("/tmp/test_eam_ctor.pot");

    // Construct from file path
    PotentialEAMPoly pot2("/tmp/test_eam_ctor.pot");

    REQUIRE(pot2.nelems == pot.nelems);
    REQUIRE(pot2.npoly_rho == pot.npoly_rho);
    std::vector<double> c1(pot.n_coeffs), c2(pot2.n_coeffs);
    pot.nl_get_coeffs(c1.data());
    pot2.nl_get_coeffs(c2.data());
    for (int i = 0; i < pot.n_coeffs; ++i) {
        REQUIRE(c2[i] == Approx(c1[i]).epsilon(1e-12));
    }
}

namespace {
struct CaptureEAMWarnings {
    std::ostringstream output;
    std::streambuf *previous = std::cerr.rdbuf(output.rdbuf());
    ~CaptureEAMWarnings() { std::cerr.rdbuf(previous); }
};
}

TEST_CASE("EAMPoly initializes after reading all tags and default cutoffs", "[eam_poly][io]") {
    const bool explicit_cutoffs = GENERATE(false, true);
    {
        std::ofstream out("eam-deferred.pot");
        out << "coeffs 0 0\n# split coefficient list\n0 1 0 0\n";
        if (explicit_cutoffs) out << "rmax_pair 3\nrmax_rho 2\n";
        out << "nelems 1\nnpoly_rho 2\nnpoly_f 1\nnpoly_pair 2\nbasis_poly 2\n";
    }
    CaptureEAMWarnings warnings;
    PotentialEAMPoly pot("eam-deferred.pot");
    REQUIRE(warnings.output.str().empty());
    const double rc = explicit_cutoffs ? 3.0 : 6.0;
    REQUIRE(pot.cutoff == Approx(rc));
    REQUIRE(pot.rmax_rho(0, 0) == Approx(explicit_cutoffs ? 2.0 : 5.0));
    auto atom = make_test_atom(20.0);
    atom.pbc_ = {false, false, false};
    atom.pos_[1] = {1.0, 0.0, 0.0};
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);
    REQUIRE(atom.energy_ == Approx(std::pow(1.0 - 1.0 / rc, 2)));
}

TEST_CASE("EAMPoly partial coefficients warn and zero missing values on reload", "[eam_poly][io]") {
    const int supplied = GENERATE(-1, 0, 1, 6); // absent, empty, partial, complete
    {
        std::ofstream out("eam-partial.pot");
        out << "nelems 1\nnpoly_rho 2\nnpoly_f 1\nnpoly_pair 2\n";
        if (supplied >= 0) {
            out << "coeffs ";
            for (int i = 0; i < supplied; ++i) out << i + 1 << " ";
            out << "\nrho_max 12\n";
        }
    }
    PotentialEAMPoly pot(2, 1, 2, 1, Eigen::MatrixXd::Constant(1, 1, 3.0),
                         Eigen::MatrixXd::Constant(1, 1, 2.0));
    std::fill(pot.regression_coeffs.begin(), pot.regression_coeffs.end(), 9.0);
    pot.rho_min = 4;
    pot.rho_max = 20;
    CaptureEAMWarnings warnings;
    pot.Load("eam-partial.pot");
    REQUIRE(pot.cutoff == Approx(6.0));
    REQUIRE(pot.rmax_rho(0, 0) == Approx(5.0));
    REQUIRE(pot.rho_min == Approx(0.0));
    REQUIRE(pot.rho_max == Approx(supplied >= 0 ? 12.0 : 0.0));
    for (int i = 0; i < pot.n_coeffs; ++i)
        REQUIRE(pot.regression_coeffs[i] == Approx(i < supplied ? i + 1.0 : 0.0));
    if (supplied >= 0 && supplied < 6) {
        REQUIRE(warnings.output.str().find("read " + std::to_string(supplied) + " of 6") != std::string::npos);
        REQUIRE(warnings.output.str().find("initialized to zero") != std::string::npos);
        REQUIRE(warnings.output.str().find("eam-partial.pot") != std::string::npos);
    } else REQUIRE(warnings.output.str().empty());
}

TEST_CASE("EAMPoly reload resizes species matrices and accepts saved matrix rows", "[eam_poly][io]") {
    Eigen::MatrixXd pair(2, 2), density(2, 2);
    pair << 3, 4, 4, 5;
    density << 2, 3, 4, 2;
    PotentialEAMPoly original(2, 1, 2, 2, pair, density, 2);
    for (int i = 0; i < original.n_coeffs; ++i) original.regression_coeffs[i] = 0.02 * (i + 1);
    original.Save("eam-multiple.pot");
    auto restored = make_test_pot();
    restored.Load("eam-multiple.pot");
    REQUIRE(restored.species_count == 2);
    REQUIRE((restored.rmax_pair - pair).norm() < 1e-12);
    REQUIRE((restored.rmax_rho - density).norm() < 1e-12);
    auto atom = make_test_atom(20);
    atom.pbc_ = {false, false, false};
    atom.pos_[1] = {1.3, 0, 0};
    atom.chemmap_[1] = 1;
    Neighborhoods nbhs(atom, original.cutoff);
    original.getEFS(atom, nbhs);
    auto copy = atom;
    restored.getEFS(copy, nbhs);
    REQUIRE(copy.energy_ == Approx(atom.energy_).margin(1e-12));
    REQUIRE((copy.forces_[0] - atom.forces_[0]).norm() < 1e-12);
    REQUIRE((copy.stress_ - atom.stress_).norm() < 1e-12);
    {
        std::ofstream out("eam-legacy-rows.pot");
        out << "nelems 2\nnpoly_rho 2\nnpoly_f 1\nnpoly_pair 2\n"
               "rmax_pair 3 4\nrmax_pair 4 5\nrmax_rho 2 3\nrmax_rho 4 2\n";
    }
    restored.Load("eam-legacy-rows.pot");
    REQUIRE((restored.rmax_pair - pair).norm() < 1e-12);
    REQUIRE((restored.rmax_rho - density).norm() < 1e-12);
}

TEST_CASE("EAMPoly accepts tags on one line and shrinks matrices on reload", "[eam_poly][io]") {
    const auto matrix = Eigen::MatrixXd::Constant(2, 2, 3.0).eval();
    PotentialEAMPoly pot(2, 1, 2, 2, matrix, matrix);
    {
        std::ofstream out("eam-one-line.pot");
        out << "coeffs 0 0 0 0 0 1 npoly_rho 2 npoly_f 1 npoly_pair 2 nelems 1";
    }
    pot.Load("eam-one-line.pot");
    REQUIRE(pot.species_count == 1);
    REQUIRE(pot.cutoff == Approx(6.0));
    REQUIRE(pot.rmax_pair.rows() == 1);
    REQUIRE(pot.rmax_rho.rows() == 1);
    REQUIRE(pot.basis_pair.size() == 1);
    REQUIRE(pot.basis_rho.size() == 1);
    auto atom = make_test_atom(20);
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);
    REQUIRE(atom.energy_ == Approx(2.0));
}

TEST_CASE("EAMPoly multi-neighbor parameter gradients match energy force and virial", "[eam_poly][gradient]") {
    const int target = GENERATE(0, 1, 2, 3);
    const auto matrix = Eigen::MatrixXd::Constant(1, 1, 5).eval();
    PotentialEAMPoly pot(3, 2, 2, 1, matrix, matrix, 2);
    std::vector<double> coeffs{0.5, 0.2, -0.1, 0.4, 0.2, 0.3, -0.2, 0.1};
    pot.nl_set_coeffs(coeffs.data());
    auto atom = make_test_atom(20);
    atom.pbc_ = {false, false, false};
    atom.pos_ = {{0, 0, 0}, {1.3, 0.1, 0}, {0.4, 2.0, 0.2}};
    atom.chemmap_ = {0, 0, 0};
    Neighborhoods nbhs(atom, pot.cutoff);
    const double ew = target == 0 || target == 3 ? 0.7 : 0.0;
    Eigen::Matrix<double, -1, 3> fw(3, 3);
    fw << 0.2, 0.3, -0.4, 0.8, -0.1, 0.2, -0.1, 0.7, 0.3;
    Eigen::Matrix3d sw;
    sw << 0.1, 0.2, -0.3, 0.2, 0.5, 0.4, -0.3, 0.4, -0.1;
    if (target != 1 && target != 3) fw.setZero();
    if (target != 2 && target != 3) sw.setZero();
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buffer;
    std::vector<double> gradient(coeffs.size(), 0.0);
    pot.AccumulateEFSCombinationGard(atom, nbhs, ew, fw, sw, buffer, gradient.data());
    auto objective = [&] {
        pot.nl_set_coeffs(coeffs.data());
        pot.getEFS(atom, nbhs);
        double value = ew * atom.energy_ + (sw.array() * atom.stress_.array()).sum();
        for (int i = 0; i < atom.size(); ++i) value += fw.row(i).dot(atom.forces_[i]);
        return value;
    };
    for (size_t i = 0; i < coeffs.size(); ++i) {
        CAPTURE(target, i);
        const double original = coeffs[i], h = 1e-6;
        coeffs[i] = original + h;
        const double plus = objective();
        coeffs[i] = original - h;
        const double minus = objective();
        coeffs[i] = original;
        REQUIRE(gradient[i] == Approx((plus - minus) / (2 * h)).margin(2e-8));
    }
}

TEST_CASE("EAMPoly exported HE tables preserve mixed-species EFS", "[eam_poly][io]") {
    const std::string suffix = GENERATE(std::string(""), std::string(".gz"), std::string(".zst"));
#ifndef USE_GZIP
    if (suffix == ".gz") SKIP("gzip support disabled");
#endif
#ifndef USE_ZSTD
    if (suffix == ".zst") SKIP("zstd support disabled");
#endif
    GPSettings settings;
    settings.Modify("nelems", "2");
    settings.Modify("npoly_rho", "2");
    settings.Modify("npoly_f", "2");
    settings.Modify("npoly_pair", "2");
    settings.Modify("basis_poly", "2");
    settings.Modify("coeffs", "0.2 0.05 -0.6 0.04 0.9 -0.08 0.3 0.01 0.4 0.2 -0.3 0.1 0.3 0.1 0.7 -0.2 0.4 0.03 1.5 -0.6");
    std::map<std::string, int> map{{"H", 1}, {"He", 0}};
    PotentialEAMPoly original(settings, map);
    const std::string file = "poly-roundtrip.eam" + suffix;
    original.OutputLAMMPSset(file, 601, 1001, -4, 4);
    if (suffix.empty()) {
        std::ifstream table(file);
        std::string line;
        int headers = 0;
        while (std::getline(table, line)) {
            if (line.find("fcc") == std::string::npos) continue;
            // LAMMPS reads each element's metadata as a separate line.
            std::istringstream metadata(line);
            int atomic_number;
            double mass, lattice;
            std::string structure, extra;
            REQUIRE(static_cast<bool>(metadata >> atomic_number >> mass >> lattice >> structure));
            REQUIRE(structure == "fcc");
            REQUIRE_FALSE(static_cast<bool>(metadata >> extra));
            ++headers;
        }
        REQUIRE(headers == 2);
    }
    auto filename = file;
    EAMFsSpline restored(filename, map);
    auto atom = make_test_atom(20);
    atom.pbc_ = {false, false, false};
    atom.pos_ = {{0, 0, 0}, {1.3, 0.2, 0}, {0.4, 1.8, 0.3}};
    atom.chemmap_ = {0, 1, 0};
    Neighborhoods nbhs(atom, original.cutoff);
    original.getEFS(atom, nbhs);
    for (bool omp : {false, true}) {
        auto copy = atom;
        if (omp) restored.getEFSOMP(copy, nbhs);
        else restored.getEFS(copy, nbhs);
        REQUIRE(copy.energy_ == Approx(atom.energy_).margin(2e-6));
        REQUIRE((copy.stress_ - atom.stress_).norm() < 2e-6);
        for (int i = 0; i < atom.size(); ++i) {
            REQUIRE((copy.forces_[i] - atom.forces_[i]).norm() < 2e-6);
            REQUIRE(copy.site_energies_[i] == Approx(atom.site_energies_[i]).margin(2e-6));
        }
    }
}

TEST_CASE("EAMPoly export reports invalid grids and output failures", "[eam_poly][io]") {
    auto pot = make_test_pot();
    REQUIRE_THROWS_WITH(pot.OutputLAMMPSset("missing-directory/out.eam", 10, 10), Catch::Matchers::ContainsSubstring("out.eam"));
    REQUIRE_THROWS(pot.OutputLAMMPSset("invalid-grid.eam", 1, 10));
    REQUIRE_THROWS(pot.OutputLAMMPSset("invalid-grid.eam", 10, 1));
    REQUIRE_THROWS(pot.OutputLAMMPSset("invalid-grid.eam", 10, 10, 2, 1));
    if (!std::filesystem::exists("/dev/full")) SKIP("/dev/full unavailable");
    for (const std::string suffix : {"", ".gz", ".zst"}) {
#ifndef USE_GZIP
        if (suffix == ".gz") continue;
#endif
#ifndef USE_ZSTD
        if (suffix == ".zst") continue;
#endif
        const std::string file = "full-eam-output" + suffix;
        std::filesystem::create_symlink("/dev/full", file);
        REQUIRE_THROWS_WITH(pot.OutputLAMMPSset(file, 10, 10), Catch::Matchers::ContainsSubstring(file));
        std::filesystem::remove(file);
    }
}

TEST_CASE("EAMPoly energy-only calculation preserves forces and stress", "[eam_poly]") {
    const bool add = GENERATE(false, true);
    auto pot = make_test_pot();
    pot.species_coeffs[0] = 1.5;
    pot.add_efs = add;
    auto atom = make_test_atom(20);
    atom.energy_ = 10;
    atom.site_energies_.assign(atom.size(), 5);
    atom.forces_.assign(atom.size(), Eigen::RowVector3d(1, 2, 3));
    atom.stress_.setConstant(4);
    atom.have_stress = true;
    const auto forces = atom.forces_;
    pot.getE(atom);
    REQUIRE(atom.energy_ == Approx(add ? 13 : 3));
    REQUIRE(atom.site_energies_[0] == Approx(add ? 6.5 : 1.5));
    REQUIRE(atom.forces_ == forces);
    REQUIRE(atom.stress_.isApprox(Eigen::Matrix3d::Constant(4)));
    REQUIRE_FALSE(atom.have_stress);
}


TEST_CASE("EAMPoly polymorphic streams round-trip model and EFS exactly",
          "[eam_poly][io][stream]") {
    const int basis = GENERATE(1, 2, 3, 4, 5);
    Eigen::MatrixXd pair(2, 2), density(2, 2);
    pair << 4.123456789012345, 4.876543210987654,
            4.876543210987654, 5.234567890123456;
    density << 3.987654321098765, 4.234567890123456,
               4.543210987654321, 3.876543210987654;
    PotentialEAMPoly original(3, 2, 4, 2, pair, density, basis, -1.2345678901234567);
    for (int i = 0; i < original.n_coeffs; ++i)
        original.regression_coeffs[i] = std::sin(i + 0.123456789012345) * 0.03;
    original.rho_min = -0.345678901234567;
    original.rho_max = 2.345678901234567;
    auto atoms = make_test_atom(20.0);
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{1.0, 1.0, 1.0}, {2.3, 1.7, 1.2}, {1.8, 3.1, 2.2}};
    atoms.chemmap_ = {0, 1, 0};
    AnyPotential &source = original;
    source.getEFS(atoms);

    std::ostringstream saved;
    saved << std::fixed << std::setprecision(3) << std::hex;
    const auto flags = saved.flags();
    source.Save(saved);
    REQUIRE(saved.flags() == flags);
    REQUIRE(saved.precision() == 3);
    REQUIRE(saved.str().starts_with("# EAM potential (polynomial basis)\n"));

    auto restored = make_test_pot();
    AnyPotential &target = restored;
    for (int reload = 0; reload < 2; ++reload) {
        CAPTURE(basis, reload);
        std::istringstream input(saved.str());
        target.Load(input);
        REQUIRE(restored.nelems == 2);
        REQUIRE(restored.npoly_rho == 3);
        REQUIRE(restored.npoly_f == 2);
        REQUIRE(restored.npoly_pair == 4);
        REQUIRE(restored.basis_poly == basis);
        REQUIRE(restored.b == original.b);
        REQUIRE(restored.rho_min == original.rho_min);
        REQUIRE(restored.rho_max == original.rho_max);
        REQUIRE(restored.regression_coeffs == original.regression_coeffs);
        REQUIRE((restored.rmax_pair.array() == pair.array()).all());
        REQUIRE((restored.rmax_rho.array() == density.array()).all());
        REQUIRE(restored.density_coeffs == restored.regression_coeffs.data());
        REQUIRE(restored.embedding_coeffs == restored.density_coeffs + restored.n_density_coeffs);
        REQUIRE(restored.pair_coeffs == restored.embedding_coeffs + restored.n_embeeding_coeffs);
        REQUIRE(restored.species_coeffs == restored.pair_coeffs + restored.n_pair_coeffs);
        auto actual = atoms;
        target.getEFS(actual);
        REQUIRE(actual.energy_ == atoms.energy_);
        REQUIRE(actual.site_energies_ == atoms.site_energies_);
        REQUIRE((actual.stress_.array() == atoms.stress_.array()).all());
        for (int i = 0; i < actual.size(); ++i)
            REQUIRE((actual.forces_[i].array() == atoms.forces_[i].array()).all());
    }
}

TEST_CASE("EAMPoly stream precision covers finite double extremes",
          "[eam_poly][io][stream]") {
    auto original = make_test_pot();
    original.regression_coeffs[0] = std::numeric_limits<double>::denorm_min();
    original.regression_coeffs[1] = std::numeric_limits<double>::min();
    original.regression_coeffs[2] = std::numeric_limits<double>::max();
    original.regression_coeffs[3] = -std::numeric_limits<double>::denorm_min();
    original.regression_coeffs[4] = std::nextafter(1.0, 2.0);
    original.regression_coeffs[5] = -0.0;
    std::stringstream stream;
    AnyPotential &source = original;
    source.Save(stream);
    auto restored = make_test_pot();
    AnyPotential &target = restored;
    target.Load(stream);
    REQUIRE(restored.regression_coeffs == original.regression_coeffs);
    REQUIRE(std::signbit(restored.regression_coeffs[5]));
}

TEST_CASE("EAMPoly stream templates retain partial coefficient compatibility",
          "[eam_poly][io][stream]") {
    const int supplied = GENERATE(-1, 0, 2, 6);
    std::ostringstream data;
    if (supplied >= 0) {
        data << "coeffs ";
        for (int i = 0; i < supplied; ++i) data << i + 1 << ' ';
    }
    data << "npoly_rho 2 npoly_f 1 npoly_pair 2 nelems 1";
    std::istringstream input(data.str());
    auto restored = make_test_pot();
    std::fill(restored.regression_coeffs.begin(), restored.regression_coeffs.end(), 7.0);
    CaptureEAMWarnings warnings;
    AnyPotential &target = restored;
    target.Load(input);
    REQUIRE(restored.n_coeffs == 6);
    for (int i = 0; i < restored.n_coeffs; ++i)
        REQUIRE(restored.regression_coeffs[i] == (i < supplied ? i + 1.0 : 0.0));
    if (supplied >= 0 && supplied < 6)
        REQUIRE(warnings.output.str().find("EAMPoly stream: read " + std::to_string(supplied)) != std::string::npos);
    else REQUIRE(warnings.output.str().empty());
}

TEST_CASE("EAMPoly malformed stream reload preserves the loaded model",
          "[eam_poly][io][stream]") {
    const std::string invalid = GENERATE(std::string(""), std::string("nelems 1 npoly_rho"),
        std::string("nelems 1x npoly_rho 2 npoly_f 1 npoly_pair 2"),
        std::string("nelems 2147483647 npoly_rho 2 npoly_f 1 npoly_pair 2"),
        std::string("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\nb"),
        std::string("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\nb nan"),
        std::string("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\nrho_max inf"),
        std::string("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\ncoeffs 1e"),
        std::string("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\ncoeffs nan"),
        std::string("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\ncoeffs +inf"),
        std::string("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\ncoeffs 1 2 3 4 5 6 7"),
        std::string("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\nrmax_pair"),
        std::string("nelems 2 npoly_rho 2 npoly_f 1 npoly_pair 2\nrmax_rho 3 4 5"),
        std::string("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\nrmax_pair 0"),
        std::string("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\nbasis_poly 6"));
    CAPTURE(invalid);
    auto pot = make_test_pot();
    for (int i = 0; i < pot.n_coeffs; ++i) pot.regression_coeffs[i] = 0.01 * (i + 1);
    AnyPotential &model = pot;
    auto atoms = make_test_atom(20.0);
    atoms.pbc_ = {false, false, false};
    atoms.pos_[1] = {1.5, 0.5, 0.2};
    model.getEFS(atoms);
    std::ostringstream before;
    model.Save(before);
    std::istringstream input(invalid);
    REQUIRE_THROWS_AS(model.Load(input), std::runtime_error);
    std::ostringstream after;
    model.Save(after);
    REQUIRE(after.str() == before.str());
    auto actual = atoms;
    model.getEFS(actual);
    REQUIRE(actual.energy_ == atoms.energy_);
    REQUIRE((actual.stress_.array() == atoms.stress_.array()).all());
    REQUIRE((actual.forces_[0].array() == atoms.forces_[0].array()).all());
}

namespace {
class FailingEAMInput : public std::stringbuf {
public:
    explicit FailingEAMInput(const std::string &text) : std::stringbuf(text) {}
protected:
    int_type underflow() override { throw std::runtime_error("injected input failure"); }
};
class FailingEAMOutput : public std::streambuf {
protected:
    int_type overflow(int_type) override { return traits_type::eof(); }
};
class CommaEAMNumbers : public std::numpunct<char> {
protected:
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '_'; }
    std::string do_grouping() const override { return "\3"; }
};
}

TEST_CASE("EAMPoly streams report transport failures",
          "[eam_poly][io][stream]") {
    auto pot = make_test_pot();
    AnyPotential &model = pot;
    const auto coefficients = pot.regression_coeffs;
    std::istringstream bad_input("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2");
    bad_input.setstate(std::ios::badbit);
    REQUIRE_THROWS_AS(model.Load(bad_input), std::runtime_error);
    FailingEAMInput buffer("nelems 1 npoly_rho 2 npoly_f 1 npoly_pair 2\n");
    std::istream interrupted(&buffer);
    REQUIRE_THROWS_AS(model.Load(interrupted), std::runtime_error);
    REQUIRE(pot.regression_coeffs == coefficients);
    FailingEAMOutput output_buffer;
    std::ostream output(&output_buffer);
    output << std::scientific << std::setprecision(4);
    const auto flags = output.flags();
    REQUIRE_THROWS_AS(model.Save(output), std::runtime_error);
    REQUIRE(output.flags() == flags);
    REQUIRE(output.precision() == 4);
}

TEST_CASE("EAMPoly file and stream persistence share the format",
          "[eam_poly][io][stream]") {
    std::vector<std::string> extensions = {".pot"};
#ifdef USE_GZIP
    extensions.push_back(".pot.gz");
#endif
#ifdef USE_ZSTD
    extensions.push_back(".pot.zst");
#endif
    auto original = make_test_pot();
    for (int i = 0; i < original.n_coeffs; ++i)
        original.regression_coeffs[i] = std::cos(i + 0.23456789012345) * 0.01;
    AnyPotential &source = original;
    std::ostringstream text;
    source.Save(text);
    for (const auto &extension : extensions) {
        const std::string filename = "eam-stream-roundtrip" + extension;
        source.Save(filename);
        auto file = open_ifile(filename);
        REQUIRE(file != nullptr);
        std::ostringstream saved;
        saved << file->rdbuf();
        REQUIRE(saved.str() == text.str());
        auto restored = make_test_pot();
        AnyPotential &target = restored;
        target.Load(filename);
        REQUIRE(restored.regression_coeffs == original.regression_coeffs);
        std::filesystem::remove(filename);
    }
}

TEST_CASE("EAMPoly stream load distinguishes normal EOF under exception masks",
          "[eam_poly][io][stream]") {
    const auto mask = static_cast<std::ios_base::iostate>(GENERATE(
        int(std::ios_base::failbit | std::ios_base::badbit),
        int(std::ios_base::eofbit | std::ios_base::failbit | std::ios_base::badbit)));
    const bool trailing_whitespace = GENERATE(false, true);
    auto original = make_test_pot();
    original.regression_coeffs[0] = 0.12345678901234567;
    std::ostringstream saved;
    original.Save(saved);
    std::string text = saved.str();
    if (!trailing_whitespace) text.erase(text.find_last_not_of(" \t\r\n") + 1);
    std::istringstream input(text);
    input.exceptions(mask);
    auto restored = make_test_pot();
    AnyPotential &target = restored;
    REQUIRE_NOTHROW(target.Load(input));
    REQUIRE(input.exceptions() == mask);
    REQUIRE(input.eof());
    REQUIRE_FALSE(input.bad());
    REQUIRE(restored.regression_coeffs == original.regression_coeffs);

    FailingEAMInput buffer(text);
    std::istream broken(&buffer);
    broken.exceptions(mask);
    REQUIRE_THROWS_AS(target.Load(broken), std::runtime_error);
    REQUIRE(broken.exceptions() == mask);
    REQUIRE(broken.bad());
    REQUIRE(restored.regression_coeffs == original.regression_coeffs);

    std::istringstream truncated("nelems");
    truncated.exceptions(mask);
    REQUIRE_THROWS_AS(target.Load(truncated), std::runtime_error);
    REQUIRE(truncated.exceptions() == mask);
    REQUIRE(restored.regression_coeffs == original.regression_coeffs);
}

TEST_CASE("EAMPoly stream format is independent of numeric locale",
          "[eam_poly][io][stream]") {
    const std::locale localized(std::locale::classic(), new CommaEAMNumbers);
    auto original = make_test_pot();
    original.regression_coeffs[0] = 1234.5678901234567;
    original.regression_coeffs[1] = std::numeric_limits<double>::denorm_min();
    std::ostringstream canonical, output;
    original.Save(canonical);
    output.imbue(localized);
    output << std::scientific << std::setprecision(3);
    const auto flags = output.flags();
    original.Save(output);
    REQUIRE(output.str() == canonical.str());
    REQUIRE(output.getloc() == localized);
    REQUIRE(output.flags() == flags);
    REQUIRE(output.precision() == 3);

    auto restored = make_test_pot();
    AnyPotential &target = restored;
    std::istringstream input(output.str());
    input.imbue(localized);
    target.Load(input);
    REQUIRE(restored.regression_coeffs == original.regression_coeffs);
    REQUIRE(input.getloc() == localized);

    // Preserve the legacy explicit-sign/hexadecimal syntax independently of
    // locale, including signed zero and the smallest representable subnormal.
    std::istringstream signed_numbers(
        "nelems +1 npoly_rho +2 npoly_f +1 npoly_pair +2\n"
        "coeffs +0x1.8p+1 -0x0p0 +4.9406564584124654e-324 -0.0 +1.25 -2.5\n");
    signed_numbers.imbue(localized);
    target.Load(signed_numbers);
    REQUIRE(restored.regression_coeffs[0] == 3.0);
    REQUIRE(restored.regression_coeffs[1] == 0.0);
    REQUIRE(std::signbit(restored.regression_coeffs[1]));
    REQUIRE(restored.regression_coeffs[2] == std::numeric_limits<double>::denorm_min());
    REQUIRE(std::signbit(restored.regression_coeffs[3]));
    REQUIRE(restored.regression_coeffs[4] == 1.25);
    REQUIRE(restored.regression_coeffs[5] == -2.5);

    FailingEAMOutput failing_buffer;
    std::ostream failing(&failing_buffer);
    failing.imbue(localized);
    failing.exceptions(std::ios_base::failbit | std::ios_base::badbit);
    REQUIRE_THROWS_AS(original.Save(failing), std::runtime_error);
    REQUIRE(failing.getloc() == localized);
    REQUIRE(failing.exceptions() == (std::ios_base::failbit | std::ios_base::badbit));
}

namespace {
GPSettings eampoly_hvp_settings(int basis,int radial_count=3) {
    GPSettings s;
    s.Modify("nelems","2"); s.Modify("basis_poly",std::to_string(basis));
    s.Modify("npoly_rho",std::to_string(radial_count));
    s.Modify("npoly_pair",std::to_string(radial_count)); s.Modify("npoly_f","3");
    s.Modify("rmax_pair_0_0","5.5"); s.Modify("rmax_pair_0_1","3.0");
    s.Modify("rmax_pair_1_0","3.0"); s.Modify("rmax_pair_1_1","4.3");
    s.Modify("rmax_rho_0_0","4.0"); s.Modify("rmax_rho_0_1","5.2");
    s.Modify("rmax_rho_1_0","2.7"); s.Modify("rmax_rho_1_1","5.5");
    return s;
}
void eampoly_hvp_coeffs(PotentialEAMPoly &p) {
    std::fill(p.regression_coeffs.begin(),p.regression_coeffs.end(),0.);
    for (int a=0;a<p.nelems;++a) {
        for (int b=0;b<p.nelems;++b) {
            double *d=p.density_coeffs+p.get_rhoid(a,b)*p.npoly_rho;
            d[0]=.2+.03*a-.02*b;
            if (p.npoly_rho>1) d[1]=.03*(a+1);
            if (p.npoly_rho>2) d[2]=-.015*(b+1);
        }
        p.embedding_coeffs[a*p.npoly_f]=.07*(a+1);
        p.embedding_coeffs[a*p.npoly_f+1]=.12+.03*a;
        p.embedding_coeffs[a*p.npoly_f+2]=-.01*(a+1);
        p.species_coeffs[a]=.1*a;
    }
    for (int id=0;id<p.nelems*(p.nelems+1)/2;++id) {
        double *c=p.pair_coeffs+id*p.npoly_pair;
        c[0]=.08+.02*id;
        if (p.npoly_pair>1) c[1]=-.03;
        if (p.npoly_pair>2) c[2]=.015;
    }
}
Atoms eampoly_hvp_cluster() {
    Atoms a; a.cell_<<9,0,0,.7,9.4,0,-.5,.4,9.8;
    a.pbc_={false,false,false};
    a.pos_={{.5,.7,.6},{2.8,.9,.7},{1.2,3.1,.9},{3.,3.4,2.8}};
    a.chemmap_={0,1,0,1};
    return a;
}
Eigen::VectorXd eampoly_hvp_direction(int n,double phase=0.) {
    Eigen::VectorXd v(3*n);
    for (int k=0;k<v.size();++k) v[k]=.3*std::sin(.7*k+phase)+.2*std::cos(.3*k-phase);
    return v;
}
Eigen::VectorXd eampoly_force_fd(PotentialEAMPoly &p,const Atoms &a,
    const Eigen::VectorXd &v,double h) {
    auto plus=a,minus=a;
    for (int i=0;i<a.size();++i) {
        plus.pos_[i]+=h*v.segment<3>(3*i).transpose();
        minus.pos_[i]-=h*v.segment<3>(3*i).transpose();
    }
    Neighborhoods np(plus,p.cutoff),nm(minus,p.cutoff);
    p.getEFS(plus,np); p.getEFS(minus,nm);
    Eigen::VectorXd out(v.size());
    for (int i=0;i<a.size();++i) out.segment<3>(3*i)=-(plus.forces_[i]-minus.forces_[i])/(2*h);
    return out;
}
Eigen::MatrixXd eampoly_dense(const HessianMatrix &hessian) {
    Eigen::MatrixXd out(3*hessian.size,3*hessian.size);
    for (int i=0;i<hessian.size;++i)
        for (int j=0;j<hessian.size;++j) out.block<3,3>(3*i,3*j)=hessian(i,j);
    return out;
}
Eigen::MatrixXd eampoly_dense_force_fd(PotentialEAMPoly &p,const Atoms &a,double h) {
    const int dimension=3*a.size();
    Eigen::MatrixXd out(dimension,dimension);
    for (int k=0;k<dimension;++k)
        out.col(k)=eampoly_force_fd(p,a,Eigen::VectorXd::Unit(dimension,k),h);
    return out;
}
std::vector<std::vector<double>> eampoly_basis_outputs(const PotentialEAMPoly &p) {
    std::vector<std::vector<double>> out;
    for (const auto *basis:p.basis_rho) {
        out.push_back(basis->vals); out.push_back(basis->ders); out.push_back(basis->derders);
    }
    for (const auto *basis:p.basis_pair) {
        out.push_back(basis->vals); out.push_back(basis->ders); out.push_back(basis->derders);
    }
    return out;
}
}

TEST_CASE("EAMPoly HVP has the independent polynomial embedding cross term", "[eam_poly][hvp]") {
    const Eigen::MatrixXd rc=Eigen::MatrixXd::Constant(2,2,5.5);
    PotentialEAMPoly p(1,3,1,2,rc,rc,2);
    for (int a=0;a<2;++a) {
        for (int b=0;b<2;++b) p.density_coeffs[p.get_rhoid(a,b)]=.3+.1*a+.05*b;
        p.embedding_coeffs[3*a]=.1; p.embedding_coeffs[3*a+1]=.4; p.embedding_coeffs[3*a+2]=.2;
    }
    Atoms a; a.cell_=20*Eigen::Matrix3d::Identity(); a.pbc_={false,false,false};
    const double r=3.5,q=r/std::sqrt(3.),cut=5.5,t=1-r/cut;
    a.pos_={{0,0,0},{q,q,q},{q,-q,-q},{-q,q,-q}}; a.chemmap_={0,1,0,1};
    Neighborhoods nbhs(a,p.cutoff);
    REQUIRE(nbhs[0].nbhcount_==3);
    for (int i=1;i<a.size();++i) REQUIRE(nbhs[i].nbhcount_==1);
    // For normalized quadratic basis size 1, f_ab=d_ab*(1-r/R)^2.
    const double rho=(.35+.3+.35)*t*t,second=.8+1.2*rho;
    const double f1=-2*.35*t/cut,f2=-2*.3*t/cut;
    Eigen::VectorXd direction=Eigen::VectorXd::Zero(12),product;
    direction.segment<3>(6)=a.pos_[2].transpose()/r;
    p.applyHessianVectorProduct(a,nbhs,direction,product);
    const Eigen::Vector3d expected=second*f1*f2*a.pos_[1].transpose()/r;
    REQUIRE(expected.norm()>1e-5);
    REQUIRE((product.segment<3>(3)-expected).norm()<1e-11);
}

TEST_CASE("EAMPoly HVP follows all radial bases and independent local cutoffs", "[eam_poly][hvp]") {
    const int basis=GENERATE(1,2,3,4,5);
    PotentialEAMPoly p(eampoly_hvp_settings(basis)); eampoly_hvp_coeffs(p);
    auto a=eampoly_hvp_cluster(); Neighborhoods nbhs(a,p.cutoff);
    const double density_only=(a.pos_[3]-a.pos_[0]).norm();
    REQUIRE(density_only>p.rmax_pair(0,1)); REQUIRE(density_only<p.rmax_rho(0,1));
    const double pair_only=(a.pos_[2]-a.pos_[1]).norm();
    REQUIRE(pair_only<p.rmax_pair(1,0)); REQUIRE(pair_only>p.rmax_rho(1,0));
    const auto u=eampoly_hvp_direction(a.size()),v=eampoly_hvp_direction(a.size(),.9);
    Eigen::VectorXd hu,hv;
    p.applyHessianVectorProduct(a,nbhs,u,hu);
    p.applyHessianVectorProduct(a,v,hv);
    for (double h:{1e-5,5e-6}) {
        CAPTURE(basis,h);
        REQUIRE((hu-eampoly_force_fd(p,a,u,h)).norm()<3e-6);
        if (basis==2 || basis==3) REQUIRE((hv-eampoly_force_fd(p,a,v,h)).norm()<3e-6);
    }
    REQUIRE(u.dot(hv)==Approx(v.dot(hu)).margin(1e-10));
}

TEST_CASE("EAMPoly single radial terms and zero or negative density retain valid HVPs", "[eam_poly][hvp]") {
    SECTION("Legacy single radial term reaches existing EFS and the new HVP") {
        const int basis=GENERATE(1,4,5);
        PotentialEAMPoly p(eampoly_hvp_settings(basis,1)); eampoly_hvp_coeffs(p);
        auto a=eampoly_hvp_cluster(); Neighborhoods nbhs(a,p.cutoff);
        p.getEFS(a,nbhs);
        REQUIRE(std::isfinite(a.energy_));
        for (const auto &f:a.forces_) REQUIRE(f.allFinite());
        const auto v=eampoly_hvp_direction(a.size(),.4);
        Eigen::VectorXd product; p.applyHessianVectorProduct(a,nbhs,v,product);
        REQUIRE(product.allFinite());
        for (double h:{1e-5,5e-6}) REQUIRE((product-eampoly_force_fd(p,a,v,h)).norm()<3e-6);
    }
    SECTION("Integer embedding powers at zero and negative raw density") {
        const bool negative=GENERATE(false,true);
        const Eigen::MatrixXd rc=Eigen::MatrixXd::Constant(2,2,3.5);
        PotentialEAMPoly p(1,3,1,2,rc,rc,2); eampoly_hvp_coeffs(p);
        for (int k=0;k<p.n_density_coeffs;++k) p.density_coeffs[k]=negative ? -.2 : .3;
        if (!negative) { p.density_coeffs[p.get_rhoid(0,0)]=1.; p.density_coeffs[p.get_rhoid(0,1)]=-1.; }
        Atoms a; a.cell_=10*Eigen::Matrix3d::Identity(); a.pbc_={false,false,false};
        a.pos_={{0,0,0},{2,0,0},{-2,0,0}}; a.chemmap_={0,0,1};
        const double rho=p.eval_density(0,0,2.)+p.eval_density(0,1,2.);
        if (negative) REQUIRE(rho<0.); else REQUIRE(rho==0.);
        Neighborhoods nbhs(a,p.cutoff); p.getEFS(a,nbhs);
        const auto v=eampoly_hvp_direction(a.size());
        Eigen::VectorXd product; p.applyHessianVectorProduct(a,nbhs,v,product);
        REQUIRE(product.allFinite()); REQUIRE(product.norm()>1e-5);
        for (double h:{1e-5,5e-6}) REQUIRE((product-eampoly_force_fd(p,a,v,h)).norm()<3e-6);
        HessianMatrix blocks;
        p.getEFSHessian(a,nbhs,blocks);
        const auto dense=eampoly_dense(blocks);
        REQUIRE(dense.allFinite());
        for (double h:{1e-5,5e-6}) REQUIRE((dense-eampoly_dense_force_fd(p,a,h)).norm()<3e-6);
    }
}

TEST_CASE("EAMPoly HVP is linear and leaves source and evaluation buffers unchanged", "[eam_poly][hvp]") {
    PotentialEAMPoly p(eampoly_hvp_settings(3)); eampoly_hvp_coeffs(p);
    auto a=eampoly_hvp_cluster(); Neighborhoods nbhs(a,p.cutoff); p.getEFS(a,nbhs);
    p.buff_site_energy_ders.setConstant(2,3,12.); p.buff_drhodx.setConstant(3,3,-7.);
    p.rho_min=-12.; p.rho_max=34.;
    const auto source=a; const auto coefficients=p.regression_coeffs;
    const auto basis_outputs=eampoly_basis_outputs(p);
    const auto u=eampoly_hvp_direction(a.size()),v=eampoly_hvp_direction(a.size(),.9);
    Eigen::VectorXd hu=Eigen::VectorXd::Constant(1,99.),hv,combined;
    REQUIRE(p.supports_analytical_hessian_vector_product());
    REQUIRE(p.supports_analytical_hessian()); REQUIRE(p.supports_analytical_cell_hessian());
    p.applyHessianVectorProduct(a,nbhs,u,hu);
    AnyPotential &base=p; base.applyHessianVectorProduct(a,v,hv);
    p.applyHessianVectorProduct(a,2*u-.3*v,combined);
    REQUIRE((combined-(2*hu-.3*hv)).norm()<1e-11);
    REQUIRE(u.dot(hv)==Approx(v.dot(hu)).margin(1e-11));
    Eigen::VectorXd translation=Eigen::VectorXd::Zero(3*a.size()),zero;
    for (int i=0;i<a.size();++i) translation.segment<3>(3*i)<<.7,-.2,.5;
    p.applyHessianVectorProduct(a,translation,zero); REQUIRE(zero.norm()<1e-12);
    p.add_efs=true;
    auto alias=u; p.applyHessianVectorProduct(a,nbhs,alias,alias);
    REQUIRE((alias-hu).norm()<1e-12);
    REQUIRE(a.pos_==source.pos_); REQUIRE(a.cell_==source.cell_); REQUIRE(a.cartesian==source.cartesian);
    REQUIRE(a.pbc_==source.pbc_); REQUIRE(a.chemmap_==source.chemmap_);
    REQUIRE(a.forces_==source.forces_); REQUIRE(a.stress_==source.stress_); REQUIRE(a.energy_==source.energy_);
    REQUIRE(a.have_energy==source.have_energy); REQUIRE(a.have_stress==source.have_stress);
    REQUIRE(a.site_energies_==source.site_energies_); REQUIRE(p.regression_coeffs==coefficients);
    REQUIRE(p.rho_min==-12.); REQUIRE(p.rho_max==34.);
    REQUIRE(p.buff_site_energy_ders.isApprox(Eigen::MatrixXd::Constant(2,3,12.)));
    REQUIRE(p.buff_drhodx.isApprox(Eigen::MatrixXd::Constant(3,3,-7.)));
    REQUIRE(eampoly_basis_outputs(p)==basis_outputs);
}

TEST_CASE("EAMPoly HVP retains periodic self-image density backgrounds", "[eam_poly][hvp]") {
    PotentialEAMPoly p(eampoly_hvp_settings(2)); eampoly_hvp_coeffs(p);
    Atoms a; a.cell_=Eigen::Vector3d(3.3,10,10).asDiagonal(); a.pbc_={true,false,false};
    a.pos_={{.2,.3,.4},{1.3,.9,.8}}; a.chemmap_={0,1}; Neighborhoods nbhs(a,p.cutoff);
    bool self=false,repeated=false;
    for (int i=0;i<a.size();++i) {
        int other=0;
        for (int j:nbhs[i].indices_) { self|=j==i; if (j!=i) ++other; }
        repeated|=other>1;
    }
    REQUIRE(self); REQUIRE(repeated);
    const auto v=eampoly_hvp_direction(a.size(),.4);
    Eigen::VectorXd product; p.applyHessianVectorProduct(a,nbhs,v,product);
    REQUIRE(product.norm()>1e-5);
    for (double h:{1e-5,5e-6}) REQUIRE((product-eampoly_force_fd(p,a,v,h)).norm()<3e-6);
}

TEST_CASE("EAMPoly dense Hessian has the independent polynomial cross-neighbor block", "[eam_poly][hessian]") {
    const Eigen::MatrixXd rc=Eigen::MatrixXd::Constant(2,2,5.5);
    PotentialEAMPoly p(1,3,1,2,rc,rc,2);
    for (int a=0;a<2;++a) {
        for (int b=0;b<2;++b) p.density_coeffs[p.get_rhoid(a,b)]=.3+.1*a+.05*b;
        p.embedding_coeffs[3*a]=.1; p.embedding_coeffs[3*a+1]=.4; p.embedding_coeffs[3*a+2]=.2;
    }
    Atoms a; a.cell_=20*Eigen::Matrix3d::Identity(); a.pbc_={false,false,false};
    const double r=3.5,q=r/std::sqrt(3.),cut=5.5,t=1-r/cut;
    a.pos_={{0,0,0},{q,q,q},{q,-q,-q},{-q,q,-q}}; a.chemmap_={0,1,0,1};
    Neighborhoods nbhs(a,p.cutoff);
    REQUIRE(nbhs[0].nbhcount_==3);
    for (int i=1;i<a.size();++i) REQUIRE(nbhs[i].nbhcount_==1);
    // With size-one quadratic bases, f_ab=d_ab*(1-r/R)^2.
    // Only the center site couples these two leaves: no pair or other site does.
    const double rho=(.35+.3+.35)*t*t,second=.8+1.2*rho;
    const double f1=-2*.35*t/cut,f2=-2*.3*t/cut;
    const Eigen::Vector3d u1=a.pos_[1].transpose()/r,u2=a.pos_[2].transpose()/r;
    const Eigen::Matrix3d expected=second*f1*f2*u1*u2.transpose();
    REQUIRE(expected.norm()>1e-5);
    REQUIRE((expected-expected.transpose()).norm()>1e-5);
    HessianMatrix hessian;
    AnyPotential &base=p;
    base.getEFSHessian(a,nbhs,hessian);
    REQUIRE((hessian(1,2)-expected).norm()<1e-11);
    REQUIRE((hessian(2,1)-expected.transpose()).norm()<1e-11);
}

TEST_CASE("EAMPoly direct dense Hessian follows force differences and EFS accumulation", "[eam_poly][hessian]") {
    PotentialEAMPoly p(eampoly_hvp_settings(3)); eampoly_hvp_coeffs(p);
    auto a=eampoly_hvp_cluster(); const auto geometry=a;
    const auto coefficients=p.regression_coeffs;
    Neighborhoods nbhs(a,p.cutoff);
    const double density_only=(a.pos_[3]-a.pos_[0]).norm();
    REQUIRE(density_only>p.rmax_pair(0,1)); REQUIRE(density_only<p.rmax_rho(0,1));
    const double pair_only=(a.pos_[2]-a.pos_[1]).norm();
    REQUIRE(pair_only<p.rmax_pair(1,0)); REQUIRE(pair_only>p.rmax_rho(1,0));
    auto fresh=a; p.getEFS(fresh,nbhs);
    HessianMatrix blocks; blocks.resize(a.size());
    for (auto &block:blocks.hessian) block.setConstant(99.);
    AnyPotential &base=p;
    base.getEFSHessian(a,nbhs,blocks);
    const Eigen::MatrixXd dense=eampoly_dense(blocks);
    REQUIRE((dense-dense.transpose()).norm()<1e-11);
    for (double h:{1e-5,5e-6}) {
        CAPTURE(h);
        REQUIRE((dense-eampoly_dense_force_fd(p,geometry,h)).norm()<3e-6);
    }
    for (int axis=0;axis<3;++axis) {
        Eigen::VectorXd translation=Eigen::VectorXd::Zero(3*a.size());
        for (int i=0;i<a.size();++i) translation[3*i+axis]=1.;
        REQUIRE((dense*translation).norm()<1e-11);
    }
    for (double phase:{0.,.9}) {
        const auto v=eampoly_hvp_direction(a.size(),phase);
        Eigen::VectorXd product; base.applyHessianVectorProduct(a,nbhs,v,product);
        REQUIRE((dense*v-product).norm()<1e-11);
    }
    REQUIRE(a.energy_==Approx(fresh.energy_).margin(1e-12));
    REQUIRE((a.stress_-fresh.stress_).norm()<1e-12);
    for (int i=0;i<a.size();++i) {
        REQUIRE((a.forces_[i]-fresh.forces_[i]).norm()<1e-12);
        REQUIRE(a.site_energies_[i]==Approx(fresh.site_energies_[i]).margin(1e-12));
    }
    // The no-neighborhood overload must overwrite H even with existing EFS.
    base.getEFSHessian(a,blocks);
    REQUIRE((eampoly_dense(blocks)-dense).norm()<1e-12);
    auto accumulated=fresh;
    p.add_efs=true;
    for (auto &block:blocks.hessian) block.setConstant(-17.);
    base.getEFSHessian(accumulated,nbhs,blocks);
    p.add_efs=false;
    REQUIRE((eampoly_dense(blocks)-dense).norm()<1e-12);
    REQUIRE(accumulated.energy_==Approx(2*fresh.energy_).margin(1e-12));
    REQUIRE((accumulated.stress_-2*fresh.stress_).norm()<1e-12);
    for (int i=0;i<a.size();++i) {
        REQUIRE((accumulated.forces_[i]-2*fresh.forces_[i]).norm()<1e-12);
        REQUIRE(accumulated.site_energies_[i]==Approx(2*fresh.site_energies_[i]).margin(1e-12));
    }
    REQUIRE(a.pos_==geometry.pos_); REQUIRE(a.cell_==geometry.cell_);
    REQUIRE(a.pbc_==geometry.pbc_); REQUIRE(a.cartesian==geometry.cartesian);
    REQUIRE(a.chemmap_==geometry.chemmap_); REQUIRE(p.regression_coeffs==coefficients);
}

TEST_CASE("EAMPoly dense Hessian retains periodic backgrounds and drives ComputeHessian", "[eam_poly][hessian]") {
    PotentialEAMPoly p(eampoly_hvp_settings(2)); eampoly_hvp_coeffs(p);
    Atoms a; a.cell_=Eigen::Vector3d(3.3,10,10).asDiagonal(); a.pbc_={true,false,false};
    a.pos_={{.2,.3,.4},{1.3,.9,.8}}; a.chemmap_={0,1};
    Neighborhoods nbhs(a,p.cutoff);
    bool self=false,repeated=false;
    for (int i=0;i<a.size();++i) {
        int other=0;
        for (int j:nbhs[i].indices_) { self|=j==i; if (j!=i) ++other; }
        repeated|=other>1;
    }
    REQUIRE(self); REQUIRE(repeated);
    HessianMatrix blocks;
    AnyPotential &base=p;
    base.getEFSHessian(a,nbhs,blocks);
    const Eigen::MatrixXd dense=eampoly_dense(blocks);
    REQUIRE((dense-dense.transpose()).norm()<1e-11);
    Eigen::MatrixXd fd;
    for (double h:{1e-5,5e-6}) {
        fd=eampoly_dense_force_fd(p,a,h);
        REQUIRE((dense-fd).norm()<3e-6);
    }
    const auto source=a;
    for (const std::string method:{"auto","analytic"}) {
        ComputeHessian compute;
        compute.set_method(method);
        Eigen::MatrixXd result;
        compute.compute_fc(a,result,&base);
        REQUIRE((result-fd).norm()<3e-6);
        compute.compute_fc_blocks(a,blocks,&base);
        REQUIRE((eampoly_dense(blocks)-fd).norm()<3e-6);
    }
    REQUIRE(a.pos_==source.pos_); REQUIRE(a.cell_==source.cell_);
    REQUIRE(a.pbc_==source.pbc_); REQUIRE(a.cartesian==source.cartesian);
    REQUIRE(a.chemmap_==source.chemmap_); REQUIRE(a.energy_==source.energy_);
    REQUIRE(a.forces_==source.forces_); REQUIRE(a.stress_==source.stress_);
    REQUIRE(a.site_energies_==source.site_energies_);

    Atoms single; single.cell_=a.cell_; single.pbc_=a.pbc_;
    single.pos_={a.pos_[0]}; single.chemmap_={0};
    REQUIRE(p.eval_density(0,0,3.3)>0.);
    base.getEFSHessian(single,blocks);
    REQUIRE(eampoly_dense(blocks).isZero(1e-12));
}

TEST_CASE("EAMPoly analytic HVP drives matrix-free calc_fc lowest modes", "[eam_poly][hvp][cli]") {
    struct Workspace {
        CMDOption::CommandLineOptions saved=CMDOption::global_cmdopts;
        std::filesystem::path original=std::filesystem::current_path(),path;
        int threads=omp_num_threads;
        Workspace() {
            char name[]="/tmp/imrmlp_eampoly_hvp_XXXXXX";
            const char *directory=mkdtemp(name);
            if (!directory) throw std::runtime_error("Cannot create EAMPoly HVP workspace");
            path=directory; std::filesystem::current_path(path);
            omp_num_threads=1; global_omp_init();
        }
        ~Workspace() {
            CMDOption::global_cmdopts=saved;
            omp_num_threads=threads; global_omp_init();
            std::filesystem::current_path(original); std::filesystem::remove_all(path);
        }
    } work;
    PotentialEAMPoly p(eampoly_hvp_settings(2)); eampoly_hvp_coeffs(p);
    auto a=eampoly_hvp_cluster(); p.Save("model.pot");
    std::ofstream("index.map")<<"Al 1.0 2\nNi 1.0 5\n";
    GPIO io; REQUIRE(io.readmap("index.map")); REQUIRE(io.write_one(a,"input.xyz","extxyz"));
    const int dimension=3*a.size();
    Eigen::MatrixXd fd(dimension,dimension);
    for (int k=0;k<dimension;++k) {
        const Eigen::VectorXd direction=Eigen::VectorXd::Unit(dimension,k);
        fd.col(k)=eampoly_force_fd(p,a,direction,5e-6);
    }
    REQUIRE((fd-fd.transpose()).norm()<1e-6);
    fd=(.5*(fd+fd.transpose())).eval();
    Eigen::VectorXd sqrt_mass(dimension);
    Eigen::MatrixXd translations=Eigen::MatrixXd::Zero(dimension,3);
    for (int i=0;i<a.size();++i) {
        const double root=std::sqrt(a.chemmap_[i]==0 ? 2. : 5.);
        sqrt_mass.segment<3>(3*i).setConstant(root);
        for (int axis=0;axis<3;++axis) translations(3*i+axis,axis)=root;
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(translations);
    const Eigen::MatrixXd q=qr.householderQ()*Eigen::MatrixXd::Identity(dimension,dimension);
    const Eigen::MatrixXd internal=q.rightCols(dimension-3);
    const Eigen::MatrixXd dynamical=sqrt_mass.cwiseInverse().asDiagonal()*fd*sqrt_mass.cwiseInverse().asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(internal.transpose()*dynamical*internal);
    REQUIRE(solver.info()==Eigen::Success);
    CMDOption::global_cmdopts=CMDOption::CommandLineOptions{};
    std::vector<std::string> arguments{"calc_fc","--potfile=model.pot","--potential-type=eampoly",
        "--input=input.xyz","--input-format=extxyz","--index=index.map",
        "--hessian-method=analytic","--lowest-modes=2","--lowest-mode-tol=1e-9"};
    std::vector<char *> argv;
    for (auto &argument:arguments) argv.push_back(argument.data());
    REQUIRE(calc_fc(argv.size(),argv.data())==0);
    REQUIRE_FALSE(std::filesystem::exists("FORCE_CONSTANTS"));
    std::ifstream output("EIGENVALS"); std::string header; REQUIRE(std::getline(output,header));
    for (int mode=0;mode<2;++mode) {
        double eigenvalue,frequency,energy,wavenumber;
        REQUIRE(output>>eigenvalue>>frequency>>energy>>wavenumber);
        REQUIRE(eigenvalue==Approx(solver.eigenvalues()[mode]).margin(2e-6));
    }
}
