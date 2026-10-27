// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/les.h"
#include "potentials/potential_mtp_LES.h"
#include "train/train_nonlinear_potential.h"
#include "base/neighborhood.h"
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <sstream>

using Catch::Approx;
using namespace Potentials;

// ═══════════════════════════════════════════════════════════════════════
//  Test helpers
// ═══════════════════════════════════════════════════════════════════════

/// Complex generalized periodic dipole in fractional lattice components.
static Eigen::Vector3cd compute_periodic_dipole_fractional(
    const Atoms& atom, const Eigen::VectorXd& q) {
    return LES::compute_periodic_dipole_fractional(atom,q);
}

/// Compute the complex periodic dipole after one atomic displacement.
static Eigen::Vector3cd displaced_dipole_pbc(
    const Atoms& atom, const Eigen::VectorXd& q,
    int atom_idx, int dir, double delta) {
    Atoms atom_copy = atom;
    atom_copy.pos_[atom_idx](dir) += delta;
    return compute_periodic_dipole_fractional(atom_copy, q);
}

/// Numerical periodic BEC: differentiate the complex generalized dipole,
/// remove the reference atom's phase, then transform to Cartesian axes.
static Eigen::Matrix3d numerical_bec_pbc(const Atoms& atom, const Eigen::VectorXd& q,
                                          int atom_idx, double delta = 1e-5) {
    Eigen::Matrix3d bec = Eigen::Matrix3d::Zero();
    const Eigen::RowVector3d fractional=atom.pos_[atom_idx]*atom.cell_.inverse();
    Eigen::Vector3cd phase;
    for (int alpha=0;alpha<3;++alpha)
        phase(alpha)=std::exp(std::complex<double>(
            0.0,2.0*M_PI*fractional(alpha)));
    for (int gamma = 0; gamma < 3; ++gamma) {
        const Eigen::Vector3cd derivative =
            (displaced_dipole_pbc(atom,q,atom_idx,gamma,delta)
             - displaced_dipole_pbc(atom,q,atom_idx,gamma,-delta))/(2.0*delta);
        Eigen::Vector3d dephased;
        for (int alpha=0;alpha<3;++alpha)
            dephased(alpha)=(std::conj(phase(alpha))*derivative(alpha)).real();
        bec.col(gamma)=atom.cell_.transpose()*dephased;
    }
    return bec;
}

// ═══════════════════════════════════════════════════════════════════════
//  PBC contract tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("LES BEC rejects non-periodic, partial, and singular cells",
          "[les][bec][pbc]") {
    Atoms atom;
    atom.cell_ = 10.0*Eigen::Matrix3d::Identity();
    atom.pos_ = {{1.0, 2.0, 3.0}};
    atom.chemmap_ = {0};
    Eigen::VectorXd q(1); q(0)=2.5;
    LES les;
    std::vector<Eigen::Matrix3d> bec;
    atom.pbc_={{false,false,false}};
    REQUIRE_THROWS(les.compute_bec(atom,q,bec,false,1.0));
    atom.pbc_={{true,true,false}};
    REQUIRE_THROWS(les.compute_bec(atom,q,bec,false,1.0));
    atom.pbc_={{true,true,true}};
    atom.cell_ = Eigen::Matrix3d::Zero();
    REQUIRE_THROWS(les.compute_bec(atom,q,bec,false,1.0));
    atom.cell_=10.0*Eigen::Matrix3d::Identity();
    Eigen::VectorXd wrong(2); wrong<<1.0,-1.0;
    REQUIRE_THROWS(les.compute_bec(atom,wrong,bec,false,1.0));
    Eigen::VectorXd nonfinite=q; nonfinite(0)=
        std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(les.compute_bec(atom,nonfinite,bec,false,1.0));
    REQUIRE_THROWS(les.compute_bec(atom,q,bec,false,0.0));
}

// ═══════════════════════════════════════════════════════════════════════
//  PBC BEC tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("BEC PBC single atom at origin", "[les][bec]") {
    // For a single atom at origin, r_frac = [0,0,0], θ = [0,0,0]
    // cos(θ_β − θ_α) = cos(0) = 1 for all α,β
    // BEC_α,γ = q * Σ_β cell_αβ * cell⁻¹_γβ = q * δ_αγ
    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{0.0, 0.0, 0.0}};
    atom.chemmap_ = {0};

    Eigen::VectorXd q(1);
    q(0) = 1.5;

    LES les;
    std::vector<Eigen::Matrix3d> bec;
    les.compute_bec(atom, q, bec, false, 1.0);

    REQUIRE(bec.size() == 1);
    Eigen::Matrix3d expected = 1.5 * Eigen::Matrix3d::Identity();
    // Analytical check — near-exact for atom at origin
    for (int a = 0; a < 3; ++a)
        for (int g = 0; g < 3; ++g)
            REQUIRE(bec[0](a, g) == Approx(expected(a, g)).margin(1e-11));

    // Numerical verification
    auto bec_num = numerical_bec_pbc(atom, q, 0);
    for (int a = 0; a < 3; ++a)
        for (int g = 0; g < 3; ++g)
            REQUIRE(bec[0](a, g) == Approx(bec_num(a, g)).margin(1e-11));
}

TEST_CASE("BEC PBC single atom off-center in orthogonal cell", "[les][bec]") {
    // Atom at (2.0, 3.0, 1.0) in a 10×10×10 orthogonal box
    // r_frac = [0.2, 0.3, 0.1]
    // θ = [0.4π, 0.6π, 0.2π]
    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2.0, 3.0, 1.0}};
    atom.chemmap_ = {0};

    Eigen::VectorXd q(1);
    q(0) = 2.0;

    LES les;
    std::vector<Eigen::Matrix3d> bec;
    les.compute_bec(atom, q, bec, false, 1.0);

    REQUIRE(bec.size() == 1);

    REQUIRE((bec[0]-2.0*Eigen::Matrix3d::Identity()).norm()<1e-11);

    // Numerical verification (most important for off-center PBC)
    auto bec_num = numerical_bec_pbc(atom, q, 0);
    for (int a = 0; a < 3; ++a)
        for (int g = 0; g < 3; ++g)
            REQUIRE(bec[0](a, g) == Approx(bec_num(a, g)).epsilon(1e-6));
}

TEST_CASE("BEC PBC non-orthogonal (triclinic) cell", "[les][bec]") {
    // Triclinic cell: non-orthogonal lattice vectors
    Atoms atom;
    atom.cell_ << 10.0, 0.0,  0.0,
                   2.0, 9.0,  0.0,
                   1.0, 3.0, 11.0;
    atom.pos_ = {{0.5, 0.5, 0.5}, {3.0, 4.0, 5.0}};
    atom.chemmap_ = {0, 0};

    Eigen::VectorXd q(2);
    q << 1.0, -1.0;  // net zero charge

    LES les;
    std::vector<Eigen::Matrix3d> bec;
    les.compute_bec(atom, q, bec, false, 1.0);

    REQUIRE(bec.size() == 2);

    REQUIRE((bec[0]-Eigen::Matrix3d::Identity()).norm()<1e-11);
    REQUIRE((bec[1]+Eigen::Matrix3d::Identity()).norm()<1e-11);

    // Numerical verification for each atom
    for (int i = 0; i < 2; ++i) {
        auto bec_num = numerical_bec_pbc(atom, q, i);
        for (int a = 0; a < 3; ++a)
            for (int g = 0; g < 3; ++g)
                REQUIRE(bec[i](a, g) == Approx(bec_num(a, g)).margin(2e-9));
    }
}

TEST_CASE("BEC PBC is invariant to origin and lattice-image choices",
          "[les][bec][pbc]") {
    Atoms atom;
    atom.cell_ << 10.0,0.0,0.0, 2.0,9.0,0.0, 1.0,3.0,11.0;
    atom.pos_={{1.0,2.0,3.0},{4.0,3.0,6.0}};
    atom.chemmap_={0,0};
    Eigen::VectorXd q(2); q<<1.0,-1.0;
    LES les;
    std::vector<Eigen::Matrix3d> reference,translated,reimaged;
    les.compute_bec(atom,q,reference,false,1.0);
    Atoms shifted=atom;
    for (auto& r:shifted.pos_) r+=Eigen::RowVector3d(1.7,-0.8,2.4);
    les.compute_bec(shifted,q,translated,false,1.0);
    Atoms image=atom;
    image.pos_[1]+=atom.cell_.row(0)-2.0*atom.cell_.row(1)+atom.cell_.row(2);
    les.compute_bec(image,q,reimaged,false,1.0);
    Eigen::Matrix3d sum=Eigen::Matrix3d::Zero();
    for (int i=0;i<2;++i) {
        REQUIRE((translated[i]-reference[i]).norm()<1e-11);
        REQUIRE((reimaged[i]-reference[i]).norm()<1e-11);
        sum+=reference[i];
    }
    REQUIRE(sum.norm()<1e-11);
}

TEST_CASE("BEC PBC remove_mean with net charge", "[les][bec]") {
    // System with net non-zero charge — remove_mean should center it
    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{0.0, 0.0, 0.0}, {5.0, 0.0, 0.0}};
    atom.chemmap_ = {0, 0};

    Eigen::VectorXd q(2);
    q << 3.0, 1.0;  // mean = 2.0

    LES les;
    // With remove_mean = true: q_centered = [1.0, -1.0]
    std::vector<Eigen::Matrix3d> bec;
    les.compute_bec(atom, q, bec, true, 1.0);

    REQUIRE(bec.size() == 2);

    // BEC should correspond to centered charges
    Eigen::VectorXd q_centered(2);
    q_centered << 1.0, -1.0;

    for (int i = 0; i < 2; ++i) {
        auto bec_num = numerical_bec_pbc(atom, q_centered, i);
        for (int a = 0; a < 3; ++a)
            for (int g = 0; g < 3; ++g)
                REQUIRE(bec[i](a, g) == Approx(bec_num(a, g)).epsilon(1e-6));
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  epsilon_factor scaling
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("BEC epsilon_factor scaling", "[les][bec]") {
    Atoms atom;
    atom.cell_ = 10.0*Eigen::Matrix3d::Identity();
    atom.pos_ = {{1.0, 0.0, 0.0}};
    atom.chemmap_ = {0};

    Eigen::VectorXd q(1);
    q(0) = 3.0;

    LES les;
    std::vector<Eigen::Matrix3d> bec1, bec4;
    les.compute_bec(atom, q, bec1, false, 1.0);
    les.compute_bec(atom, q, bec4, false, 4.0);

    // epsilon_factor=4.0 → scaling = sqrt(4) = 2.0
    double scale = 2.0;
    for (int a = 0; a < 3; ++a)
        for (int g = 0; g < 3; ++g) {
            REQUIRE(bec4[0](a, g) == Approx(bec1[0](a, g) * scale).epsilon(1e-12));
        }

    // Numerical verification for both
    auto bec_num = numerical_bec_pbc(atom, q, 0);
    // BEC with epsilon=4 should match num * sqrt(4)
    for (int a = 0; a < 3; ++a)
        for (int g = 0; g < 3; ++g) {
            REQUIRE(bec4[0](a, g) == Approx(bec_num(a, g) * scale).epsilon(1e-8));
            REQUIRE(bec1[0](a, g) == Approx(bec_num(a, g)).epsilon(1e-8));
        }
}

// ═══════════════════════════════════════════════════════════════════════
//  Empty / zero-charge edge cases
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
//  Ewald force/stress regression tests (numerical verification)
// ═══════════════════════════════════════════════════════════════════════

/// Compute the total Ewald energy as a function of displaced positions.
/// Used for numerical force verification.
/// Helper: zero out forces vector (std::vector<Eigen::Vector3d>)
static void zero_forces(std::vector<Eigen::Vector3d>& f) {
    for (auto& fi : f) fi.setZero();
}

/// Compute the total Ewald energy as a function of displaced positions.
/// Used for numerical force verification.
static double ewald_energy_at(const Atoms& atom, const Eigen::VectorXd& q, LES& les) {
    // We need a concrete AnyNeighborhoods subclass, even though it's unused
    Neighborhoods nbhs_dummy;

    Atoms atom_copy = atom;
    atom_copy.energy_ = 0.0;
    zero_forces(atom_copy.forces_);
    atom_copy.stress_.setZero();

    Eigen::VectorXd q_copy = q;
    les.generate_k_vectors(atom_copy, les.kvecs);
    les.add_latent_ewald_efs(atom_copy, nbhs_dummy, q_copy);
    return atom_copy.energy_;
}

TEST_CASE("Ewald forces numerical verification", "[les][ewald]") {
    // Two atoms in a 10×10×10 orthogonal box, with opposite charges
    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{3.0, 4.0, 2.0}, {7.0, 6.0, 8.0}};
    atom.forces_.assign(2, Eigen::Vector3d::Zero());
    atom.stress_ = Eigen::Matrix3d::Zero();
    atom.chemmap_ = {0, 0};

    Eigen::VectorXd q(2);
    q << 1.0, -1.0;

    LES les;
    les.lr_sigma = 0.5;
    les.dl = 2.0;

    // Compute forces via LES Ewald
    Neighborhoods nbhs;
    les.generate_k_vectors(atom, les.kvecs);
    atom.energy_ = 0.0;
    zero_forces(atom.forces_);
    atom.stress_.setZero();
    Eigen::VectorXd q_copy = q;
    les.add_latent_ewald_efs(atom, nbhs, q_copy);

    // Numerical force via central difference of energy
    const double delta = 1e-5;
    for (int i = 0; i < atom.size(); ++i) {
        for (int gamma = 0; gamma < 3; ++gamma) {
            // F_i,gamma = -dE/dr_i,gamma (numerical: central difference)
            Atoms atom_fwd = atom;
            atom_fwd.pos_[i](gamma) += delta;
            double E_plus = ewald_energy_at(atom_fwd, q, les);

            Atoms atom_bwd = atom;
            atom_bwd.pos_[i](gamma) -= delta;
            double E_minus = ewald_energy_at(atom_bwd, q, les);

            double num_force = -(E_plus - E_minus) / (2.0 * delta);
            REQUIRE(atom.forces_[i](gamma) == Approx(num_force).epsilon(1e-6));
        }
    }
}

TEST_CASE("Ewald analytic virial matches affine cell derivative", "[les][ewald][stress]") {
    Atoms atom;
    atom.cell_ << 9.5,0.0,0.0, 1.2,10.3,0.0, -0.4,0.8,11.1;
    atom.pos_ = {{3.0, 4.0, 2.0}, {7.0, 6.0, 8.0}};
    atom.forces_.assign(2, Eigen::Vector3d::Zero());
    atom.stress_ = Eigen::Matrix3d::Zero();
    atom.chemmap_ = {0, 0};

    Eigen::VectorXd q(2);
    q << 1.0, -1.0;

    LES les;
    les.lr_sigma = 0.5;
    les.dl = 2.0;

    Neighborhoods nbhs;
    les.generate_k_vectors(atom, les.kvecs);
    atom.energy_ = 0.0;
    zero_forces(atom.forces_);
    atom.stress_.setZero();
    Eigen::VectorXd q_copy = q;
    les.add_latent_ewald_efs(atom, nbhs, q_copy);

    const double delta=1e-6;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            Eigen::Matrix3d plus_transform=Eigen::Matrix3d::Identity();
            Eigen::Matrix3d minus_transform=Eigen::Matrix3d::Identity();
            plus_transform(a,b)+=delta;
            minus_transform(a,b)-=delta;
            Atoms plus=atom, minus=atom;
            plus.cell_=atom.cell_*plus_transform.transpose();
            minus.cell_=atom.cell_*minus_transform.transpose();
            for (int i=0;i<atom.size();++i) {
                plus.pos_[i]=atom.pos_[i]*plus_transform.transpose();
                minus.pos_[i]=atom.pos_[i]*minus_transform.transpose();
            }
            const double numeric=-(ewald_energy_at(plus,q,les)
                -ewald_energy_at(minus,q,les))/(2.0*delta);
            REQUIRE(atom.stress_(a,b)==Approx(numeric).margin(2e-6));
        }
}

TEST_CASE("LES charge Hessian matches dEdq finite difference", "[les][grad]") {
    Atoms atom;
    atom.cell_ << 8.0,0.0,0.0, 0.7,9.0,0.0, -0.2,0.5,10.0;
    atom.pos_={{1.1,2.0,3.0},{4.2,1.3,7.1},{2.7,6.2,5.4}};
    atom.chemmap_={0,0,0};
    Eigen::VectorXd q(3); q<<0.3,-0.4,0.7;
    LES les;
    const auto eval=les.evaluate(atom,q,true);
    const double delta=1e-6;
    for (int j=0;j<q.size();++j) {
        Eigen::VectorXd qp=q, qm=q;
        qp(j)+=delta; qm(j)-=delta;
        const Eigen::VectorXd numeric=(les.evaluate(atom,qp).dEdq
            -les.evaluate(atom,qm).dEdq)/(2.0*delta);
        for (int i=0;i<q.size();++i)
            REQUIRE(eval.charge_hessian(i,j)==Approx(numeric(i)).margin(2e-7));
    }
}

TEST_CASE("LES smooth cutoff removes hard grid jump", "[les][cutoff]") {
    auto energy_at=[](LES& les,double lx) {
        Atoms atom;
        atom.cell_<<lx,0,0, 0,1.7,0, 0,0,1.6;
        atom.pos_={{0.31,0.42,0.53}};
        atom.chemmap_={0};
        Eigen::VectorXd q(1); q<<1.0;
        return les.evaluate(atom,q).energy;
    };
    LES smooth;
    smooth.dl=2.0;
    smooth.remove_self_interaction=false;
    smooth.cutoff_mode=LES::CutoffMode::Smooth;
    const double smooth_jump=std::abs(energy_at(smooth,2.0+1e-4)
        -energy_at(smooth,2.0-1e-4));
    REQUIRE(smooth_jump<1e-6);

    LES hard=smooth;
    hard.cutoff_mode=LES::CutoffMode::ReferenceHard;
    const double hard_jump=std::abs(energy_at(hard,2.0+1e-4)
        -energy_at(hard,2.0-1e-4));
    REQUIRE(hard_jump>0.1);
}

TEST_CASE("LES hard cutoff includes the exact boundary and empty grids keep self energy",
          "[les][cutoff][self]") {
    Atoms boundary;
    boundary.cell_=2.0*Eigen::Matrix3d::Identity();
    boundary.pos_={{0.3,0.4,0.5}};
    boundary.chemmap_={0};
    LES hard;
    hard.dl=2.0;
    hard.cutoff_mode=LES::CutoffMode::ReferenceHard;
    hard.remove_self_interaction=false;
    hard.generate_k_vectors(boundary,hard.kvecs);
    REQUIRE(hard.kvecs.size()==3);

    LES smooth=hard;
    smooth.cutoff_mode=LES::CutoffMode::Smooth;
    smooth.generate_k_vectors(boundary,smooth.kvecs);
    REQUIRE(smooth.kvecs.empty());

    Atoms small=boundary;
    small.cell_=Eigen::Matrix3d::Identity();
    hard.remove_self_interaction=true;
    Eigen::VectorXd q(1); q<<0.7;
    const auto result=hard.evaluate(small,q);
    REQUIRE(hard.kvecs.empty());
    const double expected=-LES::coulomb_constant*q.squaredNorm()
        /(hard.lr_sigma*std::sqrt(2.0*M_PI));
    REQUIRE(result.energy==Approx(expected).margin(1e-12));
}

TEST_CASE("Ewald energy conservation under charge scaling", "[les][ewald]") {
    // Energy should scale as E ~ q² for fixed geometry
    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2.0, 3.0, 1.0}, {8.0, 7.0, 9.0}};
    atom.forces_.assign(2, Eigen::Vector3d::Zero());
    atom.stress_ = Eigen::Matrix3d::Zero();
    atom.chemmap_ = {0, 0};

    LES les;
    les.lr_sigma = 0.5;
    les.dl = 2.0;

    Neighborhoods nbhs;
    les.generate_k_vectors(atom, les.kvecs);

    // E with charges [1, -1]
    Eigen::VectorXd q1(2); q1 << 1.0, -1.0;
    atom.energy_ = 0.0; zero_forces(atom.forces_); atom.stress_.setZero();
    Eigen::VectorXd q1_copy = q1;
    les.add_latent_ewald_efs(atom, nbhs, q1_copy);
    double E1 = atom.energy_;

    // E with charges [2, -2] — should be 4× larger
    Eigen::VectorXd q2(2); q2 << 2.0, -2.0;
    atom.energy_ = 0.0; zero_forces(atom.forces_); atom.stress_.setZero();
    Eigen::VectorXd q2_copy = q2;
    les.add_latent_ewald_efs(atom, nbhs, q2_copy);
    double E2 = atom.energy_;

    REQUIRE(E2 == Approx(4.0 * E1).epsilon(1e-12));
}

TEST_CASE("Ewald zero charge zero force", "[les][ewald]") {
    // Zero charges should give zero energy, forces, stress
    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2.0, 3.0, 1.0}, {8.0, 7.0, 9.0}};
    atom.forces_.assign(2, Eigen::Vector3d::Zero());
    atom.stress_ = Eigen::Matrix3d::Zero();
    atom.chemmap_ = {0, 0};

    LES les;
    les.lr_sigma = 0.5;
    les.dl = 2.0;

    Neighborhoods nbhs;
    les.generate_k_vectors(atom, les.kvecs);

    Eigen::VectorXd q(2); q << 0.0, 0.0;
    atom.energy_ = 0.0; zero_forces(atom.forces_); atom.stress_.setZero();
    Eigen::VectorXd q_copy = q;
    les.add_latent_ewald_efs(atom, nbhs, q_copy);

    REQUIRE(atom.energy_ == Approx(0.0).margin(1e-15));
    for (int i = 0; i < atom.size(); ++i)
        for (int a = 0; a < 3; ++a)
            REQUIRE(atom.forces_[i](a) == Approx(0.0).margin(1e-15));
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            REQUIRE(atom.stress_(a, b) == Approx(0.0).margin(1e-15));
}

TEST_CASE("Ewald dEdq numerical verification with self correction", "[les][ewald][grad]") {
    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2.0, 3.0, 1.0}, {7.0, 5.0, 8.0}, {4.0, 1.5, 6.0}};
    atom.forces_.assign(3, Eigen::Vector3d::Zero());
    atom.stress_ = Eigen::Matrix3d::Zero();
    atom.chemmap_ = {0, 0, 0};

    Eigen::VectorXd q(3);
    q << 0.4, -0.7, 0.2;

    LES les;
    les.lr_sigma = 0.5;
    les.dl = 2.0;
    les.remove_self_interaction = true;

    Neighborhoods nbhs;
    Eigen::VectorXd dEdq;
    Atoms atom_eval = atom;
    Eigen::VectorXd q_eval = q;
    les.add_latent_ewald_efs(atom_eval, nbhs, q_eval, &dEdq);

    const double delta = 1e-6;
    for (int i = 0; i < q.size(); ++i) {
        Eigen::VectorXd q_plus = q;
        q_plus(i) += delta;
        double e_plus = ewald_energy_at(atom, q_plus, les);

        Eigen::VectorXd q_minus = q;
        q_minus(i) -= delta;
        double e_minus = ewald_energy_at(atom, q_minus, les);

        double num = (e_plus - e_minus) / (2.0 * delta);
        REQUIRE(dEdq(i) == Approx(num).epsilon(1e-6));
    }
}

TEST_CASE("PotMTP_LES indirect force matches total energy finite difference", "[les][mtp_les][grad]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les = true;
    pot.les_species_coeffs[0]=0.25;
    pot.les_species_coeffs[1]=-0.18;
    for (int a=0;a<pot.get_alpha_scalar_moments();++a)
        pot.les_moment_coeffs[a]=0.0;
    pot.les_moment_coeffs[0]=0.04;
    pot.les_moment_coeffs[1]=-0.025;
    pot.remove_self_interaction = true;
    pot.model_les.remove_self_interaction = true;

    Atoms atom;
    atom.cell_ = 12.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2.0, 2.0, 2.0}, {4.1, 2.2, 2.0}, {2.3, 4.2, 2.4}, {5.0, 4.8, 3.1}};
    atom.chemmap_ = {0, 1, 0, 1};
    atom.resetEFS();

    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);

    const double delta = 1e-5;
    for (int i = 0; i < atom.size(); ++i) {
        for (int gamma = 0; gamma < 3; ++gamma) {
            Atoms plus = atom;
            plus.pos_[i](gamma) += delta;
            Neighborhoods nbhs_plus(plus, pot.cutoff);
            plus.resetEFS();
            pot.getE(plus, nbhs_plus);

            Atoms minus = atom;
            minus.pos_[i](gamma) -= delta;
            Neighborhoods nbhs_minus(minus, pot.cutoff);
            minus.resetEFS();
            pot.getE(minus, nbhs_minus);

            double num_force = -(plus.energy_ - minus.energy_) / (2.0 * delta);
            REQUIRE(atom.forces_[i](gamma) == Approx(num_force).epsilon(2e-4).margin(2e-4));
        }
    }
}

TEST_CASE("MTP LES prediction entry points include identical LES terms",
          "[les][mtp_les][api]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les = true;
    pot.les_species_coeffs[0] = 0.21;
    pot.les_species_coeffs[1] = -0.16;
    std::fill_n(pot.les_moment_coeffs, pot.get_alpha_scalar_moments(), 0.0);
    pot.les_moment_coeffs[0] = 0.03;

    Atoms input;
    input.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    input.pos_ = {{1.0,1.0,1.0}, {3.0,1.2,1.1}, {2.0,3.2,1.4}};
    input.chemmap_ = {0,1,0};
    Atoms serial=input, omp=input, threaded=input, energy=input, energy_omp=input;
    pot.getEFS(serial);
    pot.getEFSOMP(omp);
    pot.getEFSOMPthread(threaded);
    pot.getE(energy);
    pot.getEOMP(energy_omp);
    REQUIRE(omp.energy_ == Approx(serial.energy_).margin(1e-10));
    REQUIRE(threaded.energy_ == Approx(serial.energy_).margin(1e-10));
    REQUIRE(energy.energy_ == Approx(serial.energy_).margin(1e-10));
    REQUIRE(energy_omp.energy_ == Approx(serial.energy_).margin(1e-10));
    REQUIRE(omp.charges_ == serial.charges_);
    REQUIRE(threaded.charges_ == serial.charges_);
    REQUIRE(energy.charges_ == serial.charges_);
    REQUIRE((omp.stress_-serial.stress_).norm() < 1e-9);
    REQUIRE((threaded.stress_-serial.stress_).norm() < 1e-9);
    REQUIRE(std::accumulate(serial.site_energies_.begin(), serial.site_energies_.end(), 0.0)
            == Approx(serial.energy_).margin(1e-9));
    for (int i=0;i<input.size();++i) for(int d=0;d<3;++d) {
        REQUIRE(omp.forces_[i](d) == Approx(serial.forces_[i](d)).margin(1e-9));
        REQUIRE(threaded.forces_[i](d) == Approx(serial.forces_[i](d)).margin(1e-9));
    }
    Neighborhoods nbhs(input, pot.cutoff);
    double site = 0.0;
    REQUIRE_THROWS(pot.CalcSiteE(nbhs[0], site));
    REQUIRE_FALSE(pot.supports_site_efs());
    REQUIRE(pot.supports_analytical_cell_hessian());
}

TEST_CASE("MTP LES response parity flips all signed responses but not EFS",
          "[les][mtp_les][parity]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les=true;
    pot.les_species_coeffs[0]=0.23;
    pot.les_species_coeffs[1]=-0.17;
    std::fill_n(pot.les_moment_coeffs,pot.get_alpha_scalar_moments(),0.0);
    pot.train_init();
    Atoms input;
    input.cell_=10.0*Eigen::Matrix3d::Identity();
    input.pos_={{1.1,2.0,3.0},{4.2,2.5,6.0}};
    input.chemmap_={0,1};

    Atoms positive=input;
    pot.les_response_parity=1;
    pot.getEFS(positive);
    pot.compute_bec(positive,false,1.0);
    Neighborhoods positive_nbhs(positive,pot.cutoff);
    std::vector<double> grad_positive(pot.get_NL_coeffs_count(),0.0);
    pot.AccumulateTotalChargeGard(positive,positive_nbhs,1.0,grad_positive.data());

    Atoms negative=input;
    pot.les_response_parity=-1;
    pot.getEFS(negative);
    pot.compute_bec(negative,false,1.0);
    Neighborhoods negative_nbhs(negative,pot.cutoff);
    std::vector<double> grad_negative(pot.get_NL_coeffs_count(),0.0);
    pot.AccumulateTotalChargeGard(negative,negative_nbhs,1.0,grad_negative.data());

    REQUIRE(negative.energy_==Approx(positive.energy_).margin(1e-11));
    REQUIRE((negative.stress_-positive.stress_).norm()<1e-10);
    for (int i=0;i<input.size();++i) {
        REQUIRE(negative.charges_[i]==Approx(-positive.charges_[i]).margin(1e-14));
        REQUIRE((negative.forces_[i]-positive.forces_[i]).norm()<1e-10);
        REQUIRE((negative.born_eff_chgs_[i]+positive.born_eff_chgs_[i]).norm()<2e-8);
    }
    REQUIRE(pot.predicted_total_charge(negative)==
            Approx(-std::accumulate(positive.charges_.begin(),positive.charges_.end(),0.0)));
    for (int i=0;i<pot.get_NL_coeffs_count();++i)
        REQUIRE(grad_negative[i]==Approx(-grad_positive[i]).margin(1e-10));

    std::ostringstream saved;
    pot.Save(saved);
    REQUIRE(saved.str().find("les_response_parity = -1")!=std::string::npos);
    PotMTP_LES reloaded;
    std::istringstream serialized(saved.str());
    reloaded.Load(serialized);
    REQUIRE(reloaded.les_response_parity==-1);

    std::string legacy_text=saved.str();
    const auto parity_begin=legacy_text.find("les_response_parity = -1\n");
    REQUIRE(parity_begin!=std::string::npos);
    legacy_text.erase(parity_begin,std::string("les_response_parity = -1\n").size());
    PotMTP_LES legacy_default;
    legacy_default.les_response_parity=-1;
    std::istringstream legacy_serialized(legacy_text);
    legacy_default.Load(legacy_serialized);
    REQUIRE(legacy_default.les_response_parity==1);

    pot.les_response_parity=0;
    Atoms invalid=input;
    REQUIRE_THROWS(pot.getEFS(invalid));
    REQUIRE_THROWS(pot.train_init());
}

TEST_CASE("MTP LES is invariant to periodic reimaging",
          "[les][mtp_les][pbc]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les=true;
    pot.les_species_coeffs[0]=0.2;
    pot.les_species_coeffs[1]=-0.1;
    std::fill_n(pot.les_moment_coeffs,pot.get_alpha_scalar_moments(),0.0);
    pot.les_moment_coeffs[0]=0.025;
    Atoms reference;
    reference.cell_<<10.0,0.0,0.0, 1.2,9.0,0.0, -0.3,0.7,11.0;
    reference.pos_={{1.0,1.0,1.0},{3.0,1.3,1.2},{2.1,3.2,1.4}};
    reference.chemmap_={0,1,0};
    Atoms image=reference;
    image.pos_[1]+=2.0*reference.cell_.row(0)-reference.cell_.row(1)
        +reference.cell_.row(2);
    pot.getEFS(reference);
    pot.getEFS(image);
    REQUIRE(image.energy_==Approx(reference.energy_).margin(1e-10));
    REQUIRE((image.stress_-reference.stress_).norm()<2e-9);
    for (int i=0;i<reference.size();++i) {
        REQUIRE(image.charges_[i]==Approx(reference.charges_[i]).margin(1e-11));
        REQUIRE((image.forces_[i]-reference.forces_[i]).norm()<2e-9);
    }
}

TEST_CASE("MTP LES BEC failure clears stale response state",
          "[les][mtp_les][bec][state]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    Atoms atom;
    atom.cell_=10.0*Eigen::Matrix3d::Identity();
    atom.pos_={{1.0,2.0,3.0}};
    atom.chemmap_={0};
    atom.born_eff_chgs_.assign(1,Eigen::Matrix3d::Identity());
    atom.have_eff_chg=true;
    pot.lr_les=false;
    REQUIRE_THROWS(pot.compute_bec(atom));
    REQUIRE(atom.born_eff_chgs_.empty());
    REQUIRE_FALSE(atom.have_eff_chg);
}

TEST_CASE("Analytic MTP LES BEC matches displaced periodic dipoles",
          "[les][mtp_les][bec][analytic]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les=true;
    pot.les_species_coeffs[0]=0.23;
    pot.les_species_coeffs[1]=-0.17;
    std::fill_n(pot.les_moment_coeffs,pot.get_alpha_scalar_moments(),0.0);
    pot.les_moment_coeffs[0]=0.04;
    pot.les_moment_coeffs[1]=-0.025;
    pot.les_moment_coeffs[3]=0.27; // product moment: exercise reverse propagation
    pot.les_moment_coeffs[4]=-0.13; // rotational scalar made from vector moments
    pot.train_init();
    Atoms input;
    input.pos_={{0.3,0.4,0.5},{2.1,0.8,0.9},{1.1,2.5,1.4}};
    input.chemmap_={0,1,0};
    // The smaller cell includes multiple images and self images inside cutoff.
    for (double length:{4.0,10.0}) for (bool triclinic:{false,true}) {
        input.cell_=length*Eigen::Matrix3d::Identity();
        if (triclinic) {
            input.cell_(1,0)=0.7;
            input.cell_(2,0)=-0.4;
            input.cell_(2,1)=0.5;
        }
        for (bool centered:{false,true}) for (int parity:{-1,1}) {
            pot.les_response_parity=parity;
            Atoms analytic=input;
            // The response API must also work without an energy call.
            pot.compute_bec(analytic,centered,4.0);
            REQUIRE(analytic.have_eff_chg);
            auto dipole=[&](Atoms displaced) {
                pot.getE(displaced);
                Eigen::VectorXd q=Eigen::Map<Eigen::VectorXd>(
                    displaced.charges_.data(),displaced.size());
                if (centered) q.array()-=q.mean();
                return LES::compute_periodic_dipole_fractional(displaced,q);
            };
            const Eigen::Matrix3d inverse=input.cell_.inverse();
            constexpr double h=1.e-5;
            for (int i=0;i<input.size();++i) for (int gamma=0;gamma<3;++gamma) {
                Atoms plus=input,minus=input;
                plus.pos_[i](gamma)+=h;
                minus.pos_[i](gamma)-=h;
                const Eigen::Vector3cd d=(dipole(plus)-dipole(minus))/(2*h);
                const Eigen::RowVector3d fractional=input.pos_[i]*inverse;
                Eigen::Vector3d dephased;
                for (int alpha=0;alpha<3;++alpha)
                    dephased(alpha)=(std::exp(std::complex<double>(
                        0.,-2*M_PI*fractional(alpha)))*d(alpha)).real();
                const Eigen::Vector3d expected=2.0*input.cell_.transpose()*dephased;
                REQUIRE((analytic.born_eff_chgs_[i].col(gamma)-expected).norm()<2.e-7);
            }
        }
    }
}

TEST_CASE("Analytic dynamic LES BEC preserves geometric and response state contracts",
          "[les][mtp_les][bec][analytic][pbc]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les=true;
    pot.les_species_coeffs[0]=0.23;
    pot.les_species_coeffs[1]=-0.17;
    std::fill_n(pot.les_moment_coeffs,pot.get_alpha_scalar_moments(),0.0);
    pot.les_moment_coeffs[0]=0.04;
    pot.les_moment_coeffs[1]=-0.025;
    pot.les_moment_coeffs[3]=0.27;
    pot.les_moment_coeffs[4]=-0.13;
    pot.train_init();
    Atoms a;
    a.cell_<<10.,0.,0.,1.2,9.,0.,-0.3,0.7,11.;
    a.pos_={{1.,1.,1.},{3.,1.3,1.2},{2.1,3.2,1.4},{4.5,3.8,2.7}};
    a.chemmap_={0,1,0,1};
    pot.getEFS(a);
    const auto energy=a.energy_;
    const auto forces=a.forces_;
    const auto stress=a.stress_;
    const auto charges=a.charges_;
    pot.compute_bec(a);
    REQUIRE(a.energy_==energy);
    REQUIRE((a.stress_-stress).norm()==0.);
    REQUIRE(a.charges_==charges);
    for (int i=0;i<a.size();++i) REQUIRE((a.forces_[i]-forces[i]).norm()==0.);

    Atoms shifted=a,image=a,rotated=a;
    for (auto& p:shifted.pos_) p+=Eigen::RowVector3d(1.7,-0.8,2.4);
    image.pos_[1]+=2*a.cell_.row(0)-a.cell_.row(1)+a.cell_.row(2);
    Eigen::Matrix3d rotation;
    rotation<<std::cos(0.37),-std::sin(0.37),0.,
              std::sin(0.37), std::cos(0.37),0.,0.,0.,1.;
    rotated.cell_=a.cell_*rotation.transpose();
    for (auto& p:rotated.pos_) p=(p*rotation.transpose()).eval();
    pot.compute_bec(shifted);
    pot.compute_bec(image);
    pot.compute_bec(rotated);
    for (int i=0;i<a.size();++i) {
        REQUIRE((shifted.born_eff_chgs_[i]-a.born_eff_chgs_[i]).norm()<1.e-11);
        REQUIRE((image.born_eff_chgs_[i]-a.born_eff_chgs_[i]).norm()<1.e-11);
        REQUIRE((rotated.born_eff_chgs_[i]
            -rotation*a.born_eff_chgs_[i]*rotation.transpose()).norm()<1.e-11);
    }
    pot.les_moment_coeffs[0]=std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(pot.compute_bec(a));
    REQUIRE_FALSE(a.have_eff_chg);
    REQUIRE(a.born_eff_chgs_.empty());
}

TEST_CASE("MTP LES coordinate Hessian uses full forces and hard-cutoff cell Hessian rejects",
          "[les][mtp_les][hessian]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les = true;
    pot.les_species_coeffs[0]=0.2; pot.les_species_coeffs[1]=-0.1;
    std::fill_n(pot.les_moment_coeffs, pot.get_alpha_scalar_moments(), 0.0);
    pot.les_moment_coeffs[0]=0.025;
    Atoms atom; atom.cell_=9.0*Eigen::Matrix3d::Identity();
    atom.pos_={{1.,1.,1.},{3.,1.3,1.2}}; atom.chemmap_={0,1};
    HessianMatrix h; pot.getEFSHessian(atom,h);
    REQUIRE(h.size == atom.size());
    for(int i=0;i<atom.size();++i) for(int j=0;j<atom.size();++j)
        REQUIRE((h(i,j)-h(j,i).transpose()).norm() < 1e-10);
    for(int i=0;i<atom.size();++i) for(int a=0;a<3;++a)
        for(int b=0;b<3;++b) {
            double rowsum=0.0;
            for(int j=0;j<atom.size();++j) rowsum += h(i,a,j,b);
            REQUIRE(std::abs(rowsum) < 2e-6);
        }
    const double delta=2e-4;
    Atoms plus=atom, minus=atom, center=atom;
    plus.pos_[0](0)+=delta; minus.pos_[0](0)-=delta;
    pot.getE(plus); pot.getE(minus); pot.getE(center);
    const double energy_hessian=(plus.energy_-2.0*center.energy_+minus.energy_)/(delta*delta);
    REQUIRE(h(0,0,0,0) == Approx(energy_hessian).margin(2e-3));
    pot.lr_les=false;
    HessianMatrix sr_h; pot.getEFSHessian(atom,sr_h);
    double les_norm=0.0;
    for(int i=0;i<atom.size();++i) for(int j=0;j<atom.size();++j)
        les_norm += (h(i,j)-sr_h(i,j)).squaredNorm();
    REQUIRE(std::sqrt(les_norm) > 1e-6);
    pot.lr_les=true;
    HessianMatrix symm_h, omp_h;
    pot.getEFSHessianSymm(atom,symm_h);
    pot.getEFSHessianSymmOMP(atom,omp_h);
    for(int i=0;i<atom.size();++i) for(int j=0;j<atom.size();++j) {
        REQUIRE((symm_h(i,j)-h(i,j)).norm() < 1e-10);
        REQUIRE((omp_h(i,j)-h(i,j)).norm() < 1e-10);
    }
    std::vector<double> diagonal;
    pot.getEFSHessianSymmDiaOMP(atom,diagonal);
    REQUIRE(diagonal.size() == 3*atom.size());
    for(int i=0;i<atom.size();++i) for(int d=0;d<3;++d)
        REQUIRE(diagonal[3*i+d] == Approx(h(i,d,i,d)).margin(1e-10));
    Neighborhoods nbhs(atom,pot.cutoff);
    HessianSparse sparse;
    pot.getHessian(atom,nbhs,{{0,0},{0,1}},sparse);
    REQUIRE((sparse(0,0)-h(0,0)).norm() < 1e-10);
    REQUIRE((sparse(0,1)-h(0,1)).norm() < 1e-10);
    Eigen::MatrixXd ch, lr;
    REQUIRE(pot.supports_analytical_cell_hessian());
    pot.les_cutoff_mode="reference_hard";
    REQUIRE_FALSE(pot.supports_analytical_cell_hessian());
    REQUIRE_THROWS(pot.getEFSHessianCell(atom,h,ch,lr));
    REQUIRE_THROWS(pot.getEFSHessianCell(atom,nbhs,h,ch,lr));
    REQUIRE_THROWS(pot.getEFSHessianCellSymm(atom,h,ch,lr));
    REQUIRE_THROWS(pot.getEFSHessianCellSymmOMP(atom,h,ch,lr));
    REQUIRE_THROWS(pot.getEFSHessianCellOMP(atom,nbhs,h,ch,lr));
}

TEST_CASE("MTP LES outer OpenMP evaluation is deterministic",
          "[les][mtp_les][api][omp]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les = true;
    pot.les_species_coeffs[0]=0.19; pot.les_species_coeffs[1]=-0.13;
    std::fill_n(pot.les_moment_coeffs, pot.get_alpha_scalar_moments(), 0.0);
    pot.les_moment_coeffs[0]=0.027; pot.les_moment_coeffs[1]=-0.011;
    Atoms input; input.cell_=10.0*Eigen::Matrix3d::Identity();
    input.pos_={{1.,1.,1.},{3.,1.2,1.1},{2.,3.1,1.3}}; input.chemmap_={0,1,0};
    std::vector<Atoms> reference(8, input), actual(8, input);
    for (int i=0;i<8;++i) {
        reference[i].pos_[1](0) += 0.01*i;
        actual[i] = reference[i];
        pot.getEFS(reference[i]);
    }
#pragma omp parallel for num_threads(2) schedule(static)
    for (int i=0;i<8;++i) pot.getEFSOMPthread(actual[i]);
    for (int i=0;i<8;++i) {
        REQUIRE(actual[i].energy_ == Approx(reference[i].energy_).margin(1e-10));
        REQUIRE(actual[i].charges_ == reference[i].charges_);
        for(int a=0;a<input.size();++a) for(int d=0;d<3;++d)
            REQUIRE(actual[i].forces_[a](d) == Approx(reference[i].forces_[a](d)).margin(1e-9));
    }
}

TEST_CASE("MTP LES zero initialization is reproducible and charge balanced",
          "[les][mtp_les][charge][init]") {
    auto make_atoms=[] {
        Atoms a; a.cell_=10.0*Eigen::Matrix3d::Identity();
        a.pos_={{1.,1.,1.},{3.,1.2,1.1},{2.,3.1,1.3}}; a.chemmap_={0,0,1};
        Atoms b=a; b.pos_[1](0)+=0.17; b.chemmap_={0,1,1};
        return std::vector<Atoms>{a,b};
    };
    auto atoms=make_atoms();
    const std::vector<double> targets={0.35,-0.20};
    PotMTP_LES first("tests/case-mtp-les/les.mtp");
    PotMTP_LES second("tests/case-mtp-les/les.mtp");
    for(auto *pot:{&first,&second}) {
        pot->lr_les=true;
        std::fill_n(pot->les_species_coeffs,pot->species_count,0.0);
        std::fill_n(pot->les_moment_coeffs,pot->get_alpha_scalar_moments(),0.0);
        REQUIRE(pot->initialize_zero_les_coeffs(atoms,targets,1e-3,5489));
    }
    double moment_norm=0.0;
    for(int i=0;i<first.get_alpha_scalar_moments();++i) {
        REQUIRE(first.les_moment_coeffs[i] == second.les_moment_coeffs[i]);
        moment_norm += first.les_moment_coeffs[i]*first.les_moment_coeffs[i];
    }
    REQUIRE(moment_norm > 0.0);
    for(int i=0;i<first.species_count;++i)
        REQUIRE(first.les_species_coeffs[i] == second.les_species_coeffs[i]);
    for(size_t i=0;i<atoms.size();++i) {
        Atoms evaluated=atoms[i]; first.getE(evaluated);
        REQUIRE(std::accumulate(evaluated.charges_.begin(),evaluated.charges_.end(),0.0)
                == Approx(targets[i]).margin(1e-9));
    }
    Atoms with_les=atoms[0], short_range=atoms[0];
    first.getEFS(with_les);
    first.lr_les=false;
    first.getEFS(short_range);
    first.lr_les=true;
    double les_force_norm=0.0;
    for(int i=0;i<with_les.size();++i)
        les_force_norm += (with_les.forces_[i]-short_range.forces_[i]).squaredNorm();
    REQUIRE(std::sqrt(les_force_norm) > 1e-10);
    std::vector<double> before(first.les_moment_coeffs,
        first.les_moment_coeffs+first.get_alpha_scalar_moments());
    REQUIRE_FALSE(first.initialize_zero_les_coeffs(atoms,targets,2e-3,1));
    REQUIRE(std::equal(before.begin(),before.end(),first.les_moment_coeffs));
    PotMTP_LES invalid("tests/case-mtp-les/les.mtp");
    invalid.lr_les=true;
    std::fill_n(invalid.les_species_coeffs,invalid.species_count,0.0);
    std::fill_n(invalid.les_moment_coeffs,invalid.get_alpha_scalar_moments(),0.0);
    REQUIRE_THROWS(invalid.initialize_zero_les_coeffs(atoms,targets,0.0,5489));
}

TEST_CASE("MTP LES combined EFS parameter gradient is numerical", "[les][mtp_les][grad]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les=true;
    pot.les_species_coeffs[0]=0.23;
    pot.les_species_coeffs[1]=-0.17;
    for (int a=0;a<pot.get_alpha_scalar_moments();++a)
        pot.les_moment_coeffs[a]=0.0;
    pot.les_moment_coeffs[0]=0.035;
    pot.les_moment_coeffs[1]=-0.021;
    pot.train_init();

    Atoms atom;
    atom.cell_<<10.0,0,0, 0.8,10.5,0, -0.3,0.6,11.0;
    atom.pos_={{2.0,2.0,2.0},{4.1,2.2,2.0},{2.3,4.2,2.4},{5.0,4.8,3.1}};
    atom.chemmap_={0,1,0,1};
    Eigen::Matrix<double,-1,3> dLdF(atom.size(),3);
    dLdF<<0.12,-0.07,0.03, -0.04,0.08,-0.02,
           0.06,0.01,-0.09, -0.03,-0.02,0.08;
    Eigen::Matrix3d dLdS;
    dLdS<<0.03,-0.01,0.005, -0.01,-0.02,0.004, 0.005,0.004,0.01;
    const double dLdE=0.37;

    Neighborhoods nbhs(atom,pot.cutoff);
    atom.resetEFS();
    pot.getEFS(atom,nbhs);
    std::vector<double> analytic(pot.get_NL_coeffs_count(),0.0);
    Eigen::Matrix<double,-1,3,Eigen::RowMajor> unused;
    pot.AccumulateEFSCombinationGard(atom,nbhs,dLdE,dLdF,dLdS,unused,analytic.data());

    std::vector<double> coeffs(pot.get_NL_coeffs_count());
    pot.nl_get_coeffs(coeffs.data());
    const int les_species_offset=pot.get_NL_coeffs_count()
        -pot.species_count-pot.get_alpha_scalar_moments();
    const int les_moment_offset=les_species_offset+pot.species_count;
    const auto loss_at=[&](const std::vector<double>& trial) {
        pot.nl_set_coeffs(trial.data());
        Atoms evaluated=atom;
        evaluated.resetEFS();
        Neighborhoods trial_nbhs(evaluated,pot.cutoff);
        pot.getEFS(evaluated,trial_nbhs);
        double loss=dLdE*evaluated.energy_;
        for (int i=0;i<evaluated.size();++i)
            loss+=dLdF.row(i).dot(evaluated.forces_[i]);
        loss+=(dLdS.array()*evaluated.stress_.array()).sum();
        return loss;
    };
    const double delta=2e-6;
    for (int idx : {0,les_species_offset,les_moment_offset,les_moment_offset+1}) {
        auto plus=coeffs, minus=coeffs;
        plus[idx]+=delta; minus[idx]-=delta;
        const double numeric=(loss_at(plus)-loss_at(minus))/(2.0*delta);
        REQUIRE(analytic[idx]==Approx(numeric).margin(3e-4));
    }
    pot.nl_set_coeffs(coeffs.data());
}

TEST_CASE("MTP LES total-charge parameter gradient is analytic",
          "[les][mtp_les][charge][grad]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les = true;
    pot.les_species_coeffs[0] = 0.19;
    pot.les_species_coeffs[1] = -0.13;
    for (int a = 0; a < pot.get_alpha_scalar_moments(); ++a)
        pot.les_moment_coeffs[a] = 0.0;
    pot.les_moment_coeffs[0] = 0.031;
    pot.les_moment_coeffs[1] = -0.017;
    pot.train_init();

    Atoms atom;
    atom.cell_ << 10.0, 0.0, 0.0, 0.4, 10.5, 0.0,
                  -0.2, 0.5, 11.0;
    atom.pos_ = {{2.0, 2.0, 2.0}, {4.1, 2.2, 2.0},
                 {2.3, 4.2, 2.4}, {5.0, 4.8, 3.1}};
    atom.chemmap_ = {0, 1, 0, 1};
    Neighborhoods nbhs(atom, pot.cutoff);
    atom.resetEFS();
    pot.getEFS(atom, nbhs);

    constexpr double dLdQ = 0.73;
    std::vector<double> analytic(pot.get_NL_coeffs_count(), 0.0);
    pot.AccumulateTotalChargeGard(atom, nbhs, dLdQ, analytic.data());

    const auto groups = pot.nl_parameter_groups();
    REQUIRE(groups.size() == 4);
    REQUIRE(groups[0].name == "sr_radial");
    REQUIRE(groups[1].name == "sr_linear");
    REQUIRE(groups[2].name == "les_species");
    REQUIRE(groups[3].name == "les_moment");
    REQUIRE(groups.front().begin == 0);
    REQUIRE(groups.back().end == pot.get_NL_coeffs_count());

    std::vector<double> coeffs(pot.get_NL_coeffs_count());
    pot.nl_get_coeffs(coeffs.data());
    const auto charge_at = [&](const std::vector<double> &trial) {
        pot.nl_set_coeffs(trial.data());
        Atoms evaluated = atom;
        evaluated.resetEFS();
        Neighborhoods trial_nbhs(evaluated, pot.cutoff);
        pot.getEFS(evaluated, trial_nbhs);
        return dLdQ * pot.predicted_total_charge(evaluated);
    };
    const double delta = 2e-6;
    for (int idx : {groups[0].begin, groups[1].begin,
                    groups[2].begin, groups[3].begin,
                    groups[3].begin + 1}) {
        auto plus = coeffs;
        auto minus = coeffs;
        plus[idx] += delta;
        minus[idx] -= delta;
        const double numeric = (charge_at(plus) - charge_at(minus))
            / (2.0 * delta);
        REQUIRE(analytic[idx] == Approx(numeric).margin(2e-6));
    }
    pot.nl_set_coeffs(coeffs.data());
}

TEST_CASE("Nonlinear trainer adds normalized per-configuration charge loss",
          "[les][mtp_les][charge][train]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.lr_les = true;
    pot.les_species_coeffs[0] = 0.16;
    pot.les_species_coeffs[1] = -0.11;
    for (int a = 0; a < pot.get_alpha_scalar_moments(); ++a)
        pot.les_moment_coeffs[a] = 0.0;
    pot.les_moment_coeffs[0] = 0.024;
    pot.les_moment_coeffs[1] = -0.013;
    pot.set_inited();

    Atoms reference;
    reference.cell_ = 11.0 * Eigen::Matrix3d::Identity();
    reference.pos_ = {{2.0, 2.0, 2.0}, {4.1, 2.2, 2.0},
                      {2.3, 4.2, 2.4}, {5.0, 4.8, 3.1}};
    reference.chemmap_ = {0, 1, 0, 1};
    reference.id = 0;
    reference.fitness = 1.0;
    reference.features_["total_charge"] = "-0.7";
    reference.global_attribs_d_["total_charge"] = 0.2;
    reference.resetEFS();

    std::vector<Atoms> atoms = {reference};
    std::vector<Neighborhoods> nbhs = {
        Neighborhoods(atoms[0], pot.cutoff)
    };
    GPSettings settings;
    settings.Modify("energy_weight", "0");
    settings.Modify("force_weight", "0");
    settings.Modify("stress_weight", "0");
    settings.Modify("charge_weight", "2");
    settings.Modify("charge_pow", "1");
    settings.Modify("penalty_weight", "0");
    settings.Modify("parameter_scaling", "none");
    TrainNonlinear trainer(settings);
    trainer.init(atoms, nbhs, &pot);

    double loss = 0.0;
    Eigen::VectorXd gradient(pot.get_NL_coeffs_count());
    trainer.AccumulateGard(loss, gradient);

    Atoms evaluated = atoms[0];
    evaluated.resetEFS();
    Neighborhoods eval_nbhs(evaluated, pot.cutoff);
    pot.getEFS(evaluated, eval_nbhs);
    const double total = pot.predicted_total_charge(evaluated);
    const double error = total - 0.2; // GlobalDouble takes precedence.
    const double weight = 2.0 / evaluated.size();
    REQUIRE(loss == Approx(weight * error * error).margin(1e-11));
    REQUIRE(trainer.qmonitor.qrms == Approx(std::abs(error)).margin(1e-12));

    std::vector<double> expected(pot.get_NL_coeffs_count(), 0.0);
    pot.AccumulateTotalChargeGard(evaluated, eval_nbhs,
                                  2.0 * weight * error, expected.data());
    for (int i = 0; i < gradient.size(); ++i)
        REQUIRE(gradient[i] == Approx(expected[i]).margin(2e-9));

    trainer.ConfigureTrainableGroups({"les_species"});
    std::vector<double> before(pot.get_NL_coeffs_count());
    std::vector<double> trial(pot.get_NL_coeffs_count());
    std::vector<double> after(pot.get_NL_coeffs_count());
    pot.nl_get_coeffs(before.data());
    for (int i = 0; i < static_cast<int>(trial.size()); ++i)
        trial[i] = before[i] + 0.123;
    trainer.SetOptimizerCoeffs(trial.data());
    pot.nl_get_coeffs(after.data());
    const auto groups = pot.nl_parameter_groups();
    for (const auto &group : groups) {
        for (int i = group.begin; i < group.end; ++i) {
            if (group.name == "les_species")
                REQUIRE(after[i] == Approx(trial[i]));
            else
                REQUIRE(after[i] == Approx(before[i]));
        }
    }
}

TEST_CASE("PotMTP_LES loads plain MTP without LES sections", "[les][mtp_les][load]") {
    PotMTP_LES pot("tests/case-mtp-les/_Trained_Potential_.mtp");

    REQUIRE_FALSE(pot.lr_les);
    REQUIRE(pot.les_species_coeffs != nullptr);
    REQUIRE(pot.les_moment_coeffs != nullptr);

    for (int t = 0; t < pot.species_count; ++t)
        REQUIRE(pot.les_species_coeffs[t] == Approx(0.0).margin(1e-30));

    for (int a = 0; a < pot.get_alpha_scalar_moments(); ++a)
        REQUIRE(pot.les_moment_coeffs[a] == Approx(0.0).margin(1e-30));

    Atoms atom;
    atom.cell_ = 12.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2.0, 2.0, 2.0}, {4.1, 2.2, 2.0}, {2.3, 4.2, 2.4}, {5.0, 4.8, 3.1}};
    atom.chemmap_ = {0, 1, 0, 1};

    Neighborhoods nbhs(atom, pot.cutoff);
    REQUIRE_NOTHROW(pot.getEFS(atom, nbhs));
    REQUIRE(atom.charges_.empty());
}

TEST_CASE("MTP LES rejects ambiguous legacy coefficients", "[les][mtp_les][load]") {
    REQUIRE_THROWS_AS(PotMTP_LES("tests/case-mtp-les/_Trained_Potential_"),
                      GPException);
}

TEST_CASE("MTP LES train_init preserves loaded zero coefficients", "[les][mtp_les][train]") {
    PotMTP_LES pot("tests/case-mtp-les/_Trained_Potential_.mtp");
    for (int t=0;t<pot.species_count;++t)
        REQUIRE(pot.les_species_coeffs[t]==Approx(0.0).margin(1e-30));
    for (int a=0;a<pot.get_alpha_scalar_moments();++a)
        REQUIRE(pot.les_moment_coeffs[a]==Approx(0.0).margin(1e-30));
    pot.train_init();
    for (int t=0;t<pot.species_count;++t)
        REQUIRE(pot.les_species_coeffs[t]==Approx(0.0).margin(1e-30));
    for (int a=0;a<pot.get_alpha_scalar_moments();++a)
        REQUIRE(pot.les_moment_coeffs[a]==Approx(0.0).margin(1e-30));
}

TEST_CASE("BEC zero charges", "[les][bec]") {
    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1.0, 2.0, 3.0}};
    atom.chemmap_ = {0};

    Eigen::VectorXd q(1);
    q(0) = 0.0;

    LES les;
    std::vector<Eigen::Matrix3d> bec;
    les.compute_bec(atom, q, bec, false, 1.0);

    REQUIRE(bec.size() == 1);
    for (int a = 0; a < 3; ++a)
        for (int g = 0; g < 3; ++g)
            REQUIRE(bec[0](a, g) == Approx(0.0).margin(1e-15));
}

// ═══════════════════════════════════════════════════════════════════════
//  Gradient numerical verification: dLdq_from_forces
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("dLdq_from_forces numerical verification", "[les][grad]") {
    // Use a simple loss L = 1/2 * Σ_i |F_i|² (force-only)
    // dL/dF_i = F_i, then compute dL/dq_i analytically via compute_les_dLdq_from_forces
    // Verify numerically: perturb q_i, recompute forces, dL/dq_i ≈ (L(q+δ)-L(q-δ))/(2δ)

    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2.0, 3.0, 1.0}, {7.0, 5.0, 8.0}};
    atom.forces_.assign(2, Eigen::Vector3d::Zero());
    atom.stress_ = Eigen::Matrix3d::Zero();
    atom.chemmap_ = {0, 0};

    Eigen::VectorXd q(2);
    q << 0.5, -0.5;

    LES les;
    les.lr_sigma = 0.5;
    les.dl = 2.0;

    // Compute forces F_i (these become dL/dF_i for L = 1/2 Σ|F|²)
    Neighborhoods nbhs;
    les.generate_k_vectors(atom, les.kvecs);
    atom.energy_ = 0.0; zero_forces(atom.forces_); atom.stress_.setZero();
    Eigen::VectorXd q_copy = q;
    les.add_latent_ewald_efs(atom, nbhs, q_copy);

    // dL/dF_i = F_i
    Eigen::Matrix<double, -1, 3> dLdF(atom.size(), 3);
    for (int i = 0; i < atom.size(); ++i)
        dLdF.row(i) = atom.forces_[i].transpose();
    Eigen::Matrix3d dLdS = Eigen::Matrix3d::Zero();

    // Analytical dL/dq_i via compute_les_dLdq_from_forces
    Eigen::VectorXd dLdq_ana(atom.size());
    les.compute_les_dLdq_from_forces(atom, q, dLdF, dLdS, dLdq_ana);

    // Numerical verification: perturb each q_i
    const double delta = 1e-6;
    for (int i = 0; i < atom.size(); ++i) {
        Eigen::VectorXd q_pert = q;
        q_pert(i) += delta;
        atom.energy_ = 0.0; zero_forces(atom.forces_); atom.stress_.setZero();
        les.add_latent_ewald_efs(atom, nbhs, q_pert);
        double L_plus = 0.0;
        for (int j = 0; j < atom.size(); ++j)
            L_plus += 0.5 * atom.forces_[j].squaredNorm();

        q_pert(i) -= 2.0 * delta;
        atom.energy_ = 0.0; zero_forces(atom.forces_); atom.stress_.setZero();
        les.add_latent_ewald_efs(atom, nbhs, q_pert);
        double L_minus = 0.0;
        for (int j = 0; j < atom.size(); ++j)
            L_minus += 0.5 * atom.forces_[j].squaredNorm();

        double dLdq_num = (L_plus - L_minus) / (2.0 * delta);
        REQUIRE(dLdq_ana(i) == Approx(dLdq_num).epsilon(1e-5));
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Gradient consistency: dLdq_from_forces vs force_stress_grad
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("dLdq_from_forces reproduces force_stress_grad sums", "[les][grad]") {
    // Verify: Σ_{i:type=t} dL/dq_i * output_scaling = dL/d(les_species[t])
    // and:    Σ_i dL/dq_i * output_scaling * B_i[a] = dL/d(les_moment[a])
    // where both sides use the same dLdF/dLdS input.

    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2.0, 3.0, 1.0}, {7.0, 5.0, 8.0}, {1.0, 6.0, 4.0}};
    atom.forces_.assign(3, Eigen::Vector3d::Zero());
    atom.stress_ = Eigen::Matrix3d::Zero();
    atom.chemmap_ = {0, 1, 0};

    const int species_count = 2;
    const int alpha_scalar = 4;  // use first 4 pseudo-descriptors
    Eigen::MatrixXd descriptors(atom.size(), alpha_scalar);
    descriptors << 1.0, 0.5, 0.2, 0.1,
                   2.0, 1.0, 0.3, 0.2,
                   1.5, 0.8, 0.4, 0.15;
    std::vector<int> atom_types = {0, 1, 0};

    Eigen::VectorXd q(atom.size());
    q << 0.3, -0.5, 0.2;

    LES les;
    les.lr_sigma = 0.5;
    les.dl = 2.0;

    // Compute forces to use as dLdF
    Neighborhoods nbhs;
    les.generate_k_vectors(atom, les.kvecs);
    atom.energy_ = 0.0; zero_forces(atom.forces_); atom.stress_.setZero();
    Eigen::VectorXd q_copy = q;
    les.add_latent_ewald_efs(atom, nbhs, q_copy);

    Eigen::Matrix<double, -1, 3> dLdF(atom.size(), 3);
    for (int i = 0; i < atom.size(); ++i)
        dLdF.row(i) = atom.forces_[i].transpose();
    Eigen::Matrix3d dLdS = Eigen::Matrix3d::Zero();

    // 1. Compute via compute_les_force_stress_grad
    std::vector<double> grad_s(species_count + alpha_scalar, 0.0);
    les.compute_les_force_stress_grad(atom, q, descriptors, atom_types,
        species_count, alpha_scalar, dLdF, dLdS, grad_s.data());

    // 2. Compute via compute_les_dLdq_from_forces
    Eigen::VectorXd dLdq_fs(atom.size());
    les.compute_les_dLdq_from_forces(atom, q, dLdF, dLdS, dLdq_fs);

    // 3. Sum dLdq to reconstruct species/moment gradients
    std::vector<double> grad_s_dq(species_count, 0.0);
    std::vector<double> grad_m_dq(alpha_scalar, 0.0);
    for (int i = 0; i < atom.size(); ++i) {
        grad_s_dq[atom_types[i]] += dLdq_fs(i);
        for (int a = 0; a < alpha_scalar; ++a)
            grad_m_dq[a] += dLdq_fs(i) * descriptors(i, a);
    }

    // 4. Compare the reconstructed species and moment gradients
    const double* grad_m = grad_s.data() + species_count;
    for (int t = 0; t < species_count; ++t) {
        REQUIRE(grad_s[t] == Approx(grad_s_dq[t]).epsilon(1e-12));
    }
    for (int a = 0; a < alpha_scalar; ++a) {
        REQUIRE(grad_m[a] == Approx(grad_m_dq[a]).epsilon(1e-12));
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Ewald force sign: energy decrease in force direction
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Ewald force points in direction of decreasing energy", "[les][ewald]") {
    // For any system, moving an atom in the force direction (F = -dE/dr)
    // should decrease the energy.
    Atoms atom;
    atom.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2.0, 3.0, 1.0}, {7.0, 5.0, 8.0}};
    atom.forces_.assign(2, Eigen::Vector3d::Zero());
    atom.stress_ = Eigen::Matrix3d::Zero();
    atom.chemmap_ = {0, 0};

    Eigen::VectorXd q(2);
    q << 0.5, -0.5;

    LES les;
    les.lr_sigma = 0.5;
    les.dl = 2.0;

    // Reference EFS
    Neighborhoods nbhs;
    les.generate_k_vectors(atom, les.kvecs);
    atom.energy_ = 0.0; zero_forces(atom.forces_); atom.stress_.setZero();
    Eigen::VectorXd q_copy = q;
    les.add_latent_ewald_efs(atom, nbhs, q_copy);
    double E0 = atom.energy_;
    auto F = atom.forces_;

    // Move atom 0 slightly in force direction and check energy decreases
    const double step = 0.001;
    for (int i = 0; i < atom.size(); ++i) {
        double F_norm = F[i].norm();
        if (F_norm < 1e-12) continue;  // skip zero-force atoms

        Eigen::Vector3d dir = F[i].normalized();
        Atoms atom_fwd = atom;
        atom_fwd.pos_[i] += step * dir;
        double E_fwd = ewald_energy_at(atom_fwd, q, les);

        // F = -dE/dr, so moving in +F direction should decrease E
        REQUIRE(E_fwd < E0);
    }
}
