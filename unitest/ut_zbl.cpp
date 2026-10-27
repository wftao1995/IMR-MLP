// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "base/neighborhood.h"
#include "analyse/compute_hessian.h"
#include "potentials/potential_zbl.h"
#include <cmath>

using Catch::Approx;
using namespace Potentials;

namespace {

GPSettings make_zbl_settings(double r0 = 1.0, double r1 = 2.0) {
    GPSettings settings;
    settings.Modify("z", "13");
    settings.Modify("r0", std::to_string(r0));
    settings.Modify("r1", std::to_string(r1));
    return settings;
}

Atoms make_dimer(double distance, int left_type = 0, int right_type = 0) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 20.0;
    atom.pos_ = {
        {0.0, 0.0, 0.0},
        {distance, 0.0, 0.0},
    };
    atom.chemmap_ = {left_type, right_type};
    return atom;
}

double energy_at(PotentialZBL &pot, double distance) {
    Atoms atom = make_dimer(distance);
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getE(atom, nbhs);
    return atom.energy_;
}

Atoms efs_at(PotentialZBL &pot, double distance) {
    Atoms atom = make_dimer(distance);
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);
    return atom;
}

Eigen::Matrix3d independent_zbl_curvature(double r) {
    if (r >= 2.) return Eigen::Matrix3d::Zero();
    const double screening = .46850 / (2 * std::pow(13., .23));
    const double coefficients[] = {.18175, .50986, .28022, .02817};
    const double exponents[] = {-3.19980, -.94229, -.40290, -.20162};
    double s0 = 0., s1 = 0., s2 = 0.;
    for (int k = 0; k < 4; ++k) {
        const double lambda = exponents[k] / screening;
        const double term = coefficients[k] * std::exp(lambda * r);
        s0 += term; s1 += lambda * term; s2 += lambda * lambda * term;
    }
    const double scale = 14.3997 * 13 * 13;
    const double energy = scale * s0 / r;
    const double first = scale * (s1 / r - s0 / (r*r));
    const double second = scale * (s2 / r - 2*s1 / (r*r) + 2*s0 / (r*r*r));
    double sw = 1., ds = 0., dds = 0.;
    if (r > 1.) {
        const double t = r - 1.;
        sw = 1 - 3*t*t + 2*t*t*t;
        ds = -6*t + 6*t*t;
        dds = -6 + 12*t;
    }
    // Return the axial form; callers rotate its longitudinal/transverse parts.
    Eigen::Matrix3d out = Eigen::Matrix3d::Identity() * ((sw*first + ds*energy) / r);
    out(0,0) = sw*second + 2*ds*first + dds*energy;
    return out;
}

Eigen::MatrixXd zbl_dense(const HessianMatrix &blocks) {
    Eigen::MatrixXd out(3*blocks.size,3*blocks.size);
    for (int i = 0; i < blocks.size; ++i)
        for (int j = 0; j < blocks.size; ++j) out.block<3,3>(3*i,3*j) = blocks(i,j);
    return out;
}

Eigen::MatrixXd zbl_force_fd(PotentialZBL &pot, const Atoms &atom, double h) {
    const int dimension = 3*atom.size();
    Eigen::MatrixXd out(dimension,dimension);
    for (int k = 0; k < dimension; ++k) {
        auto plus = atom, minus = atom;
        plus.pos_[k/3][k%3] += h;
        minus.pos_[k/3][k%3] -= h;
        Neighborhoods np(plus,pot.cutoff), nm(minus,pot.cutoff);
        pot.getEFS(plus,np); pot.getEFS(minus,nm);
        for (int i = 0; i < atom.size(); ++i)
            out.block<3,1>(3*i,k) = -(plus.forces_[i] - minus.forces_[i]) / (2*h);
    }
    return out;
}

} // namespace

TEST_CASE("ZBL force is the negative energy gradient", "[zbl]") {
    PotentialZBL pot(make_zbl_settings());
    constexpr double h = 1.0e-6;

    for (double distance : {0.8, 1.5}) {
        CAPTURE(distance);
        Atoms atom = efs_at(pot, distance);
        const double numeric_force =
            -(energy_at(pot, distance + h) - energy_at(pot, distance - h))
            / (2.0 * h);

        REQUIRE(atom.forces_[1](0)
                == Approx(numeric_force).epsilon(2.0e-6).margin(1.0e-8));
        REQUIRE(atom.forces_[0](0)
                == Approx(-numeric_force).epsilon(2.0e-6).margin(1.0e-8));
        REQUIRE(atom.forces_[0](1) == Approx(0.0).margin(1.0e-14));
        REQUIRE(atom.forces_[0](2) == Approx(0.0).margin(1.0e-14));

        // The code stores the unnormalized virial contribution.
        REQUIRE(atom.stress_(0, 0)
                == Approx(-atom.forces_[0](0) * distance)
                       .epsilon(2.0e-12).margin(1.0e-12));
        REQUIRE(atom.stress_(1, 1) == Approx(0.0).margin(1.0e-14));
        REQUIRE(atom.stress_(2, 2) == Approx(0.0).margin(1.0e-14));

        REQUIRE(atom.energy_
                == Approx(energy_at(pot, distance)).epsilon(1.0e-13));
    }
}

TEST_CASE("ZBL switches off continuously at its outer radius", "[zbl]") {
    PotentialZBL pot(make_zbl_settings());
    constexpr double h = 1.0e-7;

    Atoms just_inside = efs_at(pot, pot.r1 - h);
    Atoms at_cutoff = efs_at(pot, pot.r1);
    Atoms outside = efs_at(pot, pot.r1 + 0.1);

    REQUIRE(std::abs(just_inside.energy_) < 1.0e-10);
    REQUIRE(std::abs(just_inside.forces_[0](0)) < 1.0e-4);
    REQUIRE(at_cutoff.energy_ == Approx(0.0).margin(1.0e-14));
    REQUIRE(at_cutoff.forces_[0].norm() == Approx(0.0).margin(1.0e-14));
    REQUIRE(outside.energy_ == Approx(0.0).margin(1.0e-14));
    REQUIRE(outside.forces_[0].norm() == Approx(0.0).margin(1.0e-14));
}

TEST_CASE("ZBL validates species and switching radii", "[zbl]") {
    GPSettings missing_z;
    REQUIRE_THROWS_AS(PotentialZBL(missing_z), GPException);

    GPSettings fractional_z = make_zbl_settings();
    fractional_z.Modify("z", "13.5");
    REQUIRE_THROWS_AS(PotentialZBL(fractional_z), GPException);

    GPSettings zero_inner = make_zbl_settings();
    zero_inner.Modify("r0", "0.0");
    REQUIRE_THROWS_AS(PotentialZBL(zero_inner), GPException);

    GPSettings reversed = make_zbl_settings();
    reversed.Modify("r0", "2.0");
    reversed.Modify("r1", "1.0");
    REQUIRE_THROWS_AS(PotentialZBL(reversed), GPException);

    PotentialZBL pot(make_zbl_settings());
    Atoms invalid_species = make_dimer(1.0, 0, 1);
    Neighborhoods nbhs(invalid_species, pot.cutoff);
    REQUIRE_THROWS_AS(pot.getEFS(invalid_species, nbhs), GPException);
}

TEST_CASE("ZBL dense and matrix-free curvatures match an independent exponential oracle", "[zbl][hessian][hvp]") {
    const double r = GENERATE(.8, 1.5);
    PotentialZBL pot(make_zbl_settings());
    auto atom = make_dimer(r);
    const Eigen::Vector3d u = Eigen::Vector3d(1,2,3).normalized();
    atom.pos_[1] = r*u.transpose();
    const Eigen::Matrix3d axial = independent_zbl_curvature(r);
    const Eigen::Matrix3d curvature = axial(1,1)*Eigen::Matrix3d::Identity()
        + (axial(0,0)-axial(1,1))*u*u.transpose();
    Eigen::MatrixXd expected(6,6);
    expected << curvature,-curvature,-curvature,curvature;
    AnyPotential &base = pot;
    HessianMatrix blocks;
    Neighborhoods nbhs(atom,pot.cutoff);
    base.getEFSHessian(atom,nbhs,blocks);
    const Eigen::MatrixXd dense = zbl_dense(blocks);
    REQUIRE((dense-expected).norm() < 1e-9);
    REQUIRE((dense-dense.transpose()).norm() < 1e-11);
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(6);
        translation[axis] = translation[axis+3] = 1.;
        REQUIRE((dense*translation).norm() < 1e-11);
        Eigen::VectorXd zero;
        base.applyHessianVectorProduct(atom,nbhs,translation,zero);
        REQUIRE(zero.norm() < 1e-11);
    }
    Eigen::VectorXd udir(6),vdir(6),hu,hv;
    udir << .2,-.3,.5,-.1,.4,.7;
    vdir << -.4,.6,.1,.8,-.2,.3;
    base.applyHessianVectorProduct(atom,nbhs,udir,hu);
    base.applyHessianVectorProduct(atom,vdir,hv);
    REQUIRE((hu-expected*udir).norm() < 1e-9);
    REQUIRE((hv-expected*vdir).norm() < 1e-9);
    for (double h : {1e-6,5e-7}) {
        CAPTURE(r,h);
        const Eigen::MatrixXd fd = zbl_force_fd(pot,atom,h);
        REQUIRE((dense-fd).norm() < 2e-5);
        REQUIRE((hu-fd*udir).norm() < 2e-5);
    }
}

TEST_CASE("ZBL endpoint curvatures follow the existing C1 switch branches", "[zbl][hessian][hvp]") {
    const double r = GENERATE(1.,2.,2.1);
    PotentialZBL pot(make_zbl_settings());
    auto atom = make_dimer(r);
    HessianMatrix blocks;
    pot.getEFSHessian(atom,blocks);
    const Eigen::Matrix3d expected = independent_zbl_curvature(r);
    REQUIRE((blocks(0,0)-expected).norm() < 1e-9);
    REQUIRE((blocks(1,1)-expected).norm() < 1e-9);
    REQUIRE((blocks(0,1)+expected).norm() < 1e-9);
    REQUIRE((blocks(1,0)+expected).norm() < 1e-9);
    Eigen::VectorXd direction(6),product;
    direction << .2,.3,.4,-.5,.1,.7;
    pot.applyHessianVectorProduct(atom,direction,product);
    REQUIRE((product-zbl_dense(blocks)*direction).norm() < 1e-9);
    if (r >= 2.) REQUIRE(product.isZero(1e-12));
    // No central force FD at r0/r1: the cubic switch is only C1 there.
}

TEST_CASE("ZBL periodic self images cancel atomic curvature and repeated sources accumulate", "[zbl][hessian][hvp]") {
    PotentialZBL pot(make_zbl_settings());
    Atoms single;
    single.cell_ = Eigen::Vector3d(1.4,10,10).asDiagonal();
    single.pbc_ = {true,false,false}; single.pos_ = {{.1,.2,.3}}; single.chemmap_ = {0};
    Neighborhoods self(single,pot.cutoff);
    REQUIRE(self[0].nbhcount_ == 2);
    for (int source : self[0].indices_) REQUIRE(source == 0);
    HessianMatrix blocks;
    pot.getEFSHessian(single,self,blocks);
    REQUIRE(single.energy_ > 0.);
    REQUIRE(single.stress_.norm() > 1e-6);
    REQUIRE(zbl_dense(blocks).isZero(1e-12));
    Eigen::VectorXd direction(3),product;
    direction << .2,.4,-.1;
    pot.applyHessianVectorProduct(single,self,direction,product);
    REQUIRE(product.isZero(1e-12));

    Atoms pair = single;
    pair.pos_.push_back({.65,.4,.4}); pair.chemmap_.push_back(0);
    Neighborhoods nbhs(pair,pot.cutoff);
    bool repeated = false;
    for (int i = 0; i < pair.size(); ++i) {
        int others = 0;
        for (int source : nbhs[i].indices_) if (source != i) ++others;
        repeated |= others > 1;
    }
    REQUIRE(repeated);
    pot.getEFSHessian(pair,nbhs,blocks);
    const auto dense = zbl_dense(blocks);
    REQUIRE((dense-dense.transpose()).norm() < 1e-10);
    Eigen::VectorXd v(6);
    v << .2,-.3,.5,-.1,.4,.7;
    pot.applyHessianVectorProduct(pair,nbhs,v,product);
    REQUIRE((product-dense*v).norm() < 1e-9);
    for (double h : {1e-6,5e-7}) REQUIRE((dense-zbl_force_fd(pot,pair,h)).norm() < 2e-4);
}

TEST_CASE("ZBL dense EFS and matrix-free state contracts reach ComputeHessian", "[zbl][hessian][hvp]") {
    PotentialZBL pot(make_zbl_settings());
    auto atom = make_dimer(1.5);
    atom.pos_[1] = 1.5*Eigen::RowVector3d(1,2,3).normalized();
    Neighborhoods nbhs(atom,pot.cutoff);
    auto fresh = atom; pot.getEFS(fresh,nbhs);
    AnyPotential &base = pot;
    REQUIRE(base.supports_analytical_hessian());
    REQUIRE(base.supports_analytical_hessian_vector_product());
    REQUIRE(base.supports_analytical_cell_hessian());
    HessianMatrix blocks; blocks.resize(2);
    for (auto &block : blocks.hessian) block.setConstant(99.);
    base.getEFSHessian(atom,blocks);
    const auto dense = zbl_dense(blocks);
    REQUIRE(atom.energy_ == Approx(fresh.energy_).margin(1e-12));
    REQUIRE((atom.stress_-fresh.stress_).norm() < 1e-12);
    for (int i = 0; i < 2; ++i) {
        REQUIRE((atom.forces_[i]-fresh.forces_[i]).norm() < 1e-12);
        REQUIRE(atom.site_energies_[i] == Approx(fresh.site_energies_[i]).margin(1e-12));
    }
    auto accumulated = fresh;
    pot.add_efs = true;
    for (auto &block : blocks.hessian) block.setConstant(-7.);
    base.getEFSHessian(accumulated,nbhs,blocks);
    REQUIRE((zbl_dense(blocks)-dense).norm() < 1e-12);
    REQUIRE(accumulated.energy_ == Approx(2*fresh.energy_).margin(1e-12));
    REQUIRE((accumulated.stress_-2*fresh.stress_).norm() < 1e-12);
    for (int i = 0; i < 2; ++i) {
        REQUIRE((accumulated.forces_[i]-2*fresh.forces_[i]).norm() < 1e-12);
        REQUIRE(accumulated.site_energies_[i] == Approx(2*fresh.site_energies_[i]).margin(1e-12));
    }
    const auto source = atom;
    const auto numbers = pot.z;
    Eigen::VectorXd u(6),v(6),hu,hv,combined;
    u << .2,-.3,.5,-.1,.4,.7; v << -.4,.6,.1,.8,-.2,.3;
    base.applyHessianVectorProduct(atom,nbhs,u,hu);
    base.applyHessianVectorProduct(atom,v,hv);
    auto alias = u;
    base.applyHessianVectorProduct(atom,nbhs,alias,alias);
    REQUIRE((alias-hu).norm() < 1e-12);
    base.applyHessianVectorProduct(atom,2*u-.3*v,combined);
    REQUIRE((combined-(2*hu-.3*hv)).norm() < 1e-10);
    REQUIRE((hu-dense*u).norm() < 1e-10);
    pot.add_efs = false;
    for (const std::string method : {"auto","analytic"}) {
        ComputeHessian compute; compute.set_method(method);
        Eigen::MatrixXd result;
        compute.compute_fc(atom,result,&base);
        REQUIRE((result-dense).norm() < 1e-10);
        compute.compute_fc_blocks(atom,blocks,&base);
        REQUIRE((zbl_dense(blocks)-dense).norm() < 1e-10);
    }
    REQUIRE(atom.pos_ == source.pos_); REQUIRE(atom.cell_ == source.cell_);
    REQUIRE(atom.pbc_ == source.pbc_); REQUIRE(atom.cartesian == source.cartesian);
    REQUIRE(atom.chemmap_ == source.chemmap_); REQUIRE(atom.energy_ == source.energy_);
    REQUIRE(atom.forces_ == source.forces_); REQUIRE(atom.stress_ == source.stress_);
    REQUIRE(atom.site_energies_ == source.site_energies_);
    REQUIRE(atom.have_energy == source.have_energy); REQUIRE(atom.have_stress == source.have_stress);
    REQUIRE(pot.z == numbers); REQUIRE(pot.r0 == 1.); REQUIRE(pot.r1 == 2.);
}

TEST_CASE("Default energy-only OMP preserves ZBL force and stress data", "[zbl][energy][omp]") {
    const bool cached_neighbors = GENERATE(false, true);
    const bool additive = GENERATE(false, true);
    const bool have_stress = GENERATE(false, true);
    PotentialZBL pot(make_zbl_settings());
    AnyPotential &base = pot;
    auto expected = make_dimer(1.4);
    base.getE(expected);
    auto atom = make_dimer(1.4);
    atom.energy_ = 7.;
    atom.site_energies_ = {3., 4.};
    atom.have_energy = true;
    atom.forces_ = {{.7, -.4, .2}, {-.6, .3, .8}};
    atom.stress_ << 1., 2., 3., 4., 5., 6., 7., 8., 9.;
    atom.have_stress = have_stress;
    const auto forces = atom.forces_;
    const auto stress = atom.stress_;
    pot.add_efs = additive;
    if (cached_neighbors) {
        Neighborhoods neighbors(atom, pot.cutoff);
        base.getEOMP(atom, neighbors);
    } else base.getEOMP(atom);
    CHECK(atom.energy_ == Approx(expected.energy_ + (additive ? 7. : 0.)));
    REQUIRE(atom.site_energies_.size() == 2);
    for (int i = 0; i < 2; ++i)
        CHECK(atom.site_energies_[i] == Approx(expected.site_energies_[i] + (additive ? 3. + i : 0.)));
    CHECK(atom.forces_ == forces);
    CHECK(atom.stress_ == stress);
    // Energy-only calls follow getE's existing presence convention.
    CHECK_FALSE(atom.have_stress);
}
