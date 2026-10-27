// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "mindriver/dimer_new.h"
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace {
class SaddlePotential : public Potentials::AnyPotential {
public:
    SaddlePotential() { cutoff = 3; cutoffsq = 9; species_count = 1; }
    void getEFS(Atoms &atom, AnyNeighborhoods &) override {
        atom.resetEFS();
        for (int i = 0; i < atom.size(); ++i) {
            atom.energy_ -= atom.pos_[i].squaredNorm() / 2;
            atom.forces_[i] = atom.pos_[i].transpose();
        }
    }
};
}

TEST_CASE("Dimer convergence belongs to the current run", "[dimer]")
{
    SaddlePotential pot;
    DimerDriver driver;
    driver.pot = &pot;
    driver.max_iter = 3;
    driver.rot_max_iter = driver.trn_max_iter = 1;
    driver.summary = "";
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 20;
    atom.pos_ = {{0, 0, 0}};
    atom.chemmap_ = {0};
    driver.init(atom);
    REQUIRE(driver.run() == 0);
    REQUIRE(driver.converged);
    REQUIRE(driver.atom_center.energy_ == Catch::Approx(0));
    REQUIRE(driver.energy == Catch::Approx(driver.atom_center.energy_));

    SECTION("reinitialization clears results") {
        atom.pos_[0](0) = 1;
        driver.init(atom);
        REQUIRE_FALSE(driver.converged);
        REQUIRE(driver.iter == 0);
    }
    SECTION("running again clears convergence even without reinitialization") {
        driver.xx_center(0) = 1;
    }
    driver.max_iter = 1;
    REQUIRE(driver.run() == 1);
    REQUIRE_FALSE(driver.converged);
    REQUIRE(driver.max_trn_force > driver.dimer_gtol);
    const auto &final = driver.atom_center;
    REQUIRE(driver.energy == Catch::Approx(-final.pos_[0].squaredNorm() / 2));
    REQUIRE(final.energy_ == Catch::Approx(driver.energy));
    REQUIRE(final.forces_[0].isApprox(final.pos_[0].transpose()));
    std::filesystem::remove("dimer_modecar_final");
}

namespace {
class AnisotropicSaddlePotential : public Potentials::AnyPotential {
public:
    AnisotropicSaddlePotential() { cutoff = 3; cutoffsq = 9; species_count = 1; }
    void getEFS(Atoms &atom, AnyNeighborhoods &) override {
        atom.resetEFS();
        for (int i = 0; i < atom.size(); ++i) {
            Eigen::RowVector3d gradient = atom.pos_[i];
            gradient[0] *= -2; gradient[2] *= 3;
            atom.energy_ += .5 * atom.pos_[i].dot(gradient);
            atom.forces_[i] = -gradient.transpose();
        }
    }
};
struct DimerModeCleanup {
    ~DimerModeCleanup() { std::filesystem::remove("dimer_modecar_final"); }
};
Atoms first_step_atoms(double x) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 20;
    atom.pbc_ = {false,false,false};
    atom.pos_ = {{x,0,0}}; atom.chemmap_ = {0};
    return atom;
}
void one_step_dimer(DimerDriver &driver, Potentials::AnyPotential &pot) {
    driver.pot = &pot; driver.max_iter = 1;
    driver.rot_max_iter = driver.trn_max_iter = 1;
    driver.summary.clear();
}
}

TEST_CASE("Dimer FIRE translation consumes its first step setting and climbs physical energy", "[dimer][fire]") {
    const double first = GENERATE(.01,.004,.3);
    DimerModeCleanup cleanup;
    SaddlePotential pot; DimerDriver driver;
    one_step_dimer(driver,pot);
    driver.trn_driver = "fire"; driver.trn_first_step = first; driver.trn_max_step = .2;
    auto atom = first_step_atoms(1);
    driver.init(atom); driver.modedir << 1,0,0;
    REQUIRE(driver.run() == 1);
    const double cap = std::min(first,driver.trn_max_step);
    REQUIRE(driver.atom_center.pos_[0][0] == Catch::Approx(1-cap).margin(1e-12));
    REQUIRE(driver.atom_center.pos_[0][1] == Catch::Approx(0).margin(1e-12));
    // Translation reverses the negative mode: climbing toward the saddle is intentional.
    REQUIRE(driver.energy > -.5);
}

TEST_CASE("Dimer FIRE rotation consumes its first step setting", "[dimer][fire]") {
    const double first = GENERATE(.01,.004,.8);
    DimerModeCleanup cleanup;
    AnisotropicSaddlePotential pot; DimerDriver driver;
    one_step_dimer(driver,pot);
    driver.rot_driver = "fire"; driver.rot_first_step = first; driver.rot_max_step = .5;
    auto atom = first_step_atoms(0);
    driver.init(atom);
    driver.modedir << std::sqrt(.5),std::sqrt(.5),0;
    const Eigen::VectorXd initial = driver.modedir;
    driver.run();
    // Rotation updates the unconstrained vector, then normalizes it; its first
    // gradient is tangent to the unit direction, so tan(rotation)=step cap.
    const double parallel = initial.dot(driver.modedir);
    const double tangent = (driver.modedir-parallel*initial).norm();
    REQUIRE(tangent/parallel == Catch::Approx(std::min(first,driver.rot_max_step)).margin(1e-10));
    REQUIRE(driver.curvature < -.5);
    REQUIRE(driver.atom_center.pos_[0].norm() == Catch::Approx(0).margin(1e-12));
}

TEST_CASE("Dimer first step bridge preserves CG and LBFGS translation semantics", "[dimer]") {
    const std::string optimizer = GENERATE("cg","lbfgs");
    DimerModeCleanup cleanup;
    SaddlePotential pot;
    Eigen::RowVector3d result;
    for (int i = 0; i < 2; ++i) {
        DimerDriver driver; one_step_dimer(driver,pot);
        driver.trn_driver = optimizer;
        driver.trn_first_step = i == 0 ? .004 : .1;
        auto atom = first_step_atoms(1);
        driver.init(atom); driver.modedir << 1,0,0;
        driver.run();
        if (i == 0) result = driver.atom_center.pos_[0];
        else REQUIRE(driver.atom_center.pos_[0].isApprox(result,1e-12));
    }
    REQUIRE(result[0] < 1);
}
