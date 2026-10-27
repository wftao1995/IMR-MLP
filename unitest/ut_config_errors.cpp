// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "active_learn/al_potential.h"
#include "mindriver/relax_new.h"
#include "potentials/potential_lj.h"
#include "potentials/potential_mindist.h"
#include "train/train_linear_potential.h"
#include "train/train_mtp.h"
#include "train/train_nonlinear_potential.h"
#include <filesystem>
#include <fstream>
#include <sstream>

TEST_CASE("Training constructors propagate invalid settings", "[train][errors]") {
    GPSettings settings;
    settings.Modify("energy_weight", "bad");
    auto message = Catch::Matchers::ContainsSubstring("energy_weight");
    REQUIRE_THROWS_WITH(TrainerLinearPotential{settings}, message);
    REQUIRE_THROWS_WITH(Trainer_MTP{settings}, message);
    REQUIRE_THROWS_WITH(TrainNonlinear{settings}, message);

    settings.Modify("energy_weight", "2.5");
    REQUIRE_NOTHROW(TrainerLinearPotential{settings});
    REQUIRE_NOTHROW(Trainer_MTP{settings});
    REQUIRE_NOTHROW(TrainNonlinear{settings});
}

TEST_CASE("Relaxation stiffness requires exactly six components", "[relax_new][errors]") {
    GPSettings settings;
    settings.Modify("dprecond", "true");
    settings.Modify("relax_cell", "true");
    settings.Modify("cartesian", "false");
    settings.Modify("symm_strain", "true");
    for (auto value : {"100", "100 100 100 100 100", "100 100 100 100 100 100 100"}) {
        settings.Modify("stiffness", value);
        REQUIRE_THROWS_WITH(RelaxationDrvV2{settings}, Catch::Matchers::ContainsSubstring("stiffness"));
    }
    settings.Modify("stiffness", "100 100 100 100 100 100");
    REQUIRE_NOTHROW(RelaxationDrvV2{settings});
}

TEST_CASE("LJ pairs require epsilon and sigma", "[lj][errors]") {
    GPSettings settings;
    settings.Modify("ntypes", "1");
    settings.Modify("cut:0-0", "3");
    for (auto value : {"", "1", "1 2 3"}) {
        settings.Modify("0-0", value);
        REQUIRE_THROWS_WITH(Potentials::LJ612{settings}, Catch::Matchers::ContainsSubstring("0-0"));
    }
    settings.Modify("0-0", "1 2");
    REQUIRE_NOTHROW(Potentials::LJ612{settings});
}

TEST_CASE("Unsupported potential operations fail before producing results", "[potential][errors]") {
    Potentials::PotMDIST model(2.0);
    Potentials::AnyPotential &potential = model;
    Atoms atoms;
    AnyNeighborhoods neighbors;
    Neighborhood neighborhood;
    Potentials::HessianMatrix hessian;
    hessian.resize(1);
    hessian(0, 0).setConstant(17.0);
    Eigen::MatrixXd cell = Eigen::MatrixXd::Constant(1, 1, 19.0), mixed = cell;
    auto unsupported = Catch::Matchers::ContainsSubstring("support");
    REQUIRE_FALSE(potential.supports_analytical_hessian());
    REQUIRE_THROWS_WITH(potential.getEFSHessian(atoms, hessian), unsupported);
    REQUIRE_THROWS_WITH(potential.getEFSHessian(atoms, neighbors, hessian), unsupported);
    REQUIRE_THROWS_WITH(potential.getEFSHessianOMP(atoms, neighbors, hessian), unsupported);
    REQUIRE_THROWS_WITH(potential.getEFSHessianSymmOMP(atoms, hessian), unsupported);
    REQUIRE_THROWS_WITH(potential.getEFSHessianCellSymmOMP(atoms, hessian, cell, mixed), unsupported);
    REQUIRE(hessian(0, 0).isConstant(17.0));
    REQUIRE(cell(0, 0) == 19.0);
    REQUIRE(mixed(0, 0) == 19.0);
    double descriptor = 23.0, gradient = 29.0;
    REQUIRE_THROWS_WITH(potential.linear_compute_descriptor(neighborhood, &descriptor), unsupported);
    REQUIRE_THROWS_WITH(potential.linear_compute_descriptor_grad(neighborhood, &descriptor, &gradient), unsupported);
    REQUIRE_THROWS_WITH(potential.AccumulateFPGard(neighborhood, &gradient, 1.0, nullptr), unsupported);
    REQUIRE_THROWS_WITH(potential.predicted_total_charge(atoms), unsupported);
    REQUIRE_THROWS_WITH(potential.AccumulateTotalChargeGard(atoms, neighbors, 1.0, &gradient), unsupported);
    REQUIRE(descriptor == 23.0);
    REQUIRE(gradient == 29.0);
}

TEST_CASE("Potentials without stream serialization reject load and save", "[potential][errors]") {
    GPSettings settings;
    settings.Modify("ntypes", "1");
    settings.Modify("0-0", "1 2");
    struct NoSerialization : Potentials::AnyPotential {
        void getEFS(Atoms &, AnyNeighborhoods &) override {}
    } model;
    Potentials::AnyPotential &potential = model;
    std::stringstream stream;
    auto unsupported = Catch::Matchers::ContainsSubstring("support");
    REQUIRE_THROWS_WITH(potential.Load(stream), unsupported);
    REQUIRE_THROWS_WITH(potential.Save(stream), unsupported);
    REQUIRE(stream.str().empty());
}
