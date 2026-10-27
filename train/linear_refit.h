// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "efs_fit_weights.h"
#include "hvp_fit.h"

struct LinearRefitSettings {
    std::string mode="off", solver="qr_svd", report="linear-refit.json";
    Training::HVPFitSettings hvp, hessian;
    // Optional total-charge observation. Trainer callbacks share its target and
    // weight semantics; advertised blocks are independent of total charge.
    std::function<double(const Atoms&)> charge_weight, charge_target;
    double rtol=1e-12, atol=0., acceptance_tolerance=1e-10;
};

struct LinearRefitResult {
    int parameters=0, composition_rank=0, data_rank=0, solve_rank=0;
    std::vector<int> parameter_indices;
    std::vector<std::string> blocks;
    double before=0., after=0., predicted=0., optimality=0.;
    double data_before=0., data_after=0., penalty_before=0., penalty_after=0.;
    double charge_before=0., charge_after=0.;
    double assembly_seconds=0., solve_seconds=0., verification_seconds=0.;
    Eigen::VectorXd delta, scales, singular_values, l2;
};

// Data are already distributed across mpi.comm. Updates only selected physical
// coefficients, and restores them if collective assembly/verification fails.
LinearRefitResult linear_refit(Potentials::AnyPotential& potential,
    std::vector<Atoms>& reference, std::vector<Neighborhoods>& neighborhoods,
    const EFSFitWeights& weights, double penalty_weight,
    const LinearRefitSettings& settings);
