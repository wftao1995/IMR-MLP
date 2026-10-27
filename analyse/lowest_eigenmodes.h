// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Core>

#include <functional>

namespace MatrixFree {

struct LowestEigenmodesSettings {
    int count = 1;
    double tolerance = 1.0e-8;
    int max_matvecs = 400;
    int max_subspace = 0;
    unsigned int seed = 5489;
};

struct LowestEigenmodesResult {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
    Eigen::VectorXd residual_norms;
    int matvecs = 0;
    int converged_count = 0;
    bool converged = false;
};

using SymmetricOperator =
    std::function<void(const Eigen::VectorXd &, Eigen::VectorXd &)>;
using VectorProjector = std::function<void(Eigen::VectorXd &)>;

LowestEigenmodesResult solve_lowest_symmetric(
    int dimension, const SymmetricOperator &apply,
    const LowestEigenmodesSettings &settings,
    const VectorProjector &project = {});

} // namespace MatrixFree
