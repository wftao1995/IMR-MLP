// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "potentials/potential_mtp.h"
#include <set>
namespace NROptimizeAPP {
constexpr int VOIGT_SIZE = 6;
struct NROptions {
    bool nocell = false;
    int offset = VOIGT_SIZE;
    double eigzero_tol = 0.001;
};
void X2LatPos(const Eigen::VectorXd &, const Eigen::Matrix3d &, Atoms &, const NROptions &);
void LatPos2X(const Atoms &, const Eigen::Matrix3d &, Eigen::VectorXd &, const NROptions &);
void LatPos2G(const Atoms &, double, const Eigen::Matrix3d &, const Eigen::Matrix3d &, Eigen::VectorXd &, const NROptions &);
void compute_hessian(Atoms &, const Eigen::Matrix3d &, double, Potentials::PotMTP &, double,
                     Eigen::VectorXd &, Eigen::MatrixXd &, Eigen::MatrixXd &, const NROptions &);
void inverse_hessian(const Eigen::MatrixXd &, const Eigen::VectorXd &, const Eigen::MatrixXd &,
                     Eigen::MatrixXd &, const std::set<int> &, double);
}
