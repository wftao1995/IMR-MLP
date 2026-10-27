// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potentials/potential.h"

#include <Eigen/Dense>
#include <vector>

namespace HessianUtils {

Eigen::MatrixXd to_dense(const Potentials::HessianMatrix &hessian);
void from_dense(const Eigen::MatrixXd &dense, Potentials::HessianMatrix &hessian);
void symmetrize_blocks(Potentials::HessianMatrix &hessian);

Eigen::VectorXd sqrt_mass_coordinates(const std::vector<double> &masses);
Eigen::MatrixXd to_dynamical_matrix(const Eigen::MatrixXd &hessian,
                                    const std::vector<double> &masses);
Eigen::MatrixXd from_dynamical_matrix(const Eigen::MatrixXd &dynamical,
                                      const std::vector<double> &masses);
Eigen::MatrixXd mass_weighted_translation_basis(const std::vector<double> &masses);
Eigen::MatrixXd mass_weighted_internal_basis(const std::vector<double> &masses);
Eigen::MatrixXd acoustic_sum_rule_residual(const Eigen::MatrixXd &hessian);
double acoustic_sum_rule_error(const Eigen::MatrixXd &hessian);

} // namespace HessianUtils
