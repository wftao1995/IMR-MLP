// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace ElasticUtils {

Eigen::Matrix<double, 9, 9> voigt_to_tensor(const Eigen::Matrix<double, 6, 6> &voigt);
Eigen::Matrix<double, 6, 6> tensor_to_voigt(const Eigen::Matrix<double, 9, 9> &tensor);

Eigen::Matrix<double, 6, 6> project_by_symmetry(
    const Eigen::Matrix<double, 6, 6> &stiffness,
    const Eigen::Matrix3d &cell,
    const std::vector<Eigen::Matrix3d> &fractional_rotations);

Eigen::Matrix<double, 6, 6> born_stiffness(
    const Eigen::Matrix<double, 6, 6> &material_stiffness,
    double hydrostatic_pressure_gpa);

Eigen::Matrix<double, 6, 6> compliance(
    const Eigen::Matrix<double, 6, 6> &stiffness);

} // namespace ElasticUtils
