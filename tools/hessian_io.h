// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Dense>
#include <string>

namespace HessianIO {

Eigen::MatrixXd read_force_constants(const std::string &filename, int natoms);
Eigen::MatrixXd read_vasprun_hessian(const std::string &filename, int natoms,
                                     const std::string &key = "auto");

}
