// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef _IMRMLP_MONKOWSKI_REDUCE_H
#define _IMRMLP_MONKOWSKI_REDUCE_H

#include <Eigen/Dense>
namespace Symmetry3D
{
    bool minkowski_conditions_check(const Eigen::Matrix3d &basis, double prec);
    // Assumed basis is full ranked 3 order matrix
    // This will change basis
    bool minkowski_reduce_basis(Eigen::Matrix3d &basis, double prec, int max_iter);
    bool gaussian_reduce_two_vectors(Eigen::Vector3d &U, Eigen::Vector3d &V, double prec, int max_iter);
    bool reduce_C_in_ABC(Eigen::Matrix3d &basis, double prec);
} // Symmetry3D



#endif