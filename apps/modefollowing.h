// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "potentials/potential_mtp.h"

class RelaxationDrvV2;

namespace AppsModeFollowing {
// Cartesian: [H (row major), r]; fractional: [C-I (xx yy zz yz xz xy), s],
// with H = H_ref chol(C). --nocell omits the cell block in either convention.
extern int offset;
extern bool car_rep, nocell;
void compute_hessian(Atoms &, const Eigen::Matrix3d &, double, Potentials::PotMTP &, double,
                     Eigen::VectorXd &, Eigen::MatrixXd &, Eigen::MatrixXd &);
Eigen::VectorXd gradient(const Atoms &, const Eigen::Matrix3d &, double);
void LatPos2X(Atoms &, Eigen::RowVectorXd &);
void X2LatPos(Atoms &, Eigen::RowVectorXd &);
void LatPos2x(Atoms &, Eigen::Matrix3d &, Eigen::RowVectorXd &);
void x2LatPos(Atoms &, Eigen::Matrix3d &, Eigen::RowVectorXd &);
void NugeMode(Atoms &, Eigen::MatrixXd &, Eigen::Matrix3d &, Eigen::Matrix3d &, int, double);
int FindMode(Eigen::MatrixXd &, Eigen::VectorXd &);
void init_relaxation(RelaxationDrvV2 &, Potentials::PotMTP &, Atoms &,
                     const Eigen::Matrix3d &, double);
Eigen::MatrixXd optimizer_modes(const Eigen::MatrixXd &);
}
