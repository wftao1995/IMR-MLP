// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
#pragma once
#include <Eigen/Dense>

namespace SaddleCoordinates {
// Pull dE/dL back through C = L L^T. Components: xx, xy, xz, yy, yz, zz.
// L is either the lower Cholesky factor or the symmetric positive square root.
inline Eigen::Matrix<double, 6, 1> symmetric_pullback(
    const Eigen::Matrix3d &L, const Eigen::Matrix3d &S, bool restore_F_cholesky) {
    Eigen::Matrix<double, 6, 1> gradient;
    const Eigen::Matrix3d Linv = L.inverse();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
    if (!restore_F_cholesky) solver.compute(L);
    int index = 0;
    for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
            Eigen::Matrix3d basis = Eigen::Matrix3d::Zero(), dL;
            basis(a, b) = basis(b, a) = 1.0;
            if (restore_F_cholesky) {
                Eigen::Matrix3d phi = Linv * basis * Linv.transpose();
                phi.triangularView<Eigen::StrictlyUpper>().setZero();
                phi.diagonal() *= 0.5;
                dL = L * phi;
            } else {
                const auto &U = solver.eigenvectors();
                const auto &values = solver.eigenvalues();
                Eigen::Matrix3d rotated = U.transpose() * basis * U;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        rotated(i, j) /= values(i) + values(j);
                dL = U * rotated * U.transpose();
            }
            gradient(index++) = (S.array() * dL.array()).sum();
        }
    }
    return gradient;
}
}
