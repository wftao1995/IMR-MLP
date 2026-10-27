// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "lowest_eigenmodes.h"

#include "utils/gputils.h"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace MatrixFree {
namespace {

bool append_orthonormal(Eigen::MatrixXd &basis, Eigen::VectorXd candidate,
                        const VectorProjector &project)
{
    if (project)
        project(candidate);
    for (int pass = 0; pass < 2; ++pass) {
        if (basis.cols() != 0)
            candidate.noalias() -= basis * (basis.transpose() * candidate);
        if (project)
            project(candidate);
    }
    const double norm = candidate.norm();
    if (!std::isfinite(norm) || norm <= 1.0e-12)
        return false;
    candidate /= norm;
    const int old_columns = basis.cols();
    basis.conservativeResize(Eigen::NoChange, old_columns + 1);
    basis.col(old_columns) = candidate;
    return true;
}

void append_operator_columns(const Eigen::MatrixXd &basis, int first_column,
                             Eigen::MatrixXd &applied,
                             const SymmetricOperator &apply, int &matvecs)
{
    applied.conservativeResize(Eigen::NoChange, basis.cols());
    for (int column = first_column; column < basis.cols(); ++column) {
        Eigen::VectorXd value;
        apply(basis.col(column), value);
        if (value.size() != basis.rows() || !value.allFinite())
            GPERROR("matrix-free operator returned an invalid vector");
        applied.col(column) = value;
        ++matvecs;
    }
}

} // namespace

LowestEigenmodesResult solve_lowest_symmetric(
    int dimension, const SymmetricOperator &apply,
    const LowestEigenmodesSettings &settings,
    const VectorProjector &project)
{
    if (dimension <= 0 || settings.count <= 0 || settings.count > dimension)
        GPERROR("invalid matrix-free eigensolver dimensions");
    if (!std::isfinite(settings.tolerance) || settings.tolerance <= 0.0)
        GPERROR("matrix-free eigensolver tolerance must be finite and positive");
    if (settings.max_matvecs < settings.count)
        GPERROR("matrix-free eigensolver matvec limit is too small");

    const int wanted = settings.count;
    const int default_subspace = std::max(4 * wanted + 24, 64);
    const int max_subspace = std::min(
        dimension, settings.max_subspace > 0
                       ? settings.max_subspace
                       : default_subspace);
    if (max_subspace < wanted)
        GPERROR("matrix-free eigensolver subspace is smaller than mode count");

    std::mt19937 generator(settings.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::MatrixXd basis(dimension, 0);
    Eigen::MatrixXd applied(dimension, 0);
    const int initial_columns = std::min(max_subspace,
                                         std::max(wanted + 4, 8));
    for (int column = 0; column < initial_columns; ++column) {
        Eigen::VectorXd candidate(dimension);
        for (int i = 0; i < dimension; ++i)
            candidate(i) = normal(generator);
        append_orthonormal(basis, std::move(candidate), project);
    }
    if (basis.cols() < wanted)
        GPERROR("matrix-free projector leaves too few independent coordinates");

    LowestEigenmodesResult result;
    append_operator_columns(basis, 0, applied, apply, result.matvecs);

    while (true) {
        Eigen::MatrixXd projected = basis.transpose() * applied;
        projected = 0.5 * (projected + projected.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(projected);
        if (solver.info() != Eigen::Success)
            GPERROR("matrix-free Rayleigh-Ritz diagonalization failed");

        const Eigen::MatrixXd coefficients = solver.eigenvectors().leftCols(wanted);
        result.eigenvalues = solver.eigenvalues().head(wanted);
        result.eigenvectors = basis * coefficients;
        const Eigen::MatrixXd ax = applied * coefficients;
        Eigen::MatrixXd residuals = ax -
            result.eigenvectors * result.eigenvalues.asDiagonal();
        result.residual_norms.resize(wanted);
        result.converged_count = 0;
        for (int mode = 0; mode < wanted; ++mode) {
            if (project) {
                Eigen::VectorXd residual = residuals.col(mode);
                project(residual);
                residuals.col(mode) = residual;
            }
            result.residual_norms(mode) = residuals.col(mode).norm();
            const double scale = std::max(1.0, std::abs(result.eigenvalues(mode)));
            if (result.residual_norms(mode) <= settings.tolerance * scale)
                ++result.converged_count;
        }
        if (result.converged_count == wanted) {
            result.converged = true;
            return result;
        }
        if (result.matvecs >= settings.max_matvecs)
            return result;

        if (basis.cols() + wanted > max_subspace) {
            const int retained = std::min(
                static_cast<int>(solver.eigenvectors().cols()),
                std::min(max_subspace - 1, std::max(wanted + 4, 2 * wanted)));
            const Eigen::MatrixXd keep = solver.eigenvectors().leftCols(retained);
            basis = basis * keep;
            applied = applied * keep;
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(basis);
            const Eigen::MatrixXd transform = qr.householderQ() *
                Eigen::MatrixXd::Identity(dimension, retained);
            // Reapplying after the QR keeps A*V paired exactly with the new
            // orthonormal basis and is cheap compared with an unstable
            // coordinate reconstruction at a restart.
            basis = transform;
            applied.resize(dimension, 0);
            if (result.matvecs + retained > settings.max_matvecs)
                return result;
            append_operator_columns(basis, 0, applied, apply, result.matvecs);
        }

        const int first_new = basis.cols();
        for (int mode = 0; mode < wanted && basis.cols() < max_subspace;
             ++mode) {
            if (result.residual_norms(mode) >
                settings.tolerance *
                    std::max(1.0, std::abs(result.eigenvalues(mode))))
                append_orthonormal(basis, residuals.col(mode), project);
        }
        if (basis.cols() == first_new) {
            Eigen::VectorXd candidate(dimension);
            for (int i = 0; i < dimension; ++i)
                candidate(i) = normal(generator);
            append_orthonormal(basis, std::move(candidate), project);
        }
        const int new_columns = basis.cols() - first_new;
        if (new_columns <= 0 || result.matvecs + new_columns > settings.max_matvecs)
            return result;
        append_operator_columns(basis, first_new, applied, apply, result.matvecs);
    }
}

} // namespace MatrixFree
