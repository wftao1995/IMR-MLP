// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "tools/hessian_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace HessianUtils {
namespace {

void validate_dense_hessian(const Eigen::MatrixXd &hessian)
{
    if (hessian.rows() == 0 || hessian.rows() != hessian.cols() || hessian.rows() % 3 != 0)
        throw std::invalid_argument("Hessian must be a non-empty square 3N x 3N matrix");
    if (!hessian.allFinite())
        throw std::invalid_argument("Hessian contains non-finite values");
}

void validate_masses(const std::vector<double> &masses, Eigen::Index natoms)
{
    if (static_cast<Eigen::Index>(masses.size()) != natoms)
        throw std::invalid_argument("Mass count does not match Hessian atom count");
    for (double mass : masses) {
        if (!std::isfinite(mass) || mass <= 0.0)
            throw std::invalid_argument("All atomic masses must be finite and positive");
    }
}

} // namespace

Eigen::MatrixXd to_dense(const Potentials::HessianMatrix &hessian)
{
    if (hessian.size < 0 ||
        hessian.hessian.size() != static_cast<size_t>(hessian.size * hessian.size))
        throw std::invalid_argument("Invalid block Hessian storage");
    Eigen::MatrixXd dense(3 * hessian.size, 3 * hessian.size);
    for (int i = 0; i < hessian.size; ++i)
        for (int j = 0; j < hessian.size; ++j)
            dense.block<3, 3>(3 * i, 3 * j) = hessian(i, j);
    return dense;
}

void from_dense(const Eigen::MatrixXd &dense, Potentials::HessianMatrix &hessian)
{
    validate_dense_hessian(dense);
    const int natoms = static_cast<int>(dense.rows() / 3);
    const Eigen::MatrixXd symmetric = 0.5 * (dense + dense.transpose());
    hessian.resize(natoms);
    for (int i = 0; i < natoms; ++i) {
        for (int j = i; j < natoms; ++j) {
            const Eigen::Matrix3d block = symmetric.block<3, 3>(3 * i, 3 * j);
            hessian(i, j) = block;
            hessian(j, i) = block.transpose();
        }
    }
}

void symmetrize_blocks(Potentials::HessianMatrix &hessian)
{
    if (hessian.size < 0 ||
        hessian.hessian.size() != static_cast<size_t>(hessian.size * hessian.size))
        throw std::invalid_argument("Invalid block Hessian storage");
    for (int i = 0; i < hessian.size; ++i) {
        for (int j = i; j < hessian.size; ++j) {
            const Eigen::Matrix3d block = 0.5 *
                (hessian(i, j) + hessian(j, i).transpose());
            hessian(i, j) = block;
            hessian(j, i) = block.transpose();
        }
    }
}

Eigen::VectorXd sqrt_mass_coordinates(const std::vector<double> &masses)
{
    if (masses.empty())
        throw std::invalid_argument("Cannot construct mass coordinates without atoms");
    validate_masses(masses, static_cast<Eigen::Index>(masses.size()));
    Eigen::VectorXd roots(3 * static_cast<Eigen::Index>(masses.size()));
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(masses.size()); ++i)
        roots.segment<3>(3 * i).setConstant(std::sqrt(masses[i]));
    return roots;
}

Eigen::MatrixXd to_dynamical_matrix(const Eigen::MatrixXd &hessian,
                                    const std::vector<double> &masses)
{
    validate_dense_hessian(hessian);
    validate_masses(masses, hessian.rows() / 3);
    const Eigen::VectorXd roots = sqrt_mass_coordinates(masses);
    Eigen::MatrixXd result = 0.5 * (hessian + hessian.transpose());
    result.array().colwise() /= roots.array();
    result.array().rowwise() /= roots.transpose().array();
    return result;
}

Eigen::MatrixXd from_dynamical_matrix(const Eigen::MatrixXd &dynamical,
                                      const std::vector<double> &masses)
{
    validate_dense_hessian(dynamical);
    validate_masses(masses, dynamical.rows() / 3);
    const Eigen::VectorXd roots = sqrt_mass_coordinates(masses);
    Eigen::MatrixXd result = 0.5 * (dynamical + dynamical.transpose());
    result.array().colwise() *= roots.array();
    result.array().rowwise() *= roots.transpose().array();
    return 0.5 * (result + result.transpose());
}

Eigen::MatrixXd mass_weighted_translation_basis(const std::vector<double> &masses)
{
    if (masses.empty())
        throw std::invalid_argument("Cannot construct translations without atoms");
    validate_masses(masses, static_cast<Eigen::Index>(masses.size()));
    const Eigen::Index dim = 3 * static_cast<Eigen::Index>(masses.size());
    Eigen::MatrixXd translations = Eigen::MatrixXd::Zero(dim, 3);
    double total_mass = 0.0;
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(masses.size()); ++i) {
        total_mass += masses[i];
        const double root_mass = std::sqrt(masses[i]);
        for (int axis = 0; axis < 3; ++axis)
            translations(3 * i + axis, axis) = root_mass;
    }
    translations /= std::sqrt(total_mass);
    return translations;
}

Eigen::MatrixXd mass_weighted_internal_basis(const std::vector<double> &masses)
{
    const Eigen::Index dim = 3 * static_cast<Eigen::Index>(masses.size());
    const Eigen::MatrixXd translations = mass_weighted_translation_basis(masses);
    if (dim == 3)
        return Eigen::MatrixXd(3, 0);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(translations);
    return (qr.householderQ() * Eigen::MatrixXd::Identity(dim, dim)).rightCols(dim - 3);
}

Eigen::MatrixXd acoustic_sum_rule_residual(const Eigen::MatrixXd &hessian)
{
    validate_dense_hessian(hessian);
    const Eigen::Index natoms = hessian.rows() / 3;
    Eigen::MatrixXd residual = Eigen::MatrixXd::Zero(3 * natoms, 3);
    for (Eigen::Index i = 0; i < natoms; ++i)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (Eigen::Index j = 0; j < natoms; ++j)
                    residual(3 * i + a, b) += hessian(3 * i + a, 3 * j + b);
    return residual;
}

double acoustic_sum_rule_error(const Eigen::MatrixXd &hessian)
{
    return acoustic_sum_rule_residual(hessian).cwiseAbs().maxCoeff();
}

} // namespace HessianUtils
