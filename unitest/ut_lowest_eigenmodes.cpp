// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "analyse/lowest_eigenmodes.h"

#include <Eigen/Eigenvalues>

using Catch::Approx;

TEST_CASE("Matrix-free lowest modes match dense eigenvalues with a projector",
          "[hvp][eigenmodes]")
{
    constexpr int dimension = 12;
    Eigen::MatrixXd random = Eigen::MatrixXd::Random(dimension, dimension);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(random);
    const Eigen::MatrixXd rotation = qr.householderQ() *
        Eigen::MatrixXd::Identity(dimension, dimension);
    Eigen::VectorXd spectrum(dimension);
    spectrum << -8.0, -3.0, -1.0, 0.25, 0.5, 1.0,
                2.0, 3.0, 5.0, 8.0, 13.0, 21.0;
    const Eigen::MatrixXd matrix =
        rotation * spectrum.asDiagonal() * rotation.transpose();
    const Eigen::VectorXd excluded = rotation.col(0);

    MatrixFree::LowestEigenmodesSettings settings;
    settings.count = 3;
    settings.tolerance = 1e-10;
    settings.max_matvecs = 200;
    auto result = MatrixFree::solve_lowest_symmetric(
        dimension,
        [&](const Eigen::VectorXd &input, Eigen::VectorXd &output) {
            output = matrix * input;
            output.noalias() -= excluded * excluded.dot(output);
        },
        settings,
        [&](Eigen::VectorXd &value) {
            value.noalias() -= excluded * excluded.dot(value);
        });

    REQUIRE(result.converged);
    REQUIRE(result.eigenvalues(0) == Approx(-3.0).margin(1e-9));
    REQUIRE(result.eigenvalues(1) == Approx(-1.0).margin(1e-9));
    REQUIRE(result.eigenvalues(2) == Approx(0.25).margin(1e-9));
    REQUIRE(result.residual_norms.maxCoeff() < 1e-9);
    for (int mode = 0; mode < settings.count; ++mode)
        REQUIRE(std::abs(excluded.dot(result.eigenvectors.col(mode))) < 1e-10);
}

TEST_CASE("Matrix-free lowest modes report a matvec-limit failure",
          "[hvp][eigenmodes]")
{
    constexpr int dimension = 40;
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(dimension, dimension);
    matrix = 0.5 * (matrix + matrix.transpose());
    MatrixFree::LowestEigenmodesSettings settings;
    settings.count = 2;
    settings.tolerance = 1e-14;
    settings.max_matvecs = 8;
    const auto result = MatrixFree::solve_lowest_symmetric(
        dimension,
        [&](const Eigen::VectorXd &input, Eigen::VectorXd &output) {
            output = matrix * input;
        },
        settings);
    REQUIRE_FALSE(result.converged);
    REQUIRE(result.matvecs == 8);
}
