// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "tools/hessian_utils.h"

TEST_CASE("Dense and block Hessian conversion fills both triangles", "[hessian]")
{
    Eigen::MatrixXd dense = Eigen::MatrixXd::Random(6, 6);
    dense = 0.5 * (dense + dense.transpose().eval());
    Potentials::HessianMatrix blocks;
    HessianUtils::from_dense(dense, blocks);
    REQUIRE(blocks(1, 0).isApprox(blocks(0, 1).transpose()));
    REQUIRE(HessianUtils::to_dense(blocks).isApprox(dense));

    Potentials::HessianMatrix empty;
    empty.resize(0);
    REQUIRE(HessianUtils::to_dense(empty).size() == 0);
}

TEST_CASE("Block Hessian symmetrization averages transpose partners", "[hessian]")
{
    Potentials::HessianMatrix blocks;
    blocks.resize(2);
    blocks(0, 0) << 1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0;
    blocks(0, 1) = 2.0 * Eigen::Matrix3d::Identity();
    blocks(1, 0) = 4.0 * Eigen::Matrix3d::Identity();
    HessianUtils::symmetrize_blocks(blocks);
    const Eigen::MatrixXd dense = HessianUtils::to_dense(blocks);
    REQUIRE(dense.isApprox(dense.transpose(), 1e-14));
    REQUIRE(blocks(0, 1).isApprox(3.0 * Eigen::Matrix3d::Identity()));
}

TEST_CASE("Mass coordinates repeat each atomic square root", "[hessian][mass]")
{
    const Eigen::VectorXd roots = HessianUtils::sqrt_mass_coordinates({1.0, 4.0});
    REQUIRE(roots.segment<3>(0).isApprox(Eigen::Vector3d::Ones()));
    REQUIRE(roots.segment<3>(3).isApprox(2.0 * Eigen::Vector3d::Ones()));
}

TEST_CASE("Acoustic sum-rule residual retains Cartesian components", "[hessian][asr]")
{
    Eigen::Matrix3d spring;
    spring << 2.0, 0.2, 0.0,
              0.2, 3.0, 0.1,
              0.0, 0.1, 4.0;
    Eigen::MatrixXd hessian(6, 6);
    hessian << spring, -spring,
              -spring, spring;
    REQUIRE(HessianUtils::acoustic_sum_rule_residual(hessian).isZero(1e-14));
    hessian(0, 4) += 0.25;
    const Eigen::MatrixXd residual = HessianUtils::acoustic_sum_rule_residual(hessian);
    REQUIRE(residual(0, 1) == Catch::Approx(0.25));
    REQUIRE(HessianUtils::acoustic_sum_rule_error(hessian) == Catch::Approx(0.25));
}

TEST_CASE("Mass-weighted helpers reject inconsistent masses", "[hessian][mass]")
{
    REQUIRE_THROWS_AS(HessianUtils::sqrt_mass_coordinates({}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(HessianUtils::sqrt_mass_coordinates({1.0, 0.0}),
                      std::invalid_argument);
}
