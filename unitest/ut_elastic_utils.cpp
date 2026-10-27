// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "tools/elastic_utils.h"

using Catch::Approx;

TEST_CASE("Hydrostatic pressure produces the standard Born correction", "[elastic]")
{
    Eigen::Matrix<double, 6, 6> material = Eigen::Matrix<double, 6, 6>::Zero();
    material.diagonal().setConstant(100.0);
    material(0, 1) = material(1, 0) = 40.0;
    const auto born = ElasticUtils::born_stiffness(material, 5.0);
    REQUIRE(born(0, 0) == Approx(95.0));
    REQUIRE(born(3, 3) == Approx(95.0));
    REQUIRE(born(0, 1) == Approx(45.0));
}

TEST_CASE("Cubic tensor is invariant under axis permutation projection", "[elastic][symmetry]")
{
    Eigen::Matrix<double, 6, 6> cubic = Eigen::Matrix<double, 6, 6>::Zero();
    cubic(0, 0) = cubic(1, 1) = cubic(2, 2) = 200.0;
    cubic(0, 1) = cubic(1, 0) = cubic(0, 2) = cubic(2, 0)
        = cubic(1, 2) = cubic(2, 1) = 120.0;
    cubic(3, 3) = cubic(4, 4) = cubic(5, 5) = 80.0;
    Eigen::Matrix3d permutation;
    permutation << 0, 1, 0,
                   0, 0, 1,
                   1, 0, 0;
    const auto projected = ElasticUtils::project_by_symmetry(
        cubic, Eigen::Matrix3d::Identity(), {Eigen::Matrix3d::Identity(), permutation, permutation * permutation});
    REQUIRE(projected.isApprox(cubic, 1e-12));
}

TEST_CASE("Compliance is the inverse stiffness", "[elastic]")
{
    Eigen::Matrix<double, 6, 6> stiffness = Eigen::Matrix<double, 6, 6>::Identity() * 100.0;
    stiffness(0, 1) = stiffness(1, 0) = 20.0;
    const auto inverse = ElasticUtils::compliance(stiffness);
    REQUIRE((stiffness * inverse).isApprox(Eigen::Matrix<double, 6, 6>::Identity(), 1e-12));

    stiffness.setZero();
    REQUIRE_THROWS_AS(ElasticUtils::compliance(stiffness), std::runtime_error);
}
