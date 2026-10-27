// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "apps/fastfit_utils.h"

#include <fstream>

using Catch::Approx;
using namespace func_fastfit::detail;

TEST_CASE("fastfit removes the requested row and column", "[fastfit]") {
    Eigen::MatrixXd matrix(4, 4);
    matrix << 11, 12, 13, 14,
              12, 22, 23, 24,
              13, 23, 33, 34,
              14, 24, 34, 44;
    Eigen::VectorXd vector(4);
    vector << 1, 2, 3, 4;

    for (int removed = 0; removed < 4; ++removed) {
        Eigen::MatrixXd reduced;
        Eigen::VectorXd reduced_vector;
        RemoveOneBasis(matrix, vector, removed, reduced, reduced_vector);

        std::vector<int> keep;
        for (int i = 0; i < 4; ++i) if (i != removed) keep.push_back(i);
        REQUIRE(reduced.rows() == 3);
        REQUIRE(reduced.isApprox(reduced.transpose()));
        for (int i = 0; i < 3; ++i) {
            REQUIRE(reduced_vector(i) == vector(keep[i]));
            for (int j = 0; j < 3; ++j)
                REQUIRE(reduced(i, j) == matrix(keep[i], keep[j]));
        }
    }
}

TEST_CASE("fastfit scalar-id files are strict and round trip", "[fastfit]") {
    const std::set<int> expected{0, 2, 4};
    WriteScalarIds("fastfit-selected.dat", expected);
    REQUIRE(ReadScalarIds("fastfit-selected.dat", 5) == expected);

    SECTION("duplicates are rejected") {
        std::ofstream output("fastfit-invalid.dat");
        output << "2 1 1\n";
        output.close();
        REQUIRE_THROWS(ReadScalarIds("fastfit-invalid.dat", 5));
    }
    SECTION("shifted or out-of-range ids are rejected") {
        std::ofstream output("fastfit-invalid.dat");
        output << "1 5\n";
        output.close();
        REQUIRE_THROWS(ReadScalarIds("fastfit-invalid.dat", 5));
    }
    SECTION("trailing records are rejected") {
        std::ofstream output("fastfit-invalid.dat");
        output << "1 2 3\n";
        output.close();
        REQUIRE_THROWS(ReadScalarIds("fastfit-invalid.dat", 5));
    }
}

TEST_CASE("fastfit validates selected and removed partitions", "[fastfit]") {
    REQUIRE_NOTHROW(ValidatePartition({0, 2}, {1, 3}, 4));
    REQUIRE_THROWS(ValidatePartition({0, 1}, {1, 2, 3}, 4));
    REQUIRE_THROWS(ValidatePartition({0}, {2, 3}, 4));
    REQUIRE_THROWS(ValidatePartition({0, 2, 4}, {1, 3}, 4));
}

TEST_CASE("fastfit allocates a global descriptor sampling budget", "[fastfit]") {
    REQUIRE(AllocateSampleQuotas({2, 3}, 10) == std::vector<int>{2, 3});
    const auto quotas = AllocateSampleQuotas({100, 50, 50}, 50);
    REQUIRE(quotas == std::vector<int>{25, 13, 12});
    REQUIRE(quotas[0] + quotas[1] + quotas[2] == 50);
    REQUIRE_THROWS(AllocateSampleQuotas({1, 2}, 0));
}

TEST_CASE("fastfit selects the normalized Pareto knee", "[fastfit]") {
    const std::vector<std::pair<double, double>> objectives{
        {1.0, 10.0}, {2.0, 4.0}, {10.0, 1.0}
    };
    REQUIRE(SelectKneePoint(objectives) == 1);
}

TEST_CASE("fastfit crowding distances preserve individual identity", "[fastfit]") {
    const std::vector<std::pair<double, double>> objectives{
        {5.0, 1.0}, {1.0, 5.0}, {3.0, 3.0}, {2.0, 4.0}
    };
    const auto distances = CrowdingDistances(objectives);
    REQUIRE(std::isinf(distances[0]));
    REQUIRE(std::isinf(distances[1]));
    REQUIRE(distances[2] == Approx(1.5));
    REQUIRE(distances[3] == Approx(1.0));
}
