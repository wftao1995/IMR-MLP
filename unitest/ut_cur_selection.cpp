// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "maxvol/cur_selection.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

using Catch::Approx;

namespace {
Eigen::MatrixXd descriptor_matrix() {
    // Orthogonal feature columns with squared singular values 18, 4, 1.
    // At k=2 the leverage scores are exactly {1/4,1/2,0,1/4,0}.
    Eigen::MatrixXd matrix(5,3);
    matrix << 3,0,0, 0,2,0, 0,0,1, 3,0,0, 0,0,0;
    return matrix;
}

void check_selection(const CURResult &result, int rows, int selected) {
    REQUIRE(result.selected_indices.size() == static_cast<size_t>(selected));
    REQUIRE(result.scores.size() == static_cast<size_t>(rows));
    std::set<int> distinct;
    for (int row : result.selected_indices) {
        REQUIRE(row >= 0);
        REQUIRE(row < rows);
        distinct.insert(row);
    }
    REQUIRE(distinct.size() == result.selected_indices.size());
    for (double score : result.scores) {
        REQUIRE(std::isfinite(score));
        REQUIRE(score >= 0.);
    }
    REQUIRE(std::accumulate(result.scores.begin(), result.scores.end(), 0.) ==
            Approx(1.).margin(2e-13));
}
}

TEST_CASE("CUR empty dimensions and zero-row requests have an empty projection", "[cur][input]") {
    for (const auto shape : {std::pair{0,0}, std::pair{0,3}, std::pair{3,0}}) {
        const Eigen::MatrixXd matrix(shape.first,shape.second);
        for (bool deterministic : {false,true}) for (int k : {0,1,8}) {
            auto result = cur_select_rows(matrix,k,deterministic,42,true);
            REQUIRE(result.selected_indices.empty());
            REQUIRE(result.scores.empty());
            REQUIRE(result.reconstruction_error == 0.);
            REQUIRE(cur_select_rows(matrix,k,deterministic).reconstruction_error == -1.);
        }
    }
    for (const Eigen::MatrixXd &matrix : {descriptor_matrix(), Eigen::MatrixXd::Zero(3,2).eval()}) {
        const auto result = cur_select_rows(matrix,0,true,42,true);
        REQUIRE(result.selected_indices.empty());
        REQUIRE(result.scores.empty());
        REQUIRE(result.reconstruction_error == (matrix.isZero(0.) ? 0. : 1.));
        REQUIRE(cur_select_rows(matrix,0).reconstruction_error == -1.);
    }
}

TEST_CASE("CUR invalid requests and nonfinite descriptors throw before decomposition", "[cur][input]") {
    for (const Eigen::MatrixXd &matrix : {descriptor_matrix(), Eigen::MatrixXd(3,0)})
        REQUIRE_THROWS_AS(cur_select_rows(matrix,-1),std::invalid_argument);
    for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
        auto matrix = descriptor_matrix();
        matrix(1,2) = invalid;
        for (bool deterministic : {false,true}) for (int k : {0,1,8})
            REQUIRE_THROWS_AS(cur_select_rows(matrix,k,deterministic,42,true),std::invalid_argument);
    }
}

TEST_CASE("CUR deterministic selection matches analytic leverage and projection error", "[cur][numeric]") {
    const auto matrix = descriptor_matrix();
    auto result = cur_select_rows(matrix,2,true,42,true);
    check_selection(result,5,2);
    const std::vector<double> scores{.25,.5,0.,.25,0.};
    for (size_t i=0; i<scores.size(); ++i)
        REQUIRE(result.scores[i] == Approx(scores[i]).margin(2e-13));
    REQUIRE(result.selected_indices.front() == 1);
    REQUIRE((result.selected_indices.back() == 0 || result.selected_indices.back() == 3));
    // Only the third orthogonal feature is omitted by these rows.
    REQUIRE(result.reconstruction_error == Approx(1./std::sqrt(23.)).margin(2e-13));
    Eigen::Matrix3d rotation;
    rotation << .6,-.8,0, .8,.6,0, 0,0,1;
    auto rotated = cur_select_rows((matrix*rotation).eval(),2,true,42,true);
    for (size_t i=0; i<scores.size(); ++i)
        REQUIRE(rotated.scores[i] == Approx(scores[i]).margin(2e-13));
    REQUIRE(rotated.reconstruction_error == Approx(result.reconstruction_error).margin(2e-13));
    auto tied = cur_select_rows(Eigen::MatrixXd::Identity(3,3),3,true);
    REQUIRE(tied.selected_indices == std::vector<int>{0,1,2});
}

TEST_CASE("CUR row count can exceed feature count and is capped only by available rows", "[cur][numeric]") {
    const auto matrix = descriptor_matrix();
    for (bool deterministic : {false,true}) for (int requested : {4,5,12}) {
        auto result = cur_select_rows(matrix,requested,deterministic,27,true);
        check_selection(result,5,std::min(requested,5));
        REQUIRE(result.reconstruction_error < 1e-12);
        if (requested >= 5)
            REQUIRE(std::set<int>(result.selected_indices.begin(),result.selected_indices.end()) ==
                    std::set<int>{0,1,2,3,4});
    }
    Eigen::MatrixXd wide(2,4);
    wide << 1,2,0,-1, 0,0,2,3;
    auto result = cur_select_rows(wide,6,true,42,true);
    check_selection(result,2,2);
    REQUIRE(result.reconstruction_error < 1e-12);
}

TEST_CASE("CUR probabilistic selections repeat with a seed and handle rank deficient inputs", "[cur][numeric]") {
    Eigen::Vector4d left(1.,2.,-3.,0.);
    Eigen::RowVector3d right(2.,-1.,.5);
    const Eigen::MatrixXd matrix = left*right;
    for (int seed : {0,17,42,1927}) {
        auto first = cur_select_rows(matrix,1,false,seed,true);
        auto second = cur_select_rows(matrix,1,false,seed,true);
        check_selection(first,4,1);
        REQUIRE(first.selected_indices == second.selected_indices);
        for (int i=0; i<4; ++i)
            REQUIRE(first.scores[i] == Approx(left[i]*left[i]/14.).margin(2e-13));
        REQUIRE(first.reconstruction_error < 1e-12);
    }
    // Zero data remains valid; singular-vector scores need not be uniform.
    // Requesting more rows than features also exercises zero-weight fallback.
    for (bool deterministic : {false,true}) {
        auto result = cur_select_rows(Eigen::MatrixXd::Zero(4,2),3,deterministic,42,true);
        check_selection(result,4,3);
        REQUIRE(result.reconstruction_error == 0.);
    }
}

TEST_CASE("CUR relative reconstruction error is invariant at finite extreme scales", "[cur][numeric]") {
    const auto matrix = descriptor_matrix();
    const auto reference = cur_select_rows(matrix,2,true,42,true);
    for (double scale : {1e-200,1e200}) {
        const auto result = cur_select_rows((scale*matrix).eval(),2,true,42,true);
        check_selection(result,5,2);
        for (size_t i=0; i<reference.scores.size(); ++i)
            REQUIRE(result.scores[i] == Approx(reference.scores[i]).margin(2e-13));
        REQUIRE(std::isfinite(result.reconstruction_error));
        REQUIRE(result.reconstruction_error == Approx(reference.reconstruction_error).margin(2e-13));
    }
}
