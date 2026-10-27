// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "maxvol/crowding.h"
#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <vector>

using namespace Eigen;
using Catch::Approx;

namespace {

/// Rows of identical descriptor vectors grouped into clusters.
Matrix<double, -1, -1, RowMajor>
make_cluster_matrix(int n_dense, int n_sparse)
{
    constexpr int dim = 4;
    Matrix<double, -1, -1, RowMajor> B(n_dense + n_sparse, dim);
    // dense cluster: same vector x20
    for (int i = 0; i < n_dense; ++i) {
        B.row(i) << 1.0, 0.0, 0.0, 0.0;
    }
    // sparse cluster: identical but far away
    for (int i = 0; i < n_sparse; ++i) {
        B.row(n_dense + i) << 0.0, 1.0, 0.0, 0.0;
    }
    return B;
}

} // namespace

TEST_CASE("CrowdingWeights: alpha=0 reproduces unweighted case")
{
    auto B = make_cluster_matrix(20, 2);
    CrowdingParams p;
    p.alpha = 0.0;

    auto w = CrowdingWeights::compute(B, p);
    REQUIRE(w.size() == 22);
    for (double wi : w) REQUIRE(wi == Approx(1.0));
}

TEST_CASE("CrowdingWeights: sparse cluster gets larger weight (cosine kernel)")
{
    auto B = make_cluster_matrix(20, 2);
    CrowdingParams p;
    p.alpha = 0.5;

    auto w = CrowdingWeights::compute(B, p);

    double w_dense = 0.0, w_sparse = 0.0;
    for (int i = 0; i < 20; ++i) w_dense += w[i];
    for (int i = 20; i < 22; ++i) w_sparse += w[i];
    w_dense /= 20.0;
    w_sparse /= 2.0;

    REQUIRE(w_sparse > 2.0 * w_dense);
    // dense cluster is the majority → its weight is around the median (=1)
    REQUIRE(w_dense == Approx(1.0).margin(1e-9));
}

TEST_CASE("CrowdingWeights: gaussian kernel with auto bandwidth")
{
    auto B = make_cluster_matrix(20, 2);
    CrowdingParams p;
    p.kernel = "gaussian";
    p.alpha = 0.5;

    auto w = CrowdingWeights::compute(B, p);

    double w_dense = 0.0, w_sparse = 0.0;
    for (int i = 0; i < 20; ++i) w_dense += w[i];
    for (int i = 20; i < 22; ++i) w_sparse += w[i];
    w_dense /= 20.0;
    w_sparse /= 2.0;

    // The median-distance bandwidth deliberately gives a smooth estimate;
    // the minority cluster must still receive a strictly larger weight.
    REQUIRE(w_sparse > w_dense);
}

TEST_CASE("CrowdingWeights: element bucketing isolates species")
{
    auto B = make_cluster_matrix(20, 2); // two clusters, single matrix
    std::vector<int> buckets;
    for (int i = 0; i < 22; ++i) buckets.push_back(i < 20 ? 0 : 1);

    CrowdingParams p;
    p.alpha = 0.5;

    // Bucketed: every bucket holds identical rows → rho constant → w=1
    auto w_b = CrowdingWeights::compute(B, buckets, p);
    for (double wi : w_b) REQUIRE(wi == Approx(1.0).margin(1e-9));

    // Unbucketed: sparse cluster boosted
    auto w_u = CrowdingWeights::compute(B, p);
    REQUIRE(w_u[20] > 2.0 * w_u[0]);
}

TEST_CASE("CrowdingWeights: clipping caps extreme weights")
{
    auto B = make_cluster_matrix(20, 2);
    CrowdingParams p;
    p.kernel = "gaussian";
    p.sigma = 1e-6; // far neighbors contribute ~0 → rho ~ 0 for sparse rows
    p.alpha = 1.0;
    p.clip_max = 10.0;

    auto w = CrowdingWeights::compute(B, p);

    // dense cluster: all rows identical → rho=1 → w=1
    for (int i = 0; i < 20; ++i) REQUIRE(w[i] == Approx(1.0).margin(1e-6));
    // sparse cluster: isolated → clipped at clip_max
    REQUIRE(w[20] == Approx(10.0));
    REQUIRE(w[21] == Approx(10.0));
}

TEST_CASE("CrowdingWeights: single-row bucket yields weight 1")
{
    Matrix<double, -1, -1, RowMajor> B(3, 2);
    B.row(0) << 1.0, 0.0;
    B.row(1) << 1.0, 0.0;
    B.row(2) << 0.0, 1.0;

    std::vector<int> buckets{0, 0, 1}; // bucket 1 has a single row
    CrowdingParams p;
    p.alpha = 1.0;

    auto w = CrowdingWeights::compute(B, buckets, p);
    REQUIRE(w[2] == Approx(1.0));
    REQUIRE(w[0] == Approx(1.0)); // two identical rows → equal density
}

TEST_CASE("CrowdingWeights: bucket count must match row count")
{
    Matrix<double, -1, -1, RowMajor> B(3, 2);
    CrowdingParams p;
    std::vector<int> buckets{0, 1};
    REQUIRE_THROWS_AS(CrowdingWeights::compute(B, buckets, p),
                      std::invalid_argument);
}

TEST_CASE("CrowdingWeights: empty input returns empty weights")
{
    Matrix<double, -1, -1, RowMajor> B(0, 3);
    CrowdingParams p;
    REQUIRE(CrowdingWeights::compute(B, p).empty());
    REQUIRE(CrowdingWeights::compute(B, std::vector<int>{}, p).empty());
}

TEST_CASE("CrowdingWeights: sampled reference membership does not bias identical rows")
{
    Matrix<double, -1, -1, RowMajor> B(12, 2);
    for (int i = 0; i < B.rows(); ++i)
        B.row(i) << 1.0, 0.0;
    CrowdingParams p;
    p.max_density_points = 4;
    p.alpha = 1.0;

    const auto w = CrowdingWeights::compute(B, p);
    for (double wi : w)
        REQUIRE(wi == Approx(1.0).margin(1e-12));
}

TEST_CASE("CrowdingWeights: rows in negative buckets are left unchanged")
{
    Matrix<double, -1, -1, RowMajor> B(3, 2);
    B.setRandom();
    CrowdingParams p;
    const auto w = CrowdingWeights::compute(B, std::vector<int>{-1, -1, -1}, p);
    REQUIRE(w == std::vector<double>{1.0, 1.0, 1.0});
}

TEST_CASE("CrowdingWeights: row subsampling keeps ordering")
{
    // 100 identical rows vs 1 far-away row; sample reference = 10 rows
    constexpr int n = 101;
    Matrix<double, -1, -1, RowMajor> B(n, 2);
    for (int i = 0; i < n; ++i) {
        if (i < 100) B.row(i) << 1.0, 0.0;
        else B.row(i) << 0.0, 1.0;
    }
    CrowdingParams p;
    p.max_density_points = 10;
    p.alpha = 0.5;

    auto w = CrowdingWeights::compute(B, p);
    REQUIRE(w[100] > w[0]); // outlier still boosted
}

TEST_CASE("CrowdingWeights: alpha zero overrides isolated rows")
{
    Matrix<double, -1, -1, RowMajor> B(4, 2);
    B.row(0) << 1.0, 0.0;
    B.row(1) << 1.0, 0.0;
    B.row(2) << 1.0, 0.0;
    B.row(3) << 0.0, 1.0;
    CrowdingParams p;
    p.alpha = 0.0;
    const auto w = CrowdingWeights::compute(B, p);
    for (double value : w) REQUIRE(value == Approx(1.0));
}

TEST_CASE("CrowdingWeights: cosine treats identical zero descriptors as neighbors")
{
    Matrix<double, -1, -1, RowMajor> B(4, 2);
    B.row(0).setZero();
    B.row(1).setZero();
    B.row(2).setZero();
    B.row(3) << 1.0, 0.0;
    CrowdingParams p;
    p.alpha = 1.0;
    p.clip_max = 20.0;
    const auto w = CrowdingWeights::compute(B, p);
    REQUIRE(w[0] == Approx(1.0));
    REQUIRE(w[1] == Approx(1.0));
    REQUIRE(w[2] == Approx(1.0));
    REQUIRE(w[3] == Approx(20.0));
}

TEST_CASE("CrowdingWeights: validates public parameters")
{
    Matrix<double, -1, -1, RowMajor> B(2, 2);
    B.setRandom();
    CrowdingParams p;

    SECTION("invalid sample") {
        p.max_density_points = 0;
        REQUIRE_THROWS_AS(CrowdingWeights::compute(B, p), std::invalid_argument);
    }
    SECTION("invalid kernel") {
        p.kernel = "linear";
        REQUIRE_THROWS_AS(CrowdingWeights::compute(B, p), std::invalid_argument);
    }
    SECTION("negative alpha") {
        p.alpha = -0.1;
        REQUIRE_THROWS_AS(CrowdingWeights::compute(B, p), std::invalid_argument);
    }
    SECTION("non-finite descriptor") {
        B(0, 0) = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_AS(CrowdingWeights::compute(B, p), std::invalid_argument);
    }
}

TEST_CASE("CrowdingModel: streaming batches equal exact matrix API")
{
    Matrix<double, -1, -1, RowMajor> B(8, 3);
    B << 1.0, 0.0, 0.1,
         1.0, 0.1, 0.0,
         0.9, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.9, 0.1,
         0.1, 1.0, 0.0,
         0.0, 0.0, 1.0,
         0.0, 0.1, 1.0;
    const std::vector<int> buckets{1000000, 1000000, 1000000, 7, 7, 7, 7, 7};
    std::vector<std::uint64_t> ids(8);
    for (std::uint64_t i = 0; i < ids.size(); ++i) ids[i] = i;
    CrowdingParams p;
    p.max_density_points = -1;

    const auto expected = CrowdingWeights::compute(B, buckets, p);
    CrowdingModel model(3, p);
    model.observe(B.topRows(3),
                  std::vector<int>(buckets.begin(), buckets.begin() + 3),
                  std::vector<std::uint64_t>(ids.begin(), ids.begin() + 3));
    model.observe(B.bottomRows(5),
                  std::vector<int>(buckets.begin() + 3, buckets.end()),
                  std::vector<std::uint64_t>(ids.begin() + 3, ids.end()));
    model.finalize();
    std::vector<double> rho;
    for (int begin : {0, 2, 5}) {
        const int end = begin == 0 ? 2 : (begin == 2 ? 5 : 8);
        Matrix<double, -1, -1, RowMajor> block = B.middleRows(begin, end - begin);
        const auto part = model.density(
            block, std::vector<int>(buckets.begin() + begin, buckets.begin() + end),
            std::vector<std::uint64_t>(ids.begin() + begin, ids.begin() + end));
        rho.insert(rho.end(), part.begin(), part.end());
    }
    const auto actual = CrowdingWeights::weights_from_density(rho, buckets, p);
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        REQUIRE(actual[i] == Approx(expected[i]).margin(1e-12));
}
