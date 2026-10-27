// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "maxvol/maxvol.h"
#include <Eigen/Core>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

using Catch::Approx;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Determinant of a square matrix (absolute value, for volume)
static double abs_det(const Eigen::MatrixXd &C)
{
    return std::abs(C.determinant());
}

/// Row indices → full list of all rows not selected
static std::vector<int> complement(int m, const std::vector<int> &sel)
{
    std::set<int> s(sel.begin(), sel.end());
    std::vector<int> r;
    for (int i = 0; i < m; ++i)
        if (!s.count(i)) r.push_back(i);
    return r;
}

/// Build submatrix C = A[selected, :]
static Eigen::MatrixXd submatrix(const Eigen::MatrixXd &A,
                                 const std::vector<int> &sel)
{
    Eigen::MatrixXd C(sel.size(), A.cols());
    for (std::size_t i = 0; i < sel.size(); ++i)
        C.row(i) = A.row(sel[i]);
    return C;
}

static double generalized_volume(const Eigen::MatrixXd &C)
{
    if (C.rows() <= C.cols())
        return std::sqrt(std::max(0.0, (C * C.transpose()).determinant()));
    return std::sqrt(std::max(0.0, (C.transpose() * C).determinant()));
}

// ═══════════════════════════════════════════════════════════════════════════
// Basic correctness
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("MaxVol select_rows on identity-plus-noise matrix", "[maxvol]")
{
    // Matrix 20x5: rows 0..4 form the identity, rows 5..19 are small noise
    const int m = 20, n = 5;
    Eigen::MatrixXd A(m, n);
    A.setZero();
    for (int i = 0; i < n; ++i)
        A(i, i) = 1.0;                 // identity block
    for (int i = n; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A(i, j) = 1e-6 * (i + j);  // noise rows

    MaxVol mv;
    std::vector<int> selected = mv.select_rows(A, n);

    REQUIRE(static_cast<int>(selected.size()) == n);

    // The identity rows should be selected (or equivalent)
    Eigen::MatrixXd C = submatrix(A, selected);
    REQUIRE(abs_det(C) == Approx(1.0).margin(1e-3));
}

TEST_CASE("MaxVol determinant monotonicity", "[maxvol]")
{
    // Random 50x8 matrix
    const int m = 50, n = 8;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv;
    mv.verbose = false;
    std::vector<int> selected = mv.select_rows(A, n);

    REQUIRE(static_cast<int>(selected.size()) == n);

    // Re-run with tolerance=0 (force convergence) and check volume is reasonable
    Eigen::MatrixXd C = submatrix(A, selected);
    double vol_sel = abs_det(C);

    // Compare with a random selection
    std::vector<int> random_sel = {0,1,2,3,4,5,6,7};
    double vol_random = abs_det(submatrix(A, random_sel));

    // MaxVol should do at least as well as random
    REQUIRE(vol_sel >= vol_random * 0.99);  // allow small numerical margin
}

TEST_CASE("MaxVol convergence criterion", "[maxvol]")
{
    const int m = 30, n = 6;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv;
    mv.tolerance = 1e-6;
    mv.max_iterations = 50;
    std::vector<int> selected = mv.select_rows(A, n);

    // After convergence, compute B = A * C^{-1}
    Eigen::MatrixXd C = submatrix(A, selected);
    Eigen::MatrixXd B = A * C.inverse();

    // For non-selected rows, ALL entries of B should be <= 1+tol
    std::set<int> sel_set(selected.begin(), selected.end());
    for (int l = 0; l < m; ++l) {
        if (sel_set.count(l)) continue;
        for (int i = 0; i < n; ++i) {
            REQUIRE(std::abs(B(l, i)) <= 1.0 + mv.tolerance + 1e-10);
        }
    }
}

TEST_CASE("MaxVol reproducibility", "[maxvol]")
{
    const int m = 40, n = 5;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv;
    auto sel1 = mv.select_rows(A, n);
    auto sel2 = mv.select_rows(A, n);

    // Deterministic: same input → same output
    REQUIRE(sel1 == sel2);
}

TEST_CASE("MaxVol k=n edge case", "[maxvol]")
{
    const int m = 15, n = 5;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
    MaxVol mv;
    auto sel = mv.select_rows(A, n);
    REQUIRE(static_cast<int>(sel.size()) == n);

    // Submatrix must be non-singular
    Eigen::MatrixXd C = submatrix(A, sel);
    REQUIRE(std::abs(C.determinant()) > 1e-12);
}

TEST_CASE("MaxVol k < n rank-deficient warning", "[maxvol]")
{
    const int m = 20, n = 6;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
    MaxVol mv;
    // k < n: still works but submatrix is not square
    auto sel = mv.select_rows(A, 4);
    REQUIRE(static_cast<int>(sel.size()) == 4);
    // Verify all indices are valid
    for (int idx : sel) {
        REQUIRE(idx >= 0);
        REQUIRE(idx < m);
    }
    // Verify uniqueness
    std::set<int> s(sel.begin(), sel.end());
    REQUIRE(static_cast<int>(s.size()) == 4);
}

TEST_CASE("MaxVol rectangular selection optimizes row volume", "[maxvol]")
{
    SECTION("k less than n") {
        Eigen::MatrixXd A(2, 2);
        A << 1.0, 1.0,
             1.2, 0.0;
        MaxVol mv;
        const auto selected = mv.select_rows(A, 1);
        REQUIRE(selected == std::vector<int>{0});
    }

    SECTION("k greater than n") {
        Eigen::MatrixXd A(3, 1);
        A << 100.0, 99.0, -1.0;
        MaxVol mv;
        const auto selected = mv.select_rows(A, 2);
        const std::set<int> actual(selected.begin(), selected.end());
        const std::set<int> expected{0, 1};
        REQUIRE(actual == expected);
        REQUIRE(generalized_volume(submatrix(A, selected)) ==
                Approx(std::sqrt(100.0 * 100.0 + 99.0 * 99.0)).margin(1e-10));
    }
}

TEST_CASE("MaxVol rectangular selection reaches one-swap dominance", "[maxvol]")
{
    SECTION("wide active matrix") {
        Eigen::MatrixXd A(7, 4);
        A <<  1.471909,  0.283608, -0.237301, -1.947283,
             -1.392884,  0.767586,  0.567825, -0.025434,
             -0.721711, -0.524758,  0.546759, -0.752067,
              1.168597, -0.739976,  0.827506, -1.400822,
              0.401858, -1.100306,  0.737231, -1.336970,
             -1.711412,  0.812356,  0.125804,  1.206945,
              0.324570,  1.621873,  0.370350, -1.533726;
        MaxVol mv;
        const auto selected = mv.select_rows(A, 2);
        const Eigen::VectorXd scores = mv.compute_scores(A, selected);
        REQUIRE(scores.maxCoeff() <= 1.0 + mv.tolerance + 1e-10);
        REQUIRE(generalized_volume(submatrix(A, selected)) > 5.03);
    }

    SECTION("tall active matrix") {
        Eigen::MatrixXd A(7, 3);
        A << -1.421582, -0.217116,  0.652146,
             -1.107844, -1.091885,  0.671908,
             -1.405839,  0.469552, -1.058507,
             -0.503327,  0.491563, -2.061949,
              0.407650,  0.769123,  0.500971,
             -1.088233, -0.847103, -0.630612,
             -0.481076,  0.901375, -0.661250;
        MaxVol mv;
        const auto selected = mv.select_rows(A, 5);
        const Eigen::VectorXd scores = mv.compute_scores(A, selected);
        REQUIRE(scores.maxCoeff() <= 1.0 + mv.tolerance + 1e-10);
    }
}

TEST_CASE("MaxVol empty matrix", "[maxvol]")
{
    Eigen::MatrixXd A(0, 5);
    MaxVol mv;
    auto sel = mv.select_rows(A, 5);
    REQUIRE(sel.empty());
}

TEST_CASE("MaxVol k > m clamped", "[maxvol]")
{
    const int m = 5, n = 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
    MaxVol mv;
    // k > m: all rows returned
    auto sel = mv.select_rows(A, 100);
    REQUIRE(static_cast<int>(sel.size()) == m);
}

// ═══════════════════════════════════════════════════════════════════════════
// compute_scores
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("MaxVol compute_scores basic", "[maxvol]")
{
    const int m = 20, n = 5;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv;
    std::vector<int> selected = mv.select_rows(A, n);
    Eigen::VectorXd scores = mv.compute_scores(A, selected);

    REQUIRE(scores.size() == m);

    // Scores of selected rows should be exactly 1.0
    for (int idx : selected) {
        REQUIRE(scores[idx] == Approx(1.0).margin(1e-12));
    }

    // All scores should be ≤ 1 + tol (convergence guarantee)
    for (int i = 0; i < m; ++i) {
        REQUIRE(scores[i] <= 1.0 + mv.tolerance + 1e-10);
    }
}

TEST_CASE("MaxVol wide score includes orthogonal residual", "[maxvol]")
{
    Eigen::MatrixXd active(1, 2), candidate(1, 2);
    active << 1.0, 0.0;
    candidate << 0.0, 10.0;
    MaxVol mv;
    const std::vector<int> selected{0};
    mv.warm_cache(selected, active);
    const Eigen::VectorXd scores = mv.score_rows(candidate, selected);
    REQUIRE(scores.size() == 1);
    REQUIRE(scores[0] == Approx(10.0).margin(1e-12));
}

TEST_CASE("MaxVol tall score uses generalized replacement volume", "[maxvol]")
{
    Eigen::MatrixXd active(4, 2), candidate(1, 2);
    active << 1.0, 0.0,
              1.0, 0.0,
              0.0, 1.0,
              0.0, 1.0;
    candidate << 1.0, 1.0;
    MaxVol mv;
    const std::vector<int> selected{0, 1, 2, 3};
    mv.warm_cache(selected, active);
    const Eigen::VectorXd scores = mv.score_rows(candidate, selected);
    REQUIRE(scores.size() == 1);
    REQUIRE(scores[0] == Approx(std::sqrt(1.25)).margin(1e-12));
}

TEST_CASE("MaxVol scoring rejects invalid selected indices", "[maxvol]")
{
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2, 2);
    MaxVol mv;
    REQUIRE(mv.compute_scores(A, {-1, 1}).size() == 0);
    mv.warm_cache({0, 2}, A);
    REQUIRE(mv.score_rows(A, {0, 2}).size() == 0);
    REQUIRE(mv.compute_scores(A, {0, 0}).size() == 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// add_rows
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("MaxVol add_rows extends selection", "[maxvol]")
{
    const int m = 40, n = 5;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv;
    auto initial = mv.select_rows(A, n);    // select n rows
    int add_k = n;                           // add at least n more
    auto extended = mv.add_rows(A, initial, add_k);

    REQUIRE(static_cast<int>(extended.size()) == n + add_k);

    // All original indices must be preserved
    for (int idx : initial) {
        REQUIRE(std::find(extended.begin(), extended.end(), idx)
                != extended.end());
    }

    // All extended indices must be unique
    std::set<int> s(extended.begin(), extended.end());
    REQUIRE(static_cast<int>(s.size()) == n + add_k);
}

TEST_CASE("MaxVol add_rows to full set returns all", "[maxvol]")
{
    const int m = 10, n = 4;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv;
    auto initial = mv.select_rows(A, n);
    // Adding so many that total exceeds m
    auto result = mv.add_rows(A, initial, 100);
    REQUIRE(static_cast<int>(result.size()) == m);
}

TEST_CASE("MaxVol add_rows conditions on the current active set", "[maxvol]")
{
    Eigen::MatrixXd A(4, 2);
    A << 100.0, 0.0,
         0.0, 1.0,
         1000.0, 0.0,
         0.0, 100.0;
    MaxVol mv;
    const auto extended = mv.add_rows(A, {0, 1}, 1);
    const std::vector<int> expected{0, 1, 3};
    REQUIRE(extended == expected);
    REQUIRE(generalized_volume(submatrix(A, extended)) > 10000.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// swap_tracker / log_det / serialization
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("MaxVol swap history is recorded", "[maxvol]")
{
    const int m = 30, n = 5;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv;
    auto sel = mv.select_rows(A, n);
    const auto &history = mv.swap_history();

    // After convergence, all selected rows should be valid indices
    for (int idx : sel) {
        REQUIRE(idx >= 0);
        REQUIRE(idx < m);
    }
    // After selection we should have swap history or selected rows (or both)
    REQUIRE((!history.empty() || !sel.empty()));
    // If there were swaps, each has valid indices
    for (const auto &s : history) {
        REQUIRE(s.ind_in  >= 0);
        REQUIRE(s.ind_in  < m);
        REQUIRE(s.ind_out >= 0);
        REQUIRE(s.ind_out < m);
        REQUIRE(s.gain > 1.0 + mv.tolerance);  // swaps only happen when gain > 1+tol
    }
}

TEST_CASE("MaxVol log determinant monotonicity", "[maxvol]")
{
    const int m = 40, n = 6;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv;
    mv.select_rows(A, n);

    double ld = mv.log_determinant();
    // log_det should be finite
    REQUIRE(std::isfinite(ld));
    // For a non-singular submatrix, log_det should not be -inf
    REQUIRE(ld > -1e30);
}

TEST_CASE("MaxVol log determinant stays finite for rank-deficient small matrix", "[maxvol]")
{
    const int m = 5, n = 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, n);

    MaxVol mv;
    auto selected = mv.select_rows(A, n);

    REQUIRE(static_cast<int>(selected.size()) == n);
    REQUIRE(std::isfinite(mv.log_determinant()));
    REQUIRE(mv.log_determinant() > -1e6);

    Eigen::VectorXd scores = mv.compute_scores(A, selected);
    REQUIRE(scores.size() == m);
    for (int i = 0; i < scores.size(); ++i)
        REQUIRE(std::isfinite(scores[i]));
}

TEST_CASE("MaxVol score_rows requires prepared cache", "[maxvol]")
{
    const int m = 8, n = 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
    Eigen::MatrixXd candidates = Eigen::MatrixXd::Random(2, n);
    std::vector<int> selected = {0, 1, 2};

    MaxVol mv;
    Eigen::VectorXd cold_scores = mv.score_rows(candidates, selected);
    REQUIRE(cold_scores.size() == 0);

    mv.warm_cache(selected, A);
    Eigen::VectorXd warm_scores = mv.score_rows(candidates, selected);
    REQUIRE(warm_scores.size() == candidates.rows());
    for (int i = 0; i < warm_scores.size(); ++i)
        REQUIRE(std::isfinite(warm_scores[i]));
}

TEST_CASE("MaxVol square factorization preserves the truncated SVD fallback", "[maxvol]")
{
    Eigen::MatrixXd active = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd probes(2, 2);
    probes << 1.0, 0.0, 0.0, 1.0;
    MaxVol mv;
    const std::vector<int> selected{0, 1};
    active(1, 1) = 0.5;
    mv.warm_cache(selected, active);
    REQUIRE(mv.score_rows(probes, selected)[1] == Approx(2.0).margin(1e-12));

    // Inverting this direction with LU would produce 1e20. The rank-aware
    // fallback must still discard it using the original SVD threshold.
    active(1, 1) = 1e-20;
    mv.warm_cache(selected, active);
    const auto scores = mv.score_rows(probes, selected);
    REQUIRE(scores[0] == Approx(1.0).margin(1e-12));
    REQUIRE(scores[1] == Approx(0.0).margin(1e-12));
}

TEST_CASE("MaxVol warm_cache refreshes when active row values change", "[maxvol]")
{
    Eigen::MatrixXd A1(2, 2);
    A1 << 1.0, 0.0,
          0.0, 1.0;
    Eigen::MatrixXd A2(2, 2);
    A2 << 2.0, 0.0,
          0.0, 2.0;
    Eigen::MatrixXd candidate(1, 2);
    candidate << 1.0, 0.0;

    std::vector<int> selected = {0, 1};
    MaxVol mv;
    mv.warm_cache(selected, A1);
    Eigen::VectorXd score1 = mv.score_rows(candidate, selected);
    REQUIRE(score1.size() == 1);
    REQUIRE(score1[0] == Approx(1.0).margin(1e-12));

    mv.warm_cache(selected, A2);
    Eigen::VectorXd score2 = mv.score_rows(candidate, selected);
    REQUIRE(score2.size() == 1);
    REQUIRE(score2[0] == Approx(0.5).margin(1e-12));
}

TEST_CASE("MaxVol swap_limit caps number of swaps", "[maxvol]")
{
    const int m = 50, n = 8;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv;
    mv.max_swaps = 3;  // only allow 3 swaps

    mv.select_rows(A, n);
    // swap history should have at most 3 entries
    REQUIRE(static_cast<int>(mv.swap_history().size()) <= 3);
}

TEST_CASE("MaxVol save/load roundtrip", "[maxvol]")
{
    const int m = 30, n = 5;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv1;
    const std::vector<int> selected = mv1.select_rows(A, n);
    double ld1 = mv1.log_determinant();

    // Save
    const std::string tmpfile = "/tmp/maxvol_test_save.bin";
    mv1.save(tmpfile);

    // Load into a new object
    MaxVol mv2;
    mv2.select_rows(A, n);
    const std::vector<int> loaded = mv2.load(tmpfile);

    // log_det should match
    REQUIRE(mv2.log_determinant() == Approx(ld1).margin(1e-12));
    REQUIRE(loaded == selected);

    // Swap history is runtime diagnostic data, not checkpoint state.
    REQUIRE(mv2.swap_history().empty());

    std::remove(tmpfile.c_str());
}

TEST_CASE("MaxVol rejects version 2 checkpoints", "[maxvol][serialization]")
{
    std::stringstream input(std::ios::in | std::ios::out | std::ios::binary);
    const int magic = 0x4D56534C, version = 2, k = 1;
    const double logdet = 7.0;
    input.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    input.write(reinterpret_cast<const char *>(&version), sizeof(version));
    input.write(reinterpret_cast<const char *>(&k), sizeof(k));
    input.write(reinterpret_cast<const char *>(&logdet), sizeof(logdet));
    input.seekg(0);
    MaxVol mv;
    REQUIRE(mv.load(input).empty());
    REQUIRE(mv.log_determinant() == 0.0);
}

TEST_CASE("MaxVol subset scores", "[maxvol]")
{
    const int m = 40, n = 5;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

    MaxVol mv;
    auto sel = mv.select_rows(A, n);

    // Split rows into 4 subsets
    std::vector<int> ranges = {0, 10, 20, 30, 40};
    std::vector<double> grades;
    mv.compute_subset_scores(A, sel, ranges, grades);

    REQUIRE(grades.size() == 4);

    Eigen::VectorXd full_scores = mv.compute_scores(A, sel);
    // Each subset grade should be >= max score within that range
    for (int r = 0; r < 4; ++r) {
        double max_in_range = 0;
        for (int i = ranges[r]; i < ranges[r+1]; ++i)
            max_in_range = std::max(max_in_range, full_scores[i]);
        REQUIRE(grades[r] == Approx(max_in_range).margin(1e-12));
    }
}
