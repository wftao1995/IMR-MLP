// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "maxvol/online_maxvol.h"
#include <sstream>
#include <Eigen/QR>

using Catch::Approx;

TEST_CASE("OnlineMaxVol reusable score workspace matches allocating API",
          "[online_maxvol][workspace]")
{
    SECTION("square inverse")
    {
        OnlineMaxVolState state;
        state.reset_identity(3);
        OnlineRowMatrix candidates = OnlineRowMatrix::Random(7, 3);
        const Eigen::VectorXd expected = state.score_rows(candidates);

        OnlineMaxVolState::ScoreWorkspace workspace;
        const auto actual = state.score_rows(candidates, workspace);
        REQUIRE(actual.size() == expected.size());
        REQUIRE((actual - expected).norm() == Approx(0.0).margin(1e-12));

        OnlineRowMatrix smaller = candidates.topRows(2);
        const Eigen::VectorXd smaller_expected = state.score_rows(smaller);
        const auto smaller_actual = state.score_rows(smaller, workspace);
        REQUIRE((smaller_actual - smaller_expected).norm() == Approx(0.0).margin(1e-12));
        REQUIRE(workspace.coeff.rows() >= candidates.rows());
    }

    SECTION("wide gram inverse")
    {
        OnlineRowMatrix active(2, 4);
        active << 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0;
        OnlineMaxVolState state;
        REQUIRE(state.initialize_from_rows(active));

        OnlineRowMatrix candidates = OnlineRowMatrix::Random(9, 4);
        const Eigen::VectorXd expected = state.score_rows(candidates);
        OnlineMaxVolState::ScoreWorkspace workspace;
        const auto actual = state.score_rows(candidates, workspace);
        REQUIRE((actual - expected).norm() == Approx(0.0).margin(1e-12));
        REQUIRE(workspace.z.rows() >= candidates.rows());
        REQUIRE(workspace.coeff.rows() >= candidates.rows());
    }
}

TEST_CASE("OnlineMaxVol large batches bound scoring matrix storage", "[online_maxvol][workspace]")
{
    for (int ncols : {8, 12})
    {
        OnlineRowMatrix active = OnlineRowMatrix::Zero(8, ncols);
        active.leftCols(8).setIdentity();
        OnlineMaxVolState state;
        REQUIRE(state.initialize_from_rows(active));
        const int block_rows = state.score_block_rows();
        OnlineRowMatrix candidates = OnlineRowMatrix::Random(2 * block_rows + 7, ncols);
        OnlineMaxVolState::ScoreWorkspace workspace;
        const Eigen::VectorXd scores = state.score_rows(candidates, workspace);
        REQUIRE(scores.size() == candidates.rows());
        REQUIRE(workspace.coeff.rows() <= block_rows);
        REQUIRE(workspace.z.rows() <= block_rows);
        REQUIRE((workspace.coeff.size() + workspace.z.size()) * sizeof(double) <=
                4u * 1024u * 1024u);
        for (int row = 0; row < candidates.rows(); ++row)
        {
            const double coeff_max = candidates.row(row).head(8).cwiseAbs().maxCoeff();
            const double residual2 = candidates.row(row).tail(ncols - 8).squaredNorm();
            REQUIRE(scores[row] ==
                    Approx(std::sqrt(coeff_max * coeff_max + residual2)).margin(1e-12));
        }
    }
}

TEST_CASE("OnlineMaxVol identity update and threshold", "[online_maxvol]")
{
    OnlineMaxVolState state;
    state.reset_identity(2, 1.0);

    OnlineRowMatrix candidates(2, 2);
    candidates << 2.0, 0.0, 0.0, 3.0;
    std::vector<BRowRef> refs(2);
    refs[0].global_row = 10;
    refs[1].global_row = 11;

    REQUIRE(state.maximize_with_candidates(candidates, refs, 1.001) == 2);
    REQUIRE(state.active_rows().cwiseAbs().sum() == Approx(5.0));
    REQUIRE(state.score_rows(candidates).maxCoeff() == Approx(1.0).margin(1e-12));
    REQUIRE(state.selected_refs()[0].global_row == 10);
    REQUIRE(state.selected_refs()[1].global_row == 11);
}

TEST_CASE("OnlineMaxVol save load continue", "[online_maxvol]")
{
    OnlineRowMatrix first(2, 2), second(1, 2);
    first << 2.0, 0.0, 0.0, 3.0;
    second << 4.0, 0.0;
    std::vector<BRowRef> first_refs(2), second_refs(1);
    first_refs[0].global_row = 10;
    first_refs[1].global_row = 11;
    second_refs[0].global_row = 20;

    OnlineMaxVolState uninterrupted;
    uninterrupted.reset_identity(2);
    uninterrupted.maximize_with_candidates(first, first_refs, 1.001);

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    uninterrupted.save(buffer);
    buffer.seekg(0);
    OnlineMaxVolState resumed;
    REQUIRE(resumed.load(buffer));

    REQUIRE(uninterrupted.maximize_with_candidates(second, second_refs, 1.001) == 1);
    REQUIRE(resumed.maximize_with_candidates(second, second_refs, 1.001) == 1);
    REQUIRE((uninterrupted.active_rows() - resumed.active_rows()).norm() ==
            Approx(0.0).margin(1e-12));
    REQUIRE(resumed.selected_refs()[0].global_row == 20);
}

TEST_CASE("OnlineMaxVol rank deficient fallback restores missing direction", "[online_maxvol]")
{
    OnlineRowMatrix active(2, 2), candidate(1, 2);
    active << 1.0, 0.0, 2.0, 0.0;
    candidate << 0.0, 1.0;

    OnlineMaxVolState state;
    REQUIRE(state.initialize_from_rows(active));
    REQUIRE(state.maximize_with_candidates(candidate, {}, 1.001) == 1);
    REQUIRE(std::abs(state.active_rows().determinant()) > 0.0);
}

TEST_CASE("OnlineMaxVol wide score includes orthogonal residual", "[online_maxvol]")
{
    OnlineRowMatrix active(2, 3), candidate(1, 3);
    active << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0;
    candidate << 1.0, 1.0, 3.0;

    OnlineMaxVolState state;
    REQUIRE(state.initialize_from_rows(active));
    const Eigen::VectorXd scores = state.score_rows(candidate);
    REQUIRE(scores.size() == 1);
    // det(C C^T) changes from 4 to 40 when slot zero is replaced.
    REQUIRE(scores[0] == Approx(std::sqrt(10.0)).margin(1e-12));
    REQUIRE(state.maximize_with_candidates(candidate, {}, 1.001, 1) == 1);

    OnlineMaxVolState rebuilt;
    REQUIRE(rebuilt.initialize_from_rows(state.active_rows()));
    OnlineRowMatrix probes(3, 3);
    probes << 0.3, -0.2, 0.7, 2.0, 0.5, 0.0, -1.0, 1.0, 2.0;
    REQUIRE((state.score_rows(probes) - rebuilt.score_rows(probes)).norm() ==
            Approx(0.0).margin(1e-10));
    REQUIRE(state.log_determinant() == Approx(rebuilt.log_determinant()).margin(1e-10));
}

TEST_CASE("OnlineMaxVol wide score is invariant under uniform scaling", "[online_maxvol]")
{
    OnlineRowMatrix active(1, 2), candidate(1, 2);
    active << 1.0, 0.0;
    candidate << 0.0, 10.0;

    OnlineMaxVolState normal;
    REQUIRE(normal.initialize_from_rows(active));
    const double normal_score = normal.score_rows(candidate)[0];

    OnlineMaxVolState scaled;
    REQUIRE(scaled.initialize_from_rows(1e-6 * active));
    const OnlineRowMatrix scaled_candidate = 1e-6 * candidate;
    const double scaled_score = scaled.score_rows(scaled_candidate)[0];

    REQUIRE(normal_score == Approx(10.0).margin(1e-12));
    REQUIRE(scaled_score == Approx(normal_score).epsilon(1e-10));
}

TEST_CASE("OnlineMaxVol retains outgoing active rows in the exchange pool", "[online_maxvol]")
{
    OnlineMaxVolState state;
    state.reset_identity(3);
    const OnlineRowMatrix original = state.active_rows();

    OnlineRowMatrix candidates(2, 3);
    candidates << 0.0, 5.0, 6.0, 8.0, -5.0, 8.0;
    REQUIRE(state.maximize_with_candidates(candidates, {}, 1.001) >= 2);

    OnlineRowMatrix all_rows(original.rows() + candidates.rows(), 3);
    all_rows.topRows(original.rows()) = original;
    all_rows.bottomRows(candidates.rows()) = candidates;
    REQUIRE(state.score_rows(all_rows).maxCoeff() <= 1.001 + 1e-10);
}

TEST_CASE("OnlineMaxVol is order-dependent across candidate batches", "[online_maxvol]")
{
    // Online mode intentionally does not retain rejected historical candidates.
    // A later active-set update may therefore raise an earlier candidate's
    // score above the threshold; callers needing cumulative-pool dominance
    // must use the full or exact-stream backend.
    OnlineMaxVolState state;
    state.reset_identity(2);
    OnlineRowMatrix first(1, 2), second(1, 2);
    first << 1.0, 1.0;
    second << 2.0, -2.0;

    REQUIRE(state.maximize_with_candidates(first, {}, 1.001) == 0);
    REQUIRE(state.maximize_with_candidates(second, {}, 1.001) == 1);
    REQUIRE(state.score_rows(first)[0] == Approx(2.0).margin(1e-12));
}

TEST_CASE("OnlineMaxVol grows a wide active set before replacing rows", "[online_maxvol]")
{
    OnlineRowMatrix active(1, 3), candidates(2, 3);
    active << 1.0, 0.0, 0.0;
    candidates << 0.0, 2.0, 0.0, 0.0, 0.0, 3.0;

    OnlineMaxVolState state;
    REQUIRE(state.initialize_from_rows(active));
    REQUIRE(state.maximize_with_candidates(candidates, {}, 1.001) == 2);
    REQUIRE(state.active_rows().rows() == 3);

    OnlineRowMatrix all_rows(3, 3);
    all_rows << active, candidates;
    REQUIRE(state.score_rows(all_rows).maxCoeff() <= 1.001 + 1e-10);
}

TEST_CASE("OnlineMaxVol periodic refactor controls long update drift", "[online_maxvol]")
{
    SECTION("square Sherman Morrison")
    {
        OnlineMaxVolState state;
        state.reset_identity(4);
        for (int step = 0; step < 300; ++step)
        {
            const int slot = step % 4;
            OnlineRowMatrix candidate = OnlineRowMatrix::Zero(1, 4);
            candidate(0, slot) = 2.0 + step;
            REQUIRE(state.maximize_with_candidates(candidate, {}, 1.0001, 1) == 1);
        }
        OnlineMaxVolState rebuilt;
        REQUIRE(rebuilt.initialize_from_rows(state.active_rows()));
        REQUIRE(state.log_determinant() == Approx(rebuilt.log_determinant()).margin(1e-9));
    }

    SECTION("wide Woodbury")
    {
        OnlineRowMatrix active = OnlineRowMatrix::Zero(3, 5);
        active.leftCols(3).setIdentity();
        OnlineMaxVolState state;
        REQUIRE(state.initialize_from_rows(active));
        for (int step = 0; step < 180; ++step)
        {
            const int slot = step % 3;
            OnlineRowMatrix candidate = OnlineRowMatrix::Zero(1, 5);
            candidate(0, slot) = 2.0 + step;
            REQUIRE(state.maximize_with_candidates(candidate, {}, 1.0001, 1) == 1);
        }
        OnlineMaxVolState rebuilt;
        REQUIRE(rebuilt.initialize_from_rows(state.active_rows()));
        OnlineRowMatrix probes = OnlineRowMatrix::Random(7, 5);
        REQUIRE((state.score_rows(probes) - rebuilt.score_rows(probes)).norm() ==
                Approx(0.0).margin(1e-9));
        REQUIRE(state.log_determinant() == Approx(rebuilt.log_determinant()).margin(1e-9));
    }
}

TEST_CASE("Rank deficient ridge grading agrees with spectral residual and batching",
          "[online_maxvol][ridge]")
{
    // Rotations make the old dense Gram-inverse products catastrophically
    // cancel. The known singular system supplies an independent reference.
    constexpr int n = 48;
    Eigen::MatrixXd seed(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            seed(i, j) = std::sin(1.3 * (i + 1) * (j + 2));
    const Eigen::MatrixXd U = seed.householderQr().householderQ();
    const Eigen::MatrixXd V = seed.transpose().householderQr().householderQ();
    Eigen::VectorXd s(n);
    for (int i = 0; i < n; ++i)
        s[i] = i < n - 2 ? std::pow(10., -10. * i / (n - 3)) : 0.;
    OnlineRowMatrix active = U * s.asDiagonal() * V.transpose();
    OnlineMaxVolState state;
    REQUIRE(state.initialize_from_rows(active));
    const double lambda = 1e-12;
    const Eigen::VectorXd d = (s.array().square() + lambda).inverse();
    const Eigen::VectorXd diagonal = U.array().square().matrix() * d;
    OnlineRowMatrix candidates(137, n);
    for (int i = 0; i < candidates.rows(); ++i)
        candidates.row(i) = (1. + i / 50.) * active.row(i % n);
    // Include a missing direction, which truncated-pseudoinverse-only grades miss.
    candidates.row(136) = V.col(n - 1).transpose();
    const Eigen::VectorXd batch = state.score_rows(candidates);
    for (int i = 0; i < candidates.rows(); ++i)
    {
        OnlineRowMatrix one = candidates.middleRows(i, 1);
        const auto scalar = state.score_rows(one);
        REQUIRE(batch[i] == Approx(scalar[0]).epsilon(1e-9).margin(1e-10));
        const Eigen::RowVectorXd p = one * V;
        const Eigen::RowVectorXd c =
            (p.array() * s.transpose().array() * d.transpose().array()).matrix() * U.transpose();
        const double residual = (p.array().square() * (lambda * d).transpose().array()).sum();
        const double reference = std::sqrt((c.array().square().transpose() +
                                            residual * diagonal.array()).maxCoeff());
        REQUIRE(batch[i] == Approx(reference).epsilon(1e-7).margin(1e-9));
    }
    // Score and exchange must use the same threshold decision.
    OnlineRowMatrix one = candidates.middleRows(80, 1);
    REQUIRE(state.maximize_with_candidates(one, {}, batch[80] * (1. + 1e-7), 1) == 0);
    REQUIRE(state.maximize_with_candidates(one, {}, batch[80] * (1. - 1e-7), 1) == 1);
    std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
    state.save(saved);
    saved.seekg(0);
    OnlineMaxVolState restored;
    REQUIRE(restored.load(saved));
    REQUIRE((state.score_rows(candidates) - restored.score_rows(candidates)).norm() < 1e-7);
}

TEST_CASE("Batch ridge updates retain the current lambda and survive rebuilds",
          "[online_maxvol][ridge][batch]")
{
    constexpr int n = 24;
    Eigen::MatrixXd seed(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            seed(i, j) = std::sin(1.3 * (i + 1) * (j + 2));
    const Eigen::MatrixXd V = seed.householderQr().householderQ();
    Eigen::VectorXd singular = Eigen::VectorXd::Constant(n, 0.2);
    singular[0] = 10.;
    singular[n - 1] = 0.;
    OnlineRowMatrix active = singular.asDiagonal() * V.transpose();
    OnlineMaxVolState state;
    state.enable_batch_updates();
    REQUIRE(state.initialize_from_rows(active));
    for (int iteration = 0; iteration < 130; ++iteration)
    {
        // Alternate lambda-changing and constant-lambda replacements. A
        // rotated right basis exercises cancellation in the missing direction.
        const int slot = iteration % (n - 1);
        OnlineRowMatrix one = state.active_rows().middleRows(slot, 1);
        const double current_grade = state.score_rows(one)[0];
        REQUIRE(current_grade > 0.);
        // Active-row grades can be below one in the ridge definition.
        one *= 1.002 / current_grade;
        CAPTURE(iteration, current_grade);
        REQUIRE(state.maximize_with_candidates(one, {}, 1.001, 1) == 1);
        OnlineMaxVolState rebuilt;
        REQUIRE(rebuilt.initialize_from_rows(state.active_rows()));
        OnlineRowMatrix probes(n + 1, n);
        probes.topRows(n) = state.active_rows();
        probes.row(n) = V.col(n - 1).transpose();
        const auto expected = rebuilt.score_rows(probes);
        const auto actual = state.score_rows(probes);
        for (int row = 0; row < probes.rows(); ++row)
        {
            REQUIRE(std::abs(actual[row] - expected[row]) <=
                    1e-4 * std::max(1., std::abs(expected[row])));
            REQUIRE(actual[row] == Approx(state.score_rows(probes.middleRows(row, 1))[0])
                                       .epsilon(1e-10)
                                       .margin(1e-10));
        }
        REQUIRE(state.log_determinant() == Approx(rebuilt.log_determinant()).margin(1e-5));
    }
    REQUIRE(state.factor_stats().ridge_updates > 10);
    REQUIRE(state.factor_stats().rebuilds >= 3);
    REQUIRE(state.factor_stats().rebuilds < 15);
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    state.save(stream);
    stream.seekg(0);
    OnlineMaxVolState restored;
    REQUIRE(restored.load(stream));
    REQUIRE((state.score_rows(active) - restored.score_rows(active)).cwiseAbs().maxCoeff() < 1e-4);
}

TEST_CASE("Batch ridge updates fall back when rank or spectral separation changes",
          "[online_maxvol][ridge][batch]")
{
    OnlineRowMatrix active(3, 3), candidate(1, 3);
    active << 10., 0., 0., 0., 1., 0., 0., 0., 0.;
    candidate << 0., 0., 2.;
    OnlineMaxVolState state;
    state.enable_batch_updates();
    REQUIRE(state.initialize_from_rows(active));
    REQUIRE(state.maximize_with_candidates(candidate, {}, 1.001, 1) == 1);
    REQUIRE(state.factor_stats().ridge_updates == 0);
    REQUIRE(state.factor_stats().rebuilds == 2);
    OnlineMaxVolState rebuilt;
    REQUIRE(rebuilt.initialize_from_rows(state.active_rows()));
    REQUIRE(state.score_rows(candidate)[0] ==
            Approx(rebuilt.score_rows(candidate)[0]).margin(1e-10));
}

TEST_CASE("Batch ridge refreshes near the stop threshold and without a dominant spectrum",
          "[online_maxvol][ridge][batch]")
{
    SECTION("threshold band")
    {
        OnlineRowMatrix active = OnlineRowMatrix::Zero(3, 3);
        active(0, 0) = 10.;
        active(1, 1) = 1.;
        OnlineMaxVolState state;
        state.enable_batch_updates();
        REQUIRE(state.initialize_from_rows(active));
        OnlineRowMatrix first = 1.002 * active.middleRows(1, 1);
        REQUIRE(state.maximize_with_candidates(first, {}, 1.001, 1) == 1);
        REQUIRE(state.factor_stats().ridge_updates == 1);
        const int before = state.factor_stats().rebuilds;
        OnlineRowMatrix near = 1.00095 * state.active_rows().middleRows(1, 1);
        REQUIRE(state.maximize_with_candidates(near, {}, 1.001, 1) == 0);
        REQUIRE(state.factor_stats().rebuilds == before + 1);
    }
    SECTION("no separated largest singular value")
    {
        OnlineRowMatrix active = OnlineRowMatrix::Identity(4, 4);
        active(3, 3) = 0.;
        OnlineMaxVolState state;
        state.enable_batch_updates();
        REQUIRE(state.initialize_from_rows(active));
        OnlineRowMatrix one = 1.002 * active.middleRows(0, 1);
        REQUIRE(state.maximize_with_candidates(one, {}, 1.001, 1) == 1);
        REQUIRE(state.factor_stats().ridge_updates == 0);
        REQUIRE(state.factor_stats().rebuilds == 2);
    }
}
