// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "maxvol/online_maxvol.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

bool OnlineMaxVolState::update_regularized(int slot, const Eigen::RowVectorXd &replacement)
{
    // Small, well-conditioned update systems are essential here. General
    // states continue to use the full SVD, including wide and rank-changing C.
    if (!batch_updates_ || ridge_residual_map_.rows() != C_.cols() || C_.rows() != C_.cols() ||
        updates_since_factorization_ >= 63)
        return false;
    const auto started = std::chrono::steady_clock::now();

    struct Timer
    {
        double &seconds;
        std::chrono::steady_clock::time_point started;

        ~Timer()
        {
            seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        }
    } timer{factor_stats_.update_seconds, started};

    const Eigen::VectorXd a = inverse_.col(slot);
    const Eigen::VectorXd m = regularized_map_.col(slot);
    const Eigen::RowVectorXd c = replacement * regularized_map_;
    const Eigen::RowVectorXd r = replacement * ridge_residual_map_;
    Eigen::VectorXd t = c.transpose() + regularization_ * a;
    t[slot] -= 1.;
    const Eigen::VectorXd d = r.transpose() - regularization_ * m;
    const double q = c[slot] + regularization_ * a[slot];
    const double rho = d.squaredNorm() + regularization_ * t.squaredNorm();
    const double determinant = q * q + a[slot] * rho;
    if (!std::isfinite(determinant) || determinant < 1e-8 || a[slot] <= 0.)
        return false;

    C_.row(slot) = replacement;
    // Certify a separated largest eigenvalue without diagonalizing C. For a
    // normalized v and theta=||Cv||^2, eigenvalue_2 <= ||C||_F^2-theta.
    // Once theta exceeds this bound, the Rayleigh residual bounds the error
    // in eigenvalue_1 by ||C^T C v-theta v||^2 / (2 theta-||C||_F^2).
    const double trace = C_.squaredNorm();
    double theta = 0.;
    bool certified = false;
    for (int iteration = 0; iteration < 32; ++iteration)
    {
        const Eigen::VectorXd product = C_.transpose() * (C_ * dominant_direction_);
        const double norm = product.norm();
        if (!std::isfinite(norm) || norm == 0.)
            return false;
        dominant_direction_ = product / norm;
        const Eigen::VectorXd cv = C_ * dominant_direction_;
        theta = cv.squaredNorm();
        const double gap = 2. * theta - trace;
        if (gap > 0.)
        {
            const Eigen::VectorXd residual = C_.transpose() * cv - theta * dominant_direction_;
            if (residual.squaredNorm() / gap <= 1e-14 * theta)
            {
                certified = true;
                break;
            }
        }
    }
    if (!certified || !std::isfinite(theta) || theta <= 0.)
        return false;
    const double rank_tolerance =
        std::numeric_limits<double>::epsilon() * C_.cols() * std::sqrt(theta);
    // One certified missing direction suffices. If it was restored, an SVD
    // decides whether the matrix is now full rank or has another null vector.
    if ((C_ * null_direction_).norm() > 0.25 * rank_tolerance)
        return false;
    const double next_lambda =
        std::max(std::numeric_limits<double>::min(),
                 std::max(1e-12, std::numeric_limits<double>::epsilon() * C_.cols()) * theta);
    const double delta = next_lambda - regularization_;
    const double ratio = std::abs(delta) / regularization_;
    if (!std::isfinite(ratio) || ratio > 0.01)
        return false;

    // Rank-two row replacement, evaluated through M=C^T H and R=I-M C.
    // R is maintained directly: constructing it by subtraction (or evaluating
    // b-bMC after accumulated M updates) amplifies roundoff in deficient C.
    inverse_.noalias() -= (rho * (a * a.transpose()) + q * (a * t.transpose() + t * a.transpose()) -
                           a[slot] * (t * t.transpose())) /
                          determinant;
    regularized_map_.noalias() += d * ((q * a - a[slot] * t) / determinant).transpose() -
                                  m * ((rho * a + q * t) / determinant).transpose();
    ridge_residual_map_.noalias() +=
        (rho * (m * m.transpose()) - q * (m * d.transpose() + d * m.transpose()) -
         a[slot] * (d * d.transpose())) /
        determinant;
    double log_shift = 0.;
    if (delta != 0.)
    {
        // Change lambda as required by the score definition. The convergent
        // resolvent series has operator remainder <= ratio^(n+1)/(1-ratio).
        // This is a shift of the current factor, not a frozen-lambda score.
        const Eigen::MatrixXd E = delta * inverse_;
        Eigen::MatrixXd term = -E;
        Eigen::MatrixXd correction = term;
        log_shift = E.trace();
        double remainder = ratio * ratio / (1. - ratio);
        for (int order = 2; remainder > 1e-10; ++order)
        {
            if (order > 8)
                return false;
            term = (-term * E).eval();
            correction += term;
            log_shift -= term.trace() / order;
            remainder *= ratio;
        }
        inverse_ += (inverse_ * correction).eval();
        const Eigen::MatrixXd previous_map = regularized_map_;
        regularized_map_ += (regularized_map_ * correction).eval();
        ridge_residual_map_.noalias() += delta * previous_map * regularized_map_.transpose();
    }
    inverse_ = (0.5 * (inverse_ + inverse_.transpose())).eval();
    ridge_residual_map_ = (0.5 * (ridge_residual_map_ + ridge_residual_map_.transpose())).eval();
    if (!inverse_.allFinite() || !regularized_map_.allFinite() ||
        !ridge_residual_map_.allFinite() || inverse_.diagonal().minCoeff() <= 0.)
        return false;
    // Cheap deterministic consistency probes supplement the periodic SVD.
    // They test the two independently maintained operators, without forming
    // the ill-conditioned normal-equation residual C^T H-M.
    for (int column : {0, int(C_.cols() / 2), int(C_.cols() - 1)})
    {
        Eigen::VectorXd residual =
            ridge_residual_map_.col(column) + regularized_map_ * C_.col(column);
        residual[column] -= 1.;
        if (residual.norm() > 1e-6)
            return false;
    }
    regularization_ = next_lambda;
    log_det_ += 0.5 * (std::log(determinant) + log_shift);
    ++updates_since_factorization_;
    ++factor_stats_.ridge_updates;
    return true;
}
