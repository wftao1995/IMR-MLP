// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "maxvol/online_maxvol.h"
#include "maxvol/mpi_batch.h"
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

void OnlineMaxVolState::broadcast_runtime(MPI_Comm comm, int root)
{
#ifdef IMRMLP_MPI
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size == 1)
        return;
    MaxVolMPI::local(comm,
                     [&]
                     {
                         if (root < 0 || root >= size || (rank == root && !valid_))
                             throw std::runtime_error("Invalid canonical MaxVol state");
                     });
    const auto matrix = [&](auto &value)
    {
        int shape[2] = {int(value.rows()), int(value.cols())};
        MPI_Bcast(shape, 2, MPI_INT, root, comm);
        MaxVolMPI::local(comm,
                         [&]
                         {
                             if (rank != root)
                                 value.resize(shape[0], shape[1]);
                         });
        // Keep each MPI count in int range without a packed matrix copy.
        constexpr Eigen::Index chunk = 1 << 20;
        for (Eigen::Index first = 0; first < value.size(); first += chunk)
            MPI_Bcast(value.data() + first, int(std::min(chunk, value.size() - first)), MPI_DOUBLE,
                      root, comm);
    };
    matrix(C_);
    matrix(inverse_);
    matrix(regularized_map_);
    matrix(ridge_residual_map_);
    matrix(dominant_direction_);
    matrix(null_direction_);
    int control[] = {
        int(selected_refs_.size()), int(last_swaps_.size()),      int(factor_mode_),
        int(batch_updates_),        updates_since_factorization_, factor_stats_.rebuilds,
        factor_stats_.ridge_updates};
    MPI_Bcast(control, 7, MPI_INT, root, comm);
    MaxVolMPI::local(comm,
                     [&]
                     {
                         if (rank != root)
                         {
                             selected_refs_.resize(control[0]);
                             last_swaps_.resize(control[1]);
                         }
                     });
    // Pack only typed metadata, not the dense factors. One message avoids
    // thousands of tiny reference broadcasts on a multi-node interconnect.
    std::vector<int> references;
    std::vector<double> gains;
    MaxVolMPI::local(comm,
                     [&]
                     {
                         references.resize(6 * (selected_refs_.size() + 2 * last_swaps_.size()));
                         gains.resize(last_swaps_.size());
                     });
    size_t offset = 0;
    const auto pack = [&](const BRowRef &r)
    {
        for (int field :
             {r.cfg_index, r.local_row, r.global_row, int(r.kind), r.atom_index, r.component})
            references[offset++] = field;
    };
    if (rank == root)
    {
        for (const auto &r : selected_refs_)
            pack(r);
        for (size_t i = 0; i < last_swaps_.size(); ++i)
        {
            const auto &step = last_swaps_[i];
            pack(step.entering);
            pack(step.outgoing);
            gains[i] = step.gain;
        }
    }
    const auto send = [&](auto &values, MPI_Datatype type)
    {
        constexpr size_t chunk = 1 << 20;
        for (size_t first = 0; first < values.size(); first += chunk)
            MPI_Bcast(values.data() + first, int(std::min(chunk, values.size() - first)), type,
                      root, comm);
    };
    send(references, MPI_INT);
    send(gains, MPI_DOUBLE);
    if (rank != root)
    {
        offset = 0;
        const auto unpack = [&](BRowRef &r)
        {
            const int *f = references.data() + offset;
            r = {f[0], f[1], f[2], BRowKind(f[3]), f[4], f[5]};
            offset += 6;
        };
        for (auto &r : selected_refs_)
            unpack(r);
        for (size_t i = 0; i < last_swaps_.size(); ++i)
        {
            auto &step = last_swaps_[i];
            unpack(step.entering);
            unpack(step.outgoing);
            step.gain = gains[i];
        }
    }
    double scalars[] = {log_det_, regularization_, factor_stats_.rebuild_seconds,
                        factor_stats_.update_seconds};
    MPI_Bcast(scalars, 4, MPI_DOUBLE, root, comm);
    factor_mode_ = FactorMode(control[2]);
    batch_updates_ = control[3];
    updates_since_factorization_ = control[4];
    factor_stats_ = {control[5], control[6], scalars[2], scalars[3]};
    log_det_ = scalars[0];
    regularization_ = scalars[1];
    valid_ = true;
#else
    (void)comm;
    (void)root;
#endif
}
