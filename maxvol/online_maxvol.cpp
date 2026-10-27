// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "maxvol/online_maxvol.h"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>

namespace {

constexpr std::uint32_t kOnlineMagic = 0x4f4d5653; // OMVS
constexpr std::uint32_t kOnlineVersion = 1;
constexpr int kDriftCheckInterval = 16;
constexpr int kRefactorInterval = 128;
constexpr double kDriftTolerance = 1e-8;

void write_ref(std::ostream &os, const BRowRef &ref)
{
    const int kind = static_cast<int>(ref.kind);
    os.write(reinterpret_cast<const char *>(&ref.cfg_index), sizeof(int));
    os.write(reinterpret_cast<const char *>(&ref.local_row), sizeof(int));
    os.write(reinterpret_cast<const char *>(&ref.global_row), sizeof(int));
    os.write(reinterpret_cast<const char *>(&kind), sizeof(int));
    os.write(reinterpret_cast<const char *>(&ref.atom_index), sizeof(int));
    os.write(reinterpret_cast<const char *>(&ref.component), sizeof(int));
}

bool read_ref(std::istream &is, BRowRef &ref)
{
    int kind = static_cast<int>(BRowKind::Unknown);
    is.read(reinterpret_cast<char *>(&ref.cfg_index), sizeof(int));
    is.read(reinterpret_cast<char *>(&ref.local_row), sizeof(int));
    is.read(reinterpret_cast<char *>(&ref.global_row), sizeof(int));
    is.read(reinterpret_cast<char *>(&kind), sizeof(int));
    is.read(reinterpret_cast<char *>(&ref.atom_index), sizeof(int));
    is.read(reinterpret_cast<char *>(&ref.component), sizeof(int));
    if (!is || kind < static_cast<int>(BRowKind::Energy) ||
        kind > static_cast<int>(BRowKind::Unknown))
        return false;
    ref.kind = static_cast<BRowKind>(kind);
    return true;
}

} // namespace

void OnlineMaxVolState::clear()
{
    C_.resize(0, 0);
    inverse_.resize(0, 0);
    selected_refs_.clear();
    log_det_ = 0.0;
    regularization_ = 0.0;
    factor_mode_ = FactorMode::None;
    valid_ = false;
    updates_since_factorization_ = 0;
    last_swaps_.clear();
}

void OnlineMaxVolState::reset_identity(int ncols, double init_value)
{
    clear();
    if (ncols <= 0 || !std::isfinite(init_value) || init_value == 0.0)
        return;
    C_ = init_value * OnlineRowMatrix::Identity(ncols, ncols);
    selected_refs_.assign(static_cast<size_t>(ncols), BRowRef{});
    refactorize();
}

bool OnlineMaxVolState::initialize_from_rows(
    const Eigen::Ref<const OnlineRowMatrix> &C,
    const std::vector<BRowRef> &refs)
{
    clear();
    if (C.rows() <= 0 || C.cols() <= 0 || C.rows() > C.cols() ||
        !C.allFinite())
        return false;
    if (!refs.empty() && refs.size() != static_cast<size_t>(C.rows()))
        return false;
    C_ = C;
    selected_refs_ = refs;
    if (selected_refs_.empty())
        selected_refs_.assign(static_cast<size_t>(C.rows()), BRowRef{});
    return refactorize();
}

bool OnlineMaxVolState::refactorize()
{
    const int rows = static_cast<int>(C_.rows());
    const int cols = static_cast<int>(C_.cols());
    if (rows <= 0 || cols <= 0) {
        valid_ = false;
        return false;
    }

    const Eigen::MatrixXd dense = C_;
    valid_ = false;
    factor_mode_ = FactorMode::None;
    regularization_ = 0.0;
    if (rows == cols && refactorize_square(dense))
        return true;
    return refactorize_gram(dense);
}

bool OnlineMaxVolState::refactorize_square(const Eigen::MatrixXd &dense)
{
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(dense);
    const double rcond = lu.rcond();
    const double tol = std::numeric_limits<double>::epsilon()
                     * static_cast<double>(dense.rows());
    if (!std::isfinite(rcond) || rcond <= tol)
        return false;

    inverse_ = lu.solve(Eigen::MatrixXd::Identity(dense.rows(), dense.cols()));
    if (!inverse_.allFinite())
        return false;

    log_det_ = 0.0;
    const auto diag = lu.matrixLU().diagonal();
    for (int i = 0; i < diag.size(); ++i) {
        const double pivot = std::abs(diag[i]);
        if (!std::isfinite(pivot) || pivot <= 0.0)
            return false;
        log_det_ += std::log(pivot);
    }
    regularization_ = 0.0;
    factor_mode_ = FactorMode::SquareInverse;
    valid_ = true;
    updates_since_factorization_ = 0;
    return true;
}

bool OnlineMaxVolState::refactorize_gram(const Eigen::MatrixXd &dense)
{
    const int rows = static_cast<int>(dense.rows());
    const int cols = static_cast<int>(dense.cols());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        dense, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd &s = svd.singularValues();
    const double smax = s.size() > 0 && std::isfinite(s[0])
                      ? std::max(0.0, s[0]) : 0.0;
    const double sv_tol = std::numeric_limits<double>::epsilon()
                        * static_cast<double>(std::max(rows, cols)) * smax;
    bool exact = s.size() == rows && smax > 0.0;
    for (int i = 0; i < s.size(); ++i)
        exact = exact && std::isfinite(s[i]) && s[i] > sv_tol;

    regularization_ = 0.0;
    if (!exact) {
        const double scale2 = smax > 0.0 ? smax * smax : 1.0;
        const double relative = std::max(
            1e-12, std::numeric_limits<double>::epsilon()
                 * static_cast<double>(std::max(rows, cols)));
        regularization_ = std::max(std::numeric_limits<double>::min(),
                                   relative * scale2);
    }

    Eigen::VectorXd inv_eigs(rows);
    log_det_ = 0.0;
    for (int i = 0; i < rows; ++i) {
        const double si = i < s.size() && std::isfinite(s[i]) ? s[i] : 0.0;
        const double eig = si * si + regularization_;
        if (!std::isfinite(eig) || eig <= 0.0)
            return false;
        inv_eigs[i] = 1.0 / eig;
        log_det_ += 0.5 * std::log(eig);
    }
    inverse_ = svd.matrixU() * inv_eigs.asDiagonal() * svd.matrixU().transpose();
    inverse_ = 0.5 * (inverse_ + inverse_.transpose()).eval();
    if (!inverse_.allFinite())
        return false;
    factor_mode_ = exact ? FactorMode::GramInverse
                         : FactorMode::RegularizedGramInverse;
    valid_ = true;
    updates_since_factorization_ = 0;
    return true;
}

int OnlineMaxVolState::append_candidates(
    const Eigen::Ref<const OnlineRowMatrix> &candidates,
    const std::vector<BRowRef> &candidate_refs,
    int append_limit)
{
    const int room = static_cast<int>(C_.cols() - C_.rows());
    int count = std::min(room, static_cast<int>(candidates.rows()));
    if (append_limit > 0) count = std::min(count, append_limit);
    if (count <= 0) return 0;

    std::vector<Eigen::RowVectorXd> basis;
    basis.reserve(static_cast<size_t>(C_.cols()));
    auto residual_against_basis = [&](const Eigen::RowVectorXd &row) {
        Eigen::RowVectorXd residual = row;
        // Two passes control loss of orthogonality for nearly dependent rows.
        for (int pass = 0; pass < 2; ++pass)
            for (const auto &q : basis)
                residual.noalias() -= residual.dot(q) * q;
        return residual;
    };
    auto extend_basis = [&](const Eigen::RowVectorXd &row) {
        Eigen::RowVectorXd residual = residual_against_basis(row);
        const double norm = residual.norm();
        const double tol = std::numeric_limits<double>::epsilon()
                         * static_cast<double>(std::max(C_.rows(), C_.cols()))
                         * row.norm();
        if (std::isfinite(norm) && norm > tol)
            basis.push_back(residual / norm);
    };
    for (int i = 0; i < C_.rows(); ++i)
        extend_basis(C_.row(i));

    std::vector<char> used(static_cast<size_t>(candidates.rows()), 0);
    std::vector<int> chosen;
    chosen.reserve(static_cast<size_t>(count));
    for (int step = 0; step < count; ++step) {
        int best = -1;
        double best_residual = -1.0;
        double best_norm = -1.0;
        Eigen::RowVectorXd best_vector;
        for (int row = 0; row < candidates.rows(); ++row) {
            if (used[static_cast<size_t>(row)]) continue;
            Eigen::RowVectorXd residual = residual_against_basis(candidates.row(row));
            const double residual2 = residual.squaredNorm();
            const double norm2 = candidates.row(row).squaredNorm();
            if (residual2 > best_residual ||
                (residual2 == best_residual && norm2 > best_norm)) {
                best = row;
                best_residual = residual2;
                best_norm = norm2;
                best_vector = std::move(residual);
            }
        }
        if (best < 0) break;
        const double best_length = std::sqrt(std::max(0.0, best_residual));
        const double independence_tol = std::numeric_limits<double>::epsilon()
            * static_cast<double>(std::max(C_.rows(), C_.cols()))
            * std::sqrt(std::max(0.0, best_norm));
        if (!std::isfinite(best_length) || best_length <= independence_tol)
            break;
        used[static_cast<size_t>(best)] = 1;
        chosen.push_back(best);
        basis.push_back(best_vector / best_length);
    }

    const int old_rows = static_cast<int>(C_.rows());
    C_.conservativeResize(old_rows + static_cast<int>(chosen.size()), C_.cols());
    selected_refs_.reserve(static_cast<size_t>(C_.rows()));
    for (int i = 0; i < static_cast<int>(chosen.size()); ++i) {
        const int candidate = chosen[static_cast<size_t>(i)];
        C_.row(old_rows + i) = candidates.row(candidate);
        selected_refs_.push_back(candidate_refs.empty()
                               ? BRowRef{} : candidate_refs[static_cast<size_t>(candidate)]);
    }
    if (chosen.empty() || !refactorize())
        return 0;
    return static_cast<int>(chosen.size());
}

Eigen::VectorXd OnlineMaxVolState::score_rows(
    const Eigen::Ref<const OnlineRowMatrix> &candidates) const
{
    ScoreWorkspace workspace;
    return score_rows(candidates, workspace);
}

int OnlineMaxVolState::score_block_rows() const
{
    constexpr std::size_t matrix_budget = 4u * 1024u * 1024u;
    const std::size_t row_bytes =
        2u * sizeof(double) * static_cast<std::size_t>(std::max(1, active_size()));
    return static_cast<int>(std::clamp<std::size_t>(matrix_budget / row_bytes, 1, 4096));
}

OnlineMaxVolState::ScoreView OnlineMaxVolState::score_rows(
    const Eigen::Ref<const OnlineRowMatrix> &candidates,
    ScoreWorkspace &workspace) const
{
    workspace.rows = 0;
    if (!valid_ || candidates.cols() != C_.cols())
        return workspace.scores.head(0);
    const int rows = static_cast<int>(candidates.rows());
    if (rows == 0)
        return workspace.scores.head(0);

    const int block_rows = std::min(rows, score_block_rows());
    const auto ensure_matrix_capacity = [block_rows](
        Eigen::MatrixXd &matrix, int cols) {
        if (matrix.cols() != cols || matrix.rows() < block_rows)
            matrix.resize(block_rows, cols);
    };
    if (workspace.scores.size() < rows)
        workspace.scores.resize(rows);
    workspace.rows = rows;
    auto scores = workspace.scores.head(rows);

    if (factor_mode_ == FactorMode::SquareInverse) {
        ensure_matrix_capacity(workspace.coeff, inverse_.cols());
        for (int first = 0; first < rows; first += block_rows) {
            const int count = std::min(block_rows, rows - first);
            auto coeff = workspace.coeff.topRows(count);
            coeff.noalias() = candidates.middleRows(first, count) * inverse_;
            for (int i = 0; i < count; ++i)
                scores[first + i] = coeff.row(i).cwiseAbs().maxCoeff();
        }
        return scores;
    }

    ensure_matrix_capacity(workspace.z, C_.rows());
    ensure_matrix_capacity(workspace.coeff, inverse_.cols());
    const Eigen::VectorXd inv_diag = inverse_.diagonal().cwiseMax(0.0);
    for (int first = 0; first < rows; first += block_rows) {
        const int count = std::min(block_rows, rows - first);
        const auto block = candidates.middleRows(first, count);
        auto z = workspace.z.topRows(count);
        auto coeff = workspace.coeff.topRows(count);
        z.noalias() = block * C_.transpose();
        coeff.noalias() = z * inverse_;
        for (int row = 0; row < count; ++row) {
            const double residual2 = std::max(
                0.0, block.row(row).squaredNorm() -
                     coeff.row(row).dot(z.row(row)));
            double gain2 = 0.0;
            for (int slot = 0; slot < coeff.cols(); ++slot)
                gain2 = std::max(gain2, coeff(row, slot) * coeff(row, slot) +
                                        residual2 * inv_diag[slot]);
            scores[first + row] = std::sqrt(std::max(0.0, gain2));
        }
    }
    return scores;
}

int OnlineMaxVolState::maximize_with_candidates(
    const Eigen::Ref<const OnlineRowMatrix> &candidates,
    const std::vector<BRowRef> &candidate_refs,
    double threshold, int swap_limit)
{
    last_swaps_.clear();
    if (!valid_ || candidates.rows() <= 0 || candidates.cols() != C_.cols())
        return 0;
    if (!candidate_refs.empty() &&
        candidate_refs.size() != static_cast<size_t>(candidates.rows()))
        return 0;
    if (!candidates.allFinite() || !std::isfinite(threshold))
        return 0;

    // A persistent state may have been initialized before enough independent
    // rows were available. Fill vacant slots before doing replacement swaps.
    int swaps = 0;
    if (C_.rows() < C_.cols()) {
        const int append_limit = swap_limit > 0 ? swap_limit : 0;
        swaps = append_candidates(candidates, candidate_refs, append_limit);
    }
    if (!valid_) return 0;

    // MLIP-style A/B exchange: when a candidate enters C, the outgoing active
    // row remains in the local pool and may re-enter during a later swap.
    OnlineRowMatrix pool = candidates;
    std::vector<BRowRef> pool_refs = candidate_refs;
    if (pool_refs.empty())
        pool_refs.assign(static_cast<size_t>(pool.rows()), BRowRef{});

    const double stop = std::max(1.0, threshold);
    Eigen::MatrixXd square_coeff;
    bool square_coeff_valid = false;
    while (swap_limit <= 0 || swaps < swap_limit) {
        Eigen::Index best_row = -1, best_slot = -1;
        double best = 0.0;
        Eigen::MatrixXd coeff;
        Eigen::RowVectorXd replacement_coeff;
        if (factor_mode_ == FactorMode::SquareInverse) {
            if (!square_coeff_valid) {
                square_coeff = pool * inverse_;
                square_coeff_valid = true;
            }
            best = square_coeff.cwiseAbs().maxCoeff(&best_row, &best_slot);
            if (best_row >= 0)
                replacement_coeff = square_coeff.row(best_row);
        } else {
            const Eigen::MatrixXd z = pool * C_.transpose();
            coeff = z * inverse_;
            const Eigen::VectorXd inv_diag = inverse_.diagonal().cwiseMax(0.0);
            for (int row = 0; row < pool.rows(); ++row) {
                const double residual2 = std::max(
                    0.0, pool.row(row).squaredNorm() -
                         coeff.row(row).dot(z.row(row)));
                for (int slot = 0; slot < C_.rows(); ++slot) {
                    const double gain = std::sqrt(std::max(
                        0.0, coeff(row, slot) * coeff(row, slot) +
                             residual2 * inv_diag[slot]));
                    if (gain > best) {
                        best = gain;
                        best_row = row;
                        best_slot = slot;
                    }
                }
            }
            if (best_row >= 0)
                replacement_coeff = coeff.row(best_row);
        }
        if (!std::isfinite(best) || best <= stop)
            break;

        const Eigen::RowVectorXd replacement = pool.row(best_row);
        const Eigen::RowVectorXd outgoing = C_.row(best_slot);
        const BRowRef entering_ref = pool_refs[static_cast<size_t>(best_row)];
        const BRowRef outgoing_ref = selected_refs_[static_cast<size_t>(best_slot)];
        const FactorMode old_mode = factor_mode_;
        bool updated = old_mode == FactorMode::SquareInverse
                     ? update_square(replacement_coeff, best_slot, replacement)
                     : update_gram(best_slot, replacement);

        if (!updated) {
            C_.row(best_slot) = replacement;
            if (!refactorize()) {
                C_.row(best_slot) = outgoing;
                refactorize();
                break;
            }
        } else if (old_mode == FactorMode::RegularizedGramInverse) {
            // A missing direction may have been restored. Prefer the exact
            // square/Gram factorization as soon as it becomes available.
            if (!refactorize()) {
                C_.row(best_slot) = outgoing;
                refactorize();
                break;
            }
        }
        selected_refs_[static_cast<size_t>(best_slot)] = entering_ref;
        last_swaps_.push_back({entering_ref, outgoing_ref, best});
        pool.row(best_row) = outgoing;
        pool_refs[static_cast<size_t>(best_row)] = outgoing_ref;

        if (factor_mode_ == FactorMode::SquareInverse) {
            if (old_mode == FactorMode::SquareInverse && updated &&
                updates_since_factorization_ > 0) {
                const double denom = replacement_coeff[best_slot];
                const Eigen::RowVectorXd delta = replacement_coeff
                    - Eigen::RowVectorXd::Unit(C_.rows(), best_slot);
                const Eigen::VectorXd column = square_coeff.col(best_slot);
                square_coeff.noalias() -= (column * delta) / denom;
                square_coeff.row(best_row) =
                    Eigen::RowVectorXd::Unit(C_.rows(), best_slot) - delta / denom;
                square_coeff_valid = square_coeff.allFinite();
            } else {
                square_coeff_valid = false;
            }
        } else {
            square_coeff_valid = false;
        }
        ++swaps;
    }

    return swaps;
}

bool OnlineMaxVolState::update_square(
    const Eigen::RowVectorXd &coeff, int slot,
    const Eigen::RowVectorXd &replacement)
{
    const double denom = coeff[slot];
    if (!std::isfinite(denom) || std::abs(denom) <= 1e-12)
        return false;
    const Eigen::RowVectorXd delta = coeff -
        Eigen::RowVectorXd::Unit(C_.rows(), slot);
    const Eigen::VectorXd column = inverse_.col(slot);
    inverse_.noalias() -= (column * delta) / denom;
    C_.row(slot) = replacement;
    if (!inverse_.allFinite())
        return false;
    log_det_ += std::log(std::abs(denom));
    ++updates_since_factorization_;
    if (updates_since_factorization_ >= kRefactorInterval ||
        (updates_since_factorization_ % kDriftCheckInterval == 0 &&
         factorization_error() > kDriftTolerance))
        return refactorize();
    return true;
}

bool OnlineMaxVolState::update_gram(
    int slot, const Eigen::RowVectorXd &replacement)
{
    const int rows = static_cast<int>(C_.rows());
    const Eigen::RowVectorXd difference = replacement - C_.row(slot);
    const Eigen::VectorXd cross = C_ * difference.transpose();
    const double delta = difference.squaredNorm();

    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(rows, 2);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(rows, 2);
    U(slot, 0) = 1.0;
    U.col(1) = cross;
    V.col(0) = cross;
    V(slot, 0) += delta;
    V(slot, 1) = 1.0;

    const Eigen::MatrixXd inv_u = inverse_ * U;
    const Eigen::Matrix2d middle =
        Eigen::Matrix2d::Identity() + V.transpose() * inv_u;
    const double determinant = middle.determinant();
    if (!middle.allFinite() || !std::isfinite(determinant) ||
        std::abs(determinant) <= 1e-14)
        return false;

    const Eigen::MatrixXd right = V.transpose() * inverse_;
    inverse_.noalias() -= inv_u * middle.inverse() * right;
    inverse_ = 0.5 * (inverse_ + inverse_.transpose()).eval();
    C_.row(slot) = replacement;
    if (!inverse_.allFinite())
        return false;
    log_det_ += 0.5 * std::log(std::abs(determinant));
    ++updates_since_factorization_;
    if (updates_since_factorization_ >= kRefactorInterval ||
        (updates_since_factorization_ % kDriftCheckInterval == 0 &&
         factorization_error() > kDriftTolerance))
        return refactorize();
    return true;
}

double OnlineMaxVolState::factorization_error() const
{
    const int rows = static_cast<int>(C_.rows());
    if (rows <= 0)
        return std::numeric_limits<double>::infinity();
    const int samples[] = {0, rows / 2, rows - 1};
    double error = 0.0;
    for (int sample : samples) {
        Eigen::VectorXd product;
        if (factor_mode_ == FactorMode::SquareInverse) {
            product = C_ * inverse_.col(sample);
        } else {
            const Eigen::VectorXd projected =
                C_.transpose() * inverse_.col(sample);
            product = C_ * projected + regularization_ * inverse_.col(sample);
        }
        product[sample] -= 1.0;
        error = std::max(error, product.cwiseAbs().maxCoeff());
    }
    return error;
}

void OnlineMaxVolState::save(std::ostream &os) const
{
    const int rows = static_cast<int>(C_.rows());
    const int cols = static_cast<int>(C_.cols());
    os.write(reinterpret_cast<const char *>(&kOnlineMagic), sizeof(kOnlineMagic));
    os.write(reinterpret_cast<const char *>(&kOnlineVersion), sizeof(kOnlineVersion));
    os.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    os.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
    os.write(reinterpret_cast<const char *>(&log_det_), sizeof(log_det_));
    for (const auto &ref : selected_refs_)
        write_ref(os, ref);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            os.write(reinterpret_cast<const char *>(&C_(i, j)), sizeof(double));
}

bool OnlineMaxVolState::load(std::istream &is)
{
    clear();
    std::uint32_t magic = 0, version = 0;
    int rows = 0, cols = 0;
    double stored_log_det = 0.0;
    is.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    is.read(reinterpret_cast<char *>(&rows), sizeof(rows));
    is.read(reinterpret_cast<char *>(&cols), sizeof(cols));
    is.read(reinterpret_cast<char *>(&stored_log_det), sizeof(stored_log_det));
    if (!is || magic != kOnlineMagic || version != kOnlineVersion ||
        rows <= 0 || cols <= 0 || rows > cols || rows > 1000000 || cols > 1000000)
        return false;

    selected_refs_.resize(static_cast<size_t>(rows));
    for (auto &ref : selected_refs_)
        if (!read_ref(is, ref)) { clear(); return false; }
    C_.resize(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            is.read(reinterpret_cast<char *>(&C_(i, j)), sizeof(double));
    if (!is || !C_.allFinite()) { clear(); return false; }
    return refactorize();
}
