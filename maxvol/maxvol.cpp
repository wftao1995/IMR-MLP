// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "maxvol/maxvol.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>

namespace {

// ── helper: build a membership mask from a vector ──────────────────────────
static std::vector<char> to_mask(int n, const std::vector<int> &v)
{
    std::vector<char> mask(n, 0);
    for (int idx : v)
        if (idx >= 0 && idx < n) mask[idx] = 1;
    return mask;
}

static void hash_combine_u64(std::uint64_t &h, std::uint64_t v)
{
    h ^= v;
    h *= 1099511628211ULL;
}

template <typename MatrixType>
static std::uint64_t cache_signature(
    const std::vector<int> &selected,
    const MatrixType &A)
{
    std::uint64_t h = 1469598103934665603ULL;
    hash_combine_u64(h, static_cast<std::uint64_t>(A.rows()));
    hash_combine_u64(h, static_cast<std::uint64_t>(A.cols()));
    hash_combine_u64(h, static_cast<std::uint64_t>(selected.size()));
    for (int idx : selected) {
        hash_combine_u64(h, static_cast<std::uint64_t>(idx));
        if (idx < 0 || idx >= A.rows()) {
            hash_combine_u64(h, 0xffffffffffffffffULL);
            continue;
        }
        for (int j = 0; j < A.cols(); ++j) {
            std::uint64_t bits = 0;
            const double v = A(idx, j);
            std::memcpy(&bits, &v, sizeof(bits));
            hash_combine_u64(h, bits);
        }
    }
    return h;
}

template <typename MatrixType>
static bool valid_selection(const MatrixType &A,
                            const std::vector<int> &selected)
{
    std::vector<char> seen(static_cast<size_t>(A.rows()), 0);
    for (int idx : selected) {
        if (idx < 0 || idx >= A.rows() || seen[static_cast<size_t>(idx)])
            return false;
        seen[static_cast<size_t>(idx)] = 1;
    }
    return true;
}

struct SvdFactor
{
    Eigen::MatrixXd pinv;
    double log_volume = std::numeric_limits<double>::lowest();
    bool full_rank = false;
};

static SvdFactor factorize_for_maxvol(const Eigen::Ref<const Eigen::MatrixXd> &C)
{
    SvdFactor out;
    const int rows = static_cast<int>(C.rows());
    const int cols = static_cast<int>(C.cols());
    const int dim = std::min(rows, cols);
    out.pinv = Eigen::MatrixXd::Zero(cols, rows);

    if (rows == 0 || cols == 0 || dim == 0)
        return out;

    // A well-conditioned square state needs an inverse, not a full SVD.
    // Keep a conservative condition threshold so near-singular states retain
    // the singular-value truncation and diagnostics of the fallback below.
    if (rows == cols) {
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(C);
        const double rcond = lu.rcond();
        const double min_rcond = std::sqrt(std::numeric_limits<double>::epsilon())
                                 * static_cast<double>(rows);
        if (std::isfinite(rcond) && rcond > min_rcond) {
            Eigen::MatrixXd inverse = lu.solve(Eigen::MatrixXd::Identity(rows, cols));
            const double log_volume =
                lu.matrixLU().diagonal().cwiseAbs().array().log().sum();
            if (inverse.allFinite() && std::isfinite(log_volume)) {
                out.pinv = std::move(inverse);
                out.log_volume = log_volume;
                out.full_rank = true;
                return out;
            }
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd &s = svd.singularValues();
    const double scale = (s.size() > 0 && std::isfinite(s(0)) && s(0) > 0.0)
                       ? s(0) : 1.0;
    const double tol = std::numeric_limits<double>::epsilon()
                     * static_cast<double>(std::max(rows, cols))
                     * scale;
    int rank = 0;
    for (int i = 0; i < s.size(); ++i) {
        if (std::isfinite(s(i)) && s(i) > tol) ++rank;
        else break;
    }

    if (rank > 0) {
        out.pinv = svd.matrixV().leftCols(rank)
                 * s.head(rank).cwiseInverse().asDiagonal()
                 * svd.matrixU().leftCols(rank).transpose();
    }

    out.full_rank = (rank == dim);
    const double floor_sv = std::max(tol, std::numeric_limits<double>::min());
    out.log_volume = 0.0;
    for (int i = 0; i < dim; ++i) {
        const double si = (i < s.size() && std::isfinite(s(i))) ? s(i) : 0.0;
        out.log_volume += std::log(std::max(si, floor_sv));
    }
    return out;
}

struct RowVolumeFactor
{
    Eigen::MatrixXd coeff_map;
    Eigen::VectorXd gram_inv_diag;
};

static RowVolumeFactor factorize_row_volume(
    const Eigen::Ref<const Eigen::MatrixXd> &C)
{
    RowVolumeFactor out;
    const int rows = static_cast<int>(C.rows());
    const int cols = static_cast<int>(C.cols());
    out.coeff_map = Eigen::MatrixXd::Zero(cols, rows);
    out.gram_inv_diag = Eigen::VectorXd::Zero(rows);
    if (rows <= 0 || cols <= 0 || rows > cols)
        return out;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd &s = svd.singularValues();
    const double smax = s.size() > 0 && std::isfinite(s[0])
                      ? std::max(0.0, s[0]) : 0.0;
    const double sv_tol = std::numeric_limits<double>::epsilon()
                        * static_cast<double>(std::max(rows, cols)) * smax;
    bool full_row_rank = s.size() == rows && smax > 0.0;
    for (int i = 0; i < s.size(); ++i)
        full_row_rank = full_row_rank && std::isfinite(s[i]) && s[i] > sv_tol;

    double lambda = 0.0;
    if (!full_row_rank) {
        const double scale2 = smax > 0.0 ? smax * smax : 1.0;
        const double relative = std::max(
            1e-12, std::numeric_limits<double>::epsilon()
                 * static_cast<double>(std::max(rows, cols)));
        lambda = std::max(std::numeric_limits<double>::min(),
                          relative * scale2);
    }

    Eigen::VectorXd inv_eigs(rows);
    for (int i = 0; i < rows; ++i) {
        const double si = i < s.size() && std::isfinite(s[i]) ? s[i] : 0.0;
        const double eig = si * si + lambda;
        inv_eigs[i] = eig > 0.0 ? 1.0 / eig : 0.0;
    }
    const Eigen::MatrixXd gram_inv = svd.matrixU()
        * inv_eigs.asDiagonal() * svd.matrixU().transpose();
    out.coeff_map.noalias() = C.transpose() * gram_inv;
    out.gram_inv_diag = gram_inv.diagonal().cwiseMax(0.0);
    return out;
}

template <typename MatrixType>
static std::vector<int> augment_rows_d_optimal(
    const MatrixType &A, const std::vector<int> &current, int add_k)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (add_k <= 0 || m <= 0 || n <= 0)
        return current;

    const std::vector<char> selected_mask = to_mask(m, current);
    std::vector<int> complement;
    complement.reserve(static_cast<size_t>(m - current.size()));
    for (int i = 0; i < m; ++i)
        if (!selected_mask[static_cast<size_t>(i)]) complement.push_back(i);

    const int count = std::min(add_k, static_cast<int>(complement.size()));
    std::vector<int> result = current;
    result.reserve(current.size() + static_cast<size_t>(count));
    if (count == static_cast<int>(complement.size())) {
        result.insert(result.end(), complement.begin(), complement.end());
        return result;
    }

    Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(n, n);
    double scale = 0.0;
    for (int i = 0; i < m; ++i)
        scale = std::max(scale, A.row(i).squaredNorm());
    for (int idx : current)
        gram.noalias() += A.row(idx).transpose() * A.row(idx);

    const double relative = std::max(
        1e-12, std::numeric_limits<double>::epsilon()
             * static_cast<double>(std::max(1, n)));
    const double lambda = relative * (scale > 0.0 ? scale : 1.0);
    gram.diagonal().array() += lambda;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(gram);
    if (ldlt.info() != Eigen::Success)
        return result;
    Eigen::MatrixXd inverse = ldlt.solve(Eigen::MatrixXd::Identity(n, n));
    inverse = 0.5 * (inverse + inverse.transpose()).eval();
    if (!inverse.allFinite())
        return result;

    Eigen::MatrixXd candidates(complement.size(), n);
    for (int i = 0; i < static_cast<int>(complement.size()); ++i)
        candidates.row(i) = A.row(complement[static_cast<size_t>(i)]);
    Eigen::MatrixXd transformed = candidates * inverse;
    Eigen::VectorXd leverage = (transformed.array() * candidates.array()).rowwise().sum();
    std::vector<char> used(complement.size(), 0);

    for (int step = 0; step < count; ++step) {
        int best = -1;
        double best_value = -1.0;
        for (int i = 0; i < leverage.size(); ++i) {
            if (!used[static_cast<size_t>(i)] && leverage[i] > best_value) {
                best = i;
                best_value = leverage[i];
            }
        }
        if (best < 0) break;

        result.push_back(complement[static_cast<size_t>(best)]);
        used[static_cast<size_t>(best)] = 1;

        const Eigen::RowVectorXd transformed_best = transformed.row(best);
        const Eigen::VectorXd cross = transformed * candidates.row(best).transpose();
        const double denom = 1.0 + std::max(0.0, leverage[best]);
        if (!std::isfinite(denom) || denom <= 0.0)
            break;
        transformed.noalias() -= (cross * transformed_best) / denom;
        leverage.array() -= cross.array().square() / denom;
        leverage = leverage.cwiseMax(0.0);
        leverage[best] = -1.0;
    }
    return result;
}

struct GeneralizedSwap
{
    int row = -1;
    int slot = -1;
    double gain = 1.0;
};

/// Find the best one-row exchange for the generalized row volume
/// sqrt(det(C C^T)) (wide C) or sqrt(det(C^T C)) (tall C).
template <typename MatrixType>
static GeneralizedSwap find_generalized_swap(
    const MatrixType &A, const std::vector<int> &selected, double stop)
{
    GeneralizedSwap best;
    best.gain = stop;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = static_cast<int>(selected.size());
    if (k <= 0 || k == n)
        return best;

    Eigen::MatrixXd C(k, n);
    for (int i = 0; i < k; ++i)
        C.row(i) = A.row(selected[static_cast<size_t>(i)]);
    const std::vector<char> selected_mask = to_mask(m, selected);

    if (k < n) {
        const RowVolumeFactor factor = factorize_row_volume(C);
        const Eigen::MatrixXd coeff = A * factor.coeff_map;
        const Eigen::MatrixXd z = A * C.transpose();
        for (int row = 0; row < m; ++row) {
            if (selected_mask[static_cast<size_t>(row)]) continue;
            const double residual2 = std::max(
                0.0, A.row(row).squaredNorm() - coeff.row(row).dot(z.row(row)));
            for (int slot = 0; slot < k; ++slot) {
                const double gain2 = coeff(row, slot) * coeff(row, slot)
                                   + residual2 * factor.gram_inv_diag[slot];
                const double gain = std::sqrt(std::max(0.0, gain2));
                if (std::isfinite(gain) && gain > best.gain) {
                    best = {row, slot, gain};
                }
            }
        }
        return best;
    }

    const SvdFactor factor = factorize_for_maxvol(C);
    const Eigen::MatrixXd coeff = A * factor.pinv;
    Eigen::VectorXd active_leverage(k);
    for (int slot = 0; slot < k; ++slot)
        active_leverage[slot] = std::clamp(
            C.row(slot).dot(factor.pinv.col(slot)), 0.0, 1.0);
    for (int row = 0; row < m; ++row) {
        if (selected_mask[static_cast<size_t>(row)]) continue;
        const double candidate_leverage = coeff.row(row).squaredNorm();
        for (int slot = 0; slot < k; ++slot) {
            const double gain2 =
                (1.0 - active_leverage[slot]) * (1.0 + candidate_leverage)
                + coeff(row, slot) * coeff(row, slot);
            const double gain = std::sqrt(std::max(0.0, gain2));
            if (std::isfinite(gain) && gain > best.gain) {
                best = {row, slot, gain};
            }
        }
    }
    return best;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// QR-based initialisation
// ═══════════════════════════════════════════════════════════════════════════
template <typename MatrixType>
std::vector<int> MaxVol::qr_init(const MatrixType &A, int k)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (k > m) k = m;

    Eigen::MatrixXd At = A.transpose();
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(At);
    qr.setThreshold(0);
    const auto &pivots = qr.colsPermutation().indices();

    int npiv = static_cast<int>(pivots.size());
    k = std::min(k, npiv);

    std::vector<int> selected;
    selected.reserve(k);
    for (int i = 0; i < k; ++i)
        selected.push_back(static_cast<int>(pivots[i]));

    return selected;
}

// ═══════════════════════════════════════════════════════════════════════════
// One MaxVol iteration (Sherman-Morrison rank-1 update)
// ═══════════════════════════════════════════════════════════════════════════
template <typename MatrixType>
bool MaxVol::maxvol_iteration(const MatrixType &A,
                              std::vector<int> &selected,
                              Eigen::MatrixXd &B,
                              Eigen::MatrixXd &C,
                              Eigen::MatrixXd &C_inv,
                              int &sm_steps_since_recompute,
                              bool &sm_updates_enabled)
{
    const int m = static_cast<int>(A.rows());
    const int k = static_cast<int>(selected.size());
    const int n = static_cast<int>(A.cols());

    // ── Cold start or forced recompute ─────────────────────────────────
    bool cold_start = (B.rows() == 0);
    if (sm_recompute_every > 0 &&
        sm_steps_since_recompute >= sm_recompute_every)
        cold_start = true;

    if (cold_start) {
        C.resize(k, n);
        for (int i = 0; i < k; ++i)
            C.row(i) = A.row(selected[i]);

        SvdFactor factor = factorize_for_maxvol(C);
        C_inv = factor.pinv;
        B = A * C_inv;

        log_det_ = factor.log_volume;
        cold_start_k_ = k;
        sm_updates_enabled = factor.full_rank;
        sm_steps_since_recompute = 0;
    }

    // ── Find the best swap candidate ───────────────────────────────────
    std::vector<char> sel_mask = to_mask(m, selected);
    const double stop = 1.0 + std::max(0.0, tolerance);
    double max_val = stop;
    int    best_l  = -1;
    int    best_i  = -1;

    // B is column-major. Scan contiguous columns while preserving the old
    // row-first tie break for equal gains.
    for (int i = 0; i < k; ++i) {
        for (int l = 0; l < m; ++l) {
            if (sel_mask[l]) continue;
            double abs_val = std::abs(B(l, i));
            if (abs_val > max_val ||
                (abs_val == max_val && best_l >= 0 && l < best_l)) {
                max_val = abs_val;
                best_l  = l;
                best_i  = i;
            }
        }
    }

    if (best_l < 0) return false;   // converged

    // ── Record swap for history ────────────────────────────────────────
    int old_row = selected[best_i];
    swap_tracker_.emplace_back(best_l, old_row, max_val);

    if (verbose)
        std::cout << "MaxVol swap: row " << old_row << " out, row "
                  << best_l << " in (gain " << max_val << ")\n";

    if (!sm_updates_enabled) {
        selected[best_i] = best_l;
        B.resize(0, 0);  // force SVD recompute next iteration
        return true;
    }

    // ── Sherman-Morrison update ────────────────────────────────────────
    Eigen::RowVectorXd a_old = C.row(best_i);
    Eigen::RowVectorXd a_new = A.row(best_l);
    Eigen::RowVectorXd diff  = a_new - a_old;

    Eigen::RowVectorXd w = diff * C_inv;
    double denom = 1.0 + w[best_i];

    if (!std::isfinite(denom) || std::abs(denom) < 1e-14) {
        if (verbose)
            std::cerr << "MaxVol: SM denom=" << denom
                      << ", falling back to full recompute\n";
        selected[best_i] = best_l;
        B.resize(0, 0);  // force cold start next iteration
        return true;
    }

    // C_inv_new = C_inv - (C_inv[:, best_i] * w) / denom
    Eigen::VectorXd Ci_col = C_inv.col(best_i);
    C_inv.noalias() -= (Ci_col * w) / denom;

    // B_new = B - (B[:, best_i] * w) / denom
    Eigen::VectorXd Bi_col = B.col(best_i);
    B.noalias() -= (Bi_col * w) / denom;

    // Update C and selection
    C.row(best_i) = a_new;
    selected[best_i] = best_l;

    // Update log-determinant: |det(C_new)| = |det(C_old)| * |B[l,i]|
    log_det_ += std::log(std::abs(denom));
    if (!std::isfinite(log_det_)) {
        B.resize(0, 0);  // force SVD recompute next iteration
        log_det_ = std::numeric_limits<double>::lowest();
    }

    ++sm_steps_since_recompute;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API — select_rows
// ═══════════════════════════════════════════════════════════════════════════

template <typename MatrixType>
std::vector<int> MaxVol::select_rows_impl(const MatrixType &A, int k)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());

    swap_tracker_.clear();
    log_det_ = 0.0;
    cold_start_k_ = 0;
    cached_selected_.clear();
    cached_ncols_ = -1;
    cached_pinv_.resize(0, 0);
    cached_active_.resize(0, 0);
    cached_gram_inv_diag_.resize(0);
    cached_active_leverage_.resize(0);
    cached_wide_ = false;
    cached_signature_ = 0;

    if (m == 0 || n == 0 || k <= 0 || !A.allFinite() ||
        !std::isfinite(tolerance))
        return {};

    if (k > m) {
        if (verbose) std::cerr << "MaxVol: k=" << k << " > m=" << m
                               << ", using k=m\n";
        k = m;
    }

    int swap_count = 0;
    std::vector<int> selected;

    if (k < n) {
        // Pivoted QR on A^T greedily maximizes the row volume
        // sqrt(det(C C^T)) without reducing the problem to an arbitrary minor.
        selected = qr_init(A, k);
        if (selected.empty()) return {};
    } else {
        selected = qr_init(A, n);
        if (selected.empty()) return {};

        Eigen::MatrixXd B, C, C_inv;
        int sm_steps = 0;
        bool sm_safe = false;
        for (int iter = 0; iter < max_iterations; ++iter) {
            if (!maxvol_iteration(A, selected, B, C, C_inv, sm_steps, sm_safe))
                break;
            if (max_swaps > 0 && ++swap_count >= max_swaps) break;
        }

        if (k > n)
            selected = augment_rows_d_optimal(A, selected, k - n);

        if (k == n && C.rows() == n && C.cols() == n &&
            C_inv.rows() == n && C_inv.cols() == n && B.rows() != 0) {
            cached_selected_ = selected;
            cached_ncols_ = n;
            cached_pinv_ = C_inv;
            cached_active_ = C;
            cached_gram_inv_diag_.resize(0);
            cached_active_leverage_.resize(0);
            cached_wide_ = false;
            cached_signature_ = cache_signature(selected, A);
        }
    }

    // QR initialization (wide case) and greedy D-optimal augmentation (tall
    // case) are only starting points.  Finish with exact one-row exchanges so
    // the returned rectangular set obeys the same local dominance threshold
    // as the classic square MaxVol path.
    if (k != n) {
        const double stop = 1.0 + std::max(0.0, tolerance);
        while (static_cast<int>(swap_tracker_.size()) < max_iterations &&
               (max_swaps <= 0 || swap_count < max_swaps)) {
            const GeneralizedSwap step =
                find_generalized_swap(A, selected, stop);
            if (step.row < 0) break;
            const int old_row = selected[static_cast<size_t>(step.slot)];
            swap_tracker_.emplace_back(step.row, old_row, step.gain);
            selected[static_cast<size_t>(step.slot)] = step.row;
            ++swap_count;
            if (verbose)
                std::cout << "MaxVol generalized swap: row " << old_row
                          << " out, row " << step.row << " in (gain "
                          << step.gain << ")\n";
        }
    }

    Eigen::MatrixXd final_C(selected.size(), n);
    for (int i = 0; i < static_cast<int>(selected.size()); ++i)
        final_C.row(i) = A.row(selected[static_cast<size_t>(i)]);
    if (k != n || cached_selected_ != selected || !std::isfinite(log_det_))
        log_det_ = factorize_for_maxvol(final_C).log_volume;
    if (cached_selected_ != selected)
        ensure_cache(selected, A);
    cold_start_k_ = static_cast<int>(selected.size());
    return selected;
}

std::vector<int> MaxVol::select_rows(
    const Eigen::Ref<const Eigen::MatrixXd> &A, int k)
{
    return select_rows_impl(A, k);
}

std::vector<int> MaxVol::select_rows(
    const Eigen::Ref<const RowMatrixXd> &A, int k)
{
    return select_rows_impl(A, k);
}

// ═══════════════════════════════════════════════════════════════════════════
// Private helper — SVD cache management
// ═══════════════════════════════════════════════════════════════════════════

template <typename MatrixType>
bool MaxVol::ensure_cache(const std::vector<int> &selected, const MatrixType &A)
{
    const int n = A.cols();
    const std::uint64_t signature = cache_signature(selected, A);
    if (selected == cached_selected_ && n == cached_ncols_ &&
        signature == cached_signature_)
        return true;  // cache already valid

    cached_selected_ = selected;
    cached_ncols_ = n;
    cached_signature_ = signature;

    const int k = static_cast<int>(selected.size());
    Eigen::MatrixXd C(k, n);
    for (int i = 0; i < k; ++i)
        C.row(i) = A.row(selected[i]);
    cached_active_ = C;
    cached_wide_ = k < n;
    if (cached_wide_) {
        RowVolumeFactor factor = factorize_row_volume(C);
        cached_pinv_ = std::move(factor.coeff_map);
        cached_gram_inv_diag_ = std::move(factor.gram_inv_diag);
        cached_active_leverage_.resize(0);
    } else {
        cached_pinv_ = factorize_for_maxvol(C).pinv;
        cached_gram_inv_diag_.resize(0);
        if (k > n)
            cached_active_leverage_ =
                (C * cached_pinv_).diagonal().cwiseMax(0.0).cwiseMin(1.0);
        else
            cached_active_leverage_.resize(0);
    }

    return false;
}

template <typename MatrixType>
void MaxVol::warm_cache_impl(const std::vector<int> &selected,
                             const MatrixType &A)
{
    if (!valid_selection(A, selected) || !A.allFinite()) {
        cached_selected_.clear();
        cached_ncols_ = -1;
        cached_signature_ = 0;
        cached_pinv_.resize(0, 0);
        cached_active_.resize(0, 0);
        cached_gram_inv_diag_.resize(0);
        cached_active_leverage_.resize(0);
        cached_wide_ = false;
        return;
    }
    ensure_cache(selected, A);
}

void MaxVol::warm_cache(const std::vector<int> &selected,
                        const Eigen::Ref<const Eigen::MatrixXd> &A)
{
    warm_cache_impl(selected, A);
}

void MaxVol::warm_cache(const std::vector<int> &selected,
                        const Eigen::Ref<const RowMatrixXd> &A)
{
    warm_cache_impl(selected, A);
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API — scoring
// ═══════════════════════════════════════════════════════════════════════════

template <typename MatrixType>
Eigen::VectorXd MaxVol::compute_scores_impl(
    const MatrixType &A, const std::vector<int> &selected)
{
    const int m = static_cast<int>(A.rows());
    if (!valid_selection(A, selected) || !A.allFinite())
        return {};
    ensure_cache(selected, A);
    Eigen::VectorXd scores = score_rows_impl(A, selected);
    if (scores.size() != m) return {};
    for (int idx : selected) scores[idx] = 1.0;
    return scores;
}

Eigen::VectorXd MaxVol::compute_scores(
    const Eigen::Ref<const Eigen::MatrixXd> &A,
    const std::vector<int> &selected)
{
    return compute_scores_impl(A, selected);
}

Eigen::VectorXd MaxVol::compute_scores(
    const Eigen::Ref<const RowMatrixXd> &A,
    const std::vector<int> &selected)
{
    return compute_scores_impl(A, selected);
}

template <typename MatrixType>
Eigen::VectorXd MaxVol::score_rows_impl(
    const MatrixType &B_candidate, const std::vector<int> &selected)
{
    const int nrows = static_cast<int>(B_candidate.rows());
    const int k = static_cast<int>(selected.size());
    const int n = static_cast<int>(B_candidate.cols());

    // ── Cached pseudo-inverse (shared with compute_scores) ────────────
    if (selected != cached_selected_ || n != cached_ncols_ ||
        cached_pinv_.cols() != k || !B_candidate.allFinite()) {
        return Eigen::VectorXd();
    }

    const Eigen::MatrixXd coeff = B_candidate * cached_pinv_;
    Eigen::MatrixXd z;
    if (cached_wide_)
        z.noalias() = B_candidate * cached_active_.transpose();

    Eigen::VectorXd scores(nrows);
    for (int l = 0; l < nrows; ++l) {
        double best = 0.0;
        if (cached_wide_) {
            const double residual2 = std::max(
                0.0, B_candidate.row(l).squaredNorm()
                   - coeff.row(l).dot(z.row(l)));
            for (int i = 0; i < k; ++i) {
                const double gain2 = coeff(l, i) * coeff(l, i)
                                   + residual2 * cached_gram_inv_diag_[i];
                best = std::max(best, std::sqrt(std::max(0.0, gain2)));
            }
        } else if (k > n && cached_active_leverage_.size() == k) {
            const double candidate_leverage = coeff.row(l).squaredNorm();
            for (int i = 0; i < k; ++i) {
                const double gain2 =
                    (1.0 - cached_active_leverage_[i])
                    * (1.0 + candidate_leverage)
                    + coeff(l, i) * coeff(l, i);
                best = std::max(best, std::sqrt(std::max(0.0, gain2)));
            }
        } else {
            for (int i = 0; i < k; ++i)
                best = std::max(best, std::abs(coeff(l, i)));
        }
        scores[l] = best;
    }
    return scores;
}

Eigen::VectorXd MaxVol::score_rows(
    const Eigen::Ref<const Eigen::MatrixXd> &B_candidate,
    const std::vector<int> &selected)
{
    return score_rows_impl(B_candidate, selected);
}

Eigen::VectorXd MaxVol::score_rows(
    const Eigen::Ref<const RowMatrixXd> &B_candidate,
    const std::vector<int> &selected)
{
    return score_rows_impl(B_candidate, selected);
}

template <typename MatrixType>
void MaxVol::compute_subset_scores_impl(
    const MatrixType &A,
    const std::vector<int> &selected,
    const std::vector<int> &row_ranges,
    std::vector<double> &out_grades)
{
    const int m = static_cast<int>(A.rows());
    const int nr = static_cast<int>(row_ranges.size());

    if (nr < 2) { out_grades.clear(); return; }
    const Eigen::VectorXd scores = compute_scores_impl(A, selected);
    if (scores.size() != m) { out_grades.clear(); return; }

    out_grades.resize(nr - 1);
    for (int r = 0; r < nr - 1; ++r) {
        int start = row_ranges[r];
        int end   = row_ranges[r + 1];
        start = std::max(0, std::min(start, m));
        end = std::max(start, std::min(end, m));
        double best = 0.0;
        for (int l = start; l < end; ++l)
            best = std::max(best, scores[l]);
        out_grades[r] = best;
    }
}

void MaxVol::compute_subset_scores(
    const Eigen::Ref<const Eigen::MatrixXd> &A,
    const std::vector<int> &selected,
    const std::vector<int> &row_ranges,
    std::vector<double> &out_grades)
{
    compute_subset_scores_impl(A, selected, row_ranges, out_grades);
}

void MaxVol::compute_subset_scores(
    const Eigen::Ref<const RowMatrixXd> &A,
    const std::vector<int> &selected,
    const std::vector<int> &row_ranges,
    std::vector<double> &out_grades)
{
    compute_subset_scores_impl(A, selected, row_ranges, out_grades);
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API — incremental
// ═══════════════════════════════════════════════════════════════════════════

template <typename MatrixType>
std::vector<int> MaxVol::add_rows_impl(
    const MatrixType &A, const std::vector<int> &current, int add_k)
{
    if (add_k <= 0) return current;
    if (!valid_selection(A, current) || !A.allFinite()) return current;
    return augment_rows_d_optimal(A, current, add_k);
}

std::vector<int> MaxVol::add_rows(
    const Eigen::Ref<const Eigen::MatrixXd> &A,
    const std::vector<int> &current, int add_k)
{
    return add_rows_impl(A, current, add_k);
}

std::vector<int> MaxVol::add_rows(
    const Eigen::Ref<const RowMatrixXd> &A,
    const std::vector<int> &current, int add_k)
{
    return add_rows_impl(A, current, add_k);
}

// ═══════════════════════════════════════════════════════════════════════════
// Serialization
// ═══════════════════════════════════════════════════════════════════════════

void MaxVol::save(const std::string &filename) const
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "MaxVol::save: cannot open " << filename << "\n";
        return;
    }
    save(ofs);
}

void MaxVol::save(std::ostream &os) const
{
    const int magic = 0x4D56534C;
    const int version = 3;
    os.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    os.write(reinterpret_cast<const char *>(&version), sizeof(version));
    os.write(reinterpret_cast<const char *>(&cold_start_k_), sizeof(cold_start_k_));
    os.write(reinterpret_cast<const char *>(&log_det_), sizeof(log_det_));
    const int nselected = static_cast<int>(cached_selected_.size());
    os.write(reinterpret_cast<const char *>(&nselected), sizeof(nselected));
    for (int idx : cached_selected_)
        os.write(reinterpret_cast<const char *>(&idx), sizeof(idx));
}

std::vector<int> MaxVol::load(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "MaxVol::load: cannot open " << filename << "\n";
        return {};
    }
    return load(ifs);
}

std::vector<int> MaxVol::load(std::istream &is)
{
    auto clear_runtime_cache = [this]() {
        swap_tracker_.clear();
        cached_selected_.clear();
        cached_ncols_ = -1;
        cached_signature_ = 0;
        cached_pinv_.resize(0, 0);
        cached_active_.resize(0, 0);
        cached_gram_inv_diag_.resize(0);
        cached_active_leverage_.resize(0);
        cached_wide_ = false;
    };
    auto clear_all_state = [this, &clear_runtime_cache]() {
        clear_runtime_cache();
        cold_start_k_ = 0;
        log_det_ = 0.0;
    };

    int magic = 0, version = 0;
    is.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (!is || magic != 0x4D56534C || version != 3) {
        std::cerr << "MaxVol::load: bad magic or version\n";
        clear_all_state();
        return {};
    }
    int loaded_k = 0;
    double loaded_log_det = 0.0;
    is.read(reinterpret_cast<char *>(&loaded_k), sizeof(loaded_k));
    is.read(reinterpret_cast<char *>(&loaded_log_det), sizeof(loaded_log_det));
    if (!is || loaded_k < 0 || !std::isfinite(loaded_log_det)) {
        clear_all_state();
        return {};
    }

    std::vector<int> loaded_selected;
    {
        int nselected = 0;
        is.read(reinterpret_cast<char *>(&nselected), sizeof(nselected));
        if (!is || nselected < 0 || nselected > 100000000) {
            clear_all_state();
            return {};
        }
        loaded_selected.resize(static_cast<size_t>(nselected));
        std::set<int> unique;
        for (int &idx : loaded_selected) {
            is.read(reinterpret_cast<char *>(&idx), sizeof(idx));
            if (!is || idx < 0 || !unique.insert(idx).second) {
                clear_all_state();
                return {};
            }
        }
    }
    cold_start_k_ = loaded_k;
    log_det_ = loaded_log_det;
    clear_runtime_cache();
    cached_selected_ = loaded_selected;
    return loaded_selected;
}
