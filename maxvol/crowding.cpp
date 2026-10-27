// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "maxvol/crowding.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

using RowMatrix = CrowdingModel::RowMatrix;

void validate_params(const CrowdingParams &params)
{
    if (params.kernel != "cosine" && params.kernel != "gaussian")
        throw std::invalid_argument(
            "CrowdingWeights: kernel must be 'cosine' or 'gaussian'");
    if (!std::isfinite(params.sigma) || !std::isfinite(params.alpha) ||
        !std::isfinite(params.clip_min) || !std::isfinite(params.clip_max))
        throw std::invalid_argument("CrowdingWeights: parameters must be finite");
    if (params.alpha < 0.0)
        throw std::invalid_argument("CrowdingWeights: alpha must be non-negative");
    if (params.clip_min <= 0.0 || params.clip_max < params.clip_min)
        throw std::invalid_argument(
            "CrowdingWeights: require 0 < clip_min <= clip_max");
    if (params.max_density_points == 0 || params.max_density_points < -1)
        throw std::invalid_argument(
            "CrowdingWeights: sample must be positive or -1 for exact mode");
    if (params.batch_rows <= 0)
        throw std::invalid_argument("CrowdingWeights: batch_rows must be positive");
}

double positive_median(std::vector<double> values, double fallback)
{
    values.erase(std::remove_if(values.begin(), values.end(), [](double x) {
                     return !(x > 0.0) || !std::isfinite(x);
                 }), values.end());
    if (values.empty()) return fallback;
    auto mid = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

std::uint64_t bucket_seed(int seed, int bucket)
{
    std::uint64_t x = static_cast<std::uint32_t>(seed);
    x ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(bucket))
         + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    return x;
}

} // namespace

struct CrowdingModel::Impl
{
    struct Bucket
    {
        std::uint64_t count = 0;
        Eigen::VectorXd mean;
        Eigen::VectorXd m2;
        Eigen::VectorXd inv_scale;
        RowMatrix references;
        std::vector<std::uint64_t> reference_ids;
        std::unordered_map<std::uint64_t, int> reference_positions;
        std::vector<Eigen::VectorXd> sampled_rows;
        std::mt19937_64 rng;
        double sigma = 1.0;

        Bucket(int cols, std::uint64_t seed)
            : mean(Eigen::VectorXd::Zero(cols)),
              m2(Eigen::VectorXd::Zero(cols)),
              inv_scale(Eigen::VectorXd::Ones(cols)), rng(seed)
        {}
    };

    int cols;
    CrowdingParams params;
    bool is_finalized = false;
    std::unordered_map<int, Bucket> buckets;

    Impl(int descriptor_size, const CrowdingParams &p)
        : cols(descriptor_size), params(p)
    {
        validate_params(params);
        if (cols <= 0)
            throw std::invalid_argument(
                "CrowdingModel: descriptor size must be positive");
    }

    Bucket &bucket(int id)
    {
        auto it = buckets.find(id);
        if (it == buckets.end()) {
            it = buckets.emplace(
                std::piecewise_construct, std::forward_as_tuple(id),
                std::forward_as_tuple(cols, bucket_seed(params.seed, id))).first;
        }
        return it->second;
    }

    Eigen::RowVectorXd transform(const Eigen::Ref<const Eigen::RowVectorXd> row,
                                 const Bucket &b) const
    {
        Eigen::RowVectorXd out = row;
        if (params.standardize) {
            if (params.kernel == "gaussian")
                out -= b.mean.transpose();
            out.array() *= b.inv_scale.transpose().array();
        }
        if (params.kernel == "cosine") {
            const double norm = out.norm();
            if (norm > 1e-30) out /= norm;
            else out.setZero();
        }
        return out;
    }

    double infer_sigma(const RowMatrix &refs, int bucket_id) const
    {
        if (params.sigma > 0.0) return params.sigma;
        const int n = static_cast<int>(refs.rows());
        if (n < 2) return 1.0;
        constexpr std::size_t max_pairs = 100000;
        const std::uint64_t total = static_cast<std::uint64_t>(n) * (n - 1) / 2;
        std::vector<double> d2;
        d2.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(total, max_pairs)));
        if (total <= max_pairs) {
            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j)
                    d2.push_back((refs.row(i) - refs.row(j)).squaredNorm());
        } else {
            std::mt19937_64 rng(bucket_seed(params.seed + 1, bucket_id));
            std::uniform_int_distribution<int> pick(0, n - 1);
            while (d2.size() < max_pairs) {
                int i = pick(rng), j = pick(rng);
                if (i == j) continue;
                if (i > j) std::swap(i, j);
                d2.push_back((refs.row(i) - refs.row(j)).squaredNorm());
            }
        }
        return std::sqrt(std::max(positive_median(std::move(d2), 1.0), 1e-12));
    }
};

CrowdingModel::CrowdingModel(int descriptor_size, const CrowdingParams &params)
    : impl_(std::make_unique<Impl>(descriptor_size, params))
{}

CrowdingModel::~CrowdingModel() = default;
CrowdingModel::CrowdingModel(CrowdingModel &&) noexcept = default;
CrowdingModel &CrowdingModel::operator=(CrowdingModel &&) noexcept = default;

void CrowdingModel::observe(const RowMatrix &B, const std::vector<int> &buckets,
                            const std::vector<std::uint64_t> &row_ids)
{
    if (impl_->is_finalized)
        throw std::logic_error("CrowdingModel: observe called after finalize");
    if (B.cols() != impl_->cols || buckets.size() != static_cast<std::size_t>(B.rows()) ||
        row_ids.size() != static_cast<std::size_t>(B.rows()))
        throw std::invalid_argument("CrowdingModel: observe batch size mismatch");

    for (int i = 0; i < B.rows(); ++i) {
        if (buckets[i] < 0) continue;
        if (!B.row(i).allFinite())
            throw std::invalid_argument("CrowdingModel: descriptor contains NaN or Inf");
        auto &b = impl_->bucket(buckets[i]);
        ++b.count;
        const Eigen::VectorXd row = B.row(i).transpose();
        const Eigen::VectorXd delta = row - b.mean;
        b.mean += delta / static_cast<double>(b.count);
        b.m2.array() += delta.array() * (row - b.mean).array();

        const int cap = impl_->params.max_density_points;
        if (cap == -1 || static_cast<int>(b.sampled_rows.size()) < cap) {
            b.sampled_rows.push_back(row);
            b.reference_ids.push_back(row_ids[i]);
        } else {
            std::uniform_int_distribution<std::uint64_t> pick(0, b.count - 1);
            const std::uint64_t slot = pick(b.rng);
            if (slot < static_cast<std::uint64_t>(cap)) {
                b.sampled_rows[slot] = row;
                b.reference_ids[slot] = row_ids[i];
            }
        }
    }
}

void CrowdingModel::finalize()
{
    if (impl_->is_finalized) return;
    for (auto &[id, b] : impl_->buckets) {
        if (b.count > 0) {
            const Eigen::ArrayXd variance = b.m2.array() / static_cast<double>(b.count);
            for (int j = 0; j < impl_->cols; ++j) {
                const double sd = std::sqrt(std::max(variance[j], 0.0));
                if (sd >= 1e-12) b.inv_scale[j] = 1.0 / sd;
                else b.inv_scale[j] = impl_->params.kernel == "gaussian" ? 0.0 : 1.0;
            }
        }

        b.references.resize(static_cast<int>(b.sampled_rows.size()), impl_->cols);
        for (int r = 0; r < b.references.rows(); ++r)
            b.references.row(r) = impl_->transform(b.sampled_rows[r].transpose(), b);
        b.sampled_rows.clear();
        b.sampled_rows.shrink_to_fit();
        for (int r = 0; r < static_cast<int>(b.reference_ids.size()); ++r)
            b.reference_positions.emplace(b.reference_ids[r], r);
        if (impl_->params.kernel == "gaussian")
            b.sigma = impl_->infer_sigma(b.references, id);
    }
    impl_->is_finalized = true;
}

bool CrowdingModel::finalized() const { return impl_->is_finalized; }

std::vector<double> CrowdingModel::density(
    const RowMatrix &B, const std::vector<int> &buckets,
    const std::vector<std::uint64_t> &row_ids) const
{
    if (!impl_->is_finalized)
        throw std::logic_error("CrowdingModel: density called before finalize");
    if (B.cols() != impl_->cols || buckets.size() != static_cast<std::size_t>(B.rows()) ||
        row_ids.size() != static_cast<std::size_t>(B.rows()))
        throw std::invalid_argument("CrowdingModel: density batch size mismatch");

    std::vector<double> result(static_cast<std::size_t>(B.rows()), 0.0);
    std::unordered_map<int, std::vector<int>> query_rows;
    for (int i = 0; i < B.rows(); ++i) {
        if (buckets[i] >= 0) query_rows[buckets[i]].push_back(i);
    }

    for (const auto &[id, rows] : query_rows) {
        const auto bit = impl_->buckets.find(id);
        if (bit == impl_->buckets.end() || bit->second.references.rows() == 0) continue;
        const auto &b = bit->second;
        RowMatrix Q(static_cast<int>(rows.size()), impl_->cols);
        std::vector<bool> qzero(rows.size(), false);
        for (int q = 0; q < static_cast<int>(rows.size()); ++q) {
            if (!B.row(rows[q]).allFinite())
                throw std::invalid_argument("CrowdingModel: descriptor contains NaN or Inf");
            Q.row(q) = impl_->transform(B.row(rows[q]), b);
            qzero[q] = Q.row(q).squaredNorm() < 1e-30;
        }

        Eigen::MatrixXd K;
        if (impl_->params.kernel == "cosine") {
            K.noalias() = Q * b.references.transpose();
            K = K.cwiseMax(0.0).cwiseMin(1.0);
            std::vector<bool> rzero(static_cast<std::size_t>(b.references.rows()), false);
            for (int r = 0; r < b.references.rows(); ++r)
                rzero[r] = b.references.row(r).squaredNorm() < 1e-30;
            for (int q = 0; q < K.rows(); ++q)
                if (qzero[q])
                    for (int r = 0; r < K.cols(); ++r)
                        K(q, r) = rzero[r] ? 1.0 : 0.0;
        } else {
            K.noalias() = Q * b.references.transpose();
            const Eigen::VectorXd qn = Q.rowwise().squaredNorm();
            const Eigen::VectorXd rn = b.references.rowwise().squaredNorm();
            K = ((-2.0 * K).colwise() + qn).rowwise() + rn.transpose();
            K = (-K.cwiseMax(0.0).array() / (2.0 * b.sigma * b.sigma)).exp().matrix();
        }

        for (int q = 0; q < K.rows(); ++q) {
            double sum = K.row(q).sum();
            int contributors = K.cols();
            const auto self = b.reference_positions.find(row_ids[rows[q]]);
            if (self != b.reference_positions.end()) {
                sum -= K(q, self->second);
                --contributors;
            }
            result[rows[q]] = contributors > 0 ? sum / contributors : 0.0;
        }
    }
    return result;
}

std::vector<double> CrowdingWeights::compute(const RowMatrix &B,
                                             const std::vector<int> &buckets,
                                             const CrowdingParams &params)
{
    const int n = static_cast<int>(B.rows());
    if (!buckets.empty() && buckets.size() != static_cast<size_t>(n))
        throw std::invalid_argument(
            "CrowdingWeights: buckets size must equal number of rows");

    validate_params(params);
    if (n == 0) return {};
    std::vector<int> ids = buckets;
    if (ids.empty()) ids.assign(static_cast<std::size_t>(n), 0);
    std::vector<std::uint64_t> row_ids(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) row_ids[i] = static_cast<std::uint64_t>(i);
    CrowdingModel model(static_cast<int>(B.cols()), params);
    model.observe(B, ids, row_ids);
    model.finalize();
    return weights_from_density(model.density(B, ids, row_ids), ids, params);
}

std::vector<double> CrowdingWeights::compute(const RowMatrix &B,
                                             const CrowdingParams &params)
{
    std::vector<int> buckets;
    return compute(B, buckets, params);
}

std::vector<double> CrowdingWeights::weights_from_density(
    const std::vector<double> &density, const std::vector<int> &buckets,
    const CrowdingParams &params)
{
    validate_params(params);
    if (!buckets.empty() && buckets.size() != density.size())
        throw std::invalid_argument(
            "CrowdingWeights: buckets size must equal density size");
    std::vector<double> weights(density.size(), 1.0);
    if (density.empty() || params.alpha == 0.0) return weights;
    std::vector<int> ids = buckets;
    if (ids.empty()) ids.assign(density.size(), 0);

    std::unordered_map<int, std::vector<std::size_t>> rows_by_bucket;
    for (std::size_t i = 0; i < ids.size(); ++i)
        if (ids[i] >= 0) rows_by_bucket[ids[i]].push_back(i);
    for (const auto &[id, rows] : rows_by_bucket) {
        (void)id;
        std::vector<double> positive;
        positive.reserve(rows.size());
        for (std::size_t row : rows) {
            if (!std::isfinite(density[row]) || density[row] < 0.0)
                throw std::invalid_argument(
                    "CrowdingWeights: density must be finite and non-negative");
            if (density[row] > 0.0) positive.push_back(density[row]);
        }
        const double rho_med = positive_median(std::move(positive), 0.0);
        if (rho_med <= 0.0) continue;
        for (std::size_t row : rows) {
            const double w = density[row] > 0.0
                ? std::pow(density[row] / rho_med, -params.alpha)
                : params.clip_max;
            weights[row] = std::clamp(w, params.clip_min, params.clip_max);
        }
    }
    return weights;
}
