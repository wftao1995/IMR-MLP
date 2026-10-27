// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/// @brief Parameters for descriptor-space crowding (density) weighting.
///
/// Crowding weighting assigns a per-row weight to atomic environments
/// (descriptor vectors) that is large for rows sitting in sparse regions of
/// descriptor space and small for rows in crowded regions.  The weights are
/// meant to be stored in `Atoms::atomic_fitness` so that the existing training
/// machinery applies them as per-atom force weights.
struct CrowdingParams
{
    /// Kernel type: "cosine" (normalized dot product) or "gaussian".
    std::string kernel = "cosine";
    /// Gaussian bandwidth.  If <= 0, inferred from the median pairwise
    /// distance in each bucket (median heuristic).
    double sigma = -1.0;
    /// Scale each descriptor component by its bucket standard deviation.
    /// Gaussian descriptors are also mean-centered (z-score); cosine
    /// descriptors deliberately are not centered before row normalization.
    bool standardize = true;
    /// Weight exponent: w_i = (rho_i / rho_median)^(-alpha).  alpha = 0
    /// reproduces the unweighted case (all weights 1).
    double alpha = 0.5;
    /// Clip final weights to [clip_min, clip_max].
    double clip_min = 1e-3;
    double clip_max = 1e3;
    /// Cap the number of reference rows per bucket used for the density
    /// estimate.  Positive values use deterministic reservoir sampling;
    /// -1 keeps every row (exact mode, suitable only for small data sets).
    int max_density_points = 256;
    /// Seed for deterministic reservoir and bandwidth sampling.
    int seed = 0;
    /// Maximum number of query descriptor rows evaluated in one CLI batch.
    int batch_rows = 2048;
};

/// @brief Incremental reference model for bounded-memory crowding estimates.
///
/// Rows are observed once to accumulate per-bucket preprocessing statistics
/// and a bounded reference set.  After finalize(), density() can evaluate
/// query batches without retaining the full descriptor matrix.
class CrowdingModel
{
public:
    using RowMatrix = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

    CrowdingModel(int descriptor_size, const CrowdingParams &params);
    ~CrowdingModel();
    CrowdingModel(CrowdingModel &&) noexcept;
    CrowdingModel &operator=(CrowdingModel &&) noexcept;
    CrowdingModel(const CrowdingModel &) = delete;
    CrowdingModel &operator=(const CrowdingModel &) = delete;

    /// Observe a sequential batch. row_ids must be globally stable across
    /// observe() and density() so sampled self-contributions can be excluded.
    void observe(const RowMatrix &B, const std::vector<int> &buckets,
                 const std::vector<std::uint64_t> &row_ids);
    void finalize();
    bool finalized() const;

    /// Return one density for each query row. Negative buckets return zero and
    /// are later assigned unit weight by weights_from_density().
    std::vector<double> density(
        const RowMatrix &B, const std::vector<int> &buckets,
        const std::vector<std::uint64_t> &row_ids) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief KDE-based crowding weights over descriptor-space rows.
///
/// Each row i gets density
///   rho_i = (1 / |R_i|) * sum_{j in R_i} K(B_i, B_j),
/// where R_i is the reference set excluding i itself.  Without subsampling,
/// |R_i| = n_b - 1.
/// and weight
///   w_i = (rho_i / median_bucket(rho))^(-alpha),
/// clipped to [clip_min, clip_max].  Buckets are treated independently
/// (e.g. one bucket per chemical element), so the median weight in every
/// bucket is 1 and scales are comparable across elements.
class CrowdingWeights
{
public:
    using RowMatrix = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

    /// @brief Compute crowding weights.
    /// @param B       Row-major descriptor matrix, one row per environment.
    /// @param buckets Per-row bucket id (e.g. chemical element index);
    ///                rows with the same id share a density pool.
    ///                Rows with id < 0 are skipped (weight 1).
    /// @param params  Crowding parameters.
    /// @return n weights in [clip_min, clip_max].
    static std::vector<double> compute(const RowMatrix &B,
                                       const std::vector<int> &buckets,
                                       const CrowdingParams &params);

    /// @brief Single-bucket overload (no bucketing).
    static std::vector<double> compute(const RowMatrix &B,
                                       const CrowdingParams &params);

    /// Convert precomputed densities to clipped, bucket-normalized weights.
    static std::vector<double> weights_from_density(
        const std::vector<double> &density,
        const std::vector<int> &buckets,
        const CrowdingParams &params);
};
