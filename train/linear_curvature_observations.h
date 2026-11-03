// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "linear_efs_observations.h"

namespace Training {

struct LinearCurvatureStatistics {
    double gpu_initialize_seconds = 0, geometry_seconds = 0;
    double observation_seconds = 0, accumulation_seconds = 0;
    std::size_t tracked_device_bytes = 0, host_workspace_bytes = 0;
    std::size_t output_workspace_bytes = 0, probes = 0, blocks = 0, download_bytes = 0;
};

struct LinearCurvatureObservationBlock {
    std::size_t configuration;
    bool hessian;
    int probe, first_row, rows;
    std::span<const int> native_columns;
    std::span<const double> values; // Unweighted row-major rows x columns.
};

// Analytic, fixed-radial native linear columns, independent of the fitted
// species/scalar coefficients. requested[k] selects {Hessian,HVP}. Hessian
// probe j is Cartesian e_j and emits rows 0..j (upper triangle, no factor 2).
// HVP probe k uses the original Cartesian stored direction without target
// normalization. Probes and row blocks stream; no full Hessian/Jacobian cache.
// Column order and duplicates are retained. Each complete selected row must
// fit the workspace; callers can use the adapter for separate column blocks.
// Spans are borrowed only during callbacks; input/model mutation is forbidden.
// A callback or later block failure can follow already consumed blocks; the
// consuming fit owns its transaction. This visitor has no MPI collectives.
// CUDA FP64 ordinary MTP only. CPU defaults and finite-step losses are unchanged.
// Host cap includes external_host_bytes, copied geometry, owned neighborhoods,
// packed geometry, selected columns and one direction/result block. Borrowed
// inputs/neighborhoods, model conversion, runtime/allocator and fixed call-stack
// bookkeeping are excluded. Device cap is memory+workspace; direction/output
// storage additionally fits workspace. Both are allocation accounts, not RSS.
void visit_gpu_linear_curvature(Potentials::PotMTP &pot,
    const std::vector<Atoms> &atoms,
    const std::vector<std::array<bool, 2>> &requested,
    std::span<const int> native_columns,
    const LinearEFSOptions &options, LinearCurvatureStatistics &stats,
    const std::function<void(const LinearCurvatureObservationBlock &)> &consume,
    const std::vector<Neighborhoods> *neighborhoods = nullptr,
    std::size_t host_cap = std::size_t{128} << 20,
    std::size_t external_host_bytes = 0);

} // namespace Training
