// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potentials/potential_mtp.h"
#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace Training
{

// Backend selection applies to observation generation. Accumulation, QR and
// solves retain the caller's existing CPU numerical algorithm.
struct LinearEFSOptions
{
    std::string device = "cpu";
    int gpu_id = -1;
    int gpu_batch_atoms = 4096, gpu_batch_structures = 32;
    std::size_t gpu_memory_bytes = std::size_t{256} << 20;
    std::size_t gpu_workspace_bytes = std::size_t{64} << 20;
    void validate() const;
};

struct LinearEFSStatistics
{
    double gpu_initialize_seconds = 0, neighborhoods_seconds = 0;
    double gpu_forward_seconds = 0, gpu_observation_seconds = 0;
    double accumulation_seconds = 0;
    std::size_t tracked_device_bytes = 0, observation_workspace_bytes = 0;
    std::size_t batches = 0, observation_blocks = 0, download_bytes = 0;
};

struct LinearEFSObservationRow
{
    std::size_t configuration; // Original index in the input vector.
    int channel;               // 0: total energy, 1: force, 2: raw full virial.
    int atom = -1;             // Structure-local atom for force; otherwise -1.
    int component = -1;        // Force axis 0..2, virial 3*i+j in 0..8, energy -1.
};

struct LinearEFSObservationBlock
{
    std::span<const LinearEFSObservationRow> rows;
    std::span<const int> native_columns;
    std::span<const double> values; // Unweighted row-major rows x native_columns.
};

// Local GPU-only visitor: no collectives, weighting, residual anchoring or
// numerical factorization. Requests depend only on masks, not stored labels.
// Selected native columns retain order and duplicates. Empty selections or
// all-false masks produce no chunks after option/model/shape/column validation.
// begin_chunk (optional) receives original configuration IDs once per physical
// batch, before its observation callbacks; consume is required when work exists.
// Every span is borrowed for its callback only. Callbacks must not change model
// parameters, geometry or supplied neighborhoods. Existing neighborhoods are
// reused without modification; otherwise each bounded chunk builds its own.
// Statistics accumulate across visits; callback time is accumulation_seconds.
void visit_gpu_linear_efs(Potentials::PotMTP &pot, const std::vector<Atoms> &atoms,
                          const std::vector<std::array<bool, 3>> &requested,
                          std::span<const int> native_columns, const LinearEFSOptions &options,
                          LinearEFSStatistics &stats,
                          const std::function<void(std::span<const std::size_t>)> &begin_chunk,
                          const std::function<void(const LinearEFSObservationBlock &)> &consume,
                          const std::vector<Neighborhoods> *neighborhoods = nullptr);

} // namespace Training
