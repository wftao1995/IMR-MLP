// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace Potentials {
// Counts describe the compact model, including shared dependencies only once.
struct BasisCostStats {
    int scalars = 0, basics = 0, moments = 0, products = 0, channels = 0, max_degree = 0;
    std::size_t radial_bytes = 0;
    bool operator==(const BasisCostStats &) const = default;

    // efs-ops-v1: arithmetic operations plus explicit zero/store operations.
    // Not elapsed time. Common radial-basis/neighborhood/force-merge work is
    // excluded; the final benchmark measures those costs as well.
    double efs_work(double active_neighbors, double stored_neighbors, int rb_size) const {
        return active_neighbors * (6. * channels * rb_size + 2. * channels + 8. * max_degree + 39. * basics)
             + stored_neighbors * 12. * basics // zero (3), contraction arithmetic/stores (9)
             + 12. * products                // forward arithmetic/store (4), reverse (8)
             + 2. * moments + 4. * scalars + 2. * channels + 2. * rb_size;
    }
    // Principal per-thread EFS scratch arrays, excluding atom-level force
    // buffers, allocator capacity, training buffers and radial-basis internals.
    double scratch_bytes(std::size_t neighbors, int rb_size, int threads) const {
        return sizeof(double) * double(threads) *
            (2. * moments + 3. * basics * neighbors + 3. * neighbors +
             2. * channels + 2. * rb_size + 4. * (max_degree + 2));
    }
};

struct BasisSubsetAnalysis {
    BasisCostStats stats;
    std::vector<bool> keep;
    std::vector<int> moment_map, channel_map, channels;
};
} // namespace Potentials
