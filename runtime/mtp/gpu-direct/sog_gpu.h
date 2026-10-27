// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include "gpu_cache.h"

namespace MLIPCore::GpuDirect {

// Host views. Gaussian arrays are channel-major. Grid multiplicities already
// encode the CPU half-sphere convention, including the singly counted zero.
struct SOGGpuModelView {
    int channels = 0;
    int terms = 0;
    const double *weights = nullptr;
    const double *log_bandwidths = nullptr;
    bool onsite = true;
};

struct SOGGpuGridView {
    int atoms = 0;
    int structures = 0;
    int waves = 0;
    const int *atom_offsets = nullptr;
    const int *wave_offsets = nullptr;
    const double *volumes = nullptr;
    const double *wave_vectors = nullptr; // xyz, wave-major
    const double *multiplicities = nullptr;
    const double *taper = nullptr;
    const double *taper_derivative = nullptr; // derivative w.r.t. k^2
    // Optional promise of unchanged grid contents. Zero always uploads.
    GpuCacheToken cache_token;
};

// One-device, default-stream reciprocal operator. All evaluation arguments
// are DEVICE pointers. Owns O(N*C + K*C) storage, never an N*K phase matrix.
// Prepare copies the host views; their lifetime ends when prepare returns.
// Additive E/F/virial outputs use the MTP core's atom-major six-tensor layout.
class SOGGpu {
public:
    SOGGpu();
    ~SOGGpu();
    SOGGpu(const SOGGpu &) = delete;
    SOGGpu &operator=(const SOGGpu &) = delete;
    bool prepare(const SOGGpuModelView &, const SOGGpuGridView &,
                 std::string *error = nullptr);
    double *latents();
    double *response();
    const double *phi() const;
    const double *charge_vjp() const; // fixed-q F/W VJP + H*response
    const double *kernel_gradient() const; // physical w followed by u
    bool forward(const double *positions, double *site_energy,
                 double *force, double *site_virial,
                 std::string *error = nullptr);
    bool backward(const double *positions, const double *energy_adjoint,
                  const double *force_adjoint, const double *virial_adjoint,
                  std::string *error = nullptr);
    std::size_t memory_bytes() const;
    std::uint64_t grid_upload_bytes() const;
    std::uint64_t grid_reuses() const;
private:
    struct State;
    State *state_;
};

} // namespace MLIPCore::GpuDirect
