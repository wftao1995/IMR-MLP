// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base/structures.h"
#include "potentials/potential.h"

#include <cstddef>
#include <string>

// Execution options do not change the selected curvature method.
struct HessianExecutionSettings {
    std::string device = "cpu";
    int gpu_id = -1;
    int gpu_batch_atoms = 4096;
    int gpu_batch_structures = 32;
    int gpu_memory_mib = 256;
    // Force FD: newly owned curvature output/projection only. Analytic HVP:
    // additionally bounds owned geometry, neighborhoods and packed workspace.
    // Both exclude caller-owned inputs/old outputs, model conversion, runtime,
    // fixed call-stack/error bookkeeping and process RSS; Python's implicit
    // NumPy forcecast is also excluded.
    int host_memory_mib = 1024;
};

struct HessianExecutionReport {
    std::string backend = "cpu", method;
    bool complete = false;
    std::size_t evaluated_structures = 0, batches = 0;
    std::size_t evaluator_initializations = 0;
    std::size_t geometry_uploads = 0, geometry_reuses = 0, download_bytes = 0;
    std::size_t tracked_device_bytes = 0, host_hessian_bytes = 0;
    // Analytic HVP accounted host high-water, including conservative neighbor
    // construction reservations. This is neither process RSS nor allocator use.
    std::size_t host_workspace_bytes = 0, analytic_hvp_calls = 0;
    std::size_t direction_upload_bytes = 0;
    std::size_t peak_batch_atoms = 0, peak_batch_structures = 0, peak_batch_neighbors = 0;
    double maximum_atom_displacement = 0.0;
    double total_seconds = 0.0, setup_seconds = 0.0;
    double neighborhoods_seconds = 0.0, packing_seconds = 0.0;
    double evaluation_seconds = 0.0, reduction_seconds = 0.0, symmetry_seconds = 0.0;
};

namespace HessianGpu {

// Reset the report and validate execution settings, including CPU-only builds.
bool begin(const HessianExecutionSettings &, HessianExecutionReport &);
void validate(const Atoms &, const Potentials::AnyPotential &,
              double cutoff, const HessianExecutionSettings &);
// Copies only geometry required by ordinary MTP, never EFS/curvature labels.
Atoms geometry_copy(const Atoms &);
std::size_t check_hessian_storage(std::size_t atoms, bool blocks,
                                 bool projection, const HessianExecutionSettings &,
                                 HessianExecutionReport &);
std::size_t check_vector_storage(std::size_t atoms, const HessianExecutionSettings &,
                                HessianExecutionReport &);
void symmetrize(Eigen::MatrixXd &);
Eigen::MatrixXd dense(const Atoms &, const Potentials::AnyPotential &,
                      double step, double cutoff, bool projection,
                      const HessianExecutionSettings &, HessianExecutionReport &);
Potentials::HessianMatrix blocks(const Atoms &, const Potentials::AnyPotential &,
                                double step, double cutoff,
                                const HessianExecutionSettings &, HessianExecutionReport &);
Eigen::VectorXd hvp(const Atoms &, const Eigen::VectorXd &direction,
                    const Potentials::AnyPotential &, double delta, double cutoff,
                    const HessianExecutionSettings &, HessianExecutionReport &);

} // namespace HessianGpu
