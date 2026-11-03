// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_force_fd.h"

namespace Potentials { class PotMTP; }

namespace HessianGpu {

// Allocation-free public size guard, also usable without constructing Atoms.
std::size_t checked_analytic_hvp_dof(std::size_t atoms);

// Native second-derivative radial contract, shared by analysis and linear
// observation generation. Does not impose an analysis MPI execution policy.
void validate_analytic_radial_bases(const Potentials::PotMTP &);

// Geometry-only ownership, including the inline Atoms object. copied=true
// counts a fresh size-based copy; false counts retained vector capacities.
std::size_t analytic_geometry_bytes(const Atoms &, bool copied = false);

// Checks shape, finite direction, execution/model/layout support, and the
// minimum simultaneous host storage before any snapshot or probe allocation.
// Returns 3*N after size_t/index checks. External storage is counted, not owned.
std::size_t analytic_hvp_preflight(const Atoms &, const double *direction,
    std::size_t direction_count, const Potentials::AnyPotential &, double cutoff,
    const HessianExecutionSettings &, std::size_t external_host_bytes = 0);

// The source must already have passed analytic_hvp_preflight. Checks the
// geometry allocation before copying, keeps no EFS/curvature/metadata labels,
// and converts stored fractional positions to Cartesian without changing input.
Atoms analytic_geometry_copy(const Atoms &, const HessianExecutionSettings &,
                             std::size_t reserved_host_bytes = 0);

// Independent staged result; caller input/direction/old output are borrowed.
// Host cap includes owned geometry, neighborhood construction, packed batch,
// adapter direction/result staging and final Eigen result, including overlap.
// Model conversion/runtime, fixed call-stack/error-message bookkeeping,
// caller-owned data and NumPy forcecast are excluded.
Eigen::VectorXd analytic_hvp(const Atoms &, const Eigen::VectorXd &direction,
    const Potentials::AnyPotential &, double cutoff,
    const HessianExecutionSettings &, HessianExecutionReport &,
    std::size_t external_host_bytes = 0);

} // namespace HessianGpu
