// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base/structures.h"
#include "potentials/potential_mtp.h"

#include <memory>

namespace Symmetry3D {

struct StructureComparisonOptions {
    double symmetry_tolerance = 1e-4;
    double position_tolerance = 0.1;
    double volume_tolerance = 1e-3;
    double descriptor_tolerance = 1e-4;
    double anchor_descriptor_tolerance = 1e-4;
    bool use_descriptors = true;
    bool scale_volume = false;
    // Include improper rotations (mirror images) by default.
    bool allow_reflections = true;
};

// Cell-aware equivalence: all three cell dimensions participate, including vacuum.
// Only fully periodic structures are reduced to primitive cells. Partial/nonperiodic
// structures retain their cells and atom counts; wrapping follows their PBC flags.
// Prepare once, then reuse the cell and descriptor data for batch comparisons.
struct ComparableStructure {
    Atoms atoms;
    std::map<int, int> composition;
    Eigen::MatrixXd descriptors;
    Eigen::VectorXd fingerprint;
    std::vector<Eigen::Matrix3d> lattice_operations;
};

class StructureComparator {
public:
    explicit StructureComparator(StructureComparisonOptions options = {});
    ComparableStructure prepare(Atoms atoms);
    // Does not modify either input; concurrent comparisons may share prepared data.
    bool equivalent(const ComparableStructure &first, const ComparableStructure &second) const;

private:
    StructureComparisonOptions options_;
    std::unique_ptr<Potentials::PotMTP> descriptor_potential_;
    void describe(ComparableStructure &structure) const;
};

} // namespace Symmetry3D
