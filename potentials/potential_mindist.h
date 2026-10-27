// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential.h"

namespace Potentials {

/// \brief Minimum-distance repulsive penalty potential.
///
/// Pushes apart atoms closer than \p cutoff:
///   E = factor * (r - cutoff)^power   for r < cutoff
///   E = 0                              for r >= cutoff
///
/// Used primarily as a penalty term (e.g. in NEB) with add_efs = true.
class PotMDIST : public AnyPotential {
    void initialize(GPSettings &settings);
public:
    double power = 2;
    double factor = 1;

    PotMDIST() = default;
    explicit PotMDIST(double r) { cutoff = r; cutoffsq = r * r; }
    explicit PotMDIST(GPSettings &settings);
    ~PotMDIST() override = default;

    void InitSettings();

    using AnyPotential::getEFS;
    void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void Load(std::istream &is) override;
    void Save(std::ostream &os) override;
};

} // namespace Potentials
