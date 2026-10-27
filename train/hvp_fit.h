// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potentials/potential.h"
#include <functional>

namespace Training {
struct HVPFitSettings {
    double weight = 0.0;
    double power = 0.0;
    double step = 1e-4; // maximum per-atom displacement in Angstrom
    void validate() const;
    double observation_weight(const Atoms &atom, double normalization = 1.0) const;
};

struct HVPFitResult {
    double loss = 0.0, squared_error = 0.0;
    int observations = 0;
};

using ForcePullback = std::function<void(Atoms &, Neighborhoods &,
    Eigen::Matrix<double, -1, 3> &)>;

// The loss and pullback both use the same central force difference. The
// callback is invoked immediately after getEFS so stateful models can reuse
// their prediction cache. No coordinate Hessian or parameter Jacobian is built.
HVPFitResult finite_difference_hvp_loss(Potentials::AnyPotential &pot,
    const Atoms &reference, const HVPFitSettings &settings,
    double normalization = 1.0, const ForcePullback &pullback = {});
using ForceObservation = std::function<Eigen::MatrixXd(Atoms &, Neighborhoods &)>;
// Apply the same central-force difference to one or many force columns.
// Rows are probe-major, then atom/Cartesian component; no weights or probe
// averaging are applied. This also transforms exact affine force basis columns.
Eigen::MatrixXd finite_difference_hvp_observations(const Atoms &reference,
    double step, double cutoff, const ForceObservation &forces);

// VJP of the unweighted central-force HVP predictions. cotangents has shape
// (3N, number of directions), with caller-supplied residual weights already
// applied. Each pullback runs immediately after the matching force prediction.
void finite_difference_hvp_vjp(Potentials::AnyPotential &pot,
    const Atoms &reference, double step, const Eigen::MatrixXd &cotangents,
    const ForcePullback &pullback);
} // namespace Training
