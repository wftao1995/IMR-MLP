// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "linear_refit.h"
#include <functional>

struct VarProEvaluation {
    double objective=0., linear_gradient_inf=0., radial_penalty=0., linear_penalty=0.;
    Eigen::VectorXd coefficients, gradient, radial_norms;
    std::vector<double> linear_gradient_history; // one entry per inner solve
    // before/delta/timings cover all solves; final diagnostics describe the last solve.
    LinearRefitResult linear;
};

// Fixed diagonal coordinate change r = S z. The physical potential, its
// regularizers and the profiled linear solve are unaffected.
class VarProRadialScaling {
public:
    explicit VarProRadialScaling(Eigen::VectorXd scales);
    Eigen::VectorXd physical(const Eigen::VectorXd& z) const;
    Eigen::VectorXd optimizer(const Eigen::VectorXd& radial) const;
    Eigen::VectorXd gradient(const Eigen::VectorXd& physical_gradient) const;
    const Eigen::VectorXd& scales() const { return scales_; }
private:
    void check(const Eigen::VectorXd& v) const;
    Eigen::VectorXd scales_;
};

// Explicit model-owned affine blocks. Every probe starts from the same inner coefficients
// (including composition-null directions), and restores the complete model
// even on success. The caller explicitly installs accepted coefficients.
class MTPProfiledObjective {
public:
    using Gradient = std::function<void(double&, Eigen::VectorXd&)>;
    MTPProfiledObjective(Potentials::AnyPotential& potential,
        std::vector<Atoms>& reference, std::vector<Neighborhoods>& neighborhoods,
        const EFSFitWeights& weights, double penalty_weight,
        LinearRefitSettings settings, Gradient gradient,
        double linear_gradient_tolerance=1e-7, int linear_refinement_steps=2);
    VarProEvaluation evaluate(const Eigen::VectorXd& radial);
    // Historical name retained: these are now all complementary outer coordinates.
    int radial_size() const { return radial_size_; }
    Eigen::VectorXd outer_coefficients(const Eigen::VectorXd& full) const;
    const std::vector<int>& outer_indices() const { return outer_indices_; }
    const std::vector<int>& inner_indices() const { return inner_indices_; }
private:
    Potentials::AnyPotential& potential_;
    std::vector<Atoms>& reference_;
    std::vector<Neighborhoods>& neighborhoods_;
    EFSFitWeights weights_;
    double penalty_weight_, linear_gradient_tolerance_;
    LinearRefitSettings settings_;
    Gradient gradient_;
    Eigen::VectorXd source_, inner_l2_;
    std::vector<int> outer_indices_, inner_indices_;
    int radial_size_, linear_refinement_steps_;
};
