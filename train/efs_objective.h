// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "efs_fit_weights.h"
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace Training {
// Shared objective contract. Solving sum-normal equations is equivalent to the
// declared mean objective only when its L2 diagonal is multiplied by N.
struct EFSObjective {
    std::string convention, reference, native_reduction, l2_scope;
    std::size_t configurations = 0;
    double input_l2 = 0, mean_l2 = 0, native_l2 = 0;

    static EFSObjective resolve(const std::string& convention, const std::string& reference,
                                double l2, std::size_t count, bool sum_equations) {
        if (convention != "legacy" && convention != "mean-moments")
            throw std::invalid_argument("objective-convention must be legacy or mean-moments");
        if (reference != "native" && reference != "sum" && reference != "mean")
            throw std::invalid_argument("l2-reference must be native, sum or mean");
        if (convention == "legacy" && reference != "native")
            throw std::invalid_argument("Explicit l2-reference requires objective-convention mean-moments");
        if (!count || !std::isfinite(l2) || l2 < 0)
            throw std::invalid_argument("EFS objective requires training configurations and finite nonnegative L2");
        EFSObjective out;
        out.convention = convention; out.configurations = count; out.input_l2 = l2;
        out.native_reduction = sum_equations ? "sum" : "mean";
        out.reference = reference == "native"
            ? (convention == "legacy" ? out.native_reduction : "mean") : reference;
        out.mean_l2 = out.reference == "sum" ? l2 / count : l2;
        // Preserve the exact legacy diagonal, including floating-point rounding.
        out.native_l2 = convention == "legacy" ? l2 : (sum_equations ? out.mean_l2 * count : out.mean_l2);
        if (!std::isfinite(out.mean_l2) || !std::isfinite(out.native_l2)
            || (l2 > 0 && (out.mean_l2 == 0 || out.native_l2 == 0)))
            throw std::invalid_argument("L2 conversion overflow or underflow");
        out.l2_scope = convention == "mean-moments" ? "moments"
            : (sum_equations ? "all-linear" : "potential-defined");
        return out;
    }

    bool canonical() const { return convention == "mean-moments"; }

    static void validate_weights(const EFSFitWeights& w) {
        for (double v:{w.energy,w.force,w.stress})
            if (!std::isfinite(v) || v<0) throw std::invalid_argument("Invalid EFS objective weight");
        for (double v:{w.energy_power,w.force_power,w.stress_power,w.norm_d,w.norm_f,w.norm_p})
            if (!std::isfinite(v)) throw std::invalid_argument("Invalid EFS objective normalization");
        if (w.norm_d<=0 || w.norm_f<0 || w.norm_p<=0)
            throw std::invalid_argument("Invalid EFS force normalization");
    }

    void write(std::ostream& out, const EFSFitWeights& w, const std::string& prefix = "") const {
        out << std::setprecision(17)
            << prefix << "objective_convention\t" << convention << '\n'
            << prefix << "training_configurations\t" << configurations << '\n'
            << prefix << "native_loss_reduction\t" << native_reduction << '\n'
            << prefix << "comparison_data_loss_reduction\tmean\n"
            << prefix << "legacy_console_data_loss_reduction\t" << native_reduction << '\n'
            << prefix << "configuration_denominator\tall-loaded-training-frames\n"
            << prefix << "l2_input\t" << input_l2 << '\n'
            << prefix << "l2_reference\t" << reference << '\n'
            << prefix << "l2_scope\t" << l2_scope << '\n'
            << prefix << "l2_mean_equivalent\t" << mean_l2 << '\n'
            << prefix << "l2_native\t" << native_l2 << '\n'
            << prefix << "weights_E_F_V\t" << w.energy << ',' << w.force << ',' << w.stress << '\n'
            << prefix << "powers_E_F_V\t" << w.energy_power << ',' << w.force_power << ',' << w.stress_power << '\n'
            << prefix << "force_norm_d_f_p\t" << w.norm_d << ',' << w.norm_f << ',' << w.norm_p << '\n'
            << prefix << "virial_components\t9\n"
            << prefix << "efs_weight_semantics\tsquared-residual weights; frame fitness on E; atomic fitness on F; reference-force normalization on E/F/V\n";
    }
};
} // namespace Training
