// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "descriptors.h"
#include "descriptor_Behler.h"
#include "descriptor_BondOrder.h"
#include "descriptor_BispectrumSO3.h"
#include "descriptor_BispectrumSO4.h"
#include "descriptor_MTM.h"
#include "utils/gputils.h"

#include "potentials/potential_mtp.h"
#include <filesystem>

/// \brief Factory + wrapper for any Descriptors subclass.
///
/// Modeled on Potentials::potential_wrapper — a string-based dispatch
/// (the \c dtype setting) selects the concrete descriptor and manages its
/// lifetime via unique_ptr.
///
/// Usage (settings-only):
/// \code
///   GPSettings settings;
///   settings.Modify("dtype", "behler");
///   settings.Modify("cutoff", "5.0");
///   settings.Modify("dispcount", "4");
///   settings.Modify("dispselect", "1,2,3,4");
///   descriptor_wrapper dw(settings);
///   int n = dw.feature_size();
///   dw.compute(nbh, out);
///   dw.compute(nbh, out, opts_with_gradient);
/// \endcode
///
/// For the MTM descriptor an external or owned PotMTP is required:
/// \code
///   descriptor_wrapper dw(settings, std::move(mtp));
/// \endcode
class descriptor_wrapper : public ConnectSettings
{
public:
    std::unique_ptr<Descriptors> desc;
    std::string dtype;

    // ── Settings-bound parameters (used only for non-ConnectSettings types) ──
    int    biso3_lmax  = 4;
    double biso3_rmin  = 0.0;
    double biso3_rmax  = 5.0;
    int    biso3_rsize = 8;
    bool   biso3_real_harmonics = false;

    void InitSettings()
    {
        Connect(dtype,       "dtype",       "descriptor type (behler, bondorder, biso3, biso4, mtm)");
        Connect(biso3_lmax,  "biso3_lmax",  "maximum angular momentum l for biso3");
        Connect(biso3_rmin,  "biso3_rmin",  "minimum radius for biso3");
        Connect(biso3_rmax,  "biso3_rmax",  "maximum radius (cutoff) for biso3");
        Connect(biso3_rsize, "biso3_rsize", "radial basis size for biso3");
        Connect(biso3_real_harmonics, "biso3_real_harmonics", "use the real spherical harmonic backend");
    }

    /// Default constructor: register settings without initializing a descriptor.
    descriptor_wrapper() { InitSettings(); }

    /// Construct from GPSettings alone (for self-contained descriptor types).
    descriptor_wrapper(GPSettings settings)
    {
        InitSettings();
        ApplySettings(settings);
        if (dtype == "mtm") {
            const auto path = settings["mtm:potfile"];
            if (path.empty()) GPERROR("MTM descriptor requires mtm:potfile or an explicit PotMTP");
            owned_mtp_ = std::make_unique<Potentials::PotMTP>(
                std::filesystem::absolute(path).lexically_normal().string());
        }
        init(settings, owned_mtp_.get());
    }

    /// Construct with an owned PotMTP (takes ownership via unique_ptr).
    descriptor_wrapper(GPSettings settings,
                       std::unique_ptr<Potentials::PotMTP> mtp)
        : owned_mtp_(std::move(mtp))
    {
        InitSettings();
        ApplySettings(settings);
        init(settings, owned_mtp_.get());
    }

    /// Construct referencing an externally-owned PotMTP (no ownership transfer).
    descriptor_wrapper(GPSettings settings,
                       Potentials::PotMTP *external_mtp)
    {
        InitSettings();
        ApplySettings(settings);
        init(settings, external_mtp);
    }

    /// Destructor defined in .cpp so unique_ptr<PotMTP> can see the complete type.
    ~descriptor_wrapper();

    // ── Delegating accessors ──────────────────────────────────────────────

    int feature_size() const { return desc->feature_size(); }

    void compute(Neighborhood &nbh, Eigen::VectorXd &out)
    {
        desc->compute(nbh, out);
    }

    void compute(Neighborhood &nbh, Eigen::VectorXd &out,
                 const ComputeOptions &opts)
    {
        desc->compute(nbh, out, opts);
    }

    /// Convenience: compute descriptor values and position gradients.
    void compute_with_gradient(Neighborhood &nbh, Eigen::VectorXd &out,
                               DescriptorGradient &grad,
                               const ComputeOptions &opts = ComputeOptions{})
    {
        ComputeOptions local_opts = opts;
        local_opts.gradient = &grad;
        local_opts.compute_derivative = true;
        desc->compute(nbh, out, local_opts);
    }

private:
    std::unique_ptr<Potentials::PotMTP> owned_mtp_;

    void init(GPSettings &settings, Potentials::PotMTP *mtp)
    {
        if (dtype == "behler") {
            desc = std::make_unique<DescriptorBehler>(
                settings.ExtractSubSettings("behler"));
        } else if (dtype == "bondorder") {
            desc = std::make_unique<DescriptorBondOrder>(
                settings.ExtractSubSettings("bondorder"));
        } else if (dtype == "biso3") {
            desc = std::make_unique<Descriptor_BispectrumSO3>(
                biso3_lmax, biso3_rmin, biso3_rmax, biso3_rsize,biso3_real_harmonics);
        } else if (dtype == "biso4") {
            desc = std::make_unique<Descriptor_BispectrumSO4>(
                settings.ExtractSubSettings("biso4"));
        } else if (dtype == "mtm") {
            if (!mtp)
                GPERROR("MTM descriptor requires a PotMTP; "
                        "pass one to the descriptor_wrapper constructor");
            desc = std::make_unique<DescriptorMTM>(*mtp);
        } else {
            GPERROR("Unknown descriptor type: " + dtype);
        }
    }
};
