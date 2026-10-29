// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file potential_wrapper.h
 * @brief Factory that creates AnyPotential from a type string and settings.
 *
 * The heavy lifting (22+ concrete potential includes + the big if/else chain)
 * lives in potential_wrapper.cpp so that touching a single potential header
 * does NOT force recompilation of all 17+ consumer translation units.
 */

#include "potentials/potential.h"
#include "utils/gputils.h"
#include <map>
#include <string>

namespace Potentials {

/// True only for audited concrete CPU implementations (including local composites).
bool supports_frame_local_efs(const AnyPotential &pot);

/// @brief Factory wrapper that creates an AnyPotential from a type string.
///
/// Usage:
///   potential_wrapper pot("potential.mtp", "mtp");
///   pot.pot->getEFS(atom, nbhs);
class potential_wrapper : public ConnectSettings {
public:
    AnyPotential *pot = nullptr;

    std::string potfile;
    std::string pottype = "mtp";
    std::string typemap = "index.map";
    std::string kim_model_name;
    int plev = 2;

    void InitSettings() {
        Connect(potfile, "potfile", "Path to potential parameter file (e.g. .mtp, .eam, .lj input)");
        Connect(pottype, "pottype", "Potential type identifier: mtp, mtpnn1, mtp_m, mtp_les, hmtp, eamalloy, eamfs, eampoly, lj612, sw, tersoff, zbl, pair_table, threebody_table, shellfile, openkim, hybrid, hybrid_scale, mindist, descriptor_linear, acelmp");
        Connect(typemap, "typemap", "Path to index.map file mapping chemical species to type indices");
        Connect(kim_model_name, "kim_model", "OpenKIM model name (used when pottype=openkim), e.g. EAM_Dynamo_...");
        Connect(plev, "potlevel", "");
    }

    potential_wrapper() { InitSettings(); }

    // todo: depreach this
    potential_wrapper(std::string _potfile, std::string _pottype, GPSettings settings = GPSettings()) {
        InitSettings();
        ApplySettings(settings);
        if (settings["potfile"].empty()) potfile = _potfile;
        if (settings["pottype"].empty()) pottype = _pottype;

        settings.Modify("potfile", potfile);
        settings.Modify("pottype", pottype);
        if (pottype.empty()) {
            pottype = "mtp";
        }
        init(settings);
    }

    potential_wrapper(GPSettings settings) {
        init(settings);
    }

    /// @brief Create the concrete potential.  Implementation in .cpp.
    void init(GPSettings &settings);

    void read_map(std::string file, std::map<int, std::string> &mi2s,
                  std::map<std::string, int> &ms2i);

    ~potential_wrapper() {
        delete pot;
    }
};

} // namespace Potentials
