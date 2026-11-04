// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "potentials/potential.h"
#include <memory>
#include <string>
#include <vector>

namespace Potentials::MTP
{
struct InitializationOptions
{
    std::string input, type;
    std::vector<std::string> elements;
    bool reset = false, randomize = true;
    double lower = -.1, upper = .1, scale = 1.;
    std::string index;
    bool explicit_index = false;
    std::string dataset, format;
    bool fit = false, normalize = false;
    std::string offset_target;
};

// Construct and initialize in this process; does not publish an output model.
std::unique_ptr<AnyPotential> initialize_model(GPSettings settings,
                                               const InitializationOptions &options);
// Historical data-driven MTP-M initialization, with its original numerical order.
std::unique_ptr<AnyPotential> initialize_legacy(GPSettings model_settings,
                                                std::vector<Atoms> &atoms,
                                                const InitializationOptions &options);
// Preserve the historical native save and parameter/EFS reload checks.
void save_legacy_initialized(AnyPotential &model, const std::string &output,
                             const GPSettings &model_settings, const std::vector<Atoms> &atoms);
}
