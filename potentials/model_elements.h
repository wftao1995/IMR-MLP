// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "utils/gputils.h"

#include <string>
#include <vector>

namespace Potentials
{

/// Read the element order a model declares, without constructing the potential.
///
/// Supported sources are legacy MTP headers (potential_tag.elements), GPM
/// manifests (species table), and hybrid settings composed of those. Other
/// potential types return an empty list and keep the existing index.map path.
std::vector<std::string> probe_model_elements(const GPSettings &pot_settings);

} // namespace Potentials
