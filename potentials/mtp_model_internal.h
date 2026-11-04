// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
// Private object access shared by MTP initialization and adjustment.
#include "potential.h"

namespace Potentials
{
class PotMTP;
}

namespace Potentials::MTP::detail
{
std::map<std::string, AnyPotential *> model_children(AnyPotential &model);
std::map<int, std::string> model_element_map(AnyPotential &model);
std::vector<std::string> model_elements(AnyPotential &model);
GPSettings composite_settings(AnyPotential &model);
std::unique_ptr<AnyPotential> make_composite(const std::string &type, GPSettings settings,
                                             std::vector<std::unique_ptr<AnyPotential>> children);
std::unique_ptr<PotMTP> copy_mtp(PotMTP &model, bool short_range_only = false);

}
