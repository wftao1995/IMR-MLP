// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "potentials/potential.h"

namespace Potentials
{
class PotMTP;
}

namespace Potentials::MTP
{
struct Adjustment
{
    std::vector<std::string> appended, elements;
    std::vector<int> sources, order;
    std::string initialization, basis = "keep";
};

std::unique_ptr<Potentials::AnyPotential> adjust_tree(Potentials::AnyPotential &,
                                                      const Adjustment &);
// Historical native command: random species expansion and unconditional level sorting.
void adjust_legacy(PotMTP &model, const Adjustment &operation);
void save_legacy_adjusted(PotMTP &model, const std::string &output);
void validate_graph(const Potentials::PotMTP &model);
}
