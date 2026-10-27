// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base/neighborhood.h"
#include "io/gpio.h"

namespace Training {

std::vector<Atoms> load_training_data(GPIO &io, const std::string &filename,
                                      const std::string &format);
void restore_training_fitness(std::vector<Atoms> &frames);
void distribute_training_data(std::vector<Atoms> &frames, bool balance_atomic_fitness);
std::vector<Neighborhoods> training_neighborhoods(std::vector<Atoms> &frames, double cutoff);

} // namespace Training
