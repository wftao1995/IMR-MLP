// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "descriptor_wrapper.h"
#include "potentials/potential_mtp.h"

/// Destructor must be defined here (not in the header) because
/// unique_ptr<PotMTP> needs the complete type to invoke delete.
descriptor_wrapper::~descriptor_wrapper() = default;
