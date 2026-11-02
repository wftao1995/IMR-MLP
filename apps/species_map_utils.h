// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "io/gpio.h"
#include "utils/gputils.h"

#include <string>

// App-local helpers used while wiring model-declared species order into the
// structure reader. static inline keeps them private to each command TU.
static inline GPSettings make_potential_settings(
    const GPSettings &all,
    const std::string &potfile = {},
    const std::string &pottype = {}) {
    GPSettings pot = all.ExtractSubSettings("pot");
    if (!pot.count("potfile") && !potfile.empty()) pot.Modify("potfile", potfile);
    if (!pot.count("pottype") && !pottype.empty()) pot.Modify("pottype", pottype);
    return pot;
}

static inline void share_species_map(GPIO &target, const GPIO &source,
                                     std::string *target_index_file = nullptr) {
    target.setmap(source.getmap());
    if (target_index_file) target_index_file->clear();
}
