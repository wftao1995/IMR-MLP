// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "utils/gputils.h"

namespace Potentials {

// Settings-based potentials save `pot:key value`; also accept the older
// `key=value` stream format and unprefixed settings.
inline GPSettings read_potential_settings(std::istream &stream)
{
    GPSettings settings;
    std::string line;
    while (std::getline(stream, line)) {
        trim_comment(line, "#!");
        trim_all(line);
        if (line.empty()) continue;
        const auto separator = line.find_first_of("= \t");
        if (separator == std::string::npos)
            GPERROR("Missing potential setting value: " + line);
        auto key = line.substr(0, separator);
        auto value = line.substr(separator + 1);
        trim_all(value);
        if (!value.empty() && value.front() == '=') {
            value.erase(0, 1);
            trim_all(value);
        }
        if (key.starts_with("pot:")) key.erase(0, 4);
        if (key.empty() || value.empty())
            GPERROR("Missing potential setting key or value: " + line);
        settings.Modify(key, value);
    }
    if (stream.bad()) GPERROR("Failed to read potential settings");
    return settings;
}

} // namespace Potentials
