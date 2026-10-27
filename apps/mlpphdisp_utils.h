// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <istream>
#include <string>
#include <string_view>

namespace phonopyfc_detail {

void read_required_yaml_line(std::istream &input,
                             std::string &line,
                             std::string_view expected,
                             std::string_view filename);

void find_required_yaml_field(std::istream &input,
                              std::string &line,
                              std::string_view field,
                              std::string_view filename);

} // namespace phonopyfc_detail
