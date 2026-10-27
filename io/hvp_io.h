// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base/structures.h"
#include <fast_float/fast_float.h>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace HVPIO {
inline bool is_key(std::string_view key) {
    return key == "hvp_directions" || key == "hvp_targets";
}

// Text layout is probe-major, then atom-major xyz, independent of Eigen storage.
inline void read(Atoms &atom, std::string_view key, std::string_view text) {
    auto &matrix = key == "hvp_directions" ? atom.hvp_directions_ : atom.hvp_targets_;
    if (matrix.cols() != 0) throw std::invalid_argument("Duplicate HVP tag");
    std::vector<double> values;
    const char *p = text.data(), *end = p + text.size();
    while (p < end) {
        while (p < end && std::isspace(static_cast<unsigned char>(*p))) ++p;
        if (p == end) break;
        double value;
        auto result = fast_float::from_chars(p, end, value);
        if (result.ec != std::errc() || !std::isfinite(value) ||
            (result.ptr < end && !std::isspace(static_cast<unsigned char>(*result.ptr))))
            throw std::invalid_argument("Invalid number in HVP tag");
        values.push_back(value);
        p = result.ptr;
    }
    const size_t ndof = 3 * static_cast<size_t>(atom.size());
    if (ndof == 0 || values.empty() || values.size() % ndof != 0)
        throw std::invalid_argument("HVP tag length must be a positive multiple of 3N");
    matrix.resize(ndof, values.size() / ndof);
    for (int k = 0; k < matrix.cols(); ++k)
        for (int i = 0; i < matrix.rows(); ++i)
            matrix(i, k) = values[k * ndof + i];
}

inline std::string write(const Eigen::MatrixXd &matrix) {
    std::ostringstream out;
    out << std::setprecision(17);
    for (int k = 0; k < matrix.cols(); ++k)
        for (int i = 0; i < matrix.rows(); ++i) {
            if (k || i) out << ' ';
            out << matrix(i, k);
        }
    return out.str();
}
} // namespace HVPIO
