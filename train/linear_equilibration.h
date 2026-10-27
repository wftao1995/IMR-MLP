// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace Training {

// H already includes the physical regularizer. Change coordinates c = D*z,
// without changing H's rank or adding a numerical ridge. Only the populated
// triangle is read or written (ALGLIB uses upper, Eigen LDLT uses lower).
template<class Matrix, class Vector>
std::vector<double> equilibrate_linear_system(Matrix &h, Vector &b, int n, bool upper) {
    std::vector<double> scale(n, 1.);
    for (int i = 0; i < n; ++i) {
        const double diagonal = h(i, i);
        if (!std::isfinite(diagonal) || diagonal < 0.)
            throw std::runtime_error("Linear equilibration requires finite nonnegative diagonal entries");
        // A zero column remains zero; identifying/removing nullspaces is the
        // solver's responsibility, not a reason to invent extra regularization.
        if (diagonal > 0.) scale[i] = 1. / std::sqrt(diagonal);
    }
    for (int i = 0; i < n; ++i) {
        b[i] *= scale[i];
        if (!std::isfinite(b[i]))
            throw std::runtime_error("Non-finite equilibrated linear right-hand side");
        for (int j = upper ? i : 0; j < (upper ? n : i + 1); ++j) {
            // Do not form scale[i]*scale[j]: that product can overflow even
            // when the scaled matrix entry is representable.
            h(i, j) = (h(i, j) * scale[i]) * scale[j];
            if (!std::isfinite(h(i, j)))
                throw std::runtime_error("Non-finite equilibrated linear matrix");
        }
    }
    return scale;
}

} // namespace Training
