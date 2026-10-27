// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "spline.h"

namespace Spline {

void build_cubic_spline(int n, double delta, const std::vector<double> &f,
                        Eigen::MatrixXd &spline) {
    spline.resize(n + 1, 7);
    spline.setZero();
    for (int m = 1; m <= n; ++m)
        spline(m, 6) = f[m];

    spline(1, 5) = spline(2, 6) - spline(1, 6);
    spline(2, 5) = 0.5 * (spline(3, 6) - spline(1, 6));
    spline(n - 1, 5) = 0.5 * (spline(n, 6) - spline(n - 2, 6));
    spline(n, 5) = spline(n, 6) - spline(n - 1, 6);

    for (int m = 3; m <= n - 2; ++m) {
        spline(m, 5) = ((spline(m - 2, 6) - spline(m + 2, 6))
                     + 8.0 * (spline(m + 1, 6) - spline(m - 1, 6))) / 12.0;
    }

    for (int m = 1; m <= n - 1; ++m) {
        spline(m, 4) = 3.0 * (spline(m + 1, 6) - spline(m, 6))
                     - 2.0 * spline(m, 5) - spline(m + 1, 5);
        spline(m, 3) = spline(m, 5) + spline(m + 1, 5)
                     - 2.0 * (spline(m + 1, 6) - spline(m, 6));
    }

    spline(n, 4) = 0.0;
    spline(n, 3) = 0.0;

    for (int m = 1; m <= n; ++m) {
        spline(m, 2) = spline(m, 5) / delta;
        spline(m, 1) = 2.0 * spline(m, 4) / delta;
        spline(m, 0) = 3.0 * spline(m, 3) / delta;
    }
}

} // namespace Spline
