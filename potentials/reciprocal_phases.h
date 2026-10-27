// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace Potentials {
// Per-call harmonic workspace shared by IMRMLP and the standalone CPU adapter.
// Storage is bounded; very large tables fall back to direct phases.
class ReciprocalPhases {
    struct CS {
        double c, s;
    };
    std::vector<std::array<double, 3>> positions_;
    std::array<std::vector<CS>, 3> harmonics_;
    bool recurrence_ = false;

public:
    template <class Position>
    void prepare(int n, const std::array<int, 3>& bounds,
        const std::array<std::array<double, 3>, 3>& reciprocal, Position position, bool recurrence)
    {
        positions_.resize(n);
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a)
                positions_[i][a] = position(i, a);
        size_t count = 0;
        for (int a = 0; a < 3; ++a)
            count += size_t(bounds[a]) + 1;
        recurrence_ = recurrence && count <= size_t(64 * 1024 * 1024) / sizeof(CS) / std::max(n, 1);
        if (!recurrence_)
            return;
        constexpr double pi = 3.14159265358979323846;
        for (int a = 0; a < 3; ++a) {
            auto& h = harmonics_[a];
            h.resize((size_t(bounds[a]) + 1) * n);
            for (int i = 0; i < n; ++i) {
                double theta = 0.;
                for (int d = 0; d < 3; ++d)
                    theta += reciprocal[a][d] * positions_[i][d];
                theta = std::remainder(theta, 2 * pi);
                const double c = std::cos(theta), s = std::sin(theta);
                h[i] = { 1., 0. };
                for (int m = 1; m <= bounds[a]; ++m) {
                    const CS prev = h[size_t(m - 1) * n + i];
                    h[size_t(m) * n + i] = m % 32 == 0
                        ? CS { std::cos(m * theta), std::sin(m * theta) }
                        : CS { prev.c * c - prev.s * s, prev.s * c + prev.c * s };
                }
            }
        }
    }
    bool uses_recurrence() const { return recurrence_; }
    void evaluate(
        const std::array<int, 3>& index, const std::array<double, 3>& k, double* c, double* s) const
    {
        const int n = positions_.size();
        if (!recurrence_) {
            for (int i = 0; i < n; ++i) {
                const auto& r = positions_[i];
                const double phase = k[0] * r[0] + k[1] * r[1] + k[2] * r[2];
                c[i] = std::cos(phase);
                s[i] = std::sin(phase);
            }
            return;
        }
        for (int i = 0; i < n; ++i) {
            CS v { 1., 0. };
            for (int a = 0; a < 3; ++a) {
                CS h = harmonics_[a][size_t(std::abs(index[a])) * n + i];
                if (index[a] < 0)
                    h.s = -h.s;
                v = { v.c * h.c - v.s * h.s, v.s * h.c + v.c * h.s };
            }
            c[i] = v.c;
            s[i] = v.s;
        }
    }
};
}
