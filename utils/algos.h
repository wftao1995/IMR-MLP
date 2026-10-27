// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_ALGO_H
#define IMRMLP_ALGO_H

#include <cstddef>

template<typename T, class G>
void linspace(T start, T end, std::size_t samples, G &arr) {
    arr.resize(samples);
    double stp = (end - start) / static_cast<double>(samples-1);
    for (int i = 0; i < samples; ++i) {
        arr[i] = start + stp * i;
    }
}
#endif
