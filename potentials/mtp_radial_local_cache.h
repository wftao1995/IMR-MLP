#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace Potentials {

// The radial contraction f_mu is already computed once per neighbour. Tensor
// components also share f_mu/r^nu and its derivative; keep just those scalars
// during this site evaluation. No geometry or coefficient values survive a
// begin_site/begin_neighbor boundary.
class MTPadialLocalCache {
    int degrees_ = 0;
    std::vector<std::pair<double, double>> entries_;
    std::vector<unsigned char> ready_;
    std::size_t evaluations_ = 0;

public:
    void begin_site(int mu_count, int degree_count) {
        degrees_ = degree_count;
        entries_.resize(static_cast<std::size_t>(mu_count) * degree_count);
        ready_.resize(entries_.size());
        evaluations_ = 0;
    }
    void begin_neighbor() { std::fill(ready_.begin(), ready_.end(), 0); }

    template<bool WithDerivative>
    std::pair<double, double> get(int mu, int degree, double radial_value,
                                 double radial_derivative, double inverse_power,
                                 double inverse_distance) {
        const auto index = static_cast<std::size_t>(mu) * degrees_ + degree;
        if (!ready_[index]) {
            auto &entry = entries_[index];
            entry.first = radial_value * inverse_power;
            if constexpr (WithDerivative)
                entry.second = radial_derivative * inverse_power -
                               degree * entry.first * inverse_distance;
            else
                entry.second = 0.;
            ready_[index] = 1;
            ++evaluations_;
        }
        return entries_[index];
    }

    std::size_t evaluations() const { return evaluations_; }
    std::size_t capacity() const { return entries_.capacity(); }
};

} // namespace Potentials
