// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "mlip_transport_tally.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace {

bool close(double lhs, double rhs, double tolerance = 1.0e-12)
{
  return std::abs(lhs - rhs) <= tolerance *
         std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

}    // namespace

int main()
{
  // Finite-difference oracle for U_i(r) = k |r|^2 / 2.
  constexpr double k = 2.75;
  constexpr double epsilon = 1.0e-6;
  const std::array<double, 3> r = {1.2, -0.7, 0.4};
  std::array<double, 3> derivative = {};
  const auto energy = [](const std::array<double, 3> &x) {
    return 0.5 * k * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  };
  for (int component = 0; component < 3; ++component) {
    auto plus = r;
    auto minus = r;
    plus[component] += epsilon;
    minus[component] -= epsilon;
    derivative[component] = (energy(plus) - energy(minus)) / (2.0 * epsilon);
    assert(close(derivative[component], k * r[component], 2.0e-10));
  }

  double w[6] = {};
  MLIPTransport::standard_virial(r.data(), derivative.data(), w);
  assert(close(w[0], -r[0] * derivative[0]));
  assert(close(w[3], -0.5 * (r[0] * derivative[1] +
                             r[1] * derivative[0])));

  double b[9] = {};
  MLIPTransport::add_heat_virial(r.data(), derivative.data(), b);
  assert(close(b[3], -r[0] * derivative[1]));
  assert(close(b[6], -r[1] * derivative[0]));

  double left_to_right[3] = {};
  double right_to_left[3] = {};
  MLIPTransport::add_interface_force(0, 1, derivative.data(), left_to_right);
  MLIPTransport::add_interface_force(1, 0, derivative.data(), right_to_left);
  for (int component = 0; component < 3; ++component)
    assert(close(left_to_right[component], -right_to_left[component]));
}
