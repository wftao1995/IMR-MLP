// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include <catch2/catch_all.hpp>

#include "LAMMPS/USER-MLIP/opt/mlip_opt_virial.h"

#include <array>

TEST_CASE("LAMMPS optimized per-atom virial splits a directed edge equally",
          "[lammps][virial]")
{
    std::array<double, 6> center{};
    std::array<double, 6> neighbor{};
    const std::array<double, 3> r{{1.5, -2.0, 0.75}};
    const std::array<double, 3> derivative{{2.0, -3.0, 4.0}};
    std::array<double, 6> full{};

    MLIPOpt::standard_virial(r.data(), derivative.data(), full.data());
    MLIPOpt::add_per_atom_virial(
        center.data(), neighbor.data(), full.data());

    const std::array<double, 6> expected{{
        -r[0] * derivative[0],
        -r[1] * derivative[1],
        -r[2] * derivative[2],
        -0.5 * (r[0] * derivative[1] + r[1] * derivative[0]),
        -0.5 * (r[0] * derivative[2] + r[2] * derivative[0]),
        -0.5 * (r[1] * derivative[2] + r[2] * derivative[1]),
    }};
    for (int component = 0; component < 6; ++component) {
        REQUIRE(full[component] == Catch::Approx(expected[component]));
        REQUIRE(center[component] == Catch::Approx(0.5 * full[component]));
        REQUIRE(neighbor[component] == Catch::Approx(center[component]));
        REQUIRE(center[component] + neighbor[component]
                == Catch::Approx(full[component]));
    }
}

TEST_CASE("LAMMPS optimized per-atom virial accumulation is additive",
          "[lammps][virial]")
{
    std::array<double, 6> center{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
    std::array<double, 6> neighbor{{-1.0, -2.0, -3.0, -4.0, -5.0, -6.0}};
    const auto center_before = center;
    const auto neighbor_before = neighbor;
    const std::array<double, 3> r{{-0.4, 0.7, 1.2}};
    const std::array<double, 3> derivative{{0.6, -0.9, 1.1}};
    std::array<double, 6> full{};

    MLIPOpt::standard_virial(r.data(), derivative.data(), full.data());
    MLIPOpt::add_per_atom_virial(
        center.data(), neighbor.data(), full.data());

    for (int component = 0; component < 6; ++component) {
        REQUIRE(center[component] - center_before[component]
                == Catch::Approx(neighbor[component] - neighbor_before[component]));
    }
}
