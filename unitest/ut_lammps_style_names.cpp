// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include <array>
#include <string_view>

namespace
{
// Match LAMMPS registration: the exact spelling of each macro argument is the
// public style name. No LAMMPS headers or linked runtime are needed here.
#define PAIR_CLASS
#define PairStyle(name, type) std::string_view{#name},
constexpr std::array pair_names = {
#include "LAMMPS/USER-MLIP/pair_MLIP.h"
#include "LAMMPS/USER-MLIP/opt/pair_MLIP_opt.h"
#include "LAMMPS/USER-MLIP/opt/pair_MLIP_omp.h"
#include "LAMMPS/USER-MLIP/opt/pair_MLIP_spline.h"
#include "LAMMPS/USER-MLIP/opt-les/pair_MLIP_opt_les.h"
#include "LAMMPS/USER-MLIP/pair_MLIP_gpu_direct.h"
#include "LAMMPS/USER-MLIP/native/pair_mlip_native.h"
#include "LAMMPS/USER-MLIP/hyb-ti/pair_harmonic_reference.h"
#include "LAMMPS/USER-MLIP/transport/pair_mlip_transport.h"
#include "LAMMPS/USER-MLIP/transport/pair_mlip_transport_omp.h"
#include "LAMMPS/USER-MLIP/transport/pair_mlip_transport_spline.h"
};
#undef PairStyle
#undef PAIR_CLASS

#define COMPUTE_CLASS
#define ComputeStyle(name, type) std::string_view{#name},
constexpr std::array compute_names = {
#include "LAMMPS/USER-MLIP/native/compute_mlip_native.h"
#include "LAMMPS/USER-MLIP/opt-les/compute_mlip_les_charge.h"
#include "LAMMPS/USER-MLIP/transport/compute_mlip_heat_flux.h"
#include "LAMMPS/USER-MLIP/transport/compute_mlip_heat_current.h"
};
#undef ComputeStyle
#undef COMPUTE_CLASS

#define FIX_CLASS
#define FixStyle(name, type) std::string_view{#name},
constexpr std::array fix_names = {
#include "LAMMPS/USER-MLIP/maxvol/fix_mlip_maxvol.h"
};
#undef FixStyle
#undef FIX_CLASS
}

TEST_CASE("LAMMPS registered style names retain their exact public spelling", "[lammps][styles]")
{
    using namespace std::literals;
    constexpr std::array expected_pairs = {
        "mlip"sv,
        "mlip/opt"sv,
        "mlip/opt/omp"sv,
        "mlip/spline"sv,
        "mlip/opt-les"sv,
        "mlip/gpudirect"sv,
        "mlip/native"sv,
        "harmonic/reference"sv,
        "mlip/transport"sv,
        "mlip/transport/omp"sv,
        "mlip/transport/spline"sv,
    };
    constexpr std::array expected_computes = {"mlip/native/charge"sv, "mlip/native/latent"sv,
                                              "mlip/les/charge"sv,    "mlip/sog/latent"sv,
                                              "mlip/heat/flux"sv,     "mlip/heat/current"sv};
    constexpr std::array expected_fixes = {"mlip/maxvol"sv};
    CHECK(pair_names == expected_pairs);
    CHECK(compute_names == expected_computes);
    CHECK(fix_names == expected_fixes);
}
