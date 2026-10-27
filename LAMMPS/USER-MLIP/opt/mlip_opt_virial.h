// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#ifndef LMP_MLIP_OPT_VIRIAL_H
#define LMP_MLIP_OPT_VIRIAL_H

namespace MLIPOpt {

// r = x_neighbor - x_center, derivative = d U_center / d r.
inline void standard_virial(const double *r, const double *derivative,
                            double *virial)
{
    virial[0] = -r[0] * derivative[0];
    virial[1] = -r[1] * derivative[1];
    virial[2] = -r[2] * derivative[2];
    virial[3] = -0.5 * (r[0] * derivative[1] + r[1] * derivative[0]);
    virial[4] = -0.5 * (r[0] * derivative[2] + r[2] * derivative[0]);
    virial[5] = -0.5 * (r[1] * derivative[2] + r[2] * derivative[1]);
}

// LAMMPS stress/atom assigns half of an edge virial to each endpoint.
inline void add_per_atom_virial(double *center, double *neighbor,
                                const double *virial)
{
    for (int component = 0; component < 6; ++component) {
        const double contribution = 0.5 * virial[component];
        center[component] += contribution;
        neighbor[component] += contribution;
    }
}

} // namespace MLIPOpt

#endif
