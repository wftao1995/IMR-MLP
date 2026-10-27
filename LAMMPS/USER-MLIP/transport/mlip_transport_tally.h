// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#ifndef LMP_MLIP_TRANSPORT_TALLY_H
#define LMP_MLIP_TRANSPORT_TALLY_H

namespace MLIPTransport {

// r = x_neighbor - x_center, d = d U_center / d r.
inline void standard_virial(const double *r, const double *d, double *w)
{
  w[0] = -r[0] * d[0];
  w[1] = -r[1] * d[1];
  w[2] = -r[2] * d[2];
  w[3] = -0.5 * (r[0] * d[1] + r[1] * d[0]);
  w[4] = -0.5 * (r[0] * d[2] + r[2] * d[0]);
  w[5] = -0.5 * (r[1] * d[2] + r[2] * d[1]);
}

// Non-symmetric heat virial B = -r tensor d.
// Storage: xx, yy, zz, xy, xz, yz, yx, zx, zy.
inline void add_heat_virial(const double *r, const double *d, double *b)
{
  b[0] -= r[0] * d[0];
  b[1] -= r[1] * d[1];
  b[2] -= r[2] * d[2];
  b[3] -= r[0] * d[1];
  b[4] -= r[0] * d[2];
  b[5] -= r[1] * d[2];
  b[6] -= r[1] * d[0];
  b[7] -= r[2] * d[0];
  b[8] -= r[2] * d[1];
}

inline void add_interface_force(int center_side, int neighbor_side,
                                const double *d, double *g)
{
  const int delta = neighbor_side - center_side;
  g[0] -= delta * d[0];
  g[1] -= delta * d[1];
  g[2] -= delta * d[2];
}

}    // namespace MLIPTransport

#endif
