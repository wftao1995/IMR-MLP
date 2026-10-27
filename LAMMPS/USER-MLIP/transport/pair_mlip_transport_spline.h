// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

/* -*- c++ -*- ---------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mlip/transport/spline,PairMLIPTransportSpline)
// clang-format on
#else

#ifndef LMP_PAIR_MLIP_TRANSPORT_SPLINE_H
#define LMP_PAIR_MLIP_TRANSPORT_SPLINE_H

#include "pair_mlip_transport.h"

namespace LAMMPS_NS {

class PairMLIPTransportSpline : public PairMLIPTransport {
 public:
  explicit PairMLIPTransportSpline(class LAMMPS *);
  ~PairMLIPTransportSpline() override;

 protected:
  int spline_points_ = 0;
  double spline_delta_ = 0.0;
  double *spline_radii_ = nullptr;
  double *spline_coefficients_ = nullptr;

  void load_potential() override;
  bool evaluate_radial(int, double, Scratch &) const override;
  static void interpolate(int, double, const double *, double *);
};

}    // namespace LAMMPS_NS

#endif
#endif
