// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

/* -*- c++ -*- ---------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(mlip/heat/flux,ComputeMLIPHeatFlux)
// clang-format on
#else

#ifndef LMP_COMPUTE_MLIP_HEAT_FLUX_H
#define LMP_COMPUTE_MLIP_HEAT_FLUX_H

#include "compute.h"
#include "mlip_transport_provider.h"

namespace LAMMPS_NS {

class ComputeMLIPHeatFlux : public Compute, public MLIPTransportFluxRequest {
 public:
  ComputeMLIPHeatFlux(class LAMMPS *, int, char **);
  ~ComputeMLIPHeatFlux() override;
  void init() override;
  void compute_vector() override;

 private:
  MLIPTransportProvider *provider_ = nullptr;
};

}    // namespace LAMMPS_NS

#endif
#endif
