// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

/* Per-atom latent charges exported by pair_style mlip/opt-les. */
#ifdef COMPUTE_CLASS

ComputeStyle(mlip/les/charge,ComputeMLIPLESCharge)
ComputeStyle(mlip/sog/latent,ComputeMLIPLESCharge)

#else
#ifndef LMP_COMPUTE_MLIP_LES_CHARGE_H
#define LMP_COMPUTE_MLIP_LES_CHARGE_H

#include "compute.h"

namespace LAMMPS_NS {
class ComputeMLIPLESCharge : public Compute {
 public:
  ComputeMLIPLESCharge(class LAMMPS *, int, char **);
  ~ComputeMLIPLESCharge() override;
  void init() override;
  void compute_peratom() override;
  double memory_usage() override;

 private:
  int nmax_ = 0;
  bool latent_ = false;
  bool layout_initialized_ = false;
  class Pair *pair_ = nullptr;
};
}
#endif
#endif
