// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

/* -*- c++ -*- ---------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(mlip/heat/current,ComputeMLIPHeatCurrent)
// clang-format on
#else

#ifndef LMP_COMPUTE_MLIP_HEAT_CURRENT_H
#define LMP_COMPUTE_MLIP_HEAT_CURRENT_H

#include "compute.h"
#include "mlip_transport_provider.h"

namespace LAMMPS_NS {

class ComputeMLIPHeatCurrent : public Compute, public MLIPTransportCurrentRequest {
 public:
  ComputeMLIPHeatCurrent(class LAMMPS *, int, char **);
  ~ComputeMLIPHeatCurrent() override;
  void init() override;
  double compute_scalar() override;
  void compute_peratom() override;
  double memory_usage() override;

  int transport_left_groupbit() const override { return groupbit; }
  int transport_right_groupbit() const override { return right_groupbit_; }
  void set_transport_channel(int channel) override { channel_ = channel; }

 private:
  char *right_group_name_ = nullptr;
  int right_group_ = -1;
  int right_groupbit_ = 0;
  int channel_ = -1;
  int nmax_ = 0;
  double **interface_force_ = nullptr;
  MLIPTransportProvider *provider_ = nullptr;

  void validate_groups();
};

}    // namespace LAMMPS_NS

#endif
#endif
