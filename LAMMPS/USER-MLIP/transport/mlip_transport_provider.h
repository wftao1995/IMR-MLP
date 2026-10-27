// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

/* -*- c++ -*- ----------------------------------------------------------
   Public contract between the MTP transport pair styles and computes.
------------------------------------------------------------------------- */

#ifndef LMP_MLIP_TRANSPORT_PROVIDER_H
#define LMP_MLIP_TRANSPORT_PROVIDER_H

#include "lmptype.h"

namespace LAMMPS_NS {

class MLIPTransportProvider {
 public:
  virtual ~MLIPTransportProvider() noexcept(false) = default;

  virtual void ensure_transport_reverse(bigint) = 0;
  virtual bigint transport_timestep() const = 0;
  virtual bool transport_flux_enabled() const = 0;
  virtual int transport_interface_count() const = 0;
  virtual const double *transport_site_energy() const = 0;
  virtual const double *transport_heat_virial() const = 0;
  virtual const double *transport_interface_force(int) const = 0;
};

// Marker used by the pair style while it scans computes before force setup.
class MLIPTransportFluxRequest {
 public:
  virtual ~MLIPTransportFluxRequest() noexcept(false) = default;
};

class MLIPTransportCurrentRequest {
 public:
  virtual ~MLIPTransportCurrentRequest() noexcept(false) = default;
  virtual int transport_left_groupbit() const = 0;
  virtual int transport_right_groupbit() const = 0;
  virtual void set_transport_channel(int) = 0;
};

}    // namespace LAMMPS_NS

#endif
