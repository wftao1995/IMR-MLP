// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

/* -*- c++ -*- ---------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mlip/transport/omp,PairMLIPTransportOMP)
// clang-format on
#else

#ifndef LMP_PAIR_MLIP_TRANSPORT_OMP_H
#define LMP_PAIR_MLIP_TRANSPORT_OMP_H

#include "pair_mlip_transport.h"
#if __has_include("thr_omp.h")
#include "thr_omp.h"
#else
#include "OPENMP/thr_omp.h"
#endif

namespace LAMMPS_NS {

class PairMLIPTransportOMP : public PairMLIPTransport, public ThrOMP {
 public:
  explicit PairMLIPTransportOMP(class LAMMPS *);
  ~PairMLIPTransportOMP() override;

  void init_style() override;
  void compute(int, int) override;

 private:
  std::vector<Scratch> thread_scratch_;
  void allocate_thread_scratch();
  void destroy_thread_scratch();
};

}    // namespace LAMMPS_NS

#endif
#endif
