/* -*- c++ -*- */
#ifdef COMPUTE_CLASS
ComputeStyle(mlip/native/charge,ComputeMLIPNativeCharge)
ComputeStyle(mlip/native/latent,ComputeMLIPNativeLatent)
#else
#ifndef LMP_COMPUTE_MLIP_NATIVE_H
#define LMP_COMPUTE_MLIP_NATIVE_H
#include "compute.h"

namespace LAMMPS_NS {
class ComputeMLIPNativeObservable : public Compute {
 public:
  ComputeMLIPNativeObservable(class LAMMPS *, int, char **, bool);
  ~ComputeMLIPNativeObservable() override;
  void init() override;
  void compute_peratom() override;
  double memory_usage() override;
 private:
  class Pair *pair_ = nullptr;
  bool latent_;
  int nmax_ = 0, channels_ = 1;
};
class ComputeMLIPNativeCharge : public ComputeMLIPNativeObservable {
 public:
  ComputeMLIPNativeCharge(class LAMMPS *lmp, int argc, char **argv)
    : ComputeMLIPNativeObservable(lmp,argc,argv,false) {}
};
class ComputeMLIPNativeLatent : public ComputeMLIPNativeObservable {
 public:
  ComputeMLIPNativeLatent(class LAMMPS *lmp, int argc, char **argv)
    : ComputeMLIPNativeObservable(lmp,argc,argv,true) {}
};
}
#endif
#endif
