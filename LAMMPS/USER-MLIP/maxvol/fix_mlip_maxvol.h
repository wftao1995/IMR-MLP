// Copyright (c) 2026 atoms-agent and IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only

#ifdef FIX_CLASS

FixStyle(mlip/maxvol, FixMLIPMaxVol)

#else
#ifndef LMP_FIX_MLIP_MAXVOL_H
#define LMP_FIX_MLIP_MAXVOL_H

#include "fix.h"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

class ActiveSelector;
class GPIO;
namespace Potentials { class AnyPotential; }

namespace LAMMPS_NS {

class FixMLIPMaxVol : public Fix {
 public:
  FixMLIPMaxVol(class LAMMPS *, int, char **);
  ~FixMLIPMaxVol() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void end_of_step() override;
  double compute_scalar() override;
  double compute_vector(int) override;

 private:
  void sample(bool);
  void root_error(std::string &);

  std::string model_file_, state_file_, index_file_, save_file_ = "sampled.xyz";
  std::string potential_type_, settings_file_;
  std::vector<int> species_map_;
  double threshold_save_ = 2.1, threshold_break_ = 11.0;
  double grade_ = 0.0;
  bigint saved_ = 0, last_sample_ = -1;
  bool ready_ = false;
  // The potential must outlive the reconstructed builder.
  std::unique_ptr<Potentials::AnyPotential> model_;
  std::unique_ptr<ActiveSelector> selector_;
  std::unique_ptr<GPIO> io_;
  std::ofstream output_;
};

} // namespace LAMMPS_NS
#endif
#endif
