/* -*- c++ -*- */
#ifdef PAIR_CLASS
PairStyle(mlip/native,PairMLIPNative)
#else
#ifndef LMP_PAIR_MLIP_NATIVE_H
#define LMP_PAIR_MLIP_NATIVE_H

#include "pair.h"
#include <memory>
#include <string>
#include <vector>

class Atoms;
namespace Potentials {
class NativeInference;
struct NativeInferenceRequest;
struct NativeInferenceResult;
struct NativeCollectives;
}

namespace LAMMPS_NS {
class PairMLIPNative : public Pair {
 public:
  explicit PairMLIPNative(class LAMMPS *);
  ~PairMLIPNative() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void *extract(const char *, int &) override;
  double memory_usage() override;

 private:
  void allocate();
  void load_model();
  void collective_error(const std::string &);
  void set_geometry(Atoms &) const;
  void compute_local(const Potentials::NativeInferenceRequest &);
  void accumulate(const Potentials::NativeInferenceResult &, int, bool);
  Potentials::NativeCollectives collectives();
  void read_string(FILE *, std::string &);
  static void write_string(FILE *, const std::string &);

  struct Workspace;
  std::unique_ptr<Workspace> workspace_;
  std::unique_ptr<Potentials::NativeInference> model_;
  std::string model_file_, settings_file_, pottype_, device_ = "cpu";
  int requested_gpu_ = -1, selected_gpu_ = -1;
  std::vector<int> species_map_;
  std::vector<double> charges_, latents_;
  int channels_ = 0, has_charges_ = 0;
  double cutoff_ = 0.;
  bool ready_ = false, local_ = false, evaluated_ = false;
};
}
#endif
#endif
