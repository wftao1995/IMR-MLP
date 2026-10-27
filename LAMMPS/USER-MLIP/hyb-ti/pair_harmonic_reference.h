/* -*- c++ -*- ----------------------------------------------------------
   Fixed-cell harmonic reference pair style for thermodynamic integration.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(harmonic/reference,PairHarmonicReference)

#else

#ifndef LMP_PAIR_HARMONIC_REFERENCE_H
#define LMP_PAIR_HARMONIC_REFERENCE_H

#include "pair.h"

#include <array>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairHarmonicReference : public Pair {
 public:
  PairHarmonicReference(class LAMMPS *);
  ~PairHarmonicReference() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double memory_usage() override;

 private:
  using Block = std::array<double, 9>;
  enum class StorageMode { AUTO, REPLICATED, DISTRIBUTED };

  void allocate();
  void load_reference(const std::string &);
  void validate_system();
  void evaluate_row(int, double *) const;

  int reference_atoms_ = 0;
  double energy0_ = 0.0;
  double total_mass_ = 0.0;
  std::array<double, 9> reference_cell_{};
  std::vector<double> reference_positions_;
  std::vector<int> reference_types_;
  std::vector<double> reference_masses_;
  std::vector<double> displacements_;
  std::vector<double> responses_local_;
  std::vector<double> responses_;
  std::vector<int> response_counts_;
  std::vector<int> response_displacements_;
  std::vector<int> row_offsets_;
  std::vector<int> columns_;
  std::vector<Block> blocks_;
  StorageMode requested_storage_ = StorageMode::AUTO;
  int local_row_begin_ = 0;
  int local_row_end_ = 0;
  bool distributed_ = false;
  bool loaded_ = false;
};

} // namespace LAMMPS_NS

#endif
#endif
