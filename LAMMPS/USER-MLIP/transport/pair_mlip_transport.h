// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

/* -*- c++ -*- ----------------------------------------------------------
   CPU MTP pair style with native per-atom virial and transport channels.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mlip/transport,PairMLIPTransport)
// clang-format on
#else

#ifndef LMP_PAIR_MLIP_TRANSPORT_H
#define LMP_PAIR_MLIP_TRANSPORT_H

#include "mlip_transport_provider.h"
#include "mlip_pair_metadata.h"
#include "pair.h"
#include "radial_basis_transport.h"

#include <vector>

namespace LAMMPS_NS {

class PairMLIPTransport : public Pair, public MLIPTransportProvider {
 public:
  static constexpr int MAX_SPECIES = 20;

  struct AlphaBasic { int mu, b1, b2, b3; };
  struct AlphaTimes { int l1, l2, c, id; };
  struct AlphaTimesGroup { int begin, end, id; bool assign; };
  struct CoordPower { double x, y, z; };

  struct Scratch {
    double *moment_vals = nullptr;
    double *site_energy_derivatives = nullptr;
    double *dist_powers = nullptr;
    CoordPower *coord_powers = nullptr;
    double *moment_jacobian = nullptr;
    double *rb_vals = nullptr;
    double *rb_ders = nullptr;
    double *val_mu = nullptr;
    double *der_mu = nullptr;
    int *jlist_true = nullptr;
    int neighbor_capacity = 0;
    std::vector<double> heat_virial;
    std::vector<std::vector<double>> interface_force;

    void allocate(int, int, int, int, int, class Memory *);
    void grow_neighbors(int, int, class Memory *);
    void reset_transport(int, bool, int);
    void destroy(class Memory *);
  };

  explicit PairMLIPTransport(class LAMMPS *);
  ~PairMLIPTransport() override;

  void compute(int, int) override;
  void init_style() override;
  double init_one(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;

  void ensure_transport_reverse(bigint) override;
  bigint transport_timestep() const override { return transport_step_; }
  bool transport_flux_enabled() const override { return flux_enabled_; }
  int transport_interface_count() const override { return static_cast<int>(interfaces_.size()); }
  const double *transport_site_energy() const override { return site_energy_.data(); }
  const double *transport_heat_virial() const override { return heat_virial_.data(); }
  const double *transport_interface_force(int) const override;

 protected:
  MTPMetadata::PairArguments metadata_arguments_;
  struct InterfaceChannel {
    int left_bit;
    int right_bit;
    std::vector<double> force;
  };

  bool inited_ = false;
  bool potential_loaded_ = false;
  double cutoff = 0.0;
  double cutoffsq = 0.0;
  char potential_filename_[1000] = {};

  int species_count = 0;
  int alpha_moments_count = 0;
  int alpha_index_basic_count = 0;
  int alpha_index_times_count = 0;
  int alpha_scalar_moments = 0;
  int rb_size = 0;
  int radial_func_count = 0;
  int max_alpha_index_basic = 0;
  int max_mu = 0;
  double scaling = 1.0;
  int spmap[MAX_SPECIES + 1] = {};

  AlphaBasic *alpha_index_basic = nullptr;
  AlphaTimes *alpha_index_times = nullptr;
  AlphaTimesGroup *alpha_times_groups = nullptr;
  int alpha_times_group_count = 0;
  int *alpha_moment_mapping = nullptr;
  int *moment_zero_ids = nullptr;
  int moment_zero_count = 0;
  int *nu_arr_list = nullptr;

  double min_dist = 0.0;
  double max_dist = 0.0;
  bool radial_basis_split = false;
  std::vector<MLIPTransport::RadialBasis> radial_bases;

  double *radial_coeffs = nullptr;
  double *moment_coeffs = nullptr;
  double *specie_coeffs = nullptr;

  Scratch scratch_;

  bool flux_enabled_ = false;
  std::vector<InterfaceChannel> interfaces_;
  std::vector<double> site_energy_;
  std::vector<double> heat_virial_;
  int transport_nall_ = 0;
  bigint transport_step_ = -1;
  bigint reverse_step_ = -1;

  void allocate();
  void deallocate();
  virtual void load_potential();
  virtual bool evaluate_radial(int, double, Scratch &) const;
  void configure_transport_requests();
  void prepare_transport_storage(int);
  void reset_transport_storage(int);
  void tally_transport(Scratch &, int, int, const double *, const double *);
  void tally_standard_vatom(double **, int, int, const double *, const double *) const;
  void compute_range(int, int, Scratch &, double **, double *, double **, double &);
  void reduce_transport_scratch(const std::vector<Scratch *> &);
};

}    // namespace LAMMPS_NS

#endif
#endif
