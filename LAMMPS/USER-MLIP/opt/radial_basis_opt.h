// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#ifndef LMP_MLIP_OPT_RADIAL_BASIS_H
#define LMP_MLIP_OPT_RADIAL_BASIS_H
#include "radial_cache.h"

namespace MLIPOpt {
struct RadialBasis : MLIPCore::mlip::radial_basis_t {
  std::string name;
  bool excludes(double r) const { return MLIPCore::Radial::excludes(*this,r); }
  void evaluate(double r,double *values,double *derivatives) const {
    MLIPCore::Radial::evaluate(*this,r,values,derivatives);
  }
  // The neighbor loop checks this exact distance before calling.
  void evaluate_inside(double r,double *values,double *derivatives) const {
    MLIPCore::Radial::evaluate<false>(*this,r,values,derivatives);
  }
};
template<class T> inline void read_assignment_value(std::istream &in,const char *key,T &value) {
  MLIPCore::Radial::read_assignment_value(in,key,&value);
}
struct ParsedRadialBasis { RadialBasis basis; std::string next_token; };
inline ParsedRadialBasis read_radial_basis(std::istream &in,double scaling) {
  const auto parsed=MLIPCore::Radial::read_radial_basis(in,scaling,true);
  ParsedRadialBasis result;
  static_cast<MLIPCore::mlip::radial_basis_t &>(result.basis)=parsed.basis;
  result.basis.name=parsed.name;result.next_token=parsed.next_token;
  return result;
}
inline std::vector<RadialBasis> read_radial_bases(
    std::istream &input, std::string &token, int species_count, double scaling,
    bool &split, int &basis_size, double &max_cutoff)
{
  split = false;
  if (token == "radial_basis_split" || token == "radial_split") {
    int enabled = 0;
    read_assignment_value(input, token.c_str(), enabled);
    if (enabled != 1)
      throw std::runtime_error(token + " must equal 1 when present");
    split = true;
    if (!(input >> token)) throw std::runtime_error("missing split pair block");
  }

  if (species_count<=0 || species_count>10000) throw std::runtime_error("invalid species_count");
  const int count = species_count * species_count;
  std::vector<RadialBasis> bases(split ? count : 1);
  std::vector<bool> seen(split ? count : 0, false);
  if (split) {
    for (int block = 0; block < count; ++block) {
      if (token != "pair") throw std::runtime_error("expected 'pair i j'");
      int i = -1, j = -1;
      if (!(input >> i >> j) || i < 0 || j < 0 || i >= species_count ||
          j >= species_count)
        throw std::runtime_error("invalid ordered species pair");
      const int pair = i * species_count + j;
      if (seen[pair]) throw std::runtime_error("duplicate ordered species pair");
      seen[pair] = true;
      if (!(input >> token) || token != "radial_basis_type")
        throw std::runtime_error("pair block is missing radial_basis_type");
      ParsedRadialBasis parsed = read_radial_basis(input, scaling);
      bases[pair] = parsed.basis;
      token = parsed.next_token;
    }
  } else {
    if (token != "radial_basis_type")
      throw std::runtime_error("missing radial_basis_type");
    ParsedRadialBasis parsed = read_radial_basis(input, scaling);
    bases[0] = parsed.basis;
    token = parsed.next_token;
  }

  basis_size = bases[0].size;
  max_cutoff = bases[0].max_dist;
  for (const RadialBasis &basis : bases) {
    if (basis.size != basis_size)
      throw std::runtime_error("all ordered-pair radial bases must have the same size");
    max_cutoff = std::max(max_cutoff, basis.max_dist);
  }
  return bases;
}

} // namespace MLIPOpt
#endif
