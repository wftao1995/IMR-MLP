// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#ifndef LMP_MLIP_TRANSPORT_RADIAL_BASIS_H
#define LMP_MLIP_TRANSPORT_RADIAL_BASIS_H

#include <algorithm>
#include <cmath>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace MLIPTransport {

struct RadialBasis {
  enum Type {
    CHEBYSHEV,
    CHEBY_COS,
    CHEBYSHEV_I2,
    CHEBYSHEV_POLY2,
    CHEBYSHEV_POLY3,
    CHEBYSHEV_POLY4,
    CHEBYSHEV_POLY5,
    CHEBYSHEV_CONFIGURABLE
  } type = CHEBYSHEV;

  enum Cutoff { QUADRATIC, COSINE, POLY5 } cutoff = QUADRATIC;
  enum Shift { LINEAR, EXPONENTIAL } shift = LINEAR;

  int size = 0;
  double min_dist = 0.0;
  double max_dist = 0.0;
  double scaling = 1.0;
  double parameter = -1.5;
  double shift_lambda = 0.0;
  std::string name;

  bool excludes(double r) const
  {
    // Historical RBChebyshev and RBChebyCos potentials are allowed below
    // min_dist.  Interval-defined bases match basis/AnyBasis and are zero.
    return r >= max_dist ||
      ((type == CHEBYSHEV_CONFIGURABLE || type == CHEBYSHEV_POLY2 ||
        type == CHEBYSHEV_POLY3 || type == CHEBYSHEV_POLY4 ||
        type == CHEBYSHEV_POLY5) && r < min_dist);
  }

  void evaluate(double r, double *values, double *derivatives) const
  {
    if (size <= 0) return;
    if (excludes(r)) {
      std::fill(values, values + size, 0.0);
      std::fill(derivatives, derivatives + size, 0.0);
      return;
    }

    const double range = max_dist - min_dist;
    const double inv_range = 1.0 / range;
    double x = (2.0 * r - min_dist - max_dist) * inv_range;
    double dx = 2.0 * inv_range;
    double fc = 0.0;
    double dfc = 0.0;

    if (type == CHEBYSHEV_I2) {
      const double q = (r - max_dist) * inv_range;
      x = 2.0 * q * q - 1.0;
      dx = 4.0 * q * inv_range;
      fc = q * q;
      dfc = 2.0 * q * inv_range;
    } else if (type == CHEBY_COS) {
      // Preserve the established RBChebyCos definition, whose phase is
      // referenced to zero rather than min_dist.
      const double frequency = M_PI / max_dist;
      fc = 0.5 * (1.0 + std::cos(r * frequency));
      dfc = -0.5 * frequency * std::sin(r * frequency);
    } else if (type == CHEBYSHEV_CONFIGURABLE) {
      if (shift == EXPONENTIAL && shift_lambda != 0.0) {
        const double denominator = std::expm1(shift_lambda);
        const double exponent = shift_lambda * (max_dist - r) * inv_range;
        x = 1.0 - 2.0 * std::expm1(exponent) / denominator;
        dx = 2.0 * shift_lambda * inv_range * std::exp(exponent) /
          denominator;
      }
      configurable_cutoff(r, inv_range, fc, dfc);
    } else if (type == CHEBYSHEV_POLY2) {
      polynomial_cutoff(r, inv_range, 2, 0.0, fc, dfc);
    } else if (type == CHEBYSHEV_POLY3) {
      polynomial_cutoff(r, inv_range, 2, parameter, fc, dfc);
    } else if (type == CHEBYSHEV_POLY4) {
      polynomial_cutoff(r, inv_range, 3, parameter, fc, dfc);
    } else if (type == CHEBYSHEV_POLY5) {
      polynomial_cutoff(r, inv_range, 4, parameter, fc, dfc);
    } else {
      const double delta = r - max_dist;
      fc = delta * delta;
      dfc = 2.0 * delta;
    }

    values[0] = scaling * fc;
    derivatives[0] = scaling * dfc;
    if (size == 1) return;
    values[1] = scaling * x * fc;
    derivatives[1] = scaling * (dx * fc + x * dfc);
    for (int i = 2; i < size; ++i) {
      values[i] = 2.0 * x * values[i - 1] - values[i - 2];
      derivatives[i] = 2.0 * (dx * values[i - 1] +
                              x * derivatives[i - 1]) - derivatives[i - 2];
    }
  }

 private:
  void polynomial_cutoff(double r, double inv_range, int power, double b,
                         double &fc, double &dfc) const
  {
    const double t = (r - min_dist) * inv_range;
    const double v = 1.0 - t;
    if (type == CHEBYSHEV_POLY2) {
      fc = v * v;
      dfc = -2.0 * v * inv_range;
      return;
    }
    // (1-t)^power * (1 + (power+b)t)
    const double vp = std::pow(v, power);
    fc = vp * (1.0 + (power + b) * t);
    dfc = std::pow(v, power - 1) *
      (b * (1.0 - (power + 1.0) * t) -
       power * (power + 1.0) * t) * inv_range;
  }

  void configurable_cutoff(double r, double inv_range, double &fc,
                           double &dfc) const
  {
    if (cutoff == COSINE) {
      const double phase = (r - min_dist) * M_PI * inv_range;
      fc = 0.5 * (1.0 + std::cos(phase));
      dfc = -0.5 * M_PI * inv_range * std::sin(phase);
    } else if (cutoff == POLY5) {
      const double t = (r - min_dist) * inv_range;
      const double v = 1.0 - t;
      const double v2 = v * v;
      const double v3 = v2 * v;
      fc = v2 * v2 * (1.0 + (4.0 + parameter) * t);
      dfc = -v3 * (20.0 * t + parameter * (5.0 * t - 1.0)) *
        inv_range;
    } else {
      const double delta = r - max_dist;
      fc = delta * delta;
      dfc = 2.0 * delta;
    }
  }
};

inline void read_assignment_value(std::istream &input, const char *key,
                                  std::string &value)
{
  std::string equals;
  if (!(input >> equals >> value) || equals != "=")
    throw std::runtime_error(std::string("expected '") + key + " = ...'");
}

template <class T>
inline void read_assignment(std::istream &input, const char *key, T &value)
{
  std::string actual, equals;
  if (!(input >> actual >> equals >> value) || actual != key || equals != "=")
    throw std::runtime_error(std::string("expected '") + key + " = ...'");
}

template <class T>
inline void read_assignment_value(std::istream &input, const char *key,
                                  T &value)
{
  std::string equals;
  if (!(input >> equals >> value) || equals != "=")
    throw std::runtime_error(std::string("expected '") + key + " = ...'");
}

struct ParsedRadialBasis {
  RadialBasis basis;
  std::string next_token;
};

inline ParsedRadialBasis read_radial_basis(std::istream &input, double scaling)
{
  ParsedRadialBasis parsed;
  RadialBasis &basis = parsed.basis;
  basis.scaling = scaling;
  read_assignment_value(input, "radial_basis_type", basis.name);

  if (basis.name == "RBChebyshev") basis.type = RadialBasis::CHEBYSHEV;
  else if (basis.name == "RBChebyCos") basis.type = RadialBasis::CHEBY_COS;
  else if (basis.name == "RBChebyshevI2") basis.type = RadialBasis::CHEBYSHEV_I2;
  else if (basis.name == "RBChebyshevPoly2") basis.type = RadialBasis::CHEBYSHEV_POLY2;
  else if (basis.name == "RBChebyshevPoly3") basis.type = RadialBasis::CHEBYSHEV_POLY3;
  else if (basis.name == "RBChebyshevPoly4") basis.type = RadialBasis::CHEBYSHEV_POLY4;
  else if (basis.name == "RBChebyshevPoly5") basis.type = RadialBasis::CHEBYSHEV_POLY5;
  else if (basis.name == "RBChebyshevConfigurable")
    basis.type = RadialBasis::CHEBYSHEV_CONFIGURABLE;
  else
    throw std::runtime_error("unsupported radial_basis_type '" + basis.name +
                             "'");

  read_assignment(input, "min_dist", basis.min_dist);
  read_assignment(input, "max_dist", basis.max_dist);
  read_assignment(input, "radial_basis_size", basis.size);
  if (!std::isfinite(basis.min_dist) || !std::isfinite(basis.max_dist) ||
      !(basis.max_dist > basis.min_dist) || basis.size <= 0)
    throw std::runtime_error("invalid distance interval or radial_basis_size");
  if (!(input >> parsed.next_token))
    throw std::runtime_error("unexpected end of radial basis block");

  if (basis.type == RadialBasis::CHEBYSHEV_POLY3 ||
      basis.type == RadialBasis::CHEBYSHEV_POLY4 ||
      basis.type == RadialBasis::CHEBYSHEV_POLY5) {
    if (parsed.next_token == "rb_b") {
      read_assignment_value(input, "rb_b", basis.parameter);
      if (!(input >> parsed.next_token))
        throw std::runtime_error("unexpected end after rb_b");
    } else if (basis.type == RadialBasis::CHEBYSHEV_POLY4 ||
               basis.type == RadialBasis::CHEBYSHEV_POLY5) {
      throw std::runtime_error(basis.name + " is missing rb_b");
    }
  }

  if (basis.type == RadialBasis::CHEBYSHEV_CONFIGURABLE) {
    if (parsed.next_token != "cutoff_type")
      throw std::runtime_error("configurable basis is missing cutoff_type");
    std::string cutoff;
    read_assignment_value(input, "cutoff_type", cutoff);
    if (cutoff == "quadratic") basis.cutoff = RadialBasis::QUADRATIC;
    else if (cutoff == "cosine") basis.cutoff = RadialBasis::COSINE;
    else if (cutoff == "poly5") basis.cutoff = RadialBasis::POLY5;
    else throw std::runtime_error("unknown cutoff_type '" + cutoff + "'");
    if (!(input >> parsed.next_token))
      throw std::runtime_error("unexpected end after cutoff_type");
    if (basis.cutoff == RadialBasis::POLY5) {
      if (parsed.next_token != "cutoff_b")
        throw std::runtime_error("poly5 cutoff is missing cutoff_b");
      read_assignment_value(input, "cutoff_b", basis.parameter);
      if (!(input >> parsed.next_token))
        throw std::runtime_error("unexpected end after cutoff_b");
    }
    if (parsed.next_token != "shift_type")
      throw std::runtime_error("configurable basis is missing shift_type");
    std::string shift;
    read_assignment_value(input, "shift_type", shift);
    if (shift == "linear") basis.shift = RadialBasis::LINEAR;
    else if (shift == "exponential") basis.shift = RadialBasis::EXPONENTIAL;
    else throw std::runtime_error("unknown shift_type '" + shift + "'");
    if (!(input >> parsed.next_token))
      throw std::runtime_error("unexpected end after shift_type");
    if (basis.shift == RadialBasis::EXPONENTIAL) {
      if (parsed.next_token != "shift_lambda")
        throw std::runtime_error("exponential shift is missing shift_lambda");
      read_assignment_value(input, "shift_lambda", basis.shift_lambda);
      if (!(input >> parsed.next_token))
        throw std::runtime_error("unexpected end after shift_lambda");
    }
  }
  return parsed;
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

} // namespace MLIPTransport

#endif
