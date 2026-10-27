// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "pair_mlip_transport_spline.h"

#include "error.h"
#include "memory.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

PairMLIPTransportSpline::PairMLIPTransportSpline(LAMMPS *lmp) :
    PairMLIPTransport(lmp)
{}

void PairMLIPTransportSpline::load_potential()
{
    double ****spline_values = nullptr;
    std::ifstream ifs(potential_filename_);
    std::string rbasis_type;
    if (!ifs.is_open())
    {
        error->all(FLERR, "Cannot open potential file");
    }

    std::string tmpstr;
    char tmpline[1000];

    ifs.getline(tmpline, 1000);
    int len = (int)((std::string)tmpline).length();
    if (tmpline[len - 1] == '\r') // Ensures compatibility between Linux and Windows line endings
        tmpline[len - 1] = '\0';

    if ((std::string)tmpline != "MTP")
    {
        error->all(FLERR, "Can only read MTP format potentials");
    }

    // version reading block
    ifs.getline(tmpline, 1000);
    len = (int)((std::string)tmpline).length();
    if (tmpline[len - 1] == '\r') // Ensures compatibility between Linux and Windows line endings
        tmpline[len - 1] = '\0';
    if ((std::string)tmpline != "version = 1.1.0")
        error->all(FLERR, "MTP file must have version \"1.1.0\"");

    // name/description reading block
    ifs >> tmpstr;
    if (tmpstr == "potential_name") // optional
    {
        ifs.ignore(2);
        ifs >> tmpstr;
        ifs >> tmpstr;
    }

    if (tmpstr == "scaling") // optional
    {
        ifs.ignore(2);
        ifs >> scaling;
        ifs >> tmpstr;
    }

    if (tmpstr != "species_count")
        error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2);
    ifs >> species_count;

    //@ADD@
    ifs >> tmpstr;
    if (tmpstr == "potential_tag")
    {
        getline(ifs, tmpstr);
        ifs >> tmpstr;
    }

    if (tmpstr != "radial_basis_type")
        error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2);
    ifs >> rbasis_type;

    if (rbasis_type == "RBSpline")
    {
        // mindist
        ifs >> tmpstr;
        if (tmpstr != "min_dist")
        {
            error->all(FLERR, "Error reading mtp file");
        }
        ifs.ignore(2);
        ifs >> min_dist;
        if (ifs.fail())
        {
            error->all(FLERR, "Error reading mtp file");
        }
        // maxdist
        ifs >> tmpstr;
        if (tmpstr != "max_dist")
        {
            error->all(FLERR, "Error reading mtp file");
        }
        ifs.ignore(2);
        ifs >> max_dist;
        if (ifs.fail())
        {
            error->all(FLERR, "Error reading mtp file");
        }
        cutoff = max_dist;
    }
    else
    {
        error->all(FLERR, "Currently only RBSpline is supported");
    }

    ifs >> tmpstr;

    // TODO: read on rank 0, calculate spline coeffs, then bcast to other cores
    if (tmpstr == "radial_splines")
    {

        ifs >> spline_points_ >> spline_delta_ >> radial_func_count;
        if (spline_points_ < 2 || spline_delta_ <= 0.0)
            error->all(FLERR, "Invalid radial spline grid in MTP file");
        // double discard;
        // for (int t = 0; t < spline_points_; ++t) ifs >> discard;
        memory->create(spline_radii_, spline_points_, "pair:spline_radii_");
        for (int t = 0; t < spline_points_; ++t) {
            ifs >> spline_radii_[t];

        }
        memory->create(spline_values, species_count, species_count, radial_func_count, spline_points_, "pair:spline_values");
        // typei typej mu spline_points_ a b c d 2c 3d
        memory->create(spline_coefficients_,
                       species_count * species_count * radial_func_count * spline_points_ * 6,
                       "pair:spline_coefficients_");
        for (int mu = 0; mu < radial_func_count; ++mu) {
            for (int s1 = 0; s1 < species_count; ++s1) {
                for (int s2 = 0; s2 < species_count; ++s2) {
                    int itp, jtp, muc;
                    ifs >> muc >> itp >> jtp;
                    for (int i = 0; i < spline_points_; ++i) {
                        ifs >> spline_values[itp][jtp][muc][i];

                    }
                }
            }
        }
        // build splines
        for (int mu = 0; mu < radial_func_count; ++mu) {
            for (int s1 = 0; s1 < species_count; ++s1) {
                for (int s2 = 0; s2 < species_count; ++s2) {
                    const int offset = ((s1 * species_count + s2) * radial_func_count + mu) * spline_points_ * 6;
                    interpolate(spline_points_, spline_delta_, spline_values[s1][s2][mu], spline_coefficients_ + offset);
                }
            }
        }

        ifs >> tmpstr;

        memory->destroy(spline_values);
    }
    else
    {
        error->all(FLERR, "error reading mtp file, splines not found");
    }

    if (tmpstr != "alpha_moments_count") {
        error->all(FLERR, "Error reading .mtp file");
    }
    ifs.ignore(2);
    ifs >> alpha_moments_count;
    if (ifs.fail())
        error->all(FLERR, "Error reading .mtp file");

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_basic_count")
        error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2);
    ifs >> alpha_index_basic_count;
    if (ifs.fail())
        error->all(FLERR, "Error reading .mtp file");

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_basic")
        error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(4);

    memory->create(alpha_index_basic, alpha_index_basic_count, "pair::alpha_index_basic");
    memory->create(nu_arr_list, alpha_index_basic_count, "pair::nu_arr_list");
    if (alpha_index_basic == nullptr)
        error->all(FLERR, "Memory allocation error");

    int radial_func_max = -1;
    for (int i = 0; i < alpha_index_basic_count; i++)
    {
        char tmpch;
        ifs.ignore(1000, '{');
        ifs >> alpha_index_basic[i].mu >> tmpch >> alpha_index_basic[i].b1 >> tmpch >> alpha_index_basic[i].b2 >> tmpch >> alpha_index_basic[i].b3;
        nu_arr_list[i] = alpha_index_basic[i].b1 + alpha_index_basic[i].b2 + alpha_index_basic[i].b3;
        if (ifs.fail())
            error->all(FLERR, "Error reading .mtp file");

        if (alpha_index_basic[i].mu > radial_func_max)
            radial_func_max = alpha_index_basic[i].mu;
    }


    if (radial_func_max != radial_func_count - 1)
        error->all(FLERR, "Wrong number of radial functions specified");

    ifs.ignore(1000, '\n');

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_times_count")
        error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2);
    ifs >> alpha_index_times_count;
    if (ifs.fail())
        error->all(FLERR, "Error reading .mtp file");

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_times")
        error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(4);

    if (alpha_index_times_count > 0)
        memory->create(alpha_index_times, alpha_index_times_count, "pair::alpha_index_times");

    for (int i = 0; i < alpha_index_times_count; i++)
    {
        char tmpch;
        ifs.ignore(1000, '{');
        ifs >> alpha_index_times[i].l1 >> tmpch >> alpha_index_times[i].l2 >> tmpch >> alpha_index_times[i].c >> tmpch >> alpha_index_times[i].id;
        if (ifs.fail())
            error->all(FLERR, "Error reading .mtp file");
    }

    ifs.ignore(1000, '\n');

    ifs >> tmpstr;
    if (tmpstr != "alpha_scalar_moments")
        error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2);
    ifs >> alpha_scalar_moments;
    if (alpha_index_times_count < 0)
        error->all(FLERR, "Error reading .mtp file");

    memory->create(alpha_moment_mapping, alpha_scalar_moments, "pair::alpha_moment_mapping");
    if (alpha_moment_mapping == nullptr)
        error->all(FLERR, "Memory allocation error");

    ifs >> tmpstr;
    if (tmpstr != "alpha_moment_mapping")
        error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(4);
    for (int i = 0; i < alpha_scalar_moments; i++)
    {
        char tmpch = ' ';
        ifs >> alpha_moment_mapping[i] >> tmpch;
        if (ifs.fail())
            error->all(FLERR, "Error reading .mtp file");
    }

    std::vector<char> moment_initialized(alpha_moments_count, 0);
    std::vector<char> moment_needs_zero(alpha_moments_count, 0);
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        moment_initialized[i] = 1;
        moment_needs_zero[i] = 1;
    }
    std::vector<AlphaTimesGroup> times_groups;
    for (int begin = 0; begin < alpha_index_times_count; ) {
        const int id = alpha_index_times[begin].id;
        int end = begin + 1;
        while (end < alpha_index_times_count && alpha_index_times[end].id == id)
            ++end;

        bool reads_output = false;
        for (int i = begin; i < end; ++i) {
            const int l1 = alpha_index_times[i].l1;
            const int l2 = alpha_index_times[i].l2;
            reads_output = reads_output || l1 == id || l2 == id;
            if (!moment_initialized[l1]) {
                moment_initialized[l1] = 1;
                moment_needs_zero[l1] = 1;
            }
            if (!moment_initialized[l2]) {
                moment_initialized[l2] = 1;
                moment_needs_zero[l2] = 1;
            }
        }

        const bool assign = !moment_initialized[id] && !reads_output;
        if (!assign && !moment_initialized[id])
            moment_needs_zero[id] = 1;
        moment_initialized[id] = 1;
        times_groups.push_back({begin, end, id, assign});
        begin = end;
    }
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        const int id = alpha_moment_mapping[i];
        if (!moment_initialized[id]) {
            moment_initialized[id] = 1;
            moment_needs_zero[id] = 1;
        }
    }

    alpha_times_group_count = static_cast<int>(times_groups.size());
    memory->create(alpha_times_groups, alpha_times_group_count, "pair::alpha_times_groups");
    for (int i = 0; i < alpha_times_group_count; ++i)
        alpha_times_groups[i] = times_groups[i];

    for (int i = 0; i < alpha_moments_count; ++i)
        moment_zero_count += moment_needs_zero[i] != 0;
    memory->create(moment_zero_ids, moment_zero_count, "pair::moment_zero_ids");
    for (int i = 0, j = 0; i < alpha_moments_count; ++i)
        if (moment_needs_zero[i]) moment_zero_ids[j++] = i;

    ifs.ignore(1000, '\n');
    // Reading linear coeffs
    char foo;
    ifs >> tmpstr;
    if (tmpstr != "species_coeffs")
    {
        error->all(FLERR, "species_coeffs not found");
    }
    else
    {
        ifs.ignore(4);

        memory->create(specie_coeffs, species_count, "pair::specie_coeffs");
        for (int i = 0; i < species_count; i++)
            ifs >> specie_coeffs[i] >> foo;

        ifs >> tmpstr;

        if (tmpstr != "moment_coeffs")
            error->all(FLERR, "Cannot read linear coeffs");

        ifs.ignore(2);

        memory->create(moment_coeffs, alpha_scalar_moments, "pair::moment_coeffs");

        ifs.ignore(10, '{');

        for (int i = 0; i < alpha_scalar_moments; i++)
            ifs >> moment_coeffs[i] >> foo;
    }
    ifs.close();

    for (int i = 0; i < alpha_index_basic_count; i++)
    {
        max_alpha_index_basic =
            std::max(max_alpha_index_basic,
                     alpha_index_basic[i].b1 + alpha_index_basic[i].b2 + alpha_index_basic[i].b3);
        max_mu = std::max(max_mu, alpha_index_basic[i].mu);
    }
    max_alpha_index_basic++;
    max_mu++;

  scratch_.allocate(max_mu, alpha_moments_count, rb_size,
                    max_alpha_index_basic, alpha_index_basic_count, memory);
  potential_loaded_ = true;
}

PairMLIPTransportSpline::~PairMLIPTransportSpline()
{
  if (!copymode) {
    memory->destroy(spline_coefficients_);
    memory->destroy(spline_radii_);
  }
}

void PairMLIPTransportSpline::interpolate(int n, double delta,
                                          const double *values, double *coeff)
{
  for (int i = 0; i < n; ++i) coeff[i * 6] = values[i];
  coeff[1] = 0.0;
  coeff[(n - 1) * 6 + 1] = 0.0;

  std::vector<double> rhs(n), lower(n), diagonal(n), upper(n);
  rhs[0] = 3.0 * (values[1] - values[0]) / (delta * delta);
  for (int i = 1; i < n - 1; ++i)
    rhs[i] = 3.0 * (values[i - 1] + values[i + 1] - 2.0 * values[i]) /
             (delta * delta);
  rhs[n - 1] = -3.0 * (values[n - 1] - values[n - 2]) / (delta * delta);

  diagonal[0] = diagonal[n - 1] = 2.0;
  upper[0] = 1.0;
  lower[n - 1] = 1.0;
  for (int i = 1; i < n - 1; ++i) {
    lower[i] = 1.0;
    diagonal[i] = 4.0;
    upper[i] = 1.0;
  }
  for (int i = 1; i < n; ++i) {
    const double factor = lower[i] / diagonal[i - 1];
    diagonal[i] -= factor * upper[i - 1];
    rhs[i] -= factor * rhs[i - 1];
  }
  coeff[(n - 1) * 6 + 2] = rhs[n - 1] / diagonal[n - 1];
  for (int i = n - 2; i >= 0; --i)
    coeff[i * 6 + 2] =
        (rhs[i] - upper[i] * coeff[(i + 1) * 6 + 2]) / diagonal[i];

  for (int i = 0; i < n - 1; ++i) {
    const double c0 = coeff[i * 6 + 2];
    const double c1 = coeff[(i + 1) * 6 + 2];
    coeff[i * 6 + 1] =
        (values[i + 1] - values[i]) / delta - c0 * delta -
        (c1 - c0) * delta / 3.0;
    coeff[i * 6 + 3] = (c1 - c0) / (3.0 * delta);
    coeff[i * 6 + 4] = 2.0 * c0;
    coeff[i * 6 + 5] = 3.0 * coeff[i * 6 + 3];
  }
}

bool PairMLIPTransportSpline::evaluate_radial(int pair, double distance,
                                               Scratch &scratch) const
{
  if (distance < min_dist || distance >= max_dist) return false;
  int interval = static_cast<int>(std::floor((distance - min_dist) /
                                             spline_delta_));
  interval = std::max(0, std::min(interval, spline_points_ - 2));
  double offset = distance - spline_radii_[interval];
  if (offset < 0.0) offset = 0.0;

  const int pair_offset = pair * radial_func_count * spline_points_ * 6;
  for (int mu = 0; mu < max_mu; ++mu) {
    const double *c = spline_coefficients_ + pair_offset +
                      (mu * spline_points_ + interval) * 6;
    scratch.val_mu[mu] = c[0] + offset *
                         (c[1] + offset * (c[2] + offset * c[3]));
    scratch.der_mu[mu] = c[1] + offset * (c[4] + offset * c[5]);
  }
  return true;
}
