// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "pair_mlip_transport.h"

#include "mlip_transport_tally.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

void PairMLIPTransport::Scratch::allocate(int max_mu, int moments, int rb_size,
                                          int max_power, int basic_count,
                                          Memory *memory)
{
  memory->create(val_mu, max_mu, "mlip/transport:val_mu");
  memory->create(der_mu, max_mu, "mlip/transport:der_mu");
  memory->create(moment_vals, moments, "mlip/transport:moment_vals");
  memory->create(site_energy_derivatives, moments,
                 "mlip/transport:site_energy_derivatives");
  if (rb_size > 0) {
    memory->create(rb_vals, rb_size, "mlip/transport:rb_vals");
    memory->create(rb_ders, rb_size, "mlip/transport:rb_ders");
  }
  memory->create(dist_powers, max_power, "mlip/transport:dist_powers");
  memory->create(coord_powers, max_power + 1, "mlip/transport:coord_powers");
  dist_powers[0] = 1.0;
  coord_powers[0] = {0.0, 0.0, 0.0};
  coord_powers[1] = {1.0, 1.0, 1.0};
  grow_neighbors(64, basic_count, memory);
}

void PairMLIPTransport::Scratch::grow_neighbors(int requested, int basic_count,
                                                Memory *memory)
{
  if (requested <= neighbor_capacity) return;
  int capacity = neighbor_capacity ? neighbor_capacity + neighbor_capacity / 2 : 64;
  if (capacity < requested) capacity = requested;
  memory->grow(moment_jacobian, basic_count * capacity * 3,
               "mlip/transport:moment_jacobian");
  memory->grow(jlist_true, capacity, "mlip/transport:jlist_true");
  neighbor_capacity = capacity;
}

void PairMLIPTransport::Scratch::reset_transport(int nall, bool flux, int channels)
{
  if (flux) heat_virial.assign(static_cast<size_t>(nall) * 9, 0.0);
  else heat_virial.clear();
  interface_force.resize(channels);
  for (auto &force : interface_force)
    force.assign(static_cast<size_t>(nall) * 3, 0.0);
}

void PairMLIPTransport::Scratch::destroy(Memory *memory)
{
  memory->destroy(moment_vals);
  memory->destroy(site_energy_derivatives);
  memory->destroy(dist_powers);
  memory->destroy(coord_powers);
  memory->destroy(moment_jacobian);
  memory->destroy(rb_vals);
  memory->destroy(rb_ders);
  memory->destroy(val_mu);
  memory->destroy(der_mu);
  memory->destroy(jlist_true);
  neighbor_capacity = 0;
  heat_virial.clear();
  interface_force.clear();
}

PairMLIPTransport::PairMLIPTransport(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;
  single_enable = 0;
  centroidstressflag = CENTROID_NOTAVAIL;
  allocated = 0;
  for (int &mapping : spmap) mapping = -1;
}

PairMLIPTransport::~PairMLIPTransport()
{
  if (!copymode) deallocate();
}

void PairMLIPTransport::settings(int narg, char **arg)
{
    try {
        if (metadata_arguments_.defer(narg, arg, false)) return;
    } catch (const std::exception &e) { error->all(FLERR, e.what()); }
  if (narg < 1) error->all(FLERR, "Illegal pair_style mlip/transport command");
  if (std::strlen(arg[0]) >= sizeof(potential_filename_))
    error->all(FLERR, "MTP potential file name is too long");
  if (narg - 1 > MAX_SPECIES)
    error->all(FLERR, "Pair style mlip/transport supports at most {} atom types",
               MAX_SPECIES);

    try {
        metadata_arguments_.load(narg, arg, narg, spmap, sizeof(spmap)/sizeof(spmap[0]), false);
    } catch (const std::exception &e) { error->all(FLERR, e.what()); }
  std::strcpy(potential_filename_, arg[0]);

  if (potential_loaded_)
    error->all(FLERR, "Pair style mlip/transport potential cannot be reset");
  load_potential();
    metadata_arguments_.loaded = true;
}

void PairMLIPTransport::coeff(int narg, char **arg)
{
    try {
        auto arguments = metadata_arguments_.coefficients(narg, arg, atom->ntypes);
        if (!arguments.empty()) {
            std::vector<char *> pointers;
            for (auto &value : arguments) pointers.push_back(value.data());
            settings(static_cast<int>(pointers.size()), pointers.data());
        }
        metadata_arguments_.validate(atom->ntypes);
    } catch (const std::exception &e) { error->all(FLERR, e.what()); }
  if (!allocated) allocate();
}

void PairMLIPTransport::init_style()
{
    try { metadata_arguments_.validate(atom->ntypes); }
    catch (const std::exception &e) { error->all(FLERR, e.what()); }
  if (!potential_loaded_)
    error->all(FLERR, "Pair style mlip/transport has no loaded potential");
  if (!force->newton_pair)
    error->all(FLERR, "Pair style mlip/transport requires newton pair on");
  if (atom->ntypes > MAX_SPECIES)
    error->all(FLERR, "Pair style mlip/transport supports at most {} atom types",
               MAX_SPECIES);
  for (int type = 1; type <= atom->ntypes; ++type)
    if (spmap[type] < 0 || spmap[type] >= species_count)
      error->all(FLERR, "Invalid MTP species mapping for LAMMPS atom type {}", type);

  if (!allocated) allocate();
  cutoffsq = cutoff * cutoff;
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = 1; j <= atom->ntypes; ++j)
      cutsq[i][j] = cutoffsq;

  configure_transport_requests();
  neighbor->add_request(this, NeighConst::REQ_FULL);
  inited_ = true;
}

void PairMLIPTransport::load_potential()
{
    std::ifstream ifs(potential_filename_);
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

    try {
        radial_bases = MLIPTransport::read_radial_bases(
            ifs, tmpstr, species_count, scaling, radial_basis_split,
            rb_size, cutoff);
    } catch (const std::exception &exception) {
        const std::string message =
            std::string("Error reading MTP radial basis: ") + exception.what();
        error->all(FLERR, message.c_str());
    }
    min_dist = radial_bases.front().min_dist;
    max_dist = cutoff;


    if (tmpstr != "radial_funcs_count")
        error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2);
    ifs >> radial_func_count;

    // Radial coeffs initialization
    int pairs_count = species_count * species_count; // number of species pairs

    char foo = ' ';

    int radial_size = pairs_count * radial_func_count * rb_size;
    memory->create(radial_coeffs, radial_size, "pair::radial_coeffs");

    ifs >> tmpstr;
    if (tmpstr == "radial_coeffs")
    {

        for (int s1 = 0; s1 < species_count; s1++)
            for (int s2 = 0; s2 < species_count; s2++)
            {
                ifs >> foo >> foo >> foo;

                double t;

                for (int i = 0; i < radial_func_count; i++)
                {
                    ifs >> foo;
                    for (int j = 0; j < rb_size; j++)
                    {
                        ifs >> t >> foo;
                        radial_coeffs[(s1 * species_count + s2) * radial_func_count * rb_size +
                                              i * rb_size + j] = t;
                    }
                }
            }

        ifs >> tmpstr;
    }
    else
    {
        error->all(FLERR, "error reading mtp file, parameters not found");
    }

    if (tmpstr != "alpha_moments_count")
        error->all(FLERR, "Error reading .mtp file");
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

    // Group consecutive contractions that produce the same moment.  The first
    // production of a non-basic moment can be assigned instead of repeatedly
    // updating a zero-initialized array entry.  If a potential has an unusual
    // forward reference or self-reference, keep its original add-to-zero
    // semantics by adding that moment to moment_zero_ids.
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

void PairMLIPTransport::allocate()
{
  const int n = atom->ntypes;
  memory->create(cutsq, n + 1, n + 1, "mlip/transport:cutsq");
  memory->create(setflag, n + 1, n + 1, "mlip/transport:setflag");
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) setflag[i][j] = 1;
  allocated = 1;
}

void PairMLIPTransport::deallocate()
{
  scratch_.destroy(memory);
  memory->destroy(cutsq);
  memory->destroy(setflag);
  memory->destroy(alpha_moment_mapping);
  memory->destroy(alpha_index_times);
  memory->destroy(alpha_times_groups);
  memory->destroy(moment_zero_ids);
  memory->destroy(alpha_index_basic);
  memory->destroy(radial_coeffs);
  memory->destroy(specie_coeffs);
  memory->destroy(moment_coeffs);
  memory->destroy(nu_arr_list);
  radial_bases.clear();
  site_energy_.clear();
  heat_virial_.clear();
  interfaces_.clear();
  allocated = 0;
  potential_loaded_ = false;
}

void PairMLIPTransport::configure_transport_requests()
{
  flux_enabled_ = false;
  interfaces_.clear();
  for (Compute *compute : modify->get_compute_list()) {
    if (dynamic_cast<MLIPTransportFluxRequest *>(compute)) flux_enabled_ = true;
    auto *request = dynamic_cast<MLIPTransportCurrentRequest *>(compute);
    if (request) {
      const int channel = static_cast<int>(interfaces_.size());
      interfaces_.push_back(
          {request->transport_left_groupbit(), request->transport_right_groupbit(), {}});
      request->set_transport_channel(channel);
    }
  }
  comm_reverse = (flux_enabled_ ? 9 : 0) + 3 * static_cast<int>(interfaces_.size());
  transport_step_ = reverse_step_ = -1;
}

void PairMLIPTransport::prepare_transport_storage(int nall)
{
  transport_nall_ = nall;
  if (flux_enabled_) {
    site_energy_.assign(nall, 0.0);
    heat_virial_.assign(static_cast<size_t>(nall) * 9, 0.0);
  } else {
    site_energy_.clear();
    heat_virial_.clear();
  }
  for (auto &channel : interfaces_)
    channel.force.assign(static_cast<size_t>(nall) * 3, 0.0);
}

void PairMLIPTransport::reset_transport_storage(int nall)
{
  prepare_transport_storage(nall);
  scratch_.reset_transport(nall, flux_enabled_,
                           static_cast<int>(interfaces_.size()));
}

bool PairMLIPTransport::evaluate_radial(int pair, double distance, Scratch &scratch) const
{
  const MLIPTransport::RadialBasis &basis =
      radial_basis_split ? radial_bases[pair] : radial_bases[0];
  if (basis.excludes(distance)) return false;
  basis.evaluate(distance, scratch.rb_vals, scratch.rb_ders);
  for (int mu = 0; mu < max_mu; ++mu) {
    double value = 0.0;
    double derivative = 0.0;
    const int offset = pair * radial_func_count * rb_size + mu * rb_size;
    for (int xi = 0; xi < rb_size; ++xi) {
      const double coefficient = radial_coeffs[offset + xi];
      value += coefficient * scratch.rb_vals[xi];
      derivative += coefficient * scratch.rb_ders[xi];
    }
    scratch.val_mu[mu] = value;
    scratch.der_mu[mu] = derivative;
  }
  return true;
}

void PairMLIPTransport::tally_standard_vatom(double **target, int i, int j,
                                              const double *r, const double *d) const
{
  if (!target) return;
  double w[6];
  MLIPTransport::standard_virial(r, d, w);
  for (int k = 0; k < 6; ++k) {
    target[i][k] += 0.5 * w[k];
    target[j][k] += 0.5 * w[k];
  }
}

void PairMLIPTransport::tally_transport(Scratch &scratch, int i, int j,
                                        const double *r, const double *d)
{
  if (flux_enabled_) {
    double *b = scratch.heat_virial.data() + static_cast<size_t>(j) * 9;
    MLIPTransport::add_heat_virial(r, d, b);
  }

  const int *mask = atom->mask;
  for (size_t channel = 0; channel < interfaces_.size(); ++channel) {
    const int hi = (mask[i] & interfaces_[channel].right_bit) ? 1 : 0;
    const int hj = (mask[j] & interfaces_[channel].right_bit) ? 1 : 0;
    if (hi == hj) continue;
    double *g = scratch.interface_force[channel].data() + static_cast<size_t>(j) * 3;
    MLIPTransport::add_interface_force(hi, hj, d, g);
  }
}

void PairMLIPTransport::compute_range(int begin, int end, Scratch &scratch,
                                      double **force_array, double *site_eatom,
                                      double **site_vatom, double &energy)
{
  double **x = atom->x;
  int *type = atom->type;
  const int basic_count = alpha_index_basic_count;
  const bool need_site_energy = eflag_global || eflag_atom || flux_enabled_;

  for (int ii = begin; ii < end; ++ii) {
    const int ati = list->ilist[ii];
    const int jnum = list->numneigh[ati];
    const int type_central = spmap[type[ati]];
    const double *ivec = x[ati];
    scratch.grow_neighbors(jnum, basic_count, memory);

    for (int k = 0; k < moment_zero_count; ++k)
      scratch.moment_vals[moment_zero_ids[k]] = 0.0;

    int jnum_true = 0;
    int *neighbors = list->firstneigh[ati];
    for (int jj = 0; jj < jnum; ++jj) {
      const int atj = neighbors[jj] & NEIGHMASK;
      const double r[3] = {x[atj][0] - ivec[0], x[atj][1] - ivec[1],
                           x[atj][2] - ivec[2]};
      const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      if (rsq <= 0.0 || rsq >= cutoffsq) continue;
      const double distance = std::sqrt(rsq);
      const int pair = type_central * species_count + spmap[type[atj]];
      if (!evaluate_radial(pair, distance, scratch)) continue;

      const double rinv = 1.0 / distance;
      scratch.jlist_true[jnum_true] = atj;
      for (int k = 1; k < max_alpha_index_basic; ++k) {
        scratch.dist_powers[k] = scratch.dist_powers[k - 1] * rinv;
        scratch.coord_powers[k + 1] = {
            scratch.coord_powers[k].x * r[0],
            scratch.coord_powers[k].y * r[1],
            scratch.coord_powers[k].z * r[2]};
      }

      for (int k = 0; k < basic_count; ++k) {
        const AlphaBasic &basic = alpha_index_basic[k];
        double value = scratch.val_mu[basic.mu];
        double derivative = scratch.der_mu[basic.mu];
        const int degree = nu_arr_list[k];
        const double rpower = scratch.dist_powers[degree];
        value *= rpower;
        derivative = derivative * rpower - degree * value * rinv;
        const double px = scratch.coord_powers[basic.b1 + 1].x;
        const double py = scratch.coord_powers[basic.b2 + 1].y;
        const double pz = scratch.coord_powers[basic.b3 + 1].z;
        double monomial = px * py * pz;
        scratch.moment_vals[k] += value * monomial;
        monomial *= derivative * rinv;
        scratch.moment_jacobian[(jnum_true * 3 + 0) * basic_count + k] =
            monomial * r[0] + basic.b1 * value *
            scratch.coord_powers[basic.b1].x * py * pz;
        scratch.moment_jacobian[(jnum_true * 3 + 1) * basic_count + k] =
            monomial * r[1] + basic.b2 * value *
            scratch.coord_powers[basic.b2].y * px * pz;
        scratch.moment_jacobian[(jnum_true * 3 + 2) * basic_count + k] =
            monomial * r[2] + basic.b3 * value *
            scratch.coord_powers[basic.b3].z * px * py;
      }
      ++jnum_true;
    }

    for (int group_index = 0; group_index < alpha_times_group_count; ++group_index) {
      const AlphaTimesGroup &group = alpha_times_groups[group_index];
      if (group.assign) {
        const AlphaTimes &first = alpha_index_times[group.begin];
        double value = first.c * scratch.moment_vals[first.l1] *
                       scratch.moment_vals[first.l2];
        for (int k = group.begin + 1; k < group.end; ++k) {
          const AlphaTimes &term = alpha_index_times[k];
          value += term.c * scratch.moment_vals[term.l1] *
                   scratch.moment_vals[term.l2];
        }
        scratch.moment_vals[group.id] = value;
      } else {
        for (int k = group.begin; k < group.end; ++k) {
          const AlphaTimes &term = alpha_index_times[k];
          scratch.moment_vals[group.id] +=
              term.c * scratch.moment_vals[term.l1] *
              scratch.moment_vals[term.l2];
        }
      }
    }

    std::fill(scratch.site_energy_derivatives,
              scratch.site_energy_derivatives + alpha_moments_count, 0.0);
    double site_energy = 0.0;
    if (need_site_energy) {
      site_energy = specie_coeffs[type_central];
      for (int k = 0; k < alpha_scalar_moments; ++k)
        site_energy += moment_coeffs[k] *
                       scratch.moment_vals[alpha_moment_mapping[k]];
    }
    for (int k = 0; k < alpha_scalar_moments; ++k)
      scratch.site_energy_derivatives[alpha_moment_mapping[k]] = moment_coeffs[k];

    for (int k = alpha_index_times_count - 1; k >= 0; --k) {
      const AlphaTimes &term = alpha_index_times[k];
      scratch.site_energy_derivatives[term.l2] +=
          scratch.site_energy_derivatives[term.id] * term.c *
          scratch.moment_vals[term.l1];
      scratch.site_energy_derivatives[term.l1] +=
          scratch.site_energy_derivatives[term.id] * term.c *
          scratch.moment_vals[term.l2];
    }

    if (eflag_global) energy += site_energy;
    if (site_eatom) site_eatom[ati] += site_energy;
    if (flux_enabled_) site_energy_[ati] = site_energy;

    for (int jj = 0; jj < jnum_true; ++jj) {
      const double *jac0 = scratch.moment_jacobian + (jj * 3 + 0) * basic_count;
      const double *jac1 = scratch.moment_jacobian + (jj * 3 + 1) * basic_count;
      const double *jac2 = scratch.moment_jacobian + (jj * 3 + 2) * basic_count;
      double derivative[3] = {0.0, 0.0, 0.0};
      for (int k = 0; k < basic_count; ++k) {
        const double coefficient = scratch.site_energy_derivatives[k];
        derivative[0] += coefficient * jac0[k];
        derivative[1] += coefficient * jac1[k];
        derivative[2] += coefficient * jac2[k];
      }

      const int atj = scratch.jlist_true[jj];
      const double r[3] = {x[atj][0] - x[ati][0], x[atj][1] - x[ati][1],
                           x[atj][2] - x[ati][2]};
      for (int k = 0; k < 3; ++k) {
        force_array[ati][k] += derivative[k];
        force_array[atj][k] -= derivative[k];
      }
      tally_standard_vatom(site_vatom, ati, atj, r, derivative);
      if (flux_enabled_ || !interfaces_.empty())
        tally_transport(scratch, ati, atj, r, derivative);
    }
  }
}

void PairMLIPTransport::reduce_transport_scratch(const std::vector<Scratch *> &sources)
{
  if (flux_enabled_) {
    std::fill(heat_virial_.begin(), heat_virial_.end(), 0.0);
    for (const Scratch *source : sources)
      for (size_t i = 0; i < heat_virial_.size(); ++i)
        heat_virial_[i] += source->heat_virial[i];
  }
  for (size_t channel = 0; channel < interfaces_.size(); ++channel) {
    auto &target = interfaces_[channel].force;
    std::fill(target.begin(), target.end(), 0.0);
    for (const Scratch *source : sources)
      for (size_t i = 0; i < target.size(); ++i)
        target[i] += source->interface_force[channel][i];
  }
}

void PairMLIPTransport::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = eflag_either = eflag_global = eflag_atom =
       vflag_either = vflag_global = vflag_atom = cvflag_atom = vflag_fdotr = 0;

  const int nall = atom->nlocal + atom->nghost;
  reset_transport_storage(nall);
  double energy = 0.0;
  compute_range(0, list->inum, scratch_, atom->f,
                eflag_atom ? eatom : nullptr,
                vflag_atom ? vatom : nullptr, energy);
  reduce_transport_scratch({&scratch_});

  if (eflag_global) eng_vdwl += energy;
  if (vflag_fdotr) virial_fdotr_compute();
  transport_step_ = update->ntimestep;
  reverse_step_ = -1;
}

double PairMLIPTransport::init_one(int, int)
{
  return cutoff;
}

void PairMLIPTransport::ensure_transport_reverse(bigint timestep)
{
  if (timestep != transport_step_)
    error->all(FLERR, "MTP transport data are not current on timestep {}", timestep);
  if (reverse_step_ == timestep) return;
  if (comm_reverse > 0) comm->reverse_comm(this);
  reverse_step_ = timestep;
}

const double *PairMLIPTransport::transport_interface_force(int channel) const
{
  if (channel < 0 || channel >= static_cast<int>(interfaces_.size())) return nullptr;
  return interfaces_[channel].force.data();
}

int PairMLIPTransport::pack_reverse_comm(int n, int first, double *buffer)
{
  int m = 0;
  for (int i = first; i < first + n; ++i) {
    if (flux_enabled_)
      for (int k = 0; k < 9; ++k)
        buffer[m++] = heat_virial_[static_cast<size_t>(i) * 9 + k];
    for (const auto &channel : interfaces_)
      for (int k = 0; k < 3; ++k)
        buffer[m++] = channel.force[static_cast<size_t>(i) * 3 + k];
  }
  return m;
}

void PairMLIPTransport::unpack_reverse_comm(int n, int *list, double *buffer)
{
  int m = 0;
  for (int ii = 0; ii < n; ++ii) {
    const int i = list[ii];
    if (flux_enabled_)
      for (int k = 0; k < 9; ++k)
        heat_virial_[static_cast<size_t>(i) * 9 + k] += buffer[m++];
    for (auto &channel : interfaces_)
      for (int k = 0; k < 3; ++k)
        channel.force[static_cast<size_t>(i) * 3 + k] += buffer[m++];
  }
}

double PairMLIPTransport::memory_usage()
{
  double bytes = static_cast<double>(site_energy_.capacity() +
                                     heat_virial_.capacity()) * sizeof(double);
  for (const auto &channel : interfaces_)
    bytes += static_cast<double>(channel.force.capacity()) * sizeof(double);
  return bytes;
}
