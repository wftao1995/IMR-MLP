// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "pair_MLIP_opt.h"
#include "mlip_opt_virial.h"
#include "mlip_model_mpi.h"
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <vector>
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "update.h"

using namespace LAMMPS_NS;

void PairMLIPOPT::settings(int narg, char **arg)
{
    try {
        const bool options_only = narg > 0 && std::string(arg[0]) == "derivative_mode";
        if (metadata_arguments_.defer(narg, arg, options_only)) {
            if (options_only) {
                if (narg != 2) throw std::invalid_argument("Expected derivative_mode auto|cached|recompute");
                derivative_mode_ = MLIPCore::Cpu::derivative_mode(arg[1]);
            }
            return;
        }
        if (strlen(arg[0]) > 999) throw std::invalid_argument("MLIP settings file name too long");
        int map_end = narg;
        for (int i = 1; i < narg; ++i) if (std::string(arg[i]) == "derivative_mode") {
            map_end = i;
            if (i + 2 != narg) throw std::invalid_argument("Expected derivative_mode auto|cached|recompute");
            derivative_mode_ = MLIPCore::Cpu::derivative_mode(arg[i+1]);
            break;
        }
        std::string model_text;
        if (pending_restart_model_text_.empty()) model_text = MLIPCore::read_model_text_mpi(world, arg[0]);
        else model_text.swap(pending_restart_model_text_);
        std::istringstream header_input(model_text);
        const auto header = MTPMetadata::read_header(header_input);
        mtp_species_map_.assign(map_end, -1);
        metadata_arguments_.load(narg, arg, map_end, mtp_species_map_.data(),
                                 mtp_species_map_.size(), false, true, &header);
        bool mtp_m = header.tag.structured() &&
            (header.tag.text("type") == "MTP_M" || header.tag.text("model") != "linear" || header.tag.fields.count("gpm_format"));
        std::istringstream body(model_text);
        std::string body_line;
        while (std::getline(body, body_line)) {
            body_line = MTPMetadata::trim(body_line);
            if (!body_line.empty() && body_line.front() == '#')
                body_line = MTPMetadata::trim(body_line.substr(1));
            const auto equals = body_line.find('=');
            const auto key = MTPMetadata::trim(body_line.substr(0, equals));
            if (key == "model_coeffs" || key == "mtp_model") mtp_m = true;
        }
        strcpy(MLIPpot_filename, arg[0]);
        if (mtp_m) {
            MLIPCore::collective_try(world, [&] {
                mtp_model_.reset(MLIPCore::setup_from_text(model_text, mtp_species_map_.data(),
                                                &cutoff, mtp_species_map_.size()));
                mtp_executor_ = std::make_unique<MLIPCore::Cpu::Executor>(*mtp_model_);
            });
            no_virial_fdotr_compute = 1;
            restart_state_.filename = arg[0];
            restart_state_.model_text = model_text;
            restart_state_.arguments.clear();
            for (int i = 0; i < narg; ++i) restart_state_.arguments.emplace_back(arg[i]);
            restartinfo = 1;
        } else {
            if (map_end > MLIP_G_S) throw std::invalid_argument("Too many legacy MLIP atom-type mappings");
            if (derivative_mode_ != MLIPCore::Cpu::DerivativeMode::Auto)
                throw std::invalid_argument("derivative_mode currently requires an MTP_M model");
            std::fill_n(spmap, MLIP_G_S, -1);
            std::copy(mtp_species_map_.begin(), mtp_species_map_.end(), spmap);
            LoadPot();
        }
        metadata_arguments_.loaded = true;
    } catch (const std::exception &e) { error->one(FLERR, e.what()); }
}

void PairMLIPOPT::coeff(int narg, char **arg)
{
    try {
        auto arguments = metadata_arguments_.coefficients(narg, arg, atom->ntypes);
        if (!arguments.empty()) {
            std::vector<char *> pointers;
            for (auto &value : arguments) pointers.push_back(value.data());
            settings(static_cast<int>(pointers.size()), pointers.data());
        }
        metadata_arguments_.validate(atom->ntypes);
    } catch (const std::exception &e) { error->one(FLERR, e.what()); }
    if (!allocated)
    {
        allocate();
    }
}

void PairMLIPOPT::write_restart(FILE *file)
{
    try { MLIPCore::write_restart_state(file, restart_state_); }
    catch (const std::exception &e) { error->one(FLERR, e.what()); }
}

void PairMLIPOPT::read_restart(FILE *file)
{
    try {
        auto state = MLIPCore::read_restart_state(file, world);
        pending_restart_model_text_ = std::move(state.model_text);
        std::vector<char *> arguments;
        for (auto &argument : state.arguments) arguments.push_back(argument.data());
        settings(static_cast<int>(arguments.size()), arguments.data());
        metadata_arguments_.validate(atom->ntypes);
        if (!allocated) allocate();
    } catch (const std::exception &e) { error->one(FLERR, e.what()); }
}

void PairMLIPOPT::init_style()
{
    try { metadata_arguments_.validate(atom->ntypes); }
    catch (const std::exception &e) { error->one(FLERR, e.what()); }
    if (mtp_executor_) {
        if (std::strcmp(update->unit_style, "metal") || domain->dimension != 3)
            error->all(FLERR, "MTP_M requires units metal and dimension 3");
        // Neighbor::init derives exclude after pair initialization. Inspect
        // requested settings, including commands changed between runs.
        if (neighbor->nex_type || neighbor->nex_group || neighbor->nex_mol ||
            neighbor->includegroup)
            error->all(FLERR, "MTP_M requires a complete environment; neighbor exclusions and neigh_modify include groups are unsupported");
        if (atom->molecular) for (int k = 1; k <= 3; ++k)
            if (force->special_lj[k] != 1.0 || force->special_coul[k] != 1.0)
                error->all(FLERR, "MTP_M molecular systems require special_bonds lj/coul 1 1 1");
    }
    if (force->newton_pair != 1)
        error->all(FLERR, "Pair style MLIP requires Newton pair on");

    if (inited)
    {
        // deallocate();
    }

    // LoadPot();

    if (!allocated)
    {
        allocate();
    }
    cutoffsq = cutoff * cutoff;
    int n = atom->ntypes;
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            cutsq[i][j] = cutoffsq;
        }
    }

    inited = true;
    //   In a full list, all the neighbors of atom I are stored with atom I.
    // In a half list, the I-J interaction is stored either with atom I or J
    // but not both.  See the neigh_half*cpp and neigh_full*cpp files
    // for details.


    neighbor->add_request(this, NeighConst::REQ_FULL);


}

PairMLIPOPT::PairMLIPOPT(LAMMPS *lmp) : Pair(lmp)
{
    restartinfo = 0;
    manybody_flag = 1;
    centroidstressflag = CENTROID_NOTAVAIL;

    single_enable = 0;

    inited = false;
    allocated = 0;
    scaling = 1.0;
    scaling2 = 2.0;

    // born_matrix_enable = 1;


}

void PairMLIPOPT::LoadPot(void)
{
    std::ifstream ifs(MLIPpot_filename);
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
        scaling2 = 2. * scaling;
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
        radial_bases = MLIPOpt::read_radial_bases(
            ifs, tmpstr, species_count, scaling, radial_basis_split,
            rb_size, cutoff);
    } catch (const std::exception &exception) {
        const std::string message =
            std::string("Error reading MTP radial basis: ") + exception.what();
        error->all(FLERR, message.c_str());
    }
    {
        std::vector<MLIPCore::mlip::radial_basis_t *> bases;
        for (auto &basis : radial_bases) bases.push_back(&basis);
        MLIPCore::Radial::finish_bases(MLIPCore::read_model_text(MLIPpot_filename),
                                     bases, radial_basis_split, species_count);
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

    memory->create(alpha_index_times, alpha_index_times_count, "pair::alpha_index_times");
    if (alpha_index_times_count > 0 && alpha_index_times == nullptr)
        error->all(FLERR, "Memory allocation error");

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
    std::vector<alpha_times_group_t> times_groups;
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

    memory->create(val_mu, max_mu, "pair::val_mu");
    memory->create(der_mu, max_mu, "pair::der_mu");
    memory->create(moment_vals, alpha_moments_count, "pair::moment_vals");
    memory->create(site_energy_ders_wrt_moments_, alpha_moments_count, "pair::site_energy_ders_wrt_moments_");
    memory->create(rb_vals, rb_size, "pair::rb_vals");
    memory->create(rb_ders, rb_size, "pair::rb_ders");
    memory->create(dist_powers, max_alpha_index_basic, "pair::dist_powers");
    dist_powers[0] = 1;
    // coord_powers is offset by 3 to avoid if-statement in main loop
    memory->create(coords_powers, max_alpha_index_basic + 1, "pair::coords_powers");
    coords_powers[0].x = 0;
    coords_powers[0].y = 0;
    coords_powers[0].z = 0;
    coords_powers[1].x = 1;
    coords_powers[1].y = 1;
    coords_powers[1].z = 1;
    memory->create(jlist_true, jnum_capacity, "pair::jlist_true");
    memory->create(moment_jacobian, alpha_index_basic_count * jnum_capacity * 3, "pair::moment_jacobian");
}

void PairMLIPOPT::allocate()
{
    int n = atom->ntypes;
    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            setflag[i][j] = 1;

    allocated = 1;
}

void PairMLIPOPT::deallocate()
{
    if (allocated)
    {
        memory->destroy(rb_vals);
        memory->destroy(rb_ders);
        memory->destroy(val_mu);
        memory->destroy(der_mu);
        memory->destroy(cutsq);
        memory->destroy(setflag);
        memory->destroy(moment_vals);
        memory->destroy(site_energy_ders_wrt_moments_);
        memory->destroy(alpha_moment_mapping);
        memory->destroy(alpha_index_times);
        memory->destroy(alpha_times_groups);
        memory->destroy(moment_zero_ids);
        memory->destroy(alpha_index_basic);
        memory->destroy(radial_coeffs);
        memory->destroy(specie_coeffs);
        memory->destroy(moment_coeffs);
        memory->destroy(moment_jacobian);
        memory->destroy(jlist_true);
        memory->destroy(dist_powers);
        memory->destroy(coords_powers);
        memory->destroy(nu_arr_list);
    }

    allocated = 0;
}

PairMLIPOPT::~PairMLIPOPT()
{
    if (copymode)
        return;

    if (allocated)
    {
        deallocate();
    }
}
// todo: use template function to handle radial basis for different types
void PairMLIPOPT::compute(int eflag, int vflag)
{
    ev_init(eflag, vflag);
    if (mtp_executor_) { compute_mtp_m(); return; }

    const bool need_energy = eflag_global || eflag_atom;
    double en = 0.0;
    double *p_site_en = NULL;
    double **p_site_virial = NULL;

    if (eflag_atom)
        p_site_en = &eatom[0];
    if (vflag_atom)
        p_site_virial = vatom;

    double **x = atom->x;
    double **f = atom->f;
    int *type = atom->type;
    const int basic_count = alpha_index_basic_count;

    // list->inum (int) is number of i atoms in i-j pairs
    // list->numneigh (int *) is number of j atoms for each i atom in i-j pairs
    for (int ii = 0; ii < list->inum; ii++)
    {
        int ati = list->ilist[ii];
        int jnum = list->numneigh[ati];
        int type_central = spmap[type[ati]];
        // if (type_central < 0)
        //     continue;
        double *ivec = x[ati];
        if (jnum_capacity < jnum)
        {
            int new_capacity = jnum_capacity ? jnum_capacity + jnum_capacity / 2 : 64;
            if (new_capacity < jnum) new_capacity = jnum;
            memory->grow(moment_jacobian, basic_count * new_capacity * 3, "pair::moment_jacobian");
            memory->grow(jlist_true, new_capacity, "pair::jlist_true");
            jnum_capacity = new_capacity;
        }
        for (int i = 0; i < moment_zero_count; ++i)
            moment_vals[moment_zero_ids[i]] = 0.0;

        int jnum_true = 0;
        int *neighbors = list->firstneigh[ati];
        for (int jj = 0; jj < jnum; jj++)
        {
            int atj = neighbors[jj] & NEIGHMASK;
            double *jvec = x[atj];
            int jtype = spmap[type[atj]];
            // if (jtype < 0)
            //     continue;
            double rvec[3] = {jvec[0] - ivec[0], jvec[1] - ivec[1], jvec[2] - ivec[2]};
            double rijsq = rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2];
            if (rijsq > 0.0 && rijsq < cutoffsq)
            {
                #ifdef COUNT_TIME
                t2 = clock();
                #endif
                double rij = sqrt(rijsq);
                const int pair = type_central * species_count + jtype;
                const MLIPOpt::RadialBasis &basis = radial_basis_split
                    ? radial_bases[pair] : radial_bases[0];
                if (basis.excludes(rij)) continue;
                double rijinv = 1.0 / rij;
                jlist_true[jnum_true] = atj;
                basis.evaluate_inside(rij, rb_vals, rb_ders);

                // dist_powers[0] = 1;
                // coords_powers[0] = coords_powers[1] = coords_powers[2] = 1;
                for (int k = 1; k < max_alpha_index_basic; k++)
                {
                    dist_powers[k] = dist_powers[k - 1] * rijinv;
                    coords_powers[k+1] = {
                        coords_powers[k].x * rvec[0],
                        coords_powers[k].y * rvec[1],
                        coords_powers[k].z * rvec[2]
                    };
                }

                for (int mu = 0; mu < max_mu; ++mu)
                {
                    double val = 0.0;
                    double der = 0.0;
                    const int idxoffset = pair * radial_func_count * rb_size + mu * rb_size;
                    #pragma ivdep
                    for (int xi = 0; xi < rb_size; ++xi) {
                        const double coeff = radial_coeffs[idxoffset + xi];
                        val += coeff * rb_vals[xi];
                        der += coeff * rb_ders[xi];
                    }
                    val_mu[mu] = val;
                    der_mu[mu] = der;
                }

                for (int i = 0; i < basic_count; i++)
                {
                    const int mu = alpha_index_basic[i].mu;
                    double val = val_mu[mu], der = der_mu[mu];

                    const int k = nu_arr_list[i];
                    const double rpowk = dist_powers[k];
                    val *= rpowk;
                    der = der * rpowk - k * val * rijinv;
                    const double pow0 = coords_powers[alpha_index_basic[i].b1 + 1].x;
                    const double pow1 = coords_powers[alpha_index_basic[i].b2 + 1].y;
                    const double pow2 = coords_powers[alpha_index_basic[i].b3 + 1].z;

                    double mult0 = pow0 * pow1 * pow2;
                    moment_vals[i] += val * mult0;
                    mult0 *= der * rijinv;
                    moment_jacobian[(jnum_true * 3 + 0) * basic_count + i] = mult0 * rvec[0] +
                        alpha_index_basic[i].b1 * val *
                        coords_powers[alpha_index_basic[i].b1].x *
                        pow1 * pow2;
                    moment_jacobian[(jnum_true * 3 + 1) * basic_count + i] = mult0 * rvec[1] +
                        alpha_index_basic[i].b2 * val *
                        coords_powers[alpha_index_basic[i].b2].y *
                        pow0 * pow2;
                    moment_jacobian[(jnum_true * 3 + 2) * basic_count + i] = mult0 * rvec[2] +
                        alpha_index_basic[i].b3 * val *
                        coords_powers[alpha_index_basic[i].b3].z *
                        pow0 * pow1;
                }

                jnum_true++;
            }
        }

        // Non-elementary moments.  Assignment groups avoid clearing and then
        // repeatedly loading/storing newly produced moments.  Additive groups
        // retain the original operation order for unusual potential graphs.
        for (int g = 0; g < alpha_times_group_count; ++g)
        {
            const alpha_times_group_t &group = alpha_times_groups[g];
            if (group.assign) {
                const alpha_times_t &first = alpha_index_times[group.begin];
                double value = first.c * moment_vals[first.l1] * moment_vals[first.l2];
                for (int i = group.begin + 1; i < group.end; ++i) {
                    const alpha_times_t &term = alpha_index_times[i];
                    value += term.c * moment_vals[term.l1] * moment_vals[term.l2];
                }
                moment_vals[group.id] = value;
            } else {
                for (int i = group.begin; i < group.end; ++i) {
                    const alpha_times_t &term = alpha_index_times[i];
                    moment_vals[group.id] += term.c * moment_vals[term.l1] * moment_vals[term.l2];
                }
            }
        }

        // convolving with coefficients && Backpropagation step 1: site energy derivative is the corresponding linear combination
        memset(&site_energy_ders_wrt_moments_[0], 0, alpha_moments_count * sizeof(site_energy_ders_wrt_moments_[0]));
        double buff_site_energy = 0.0;
        if (need_energy) {
            buff_site_energy = specie_coeffs[type_central];
            #pragma ivdep
            for (int i = 0; i < alpha_scalar_moments; i++)
                buff_site_energy += moment_coeffs[i] * moment_vals[alpha_moment_mapping[i]];
        }
        #pragma ivdep
        for (int i = 0; i < alpha_scalar_moments; ++i) {
            site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = moment_coeffs[i];
        }

        // Backpropagation step 2: expressing through basic moments:
        // TODO: how to make this parallel??
        // Parallel: parallel at level, and do reduction after
        for (int i = alpha_index_times_count - 1; i >= 0; i--)
        {
            double val0 = moment_vals[alpha_index_times[i].l1];
            double val1 = moment_vals[alpha_index_times[i].l2];
            int mulc = alpha_index_times[i].c;

            site_energy_ders_wrt_moments_[alpha_index_times[i].l2] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i].id] * mulc * val0;
            site_energy_ders_wrt_moments_[alpha_index_times[i].l1] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i].id] * mulc * val1;
        }

        if (need_energy) en += buff_site_energy;
        if (p_site_en != NULL)
            p_site_en[ati] = buff_site_energy;

        // Multiply by the neighbor-major Jacobian.  Each component is a
        // contiguous reduction over basic moments.
        for (int j = 0; j < jnum_true; ++j)
        {
            const double *jac0 = moment_jacobian + (j * 3 + 0) * basic_count;
            const double *jac1 = moment_jacobian + (j * 3 + 1) * basic_count;
            const double *jac2 = moment_jacobian + (j * 3 + 2) * basic_count;
            double der0 = 0.0;
            double der1 = 0.0;
            double der2 = 0.0;
            #pragma omp simd reduction(+:der0,der1,der2)
            for (int i = 0; i < basic_count; ++i) {
                const double coeff = site_energy_ders_wrt_moments_[i];
                der0 += coeff * jac0[i];
                der1 += coeff * jac1[i];
                der2 += coeff * jac2[i];
            }
            const int atj = jlist_true[j];
            f[ati][0] += der0;
            f[ati][1] += der1;
            f[ati][2] += der2;
            f[atj][0] -= der0;
            f[atj][1] -= der1;
            f[atj][2] -= der2;

            if (vflag_global || p_site_virial != NULL) {
                const double rvec[3] = {
                    x[atj][0] - ivec[0],
                    x[atj][1] - ivec[1],
                    x[atj][2] - ivec[2]
                };
                const double derivative[3] = {der0, der1, der2};
                double edge_virial[6];
                MLIPOpt::standard_virial(rvec, derivative, edge_virial);
                if (vflag_global)
                    for (int component = 0; component < 6; ++component)
                        virial[component] += edge_virial[component];
                if (p_site_virial != NULL)
                    MLIPOpt::add_per_atom_virial(
                        p_site_virial[ati], p_site_virial[atj], edge_virial);
            }
        }
    }

    if (eflag_global)
        eng_vdwl += en;
    if (vflag_fdotr)
        virial_fdotr_compute();
}

void PairMLIPOPT::compute_mtp_m()
{
    try {
        if (!mtp_workspace_) mtp_workspace_ = std::make_unique<MLIPCore::Cpu::Workspace>(*mtp_model_);
        int maximum = 0;
        for (int ii = 0; ii < list->inum; ++ii)
            maximum = std::max(maximum, list->numneigh[list->ilist[ii]]);
        mtp_workspace_->reserve(*mtp_model_, maximum, derivative_mode_);
        const bool need_energy = eflag_global || eflag_atom;
        auto &workspace = *mtp_workspace_;
        for (int ii = 0; ii < list->inum; ++ii) {
            const int i = list->ilist[ii];
            const double energy = mtp_executor_->evaluate_site(i, list->numneigh[i],
                list->firstneigh[i], atom->type, atom->x, mtp_species_map_.data(),
                need_energy, true, workspace);
            if (eflag_global) eng_vdwl += energy;
            if (eflag_atom) eatom[i] += energy;
            for (int edge = 0; edge < workspace.edge_count(); ++edge) {
                const int j = workspace.neighbor(edge);
                const double *gradient = workspace.gradient(edge);
                for (int k = 0; k < 3; ++k) {
                    atom->f[i][k] += gradient[k];
                    atom->f[j][k] -= gradient[k];
                }
                if (vflag_global || vflag_atom) {
                    double edge_virial[6];
                    MLIPOpt::standard_virial(workspace.displacement(edge), gradient, edge_virial);
                    if (vflag_global) for (int k = 0; k < 6; ++k) virial[k] += edge_virial[k];
                    if (vflag_atom) MLIPOpt::add_per_atom_virial(vatom[i], vatom[j], edge_virial);
                }
            }
        }
    } catch (const std::exception &e) { error->one(FLERR, e.what()); }
}

double PairMLIPOPT::init_one(int i, int j)
{
    return cutoff;
}
