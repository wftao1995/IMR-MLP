// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "pair_MLIP.h"
#include "mlip_legacy_reader.h"
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

#include <fstream>

using namespace LAMMPS_NS;

//#include <chrono>

#define DEBUG_MARCO(a) \
    if (comm->me == 0) \
        a;

#ifdef COUNT_TIME
#include <time.h>
#endif


void PairMLIP::settings(int narg, char **arg)
{
    try {
        if (metadata_arguments_.defer(narg, arg, false)) return;
    } catch (const std::exception &e) { error->all(FLERR, e.what()); }
    if (narg == 0)
    {
        error->all(FLERR, "INCORRECT PAIR STYLE");
    }

    if (strlen(arg[0]) > 999)
    {
        error->all(FLERR, "MLIP settings file name too long");
    }

    for (int i = 0; i < MLIP_G_S; ++i)
    {
        spmap[i] = -1;
    }

    try {
        metadata_arguments_.load(narg, arg, narg, spmap, sizeof(spmap)/sizeof(spmap[0]), false);
    } catch (const std::exception &e) { error->all(FLERR, e.what()); }
    // MPI_Bcast(&spmap[0], );

    strcpy(MLIPpot_filename, arg[0]);
    LoadPot();
    metadata_arguments_.loaded = true;
}

void PairMLIP::coeff(int narg, char **arg)
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
    if (!allocated)
    {
        allocate();
    }
}

void PairMLIP::init_style()
{
    try { metadata_arguments_.validate(atom->ntypes); }
    catch (const std::exception &e) { error->all(FLERR, e.what()); }
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
    for (int i = 1; i < n; i++)
    {
        for (int j = 1; j < n; j++)
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
    // neighbor->requests[irequest]->enable_full();
}

PairMLIP::PairMLIP(LAMMPS *lmp) : Pair(lmp)
{
    restartinfo = 0;
    manybody_flag = 1;

    single_enable = 0;

    inited = false;
    allocated = 0;


}

void PairMLIP::LoadPot(void)
{
    std::ifstream ifs(MLIPpot_filename);
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

    scaling = 1.0;
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

    try {
        rbasis_type = MLIPLegacyReader::read_radial_basis_type(ifs);
    } catch (const std::exception &e) {
        error->all(FLERR, e.what());
    }

    // TODO: more basis, now is only RBChebyshev
    if (rbasis_type == "RBChebyshev")
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
        mult = 2.0 / (max_dist - min_dist);
        plus_r = min_dist + max_dist;
        sub_r = max_dist - min_dist;
        // rb_size
        ifs >> tmpstr;
        if (tmpstr != "radial_basis_size")
        {
            error->all(FLERR, "Error reading mtp file");
        }
        ifs.ignore(2);
        ifs >> rb_size;
        if (ifs.fail())
        {
            error->all(FLERR, "Error reading mtp file");
        }
    }
    else
    {
        error->all(FLERR, "Currently only RBChebyshev is supported");
    }

    ifs >> tmpstr;
    if (tmpstr != "radial_funcs_count")
        error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2);
    ifs >> radial_func_count;

    // Radial coeffs initialization
    int pairs_count = species_count * species_count; // number of species pairs

    char foo = ' ';

    // regression_coeffs.resize(pairs_count * radial_func_count * (p_RadialBasis->rb_size));
    double *tmp_regression_coeffs;
    tmp_regression_coeffs = new double[pairs_count * radial_func_count * rb_size];

    // inited = true;

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
                        tmp_regression_coeffs[(s1 * species_count + s2) * radial_func_count * rb_size +
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

    memory->create(alpha_index_basic, alpha_index_basic_count * 4, "pair::alpha_index_basic");
    memory->create(nu_arr_list, alpha_index_basic_count, "pair::nu_arr_list");
    if (alpha_index_basic == nullptr)
        error->all(FLERR, "Memory allocation error");

    int radial_func_max = -1;
    for (int i = 0; i < alpha_index_basic_count; i++)
    {
        char tmpch;
        ifs.ignore(1000, '{');
        ifs >> alpha_index_basic[i * 4] >> tmpch >> alpha_index_basic[i * 4 + 1] >> tmpch >> alpha_index_basic[i * 4 + 2] >> tmpch >> alpha_index_basic[i * 4 + 3];
        nu_arr_list[i] = alpha_index_basic[i * 4 + 1] + alpha_index_basic[i * 4 + 2] + alpha_index_basic[i * 4 + 3];
        if (ifs.fail())
            error->all(FLERR, "Error reading .mtp file");

        if (alpha_index_basic[i * 4] > radial_func_max)
            radial_func_max = alpha_index_basic[i * 4];
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

    memory->create(alpha_index_times, alpha_index_times_count * 4, "pair::alpha_index_times");
    if (alpha_index_times_count > 0 && alpha_index_times == nullptr)
        error->all(FLERR, "Memory allocation error");

    for (int i = 0; i < alpha_index_times_count; i++)
    {
        char tmpch;
        ifs.ignore(1000, '{');
        ifs >> alpha_index_times[i * 4] >> tmpch >> alpha_index_times[i * 4 + 1] >> tmpch >> alpha_index_times[i * 4 + 2] >> tmpch >> alpha_index_times[i * 4 + 3];
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
    ifs.ignore(1000, '\n');

    alpha_count = alpha_scalar_moments + 1;

    // Reading linear coeffs
    ifs >> tmpstr;
    if (tmpstr != "species_coeffs")
    {
        error->all(FLERR, "species_coeffs not found");
    }
    else
    {
        ifs.ignore(4);

        memory->create(linear_coeffs, alpha_count + species_count - 1, "pair::linear_coeffs");
        for (int i = 0; i < species_count; i++)
            ifs >> linear_coeffs[i] >> foo;

        ifs >> tmpstr;

        if (tmpstr != "moment_coeffs")
            error->all(FLERR, "Cannot read linear coeffs");

        ifs.ignore(2);

        // linear_coeffs.resize(alpha_count + species_count - 1);

        ifs.ignore(10, '{');

        for (int i = 0; i < alpha_count - 1; i++)
            ifs >> linear_coeffs[i + species_count] >> foo;
    }
    ifs.close();

    // DistributeCoeffs();
    int radial_size = pairs_count * radial_func_count * rb_size;
    int lin_coeff_size = alpha_count + species_count - 1;
    int reg_coeff_size = radial_size + lin_coeff_size;
    memory->create(regression_coeffs, reg_coeff_size, "pair::regression_coeffs");
    memcpy(&regression_coeffs[0], &tmp_regression_coeffs[0], radial_size * sizeof(tmp_regression_coeffs[0]));
    if (lin_coeff_size == alpha_count)
    {
        for (int i = 0; i <= species_count - 1; i++)
        {
            regression_coeffs[radial_size + i] = linear_coeffs[0];
        }
        for (int i = 1; i < alpha_count; i++)
        {
            regression_coeffs[radial_size + i + species_count - 1] = linear_coeffs[i];
        }
    }
    else
    {
        for (int i = 0; i < lin_coeff_size; i++)
        {
            regression_coeffs[radial_size + i] = linear_coeffs[i];
        }
    }

    // int max_alpha_index_basic = 0;
    for (int i = 0; i < alpha_index_basic_count; i++)
    {
        max_alpha_index_basic =
            std::max(max_alpha_index_basic,
                     alpha_index_basic[i * 4 + 1] + alpha_index_basic[i * 4 + 2] + alpha_index_basic[i * 4 + 3]);
        max_mu = std::max(max_mu, alpha_index_basic[i * 4]);
    }
    max_alpha_index_basic++;
    max_mu++;

    delete[] tmp_regression_coeffs;

    memory->create(val_mu, max_mu, "pair::val_mu");
    memory->create(der_mu, max_mu, "pair::der_mu");
    memory->create(moment_vals, alpha_moments_count, "pair::moment_vals");
    memory->create(site_energy_ders_wrt_moments_, alpha_moments_count, "pair::site_energy_ders_wrt_moments_");
    memory->create(rb_vals, rb_size, "pair::rb_vals");
    memory->create(rb_ders, rb_size, "pair::rb_ders");
    memory->create(dist_powers, max_alpha_index_basic, "pair::dist_powers");
    // coord_powers is offset by 3 to avoid if-statement in main loop
    memory->create(coords_powers, max_alpha_index_basic * 3 + 3, "pair::coords_powers");
    coords_powers[0] = 0;
    coords_powers[1] = 0;
    coords_powers[2] = 0;
    coords_powers[3] = 1;
    coords_powers[4] = 1;
    coords_powers[5] = 1;
    memory->create(moment_jacobian, alpha_index_basic_count * jnum_old * 3, "pair::moment_jacobian");
    memory->create(jlist_true, jnum_old, "pair::jlist_true");
    memory->create(buff_site_energy_ders, jnum_old * 3, "pair::buff_site_energy_ders");
}

void PairMLIP::allocate()
{
    int n = atom->ntypes;
    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            setflag[i][j] = 1;

    allocated = 1;
}

void PairMLIP::deallocate()
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
        memory->destroy(alpha_index_basic);
        memory->destroy(linear_coeffs);
        memory->destroy(regression_coeffs);
        memory->destroy(moment_jacobian);
        memory->destroy(jlist_true);
        memory->destroy(dist_powers);
        memory->destroy(coords_powers);
        memory->destroy(buff_site_energy_ders);
        memory->destroy(nu_arr_list);
    }
    allocated = 0;
}

PairMLIP::~PairMLIP()
{
    if (copymode)
        return;

    if (allocated)
    {
        deallocate();
    }

    #ifdef COUNT_TIME
        printf("Time j loop: %f\n", ts1);
        printf("Time j init: %f\n", ts2);
        printf("Time j loop alpha: %f\n", ts3);
        printf("Time post process: %f\n", ts4);
        printf("Time TOTAL: %f\n", ts5);
    #endif
}

void PairMLIP::compute(int eflag, int vflag)
{
    if (eflag || vflag)
        ev_setup(eflag, vflag);
    else
        eflag = vflag = eflag_global = eflag_atom = 0;

    double en = 0.0;
    double virstr[9];
    double *p_site_en = NULL;
    double **p_site_virial = NULL;

    if (eflag_atom)
        p_site_en = &eatom[0];
    if (vflag_atom)
        p_site_virial = vatom;

    // list->inum (int) is number of i atoms in i-j pairs
    // list->numneigh (int *) is number of j atoms for each i atom in i-j pairs
    for (int ii = 0; ii < list->inum; ii++)
    {
        #ifdef COUNT_TIME
        clock_t t1, t2, t3, t4, t5, t6;
        clock_t spend1 = 0, spend2 = 0;
        t1 = clock();
        #endif
        int ati = list->ilist[ii];
        int jnum = list->numneigh[ati];
        int type_central = spmap[atom->type[ati]];
        // if (type_central < 0)
        //     continue;
        double *ivec = atom->x[ati];
        // note that the distance between i-j atoms may larger than cutoffs, \
             set everything to zero to aviod this further
        double buff_site_energy = 0;
        if (jnum_old < jnum)
        {
            memory->grow(moment_jacobian, alpha_index_basic_count * jnum * 3, "pair::moment_jacobian");
            memory->grow(jlist_true, jnum, "pair::jlist_true");
            memory->grow(buff_site_energy_ders, jnum * 3, "pair::buff_site_energy_ders");
            jnum_old = jnum;
        }
        memset(&moment_jacobian[0], 0, alpha_index_basic_count * jnum * 3 * sizeof(moment_jacobian[0]));
        memset(&buff_site_energy_ders[0], 0, jnum * 3 * sizeof(buff_site_energy_ders[0]));
        memset(&moment_vals[0], 0, alpha_moments_count * sizeof(moment_vals[0]));

        int jnum_true = 0;
        for (int jj = 0; jj < jnum; jj++)
        {
            int atj = list->firstneigh[ati][jj];
            double *jvec = atom->x[atj];
            int jtype = spmap[atom->type[atj]];
            // if (jtype < 0)
            //     continue;
            double rvec[3] = {jvec[0] - ivec[0], jvec[1] - ivec[1], jvec[2] - ivec[2]};
            double rijsq = rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2];
            if (rijsq < cutoffsq)
            {
                #ifdef COUNT_TIME
                t2 = clock();
                #endif
                double rij = sqrt(rijsq);
                double rijinv = 1.0 / rij;
                jlist_true[jnum_true] = atj;
                RB_CalcDer(rij);

                dist_powers[0] = 1;
                // coords_powers[0] = coords_powers[1] = coords_powers[2] = 1;
                for (int k = 1; k < max_alpha_index_basic; k++)
                {
                    dist_powers[k] = dist_powers[k - 1] * rijinv;
                    // offset by 3 due to coords_powers [-1] offset
                    for (int a = 0; a < 3; a++)
                    {
                        coords_powers[k * 3 + a + 3] = coords_powers[(k - 1) * 3 + a + 3] * rvec[a];
                    }
                }

                for (int mu = 0; mu < max_mu; ++mu)
                {
                    val_mu[mu] = 0;
                    der_mu[mu] = 0;
                    int idxoffset = (type_central * species_count + jtype) * radial_func_count * rb_size + mu * rb_size;
                    #pragma ivdep
                    for (int xi = 0; xi < rb_size; xi++)
                    {
                        val_mu[mu] += regression_coeffs[idxoffset + xi] * rb_vals[xi];
                        der_mu[mu] += regression_coeffs[idxoffset + xi] * rb_ders[xi];
                    }
                }
                #ifdef COUNT_TIME
                t3 = clock();
                spend1 += t3 - t2;
                #endif

                for (int i = 0; i < alpha_index_basic_count; i++)
                {
                    int mu = alpha_index_basic[i * 4];
                    double val = val_mu[mu], der = der_mu[mu];

                    // int k = alpha_index_basic[i * 4 + 1] + alpha_index_basic[i * 4 + 2] + alpha_index_basic[i * 4 + 3];
                    int k = nu_arr_list[i];
                    double rpowk = dist_powers[k];
                    val *= rpowk;
                    der = der * rpowk - k * val * rijinv;
                    // offset by 3
                    double pow0 = coords_powers[alpha_index_basic[i * 4 + 1] * 3 + 0 + 3];
                    double pow1 = coords_powers[alpha_index_basic[i * 4 + 2] * 3 + 1 + 3];
                    double pow2 = coords_powers[alpha_index_basic[i * 4 + 3] * 3 + 2 + 3];

                    double mult0 = pow0 * pow1 * pow2;

                    moment_vals[i] += val * mult0;
                    mult0 *= der * rijinv;
                    moment_jacobian[i * jnum * 3 + 3 * jnum_true + 0] += mult0 * rvec[0];
                    moment_jacobian[i * jnum * 3 + 3 * jnum_true + 1] += mult0 * rvec[1];
                    moment_jacobian[i * jnum * 3 + 3 * jnum_true + 2] += mult0 * rvec[2];

                    moment_jacobian[i * jnum * 3 + 3 * jnum_true + 0] +=
                        alpha_index_basic[i * 4 + 1] * val *
                        coords_powers[alpha_index_basic[i * 4 + 1] * 3 + 0] *
                        pow1 * pow2;
                    moment_jacobian[i * jnum * 3 + 3 * jnum_true + 1] +=
                        alpha_index_basic[i * 4 + 2] * val *
                        coords_powers[alpha_index_basic[i * 4 + 2] * 3 + 1] *
                        pow0 * pow2;
                    moment_jacobian[i * jnum * 3 + 3 * jnum_true + 2] +=
                        alpha_index_basic[i * 4 + 3] * val *
                        coords_powers[alpha_index_basic[i * 4 + 3] * 3 + 2] *
                        pow0 * pow1;
                }

                jnum_true++;
                #ifdef COUNT_TIME
                t4 = clock();
                spend2 += t4 - t3;
                #endif
            }
        }

        #ifdef COUNT_TIME
        t5 = clock();
        #endif

        // non-elementary b_i
        for (int i = 0; i < alpha_index_times_count; i++)
        {
            double val0 = moment_vals[alpha_index_times[i * 4]];
            double val1 = moment_vals[alpha_index_times[i * 4 + 1]];
            int val2 = alpha_index_times[i * 4 + 2];
            moment_vals[alpha_index_times[i * 4 + 3]] += val2 * val0 * val1;
        }

        // convolving with coefficients && Backpropagation step 1: site energy derivative is the corresponding linear combination
        buff_site_energy += linear_coeffs[type_central];
        memset(&site_energy_ders_wrt_moments_[0], 0, alpha_moments_count * sizeof(site_energy_ders_wrt_moments_[0]));
        #pragma ivdep
        for (int i = 0; i < alpha_scalar_moments; i++)
        {
            buff_site_energy += linear_coeffs[species_count + i] * moment_vals[alpha_moment_mapping[i]];
            site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = linear_coeffs[species_count + i];
        }

        // Backpropagation step 2: expressing through basic moments:
        for (int i = alpha_index_times_count - 1; i >= 0; i--)
        {
            double val0 = moment_vals[alpha_index_times[i * 4 + 0]];
            double val1 = moment_vals[alpha_index_times[i * 4 + 1]];
            int val2 = alpha_index_times[i * 4 + 2];

            site_energy_ders_wrt_moments_[alpha_index_times[i * 4 + 1]] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i * 4 + 3]] * val2 * val0;
            site_energy_ders_wrt_moments_[alpha_index_times[i * 4 + 0]] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i * 4 + 3]] * val2 * val1;
        }

        // Backpropagation step 3: multiply by the Jacobian:
        for (int i = 0; i < alpha_index_basic_count; i++) {
            for (int j = 0; j < jnum_true; j++) {
                #pragma ivdep
                for (int a = 0; a < 3; a++)
                {
                    buff_site_energy_ders[j * 3 + a] += site_energy_ders_wrt_moments_[i] * moment_jacobian[i * jnum * 3 + j * 3 + a];
                }
            }
        }

        // extract energy, force, virual, etc
        en += buff_site_energy;
        if (p_site_en != NULL)
        {
            p_site_en[ati] = buff_site_energy;
        }
        for (int i = 0; i < jnum_true; i++)
        {
            atom->f[ati][0] += buff_site_energy_ders[i * 3 + 0];
            atom->f[ati][1] += buff_site_energy_ders[i * 3 + 1];
            atom->f[ati][2] += buff_site_energy_ders[i * 3 + 2];
            atom->f[jlist_true[i]][0] -= buff_site_energy_ders[i * 3 + 0];
            atom->f[jlist_true[i]][1] -= buff_site_energy_ders[i * 3 + 1];
            atom->f[jlist_true[i]][2] -= buff_site_energy_ders[i * 3 + 2];
        }
        if (p_site_virial != NULL)
        {
            for (int i = 0; i < jnum_true; i++)
            {
                // int &atj = list->ilist[i];
                int atj = jlist_true[i];
                double jvec[3] = {atom->x[atj][0] - atom->x[ati][0], atom->x[atj][1] - atom->x[ati][1], atom->x[atj][2] - atom->x[ati][2]};
                p_site_virial[ati][0] -= buff_site_energy_ders[i * 3 + 0] * jvec[0];
                p_site_virial[ati][1] -= buff_site_energy_ders[i * 3 + 1] * jvec[1];
                p_site_virial[ati][2] -= buff_site_energy_ders[i * 3 + 2] * jvec[2];
                p_site_virial[ati][3] -= 0.5 * (buff_site_energy_ders[i * 3 + 1] * jvec[0] +
                                                buff_site_energy_ders[i * 3 + 0] * jvec[1]);
                p_site_virial[ati][4] -= 0.5 * (buff_site_energy_ders[i * 3 + 2] * jvec[0] +
                                                buff_site_energy_ders[i * 3 + 0] * jvec[2]);
                p_site_virial[ati][5] -= 0.5 * (buff_site_energy_ders[i * 3 + 2] * jvec[1] +
                                                buff_site_energy_ders[i * 3 + 1] * jvec[2]);
            }
        }

        #ifdef COUNT_TIME
        t6 = clock();





        ts1 += (double)(t5-t1)/CLOCKS_PER_SEC;
        ts2 += (double)spend1/CLOCKS_PER_SEC;
        ts3 += (double)spend2/CLOCKS_PER_SEC;
        ts4 += (double)(t6-t5)/CLOCKS_PER_SEC;
        ts5 += (double)(t6-t1)/CLOCKS_PER_SEC;
        #endif
    }

    if (eflag_global)
        eng_vdwl += en;
    if (vflag_fdotr)
        virial_fdotr_compute();
}

double PairMLIP::init_one(int i, int j)
{
    return cutoff;
}

void PairMLIP::RB_Calc(const double &r)
{
    // if (rb_type == 1)
    double ksi = (2 * r - plus_r) / sub_r;
    double rm = r - max_dist;
    rb_vals[0] = rm * rm;
    rb_vals[1] = ksi * rm * rm;
    for (int i = 0; i < rb_size; i++)
    {
        rb_vals[i] = 2 * ksi * rb_vals[i - 1] - rb_vals[i - 2];
    }
}

void PairMLIP::RB_CalcDer(const double &r)
{
    // if (rb_type == 1)
    double ksi = (2 * r - plus_r) / sub_r;
    double rm = r - max_dist;
    rb_vals[0] = scaling * rm * rm;
    rb_vals[1] = scaling * ksi * rm * rm;

    rb_ders[0] = scaling * 2 * rm;
    rb_ders[1] = scaling * (mult * rm * rm + 2 * ksi * rm);

    for (int i = 2; i < rb_size; i++)
    {
        rb_vals[i] = 2 * ksi * rb_vals[i - 1] - rb_vals[i - 2];
        rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
    }
}

// int PairMLIP::pack_reverse_comm(int n, int first, double *buf)
// {
//   int i,m,last;

//   m = 0;
//   last = first + n;
//   for (i = first; i < last; i++)
//   {
//     buf[m++] = comm_arr[3*i+0];
//     buf[m++] = comm_arr[3*i+1];
//     buf[m++] = comm_arr[3*i+2];
//   }
//   return m;
// }

// /* ---------------------------------------------------------------------- */

// void PairMLIP::unpack_reverse_comm(int n, int *list, double *buf)
// {
//   int i,j,m;

//   m = 0;
//   for (i = 0; i < n; i++)
//   {
//     j = list[i];
//     comm_arr[3*j+0] += buf[m++];
//     comm_arr[3*j+1] += buf[m++];
//     comm_arr[3*j+2] += buf[m++];
//   }
// }
