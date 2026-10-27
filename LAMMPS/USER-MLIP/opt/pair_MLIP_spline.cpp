// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "pair_MLIP_spline.h"
#include "mlip_opt_virial.h"
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

using namespace LAMMPS_NS;

PairMLIPSpline::PairMLIPSpline(class LAMMPS * lmpr) : PairMLIPOPT(lmpr) {
    ;
};

void PairMLIPSpline::LoadPot(void)
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

        ifs >> Nr >> deltar >> radial_func_count;
        if (Nr < 2 || deltar <= 0.0)
            error->all(FLERR, "Invalid radial spline grid in MTP file");
        // double discard;
        // for (int t = 0; t < Nr; ++t) ifs >> discard;
        memory->create(rvals, Nr, "pair:rvals");
        for (int t = 0; t < Nr; ++t) {
            ifs >> rvals[t];

        }
        memory->create(spdata, species_count, species_count, radial_func_count, Nr, "pair:spdata");
        // typei typej mu Nr a b c d 2c 3d
        memory->create(spcoeffs,
                       species_count * species_count * radial_func_count * Nr * 6,
                       "pair:spcoeffs");
        for (int mu = 0; mu < radial_func_count; ++mu) {
            for (int s1 = 0; s1 < species_count; ++s1) {
                for (int s2 = 0; s2 < species_count; ++s2) {
                    int itp, jtp, muc;
                    ifs >> muc >> itp >> jtp;
                    for (int i = 0; i < Nr; ++i) {
                        ifs >> spdata[itp][jtp][muc][i];

                    }
                }
            }
        }
        // build splines
        for (int mu = 0; mu < radial_func_count; ++mu) {
            for (int s1 = 0; s1 < species_count; ++s1) {
                for (int s2 = 0; s2 < species_count; ++s2) {
                    const int offset = ((s1 * species_count + s2) * radial_func_count + mu) * Nr * 6;
                    interpolate(Nr, deltar, spdata[s1][s2][mu], spcoeffs + offset);
                }
            }
        }

        ifs >> tmpstr;

        memory->destroy(spdata);
    }
    else
    {
        error->all(FLERR, "error reading mtp file, splines not found");
    }

    if (tmpstr != "alpha_moments_count") {
        printf("%s\n", tmpstr.c_str());
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

    memory->create(val_mu, max_mu, "pair::val_mu");
    memory->create(der_mu, max_mu, "pair::der_mu");
    memory->create(moment_vals, alpha_moments_count, "pair::moment_vals");
    memory->create(site_energy_ders_wrt_moments_, alpha_moments_count, "pair::site_energy_ders_wrt_moments_");
    memory->create(rb_vals, rb_size, "pair::rb_vals");
    memory->create(rb_ders, rb_size, "pair::rb_ders");
    memory->create(dist_powers, max_alpha_index_basic, "pair::dist_powers");
    dist_powers[0] = 1.0;
    // coord_powers is offset by 3 to avoid if-statement in main loop
    memory->create(coords_powers, max_alpha_index_basic + 1, "pair::coords_powers");
    coords_powers[0].x = 0;
    coords_powers[0].y = 0;
    coords_powers[0].z = 0;
    coords_powers[1].x = 1;
    coords_powers[1].y = 1;
    coords_powers[1].z = 1;
    memory->create(jlist_true, jnum_capacity, "pair::jlist_true");
    memory->create(moment_jacobian,
                   alpha_index_basic_count * jnum_capacity * 3,
                   "pair::moment_jacobian");
}

PairMLIPSpline::~PairMLIPSpline() {
    if (spcoeffs)
        memory->destroy(spcoeffs);
    if (rvals)
        memory->destroy(rvals);
}
// a + b x + c x^2 + d x^3
// b + 2c x + 3d x^2
// in order {a,b,c,d}
// a->0, b->1, c->2, d->3
// let h to be inverval
// ci h + 4 ci+1 h + ci+2 h = 3*(yi + yi+2 - 2yi+1) / h
// c0 = 0, cn = 0 in nature BC
// in clamed BC,
// 2h c0 + h c1 = 3(y1-y0)/h
// 2h cn - h cn-1 = 3(yn - yn-1) / h
// TODO: cubic hermite spline, so that force is accurate
void PairMLIPSpline::interpolate(int n, double delta, double *f, double *spc) {
    // a = yi
    for (int i = 0; i < n; ++i) {
        spc[i * 6 + 0] = f[i];
    }
    // clamed BC, f' = 0
    // b0 = 0 , bn-1 = 0
    // int fn = n-1;
    spc[0 * 6 + 1] = 0;
    spc[(n-1) * 6 + 1] = 0; // non-exist actually, will not be used
    std::vector<double> yarr(n), m1(n), m2(n), m3(n);
    yarr[0] = 3*(f[1] - f[0])/delta/delta;
    for (int i = 1; i < n-1; ++i) {
        yarr[i] = 3*(f[i-1]+f[i+1]-2*f[i])/delta/delta;
    }
    yarr[n-1] = -3*(f[n-1]-f[n-2])/delta/delta;
    // matrix
    for (int i = 1; i < n-1; ++i) {
        m1[i] = 1;
        m2[i] = 4;
        m3[i] = 1;
    }
    m2[0] = 2;
    m3[0] = 1;
    m1[n-1] = 1;
    m2[n-1] = 2;
    m1[0] = 0; // non-exist
    m3[n-1] = 0; // non-exist
    // Eliminate both the lower diagonal and right-hand side.
    for (int i = 1; i < n; ++i) {
        const double factor = m1[i] / m2[i-1];
        m2[i] -= factor * m3[i-1];
        yarr[i] -= factor * yarr[i-1];
    }
    // back to forth
    spc[(n-1) * 6 + 2] = yarr[n-1] / m2[n-1];
    for (int i = n-2; i >= 0; --i) {
        spc[i * 6 + 2] = (yarr[i] - m3[i] * spc[(i+1) * 6 + 2]) / m2[i];
    }
    // fill others (b->1, d->3, 2c->4, 3d->5)
    for (int i = 0; i < n-1; ++i) {
        const double ci = spc[i * 6 + 2];
        const double ci1 = spc[(i+1) * 6 + 2];
        spc[i * 6 + 1] = (f[i+1]-f[i])/delta - ci*delta - (ci1-ci)*delta/3;
        spc[i * 6 + 3] = (ci1-ci)/3.0/delta;
        spc[i * 6 + 4] = 2*ci;
        spc[i * 6 + 5] = 3*spc[i * 6 + 3];
    }
}

// compute
void PairMLIPSpline::compute(int eflag, int vflag)
{
    ev_init(eflag, vflag);

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

    for (int ii = 0; ii < list->inum; ii++)
    {
        const int ati = list->ilist[ii];
        const int jnum = list->numneigh[ati];
        const int type_central = spmap[type[ati]];
        double *ivec = x[ati];
        if (jnum_capacity < jnum) {
            int capacity = jnum_capacity ? jnum_capacity + jnum_capacity / 2 : 64;
            if (capacity < jnum) capacity = jnum;
            memory->grow(moment_jacobian, basic_count * capacity * 3, "pair::moment_jacobian");
            memory->grow(jlist_true, capacity, "pair::jlist_true");
            jnum_capacity = capacity;
        }
        for (int i = 0; i < moment_zero_count; ++i)
            moment_vals[moment_zero_ids[i]] = 0.0;

        int jnum_true = 0;
        int *neighbors = list->firstneigh[ati];
        for (int jj = 0; jj < jnum; jj++)
        {
            const int atj = neighbors[jj] & NEIGHMASK;
            double *jvec = x[atj];
            const int jtype = spmap[type[atj]];
            double rvec[3] = {jvec[0] - ivec[0], jvec[1] - ivec[1], jvec[2] - ivec[2]};
            const double rijsq = rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2];
            if (rijsq < cutoffsq)
            {
                const double rij = sqrt(rijsq);
                const double rijinv = 1.0 / rij;
                jlist_true[jnum_true] = atj;

                for (int k = 1; k < max_alpha_index_basic; k++)
                {
                    dist_powers[k] = dist_powers[k - 1] * rijinv;
                    coords_powers[k+1] = {
                        coords_powers[k].x * rvec[0],
                        coords_powers[k].y * rvec[1],
                        coords_powers[k].z * rvec[2]
                    };
                }

                int interval = static_cast<int>(floor((rij - min_dist) / deltar));
                if (interval < 0) interval = 0;
                else if (interval > Nr - 2) interval = Nr - 2;
                double xx = rij - rvals[interval];
                if (xx < 0.0) xx = 0.0;
                const int pair_offset =
                    (type_central * species_count + jtype) * radial_func_count * Nr * 6;
                for (int mu = 0; mu < max_mu; ++mu)
                {
                    const double *coeffs = spcoeffs + pair_offset + (mu * Nr + interval) * 6;
                    val_mu[mu] = coeffs[0] + xx * (coeffs[1] + xx * (coeffs[2] + xx * coeffs[3]));
                    der_mu[mu] = coeffs[1] + xx * (coeffs[4] + xx * coeffs[5]);
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

        for (int i = alpha_index_times_count - 1; i >= 0; i--)
        {
            const double val0 = moment_vals[alpha_index_times[i].l1];
            const double val1 = moment_vals[alpha_index_times[i].l2];
            const int mulc = alpha_index_times[i].c;

            site_energy_ders_wrt_moments_[alpha_index_times[i].l2] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i].id] * mulc * val0;
            site_energy_ders_wrt_moments_[alpha_index_times[i].l1] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i].id] * mulc * val1;
        }

        if (need_energy) en += buff_site_energy;
        if (p_site_en != NULL)
            p_site_en[ati] = buff_site_energy;

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

void PairMLIPSpline::settings(int narg, char **arg)
{
    PairMLIPOPT::settings(narg,arg);
}
