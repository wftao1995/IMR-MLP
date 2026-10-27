// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mlip, PairMLIP)

#else

#ifndef LMP_PAIR_MLIP_H
#define LMP_PAIR_MLIP_H

//#define MLIP_DEBUG
#define MLIP_G_S 20
// #define COUNT_TIME

#include "mlip_pair_metadata.h"
#include "pair.h"
namespace LAMMPS_NS
{

    class PairMLIP : public Pair
    {
    public:
        double cutoff;
        #ifdef MLIP_DEBUG
        int debug_count = 0;
        #endif

        #ifdef COUNT_TIME
        double ts1 = 0, ts2 = 0, ts3 = 0, ts4 = 0, ts5 = 0;
        #endif

        PairMLIP(class LAMMPS *);
        void LoadPot();
        virtual ~PairMLIP();
        virtual void compute(int, int);
        void init_style();
        double init_one(int, int);
        void settings(int, char **);
        virtual void coeff(int, char **);

        //void reverse_comm_callback(double *);

    protected:
  MTPMetadata::PairArguments metadata_arguments_;
        bool inited;

        double cutoffsq;
        char MLIPpot_filename[1000];
        void allocate();
        void deallocate();

        // coeffs
        double scaling;
        int species_count;
        int spmap[MLIP_G_S];

        int alpha_moments_count;
        int alpha_index_basic_count;
        int *alpha_index_basic = NULL;
        int *alpha_index_times = NULL;
        int alpha_index_times_count;
        int *alpha_moment_mapping = NULL;

        int alpha_count;
        int alpha_scalar_moments;

        // radical basis
        double min_dist;
        double max_dist;
        int rb_size;
        int rb_type = 1;
        double mult;
        double plus_r;
        double sub_r;
        int radial_func_count;
        void RB_Calc(const double &r);
        void RB_CalcDer(const double &r);

        // regression coeffs
        double *regression_coeffs = NULL;
        double *linear_coeffs = NULL;
        int max_alpha_index_basic = 0;
        int max_mu = 0;

        // usfule ones
        int *nu_arr_list = NULL;

        //template variables
        double *moment_vals = NULL;                   // tmp
        double *site_energy_ders_wrt_moments_ = NULL; // tmp
        double *dist_powers = NULL;
        double *coords_powers = NULL;
        double *moment_jacobian = NULL;             // tmp
        double *buff_site_energy_ders = NULL;
        int jnum_old = 40;                             // tmp
        double *rb_vals = NULL;                       // tmp
        double *rb_ders = NULL;                       // tmp
        double *val_mu = NULL;
        double *der_mu = NULL;
        int *jlist_true = NULL;

        //int pack_reverse_comm(int, int, double *);
        //void unpack_reverse_comm(int, int *, double *);
    };

}

#endif
#endif

    /* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

*/
