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

   MTP+LES optimized pair style (based on pair_MLIP_opt)
   Adds Latent Ewald Summation for long-range electrostatics.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mlip/opt-les, PairMLIPOPTLES)

#else

#ifndef LMP_PAIR_MLIPOPTLES_H
#define LMP_PAIR_MLIPOPTLES_H

#define MLIP_G_S 20

#include "mlip_pair_metadata.h"
#include "pair.h"
#include "sog_model.h"
#include "reciprocal_phases.h"
#include <vector>
namespace LAMMPS_NS
{

    class PairMLIPOPTLES : public Pair
    {
    public:
        double cutoff;

        struct alpha_basic_t {
            int mu, b1, b2, b3;
        };
        struct alpha_times_t {
            int l1,l2,c,id;
        };

        PairMLIPOPTLES(class LAMMPS *);
        void LoadPot();
        virtual ~PairMLIPOPTLES();
        virtual void compute(int, int);
        void init_style();
        double init_one(int, int);
        void settings(int, char **);
        virtual void coeff(int, char **);
        void *extract(const char *, int &) override;

    protected:
  MTPMetadata::PairArguments metadata_arguments_;
        bool inited;

        double cutoffsq;
        char MLIPpot_filename[1000];
        void allocate();
        void deallocate();
        void clear_model();

        // --- MTP parameters ---
        int species_count;
        int alpha_moments_count;
        int alpha_index_basic_count;
        int alpha_index_times_count;
        int alpha_scalar_moments;
        int rb_size;
        int radial_func_count;
        int max_alpha_index_basic = 0;
        int max_mu = 0;

        double scaling;
        double scaling2;
        int spmap[MLIP_G_S + 1];
        alpha_basic_t *alpha_index_basic = NULL;
        alpha_times_t *alpha_index_times = NULL;
        int *alpha_moment_mapping = NULL;

        // radial basis
        double min_dist;
        double max_dist;
        int rb_type = 1;
        double mult;
        double plus_r;
        double sub_r;
        double inv_subr;
        double irpi;

        // poly5
        double p0, pp;

        // poly3
        double rb_b = -1.5;

        void (PairMLIPOPTLES::*rbcalc)(const double &);
        void RB_CalcDer(const double &r);
        void RB_CalcDerCos(const double &r);
        void RB_CalcDerPoly5(const double &r);
        void RB_CalcDerPoly3(const double &r);

        // regression coeffs
        double *radial_coeffs = NULL;
        double *moment_coeffs = NULL;
        double *specie_coeffs = NULL;

        int *nu_arr_list = NULL;

        // MTP compute buffers
        double *moment_vals = NULL;
        double *site_energy_ders_wrt_moments_ = NULL;
        double *dist_powers = NULL;
        double *coords_powers_x = NULL;
        double *coords_powers_y = NULL;
        double *coords_powers_z = NULL;
        double *moment_jacobian = NULL;
        double *buff_site_energy_ders = NULL;
        int jnum_old = 0;
        double *rb_vals = NULL;
        double *rb_ders = NULL;
        double *val_mu = NULL;
        double *der_mu = NULL;
        int *jlist_true = NULL;

        // --- LES parameters ---
        bool lr_les = false;
        bool use_sog = false;
        Potentials::SOGModel sog;
        int les_version = 0;
        int sog_channels=1, active_channel=0;
        double** latent_channels=nullptr;
        std::vector<double> channel_values,channel_derivatives;
        double lr_sigma = 0.5;
        double dl = 2.0;
        double taper_fraction = 0.1;
        bool smooth_cutoff = true;
        bool remove_self_interaction = true;
        int les_response_parity = 1;
        static constexpr double coulomb_constant = 14.3996454784255;
        double *les_species_coeffs = NULL; // [species_count]
        double *les_moment_coeffs  = NULL; // [alpha_scalar_moments]

        // LES buffers
        double *latent_charges = NULL;      // [nmax] signed physical-response charge
        int latent_charges_size = 0;
        double *site_les_desc = NULL;       // [nmax * alpha_scalar_moments] per-atom scalar descriptors
        int site_les_desc_size = 0;
        double *cached_moment_vals = NULL;  // [nmax * alpha_moments_count] per-atom full moment tree
        int cached_moment_vals_size = 0;

        struct kvector_t { double x, y, z, kernel, kernel_derivative; std::array<int,3> index; };
        bool use_recurrence=true, grid_valid=false;
        std::array<double,6> cached_h{};
        std::array<int,3> phase_bounds{};
        std::array<std::array<double,3>,3> reciprocal{};
        std::vector<kvector_t> kvecs;

        // LES helper methods
        void generate_k_vectors();
        void ewald_lr_compute(double **x, double **f, int *type, int nlocal, int nghost,
                              double &eng_vdwl);
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
