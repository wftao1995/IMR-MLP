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

PairStyle(mlip/opt/omp, PairMLIPOMP)

#else

#ifndef LMP_PAIR_MLIPOMP_H
#define LMP_PAIR_MLIPOMP_H

//#define MLIP_DEBUG
#define MLIP_G_S 20
// #define COUNT_TIME

#include "memory.h"

#include "pair.h"
#include "mlip_pair_metadata.h"
#include "radial_basis_opt.h"
#if __has_include("mlip_restart.h")
#include "mlip_restart.h"
#else
#include "../mlip_restart.h"
#endif
#if __has_include("cpu/mtp_cpu.h")
#include "cpu/mtp_cpu.h"
#else
#include "mtp_cpu.h"
#endif
#include "thr_omp.h"
#include <vector>
namespace LAMMPS_NS
{

    class PairMLIPOMP : public Pair, public ThrOMP
    {
    public:
        double cutoff;

        struct alpha_basic_t {
            int mu, b1, b2, b3;
        };
        struct alpha_times_t {
            int l1,l2,c,id;
        };
        struct alpha_times_group_t {
            int begin, end, id;
            bool assign;
        };
        struct powk_t {
            double rpowk, pow0, pow1, pow2;
        };
        struct coordpow_t {
            double x, y, z;
        };

        PairMLIPOMP(class LAMMPS *);
        virtual void LoadPot();
        virtual ~PairMLIPOMP();
        virtual void compute(int, int);
        void init_style();
        double init_one(int, int);
        void settings(int, char **);
        virtual void coeff(int, char **);
        void write_restart(FILE *) override;
        void read_restart(FILE *) override;

        //void reverse_comm_callback(double *);

    protected:
        MLIPCore::Cpu::Model mtp_model_;
        std::unique_ptr<MLIPCore::Cpu::Executor> mtp_executor_;
        std::vector<std::unique_ptr<MLIPCore::Cpu::Workspace>> mtp_workspaces_;
        std::vector<int> mtp_species_map_;
        MLIPCore::Cpu::DerivativeMode derivative_mode_ = MLIPCore::Cpu::DerivativeMode::Auto;
        void compute_mtp_m(int, int);
        MLIPCore::RestartState restart_state_;
        std::string pending_restart_model_text_;
        bool inited;

        double cutoffsq;
        char MLIPpot_filename[1000];
        void allocate();
        void deallocate();
        // read constants
        int species_count;
        int alpha_moments_count;
        int alpha_index_basic_count;
        int alpha_index_times_count;
        int alpha_scalar_moments;
        int rb_size;
        int radial_func_count;
        // to be determined by potential
        int max_alpha_index_basic = 0;
        int max_mu = 0;

        double scaling;
        double scaling2;
        int spmap[MLIP_G_S];
        MTPMetadata::PairArguments metadata_arguments_;
        alpha_basic_t *alpha_index_basic = NULL;
        alpha_times_t *alpha_index_times = NULL;
        alpha_times_group_t *alpha_times_groups = NULL;
        int alpha_times_group_count = 0;
        int *alpha_moment_mapping = NULL;
        int *moment_zero_ids = NULL;
        int moment_zero_count = 0;

        // radical basis
        double min_dist;
        double max_dist;
        bool radial_basis_split = false;
        std::vector<MLIPOpt::RadialBasis> radial_bases;

        // regression coeffs
        double *radial_coeffs = NULL;
        double *moment_coeffs = NULL;
        double *specie_coeffs = NULL;


        // usfule ones
        int *nu_arr_list = NULL;

        //template variables
        int omp_num_threads = 1;
        // int ntotal_old = 0;
        struct ompbuffers {
            double *moment_vals = NULL;                   // tmp
            double *site_energy_ders_wrt_moments_ = NULL; // tmp
            double *dist_powers = NULL;
            coordpow_t *coords_powers = NULL;
            // Neighbor-major layout: [neighbor][xyz][basic moment].
            double *moment_jacobian = NULL;             // tmp
            int jnum_capacity = 0;                      // tmp
            double *rb_vals = NULL;                       // tmp
            double *rb_ders = NULL;                       // tmp
            double *val_mu = NULL;
            double *der_mu = NULL;
            int *jlist_true = NULL;
            // opt template variables
            powk_t *powks;
            double *mults;
            double *valmus;
            double *dermus;

            void alloc(int, int, int, int, int, int, Memory*);
            void destroy(Memory*);
        };
        ompbuffers *buffers = NULL;

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
