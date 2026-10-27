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

PairStyle(mlip/spline, PairMLIPSpline)

#else

#ifndef LMP_PAIR_MLIPSPLINE_H
#define LMP_PAIR_MLIPSPLINE_H

//#define MLIP_DEBUG
#define MLIP_G_S 20
// #define COUNT_TIME

#include "pair.h"
#include "pair_MLIP_opt.h"
namespace LAMMPS_NS
{

    class PairMLIPSpline : public PairMLIPOPT
    {
    public:
        PairMLIPSpline(class LAMMPS *);
        void LoadPot() override;
        virtual ~PairMLIPSpline();
        virtual void compute(int, int);
        void settings(int, char **);

    protected:
        // spline coeffs
        // typei, typej, mu, sp_rval, sp_coeff
        int Nr;
        double deltar;
        double *rvals = NULL;
        // typei, typej, mu, id
        double ****spdata = NULL;
        // Flat layout: [type_i][type_j][mu][interval][6 coefficients].
        double *spcoeffs = NULL;
        void interpolate(int n, double delta, double *vals, double *spcoeffs_data);
    };
};

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
