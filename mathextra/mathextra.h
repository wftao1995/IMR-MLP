// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_MATH_EXTRA
#define IMRMLP_MATH_EXTRA
#include "angular_functions.h"
#include "combinatorics.h"

/**
 * Standard library functions:
 * All discussions are based on double precision (64bit float)
 * Standard values are computed using Wolfram Mathematica, with 50 demical digits precision.
 *  tgamma: both boost and std version usually < 150\epsilon when x not exceed 100 and not smaller than 0.02
 *      and < 4 when x not exceed 10
 *      usually (not always) boost::maht::tgamma is a little more accurate than std::tgamma
 * 
 *  tgamma1mp1: only boost have this function, returns gamma(z+1)-1 by definition, but give a little more accurate 
 *      result than directly using definition. This is more accurate when z 
 *      is very small
 * 
 *  erf: boost and std not exceed 1\epsilon when x value is small than 5
 * 
 *  erfc: 
 * */

#endif