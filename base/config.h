// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef _IMRMLP_BASE_CONFIG_H_
#define _IMRMLP_BASE_CONFIG_H_

/* ---- Switch ---- */
// Weather modify prealloc size of Atoms object
#define IMRMLP_STRPREALLOC

// Weather use boxed version of neighborhood algorimthm
// Only appears in neighborhood.h and neighborhood.cpp
// #define BOXED_NEBH
// #define NEIGHBOR_ASSUME_ALLRIGHT

/* ---- Config ---- */
// Prealloc size of Atoms object
// #define STR_PREALLOC 1000

#define PRE_ALLOC_NBH_SIZE 100

#endif
