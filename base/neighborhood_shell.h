// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_NEIGHBORHOOD_SHELL
#define IMRMLP_NEIGHBORHOOD_SHELL

#include "anyneighborhoods.h"
#include "neighborhoods_base.h"

class NeighborhoodsShell : public NeighborhoodsBase<NeighborhoodsShell, AnyNeighborhoods>
{
public:
    void construct(Atoms &atom, const double cutoff_inner, const double cutoff_outer);

    NeighborhoodsShell() = default;
    ~NeighborhoodsShell() = default;
    NeighborhoodsShell(Atoms &atom, const double cutoff_inner, const double cutoff_outer);
    void init(Atoms &atom, const double cutoff_inner, const double cutoff_outer);
};

#endif
