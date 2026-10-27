// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "anyneighborhoods.h"
#include "neighborhoods_base.h"

// with ghost atoms
class NeighborhoodsGhost : public NeighborhoodsBase<NeighborhoodsGhost, AnyNeighborhoods>
{
public:
    void construct(Atoms &atom, const double cutoff);

    NeighborhoodsGhost() = default;
    ~NeighborhoodsGhost() = default;
    NeighborhoodsGhost(Atoms &atom, const double cutoff);
    void init(Atoms &atom, const double cutoff);
};
