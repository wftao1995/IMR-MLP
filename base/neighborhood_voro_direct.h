// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef _MLGP_NEIGHBORHOOD_VORO_DIRECT_H
#define _MLGP_NEIGHBORHOOD_VORO_DIRECT_H

#include "neighborhood_voro.h"
#include <voro++/voro++.hh>
#include <string>

class NeighborhoodsVoroDir : public NeighborhoodsVoro
{
private:
    ;

public:
    NeighborhoodsVoroDir(){};
    ~NeighborhoodsVoroDir(){};
    NeighborhoodsVoroDir(Atoms &atom, const int voro_edge_upto, const double surftol, const double edgetol, const int numneigh);
    void construct(Atoms &atom, const double surftol, const double edgetol, const int numneigh);
};

#endif // _MLGP_NEIGHBORHOOD_VORO_DIRECT_H