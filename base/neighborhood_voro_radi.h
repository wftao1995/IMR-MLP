// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef _MLGP_NEIGHBORHOOD_VORO_RADI_H
#define _MLGP_NEIGHBORHOOD_VORO_RADI_H

#include "neighborhood_voro.h"
#include <voro++/voro++.hh>
#include <string>

class NeighborhoodsVoroRad : public NeighborhoodsVoro
{
private:
    ;

public:
    NeighborhoodsVoroRad(){};
    ~NeighborhoodsVoroRad(){};
    NeighborhoodsVoroRad(Atoms &atom, std::map<int, double> &radii, const int voro_edge_upto, const double surftol, const double edgetol, const int numneigh);
    void construct(Atoms &atom, std::map<int, double> &radii, const double surftol, const double edgetol, const int numneigh);
};

#endif // _MLGP_NEIGHBORHOOD_VORO_RADI_H