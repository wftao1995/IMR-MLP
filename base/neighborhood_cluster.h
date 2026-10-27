// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_NEIGHBORHOOD_CLUSTER
#define IMRMLP_NEIGHBORHOOD_CLUSTER
/******
 * This version of neighborhood will only store one pair.
 *
 * i.e. if A is neighbor of B, and index of A is smaller than B, then B will appear
 *  in A's neighbor list but A won't be in B's neighbor list.
 *
 * *****/
#include "anyneighborhoods.h"
#include "neighborhoods_base.h"

class NeighborhoodsCluster : public NeighborhoodsBase<NeighborhoodsCluster, AnyNeighborhoods>
{
public:
    void construct(Atoms &atom, const double cutoff);

    NeighborhoodsCluster() = default;
    ~NeighborhoodsCluster() = default;
    NeighborhoodsCluster(Atoms &atom, const double cutoff);
    void init(Atoms &atom, const double cutoff);
};

#endif
