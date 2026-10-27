// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_NEIGHBORHOOD_CLUSTER_SIMPLE
#define IMRMLP_NEIGHBORHOOD_CLUSTER_SIMPLE
/******
 * This version of neighborhood will only store one pair.
 *
 * i.e. if A is neighbor of B, and index of A is smaller than B, then B will appear
 *  in A's neighbor list but A won't be in B's neighbor list.
 *
 * *****/
#include "anyneighborhoods_simple.h"
#include "neighborhoods_base.h"

class NeighborhoodsClusterSimple : public NeighborhoodsBase<NeighborhoodsClusterSimple, AnyNeighborhoodsSimple>
{
public:
    void construct(Atoms &atom, const double cutoff);

    NeighborhoodsClusterSimple() = default;
    ~NeighborhoodsClusterSimple() = default;
    NeighborhoodsClusterSimple(Atoms &atom, const double cutoff);
    void init(Atoms &atom, const double cutoff);
};

#endif
