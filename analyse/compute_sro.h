// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef COMPUTE_SRO_H
#define COMPUTE_SRO_H

#include "base/structures.h"
#include "base/anyneighborhoods.h"
#include "base/anyneighborhoods_simple.h"
#include "utils/gputils.h"
#include "sro_common.h"

/***
 * Backgrounds:
 *  1. Warren-Cowley parameters (WC-SRO)
 *      This is the simplest and oldest method of sro parameter,
 *      which describs short-range order in a statistical linking manner.
 *    Definition:
 *      \alpha^{AB}_{lmn} = 1 - P^{AB}_{lmn}/c_{B} = 1 - P{BA}_{lmn}/c_{A}
 *    where lmn means for a relative position of an atom {l,m,n}
 *    and c_{A}, c_{B} are concentrations of species A and B.
 *    P^{AB} is the probability of finding a B atom at position lmn.
 *    Average over the ensumble, we get
 *      \alpha^{AB}_{m} = 1 - \frac{y_{m}^{AB}}{c_{A}c_{B}}
 *    where m stands for m-th nearst neighbor, and y^{AB} stands for
 *    the probability of atom type A setting at first place and B at
 *    second place.
 *    The Warren-Cowley parameters only describs binary correlations.
 *  2. Pairwise multicomponent short-range order (PM-SRO) parameters.
 *      Actually same as WC-SRO, only differnet when A==B
 *      \alpha^{AB}_{m} = \frac{P_m^{AB} - c_B}{\delta^{AB} - c_B}
 * */

struct pair_srotype
{
    int chema;
    int chemb;
    int order;
    double sro;

    bool operator<(const pair_srotype &opt) const
    {
        if (order < opt.order)
            return true;
        if (order > opt.order)
            return false;
        if (chema < opt.chema)
            return true;
        if (chema > opt.chema)
            return false;
        if (chemb < opt.chemb)
            return true;
        return false;
    }
};

struct sro_counter
{
    int order;
    int typea;
    int typeb;
    bool operator<(const sro_counter &opt) const
    {
        if (order < opt.order)
            return true;
        if (order > opt.order)
            return false;
        if (typea < opt.typea)
            return true;
        if (typea > opt.typea)
            return false;
        if (typeb < opt.typeb)
            return true;
        return false;
    };
};
// compute WC-SRO
class ComputeSRO : public ConnectSettings
{
private:
    std::string type_tmp;
    double width = 0.05;
    SROType sro_type = WC;
    std::string tmpsrotype = "";

public:
    int nnorder = 1;
    std::vector<int> nnorders;
    void InitSettings()
    {
        Connect(nnorder, "neighbor_order");
        Connect(width, "width");
        Connect(tmpsrotype, "sro_type", "WC or PM");
    }
    void ExtraSettings(const GPSettings &_settings);

    std::vector<pair_srotype> compute(const Atoms &atom, const AnyNeighborhoods &nbhs, std::map<sro_counter, size_t> *pcount = nullptr); // compute all orders
    // for very large structures, where memory is limited and still need a parallel computation. Not flexable as previous one.
    std::vector<pair_srotype> compute(const Atoms &atom, const AnyNeighborhoodsSimple &nbhs, std::map<sro_counter, size_t> *pcount = nullptr, bool p = true); // compute all orders
    ComputeSRO(){};
    ComputeSRO(const GPSettings &_settings);
    ~ComputeSRO(){};
};

#endif
