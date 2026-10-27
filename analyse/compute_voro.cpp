// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_voro.h"
#include "base/neighborhood_voro_direct.h"
#include "base/neighborhood_voro_radi.h"

ComputeVoro::ComputeVoro(const GPSettings &_settings, const std::map<int, double> &atradii)
{
    InitSettings();
    ApplySettings(_settings);
    if (mode_tmp[0] == 'r' || mode_tmp[0] == 'R')
    {
        mode = RAD;
    }
    else if (mode_tmp[0] == 'd' || mode_tmp[0] == 'D')
    {
        mode = DIR;
    }
    atomradii = atradii;
    ++num_edges;
    // PrintSettings();
}

VoroInfo ComputeVoro::compute(Atoms &str)
{
    VoroInfo result;
    result.total_count = str.size();
    if (mode == RAD)
    {
        NeighborhoodsVoroRad nbhs(str, atomradii, num_edges, strip_surf_tol, strip_edge_tol, min_neighbor);
        result.tessencount_all = nbhs.tessencount_;
        result.tessencount_partial = nbhs.tessencount_type_;
#ifndef NO_SAMPLE_STR
        result.str = nbhs.str_;
#endif
        result.meanvol = nbhs.vorovol_;
    }
    else
    {
        NeighborhoodsVoroDir nbhs(str, num_edges, strip_surf_tol, strip_edge_tol, min_neighbor);
        result.tessencount_all = nbhs.tessencount_;
        result.tessencount_partial = nbhs.tessencount_type_;
#ifndef NO_SAMPLE_STR
        result.str = nbhs.str_;
#endif
        result.meanvol = nbhs.vorovol_;
    }

    return result;
}

VoroInfo ComputeVoro::compute(const NeighborhoodsVoro &nbhs)
{
    VoroInfo result;
    result.total_count = nbhs.size();
    result.tessencount_all = nbhs.tessencount_;
    result.tessencount_partial = nbhs.tessencount_type_;
#ifndef NO_SAMPLE_STR
    result.str = nbhs.str_;
#endif
    result.meanvol = nbhs.vorovol_;
    return result;
}
// Be sure to delete !!!
VoroInfo ComputeVoro::compute(Atoms &str, NeighborhoodsVoro **pnbhs)
{
    VoroInfo result;
    if (mode == RAD)
    {
        *pnbhs = new NeighborhoodsVoroRad(str, atomradii, num_edges, strip_surf_tol, strip_edge_tol, min_neighbor);
    }
    else
    {
        /*REMEBER TO DELETE THIS POINTER!*/
        *pnbhs = new NeighborhoodsVoroDir(str, num_edges, strip_surf_tol, strip_edge_tol, min_neighbor);
    }
    result.total_count = (*pnbhs)->size();
    result.tessencount_all = (*pnbhs)->tessencount_;
    result.tessencount_partial = (*pnbhs)->tessencount_type_;
#ifndef NO_SAMPLE_STR
    result.str = (*pnbhs)->str_;
#endif
    result.meanvol = (*pnbhs)->vorovol_;
    return result;
}
