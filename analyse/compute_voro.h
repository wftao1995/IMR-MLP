// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_COMPUTE_VORO
#define IMRMLP_COMPUTE_VORO

#include "base/neighborhood_voro.h"
#include "utils/gputils.h"

/**
 * Notes about surf_tol and edge_tol:
 *  Surf_tol will remove small faces, which makes ie. 0 1 1 2 4 -> 0 0 0 2 4
 *  Edge_tol will remove small edges, which will reduce artifacts, makes cells
 *      i.e nearly equavelent to 0 0 12 0 0 to 0 0 12 0 0 
 * 
 * **/

/**
 * @brief Voronoi tessellation analyzer.
 * @ingroup analyse
 *
 * Computes Voronoi cell volumes, face areas, edge lengths, and coordination numbers.
 * Supports direct (DIR) and radius-based (RAD) modes with face/edge tolerance stripping.
 */
class ComputeVoro : public ConnectSettings
{
private:
    enum voromode
    {
        DIR,
        RAD
    } mode = RAD;      // dir or rad
    int num_edges = 9; // 0-0-0-12-0-0-0
    double strip_surf_tol = -0.02;
    double strip_edge_tol = -0.02;
    int min_neighbor = 0; // minimum retained neighbors during refinement; 0 disables the limit
    std::string mode_tmp;
    std::map<int, double> atomradii;
    // bool sample_one = false;
    // bool sample_all = false;

public:
    ComputeVoro(){};
    ComputeVoro(const GPSettings &_settings, const std::map<int, double> &atradii);
    ~ComputeVoro(){};

    int get_num_edges() const { return num_edges; };

    VoroInfo compute(Atoms &);
    VoroInfo compute(const NeighborhoodsVoro &);                      // just a wrapper
    VoroInfo compute(Atoms &str, NeighborhoodsVoro **); // build neighbors

    void InitSettings()
    {
        Connect(mode_tmp, "mode");
        Connect(num_edges, "num_edges");
        Connect(strip_surf_tol, "strip_surf_tol");
        Connect(strip_edge_tol, "strip_edge_tol");
        Connect(min_neighbor, "min_neighbor");
    }
};

#endif
