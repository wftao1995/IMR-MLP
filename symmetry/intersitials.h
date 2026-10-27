// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_INTERSITIALS_H
#define IMRMLP_INTERSITIALS_H

#include <voro++/voro++.hh>
#include "external/spglib_driver.h"
#include "base/structures.h"
#include "base/anyneighborhoods.h"

class NeighborhoodKD;

/**
 * 1. compute voronoi tessellation
 * 2. find center of face, center of edge and vertices
 * 3. postprocess of very-very close points
 * */

void get_intersitial_sites(Atoms &atom, std::vector<Eigen::RowVector3d> &intpos,
                           std::map<int, double> atomradii,
                           bool count_vec = true,
                           bool count_cedg = true,
                           bool count_cface = true,
                           double mindist = 0.6,
                           double volprec = 1.0,
                           double edgeprec = 0.1,
                           double faceprec = 0.1);

// FFF candidates strictly inside the host convex hull. Cartesian geometry is
// independent of a simulation box. Artificial container-wall features are
// excluded. Local queries use Euclidean neighbors and the global host hull.
void get_intersitial_sites_nonperiodic(Atoms &atom,
    std::vector<Eigen::RowVector3d> &intpos, std::map<int,double> atomradii,
    bool count_vec=true, bool count_cedg=true, bool count_cface=true,
    double mindist=0.6, double volprec=1.0, double edgeprec=0.1, double faceprec=0.1,
    const Eigen::RowVector3d *query=nullptr, double search_radius=8.0);

/**
 * @brief Find interstitial sites near a given position using a local Voronoi approach.
 *
 * Unlike get_intersitial_sites() which builds a full-cell Voronoi diagram,
 * this function uses a KD-tree (NeighborhoodKD) to find atoms near the query
 * position, then builds a local non-periodic voro++ container with only those
 * atoms. This avoids iterating over all atoms in the cell.
 *
 * @param atom        Input structure
 * @param query_pos   Query position in Cartesian coordinates
 * @param intpos      Output: interstitial site positions (Cartesian)
 * @param atomradii   Map from atomic number → covalent radius
 * @param search_radius  Cutoff radius for KD-tree neighbor search (default: 8.0 A)
 * @param count_vec   Include Voronoi vertices (default: true)
 * @param count_cedg  Include Voronoi edge centers (default: false)
 * @param count_cface Include Voronoi face centers (default: false)
 * @param mindist     Minimum distance from any host atom (default: 0.6)
 * @param volprec     Minimum Voronoi cell volume (default: 1.0)
 * @param edgeprec    Minimum edge length (default: 0.1)
 * @param faceprec    Minimum face area (default: 0.1)
 */
void get_intersitial_sites_near(Atoms &atom,
                                const Eigen::RowVector3d &query_pos,
                                std::vector<Eigen::RowVector3d> &intpos,
                                std::map<int, double> atomradii,
                                double search_radius = 8.0,
                                bool count_vec = true,
                                bool count_cedg = true,
                                bool count_cface = true,
                                double mindist = 0.6,
                                double volprec = 1.0,
                                double edgeprec = 0.1,
                                double faceprec = 0.1,
                                NeighborhoodKD *nkd_external = nullptr);

// todo: add an option merge:
// 4. clustering candidate points
// 5. select one point in each cluster, which has the maxium volume (voro vol)

#endif
