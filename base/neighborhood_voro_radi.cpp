// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "neighborhood_voro_radi.h"
#include <iostream>
#include "voro_geometry.h"
#include "tools/tools.h"
#include "tools/chem.h"
#include <Eigen/Dense>
#include <voro++/c_loops.hh>

#ifdef TOL
#undef TOL
#endif
#define TOL 10e-10

/*****
 * Build voronoi neighbors
 * 
 * need atom radiis
 * 
 * Note: positions should be wrapped before calling this code.
 * 
 * atom: Atoms object
 * radii: radii of each atoms
 * surftol: refine surface, =0 for count all, <0 to remove min, > 0 to set 
 *  threoldsd
 * edgetol: refine edge, less than 0 means count all
 * numneigh: minium number of neighbors, 0 means no restriction
 * 
 * 
 * ***/
void NeighborhoodsVoroRad::construct(Atoms &atom,
                                     std::map<int, double> &radii,
                                     const double surftol,
                                     const double edgetol,
                                     const int numneigh)
{
    //test if orth
    const Eigen::Matrix3d diagonal = atom.cell_.diagonal().asDiagonal();
    const bool orthcell = atom.cell_.isApprox(diagonal, 1e-12) && atom.cell_.diagonal().minCoeff() > 0;
    bool warned_refinement = false;

    // variables used in loop
    double vol;
    std::vector<double> fs;     // face areas
    std::vector<int> facecount; // number of faces, for face frquency
    int id;
    std::vector<int> index;
    index.resize(voro_edge_upto_);
    nbhs.resize(atom.size());
    voro_volume_each_.resize(atom.size());

    if (orthcell)
    {
        //get box boundary
        const double lx = atom.cell_(0, 0);
        const double ly = atom.cell_(1, 1);
        const double lz = atom.cell_(2, 2);
        //split into boxes
        const double l = pow(double(atom.size()) / (5.6 * lx * ly * lz), 1. / 3.);
        const int nx = floor(lx * l + 1);
        const int ny = floor(ly * l + 1);
        const int nz = floor(lz * l + 1);
        //set up a voro container
        voro::container_poly con(0, lx, 0, ly, 0, lz, nx, ny, nz, true, true, true, VORO_INIT_MEM);
        //put atoms into container
        for (int i = 0; i < atom.size(); ++i)
        {
            con.put(i, atom.pos_[i][0], atom.pos_[i][1], atom.pos_[i][2], radii[atom.chemmap_[i]]);
        }

        voro::voronoicell_neighbor ca, cb, *cell;
        voro::c_loop_all cl(con);

        if (cl.start())
        {
            do
            {
                if (con.compute_cell(ca, cl))
                {
                    index.resize(voro_edge_upto_);
                    for (int i = 0; i < voro_edge_upto_; ++i)
                    {
                        index[i] = 0;
                    }
                    id = cl.pid();
                    const auto faces = voro_geometry::faces(ca, atom, id, &radii);
                    std::vector<int> selected;
                    cell = voro_geometry::refine(ca, cb, faces, surftol, numneigh, selected, warned_refinement);
                    cell->face_areas(fs);
                    vol = cell->volume();
                    voro_volume_each_[id] = vol;
                    facecount.clear();
                    cell->face_freq_table(facecount);
                    for (int i = 3; i < __min(voro_edge_upto_, facecount.size()); ++i)
                    {
                        index[i] = facecount[i];
                    }

                    //refine edges, when counting edges are nessesary
                    if (fabs(edgetol) > TOL)
                    {
                        RefineEdge(fs.size(), cell, index, edgetol);
                    }

                    // store neighborhoods and voro index
                    Neighborhood nbh = voro_geometry::neighborhood(faces, selected, atom, id, Eigen::Matrix3d::Identity());
                    nbhs[id] = nbh;
                    // count voro index
                    voro_index_type indname(index);
                    tessencount_[indname]++;
                    vorovol_[indname] += vol;
                    voro_index_bond_type indtpname(index, atom.chemmap_[id]);
                    tessencount_type_[indtpname]++;
#ifndef NO_SAMPLE_STR
                    if (tessencount_[indname] == 0)
                    //if (tessencount_[indname] == 1 || (tessencount_[indname] > 1 && rand() % 100 < 5))
                    {
                        str_[indname].clear();
                        str_[indname].push_back(Eigen::RowVector3d::Zero());
                        str_[indname].insert(str_[indname].end(), nbh.pos_.begin(), nbh.pos_.end());
                    }
#endif
                }
                else
                {
                    std::cout << "Warning: cell cannot be computed.\n";
                }
            } while (cl.inc());
        }
        else
        {
            std::cout << "Warning: isolate particle, atom radius setting might incorrect." << std::endl;
        }
    }
    else
    {
        /***
         * For non-orthgonal box
         * see https://lammps.sandia.gov/doc/Howto_triclinic.html
         * ***/
        //set up a voro container
        Atoms atomc(atom);
        atomc.to_lower_standard();
        Eigen::Matrix3d original_cell = atom.cell_;
        if (original_cell.determinant() < 0) original_cell.row(2) *= -1;
        const Eigen::Matrix3d rotation = atomc.cell_.inverse() * original_cell;
        double lx = atomc.cell_(0, 0);
        double ly = atomc.cell_(1, 1);
        double lz = atomc.cell_(2, 2);
        double xy = atomc.cell_(1, 0);
        double xz = atomc.cell_(2, 0);
        double yz = atomc.cell_(2, 1);
        //split into boxes
        const double l = pow(double(atomc.size()) / (5.6 * lx * ly * lz), 1. / 3.);
        const int nx = floor(lx * l + 1);
        const int ny = floor(ly * l + 1);
        const int nz = floor(lz * l + 1);
        voro::container_periodic_poly con(lx, xy, ly, xz, yz, lz, nx, ny, nz, VORO_INIT_MEM);
        //put atoms into container
        for (int i = 0; i < atom.size(); ++i)
        {
            con.put(i, atomc.pos_[i][0], atomc.pos_[i][1], atomc.pos_[i][2], radii[atomc.chemmap_[i]]);
        }

        voro::voronoicell_neighbor ca, cb, *cell;
        voro::c_loop_all_periodic cl(con);

        if (cl.start())
        {
            do
            {
                if (con.compute_cell(ca, cl))
                {
                    index.resize(voro_edge_upto_);
                    for (int i = 0; i < voro_edge_upto_; ++i)
                    {
                        index[i] = 0;
                    }

                    id = cl.pid();
                    const auto faces = voro_geometry::faces(ca, atomc, id, &radii);
                    std::vector<int> selected;
                    cell = voro_geometry::refine(ca, cb, faces, surftol, numneigh, selected, warned_refinement);
                    cell->face_areas(fs);
                    vol = cell->volume();
                    voro_volume_each_[id] = vol;
                    cell->face_freq_table(facecount);
                    for (int i = 3; i < __min(voro_edge_upto_, facecount.size()); ++i)
                    {
                        index[i] = facecount[i];
                    }

                    //refine edges, when counting edges are nessesary
                    if (fabs(edgetol) > 0)
                    {
                        RefineEdge(fs.size(), cell, index, edgetol);
                    }

                    // store neighborhoods and voro index
                    Neighborhood nbh = voro_geometry::neighborhood(faces, selected, atom, id, rotation);
                    nbhs[id] = nbh;
                    // count voro index
                    voro_index_type indname(index);
                    tessencount_[indname]++;
                    vorovol_[indname] += vol;
                    voro_index_bond_type indtpname(index, atom.chemmap_[id]);
                    tessencount_type_[indtpname]++;
#ifndef NO_SAMPLE_STR
                    if (tessencount_[indname] == 0)
                    //if (tessencount_[indname] == 0 || (tessencount_[indname] != 0 && rand() % 100 < 5))
                    {
                        str_[indname].clear();
                        str_[indname].push_back(Eigen::RowVector3d::Zero());
                        str_[indname].insert(str_[indname].end(), nbh.pos_.begin(), nbh.pos_.end());
                    }
#endif
                }
                else
                {
                    std::cout << "Warning: cell cannot be computed.\n";
                }
            } while (cl.inc());
        }
        else
        {
            std::cout << "Warning: isolate particle." << std::endl;
        }
    }
}

NeighborhoodsVoroRad::NeighborhoodsVoroRad(Atoms &atom,
                                           std::map<int, double> &radii,
                                           const int voro_edge_upto,
                                           const double surftol,
                                           const double edgetol,
                                           const int numneigh)
{
    if (radii.size() == 0)
    {
        std::cerr << "Error: empty radii\n";
        exit(128);
    }
    voro_edge_upto_ = voro_edge_upto;
    //voro_edge_count_upto_ = voro_edge_cont_upto;
    construct(atom, radii, surftol, edgetol, numneigh);
}
