// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/*******
 * The atom object should be wrapped before boxing, i won't do this here
 *
 * Method to calculate neighborhood atoms
 *
 * Written by wftao (wftao1995@gmail.com)
 *
 * ******/

#include "neighborhood_cluster.h"
#include "tools/tools.h"
#include "tools/chem.h"

#ifndef BOXED_NEBH
void NeighborhoodsCluster::construct(Atoms &atom, const double cutoff)
{
    const int size = atom.size();

    if (!nbhs.empty())
    {
        nbhs.clear();
    }
    nbhs.resize(size);

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < pos.size(); ++j)
        {
            Eigen::RowVector3d rij = pos[j] - pos[i];
            double dij = rij.norm();

            if ((dij < cutoff) && (dij > 10e-4) && (i <= indices_orig[j]))
            {
                Neighborhood &nebi = nbhs[i]; //nbhs[i] and nbhs[orig[i]] are the same
                Neighborhood &nebj = nbhs[indices_orig[j]];

                nebi.nbhcount_++;
                nebi.indices_.emplace_back(indices_orig[j]);
                nebi.pos_.emplace_back(rij);
                nebi.distances_.emplace_back(dij);
                nebi.chemmap_.emplace_back(atom.chemmap_[indices_orig[j]]);
            }
        }
        nbhs[i].centertype = atom.chemmap_[i];
#ifdef _ENABLE_FORCE_ONLY_PREDICTION_
        nbhs[i].force_ = atom.forces_[i];
#endif
    }
}
#else
void NeighborhoodsCluster::construct(Atoms &atom, const double cutoff)
{
    const int size = atom.size();

    if (!nbhs.empty())
    {
        nbhs.clear();
    }
    nbhs.resize(size);

    for (int i = 1; i < boxle_[0] - 1; ++i)
    {
        for (int j = 1; j < boxle_[1] - 1; ++j)
        {
            for (int k = 1; k < boxle_[2] - 1; ++k)
            {
                for (int s = 0; s < boxes[i][j][k].size(); s++)
                {
                    int indi = boxes[i][j][k][s];
                    if (indi >= size)
                        continue;
                    nbhs[indi].centertype = atom.chemmap_[indi];
                    for (int l1 = -1; l1 <= 1; l1++)
                    {
                        for (int l2 = -1; l2 <= 1; l2++)
                        {
                            for (int l3 = -1; l3 <= 1; l3++)
                            {
                                for (int t = 0; t < boxes[i + l1][j + l2][k + l3].size(); t++)
                                {
                                    int indj = boxes[i + l1][j + l2][k + l3][t];
                                    Eigen::RowVector3d rij = pos[indj] - pos[indi];
                                    double dij = rij.norm();
                                    if ((dij < cutoff) && (indi != indj) && (indi <= indices_orig[indj]))
                                    {
                                        Neighborhood &nebi = nbhs[indi];

                                        nebi.nbhcount_++;
                                        nebi.indices_.emplace_back(indices_orig[indj]);
                                        nebi.pos_.emplace_back(rij);
                                        nebi.distances_.emplace_back(dij);
                                        nebi.chemmap_.emplace_back(atom.chemmap_[indices_orig[indj]]);
                                    }
                                }
                            }
                        }
                    }
#ifdef _ENABLE_FORCE_ONLY_PREDICTION_
                    nbhs[indi].force_ = atom.forces_[indi];
#endif
                }
            }
        }
    }
}
#endif //BOXED_NEBH

void NeighborhoodsCluster::init(Atoms &atom, const double cutoff)
{
    this->init_single(atom, cutoff);
}

NeighborhoodsCluster::NeighborhoodsCluster(Atoms &atom, const double cutoff)
{
    init(atom, cutoff);
}
