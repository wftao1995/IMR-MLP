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

#include "neighborhood_cluster_shell_simple.h"
#include "tools/tools.h"
#include "tools/chem.h"

void NeighborhoodsClusterShellSimple::construct(Atoms &atom, const double cutoff_inner, const double cutoff_outer)
{
    const int size = atom.size();

    if (!nbhs.empty())
    {
        nbhs.clear();
    }
    nbhs.resize(size);

#ifdef BOXED_NEBH
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
                                    if ((dij < cutoff_outer) && (dij > cutoff_inner) && (indi != indj) && (indi <= indices_orig[indj]))
                                    {
                                        NeighborhoodSimple &nebi = nbhs[indi];

                                        nebi.indices_.emplace_back(indices_orig[indj]);
                                        nebi.distances_.emplace_back(dij);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#else  // BOXED_NEBH
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < pos.size(); ++j)
        {
            Eigen::RowVector3d rij = pos[j] - pos[i];
            double dij = rij.norm();

            if ((dij < cutoff_outer) && (dij > cutoff_inner) && (i <= indices_orig[j]))
            {
                NeighborhoodSimple &nebi = nbhs[i];

                nebi.indices_.emplace_back(indices_orig[j]);
                nebi.distances_.emplace_back(dij);
            }
        }
    }
#endif // BOXED_NEBH
}

void NeighborhoodsClusterShellSimple::init(Atoms &atom, const double cutoff_inner, const double cutoff_outer)
{
    this->init_double(atom, cutoff_inner, cutoff_outer);
}

NeighborhoodsClusterShellSimple::NeighborhoodsClusterShellSimple(Atoms &atom, const double cutoff_inner, const double cutoff_outer)
{
    init(atom, cutoff_inner, cutoff_outer);
}
