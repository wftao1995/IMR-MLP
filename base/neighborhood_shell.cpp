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

#include "neighborhood_shell.h"
#include "tools/tools.h"
#include "tools/chem.h"

#ifdef BOXED_NEBH
void NeighborhoodsShell::construct(Atoms &atom, const double cutoff_inner, const double cutoff_outer)
{
    const int size = atom.size();
    const double cutoff_isq = cutoff_inner * cutoff_inner;
    const double cutoff_osq = cutoff_outer * cutoff_outer;

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
                                    const double dijsq = rij(0) * rij(0) + rij(1) * rij(1) + rij(2) * rij(2);
                                    if ((dijsq < cutoff_osq) && (dijsq > cutoff_isq) && (indi != indj) && (indi <= indices_orig[indj]))
                                    {
                                        Neighborhood &nebi = nbhs[indi];
                                        const double dij = sqrt(dijsq);

                                        nebi.nbhcount_++;
                                        nebi.indices_.emplace_back(indices_orig[indj]);
                                        nebi.pos_.emplace_back(rij);
                                        nebi.distances_.emplace_back(dij);
                                        nebi.chemmap_.emplace_back(atom.chemmap_[indices_orig[indj]]);
                                        if (indi != indices_orig[indj])
                                        {
                                            Neighborhood &nebj = nbhs[indices_orig[indj]];
                                            nebj.nbhcount_++;
                                            nebj.indices_.emplace_back(indi);
                                            nebj.pos_.emplace_back(-rij);
                                            nebj.distances_.emplace_back(dij);
                                            nebj.chemmap_.emplace_back(atom.chemmap_[indi]);
                                        }
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
#else  // BOXED_NEBH
void NeighborhoodsShell::construct(Atoms &atom, const double cutoff_inner, const double cutoff_outer)
{
    const int size = atom.size();
    const double cutoff_isq = cutoff_inner * cutoff_inner;
    const double cutoff_osq = cutoff_outer * cutoff_outer;

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
            double dijsq = rij(0) * rij(0) + rij(1) * rij(1) + rij(2) * rij(2);
            if ((dijsq < cutoff_osq) && (dijsq > cutoff_isq) && (i <= indices_orig[j]))
            {
                Neighborhood &nebi = nbhs[i];
                double dij = sqrt(dijsq);

                nebi.nbhcount_++;
                nebi.indices_.emplace_back(indices_orig[j]);
                nebi.pos_.emplace_back(rij);
                nebi.distances_.emplace_back(dij);
                nebi.chemmap_.emplace_back(atom.chemmap_[indices_orig[j]]);

                if (i != indices_orig[j])
                {
                    Neighborhood &nebj = nbhs[indices_orig[j]];
                    nebj.nbhcount_++;
                    nebj.indices_.emplace_back(i);
                    nebj.pos_.emplace_back(-rij);
                    nebj.distances_.emplace_back(dij);
                    nebj.chemmap_.emplace_back(atom.chemmap_[i]);
                }
            }
        }
        nbhs[i].centertype = atom.chemmap_[i];
#ifdef _ENABLE_FORCE_ONLY_PREDICTION_
        nbhs[i].force_ = atom.forces_[i];
#endif
    }
}
#endif // BOXED_NEBH

void NeighborhoodsShell::init(Atoms &atom, const double cutoff_inner, const double cutoff_outer)
{
    this->init_double(atom, cutoff_inner, cutoff_outer);
}

NeighborhoodsShell::NeighborhoodsShell(Atoms &atom, const double cutoff, const double width)
{
    init(atom, cutoff, width);
}
