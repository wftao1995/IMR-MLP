// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_ANY_NEIGHBORHOODS_SIMPLE_H
#define IMRMLP_ANY_NEIGHBORHOODS_SIMPLE_H

#include "structures.h"
#include "tools/chem.h"
#include "config.h"

struct NeighborhoodSimple
{
    std::vector<int> indices_;     // indices of each atom
    std::vector<double> distances_; //relative distance about center atom
};

class AnyNeighborhoodsSimple
{
protected:
    std::vector<NeighborhoodSimple> nbhs;

public:
    inline int size() const
    {
        return nbhs.size();
    };

    inline NeighborhoodSimple &operator[](int index)
    {
        return nbhs[index];
    }

    inline const NeighborhoodSimple &operator[](int index) const
    {
        return nbhs[index];
    }

    AnyNeighborhoodsSimple get_neighbor_shell(double rshell, double width = 0.05) const
    {
        AnyNeighborhoodsSimple ret;
        for (int i = 0; i < nbhs.size(); ++i)
        {
            const NeighborhoodSimple &nbh = nbhs[i];
            NeighborhoodSimple nbhnew;
            for (int j = 0; j < nbh.indices_.size(); ++j)
            {
                if (fabs(nbh.distances_[j] - rshell) < width)
                {
                    nbhnew.indices_.push_back(nbh.indices_[j]);
                    nbhnew.distances_.push_back(nbh.distances_[j]);
                }
            }
            ret.nbhs.push_back(nbhnew);
        }
        return ret;
    }
};

#endif //IMRMLP_ANY_NEIGHBORHOODS_H