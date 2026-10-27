// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_ANY_NEIGHBORHOODS_H
#define IMRMLP_ANY_NEIGHBORHOODS_H

#include <Eigen/Dense>
#include <iosfwd>
#include <vector>

// #define PRE_ALLOC_NBH_SIZE 100
// #define NBH_PREALLOC

struct NeighborhoodSub
{
    int nbhcount_ = 0;                    // number of neighborhoods
    std::vector<int> indices_;           // indices of each atom
    std::vector<Eigen::RowVector3d> pos_; //relative positons of each atom about center atom
    std::vector<double> distances_;       //relative distance about center atom
    int centertype;                       // type of center atom
    int outertype;

    friend std::ostream &operator<<(std::ostream &os, const NeighborhoodSub &nbh);
};

struct Neighborhood
{
    int nbhcount_ = 0;                    // number of neighborhoods
    std::vector<int> indices_;           // indices of each atom
    std::vector<int> chemmap_;            // atomic numbers of each atom
    std::vector<Eigen::RowVector3d> pos_; //relative positons of each atom about center atom
    std::vector<double> distances_;       //relative distance about center atom
    int centertype;                       // type of center atom
#ifdef _ENABLE_FORCE_ONLY_PREDICTION_
    Vector3d force_; //force on each atom
#endif
#ifdef NBH_PREALLOC
    Neighborhood()
    {
        indices_.reserve(PRE_ALLOC_NBH_SIZE);
        chemmap_.reserve(PRE_ALLOC_NBH_SIZE);
        pos_.reserve(PRE_ALLOC_NBH_SIZE);
        distances_.reserve(PRE_ALLOC_NBH_SIZE);
#ifdef _ENABLE_FORCE_ONLY_PREDICTION_
        force_.reserve(PRE_ALLOC_NBH_SIZE);
#endif
    }
#endif
    friend std::ostream &operator<<(std::ostream &os, const Neighborhood &nhb);

    NeighborhoodSub get_sub_by_type(const int &t) const
    {
        NeighborhoodSub ret;
        ret.centertype = centertype;
        ret.outertype = t;
        for (int i = 0; i < nbhcount_; ++i)
        {
            if (chemmap_[i] == t)
            {
                ret.distances_.push_back(distances_[i]);
                ret.indices_.push_back(indices_[i]);
                ret.pos_.push_back(pos_[i]);
                ++ret.nbhcount_;
            }
        }
        return ret;
    }
};

class AnyNeighborhoods
{
protected:
    std::vector<Neighborhood> nbhs;

public:
    inline int size() const
    {
        return nbhs.size();
    };

    inline void resize(size_t t)
    {
        nbhs.resize(t);
    }

    inline Neighborhood &operator[](int index)
    {
        return nbhs[index];
    }

    inline const Neighborhood &operator[](int index) const
    {
        return nbhs[index];
    }
    //friend ostream &operator<<(ostream &os, const Neighborhoods &desp);
    friend std::ostream &operator<<(std::ostream &os, const AnyNeighborhoods &nhb);
    AnyNeighborhoods(){};
    virtual ~AnyNeighborhoods(){};
    AnyNeighborhoods(const AnyNeighborhoods&) = default;
    AnyNeighborhoods& operator=(const AnyNeighborhoods&) = default;
    AnyNeighborhoods(AnyNeighborhoods&&) = default;
    AnyNeighborhoods& operator=(AnyNeighborhoods&&) = default;

    AnyNeighborhoods get_neighbor_shell(double rshell, double width = 0.05) const;
    AnyNeighborhoods sub_neighborhoods(const double rcut) const;
    AnyNeighborhoods sub_neighborhoods(const double rinn, const double rout) const;
    AnyNeighborhoods sub_neighborhoods(const int &nn) const;
};

#endif //IMRMLP_ANY_NEIGHBORHOODS_H
