// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "anyneighborhoods.h"
#include "tools/chem.h"
#include "tools/tools.h"
#include <algorithm>
#include <numeric>

std::ostream &operator<<(std::ostream &os, const NeighborhoodSub &nhb)
{
    os.setf(std::ios_base::fixed, std::ios_base::floatfield);
    os << "Size: " << nhb.nbhcount_ << '\n';
    os << "Center Type: " << Chemicals::chemical_symbols[nhb.centertype] << '\n';
    for (int i = 0; i < nhb.nbhcount_; ++i)
    {
        os << " ID: " << i
           << "  " << nhb.distances_[i]
           << "  " << nhb.indices_[i]
           << "  " << nhb.pos_[i]
           << '\n';
    }
    os << '\n';
    return os;
}

std::ostream &operator<<(std::ostream &os, const Neighborhood &nhb)
{
    os.setf(std::ios_base::fixed, std::ios_base::floatfield);
    os << "Size: " << nhb.nbhcount_ << '\n';
    os << "Center Type: " << Chemicals::chemical_symbols[nhb.centertype] << '\n';
    for (int i = 0; i < nhb.nbhcount_; ++i)
    {
        os << " ID: " << i
           << "  " << Chemicals::chemical_symbols[nhb.chemmap_[i]]
           << "  " << nhb.distances_[i]
           << "  " << nhb.indices_[i]
           << "  " << nhb.pos_[i]
           << '\n';
    }
    os << '\n';
    return os;
}

std::ostream &operator<<(std::ostream &os, const AnyNeighborhoods &nhb)
{
    os << "Neighborhoods: " << nhb.size() << std::endl;
    os.setf(std::ios_base::fixed, std::ios_base::floatfield);
    //os.width(10);
    os << std::endl;
    for (int i = 0; i < nhb.size(); ++i)
    {
        os << " ID: " << i << std::endl;
        os << " Count: " << nhb[i].nbhcount_ << std::endl;
        os << " Centertype: " << Chemicals::chemical_symbols[nhb[i].centertype] << std::endl;
        os << " Atoms: " << std::endl;
        for (int j = 0; j < nhb[i].nbhcount_; ++j)
        {
            os << "  " << Chemicals::chemical_symbols[nhb[i].chemmap_[j]]
               << "  " << nhb[i].distances_[j]
               << "  " << nhb[i].indices_[j]
               << "  " << nhb[i].pos_[j]
               << "  " << nhb[i].distances_[j]
               << std::endl;
        }
        os << std::endl;
    }
    return os;
}

AnyNeighborhoods AnyNeighborhoods::get_neighbor_shell(double rshell, double width) const
{
    AnyNeighborhoods ret;
    for (int i = 0; i < nbhs.size(); ++i)
    {
        const Neighborhood &nhb = nbhs[i];
        Neighborhood nbhnew;
        nbhnew.centertype = nhb.centertype;
        nbhnew.nbhcount_ = 0;
        for (int j = 0; j < nhb.nbhcount_; ++j)
        {
            if (fabs(nhb.distances_[j] - rshell) < width)
            {
                nbhnew.nbhcount_++;
                nbhnew.chemmap_.push_back(nhb.chemmap_[j]);
                nbhnew.indices_.push_back(nhb.indices_[j]);
                nbhnew.distances_.push_back(nhb.distances_[j]);
                nbhnew.pos_.push_back(nhb.pos_[j]);
            }
        }
        ret.nbhs.push_back(nbhnew);
    }
    return ret;
}

AnyNeighborhoods AnyNeighborhoods::sub_neighborhoods(const double rcut) const
{
    AnyNeighborhoods ret;
    ret.resize(nbhs.size());
    for (int i = 0; i < nbhs.size(); ++i)
    {
        Neighborhood nbhnew;
        const Neighborhood &nhb = nbhs[i];
        nbhnew.centertype = nhb.centertype;
        nbhnew.nbhcount_ = 0;
        for (int j = 0; j < nhb.nbhcount_; ++j)
        {
            if (nhb.distances_[j] <= rcut)
            {
                nbhnew.nbhcount_++;
                nbhnew.chemmap_.push_back(nhb.chemmap_[j]);
                nbhnew.distances_.push_back(nhb.distances_[j]);
                nbhnew.pos_.push_back(nhb.pos_[j]);
                nbhnew.indices_.push_back(nhb.indices_[j]);
            }
        }
        // ret.nbhs.push_back(nbhnew);
        ret.nbhs[i] = nbhnew;
    }
    return ret;
}

AnyNeighborhoods AnyNeighborhoods::sub_neighborhoods(const int &nn) const {
    AnyNeighborhoods ret;
    ret.resize(nbhs.size());
    for (int i = 0; i < nbhs.size(); ++i)
    {
        Neighborhood nbhnew;
        nbhnew.nbhcount_ = 0;
        const Neighborhood &nhb = nbhs[i];
        nbhnew.centertype = nhb.centertype;
        if (nhb.nbhcount_ <= nn) {
            // ret.nbhs.push_back(nbhs[i]);
            ret.nbhs[i] = nbhs[i];
        }
        else {
            std::vector<int> inds(nbhs[i].nbhcount_);
            auto &rdists = nbhs[i].distances_;
            auto &pos = nbhs[i].pos_;
            auto &indices = nbhs[i].indices_;
            auto &chemmap = nbhs[i].chemmap_;
            std::iota(inds.begin(), inds.end(), 0);
            std::partial_sort(inds.begin(), inds.begin()+nn, inds.end(),
                              [&rdists](int i, int j) {return rdists[i] < rdists[j];});
            nbhnew.nbhcount_ = nn;
            nbhnew.indices_.resize(nn);
            nbhnew.chemmap_.resize(nn);
            nbhnew.distances_.resize(nn);
            nbhnew.pos_.resize(nn);
            for (int k = 0; k < nn; ++k) {
                int idx = inds[k];
                nbhnew.indices_[k] = indices[idx];
                nbhnew.chemmap_[k] = chemmap[idx];
                nbhnew.distances_[k] = rdists[idx];
                nbhnew.pos_[k] = pos[idx];
            }
            ret.nbhs[i] = nbhnew;
        }
    }
    return ret;
}

AnyNeighborhoods AnyNeighborhoods::sub_neighborhoods(const double rinn, const double rout) const
{
    AnyNeighborhoods ret;
    ret.resize(nbhs.size());
    for (int i = 0; i < nbhs.size(); ++i)
    {
        Neighborhood nbhnew;
        const Neighborhood &nhb = nbhs[i];
        nbhnew.centertype = nhb.centertype;
        nbhnew.nbhcount_ = 0;
        for (int j = 0; j < nhb.nbhcount_; ++j)
        {
            if (nhb.distances_[j] <= rout && nhb.distances_[j] >= rinn)
            {
                nbhnew.nbhcount_++;
                nbhnew.chemmap_.push_back(nhb.chemmap_[j]);
                nbhnew.distances_.push_back(nhb.distances_[j]);
                nbhnew.pos_.push_back(nhb.pos_[j]);
                nbhnew.indices_.push_back(nhb.indices_[j]);
            }
        }
        // ret.nbhs.push_back(nbhnew);
        ret.nbhs[i] = nbhnew;
    }
    return ret;
}