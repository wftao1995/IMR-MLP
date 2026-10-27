// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_COMPUTE_BOND_ANGLE_H
#define IMRMLP_COMPUTE_BOND_ANGLE_H

#include <vector>
#include <map>
#include <set>
#include "base/anyneighborhoods.h"
#include "utils/gputils.h"

struct bondangle_type_index
{
    int chem[3];

    bondangle_type_index(int s1, int s2, int s3)
    {
        chem[1] = s2;
        if (s1 <= s3)
        {
            chem[0] = s1;
            chem[2] = s3;
        }
        else
        {
            chem[0] = s3;
            chem[2] = s1;
        }
    }

    bool operator<(const bondangle_type_index &opt) const
    {
        if (chem[1] < opt.chem[1])
            return true;
        if (chem[1] > opt.chem[1])
            return false;
        if (chem[0] < opt.chem[0])
            return true;
        if (chem[0] > opt.chem[0])
            return false;
        if (chem[2] < opt.chem[2])
            return true;
        return false;
    }
};

struct dihedral_type_index
{
    int chem[4];

    dihedral_type_index(int s1, int s2, int s3, int s4)
    {
        // s2 <= s3, s2,s3 are center, s1, s4 are bonded atoms, when s2 == s3, s1 <= s4
        if (s2 > s3)
        {
            chem[1] = s3;
            chem[2] = s2;
            chem[0] = s4;
            chem[3] = s1;
        }
        else if (s2 == s3)
        {
            chem[1] = s2;
            chem[2] = s3;
            if (s1 > s4)
            {
                chem[0] = s4;
                chem[3] = s1;
            }
            else
            {
                chem[0] = s1;
                chem[3] = s4;
            }
        }
        else
        {
            chem[0] = s1;
            chem[1] = s2;
            chem[2] = s3;
            chem[3] = s4;
        }
    }

    bool operator<(const dihedral_type_index &opt) const
    {
        if (chem[1] < opt.chem[1])
            return true;
        if (chem[1] > opt.chem[1])
            return false;
        if (chem[2] < opt.chem[2])
            return true;
        if (chem[2] > opt.chem[2])
            return false;
        if (chem[0] < opt.chem[0])
            return true;
        if (chem[0] > opt.chem[0])
            return false;
        if (chem[3] < opt.chem[3])
            return true;
        return false;
    }
};

typedef std::map<bondangle_type_index, std::vector<int>> bondangle_type;
typedef std::map<bondangle_type_index, ulong> bondangle_count;
typedef std::map<dihedral_type_index, std::vector<int>> dihedral_type;
typedef std::map<dihedral_type_index, ulong> dihedral_count;

void add2t1(bondangle_type &a, bondangle_type &b);
void add2t1(bondangle_count &a, bondangle_count &b);
void add2t1(dihedral_type &a, dihedral_type &b);
void add2t1(dihedral_count &a, dihedral_count &b);

class ComputeAngles : public ConnectSettings
{
private:
    double sep = M_PI / 180; // in rad
    double low = 0.0;
    double high = M_PI;
    int nsize;
    double prec = 1e-5;
    std::set<bondangle_type_index> bond_types_;
    std::set<dihedral_type_index> dihedral_types_;

    int get_id(double x) const;
    bondangle_type compute_selected(const AnyNeighborhoods &, bondangle_count &, double **) const;
    dihedral_type compute_selected(const AnyNeighborhoods &, dihedral_count &, double **) const;

public:
    std::vector<double> angles;
    int ntypes = 3;
    ComputeAngles(/* args */) { init(); };
    ComputeAngles(double sep, double low, double high, int ntypes);
    ~ComputeAngles(){};
    void init();

    void InitSettings()
    {
        Connect(sep, "sep");
        Connect(low, "low");
        Connect(high, "high");
        Connect(ntypes, "ntypes");
        Connect(prec, "prec");
    }
    ComputeAngles(const GPSettings &_settings);

    // Empty selections retain all combinations. Existing keys canonicalize
    // bond endpoints and reversal of a complete dihedral chain.
    void set_bondangle_types(const std::set<bondangle_type_index> &types) { bond_types_ = types; }
    void set_dihedral_types(const std::set<dihedral_type_index> &types) { dihedral_types_ = types; }

    bondangle_type compute(const AnyNeighborhoods &nhbs, bondangle_count &count) const;
    bondangle_type compute(const AnyNeighborhoods &nhbs, bondangle_count &count, double **rcutij) const;
    dihedral_type compute(const AnyNeighborhoods &nbhs, dihedral_count &count) const;
    dihedral_type compute(const AnyNeighborhoods &nbhs, dihedral_count &count, double **rcutij) const;
    void initbret(bondangle_type &bret) const;
    void initdret(dihedral_type &dret) const;
};

#endif
