// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_COMPUTE_CNA_H
#define IMRMLP_COMPUTE_CNA_H

#include <vector>
#include <map>
#include "base/anyneighborhoods.h"
#include "utils/gputils.h"

/****
 * Compute CNA indices for each atom
 * 
 * the basic CNA index contains 4 numbers: ijkl, 
 *      for bonded atoms, i = 1, in this compute only bond atoms, for unbond i = 2
 *      j is common neighbor counts
 *      k is bonds formed by common neighbors
 *      l is the number of bonds in the longest chain of connected bonds
 * or use a CNA variant:
 *      i,j,k as before
 *      l is max number of bonds an atom bond to
 *      m is min number of bonds an atom bond to
 * the later one is simpler to reliaze
 * 
 * if we take only 1nn neighbor and count ubond
 * then for variant 1:
 *  FCC: 12x1421 (bond only)
 *  HCP: 6x1421, 6x2422
 *  ICOS: 
 *  BCC: 8x1666, 6x2422
 *  DIA: 12x1543, 4x2663
 * for variant 2: 
 *  FCC: 12x14211 (1nn only)
 *  HCP: 6x14211, 6x24222
 *  ICOS: 12x(?, maybe 1)5522
 *  BCC: 8x16622, 6x24422
 *  DIA:
 * 
 * for multispecies system, if species info is required, the CNA index contains additional information: ijkluvw
 *      ijkl is same as before, (ii, jj, ll, mm in code)
 *      u means spcie counts
 *      v means number same-type bonds, w means number of diffenerent type bonds
 *      ijkl and uvw are stored separately
 * 
 * if no species info required, then compute HA index for each neighbor pair of one atom, info stores as follows:
 *      index, counts : i.e. 
 *          12 331
 *          5 441
 * 
 * if species info is required, the species info is stored as follows:
 *      mytype bondtype index counts : i.e for Fe-Al
 *          26 13 6 421
 * 
 * TODO: acceleartion, i.e. in some cases one can ensure that l = 1.
 * TODO: A simple CNA, which only recongise fcc, bcc, hcp, diamond, icosahedral, unknown, etc.
 *      BCC: 14 nn atoms
 *      FCC,HCP: 14
 * TODO: smart CNA, choose r_cut automatically
 * TODO: variants in runtime
 * TODO: crystal identification, use variable n1441 etc to replace std::map
 * 
 * ***/

#define CNA_VARIANT_1
//#define CNA_VARIANT_2

//typedef uint64_t cna_index;
typedef int cna_index; // faster

struct cna_index_sp
{
    int chem[2];
    cna_index index;      // ijkl
    cna_index index_type; // uvw
    bool operator<(const cna_index_sp &opt) const
    {
        if (index < opt.index)
            return true;
        if (index > opt.index)
            return false;
        if (index_type < opt.index_type)
            return true;
        if (index_type > opt.index_type)
            return false;
        if (chem[0] < opt.chem[0])
            return true;
        if (chem[0] > opt.chem[0])
            return false;
        if (chem[1] < opt.chem[1])
            return true;
        return false;
    }
};

typedef std::map<cna_index, int> cna_atom;
typedef std::map<cna_index_sp, int> cna_atom_sp;

class ComputeCNA : public ConnectSettings
{
private:
    bool unbond = false;
    bool countsp = false;

public:
    ComputeCNA(/* args */);
    ~ComputeCNA();

    void InitSettings()
    {
        Connect(unbond, "unbond");
        Connect(countsp, "specie_info");
    }
    ComputeCNA(const GPSettings &_settings);
    bool species_information() const { return countsp; }

    std::vector<cna_atom> compute(const AnyNeighborhoods &nhbs); 
    std::vector<cna_atom_sp> compute_sp(const AnyNeighborhoods &nbhs);
};

#endif
