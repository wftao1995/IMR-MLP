// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_cna.h"
#include "common_neighbor_geometry.h"
#ifdef CNA_VARIANT_1
namespace
{
int max_b_length(const std::vector<std::vector<int>> &bonds, int node,
                 std::vector<bool> &used)
{
    int longest = 0;
    const int size = bonds.size();
    for (int next : bonds[node])
    {
        if (used[node * size + next])
            continue;
        // Each undirected bond can occur only once in the chain, including rings.
        used[node * size + next] = used[next * size + node] = true;
        longest = std::max(longest, 1 + max_b_length(bonds, next, used));
        used[node * size + next] = used[next * size + node] = false;
    }
    return longest;
}
}
#endif
ComputeCNA::ComputeCNA(/* args */)
{
}

ComputeCNA::~ComputeCNA()
{
}

ComputeCNA::ComputeCNA(const GPSettings &_settings)
{
    InitSettings();
    ApplySettings(_settings);
    // PrintSettings();
}
// A reference full CNA inplementation, no type info
std::vector<cna_atom> ComputeCNA::compute(const AnyNeighborhoods &nbhs)
{
    std::vector<cna_atom> result;
    int size = nbhs.size();
    result.resize(size);
    int ii, jj, ll, mm;
    namespace Geometry = CommonNeighborGeometry;
#ifdef CNA_VARIANT_2
    int kk;
    std::vector<int> maxbond;
#endif
    for (int i = 0; i < size; ++i)
    {
        const CommonNeighborGeometry::Image root{i, Eigen::RowVector3d::Zero()};
        for (const auto &pair : Geometry::candidates(nbhs, i, unbond))
        {
            if (!Geometry::owns_pair(i, pair)) continue;
            const auto common = Geometry::common(nbhs, i, pair);
            ii = unbond && !Geometry::bonded(nbhs, root, pair) ? 2 : 1;
            jj = common.size();
#ifdef CNA_VARIANT_1
            // Build the common-neighbor graph entirely in local indices.
            std::vector<std::vector<int>> bonds(jj);
            ll = 0;
            for (int k = 0; k < jj; ++k)
            {
                for (int l = k + 1; l < jj; ++l)
                {
                    if (Geometry::bonded(nbhs, common[k], common[l]))
                    {
                        ++ll;
                        bonds[k].push_back(l);
                        bonds[l].push_back(k);
                    }
                }
            }
            mm = 0;
            std::vector<bool> used(jj * jj, false);
            for (int k = 0; k < jj && mm < ll; ++k)
            {
                mm = std::max(mm, max_b_length(bonds, k, used));
            }
            cna_index index;
            // if (mm > 9 || kk > 9 || jj > 9)
            // {

            //     // int_32 can't handle large numbers
            //     index = kk + mm * 50 + ll * 600 + jj * 1000 + ii * 10000;
            // }
            // else
            // {
            index = mm + ll * 10 + jj * 100 + ii * 1000;
            // }
#endif
#ifdef CNA_VARIANT_2
            ll = 0;
            maxbond.clear();
            maxbond.resize(jj);
            for (int k = 0; k < jj; ++k)
            {
                for (int l = k + 1; l < jj;  ++l)
                {
                    if (Geometry::bonded(nbhs, common[k], common[l]))
                    {
                        ++ll;
                        ++maxbond[k];
                    }
                }
            }
            mm = *std::min_element(maxbond.begin(), maxbond.end());
            kk = *std::max_element(maxbond.begin(), maxbond.end());
            cna_index index;
            // if (mm > 9 || kk > 9 || jj > 9)
            // {

            //     // int_32 can't handle large numbers
            //     index = kk + mm * 50 + ll * 600 + jj * 1000 + ii * 10000;
            // }
            // else
            // {
            index = kk + mm * 10 + ll * 100 + jj * 1000 + ii * 10000;
            // }

#endif
            ++result[i][index];
            ++result[pair.source][index];
        }
    }
    return result;
}

// Species-resolved CNA: adds chem[2] and bond-type info to the index
std::vector<cna_atom_sp> ComputeCNA::compute_sp(const AnyNeighborhoods &nbhs)
{
    std::vector<cna_atom_sp> result;
    int size = nbhs.size();
    result.resize(size);
    int ii, jj, ll;
    namespace Geometry = CommonNeighborGeometry;
    for (int i = 0; i < size; ++i)
    {
        const Neighborhood &nbhi = nbhs[i];
        const CommonNeighborGeometry::Image root{i, Eigen::RowVector3d::Zero()};
        for (const auto &pair : CommonNeighborGeometry::candidates(nbhs, i, unbond))
        {
            if (!CommonNeighborGeometry::owns_pair(i, pair)) continue;
            const auto &nbhj = nbhs[pair.source];
            const auto common = CommonNeighborGeometry::common(nbhs, i, pair);
            ii = unbond && !CommonNeighborGeometry::bonded(nbhs, root, pair) ? 2 : 1;
            jj = common.size();
            // Count bonds among common neighbors
            ll = 0;
            std::vector<std::vector<int>> bonds(jj);
            for (int k = 0; k < jj; ++k)
            {
                for (int l = k + 1; l < jj; ++l)
                {
                    if (Geometry::bonded(nbhs, common[k], common[l]))
                    {
                        ++ll;
                        bonds[k].push_back(l);
                        bonds[l].push_back(k);
                    }
                }
            }
            int longest = 0;
            std::vector<bool> used(jj * jj, false);
            for (int k = 0; k < jj && longest < ll; ++k)
                longest = std::max(longest, max_b_length(bonds, k, used));
            // Count same-type vs different-type bonds among common neighbors
            int same_bonds = 0, diff_bonds = 0;
            for (int k = 0; k < jj; ++k)
            {
                for (int l = k + 1; l < jj; ++l)
                {
                    if (Geometry::bonded(nbhs, common[k], common[l]))
                    {
                        if (nbhs[common[k].source].centertype == nbhs[common[l].source].centertype)
                            ++same_bonds;
                        else
                            ++diff_bonds;
                    }
                }
            }
            cna_index_sp idx;
            idx.chem[0] = nbhi.centertype;
            idx.chem[1] = nbhj.centertype;
            idx.index = ii * 1000 + jj * 100 + ll * 10 + longest;
            idx.index_type = jj * 100 + same_bonds * 10 + diff_bonds; // encode type info
            ++result[i][idx];
            std::swap(idx.chem[0], idx.chem[1]);
            ++result[pair.source][idx];
        }
    }
    return result;
}
