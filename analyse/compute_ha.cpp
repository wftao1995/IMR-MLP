// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/****
 * This is an C++ implementation of PA.F with many imporvements
 *
 * Written by Jiantao Wang (wftao1995@gmail.com)
 *
*****/

/*------------------------------------------------------------------------------
 * Method to compute the Honeycutt-Andersen index based on the pca neighbors.
 * The first index 1 means the selected pair are bonded, the second gives the
 * number of common neighbors between these two, and the third gives the number
 * of bonds formed between the common neighbors; the fourth are generally 1, while
 * for 144?, if all common neighbors are linked together, it is set to 2.
 * For 142?, i-j, k-l are count as 1, i-j-k l are count as 1.
 *
 *----------------------------------------------------------------------------*/

#include "compute_ha.h"
#include "common_neighbor_geometry.h"
#include "tools/tools.h"
#include <cstdlib>
using namespace std;
namespace geometry = CommonNeighborGeometry;
//n1201, n1211, n1301, n1311, n1321, n1331, n1421, n1422, n1431, n1441, n1541, n1551, n1661, n1771;
ComputeHA::ComputeHA(const GPSettings &_settings)
{
    InitSettings();
    ApplySettings(_settings);
    // PrintSettings();
}

ComputeHA::Selection::Selection(const std::vector<haindexT> &indices) {
    for (int code : indices) {
        if (code <= 0) GPERROR("HA indices must be positive category codes, not atom indices");
        codes.insert(code);
        const int ii=code/1000, jj=(code/100)%10, ll=(code/10)%10, mm=code%10;
        if ((ii != 1 && ii != 2) || (mm != 1 && mm != 2)) ordinary_only=false;
        first.insert(ii); common.emplace(ii,jj); bonds.emplace(ii,jj,ll);
    }
}

ha_index_image ComputeHA::compute_selected(const AnyNeighborhoods &nbhs, HAWorkStats *work) {
    const Selection selection(selected_indices_);
    if (work) *work = HAWorkStats{};
    ha_index_image result;
    result.all_count = 0;
    if (selection.active()) for (int code : selection.codes) result.total[code]=0;
    else result.total = INITIAL_HA_MAP;
    for (int i = 0; i < nbhs.size(); ++i) compute_impl(nbhs,result,i,selection,work);
    return result;
}

ha_index_image ComputeHA::compute(AnyNeighborhoods &nbhs) {
    return compute_selected(nbhs,nullptr);
}
ha_index_image ComputeHA::compute(AnyNeighborhoods &nbhs, HAWorkStats &work) {
    return compute_selected(nbhs,&work);
}

namespace {
AnyNeighborhoods filter_ha_graph(const AnyNeighborhoods &nbhs, double **cutoffij) {

    // Filter the entire bond graph so candidate pairs, common neighbors, and
    // bonds between common neighbors all use the same pair cutoffs.
    AnyNeighborhoods filtered;
    filtered.resize(nbhs.size());
    for (int i = 0; i < nbhs.size(); ++i)
    {
        const auto &source = nbhs[i];
        auto &target = filtered[i];
        target.centertype = source.centertype;
        for (int j = 0; j < source.nbhcount_; ++j)
        {
            if (source.distances_[j] > cutoffij[source.centertype][source.chemmap_[j]])
                continue;
            target.indices_.push_back(source.indices_[j]);
            target.chemmap_.push_back(source.chemmap_[j]);
            target.pos_.push_back(source.pos_[j]);
            target.distances_.push_back(source.distances_[j]);
            ++target.nbhcount_;
        }
    }
    return filtered;
}
}
ha_index_image ComputeHA::compute(AnyNeighborhoods &nbhs, double **cutoffij) {
    const auto filtered=filter_ha_graph(nbhs,cutoffij);
    return compute_selected(filtered,nullptr);
}
ha_index_image ComputeHA::compute(AnyNeighborhoods &nbhs, double **cutoffij, HAWorkStats &work) {
    const auto filtered=filter_ha_graph(nbhs,cutoffij);
    return compute_selected(filtered,&work);
}

// ── Shared implementation for both compute() overloads ──────────────────
void ComputeHA::compute_impl(const AnyNeighborhoods &nbhs,
                              ha_index_image &result, int i, const Selection &selection, HAWorkStats *work)
{
    const auto &nbhi = nbhs[i];
    const geometry::Image root{i, Eigen::RowVector3d::Zero()};
    for (const auto &pair : geometry::candidates(nbhs, i, count_ubond))
    {
        // counting pairs — avoid two-fold processing
        if (!geometry::owns_pair(i, pair))
            continue;
        ++result.all_count;
        if (work) ++work->owned_pairs;
        const Neighborhood &nbhj = nbhs[pair.source];
        int ii = 1;
        if (count_ubond)
        {
            if (!geometry::bonded(nbhs, root, pair))
                ii = 2;
        }
        if (selection.active() && selection.ordinary_only && !selection.first.contains(ii)) continue;
        if (work) ++work->common_sets;
        const auto common = geometry::common(nbhs, i, pair);
        int jj = (int)common.size();
        if (selection.active() && selection.ordinary_only && !selection.common.contains({ii,jj})) continue;
        // counting bonds among common neighbors
        int ll = 0;
        bool all_link = true;
        bool if422 = false;
        if (jj == 4)
        {
            for (int k = 0; k < jj; ++k)
            {
                int ct = 0;
                for (int l = 0; l < jj; ++l)
                {
                    if (work) ++work->common_bond_checks;
                    if (geometry::bonded(nbhs, common[k], common[l]))
                    {
                        ++ll;
                        ++ct;
                    }
                    else
                        all_link = false;
                }
                if (ct == 2)
                    if422 = true;
            }
            ll = ll / 2;
        }
        else
        {
            for (int k = 0; k < jj; ++k)
            {
                for (int l = k + 1; l < jj; ++l)
                {
                    if (work) ++work->common_bond_checks;
                    if (geometry::bonded(nbhs, common[k], common[l]))
                        ++ll;
                    else
                        all_link = false;
                }
            }
        }
        if (selection.active() && selection.ordinary_only && !selection.bonds.contains({ii,jj,ll})) continue;
        if (work) ++work->fourth_classifications;
        // for ?44? and ?42?, find mm
        int mm = 1;
        if (all_link && jj == 4 && ll == 4)
            mm = 2;
        if (jj == 4 && ll == 2 && if422)
            mm = 2;
        // merge ijlm → HA index
        haindexT ind;
        if (jj > 9 || ll > 9)
            ind = 100000 * ii + 1000 * jj + 10 * ll + mm;
        else
            ind = 1000 * ii + 100 * jj + 10 * ll + mm;
        if (selection.active() && !selection.codes.contains(ind)) continue;
        result.total[ind] += 1;
#ifndef NO_SAMPLE_STR
        if (sample_one && result.structure.count(ind) == 0)
        {
            haindex_pos indpos;
            indpos.index = ind;
            indpos.pos.clear();
            indpos.pos.resize(common.size() + 2);
            indpos.pos[0] = Eigen::RowVector3d::Zero();
            indpos.pos[1] = pair.position;
            for (int tmp = 0; tmp < (int)common.size(); ++tmp)
                indpos.pos[tmp + 2] = common[tmp].position;
            result.structure[ind] = indpos;
        }
#endif
        // bond-type partial
        haindex_bond_type haindb;
        haindb.chem[0] = nbhi.centertype;
        haindb.chem[1] = nbhj.centertype;
        if (haindb.chem[0] > haindb.chem[1])
        {
            int tmp = haindb.chem[1];
            haindb.chem[1] = haindb.chem[0];
            haindb.chem[0] = tmp;
        }
        haindb.index = ind;
        result.partial[haindb] += 1;
    }
}
