// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_sro.h"
#include "tools/procstr.h"
#include <functional>

namespace
{
// get_orders stores each shell's smallest distance. Use those same groups when
// counting, so a bond cannot also enter the next shell's overlapping window.
int shell_order(const std::vector<double> &radii, double distance)
{
    const auto next = std::upper_bound(radii.begin(), radii.end(), distance);
    return next == radii.begin() ? -1 : static_cast<int>(next - radii.begin()) - 1;
}
}

void ComputeSRO::ExtraSettings(const GPSettings &_settings)
{
    if (_settings["nnorders"] != "")
    {
        std::vector<std::string> splstr;
        str_split(_settings["nnorders"], splstr);
        nnorders.clear();
        for (int i = 1; i < (int)splstr.size(); ++i)
        {
            nnorders.push_back(std::stoi(splstr[i]));
        }
    }
}

ComputeSRO::ComputeSRO(const GPSettings &_settings)
{
    InitSettings();
    ApplySettings(_settings);
    ExtraSettings(_settings);
    if (tmpsrotype != "")
    {
        sro_type = parse_sro_type(tmpsrotype);
    }
    std::cout << "SRO type is " << tmpsrotype << std::endl;
    // PrintSettings();
}

// get_orders is now a template in sro_common.h — works for both
// Neighborhood and NeighborhoodSimple

std::vector<pair_srotype> ComputeSRO::compute(const Atoms &atom, const AnyNeighborhoods &nbhs, std::map<sro_counter, size_t> *pcount)
{
    std::map<int, int> spmap = atom.types_count_map();
    int total_atoms = atom.size();
    std::map<int, double> concen = compute_concentrations(spmap, total_atoms);
    std::vector<double> mdist;
    int norder;
    get_orders(nbhs, mdist, norder, width);
    // init
    std::vector<pair_srotype> ret;
    std::map<sro_counter, size_t> counter;
    sro_counter cttmp;

    // size_t total_pairs = 0;
    std::map<int, size_t> total_pairs; // order : count
    for (int j = 0; j < nbhs.size(); ++j)
    {
        const Neighborhood &nbh = nbhs[j];
        cttmp.typea = nbh.centertype;
        for (int k = 0; k < nbh.nbhcount_; ++k)
        {
            cttmp.order = shell_order(mdist, nbh.distances_[k]);
            if (cttmp.order < 0)
                continue;
            cttmp.typeb = nbh.chemmap_[k];
            counter[cttmp]++;
            total_pairs[cttmp.order]++;
        }
    }
    // analyse and return results
    for (int o = 0; o < norder; ++o)
    {
        for (auto it = spmap.begin(); it != spmap.end(); ++it)
        {
            for (auto it2 = it; it2 != spmap.end(); ++it2)
            {
                pair_srotype tmp;
                cttmp.order = o;
                cttmp.typea = it->first;
                cttmp.typeb = it2->first;
                tmp.chema = it->first;
                tmp.chemb = it2->first;
                tmp.order = o;
                tmp.sro = compute_sro_value(sro_type,
                    static_cast<double>(counter[cttmp]),
                    static_cast<double>(total_pairs[o]),
                    concen[it->first], concen[it2->first],
                    tmp.chema == tmp.chemb);
                ret.push_back(tmp);
            }
        }
    }
    if (pcount != nullptr)
    {
        *pcount = counter;
    }
    return ret;
}
// forward set to true if use neighborhood_cluster
std::vector<pair_srotype> ComputeSRO::compute(const Atoms &atom, const AnyNeighborhoodsSimple &nbhs, std::map<sro_counter, size_t> *pcount, bool forward)
{
    std::map<int, int> spmap = atom.types_count_map();
    int total_atoms = atom.size();
    std::map<int, double> concen = compute_concentrations(spmap, total_atoms);
    std::vector<double> mdist;
    int norder;
    get_orders(nbhs, mdist, norder, width);
    // init
    std::vector<pair_srotype> ret;
    std::map<sro_counter, size_t> counter;
    sro_counter cttmp;

    // size_t total_pairs = 0;
    std::map<int, size_t> total_pairs; // order : count

    for (int j = 0; j < nbhs.size(); ++j)
    {
        const NeighborhoodSimple &nbh = nbhs[j];
        for (int k = 0; k < nbh.indices_.size(); ++k)
        {
            cttmp.order = shell_order(mdist, nbh.distances_[k]);
            if (cttmp.order < 0)
                continue;
            cttmp.typea = atom.chemmap_[j];
            cttmp.typeb = atom.chemmap_[nbh.indices_[k]];
            counter[cttmp]++;
            total_pairs[cttmp.order]++;
            if (forward)
            {
                std::swap(cttmp.typea, cttmp.typeb);
                counter[cttmp]++;
                total_pairs[cttmp.order]++;
            }
        }
    }
    // analyse and return results
    for (int o = 0; o < norder; ++o)
    {
        for (auto it = spmap.begin(); it != spmap.end(); ++it)
        {
            for (auto it2 = it; it2 != spmap.end(); ++it2)
            {
                pair_srotype tmp;
                cttmp.order = o;
                cttmp.typea = it->first;
                cttmp.typeb = it2->first;
                tmp.chema = it->first;
                tmp.chemb = it2->first;
                tmp.order = o;
                tmp.sro = compute_sro_value(sro_type,
                    static_cast<double>(counter[cttmp]),
                    static_cast<double>(total_pairs[o]),
                    concen[it->first], concen[it2->first],
                    tmp.chema == tmp.chemb);
                ret.push_back(tmp);
            }
        }
    }
    if (pcount != nullptr)
    {
        *pcount = counter;
    }
    return ret;
}
