// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_bondangle.h"
#include "common_neighbor_geometry.h"
#include <algorithm>
#include <cmath>
#include <limits>

ComputeAngles::ComputeAngles(const GPSettings &_settings)
{
    InitSettings();
    ApplySettings(_settings);
    // PrintSettings();
    init();
}

ComputeAngles::ComputeAngles(double _sep, double _low, double _high, int _ntypes) : sep(_sep), low(_low), high(_high), ntypes(_ntypes)
{
    init();
}

void ComputeAngles::init()
{
    if (!std::isfinite(sep) || sep <= 0 || !std::isfinite(low) ||
        !std::isfinite(high) || high < low)
        GPERROR("Angle histogram requires a positive finite interval and finite ordered bounds (radians)");
    // Grid points include the upper bound when it is a multiple of sep.
    const double count = std::floor((high - low) / sep) + 1;
    if (count > std::numeric_limits<int>::max())
        GPERROR("Angle histogram has too many bins");
    angles.clear();
    nsize = static_cast<int>(count);
    angles.resize(nsize);
    for (int i = 0; i < nsize; ++i)
    {
        angles[i] = low + sep * i;
    }
}

int ComputeAngles::get_id(double x) const
{
    // A partial final interval still belongs to the last available grid point.
    return std::clamp(static_cast<int>(std::round((x - low) / sep)), 0, nsize - 1);
}

void ComputeAngles::initbret(bondangle_type &bret) const
{
    bret.clear();
    std::vector<int> ivec(nsize, 0);
    if (!bond_types_.empty()) {
        for (const auto &key : bond_types_) bret[key] = ivec;
        return;
    }
    // j-i-k
    for (int i = 0; i < ntypes; i++)
    {
        for (int j = 0; j < ntypes; j++)
        {
            for (int k = j; k < ntypes; k++)
            {
                bondangle_type_index bind(j, i, k);
                bret[bind] = ivec;
            }
        }
    }
}
void ComputeAngles::initdret(dihedral_type &dret) const
{
    dret.clear();
    std::vector<int> ivec(nsize, 0);
    if (!dihedral_types_.empty()) {
        for (const auto &key : dihedral_types_) dret[key] = ivec;
        return;
    }
    // k-i-j-l, i<=j
    for (int i = 0; i < ntypes; i++)
    {
        for (int j = i; j < ntypes; j++)
        {
            for (int k = 0; k < ntypes; k++)
            {
                for (int l = 0; l < ntypes; l++)
                {
                    if (i == j && k > l)
                        continue;
                    dihedral_type_index dind(k, i, j, l);
                    dret[dind] = ivec;
                }
            }
        }
    }
}
namespace {
// Buckets contain original edge slots, never source atom IDs: different images
// and even repeated slots must keep their original multiplicity.
using SpeciesSlots = std::map<int, std::vector<int>>;
SpeciesSlots species_slots(const Neighborhood &site, double **cutoffs) {
    SpeciesSlots slots;
    for (int edge = 0; edge < site.nbhcount_; ++edge) {
        if (cutoffs && site.distances_[edge] > cutoffs[site.centertype][site.chemmap_[edge]]) continue;
        slots[site.chemmap_[edge]].push_back(edge);
    }
    return slots;
}
const std::vector<int> &slots_for(const SpeciesSlots &slots, int species) {
    static const std::vector<int> empty;
    const auto found = slots.find(species);
    return found == slots.end() ? empty : found->second;
}
}

bondangle_type ComputeAngles::compute_selected(const AnyNeighborhoods &nhbs,
                                               bondangle_count &count, double **cutoffs) const {
    bondangle_type ret;
    initbret(ret);
    for (int i = 0; i < nhbs.size(); ++i) {
        const auto &nbh = nhbs[i];
        // Do not build buckets for centers absent from the selection.
        const auto first = bond_types_.lower_bound(bondangle_type_index(
            std::numeric_limits<int>::min(), nbh.centertype, std::numeric_limits<int>::min()));
        if (first == bond_types_.end() || first->chem[1] != nbh.centertype) continue;
        const auto slots = species_slots(nbh, cutoffs);
        for (auto key = first; key != bond_types_.end() && key->chem[1] == nbh.centertype; ++key) {
            const auto &right = slots_for(slots, key->chem[2]);
            for (int a : slots_for(slots, key->chem[0])) {
                auto begin = key->chem[0] == key->chem[2]
                    ? std::upper_bound(right.begin(), right.end(), a) : right.begin();
                for (auto edge = begin; edge != right.end(); ++edge) {
                    const int b = *edge;
                    // Retain the original arithmetic ordering even for unlike species.
                    const int j = std::min(a, b), k = std::max(a, b);
                    double cval = nbh.pos_[j].dot(nbh.pos_[k]) / nbh.pos_[j].norm() / nbh.pos_[k].norm();
                    double angle = std::acos(std::clamp(cval, -1.0, 1.0));
                    if (angle < low || angle > high) continue;
                    ret.at(*key)[get_id(angle)]++;
                    count[*key]++;
                }
            }
        }
    }
    return ret;
}

dihedral_type ComputeAngles::compute_selected(const AnyNeighborhoods &nbhs,
                                              dihedral_count &count, double **cutoffs) const {
    dihedral_type ret;
    initdret(ret);
    // Explicit orientations preserve whole-chain reversal, including equal
    // central species with unlike endpoints. Sets prevent double enumeration.
    using Endpoints = std::set<std::pair<int, int>>;
    std::map<int, std::map<int, Endpoints>> selected;
    for (const auto &key : dihedral_types_) {
        selected[key.chem[1]][key.chem[2]].insert({key.chem[0], key.chem[3]});
        selected[key.chem[2]][key.chem[1]].insert({key.chem[3], key.chem[0]});
    }
    std::vector<SpeciesSlots> buckets(nbhs.size());
    for (int i = 0; i < nbhs.size(); ++i)
        if (selected.contains(nbhs[i].centertype)) buckets[i] = species_slots(nbhs[i], cutoffs);
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &nbhi = nbhs[i];
        const auto center = selected.find(nbhi.centertype);
        if (center == selected.end()) continue;
        SpeciesSlots central_edges;
        for (int j = 0; j < nbhi.nbhcount_; ++j) {
            if (cutoffs && nbhi.distances_[j] > cutoffs[nbhi.centertype][nbhi.chemmap_[j]]) continue;
            const int pair_type = nbhs[nbhi.indices_[j]].centertype;
            if (center->second.contains(pair_type)) central_edges[pair_type].push_back(j);
        }
        const CommonNeighborGeometry::Image root{i, Eigen::RowVector3d::Zero()};
        for (const auto &[pair_type, endpoints] : center->second) {
            for (int j : slots_for(central_edges, pair_type)) {
                const CommonNeighborGeometry::Image pair{nbhi.indices_[j], nbhi.pos_[j]};
                if (!CommonNeighborGeometry::owns_pair(i, pair)) continue;
                const auto &nbhj = nbhs[nbhi.indices_[j]];
                for (const auto &[left, right] : endpoints) {
                    for (int k : slots_for(buckets[i], left)) {
                        if (k == j) continue;
                        for (int l : slots_for(buckets[nbhi.indices_[j]], right)) {
                            const CommonNeighborGeometry::Image endpoint{
                                nbhj.indices_[l], pair.position + nbhj.pos_[l]};
                            if (CommonNeighborGeometry::same_image(endpoint, root)) continue;
                            dihedral_type_index dindex(nbhi.chemmap_[k], nbhi.centertype, nbhj.centertype, nbhj.chemmap_[l]);
                            auto v1 = nbhi.pos_[j].cross(nbhi.pos_[k]);
                            auto v2 = (-nbhi.pos_[j]).cross(nbhj.pos_[l]);
                            double v1n = v1.norm(), v2n = v2.norm();
                            if (fabs(v1n) < 1e-8 || fabs(v2n) < 1e-8) continue;
                            double cval = v1.dot(v2) / v1n / v2n;
                            double angle = std::acos(std::clamp(cval, -1.0, 1.0));
                            if (angle < low || angle > high) continue;
                            ret.at(dindex)[get_id(angle)]++;
                            count[dindex]++;
                        }
                    }
                }
            }
        }
    }
    return ret;
}

// types must be 0 1 2 3 4 ..
bondangle_type ComputeAngles::compute(const AnyNeighborhoods &nhbs, bondangle_count &count) const
{
    if (!bond_types_.empty()) return compute_selected(nhbs, count, nullptr);
    bondangle_type ret;
    initbret(ret);
    for (int i = 0; i < nhbs.size(); ++i)
    {
        const Neighborhood &nbh = nhbs[i];
        for (int j = 0; j < nbh.nbhcount_; ++j)
        {
            for (int k = j + 1; k < nbh.nbhcount_; ++k)
            {
                bondangle_type_index bind(nbh.chemmap_[j], nbh.centertype, nbh.chemmap_[k]);
                if (!bond_types_.empty() && !bond_types_.contains(bind)) continue;
                double cval = nbh.pos_[j].dot(nbh.pos_[k]) / nbh.pos_[j].norm() / nbh.pos_[k].norm();
                double angle = std::acos(std::clamp(cval, -1.0, 1.0));
                if (angle < low || angle > high)
                    continue;
                int ida = get_id(angle);
                ret[bind][ida]++;
                count[bind]++;
            }
        }
    }
    return ret;
}

bondangle_type ComputeAngles::compute(const AnyNeighborhoods &nhbs, bondangle_count &count, double **rcutij) const
{
    if (!bond_types_.empty()) return compute_selected(nhbs, count, rcutij);
    bondangle_type ret;
    initbret(ret);
    for (int i = 0; i < nhbs.size(); ++i)
    {
        const Neighborhood &nbh = nhbs[i];
        for (int j = 0; j < nbh.nbhcount_; ++j)
        {
            if (nbh.distances_[j] > rcutij[nbh.centertype][nbh.chemmap_[j]])
                continue;
            for (int k = j + 1; k < nbh.nbhcount_; ++k)
            {
                if (nbh.distances_[k] > rcutij[nbh.centertype][nbh.chemmap_[k]])
                    continue;
                bondangle_type_index bind(nbh.chemmap_[j], nbh.centertype, nbh.chemmap_[k]);
                if (!bond_types_.empty() && !bond_types_.contains(bind)) continue;
                double cval = nbh.pos_[j].dot(nbh.pos_[k]) / nbh.pos_[j].norm() / nbh.pos_[k].norm();
                double angle = std::acos(std::clamp(cval, -1.0, 1.0));
                if (angle < low || angle > high)
                    continue;
                int ida = get_id(angle);
                ret[bind][ida]++;
                count[bind]++;
            }
        }
    }
    return ret;
}

dihedral_type ComputeAngles::compute(const AnyNeighborhoods &nbhs, dihedral_count &count) const
{
    if (!dihedral_types_.empty()) return compute_selected(nbhs, count, nullptr);
    dihedral_type ret;
    initdret(ret);
    for (int i = 0; i < nbhs.size(); ++i)
    {
        const Neighborhood &nbhi = nbhs[i];
        const CommonNeighborGeometry::Image root{i, Eigen::RowVector3d::Zero()};
        for (int j = 0; j < nbhi.nbhcount_; ++j)
        {
            // avoid two-way analyse
            const CommonNeighborGeometry::Image pair{nbhi.indices_[j], nbhi.pos_[j]};
            if (!CommonNeighborGeometry::owns_pair(i, pair))
                continue;
            const Neighborhood &nbhj = nbhs[nbhi.indices_[j]];
            for (int k = 0; k < nbhi.nbhcount_; ++k)
            {
                if (k == j)
                    continue;
                for (int l = 0; l < nbhj.nbhcount_; ++l)
                {
                    const CommonNeighborGeometry::Image endpoint{
                        nbhj.indices_[l], pair.position + nbhj.pos_[l]};
                    if (CommonNeighborGeometry::same_image(endpoint, root))
                        continue;
                    dihedral_type_index dindex(nbhi.chemmap_[k], nbhi.centertype, nbhj.centertype, nbhj.chemmap_[l]);
                    if (!dihedral_types_.empty() && !dihedral_types_.contains(dindex)) continue;
                    auto v1 = nbhi.pos_[j].cross(nbhi.pos_[k]);
                    auto v2 = (-nbhi.pos_[j]).cross(nbhj.pos_[l]);
                    double v1n = v1.norm();
                    double v2n = v2.norm();
                    // A-B-C-D may form a line
                    if (fabs(v1n) < 1e-8 || fabs(v2n) < 1e-8)
                        continue;
                    double cval = v1.dot(v2) / v1n / v2n;
                    double angle = std::acos(std::clamp(cval, -1.0, 1.0));
                    if (angle < low || angle > high)
                        continue;
                    int ida = get_id(angle);
                    ret[dindex][ida]++;
                    count[dindex]++;
                }
            }
        }
    }
    return ret;
}

dihedral_type ComputeAngles::compute(const AnyNeighborhoods &nbhs, dihedral_count &count, double **rcutij) const
{
    if (!dihedral_types_.empty()) return compute_selected(nbhs, count, rcutij);
    dihedral_type ret;
    initdret(ret);
    for (int i = 0; i < nbhs.size(); ++i)
    {
        const Neighborhood &nbhi = nbhs[i];
        const CommonNeighborGeometry::Image root{i, Eigen::RowVector3d::Zero()};
        for (int j = 0; j < nbhi.nbhcount_; ++j)
        {
            // avoid two-way analyse
            const CommonNeighborGeometry::Image pair{nbhi.indices_[j], nbhi.pos_[j]};
            if (!CommonNeighborGeometry::owns_pair(i, pair) || nbhi.distances_[j] > rcutij[nbhi.centertype][nbhi.chemmap_[j]])
                continue;
            const Neighborhood &nbhj = nbhs[nbhi.indices_[j]];
            for (int k = 0; k < nbhi.nbhcount_; ++k)
            {
                if (k == j || nbhi.distances_[k] > rcutij[nbhi.centertype][nbhi.chemmap_[k]])
                    continue;
                for (int l = 0; l < nbhj.nbhcount_; ++l)
                {
                    const CommonNeighborGeometry::Image endpoint{
                        nbhj.indices_[l], pair.position + nbhj.pos_[l]};
                    if (CommonNeighborGeometry::same_image(endpoint, root) || nbhj.distances_[l] > rcutij[nbhj.centertype][nbhj.chemmap_[l]])
                        continue;
                    dihedral_type_index dindex(nbhi.chemmap_[k], nbhi.centertype, nbhj.centertype, nbhj.chemmap_[l]);
                    if (!dihedral_types_.empty() && !dihedral_types_.contains(dindex)) continue;
                    auto v1 = nbhi.pos_[j].cross(nbhi.pos_[k]);
                    auto v2 = (-nbhi.pos_[j]).cross(nbhj.pos_[l]);
                    double v1n = v1.norm();
                    double v2n = v2.norm();
                    // A-B-C-D may form a line
                    if (fabs(v1n) < 1e-8 || fabs(v2n) < 1e-8)
                        continue;
                    double cval = v1.dot(v2) / v1n / v2n;
                    double angle = std::acos(std::clamp(cval, -1.0, 1.0));
                    if (angle < low || angle > high)
                        continue;
                    int ida = get_id(angle);
                    ret[dindex][ida]++;
                    count[dindex]++;
                }
            }
        }
    }
    return ret;
}

void add2t1(bondangle_type &a, bondangle_type &b)
{
    for (auto & it : b)
    {
        for (int i = 0; i < a[it.first].size(); ++i)
        {
            a[it.first][i] += b[it.first][i];
        }
    }
}
void add2t1(bondangle_count &a, bondangle_count &b)
{
    for (auto &it : b)
    {
        a[it.first] += b[it.first];
    }
}
void add2t1(dihedral_type &a, dihedral_type &b)
{
    for (auto & it : b)
    {
        for (int i = 0; i < a[it.first].size(); ++i)
        {
            a[it.first][i] += b[it.first][i];
        }
    }
}
void add2t1(dihedral_count &a, dihedral_count &b)
{
    for (auto &it : b)
    {
        a[it.first] += b[it.first];
    }
}
