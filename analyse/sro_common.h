// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_SRO_COMMON_H
#define IMRMLP_SRO_COMMON_H

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <stdexcept>

// ── Shared SRO type enum (was duplicated in sro, tsro, wcsro) ─────────
enum SROType {
    WC = 0,
    PM
};

// ── Parse SRO type string → enum ──────────────────────────────────────
inline SROType parse_sro_type(const std::string &s) {
    if (s == "WC")   return WC;
    if (s == "PM")   return PM;
    throw std::invalid_argument("SRO sro_type must be WC or PM; unsupported definition: " + s);
}

// ── Compute a single SRO value ────────────────────────────────────────
// Shared Warren-Cowley and pairwise multicomponent definitions.
inline double compute_sro_value(SROType sro_type,
                                 double count, double total_pairs,
                                 double concen_a, double concen_b,
                                 bool is_same_type)
{
    if (sro_type == WC) {
        return 1.0 - count / total_pairs / concen_a / concen_b;
    } else if (sro_type == PM) {
        double delta_ab = is_same_type ? 1.0 : 0.0;
        return (count / total_pairs / concen_a - concen_b) / (delta_ab - concen_b);
    }
    throw std::invalid_argument("SRO definition is not implemented; use WC or PM");
}

// ── Concentration map from type-count map ─────────────────────────────
inline std::map<int, double> compute_concentrations(const std::map<int, int> &type_counts,
                                                      int total_atoms)
{
    std::map<int, double> concen;
    for (auto &it : type_counts)
        concen[it.first] = static_cast<double>(it.second) / static_cast<double>(total_atoms);
    return concen;
}

// ── Helper: get neighbor count for any neighborhood type ────────────────
// Detects nbhcount_ member (Neighborhood) vs indices_.size() (NeighborhoodSimple)
template<typename T>
int nbh_size(const T &nbh) {
    if constexpr (requires { nbh.nbhcount_; }) {
        return nbh.nbhcount_;
    } else {
        return static_cast<int>(nbh.indices_.size());
    }
}

// Determine common shell radii from all sites in a deterministic distance order.
// Both full and compact neighborhoods use this path.
template<typename NbhsT>
void get_orders(const NbhsT &nbhs, std::vector<double> &mdist,
                int &norder, const double &width)
{
    std::vector<double> distances;
    for (int i = 0; i < nbhs.size(); ++i)
        distances.insert(distances.end(), nbhs[i].distances_.begin(), nbhs[i].distances_.end());
    std::sort(distances.begin(), distances.end());
    mdist.clear();
    for (double distance : distances)
    {
        // Preserve the shell-width convention, including exclusion of self distances.
        const double previous = mdist.empty() ? 0.0 : mdist.back();
        if (distance > previous - width && distance < previous + width)
            continue;
        mdist.push_back(distance);
    }
    norder = mdist.size();
}

#endif // IMRMLP_SRO_COMMON_H
