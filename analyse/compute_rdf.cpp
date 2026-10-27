// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_rdf.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

int ComputeRDF::getid(double x) const
{
    return round(x / delta_r);
}

void ComputeRDF::initret(rdf_count_type &val)
{
    val.clear();
    val.resize(nsize);
}

ComputeRDF::ComputeRDF(double _delta, double _cutoff)
{
    r_limit = _cutoff;
    delta_r = _delta;
    init();
}

ComputeRDF::ComputeRDF(const GPSettings &_settings, double rlimit)
{
    r_limit = rlimit;
    InitSettings();
    ApplySettings(_settings);
    // PrintSettings();
    init();
}

void ComputeRDF::initret(rdf_pcount_type &val)
{
    for (int i = 0; i < ntypes; ++i)
    {
        for (int j = i; j < ntypes; ++j)
        {
            rdf_partial_type ind(i, j);
            val[ind] = std::vector<unsigned long>(nsize, 0);
        }
    }
}

void ComputeRDF::init()
{
    if (!std::isfinite(delta_r) || delta_r <= 0.0 ||
        !std::isfinite(r_limit) || r_limit <= 0.0)
        throw std::invalid_argument("RDF bin width and cutoff must be finite and positive");
    const double bins = std::ceil(r_limit / delta_r) + 1.0;
    if (!std::isfinite(bins) || bins > std::numeric_limits<int>::max())
        throw std::invalid_argument("RDF requires too many bins");
    r_dists.clear();
    shell_vol.clear();
    nsize = static_cast<int>(bins);
    r_dists.resize(nsize);
    shell_vol.resize(nsize);
    const double pref = 4.0 / 3.0 * M_PI;
    for (int i = 0; i < nsize; ++i)
    {
        r_dists[i] = delta_r * i;
        // Keep the rounded bins, but normalize only over their sampled
        // radial support. ceil+1 may retain a final bin with no volume.
        const double lower = std::max(0.0, (i - 0.5) * delta_r);
        const double upper = std::min(r_limit, (i + 0.5) * delta_r);
        shell_vol[i] = upper > lower
            ? pref * (upper - lower) * (upper * upper + upper * lower + lower * lower)
            : 0.0;
    }
}
// make sure that nbhs is constructed with cutoff of r_limit or greater
void ComputeRDF::count_pcf(AnyNeighborhoods &nbhs, rdf_count_type &total, rdf_pcount_type &partial, unsigned long &total_num, std::vector<unsigned long> &partial_num) const
{
    total_num += nbhs.size();
    for (int i = 0; i < nbhs.size(); ++i)
    {
        Neighborhood &nbh = nbhs[i];
        partial_num[nbh.centertype]++;
        for (int j = 0; j < nbh.nbhcount_; ++j)
        {
            if (nbh.distances_[j] > r_limit) continue;
            int id = getid(nbh.distances_[j]);
            rdf_partial_type ind(nbh.centertype, nbh.chemmap_[j]);
            partial[ind][id]++;
            total[id]++;
        }
    }
}

void ComputeRDF::count_pcf(NeighborhoodKD &nbhs, rdf_count_type &total, rdf_pcount_type &partial, unsigned long &total_num, std::vector<unsigned long> &partial_num) const
{
    total_num += nbhs.size();
    for (int i = 0; i < nbhs.size(); ++i)
    {
        Neighborhood nbh = nbhs.query(i, r_limit);
        partial_num[nbh.centertype]++;
        for (int j = 0; j < nbh.nbhcount_; ++j)
        {
            if (nbh.distances_[j] > r_limit) continue;
            int id = getid(nbh.distances_[j]);
            rdf_partial_type ind(nbh.centertype, nbh.chemmap_[j]);
            partial[ind][id]++;
            total[id]++;
        }
    }
}

void ComputeRDF::compute_pcf(rdf_count_type &total, unsigned long &total_num, double &rho, rdf_result_type &res_total) const
{
    res_total.assign(nsize, 0.0);
    // Zero is the finite placeholder for an empty statistical denominator.
    if (total_num == 0 || rho == 0.0) return;
    if (!std::isfinite(rho) || rho < 0.0)
        throw std::invalid_argument("RDF normalization requires a finite positive density");
    for (int i = 0; i < nsize; ++i)
    {
        if (shell_vol[i] > 0.0)
            res_total[i] = total[i] / (rho * shell_vol[i] * (double)total_num);
    }
}

void ComputeRDF::compute_pcf(const int type1, const int type2, const unsigned long num_type1, const double rho_type2, rdf_pcount_type &partial, rdf_result_type &result) const
{
    rdf_partial_type rind(type1, type2);
    result.assign(nsize, 0.0);
    if (num_type1 == 0 || rho_type2 == 0.0) return;
    if (!std::isfinite(rho_type2) || rho_type2 < 0.0)
        throw std::invalid_argument("RDF normalization requires a finite positive density");
    const double pair_weight = type1 == type2 ? 1.0 : 0.5;
    for (int i = 0; i < nsize; ++i)
    {
        if (shell_vol[i] > 0.0)
            result[i] = partial[rind][i] / (rho_type2 * shell_vol[i] * (double)num_type1) * pair_weight;
    }
}

// TODO: refine this
double ComputeRDF::get_rdfcut(rdf_result_type &res)
{
    int step_compare = 1;
    int step_scan = 1;
    double first_top, first_ridge;
    bool clip = true;
    int i = 0;
    for (; i < nsize * 0.6 && i + step_scan < nsize; i += step_scan)
    {
        if (res[step_scan + i] - res[i] > 0 && clip)
        {
            continue;
        }
        else
        {
            clip = false;
            break;
        }
    }
    first_top = r_dists[i];
    if (i + 1 < nsize) ++i;
    for (; i < nsize * 0.6 && i + step_scan < nsize; i += step_scan)
    {
        if (res[step_scan + i] - res[i] < 0 && (!clip))
        {
            continue;
        }
        else
        {
            clip = true;
            break;
        }
    }
    first_ridge = r_dists[i];
    return first_ridge;
}

void add2t1(rdf_count_type &p1, rdf_count_type &p2)
{
    for (int i = 0; i < p1.size(); ++i)
    {
        p1[i] += p2[i];
    }
}
void add2t1(rdf_pcount_type &p1, rdf_pcount_type &p2)
{
    for (auto &it : p2)
    {
        for (int i = 0; i < p1[it.first].size(); ++i)
        {
            p1[it.first][i] += p2[it.first][i];
        }
    }
}
