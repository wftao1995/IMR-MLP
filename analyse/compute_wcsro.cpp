// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_wcsro.h"
#include "tools/procstr.h"
#include <algorithm>
#include <functional>
#include <cmath>
#include <stdexcept>

void ComputeWCSRO::validate_settings() const
{
    if ((!std::isfinite(r_inn) || (r_inn < 0.0 && r_inn != -1.0)) ||
        (!std::isfinite(r_out) || (r_out <= 0.0 && r_out != -1.0)))
        throw std::invalid_argument("WCSRO radii must be finite: r_inn >= 0, r_out > 0, or -1 for unspecified");
    if (r_inn >= 0.0 && r_out >= 0.0 && r_out <= r_inn)
        throw std::invalid_argument("WCSRO requires r_out > r_inn");
    if (!tmpsrotype.empty() && tmpsrotype != "WC" && tmpsrotype != "PM")
        throw std::invalid_argument("WCSRO sro_type must be WC or PM; other definitions are not implemented");
}

double ComputeWCSRO::inner_radius(double fallback) const
{
    validate_settings();
    const double radius = r_inn >= 0.0 ? r_inn : fallback;
    if (!std::isfinite(radius) || radius < 0.0)
        throw std::invalid_argument("WCSRO inner radius must be finite and nonnegative");
    if (r_out > 0.0 && radius >= r_out)
        throw std::invalid_argument("WCSRO requires r_out > r_inn");
    return radius;
}

double ComputeWCSRO::outer_radius(double fallback) const
{
    validate_settings();
    const double radius = r_out > 0.0 ? r_out : fallback;
    if (!std::isfinite(radius) || radius <= 0.0)
        throw std::invalid_argument("WCSRO outer radius must be finite and positive");
    if (r_inn >= 0.0 && radius <= r_inn)
        throw std::invalid_argument("WCSRO requires r_out > r_inn");
    return radius;
}

ComputeWCSRO::ComputeWCSRO(const GPSettings &_settings)
{
    InitSettings();
    ApplySettings(_settings);
    validate_settings();
    if (tmpsrotype != "")
    {
        sro_type = parse_sro_type(tmpsrotype);
    }
// PrintSettings();
}

void ComputeWCSRO::compute(const AnyNeighborhoods &nbhs, 
                 std::vector<size_t> &counter,
                 std::vector<size_t> &type_counter,
                 size_t &total_atoms, 
                 size_t &total_pairs) {
    validate_settings();
    if (counter.size() < 3 || type_counter.size() < 2)
        throw std::invalid_argument("WCSRO requires at least three pair counters and two species counters");
    // Validate before changing cumulative counts, including every neighbor's
    // type even when an explicit radial shell will exclude that neighbor.
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &nbh = nbhs[i];
        if (nbh.centertype < 0 || nbh.centertype > 1)
            throw std::invalid_argument("WCSRO requires atom types 0 and 1");
        if (nbh.nbhcount_ < 0 || nbh.chemmap_.size() < static_cast<size_t>(nbh.nbhcount_) ||
            nbh.distances_.size() < static_cast<size_t>(nbh.nbhcount_))
            throw std::invalid_argument("WCSRO neighbor count differs from neighbor data lengths");
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            if (nbh.chemmap_[j] < 0 || nbh.chemmap_[j] > 1)
                throw std::invalid_argument("WCSRO requires neighbor types 0 and 1");
            if (!std::isfinite(nbh.distances_[j]) || nbh.distances_[j] < 0.0)
                throw std::invalid_argument("WCSRO neighbor distances must be finite and nonnegative");
        }
    }
    total_atoms += nbhs.size();
    for (int i = 0; i < nbhs.size(); ++i)
    {
        const auto &nbh = nbhs[i];
        type_counter[nbh.centertype]++;
        for (int j = 0; j < nbh.nbhcount_; ++j)
        {
            const double distance = nbh.distances_[j];
            if ((r_inn >= 0.0 && distance <= r_inn) ||
                (r_out > 0.0 && distance >= r_out))
                continue;
            counter[nbh.centertype+nbh.chemmap_[j]]++;
            total_pairs++;
        }
    }
}

void ComputeWCSRO::postprocess(std::vector<size_t> &counter, 
                               std::vector<size_t> &type_counter, 
                               size_t &total_atoms, 
                               size_t &total_pairs,
                               std::vector<std::vector<double>> &sro_results) {
    validate_settings();
    if (counter.size() < 3 || type_counter.size() < 2)
        throw std::invalid_argument("WCSRO requires at least three pair counters and two species counters");
    if (total_atoms == 0 || type_counter[0] == 0 || type_counter[1] == 0)
        throw std::invalid_argument("WCSRO requires both species to be present");
    if (type_counter[0] + type_counter[1] != total_atoms ||
        counter[0] + counter[1] + counter[2] != total_pairs)
        throw std::invalid_argument("WCSRO cumulative counts are inconsistent with their totals");
    if (total_pairs == 0)
        throw std::invalid_argument("WCSRO requires at least one neighbor pair in the selected radial shell");
    if (counter[1] % 2 != 0)
        throw std::invalid_argument("WCSRO unlike-pair count requires reciprocal neighbor lists");
    double c0 = static_cast<double>(type_counter[0]) / static_cast<double>(total_atoms);
    double c1 = static_cast<double>(type_counter[1]) / static_cast<double>(total_atoms);
    sro_results.resize(2);
    for (int i = 0; i < 2; ++i) {
        sro_results[i].resize(2);
        std::fill(sro_results[i].begin(), sro_results[i].end(), 0);
    }
    std::vector<double> x{c0, c1};
    std::vector<std::vector<double>> CN(2);
    for (auto &it : CN)
        it.resize(2);
    const double unlike = static_cast<double>(counter[1]) / 2.0;
    const double row_totals[2] = {counter[0] + unlike, counter[2] + unlike};
    if (row_totals[0] == 0.0 || row_totals[1] == 0.0)
        throw std::invalid_argument("WCSRO requires neighbors around both species in the selected radial shell");
    const SROType requested_type = tmpsrotype == "PM" ? PM : WC;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            // Only the unlike-species bin combines both AB and BA directions.
            CN[i][j] = static_cast<double>(counter[i+j]) / (i == j ? 1.0 : 2.0);
            // P(j|i) is normalized by neighbors around species i. A global
            // pair normalization is equivalent only at equal coordination.
            sro_results[i][j] = compute_sro_value(requested_type,
                CN[i][j], row_totals[i], 1.0, x[j], i == j);
        }
    }
}
