// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef COMPUTE_WCSRO_H
#define COMPUTE_WCSRO_H

#include "base/structures.h"
#include "base/anyneighborhoods.h"
#include "base/anyneighborhoods_simple.h"
#include "utils/gputils.h"
#include "sro_common.h"

/***
 * Backgrounds:
 *  1. Warren-Cowley parameters (WC-SRO)
 *    Unlike the case in compute_sro.h which only process on-lattice, this version will 
 *    process all cases
 *  2. two comp only
 *   0-0 : 0
 *   0-1 : 1
 *   1-0 : 1
 *   1-1 : 2
 * Note: 
 *  Require a index file!
 * */

// compute WC-SRO
class ComputeWCSRO : public ConnectSettings
{
private:
    std::string type_tmp;
    double r_inn = -1.0;
    double r_out = -1.0;
    SROType sro_type = WC;
    std::string tmpsrotype = "";

    void validate_settings() const;

public:
    void InitSettings()
    {
        Connect(r_inn, "r_inn");
        Connect(r_out, "r_out");
        Connect(tmpsrotype, "sro_type");
    }

    void compute(const AnyNeighborhoods &nbhs, 
                 std::vector<size_t> &counter,
                 std::vector<size_t> &type_counter,
                 size_t &total_atoms,
                 size_t &total_pairs); // compute all orders
    // The input must contain reciprocal neighbor lists. Counts retain the
    // historical directed bins AA, AB+BA, BB; AB and BA each have half bin 1.
    // Negative-one radii mean unspecified; callers supply their global cutoffs.
    double inner_radius(double fallback) const;
    double outer_radius(double fallback) const;
    ComputeWCSRO(){};
    ComputeWCSRO(const GPSettings &_settings);
    ~ComputeWCSRO(){};

    void postprocess(std::vector<size_t> &counter, 
                     std::vector<size_t> &type_counter,
                     size_t &total_atoms, 
                     size_t &total_pairs,
                     std::vector<std::vector<double>> &sro_results);
};

#endif
