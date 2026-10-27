// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef _IMRMLP_COMPUTE_RDF_H
#define _IMRMLP_COMPUTE_RDF_H

#include "base/anyneighborhoods.h"
#include "base/neighborhoodKD.h"
#include "utils/gputils.h"

/**
 * Compute radial distribution function of order 2
 * Compute pair correlation function of order 2
 * 
 * This code applies all pbc conditions
 * 
 * Ensure that rho is constant over all images
 * 
 * TODO: non-pbc consideration
 * **/

typedef std::vector<unsigned long> rdf_count_type;
typedef std::vector<double> rdf_result_type;
struct rdf_partial_type
{
    int chem[2];

    rdf_partial_type(int s1, int s2)
    {
        if (s1 < s2)
        {
            chem[0] = s1;
            chem[1] = s2;
        }
        else
        {
            chem[0] = s2;
            chem[1] = s1;
        }
    }

    bool operator<(const rdf_partial_type &opt) const
    {
        if (chem[0] < opt.chem[0])
            return true;
        if (chem[0] > opt.chem[0])
            return false;
        if (chem[1] < opt.chem[1])
            return true;
        return false;
    }
};
typedef std::map<rdf_partial_type, rdf_count_type> rdf_pcount_type;

void add2t1(rdf_count_type &p1, rdf_count_type &p2);
void add2t1(rdf_pcount_type &p1, rdf_pcount_type &p2);

class ComputeRDF : public ConnectSettings
{
private:
    double delta_r = 0.1;
    double r_limit = 15.0;
    int nsize;

    int getid(double x) const;

public:
    int ntypes = 0;
    std::vector<double> r_dists;
    std::vector<double> shell_vol;
    ComputeRDF(double _delta, double _cutoff);
    ComputeRDF(const GPSettings &_settings, double maxr);

    void init();

    void initret(rdf_pcount_type &val);
    void initret(rdf_count_type &val);

    void InitSettings()
    {
        Connect(delta_r, "delta_r");
    }

    // only gives counts
    // PCF
    void count_pcf(AnyNeighborhoods &nbhs, rdf_count_type &total, rdf_pcount_type &partial, unsigned long &total_num, std::vector<unsigned long> &partial_num) const;
    void count_pcf(NeighborhoodKD &nbhs, rdf_count_type &total, rdf_pcount_type &partial, unsigned long &total_num, std::vector<unsigned long> &partial_num) const;
    // for large structures
    // void conut_pcf(Atoms &atom, rdf_count_type &total, rdf_pcount_type &partial, unsigned long &total_num, std::vector<unsigned long> &partial_num) const;
    void compute_pcf(rdf_count_type &total, unsigned long &total_num, double &rho, rdf_result_type &res_total) const;
    void compute_pcf(const int type1, const int type2, const unsigned long num_type1, const double rho_type1, rdf_pcount_type &partial, rdf_result_type &result) const;
    double get_rdfcut(rdf_result_type &res);
};

#endif
