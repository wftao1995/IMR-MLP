// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_COMPUTE_BOO_H
#define IMRMLP_COMPUTE_BOO_H

#include "base/anyneighborhoods.h"
#include "base/structures.h"
#include "utils/gputils.h"
#include "mathextra/angular_functions.h"

class ComputeBondOrder : public ConnectSettings
{
private:
    int maxl_ = 4;
    int qlmsize;

    bool average = true;
    bool corase = false;

    double r_cut = 5.0;
    int nnn = 0;
    int min_nn = 0; // only used when using r_cut
    int max_nn = 0; // as above

    bool use_nnn = false;

    std::vector<std::complex<double>> Qlmc;
    AngularFuncs::ComplexSphericalHarmoics csph;

public:
    ComputeBondOrder(){};
    ComputeBondOrder(const GPSettings &_settings)
    {
        InitSettings();
        ApplySettings(_settings);
        init(maxl_);
        if (nnn > 0)
            use_nnn = true;
        // PrintSettings();
    }

    void init(int ml)
    {
        maxl_ = ml;
        qlmsize = (maxl_ + 1) * (maxl_ + 1);
        Qlmc.resize(qlmsize); 
        csph.init(maxl_, maxl_);
    }

    inline void set_rcut(const double rcut) { r_cut = rcut; }

    inline int get_idx(const int &l, const int &m) const
    {
        return m + l * (l + 1);
    }

    void InitSettings()
    {
        Connect(maxl_, "maxl");
        Connect(average, "average");
        Connect(corase, "corase");
        Connect(r_cut, "cutoff");
        Connect(nnn, "nnn");
        Connect(min_nn, "minnn");
        Connect(max_nn, "maxnn");
    }
    // Use supplied neighborhoods as-is; the one-argument overload selects neighbors from settings.
    void compute(Atoms &atom, const AnyNeighborhoods &nbhs);
    void compute(Atoms &atom);

};


#endif