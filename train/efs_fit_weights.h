// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potentials/potential.h"
#include <cmath>

// Weights multiply squared residuals (not residuals). stress_ is the full
// 3x3 virial tensor. Keep this contract shared by nonlinear and linear fits.
struct EFSFitWeights {
    double energy=1., force=.01, stress=.0001;
    double energy_power=1., force_power=0., stress_power=1.;
    double norm_d=.1, norm_f=0., norm_p=1.;

    double energy_weight(const Atoms& a) const {
        return energy/std::pow(a.size(),energy_power)*a.fitness;
    }
    double force_weight(const Atoms& a) const {
        return force/std::pow(a.size(),force_power);
    }
    double stress_weight(const Atoms& a) const {
        return stress/std::pow(a.size(),stress_power);
    }
    double normalization(const Atoms& a) const {
        if (norm_f<=0. || !a.has_forces()) return 1.;
        double mean=0.;
        for (const auto& f:a.forces_) mean+=f.squaredNorm();
        mean/=a.size();
        // Residual-core training keeps the normalization of the original labels.
        if (const auto it=a.global_attribs_d_.find("__imrmlp_fixed_core_force_norm2");
            it!=a.global_attribs_d_.end()) mean=it->second;
        return norm_d/(norm_d+std::pow(norm_f*mean,norm_p));
    }
    static double atomic_weight(const Atoms& a,int i,double weight) {
        return a.atomic_fitness.empty() ? weight : weight*a.atomic_fitness[i];
    }
};
