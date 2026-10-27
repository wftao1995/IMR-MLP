// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "hvp_fit.h"

class ErrorMonitor
{
public:
    double sq_energy_at_error;
    double sq_force_error;
    double sq_stress_error;

    int nstresstensor;
    int nforces;
    int nenergy;

    double loss_e, loss_f, loss_s;
    double sq_hessian_error;
    double loss_h;
    double sq_hvp_error = 0, loss_hvp = 0, hvp_rms = 0;
    int nhvp = 0;
    void add_hvp(const Training::HVPFitResult &r) {
        sq_hvp_error += r.squared_error; loss_hvp += r.loss; nhvp += r.observations;
    }

    int t_size_all = 0;

public:
    ErrorMonitor(){
        init();
    };
    void init()
    {
        sq_energy_at_error = 0;
        sq_force_error = 0;
        sq_stress_error = 0;
        sq_hessian_error = 0;
        nstresstensor = 0;
        nforces = 0;
        nenergy = 0;
        nhessian = 0;
        sq_hvp_error = loss_hvp = hvp_rms = 0; nhvp = 0;
        loss_e = 0; loss_f = 0; loss_s = 0; loss_h = 0;
        erms = frms = srms = hrms = 0;
    }
    int nhessian = 0;
    double erms, frms, srms, hrms;

    void report();
};
