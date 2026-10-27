// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "utils/gputils.h"


// Settings catalog for the relaxation L-BFGS interface.
class SimpleLBFGS : public ConnectSettings {
public:
    double gtol = 1e-2;
    int max_iter = 1000;
    double max_step = 1e99;

    double lbfgs_m = 5;

    void InitSettings() {
        Connect(gtol, "gtol", "force tolerance");
        Connect(max_step, "max_move");
        Connect(max_iter, "maxiteration");
    }

    SimpleLBFGS() { InitSettings(); }
    SimpleLBFGS(GPSettings settings) {
        InitSettings();
        ApplySettings(settings);
    }

};
