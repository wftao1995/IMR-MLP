// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "utils/gputils.h"
#include <functional>


// 
class SimpleSGD : public ConnectSettings {
public:
    // init by user
    double max_step = 0.1;
    double min_step = 0.0;
    double max_step_init = 0.1;
    double gtol = 1e-3;
    int max_iter = 200;
    bool gonly = true;

    double step_upscale = 1.2;
    double step_downscale = 0.2;

    // Optional callback for external convergence criteria (e.g., NEB perpendicular force)
    std::function<bool()> converged_callback;

    void InitSettings() {
        Connect(gtol, "gtol", "force tolerance");
        Connect(max_step, "max_move");
        Connect(min_step, "min_move");
        Connect(max_step_init, "max_step_init");
        Connect(max_iter, "maxiteration");
        Connect(step_upscale, "step_upscale");
        Connect(step_downscale, "step_downscale");
        Connect(gonly, "gonly", "do not use function value");
    }

    SimpleSGD(GPSettings settings) {
        InitSettings();
        ApplySettings(settings);
    }

    template<class T>
    int minimize(T &functor, Eigen::VectorXd &x, double &f) {
        std::vector<double> step_max(x.size(), max_step_init);
        double fnmax = 0.0;
        Eigen::VectorXd g, x_prev, g_prev;
        double f_prev;
        g.resize(x.size());

        f = functor(x, g);
        if ((fabs(g.maxCoeff()) < gtol && fabs(g.minCoeff()) < gtol) || (converged_callback && converged_callback())) {
            return 0;
        }
        // run
        int nl = x.size();
        if (!gonly)
            fnmax = 1.0 / fabs(*std::max_element(g.begin(), g.end(), [](const double &a, const double &b) {
                    return fabs(a) < fabs(b);}));
        else 
            fnmax = 1.0;
        for (int currit = 0; currit < max_iter; ++currit) {
            x_prev = x;
            g_prev = g;
            f_prev = f;
            for (int i = 0; i < nl; ++i) {
                double gd_step = g[i] * step_max[i] * fnmax;
                if (gd_step > max_step) x[i] -= max_step;
                else if (gd_step < -max_step) x[i] += max_step;
                else x[i] -= gd_step;
            }

            f = functor(x, g);
            if ((fabs(g.maxCoeff()) < gtol && fabs(g.minCoeff()) < gtol) || (converged_callback && converged_callback())) {
                return currit + 1;
            }
            // check if accept or revert
            if (!gonly) {
                if (f < f_prev) {
                    // accept
                    fnmax = 1.0 / fabs(*std::max_element(g.begin(), g.end(), [](const double &a, const double &b) {
                    return fabs(a) < fabs(b);}));
                    std::for_each(step_max.begin(), step_max.end(), [&](double &a){ 
                        if (a < max_step)
                            a *= step_upscale;
                        });
                } else {
                    // revert
                    x = x_prev; 
                    g = g_prev;
                    f = f_prev;
                    std::for_each(step_max.begin(), step_max.end(), [&](double &a){ 
                        if (a > min_step)
                            a *= step_downscale;
                        });
                }
            }
        }
        return max_iter;
    };
};