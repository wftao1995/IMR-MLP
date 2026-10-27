// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "utils/gputils.h"
#include <functional>
/**
 * @brief CG method without energy info
 * 
 */
class SimpleCGF : public ConnectSettings {
public: 
    double gtol = 1e-2;
    int max_iter = 1000;
    double finite_step = 0.002;
    double max_step = 1.0; // not the same as fire, this is max move

    // Optional callback for external convergence criteria (e.g., NEB perpendicular force)
    std::function<bool()> converged_callback;

    void InitSettings() {
        Connect(gtol, "gtol");
        Connect(max_iter, "maxiteration");
        Connect(finite_step, "finite_step");
        Connect(max_step, "max_move");
    }

    SimpleCGF(GPSettings settings) {
        InitSettings();
        ApplySettings(settings);
// PrintSettings();
    }

    bool check_conv(Eigen::VectorXd &g) {
        if (fabs(g.maxCoeff()) < gtol && fabs(g.minCoeff()) < gtol) return true;
        if (converged_callback && converged_callback()) return true;
        return false;
    }

    // Force-only line search. Functors return the energy gradient, as in
    // FIRE/SGD, so search directions must point against that gradient.
    template<class T>
    int minimize(T &functor, Eigen::VectorXd &x, double &f) {
        Eigen::VectorXd g, d, d_unit; 
        g.resize(x.size());
        bool fdstep = true;
        bool optflag = false;
        f = functor(x, g);
        if (check_conv(g)) return 0;
        Eigen::VectorXd g_prev;
        g_prev.resize(x.size());
        d.resize(x.size());
        d_unit.resize(x.size());
        d.setZero();
        g_prev.setZero();
        double a1, a2, gam, step_size;
        double fp1, fp2, gavg, curv;
        auto sign = [](double a, double b) {
            if (b >= 0) return abs(a);
            else return -abs(a);
        };
        // start opt form a finite difference step
        for (int i = 0; i < max_iter; ++i) {
            if (fdstep) {
                fdstep = false;
                optflag = true;
                a1 = abs(g.dot(g_prev));
                a2 = g_prev.dot(g_prev);
                if (a1 <= 0.5 * a2 && a2 != 0.) {
                    gam = g.dot(g-g_prev) / a2;
                } else {
                    gam = 0;
                }
                d = -g + gam * d;
                if (d.dot(g) >= 0) d = -g;
                d_unit = d; d_unit.normalize();
                g_prev = g;
                x += d_unit * finite_step;
            } else {
                fdstep = true;
                optflag = false;
                fp1 = g_prev.dot(d_unit);
                fp2 = g.dot(d_unit);
                curv = (fp2 - fp1) / finite_step;
                if (curv <= 0) {
                    step_size = max_step;
                } else {
                    gavg = 0.5 * (fp1 + fp2);
                    step_size = -gavg / curv;
                    if (step_size > max_step) {
                        step_size = sign(max_step, step_size) - sign(finite_step, step_size);
                    } else {
                        step_size = step_size - 0.5 * finite_step;
                    }
                }
                x += d_unit * step_size;
            }
            f = functor(x, g);
            if (check_conv(g)) return i + 1;
        }

        
        return max_iter;
    };
};
