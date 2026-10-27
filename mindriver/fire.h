// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "utils/gputils.h"
#include <fmt/format.h>


/**
 * @brief FIRE (Fast Inertial Relaxation Engine) optimizer.
 * @ingroup mindriver
 *
 * Used as a fallback when L-BFGS fails. Implements adaptive timestep with
 * velocity damping and power-based restarts. The force-only entry point
 * does not require the reported value to decrease (e.g. NEB effective forces).
 */
class SimpleFireF : public ConnectSettings {
public:
    // init by user
    // one can modify this to weight different type of atoms
    double ts_max = 10.0; // choice: 1.0
    double ts_min = 0.001;
    double ts_init = 2; // choice: 1.0
    double alpha_start = 0.25; // or 0.1
    double alpha_reduce = 0.99;
    double ts_grow = 1.1;
    double ts_reduce = 0.5;
    bool initaildeyal = true;
    bool downhillcheck = false;
    bool assignmass = false;
    bool massactall = true;
    int ndelay = 20;
    double gtol = 1e-2;
    int max_iter = 1000;
    double max_step = 100;
    double first_step = 1e-2;
    int first_delay = 0;
    bool half_step_back = true;
    double backward_ratio = 0.5;
    std::vector<double> mass;

    bool freeze_small_gd = false;
    double freeze_gd_tol = 1e-6;

    // Optional callback for external convergence criteria (e.g., NEB perpendicular force)
    std::function<bool()> converged_callback;

    // change during run
    double T;
    double cmax;
    double pmax;
    double ts;
    double alpha;

    size_t vdfitmax = 2000; // stop if consecutive iterations with P(t) < 0, avoid inifite loops

    // buffers
    double num_it_ppostive = 0;
    double num_it_pnegative = 0;
    bool is_uphill = false;

    void InitSettings() {
        Connect(gtol, "gtol", "force tolerance");
        Connect(max_step, "max_move");
        Connect(max_iter, "maxiteration");
        Connect(ts_max, "ts_max");
        Connect(ts_min, "ts_min");
        Connect(ts_init, "ts_init");
        Connect(alpha_start, "alpha_start");
        Connect(alpha_reduce, "alpha_reduce");
        Connect(ts_grow, "ts_grow");
        Connect(ts_reduce, "ts_reduce");
        Connect(initaildeyal, "initialdelay");
        Connect(ndelay, "ndelay");
        Connect(downhillcheck, "downhillcheck");
        Connect(assignmass, "assign_mass", "", 1, 10);
        Connect(massactall, "mass_actall", "", 1, 10);
        Connect(half_step_back, "halfstepback");
        Connect(backward_ratio, "backward_ratio");
        Connect(first_step, "first_step");
        Connect(first_delay, "first_delay");
        Connect(freeze_small_gd, "freeze_small_gd");
        Connect(freeze_gd_tol, "freeze_gd_tol");
        if (first_step > max_step) first_step = max_step;
    }

    SimpleFireF() { InitSettings(); }
    SimpleFireF(GPSettings settings) {
        InitSettings();
        ApplySettings(settings);
// PrintSettings();
    }

    // with energy
    // todo: check and remove one functor call.
    template<class T>
    int minimizef(T &functor, Eigen::VectorXd &x, double &f) {
        // init mass
        Eigen::VectorXd ivmass;
        if (mass.size() != x.size()) {
            ivmass.resize(x.size());
            ivmass.setOnes();
        } else {
            ivmass.resize(x.size());
            for (int i = 0; i < mass.size(); ++i) {
                ivmass[i] = 1.0 / mass[i];
            }
        }
        Eigen::VectorXd g;
        g.resize(x.size());
        // init parameters
        alpha = alpha_start;
        ts = ts_init;
        num_it_pnegative = 0;
        num_it_ppostive = 0;
        double workP = 0;
        // run
        int nl = x.size();
        // f = functor(x, g);
        // if (fabs(g.maxCoeff()) < gtol && fabs(g.minCoeff()) < gtol) {
        //     return 0;
        // }
        Eigen::VectorXd v, dr, v_prev, x_prev, g_prev;
        double f_prev;
        // double f_prev;
        double drnorm = 0;
        int Nsteps = 0;
        // f_prev = functor(x, g);
        for (int currit = 0; currit < max_iter; ++currit) {
            // if (fabs(g.maxCoeff()) < gtol && fabs(g.minCoeff()) < gtol) return currit;
            if (v.size() == 0) {
                f_prev = functor(x, g);
                v.resize(x.size());
                v.setZero();
                // f_prev = f;
                x_prev = x;
                g_prev = g;
                v_prev = v;
            } else {
                is_uphill = false;
                if (downhillcheck) {
                    f = functor(x, g);
                    if ((fabs(g.maxCoeff()) < gtol && fabs(g.minCoeff()) < gtol) || (converged_callback && converged_callback())) return currit;
                    if (f > f_prev) {
                        x = x_prev;
                        g = g_prev;
                        f = f_prev;
                        is_uphill = true;
                    }
                    f_prev = f;
                    x_prev = x;
                    v_prev = v;
                }

                workP = -g.dot(v);
                if (workP > 0.0 && !is_uphill) {
                    double vv = v.norm();
                    v *= (1.0 - alpha);
                    double gnorm = g.norm();
                    if (gnorm > 0.0) {
                        v -= alpha * g * vv / gnorm;
                    }
                    if (!initaildeyal || Nsteps > ndelay) {
                        ts = std::min(ts_max, ts_grow * ts);
                        alpha = alpha * alpha_reduce;
                    } 
                    ++Nsteps;
                } else {
                    v.setZero();
                    alpha = alpha_start;
                    ts *= ts_reduce;
                    Nsteps = 0;
                }
            }
            v += -ts * g;
            dr = ts * v;
            drnorm = dr.norm();
            if (drnorm > max_step) {
                dr *= max_step / drnorm;
            }
            x_prev = x;
            x += dr;
            f = functor(x, g);
        }
        return max_iter;
    }

    // without energy
    template<class T>
    int minimize(T &functor, Eigen::VectorXd &x, double &f) {
        // init mass
        Eigen::VectorXd ivmass;
        ivmass.resize(x.size());
        ivmass.setOnes();
        if (assignmass) {
            for (int i = 0; i < ivmass.size(); ++i) {
                ivmass[i] = 1.0 / mass[i];
            }
        }
        Eigen::VectorXd g;
        g.resize(x.size());
        // init parameters
        alpha = alpha_start;
        ts = ts_init;
        // Explicit small initial timesteps remain valid with the default floor.
        const double effective_ts_min = std::min(ts_min, ts_init);
        // Settings are applied after InitSettings; enforce the cap at use time.
        const double first_move = std::min(first_step, max_step);
        num_it_pnegative = 0;
        num_it_ppostive = 0;
        double workP = 0;
        // run
        int nl = x.size();
        // f = functor(x, g);
        // if (fabs(g.maxCoeff()) < gtol && fabs(g.minCoeff()) < gtol) {
        //     return 0;
        // }
        Eigen::VectorXd v, dr;
        // double f_prev;
        double drnorm = 0;
        int Nsteps = 0;
        for (int currit = 0; currit < max_iter; ++currit) {
            f = functor(x, g);
            if (freeze_small_gd) {
                for (int i = 0; i < g.size(); ++i) {
                    if (fabs(g[i]) < freeze_gd_tol) {
                        g[i] = 0;
                    }
                }
            }
            if ((fabs(g.maxCoeff()) < gtol && fabs(g.minCoeff()) < gtol) || (converged_callback && converged_callback())) return currit;
            if (assignmass && massactall) {
                for (int i = 0; i < g.size(); ++i) {
                    g(i) *= ivmass(i);
                }
            }
            if (v.size() == 0) {
                v.resize(x.size());
                v.setZero();
            } else {
                workP = -g.dot(v);
                if (workP > 0.0) {
                    double vv = v.norm();
                    v *= (1.0 - alpha);
                    double gnorm = g.norm();
                    if (gnorm > 0.0) {
                        v -= alpha * g * vv / gnorm;
                    }
                    if (!initaildeyal || Nsteps > ndelay) {
                        ts = std::min(ts_max, ts_grow * ts);
                        alpha = alpha * alpha_reduce;
                    } 
                    ++Nsteps;
                } else {
                    Message("Uphill - P = " + std::to_string(workP));
                    v.setZero();
                    alpha = alpha_start;
                    ts = std::max(effective_ts_min, ts * ts_reduce);
                    Nsteps = 0; 
                    if (half_step_back) {
                        // Roll back a fraction of the actual preceding move,
                        // before using the new timestep for the next trial.
                        x -= backward_ratio * dr;
                        f = functor(x, g);
                        if (freeze_small_gd) {
                            for (int i = 0; i < g.size(); ++i) {
                                if (fabs(g[i]) < freeze_gd_tol) {
                                    g[i] = 0;
                                }
                            }
                        }
                        if (assignmass && massactall) {
                            for (int i = 0; i < g.size(); ++i) {
                                g(i) *= ivmass(i);
                            }
                        }
                        v.setZero();
                    }
                }
            }
            if (massactall || !assignmass) {
                v += -ts * g;
            } else {
                for (int i = 0; i < v.size(); ++i) {
                    v(i) += -ts * g(i) * ivmass(i);
                }
            }
            dr = ts * v;
            drnorm = dr.norm();
            if (currit < first_delay) {
                if (drnorm > first_move) {
                    dr *= first_move / drnorm;
                }
            } else {
                if (drnorm > max_step) {
                    dr *= max_step / drnorm;
                }
            }
            x += dr;
        }
        return max_iter;
    };
};
