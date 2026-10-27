// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once 

#include "libalglib/ap.h"
#include "utils/gputils.h"
#include "base/structures.h"
/**
 * @brief Smoothing a trajectory using fft.
 * Need trajectory with uniform sampling
 * Need timestep info to determine frequency.
 * Species in atom order and PBC flags must match in every frame; atoms are
 * not aligned or reordered by this filter.
 * Fixed partially periodic cells may omit open-direction lattice rows when
 * positions are Cartesian and PBC flags stay fixed. Such trajectories are
 * unwrapped and filtered in Cartesian coordinates, preserving open components.
 * Variable singular partial cells are unsupported; full-rank variable cells
 * retain the fractional-coordinate smoothing convention.
 * 
 * compute2() rebuilds velocity from the smoothed Cartesian trajectory,
 * including cell motion. The legacy compute() velocity branch is unfinished.
 * Second derivatives and forces can be reconstructed with explicit masses.
 * @todo stress reconstruction needs
 *       a separate formulation.
 * f'(x) -> iq*F(q)
 * f''(x) -> -q^2*F(q)
 * When rebuilding, cell should be smoothed first
 * v = dx/dt
 * F = ma
 * a = d2x/dt2
 * mass should be given if force is required
 * Existing energy, force and stress data can be smoothed; this does not
 * reconstruct missing values for those quantities.
 * compute2 interpolation_factor>1 inserts Fourier samples without another window;
 * Cartesian positions and cell are interpolated after original-grid smoothing.
 * 
 * @todo compressive sensing to restore more info from non-uniform sampling.
 */
class ComputeSmoothFFT : public ConnectSettings
{
public: 

    enum {
        BLOCK = 0, // simply set zero
        SUPERGAUSSIAN = 1,
    } window;

    bool even_seq = true;
    bool rebuild_velo = false;
    bool rebuild_force = false;
    int interpolation_factor = 1; // q>1: Fourier samples at dt/q, same mirrored period

    struct param_type {
        // gaussian
        double m = 1;
        double sigma = 0.5; // buffer, will be calculated from freq_cutoff
        double x0 = 0; // do not set this
        // block
        double freq_cutoff = 1; // fs^-1; block cutoff or half-power point for other windows
        double dt = 1; // buffer, do not set this
        int ncutoff; // buffer, do not set this
    } params;
    
    void InitSettings() {
        Connect(params.m, "m", "Gaussian m");
        Connect(params.freq_cutoff, "freq_cutoff", "cutoff frequency in fs^-1");
        Connect(rebuild_velo, "rebuild_velo");
        Connect(interpolation_factor,"interpolation_factor","positive integer Fourier interpolation factor (compute2 only)");
        Connect(rebuild_force, "rebuild_force", "reconstruct Cartesian forces using explicit per-atom masses in amu");
    }

    ComputeSmoothFFT(GPSettings settings) {
        InitSettings();
        ApplySettings(settings);
        // PrintSettings();
    }

    ComputeSmoothFFT(){};
    ~ComputeSmoothFFT(){};

    
    void compute(std::vector<Atoms> &atoms, double dt, int windowtype);

    void compute2(std::vector<Atoms> &atoms, double dt, int windowtype);

    void applywindow(alglib::complex_1d_array &arr, int wtype, param_type &param);
    void applywindow1(alglib::complex_1d_array &arr, int wtype, param_type &param);

    void fill_data(alglib::real_1d_array &arr, int len);

private:
    void compute2_base(std::vector<Atoms>&,double,int,bool reconstruct);

};
