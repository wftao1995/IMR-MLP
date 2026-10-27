// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
// mtp_kim_core.h — Self-contained MTP computation core
// No dependencies beyond C++ standard library and KIM API headers.
// Replaces Eigen/IMRMLP/alglib with plain C++ for KIM portable model use.

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <map>
#include <memory>

#include "mtp_kim_radial.h"

namespace mtp_kim {

// ── Small vector / linear algebra helpers (replace Eigen) ────────────

struct Vec3 { double x, y, z; };
inline Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline double dot(const Vec3 &a, const Vec3 &b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline double len2(const Vec3 &v) { return v.x*v.x + v.y*v.y + v.z*v.z; }
inline double len(const Vec3 &v) { return std::sqrt(len2(v)); }

// ── Main MTP Core class ────────────────────────────────────────────────

class MTPCore {
public:
    MTPCore() = default;
    ~MTPCore();
    MTPCore(const MTPCore &) = delete;
    MTPCore &operator=(const MTPCore &) = delete;

    // ── Load from .mtp file (or istream) ──────────────────────────────
    bool load(std::istream &ifs);
    bool load(const std::string &filepath);

    // ── Accessors ─────────────────────────────────────────────────────
    int    speciesCount()     const { return species_count; }
    double maxCutoff()        const { return max_cutoff; }
    int    alphaMomentsCount() const { return alpha_moments_count; }
    int    alphaBasicCount()  const { return alpha_index_basic_count; }
    int    radialFuncCount()  const { return radial_func_count; }
    int    basisSize()       const { return rb_size; }

    // ── Radial basis evaluation ───────────────────────────────────────
    // Evaluate radial basis for a species pair
    void evalRadial(int itype, int jtype, double r,
                    double *vals_out, double *ders_out) const;

    // ── Per-thread computation buffer (thread-safe pool) ──────────────
    struct ThreadBuffer {
        std::vector<double> rb_vals;
        std::vector<double> rb_ders;
        std::vector<double> val_mu;
        std::vector<double> der_mu;
        std::vector<double> moment_vals;
        std::vector<double> dist_powers;
        std::vector<double> coords_powers;
        std::vector<double> moment_jacobian;
        std::vector<double> moment_ders; // site_energy_ders_wrt_moments

        void ensure(int rbSize, int maxMu, int maxAlphaBasic,
                    int alphaMoments, int maxJacSize) {
            rb_vals.resize(rbSize);
            rb_ders.resize(rbSize);
            val_mu.resize(maxMu);
            der_mu.resize(maxMu);
            moment_vals.assign(alphaMoments, 0.0);
            dist_powers.resize(maxAlphaBasic);
            coords_powers.resize(maxAlphaBasic * 3);
            moment_jacobian.resize(maxJacSize);
            moment_ders.assign(alphaMoments, 0.0);
        }
    };

    // Get a buffer from the pool (thread-safe, modifies mutable pool state)
    ThreadBuffer &acquireBuffer() const;

    // ── Core EFS computation for one atom ──────────────────────────────
    void calcSiteEFS(int centertype,
                     int nNeighbors,
                     const Vec3 *relPos,
                     const double *dists,
                     const int *jtypes,
                     double &site_energy,
                     double *site_energy_ders) const;

    // Thread-safe variant using external buffer
    void calcSiteEFS(int centertype, int nNeighbors,
                     const Vec3 *relPos, const double *dists,
                     const int *jtypes, double &site_energy,
                     double *site_energy_ders,
                     ThreadBuffer &buf) const;

    // ── Compute total EFS for a system ─────────────────────────────────
    // coordinates: flat array [3*nAtoms] in model length units
    // speciesCodes: per-atom species codes
    // contributing: per-atom contributing flags
    // nAtoms: total number of particles
    // getNeighborList: callback to get neighbors for particle i
    // energy (out): total energy
    // forces (out): flat array [3*nAtoms]
    // virial (out): KIM order [xx, yy, zz, yz, xz, xy]
    void computeEFS(int nAtoms,
                    const int *speciesCodes,
                    const int *contributing,
                    const double *coordinates,
                    double &energy,
                    double *forces,
                    double *virial);

    // ── KIM neighbor list callback context ─────────────────────────────
    struct NbhCallback {
        const MTPCore *core;
        const double *coords;
        const int *speciesCodes;
        int nAtoms;
        mutable std::vector<int> resultBuf;  // reusable buffer
    };

    // Static callback suitable for KIM GetNeighborList
    // Uses binning for O(N·bins_per_cutoff) performance
    static int getNbhCallback(void *data, int nLists, const double *cutoffs,
                              int listIdx, int particleNum,
                              int *nNeigh, int const **neighbors);

    // Build cell-linked list for fast neighbor queries
    void buildBins(int nAtoms, const double *coords, double binSize) const;

    // ── .mtp file parameters (public for KIM driver access) ──────────
    int species_count = 0;
    int radial_func_count = 0;   // max_mu
    int rb_size = 0;            // radial basis size
    int alpha_moments_count = 0;
    int alpha_index_basic_count = 0;
    int alpha_index_times_count = 0;
    int alpha_scalar_moments = 0;
    int max_alpha_index_basic = 0;  // max(nu) among basic moments
    int max_mu = 0;                // max radial function index

    double max_cutoff = 0.0;

    // ── Radial basis ──────────────────────────────────────────────────
    bool radial_basis_split = false;
    RadialBasis *p_radial_basis = nullptr;               // shared basis
    std::vector<RadialBasis *> p_radial_basis_pair;       // per-pair bases

    // ── Radial coefficients ───────────────────────────────────────────
    // Layout: radial_coeffs[(s1*sp+s2)*rf_count*rb_size + mu*rb_size + xi]
    std::vector<double> radial_coeffs;

    // ── Cutoff matrix (per species pair) ──────────────────────────────
    std::vector<double> cutoffs;      // flat: [s1*sp + s2]
    std::vector<double> r_mins;       // flat: [s1*sp + s2]

    // ── Moment basis definition ───────────────────────────────────────
    // alpha_index_basic: [count][4] = {mu, nu_x, nu_y, nu_z}
    std::vector<int> alpha_index_basic;  // flat: count*4
    std::vector<int> nu_arr;             // precomputed nu_x+nu_y+nu_z per basic moment

    // alpha_index_times: [count][4] = {id1, id2, times, bid}
    std::vector<int> alpha_index_times;  // flat: count*4

    // alpha_moment_mapping: [alpha_scalar_moments] -> moment index
    std::vector<int> alpha_moment_mapping;

    // ── Linear coefficients ───────────────────────────────────────────
    std::vector<double> species_coeffs;  // per-species constant
    std::vector<double> moment_coeffs;   // per scalar moment coefficient

    // ── Thread buffer pool (lock-free for OMP) ──────────────────────
    mutable std::vector<ThreadBuffer> m_bufferPool;
    mutable int m_nextBuffer = 0;
    int m_maxBufCount = 0;

    void ensureBufferPool(int bufCount) const;

    // ── Binning state (mutable for re-building per call) ─────────────
    mutable std::vector<int> m_binHead;
    mutable std::vector<int> m_binNext;
    mutable std::vector<int> m_binCells;
    mutable double m_binSize = 0.0;
    mutable int m_binNx = 0, m_binNy = 0, m_binNz = 0;
    mutable double m_bboxMin[3] = {0,0,0};
    mutable double m_bboxMax[3] = {0,0,0};
};

} // namespace mtp_kim
