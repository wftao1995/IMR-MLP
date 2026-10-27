// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "utils/gputils.h"
#include "base/structures.h"
#include "potentials/potential.h"
#include "io/gpio.h"
#include "periodic_subspace.h"
#include <fstream>

/**
 * @brief DIMER method for saddle point (transition state) search.
 *
 * Creates a dimer (two images separated by ΔR along direction n̂) and
 * iteratively rotates (to find the lowest curvature mode) and translates
 * (to climb toward the saddle point).
 *
 * Supports:
 *   - Fixed-cell and variable-cell (VC-DIMER) calculations
 *   - Multiple optimizers for rotation/translation sub-steps (FIRE, LBFGS, CG, SGD)
 *   - Strict MODECAR input; random direction only when no file is specified
 *
 * Usage via CLI: imrmlp dimer dimer.ini
 */
class DimerDriver : public ConnectSettings {
public:
    // ── Dimer geometry ──────────────────────────────────────────────
    double ddr = 5e-3;            // Dimer separation (ΔR)
    double rot_force_min = 1e-2;  // |F_rot| below this: skip rotation entirely

    // ── MODECAR ─────────────────────────────────────────────────────
    int random_seed = 0;          // Reproducible default; -1 requests random_device.
    std::string modecarf = "";    // MODECAR file for initial mode direction

    // ── VC-DIMER flags ──────────────────────────────────────────────
    bool vc_flag = false;         // Variable-cell DIMER
    bool periodic_subspace = false; // Explicit mixed coordinates for partial PBC VC.
    bool subspace_active = false;
    PeriodicSubspaceCoordinates subspace;
    bool symm_strain = true;      // Use symmetric strain (6-component Voigt)
    bool restore_F_cholesky = true;
    bool recast_atomic_force = true;
    double pressure = 0.0;        // External pressure (GPa)
    double cell_factor = 1.0;     // Cell preconditioning factor

    // ── Rotation sub-optimizer ──────────────────────────────────────
    std::string rot_driver = "fire";
    int rot_max_iter = 50;
    double rot_gtol = 1e-3;
    double rot_max_step = 0.5;
    double rot_first_step = 0.01;

    // ── Translation sub-optimizer ───────────────────────────────────
    std::string trn_driver = "fire";
    int trn_max_iter = 200;
    double trn_gtol = 1e-2;
    double trn_max_step = 0.2;
    double trn_first_step = 0.01;

    // ── Outer loop convergence ──────────────────────────────────────
    double dimer_gtol = 1e-2;     // Max |g| component for convergence
    int max_iter = 1000;
    double min_curvature = -1e-6; // Must be negative (saddle indicator)

    // ── Output ──────────────────────────────────────────────────────
    bool write_traj = false;
    std::string trajfile = "dimer.trj";
    std::string summary = "dimer.dat";
    int nblock = 10;

    // ── Debug ───────────────────────────────────────────────────────
    int debug_level = 0;

    // ═══════════════════════════════════════════════════════════════
    // Internal state (set after init)
    // ═══════════════════════════════════════════════════════════════

    Potentials::AnyPotential *pot = nullptr;
    GPIO *iobj = nullptr;

    // Optimization variable dimensions
    int nlen = 0;           // total length of optimization vectors
    int offset = 0;         // cell DOF count (0, 6, or 9)
    int natom = 0;          // number of atoms
    int npx = 0;            // 3 * natom

    // Coordinate transform reference cell
    Eigen::Matrix3d refcell = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d ivrefcell = Eigen::Matrix3d::Identity();

    // Center structure and dimer endpoints
    Atoms atom_center;       // center structure (current best estimate)
    Atoms atom_plus;         // center + ddr * modedir
    Atoms atom_minus;        // center - ddr * modedir

    // Optimization vectors
    Eigen::VectorXd xx_center;      // center point in optimization space
    Eigen::VectorXd modedir;        // current mode direction (unit vector)
    Eigen::VectorXd modedir_init;   // initial mode direction

    // Force buffers
    Eigen::VectorXd g_plus;         // gradient on atom_plus
    Eigen::VectorXd g_minus;        // gradient on atom_minus

    // Convergence tracking
    int iter = 0;
    double curvature = 0.0;
    double max_rot_force = 0.0;
    double max_trn_force = 0.0;
    double max_center_gradient = 0.0;
    double energy = 0.0;
    bool converged = false;

    // Trajectory
    std::ofstream trjofs;

    // ═══════════════════════════════════════════════════════════════
    // Method declarations
    // ═══════════════════════════════════════════════════════════════

    DimerDriver() { InitSettings(); }
    DimerDriver(GPSettings settings);
    void InitSettings();

    // Initialization
    void validate_settings() const;
    void refresh_state();
    void init(Atoms &center_atom);
    void read_modecar(const std::string &fnm);
    void random_modedir();

    // Coordinate transforms (mirror RelaxationDrvV2 patterns)
    void LatPos2X(const Atoms &at, Eigen::VectorXd &x);
    void X2LatPos(const Eigen::VectorXd &x, Atoms &at);

    // Gradient mapping: Atom forces + stress → gradient vector g
    void LatPos2G(Atoms &at, Eigen::VectorXd &g, double &ene);

    // Dimer endpoint management
    void build_dimer_endpoints();
    void eval_endpoints();

    // Force/curvature computation
    double compute_curvature();
    void compute_rotational_force(Eigen::VectorXd &g_out);
    void compute_translational_force(Eigen::VectorXd &g_out);

    // Functor interfaces for optimizers
    // Rotation functor: minimize curvature w.r.t. mode direction
    double operator_rot(const Eigen::VectorXd &mode_in, Eigen::VectorXd &g_out);
    // Translation functor: climb toward saddle
    double operator_trn(const Eigen::VectorXd &x_in, Eigen::VectorXd &g_out);

    // Main algorithm
    int run();

    // Convergence check
    bool check_convergence();

    // Output
    // Versioned MODECAR stores the coordinate convention and VC reference cell.
    void write_modecar(const std::string &fnm) const;
    void write_summary(const std::string &fnm);
    void print_status();
};
