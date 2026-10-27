// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once 
#include "base/structures.h"

/*
* reg : {
*   inputs : ,
*   outputs : ,
*   weight : ,
*   type : ,
*   internal : ,
* }
* 24/10/21: use minnlc
*/
namespace Constraints { class Expression; }

// when ne, g <= 0
struct regulator {
    enum {
        NONE,
        CONSTANT_VOLUME, // fix volume
        CONSTANT_VECTOR, // fix lattice vector length
        CONSTANT_ANGLE,  // fix lattice angles
        COUPLE_VECTOR,   // fix lattice vector length ratio
        MINDIST,         // fix minimal distance
        DISTANCE,        // fixed initial-image distance, two atoms
        ANGLE,           // fixed initial-image angle, three atoms \theta{i-j-k}
        FORMULA,         // Parsed physical expression with analytic derivatives.
    } regulator_t = NONE;
    // to be inited
    std::vector<int> inputs;  // input of x
    std::vector<int> outputs; // outputs for g, usually same as inputs
    std::vector<int> integer_specs;  // indices
    std::vector<double> double_specs; // i.e hold upper bound
    double double_target = 0; // i.e target volume
    double weight_init = 1.0;
    double weight = 1;
    double weight_step = 1;
    double weight_multply = 1.0;
    // 0: quadratic equality penalty; +/-1: one-sided quadratic penalty.
    // 10: augmented-Lagrangian equality. Types 9/11 are unsupported.
    int type = 0;
    // int lpos; // pos of lambda in x vector, auto init in relax class
    void *data = NULL;
    std::vector<double> internal_datas; // i.e volume for constant_volume

    std::shared_ptr<const Constraints::Expression> expression;
    bool bounded = false;
    double lower = 0, upper = 0; // Bounds on the evaluated residual, not squared physical values.

    int curr_constraint_iter = 0;
    double tol = 1e-4;

    // vals
    double lambda_init = 0.0; // start from pure penately
    double lambda = 0;

    // buffers 
    std::vector<double> x;
    std::vector<double> g;
    double voliate = 0;
    double mos = 0, mos2 = 0;
    /*
    for ieq constraint, f must be 0 if conditions are meet
    */
    double f = 0;
    double g0 = 0;
    bool to_target_g0 = false;
    bool update = true;

    // Fixed initial nearest-image branches; copied with regulator/checkpoints.
    bool atomic_geometry_bound = false;
    std::array<Eigen::RowVector3i, 2> atomic_images{Eigen::RowVector3i::Zero(), Eigen::RowVector3i::Zero()};

    // functions
    void (*compute)(double *,        // x
                    int,             // nx
                    double *,        // g
                    int,             // ng
                    double *,        // f, scalar
                    double,          // g0
                    Atoms *,         // atoms object
                    int *,           // indices
                    void *           // other data if nesseary
    ) = nullptr;
};


namespace Constraints {
    struct AtomicGeometryEvaluation {
        double residual = 0;
        double physical_value = 0; // Angstrom or degrees
        std::vector<Eigen::RowVector3d> cartesian_gradients; // integer_specs order
        Eigen::Matrix3d explicit_cell_gradient = Eigen::Matrix3d::Zero();
    };
    void atomic_geometry_setup(regulator &, const Atoms &);
    AtomicGeometryEvaluation evaluate_atomic_geometry(const regulator &, const Atoms &);

    struct symop {
        int *mapdata = nullptr;
        int **mapping = nullptr;
        Eigen::Matrix3d rotation; // matrix P
        Eigen::RowVector3d translation; // vector W, in direct representation
        ~symop(){
            if (mapdata) delete [] mapdata;
            if (mapping) delete [] mapping;
        }
        Eigen::Matrix3d cell;
        std::vector<Eigen::RowVector3d> pos_buffer, pos_orig, pos_new;
        int natoms;
        // X' = XR + T
        void apply(const double *x, int _natoms) {
            // resize buffer if required
            pos_buffer.resize(_natoms);
            pos_new.resize(_natoms);
            pos_orig.resize(_natoms);
            // apply on cell, L' = LR
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    cell(a, b) = x[3*a+b];
                } 
            }
            // apply on pos, R' = RP
            Eigen::RowVector3d pos;
            for (int i = 0; i < _natoms; ++i) {
                pos(0) = x[9+3*i];
                pos(1) = x[9+3*i+1];
                pos(2) = x[9+3*i+2];
                pos = pos * cell;
                pos_orig[i] = pos;
                pos_buffer[i] = pos * rotation;
            }
            // reorder pos_buffer by mapping
            // map is orig id -> id at new pos
            for (int i = 0; i < _natoms; ++i) {
                pos_new[mapping[i][1]] = pos_buffer[mapping[i][0]];
            }
        }
    };
    // l11, l12, l13, l21, ...
    /* // Input: lattice matrix, length 9
        Input: None, just Atoms object, or just pass x.data(), nx = 0 or 9
     */
    void constant_volume(double *cell, int nx, double *g, int ng, double *f, 
                         double g0,
                         Atoms *patom = nullptr,
                         int *ids = nullptr,  
                         void *pdata = nullptr);
    void constant_volume_setup(regulator &reg, Atoms &atom);
    /* Input: lattice vector, length 3
     */
    void constant_vector(double *cell, int nx, double *g, int ng, double *f, 
                         double g0,
                         Atoms *patom = nullptr,
                         int *ids = nullptr,  
                         void *pdata = nullptr);
    void constant_vector_setup(regulator &reg, Atoms &atom, int l);
    /* Input: two lattice vectors, length 6
       alpha: b c
       beta: c a
       gamma: a b
     */
    void constant_angle(double *cell, int nx, double *g, int ng, double *f, 
                         double g0,
                         Atoms *patom = nullptr,
                         int *ids = nullptr,  
                         void *pdata = nullptr);
    void constant_angle_setup(regulator &reg, Atoms &atom, int l1, int l2);
    /* Input: two lattice vectors to be coupled, length 6
     */
    void couple_vector(double *x, int nx, double *g, int ng, double *f,
                         double g0,
                         Atoms *patom = nullptr,
                         int *ids = nullptr,  
                         void *pdata = nullptr);
    void couple_vector_setup(regulator &reg, Atoms &atom, int l1, int l2);
    


    // These constraints are related with atom positions, constraints with _f suffix means 
    // car mode
    // Just need to specify which version to use in setup function
    // Just need to call setup function in INIT method


    /* Input:
        x: selected lattice positions
    */
    void mindist(double *x, int nx, double *g, int ng, double *f,
                         double g0,
                         Atoms *patom = nullptr,
                         int *ids = nullptr,  
                         void *pdata = nullptr);        
    void mindist_setup(regulator &reg, Atoms &atom, double min_dist, int xoffset);
    void mindist_f(double *x, int nx, double *g, int ng, double *f,
                         double g0,
                         Atoms *patom = nullptr,
                         int *ids = nullptr,  
                         void *pdata = nullptr);   
    void mindist_f_setup(regulator &reg, Atoms &atom, double min_dist, int xoffset);
};
