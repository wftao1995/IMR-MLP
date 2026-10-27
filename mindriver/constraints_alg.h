// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once 
#include "base/structures.h"
#include "libalglib/ap.h"

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
// loop over all regulators, do not set box constraint here
struct regulator_alg {
    // to be inited
    // if of size 0, then use all x
    int n_output = 1; // number of constraints functions in this regulator
    std::vector<double> bndl, bndu; // lower/upper bounds of output value, may be +inf or -inf
    Eigen::VectorXi inputs_id;  // input indices of x, used for copy values to inputs_x
    Eigen::VectorXd inputs_x;   // input value of x
    Eigen::VectorXd outputs_f; // outputs for function value, usually same as inputs
    double **outputs_j_ptr; // pointing to outputs_j
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> outputs_j; // outputs for jac, length(f) * length(x)
    std::vector<int> integer_specs;  // indices
    std::vector<double> double_specs; // i.e hold upper bound
    void *data = NULL; // data

    // functions
    void (*compute)(const double *x,  // x, vector
                    double *f,        // f, vector
                    double **j,       // jac, matrix, row major, each row is dfi/dxj
                    const int *,     // inter_specs
                    const double *,  // double_specs
                    void *p_driver,  // p_driver
                    void *data
    ) = nullptr;
};


namespace Constraints {
    ;
    
    // int phrase_json_config_alg(nlohmann::json &jsonobj, std::vector<regulator_alg> &regs);
};
