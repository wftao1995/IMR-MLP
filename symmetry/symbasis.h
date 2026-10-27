// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once 
#include "external/spglib_driver.h"

namespace Symmetry3D {
    void get_symm_basis(symdataset &dataset, Eigen::MatrixXd &result);

    bool symmetrize(Atoms &atom, std::vector<symopt_type> &symops, double prec = 0.1);

    void apply_symop(symopt_type &op, Atoms &atom);
};