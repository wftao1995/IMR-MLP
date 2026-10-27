#pragma once

#include "base/structures.h"

#include <Eigen/Dense>

#include <limits>
#include <string>
#include <vector>

namespace TIReferenceIO {

struct SparseOptions {
    double cutoff_angstrom = 0.0;
    double block_threshold = 0.0;
    double minimum_frequency_thz = 0.0;
    bool stabilize = true;
};

struct SparseDiagnostics {
    int total_offdiagonal_blocks = 0;
    int retained_offdiagonal_blocks = 0;
    double retained_fraction = 1.0;
    double relative_hessian_change = 0.0;
    double acoustic_sum_rule_error = 0.0;
    double minimum_frequency_before_thz = std::numeric_limits<double>::quiet_NaN();
    double minimum_frequency_after_thz = std::numeric_limits<double>::quiet_NaN();
    double isotropic_stabilizer = 0.0;
};

struct SparseResult {
    Eigen::MatrixXd hessian;
    SparseDiagnostics diagnostics;
};

SparseResult truncate_force_constants(const Atoms &reference,
                                      const Eigen::MatrixXd &hessian,
                                      const std::vector<double> &masses,
                                      const SparseOptions &options);

Eigen::VectorXd internal_mode_eigenvalues(const Eigen::MatrixXd &hessian,
                                          const std::vector<double> &masses);

void write_force_constants(const std::string &filename,
                           const Eigen::MatrixXd &hessian);

Atoms write_lammps_reference(const std::string &filename,
                             const Atoms &reference,
                             const Eigen::MatrixXd &hessian,
                             double energy0);

} // namespace TIReferenceIO
