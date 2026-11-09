// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "efs_fit_weights.h"
#include "potentials/potential_mtp.h"
#include <array>
#include <string>
#include <vector>

namespace Training
{

// Residuals are centered at the input model's coefficients to avoid subtracting
// two large total energies when evaluating a good fit. No regularization here.
struct EFSQuadratic
{
    Eigen::MatrixXd gram;
    Eigen::VectorXd residual_rhs;
    double residual_norm = 0;
    double observations = 0;
    // Empty on the default path. Optional QR factor of [design, residual].
    Eigen::MatrixXd square_root;

    double loss(const Eigen::VectorXd &delta) const;
};

class LinearEFSSystem
{
  public:
    std::array<EFSQuadratic, 3> channels; // E, F, full 3x3 virial
    Eigen::VectorXd anchor;

    // Local assembly, using bounded row batches instead of the full design
    // matrix. Global channel weights may be set to one for reweightable export.
    LinearEFSSystem(Potentials::PotMTP &pot, std::vector<Atoms> &atoms,
                    const EFSFitWeights &weights, int coefficient_offset = 0,
                    bool square_root = false);
    void reduce(); // collective over mpi.comm, same result on every rank
    Eigen::MatrixXd matrix(double we, double wf, double ws) const;
    Eigen::VectorXd rhs(double we, double wf, double ws) const;
    Eigen::MatrixXd root_factor(double we, double wf, double ws) const;
    double loss(const Eigen::VectorXd &coeffs, double we, double wf, double ws) const;
    void require_observations(double we, double wf, double ws) const;
};

void BuildReducedSystem(const Eigen::MatrixXd &mat, const Eigen::VectorXd &vec,
                        const std::vector<int> &indices, Eigen::MatrixXd &reduced_mat,
                        Eigen::VectorXd &reduced_vec);

void RemoveOneBasis(const Eigen::MatrixXd &mat, const Eigen::VectorXd &vec, int removed,
                    Eigen::MatrixXd &reduced_mat, Eigen::VectorXd &reduced_vec);

bool solve_normal_equations(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, Eigen::VectorXd &x,
                            const std::string &method);

// Expand an already validated selection. prefix coefficients (species biases)
// remain fixed; selected IDs are relative to the remaining coefficient block.
template <class Indices>
Eigen::VectorXd expand_linear_coefficients(const Eigen::VectorXd &reduced, const Indices &indices,
                                           int total, int prefix = 0)
{
    Eigen::VectorXd full = Eigen::VectorXd::Zero(total);
    full.head(prefix) = reduced.head(prefix);
    int column = prefix;
    for (int id : indices)
        full[prefix + id] = reduced[column++];
    return full;
}
} // namespace Training
