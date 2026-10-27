// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "train_linear_potential.h"
#include "utils/comm.h"
#include "train/collective_error.h"
#include <cmath>
#include <cstring>
#include <fmt/format.h>

#ifdef IMRMLP_MPI
#include <mpi.h>
#endif

namespace {
void validate_data(const std::vector<Atoms> &atoms, const std::vector<Neighborhoods> &nbhs)
{
    int invalid = atoms.size() != nbhs.size();
    if (!invalid) {
        for (size_t i = 0; i < atoms.size(); ++i) {
            const auto &a = atoms[i];
            if (a.size() == 0 || a.size() != nbhs[i].size() ||
                a.chemmap_.size() != a.pos_.size() ||
                (a.has_forces() && a.forces_.size() != a.pos_.size()) ||
                (!a.atomic_fitness.empty() && a.atomic_fitness.size() != a.pos_.size()))
                invalid = 1;
        }
    }
    if (invalid)
        GPERROR("Linear training requires matching nonempty structures, neighborhoods and per-atom labels");
    for (const auto &a : atoms) {
        if ((a.has_energy() && !std::isfinite(a.energy_)) ||
            (a.has_stresses() && !a.stress_.allFinite()) ||
            !std::isfinite(a.fitness) || a.fitness < 0)
            invalid = 1;
        if (a.has_forces())
            for (const auto &force : a.forces_)
                if (!force.allFinite()) invalid = 1;
        for (double weight : a.atomic_fitness)
            if (!std::isfinite(weight) || weight < 0) invalid = 1;
    }
    if (invalid)
        GPERROR("Linear training requires finite labels and finite nonnegative fitness weights");
}

// Match the geometry used by finite_difference_hvp_loss without copying labels
// or an optional dense coordinate Hessian into every displaced structure.
Atoms hvp_geometry(const Atoms &a)
{
    Atoms out;
    out.cell_ = a.cell_;
    out.pos_ = a.pos_;
    out.chemmap_ = a.chemmap_;
    out.pbc_ = a.pbc_;
    out.cartesian = a.cartesian;
    out.id = 0;
    out.charges_ = a.charges_;
    out.collinear_ = a.collinear_;
    out.magmom_scalar_ = a.magmom_scalar_;
    out.magmoms_ = a.magmoms_;
    out.global_attribs_d_ = a.global_attribs_d_;
    out.global_attribs_i_ = a.global_attribs_i_;
    out.global_attribs_v_ = a.global_attribs_v_;
    out.global_attribs_vi_ = a.global_attribs_vi_;
    out.features_ = a.features_;
    return out;
}

Eigen::MatrixXd force_design(Potentials::AnyPotential &pot, Atoms &atom, int features)
{
    for (const auto &position : atom.pos_)
        if (!position.allFinite()) GPERROR("Nonfinite position in linear HVP displacement");
    Neighborhoods nbhs(atom, pot.cutoff);
    Eigen::MatrixXd design = Eigen::MatrixXd::Zero(3 * atom.size(), features);
    std::vector<double> descriptor(features), gradient;
    for (int i = 0; i < atom.size(); ++i) {
        auto &nbh = nbhs[i];
        const int nnb = nbh.nbhcount_;
        if (nnb == 0) continue;
        gradient.assign(size_t(features) * nnb * 3, 0.);
        pot.linear_compute_descriptor_grad(nbh, descriptor.data(), gradient.data());
        for (int f = 0; f < features; ++f)
            for (int j = 0; j < nnb; ++j)
                for (int axis = 0; axis < 3; ++axis) {
                    const double derivative = gradient[size_t(f) * nnb * 3 + j * 3 + axis];
                    design(3 * i + axis, f) += derivative;
                    design(3 * nbh.indices_[j] + axis, f) -= derivative;
                }
    }
    if (!design.allFinite()) GPERROR("Nonfinite force design matrix in linear HVP fitting");
    return design;
}
}

double TrainerLinearPotential::energy_weight(const Atoms &atom) const
{
    return weight_eng / std::pow(atom.size(), weight_eng_power) * atom.fitness;
}

double TrainerLinearPotential::force_weight(const Atoms &atom) const
{
    return weight_force / std::pow(atom.size(), weight_force_power);
}

double TrainerLinearPotential::stress_weight(const Atoms &atom) const
{
    return weight_stress / std::pow(atom.size(), weight_stress_power);
}

void TrainerLinearPotential::init(std::vector<Atoms> &atoms,
                                   std::vector<Neighborhoods> &nbhs,
                                   Potentials::AnyPotential *pot)
{
    initialized_ = assembled_ = false;
    n_features_ = 0;
    emonitor.init();
    if (!pot) GPERROR("Linear training requires a potential");
    Training::collective_try([&] { validate_data(atoms, nbhs); });
    if (debug) {
        debug_guard_ = std::make_unique<ScopedStreamRedirect>(
            global_debugstream,
            new std::ofstream("DEBUG_" + std::to_string(mpi.rank)));
    }
    p_pot = pot;
    if (!p_pot->supports_linear_fit())
        GPERROR("Potential does not support linear fitting");

    n_features_ = p_pot->get_NL_coeffs_count();
    if (n_features_ <= 0)
        GPERROR("Potential reports zero or negative feature count");

    training_set_ = atoms;
    efs_set_ = atoms;
    trainint_set_nbhs_ = nbhs;
    train_set_size = training_set_.size();
    train_set_size_all = train_set_size;
#ifdef IMRMLP_MPI
    MPI_Allreduce(&train_set_size, &train_set_size_all, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, mpi.comm);
#endif
    if (train_set_size_all == 0) GPERROR("Linear training requires a nonempty training set");
    emonitor.t_size_all = train_set_size_all;

    ATA_.resize(n_features_, n_features_);
    ATy_.resize(n_features_);
    ATA_mpi_.resize(n_features_, n_features_);
    ATy_mpi_.resize(n_features_);
    initialized_ = true;
}

void TrainerLinearPotential::init_valid(std::vector<Atoms> &atoms,
                                         std::vector<Neighborhoods> &nbhs)
{
    validate_data(atoms, nbhs);
    valid_set_ = atoms;
    efs_valid_set_ = atoms;
    valid_set_nbhs_ = nbhs;
    valid_set_size = valid_set_.size();
}

void TrainerLinearPotential::Assemble()
{
    if (!initialized_) GPERROR("Linear trainer must be initialized before assembly");
    assembled_ = false;
    ATA_.setZero();
    ATy_.setZero();

    Training::collective_try([&] {
    if (!std::isfinite(weight_eng) || weight_eng < 0 ||
        !std::isfinite(weight_force) || weight_force < 0 ||
        !std::isfinite(weight_stress) || weight_stress < 0 ||
        !std::isfinite(weight_eng_power) || !std::isfinite(weight_force_power) ||
        !std::isfinite(weight_stress_power) || !std::isfinite(reg_L2) || reg_L2 < 0)
        GPERROR("Linear training weights and regularization must be finite and nonnegative");
    hvp_fit.validate();
    for (size_t s = 0; s < train_set_size; ++s) {
        Atoms &atom = training_set_[s];
        Neighborhoods &nbhs = trainint_set_nbhs_[s];
        const int natoms = atom.size();
        if (natoms == 0) continue;

        const bool has_E = weight_eng > 0 && atom.has_energy();
        const bool has_F = weight_force > 0 && atom.has_forces();
        const bool has_S = weight_stress > 0 && atom.has_stresses();

        const double w_E = has_E ? energy_weight(atom) : 0;
        const double w_F = has_F ? force_weight(atom) : 0;
        const double w_S = has_S ? stress_weight(atom) : 0;
        if (!std::isfinite(w_E) || !std::isfinite(w_F) || !std::isfinite(w_S))
            GPERROR("Linear training produced a nonfinite effective weight");

        // ── Phase 1: Compute all per-atom descriptors and gradients ────
        // desc_all[i][f] = descriptor f for center atom i
        // grad_all[i]   = gradient tensor for center atom i (flat row-major)
        std::vector<std::vector<double>> desc_all(natoms);
        std::vector<std::vector<double>> grad_all(natoms);
        for (int i = 0; i < natoms; ++i)
            desc_all[i].resize(n_features_);

        for (int i = 0; i < natoms; ++i) {
            Neighborhood &nbh = nbhs[i];
            if (has_F || has_S) {
                const int nnb = nbh.nbhcount_;
                if (nnb == 0) {
                    if (has_E) p_pot->linear_compute_descriptor(nbh, desc_all[i].data());
                    continue;
                }
                grad_all[i].resize(n_features_ * nnb * 3);
                p_pot->linear_compute_descriptor_grad(nbh, desc_all[i].data(), grad_all[i].data());
            } else {
                p_pot->linear_compute_descriptor(nbh, desc_all[i].data());
            }
        }

        // ── Phase 2: Energy: A_E_row = sum_i desc_i, y = E_ref ────────
        if (has_E) {
            Eigen::VectorXd A_E_row = Eigen::VectorXd::Zero(n_features_);
            for (int i = 0; i < natoms; ++i)
                for (int f = 0; f < n_features_; ++f)
                    A_E_row[f] += desc_all[i][f];
            for (int f = 0; f < n_features_; ++f) {
                for (int g = 0; g < n_features_; ++g)
                    ATA_(f, g) += w_E * A_E_row[f] * A_E_row[g];
                ATy_[f] += w_E * A_E_row[f] * atom.energy_;
            }
        }

        // ── Phase 3: Forces ────────────────────────────────────────────
        if (has_F) {
            // Build reverse mapping: global_j -> list of (center_i, local_idx)
            std::vector<std::vector<std::pair<int,int>>> rev(natoms);
            for (int i = 0; i < natoms; ++i) {
                Neighborhood &nbh = nbhs[i];
                for (int j = 0; j < nbh.nbhcount_; ++j)
                    rev[nbh.indices_[j]].emplace_back(i, j);
            }

            // Accumulate per force component
            // F_{j,a} = sum_f w_f * [
            //     + sum_k grad_j(f, k, a)            (center-role: j as center)
            //     - sum_{i: j in nbh_i} grad_i(f, j_local, a)  (neighbor-role)
            // ]
            Eigen::VectorXd A_F_row(n_features_);
            for (int j = 0; j < natoms; ++j) {
                const double w_F_j = (atom.atomic_fitness.size() > 0)
                    ? w_F * atom.atomic_fitness[j] : w_F;
                if (w_F_j <= 0) continue;

                // Quick check: any contribution?
                if (rev[j].empty() && nbhs[j].nbhcount_ == 0) continue;

                for (int a = 0; a < 3; ++a) {
                    A_F_row.setZero();
                    // Forward (center-role): atom j as center
                    const int nnb_j = nbhs[j].nbhcount_;
                    const double *g_j = grad_all[j].data();
                    for (int k = 0; k < nnb_j; ++k)
                        for (int f = 0; f < n_features_; ++f)
                            A_F_row[f] += g_j[f * nnb_j * 3 + k * 3 + a];
                    // Reverse (neighbor-role): other centers have j as neighbor
                    for (auto &[ci, local_j] : rev[j]) {
                        const int nnb = nbhs[ci].nbhcount_;
                        const double *g = grad_all[ci].data();
                        for (int f = 0; f < n_features_; ++f)
                            A_F_row[f] -= g[f * nnb * 3 + local_j * 3 + a];
                    }
                    double F_ref = atom.forces_[j][a];
                    for (int f = 0; f < n_features_; ++f) {
                        for (int g = 0; g < n_features_; ++g)
                            ATA_(f, g) += w_F_j * A_F_row[f] * A_F_row[g];
                        ATy_[f] += w_F_j * A_F_row[f] * F_ref;
                    }
                }
            }
        }

        // ── Phase 4: Stress ────────────────────────────────────────────
        if (has_S) {
            // S_pred_{a,b} = sum_i sum_f w_f * (-sum_j grad_i[f,j,a] * pos_j[b])
            // For each (a,b): A_row_f = -sum_i sum_j grad_i[f,j,a] * pos_j[b]
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    Eigen::VectorXd A_S_row = Eigen::VectorXd::Zero(n_features_);
                    for (int i = 0; i < natoms; ++i) {
                        Neighborhood &nbh = nbhs[i];
                        const int nnb = nbh.nbhcount_;
                        if (nnb == 0) continue;
                        const double *g = grad_all[i].data();
                        for (int f = 0; f < n_features_; ++f)
                            for (int j = 0; j < nnb; ++j)
                                A_S_row[f] -= g[f * nnb * 3 + j * 3 + a] * nbh.pos_[j][b];
                    }
                    double S_ref = atom.stress_(a, b);
                    for (int f = 0; f < n_features_; ++f) {
                        for (int g = 0; g < n_features_; ++g)
                            ATA_(f, g) += w_S * A_S_row[f] * A_S_row[g];
                        ATy_[f] += w_S * A_S_row[f] * S_ref;
                    }
                }
            }
        }

        const double w_H = hvp_fit.observation_weight(atom);
        if (w_H > 0) {
            if (!atom.cartesian) GPERROR("HVP fitting requires Cartesian positions");
            for (int k = 0; k < atom.hvp_directions_.cols(); ++k) {
                const auto direction = atom.hvp_directions_.col(k);
                double max_norm = 0;
                for (int i = 0; i < natoms; ++i)
                    max_norm = std::max(max_norm, direction.segment<3>(3 * i).stableNorm());
                const double epsilon = hvp_fit.step / max_norm;
                if (!std::isfinite(epsilon) || epsilon <= 0)
                    GPERROR("Invalid HVP displacement scale");
                Atoms plus = hvp_geometry(atom), minus = hvp_geometry(atom);
                for (int i = 0; i < natoms; ++i) {
                    const Eigen::RowVector3d delta = epsilon * direction.segment<3>(3 * i).transpose();
                    plus.pos_[i] += delta;
                    minus.pos_[i] -= delta;
                }
                // H v = -dF/dR v. Rebuild both neighborhoods at fixed cell.
                const Eigen::MatrixXd rows = -(force_design(*p_pot, plus, n_features_) -
                    force_design(*p_pot, minus, n_features_)) / (2 * epsilon);
                ATA_.noalias() += w_H * rows.transpose() * rows;
                ATy_.noalias() += w_H * rows.transpose() * atom.hvp_targets_.col(k);
            }
        }
    }

    if (!ATA_.allFinite() || !ATy_.allFinite())
        GPERROR("Linear training produced a nonfinite design matrix or target");
    });

    // ── MPI reduce ─────────────────────────────────────────────────────
#ifdef IMRMLP_MPI
    MPI_Reduce(ATA_.data(), ATA_mpi_.data(), n_features_ * n_features_,
               MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(ATy_.data(), ATy_mpi_.data(), n_features_,
               MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    if (mpi.rank == 0) {
        ATA_ = ATA_mpi_;
        ATy_ = ATy_mpi_;
    }
#endif
    Training::collective_check(mpi.rank == 0 && (!ATA_.allFinite() || !ATy_.allFinite()) ?
        "Linear training produced a nonfinite reduced design matrix or target" : "");
    Training::collective_check(mpi.rank == 0 && ATA_.diagonal().maxCoeff() <= 0 ?
        "Linear training has no effective labels with positive weight" : "");
    assembled_ = true;
}

void TrainerLinearPotential::Solve()
{
    if (!initialized_ || !assembled_)
        GPERROR("Linear trainer must be initialized and assembled before solving");
    Eigen::VectorXd w(n_features_);
    Training::collective_try([&] {
    if (mpi.rank == 0) {
        if (!std::isfinite(reg_L2) || reg_L2 < 0)
            GPERROR("Linear training regularization must be finite and nonnegative");
        // Apply L2 regularisation
        Eigen::MatrixXd ATA_reg = ATA_;
        if (reg_L2 > 0) {
            for (int i = 0; i < n_features_; ++i)
                ATA_reg(i, i) += reg_L2;
        }
        if (!ATA_reg.allFinite()) GPERROR("Nonfinite regularized linear training matrix");

        // Solve via LDLT
        Eigen::LDLT<Eigen::MatrixXd> solver(ATA_reg);
        if (solver.info() != Eigen::Success) {
            Warning("LDLT decomposition failed, trying full LU");
            Eigen::FullPivLU<Eigen::MatrixXd> lu(ATA_reg);
            w = lu.solve(ATy_);
        } else {
            w = solver.solve(ATy_);
            Message(fmt::format("Solved {} coefficients, ATA condition = {:.2e}",
                      n_features_, solver.vectorD().maxCoeff() /
                      std::max(1e-300, solver.vectorD().minCoeff())));
        }
        if (!w.allFinite()) GPERROR("Linear training solver produced nonfinite coefficients");
    }
    });

#ifdef IMRMLP_MPI
    MPI_Bcast(w.data(), n_features_, MPI_DOUBLE, 0, mpi.comm);
#endif
    p_pot->nl_set_coeffs(w.data());
    evaluate_errors(training_set_, efs_set_, trainint_set_nbhs_, emonitor);
}

void TrainerLinearPotential::Train()
{
    Message("Assembling design matrix ...");
    Assemble();

    Message("Solving linear system ...");
    Solve();

    Training::collective_try([&] {
        if (mpi.rank == 0)
            save_current(potfilesave);
    });

    int validate = monit_valid_err && valid_set_size > 0;
#ifdef IMRMLP_MPI
    MPI_Allreduce(MPI_IN_PLACE, &validate, 1, MPI_INT, MPI_MAX, mpi.comm);
#endif
    if (validate)
        evaluate_errors(valid_set_, efs_valid_set_, valid_set_nbhs_, emonitor_valid);
}

void TrainerLinearPotential::evaluate_errors(const std::vector<Atoms> &references,
                                             std::vector<Atoms> &predictions,
                                             std::vector<Neighborhoods> &nbhs,
                                             ErrorMonitor &monitor)
{
    monitor.init();
    monitor.t_size_all = references.size();
#ifdef IMRMLP_MPI
    MPI_Allreduce(MPI_IN_PLACE, &monitor.t_size_all, 1, MPI_INT, MPI_SUM, mpi.comm);
#endif
    Training::collective_try([&] {
    for (size_t i = 0; i < references.size(); ++i) {
        const auto &atom = references[i];
        auto &prediction = predictions[i];
        p_pot->getEFS(prediction, nbhs[i]);
        if (atom.has_energy()) {
            double error = (prediction.energy_ - atom.energy_) / atom.size();
            monitor.sq_energy_at_error += error * error;
            ++monitor.nenergy;
        }
        if (atom.has_forces()) {
            for (int j = 0; j < atom.size(); ++j) {
                monitor.sq_force_error += (prediction.forces_[j] - atom.forces_[j]).squaredNorm();
                ++monitor.nforces;
            }
        }
        if (atom.has_stresses()) {
            monitor.sq_stress_error += (prediction.stress_ - atom.stress_).squaredNorm();
            ++monitor.nstresstensor;
        }
        monitor.add_hvp(Training::finite_difference_hvp_loss(*p_pot, atom, hvp_fit));
    }
    });
    monitor.report();
#ifdef IMRMLP_MPI
    MPI_Bcast(&monitor.erms, 1, MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(&monitor.frms, 1, MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(&monitor.srms, 1, MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(&monitor.hvp_rms, 1, MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(&monitor.loss_hvp, 1, MPI_DOUBLE, 0, mpi.comm);
#endif
}

void TrainerLinearPotential::save_current(const std::string &fnm)
{
    if (!initialized_) GPERROR("Linear trainer must be initialized before saving");
    p_pot->Save(fnm);
}
