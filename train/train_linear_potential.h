// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "error_monitor.h"
#include "hvp_fit.h"
#include "potentials/potential.h"
#include "utils/gputils.h"
#include <Eigen/Dense>

class TrainerLinearPotential : public ConnectSettings
{
private:
    Potentials::AnyPotential *p_pot = nullptr;
    int n_features_ = 0;
    bool initialized_ = false;
    bool assembled_ = false;

    // Design matrix: A^T A and A^T y, assembled incrementally
    Eigen::MatrixXd ATA_;
    Eigen::VectorXd ATy_;

    // MPI buffers
    Eigen::MatrixXd ATA_mpi_;
    Eigen::VectorXd ATy_mpi_;

    // Training data
    std::vector<Atoms> training_set_, efs_set_;
    std::vector<Atoms> valid_set_, efs_valid_set_;
    std::vector<Neighborhoods> trainint_set_nbhs_, valid_set_nbhs_;

    size_t train_set_size = 0;
    size_t train_set_size_all = 0;
    size_t valid_set_size = 0;

    // Weight parameters
    double weight_eng = 1;
    double weight_force = 0.01;
    double weight_stress = 0.0001;
    double weight_eng_power = 1.0;
    double weight_force_power = 0.0;
    double weight_stress_power = 1.0;
    Training::HVPFitSettings hvp_fit;

    double reg_L2 = 1e-10;

    std::string potfilesave = "potential_fitted";
    bool debug = false;
    std::unique_ptr<ScopedStreamRedirect> debug_guard_;

    double energy_weight(const Atoms &atom) const;
    double force_weight(const Atoms &atom) const;
    double stress_weight(const Atoms &atom) const;
    void evaluate_errors(const std::vector<Atoms> &references,
                         std::vector<Atoms> &predictions,
                         std::vector<Neighborhoods> &nbhs, ErrorMonitor &monitor);

public:
    ErrorMonitor emonitor, emonitor_valid;
    bool monit_valid_err = false;

    TrainerLinearPotential() { InitSettings(); }
    TrainerLinearPotential(const GPSettings &_settings)
    {
        InitSettings();
        ApplySettings(_settings);
    }

    void InitSettings()
    {
        Connect(debug, "debug", "", 1, 10);
        Connect(weight_eng, "energy_weight");
        Connect(weight_force, "force_weight");
        Connect(weight_stress, "stress_weight");
        Connect(weight_eng_power, "energy_pow");
        Connect(weight_force_power, "force_pow");
        Connect(weight_stress_power, "stress_pow");
        Connect(hvp_fit.weight, "hvp_weight", "Weight of fixed-cell Hessian-vector observations");
        Connect(hvp_fit.power, "hvp_pow", "Atom-count power for HVP loss normalization");
        Connect(hvp_fit.step, "hvp_fd_step", "Maximum per-atom displacement for HVP force differences (Angstrom)");
        Connect(reg_L2, "weight_L2");
        Connect(potfilesave, "save_to");
    }

    /// Distribute training set, allocate buffers
    void init(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs,
              Potentials::AnyPotential *pot);
    void init_valid(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs);

    /// Assemble the design-matrix products A^T A and A^T y
    void Assemble();

    /// Solve the regularised normal equations (A^T A + lambda I) w = A^T y
    void Solve();

    /// Assemble + Solve + optionally validate
    void Train();

    void save_current(const std::string &fnm);

    int get_feature_count() const { return n_features_; }
};
