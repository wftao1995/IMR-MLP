// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/***
 * Training a non-linear potential
 * ***/

#pragma once

#include "runtime/mtp/gpu-direct/gpu_backend.h"
#include "error_monitor.h"
#include "potentials/potential.h"
#include "libalglib/optimization.h"
#include "utils/gputils.h"
#ifdef IMRMLP_GPU
#include "potentials/mtp_gpu_adapter.h"
#endif
#include "train/sgd_scheduler.h"
#include "train/efs_fit_weights.h"
#include "train/efs_objective.h"
#include "train/linear_refit.h"
#include "train/nonlinear_workflow.h"
#include "libalglib/stdafx.h"
#include "libalglib/interpolation.h"
#include "utils/gputils.h"
#include <random>
#include <limits>

class ChargeErrorMonitor
{
public:
    double sq_error = 0.0;
    double loss_q = 0.0;
    double q_min = std::numeric_limits<double>::infinity();
    double q_max = -std::numeric_limits<double>::infinity();
    double total_min = std::numeric_limits<double>::infinity();
    double total_max = -std::numeric_limits<double>::infinity();
    int nconfig = 0;

    double qrms = 0.0;
    double report_q_min = 0.0;
    double report_q_max = 0.0;
    double report_total_min = 0.0;
    double report_total_max = 0.0;

    void init();
    void add(const Atoms &atom, double predicted, double target, double loss);
    void report(size_t training_size_all);
};

class TrainNonlinear : public ConnectSettings
{
private:
    LinearRefitSettings linear_refit_settings;
    std::string varpro_report = "varpro.jsonl";
    std::string varpro_radial_scale_file;
    std::string validation_metric = "force";
    double varpro_linear_gradient_tolerance = 1e-7;
    int varpro_linear_refinement_steps = 2;
    double validation_max_energy = -1., validation_max_force = -1., validation_max_stress = -1.;
    EFSFitWeights efs_weights() const {
        return {weight_eng,weight_force,weight_stress,
                weight_eng_power,weight_force_power,weight_stress_power,
                weight_force_norm_d,weight_force_norm_f,weight_force_norm_p};
    }
    // internal variables
    int n_relax_coeffs = 0;
    void RequireCurrentParameterLayout() const {
        if (!p_pot || p_pot->get_NL_coeffs_count()!=n_relax_coeffs)
            GPERROR("Potential parameter layout changed; reinitialize the trainer after appending species");
    }

    Eigen::VectorXd loss_gard; // local one
    Eigen::VectorXd update_vector;
    Eigen::VectorXd update_vector_last;
    alglib::real_1d_array scale;
    Eigen::VectorXd x; //, g use Eigen::Map to threat this as vector
    Eigen::VectorXd x_in, g_in;    
    Eigen::VectorXd scaling_coeffs; // scaling of problem, scale each parameter to about magnitute of 1.0

    double loss;
    double loss_mpi;
    Eigen::VectorXd loss_gard_mpi;

    double energy_weight(const Atoms &atom) const;
    double force_weight(const Atoms &atom) const;
    double stress_weight(const Atoms &atom) const;
    double charge_weight(const Atoms &atom) const;
    double total_charge_target(const Atoms &atom) const;
    void AddChargeLoss(Atoms &prediction, const Atoms &reference,
                       AnyNeighborhoods &nbhs, double &loss,
                       double *gradient, ChargeErrorMonitor &monitor);

    // Control flags
    int maxiter = 1000;
    std::string potfilesave = "potential_fitted";
    bool random_init = false;
    bool force_random_init = false;
    std::string training_device = "cpu";
    int gpu_device_id = -1;
    int gpu_batch_atoms = 0;
    int gpu_batch_structures = 0;
    double gpu_memory_fraction = 0.75;
    bool gpu_device_loss = true;
    bool gpu_deterministic_reduction = false;
    int gpu_radial_parameter_tile = 128;
    bool gpu_cache_batches = true;
    int validation_interval = 1;
    bool defer_validation_ = false;
    bool collective_evaluation_errors_ = false;
    bool cg_active_ = false;
    Potentials::AnyPotential *cg_result_potential_ = nullptr;
    std::vector<double> cg_result_coefficients_;
    void InvalidateCGResult();
    void ValidateCGSettingsImpl(int limit, bool terminal_refit = false) const;
    void ValidateLinearRefitSettings(bool respect_mask = false) const;
    void TrainLinearRefitImpl(bool validate_final);
    void ValidateCGData() const;
    void ValidateCGPrediction(const Atoms &, const Atoms &, bool validation) const;
    void SaveCGPoint(const std::string &path);
    bool validation_current_ = false;

    std::vector<double> validation_coefficients_;
    void EvaluateValidation();
    void DisplayValidation() const;

    // Common variables
    size_t train_set_size;     // on current node
    size_t train_set_size_aligned;   // upper bound
    size_t train_set_size_all; // all nodes

    size_t valid_set_size = 0;
    size_t valid_set_size_all = 0;

    // Weight flags
    bool reg_init = false;
    double reg_L2 = 1e-10; // L2, @todo: rename to reg_L2
    std::string objective_convention = "legacy", l2_reference = "native", objective_report;
    Training::EFSObjective objective_contract_;
    double reg_L1 = 0; // L1
    Training::HVPFitSettings hessian_fit; // Full Frobenius matrix loss through force finite differences.
    void AddHessianLoss(const Atoms &, double, double &, bool, ErrorMonitor &);
    Training::HVPFitSettings hvp_fit;
    void AddHVPLoss(const Atoms &, double, double &, bool, ErrorMonitor &);
    double weight_eng = 1;           // energy weight
    double weight_force = 0.01;      // force weight
    double weight_stress = 0.0001;   // stress weight
    double weight_charge = 0.0;      // per-configuration total-charge weight
    double weight_eng_power = 1.0;   // weight = weight_eng / (atom.size())^weight_eng_power
    double weight_force_power = 0.0; // weight = weight_force / (atom.size())^weight_force_power
    double weight_reg_w0 = 1e-8;
    double weight_reg_w1 = 0;
    double weight_reg_w2 = 0;
    double randomlow = -0.1;
    double randomupw = 0.1;
    double weight_stress_power = 1.0; // weight = weight_stress / (atom.size())^weight_stress_power
    double weight_charge_power = 1.0;
    std::string charge_target_attrib = "total_charge";
    double charge_default_target = 0.0;
    double weight_penatly = 1.0e-6;   // make basis orthogonal
    double weight_force_norm_d = 0.1; // to target, wgt *= weight_force_norm_d / (weight_force_norm_d + weight_force_norm_f^weight_force_norm_p)
    double weight_force_norm_f = 0;
    double weight_force_norm_p = 1.0;

    // SGD related
    double learning_rate = 0.0001;
    double momentum = 0.1;
    int batch_size = 100;
    int maxepoches = 100;
    int batches_per_epoch;
    int batch_size_core;
    // SGD learning rate scheduler
    std::string sgd_scheduler_type = "none";
    double sgd_decay_rate = 0.95;
    int sgd_decay_steps = 10;
    double sgd_final_lr = 0.0;
    double clip_gradient = 0.0;       // 0 = no clipping
    int sgd_warmup_steps = 0;         // 0 = no warmup
    double sgd_warmup_lr = 0.0;       // starting LR for warmup

    // BFGS|CG related
    double bfgs_conv_tol = 1e-3;
    double bfgs_grad_tol = 1e-3;
    double bfgs_step_tol = 1e-3;
    int bfgs_m = 5;

    // flag
    bool sgd_pretrain = false;

    // buffers
    Eigen::Matrix<double, -1, 3> dLdF;
    Eigen::Matrix3d dLdS;
    double ploss = 0;
    double ploss_mpi;
    Eigen::VectorXd pgard;
    Eigen::VectorXd pgard_mpi;

    bool bfgs_set_scale = false;
    bool schedule_active_ = false;
    std::string training_schedule = "single";
    std::vector<std::string> trainable_groups;
    int sr_stage_iterations = 100;
    int les_stage_iterations = 50;
    int joint_stage_iterations = 0;
    double charge_warmup_factor = 10.0;
    double les_zero_init_range = 1e-3;
    double sog_zero_init_range = 0.1;
    int les_zero_init_seed = 5489;
    std::string curvature_precondition = "none";
    int gn_probes = 8, gn_update = 20, gn_seed = 5489;
    double gn_floor = 1e-12;
    void ValidateCurvatureSettings() const;
    void PrepareParameterScale(const double *coeffs);
    Eigen::VectorXd L2CurvatureDiagonal();
    std::string parameter_scaling = "none";
    double scale_floor = 1e-3;
    double scale_ceiling = 0.1;
    double scaled_step_max = 0.0;
    std::vector<char> trainable_mask;
    Eigen::VectorXd frozen_coeffs;

#ifdef IMRMLP_GPU
    std::unique_ptr<Potentials::MTPGpuEvaluator> gpu_evaluator_;
    std::unique_ptr<Potentials::MTPGpuBatch> gpu_batch_;
    struct GpuCachedBatch {
        std::vector<int> indices;
        std::unique_ptr<Potentials::MTPGpuBatch> batch;
        std::vector<double> target_energy;
        std::vector<double> energy_weight;
        std::vector<double> target_force;
        std::vector<double> force_weight;
        std::vector<double> target_stress;
        std::vector<double> stress_weight;
        int energy_count = 0;
        int force_count = 0;
        int stress_count = 0;
    };
    std::vector<GpuCachedBatch> gpu_cached_batches_;
    std::vector<GpuCachedBatch> gpu_validation_batches_;
    std::vector<double> gpu_dloss_denergy_;
    std::vector<double> gpu_dloss_dforce_;
    std::vector<double> gpu_dloss_dstress_;
    std::vector<double> gpu_target_energy_;
    std::vector<double> gpu_energy_weight_;
    std::vector<double> gpu_target_force_;
    std::vector<double> gpu_force_weight_;
    std::vector<double> gpu_target_stress_;
    std::vector<double> gpu_stress_weight_;
    size_t gpu_free_memory_bytes_ = 0;
    std::vector<size_t> gpu_reciprocal_bytes_;
    size_t gpu_memory_budget_bytes_ = 0;
    size_t EstimateGpuBytes(size_t atoms, size_t neighbors,
                             size_t structures, size_t reciprocal_bytes = 0) const;
    size_t EstimateGpuChunkBytes(const std::vector<int> &indices) const;
    std::vector<std::vector<int>> MakeGpuChunks(const std::vector<int> &indices) const;
    std::vector<std::vector<int>> MakeGpuChunks(
        const std::vector<int> &indices, const std::vector<Atoms> &atoms,
        const std::vector<Neighborhoods> &neighbors,
        const std::vector<size_t> &reciprocal_bytes) const;
    void EvaluateGpuValidation();
    void PrepareGpuDeviceLabels(
        const std::vector<int> &indices,
        std::vector<double> &target_energy,
        std::vector<double> &energy_weights,
        std::vector<double> &target_force,
        std::vector<double> &force_weights,
        std::vector<double> &target_stress,
        std::vector<double> &stress_weights,
        int &energy_count, int &force_count, int &stress_count) const;
    void AccumulateGpuDeviceChunk(Potentials::MTPGpuBatch &batch,
                                   const std::vector<int> &indices,
                                   const GpuCachedBatch *cached,
                                   bool reset_gradient,
                                   bool finalize_gradient,
                                   double &batch_loss);
    void AccumulateGpuBatch(const std::vector<int> &indices, double &batch_loss);
#endif

    double bfgs_f;
    Eigen::VectorXd bfgs_g;
    double linstop = 1e-8;
    bool do_orthgonalization = true;

public:
    ErrorMonitor emonitor, emonitor_valid;
    double hvp_fd_step() const { return hvp_fit.step; }
    double hessian_fd_step() const { return hessian_fit.step; }
    Training::WorkflowOptions workflow_options;
    void ValidateLinearRefitOnly(const Potentials::AnyPotential& potential) const;
    void TrainLinearRefit();
    // Collective native publication, including verified multi-file models.
    void SaveConditionalResult(const std::string &path);
    ChargeErrorMonitor qmonitor, qmonitor_valid;
    bool monit_valid_err = false;
    int radial_reg_type = 0;
    std::vector<Atoms> training_set_, valid_set_;
    std::vector<Atoms> efs_set_, efs_valid_set_;
    std::vector<Neighborhoods> trainint_set_nbhs_, valid_set_nbhs_;
    std::vector<int> train_set_inds_aligned;
    double loss_f_prev = std::numeric_limits<double>::infinity();

    inline double get_ploss() {return ploss ;};

    bool apply_disp = false;
    double disp_sigma = 1e-6;

    // mtp potential
    Potentials::AnyPotential *p_pot = nullptr;
    // ALGLIB completion code; 0 means no completed run. Budget/stagnation
    // stops preserve a finite accepted point but do not imply convergence.
    int bfgs_termination_type = 0;
    bool bfgs_converged() const {
        return bfgs_termination_type == 1 || bfgs_termination_type == 2
            || bfgs_termination_type == 4;
    }
    int bfgs_iter = 0;
    int bfgs_accepted_iter = 0;
    bool bfgs_initial_reported = false;
    size_t validation_evaluations = 0;
    void ValidateAcceptedPoint(const double *coeffs, int iteration, bool force = false);
    // Adds a cumulative label to accepted-iteration logs and checkpoint
    // filenames when a coefficient checkpoint is used to start a new run.
    // ALGLIB's internal L-BFGS history is deliberately not serialized.
    int bfgs_iteration_offset = 0;
    alglib::minlbfgsstate bfgsstate;
    alglib::mincgstate cgstate;
    alglib::mindfstate dfstate;
    int df_popsize = 20;
    int df_algo = 0; // 0 is auto, 1 is fix
    int df_strategy = 2;
    double df_cprob = 0.4;
    double df_dweight = 1.0;
    int save_every_it = 10;
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> fweights;

    // use matrixA
    bool fast_lfit = true;

    bool debug = false;
    std::unique_ptr<ScopedStreamRedirect> debug_guard_;
    std::unique_ptr<Training::SGDScheduler> sgd_scheduler;

    TrainNonlinear();
    TrainNonlinear(const GPSettings &_settings)
    {
        InitSettings();
        ApplySettings(_settings);
    }
    ~TrainNonlinear();

    void ReBalance();
    void AddLossGard(int ind, double &_loss);
    void AddLoss(int ind, double &_loss);
    void InitPenaltyGard();

    void InitSettings();
    void ApplySettings(const GPSettings &settings, const std::set<std::string> &skip = {});
    // accept all, distributed to mpi cores
    void init(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs, Potentials::AnyPotential *pot);
    void init_valid(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs);

    // SGD Trainer
    void TrainSGD();
    void TrainScheduled(const std::string &optimizer);
    void TrainSingleStage(const std::string &optimizer, int iterations);
    void TrainNonLinearSGD(int epochs);
    void SGDUpdate(int local_batch_size); // SGD with Momentum

    // L-BFGS, alglib as backend
    void TrainNonLinearBFGS(int it_limit);
    void TrainBFGS();
    void TrainVarPro();
    void ConfigureTrainableGroups(const std::vector<std::string> &groups);
    void ApplyTrainableMask(alglib::real_1d_array &gradient) const;
    Eigen::Index WeightedResidualCount() const;
    void AccumulateResidualVJP(const Eigen::VectorXd &cotangents, Eigen::VectorXd &gradient);
    Eigen::VectorXd EstimateGNDiagonal();
    Eigen::VectorXd gn_diagonal;
    int gn_refreshes = 0;
    void SetOptimizerCoeffs(const double *coeffs);

    void AccumulateGard(double &f, Eigen::VectorXd &gard);
    void AccumulateGard(double &f, alglib::real_1d_array &gard);
    void AccumulateLoss(double &f);
    void AccumulateTestErr(double &f);


    // for external optimizer
    double operator()(const Eigen::VectorXd &x_in, Eigen::VectorXd &g_in);

    // CG
    void TrainNonLinearCG(int it_limit);
    void TrainCG();
    void ValidateCGSettings() const;
    // Collective single-file publication of a completed CG result.
    void SaveCGResult(const std::string &path);
    int cg_termination_type = 0;
    bool cg_converged() const {
        return cg_termination_type == 1 || cg_termination_type == 2 || cg_termination_type == 4;
    }

    void save_current(std::string fnm);
    void save_accepted_checkpoint(const alglib::real_1d_array &coeffs);

    // compute curvture of a single compoment, give val, der and estimated curvture
    void test_curv(int ind, double nudge);
    // check symoblic results and numerically results
    void check_grad(int ind, double nudge);

    int get_relaxed_coeffs_count(void) const {
        return n_relax_coeffs;
    }

    void display_iteration_log(double &f);

    inline int nconf() const { return training_set_.size(); };
};
