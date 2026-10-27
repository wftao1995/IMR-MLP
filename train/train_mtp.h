// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/***
 * Training a mtp potential
 *
 * SGD with momentum (heavy-ball method).
 *
 * When basis is large, and have many data, one can choose energy-only scaling method to do rescaling.
 *
 * TODO: adaptive regulator in linear training
 * ***/

#include <unsupported/Eigen/CXX11/Tensor>
#ifndef IMRMLP_TRAIN_MTP
#define IMRMLP_TRAIN_MTP

#include "potentials/potential_mtp.h"
#include "libalglib/optimization.h"
#include "utils/gputils.h"
#include "libalglib/stdafx.h"
#include "libalglib/interpolation.h"
#include "utils/gputils.h"
#include <random>
#include "hvp_fit.h"
#include "deplibs/LBFGSpp/include/LBFGS.h"
#include "train/sgd_scheduler.h"

// only to switch off vscode error
#include <cstddef>
using size_t = std::size_t;

#include "error_monitor.h"

class Trainer_MTP : public ConnectSettings
{
private:
    // std::vector<double> mask; // set zero to constraint some coeffs

    int n_size_mat = 0;
    int n_radial_coeffs = 0;
    int n_relax_coeffs = 0;
    struct LayoutSnapshot {
        const Potentials::PotMTP *potential=nullptr;
        int coefficients=0, species=0;
        bool current() const {
            return potential && potential->get_coeff_size()==coefficients && potential->species_count==species;
        }
    };
    LayoutSnapshot training_layout_, ladder_output_layout_;
    void CaptureParameterLayout();
    void RequireCurrentParameterLayout(bool allow_ladder_output=false) const;


    alglib::real_2d_array lin_matrix, lin_matrix_F, lin_matrix_S; // A^T * A
    alglib::real_2d_array lin_matrix_mpi, lin_matrix_F_mpi, lin_matrix_S_mpi;
    alglib::real_1d_array lin_vector, lin_vector_F, lin_vector_S; // y
    alglib::real_1d_array lin_vector_mpi, lin_vector_F_mpi, lin_vector_S_mpi;

    // for fast linear fit, memory usage is high
    // use Eigen's LDLT solver
    int n_lfit_data = 0; // for each core
    int n_lfit_data_all = 0;
    int n_lfit_data_E = 0, n_lfit_data_F = 0, n_lfit_data_S = 0;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> eigen_lin_matrix_A, eigen_lin_matrix, eigen_lin_matrix_mpi;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> eigen_lin_matrix_F, eigen_lin_matrix_S;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> eigen_lin_matrix_F_mpi, eigen_lin_matrix_S_mpi;
    Eigen::VectorXd eigen_lin_vector_A, eigen_lin_vector, eigen_lin_vector_mpi, weight_list;
    Eigen::VectorXd eigen_lin_vector_F, eigen_lin_vector_S;
    Eigen::VectorXd eigen_lin_vector_F_mpi, eigen_lin_vector_S_mpi;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> eigen_lin_matrix_A_F, eigen_lin_matrix_A_S;
    Eigen::VectorXd eigen_lin_vector_A_F, eigen_lin_vector_A_S;

    alglib::real_1d_array loss_gard;
    alglib::real_1d_array optimization_coeffs;
    alglib::real_1d_array update_vector;
    alglib::real_1d_array update_vector_last;
    alglib::real_1d_array scale;
    Eigen::VectorXd x_in, g_in;
    Eigen::VectorXd scaling_coeffs; // scaling of problem, scale each parameter to about magnitute of 1.0

    double loss;
    double loss_mpi;
    alglib::real_1d_array loss_gard_mpi;
    Eigen::VectorXd linear_coeff_buffer;
    std::vector<double> linear_reduce_send, linear_reduce_receive;

    double energy_weight(const Atoms &atom) const;
    double force_weight(const Atoms &atom) const;
    double stress_weight(const Atoms &atom) const;

    // Control flags
    bool safe_linear_fitting = true;
    std::string linear_precondition = "none";
    std::string rescale_mode = "search";
    int maxiter = 1000;
    int pre_init_iter = 75;
    bool skip_pre_init = false;
    std::string potfilesave = "potential.mtp";
    bool random_init = false;
    bool force_linear_randominit = false;
    bool force_radial_randominit = false;

    // Common variables
    size_t train_set_size;     // on current node
    size_t train_set_size_aligned;   // upper bound
    size_t train_set_size_all; // all nodes

    size_t valid_set_size = 0;
    size_t valid_set_size_all = 0;

    // Weight flags
    bool reg_init = false;
    double reg_L2 = 1e-10; // L2, @todo: rename to reg_L2
    double reg_L1 = 0; // L1
    Training::HVPFitSettings hvp_fit;
    double weight_hessian = 0;
    double weight_hessian_power = 0.0;
    double hessian_fd_step = 1e-4;
    void AddHessianLoss(const Atoms &, double, double &, bool, ErrorMonitor &);
    void AddHVPLoss(const Atoms &, double, double &, bool, ErrorMonitor &);
    double weight_eng = 1;           // energy weight
    double weight_force = 0.01;      // force weight
    double weight_stress = 0.0001;   // stress weight
    double weight_eng_power = 1.0;   // weight = weight_eng / (atom.size())^weight_eng_power
    double weight_force_power = 0.0; // weight = weight_force / (atom.size())^weight_force_power
    double weight_reg_w0 = 1e-8;
    double weight_reg_w1 = 0;
    double weight_reg_w2 = 0;
    int quad_count = 100;
    double randomlow = -0.1;
    double randomupw = 0.1;
    double weight_stress_power = 1.0; // weight = weight_stress / (atom.size())^weight_stress_power
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
    bool bfgs_rescale_final = false;
    LBFGSpp::LBFGSParam<double> lbfgs_param;

    void allocate();

    // flag
    bool sgd_pretrain = false;

    // buffers
    std::vector<double> energy_cmpts;
    Eigen::Tensor<double, 3> force_cmpts;
    Eigen::Tensor<double, 3> stress_cmpts;
    Eigen::Tensor<double, 3, Eigen::RowMajor> force_cmpts_R;
    Eigen::Tensor<double, 3, Eigen::RowMajor> stress_cmpts_R;
    Eigen::Matrix<double, -1, 3> dLdF;
    Eigen::Matrix3d dLdS;
    double ploss = 0;
    double ploss_mpi;
    alglib::real_1d_array pgard;
    alglib::real_1d_array pgard_mpi;

    bool bfgs_set_scale = false;

    double bfgs_f;
    Eigen::VectorXd bfgs_g;
    double linstop = 1e-8;
    bool do_orthgonalization = true;
    int bfgs_last_linear_fit_iter = -1;
    void RefitBFGS(const char *reason);

public:
    ErrorMonitor emonitor, emonitor_valid;
    bool fix_radial_coeffs = false;
    bool fix_linear_coeffs = false;
    bool fix_species_coeffs = false;
    bool monit_valid_err = false;
    int radial_reg_type = 0;
    std::vector<Atoms> training_set_, valid_set_;
    std::vector<Atoms> efs_set_, efs_valid_set_;
    std::vector<Neighborhoods> trainint_set_nbhs_, valid_set_nbhs_;
    std::vector<int> train_set_inds_aligned;
    double loss_f_prev;

    inline double get_ploss() {return ploss ;};

    // ladder fitting
    bool do_ladder = false;
    int ladder_cutoff = 10;
    int ladder_last   = 0;
    int n_scalars_save;
    Potentials::PotMTP *p_ladder = nullptr;

    bool apply_disp = false;
    double disp_sigma = 1e-6;

    // mtp potential
    Potentials::PotMTP *p_mtp = nullptr;
    int bfgs_iter = 0;
    bool linear_fit_eng_only = false;
    bool linear_fit_during_bfgs = false;
    alglib::minlbfgsstate bfgsstate;
    alglib::mincgstate cgstate;
    alglib::mindfstate dfstate;
    int df_popsize = 20;
    int df_algo = 0; // 0 is auto, 1 is fix
    int df_strategy = 2;
    bool df_orthogonal_constraints = false;
    double df_cprob = 0.4;
    double df_dweight = 1.0;
    int save_every_it = 10;

    // use matrixA
    bool fast_lfit = true;

    bool debug = false;
    std::unique_ptr<ScopedStreamRedirect> debug_guard_;
    std::unique_ptr<Training::SGDScheduler> sgd_scheduler;

    Trainer_MTP();
    Trainer_MTP(const GPSettings &_settings)
    {
        InitSettings();
        ApplySettings(_settings);
    }
    ~Trainer_MTP();

    void ReBalance();
    void SymLinearMatrix();
    void SymLinearMatrixSep();
    void AddLossGard(int ind, double &_loss);
    void AddLoss(int ind, double &_loss);
    void InitPenaltyGard();
    void AddPenaltyGard();

    void InitSettings();

    // Common, fast version use eigen fully, linear fit use LDLT
    int position_ladd = 0;
    int position_ladd_E = 0;
    int position_ladd_F = 0;
    int position_ladd_S = 0;
    void AddLinear(int ind);
    void AddLinearFast(int ind);
    void AddLinearE(int ind);
    void AddLinearEFast(int ind);
    void AddLinearSep(int ind);
    void AddLinearSepFast(int ind);
    void TrainLinear();
    void TrainLinearAlg();
    void TrainLinearFast();
    void Rescale();
    // accept all, distributed to mpi cores
    void init(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs, Potentials::PotMTP *pot);
    void init_valid(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs);

    // Randon training, for searching radial coeffs
    void TrainRandom();

    // SGD Trainer
    void TrainSGD();
    void SGDUpdate(int local_batch_size); // SGD with Momentum

    // L-BFGS, alglib as backend
    void TrainNonLinearBFGS(int it_limit);
    void TrainBFGS();
    void ValidateSpeciesOnlySettings() const;
    void TrainSpeciesOnly();

    // MINDF, alglib as backend, only optimize radial coeffs, linear coeffs are fitted by TrainLinear()
    void TrainNonLinearDF(int it_limit);
    void TrainDF();
    int DFConstraintCount() const;
    void DFConstraintBounds(alglib::real_1d_array &bounds) const;
    void EvaluateDFConstraints(const alglib::real_1d_array &x,
                               alglib::real_1d_array &values) const;
    void FitDFLinear();

    void AccumulateGard(double &f, alglib::real_1d_array &gard);
    void AccumulateLoss(double &f);
    void AccumulateTestErr(double &f);


    // LBFGSpp with LO linesearch, or others
    void TrainNonLinearLBFGS(int it_limit);
    void TrainLBFGS();
    void TrainRadialLBFGSThenLinear(int it_limit);
    double operator()(const Eigen::VectorXd &x_in, Eigen::VectorXd &g_in);

    // CG
    void TrainNonLinearCG(int it_limit);
    void TrainCG();

    alglib::real_1d_array hess_diag;
    bool precond = false;
    void diag_hess(alglib::real_1d_array &x);

    void save_current(std::string fnm);


    // compute curvture of a single compoment, give val, der and estimated curvture
    void test_curv(int ind, double nudge);
    // check symoblic results and numerically results
    void check_grad(int ind, double nudge);
    // get matrix A^T*A
    void get_ATA();
    void get_A();

    int get_relaxed_coeffs_count(void) const {
        return n_relax_coeffs;
    }

    void ladder_fin();
    void ladder_up();

    void display_iteration_log(double &f);

    inline int nconf() const { return training_set_.size(); };
};


#endif
