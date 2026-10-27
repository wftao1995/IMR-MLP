// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "train_mtp.h"
#include "linear_efs_system.h"
#include "linear_equilibration.h"
#include "mtp_linear_reduce.h"
#include "collective_error.h"
#include "train_nonlinear_potential.h"
#include "tools/tools.h"
#include "libalglib/solvers.h"
#include "utils/gputils.h"
#include "utils/gputils_eigen.h"
#include "libalglib/alglibmisc.h"

#include <exception>

namespace {
// A root-only solver error must reach all ranks before the coefficient Bcast.
void check_linear_solve_error(std::string error) {
#ifdef IMRMLP_MPI
    int length = error.size();
    MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
    error.resize(length);
    if (length) MPI_Bcast(error.data(), length, MPI_CHAR, 0, mpi.comm);
#endif
    if (!error.empty()) GPERROR(error);
}

bool bfgs_linear_fit_step(int iteration) {
    return iteration == 25 || iteration == 100 || iteration == 200;
}
}

void Trainer_MTP::InitSettings() 
{
    Connect(debug, "debug", "", 1, 10);
    // ladder
    Connect(do_ladder, "do_ladder");
    Connect(ladder_cutoff, "ladder_cut");
    Connect(ladder_last, "ladder_last");
    Connect(apply_disp, "apply_disp", "", 1, 10);
    Connect(disp_sigma, "disp_sigma", "", 1, 10);
    // General control
    // regulation weight L2
    Connect(reg_L2, "weight_reg"); // todo: discard this
    Connect(reg_L1, "weight_L1");
    Connect(reg_L2, "weight_L2");
    Connect(weight_reg_w0, "weight_rad_w0");
    Connect(weight_reg_w1, "weight_rad_w1");
    Connect(weight_reg_w2, "weight_rad_w2");
    Connect(radial_reg_type, "radial_reg_type");
    Connect(quad_count, "quad_count");
    Connect(randomlow, "rand_low");
    Connect(randomupw, "rand_up");
    Connect(safe_linear_fitting, "safe_fit");
    Connect(linear_precondition, "linear_precondition", "Linear solve coordinates: none or diagonal");
    Connect(rescale_mode, "rescale_mode", "Rescale calls: search or fixed (one linear fit at the current radial scaling)");
    Connect(potfilesave, "save_to");
    Connect(random_init, "random_init");
    Connect(force_linear_randominit, "force_randomlinear");
    Connect(force_radial_randominit, "force_randomradial");
    Connect(skip_pre_init, "skip_preinit");
    Connect(pre_init_iter, "pre_init_iter");
    Connect(linear_fit_eng_only, "lfit_eng_only");
    Connect(fast_lfit, "lfit_fastalgo");
    Connect(save_every_it, "save_every");
    // Weights
    Connect(hvp_fit.weight, "hvp_weight");
    Connect(hvp_fit.power, "hvp_pow");
    Connect(hvp_fit.step, "hvp_fd_step");
    Connect(weight_eng, "energy_weight");
    Connect(weight_force, "force_weight");
    Connect(weight_stress, "stress_weight");
    Connect(weight_hessian, "hessian_weight");
    Connect(weight_hessian_power, "hessian_pow");
    Connect(hessian_fd_step, "hessian_fd_step");
    Connect(weight_eng_power, "energy_pow");
    Connect(weight_force_power, "force_pow");
    Connect(weight_stress_power, "stress_pow");
    // weight to keep orthgonal
    Connect(weight_penatly, "penalty_weight");
    Connect(weight_force_norm_d, "force_wgt_d");
    Connect(weight_force_norm_f, "force_wgt_f");
    Connect(weight_force_norm_p, "force_wgt_p");
    // SGD
    Connect(learning_rate, "learning_rate");
    Connect(momentum, "momentum");
    Connect(batch_size, "batch_size");
    Connect(maxepoches, "epoch_limit");
    Connect(sgd_scheduler_type, "lr_scheduler");
    Connect(sgd_decay_rate, "lr_decay_rate");
    Connect(sgd_decay_steps, "lr_decay_steps");
    Connect(sgd_final_lr, "lr_final");
    Connect(clip_gradient, "clip_gradient");
    Connect(sgd_warmup_steps, "lr_warmup_steps");
    Connect(sgd_warmup_lr, "lr_warmup_init");
    // L-BFGS
    Connect(linear_fit_during_bfgs, "linear_fit_bfgs");
    Connect(bfgs_m, "bfgs_m");
    Connect(bfgs_conv_tol, "bfgs_conv_tol");
    Connect(bfgs_grad_tol, "bfgs_grad_tol");
    Connect(bfgs_step_tol, "bfgs_step_tol");
    Connect(maxiter, "bfgs_maxiteration");
    Connect(bfgs_rescale_final, "bfgs_rescale", "", 1, 10);
    Connect(bfgs_set_scale, "set_scale", "", 1, 10);
    Connect(precond, "precondition", "", 1, 10);
    // L-BFGSpp
    Connect(lbfgs_param.epsilon, "lbfgspp:epsilon", "", 1, 10);
    Connect(lbfgs_param.epsgmax, "lbfgspp:epsgmax", "", 1, 10);
    Connect(lbfgs_param.epsilon_rel, "lbfgspp:epsrel", "", 1, 10);
    Connect(lbfgs_param.m, "lbfgspp:m", "", 1, 10);
    Connect(lbfgs_param.damping, "lbfgspp:damping", "", 1, 10);
    Connect(lbfgs_param.initial_step, "lbfgspp:init_step", "", 1, 10);
    Connect(lbfgs_param.first_initial_step, "lbfgspp:first_step", "", 1, 10);
    Connect(lbfgs_param.past, "lbfgspp:past", "", 1, 10);
    Connect(lbfgs_param.delta, "lbfgspp:delta", "", 1, 10);
    Connect(lbfgs_param.max_iterations, "lbfgspp:maxit", "", 1, 10);
    Connect(lbfgs_param.linesearch, "lbfgspp:linesearch", "", 1, 10);
    Connect(lbfgs_param.max_linesearch, "lbfgspp:max_linesearch", "", 1, 10);
    Connect(lbfgs_param.min_step, "lbfgspp:min_step", "", 1, 10);
    Connect(lbfgs_param.max_step, "lbfgspp:max_step", "", 1, 10);
    Connect(lbfgs_param.ftol, "lbfgspp:ftol", "", 1, 10);
    Connect(lbfgs_param.wolfe, "lbfgspp:wolfe", "", 1, 10);
    Connect(lbfgs_param.f_dec_coeff, "lbfgspp:f_dec_coeff", "", 1, 10);
    Connect(lbfgs_param.s_curv_coeff, "lbfgspp:s_curv_coeff", "", 1, 10);
    Connect(lbfgs_param.cautious_factor, "lbfgspp:cautious_factor", "", 1, 10);
    Connect(lbfgs_param.max_step_move, "lbfgspp:max_move", "", 1, 10);
    // fitting control
    Connect(fix_linear_coeffs, "fix_linear_coeffs", "fix linear coeffs", 1, 10);
    Connect(fix_species_coeffs, "fix_species_coeffs", "fix species coeffs", 1, 10);
    Connect(fix_radial_coeffs, "fix_radial_coeffs", "fix radial coeffs", 1, 10);
    Connect(do_orthgonalization, "do_orthogonalization");
    // df
    Connect(df_popsize, "popsize");
    Connect(df_orthogonal_constraints, "df_orthogonal_constraints",
            "Constrain radial channels within each species pair to be orthogonal during DF");
}

void Trainer_MTP::CaptureParameterLayout() {
    training_layout_={p_mtp,p_mtp->get_coeff_size(),p_mtp->species_count};
    ladder_output_layout_=p_ladder ? LayoutSnapshot{p_ladder,p_ladder->get_coeff_size(),p_ladder->species_count} : LayoutSnapshot{};
}

void Trainer_MTP::RequireCurrentParameterLayout(bool allow_ladder_output) const {
    if(!training_layout_.current() || (ladder_output_layout_.potential && !ladder_output_layout_.current())
        || (p_mtp!=training_layout_.potential && !(allow_ladder_output && p_mtp==ladder_output_layout_.potential)))
        GPERROR("Potential parameter layout changed; reinitialize the trainer after appending species");
}

void Trainer_MTP::Rescale() {
    RequireCurrentParameterLayout();
    if (rescale_mode == "fixed") {
        Message("Rescale: fixed radial scaling; one linear fit");
        TrainLinear();
        return;
    }
    if (rescale_mode != "search") GPERROR("Unknown rescale_mode: " + rescale_mode);
    double min_scaling = p_mtp->get_scale();
    double max_scaling = p_mtp->get_scale();
    int ind;
    int soffset = 0;
    if (fix_species_coeffs) soffset = p_mtp->species_count;
    do {
        double condition_number[5];
        double scaling = p_mtp->get_scale();
        double scalings[5] = {scaling / 1.2, scaling / 1.1, scaling, scaling * 1.1, scaling * 1.2}; // TODO: what's this
        std::vector<double> coeffs(n_size_mat); // linear coeffs
        for (int j = 0; j < 5; ++j) {
            Message("Rescaling to " + std::to_string(scalings[j]));
            p_mtp->set_scale(scalings[j]);
            TrainLinear();
            double rms = 0;

            for (int i = 0; i < n_size_mat; ++i) {
                coeffs[i] = std::abs(p_mtp->get_linear_coeff(i + soffset));
                rms += coeffs[i] * coeffs[i];
            }
            rms = std::sqrt(rms);
            std::sort(coeffs.begin(), coeffs.end());
            double median = coeffs[coeffs.size() / 2];
            condition_number[j] = rms / median;
            Message("   Condition number is " + std::to_string(condition_number[j]));
        }
        // finds minimal condition number
        ind = 2;
        for (int j = 0; j < 5; j++)
            if (condition_number[j] < condition_number[ind])
                ind = j;
        p_mtp->set_scale(scalings[ind]);
        Message("Scaling to " + std::to_string(scalings[ind]));
        TrainLinear();
    } while (ind != 2);
}

void Trainer_MTP::SymLinearMatrix() {
    RequireCurrentParameterLayout();
    if (mpi.rank == 0)
        for (int i = 0; i < n_size_mat; ++i) {
            for (int j = i + 1; j < n_size_mat; ++j) {
                lin_matrix_mpi(j, i) = lin_matrix_mpi(i, j);
            }
        }
}

void Trainer_MTP::SymLinearMatrixSep() {
    RequireCurrentParameterLayout();
    if (mpi.rank == 0)
        for (int i = 0; i < n_size_mat; ++i) {
            for (int j = i + 1; j < n_size_mat; ++j) {
                lin_matrix_mpi(j, i) = lin_matrix_mpi(i, j);
                lin_matrix_F_mpi(j, i) = lin_matrix_F_mpi(i, j);
                lin_matrix_S_mpi(j, i) = lin_matrix_S_mpi(i, j);
            }
        }
}

void Trainer_MTP::AddLinear(int ind) {
    RequireCurrentParameterLayout();
    Atoms &atm = training_set_[ind];
    Neighborhoods &nbhs = trainint_set_nbhs_[ind];
    p_mtp->CalcEFSCmpt(atm, nbhs, energy_cmpts, force_cmpts, stress_cmpts);

    double avef = 0;
    if (atm.has_forces() && weight_force_norm_f > 0) {
        for (auto &it: atm.forces_) {
            avef += it.squaredNorm();
        }
        avef /= atm.size();
    }

    // Note
    /**
     * One can change this for weighting
     * **/
    double cc;
    if (weight_force_norm_f > 0)
        cc = weight_force_norm_d / (weight_force_norm_d + std::pow(weight_force_norm_f * avef, weight_force_norm_p));
    else
        cc = 1.0;

    int soffset = 0;
    if (fix_species_coeffs) soffset = p_mtp->species_count;
    // for force and stress cmpts, ders of species is zero
    // force, todo: adjust memory layout for faster access
    if (weight_force > 0 && atm.has_forces()) {
        double weight_force_i = force_weight(atm) * cc;
        for (int iat = 0; iat < atm.size(); ++iat) {
            double weight_force_iat = weight_force_i;
            if (atm.atomic_fitness.size() > 0) {
                weight_force_iat *= atm.atomic_fitness[iat];
            }
            if (weight_force_iat > 0) {
                for (int a = 0; a < 3; ++a) {
                    for (int i = 0; i < n_size_mat; ++i) {
                        for (int j = i; j < n_size_mat; ++j) {
                            lin_matrix(i, j) += weight_force_iat * force_cmpts(iat, i + soffset, a) * force_cmpts(iat, j + soffset, a);
                        }
                        lin_vector[i] += weight_force_iat * force_cmpts(iat, i + soffset, a) * atm.forces_[iat][a];
                    }
                }
            }
        }
    }
    // stress
    if (weight_stress > 0 && atm.has_stresses()) {
        double weight_stress_i = stress_weight(atm) * cc;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int i = 0; i < n_size_mat; ++i) {
                    for (int j = i; j < n_size_mat; ++j) {
                        lin_matrix(i, j) += weight_stress_i * stress_cmpts(i + soffset, a, b) * stress_cmpts(j + soffset, a, b);
                    }
                    lin_vector[i] += weight_stress_i * stress_cmpts(i + soffset, a, b) * atm.stress_(a, b);
                }
    }
    if (hvp_fit.weight > 0 && atm.has_hvp()) {
        const double weight = hvp_fit.observation_weight(atm, cc);
        Eigen::MatrixXd components;
        for (int k = 0; k < atm.hvp_directions_.cols(); ++k) {
            p_mtp->CalcHessianVectorCmpt(atm, nbhs, atm.hvp_directions_.col(k), components);
            const auto rows = components.rightCols(n_size_mat);
            const Eigen::MatrixXd gram = weight * rows.transpose() * rows;
            const Eigen::VectorXd rhs = weight * rows.transpose() * atm.hvp_targets_.col(k);
            for (int i = 0; i < n_size_mat; ++i) {
                lin_vector[i] += rhs[i];
                for (int j = i; j < n_size_mat; ++j) lin_matrix(i, j) += gram(i, j);
            }
        }
    }
    // energy, if fix_species_coeffs, need subtract related terms
    if (weight_eng > 0 && atm.has_energy()) {
        double weight_eng_i = energy_weight(atm) * cc; // gain more weight near equ
        for (int i = 0; i < n_size_mat; ++i) {
            for (int j = i; j < n_size_mat; ++j) {
                lin_matrix(i, j) += weight_eng_i * energy_cmpts[i + soffset] * energy_cmpts[j + soffset];
            }
            if (fix_species_coeffs) {
                double sub = 0;
                for (int s = 0; s < p_mtp->species_count; ++s) {
                    sub += energy_cmpts[s] * p_mtp->species_coeffs[s];
                }
                lin_vector[i] += weight_eng_i * energy_cmpts[i+soffset] * (atm.energy_ - sub);
            } else {
                lin_vector[i] += weight_eng_i * energy_cmpts[i] * atm.energy_;
            }
        }
    }
}

void Trainer_MTP::AddLinearSepFast(int ind) {
    RequireCurrentParameterLayout();
    Atoms &atm = training_set_[ind];
    Neighborhoods &nbhs = trainint_set_nbhs_[ind];
    p_mtp->CalcEFSCmpt_R(atm, nbhs, energy_cmpts, force_cmpts_R, stress_cmpts_R);

    int soffset = 0;
    if (fix_species_coeffs) soffset = p_mtp->species_count;
    // for force and stress cmpts, ders of species is zero
    // force, todo: adjust memory layout for faster access
    if (weight_force > 0 && atm.has_forces()) {
        for (int iat = 0; iat < atm.size(); ++iat) {
            if ((atm.atomic_fitness.size() == 0) || (atm.atomic_fitness.size() > 0 && atm.atomic_fitness[iat] > 0)) {
                for (int a = 0; a < 3; ++a) {
                    double weight_force_iat = weight_list(position_ladd);
                    memcpy(eigen_lin_matrix_A_F.row(position_ladd_F).data(), &force_cmpts_R(iat, a, 0) + soffset, n_size_mat * sizeof(double));
                    eigen_lin_matrix_A_F.row(position_ladd_F) *= weight_force_iat;
                    eigen_lin_vector_A_F(position_ladd_F) = weight_force_iat * atm.forces_[iat][a];
                    ++position_ladd_F;
                    ++position_ladd;
                }
            }
        }
    }
    // stress
    if (weight_stress > 0 && atm.has_stresses()) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double weight_stress_i = weight_list(position_ladd);
                memcpy(eigen_lin_matrix_A_S.row(position_ladd_S).data(), &stress_cmpts_R(a, b, 0) + soffset, n_size_mat * sizeof(double));
                eigen_lin_matrix_A_S.row(position_ladd_S) *= weight_stress_i;
                eigen_lin_vector_A_S[position_ladd_S] = weight_stress_i * atm.stress_(a, b);
                ++position_ladd_S;
                ++position_ladd;
            }
    }
    // energy, if fix_species_coeffs, need subtract related terms
    if (weight_eng > 0 && atm.has_energy()) {
        double weight_eng_i = weight_list(position_ladd); 
        memcpy(eigen_lin_matrix_A.row(position_ladd_E).data(), energy_cmpts.data() + soffset, n_size_mat * sizeof(double));
        eigen_lin_matrix_A.row(position_ladd_E) *= weight_eng_i;
        double sub = 0;
        if (fix_species_coeffs) {
            for (int s = 0; s < p_mtp->species_count; ++s) {
                sub += energy_cmpts[s] * p_mtp->species_coeffs[s];
            }
        }
        eigen_lin_vector_A[position_ladd_E] = weight_eng_i * (atm.energy_ - sub);
        ++position_ladd;
        ++position_ladd_E;
    }
}

void Trainer_MTP::AddLinearFast(int ind) {
    RequireCurrentParameterLayout();
    Atoms &atm = training_set_[ind];
    Neighborhoods &nbhs = trainint_set_nbhs_[ind];
    p_mtp->CalcEFSCmpt_R(atm, nbhs, energy_cmpts, force_cmpts_R, stress_cmpts_R);

    int soffset = 0;
    if (fix_species_coeffs) soffset = p_mtp->species_count;
    // for force and stress cmpts, ders of species is zero
    // force, todo: adjust memory layout for faster access
    if (weight_force > 0 && atm.has_forces()) {
        for (int iat = 0; iat < atm.size(); ++iat) {
            if ((atm.atomic_fitness.size() == 0) || (atm.atomic_fitness.size() > 0 && atm.atomic_fitness[iat] > 0)) {
                for (int a = 0; a < 3; ++a) {
                    double weight_force_iat = weight_list(position_ladd);
                    memcpy(eigen_lin_matrix_A.row(position_ladd).data(), &force_cmpts_R(iat, a, 0) + soffset, n_size_mat * sizeof(double));
                    eigen_lin_matrix_A.row(position_ladd) *= weight_force_iat;
                    eigen_lin_vector_A(position_ladd) = weight_force_iat * atm.forces_[iat][a];
                    ++position_ladd;
                }
            }
        }
    }
    // stress
    if (weight_stress > 0 && atm.has_stresses()) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double weight_stress_i = weight_list(position_ladd);
                memcpy(eigen_lin_matrix_A.row(position_ladd).data(), &stress_cmpts_R(a, b, 0) + soffset, n_size_mat * sizeof(double));
                eigen_lin_matrix_A.row(position_ladd) *= weight_stress_i;
                eigen_lin_vector_A[position_ladd] = weight_stress_i * atm.stress_(a, b);
                ++position_ladd;
            }
    }
    if (hvp_fit.weight > 0 && atm.has_hvp()) {
        Eigen::MatrixXd components;
        for (int k = 0; k < atm.hvp_directions_.cols(); ++k) {
            p_mtp->CalcHessianVectorCmpt(atm, nbhs, atm.hvp_directions_.col(k), components);
            for (int row = 0; row < components.rows(); ++row) {
                const double weight = weight_list(position_ladd);
                eigen_lin_matrix_A.row(position_ladd) = weight * components.row(row).tail(n_size_mat);
                eigen_lin_vector_A[position_ladd] = weight * atm.hvp_targets_(row, k);
                ++position_ladd;
            }
        }
    }
    // energy, if fix_species_coeffs, need subtract related terms
    if (weight_eng > 0 && atm.has_energy()) {
        double weight_eng_i = weight_list(position_ladd); 
        memcpy(eigen_lin_matrix_A.row(position_ladd).data(), energy_cmpts.data() + soffset, n_size_mat * sizeof(double));
        eigen_lin_matrix_A.row(position_ladd) *= weight_eng_i;
        double sub = 0;
        if (fix_species_coeffs) {
            for (int s = 0; s < p_mtp->species_count; ++s) {
                sub += energy_cmpts[s] * p_mtp->species_coeffs[s];
            }
        }
        eigen_lin_vector_A[position_ladd] = weight_eng_i * (atm.energy_ - sub);
        ++position_ladd;
    }
}

void Trainer_MTP::AddLinearSep(int ind) {
    RequireCurrentParameterLayout();
    Atoms &atm = training_set_[ind];
    Neighborhoods &nbhs = trainint_set_nbhs_[ind];
    p_mtp->CalcEFSCmpt(atm, nbhs, energy_cmpts, force_cmpts, stress_cmpts);

    double avef = 0;
    if (atm.has_forces() && weight_force_norm_f > 0) {
        for (auto &it: atm.forces_) {
            avef += it.squaredNorm();
        }
        avef /= atm.size();
    }
    double cc;
    if (weight_force_norm_f > 0)
        cc = weight_force_norm_d / (weight_force_norm_d + std::pow(weight_force_norm_f * avef, weight_force_norm_p));
    else
        cc = 1.0;
    int soffset = 0;
    if (fix_species_coeffs) soffset = p_mtp->species_count;

    // force
    if (weight_force > 0 && atm.has_forces()) {
        double weight_force_i = force_weight(atm) * cc;
        for (int iat = 0; iat < atm.size(); ++iat) {
            double weight_force_iat = weight_force_i;
            if (atm.atomic_fitness.size() > 0) {
                weight_force_iat *= atm.atomic_fitness[iat];
            }
            if (weight_force_iat > 0) {
                for (int a = 0; a < 3; ++a) {
                    for (int i = 0; i < n_size_mat; ++i) {
                        for (int j = i; j < n_size_mat; ++j) {
                            lin_matrix_F(i, j) += weight_force_iat * force_cmpts(iat, i + soffset, a) * force_cmpts(iat, j + soffset, a);
                        }
                        lin_vector_F[i] += weight_force_iat * force_cmpts(iat, i + soffset, a) * atm.forces_[iat][a];
                    }
                }
            }
        }
    }
    // stress
    if (weight_stress > 0 && atm.has_stresses()) {
        double weight_stress_i = stress_weight(atm) * cc;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int i = 0; i < n_size_mat; ++i) {
                    for (int j = i; j < n_size_mat; ++j) {
                        lin_matrix_S(i, j) += weight_stress_i * stress_cmpts(i + soffset, a, b) * stress_cmpts(j + soffset, a, b);
                    }
                    lin_vector_S[i] += weight_stress_i * stress_cmpts(i + soffset, a, b) * atm.stress_(a, b);
                }
    }
    // energy
    if (weight_eng > 0 && atm.has_energy()) {
        double weight_eng_i = energy_weight(atm); // gain more weight near equ
        for (int i = 0; i < n_size_mat; ++i) {
            for (int j = i; j < n_size_mat; ++j) {
                lin_matrix(i, j) += weight_eng_i * energy_cmpts[i + soffset] * energy_cmpts[j + soffset];
            }
            if (fix_species_coeffs) {
                double sub = 0;
                for (int s = 0; s < p_mtp->species_count; ++s) {
                    sub += energy_cmpts[s] * p_mtp->species_coeffs[s];
                }
                lin_vector[i] += weight_eng_i * energy_cmpts[i+soffset] * (atm.energy_ - sub);
            } else {
                lin_vector[i] += weight_eng_i * energy_cmpts[i] * atm.energy_;
            }
        }
    }
}

void Trainer_MTP::AddLinearE(int ind) {
    RequireCurrentParameterLayout();
    Atoms &atm = training_set_[ind];
    Neighborhoods &nbhs = trainint_set_nbhs_[ind];
    p_mtp->CalcECmpt(atm, nbhs, energy_cmpts);

    double avef = 0;
    if (atm.has_forces() && weight_force_norm_f > 0) {
        for (auto &it: atm.forces_) {
            avef += it.squaredNorm();
        }
        avef /= atm.size();
    }
    int soffset = 0;
    if (fix_species_coeffs) soffset = p_mtp->species_count;

    // Note
    /**
     * One can change this for weighting
     * **/
    double cc;
    if (weight_force_norm_f > 0)
        cc = weight_force_norm_d / (weight_force_norm_d + std::pow(weight_force_norm_f * avef, weight_force_norm_p));
    else
        cc = 1.0;

    // energy
    if (weight_eng > 0 && atm.has_energy()) {
        double weight_eng_i = energy_weight(atm) * cc; // gain more weight near equ
        for (int i = 0; i < n_size_mat; ++i) {
            for (int j = i; j < n_size_mat; ++j) {
                lin_matrix(i, j) += weight_eng_i * energy_cmpts[i+soffset] * energy_cmpts[j+soffset];
            }
            if (fix_species_coeffs) {
                double sub = 0;
                for (int s = 0; s < p_mtp->species_count; ++s) {
                    sub += energy_cmpts[s] * p_mtp->species_coeffs[s];
                }
                lin_vector[i] += weight_eng_i * energy_cmpts[i+soffset] * (atm.energy_ - sub);
            } else {
                lin_vector[i] += weight_eng_i * energy_cmpts[i] * atm.energy_;
            }
        }
    }
}

void Trainer_MTP::AddLinearEFast(int ind)
{
    RequireCurrentParameterLayout();
    Atoms &atm = training_set_[ind];
    Neighborhoods &nbhs = trainint_set_nbhs_[ind];
    p_mtp->CalcECmpt(atm, nbhs, energy_cmpts);

    double avef = 0;
    if (atm.has_forces() && weight_force_norm_f > 0) {
        for (auto &it: atm.forces_) {
            avef += it.squaredNorm();
        }
        avef /= atm.size();
    }
    int soffset = 0;
    if (fix_species_coeffs) soffset = p_mtp->species_count;

    // Note
    /**
     * One can change this for weighting
     * **/
    double cc;
    if (weight_force_norm_f > 0)
        cc = weight_force_norm_d / (weight_force_norm_d + std::pow(weight_force_norm_f * avef, weight_force_norm_p));
    else
        cc = 1.0;

    // energy
    if (weight_eng > 0 && atm.has_energy()) {
        double weight_eng_i = energy_weight(atm) * cc; // gain more weight near equ
        for (int i = 0; i < n_size_mat; ++i) {
            for (int j = i; j < n_size_mat; ++j) {
                eigen_lin_matrix(i, j) += weight_eng_i * energy_cmpts[i+soffset] * energy_cmpts[j+soffset];
            }
            if (fix_species_coeffs) {
                double sub = 0;
                for (int s = 0; s < p_mtp->species_count; ++s) {
                    sub += energy_cmpts[s] * p_mtp->species_coeffs[s];
                }
                eigen_lin_vector[i] += weight_eng_i * energy_cmpts[i+soffset] * (atm.energy_ - sub);
            } else {
                eigen_lin_vector[i] += weight_eng_i * energy_cmpts[i] * atm.energy_;
            }
        }
    }
}

void Trainer_MTP::TrainLinear() {
    RequireCurrentParameterLayout();
    if (linear_precondition != "none" && linear_precondition != "diagonal")
        GPERROR("Unknown linear_precondition: " + linear_precondition);
    if (fast_lfit) {
        TrainLinearFast();
    } else {
        TrainLinearAlg();
    }
}

// Solving A^T*A*x = A^T*y, or (\lambda * I + A^T * A) * x = A^T * y
void Trainer_MTP::TrainLinearAlg() {
    RequireCurrentParameterLayout();
    // reg_init = false;
    if (do_orthgonalization)
        p_mtp->Orthogonalize();
    // matrix, vector
    for (int i = 0; i < n_size_mat; ++i)
        for (int j = i; j < n_size_mat; ++j) {
            lin_matrix(i, j) = 0;
            if (mpi.rank == 0)
                lin_matrix_mpi(i, j) = 0;
        }
    for (int i = 0; i < n_size_mat; ++i) {
        lin_vector[i] = 0;
        if (mpi.rank == 0)
            lin_vector_mpi[i] = 0;
    }
    // Add energy, force, stress to linear matrix
    for (int i = 0; i < training_set_.size(); ++i) {
        if (linear_fit_eng_only)
            AddLinearE(i);
        else
            AddLinear(i);
        if (i%20 == 1 && mpi.rank == 0)
            std::cout <<  " on proc " + std::to_string(mpi.rank) << " linear add " << i << " / " << train_set_size << " added." << std::endl;
    }

#ifdef IMRMLP_MPI
    Training::reduce_mtp_linear_system(lin_matrix, lin_vector, lin_matrix_mpi,
        lin_vector_mpi, n_size_mat, true, mpi.rank == 0,
        linear_reduce_send, linear_reduce_receive,
        [](double *send, double *receive, int count) {
            MPI_Reduce(send, receive, count, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
        });
#else
    lin_vector_mpi = lin_vector;
    lin_matrix_mpi = lin_matrix;
#endif
    // solve the matrix on one processor and broadcast to all others
    std::string solve_error;
    if (mpi.rank == 0) try {
        const int coeff_offset = fix_species_coeffs ? p_mtp->species_count : 0;
        // Regularization uses the full linear coefficient indices.
        if (!reg_init) {
            /**
             * @brief 
             * adjust regulator coeffs.
             * B might be very large or very small even when using poly5 or cos smooth
             * function
             * We need to scale it here.
             */
            for (int i = 0; i < n_size_mat; ++i) {
                if (p_mtp->reg_L2_vector[coeff_offset + i] <
                    1e-2 * reg_L2 * 
                        std::max(1.0, lin_matrix_mpi(i, i) / (double) train_set_size_all) ||
                    p_mtp->reg_L2_vector[coeff_offset + i] >
                    1e2 * reg_L2 * 
                        std::max(1.0, lin_matrix_mpi(i, i) / (double) train_set_size_all)) {
                    reg_init = true;
                    break;
                }
            }
        }

        if (reg_init)
            for (int i = 0; i < n_size_mat; ++i) {
                p_mtp->reg_L2_vector[coeff_offset + i] = reg_L2 * std::max(1.0, lin_matrix_mpi(i, i)) / (double) train_set_size_all;
            }
        // Apply ridge regression to every coefficient in the reduced system.
        for (int i = 0; i < n_size_mat; ++i)
            lin_matrix_mpi(i, i) += p_mtp->reg_L2_vector[coeff_offset + i] * (double) train_set_size_all;

        std::vector<double> coordinate_scale(n_size_mat, 1.);
        if (linear_precondition == "diagonal")
            coordinate_scale = Training::equilibrate_linear_system(
                lin_matrix_mpi, lin_vector_mpi, n_size_mat, true);

        alglib::densesolverreport rep;
        alglib::real_1d_array x;
        x.setlength(n_size_mat);
        if (safe_linear_fitting) {
            SymLinearMatrix();
            alglib::rmatrixsolve(lin_matrix_mpi, n_size_mat, lin_vector_mpi, x, rep);
        }
        else {
            bool res = alglib::spdmatrixsolvefast(lin_matrix_mpi, n_size_mat, true, lin_vector_mpi);
            if (!res) rep.terminationtype = -3;
            x = lin_vector_mpi;
        }

        if (rep.terminationtype == -3) {
            GPERROR("Linear fitting failed: singular or ill-conditioned system");
        } else {
            for (int i = 0; i < n_size_mat; ++i) {
                linear_coeff_buffer[i] = x[i] * coordinate_scale[i];
            }
        }
        // **/
        if (!linear_coeff_buffer.allFinite()) GPERROR("Non-finite linear fit coefficients");
    } catch (const std::exception &e) { solve_error = e.what(); }
    check_linear_solve_error(solve_error);
#ifdef IMRMLP_MPI
    MPI_Bcast(linear_coeff_buffer.data(), n_size_mat, MPI_DOUBLE, 0, mpi.comm);
#endif
    if (fix_species_coeffs) {
        p_mtp->set_moment_coeff(linear_coeff_buffer.data());
    } else {
        p_mtp->set_linear_coeff(linear_coeff_buffer.data());
    }
}

void Trainer_MTP::TrainLinearFast()
{
    RequireCurrentParameterLayout();
    // reg_init = false;
    if (do_orthgonalization)
        p_mtp->Orthogonalize();
    // matrix, vector
    if (linear_fit_eng_only) {
        eigen_lin_matrix_A.resize(training_set_.size(), n_size_mat);
        eigen_lin_vector_A.resize(training_set_.size());
    } else {
        eigen_lin_matrix_A.resize(n_lfit_data, n_size_mat);
        eigen_lin_vector_A.resize(n_lfit_data);
    }
    eigen_lin_matrix.setZero();
    eigen_lin_matrix_mpi.setZero();
    eigen_lin_vector.setZero();
    eigen_lin_vector_mpi.setZero();
    position_ladd = 0;


    // Add energy, force, stress to linear matrix
    for (int i = 0; i < training_set_.size(); ++i) {
        if (linear_fit_eng_only)
            AddLinearEFast(i);
        else
            AddLinearFast(i);
        if (i%20 == 1 && mpi.rank == 0)
            std::cout <<  " on proc " + std::to_string(mpi.rank) << " linear add " << i << " / " << train_set_size << " added." << std::endl;
    }


        if (linear_fit_eng_only) {
            eigen_lin_matrix = eigen_lin_matrix.selfadjointView<Eigen::Upper>();
        } else {
            eigen_lin_matrix = eigen_lin_matrix_A.transpose() * eigen_lin_matrix_A;
            eigen_lin_vector = eigen_lin_matrix_A.transpose() * eigen_lin_vector_A;
        }

#ifdef IMRMLP_MPI
    // Eigen LDLT consumes the lower triangle; the energy-only path mirrors
    // its directly accumulated upper triangle before reaching this point.
    Training::reduce_mtp_linear_system(eigen_lin_matrix, eigen_lin_vector,
        eigen_lin_matrix_mpi, eigen_lin_vector_mpi, n_size_mat, false,
        mpi.rank == 0, linear_reduce_send, linear_reduce_receive,
        [](double *send, double *receive, int count) {
            MPI_Reduce(send, receive, count, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
        });
#else
    eigen_lin_vector_mpi = eigen_lin_vector;
    eigen_lin_matrix_mpi = eigen_lin_matrix;
#endif
    // solve the matrix on one processor and broadcast to all others
    std::string solve_error;
    if (mpi.rank == 0) try {
        const int coeff_offset = fix_species_coeffs ? p_mtp->species_count : 0;
        // Regularization uses the full linear coefficient indices.
        if (!reg_init) {
            /**
             * @brief 
             * adjust regulator coeffs.
             * B might be very large or very small even when using poly5 or cos smooth
             * function
             * We need to scale it here.
             */
            for (int i = 0; i < n_size_mat; ++i) {
                if (p_mtp->reg_L2_vector[coeff_offset + i] <
                    1e-2 * reg_L2 * 
                        std::max(1.0, eigen_lin_matrix_mpi(i, i) / (double) train_set_size_all) ||
                    p_mtp->reg_L2_vector[coeff_offset + i] >
                    1e2 * reg_L2 * 
                        std::max(1.0, eigen_lin_matrix_mpi(i, i) / (double) train_set_size_all)) {
                    reg_init = true;
                    break;
                }
            }
        }

        if (reg_init)
            for (int i = 0; i < n_size_mat; ++i) {
                p_mtp->reg_L2_vector[coeff_offset + i] = reg_L2 * std::max(1.0, eigen_lin_matrix_mpi(i, i)) / (double) train_set_size_all;
            }
        // Apply ridge regression to every coefficient in the reduced system.
        for (int i = 0; i < n_size_mat; ++i)
            eigen_lin_matrix_mpi(i, i) += p_mtp->reg_L2_vector[coeff_offset + i] * (double) train_set_size_all;
        if (debug) {
            writeMatrixToFile("MATATA", eigen_lin_matrix_mpi);
            writeMatrixToFile("VECATA", eigen_lin_vector_mpi);
        }
        std::vector<double> coordinate_scale(n_size_mat, 1.);
        if (linear_precondition == "diagonal")
            coordinate_scale = Training::equilibrate_linear_system(
                eigen_lin_matrix_mpi, eigen_lin_vector_mpi, n_size_mat, false);
        auto solver = eigen_lin_matrix_mpi.ldlt();
        if (solver.info() != Eigen::Success) GPERROR("Linear LDLT factorization failed");
        linear_coeff_buffer = solver.solve(eigen_lin_vector_mpi);
        for (int i = 0; i < n_size_mat; ++i) linear_coeff_buffer[i] *= coordinate_scale[i];
        if (!linear_coeff_buffer.allFinite()) GPERROR("Non-finite linear fit coefficients");
    } catch (const std::exception &e) { solve_error = e.what(); }
    check_linear_solve_error(solve_error);


#ifdef IMRMLP_MPI
    MPI_Bcast(linear_coeff_buffer.data(), n_size_mat, MPI_DOUBLE, 0, mpi.comm);
#endif
    if (fix_species_coeffs) {
        p_mtp->set_moment_coeff(linear_coeff_buffer.data());
    } else {
        p_mtp->set_linear_coeff(linear_coeff_buffer.data());
    }
}

double Trainer_MTP::energy_weight(const Atoms &atom) const {
    double wgt = weight_eng / pow(atom.size(), weight_eng_power);
    return wgt * atom.fitness;
}

double Trainer_MTP::force_weight(const Atoms &atom) const {
    double wgt = weight_force / pow(atom.size(), weight_force_power);
    return wgt;
}

double Trainer_MTP::stress_weight(const Atoms &atom) const {
    double wgt = weight_stress / pow(atom.size(), weight_stress_power);
    return wgt;
}

void Trainer_MTP::allocate() {
    RequireCurrentParameterLayout();
    n_size_mat = p_mtp->get_lin_coeff_size();
    if (fix_species_coeffs) n_size_mat -= p_mtp->species_count;
    if (!fast_lfit) {
        lin_matrix.setlength(n_size_mat, n_size_mat);
        lin_vector.setlength(n_size_mat);
        if (mpi.rank == 0) {
            lin_matrix_mpi.setlength(n_size_mat, n_size_mat);
            lin_vector_mpi.setlength(n_size_mat);
        }
        else {
            lin_matrix_mpi.setlength(1, 1);
            lin_vector_mpi.setlength(1);
        }
    } else {
        eigen_lin_matrix.resize(n_size_mat, n_size_mat);
        eigen_lin_vector.resize(n_size_mat);
        eigen_lin_matrix_A.resize(n_lfit_data, n_size_mat);
        eigen_lin_vector_A.resize(n_lfit_data);
        if (mpi.rank == 0) {
            eigen_lin_matrix_mpi.resize(n_size_mat, n_size_mat);
            eigen_lin_vector_mpi.resize(n_size_mat);
        } else {
            eigen_lin_matrix_mpi.resize(1, 1);
            eigen_lin_vector_mpi.resize(1);
        }
        weight_list.resize(n_lfit_data);
        weight_list.setZero();
        // pre-compute all weight data
        position_ladd = 0;
        for (int ind = 0; ind < training_set_.size(); ++ind) {
            auto &atom = training_set_[ind];
            double cc;
            double avef = 0;
            if (weight_force_norm_f > 0 && atom.has_forces()) [[unlikely]] {
                for (auto &it: atom.forces_) {
                    avef += it.squaredNorm();
                }
                avef /= atom.size();
                cc = weight_force_norm_d / (weight_force_norm_d + std::pow(weight_force_norm_f * avef, weight_force_norm_p));
            }
            else
                cc = 1.0;
            if (weight_force > 0 && atom.has_forces()) [[likely]] {
                double weight_force_i = force_weight(atom) * cc;
                for (int i = 0; i < atom.size(); ++i) {
                    double weight_force_iat = weight_force_i;
                    if (atom.atomic_fitness.size() > 0) {
                        weight_force_iat *= atom.atomic_fitness[i];
                    }
                    if (weight_force_iat <= 0) continue;
                    for (int a = 0; a < 3; ++a) {
                        weight_list(position_ladd) = sqrt(weight_force_iat);
                        ++position_ladd;
                    }
                }
            } 

            if (weight_stress > 0 && atom.has_stresses()) {
                double weight_stress_i = stress_weight(atom) * cc;
                for (int a = 0; a < 9; ++a) {
                    weight_list(position_ladd) = sqrt(weight_stress_i);
                    ++position_ladd;
                }
            }

            if (hvp_fit.weight > 0 && atom.has_hvp()) {
                const double weight = std::sqrt(hvp_fit.observation_weight(atom, cc));
                for (int k = 0; k < atom.hvp_directions_.size(); ++k)
                    weight_list(position_ladd++) = weight;
            }

            if (weight_eng > 0 && atom.has_energy()) [[likely]] {
                double weight_eng_i = energy_weight(atom) * cc;
                weight_list(position_ladd) = sqrt(weight_eng_i);
                ++position_ladd;
            }
        }
    }
    // Penalty evaluation indexes these by the full linear coefficient index.
    p_mtp->reg_L1_vector.assign(p_mtp->get_lin_coeff_size(), reg_L1);
    p_mtp->reg_L2_vector.assign(p_mtp->get_lin_coeff_size(), reg_L2);
    linear_coeff_buffer.resize(n_size_mat);
}

/**
 * Distribute training set into each mpi processor core
 * **/
void Trainer_MTP::init(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs, Potentials::PotMTP *pot) {
    if (hvp_fit.weight > 0 && linear_fit_eng_only)
        GPERROR("HVP fitting cannot be combined with lfit_eng_only");

    hvp_fit.validate();
    Training::HVPFitSettings{weight_hessian, weight_hessian_power, hessian_fd_step}.validate();
    for (const auto &a : atoms) {
        a.validate_hvp();
        if (a.have_hessian && (a.size() == 0 || !a.has_hessian() || !a.hessian_.allFinite()))
            GPERROR("Full Hessian labels require a finite Cartesian 3N by 3N matrix");
    }

    if (linear_precondition != "none" && linear_precondition != "diagonal")
        GPERROR("Unknown linear_precondition: " + linear_precondition);
    if (rescale_mode != "search" && rescale_mode != "fixed")
        GPERROR("Unknown rescale_mode: " + rescale_mode);
    if (debug) {
        debug_guard_ = std::make_unique<ScopedStreamRedirect>(
            global_debugstream,
            new std::ofstream("DEBUG_" + std::to_string(mpi.rank)));
    }
    p_mtp = pot;
    p_mtp->radial_mode = radial_reg_type;
    p_mtp->weight_rad_w0 = weight_reg_w0;
    p_mtp->weight_rad_w1 = weight_reg_w1;
    p_mtp->weight_rad_w2 = weight_reg_w2;
    p_mtp->random_lower = randomlow;
    p_mtp->random_upper = randomupw;
    p_mtp->quad_point = quad_count;
    if (!p_mtp->inited()) {
        if (random_init) p_mtp->random_init();
    }
    if (force_linear_randominit) {
        p_mtp->random_init_linear();
    }
    if (force_radial_randominit) {
        p_mtp->random_init_radial();
    }
    if (apply_disp) {
        std::random_device rd;
        std::mt19937 mtd(rd());
        std::normal_distribution<double> ncf(0, disp_sigma);
        for (int i = 0; i < p_mtp->reg_coeff_size; ++i) {
            p_mtp->regression_coeffs[i] += ncf(mtd);
        }
    }
    if (do_ladder) {
        if (ladder_cutoff > p_mtp->alpha_scalar_moments) {
            ladder_cutoff = p_mtp->alpha_scalar_moments;
        }
        p_ladder = new Potentials::PotMTP(*p_mtp);
        std::swap(p_ladder, p_mtp);
        // remove scalars after ladder_cut
        n_scalars_save = p_mtp->alpha_scalar_moments;
        std::set<int> ids_removed;
        for (int i = ladder_cutoff; i < n_scalars_save; ++i) {
            ids_removed.insert(i);
        }
        p_mtp->remove_scalar_by_id(ids_removed);
        p_mtp->remove_orphans();
        if (mpi.rank == 0) p_mtp->Save("LADDER_TEMPLATE.mtp");
        MPI_Barrier(mpi.comm);
        p_mtp->Load("LADDER_TEMPLATE.mtp");
        // fill ladder_last and ladder_cut
        std::random_device rd;
        std::mt19937 mtd(rd());
        std::uniform_real_distribution<double> ucf(randomlow, randomupw);
        for (int i = ladder_last; i < ladder_cutoff; ++i) {
            p_mtp->moment_coeffs[i] = ucf(mtd);
        }
    }
    p_mtp->train_init();
    training_set_ = atoms;
    efs_set_ = atoms;
    trainint_set_nbhs_ = nbhs;
    train_set_size = training_set_.size();
    train_set_size_aligned = train_set_size;
    train_set_size_all = train_set_size;
    n_radial_coeffs = p_mtp->radial_coeff_size;
    n_relax_coeffs = p_mtp->get_coeff_size();
    if (fix_linear_coeffs) n_relax_coeffs -= p_mtp->lin_coeff_size;
    if (fix_radial_coeffs) n_relax_coeffs -= p_mtp->radial_coeff_size;
    if (fix_linear_coeffs && fix_radial_coeffs) {
        GPERROR("Nothing to be optimized");
    }
    #ifdef IMRMLP_MPI
    MPI_Allreduce(&train_set_size, &train_set_size_all, 1, MPI_UNSIGNED_LONG, MPI_SUM, mpi.comm);
    MPI_Allreduce(&train_set_size, &train_set_size_aligned, 1, MPI_UNSIGNED_LONG, MPI_MAX, mpi.comm);
    #endif
    emonitor.t_size_all = train_set_size_all;
    train_set_inds_aligned.assign(train_set_size_aligned, 0);
    std::iota(train_set_inds_aligned.begin(), train_set_inds_aligned.begin() + train_set_size, 0);
    optimization_coeffs.setlength(n_relax_coeffs);
    p_mtp->ae_get_coeffs(optimization_coeffs, fix_radial_coeffs, fix_linear_coeffs);
    update_vector.setlength(n_relax_coeffs);
    update_vector_last.setlength(n_relax_coeffs);
    for (int i = 0; i < n_relax_coeffs; ++i) {
        update_vector[i] = 0;
        update_vector_last[i] = 0;
    }
    loss = 0;
    loss_gard.setlength(n_relax_coeffs);
    loss_mpi = 0;
    ploss = 0;
    loss_gard_mpi.setlength(n_relax_coeffs);
    pgard.setlength(n_relax_coeffs);
    for (int i = 0; i < n_relax_coeffs; ++i) {
        loss_gard[i] = 0;
        loss_gard_mpi[i] = 0;
        pgard[i] = 0;
    }
    // for linear fit (fast linear fit)
    n_lfit_data = n_lfit_data_E = n_lfit_data_F = n_lfit_data_S = 0;
    for (int i = 0; i < training_set_.size(); ++i) {
        auto &atom = training_set_[i];
        if (weight_eng > 0 && atom.has_energy()) {
            double wgt = energy_weight(atom);
            if (wgt >= 0) {
                n_lfit_data++;
                n_lfit_data_E++;
            }
        }
        if (weight_force > 0 && atom.has_forces()) {
            double weight_force_i = force_weight(atom);
            for (int i = 0; i < atom.size(); ++i) {
                double weight_force_iat = weight_force_i;
                if (atom.atomic_fitness.size() > 0) {
                    weight_force_iat *= atom.atomic_fitness[i];
                }
                if (weight_force_iat > 0) {
                    n_lfit_data += 3;
                    n_lfit_data_F += 3;
                }
            }
        }
        if (weight_stress > 0 && atom.has_stresses()) {
            double wgt = stress_weight(atom);
            if (wgt >= 0) {
                n_lfit_data += 9;
                n_lfit_data_S += 9;
            }
        }
        if (hvp_fit.weight > 0 && atom.has_hvp())
            n_lfit_data += atom.hvp_directions_.size();
    }
    MPI_Allreduce(&n_lfit_data, &n_lfit_data_all, 1, MPI_INT, MPI_SUM, mpi.comm);
    // Ladder graph truncation above is an intentional layout change. Cache
    // dimensions and snapshots are initialized together, before any fitting.
    CaptureParameterLayout();
    if (pot != nullptr) {
        allocate();
    }
    InitPenaltyGard();
    // update batch size
#ifdef IMRMLP_MPI
    batch_size_core = ceil(batch_size / mpi.size);
    batch_size = batch_size_core * mpi.size;
#else
    batch_size_core = batch_size;
#endif
    if (batch_size_core > train_set_size_aligned)
        batch_size_core = train_set_size_aligned;
    if (batch_size_core == 0)
        batch_size_core++;
    batches_per_epoch = std::ceil(static_cast<double>(train_set_size_aligned) / batch_size_core);
}

void Trainer_MTP::init_valid(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs) {
    RequireCurrentParameterLayout();
    for (const auto &a : atoms) a.validate_hvp();

    valid_set_ = atoms;
    efs_valid_set_ = atoms;
    valid_set_nbhs_ = nbhs;
    valid_set_size  = valid_set_.size();
    valid_set_size_all = valid_set_size;
    #ifdef IMRMLP_MPI
    MPI_Allreduce(&valid_set_size, &valid_set_size_all, 1, MPI_UNSIGNED_LONG, MPI_SUM, mpi.comm);
    #endif
}

void Trainer_MTP::AccumulateGard(double &f, alglib::real_1d_array &gard) {
    RequireCurrentParameterLayout();
    loss = 0;
    f = 0;
    emonitor.init();
    emonitor_valid.init();
    memset(loss_gard.getcontent(), 0, n_relax_coeffs * sizeof(double));
    memset(gard.getcontent(), 0, n_relax_coeffs * sizeof(double));
    for (int i = 0; i < train_set_size; ++i)
        AddLossGard(i, loss);
    #ifdef IMRMLP_MPI
    MPI_Reduce(&loss, &f, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(loss_gard.getcontent(), gard.getcontent(), n_relax_coeffs, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    #else
    gard = loss_gard;
    f = loss;
    #endif
    InitPenaltyGard();
    if (mpi.rank == 0) {
        f /= train_set_size_all;
        f += ploss;
        for (int i = 0; i < n_relax_coeffs; ++i) {
            gard[i] = gard[i] / train_set_size_all + pgard[i];
        }
    }
    // for alglib version > 4
    #ifdef IMRMLP_MPI
    MPI_Bcast(&f, 1, MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(gard.getcontent(), gard.length(), MPI_DOUBLE, 0, mpi.comm);
    #endif 
    emonitor.report();
    if (valid_set_size_all > 0) {
        double ff = 0;
        AccumulateTestErr(ff);
        emonitor_valid.report();
    }
}

void Trainer_MTP::AccumulateLoss(double &f) {
    RequireCurrentParameterLayout();
    loss = 0;
    f = 0;
    emonitor.init();
    emonitor_valid.init();
    for (int i = 0; i < train_set_size; ++i)
        AddLoss(i, loss);
    #ifdef IMRMLP_MPI
    MPI_Reduce(&loss, &f, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    #else
    f = loss;
    #endif
    InitPenaltyGard();
    if (mpi.rank == 0) {
        f = f / train_set_size_all + ploss;
    }
    // for alglib version > 4
    #ifdef IMRMLP_MPI
    MPI_Bcast(&f, 1, MPI_DOUBLE, 0, mpi.comm);
    #endif 
    emonitor.report();
    if (valid_set_size_all > 0) {
        double ff = 0;
        AccumulateTestErr(ff);
        emonitor_valid.report();
    }
}

void Trainer_MTP::AccumulateTestErr(double &f) {
    RequireCurrentParameterLayout();
    double loss_val = 0;
    f = 0;
    for (int i = 0; i < valid_set_size; ++i) {
        Atoms &atom = valid_set_[i];
        Neighborhoods &nbhs = valid_set_nbhs_[i];
        if (atom.size() == 0)
            return;

        Atoms &atomn = efs_valid_set_[i];
        p_mtp->getEFS(atomn, nbhs);

        if (atom.has_forces()) [[likely]] {
            for (int i = 0; i < atom.size(); ++i) {
                double ff = (atomn.forces_[i] - atom.forces_[i]).squaredNorm();
                emonitor_valid.sq_force_error += ff;
                emonitor_valid.nforces++;
            }
        } 

        if (atom.has_stresses()) {
            double ff = (atomn.stress_ - atom.stress_).squaredNorm();
            emonitor_valid.sq_stress_error += ff;
            emonitor_valid.nstresstensor++;
        }

        if (weight_eng > 0 && atom.has_energy()) [[likely]] {
            double ff = (atomn.energy_ - atom.energy_) * (atomn.energy_ - atom.energy_);
            emonitor_valid.sq_energy_at_error += ff / atom.size() / atom.size();
            emonitor_valid.nenergy++;
        }
        AddHessianLoss(atom, 1.0, f, false, emonitor_valid);
        AddHVPLoss(atom, 1.0, f, false, emonitor_valid);
    }
}

void bfgsgard(const alglib::real_1d_array &x, 
                            double &func, 
                            alglib::real_1d_array &gard, 
                            void *ptr) {
    Trainer_MTP *ptrainer = (Trainer_MTP *) ptr;
    ptrainer->p_mtp->ae_set_coeffs(x, ptrainer->fix_radial_coeffs, ptrainer->fix_linear_coeffs);
    ptrainer->AccumulateGard(func, gard);
    if (func < ptrainer->loss_f_prev) {
        ptrainer->loss_f_prev = func;
        ptrainer->display_iteration_log(func);
        ptrainer->bfgs_iter += 1;
    } else {
        Message("  Loss: " + std::to_string(func));
    }
    if (ptrainer->bfgs_iter % ptrainer->save_every_it == 0 && mpi.rank == 0) {
        ptrainer->save_current("_Current_potential_save_" + std::to_string(ptrainer->bfgs_iter) + ".mtp");
    }
}

void bfgsrep(const alglib::real_1d_array &, double, void *) {
    _DEBUG_OUTPUT_("REP CALLED");
}
int Trainer_MTP::DFConstraintCount() const {
    const int pairs = p_mtp->species_count * p_mtp->species_count;
    const int channels = p_mtp->radial_func_count;
    return pairs * channels + (df_orthogonal_constraints ? pairs * channels * (channels - 1) / 2 : 0);
}

void Trainer_MTP::DFConstraintBounds(alglib::real_1d_array &bounds) const {
    bounds.setlength(DFConstraintCount());
    const int norms = p_mtp->species_count * p_mtp->species_count * p_mtp->radial_func_count;
    for (int i = 0; i < bounds.length(); ++i) bounds[i] = i < norms ? 1.0 : 0.0;
}

void Trainer_MTP::EvaluateDFConstraints(const alglib::real_1d_array &x,
                                        alglib::real_1d_array &values) const {
    const int pairs = p_mtp->species_count * p_mtp->species_count;
    const int channels = p_mtp->radial_func_count;
    const int basis = p_mtp->rb_size;
    if (x.length() != pairs * channels * basis || values.length() != DFConstraintCount() + 1)
        GPERROR("DF radial coefficient or constraint size mismatch");
    // Keep the original pair-major norm rows; append off-diagonal Gram entries.
    int row = 1;
    for (int pair = 0; pair < pairs; ++pair)
        for (int k = 0; k < channels; ++k) {
            values[row] = 0.0;
            for (int l = 0; l < basis; ++l) {
                const double coefficient = x[(pair * channels + k) * basis + l];
                values[row] += coefficient * coefficient;
            }
            ++row;
        }
    if (df_orthogonal_constraints)
        for (int pair = 0; pair < pairs; ++pair)
            for (int k = 0; k < channels; ++k)
                for (int other = k + 1; other < channels; ++other) {
                    values[row] = 0.0;
                    for (int l = 0; l < basis; ++l)
                        values[row] += x[(pair * channels + k) * basis + l]
                                     * x[(pair * channels + other) * basis + l];
                    ++row;
                }
}

void Trainer_MTP::FitDFLinear() {
    RequireCurrentParameterLayout();
    // DF constraints apply to x itself, so the inner fit must retain that x.
    struct RestoreFlag {
        bool &flag;
        bool previous;
        ~RestoreFlag() { flag = previous; }
    } restore{do_orthgonalization, do_orthgonalization};
    do_orthgonalization = false;
    TrainLinear();
}

// Return the loss and radial Gram entries for the DF constraints.
void dffvec(const alglib::real_1d_array &x, 
                            alglib::real_1d_array &funci, 
                            void *ptr) {
    Trainer_MTP *ptrainer = (Trainer_MTP *) ptr;
    if (ptrainer->debug) {
        for (int i = 0; i < x.length(); ++i) {
            *global_debugstream << fmt::format("{:10}", x[i]);
        }
        *global_debugstream << std::endl;
    }
    ptrainer->p_mtp->ae_set_coeffs(x, false, true);
    ptrainer->FitDFLinear();
    ptrainer->AccumulateLoss(funci[0]);
    ptrainer->EvaluateDFConstraints(x, funci);
    if (funci[0] < ptrainer->loss_f_prev) {
        ptrainer->loss_f_prev = funci[0];
        double ff;
        ptrainer->AccumulateTestErr(ff);
        ptrainer->display_iteration_log(funci[0]);
        ptrainer->bfgs_iter += 1;
        if (mpi.rank == 0)
            ptrainer->save_current("_DF_save_" + std::to_string(ptrainer->bfgs_iter) + ".mtp");
    } else {
        Message("  Loss: " + std::to_string(funci[0]));
    }
}

void Trainer_MTP::RefitBFGS(const char *reason) {
    RequireCurrentParameterLayout();
    double before;
    AccumulateGard(before, loss_gard_mpi);
    TrainLinear();
    AccumulateGard(loss_f_prev, loss_gard_mpi);
    bfgs_last_linear_fit_iter = bfgs_iter;
    Message(fmt::format("BFGS linear fit at iteration {} ({}): loss {} -> {}",
                        bfgs_iter, reason, before, loss_f_prev));
}


// Optional diagonal Hessian preconditioning is applied to each L-BFGS solve.
void Trainer_MTP::TrainNonLinearBFGS(int limit) {
    RequireCurrentParameterLayout();
    if (limit < 0)
        GPERROR("BFGS iteration limit must be nonnegative");
    if (save_every_it <= 0)
        GPERROR("fit:save_every must be positive");

    // Keep the legacy CG callbacks separate: only accepted L-BFGS iterations
    // advance bfgs_iter, including across linear-fit restarts.
    struct CallbackState {
        Trainer_MTP *trainer;
        bool initial_report = true;
        bool refit_pending = false;
    } callback{this};
    auto gradient = [](const alglib::real_1d_array &x, double &f,
                       alglib::real_1d_array &g, void *ptr) {
        auto &trainer = *static_cast<CallbackState *>(ptr)->trainer;
        trainer.p_mtp->ae_set_coeffs(x, trainer.fix_radial_coeffs,
                                     trainer.fix_linear_coeffs);
        trainer.AccumulateGard(f, g);
    };
    auto accepted = [](const alglib::real_1d_array &x, double f, void *ptr) {
        auto &callback = *static_cast<CallbackState *>(ptr);
        auto &trainer = *callback.trainer;
        if (callback.initial_report) {
            callback.initial_report = false;
            return;
        }
        // A termination request may be followed by more callbacks.
        if (callback.refit_pending)
            return;
        ++trainer.bfgs_iter;
        trainer.loss_f_prev = f;
        trainer.p_mtp->ae_set_coeffs(x, trainer.fix_radial_coeffs,
                                     trainer.fix_linear_coeffs);
        trainer.display_iteration_log(f);
        if (mpi.rank == 0 && trainer.bfgs_iter % trainer.save_every_it == 0)
            trainer.save_current("_Current_potential_save_"
                + std::to_string(trainer.bfgs_iter) + ".mtp");
        if (trainer.linear_fit_during_bfgs && bfgs_linear_fit_step(trainer.bfgs_iter)) {
            callback.refit_pending = true;
            alglib::minlbfgsrequesttermination(trainer.bfgsstate);
        }
    };

    alglib::real_1d_array x;
    alglib::minlbfgsreport report;
    x.setlength(n_relax_coeffs);
    p_mtp->ae_get_coeffs(x, fix_radial_coeffs, fix_linear_coeffs);
    bfgs_iter = 0;
    bfgs_last_linear_fit_iter = -1;
    try {
        alglib::minlbfgscreate(n_relax_coeffs, bfgs_m, x, bfgsstate);
        alglib::minlbfgssetxrep(bfgsstate, true);
        AccumulateGard(loss_f_prev, loss_gard_mpi);
        if (bfgs_set_scale) {
            scale.setlength(x.length());
            if (mpi.rank == 0) {
                std::ifstream ifs("scale_data");
                for (int i = 0; i < x.length(); ++i)
                    ifs >> scale[i];
            }
            MPI_Bcast(scale.getcontent(), x.length(), MPI_DOUBLE, 0, mpi.comm);
            Message("Setting scale");
            alglib::minlbfgssetscale(bfgsstate, scale);
            alglib::minlbfgssetprecscale(bfgsstate);
        }
        Message("BFGS start");
        Message("Initial Loss: " + std::to_string(loss_f_prev));
        while (true) {
            const int iteration_start = bfgs_iter;
            callback.initial_report = true;
            callback.refit_pending = false;
            // ALGLIB uses zero for an unlimited iteration budget.
            const int remaining = limit > 0 ? limit - bfgs_iter : 0;
            alglib::minlbfgssetcond(bfgsstate, bfgs_grad_tol,
                                   bfgs_step_tol, bfgs_conv_tol, remaining);
            if (precond) {
                hess_diag.setlength(n_relax_coeffs);
                diag_hess(x);
                alglib::minlbfgssetprecdiag(bfgsstate, hess_diag);
            }
            alglib::minlbfgsoptimize(bfgsstate, gradient, accepted, &callback);
            alglib::minlbfgsresults(bfgsstate, x, report);
            MPI_Bcast(x.getcontent(), n_relax_coeffs, MPI_DOUBLE, 0, mpi.comm);
            p_mtp->ae_set_coeffs(x, fix_radial_coeffs, fix_linear_coeffs);
            if (report.iterationscount != bfgs_iter - iteration_start)
                GPERROR("L-BFGS accepted iteration count disagrees with ALGLIB");

            // Natural convergence or the stage limit takes precedence over
            // a scheduled refit; TrainBFGS handles the final fit once.
            if (!callback.refit_pending || report.terminationtype != 8
                || (limit > 0 && bfgs_iter >= limit))
                break;
            RefitBFGS("scheduled");
            reg_init = false;
            p_mtp->ae_get_coeffs(x, fix_radial_coeffs, fix_linear_coeffs);
            alglib::minlbfgsrestartfrom(bfgsstate, x);
            Message(fmt::format("BFGS restart after iteration {} (history reset)", bfgs_iter));
        }
    } catch (const alglib::ap_error &e) {
        GPERROR("ALGLIB L-BFGS failed: " + e.msg);
    }
    switch (report.terminationtype)
    {
    case -8:
        Warning("Inf or NaN in function/gradient");
        break;
    case -2:
        Warning("Rounding errors prevent further imporvement.");
        break;
    case -1:
        Warning("Incorrect parameters");
        break;
    case 1:
        Message("Relative function imporvement is no more than EpsF");
        break;
    case 2:
        Message("Relative step is no more than EpsX, stop optimization");
        break;
    case 4:
        Message("Gradient norm is no more than EpsG, stop optimization");
        break;
    case 5:
        Message("Max iteration limit reached, stop optimization");
        break;
    case 7:
        Message("Stopping conditions are too stringent");
        break;
    case 8:
        Message("Terminate optimization by callback function");
        break;
    default:
        break;
    }
    #ifdef IMRMLP_MPI
    MPI_Barrier(mpi.comm);
    #endif
}

void Trainer_MTP::TrainBFGS() {
    RequireCurrentParameterLayout();
    if (!p_mtp->inited()) {
        if (random_init)
            p_mtp->random_init();
        if (bfgs_rescale_final) {
            Message("Rescaling");
            Rescale();
        }
    }

    // pre training
    if (!skip_pre_init) {
        TrainNonLinearBFGS(pre_init_iter);
        // rescaling
        if (bfgs_rescale_final)
            Rescale();
        bfgs_iter = 0;
    }

    TrainNonLinearBFGS(maxiter);
    if (!linear_fit_eng_only && bfgs_rescale_final)
        Rescale();
    else if (linear_fit_during_bfgs && bfgs_last_linear_fit_iter != bfgs_iter)
        RefitBFGS("end");
    save_current(potfilesave);
}

void Trainer_MTP::TrainDF() {
    RequireCurrentParameterLayout();
    if (!p_mtp->inited()) {
        if (random_init)
            p_mtp->random_init();
    }
    TrainNonLinearDF(maxiter);
    save_current(potfilesave);
}

void Trainer_MTP::TrainNonLinearDF(int limit) {
    RequireCurrentParameterLayout();
    alglib::real_1d_array x;
    alglib::mindfreport report;
    x.setlength(n_radial_coeffs);
    // ignore if editor reports error, this class is a friend of potential class.
    // TODO: find a more 'clean' way to solve this problem.
    p_mtp->ae_get_coeffs(x, false, true);
    bfgs_iter = 0;
    try
    {
        /* code */
        alglib::mindfcreate(n_radial_coeffs, x, dfstate);
        alglib::mindfsetcondf(dfstate, bfgs_conv_tol);
        int seed;
        if (mpi.rank == 0) {
            auto rd = std::random_device();
            seed = rd();
        }
        MPI_Bcast(&seed, 1, MPI_INT, 0, mpi.comm);
        alglib::mindfsetseed(dfstate, seed);
        // set bc
        alglib::real_1d_array boundl, boundu;
        boundl.setlength(n_radial_coeffs);
        boundu.setlength(n_radial_coeffs);
        for (int i = 0; i < n_radial_coeffs; ++i) {
            boundl[i] = -1;
            boundu[i] = 1;
        }
        alglib::mindfsetbc(dfstate, boundl, boundu);
        // Norm rows equal one; optional pairwise dot products equal zero.
        {
            alglib::real_1d_array nl;
            DFConstraintBounds(nl);
            alglib::mindfsetnlc2(dfstate, nl, nl);
        }
        // set driver
        if (df_algo == 0)
            alglib::mindfsetalgogdemo(dfstate, maxiter, df_popsize);
        else 
            alglib::mindfsetalgogdemofixed(dfstate, maxiter, df_strategy, df_cprob, df_dweight, df_popsize);
    }
    catch(const alglib::ap_error& e)
    {
        std::cerr << e.msg << '\n';
    }
    // init
    AccumulateLoss(loss_f_prev);
    // Optionally load parameter scales for the derivative-free optimizer.
    if (bfgs_set_scale) {
        scale.setlength(x.length());
        // read scaling data
        if (mpi.rank == 0) {
            std::ifstream ifs("scale_data");
            for (int i = 0; i < x.length(); ++i) {
                ifs >> scale[i];
            }
        }
        #ifdef IMRMLP_MPI
        MPI_Bcast(scale.getcontent(), x.length(), MPI_DOUBLE, 0, mpi.comm);
        #endif
        Message("Setting scale");
        alglib::mindfsetscale(dfstate, scale);
    }
        Message("DF start");
        Message("Initial Loss: " + std::to_string(loss_f_prev));
        MPI_Barrier(mpi.comm);
        try
        {
            alglib::mindfoptimize(dfstate, dffvec, bfgsrep, this);
            alglib::mindfresults(dfstate, x, report);
        }
        catch(const alglib::ap_error& e)
        {
            std::cerr << e.msg << '\n';
        }
    MPI_Bcast(x.getcontent(), n_radial_coeffs, MPI_DOUBLE, 0, mpi.comm);
    p_mtp->ae_set_coeffs(x, false, true);
    switch (report.terminationtype)
    {
    case -8:
        Warning("Inf or NaN in function/gradient");
        break;
    case -2:
        Warning("Rounding errors prevent further imporvement.");
        break;
    case -1:
        Warning("Incorrect parameters");
        break;
    case 1:
        Message("Relative function imporvement is no more than EpsF");
        break;
    case 2:
        Message("Relative step is no more than EpsX, stop optimization");
        break;
    case 4:
        Message("Gradient norm is no more than EpsG, stop optimization");
        break;
    case 5:
        Warning("Max iteration limit reached, stop optimization");
        break;
    case 7:
        Warning("Stopping conditions are too stringent");
        break;
    case 8:
        Message("Terminate optimization by callback function");
        break;
    default:
        break;
    }
    #ifdef IMRMLP_MPI
    MPI_Barrier(mpi.comm);
    #endif
}

void Trainer_MTP::TrainRandom() {
    RequireCurrentParameterLayout();
    bfgs_iter = 0;
    double loss_prev = std::numeric_limits<double>::max();
    double loss_curr = 0;
    for (bfgs_iter = 0; bfgs_iter < maxiter; ++bfgs_iter) {
        Message("Iteration: " + std::to_string(bfgs_iter));
        p_mtp->random_init();
        TrainLinear();
        AccumulateLoss(loss_curr);
        display_iteration_log(loss_curr);
        if (loss_curr < loss_prev) {
            loss_prev = loss_curr;
            p_mtp->Save("_RANDOM_IT_" + std::to_string(bfgs_iter) + ".mtp");
        }
    }
}

/**
 * Options provided:
 * 1. A linear fit carried out after 10 epochs, all parameters are updated in SGD iteration
 * 2. A linear fit carried out after 10 epochs, only radial parameters are updated in SGD iteration
 *      Different init state may produce very different results.
 * 3. SGD without linear fit
 *
 * Here we choose 1.
 * **/
void Trainer_MTP::TrainSGD() {
    RequireCurrentParameterLayout();
    Training::collective_try([&] {
        if (maxepoches < 0 || save_every_it <= 0 || batches_per_epoch <= 0 || batch_size_core <= 0)
            GPERROR("SGD requires nonnegative epochs and positive save/batch counts");
        if (maxepoches > std::numeric_limits<int>::max() / batches_per_epoch)
            GPERROR("SGD epoch and batch counts exceed the iteration range");
        for (double value : {learning_rate, momentum, clip_gradient})
            if (!std::isfinite(value) || value < 0.)
                GPERROR("SGD learning rate, momentum and gradient clipping must be finite and nonnegative");
        // A zero-epoch run validates settings without stepping a scheduler.
        sgd_scheduler = Training::SGDScheduler::create_with_warmup(
            sgd_scheduler_type, learning_rate, sgd_final_lr,
            std::max(1, maxepoches * batches_per_epoch), sgd_decay_rate, sgd_decay_steps,
            sgd_warmup_steps, sgd_warmup_lr);
    });
    if (maxepoches == 0) return;
    // init
    if (!p_mtp->inited()) {
        Message("Init potential");
        if (random_init) {
            Message("Random init");
            p_mtp->random_init();
        }
        Message("Rescaling");
        Rescale();
        if (mpi.rank == 0)
            save_current(potfilesave);
    }

    bfgs_iter = 0;
    for (int iepoch = 0; iepoch < maxepoches; ++iepoch) {
        Message("Epoch: " + std::to_string(iepoch));
        ReBalance();
        int icurrent = 0;
        for (int ibatch = 0; ibatch < batches_per_epoch; ++ibatch) {
            // Padding synchronizes batch cadence, but contributes no samples.
            int this_batch = std::max(0, std::min(batch_size_core,
                                      static_cast<int>(train_set_size) - icurrent));
            Message(" batch: " + std::to_string(ibatch));
            loss = 0;
            memset(loss_gard.getcontent(), 0, n_relax_coeffs * sizeof(double));
            for (int i = 0; i < this_batch; ++i) {
                int id = train_set_inds_aligned[icurrent];
                ++icurrent;
                AddLossGard(id, loss);
            }
            Message(" loss: " + std::to_string(loss_mpi));
            SGDUpdate(this_batch);
            ++bfgs_iter;
        }
        if (mpi.rank == 0 && iepoch % save_every_it == 0) {
            save_current(potfilesave + std::to_string(iepoch));
        }
    }

    // postprocessing
    if (!linear_fit_eng_only && bfgs_rescale_final)
        Rescale();
    if (mpi.rank == 0)
        save_current(potfilesave);
    if (!p_mtp->inited())
        p_mtp->set_empty_flag();
    sgd_pretrain = true;
}

Trainer_MTP::~Trainer_MTP() {
    if (p_ladder) {
        delete p_ladder;
        p_ladder = nullptr;
    }
    // debug_guard_ (if set) restores global_debugstream and deletes ofstream automatically
}

void Trainer_MTP::AddLossGard(int id, double &_loss) {
    RequireCurrentParameterLayout();
    Atoms &atom = training_set_[id];
    Neighborhoods &nbhs = trainint_set_nbhs_[id];
    if (atom.size() == 0)
        return;

    Atoms &atomn = efs_set_[id];
    p_mtp->getEFS(atomn, nbhs);

    double dLdE = 0;
    dLdF.resize(atom.size(), 3);

    double cc;
    double avef = 0;
    if (weight_force_norm_f > 0 && atom.has_forces()) [[unlikely]] {
        for (auto &it: atom.forces_) {
            avef += it.squaredNorm();
        }
        avef /= atom.size();
        cc = weight_force_norm_d / (weight_force_norm_d + std::pow(weight_force_norm_f * avef, weight_force_norm_p));
    }
    else
        cc = 1.0;

    // add to loss, energy in the last to reduce round-off errors
    if (weight_force > 0 && atom.has_forces()) [[likely]] {
        double weight_force_i = force_weight(atom) * cc;
        for (int i = 0; i < atom.size(); ++i) {
            double weight_force_iat = weight_force_i;
            if (atom.atomic_fitness.size() > 0) {
                weight_force_iat *= atom.atomic_fitness[i];
            }
            double ff = (atomn.forces_[i] - atom.forces_[i]).squaredNorm();
            emonitor.sq_force_error += ff;
            emonitor.nforces++;
            double ll = weight_force_iat * ff;
            _loss += ll;
            emonitor.loss_f += ll;
            for (int a = 0; a < 3; ++a) {
                dLdF(i, a) = 2.0 * weight_force_iat * (atomn.forces_[i][a] - atom.forces_[i][a]);
            }
        }
    } else
        dLdF.setZero();

    if (weight_stress > 0 && atom.has_stresses()) {
        double weight_stress_i = stress_weight(atom) * cc;
        double ff = (atomn.stress_ - atom.stress_).squaredNorm();
        emonitor.sq_stress_error += ff;
        emonitor.nstresstensor++;
        double ll = weight_stress_i * ff;
        _loss += ll;
        emonitor.loss_s += ll;
        dLdS = 2 * weight_stress_i * (atomn.stress_ - atom.stress_); // Loss derivative with respect to predicted stress.
    } else {
        dLdS.setZero();
    }

    if (weight_eng > 0 && atom.has_energy()) [[likely]] {
        double weight_eng_i = energy_weight(atom) * cc;
        double ff = (atomn.energy_ - atom.energy_) * (atomn.energy_ - atom.energy_);
        emonitor.sq_energy_at_error += ff / atom.size() / atom.size();
        emonitor.nenergy++;
        double ll = weight_eng_i * ff;
        _loss += ll;
        emonitor.loss_e += ll;
        dLdE = 2 * weight_eng_i * (atomn.energy_ - atom.energy_);
    }

    if (fix_radial_coeffs) {
        p_mtp->AccumulateEFSCombinationGardLonly(atomn, nbhs, dLdE, dLdF, dLdS, loss_gard.getcontent());
        if (fix_species_coeffs) {
            for (int i = 0; i < p_mtp->species_count; ++i) {
                loss_gard[i] = 0;
            }
        }
    } else if (fix_linear_coeffs) {
        p_mtp->AccumulateEFSCombinationGardRonly(atomn, nbhs, dLdE, dLdF, dLdS, loss_gard.getcontent());
    } else {
        p_mtp->AccumulateEFSCombinationGard(atomn, nbhs, dLdE, dLdF, dLdS, loss_gard.getcontent());
        if (fix_species_coeffs) {
            for (int i = 0; i < p_mtp->species_count; ++i) {
                loss_gard[n_radial_coeffs + i] = 0;
            }
        }
    }
    // std::ofstream ofs("FUNC_MY_GARD");
    // ofs << loss << std::endl;
    // for (int i = 0; i < loss_gard.length(); ++i)
    // {
    //     if (i % 5 == 0)
    //         ofs << std::endl;
    //     ofs << loss_gard[i] << "   ";
    // }
    AddHessianLoss(atom, cc, _loss, true, emonitor);
    AddHVPLoss(atom, cc, _loss, true, emonitor);
    // Displaced-force pullbacks share the same frozen species block as E/F/S.
    if (fix_species_coeffs && !fix_linear_coeffs) {
        const int offset = fix_radial_coeffs ? 0 : n_radial_coeffs;
        for (int i = 0; i < p_mtp->species_count; ++i) loss_gard[offset + i] = 0;
    }
}

void Trainer_MTP::AddLoss(int id, double &_loss) {
    RequireCurrentParameterLayout();
    Atoms &atom = training_set_[id];
    Neighborhoods &nbhs = trainint_set_nbhs_[id];
    if (atom.size() == 0)
        return;

    Atoms &atomn = efs_set_[id];
    p_mtp->getEFS(atomn, nbhs);

    double cc;
    double avef = 0;
    if (weight_force_norm_f > 0 && atom.has_forces()) [[unlikely]] {
        for (auto &it: atom.forces_) {
            avef += it.squaredNorm();
        }
        avef /= atom.size();
        cc = weight_force_norm_d / (weight_force_norm_d + std::pow(weight_force_norm_f * avef, weight_force_norm_p));
    }
    else
        cc = 1.0;

    // add to loss, energy in the last to reduce round-off errors
    // add a atomic weight
    if (weight_force > 0 && atom.has_forces()) [[likely]] {
        double weight_force_i = force_weight(atom) * cc;
        for (int i = 0; i < atom.size(); ++i) {
            double weight_force_iat = weight_force_i;
            if (atom.atomic_fitness.size() > 0) {
                weight_force_iat *= atom.atomic_fitness[i];
            }
            double ff = (atomn.forces_[i] - atom.forces_[i]).squaredNorm();
            double ll = weight_force_iat * ff;
            _loss += ll;
            emonitor.sq_force_error += ff;
            emonitor.nforces++;
            emonitor.loss_f += ll;
        }
    } 

    if (weight_stress > 0 && atom.has_stresses()) {
        double weight_stress_i = stress_weight(atom) * cc;
        double ff = (atomn.stress_ - atom.stress_).squaredNorm();
        double ll = weight_stress_i * ff;
        _loss += ll;
        emonitor.sq_stress_error += ff;
        emonitor.nstresstensor++;
        emonitor.loss_s += ll;
    } 

    AddHessianLoss(atom, cc, _loss, false, emonitor);

    if (weight_eng > 0 && atom.has_energy()) [[likely]] {
        double weight_eng_i = energy_weight(atom) * cc;
        double ff = (atomn.energy_ - atom.energy_) * (atomn.energy_ - atom.energy_);
        double ll = weight_eng_i * ff;
        _loss += ll;
        emonitor.sq_energy_at_error += ff / atom.size() / atom.size();
        emonitor.nenergy++;
        emonitor.loss_e += ll;
    }
    AddHVPLoss(atom, cc, _loss, false, emonitor);
}


void Trainer_MTP::ReBalance() {
    RequireCurrentParameterLayout();
#ifdef IMRMLP_MPI
#ifdef ALL_IN_ONE
    if (mpi.rank == 0)
        std::shuffle(train_set_inds_all.begin(), train_set_inds_all.end(), RNG.mtd);
    MPI_Bcast(train_set_inds_all.data(),
              (int) train_set_inds_all.size(),
              MPI_INT,
              0,
              mpi.comm);
    // balance all training set to nodes
    train_set_inds.clear();
    for (int i = 0; i < train_set_inds_all.size(); ++i) {
        if (i % mpi.size == mpi.rank) {
            train_set_inds.push_back(train_set_inds_all[i]);
        }
    }
    if (train_set_inds.size() < train_set_size) {
        train_set_inds.push_back(train_set_inds[0]);
    }
#else 
    ;
#endif
#endif
    std::shuffle(train_set_inds_aligned.begin(), train_set_inds_aligned.begin() + train_set_size, RNG.mtd);
}

void Trainer_MTP::InitPenaltyGard() {
    RequireCurrentParameterLayout();
    // pgard.setlength(p_mtp->get_coeff_size());
    ploss = 0;
    for (int i = 0; i < n_relax_coeffs; ++i)
        pgard[i] = 0;
    p_mtp->AddPenaltyGard(weight_penatly, ploss, pgard.getcontent(), fix_radial_coeffs, fix_linear_coeffs, fix_species_coeffs);
}

Trainer_MTP::Trainer_MTP() {
    InitSettings();
}


void Trainer_MTP::SGDUpdate(int local_batch_size) {
    RequireCurrentParameterLayout();
    // Compute global batch size (total samples across all MPI ranks)
    int global_batch_size = local_batch_size;
#ifdef IMRMLP_MPI
    MPI_Allreduce(&local_batch_size, &global_batch_size, 1, MPI_INT, MPI_SUM, mpi.comm);
#endif

    if (global_batch_size == 0) {
        loss_mpi = 0.;
        memset(loss_gard_mpi.getcontent(), 0, n_relax_coeffs * sizeof(double));
        return;
    }

    // sum all loss and penalty, and broadcast
    loss_mpi = 0;
    memset(loss_gard_mpi.getcontent(), 0, n_relax_coeffs * sizeof(double));
    MPI_Reduce(&loss,
               &loss_mpi,
               1,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               mpi.comm);
    MPI_Reduce(loss_gard.getcontent(),
               loss_gard_mpi.getcontent(),
               n_relax_coeffs,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               mpi.comm);
    // update coeffs
    if (mpi.rank == 0) {
        double inv_batch = 1.0 / static_cast<double>(global_batch_size);
        // Frozen species already have zero gradients; radial coordinates come first.
        for (int i = 0; i < n_relax_coeffs; ++i) {
            loss_gard_mpi[i] *= inv_batch;
            loss_gard_mpi[i] += pgard[i];
        }
        // Gradient clipping
        if (clip_gradient > 0.0) {
            for (int i = 0; i < n_relax_coeffs; ++i) {
                loss_gard_mpi[i] = std::max(-clip_gradient,
                                  std::min(loss_gard_mpi[i], clip_gradient));
            }
        }
        p_mtp->ae_get_coeffs(optimization_coeffs, fix_radial_coeffs, fix_linear_coeffs);
        double current_lr = learning_rate;
        if (sgd_scheduler) {
            sgd_scheduler->on_step(bfgs_iter);
            current_lr = sgd_scheduler->get_lr();
        }
        for (int i = 0; i < n_relax_coeffs; ++i) {
            update_vector[i] = current_lr * loss_gard_mpi[i] + momentum * update_vector_last[i];
            optimization_coeffs[i] -= update_vector[i];
            update_vector_last[i] = update_vector[i];
        }
    }
    // Broadcast
#ifdef IMRMLP_MPI
    MPI_Bcast(optimization_coeffs.getcontent(),
              n_relax_coeffs,
              MPI_DOUBLE,
              0,
              mpi.comm);
#endif
    p_mtp->ae_set_coeffs(optimization_coeffs, fix_radial_coeffs, fix_linear_coeffs);
    InitPenaltyGard();
}

void Trainer_MTP::TrainCG() {
    RequireCurrentParameterLayout();
    if (!p_mtp->inited()) {
        if (random_init)
            p_mtp->random_init();
        Message("Rescaling");
        Rescale();
    }

    // pre training
    if (!skip_pre_init) {
        TrainNonLinearCG(pre_init_iter);
        // rescaling
        Rescale();
        bfgs_iter = 0;
    }

    TrainNonLinearCG(maxiter);
    if (!linear_fit_eng_only && bfgs_rescale_final)
        Rescale();
    else if (!linear_fit_eng_only)
        TrainLinear();
    save_current(potfilesave);
}

double Trainer_MTP::operator()(const Eigen::VectorXd &x_, Eigen::VectorXd &g_in) {
    RequireCurrentParameterLayout();
    // scaling back
    x_in = x_;
    for (int i = 0; i < x_in.size(); ++i) {
        x_in[i] /= scaling_coeffs[i];
    }
    p_mtp->set_coeffs(x_in, fix_radial_coeffs, fix_linear_coeffs);
    AccumulateGard(loss_mpi, loss_gard_mpi);

    // apply scaling
    // if (mpi.rank == 0) {
        memcpy(g_in.data(), loss_gard_mpi.getcontent(), n_relax_coeffs * sizeof(double));

        for (int i = 0; i < g_in.size(); ++i) {
            g_in[i] *= scaling_coeffs[i];
        }

        if (loss_mpi < loss_f_prev) {
            loss_f_prev = loss_mpi;
            Message("NonLinear iteration " + std::to_string(bfgs_iter));
            Message("  Loss: " + std::to_string(loss_mpi));
            ++bfgs_iter;
        } else {
            Message("  Loss: " + std::to_string(loss_mpi));
        }

        if (bfgs_iter % save_every_it == 0 && mpi.rank == 0)
            save_current("_Current_potential_save_" + std::to_string(bfgs_iter) + ".mtp");

    MPI_Barrier(mpi.comm);

    return loss_mpi;
}

void Trainer_MTP::TrainNonLinearLBFGS(int it_limit)
{
    RequireCurrentParameterLayout();
    bfgs_iter = 0;
    int nppiter;
    int olditmax = lbfgs_param.max_iterations;
    lbfgs_param.max_iterations = it_limit;
    double emf;

    if (lbfgs_param.linesearch == LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING) {
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchBacktracking> solver(lbfgs_param);
        nppiter = solver.minimize(*this, x_in, emf);
    } else if (lbfgs_param.linesearch == LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO) {
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchBracketing> solver(lbfgs_param);
        nppiter = solver.minimize(*this, x_in, emf);
    } else if (lbfgs_param.linesearch == LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchNocedalWright> solver(lbfgs_param);
        nppiter = solver.minimize(*this, x_in, emf);
    } else if (lbfgs_param.linesearch == LBFGSpp::LBFGS_LINESEARCH_LEWIS_OVERTON) {
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchLewisOverton> solver(lbfgs_param);
        nppiter = solver.minimize(*this, x_in, emf);
    } else {
        throw std::invalid_argument("unknown linesearch method");
    }

    lbfgs_param.max_iterations = olditmax;

}

void Trainer_MTP::TrainLBFGS()
{
    RequireCurrentParameterLayout();
    // set scaling
    scaling_coeffs.resize(n_relax_coeffs);
    scaling_coeffs.setOnes();

    if (!p_mtp->inited()) {
        if (random_init)
            p_mtp->random_init();
        Message("Rescaling");
        Rescale();
    }

    x_in.resize(n_relax_coeffs);
    g_in.resize(n_relax_coeffs);
    p_mtp->get_coeffs(x_in, fix_radial_coeffs, fix_linear_coeffs);
    for (int i = 0; i < x_in.size(); ++i) {
        x_in[i] *= scaling_coeffs[i];
    }

    // pre training
    if (!skip_pre_init) {
        TrainNonLinearLBFGS(pre_init_iter);
        // rescaling
        Rescale();
        bfgs_iter = 0;
    }

    for (int i = 0; i < n_relax_coeffs; ++i) {
        if (fabs(x_in[i]) < 1e-6) 
            scaling_coeffs[i] = 1e6;
        else if (fabs(x_in[i]) < 1e-2) 
            scaling_coeffs[i] = 100;
        else if (fabs(x_in[i]) < 1e-1) 
            scaling_coeffs[i] = 10;
        else if (fabs(x_in[i]) > 1e4)
            scaling_coeffs[i] = 1e-4;
        else 
            scaling_coeffs[i] = 1.0 / fabs(x_in[i]);
    }

    TrainNonLinearLBFGS(maxiter);
    if (!linear_fit_eng_only && bfgs_rescale_final)
        Rescale();
    else if (!linear_fit_eng_only)
        TrainLinear();
    save_current(potfilesave);
}

void Trainer_MTP::TrainRadialLBFGSThenLinear(int it_limit)
{
    RequireCurrentParameterLayout();
    if (it_limit > 0) {
        loss_f_prev = 9e99;
        const int old_maxiter = maxiter;
        maxiter = it_limit;

        try {
            TrainNonLinearBFGS(it_limit);
            maxiter = old_maxiter;
        } catch (const std::exception &e) {
            maxiter = old_maxiter;
            Warning("Radial alglib LBFGS failed, continue with linear fit: " + std::string(e.what()));
        } catch (...) {
            maxiter = old_maxiter;
            Warning("Radial alglib LBFGS failed with unknown error, continue with linear fit");
        }
    }

    TrainLinear();
}

void Trainer_MTP::TrainNonLinearCG(int it_limit)
{
    RequireCurrentParameterLayout();
    int ncoeff = p_mtp->get_coeff_size();
    alglib::real_1d_array x;
    alglib::mincgreport report;
    x.setlength(n_relax_coeffs);
    // ignore if editor reports error, this class is a friend of potential class.
    // TODO: find a more 'clean' way to solve this problem.
    p_mtp->ae_get_coeffs(x, fix_radial_coeffs, fix_linear_coeffs);
    bfgs_iter = 0;
    alglib::mincgcreate(n_relax_coeffs, x, cgstate);
    alglib::mincgsetcond(cgstate, bfgs_grad_tol, bfgs_step_tol, bfgs_conv_tol, maxiter);
        Message("CG start");
        MPI_Barrier(mpi.comm);
        alglib::mincgoptimize(cgstate, bfgsgard, bfgsrep, this);
        alglib::mincgresults(cgstate, x, report);
    MPI_Bcast(x.getcontent(), n_relax_coeffs, MPI_DOUBLE, 0, mpi.comm);
    p_mtp->ae_set_coeffs(x, fix_radial_coeffs, fix_linear_coeffs);
    switch (report.terminationtype)
    {
    case -8:
        Warning("Inf or NaN in function/gradient");
        break;
    case -7:
        Warning("Gradient verification failed");
        break;
    case 1:
        Message("Relative function imporvement is no more than EpsF");
        break;
    case 2:
        Message("Relative step is no more than EpsX, stop optimization");
        break;
    case 4:
        Message("Gradient norm is no more than EpsG, stop optimization");
        break;
    case 5:
        Warning("Max iteration limit reached, stop optimization");
        break;
    case 7:
        Warning("Stopping conditions are too stringent");
        break;
    case 8:
        Message("Terminate optimization by callback function");
        break;
    default:
        break;
    }
}

void Trainer_MTP::diag_hess(alglib::real_1d_array &x) {
    RequireCurrentParameterLayout();
    const double nudge_rate = 0.003;
    const double min_abs_nudge = 1e-4;
    auto coeff_orig = x;
    double loss0;
    AccumulateLoss(loss0);
    Message("Preconditioning");
    for (int i = 0; i < n_relax_coeffs; ++i) {
        const double nugeval = std::max(nudge_rate * std::abs(x[i]), min_abs_nudge);
        double lossp, lossn;
        x[i] += nugeval;
        p_mtp->ae_set_coeffs(x, fix_radial_coeffs, fix_linear_coeffs);
        AccumulateLoss(lossp);
        x[i] -= 2.0 * nugeval;
        p_mtp->ae_set_coeffs(x, fix_radial_coeffs, fix_linear_coeffs);
        AccumulateLoss(lossn);
        x = coeff_orig;
        hess_diag[i] = (lossp + lossn - loss0 - loss0) / nugeval / nugeval;
        if (mpi.rank == 0)
            std::cout << " ON PARAM " << i << " GET " << hess_diag[i] << std::endl;
    }

    p_mtp->ae_set_coeffs(coeff_orig, fix_radial_coeffs, fix_linear_coeffs);
}

void Trainer_MTP::save_current(std::string fnm) {
    RequireCurrentParameterLayout();
    if (do_ladder) {
        ladder_fin();
    }
    p_mtp->Save(fnm);
    if (do_ladder) {
        ladder_up();
    }
}

void Trainer_MTP::test_curv(int ind, double nudge) {
    RequireCurrentParameterLayout();
    alglib::real_1d_array x_coeffs, x_gard0, x_gard1, x_gard2;
    x_coeffs.setlength(n_relax_coeffs);
    x_gard0.setlength(n_relax_coeffs);
    x_gard2.setlength(n_relax_coeffs);
    x_gard1.setlength(n_relax_coeffs);
    p_mtp->ae_get_coeffs(x_coeffs, fix_radial_coeffs, fix_linear_coeffs);
    double f0, f1, f2;
    AccumulateGard(f0, x_gard0);
    double x0 = x_coeffs[ind];
    
    x_coeffs[ind] = x0 + nudge;
    p_mtp->ae_set_coeffs(x_coeffs, fix_radial_coeffs, fix_linear_coeffs);
    AccumulateGard(f1, x_gard1);
    x_coeffs[ind] = x0 - nudge;
    p_mtp->ae_set_coeffs(x_coeffs, fix_radial_coeffs, fix_linear_coeffs);
    AccumulateGard(f2, x_gard2);

    double hh = (f1 + f2 - 2.0 * f0) / nudge / nudge;

    if (mpi.rank == 0)
    std::cout << std::endl
            << " On coeff " << ind << " WITH COEFF = " << x_coeffs[ind] << " AND NUDGE = " << nudge << std::endl 
            << " F0 = " << f0 << " F1 = " << f1 << " F2 = " << f2 << std::endl
            << " G0 = " << x_gard0[ind] << " G1 = " << x_gard1[ind] << " G2 = " << x_gard2[ind] << std::endl
            << " HE = " << hh << std::endl;

    x_coeffs[ind] = x0;
    p_mtp->ae_set_coeffs(x_coeffs, fix_radial_coeffs, fix_linear_coeffs);
}

void Trainer_MTP::check_grad(int ind, double nudge) {
    RequireCurrentParameterLayout();
    alglib::real_1d_array x_coeffs, x_gard0, x_gard1, x_gard2;
    x_coeffs.setlength(n_relax_coeffs);
    x_gard0.setlength(n_relax_coeffs);
    x_gard2.setlength(n_relax_coeffs);
    x_gard1.setlength(n_relax_coeffs);
    p_mtp->ae_get_coeffs(x_coeffs, fix_radial_coeffs, fix_linear_coeffs);
    double f0, f1, f2;
    AccumulateGard(f0, x_gard0);
    double x0 = x_coeffs[ind];
    
    x_coeffs[ind] = x0 + nudge;
    p_mtp->ae_set_coeffs(x_coeffs, fix_radial_coeffs, fix_linear_coeffs);
    AccumulateGard(f1, x_gard1);
    x_coeffs[ind] = x0 - nudge;
    p_mtp->ae_set_coeffs(x_coeffs, fix_radial_coeffs, fix_linear_coeffs);
    AccumulateGard(f2, x_gard2);
    x_coeffs[ind] = x0;

    double hh = (f1 + f2 - 2.0 * f0) / nudge / nudge;
    double n0 = (f1 - f2) / 2.0 / nudge;
    double n1 = (f1 - f0) / nudge;
    double n2 = (f0 - f2) / nudge;

    if (mpi.rank == 0)
    std::cout << std::endl
            << " On coeff " << ind << " WITH COEFF = " << x_coeffs[ind] << " AND NUDGE = " << nudge << std::endl 
            << " F0 = " << f0 << " F1 = " << f1 << " F2 = " << f2 << std::endl
            << " G0 = " << x_gard0[ind] << " G1 = " << x_gard1[ind] << " G2 = " << x_gard2[ind] << std::endl
            << " N0 = " << n0 << " N1 = " << n1 << " N2 = " << n2 << std::endl
            << " N_CURV = " << hh << std::endl;

    p_mtp->ae_set_coeffs(x_coeffs, fix_radial_coeffs, fix_linear_coeffs);
}
// compute and output matrixA
void Trainer_MTP::get_A() {
    RequireCurrentParameterLayout();
    std::ofstream ofsE, ofsF, ofsS;
    std::ofstream ofsEv, ofsFv, osfSv;
    // header
    // NSCALAR NCONFS
    // for each conf:
    // ID NATOM
    // DATA
    int nconfs = training_set_.size();
    if (weight_eng > 0) {
        // CONFID DATA TARGET
        // do not give NSIZEMAT in this file
        ofsE = std::ofstream("MatrixA_Energy_CMPT_ON_CORE" + std::to_string(mpi.rank), std::ios::binary);
    }
    if (weight_force > 0) {
        // CONFID DATA TARGET
        ofsF = std::ofstream("MatrixA_Force_CMPT_ON_CORE" + std::to_string(mpi.rank), std::ios::binary);
    }
    if (weight_stress > 0) {
        // CONFID DATA TARGET
        ofsS = std::ofstream("MatrixA_Stress_CMPT_ON_CORE" + std::to_string(mpi.rank), std::ios::binary);
    }
    bool compute_all = false;
    if (weight_force > 0 || weight_stress > 0) compute_all = true;
    for (int ind = 0; ind < training_set_.size(); ++ind) {
        Atoms &atm = training_set_[ind];
        int idthis = atm.id;
        Neighborhoods &nbhs = trainint_set_nbhs_[ind];
        if (compute_all)
            p_mtp->CalcEFSCmpt(atm, nbhs, energy_cmpts, force_cmpts, stress_cmpts);
        else 
            p_mtp->CalcECmpt(atm, nbhs, energy_cmpts);

        int id = atm.id;
        int natom = atm.size();
        
        if (weight_eng > 0 && atm.has_energy()) {
            ofsE.write(reinterpret_cast<char*>(&idthis), sizeof(idthis));
            ofsE.write(reinterpret_cast<char*>(energy_cmpts.data()), sizeof(double)*n_size_mat);
            ofsE.write(reinterpret_cast<char*>(&atm.energy_), sizeof(double));
        }
        if (weight_force > 0 && atm.forces_.size() > 0) {
            for (int i = 0; i < atm.size(); ++i) {
                for (int a = 0; a < 3; ++a) {
                    ofsF.write(reinterpret_cast<char*>(&idthis), sizeof(idthis));
                    for (int l = 0; l < energy_cmpts.size(); ++l) {
                        // data
                        ofsF.write(reinterpret_cast<char*>(&force_cmpts(i, l, a)), sizeof(double));
                    }
                    // target
                    ofsF.write(reinterpret_cast<char*>(&atm.forces_[i](a)), sizeof(double));
                }
            }
        }
        if (weight_stress > 0 && atm.has_stresses()) {
            for (int a = 0; a < 3; ++a) {
                for (int b = a; b < 3; ++b) {
                    // Six rows represent the full symmetric tensor norm.
                    const double scale = a == b ? 1.0 : std::sqrt(2.0);
                    ofsS.write(reinterpret_cast<char*>(&idthis), sizeof(idthis));
                    for (int l = 0; l < energy_cmpts.size(); ++l) {
                        const double value = scale * stress_cmpts(l, a, b);
                        ofsS.write(reinterpret_cast<const char*>(&value), sizeof(double));
                    }
                    const double target = scale * atm.stress_(a, b);
                    ofsS.write(reinterpret_cast<const char*>(&target), sizeof(double));
                }
            }
        }
    }
    if (ofsE.is_open()) ofsE.close();
    if (ofsF.is_open()) ofsF.close();
    if (ofsS.is_open()) ofsS.close();
}

void Trainer_MTP::ladder_fin() {
    if (do_ladder) {
        RequireCurrentParameterLayout();
        // p_mtp is ladder potential created by this module
        // p_ladder is original potential
        // now swap back
        std::swap(p_mtp, p_ladder);
        // set zero
        memset(p_mtp->regression_coeffs, 0, p_mtp->reg_coeff_size * sizeof(double));
        // copy coeffs
        memcpy(p_mtp->regression_coeffs, p_ladder->regression_coeffs, p_ladder->reg_coeff_size * sizeof(double));
    }
}
// only in pair with ladder_fin for temp save
void Trainer_MTP::ladder_up() {
    RequireCurrentParameterLayout(true);
    std::swap(p_mtp, p_ladder);
}

void Trainer_MTP::display_iteration_log(double &f) {
    if (hvp_fit.weight > 0) {
        Message(fmt::format(" HVP monitor: loss = {} RMS = {}", emonitor.loss_hvp, emonitor.hvp_rms));
        if (valid_set_size > 0)
            Message(fmt::format(" Test HVP: RMS = {}", emonitor_valid.hvp_rms));
    }

    Message(fmt::format("Iteration {} : loss_f = {} f_real = {} f_eng = {} f_fce = {} f_str = {} f_pnal = {} ", bfgs_iter, f, f-ploss, emonitor.loss_e, emonitor.loss_f, emonitor.loss_s, ploss));
    Message(fmt::format(" Train monitor: EATRMS = {:.5f} FRMS = {:.5f} SRMS = {:.5f}", emonitor.erms * 1000, emonitor.frms * 1000, emonitor.srms));
    if (valid_set_size > 0) {
        Message(fmt::format(" Test monitor: EATRMS = {:.5f} FRMS = {:.5f} SRMS = {:.5f}", emonitor_valid.erms * 1000, emonitor_valid.frms * 1000, emonitor_valid.srms));
    }
}

// Reweightable blocks: global E/F/S weights are applied by fastfit, not here.
// Both trainer backends use this bounded-memory assembly for export.
void Trainer_MTP::get_ATA() {
    // Matrix export assembles a fresh system and is also valid after the
    // ladder's final swap back to its original output model.
    RequireCurrentParameterLayout(true);
    EFSFitWeights weights;
    weights.energy = 1.;
    weights.force = weights.stress = linear_fit_eng_only ? 0. : 1.;
    weights.energy_power = weight_eng_power;
    weights.force_power = weight_force_power;
    weights.stress_power = weight_stress_power;
    weights.norm_d = weight_force_norm_d;
    weights.norm_f = weight_force_norm_f;
    weights.norm_p = weight_force_norm_p;
    if (hvp_fit.weight > 0) GPERROR("E/F/S matrix export does not include HVP observations");
    std::unique_ptr<Training::LinearEFSSystem> system;
    int failed = 0, any_failed = 0;
    std::string error;
    try {
        system = std::make_unique<Training::LinearEFSSystem>(*p_mtp, training_set_, weights,
            fix_species_coeffs ? p_mtp->species_count : 0);
    } catch (const std::exception &e) { failed = 1; error = e.what(); }
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, mpi.comm);
    if (any_failed) GPERROR("EFS matrix assembly failed: " + error);
    system->reduce();
    if (mpi.rank == 0) {
        const char *names[] = {"E", "F", "S"};
        for (int i = 0; i < 3; ++i) {
            auto &block = system->channels[i];
            Eigen::VectorXd rhs = block.residual_rhs + block.gram * system->anchor;
            writeMatrixToBinFile(std::string("MATATA_") + names[i], block.gram);
            writeVectorToBinFile(std::string("VECATA_") + names[i], rhs);
        }
        std::ofstream metadata("MATATA.info");
        metadata << std::setprecision(17)
            << "# Unweighted E/F/S blocks, full 3x3 virial; fastfit format 2\n"
            << "energy-power " << weights.energy_power << '\n'
            << "force-power " << weights.force_power << '\n'
            << "stress-power " << weights.stress_power << '\n'
            << "force-norm-d " << weights.norm_d << '\n'
            << "force-norm-f " << weights.norm_f << '\n'
            << "force-norm-p " << weights.norm_p << '\n';
        if (!metadata) GPERROR("Failed to write MATATA.info");
    }
}

void Trainer_MTP::AddHVPLoss(const Atoms &atom, double normalization, double &loss,
    bool gradient, ErrorMonitor &monitor) {
    if (hvp_fit.weight == 0 || !atom.has_hvp()) return;
    Training::ForcePullback backward;
    if (gradient) backward = [&](Atoms &prediction, Neighborhoods &nbhs,
                                 Eigen::Matrix<double, -1, 3> &weights) {
        Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();
        if (fix_radial_coeffs)
            p_mtp->AccumulateEFSCombinationGardLonly(prediction, nbhs, 0., weights, stress, loss_gard.getcontent());
        else if (fix_linear_coeffs)
            p_mtp->AccumulateEFSCombinationGardRonly(prediction, nbhs, 0., weights, stress, loss_gard.getcontent());
        else
            p_mtp->AccumulateEFSCombinationGard(prediction, nbhs, 0., weights, stress, loss_gard.getcontent());
    };
    const auto result = Training::finite_difference_hvp_loss(*p_mtp, atom, hvp_fit, normalization, backward);
    loss += result.loss;
    monitor.add_hvp(result);
}

void Trainer_MTP::AddHessianLoss(const Atoms &atom, double normalization, double &loss,
    bool gradient, ErrorMonitor &monitor) {
    if (weight_hessian == 0 || !atom.has_hessian()) return;
    const int ndof = 3 * atom.size();
    Atoms probes = atom;
    probes.hvp_directions_ = Eigen::MatrixXd::Identity(ndof, ndof);
    probes.hvp_targets_ = atom.hessian_;
    // Cancel the shared HVP operator's probe average for the full Frobenius sum.
    Training::HVPFitSettings fit{weight_hessian * ndof, weight_hessian_power, hessian_fd_step};
    Training::ForcePullback backward;
    if (gradient) backward = [&](Atoms &prediction, Neighborhoods &nbhs,
                                 Eigen::Matrix<double, -1, 3> &weights) {
        Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();
        if (fix_radial_coeffs)
            p_mtp->AccumulateEFSCombinationGardLonly(prediction, nbhs, 0., weights, stress, loss_gard.getcontent());
        else if (fix_linear_coeffs)
            p_mtp->AccumulateEFSCombinationGardRonly(prediction, nbhs, 0., weights, stress, loss_gard.getcontent());
        else
            p_mtp->AccumulateEFSCombinationGard(prediction, nbhs, 0., weights, stress, loss_gard.getcontent());
    };
    const auto result = Training::finite_difference_hvp_loss(*p_mtp, probes, fit, normalization, backward);
    loss += result.loss;
    monitor.sq_hessian_error += result.squared_error;
    monitor.loss_h += result.loss;
    monitor.nhessian += result.observations;
}

void Trainer_MTP::ValidateSpeciesOnlySettings() const {
    if (fix_species_coeffs || fix_linear_coeffs || fix_radial_coeffs)
        GPERROR("trainmtp --species-only selects its own species parameter group; remove fix_*_coeffs");
    if (do_ladder || apply_disp || random_init || force_linear_randominit || force_radial_randominit)
        GPERROR("trainmtp --species-only cannot initialize, displace or ladder non-species coefficients");
    if (linear_fit_during_bfgs || bfgs_rescale_final || bfgs_set_scale || precond)
        GPERROR("trainmtp --species-only cannot use legacy refits, rescaling or preconditioning");
    if (maxiter <= 0)
        GPERROR("trainmtp --species-only requires positive bfgs_maxiteration");
}

void Trainer_MTP::TrainSpeciesOnly() {
    RequireCurrentParameterLayout();
    ValidateSpeciesOnlySettings();
    // Translate effective legacy objective/optimizer values, including defaults,
    // while the generic trainer owns group masks and accepted optimizer steps.
    GPSettings translated;
    const auto effective = CurrentSettings();
    for (const auto *key : {"energy_weight", "force_weight", "stress_weight",
        "energy_pow", "force_pow", "stress_pow", "hessian_weight", "hessian_pow",
        "hessian_fd_step", "hvp_weight", "hvp_pow", "hvp_fd_step",
        "force_wgt_d", "force_wgt_f", "force_wgt_p", "weight_L1", "weight_L2",
        "penalty_weight", "bfgs_m", "bfgs_conv_tol", "bfgs_grad_tol",
        "bfgs_step_tol", "bfgs_maxiteration", "save_every", "save_to"})
        translated.Modify(key, effective[key]);
    translated.Modify("trainable_groups", "species");
    translated.Modify("do_orthogonalization", "false");
    translated.Modify("skip_preinit", "true");
    TrainNonlinear trainer(translated);
    trainer.init(training_set_, trainint_set_nbhs_, p_mtp);
    trainer.init_valid(valid_set_, valid_set_nbhs_);
    trainer.TrainBFGS();
    emonitor = trainer.emonitor;
    emonitor_valid = trainer.emonitor_valid;
}
