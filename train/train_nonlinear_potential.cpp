// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "runtime/mtp/gpu-direct/gpu_backend.h"
#include "train_nonlinear_potential.h"
#include "conditional_linear_persistence.h"
#include "collective_error.h"
#include "gn_diagonal.h"
#include "potentials/potential_mtp_LES.h"
#include "tools/tools.h"
#include "utils/comm.h"
#include "libalglib/solvers.h"
#include "utils/gputils.h"
#include "libalglib/alglibmisc.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fmt/format.h>
#include <fmt/ostream.h>

// #include <chrono>
#include <functional>

#include <fstream>
#include <filesystem>
#include <string>
#include <limits>
#include <cstdlib>
#include <cctype>
#include <typeinfo>
#include <chrono>
#include <unistd.h>
#ifdef IMRMLP_MPI
#include <mpi.h>
#ifdef IMRMLP_GPU
#include "runtime/mtp/gpu-direct/gpu_mpi.h"
#endif
#endif

namespace {

struct ValidationDeferral {
    bool &flag;
    bool previous;
    explicit ValidationDeferral(bool &value, bool enable = true) : flag(value), previous(value) { flag = previous || enable; }
    ~ValidationDeferral() { flag = previous; }
};

#ifdef IMRMLP_GPU
int gpu_local_rank()
{
    const char *names[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID", "I_MPI_LOCAL_RANK", "PMI_LOCAL_RANK",
        "SLURM_LOCALID"
    };
    for (const char *name : names) {
        if (const char *value = std::getenv(name)) {
            char *end = nullptr;
            const long rank = std::strtol(value, &end, 10);
            if (end != value && *end == '\0' && rank >= 0 &&
                rank <= std::numeric_limits<int>::max()) {
                return static_cast<int>(rank);
            }
        }
    }
    return 0;
}

int select_gpu_device(int requested, std::string *error)
{
    if (requested >= 0) return requested;
    int count = 0;
    if (!MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count, error)) return -1;
    if (count <= 0) {
        if (error != nullptr) *error = "no visible GPU device";
        return -1;
    }
    return gpu_local_rank() % count;
}
#endif

} // namespace

void ChargeErrorMonitor::init() {
    sq_error = 0.0;
    loss_q = 0.0;
    q_min = std::numeric_limits<double>::infinity();
    q_max = -std::numeric_limits<double>::infinity();
    total_min = std::numeric_limits<double>::infinity();
    total_max = -std::numeric_limits<double>::infinity();
    nconfig = 0;
    qrms = 0.0;
}

void ChargeErrorMonitor::add(const Atoms &atom, double predicted,
                             double target, double qloss) {
    const double err = predicted - target;
    sq_error += err * err;
    loss_q += qloss;
    ++nconfig;
    total_min = std::min(total_min, predicted);
    total_max = std::max(total_max, predicted);
    for (double q : atom.charges_) {
        q_min = std::min(q_min, q);
        q_max = std::max(q_max, q);
    }
}

void ChargeErrorMonitor::report(size_t training_size_all) {
    double sq_all = 0.0, loss_all = 0.0;
    double qmin_all = 0.0, qmax_all = 0.0;
    double tmin_all = 0.0, tmax_all = 0.0;
    int n_all = 0;
    MPI_Reduce(&sq_error, &sq_all, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&loss_q, &loss_all, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&nconfig, &n_all, 1, MPI_INT, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&q_min, &qmin_all, 1, MPI_DOUBLE, MPI_MIN, 0, mpi.comm);
    MPI_Reduce(&q_max, &qmax_all, 1, MPI_DOUBLE, MPI_MAX, 0, mpi.comm);
    MPI_Reduce(&total_min, &tmin_all, 1, MPI_DOUBLE, MPI_MIN, 0, mpi.comm);
    MPI_Reduce(&total_max, &tmax_all, 1, MPI_DOUBLE, MPI_MAX, 0, mpi.comm);
    if (mpi.rank == 0) {
        qrms = n_all > 0 ? std::sqrt(sq_all / n_all) : 0.0;
        loss_q = training_size_all > 0 ? loss_all / training_size_all : 0.0;
        report_q_min = n_all > 0 ? qmin_all : 0.0;
        report_q_max = n_all > 0 ? qmax_all : 0.0;
        report_total_min = n_all > 0 ? tmin_all : 0.0;
        report_total_max = n_all > 0 ? tmax_all : 0.0;
    }
}

namespace {
void validate_hessian_settings(const Training::HVPFitSettings &fit) {
    if (!std::isfinite(fit.weight) || fit.weight < 0. || !std::isfinite(fit.power) ||
        !std::isfinite(fit.step) || fit.step <= 0.)
        GPERROR("Full Hessian fitting requires finite hessian_weight >= 0, hessian_pow and hessian_fd_step > 0");
}
void validate_hessian_label(const Atoms &atom) {
    if (atom.have_hessian && (atom.size() == 0 || !atom.has_hessian() || !atom.hessian_.allFinite()))
        GPERROR("Full Hessian labels require a finite Cartesian 3N by 3N matrix");
}
constexpr const char *unsupported_nonlinear_settings[] = {
    "weight_rad_w0", "weight_rad_w1", "weight_rad_w2", "radial_reg_type",
    "rand_low", "rand_up", "lfit_fastalgo", "popsize"
};
}

void TrainNonlinear::ApplySettings(const GPSettings &settings, const std::set<std::string> &skip) {
    // Preflight the complete update, including explicitly supplied defaults and
    // empty strings, before a supported binding can change the live trainer.
    for (const auto *name : unsupported_nonlinear_settings)
        if (!skip.count(name) && settings.count(name))
            GPERROR(std::string("Generic nonlinear training does not support setting: ") + name);
    ConnectSettings::ApplySettings(settings, skip);
}

void TrainNonlinear::InitSettings()
{
    auto unsupported = [&](auto &value, const char *name) {
        auto binding = Connect(value, name, "Unsupported by generic nonlinear training");
        // ConnectSettings::ApplySettings is not virtual. The binding also guards
        // callers using a base reference; snapshots omit these unsupported keys.
        binding->convert = [name](const Parameters::Occurrences &, Parameters::Source) -> Parameters::Commit {
            throw Parameters::Error(std::string("Generic nonlinear training does not support setting: ") + name);
        };
        binding->format = [] { return std::string(); };
    };
    Connect(debug, "debug", "", 1, 10);
    Connect(apply_disp, "apply_disp", "", 1, 10);
    Connect(disp_sigma, "disp_sigma", "", 1, 10);
    // General control
    // regulation weight L2
    Connect(reg_L2, "weight_reg"); // todo: discard this
    Connect(reg_L1, "weight_L1");
    Connect(reg_L2, "weight_L2");
    Connect(objective_convention, "objective-convention");
    Connect(l2_reference, "l2-reference");
    Connect(objective_report, "objective-report");
    unsupported(weight_reg_w0, "weight_rad_w0");
    unsupported(weight_reg_w1, "weight_rad_w1");
    unsupported(weight_reg_w2, "weight_rad_w2");
    unsupported(radial_reg_type, "radial_reg_type");
    unsupported(randomlow, "rand_low");
    unsupported(randomupw, "rand_up");
    Connect(potfilesave, "save_to");
    Connect(random_init, "random_init");
    Connect(force_random_init, "force_random");
    Connect(training_device, "device");
    Connect(gpu_device_id, "gpu_id");
    Connect(gpu_batch_atoms, "gpu_batch_atoms");
    Connect(gpu_batch_structures, "gpu_batch_structures");
    Connect(gpu_memory_fraction, "gpu_memory_fraction");
    Connect(gpu_device_loss, "gpu_device_loss");
    Connect(gpu_deterministic_reduction, "gpu_deterministic_reduction");
    Connect(gpu_radial_parameter_tile, "gpu_radial_parameter_tile");
    Connect(gpu_cache_batches, "gpu_cache_batches");
    Connect(validation_interval, "validation_interval",
            "Validate every N accepted L-BFGS/CG iterations; initial and final points always validate");
    unsupported(fast_lfit, "lfit_fastalgo");
    Connect(save_every_it, "save_every");
    // Weights
    Connect(hessian_fit.weight, "hessian_weight");
    Connect(hessian_fit.power, "hessian_pow");
    Connect(hessian_fit.step, "hessian_fd_step");
    Connect(hvp_fit.weight, "hvp_weight");
    Connect(hvp_fit.power, "hvp_pow");
    Connect(hvp_fit.step, "hvp_fd_step");
    Connect(weight_eng, "energy_weight");
    Connect(weight_force, "force_weight");
    Connect(weight_stress, "stress_weight");
    Connect(weight_charge, "charge_weight");
    Connect(weight_eng_power, "energy_pow");
    Connect(weight_force_power, "force_pow");
    Connect(weight_stress_power, "stress_pow");
    Connect(weight_charge_power, "charge_pow");
    Connect(charge_target_attrib, "charge_target_attrib");
    Connect(charge_default_target, "charge_default_target");
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
    Connect(bfgs_m, "bfgs_m");
    Connect(bfgs_conv_tol, "bfgs_conv_tol");
    Connect(bfgs_grad_tol, "bfgs_grad_tol");
    Connect(bfgs_step_tol, "bfgs_step_tol");
    Connect(maxiter, "bfgs_maxiteration");
    Connect(bfgs_iteration_offset, "bfgs_iteration_offset");
    Connect(bfgs_set_scale, "set_scale", "", 1, 10);
    Connect(training_schedule, "training_schedule");
    Connect(varpro_report, "varpro_report", "VP JSONL report (new file for each run)");
    Connect(varpro_radial_scale_file, "varpro_radial_scale_file", "Optional positive diagonal radial coordinate scales for VP");
    Connect(varpro_linear_refinement_steps, "varpro_linear_refinement_steps", "Maximum residual corrections at fixed radial parameters (0 disables)");
    Connect(varpro_linear_gradient_tolerance, "varpro_linear_gradient_tolerance");
    Connect(validation_metric, "validation_metric", "VP best-validation metric: energy, force or stress");
    Connect(workflow_options.refit_train_valid, "refit_train_valid",
            "After selection, refit train+valid with the fixed budget and no validation callbacks");
    Connect(workflow_options.regularization_search, "regularization_search",
            "Select a fixed L2 multiplier by validation before final training");
    Connect(workflow_options.regularization_multipliers, "regularization_multipliers",
            "Validation L2 candidates relative to weight_L2 (default 0.1 1 10)", -1);
    Connect(validation_max_energy, "validation_max_energy", "VP validation energy cap in eV/atom; negative disables");
    Connect(validation_max_force, "validation_max_force", "VP validation force cap in eV/Angstrom; negative disables");
    Connect(validation_max_stress, "validation_max_stress", "VP validation stress cap in GPa; negative disables");
    Connect(linear_refit_settings.mode, "linear_refit");
    Connect(linear_refit_settings.solver, "linear_refit_solver");
    Connect(linear_refit_settings.report, "linear_refit_report");
    Connect(linear_refit_settings.rtol, "linear_refit_rtol");
    Connect(linear_refit_settings.atol, "linear_refit_atol");
    Connect(linear_refit_settings.acceptance_tolerance, "linear_refit_tolerance");
    Connect(trainable_groups, "trainable_groups", "", -1);
    Connect(sr_stage_iterations, "sr_stage_iterations");
    Connect(les_stage_iterations, "les_stage_iterations");
    Connect(joint_stage_iterations, "joint_stage_iterations");
    Connect(charge_warmup_factor, "charge_warmup_factor");
    Connect(les_zero_init_range, "les_zero_init_range");
    Connect(sog_zero_init_range, "sog_zero_init_range");
    Connect(les_zero_init_seed, "les_zero_init_seed");
    Connect(curvature_precondition, "curvature_precondition", "none or gn_diag (local weighted residual VJP probes)");
    Connect(gn_probes, "gn_probes");
    Connect(gn_update, "gn_update", "Accepted iterations between curvature refreshes; 0 fixes the initial metric");
    Connect(gn_seed, "gn_seed");
    Connect(gn_floor, "gn_floor");
    Connect(parameter_scaling, "parameter_scaling");
    Connect(scale_floor, "scale_floor");
    Connect(scale_ceiling, "scale_ceiling");
    Connect(scaled_step_max, "scaled_step_max");
    // df
    unsupported(df_popsize, "popsize");
}

double TrainNonlinear::energy_weight(const Atoms &atom) const {
    return efs_weights().energy_weight(atom);
}

double TrainNonlinear::force_weight(const Atoms &atom) const {
    return efs_weights().force_weight(atom);
}

double TrainNonlinear::stress_weight(const Atoms &atom) const {
    return efs_weights().stress_weight(atom);
}

double TrainNonlinear::charge_weight(const Atoms &atom) const {
    return weight_charge / std::pow(atom.size(), weight_charge_power)
        * atom.fitness;
}

double TrainNonlinear::total_charge_target(const Atoms &atom) const {
    auto dit = atom.global_attribs_d_.find(charge_target_attrib);
    if (dit != atom.global_attribs_d_.end())
        return dit->second;
    auto fit = atom.features_.find(charge_target_attrib);
    if (fit == atom.features_.end())
        return charge_default_target;
    try {
        size_t parsed = 0;
        const double value = std::stod(fit->second, &parsed);
        if (parsed != fit->second.size() || !std::isfinite(value))
            throw std::invalid_argument("trailing characters");
        return value;
    } catch (const std::exception &) {
        GPERROR("Invalid total-charge target '" + fit->second
            + "' in Feature " + charge_target_attrib);
    }
    return charge_default_target;
}

/**
 * Distribute training set into each mpi processor core
 * **/
void TrainNonlinear::init(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs, Potentials::AnyPotential *pot) {
    InvalidateCGResult();
    auto local_init = [&](auto &&operation) {
        Training::collective_try(operation);
    };
    local_init([&] {
        if (!pot) GPERROR("Nonlinear training requires a potential");
        if (atoms.size() != nbhs.size()) GPERROR("Nonlinear training data/neighborhood counts differ");
        if (validation_interval < 1)
            GPERROR("fit:validation_interval must be positive");
        hvp_fit.validate();
        ValidateCurvatureSettings();
        validate_hessian_settings(hessian_fit);
        for (const auto &a : atoms) { a.validate_hvp(); validate_hessian_label(a); }
        validation_coefficients_.clear();
        validation_current_ = false;

        if (debug) {
            debug_guard_ = std::make_unique<ScopedStreamRedirect>(
                global_debugstream,
                new std::ofstream("DEBUG_" + std::to_string(mpi.rank)));
        }
        p_pot = pot;
        std::transform(training_device.begin(), training_device.end(), training_device.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        training_device = MLIPCore::GpuDirect::canonical_device(training_device);
        if ((hessian_fit.weight > 0 || hvp_fit.weight > 0) && training_device != "cpu")
            GPERROR("HVP and full Hessian fitting require CPU force-difference observations");
        if (linear_refit_settings.mode != "off") {
            // Validate the explicit model-owned block claim before initialization
            // or an output snapshot can mutate the caller's model or filesystem.
            Training::ConditionalLinearAdapter selected(*p_pot,
                Training::conditional_linear_selection(*p_pot,linear_refit_settings.mode),weight_penatly);
        }
        if (!MLIPCore::GpuDirect::valid_device(training_device)) {
            GPERROR("fit:device must be cpu, cuda, hip, or dcu");
        }
        if (!MLIPCore::GpuDirect::device_supported(training_device)) {
            GPERROR("fit:device=" + training_device + " is unavailable in this " +
                    MLIPCore::GpuDirect::compiled_backend() + " build");
        }
        if (gpu_batch_atoms < 0) {
            GPERROR("fit:gpu_batch_atoms must be zero (automatic) or positive");
        }
        if (gpu_batch_structures < 0) {
            GPERROR("fit:gpu_batch_structures must be zero (automatic) or positive");
        }
        if (!(gpu_memory_fraction > 0.0 && gpu_memory_fraction <= 1.0)) {
            GPERROR("fit:gpu_memory_fraction must be in (0, 1]");
        }
        if (gpu_radial_parameter_tile <= 0) {
            GPERROR("fit:gpu_radial_parameter_tile must be positive");
        }

        // 1. get relax coeffs
        n_relax_coeffs = p_pot->get_NL_coeffs_count();
        x.resize(n_relax_coeffs);
        p_pot->nl_get_coeffs(x.data());
        if (!p_pot->inited()) {
            if (random_init) p_pot->random_init();
            else p_pot->init_coeffs();
        }
        if (force_random_init) p_pot->random_init();
        // Constructors/random_init/init_coeffs mutate the potential, not x.
        // Keep the optimizer's initial vector synchronized so the first callback
        // cannot silently restore stale checkpoint coefficients.
        p_pot->nl_get_coeffs(x.data());

        if (apply_disp) {
            std::random_device rd;
            std::mt19937 mtd(rd());
            std::normal_distribution<double> ncf(0, disp_sigma);
            for (int i = 0; i < n_relax_coeffs; ++i) {
                x(i) = ncf(mtd);
            }
        }
    });
    if (apply_disp) {
        MPI_Bcast(x.data(), x.size(), MPI_DOUBLE, 0, mpi.comm);
    }
    local_init([&] {
        if (apply_disp) p_pot->nl_set_coeffs(x.data());
        p_pot->train_init();
        if (weight_charge > 0.0 && !p_pot->supports_total_charge_training())
            GPERROR("fit:charge_weight requires a potential with analytic total-charge training support");
        training_set_ = atoms;
        // Optional initialization aid for a newly randomized LES model.  It must
        // never run implicitly when resuming a trained potential.
        {
            auto* ples = dynamic_cast<Potentials::PotMTP_LES*>(p_pot);
            if (ples && ples->lr_les && ples->les_balance_initial_charges
                && (random_init || force_random_init))
                ples->balance_les_charges(atoms);
        }
        p_pot->nl_get_coeffs(x.data());
        efs_set_ = atoms;
        trainint_set_nbhs_ = nbhs;
        train_set_size = training_set_.size();
        train_set_size_aligned = train_set_size;
        train_set_size_all = train_set_size;
    });
    #ifdef IMRMLP_MPI
    MPI_Allreduce(&train_set_size, &train_set_size_all, 1, MPI_UNSIGNED_LONG, MPI_SUM, mpi.comm);
    MPI_Allreduce(&train_set_size, &train_set_size_aligned, 1, MPI_UNSIGNED_LONG, MPI_MAX, mpi.comm);
    #endif
    Training::collective_try([&] {
        objective_contract_ = Training::EFSObjective::resolve(objective_convention,
            l2_reference, reg_L2, train_set_size_all, false);
        if (objective_contract_.canonical()) {
            Training::EFSObjective::validate_weights(efs_weights());
            if (typeid(*p_pot) != typeid(Potentials::PotMTP)
                || static_cast<Potentials::PotMTP*>(p_pot)->radial_mode != 0
                || training_device != "cpu" || reg_L1 != 0 || weight_penatly != 0
                || weight_charge != 0 || hvp_fit.weight != 0 || hessian_fit.weight != 0)
                GPERROR("mean-moments requires ordinary CPU MTP radial_mode=0 and zero L1, radial penalty, charge, HVP and full-Hessian weights");
        }
        if (typeid(*p_pot) == typeid(Potentials::PotMTP)) objective_contract_.l2_scope = "moments";
        if (mpi.rank == 0) {
            objective_contract_.write(std::cout, efs_weights(), "# ");
            if (!objective_report.empty()) {
                if (std::filesystem::exists(objective_report)) GPERROR("objective-report must be a fresh path");
                std::ofstream out(objective_report);
                if (!out) GPERROR("Cannot open objective report: " + objective_report);
                objective_contract_.write(out, efs_weights());
                out << "L1_weight\t" << reg_L1 << "\nradial_penalty_weight\t" << weight_penatly
                    << "\ncharge_weight\t" << weight_charge << "\nHVP_weight\t" << hvp_fit.weight << '\n';
                out.close();
                if (!out) GPERROR("Cannot write objective report: " + objective_report);
            }
        }
    });
    local_init([&] {
        emonitor.t_size_all = train_set_size_all;
        train_set_inds_aligned.assign(train_set_size_aligned, 0);
        std::iota(train_set_inds_aligned.begin(), train_set_inds_aligned.begin() + train_set_size, 0);
        update_vector.resize(n_relax_coeffs);
        update_vector_last.resize(n_relax_coeffs);
        for (int i = 0; i < n_relax_coeffs; ++i) {
            update_vector[i] = 0;
            update_vector_last[i] = 0;
        }
        loss = 0;
        loss_gard.resize(n_relax_coeffs);
        loss_mpi = 0;
        ploss = 0;
        loss_gard_mpi.resize(n_relax_coeffs);
        pgard.resize(n_relax_coeffs);
        for (int i = 0; i < n_relax_coeffs; ++i) {
            loss_gard[i] = 0;
            loss_gard_mpi[i] = 0;
            pgard[i] = 0;
        }
        std::fill(p_pot->reg_L1_vector.begin(), p_pot->reg_L1_vector.end(), reg_L1);
        std::fill(p_pot->reg_L2_vector.begin(), p_pot->reg_L2_vector.end(), objective_contract_.native_l2);
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

    });
#ifdef IMRMLP_GPU
    gpu_evaluator_.reset();
    gpu_batch_.reset();
    gpu_cached_batches_.clear();
    gpu_validation_batches_.clear();
    if (training_device != "cpu") {
        collective_evaluation_errors_ = true;
        local_init([&] {
            Message("GPU backend: " + training_device);
            auto *mtp = dynamic_cast<Potentials::PotMTP *>(p_pot);
            if (mtp == nullptr) {
                GPERROR("GPU training requires PotMTP or a supported PotMTP_M model");
            }
            if (p_pot->supports_total_charge_training() || weight_charge > 0.0) {
                GPERROR("GPU MTP training does not yet support charge/LES loss terms");
            }
        });
        std::string gpu_error;
        int device = -1;
#ifdef IMRMLP_MPI
        if (!MLIPCore::GpuDirect::select_mpi_device(
                mpi.comm, gpu_device_id, &device, &gpu_error)) {
            GPERROR("Unable to select GPU device for MTP training: " + gpu_error);
        }
#else
        device = select_gpu_device(gpu_device_id, &gpu_error);
        if (device < 0) {
            GPERROR("Unable to select GPU device for MTP training: " + gpu_error);
        }
#endif
        local_init([&] {
            MLIPCore::GpuDirect::MTPGpuDeviceInfo device_info;
            if (!MLIPCore::GpuDirect::MTPGpuDirect::device_info(device, &device_info, &gpu_error)) {
                GPERROR(gpu_error);
            }
            Message("GPU device: " + device_info.name + " arch=" + device_info.architecture +
                    " identity=" + device_info.identity);
            gpu_evaluator_ = std::make_unique<Potentials::MTPGpuEvaluator>();
            gpu_batch_ = std::make_unique<Potentials::MTPGpuBatch>();
            gpu_batch_->configure_gradient_reduction(
                gpu_deterministic_reduction, gpu_radial_parameter_tile);
            if (!gpu_evaluator_->initialize(*p_pot, device, &gpu_error)) {
                GPERROR("Unable to initialize GPU MTP training: " + gpu_error);
            }
        });
#ifdef IMRMLP_MPI
        {
            char processor_name[MPI_MAX_PROCESSOR_NAME] = {};
            int processor_name_length = 0;
            MPI_Get_processor_name(processor_name, &processor_name_length);
            int visible_devices = 0;
            local_init([&] {
                if (!MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(
                        &visible_devices, &gpu_error)) {
                    GPERROR("Unable to query visible GPU devices: " + gpu_error);
                }
            });
            const int local_rank = gpu_local_rank();
            std::vector<char> processor_names(
                mpi.rank == 0
                    ? static_cast<size_t>(mpi.size) * MPI_MAX_PROCESSOR_NAME
                    : 0);
            std::vector<int> local_ranks(
                mpi.rank == 0 ? static_cast<size_t>(mpi.size) : 0);
            std::vector<int> device_ids(
                mpi.rank == 0 ? static_cast<size_t>(mpi.size) : 0);
            std::vector<int> visible_device_counts(
                mpi.rank == 0 ? static_cast<size_t>(mpi.size) : 0);
            MPI_Gather(processor_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                       processor_names.data(), MPI_MAX_PROCESSOR_NAME,
                       MPI_CHAR, 0, mpi.comm);
            MPI_Gather(&local_rank, 1, MPI_INT, local_ranks.data(), 1,
                       MPI_INT, 0, mpi.comm);
            MPI_Gather(&device, 1, MPI_INT, device_ids.data(), 1,
                       MPI_INT, 0, mpi.comm);
            MPI_Gather(&visible_devices, 1, MPI_INT,
                       visible_device_counts.data(), 1, MPI_INT, 0,
                       mpi.comm);
            if (mpi.rank == 0) {
                std::string placements = "[";
                for (int rank = 0; rank < mpi.size; ++rank) {
                    if (rank != 0) placements += ',';
                    const char *name = processor_names.data()
                        + static_cast<size_t>(rank) * MPI_MAX_PROCESSOR_NAME;
                    placements += fmt::format(
                        "{}@{}:local_rank={}:device={}:visible_devices={}",
                        rank, name, local_ranks[rank], device_ids[rank],
                        visible_device_counts[rank]);
                }
                placements += ']';
                Message("GPU MTP MPI placement: " + placements);
            }
        }
#endif
        local_init([&] {
            const auto potential_stats =
                gpu_evaluator_->context().potential_stats();
            Message(fmt::format(
                "GPU MTP moment workspace: declared_slots={}, active_slots={}, "
                "compaction_ratio={:.6f}, block_groups_forward={}, "
                "block_groups_reverse={}, neural_cache_doubles_per_center={}",
                potential_stats.declared_moment_slots,
                potential_stats.active_moment_slots,
                static_cast<double>(potential_stats.active_moment_slots) /
                    potential_stats.declared_moment_slots,
                potential_stats.forward_block_groups,
                potential_stats.reverse_block_groups,
                potential_stats.neural_cache_doubles_per_center));
            size_t total_memory = 0;
            if (!gpu_evaluator_->context().memory_info(
                    &gpu_free_memory_bytes_, &total_memory, &gpu_error)) {
                GPERROR("Unable to query GPU memory for MTP training: " + gpu_error);
            }
            gpu_memory_budget_bytes_ = static_cast<size_t>(
                static_cast<long double>(gpu_free_memory_bytes_) * gpu_memory_fraction);
            Message("GPU MTP training initialized on device " + std::to_string(device) +
                    " with " + std::to_string(train_set_size) + " local structures; " +
                    std::to_string(gpu_memory_budget_bytes_ / (1024 * 1024)) +
                    " MiB batch budget from " +
                    std::to_string(gpu_free_memory_bytes_ / (1024 * 1024)) +
                    " MiB free");
        });
#ifdef IMRMLP_MPI
        const unsigned long local_structure_count =
            static_cast<unsigned long>(train_set_size);
        std::vector<unsigned long> rank_structure_counts(
            mpi.rank == 0 ? static_cast<size_t>(mpi.size) : 0);
        MPI_Gather(&local_structure_count, 1, MPI_UNSIGNED_LONG,
                   rank_structure_counts.data(), 1, MPI_UNSIGNED_LONG,
                   0, mpi.comm);
        if (mpi.rank == 0) {
            std::string counts = "[";
            for (int rank = 0; rank < mpi.size; ++rank) {
                if (rank != 0) counts += ',';
                counts += std::to_string(rank_structure_counts[rank]);
            }
            counts += ']';
            Message(fmt::format(
                "GPU MTP MPI partition: ranks={}, local_structures={}, "
                "total_structures={}",
                mpi.size, counts, train_set_size_all));
        }
#endif
        local_init([&] {
            gpu_reciprocal_bytes_.resize(train_set_size);
            for(size_t i=0;i<train_set_size;++i)
                gpu_reciprocal_bytes_[i]=gpu_evaluator_->reciprocal_bytes(training_set_[i]);
            if (train_set_size > 0) {
                std::vector<int> all_indices(train_set_size);
                std::iota(all_indices.begin(), all_indices.end(), 0);
                const auto chunks = MakeGpuChunks(all_indices);
                size_t local_atoms = 0;
                size_t local_neighbors = 0;
                for (size_t index = 0; index < train_set_size; ++index) {
                    local_atoms += training_set_[index].size();
                    for (int atom = 0; atom < trainint_set_nbhs_[index].size(); ++atom) {
                        local_neighbors += static_cast<size_t>(
                            trainint_set_nbhs_[index][atom].nbhcount_);
                    }
                }
                Message("GPU MTP training planned " + std::to_string(chunks.size()) +
                        " structure-preserving chunk(s) for the local training set");
                Message(fmt::format(
                    "GPU MTP local inventory: structures={}, atoms={}, "
                    "directed_neighbors={}, estimated_peak_bytes={}",
                    train_set_size, local_atoms, local_neighbors,
                    EstimateGpuChunkBytes(all_indices)));
            }
        });
    }
#endif
}

void TrainNonlinear::init_valid(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs) {
    RequireCurrentParameterLayout();
    auto prepare = [&] {
        if (atoms.size() != nbhs.size())
            GPERROR("Validation structures and neighborhoods must have matching sizes");
        validation_coefficients_.clear();
        validation_current_ = false;
    #ifdef IMRMLP_GPU
        gpu_validation_batches_.clear();
    #endif
        for (const auto &a : atoms) { a.validate_hvp(); validate_hessian_label(a); }

        for (const auto &a : atoms) a.validate_hvp();
        valid_set_ = atoms;
        efs_valid_set_ = atoms;
        valid_set_nbhs_ = nbhs;
        valid_set_size  = valid_set_.size();
        valid_set_size_all = valid_set_size;
    };
    Training::collective_try(prepare);
    #ifdef IMRMLP_MPI
    MPI_Allreduce(&valid_set_size, &valid_set_size_all, 1, MPI_UNSIGNED_LONG, MPI_SUM, mpi.comm);
    #endif
    emonitor_valid.t_size_all = valid_set_size_all;
}

#ifdef IMRMLP_GPU
size_t TrainNonlinear::EstimateGpuBytes(size_t atoms, size_t neighbors,
                                         size_t structures, size_t reciprocal_bytes) const
{
    const auto *mtp = dynamic_cast<const Potentials::PotMTP *>(p_pot);
    if (mtp == nullptr) return std::numeric_limits<size_t>::max();

    // Forward and reverse retain four moment-sized arrays. The remaining
    // terms mirror MTPGpuDirect::State buffers and the contiguous batch CSR.
    const auto potential_stats = gpu_evaluator_ != nullptr
        ? gpu_evaluator_->context().potential_stats()
        : MLIPCore::GpuDirect::MTPGpuPotentialStats{};
    const int moment_slots = potential_stats.active_moment_slots > 0
        ? potential_stats.active_moment_slots
        : mtp->alpha_moments_count;
    const long double moment_bytes = 4.0L * atoms * moment_slots
        * sizeof(double);
    const long double neural_cache_bytes = static_cast<long double>(atoms) *
        potential_stats.neural_cache_doubles_per_center * sizeof(double);
    const long double radial_partial_bytes = static_cast<long double>(atoms) *
        mtp->get_species_count() * mtp->get_species_count() *
        mtp->radial_func_count * mtp->rb_size * sizeof(double);
    const long double atom_bytes = atoms * (
        4.0L * sizeof(int) + // centers, offsets, types, center_structure
        20.0L * sizeof(double)); // x/F/E/S, labels, weights, adjoints
    const long double neighbor_bytes = neighbors * (
        sizeof(int) + 3.0L * sizeof(double));
    // A second input slot lets topology/type/coordinate uploads for chunk
    // n+1 overlap the retained gradient work for chunk n.
    const long double pipeline_input_bytes = atoms * (
        3.0L * sizeof(int) + 3.0L * sizeof(double)) + neighbor_bytes;
    const long double structure_bytes = structures *
        (sizeof(int) + 22.0L * sizeof(double));
    const long double fixed_bytes =
        static_cast<long double>(gpu_evaluator_ ? gpu_evaluator_->full_gradient_count() : n_relax_coeffs) * sizeof(double) +
        2.0L * 1024.0L * 1024.0L;
    // Keep allocator alignment and temporary growth below the configured
    // budget even when the analytical layout changes slightly.
    const long double estimate =
        (moment_bytes + neural_cache_bytes + radial_partial_bytes + atom_bytes +
         neighbor_bytes + pipeline_input_bytes + structure_bytes + fixed_bytes + reciprocal_bytes)
        * 1.10L;
    if (estimate >= std::numeric_limits<size_t>::max())
        return std::numeric_limits<size_t>::max();
    return static_cast<size_t>(std::ceil(estimate));
}

size_t TrainNonlinear::EstimateGpuChunkBytes(
    const std::vector<int> &indices) const
{
    size_t atoms = 0, reciprocal_bytes = 0;
    size_t neighbors = 0;
    for (int index : indices) {
        if (index < 0 || static_cast<size_t>(index) >= train_set_size)
            return std::numeric_limits<size_t>::max();
        atoms += training_set_[index].size();
        if (gpu_reciprocal_bytes_.at(index) > std::numeric_limits<size_t>::max()-reciprocal_bytes)
            return std::numeric_limits<size_t>::max();
        reciprocal_bytes += gpu_reciprocal_bytes_.at(index);
        const auto &nbhs = trainint_set_nbhs_[index];
        for (int i = 0; i < nbhs.size(); ++i)
            neighbors += static_cast<size_t>(nbhs[i].nbhcount_);
    }
    return EstimateGpuBytes(atoms, neighbors, indices.size(), reciprocal_bytes);
}

std::vector<std::vector<int>> TrainNonlinear::MakeGpuChunks(
    const std::vector<int> &indices) const
{
    return MakeGpuChunks(indices, training_set_, trainint_set_nbhs_, gpu_reciprocal_bytes_);
}

std::vector<std::vector<int>> TrainNonlinear::MakeGpuChunks(
    const std::vector<int> &indices, const std::vector<Atoms> &atoms,
    const std::vector<Neighborhoods> &neighbors,
    const std::vector<size_t> &reciprocal_bytes) const
{
    std::vector<std::vector<int>> chunks;
    std::vector<int> current;
    size_t current_atoms = 0;
    size_t current_neighbors = 0, current_reciprocal = 0;

    for (int index : indices) {
        if (index < 0 || static_cast<size_t>(index) >= atoms.size()) {
            GPERROR("GPU MTP training batch contains an invalid structure index");
        }
        const size_t reciprocal = reciprocal_bytes.at(index);
        const size_t structure_atoms = atoms[index].size();
        size_t structure_neighbors = 0;
        const auto &nbhs = neighbors[index];
        for (int i = 0; i < nbhs.size(); ++i)
            structure_neighbors += static_cast<size_t>(nbhs[i].nbhcount_);

        const auto exceeds_limit = [&](size_t atoms, size_t neighbors,
                                       size_t structures, size_t long_range) {
            const bool atom_limit = gpu_batch_atoms > 0 &&
                atoms > static_cast<size_t>(gpu_batch_atoms);
            const bool structure_limit = gpu_batch_structures > 0 &&
                structures > static_cast<size_t>(gpu_batch_structures);
            return atom_limit || structure_limit ||
                EstimateGpuBytes(atoms, neighbors, structures, long_range) >
                    gpu_memory_budget_bytes_;
        };

        if (!current.empty() &&
            exceeds_limit(current_atoms + structure_atoms,
                          current_neighbors + structure_neighbors,
                          current.size() + 1, reciprocal > std::numeric_limits<size_t>::max()-current_reciprocal
                              ? std::numeric_limits<size_t>::max() : current_reciprocal + reciprocal)) {
            chunks.push_back(std::move(current));
            current.clear();
            current_atoms = 0;
            current_neighbors = 0; current_reciprocal = 0;
        }

        const size_t single_estimate = EstimateGpuBytes(
            structure_atoms, structure_neighbors, 1, reciprocal);
        if (exceeds_limit(structure_atoms, structure_neighbors, 1, reciprocal)) {
            GPERROR(fmt::format(
                "GPU MTP structure {} cannot fit one GPU chunk: atoms={}, "
                "directed_neighbors={}, estimated_bytes={}, budget_bytes={}, "
                "free_bytes={}, gpu_batch_atoms={}, gpu_batch_structures={}",
                index, structure_atoms, structure_neighbors, single_estimate,
                gpu_memory_budget_bytes_, gpu_free_memory_bytes_,
                gpu_batch_atoms, gpu_batch_structures));
        }
        current.push_back(index);
        current_atoms += structure_atoms;
        current_neighbors += structure_neighbors; current_reciprocal += reciprocal;
    }
    if (!current.empty()) chunks.push_back(std::move(current));
    return chunks;
}

void TrainNonlinear::PrepareGpuDeviceLabels(
    const std::vector<int> &indices,
    std::vector<double> &target_energy,
    std::vector<double> &energy_weights,
    std::vector<double> &target_force,
    std::vector<double> &force_weights,
    std::vector<double> &target_stress,
    std::vector<double> &stress_weights,
    int &energy_count, int &force_count, int &stress_count) const
{
    std::vector<int> atom_offsets(indices.size() + 1, 0);
    for (size_t s = 0; s < indices.size(); ++s) {
        atom_offsets[s + 1] = atom_offsets[s] +
            static_cast<int>(training_set_[indices[s]].size());
    }
    const size_t atom_count = static_cast<size_t>(atom_offsets.back());
    target_energy.assign(indices.size(), 0.0);
    energy_weights.assign(indices.size(), 0.0);
    target_force.assign(atom_count * 3, 0.0);
    force_weights.assign(atom_count, 0.0);
    target_stress.assign(indices.size() * 9, 0.0);
    stress_weights.assign(indices.size(), 0.0);

    energy_count = 0;
    force_count = 0;
    stress_count = 0;
    for (size_t local_structure = 0;
         local_structure < indices.size(); ++local_structure) {
        const Atoms &atom = training_set_[indices[local_structure]];
        if (atom.size() == 0) continue;

        const double cc = efs_weights().normalization(atom);

        if (weight_eng > 0 && atom.has_energy()) [[likely]] {
            target_energy[local_structure] = atom.energy_;
            energy_weights[local_structure] = energy_weight(atom) * cc;
            ++energy_count;
        }
        if (weight_stress > 0 && atom.has_stresses()) {
            stress_weights[local_structure] = stress_weight(atom) * cc;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    target_stress[9 * local_structure + 3 * a + b] =
                        atom.stress_(a, b);
                }
            }
            ++stress_count;
        }
        if (weight_force > 0 && atom.has_forces()) [[likely]] {
            const double structure_weight = force_weight(atom) * cc;
            const int atom_base = atom_offsets[local_structure];
            for (int i = 0; i < static_cast<int>(atom.size()); ++i) {
                double atom_weight = structure_weight;
                if (!atom.atomic_fitness.empty())
                    atom_weight *= atom.atomic_fitness[i];
                force_weights[atom_base + i] = atom_weight;
                for (int a = 0; a < 3; ++a) {
                    target_force[3 * (atom_base + i) + a] =
                        atom.forces_[i][a];
                }
            }
            force_count += static_cast<int>(atom.size());
        }
    }
}

void TrainNonlinear::AccumulateGpuDeviceChunk(
    Potentials::MTPGpuBatch &batch, const std::vector<int> &indices,
    const GpuCachedBatch *cached, bool reset_gradient,
    bool finalize_gradient, double &batch_loss)
{
    int energy_count = 0;
    int force_count = 0;
    int stress_count = 0;
    const std::vector<double> *target_energy = nullptr;
    const std::vector<double> *energy_weights = nullptr;
    const std::vector<double> *target_force = nullptr;
    const std::vector<double> *force_weights = nullptr;
    const std::vector<double> *target_stress = nullptr;
    const std::vector<double> *stress_weights = nullptr;
    if (cached != nullptr) {
        target_energy = &cached->target_energy;
        energy_weights = &cached->energy_weight;
        target_force = &cached->target_force;
        force_weights = &cached->force_weight;
        target_stress = &cached->target_stress;
        stress_weights = &cached->stress_weight;
        energy_count = cached->energy_count;
        force_count = cached->force_count;
        stress_count = cached->stress_count;
    } else {
        PrepareGpuDeviceLabels(
            indices, gpu_target_energy_, gpu_energy_weight_,
            gpu_target_force_, gpu_force_weight_, gpu_target_stress_,
            gpu_stress_weight_, energy_count, force_count, stress_count);
        target_energy = &gpu_target_energy_;
        energy_weights = &gpu_energy_weight_;
        target_force = &gpu_target_force_;
        force_weights = &gpu_force_weight_;
        target_stress = &gpu_target_stress_;
        stress_weights = &gpu_stress_weight_;
    }

    std::array<double,
        MLIPCore::GpuDirect::MTPGpuLossRequest::metric_count> metrics{};
    std::string gpu_error;
    if (!batch.evaluate_loss(
            *gpu_evaluator_, *target_energy, *energy_weights,
            *target_force, *force_weights, *target_stress,
            *stress_weights, metrics, reset_gradient,
            finalize_gradient, &gpu_error)) {
        GPERROR("GPU MTP device loss evaluation failed: " + gpu_error);
    }
    if (finalize_gradient) {
        batch_loss += metrics[0];
        emonitor.loss_e += metrics[1];
        emonitor.loss_f += metrics[2];
        emonitor.loss_s += metrics[3];
        emonitor.sq_energy_at_error += metrics[4];
        emonitor.sq_force_error += metrics[5];
        emonitor.sq_stress_error += metrics[6];
    }
    emonitor.nenergy += energy_count;
    emonitor.nforces += force_count;
    emonitor.nstresstensor += stress_count;

    if (!batch.accumulate_prepared_gradient(
            *gpu_evaluator_, loss_gard.data(), n_relax_coeffs,
            reset_gradient, finalize_gradient, &gpu_error)) {
        GPERROR("GPU MTP prepared gradient evaluation failed: " + gpu_error);
    }
}

void TrainNonlinear::AccumulateGpuBatch(const std::vector<int> &indices,
                                         double &batch_loss)
{
    if (train_set_size == 0 || indices.empty()) return;
    const auto wall_start = std::chrono::steady_clock::now();
    if (dynamic_cast<Potentials::PotMTP *>(p_pot) == nullptr ||
        gpu_evaluator_ == nullptr || gpu_batch_ == nullptr) {
        GPERROR("GPU MTP training was not initialized");
    }

    std::string gpu_error;
    if (!gpu_evaluator_->update_coefficients(*p_pot, &gpu_error)) {
        GPERROR("GPU MTP coefficient update failed: " + gpu_error);
    }
    const auto runtime_stats_before = gpu_evaluator_->context().runtime_stats();

    const auto chunks = MakeGpuChunks(indices);
    const bool full_ordered_selection = [&]() {
        if (!gpu_cache_batches || indices.size() != train_set_size)
            return false;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] != static_cast<int>(i)) return false;
        }
        return true;
    }();
    std::string cache_status = "off";
    if (full_ordered_selection && gpu_cached_batches_.empty()) {
        gpu_cached_batches_.reserve(chunks.size());
        for (const auto &chunk : chunks) {
            GpuCachedBatch cached;
            cached.indices = chunk;
            cached.batch = std::make_unique<Potentials::MTPGpuBatch>();
            cached.batch->configure_gradient_reduction(
                gpu_deterministic_reduction, gpu_radial_parameter_tile);
            if (!cached.batch->build(training_set_, trainint_set_nbhs_, chunk,
                                     &gpu_error)) {
                gpu_cached_batches_.clear();
                GPERROR("GPU MTP cached chunk construction failed: " +
                        gpu_error);
            }
            PrepareGpuDeviceLabels(
                chunk, cached.target_energy, cached.energy_weight,
                cached.target_force, cached.force_weight,
                cached.target_stress, cached.stress_weight,
                cached.energy_count, cached.force_count,
                cached.stress_count);
            gpu_cached_batches_.push_back(std::move(cached));
        }
        cache_status = "built";
    } else if (full_ordered_selection &&
               gpu_cached_batches_.size() == chunks.size()) {
        cache_status = "hit";
    }
    size_t selected_atoms = 0;
    size_t selected_neighbors = 0;
    for (int index : indices) {
        selected_atoms += training_set_[index].size();
        for (int atom = 0; atom < trainint_set_nbhs_[index].size(); ++atom) {
            selected_neighbors += static_cast<size_t>(
                trainint_set_nbhs_[index][atom].nbhcount_);
        }
    }
    bool accumulated_gradient = false;
    for (size_t chunk_index = 0; chunk_index < chunks.size(); ++chunk_index) {
        const auto &chunk = chunks[chunk_index];
        Potentials::MTPGpuBatch *batch = gpu_batch_.get();
        if (full_ordered_selection &&
            gpu_cached_batches_.size() == chunks.size() &&
            gpu_cached_batches_[chunk_index].indices == chunk) {
            batch = gpu_cached_batches_[chunk_index].batch.get();
        } else if (!batch->build(training_set_, trainint_set_nbhs_, chunk,
                                 &gpu_error)) {
            GPERROR("GPU MTP chunk construction failed: " + gpu_error);
        }
        if (gpu_device_loss) {
            const GpuCachedBatch *cached =
                full_ordered_selection &&
                gpu_cached_batches_.size() == chunks.size() &&
                gpu_cached_batches_[chunk_index].indices == chunk
                ? &gpu_cached_batches_[chunk_index] : nullptr;
            AccumulateGpuDeviceChunk(
                *batch, chunk, cached, !accumulated_gradient,
                chunk_index + 1 == chunks.size(), batch_loss);
            accumulated_gradient = true;
            continue;
        }
        if (!batch->evaluate(*gpu_evaluator_, efs_set_, chunk, true,
                             &gpu_error)) {
            GPERROR("GPU MTP chunk forward evaluation failed: " + gpu_error);
        }

        std::vector<int> atom_offsets(chunk.size() + 1, 0);
        for (size_t s = 0; s < chunk.size(); ++s) {
            atom_offsets[s + 1] = atom_offsets[s] +
                static_cast<int>(training_set_[chunk[s]].size());
        }
        gpu_dloss_denergy_.assign(chunk.size(), 0.0);
        gpu_dloss_dforce_.assign(
            static_cast<size_t>(atom_offsets.back()) * 3, 0.0);
        gpu_dloss_dstress_.assign(chunk.size() * 9, 0.0);

        for (size_t local_structure = 0;
             local_structure < chunk.size(); ++local_structure) {
            const int id = chunk[local_structure];
            Atoms &atom = training_set_[id];
            Atoms &prediction = efs_set_[id];
            if (atom.size() == 0) continue;

            const double cc = efs_weights().normalization(atom);

            const int atom_base = atom_offsets[local_structure];
            if (weight_force > 0 && atom.has_forces()) [[likely]] {
                const double structure_weight = force_weight(atom) * cc;
                for (int i = 0; i < static_cast<int>(atom.size()); ++i) {
                    double atom_weight = structure_weight;
                    if (!atom.atomic_fitness.empty())
                        atom_weight *= atom.atomic_fitness[i];
                    const double ff =
                        (prediction.forces_[i] - atom.forces_[i]).squaredNorm();
                    emonitor.sq_force_error += ff;
                    emonitor.nforces++;
                    const double ll = atom_weight * ff;
                    batch_loss += ll;
                    emonitor.loss_f += ll;
                    for (int a = 0; a < 3; ++a) {
                        gpu_dloss_dforce_[3 * (atom_base + i) + a] +=
                            2.0 * atom_weight *
                            (prediction.forces_[i][a] - atom.forces_[i][a]);
                    }
                }
            }

            if (weight_stress > 0 && atom.has_stresses()) {
                const double structure_weight = stress_weight(atom) * cc;
                const Eigen::Matrix3d error_tensor =
                    prediction.stress_ - atom.stress_;
                const double ff = error_tensor.squaredNorm();
                emonitor.sq_stress_error += ff;
                emonitor.nstresstensor++;
                const double ll = structure_weight * ff;
                batch_loss += ll;
                emonitor.loss_s += ll;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        gpu_dloss_dstress_[9 * local_structure + 3 * a + b] +=
                            2.0 * structure_weight * error_tensor(a, b);
                    }
                }
            }

            if (weight_eng > 0 && atom.has_energy()) [[likely]] {
                const double structure_weight = energy_weight(atom) * cc;
                const double error_energy = prediction.energy_ - atom.energy_;
                const double ff = error_energy * error_energy;
                emonitor.sq_energy_at_error += ff / atom.size() / atom.size();
                emonitor.nenergy++;
                const double ll = structure_weight * ff;
                batch_loss += ll;
                emonitor.loss_e += ll;
                gpu_dloss_denergy_[local_structure] +=
                    2.0 * structure_weight * error_energy;
            }
        }

        if (!batch->accumulate_gradient_chunks(
                *gpu_evaluator_, gpu_dloss_denergy_, gpu_dloss_dforce_,
                gpu_dloss_dstress_, loss_gard.data(), n_relax_coeffs,
                !accumulated_gradient, chunk_index + 1 == chunks.size(),
                &gpu_error)) {
            GPERROR("GPU MTP chunk gradient evaluation failed: " + gpu_error);
        }
        accumulated_gradient = true;
    }
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start).count();
    Message(fmt::format(
        "GPU MTP batch timing: structures={}, atoms={}, "
        "directed_neighbors={}, chunks={}, wall_s={:.6f}, atoms_per_s={:.3f}, "
        "device_loss={}, deterministic_reduction={}, host_batch_cache={}, "
        "topology_uploads={}, topology_reuses={}, async_input_uploads={}, "
        "geometry_upload_bytes={}, label_upload_bytes={}, sog_grid_upload_bytes={}, "
        "input_reuses={}, label_reuses={}, sog_grid_reuses={}, tracked_device_bytes={}",
        indices.size(), selected_atoms, selected_neighbors, chunks.size(),
        wall_seconds, selected_atoms / wall_seconds,
        gpu_device_loss, gpu_deterministic_reduction, cache_status,
        gpu_evaluator_->context().runtime_stats().topology_uploads -
            runtime_stats_before.topology_uploads,
        gpu_evaluator_->context().runtime_stats().topology_reuses -
            runtime_stats_before.topology_reuses,
        gpu_evaluator_->context().runtime_stats().pipelined_input_uploads -
            runtime_stats_before.pipelined_input_uploads,
        gpu_evaluator_->context().runtime_stats().coordinate_upload_bytes - runtime_stats_before.coordinate_upload_bytes,
        gpu_evaluator_->context().runtime_stats().label_upload_bytes - runtime_stats_before.label_upload_bytes,
        gpu_evaluator_->context().runtime_stats().sog_grid_upload_bytes - runtime_stats_before.sog_grid_upload_bytes,
        gpu_evaluator_->context().runtime_stats().input_reuses - runtime_stats_before.input_reuses,
        gpu_evaluator_->context().runtime_stats().label_reuses - runtime_stats_before.label_reuses,
        gpu_evaluator_->context().runtime_stats().sog_grid_reuses - runtime_stats_before.sog_grid_reuses,
        gpu_evaluator_->context().memory_stats().tracked_device_bytes));
}
#endif

#ifdef IMRMLP_GPU
void TrainNonlinear::EvaluateGpuValidation() {
    if (valid_set_size == 0) return;
    if (!gpu_evaluator_)
        GPERROR("GPU validation requires an initialized training evaluator");
    std::string error;
    if (!gpu_evaluator_->update_coefficients(*p_pot, &error)) GPERROR(error);
    if (!gpu_cache_batches) gpu_validation_batches_.clear();
    if (gpu_validation_batches_.empty()) {
        std::vector<int> indices;
        std::vector<size_t> reciprocal_bytes(valid_set_size);
        for (size_t i = 0; i < valid_set_size; ++i) {
            if (valid_set_[i].size() == 0) continue;
            indices.push_back(static_cast<int>(i));
            reciprocal_bytes[i] = gpu_evaluator_->reciprocal_bytes(valid_set_[i]);
        }
        for (const auto &chunk : MakeGpuChunks(
                 indices, valid_set_, valid_set_nbhs_, reciprocal_bytes)) {
            GpuCachedBatch cached;
            cached.indices = chunk;
            cached.batch = std::make_unique<Potentials::MTPGpuBatch>();
            cached.batch->configure_gradient_reduction(
                gpu_deterministic_reduction, gpu_radial_parameter_tile);
            if (!cached.batch->build(valid_set_, valid_set_nbhs_, chunk, &error))
                GPERROR("GPU validation batch construction failed: " + error);
            gpu_validation_batches_.push_back(std::move(cached));
        }
    }
    for (auto &cached : gpu_validation_batches_) {
        if (!cached.batch->evaluate(*gpu_evaluator_, efs_valid_set_,
                                    cached.indices, true, &error))
            GPERROR("GPU validation EFS failed: " + error);
    }
}
#endif

void TrainNonlinear::AccumulateGard(double &f, Eigen::VectorXd &gard) {
    RequireCurrentParameterLayout();
    loss = 0;
    f = 0;
    emonitor.init();
    validation_current_ = false;
    qmonitor.init();
    memset(loss_gard.data(), 0, n_relax_coeffs * sizeof(double));
    memset(gard.data(), 0, n_relax_coeffs * sizeof(double));
#ifdef IMRMLP_GPU
    if (training_device != "cpu") {
        std::vector<int> indices(train_set_size);
        std::iota(indices.begin(), indices.end(), 0);
        Training::collective_try([&] {AccumulateGpuBatch(indices, loss);});
    } else
#endif
    {
        auto accumulate = [&] {
            for (int i = 0; i < train_set_size; ++i) AddLossGard(i, loss);
        };
        if (collective_evaluation_errors_) Training::collective_try(accumulate);
        else accumulate();
    }
    #ifdef IMRMLP_MPI
    const char *mpi_profile_value = std::getenv("IMRMLP_PROFILE_MPI_COMM");
    const bool profile_mpi_communication =
        training_device != "cpu"
        && mpi_profile_value != nullptr
        && std::string(mpi_profile_value) != "0";
    double mpi_sync_seconds = 0.0;
    if (profile_mpi_communication) {
        const auto mpi_sync_start = std::chrono::steady_clock::now();
        MPI_Barrier(mpi.comm);
        mpi_sync_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - mpi_sync_start).count();
    }
    const auto mpi_reduce_start = std::chrono::steady_clock::now();
    MPI_Reduce(&loss, &f, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(loss_gard.data(), gard.data(), n_relax_coeffs, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    const double mpi_reduce_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - mpi_reduce_start).count();
    #else
    gard = loss_gard;
    f = loss;
    #endif
    if (collective_evaluation_errors_) Training::collective_try([&] {InitPenaltyGard();});
    else InitPenaltyGard();
    if (mpi.rank == 0) {
        f /= train_set_size_all;
        f += ploss;
        for (int i = 0; i < n_relax_coeffs; ++i) {
            gard[i] = gard[i] / train_set_size_all + pgard[i];
        }
    }
    // for alglib version > 4
    #ifdef IMRMLP_MPI
    const auto mpi_broadcast_start = std::chrono::steady_clock::now();
    MPI_Bcast(&f, 1, MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(gard.data(), gard.size(), MPI_DOUBLE, 0, mpi.comm);
    const double mpi_broadcast_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - mpi_broadcast_start).count();
    if (mpi.rank == 0 && profile_mpi_communication) {
        Message(fmt::format(
            "MTP MPI gradient communication: coefficients={}, "
            "bytes={}, sync_s={:.9f}, reduce_s={:.9f}, "
            "broadcast_s={:.9f}",
            n_relax_coeffs,
            static_cast<size_t>(n_relax_coeffs) * sizeof(double),
            mpi_sync_seconds, mpi_reduce_seconds,
            mpi_broadcast_seconds));
    }
    #endif 
    emonitor.report();
    qmonitor.report(train_set_size_all);
    if (!defer_validation_) EvaluateValidation();
}

void TrainNonlinear::AccumulateGard(double &f, alglib::real_1d_array &gard)
{
    RequireCurrentParameterLayout();
    AccumulateGard(f, loss_gard_mpi);
    for (int i = 0; i < loss_gard_mpi.size(); ++i) {
        gard[i] = loss_gard_mpi[i];
    }
}

void TrainNonlinear::AccumulateLoss(double &f) {
    RequireCurrentParameterLayout();
    loss = 0;
    f = 0;
    emonitor.init();
    validation_current_ = false;
    qmonitor.init();
#ifdef IMRMLP_GPU
    if (training_device != "cpu") {
        loss_gard.setZero();
        std::vector<int> indices(train_set_size);
        std::iota(indices.begin(),indices.end(),0);
        Training::collective_try([&] {AccumulateGpuBatch(indices,loss);});
    } else
#endif
    {
        for (int i = 0; i < train_set_size; ++i) AddLoss(i,loss);
    }
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
    qmonitor.report(train_set_size_all);
    if (!defer_validation_) EvaluateValidation();
}

void TrainNonlinear::EvaluateValidation() {
    RequireCurrentParameterLayout();
    if (valid_set_size_all == 0) return;
    const auto start = std::chrono::steady_clock::now();
    emonitor_valid.init();
    qmonitor_valid.init();
    double unused = 0.;
    if (collective_evaluation_errors_) Training::collective_try([&] {AccumulateTestErr(unused);});
    else AccumulateTestErr(unused);
    if (cg_active_) Training::collective_try([&] {
        if (!std::isfinite(emonitor_valid.sq_energy_at_error)
            || !std::isfinite(emonitor_valid.sq_force_error)
            || !std::isfinite(emonitor_valid.sq_stress_error)
            || !std::isfinite(emonitor_valid.sq_hessian_error) || !std::isfinite(emonitor_valid.loss_h)
            || !std::isfinite(emonitor_valid.sq_hvp_error) || !std::isfinite(emonitor_valid.loss_hvp)
            || !std::isfinite(qmonitor_valid.sq_error) || !std::isfinite(qmonitor_valid.loss_q))
            GPERROR("CG encountered non-finite validation errors");
    });
    emonitor_valid.report();
    qmonitor_valid.report(valid_set_size_all);
    if (cg_active_) Training::collective_try([&] {
        if (!std::isfinite(emonitor_valid.erms) || !std::isfinite(emonitor_valid.frms)
            || !std::isfinite(emonitor_valid.srms) || !std::isfinite(emonitor_valid.hrms)
            || !std::isfinite(emonitor_valid.loss_h) || !std::isfinite(emonitor_valid.hvp_rms)
            || !std::isfinite(emonitor_valid.loss_hvp)
            || !std::isfinite(qmonitor_valid.qrms) || !std::isfinite(qmonitor_valid.loss_q))
            GPERROR("CG encountered non-finite reduced validation errors");
    });
    validation_coefficients_.resize(n_relax_coeffs);
    p_pot->nl_get_coeffs(validation_coefficients_.data());
    if (cg_active_ && !Eigen::Map<const Eigen::VectorXd>(
            validation_coefficients_.data(), n_relax_coeffs).allFinite())
        GPERROR("CG validation returned non-finite coefficients");
    validation_current_ = true;
    ++validation_evaluations;
    Message(fmt::format(
        "Validation timing: device={}, structures={}, local_structures={}, evaluation={}, wall_s={:.6f}",
        training_device, valid_set_size_all, valid_set_size, validation_evaluations,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()));
}

void TrainNonlinear::ValidateAcceptedPoint(const double *coeffs, int iteration, bool force) {
    if (valid_set_size_all == 0 || (!force && iteration % validation_interval != 0)) return;
    // The last line-search probe need not be the accepted optimizer point.
    std::vector<double> current(n_relax_coeffs);
    auto set_current = [&] {
        SetOptimizerCoeffs(coeffs);
        p_pot->nl_get_coeffs(current.data());
    };
    if (collective_evaluation_errors_) Training::collective_try(set_current);
    else set_current();
    int changed = current != validation_coefficients_;
#ifdef IMRMLP_MPI
    // Every rank must participate, even ranks with no local validation data.
    MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, mpi.comm);
#endif
    if (changed) EvaluateValidation();
    else validation_current_ = true;
    Message(fmt::format("Validation at accepted iteration {}{}", iteration,
                        changed ? "" : " (reused current metrics)"));
    DisplayValidation();
}

void TrainNonlinear::AccumulateTestErr(double &f) {
    RequireCurrentParameterLayout();
    f = 0;
#ifdef IMRMLP_GPU
    if (training_device != "cpu") EvaluateGpuValidation();
#endif
    for (int i = 0; i < valid_set_size; ++i) {
        Atoms &atom = valid_set_[i];
        Neighborhoods &nbhs = valid_set_nbhs_[i];
        if (atom.size() == 0)
            continue;

        Atoms &atomn = efs_valid_set_[i];
        if (training_device == "cpu") p_pot->getEFS(atomn, nbhs);
        if (cg_active_) ValidateCGPrediction(atomn, atom, true);

        if (p_pot->supports_total_charge_training()) {
            const double predicted = p_pot->predicted_total_charge(atomn);
            const double target = total_charge_target(atom);
            const double err = predicted - target;
            const double qloss = charge_weight(atom) * err * err;
            qmonitor_valid.add(atomn, predicted, target, qloss);
        }

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
        AddHVPLoss(atom, 1.0, f, false, emonitor_valid);
        AddHessianLoss(atom, 1.0, f, false, emonitor_valid);
    }
}

void gnl_bfgsgard(const alglib::real_1d_array &x, 
                            double &func, 
                            alglib::real_1d_array &gard, 
                            void *ptr) {
    TrainNonlinear *ptrainer = (TrainNonlinear *) ptr;
    ptrainer->SetOptimizerCoeffs(x.getcontent());
    ptrainer->AccumulateGard(func, gard);
    ptrainer->ApplyTrainableMask(gard);
    if (func < ptrainer->loss_f_prev) {
        ptrainer->loss_f_prev = func;
        ptrainer->display_iteration_log(func);
        ptrainer->bfgs_iter += 1;
    } else {
        Message("  Loss: " + std::to_string(func));
    }
}

void TrainNonlinear::ConfigureTrainableGroups(
    const std::vector<std::string> &groups) {
    RequireCurrentParameterLayout();
    trainable_mask.assign(n_relax_coeffs, groups.empty() ? 1 : 0);
    const auto available = p_pot->nl_parameter_groups();
    for (const auto &group : available) {
        if (group.begin < 0 || group.end < group.begin
            || group.end > n_relax_coeffs)
            GPERROR("Potential returned an invalid nonlinear parameter group: "
                + group.name);
    }
    for (const std::string &requested : groups) {
        auto it = std::find_if(available.begin(), available.end(),
            [&](const Potentials::NLParameterGroup &group) {
                return group.name == requested;
            });
        if (it == available.end())
            GPERROR("Unknown nonlinear parameter group: " + requested);
        std::fill(trainable_mask.begin() + it->begin,
                  trainable_mask.begin() + it->end, 1);
    }
    if (std::none_of(trainable_mask.begin(), trainable_mask.end(),
                     [](char value) { return value != 0; }))
        GPERROR("No nonlinear coefficients selected for optimization");
    frozen_coeffs.resize(n_relax_coeffs);
    p_pot->nl_get_coeffs(frozen_coeffs.data());
}

void TrainNonlinear::ApplyTrainableMask(
    alglib::real_1d_array &gradient) const {
    if (trainable_mask.empty())
        return;
    for (int i = 0; i < n_relax_coeffs; ++i)
        if (!trainable_mask[i])
            gradient[i] = 0.0;
}

void TrainNonlinear::SetOptimizerCoeffs(const double *coeffs) {
    RequireCurrentParameterLayout();
    if (!cg_active_) InvalidateCGResult();
    validation_current_ = false;
    if (trainable_mask.empty()) {
        p_pot->nl_set_coeffs(coeffs);
        return;
    }
    x = Eigen::Map<const Eigen::VectorXd>(coeffs, n_relax_coeffs);
    for (int i = 0; i < n_relax_coeffs; ++i)
        if (!trainable_mask[i])
            x[i] = frozen_coeffs[i];
    p_pot->nl_set_coeffs(x.data());
}

void gnl_bfgsrep(const alglib::real_1d_array &x,
                           double func, 
                           void *ptr) {
    TrainNonlinear *ptrainer = (TrainNonlinear *) ptr;
    // ALGLIB invokes the x-report callback once for the initial point before
    // completing any optimizer iteration.  Keep that state out of accepted
    // iteration counts and periodic checkpoint names.
    if (!ptrainer->bfgs_initial_reported) {
        ptrainer->bfgs_initial_reported = true;
        ptrainer->ValidateAcceptedPoint(x.getcontent(), ptrainer->bfgs_iteration_offset, true);
        if (mpi.rank == 0) {
            Message(fmt::format(
                "LBFGS initial point: loss_f = {:.17g}", func));
        }
        return;
    }
    ++ptrainer->bfgs_accepted_iter;
    const int cumulative_iteration = ptrainer->bfgs_iteration_offset +
        ptrainer->bfgs_accepted_iter;
    if (mpi.rank == 0) {
        Message(fmt::format(
            "LBFGS accepted iteration {}: loss_f = {:.17g}",
            cumulative_iteration, func));
    }
    ptrainer->ValidateAcceptedPoint(x.getcontent(), cumulative_iteration);
    ptrainer->save_accepted_checkpoint(x);
}


// TODO: adopt a diagonal Hessian-based preconditioner
void TrainNonlinear::TrainNonLinearBFGS(int limit) {
    RequireCurrentParameterLayout();
    InvalidateCGResult();
    ValidationDeferral defer(defer_validation_);
    ValidationDeferral collective_errors(collective_evaluation_errors_);
    validation_coefficients_.clear();
    validation_current_ = false;
    bfgs_termination_type = 0;
    bfgs_iter = 0;
    bfgs_accepted_iter = 0;
    bfgs_initial_reported = false;
    Training::collective_try([&] {
        if (!p_pot || n_relax_coeffs <= 0 || x.size() != n_relax_coeffs)
            GPERROR("L-BFGS requires an initialized potential with nonlinear coefficients");
        if (bfgs_m <= 0 || bfgs_m > n_relax_coeffs)
            GPERROR("fit:bfgs_m must be between 1 and the nonlinear coefficient count");
        if (limit < 0)
            GPERROR("L-BFGS iteration limit must be nonnegative (0 means unlimited)");
        if (bfgs_iteration_offset < 0 || limit > std::numeric_limits<int>::max()-bfgs_iteration_offset)
            GPERROR("fit:bfgs_iteration_offset and iteration limit must fit a nonnegative int");
        if (save_every_it <= 0)
            GPERROR("fit:save_every must be positive");
        for (double tolerance : {bfgs_grad_tol, bfgs_step_tol, bfgs_conv_tol})
            if (!std::isfinite(tolerance) || tolerance < 0.)
                GPERROR("L-BFGS tolerances must be finite and nonnegative");
        ValidateCurvatureSettings();
        if (!std::isfinite(scaled_step_max) || scaled_step_max < 0.)
            GPERROR("fit:scaled_step_max must be finite and nonnegative");
        if (train_set_size_all == 0)
            GPERROR("L-BFGS requires a nonempty training set");
        p_pot->nl_get_coeffs(x.data());
        if (!x.allFinite()) GPERROR("L-BFGS initial coefficients must be finite");
    });
    int root_count = n_relax_coeffs;
    MPI_Bcast(&root_count, 1, MPI_INT, 0, mpi.comm);
    Training::collective_check(root_count == n_relax_coeffs ? "" :
        "L-BFGS coefficient counts differ between ranks");
    const Eigen::VectorXd initial_coeffs = x;
    alglib::real_1d_array alg_x, gradient;
    alglib::minlbfgsreport report;
    // Only rank 0 advances ALGLIB. Local solver failures are agreed before
    // broadcasting the next callback, so peers cannot enter a stale collective.
    auto solver_step = [&](auto &&operation) {
        Training::collective_try([&] {
            if (mpi.rank != 0) return;
            try { operation(); }
            catch (const alglib::ap_error &e) {
                throw std::runtime_error("L-BFGS solver: " + e.msg);
            }
        });
    };
    try {
        Training::collective_try([&] {
            alg_x.setcontent(n_relax_coeffs, x.data());
            gradient.setlength(n_relax_coeffs);
        });
        MPI_Bcast(alg_x.getcontent(), n_relax_coeffs, MPI_DOUBLE, 0, mpi.comm);
        Training::collective_try([&] { SetOptimizerCoeffs(alg_x.getcontent()); });
        solver_step([&] {
            Message("Create solver");
            alglib::minlbfgscreate(n_relax_coeffs, bfgs_m, alg_x, bfgsstate);
            alglib::minlbfgssetcond(bfgsstate, bfgs_grad_tol, bfgs_conv_tol, bfgs_step_tol, limit);
            alglib::minlbfgssetxrep(bfgsstate, true);
            PrepareParameterScale(alg_x.getcontent());
            alglib::minlbfgssetscale(bfgsstate,scale);
            if (bfgs_set_scale || parameter_scaling != "none")
                alglib::minlbfgssetprecscale(bfgsstate);
            if (scaled_step_max > 0.) alglib::minlbfgssetstpmax(bfgsstate, scaled_step_max);
        });
        gn_refreshes = 0;
        auto update_curvature = [&] {
            gn_diagonal = EstimateGNDiagonal();
            alglib::real_1d_array diagonal;
            solver_step([&] {
                diagonal.setlength(n_relax_coeffs);
                for (int i = 0; i < n_relax_coeffs; ++i)
                    diagonal[i] = !trainable_mask.empty() && !trainable_mask[i]
                        ? 1. : std::max(gn_floor,gn_diagonal[i]);
                alglib::minlbfgssetprecdiag(bfgsstate,diagonal);
            });
            ++gn_refreshes;
            Message(fmt::format("GN diagonal refresh {} at accepted iteration {}: probes={}, min={}, max={}, floor={}; data VJP + model-defined L2 gradient differences",
                gn_refreshes,bfgs_iteration_offset+bfgs_accepted_iter,gn_probes,
                gn_diagonal.minCoeff(),gn_diagonal.maxCoeff(),gn_floor));
        };
        if (curvature_precondition == "gn_diag") update_curvature();
        auto evaluate = [&](double &value) {
            Training::collective_try([&] { SetOptimizerCoeffs(alg_x.getcontent()); });
            AccumulateGard(value, gradient);
            Training::collective_try([&] {
                // Check before masking, including derivatives of frozen groups.
                if (!std::isfinite(value) || !Eigen::Map<const Eigen::VectorXd>(
                        gradient.getcontent(), n_relax_coeffs).allFinite())
                    GPERROR("L-BFGS encountered a non-finite objective or gradient");
                ApplyTrainableMask(gradient);
            });
        };
        evaluate(loss_f_prev);
        Message("BFGS start");
        Message(fmt::format("LBFGS coefficient-restart offset: {} (optimizer history reset)", bfgs_iteration_offset));
        Message(fmt::format("Initial Loss: {:.17g}", loss_f_prev));
        // Keep ALGLIB's public optimize API and callback protocol unchanged.
        // Workers service the root's callbacks; even an exception raised by
        // ALGLIB between callbacks can therefore send a stop event first.
        int refresh_requested = 0;
        int cycle_start = 0;
        struct Callbacks {
            std::function<void(const alglib::real_1d_array &, double &, alglib::real_1d_array &)> gradient;
            std::function<void(const alglib::real_1d_array &, double)> report;
        } callbacks;
        auto service = [&](int event, double &value) {
            MPI_Bcast(alg_x.getcontent(), n_relax_coeffs, MPI_DOUBLE, 0, mpi.comm);
            Training::collective_try([&] {
                if (!Eigen::Map<const Eigen::VectorXd>(alg_x.getcontent(), n_relax_coeffs).allFinite())
                    GPERROR("L-BFGS proposed non-finite coefficients");
            });
            if (event == 1) {
                evaluate(value);
                if (value < loss_f_prev) {
                    loss_f_prev = value;
                    display_iteration_log(value);
                    ++bfgs_iter;
                } else Message("  Loss: " + std::to_string(value));
            } else {
                MPI_Bcast(&value, 1, MPI_DOUBLE, 0, mpi.comm);
                // Validation handles its own collective phases. A rank-0-only
                // checkpoint failure must also reach every peer before stepping.
                Training::collective_try([&] { gnl_bfgsrep(alg_x, value, this); });
            }
        };
        callbacks.gradient = [&](const alglib::real_1d_array &point, double &value,
                                 alglib::real_1d_array &out_gradient) {
            alg_x = point;
            int event = 1;
            MPI_Bcast(&event, 1, MPI_INT, 0, mpi.comm);
            service(event, value);
            out_gradient = gradient;
        };
        callbacks.report = [&](const alglib::real_1d_array &point, double value) {
            alg_x = point;
            int event = 2;
            MPI_Bcast(&event, 1, MPI_INT, 0, mpi.comm);
            service(event, value);
            if (curvature_precondition == "gn_diag" && gn_update > 0
                && bfgs_accepted_iter > cycle_start && bfgs_accepted_iter % gn_update == 0
                && (limit == 0 || bfgs_accepted_iter < limit)) {
                refresh_requested = 1;
                alglib::minlbfgsrequesttermination(bfgsstate);
            }
        };
        for (;;) {
            refresh_requested = 0;
            cycle_start = bfgs_accepted_iter;
            std::string optimizer_error;
            if (mpi.rank == 0) {
                try {
                    alglib::minlbfgsoptimize(bfgsstate,
                        [](const alglib::real_1d_array &point, double &value,
                           alglib::real_1d_array &gradient, void *context) {
                            static_cast<Callbacks *>(context)->gradient(point, value, gradient);
                        },
                        [](const alglib::real_1d_array &point, double value, void *context) {
                            static_cast<Callbacks *>(context)->report(point, value);
                        }, &callbacks);
                } catch (const alglib::ap_error &e) {
                    optimizer_error = "L-BFGS solver: " + e.msg;
                } catch (const std::exception &e) {
                    optimizer_error = e.what();
                } catch (...) {
                    optimizer_error = "Unknown L-BFGS optimizer exception";
                }
                int event = 0;
                MPI_Bcast(&event, 1, MPI_INT, 0, mpi.comm);
            } else {
                for (;;) {
                    int event = 0;
                    MPI_Bcast(&event, 1, MPI_INT, 0, mpi.comm);
                    if (event == 0) break;
                    double value = 0.;
                    try { service(event, value); }
                    catch (const std::exception &e) { optimizer_error = e.what(); }
                    catch (...) { optimizer_error = "Unknown L-BFGS worker exception"; }
                    // A callback failure is collective. Wait for root to unwind
                    // optimize and send stop, rather than entering another reduction.
                }
            }
            Training::collective_check(optimizer_error);
            solver_step([&] {
                alglib::minlbfgsresults(bfgsstate, alg_x, report);
                bfgs_termination_type = static_cast<int>(report.terminationtype);
                Message(fmt::format("LBFGS report: iterations={}, gradient_evaluations={}, termination_type={}",
                                    report.iterationscount, report.nfev, report.terminationtype));
            });
            MPI_Bcast(&bfgs_termination_type, 1, MPI_INT, 0, mpi.comm);
            MPI_Bcast(alg_x.getcontent(), n_relax_coeffs, MPI_DOUBLE, 0, mpi.comm);
            MPI_Bcast(&refresh_requested,1,MPI_INT,0,mpi.comm);
            if (!refresh_requested || bfgs_termination_type != 8) break;
            Training::collective_try([&] { SetOptimizerCoeffs(alg_x.getcontent()); });
            update_curvature();
            solver_step([&] {
                alglib::minlbfgsrestartfrom(bfgsstate,alg_x);
                alglib::minlbfgssetcond(bfgsstate,bfgs_grad_tol,bfgs_conv_tol,bfgs_step_tol,
                    limit == 0 ? 0 : limit-bfgs_accepted_iter);
            });
            bfgs_initial_reported = false;
            Message(fmt::format("Curvature history restart at accepted iteration {}; remaining budget={}",
                bfgs_iteration_offset+bfgs_accepted_iter,limit == 0 ? 0 : limit-bfgs_accepted_iter));
        }
        Training::collective_try([&] {
            if (!Eigen::Map<const Eigen::VectorXd>(alg_x.getcontent(), n_relax_coeffs).allFinite())
                GPERROR("L-BFGS returned non-finite coefficients");
            switch (bfgs_termination_type) {
            case 1: Message("Relative function improvement is no more than EpsF"); break;
            case 2: Message("Relative step is no more than EpsX, stop optimization"); break;
            case 4: Message("Gradient norm is no more than EpsG, stop optimization"); break;
            case 5: Warning("Max iteration limit reached; retaining the finite accepted point (not converged)"); break;
            case -2: Warning("Rounding errors prevent improvement; retaining the finite accepted point (not converged)"); break;
            case 7: Warning("Stopping conditions are too stringent; retaining the finite accepted point (not converged)"); break;
            case 8: Message("Optimization stopped by callback; retaining the finite accepted point"); break;
            default: GPERROR("L-BFGS failed with termination type " + std::to_string(bfgs_termination_type));
            }
            SetOptimizerCoeffs(alg_x.getcontent());
            p_pot->nl_get_coeffs(x.data());
        });
        Training::collective_try([&] {
            ValidateAcceptedPoint(x.data(), bfgs_iteration_offset + bfgs_accepted_iter, true);
        });
    } catch (...) {
        // Failed line-search probes must not leak into the caller's live model.
        // Already written checkpoints contain only previously accepted points.
        Training::collective_try([&] { p_pot->nl_set_coeffs(initial_coeffs.data()); });
        x = initial_coeffs;
        validation_current_ = false;
        validation_coefficients_.clear();
        bfgs_termination_type = 0;
        throw;
    }
}

void TrainNonlinear::TrainBFGS() {
    TrainScheduled("bfgs");
}

void TrainNonlinear::ValidateLinearRefitOnly(const Potentials::AnyPotential& potential) const {
    if(linear_refit_settings.mode=="off") GPERROR("--linear-refit-only requires fit:linear_refit");
    if(random_init || force_random_init || apply_disp || training_schedule!="single")
        GPERROR("--linear-refit-only requires initialized coefficients, no randomization and training_schedule=single");
    // inited() is historically non-const in AnyPotential.
    if(!const_cast<Potentials::AnyPotential&>(potential).inited())
        GPERROR("--linear-refit-only requires an initialized input model");
}

void TrainNonlinear::ValidateLinearRefitSettings(bool respect_mask) const {
    if(linear_refit_settings.mode=="off") return;
    if(training_device!="cpu") GPERROR("Linear refit requires CPU observations");
    if(linear_refit_settings.solver!="qr_svd") GPERROR("Unknown linear refit solver");
    if(!std::isfinite(linear_refit_settings.rtol) || linear_refit_settings.rtol<0. || linear_refit_settings.rtol>=1.
        || !std::isfinite(linear_refit_settings.atol) || linear_refit_settings.atol<0.
        || !std::isfinite(linear_refit_settings.acceptance_tolerance) || linear_refit_settings.acceptance_tolerance<=0.)
        GPERROR("Invalid linear refit tolerances");
    if(!p_pot) return; // CLI option validation precedes model construction.
    Training::ConditionalLinearAdapter selected(*p_pot,
        Training::conditional_linear_selection(*p_pot,linear_refit_settings.mode),weight_penatly);
    if((selected.l1.array()!=0.).any())
        GPERROR("Linear refit does not support nonzero selected L1 regularization");
    if(!selected.l2.allFinite() || selected.l2.minCoeff()<0.)
        GPERROR("Invalid linear refit L2 regularization");
    for(int index:selected.indices)
        if(respect_mask && !trainable_mask.empty() && !trainable_mask[index])
            GPERROR("Linear refit selects a frozen nonlinear parameter");
}

void TrainNonlinear::TrainLinearRefit() {
    TrainLinearRefitImpl(false);
}

void TrainNonlinear::TrainLinearRefitImpl(bool validate_final) {
    RequireCurrentParameterLayout();
    if(linear_refit_settings.mode=="off") return;
    ValidationDeferral active(cg_active_, validate_final);
    ValidationDeferral collective_errors(collective_evaluation_errors_, validate_final);
    std::unique_ptr<Training::ConditionalLinearAdapter> selection;
    Training::collective_try([&] {
        ValidateLinearRefitSettings(validate_final);
        if(!p_pot) GPERROR("Linear refit requires an initialized trainer");
        selection=std::make_unique<Training::ConditionalLinearAdapter>(*p_pot,
            Training::conditional_linear_selection(*p_pot,linear_refit_settings.mode),weight_penatly);
    });
    auto &selected=*selection;
    const auto original=selected.get();
    const Atoms *sample=nullptr;
    for(const auto &atom:training_set_) if(atom.size()) {sample=&atom;break;}
    try {
        Training::collective_try([&] {
            if(mpi.rank==0) Training::publish_conditional_linear(*p_pot,potfilesave+".pre-refit",sample);
        });
        auto observation_settings=linear_refit_settings;
        observation_settings.hvp=hvp_fit;observation_settings.hessian=hessian_fit;
        if(weight_charge>0.) {
            observation_settings.charge_weight=[&](const Atoms &atom){return charge_weight(atom);};
            observation_settings.charge_target=[&](const Atoms &atom){return total_charge_target(atom);};
        }
        linear_refit(*p_pot,training_set_,trainint_set_nbhs_,efs_weights(),weight_penatly,observation_settings);
        validation_current_=false;
        validation_coefficients_.clear();
        if(validate_final) EvaluateValidation();
        Training::collective_try([&] {
            if(mpi.rank==0) Training::publish_conditional_linear(*p_pot,potfilesave,sample);
        });
        p_pot->nl_get_coeffs(x.data());
        ConfigureTrainableGroups(trainable_groups);
        // A terminal refit is no longer the exact CG endpoint. Preserve the
        // historical termination reason, but revoke its publication token.
        cg_result_potential_=nullptr;
        cg_result_coefficients_.clear();
    } catch(...) {
        Training::collective_try([&] {selected.set(original);p_pot->nl_get_coeffs(x.data());});
        validation_current_=false;
        validation_coefficients_.clear();
        throw;
    }
}

void TrainNonlinear::SaveConditionalResult(const std::string &path) {
    RequireCurrentParameterLayout();
    Training::collective_try([&] {
        if(!p_pot || n_relax_coeffs<=0 || p_pot->get_NL_coeffs_count()!=n_relax_coeffs
            || !p_pot->inited() || p_pot->conditional_linear_blocks(weight_penatly).empty())
            GPERROR("Conditional publication requires an initialized supported model");
        Eigen::VectorXd coefficients(n_relax_coeffs);p_pot->nl_get_coeffs(coefficients.data());
        if(!coefficients.allFinite()) GPERROR("Cannot publish nonfinite conditional model coefficients");
    });
    const Atoms *sample=nullptr;
    for(const auto &atom:training_set_) if(atom.size()) {sample=&atom;break;}
    Training::collective_try([&] {
        if(mpi.rank==0) Training::publish_conditional_linear(*p_pot,path,sample);
    });
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
void TrainNonlinear::TrainSGD() {
    TrainScheduled("sgd");
}

void TrainNonlinear::TrainNonLinearSGD(int epochs) {
    RequireCurrentParameterLayout();
    InvalidateCGResult();
    ValidationDeferral collective_errors(collective_evaluation_errors_);
    Training::collective_try([&] {
        if (epochs < 0 || save_every_it <= 0 || batches_per_epoch <= 0 || batch_size_core <= 0)
            GPERROR("SGD requires nonnegative epochs and positive save/batch counts");
        if (epochs > std::numeric_limits<int>::max() / batches_per_epoch)
            GPERROR("SGD epoch and batch counts exceed the iteration range");
        if (curvature_precondition != "none")
            GPERROR("curvature_precondition requires ordinary L-BFGS or CG");
        for (double value : {learning_rate, momentum, clip_gradient})
            if (!std::isfinite(value) || value < 0.)
                GPERROR("SGD learning rate, momentum and gradient clipping must be finite and nonnegative");
        // No scheduler step is performed for a zero-epoch run.
        sgd_scheduler = Training::SGDScheduler::create_with_warmup(
            sgd_scheduler_type, learning_rate, sgd_final_lr,
            std::max(1, epochs * batches_per_epoch), sgd_decay_rate, sgd_decay_steps,
            sgd_warmup_steps, sgd_warmup_lr);
    });
    if (epochs == 0) return;
    bfgs_iter = 0;
    for (int iepoch = 0; iepoch < epochs; ++iepoch) {
        Message("Epoch: " + std::to_string(iepoch));
        ReBalance();
        int icurrent = 0;
        for (int ibatch = 0; ibatch < batches_per_epoch; ++ibatch) {
            // All ranks keep the aligned batch cadence, but only real local
            // samples contribute observations or the global averaging count.
            int this_batch = std::max(0,std::min(batch_size_core,
                                      static_cast<int>(train_set_size) - icurrent));
            Message(" batch: " + std::to_string(ibatch));
            loss = 0;
            memset(loss_gard.data(), 0, n_relax_coeffs * sizeof(double));
            std::vector<int> current_indices;
            current_indices.reserve(this_batch);
            for (int i = 0; i < this_batch; ++i) {
                int id = train_set_inds_aligned[icurrent];
                ++icurrent;
                current_indices.push_back(id);
            }
#ifdef IMRMLP_GPU
            if (training_device != "cpu") {
                Training::collective_try([&] {AccumulateGpuBatch(current_indices, loss);});
            } else
#endif
            {
                Training::collective_try([&] {
                    for (int id : current_indices) AddLossGard(id, loss);
                });
            }
            // AddPenaltyGard();
            Training::collective_try([&] {
                if (!std::isfinite(loss) || !loss_gard.allFinite())
                    GPERROR("SGD encountered a non-finite objective or gradient");
            });
            SGDUpdate(this_batch);
            Training::collective_try([&] {
                if (!x.allFinite()) GPERROR("SGD produced non-finite coefficients");
            });
            Message(" loss: " + std::to_string(loss_mpi));
            ++bfgs_iter;
        }
        if (iepoch % save_every_it == 0)
            Training::collective_try([&] { save_current(potfilesave + std::to_string(iepoch)); });
    }

    // postprocessing
    Training::collective_try([&] { save_current(potfilesave); });
    sgd_pretrain = true;
}

TrainNonlinear::~TrainNonlinear() {
    // debug_guard_ (if set) restores global_debugstream and deletes ofstream automatically
}

void TrainNonlinear::AddChargeLoss(Atoms &prediction, const Atoms &reference,
    AnyNeighborhoods &nbhs, double &accumulated_loss, double *gradient,
    ChargeErrorMonitor &monitor) {
    if (!p_pot->supports_total_charge_training())
        return;
    const double predicted_total = p_pot->predicted_total_charge(prediction);
    const double target = total_charge_target(reference);
    const double error = predicted_total - target;
    const double weight = charge_weight(reference);
    const double qloss = weight * error * error;
    accumulated_loss += qloss;
    monitor.add(prediction, predicted_total, target, qloss);
    if (gradient && weight > 0.0)
        p_pot->AccumulateTotalChargeGard(prediction, nbhs,
                                         2.0 * weight * error, gradient);
}

void TrainNonlinear::AddLossGard(int id, double &_loss) {
    RequireCurrentParameterLayout();
    Atoms &atom = training_set_[id];
    Neighborhoods &nbhs = trainint_set_nbhs_[id];
    if (atom.size() == 0)
        return;

    Atoms &atomn = efs_set_[id];
    p_pot->getEFS(atomn, nbhs);
    if (cg_active_) ValidateCGPrediction(atomn, atom, false);

    AddChargeLoss(atomn, atom, nbhs, _loss, loss_gard.data(), qmonitor);

    double dLdE = 0;
    dLdF.resize(atom.size(), 3);

    const double cc = efs_weights().normalization(atom);

    // add to loss, energy in the last to reduce round-off errors
    if (weight_force > 0 && atom.has_forces()) [[likely]] {
        double weight_force_i = force_weight(atom) * cc;
        for (int i = 0; i < atom.size(); ++i) {
            const double weight_force_iat =
                EFSFitWeights::atomic_weight(atom,i,weight_force_i);
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
        dLdS = 2 * weight_stress_i * (atomn.stress_ - atom.stress_); // 
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

    if (weight_force > 0 || weight_stress > 0) {
        p_pot->AccumulateEFSCombinationGard(atomn, nbhs, dLdE, dLdF, dLdS, fweights, loss_gard.data());
    }
    else if (dLdE != 0.0) {
        p_pot->AccumulateEFSCombinationGard(atomn, nbhs, dLdE, loss_gard.data());
    }
    AddHVPLoss(atom, cc, _loss, true, emonitor);
    AddHessianLoss(atom, cc, _loss, true, emonitor);
}

void TrainNonlinear::AddLoss(int id, double &_loss) {
    RequireCurrentParameterLayout();
    Atoms &atom = training_set_[id];
    Neighborhoods &nbhs = trainint_set_nbhs_[id];
    if (atom.size() == 0)
        return;

    Atoms &atomn = efs_set_[id];
    p_pot->getEFS(atomn, nbhs);

    AddChargeLoss(atomn, atom, nbhs, _loss, nullptr, qmonitor);

    const double cc = efs_weights().normalization(atom);

    // add to loss, energy in the last to reduce round-off errors
    // add a atomic weight
    if (weight_force > 0 && atom.has_forces()) [[likely]] {
        double weight_force_i = force_weight(atom) * cc;
        for (int i = 0; i < atom.size(); ++i) {
            const double weight_force_iat =
                EFSFitWeights::atomic_weight(atom,i,weight_force_i);
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
    AddHessianLoss(atom, cc, _loss, false, emonitor);
}


void TrainNonlinear::ReBalance() {
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

void TrainNonlinear::InitPenaltyGard() {
    RequireCurrentParameterLayout();
    // pgard.setlength(p_pot->get_coeff_size());
    ploss = 0;
    for (int i = 0; i < n_relax_coeffs; ++i)
        pgard[i] = 0;
    p_pot->AddPenaltyGard(weight_penatly, ploss, pgard.data());
    if (cg_active_ && (!std::isfinite(ploss) || !pgard.allFinite()))
        GPERROR("CG encountered a non-finite local penalty or penalty gradient");
    // AddPenaltyGard above selects and accumulates the model-specific regularization.
    // for (int i = 0; i < n_relax_coeffs; ++i) {
    //     ploss += reg_L1vec(i) * fabs(x[i]);
    //     ploss += reg_L2vec(i) * x[i] * x[i];
    //     if (x[i] > 0) pgard[i] -= reg_L1vec[i];
    //     else if (x[i] < 0) pgard[i] += reg_L1vec[i];
    //     pgard[i] += 2 * x[i] * reg_L2vec[i];
    // }
}

TrainNonlinear::TrainNonlinear() {
    InitSettings();
}

void TrainNonlinear::SGDUpdate(int local_batch_size) {
    // Compute global batch size (total samples across all MPI ranks)
    int global_batch_size = local_batch_size;
#ifdef IMRMLP_MPI
    MPI_Allreduce(&local_batch_size, &global_batch_size, 1, MPI_INT, MPI_SUM, mpi.comm);
#endif
    // Every rank takes this branch together. A globally empty batch has no
    // update, including no momentum drift or regularization-only step.
    if (global_batch_size == 0) {
        loss_mpi = 0.;
        loss_gard_mpi.setZero();
        return;
    }

    // sum all loss and penalty, and broadcast
    loss_mpi = 0;
    memset(loss_gard_mpi.data(), 0, n_relax_coeffs * sizeof(double));
    MPI_Reduce(&loss,
               &loss_mpi,
               1,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               mpi.comm);
    MPI_Reduce(loss_gard.data(),
               loss_gard_mpi.data(),
               n_relax_coeffs,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               mpi.comm);
    // update coeffs
    if (mpi.rank == 0) {
        int soffset = 0;
        double inv_batch = 1.0 / static_cast<double>(global_batch_size);
        for (int i = soffset; i < n_relax_coeffs; ++i) {
            loss_gard_mpi[i] *= inv_batch;
            loss_gard_mpi[i] += pgard[i];
            if (!trainable_mask.empty() && !trainable_mask[i])
                loss_gard_mpi[i] = 0.0;
        }
        // Gradient clipping
        if (clip_gradient > 0.0) {
            for (int i = soffset; i < n_relax_coeffs; ++i) {
                loss_gard_mpi[i] = std::max(-clip_gradient,
                                  std::min(loss_gard_mpi[i], clip_gradient));
            }
        }
        p_pot->nl_get_coeffs(x.data());
        double current_lr = learning_rate;
        if (sgd_scheduler) {
            sgd_scheduler->on_step(bfgs_iter);
            current_lr = sgd_scheduler->get_lr();
        }
        for (int i = 0; i < n_relax_coeffs; ++i) {
            if (!trainable_mask.empty() && !trainable_mask[i]) {
                update_vector[i] = update_vector_last[i] = 0.;
                continue;
            }
            update_vector[i] = current_lr * loss_gard_mpi[i] + momentum * update_vector_last[i];
            x[i] -= update_vector[i];
            update_vector_last[i] = update_vector[i];
        }
    }
    // Broadcast
#ifdef IMRMLP_MPI
    MPI_Bcast(x.data(),
              n_relax_coeffs,
              MPI_DOUBLE,
              0,
              mpi.comm);
#endif
    Training::collective_try([&] {
        p_pot->nl_set_coeffs(x.data());
        InitPenaltyGard();
    });
}

void TrainNonlinear::InvalidateCGResult() {
    cg_termination_type = 0;
    cg_result_potential_ = nullptr;
    cg_result_coefficients_.clear();
}

void TrainNonlinear::ValidateCGSettingsImpl(int limit, bool terminal_refit) const {
    if (training_device != "cpu") {
        if (!MLIPCore::GpuDirect::valid_device(training_device)
            || !MLIPCore::GpuDirect::device_supported(training_device))
            GPERROR("CG requested an unavailable GPU backend");
        if (weight_charge > 0. || hvp_fit.weight > 0. || hessian_fit.weight > 0.)
            GPERROR("GPU CG supports E/F/S objectives; charge, HVP and full-Hessian losses require CPU");
#ifdef IMRMLP_GPU
        if (p_pot && (!gpu_evaluator_ || !gpu_evaluator_->initialized()))
            GPERROR("Reinitialize the trainer before switching CG to a GPU backend");
#endif
    }
    hvp_fit.validate();
    validate_hessian_settings(hessian_fit);
    if (!std::isfinite(weight_charge) || weight_charge < 0. ||
        !std::isfinite(weight_charge_power) || !std::isfinite(charge_default_target))
        GPERROR("CG requires finite charge weight, power and default target, with charge_weight >= 0");
    if (linear_refit_settings.mode != "off") {
        if(!terminal_refit) GPERROR("Direct CG requires fit:linear_refit=off; use TrainCG for terminal refit");
        if(training_schedule!="single") GPERROR("CG terminal refit requires training_schedule=single");
        ValidateLinearRefitSettings(true);
    }
    if (objective_convention != "legacy" || l2_reference != "native")
        GPERROR("CG requires objective-convention=legacy and l2-reference=native");
    ValidateCurvatureSettings();
    if (!std::isfinite(scaled_step_max) || scaled_step_max < 0.)
        GPERROR("fit:scaled_step_max must be finite and nonnegative");
    if (limit <= 0) GPERROR("CG iteration limit (bfgs_maxiteration or it_limit) must be positive (unlimited runs are unsupported)");
    if (bfgs_iteration_offset < 0 || limit > std::numeric_limits<int>::max()-bfgs_iteration_offset)
        GPERROR("CG iteration offset and limit must fit a nonnegative int");
    if (save_every_it <= 0 || validation_interval <= 0)
        GPERROR("CG save_every and validation_interval must be positive");
    for (double value : {bfgs_grad_tol, bfgs_conv_tol, bfgs_step_tol, reg_L1, reg_L2, weight_penatly})
        if (!std::isfinite(value) || value < 0.)
            GPERROR("CG tolerances and regularization weights must be finite and nonnegative");
    Training::EFSObjective::validate_weights(efs_weights());
    if (!(weight_eng > 0. || weight_force > 0. || weight_stress > 0. || hvp_fit.weight > 0. || hessian_fit.weight > 0. || weight_charge > 0.))
        GPERROR("CG requires a positive E/F/S, HVP, full-Hessian or total-charge weight");
    if (potfilesave.empty()) GPERROR("CG save_to must not be empty");
}

void TrainNonlinear::ValidateCGSettings() const {
    if (training_schedule != "single"
        && training_schedule != "mtp_les_from_scratch" && training_schedule != "mtp_les_resume"
        && training_schedule != "mtp_sog_from_scratch" && training_schedule != "mtp_sog_resume")
        GPERROR("Unknown fit:training_schedule: " + training_schedule);
    ValidateCGSettingsImpl(maxiter, true);
}

void TrainNonlinear::ValidateCGData() const {
    Training::collective_try([&] {
        if (weight_charge > 0. && !p_pot->supports_total_charge_training())
            GPERROR("CG charge_weight requires analytic total-charge training support");
    });
    auto validate = [&](const std::vector<Atoms> &data,
                        const std::vector<Neighborhoods> &neighbors, bool validation) {
        if (data.size() != neighbors.size()) GPERROR("CG data/neighborhood counts differ");
        bool supervised = false;
        for (const auto &a : data) {
            if (a.size() == 0 || a.chemmap_.size() != a.pos_.size()
                || !a.cell_.allFinite() || !std::isfinite(a.cell_.determinant())
                || a.cell_.determinant() == 0.)
                GPERROR("CG requires nonempty finite structures with valid species and cells");
            for (const auto &position : a.pos_)
                if (!position.allFinite()) GPERROR("CG positions must be finite");
            for (int species : a.chemmap_)
                if (species < 0 || (p_pot->species_count > 0 && species >= p_pot->species_count))
                    GPERROR("CG species index is outside the potential's species range");
            if (!std::isfinite(a.fitness) || a.fitness < 0.) GPERROR("CG fitness must be finite and nonnegative");
            if (!a.atomic_fitness.empty() && a.atomic_fitness.size() != a.pos_.size())
                GPERROR("CG atomic fitness count differs from atom count");
            for (double weight : a.atomic_fitness)
                if (!std::isfinite(weight) || weight < 0.) GPERROR("CG atomic fitness must be finite and nonnegative");
            if (a.has_energy() && !std::isfinite(a.energy_)) GPERROR("CG reference energy must be finite");
            if (a.has_forces()) {
                if (a.forces_.size() != a.pos_.size()) GPERROR("CG reference force count differs from atom count");
                for (const auto &force : a.forces_)
                    if (!force.allFinite()) GPERROR("CG reference forces must be finite");
            }
            if (a.has_stresses() && !a.stress_.allFinite()) GPERROR("CG reference stress must be finite");
            const auto weights = efs_weights();
            const double normalization = weights.normalization(a);
            if (!std::isfinite(normalization) || normalization <= 0.)
                GPERROR("CG effective force normalization must be finite and positive");
            auto positive = [&](double weight) {
                weight *= normalization;
                if (!std::isfinite(weight) || weight < 0.) GPERROR("CG effective weights must be finite and nonnegative");
                return weight > 0.;
            };
            if (p_pot->supports_total_charge_training()) {
                const double target = total_charge_target(a);
                const double weight = charge_weight(a);
                if (!std::isfinite(target) || !std::isfinite(weight) || weight < 0.)
                    GPERROR("CG total-charge targets and effective weights must be finite, with nonnegative weights");
                if (weight > 0.) supervised = true;
            }
            if (hessian_fit.weight > 0. && a.have_hessian) {
                validate_hessian_label(a);
                if (!a.cartesian) GPERROR("CG full Hessian fitting requires Cartesian positions");
                if (positive(hessian_fit.weight * a.fitness / std::pow(a.size(), hessian_fit.power))) supervised = true;
            }
            if (hvp_fit.weight > 0. && hvp_fit.observation_weight(a, normalization) > 0.) {
                if (!a.cartesian) GPERROR("CG HVP fitting requires Cartesian positions");
                supervised = true;
            }
            if (a.has_energy() && positive(weights.energy_weight(a))) supervised = true;
            if (a.has_stresses() && positive(weights.stress_weight(a))) supervised = true;
            if (a.has_forces()) {
                const double weight = weights.force_weight(a);
                for (int i = 0; i < a.size(); ++i)
                    if (positive(EFSFitWeights::atomic_weight(a, i, weight))) supervised = true;
            }
        }
        return validation || supervised;
    };
    // A rank may legitimately own an empty shard. Agree on supervision only
    // after every rank has finished all local validation without exceptions.
    int supervised = 0;
    Training::collective_try([&] {
        supervised = validate(training_set_, trainint_set_nbhs_, false);
        validate(valid_set_, valid_set_nbhs_, true);
    });
#ifdef IMRMLP_MPI
    MPI_Allreduce(MPI_IN_PLACE, &supervised, 1, MPI_INT, MPI_MAX, mpi.comm);
#endif
    if (!supervised) GPERROR("CG requires effective E/F/S, HVP, full-Hessian or total-charge supervision");
}

void TrainNonlinear::ValidateCGPrediction(const Atoms &prediction, const Atoms &reference, bool validation) const {
    if (p_pot->supports_total_charge_training() &&
        !std::isfinite(p_pot->predicted_total_charge(prediction)))
        GPERROR("CG encountered non-finite predicted total charge");
    if (reference.has_energy() && (validation || weight_eng > 0.)
        && (!prediction.has_energy() || !std::isfinite(prediction.energy_)))
        GPERROR("CG encountered non-finite or missing predicted energy");
    if (reference.has_forces() && (validation || weight_force > 0.)) {
        if (!prediction.has_forces() || prediction.forces_.size() != reference.pos_.size())
            GPERROR("CG predicted force count differs from atom count");
        for (const auto &force : prediction.forces_)
            if (!force.allFinite()) GPERROR("CG encountered non-finite predicted forces");
    }
    if (reference.has_stresses() && (validation || weight_stress > 0.)
        && (!prediction.has_stresses() || !prediction.stress_.allFinite()))
        GPERROR("CG encountered non-finite or missing predicted stress");
}

void TrainNonlinear::SaveCGPoint(const std::string &path) {
    if (mpi.rank != 0) return;
    namespace fs = std::filesystem;
    const fs::path destination(path);
    if (destination.empty() || destination.filename().empty()) GPERROR("CG output path must name a file");
    auto check_destination = [&] {
        std::error_code error;
        const auto status = fs::symlink_status(destination, error);
        if (error && error != std::errc::no_such_file_or_directory)
            GPERROR("Cannot inspect CG output: " + path + ": " + error.message());
        if (status.type() != fs::file_type::not_found && !fs::is_regular_file(status))
            GPERROR("CG output must be a regular file or a new path: " + path);
    };
    check_destination();
    fs::path parent = destination.parent_path();
    if (parent.empty()) parent = ".";
    std::string pattern = (parent / ".imrmlp-cg-save-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
    if (!::mkdtemp(name.data())) GPERROR("Cannot create CG staging directory beside: " + path);
    struct Stage {
        fs::path path;
        ~Stage() { std::error_code error; fs::remove_all(path, error); }
    } stage{fs::path(name.data())};
    // Preserve the entire filename, including compression suffixes. Save()
    // implementations with sidecars are not part of CG's publication contract.
    const fs::path temporary = stage.path / destination.filename();
    p_pot->Save(temporary.string());
    std::size_t entries = 0;
    for (const auto &entry : fs::directory_iterator(stage.path)) {
        ++entries;
        if (entry.path() != temporary || !fs::is_regular_file(entry.symlink_status()))
            GPERROR("CG supports only a single-file potential serializer");
    }
    if (entries != 1 || fs::file_size(temporary) == 0)
        GPERROR("CG potential serializer did not produce a nonempty regular file");
    check_destination();
    fs::rename(temporary, destination);
}

void TrainNonlinear::SaveCGResult(const std::string &path) {
    RequireCurrentParameterLayout();
    Training::collective_try([&] {
        if (!p_pot || cg_termination_type == 0 || p_pot != cg_result_potential_
            || n_relax_coeffs <= 0 || p_pot->get_NL_coeffs_count() != n_relax_coeffs
            || cg_result_coefficients_.size() != static_cast<std::size_t>(n_relax_coeffs))
            GPERROR("CG has no completed result to save");
        Eigen::VectorXd current(n_relax_coeffs);
        p_pot->nl_get_coeffs(current.data());
        if (!current.allFinite() || current != Eigen::Map<const Eigen::VectorXd>(
                cg_result_coefficients_.data(), n_relax_coeffs))
            GPERROR("CG result coefficients changed after optimization");
    });
    Training::collective_try([&] { SaveCGPoint(path); });
}

void TrainNonlinear::TrainCG() {
    RequireCurrentParameterLayout();
#ifdef IMRMLP_MPI
    // Check the original mode before either branching or hiding it from the
    // pure optimizer's existing all-settings agreement below.
    int mode_size=0;
    Training::collective_try([&] {
        if(linear_refit_settings.mode.size()>static_cast<size_t>(std::numeric_limits<int>::max()))
            GPERROR("CG refit mode exceeds the MPI message limit");
        mode_size=static_cast<int>(linear_refit_settings.mode.size());
    });
    MPI_Bcast(&mode_size,1,MPI_INT,0,mpi.comm);
    std::string root_mode;
    Training::collective_try([&] {root_mode=mpi.rank==0 ? linear_refit_settings.mode : std::string(mode_size,'\0');});
    MPI_Bcast(root_mode.data(),mode_size,MPI_CHAR,0,mpi.comm);
    Training::collective_check(root_mode==linear_refit_settings.mode ? "" : "CG terminal refit modes differ between ranks");
#endif
    if(linear_refit_settings.mode=="off") {TrainScheduled("cg");return;}
    Training::collective_try([&] {
        ConfigureTrainableGroups(trainable_groups);
        ValidateCGSettings();
    });
    const auto mode=linear_refit_settings.mode;
    const auto destination=potfilesave;
    // The low-level optimizer stays pure CG. Its accepted checkpoints and final
    // endpoint cannot overwrite the destination of the conditional result.
    linear_refit_settings.mode="off";
    potfilesave=destination+".cg.snapshot";
    try {
        TrainScheduled("cg");
    } catch(...) {
        linear_refit_settings.mode=mode;potfilesave=destination;
        throw;
    }
    linear_refit_settings.mode=mode;potfilesave=destination;
    TrainLinearRefitImpl(true);
}

double TrainNonlinear::operator()(const Eigen::VectorXd &x_, Eigen::VectorXd &g_in) {
    // scaling back
    x_in = x_;
    for (int i = 0; i < x_in.size(); ++i) {
        x_in[i] /= scaling_coeffs[i];
    }
    p_pot->nl_set_coeffs(x_in.data());
    AccumulateGard(loss_mpi, loss_gard_mpi);

    #ifdef IMRMLP_MPI
    MPI_Bcast(&loss_mpi, 1, MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(loss_gard_mpi.data(), loss_gard_mpi.size(), MPI_DOUBLE, 0, mpi.comm);
    #endif
    
        memcpy(g_in.data(), loss_gard_mpi.data(), n_relax_coeffs * sizeof(double));

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

void TrainNonlinear::TrainNonLinearCG(int limit) {
    RequireCurrentParameterLayout();
    ValidationDeferral active(cg_active_);
    ValidationDeferral defer(defer_validation_);
    ValidationDeferral collective_errors(collective_evaluation_errors_);
    validation_coefficients_.clear();
    validation_current_ = false;
    InvalidateCGResult();
    validation_evaluations = 0;
    loss_f_prev = std::numeric_limits<double>::infinity();
    bfgs_iter = 0;
    bfgs_accepted_iter = 0;
    bfgs_initial_reported = false;
    Training::collective_try([&] {
        if (!p_pot || n_relax_coeffs <= 0 || x.size() != n_relax_coeffs
            || p_pot->get_NL_coeffs_count() != n_relax_coeffs)
            GPERROR("CG requires an initialized potential with unchanged nonlinear coefficient count");
        if (training_schedule != "single")
            GPERROR("Direct CG execution requires training_schedule=single; use TrainCG for staged training");
        ValidateCGSettingsImpl(limit);
        if (train_set_size_all == 0) GPERROR("CG requires a nonempty training set");
        ConfigureTrainableGroups(trainable_groups);
        p_pot->nl_get_coeffs(x.data());
        if (!x.allFinite()) GPERROR("CG initial coefficients must be finite");
    });
#ifdef IMRMLP_MPI
    // Accepted-point validation contains collectives; its cadence and all
    // other controls must agree before any rank can enter an optimizer phase.
    std::string local_settings;
    Training::collective_try([&] {
        std::ostringstream out;
        out << "it_limit=" << limit << '\n';
        for (const auto &[key, value] : CurrentSettings().settings)
            out << key << '=' << value << '\n';
        local_settings = out.str();
        if (local_settings.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            GPERROR("CG settings exceed the MPI message limit");
    });
    int settings_size = static_cast<int>(local_settings.size());
    MPI_Bcast(&settings_size, 1, MPI_INT, 0, mpi.comm);
    std::string root_settings;
    Training::collective_try([&] {
        root_settings = mpi.rank == 0 ? local_settings : std::string(settings_size, '\0');
    });
    MPI_Bcast(root_settings.data(), settings_size, MPI_CHAR, 0, mpi.comm);
    Training::collective_check(root_settings == local_settings ? "" : "CG settings differ between ranks");
#endif
    ValidateCGData();
    int root_count = n_relax_coeffs;
    MPI_Bcast(&root_count, 1, MPI_INT, 0, mpi.comm);
    Training::collective_check(root_count == n_relax_coeffs ? "" :
        "CG coefficient counts differ between ranks");
    const Eigen::VectorXd initial_coeffs = x;
    alglib::real_1d_array alg_x, gradient;
    alglib::mincgreport report;
    // Only rank 0 advances ALGLIB. Local solver failures are agreed before
    // broadcasting the next callback, so peers cannot enter a stale collective.
    auto solver_step = [&](auto &&operation) {
        Training::collective_try([&] {
            if (mpi.rank != 0) return;
            try { operation(); }
            catch (const alglib::ap_error &e) {
                throw std::runtime_error("CG solver: " + e.msg);
            }
        });
    };
    try {
        Training::collective_try([&] {
            alg_x.setcontent(n_relax_coeffs, x.data());
            gradient.setlength(n_relax_coeffs);
        });
        MPI_Bcast(alg_x.getcontent(), n_relax_coeffs, MPI_DOUBLE, 0, mpi.comm);
        Training::collective_try([&] {
            // Synchronize the root's starting point before capturing frozen
            // groups, including independently randomized initial potentials.
            p_pot->nl_set_coeffs(alg_x.getcontent());
            ConfigureTrainableGroups(trainable_groups);
        });
        solver_step([&] {
            Message("Create solver");
            alglib::mincgcreate(n_relax_coeffs, alg_x, cgstate);
            alglib::mincgsetcond(cgstate, bfgs_grad_tol, bfgs_conv_tol, bfgs_step_tol, limit);
            alglib::mincgsetxrep(cgstate, true);
            PrepareParameterScale(alg_x.getcontent());
            alglib::mincgsetscale(cgstate,scale);
            if (bfgs_set_scale || parameter_scaling != "none")
                alglib::mincgsetprecscale(cgstate);
            if (scaled_step_max > 0.) alglib::mincgsetstpmax(cgstate,scaled_step_max);

        });
        gn_refreshes = 0;
        auto update_curvature = [&] {
            gn_diagonal = EstimateGNDiagonal();
            alglib::real_1d_array diagonal;
            solver_step([&] {
                diagonal.setlength(n_relax_coeffs);
                for (int i = 0; i < n_relax_coeffs; ++i)
                    diagonal[i] = !trainable_mask.empty() && !trainable_mask[i]
                        ? 1. : std::max(gn_floor,gn_diagonal[i]);
                alglib::mincgsetprecdiag(cgstate,diagonal);
            });
            ++gn_refreshes;
            Message(fmt::format("GN diagonal refresh {} at accepted iteration {}: probes={}, min={}, max={}, floor={}; data VJP + model-defined L2 gradient differences",
                gn_refreshes,bfgs_iteration_offset+bfgs_accepted_iter,gn_probes,
                gn_diagonal.minCoeff(),gn_diagonal.maxCoeff(),gn_floor));
        };
        if (curvature_precondition == "gn_diag") update_curvature();
        auto evaluate = [&](double &value) {
            Training::collective_try([&] { SetOptimizerCoeffs(alg_x.getcontent()); });
            AccumulateGard(value, gradient);
            Training::collective_try([&] {
                // Check before masking, including derivatives of frozen groups.
                if (!std::isfinite(value) || !Eigen::Map<const Eigen::VectorXd>(
                        gradient.getcontent(), n_relax_coeffs).allFinite())
                    GPERROR("CG encountered a non-finite objective or gradient");
                ApplyTrainableMask(gradient);
            });
        };
        evaluate(loss_f_prev);
        Message("CG start");
        Message(fmt::format("CG coefficient-restart offset: {} (optimizer history reset)", bfgs_iteration_offset));
        Message(fmt::format("Initial Loss: {:.17g}", loss_f_prev));
        // Keep ALGLIB's public optimize API and callback protocol unchanged.
        // Workers service the root's callbacks; even an exception raised by
        // ALGLIB between callbacks can therefore send a stop event first.
        int refresh_requested = 0;
        int cycle_start = 0;
        struct Callbacks {
            std::function<void(const alglib::real_1d_array &, double &, alglib::real_1d_array &)> gradient;
            std::function<void(const alglib::real_1d_array &, double)> report;
        } callbacks;
        auto service = [&](int event, double &value) {
            MPI_Bcast(alg_x.getcontent(), n_relax_coeffs, MPI_DOUBLE, 0, mpi.comm);
            Training::collective_try([&] {
                if (!Eigen::Map<const Eigen::VectorXd>(alg_x.getcontent(), n_relax_coeffs).allFinite())
                    GPERROR("CG proposed non-finite coefficients");
            });
            if (event == 1) {
                evaluate(value);
                if (value < loss_f_prev) {
                    loss_f_prev = value;
                    display_iteration_log(value);
                    ++bfgs_iter;
                } else Message("  Loss: " + std::to_string(value));
            } else {
                MPI_Bcast(&value, 1, MPI_DOUBLE, 0, mpi.comm);
                // Validation handles its own collective phases. A rank-0-only
                // checkpoint failure must also reach every peer before stepping.
                Training::collective_try([&] {
                    if (!std::isfinite(value)) GPERROR("CG accepted a non-finite objective");
                    SetOptimizerCoeffs(alg_x.getcontent());
                });
                if (!bfgs_initial_reported) {
                    bfgs_initial_reported = true;
                    Training::collective_try([&] {
                        ValidateAcceptedPoint(alg_x.getcontent(), bfgs_iteration_offset, true);
                    });
                    Message(fmt::format("CG initial point: loss_f = {:.17g}", value));
                    return;
                }
                ++bfgs_accepted_iter;
                const int iteration = bfgs_iteration_offset + bfgs_accepted_iter;
                Message(fmt::format("CG accepted iteration {}: loss_f = {:.17g}", iteration, value));
                Training::collective_try([&] { ValidateAcceptedPoint(alg_x.getcontent(), iteration); });
                Training::collective_try([&] {
                    if (iteration % save_every_it == 0)
                        SaveCGPoint(fmt::format("{}.iter-{:06d}.mtp", potfilesave, iteration));
                });
            }
        };
        callbacks.gradient = [&](const alglib::real_1d_array &point, double &value,
                                 alglib::real_1d_array &out_gradient) {
            alg_x = point;
            int event = 1;
            MPI_Bcast(&event, 1, MPI_INT, 0, mpi.comm);
            service(event, value);
            out_gradient = gradient;
        };
        callbacks.report = [&](const alglib::real_1d_array &point, double value) {
            alg_x = point;
            int event = 2;
            MPI_Bcast(&event, 1, MPI_INT, 0, mpi.comm);
            service(event, value);
            if (curvature_precondition == "gn_diag" && gn_update > 0
                && bfgs_accepted_iter > cycle_start && bfgs_accepted_iter % gn_update == 0
                && (limit == 0 || bfgs_accepted_iter < limit)) {
                refresh_requested = 1;
                alglib::mincgrequesttermination(cgstate);
            }
        };
        for (;;) {
            refresh_requested = 0;
            cycle_start = bfgs_accepted_iter;
            std::string optimizer_error;
            if (mpi.rank == 0) {
                try {
                    alglib::mincgoptimize(cgstate,
                        [](const alglib::real_1d_array &point, double &value,
                           alglib::real_1d_array &gradient, void *context) {
                            static_cast<Callbacks *>(context)->gradient(point, value, gradient);
                        },
                        [](const alglib::real_1d_array &point, double value, void *context) {
                            static_cast<Callbacks *>(context)->report(point, value);
                        }, &callbacks);
                } catch (const alglib::ap_error &e) {
                    optimizer_error = "CG solver: " + e.msg;
                } catch (const std::exception &e) {
                    optimizer_error = e.what();
                } catch (...) {
                    optimizer_error = "Unknown CG optimizer exception";
                }
                int event = 0;
                MPI_Bcast(&event, 1, MPI_INT, 0, mpi.comm);
            } else {
                for (;;) {
                    int event = 0;
                    MPI_Bcast(&event, 1, MPI_INT, 0, mpi.comm);
                    if (event == 0) break;
                    double value = 0.;
                    try { service(event, value); }
                    catch (const std::exception &e) { optimizer_error = e.what(); }
                    catch (...) { optimizer_error = "Unknown CG worker exception"; }
                    // A callback failure is collective. Wait for root to unwind
                    // optimize and send stop, rather than entering another reduction.
                }
            }
            Training::collective_check(optimizer_error);
            solver_step([&] {
                alglib::mincgresults(cgstate, alg_x, report);
                cg_termination_type = static_cast<int>(report.terminationtype);
                Message(fmt::format("CG report: iterations={}, gradient_evaluations={}, termination_type={}",
                                    report.iterationscount, report.nfev, report.terminationtype));
            });
            MPI_Bcast(&cg_termination_type, 1, MPI_INT, 0, mpi.comm);
            MPI_Bcast(alg_x.getcontent(), n_relax_coeffs, MPI_DOUBLE, 0, mpi.comm);
            MPI_Bcast(&refresh_requested,1,MPI_INT,0,mpi.comm);
            if (!refresh_requested || cg_termination_type != 8) break;
            Training::collective_try([&] { SetOptimizerCoeffs(alg_x.getcontent()); });
            update_curvature();
            solver_step([&] {
                alglib::mincgrestartfrom(cgstate,alg_x);
                alglib::mincgsetcond(cgstate,bfgs_grad_tol,bfgs_conv_tol,bfgs_step_tol,
                    limit == 0 ? 0 : limit-bfgs_accepted_iter);
            });
            bfgs_initial_reported = false;
            Message(fmt::format("Curvature history restart at accepted iteration {}; remaining budget={}",
                bfgs_iteration_offset+bfgs_accepted_iter,limit == 0 ? 0 : limit-bfgs_accepted_iter));
        }
        Training::collective_try([&] {
            if (!Eigen::Map<const Eigen::VectorXd>(alg_x.getcontent(), n_relax_coeffs).allFinite())
                GPERROR("CG returned non-finite coefficients");
            switch (cg_termination_type) {
            case 1: Message("Relative function improvement is no more than EpsF"); break;
            case 2: Message("Relative step is no more than EpsX, stop optimization"); break;
            case 4: Message("Gradient norm is no more than EpsG, stop optimization"); break;
            case 5: Warning("Max iteration limit reached; retaining the finite accepted point (not converged)"); break;
            case 7: Warning("Stopping conditions are too stringent; retaining the finite accepted point (not converged)"); break;
            default: GPERROR("CG failed with termination type " + std::to_string(cg_termination_type));
            }
            SetOptimizerCoeffs(alg_x.getcontent());
            p_pot->nl_get_coeffs(x.data());
        });
        double final_loss = 0.;
        evaluate(final_loss);
        Training::collective_try([&] {
            ValidateAcceptedPoint(x.data(), bfgs_iteration_offset + bfgs_accepted_iter, true);
        });
        Training::collective_try([&] {
            // Final evaluation/validation installs the accepted coordinates.
            // Nonlinear coordinate maps (e.g. SOG bandwidths) may round-trip
            // them, so publication must snapshot the resulting live model.
            p_pot->nl_get_coeffs(x.data());
            cg_result_coefficients_.assign(x.data(), x.data()+n_relax_coeffs);
            cg_result_potential_ = p_pot;
        });
    } catch (...) {
        // Failed line-search probes must not leak into the caller's live model.
        // Already written checkpoints contain only previously accepted points.
        Training::collective_try([&] { p_pot->nl_set_coeffs(initial_coeffs.data()); });
        x = initial_coeffs;
        validation_current_ = false;
        validation_coefficients_.clear();
        InvalidateCGResult();
        throw;
    }
}

void TrainNonlinear::save_current(std::string fnm) {
    RequireCurrentParameterLayout();
    if (mpi.rank == 0)
        p_pot->Save(fnm);
}

void TrainNonlinear::save_accepted_checkpoint(
    const alglib::real_1d_array &coeffs) {
    if (mpi.rank != 0)
        return;
    const int cumulative_iteration = bfgs_iteration_offset +
        bfgs_accepted_iter;
    if (cumulative_iteration <= 0 ||
        cumulative_iteration % save_every_it != 0)
        return;
    SetOptimizerCoeffs(coeffs.getcontent());
    save_current(fmt::format(
        "{}.iter-{:06d}.mtp", potfilesave, cumulative_iteration));
}

void TrainNonlinear::test_curv(int ind, double nudge) {
    alglib::real_1d_array x_coeffs, x_gard0, x_gard1, x_gard2;
    x_coeffs.setlength(n_relax_coeffs);
    x_gard0.setlength(n_relax_coeffs);
    x_gard2.setlength(n_relax_coeffs);
    x_gard1.setlength(n_relax_coeffs);
    p_pot->nl_get_coeffs(x_coeffs.getcontent());
    double f0, f1, f2;
    AccumulateGard(f0, x_gard0);
    double x0 = x_coeffs[ind];

    double ngval = x0 * nudge;
    const double min_abs_nudge = 1e-4;
    if (fabs(ngval) < min_abs_nudge)
        ngval = std::copysign(min_abs_nudge, x0 + 1e-100);

    x_coeffs[ind] = x0 + ngval;
    p_pot->nl_set_coeffs(x_coeffs.getcontent());
    AccumulateGard(f1, x_gard1);
    x_coeffs[ind] = x0 - ngval;
    p_pot->nl_set_coeffs(x_coeffs.getcontent());
    AccumulateGard(f2, x_gard2);

    double hh = (f1 + f2 - 2.0 * f0) / (ngval * ngval);

    if (mpi.rank == 0)
    std::cout << std::endl
            << " On coeff " << ind << " WITH COEFF = " << x_coeffs[ind]
            << " AND NUDGE = " << ngval << " (abs_min=" << min_abs_nudge << ")" << std::endl
            << " F0 = " << f0 << " F1 = " << f1 << " F2 = " << f2 << std::endl
            << " G0 = " << x_gard0[ind] << " G1 = " << x_gard1[ind] << " G2 = " << x_gard2[ind] << std::endl
            << " HE = " << hh << std::endl;

    x_coeffs[ind] = x0;
    p_pot->nl_set_coeffs(x_coeffs.getcontent());
}

void TrainNonlinear::check_grad(int ind, double nudge) {
    alglib::real_1d_array x_coeffs, x_gard0, x_gard1, x_gard2;
    x_coeffs.setlength(n_relax_coeffs);
    x_gard0.setlength(n_relax_coeffs);
    x_gard2.setlength(n_relax_coeffs);
    x_gard1.setlength(n_relax_coeffs);
    p_pot->nl_get_coeffs(x_coeffs.getcontent());
    double f0, f1, f2;
    AccumulateGard(f0, x_gard0);
    double x0 = x_coeffs[ind];
    // Minimum absolute perturbation to avoid vanishing FD for small coefficients.
    // When |x0 * nudge| < min_abs_nudge, use min_abs_nudge as the perturbation.
    double ngval = x0 * nudge;
    const double min_abs_nudge = 1e-4;
    if (fabs(ngval) < min_abs_nudge)
        ngval = std::copysign(min_abs_nudge, x0 + 1e-100);

    x_coeffs[ind] = x0 + ngval;
    p_pot->nl_set_coeffs(x_coeffs.getcontent());;
    AccumulateGard(f1, x_gard1);
    x_coeffs[ind] = x0 - ngval;
    p_pot->nl_set_coeffs(x_coeffs.getcontent());
    AccumulateGard(f2, x_gard2);
    x_coeffs[ind] = x0;

    double hh = (f1 + f2 - 2.0 * f0) / (ngval * ngval);
    double n0 = (f1 - f2) / (2.0 * ngval);
    double n1 = (f1 - f0) / ngval;
    double n2 = (f0 - f2) / ngval;

    if (mpi.rank == 0)
    std::cout << "Total coeffs count is " << n_relax_coeffs << std::endl
            << " On coeff " << ind << " WITH COEFF = " << x_coeffs[ind]
            << " AND NUDGE = " << ngval << " (abs_min=" << min_abs_nudge << ")" << std::endl
            << " F0 = " << f0 << " F1 = " << f1 << " F2 = " << f2 << std::endl
            << " G0 = " << x_gard0[ind] << " G1 = " << x_gard1[ind] << " G2 = " << x_gard2[ind] << std::endl
            << " N0 = " << n0 << " (G0~N0 ratio: " << (fabs(n0) > 1e-30 ? x_gard0[ind]/n0 : 0) << ")" << std::endl
            << " N_CURV = " << hh << std::endl;

    p_pot->nl_set_coeffs(x_coeffs.getcontent());
}

void TrainNonlinear::display_iteration_log(double &f) {
    if (hessian_fit.weight > 0)
        Message(fmt::format(" Full Hessian FD monitor: loss = {} RMS = {}", emonitor.loss_h, emonitor.hrms));
    if (hvp_fit.weight > 0) {
        Message(fmt::format(" HVP monitor: loss = {} RMS = {}", emonitor.loss_hvp, emonitor.hvp_rms));
    }

    Message(fmt::format("Iteration {} : loss_f = {} f_real = {} f_eng = {} f_fce = {} f_str = {} f_charge = {} f_pnal = {} ", bfgs_iter, f, f-ploss, emonitor.loss_e, emonitor.loss_f, emonitor.loss_s, qmonitor.loss_q, ploss));
    Message(fmt::format(" Train monitor: EATRMS = {:.5f} FRMS = {:.5f} SRMS = {:.5f}", emonitor.erms * 1000, emonitor.frms * 1000, emonitor.srms));
    if (p_pot->supports_total_charge_training())
        Message(fmt::format(" Charge monitor: QRMS = {:.6f} Q = [{:.6f}, {:.6f}] q = [{:.6f}, {:.6f}]",
            qmonitor.qrms, qmonitor.report_total_min, qmonitor.report_total_max,
            qmonitor.report_q_min, qmonitor.report_q_max));
    DisplayValidation();
}

void TrainNonlinear::DisplayValidation() const {
    if (validation_current_ && valid_set_size_all > 0) {
        if (hessian_fit.weight > 0)
            Message(fmt::format(" Test full Hessian FD: RMS = {}", emonitor_valid.hrms));
        if (hvp_fit.weight > 0)
            Message(fmt::format(" Test HVP: RMS = {}", emonitor_valid.hvp_rms));
        Message(fmt::format(" Test monitor: EATRMS = {:.5f} FRMS = {:.5f} SRMS = {:.5f}", emonitor_valid.erms * 1000, emonitor_valid.frms * 1000, emonitor_valid.srms));
        if (p_pot->supports_total_charge_training())
            Message(fmt::format(" Test charge: QRMS = {:.6f} Q = [{:.6f}, {:.6f}] q = [{:.6f}, {:.6f}]",
                qmonitor_valid.qrms, qmonitor_valid.report_total_min,
                qmonitor_valid.report_total_max, qmonitor_valid.report_q_min,
                qmonitor_valid.report_q_max));
    }
}

void TrainNonlinear::AddHVPLoss(const Atoms &atom, double normalization, double &loss,
    bool gradient, ErrorMonitor &monitor) {
    if (hvp_fit.weight == 0 || !atom.has_hvp()) return;
    Training::ForcePullback backward;
    if (gradient) backward = [&](Atoms &prediction, Neighborhoods &nbhs,
                                 Eigen::Matrix<double, -1, 3> &weights) {
        Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();
        p_pot->AccumulateEFSCombinationGard(prediction, nbhs, 0., weights, stress, fweights, loss_gard.data());
    };
    const auto result = Training::finite_difference_hvp_loss(*p_pot, atom, hvp_fit, normalization, backward);
    loss += result.loss;
    monitor.add_hvp(result);
}

void TrainNonlinear::AddHessianLoss(const Atoms &atom, double normalization, double &loss,
    bool gradient, ErrorMonitor &monitor) {
    validate_hessian_settings(hessian_fit);
    if (hessian_fit.weight == 0 || !atom.have_hessian) return;
    validate_hessian_label(atom);
    const int ndof = 3 * atom.size();
    Atoms probes = atom;
    probes.hvp_directions_ = Eigen::MatrixXd::Identity(ndof, ndof);
    probes.hvp_targets_ = atom.hessian_;
    auto fit = hessian_fit;
    // HVP loss averages over probes; a full matrix uses its Frobenius sum.
    fit.weight *= ndof;
    Training::ForcePullback backward;
    if (gradient) backward = [&](Atoms &prediction, Neighborhoods &nbhs,
                                 Eigen::Matrix<double, -1, 3> &weights) {
        Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();
        p_pot->AccumulateEFSCombinationGard(prediction, nbhs, 0., weights, stress, fweights, loss_gard.data());
    };
    const auto result = Training::finite_difference_hvp_loss(*p_pot, probes, fit, normalization, backward);
    loss += result.loss;
    monitor.sq_hessian_error += result.squared_error;
    monitor.loss_h += result.loss;
    monitor.nhessian += result.observations;
}

void TrainNonlinear::ValidateCurvatureSettings() const {
    if (curvature_precondition != "none" && curvature_precondition != "gn_diag")
        GPERROR("curvature_precondition must be none or gn_diag");
    if (gn_probes <= 0 || gn_update < 0 || gn_seed < 0 || !std::isfinite(gn_floor) || gn_floor <= 0.)
        GPERROR("GN requires positive gn_probes and gn_floor, nonnegative gn_update and gn_seed");
    if (curvature_precondition == "gn_diag") {
        std::string device = training_device;
        std::transform(device.begin(),device.end(),device.begin(),[](unsigned char c) { return std::tolower(c); });
        if (device != "cpu") GPERROR("gn_diag requires CPU residual pullbacks");
    }
    const std::string mode = bfgs_set_scale ? "file" : parameter_scaling;
    if (mode != "none" && mode != "file" && mode != "group_auto")
        GPERROR("parameter_scaling must be none, file or group_auto");
    if (mode == "group_auto" && (!std::isfinite(scale_floor) || !std::isfinite(scale_ceiling)
        || scale_floor <= 0. || scale_ceiling < scale_floor))
        GPERROR("Require finite 0 < scale_floor <= scale_ceiling");
}

void TrainNonlinear::PrepareParameterScale(const double *coeffs) {
    ValidateCurvatureSettings();
    const std::string mode = bfgs_set_scale ? "file" : parameter_scaling;
    scale.setlength(n_relax_coeffs);
    for (int i = 0; i < n_relax_coeffs; ++i) scale[i] = 1.;
    if (mode == "file") {
        std::ifstream input("scale_data");
        if (!input) GPERROR("parameter_scaling=file requires scale_data");
        for (int i = 0; i < n_relax_coeffs; ++i)
            if (!(input >> scale[i]) || !std::isfinite(scale[i]) || scale[i] == 0.)
                GPERROR("Invalid or incomplete scale_data");
        std::string extra;
        if (input >> extra) GPERROR("Trailing data in scale_data");
    } else if (mode == "group_auto") {
        for (int i = 0; i < n_relax_coeffs; ++i) scale[i] = scale_floor;
        for (const auto &group : p_pot->nl_parameter_groups()) {
            if (group.begin < 0 || group.end < group.begin || group.end > n_relax_coeffs)
                GPERROR("Invalid parameter scaling group: " + group.name);
            const int count = group.end - group.begin;
            const double norm = Eigen::Map<const Eigen::VectorXd>(coeffs+group.begin,count).stableNorm();
            const double value = std::clamp(count ? norm/std::sqrt(double(count)) : scale_floor,
                                           scale_floor,scale_ceiling);
            for (int i = group.begin; i < group.end; ++i) scale[i] = value;
            Message(fmt::format("Parameter scale {} = {}",group.name,value));
        }
    }
}

Eigen::Index TrainNonlinear::WeightedResidualCount() const {
    Eigen::Index count = 0;
    for (const auto &a : training_set_) {
        if (!a.size()) continue;
        const Eigen::Index ndof = 3 * Eigen::Index(a.size());
        if (weight_eng > 0. && a.has_energy()) ++count;
        if (weight_force > 0. && a.has_forces()) count += ndof;
        if (weight_stress > 0. && a.has_stresses()) count += 9;
        if (weight_charge > 0. && p_pot->supports_total_charge_training()) ++count;
        if (hvp_fit.weight > 0. && a.has_hvp()) count += ndof * a.hvp_directions_.cols();
        if (hessian_fit.weight > 0. && a.has_hessian()) count += ndof * ndof;
    }
    return count;
}

void TrainNonlinear::AccumulateResidualVJP(const Eigen::VectorXd &cotangents, Eigen::VectorXd &gradient) {
    RequireCurrentParameterLayout();
    if (!p_pot || !train_set_size_all || cotangents.size() != WeightedResidualCount() || !cotangents.allFinite())
        GPERROR("Weighted residual VJP requires initialized training data and finite matching cotangents");
    gradient.setZero(n_relax_coeffs);
    Eigen::Index row = 0;
    auto root_weight = [&](double weight) {
        const double mean_weight = weight / double(train_set_size_all);
        if (!std::isfinite(mean_weight) || mean_weight < 0.) GPERROR("Invalid weighted residual observation");
        return std::sqrt(mean_weight);
    };
    for (int k = 0; k < training_set_.size(); ++k) {
        const auto &a = training_set_[k];
        if (!a.size()) continue;
        const double cc = efs_weights().normalization(a);
        auto &prediction = efs_set_[k];
        auto &nbhs = trainint_set_nbhs_[k];
        p_pot->getEFS(prediction,nbhs);
        double energy = 0.;
        Eigen::Matrix<double,-1,3> force = Eigen::Matrix<double,-1,3>::Zero(a.size(),3);
        Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();
        if (weight_eng > 0. && a.has_energy()) energy = root_weight(energy_weight(a)*cc)*cotangents[row++];
        const bool forces = weight_force > 0. && a.has_forces();
        const bool stresses = weight_stress > 0. && a.has_stresses();
        if (forces) for (int i = 0; i < a.size(); ++i) {
            const double weight = root_weight(EFSFitWeights::atomic_weight(a,i,force_weight(a)*cc));
            for (int j = 0; j < 3; ++j) force(i,j) = weight*cotangents[row++];
        }
        if (stresses) {
            const double weight = root_weight(stress_weight(a)*cc);
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
                stress(i,j) = weight*cotangents[row++];
        }
        if (weight_charge > 0. && p_pot->supports_total_charge_training())
            p_pot->AccumulateTotalChargeGard(prediction,nbhs,
                root_weight(charge_weight(a))*cotangents[row++],gradient.data());
        if (forces || stresses)
            p_pot->AccumulateEFSCombinationGard(prediction,nbhs,energy,force,stress,fweights,gradient.data());
        else
            p_pot->AccumulateEFSCombinationGard(prediction,nbhs,energy,gradient.data());
        Training::ForcePullback backward = [&](Atoms &scratch, Neighborhoods &neighbors,
                                               Eigen::Matrix<double,-1,3> &weights) {
            Eigen::Matrix3d zero = Eigen::Matrix3d::Zero();
            p_pot->AccumulateEFSCombinationGard(scratch,neighbors,0.,weights,zero,fweights,gradient.data());
        };
        const int ndof = 3 * a.size();
        if (hvp_fit.weight > 0. && a.has_hvp()) {
            Eigen::MatrixXd weights(ndof,a.hvp_directions_.cols());
            const double weight = root_weight(hvp_fit.observation_weight(a,cc));
            for (int j = 0; j < weights.cols(); ++j) for (int i = 0; i < ndof; ++i)
                weights(i,j) = weight*cotangents[row++];
            Training::finite_difference_hvp_vjp(*p_pot,a,hvp_fit.step,weights,backward);
        }
        if (hessian_fit.weight > 0. && a.has_hessian()) {
            Atoms probes = a;
            probes.hvp_directions_ = Eigen::MatrixXd::Identity(ndof,ndof);
            probes.hvp_targets_ = a.hessian_;
            Eigen::MatrixXd weights(ndof,ndof);
            const double weight = root_weight(hessian_fit.weight*a.fitness*cc/std::pow(a.size(),hessian_fit.power));
            for (int j = 0; j < ndof; ++j) for (int i = 0; i < ndof; ++i)
                weights(i,j) = weight*cotangents[row++];
            Training::finite_difference_hvp_vjp(*p_pot,probes,hessian_fit.step,weights,backward);
        }
    }
    if (!gradient.allFinite()) GPERROR("Non-finite weighted residual VJP");
    if (!trainable_mask.empty()) for (int i = 0; i < n_relax_coeffs; ++i)
        if (!trainable_mask[i]) gradient[i] = 0.;
}

Eigen::VectorXd TrainNonlinear::L2CurvatureDiagonal() {
    Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(n_relax_coeffs);
    const auto l2 = p_pot->reg_L2_vector;
    if (std::none_of(l2.begin(),l2.end(),[](double value) { return value != 0.; })) return diagonal;
    Eigen::VectorXd coefficients(n_relax_coeffs), trial, plus(n_relax_coeffs), minus(n_relax_coeffs);
    p_pot->nl_get_coeffs(coefficients.data()); trial = coefficients;
    auto isolated_gradient = [&](Eigen::VectorXd &result) {
        Eigen::VectorXd without = Eigen::VectorXd::Zero(n_relax_coeffs);
        result.setZero(); double value = 0.;
        p_pot->reg_L2_vector = l2;
        p_pot->AddPenaltyGard(weight_penatly,value,result.data());
        std::fill(p_pot->reg_L2_vector.begin(),p_pot->reg_L2_vector.end(),0.);
        value = 0.; p_pot->AddPenaltyGard(weight_penatly,value,without.data());
        result -= without;
        if (!result.allFinite()) GPERROR("Non-finite isolated L2 gradient");
    };
    try {
        for (int i = 0; i < n_relax_coeffs; ++i) {
            if (!trainable_mask.empty() && !trainable_mask[i]) continue;
            const double step = std::cbrt(std::numeric_limits<double>::epsilon())*std::max(1.,std::abs(coefficients[i]));
            trial[i] = coefficients[i]+step; p_pot->nl_set_coeffs(trial.data()); isolated_gradient(plus);
            trial[i] = coefficients[i]-step; p_pot->nl_set_coeffs(trial.data()); isolated_gradient(minus);
            diagonal[i] = (plus[i]-minus[i])/(2.*step);
            trial[i] = coefficients[i];
        }
    } catch (...) {
        p_pot->reg_L2_vector = l2; p_pot->nl_set_coeffs(coefficients.data()); throw;
    }
    p_pot->reg_L2_vector = l2; p_pot->nl_set_coeffs(coefficients.data());
    if (!diagonal.allFinite()) GPERROR("Non-finite model-defined L2 diagonal");
    return diagonal;
}

Eigen::VectorXd TrainNonlinear::EstimateGNDiagonal() {
    Eigen::VectorXd local;
    Training::collective_try([&] {
        ValidateCurvatureSettings();
        const std::uint64_t seed = std::uint64_t(gn_seed) + 0x9e3779b97f4a7c15ULL*std::uint64_t(mpi.rank);
        local = Training::estimate_gn_diagonal(WeightedResidualCount(),n_relax_coeffs,gn_probes,seed,
            [&](const Eigen::VectorXd &z, Eigen::VectorXd &out) { AccumulateResidualVJP(z,out); });
    });
    Eigen::VectorXd result = local;
#ifdef IMRMLP_MPI
    MPI_Allreduce(local.data(),result.data(),n_relax_coeffs,MPI_DOUBLE,MPI_SUM,mpi.comm);
#endif
    Training::collective_try([&] { result += L2CurvatureDiagonal(); });
    if (!result.allFinite()) GPERROR("Non-finite GN diagonal");
    return result;
}
