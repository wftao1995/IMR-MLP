// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "train_mtp.h"
#include "linear_efs_system.h"
#include "linear_equilibration.h"
#include "mtp_linear_reduce.h"
#include "collective_error.h"
#include "conditional_linear_persistence.h"
#include "gpu_validation_storage.h"
#include "analyse/gpu_analytic_hvp.h"
#include "train_nonlinear_potential.h"
#include "tools/tools.h"
#include "libalglib/solvers.h"
#include "utils/gputils.h"
#include "utils/gputils_eigen.h"
#include "libalglib/alglibmisc.h"
#include "potentials/mtp_gpu_adapter.h"
#if defined(IMRMLP_GPU) && defined(IMRMLP_MPI)
#include "runtime/mtp/gpu-direct/gpu_mpi.h"
#endif

#include <exception>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <typeinfo>

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
    // The vendored LBFGSpp constructor leaves this exposed option unset.
    // Settings snapshots (including MPI agreement) must never read it uninitialized.
    lbfgs_param.epsgmax = 0.;
    Connect(linear_device_, "device", "Training backend: cpu, cuda, hip or dcu");
    Connect(gpu_device_loss_, "gpu_device_loss", "Compute nonlinear residuals and loss on device");
    Connect(gpu_deterministic_reduction_, "gpu_deterministic_reduction");
    Connect(linear_gpu_curvature_host_mib_, "gpu_curvature_host_mib", "Owned host curvature workspace budget in MiB");
    Connect(linear_gpu_id_, "gpu_id");
    Connect(linear_gpu_batch_atoms_, "gpu_batch_atoms");
    Connect(linear_gpu_batch_structures_, "gpu_batch_structures");
    Connect(linear_gpu_memory_mib_, "gpu_memory_mib");
    Connect(linear_gpu_workspace_mib_, "gpu_workspace_mib");
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

Training::LinearEFSOptions Trainer_MTP::GpuLinearOptions() const {
    Training::LinearEFSOptions options;
    options.device = linear_device_;
    std::transform(options.device.begin(), options.device.end(), options.device.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    options.device = MLIPCore::GpuDirect::canonical_device(options.device);
    options.gpu_id = linear_gpu_id_;
    options.gpu_batch_atoms = linear_gpu_batch_atoms_;
    options.gpu_batch_structures = linear_gpu_batch_structures_;
    if (linear_gpu_memory_mib_ <= 0 || linear_gpu_workspace_mib_ <= 0 ||
        size_t(linear_gpu_memory_mib_) > std::numeric_limits<size_t>::max() / (size_t{1} << 20) ||
        size_t(linear_gpu_workspace_mib_) > std::numeric_limits<size_t>::max() / (size_t{1} << 20))
        GPERROR("GPU trainmtp memory budgets must be positive, representable MiB values");
    options.gpu_memory_bytes = size_t(linear_gpu_memory_mib_) << 20;
    options.gpu_workspace_bytes = size_t(linear_gpu_workspace_mib_) << 20;
    return options;
}

std::string Trainer_MTP::GpuLinearAllocationSignature() const {
    std::ostringstream out;
    out << std::setprecision(17) << fast_lfit << ' ' << fix_species_coeffs << ' ' << linear_fit_eng_only << ' '
        << weight_eng << ' ' << weight_force << ' ' << weight_stress << ' '
        << weight_eng_power << ' ' << weight_force_power << ' ' << weight_stress_power << ' '
        << weight_force_norm_d << ' ' << weight_force_norm_f << ' ' << weight_force_norm_p << ' '
        << weight_hessian << ' ' << weight_hessian_power << ' ' << hvp_fit.weight << ' ' << hvp_fit.power;
    return out.str();
}

bool Trainer_MTP::UsesGpuLinear() const {
    std::string device = linear_device_;
    std::transform(device.begin(), device.end(), device.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return device != "cpu";
}

bool Trainer_MTP::UsesGpuCurvature() const {
    return UsesGpuLinear() && (weight_hessian > 0 || hvp_fit.weight > 0 ||
        std::any_of(training_set_.begin(), training_set_.end(), [](const Atoms &a) {
            return a.have_hessian || a.hvp_directions_.size() || a.hvp_targets_.size();
        }));
}

namespace {
using Training::gpu_validation_checked_add;
using Training::gpu_validation_checked_mul;
size_t curvature_dof(const Atoms &a) {
    if (a.pos_.size() > size_t(std::numeric_limits<int>::max()) / 3)
        GPERROR("GPU trainmtp Cartesian dimension exceeds the native index range");
    const size_t d = 3 * a.pos_.size();
    if (a.hvp_directions_.cols() > std::numeric_limits<int>::max() ||
        a.hvp_targets_.cols() > std::numeric_limits<int>::max())
        GPERROR("GPU trainmtp HVP probe count exceeds the native index range");
    return d;
}
size_t upper_rows(size_t d) {
    return gpu_validation_checked_mul(d, gpu_validation_checked_add(d, 1)) / 2;
}
size_t prediction_geometry_bytes(const std::vector<Atoms> &data) {
    size_t bytes = gpu_validation_checked_mul(data.size(), sizeof(Atoms));
    for (const auto &a : data)
        bytes = gpu_validation_checked_add(bytes,
            gpu_validation_checked_mul(a.pos_.size(), 7 * sizeof(double) + sizeof(int)));
    return bytes;
}
std::vector<Atoms> prediction_geometries(const std::vector<Atoms> &data) {
    std::vector<Atoms> result;
    result.reserve(data.size());
    for (const auto &a : data) result.push_back(Training::gpu_hvp_geometry_copy(a));
    return result;
}
template<class Buffer>
size_t curvature_cpu_retained(const Buffer &b) {
    size_t bytes = 0;
    auto add = [&](size_t count, size_t unit = sizeof(double)) {
        bytes = gpu_validation_checked_add(bytes, gpu_validation_checked_mul(count, unit));
    };
    for (const auto *v : {&b.basis_vals, &b.moment_vals, &b.dist_powers, &b.rb_vals, &b.rb_ders,
                         &b.rb_derders, &b.val_mu, &b.der_mu, &b.derder_mu,
                         &b.moment_adjoint, &b.site_energy_ders_wrt_moments_}) add(v->capacity());
    add(size_t(b.coords_powers.size())); add(size_t(b.buff_site_energy_ders.size()));
    add(size_t(b.buff_site_hvp.size())); add(size_t(b.buff_site_hvp_direction.size()));
    add(size_t(b.moment_jacobian.size())); add(size_t(b.basis_ders.size())); add(size_t(b.moment_ders.size()));
    add(size_t(b.moment_jacobian_coeffs.size())); add(size_t(b.moment_jacobian_coeff_coord.size()));
    add(b.dloss_dmom.capacity()); add(b.dloss_dsenders.capacity());
    add(size_t(b.ext_dUdB.size())); add(size_t(b.ext_d2UdBdB.size()));
    add(size_t(b.Q_jacobian.size()));
    add(size_t(b.arraybuffer3d.size())); add(size_t(b.matrixx3dbuffer.size()));
    add(b.dSEdsi.capacity()); add(size_t(b.tmp_se_ders_wgts.size()));
    add(size_t(b.ext_dUdQ.size())); add(size_t(b.ext_d2udqdt.size())); add(size_t(b.Qs.size()));
    // Cache entries/ready flags grow together from the same begin_site count;
    // allow two copies, including the flag array and alignment, if omp_init
    // reallocates the vector of complete buffers. Verification disables new use.
    add(b.radial_cache.capacity(), 2 * (sizeof(std::pair<double, double>) + 1));
    add(b.basic_moment_hessian_diag.capacity(), sizeof(Eigen::Matrix3d));
    add(b.buff_site_hessian.hessian.capacity(), sizeof(Eigen::Matrix3d));
    for (const auto *matrices : {&b.moment_hessian, &b.basis_derders, &b.Q_hessian,
                                 &b.moment_hessian_sparse, &b.basis_derders_sparse}) {
        add(matrices->capacity(), sizeof(Potentials::HessianMatrix));
        for (const auto &matrix : *matrices) add(matrix.hessian.capacity(), sizeof(Eigen::Matrix3d));
    }
    if constexpr (requires { b.force; }) {
        add(b.force.capacity(), sizeof(Eigen::Vector3d)); add(b.siteenergy.capacity());
        add(b.hessian.hessian.capacity(), sizeof(Eigen::Matrix3d)); add(b.hessian_diag.capacity());
        add(b.buff_site_hessian_diag.capacity());
        for (const auto *nested : {&b.buff_moment_hessian_diag, &b.buff_basis_derders_diag}) {
            add(nested->capacity(), sizeof(std::vector<double>));
            for (const auto &v : *nested) add(v.capacity());
        }
        add(size_t(b.cellhess.size())); add(size_t(b.lrhess.size()));
    }
    return bytes;
}
}

size_t Trainer_MTP::GpuCurvatureHostCap() const {
    if (linear_gpu_curvature_host_mib_ <= 0 ||
        size_t(linear_gpu_curvature_host_mib_) > std::numeric_limits<size_t>::max() / (size_t{1} << 20))
        GPERROR("GPU trainmtp curvature host budget must be a positive representable MiB value");
    return size_t(linear_gpu_curvature_host_mib_) << 20;
}

size_t Trainer_MTP::GpuCurvatureOwnedBytes() const {
    size_t bytes = 0;
    auto add = [&](size_t n) { bytes = gpu_validation_checked_add(bytes, n); };
    for (const auto *data : {&training_set_, &efs_set_, &valid_set_, &efs_valid_set_})
        add(Training::gpu_validation_atoms_bytes(*data));
    for (const auto *nbhs : {&trainint_set_nbhs_, &valid_set_nbhs_})
        add(Training::gpu_validation_neighbors_bytes(*nbhs));
    for (const auto *v : {&linear_reduce_send, &linear_reduce_receive})
        add(gpu_validation_checked_mul(v->capacity(), sizeof(double)));
    add(gpu_validation_checked_mul(train_set_inds_aligned.capacity(), sizeof(int)));
    add(gpu_validation_checked_mul(gpu_linear_report_.effective_l2.capacity(), sizeof(double)));
    add(gpu_validation_checked_mul(energy_cmpts.capacity(), sizeof(double)));
    auto eigen = [&](const auto &v) { add(gpu_validation_checked_mul(size_t(v.size()), sizeof(double))); };
    eigen(gpu_linear_gram_); eigen(gpu_linear_rhs_);
    eigen(eigen_lin_matrix_A); eigen(eigen_lin_matrix); eigen(eigen_lin_matrix_mpi);
    eigen(eigen_lin_matrix_F); eigen(eigen_lin_matrix_S); eigen(eigen_lin_matrix_F_mpi); eigen(eigen_lin_matrix_S_mpi);
    eigen(eigen_lin_matrix_A_F); eigen(eigen_lin_matrix_A_S);
    eigen(eigen_lin_vector_A); eigen(eigen_lin_vector); eigen(eigen_lin_vector_mpi); eigen(weight_list);
    eigen(eigen_lin_vector_F); eigen(eigen_lin_vector_S); eigen(eigen_lin_vector_F_mpi); eigen(eigen_lin_vector_S_mpi);
    eigen(eigen_lin_vector_A_F); eigen(eigen_lin_vector_A_S);
    eigen(linear_coeff_buffer); eigen(x_in); eigen(g_in); eigen(scaling_coeffs);
    eigen(force_cmpts); eigen(stress_cmpts); eigen(force_cmpts_R); eigen(stress_cmpts_R); eigen(dLdF);
    for (const auto *v : {&lin_vector, &lin_vector_F, &lin_vector_S, &lin_vector_mpi, &lin_vector_F_mpi, &lin_vector_S_mpi,
                         &loss_gard, &optimization_coeffs, &update_vector, &update_vector_last, &scale, &loss_gard_mpi, &pgard, &pgard_mpi})
        if (v->length()) add(gpu_validation_checked_add(gpu_validation_checked_mul(size_t(v->length()), sizeof(double)), 63));
    for (const auto *m : {&lin_matrix, &lin_matrix_F, &lin_matrix_S, &lin_matrix_mpi, &lin_matrix_F_mpi, &lin_matrix_S_mpi})
        if (m->rows()) add(gpu_validation_checked_add(gpu_validation_checked_mul(size_t(m->rows()),
            gpu_validation_checked_add(sizeof(void *), gpu_validation_checked_mul(size_t(m->c_ptr()->stride), sizeof(double)))), 63));
    return bytes;
}

size_t Trainer_MTP::GpuCurvatureRowCount(const std::vector<Atoms> &data) const {
    size_t rows = 0;
    for (const auto &a : data) {
        const size_t d = curvature_dof(a);
        a.validate_hvp();
        Training::validate_hessian_fit_label(a);
        if (weight_eng > 0 && a.has_energy()) rows = gpu_validation_checked_add(rows, 1);
        if (weight_force > 0 && a.has_forces())
            for (size_t i = 0; i < a.pos_.size(); ++i)
                if (a.atomic_fitness.empty() || a.atomic_fitness[i] > 0) rows = gpu_validation_checked_add(rows, 3);
        if (weight_stress > 0 && a.has_stresses()) rows = gpu_validation_checked_add(rows, 9);
        if (weight_hessian > 0 && a.has_hessian()) rows = gpu_validation_checked_add(rows, upper_rows(d));
        if (hvp_fit.weight > 0 && a.has_hvp())
            rows = gpu_validation_checked_add(rows, gpu_validation_checked_mul(d, size_t(a.hvp_directions_.cols())));
    }
    if (rows > size_t(std::numeric_limits<int>::max())) GPERROR("GPU trainmtp observation count exceeds the native index range");
    return rows;
}

size_t Trainer_MTP::GpuCurvatureReserve(const Potentials::PotMTP &potential,
    const std::vector<Atoms> &data, const std::vector<Neighborhoods> &neighbors, bool allocating) const {
    const size_t c = size_t(potential.lin_coeff_size - (fix_species_coeffs ? potential.species_count : 0));
    const size_t full = size_t(potential.get_coeff_size()), rows = GpuCurvatureRowCount(data);
    auto mul = gpu_validation_checked_mul;
    auto add = gpu_validation_checked_add;
    const size_t square = mul(c, c), triangle = add(c, mul(c, add(c, 1)) / 2);
    // Conservative concurrent CPU solve/equilibration copies, fit snapshots,
    // verification Gram/RHS and MPI send/receive buffers. This is owned storage
    // accounting, not an RSS or allocator-overhead claim.
    size_t doubles = add(mul(8, square), add(mul(16, c), add(mul(8, full), mul(2, triangle))));
    if (allocating) {
        doubles = add(doubles, add(mul(2, square), add(mul(12, full), mul(5, c))));
        if (fast_lfit) doubles = add(doubles, mul(rows, add(c, 2)));
    }
    size_t bytes = mul(doubles, sizeof(double));
    // ALGLIB rows include aligned strides and pointer tables, not just C*C.
    bytes = add(bytes, mul(12, add(mul(c, sizeof(void *) + 63), 63)));
    bytes = add(bytes, mul(gpu_curvature_aligned_size_, sizeof(int)));
    bytes = add(bytes, mul(data.size(), 512)); // request masks, row descriptors, aligned indices
    bytes = add(bytes, mul(c, sizeof(int)));
    size_t maximum_frame = 0, maximum_neighbors = 0;
    size_t total_atoms = 0, total_edges = 0;
    for (size_t k = 0; k < data.size(); ++k) {
        const size_t n = data[k].pos_.size();
        bytes = add(bytes, mul(n, sizeof(int))); // fast per-atom force row map
        // Geometry, CPU EFS outputs, one direction/product, and rescaling scratch.
        maximum_frame = std::max(maximum_frame, add(sizeof(Atoms), mul(n, 24 * sizeof(double) + sizeof(int))));
        if (n > size_t(linear_gpu_batch_atoms_)) GPERROR("GPU trainmtp structure exceeds atom batch cap");
        total_atoms = add(total_atoms, n);
        for (int i = 0; i < neighbors[k].size(); ++i) {
            if (neighbors[k][i].nbhcount_ < 0) GPERROR("Negative GPU trainmtp neighbor count");
            total_edges = add(total_edges, size_t(neighbors[k][i].nbhcount_));
            maximum_neighbors = std::max(maximum_neighbors, size_t(neighbors[k][i].nbhcount_));
        }
    }
    // EFS filters unlabeled frames and can split a batch early for workspace.
    // Bound every possible resulting batch, rather than a particular greedy
    // partition of the original data. Independent maxima cover retained buffers.
    const size_t maximum_packed_atoms = std::min(total_atoms, size_t(linear_gpu_batch_atoms_));
    const size_t maximum_packed_structures = std::min(data.size(), size_t(linear_gpu_batch_structures_));
    const size_t maximum_packed_edges = std::min(total_edges, mul(maximum_packed_atoms, maximum_neighbors));
    size_t maximum_pack = add(mul(sizeof(int), add(add(mul(4, maximum_packed_atoms), maximum_packed_structures + 2), maximum_packed_edges)),
                             mul(sizeof(double), add(mul(24, maximum_packed_atoms), mul(3, maximum_packed_edges))));
    maximum_pack = add(maximum_pack, mul(maximum_packed_structures, sizeof(Atoms) + 2 * sizeof(void *) + 2 * sizeof(int)));
    // The EFS visitor retains per-container capacity between unequal batches.
    // Independent shape maxima cover coexistence; factor three covers vector
    // geometric capacity and old/new allocation overlap.
    maximum_pack = mul(3, maximum_pack);
    size_t efs_rows = 0;
    for (const auto &a : data) {
        if (weight_eng > 0 && a.has_energy()) efs_rows = std::max(efs_rows, maximum_packed_structures);
        if (!linear_fit_eng_only && weight_force > 0 && a.has_forces()) efs_rows = std::max(efs_rows, mul(3, maximum_packed_atoms));
        if (!linear_fit_eng_only && weight_stress > 0 && a.has_stresses()) efs_rows = std::max(efs_rows, mul(9, maximum_packed_structures));
    }
    efs_rows = std::min(efs_rows, GpuLinearOptions().gpu_workspace_bytes / mul(c, sizeof(double)));
    maximum_pack = add(maximum_pack, add(mul(mul(efs_rows, c), sizeof(double)), mul(efs_rows, sizeof(Training::LinearEFSObservationRow))));
    for (int count : {potential.alpha_moments_count, potential.alpha_index_basic_count,
                      potential.alpha_scalar_moments, potential.rb_size, potential.max_mu, potential.max_alpha_index_basic,
                      potential.species_count, potential.radial_func_count})
        if (count < 0) GPERROR("Invalid GPU trainmtp curvature model dimensions");
    const size_t m = size_t(potential.alpha_moments_count), b = size_t(potential.alpha_index_basic_count);
    const size_t s = size_t(potential.alpha_scalar_moments), r = size_t(potential.rb_size);
    const size_t u = size_t(potential.max_mu), alpha = size_t(potential.max_alpha_index_basic);
    const size_t threads = size_t(std::max(1, get_omp_max_threads()));
    // Complete model-dependent matrix-free CPU oracle scratch, including the
    // moment Jacobian/basic 3x3 blocks and radial second-derivative temporaries.
    size_t base = add(add(mul(3, alpha + 1), alpha + 6), add(mul(2, m), add(mul(4, r), mul(4, u))));
    size_t per_thread = add(base, add(mul(add(add(mul(3, m), mul(9, b)), 9), maximum_neighbors),
        add(mul(2, m), add(mul(2, s + 1), mul(4, r + 2)))));
    size_t cpu_bytes = add(mul(threads, sizeof(Potentials::PotMTP::ompbuffers)), mul(mul(threads + 1, per_thread), sizeof(double)));
    size_t retained = curvature_cpu_retained(potential);
    retained = add(retained, mul(potential.omp_buffers_list.capacity(), sizeof(Potentials::PotMTP::ompbuffers)));
    for (const auto &buffer : potential.omp_buffers_list)
        retained = add(retained, curvature_cpu_retained(buffer));
    // A thread-count change can copy old buffers while growing the pool.
    cpu_bytes = add(cpu_bytes, mul(2, retained));
    cpu_bytes = add(cpu_bytes, mul(add(potential.reg_L1_vector.capacity(), potential.reg_L2_vector.capacity()), sizeof(double)));
    if (allocating) {
        // Legacy train_init's two radial-parameter Jacobians are retained even
        // though this linear consumer only changes the final coefficient head.
        const size_t jac = mul(mul(mul(b, size_t(potential.species_count)), size_t(potential.radial_func_count)), r);
        cpu_bytes = add(cpu_bytes, mul(add(mul(2, jac), add(mul(2, m), mul(2, full))), sizeof(double)));
    }
    return add(bytes, add(cpu_bytes, add(maximum_frame, maximum_pack)));
}

void Trainer_MTP::CheckGpuCurvatureHost(Potentials::PotMTP &potential,
    const std::vector<Atoms> &data, const std::vector<Neighborhoods> &neighbors, bool initialized) {
    const size_t cap = GpuCurvatureHostCap();
    size_t bytes = gpu_validation_checked_add(GpuCurvatureOwnedBytes(), GpuCurvatureReserve(potential, data, neighbors, !initialized));
    if (!initialized) {
        bytes = gpu_validation_checked_add(bytes, Training::gpu_validation_atoms_bytes(data, true));
        bytes = gpu_validation_checked_add(bytes, Training::gpu_validation_neighbors_bytes(neighbors, true));
        bytes = gpu_validation_checked_add(bytes, prediction_geometry_bytes(data));
    }
    if (bytes > cap) GPERROR("GPU trainmtp curvature owned host storage exceeds gpu_curvature_host_mib");
    gpu_curvature_host_peak_ = std::max(gpu_curvature_host_peak_, bytes);
    gpu_curvature_external_bytes_ = bytes;
}

void Trainer_MTP::ValidateGpuSettings() const {
    if (!UsesGpuLinear()) return;
    GpuLinearOptions().validate();
    if (radial_reg_type != 0 || do_ladder)
        GPERROR("GPU trainmtp requires radial_mode=0 without ladder topology changes");
    hvp_fit.validate();
    Training::HVPFitSettings{weight_hessian, weight_hessian_power, hessian_fd_step}.validate();
    if (UsesGpuCurvature()) {
        if (GpuLinearOptions().device != "cuda") GPERROR("GPU trainmtp curvature observations require CUDA");
        if (linear_fit_eng_only && (weight_hessian > 0 || hvp_fit.weight > 0))
            GPERROR("GPU trainmtp curvature fitting cannot be combined with lfit_eng_only");
        (void)GpuCurvatureHostCap();
    }

    if (linear_precondition != "none" && linear_precondition != "diagonal")
        GPERROR("Unknown linear_precondition: " + linear_precondition);
    if (fix_linear_coeffs && fix_radial_coeffs) GPERROR("Nothing to be optimized");
    for (double value : {weight_eng, weight_force, weight_stress, reg_L1, reg_L2})
        if (!std::isfinite(value) || value < 0) GPERROR("GPU trainmtp requires finite nonnegative channel weights and L2");
    for (double value : {weight_eng_power, weight_force_power, weight_stress_power,
                         weight_force_norm_d, weight_force_norm_f, weight_force_norm_p})
        if (!std::isfinite(value)) GPERROR("GPU trainmtp requires finite weighting parameters");
    if (weight_force_norm_f < 0 || (weight_force_norm_f > 0 && weight_force_norm_d <= 0))
        GPERROR("GPU trainmtp force normalization requires f >= 0 and positive d when enabled");
}

void Trainer_MTP::ValidateGpuLinearSettings() const {
    ValidateGpuSettings();
    if (UsesGpu() && reg_L1 != 0)
        GPERROR("GPU trainmtp linear fitting does not support L1 regularization");
}

void Trainer_MTP::RejectGpuLinearAction(const char *action) const {
    int requested = UsesGpuLinear(), any = requested;
    MPI_Allreduce(&requested, &any, 1, MPI_INT, MPI_MAX, mpi.comm);
    if (any) GPERROR(std::string("Unsupported GPU trainmtp action: ") + action);
}

double Trainer_MTP::GpuLinearNormalization(const Atoms &atom, bool fast) const {
    // Preserve the legacy fast full-EFS missing-force branch, including p=0.
    if (weight_force_norm_f <= 0 || (fast && !linear_fit_eng_only && !atom.has_forces())) return 1.;
    double average = 0;
    if (atom.has_forces()) {
        for (const auto &force : atom.forces_) average += force.squaredNorm();
        average /= atom.size();
    }
    const double weight = weight_force_norm_d /
        (weight_force_norm_d + std::pow(weight_force_norm_f * average, weight_force_norm_p));
    if (!std::isfinite(weight) || weight <= 0) GPERROR("Invalid GPU trainmtp force normalization");
    return weight;
}

bool Trainer_MTP::PreflightGpuLinear(Potentials::PotMTP *potential,
    const std::vector<Atoms> &data, const std::vector<Neighborhoods> &neighbors, bool initialized, bool nonlinear) {
    if (!nonlinear) gpu_linear_report_.verified = false;
    int requested = UsesGpuLinear(), any = requested, all = requested;
    MPI_Allreduce(&requested, &any, 1, MPI_INT, MPI_MAX, mpi.comm);
    MPI_Allreduce(&requested, &all, 1, MPI_INT, MPI_MIN, mpi.comm);
    if (any != all) GPERROR("All trainmtp ranks must select the same observation backend");
    if (!any) return false;
    const size_t local_frames = data.size();
    gpu_curvature_aligned_size_ = local_frames;
#ifdef IMRMLP_MPI
    MPI_Allreduce(&local_frames, &gpu_curvature_aligned_size_, 1, MPI_UNSIGNED_LONG, MPI_MAX, mpi.comm);
#endif
    std::string signature;
    Training::collective_try([&] {
        if (nonlinear) ValidateGpuNonlinearSettings();
        else if (initialized) ValidateGpuLinearSettings();
        else ValidateGpuSettings();
        gpu_linear_options_ = GpuLinearOptions();
        if (!potential || typeid(*potential) != typeid(Potentials::PotMTP) || potential->radial_mode != 0)
            GPERROR("GPU trainmtp requires ordinary PotMTP with radial_mode=0");
        if (initialized) {
            RequireCurrentParameterLayout();
            const int expected = potential->get_coeff_size() - (fix_radial_coeffs ? potential->radial_coeff_size : 0)
                - (fix_linear_coeffs ? potential->lin_coeff_size : 0);
            if (nonlinear && expected != n_relax_coeffs)
                GPERROR("GPU trainmtp parameter selection changed; reinitialize the trainer");
            if (gpu_linear_allocation_signature_ != GpuLinearAllocationSignature())
                GPERROR("GPU trainmtp allocation or cached-weight settings changed; reinitialize the trainer");
            if (!nonlinear && fast_lfit && (!weight_list.allFinite() || (weight_list.size() && weight_list.minCoeff() < 0)))
                GPERROR("Invalid GPU trainmtp cached observation weights");
        }
        if (potential->hc_correction || potential->rR_correction || potential->add_efs)
            GPERROR("GPU trainmtp requires ordinary MTP without corrections or additive evaluation");
        if (data.size() != neighbors.size()) GPERROR("GPU trainmtp requires one neighborhood set per structure");
        if (data.size() > size_t(std::numeric_limits<int>::max())) GPERROR("GPU trainmtp structure count exceeds the native index range");
        if (!nonlinear && potential->lin_coeff_size <= (fix_species_coeffs ? potential->species_count : 0))
            GPERROR("GPU trainmtp has no selected linear coefficients");
        Eigen::VectorXd coefficients;
        potential->get_coeffs(coefficients);
        if (!coefficients.allFinite()) GPERROR("GPU trainmtp requires finite potential coefficients");
        if (initialized) {
            if (potential->reg_L1_vector.size() != size_t(potential->lin_coeff_size) ||
                potential->reg_L2_vector.size() != size_t(potential->lin_coeff_size))
                GPERROR("GPU trainmtp regularization layout changed; reinitialize the trainer");
            for (int i = 0; i < potential->lin_coeff_size; ++i) {
                if (!std::isfinite(potential->reg_L1_vector[i]) || potential->reg_L1_vector[i] < 0 || (!nonlinear && potential->reg_L1_vector[i] != 0) ||
                    !std::isfinite(potential->reg_L2_vector[i]) || potential->reg_L2_vector[i] < 0)
                    GPERROR("GPU trainmtp requires zero L1 and finite nonnegative effective L2");
            }
            if (!train_set_size_all) GPERROR("GPU trainmtp requires a nonempty global training set");
        }
        for (size_t k = 0; k < data.size(); ++k) {
            const auto &atom = data[k];
            (void)curvature_dof(atom); // Before atom.size() narrowing or validate_hvp's int probe loop.
            if (atom.size() <= 0 || atom.chemmap_.size() != size_t(atom.size()) ||
                neighbors[k].size() != atom.size()) GPERROR("Invalid GPU trainmtp structure or neighborhood shape");
            atom.validate_hvp();
            Training::validate_hessian_fit_label(atom);
            if (!atom.cell_.allFinite() || !std::isfinite(atom.fitness) || atom.fitness < 0)
                GPERROR("GPU trainmtp requires finite geometry and nonnegative fitness");
            if (atom.atomic_fitness.size() && atom.atomic_fitness.size() != size_t(atom.size()))
                GPERROR("Invalid GPU trainmtp atomic fitness count");
            for (double value : atom.atomic_fitness)
                if (!std::isfinite(value) || value < 0) GPERROR("GPU trainmtp requires finite nonnegative atomic fitness");
            for (int i = 0; i < atom.size(); ++i) {
                if (!atom.pos_[i].allFinite() || atom.chemmap_[i] < 0 || atom.chemmap_[i] >= potential->species_count ||
                    neighbors[k][i].centertype != atom.chemmap_[i]) GPERROR("Invalid GPU trainmtp position or species");
            }
            if ((atom.has_energy() && !std::isfinite(atom.energy_)) ||
                (atom.has_stresses() && !atom.stress_.allFinite()) ||
                (atom.has_forces() && atom.forces_.size() != size_t(atom.size())))
                GPERROR("Invalid GPU trainmtp EFS labels");
            for (const auto &force : atom.forces_)
                if (!force.allFinite()) GPERROR("GPU trainmtp requires finite force labels");
            const double normalization = GpuLinearNormalization(atom, fast_lfit);
            for (double weight : {energy_weight(atom) * normalization,
                                  force_weight(atom) * normalization, stress_weight(atom) * normalization,
                                  hessian_weight(atom) * normalization, hvp_fit.observation_weight(atom, normalization)})
                if (!std::isfinite(weight) || weight < 0) GPERROR("Invalid GPU trainmtp observation weight");
        }
        const bool curvature = UsesGpuCurvature() || std::any_of(data.begin(), data.end(), [](const Atoms &a) {
            return a.have_hessian || a.hvp_directions_.size() || a.hvp_targets_.size();
        });
        if (curvature) {
            if (nonlinear) GPERROR("Nonlinear GPU trainmtp does not support Hessian/HVP labels");
            if (gpu_linear_options_.device != "cuda") GPERROR("GPU trainmtp curvature observations require CUDA");
            if (potential->hc_correction || potential->rR_correction || potential->add_efs)
                GPERROR("GPU trainmtp curvature requires ordinary MTP without corrections or additive evaluation");
            if (weight_hessian > 0 || hvp_fit.weight > 0) HessianGpu::validate_analytic_radial_bases(*potential);
            const size_t rows = GpuCurvatureRowCount(data);
            if (initialized && rows != size_t(n_lfit_data))
                GPERROR("GPU trainmtp curvature observation layout changed; reinitialize the trainer");
            if (initialized && fast_lfit) {
                size_t cursor = 0;
                auto check_weight = [&](size_t count, double weight) {
                    const double expected = std::sqrt(weight);
                    if (!std::isfinite(expected) || count > size_t(weight_list.size()) - cursor)
                        GPERROR("GPU trainmtp cached curvature row count changed; reinitialize the trainer");
                    for (size_t j = 0; j < count; ++j)
                        if (weight_list[Eigen::Index(cursor++)] != expected)
                            GPERROR("GPU trainmtp cached observation weights changed; reinitialize the trainer");
                };
                for (const auto &a : data) {
                    const double cc = GpuLinearNormalization(a, true);
                    if (weight_force > 0 && a.has_forces()) for (size_t i = 0; i < a.pos_.size(); ++i) {
                        const double w = force_weight(a) * cc * (a.atomic_fitness.empty() ? 1. : a.atomic_fitness[i]);
                        if (w > 0) check_weight(3, w);
                    }
                    if (weight_stress > 0 && a.has_stresses()) check_weight(9, stress_weight(a) * cc);
                    if (weight_hessian > 0 && a.has_hessian()) check_weight(upper_rows(curvature_dof(a)), hessian_weight(a) * cc);
                    if (hvp_fit.weight > 0 && a.has_hvp()) check_weight(size_t(a.hvp_directions_.size()), hvp_fit.observation_weight(a, cc));
                    if (weight_eng > 0 && a.has_energy()) check_weight(1, energy_weight(a) * cc);
                }
                if (cursor != size_t(weight_list.size())) GPERROR("GPU trainmtp cached observation row count changed; reinitialize the trainer");
            }
            CheckGpuCurvatureHost(*potential, data, neighbors, initialized);
        }
        // Exclude rank-local batching and GPU placement, but require one mathematical problem.
        std::ostringstream out;
        out << std::setprecision(17) << gpu_linear_options_.device << ' ' << fast_lfit << ' '
            << linear_fit_eng_only << ' ' << fix_species_coeffs << ' ' << fix_linear_coeffs << ' '
            << fix_radial_coeffs << ' ' << do_orthgonalization << ' ' << safe_linear_fitting << ' '
            << linear_precondition << ' ' << weight_eng << ' ' << weight_force << ' ' << weight_stress << ' '
            << weight_eng_power << ' ' << weight_force_power << ' ' << weight_stress_power << ' '
            << weight_hessian << ' ' << weight_hessian_power << ' ' << hvp_fit.weight << ' ' << hvp_fit.power << ' '
            << hessian_fd_step << ' ' << hvp_fit.step << ' '
            << weight_force_norm_d << ' ' << weight_force_norm_f << ' ' << weight_force_norm_p << ' ' << reg_L2 << ' '
            << potential->species_count << ' ' << potential->radial_coeff_size << ' ' << potential->lin_coeff_size;
        for (double value : coefficients) out << ' ' << value;
        if (initialized) {
            for (double value : potential->reg_L2_vector) out << ' ' << value;
            out << ' ' << reg_init;
        }
        // Native serialization also captures radial basis parameters and the moment graph.
        potential->Save(out);
        if (potential->radial_basis_split)
            for (const auto *basis : potential->p_radical_basis_pair) out << ' ' << basis->scaling;
        if (nonlinear) {
            // Mathematical and optimizer settings must agree; local device and
            // batching choices deliberately remain rank-local.
            for (const auto &[key, value] : CurrentSettings().settings)
                if (key != "gpu_id" && key != "gpu_batch_atoms" && key != "gpu_batch_structures" &&
                    key != "gpu_memory_mib" && key != "gpu_workspace_mib" && key != "device")
                    out << '\n' << key << '=' << value;
            for (double value : potential->reg_L1_vector) out << ' ' << value;
        }
        signature = out.str();
        if (signature.size() > size_t(std::numeric_limits<int>::max())) GPERROR("GPU trainmtp MPI signature exceeds count range");
    });
#ifdef IMRMLP_MPI
    int length = mpi.rank == 0 ? static_cast<int>(signature.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
    std::string reference;
    Training::collective_try([&] { reference = mpi.rank == 0 ? signature : std::string(length, '\0'); });
    MPI_Bcast(reference.data(), length, MPI_CHAR, 0, mpi.comm);
    Training::collective_check(signature == reference ? "" : "GPU trainmtp ranks disagree on model or linear fitting settings");
#endif
    size_t local_count = data.size(), global_count = local_count;
#ifdef IMRMLP_MPI
    MPI_Allreduce(&local_count, &global_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, mpi.comm);
#endif
    if (!global_count) GPERROR("GPU trainmtp requires a nonempty global training set");
    size_t local_rows = 0, global_rows = 0;
    Training::collective_try([&] { local_rows = GpuCurvatureRowCount(data); });
    global_rows = local_rows;
#ifdef IMRMLP_MPI
    MPI_Allreduce(&local_rows, &global_rows, 1, MPI_UNSIGNED_LONG, MPI_SUM, mpi.comm);
#endif
    if (global_rows > size_t(std::numeric_limits<int>::max())) GPERROR("GPU trainmtp global observation count exceeds the native index range");
#ifdef IMRMLP_GPU
    int device = gpu_linear_options_.gpu_id;
    std::string error;
#ifdef IMRMLP_MPI
    if (!MLIPCore::GpuDirect::select_mpi_device(mpi.comm, device, &device, &error))
        GPERROR("Unable to select GPU for trainmtp: " + error);
#else
    if (device < 0) device = 0;
    Training::collective_try([&] {
        int count = 0;
        if (!MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count, &error)) GPERROR(error);
        if (device < 0 || device >= count) GPERROR("GPU trainmtp device index is outside the visible device range");
    });
#endif
    gpu_linear_options_.gpu_id = device;
#endif
    return true;
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
    if (!gpu_nonlinear_running_ &&
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true)) {
        RunGpuNonlinear([&] { Rescale(); });
        return;
    }
    RequireCurrentParameterLayout();
    if (UsesGpu() && gpu_nonlinear_running_ && fix_linear_coeffs) return;
    if (UsesGpu() && gpu_nonlinear_running_ && fix_radial_coeffs) {
        TrainLinear();
        return;
    }
    if (rescale_mode == "fixed") {
        Message("Rescale: fixed radial scaling; one linear fit");
        TrainLinear();
        return;
    }
    if (rescale_mode != "search") GPERROR("Unknown rescale_mode: " + rescale_mode);
    // Split radial models have no shared p_radical_basis. Search a common
    // multiplicative scale while preserving each pair's relative scaling.
    auto get_scaling = [&] {
        return p_mtp->radial_basis_split
            ? p_mtp->p_radical_basis_pair.front()->scaling
            : p_mtp->p_radical_basis->scaling;
    };
    auto set_scaling = [&](double value) {
        if (p_mtp->radial_basis_split) {
            const double ratio = value / get_scaling();
            for (auto *basis : p_mtp->p_radical_basis_pair) basis->scaling *= ratio;
        } else p_mtp->p_radical_basis->scaling = value;
    };
    double min_scaling = get_scaling();
    double max_scaling = get_scaling();
    int ind;
    int soffset = 0;
    if (fix_species_coeffs) soffset = p_mtp->species_count;
    do {
        double condition_number[5];
        double scaling = get_scaling();
        double scalings[5] = {scaling / 1.2, scaling / 1.1, scaling, scaling * 1.1, scaling * 1.2}; // TODO: what's this
        std::vector<double> coeffs(n_size_mat); // linear coeffs
        for (int j = 0; j < 5; ++j) {
            Message("Rescaling to " + std::to_string(scalings[j]));
            set_scaling(scalings[j]);
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
        set_scaling(scalings[ind]);
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
    if (UsesGpuLinear()) GPERROR("GPU trainmtp row assembly requires TrainLinear, TrainLinearAlg or TrainLinearFast");
    RequireCurrentParameterLayout();
    Atoms &atm = training_set_[ind];
    Neighborhoods &nbhs = trainint_set_nbhs_[ind];
    p_mtp->CalcEFSCmpt(atm, nbhs, energy_cmpts, force_cmpts, stress_cmpts);
    // DebugWrite(fmt::format("{}\n", fmt::join(energy_cmpts, " ")));

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

    // hessian

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
    // Hessian
    if (weight_hessian > 0 && atm.has_hessian()) {
        const double weight = hessian_weight(atm) * cc;
        const int ndof = 3 * atm.size();
        Eigen::VectorXd direction = Eigen::VectorXd::Zero(ndof);
        Eigen::MatrixXd components;
        for (int col = 0; col < ndof; ++col) {
            direction[col] = 1.;
            p_mtp->CalcHessianVectorCmpt(atm, nbhs, direction, components);
            direction[col] = 0.;
            for (int row = 0; row <= col; ++row) {
                for (int i = 0; i < n_size_mat; ++i) {
                    const double hi = components(row, i + soffset);
                    for (int j = i; j < n_size_mat; ++j)
                        lin_matrix(i, j) += weight * hi * components(row, j + soffset);
                    lin_vector[i] += weight * hi * atm.hessian_(row, col);
                }
            }
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
    if (UsesGpuLinear()) GPERROR("GPU trainmtp row assembly requires TrainLinear, TrainLinearAlg or TrainLinearFast");
    RequireCurrentParameterLayout();
    Atoms &atm = training_set_[ind];
    Neighborhoods &nbhs = trainint_set_nbhs_[ind];
    p_mtp->CalcEFSCmpt_R(atm, nbhs, energy_cmpts, force_cmpts_R, stress_cmpts_R);
    // DebugWrite(fmt::format("{}\n", fmt::join(energy_cmpts, " ")));

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
    // Hessian
    if (weight_hessian > 0 && atm.has_hessian()) {
        const int ndof = 3 * atm.size();
        const int start = position_ladd;
        Eigen::VectorXd direction = Eigen::VectorXd::Zero(ndof);
        Eigen::MatrixXd components;
        for (int col = 0; col < ndof; ++col) {
            direction[col] = 1.;
            p_mtp->CalcHessianVectorCmpt(atm, nbhs, direction, components);
            direction[col] = 0.;
            for (int row = 0; row <= col; ++row) {
                const int index = start + row * ndof - row * (row - 1) / 2 + col - row;
                for (int i = 0; i < n_size_mat; ++i)
                    eigen_lin_matrix_A(index, i) = weight_list(index) * components(row, i + soffset);
                eigen_lin_vector_A[index] = weight_list(index) * atm.hessian_(row, col);
            }
        }
        position_ladd += ndof * (ndof + 1) / 2;

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

void Trainer_MTP::AssembleGpuLinear(bool fast) {
    const int offset = fix_species_coeffs ? p_mtp->species_count : 0;
    std::vector<int> columns(n_size_mat);
    std::iota(columns.begin(), columns.end(), offset);
    std::vector<std::array<bool, 3>> requested(training_set_.size());
    std::vector<std::array<bool, 2>> curvature_requested(training_set_.size());
    // Per-configuration row offsets preserve the existing fast A layout (F,V,H,HVP,E),
    // independently of the GPU visitor's E,F,V blocks and physical batches.
    struct Rows {
        std::vector<int> forces;
        int virial = -1, energy = -1;
        int hessian = -1, hvp = -1;
        double energy_weight = 0, force_weight = 0, stress_weight = 0;
        double hessian_weight = 0, hvp_weight = 0;
    };
    std::vector<Rows> rows(training_set_.size());
    int cursor = 0;
    for (size_t k = 0; k < training_set_.size(); ++k) {
        const auto &atom = training_set_[k];
        const double normalization = GpuLinearNormalization(atom, fast);
        rows[k].energy_weight = energy_weight(atom) * normalization;
        rows[k].force_weight = force_weight(atom) * normalization;
        rows[k].stress_weight = stress_weight(atom) * normalization;
        rows[k].hessian_weight = hessian_weight(atom) * normalization;
        rows[k].hvp_weight = hvp_fit.observation_weight(atom, normalization);
        curvature_requested[k] = {!linear_fit_eng_only && weight_hessian > 0 && atom.has_hessian(),
                                  !linear_fit_eng_only && hvp_fit.weight > 0 && atom.has_hvp()};
        requested[k] = {weight_eng > 0 && atom.has_energy(),
            !linear_fit_eng_only && weight_force > 0 && atom.has_forces(),
            !linear_fit_eng_only && weight_stress > 0 && atom.has_stresses()};
        if (fast && !linear_fit_eng_only) {
            auto &indices = rows[k];
            indices.forces.assign(atom.size(), -1);
            if (requested[k][1]) for (int i = 0; i < atom.size(); ++i)
                if (atom.atomic_fitness.empty() || atom.atomic_fitness[i] > 0) {
                    indices.forces[i] = cursor; cursor += 3;
                }
            if (requested[k][2]) { indices.virial = cursor; cursor += 9; }
            if (curvature_requested[k][0]) { indices.hessian = cursor; cursor += static_cast<int>(upper_rows(curvature_dof(atom))); }
            if (curvature_requested[k][1]) { indices.hvp = cursor; cursor += static_cast<int>(atom.hvp_directions_.size()); }
            if (requested[k][0]) indices.energy = cursor++;
        }
    }
    if (fast && !linear_fit_eng_only && cursor != n_lfit_data)
        GPERROR("GPU trainmtp observation layout changed; reinitialize the trainer");
    gpu_linear_target_squared_ = 0;
    Training::visit_gpu_linear_efs(*p_mtp, training_set_, requested, columns,
        gpu_linear_options_, gpu_linear_report_.gpu_stats, {},
        [&](const Training::LinearEFSObservationBlock &block) {
            for (size_t r = 0; r < block.rows.size(); ++r) {
                const auto &ref = block.rows[r];
                const auto &atom = training_set_[ref.configuration];
                const auto values = block.values.subspan(r * n_size_mat, n_size_mat);
                double target = 0, weight = 0;
                int index = -1;
                const auto &weights = rows[ref.configuration];
                if (ref.channel == 0) {
                    target = atom.energy_;
                    if (fix_species_coeffs) for (int type : atom.chemmap_) target -= p_mtp->species_coeffs[type];
                    weight = weights.energy_weight;
                    index = rows[ref.configuration].energy;
                } else if (ref.channel == 1) {
                    target = atom.forces_[ref.atom][ref.component];
                    weight = weights.force_weight;
                    if (!atom.atomic_fitness.empty()) weight *= atom.atomic_fitness[ref.atom];
                    if (weight <= 0) continue;
                    if (fast && !linear_fit_eng_only) index = rows[ref.configuration].forces[ref.atom] + ref.component;
                } else {
                    target = atom.stress_(ref.component / 3, ref.component % 3);
                    weight = weights.stress_weight;
                    index = rows[ref.configuration].virial + ref.component;
                }
                for (double value : values) if (!std::isfinite(value)) GPERROR("Non-finite GPU trainmtp observation");
                if (fast && !linear_fit_eng_only) {
                    if (index < 0 || index >= n_lfit_data) GPERROR("Invalid GPU trainmtp design row");
                    // Match the legacy fast path's cached sqrt weights from allocate().
                    const double root_weight = weight_list[index];
                    weight = root_weight * root_weight;
                    for (int i = 0; i < n_size_mat; ++i) eigen_lin_matrix_A(index, i) = root_weight * values[i];
                    eigen_lin_vector_A[index] = root_weight * target;
                } else for (int i = 0; i < n_size_mat; ++i) {
                    if (fast) eigen_lin_vector[i] += weight * values[i] * target;
                    else lin_vector[i] += weight * values[i] * target;
                    for (int j = i; j < n_size_mat; ++j) {
                        const double product = weight * values[i] * values[j];
                        if (fast) eigen_lin_matrix(i, j) += product;
                        else lin_matrix(i, j) += product;
                    }
                }
                gpu_linear_target_squared_ += static_cast<long double>(weight) * target * target;
            }
        }, &trainint_set_nbhs_);
    if (UsesGpuCurvature()) {
        // No complete per-probe matrix is retained. Each callback has a whole
        // selected row so ordinary Gram cross-products include every column.
        CheckGpuCurvatureHost(*p_mtp, training_set_, trainint_set_nbhs_, true);
        Training::visit_gpu_linear_curvature(*p_mtp, training_set_, curvature_requested, columns,
            gpu_linear_options_, gpu_linear_report_.curvature_stats,
            [&](const Training::LinearCurvatureObservationBlock &block) {
                if (block.configuration >= training_set_.size() || block.native_columns.size() != columns.size() ||
                    !std::equal(columns.begin(), columns.end(), block.native_columns.begin()))
                    GPERROR("Invalid GPU trainmtp curvature block identity");
                const auto &atom = training_set_[block.configuration];
                const auto &info = rows[block.configuration];
                const size_t d = curvature_dof(atom);
                for (int r = 0; r < block.rows; ++r) {
                    const size_t coordinate = size_t(block.first_row + r);
                    const auto values = block.values.subspan(size_t(r) * n_size_mat, n_size_mat);
                    if (coordinate >= d || block.probe < 0 ||
                        (block.hessian ? size_t(block.probe) >= d || coordinate > size_t(block.probe)
                                       : block.probe >= atom.hvp_targets_.cols()))
                        GPERROR("Invalid GPU trainmtp curvature coordinate");
                    const double target = block.hessian ? atom.hessian_(coordinate, block.probe)
                                                         : atom.hvp_targets_(coordinate, block.probe);
                    double weight = block.hessian ? info.hessian_weight : info.hvp_weight;
                    for (double v : values) if (!std::isfinite(v)) GPERROR("Non-finite GPU trainmtp curvature observation");
                    if (fast) {
                        const size_t offset = block.hessian ? coordinate * d - (coordinate ? coordinate * (coordinate - 1) / 2 : 0)
                                                               + size_t(block.probe) - coordinate
                                                           : size_t(block.probe) * d + coordinate;
                        const int start = block.hessian ? info.hessian : info.hvp;
                        if (start < 0 || offset >= size_t(n_lfit_data) - size_t(start))
                            GPERROR("Invalid GPU trainmtp curvature design row");
                        const size_t index = size_t(start) + offset;
                        const double root_weight = weight_list[Eigen::Index(index)];
                        weight = root_weight * root_weight;
                        for (int i = 0; i < n_size_mat; ++i) eigen_lin_matrix_A(index, i) = root_weight * values[i];
                        eigen_lin_vector_A[index] = root_weight * target;
                    } else for (int i = 0; i < n_size_mat; ++i) {
                        lin_vector[i] += weight * values[i] * target;
                        for (int j = i; j < n_size_mat; ++j) lin_matrix(i, j) += weight * values[i] * values[j];
                    }
                    gpu_linear_target_squared_ += static_cast<long double>(weight) * target * target;
                }
            }, &trainint_set_nbhs_, GpuCurvatureHostCap(), gpu_curvature_external_bytes_);
        gpu_curvature_host_peak_ = std::max(gpu_curvature_host_peak_, gpu_linear_report_.curvature_stats.host_workspace_bytes);
        gpu_linear_report_.curvature_host_peak_bytes = gpu_curvature_host_peak_;
    }
    if (!std::isfinite(gpu_linear_target_squared_)) GPERROR("Non-finite GPU trainmtp target norm");
    const size_t count = size_t(n_size_mat) + size_t(n_size_mat) * (n_size_mat + size_t{1}) / 2;
    if (count > size_t(std::numeric_limits<int>::max())) GPERROR("GPU trainmtp reduction exceeds MPI count range");
    // Allocate before leaving the local collective failure boundary.
    linear_reduce_send.resize(count);
    linear_reduce_receive.resize(mpi.rank == 0 ? count : 0);
    if (fast) {
        if (!linear_fit_eng_only && (!eigen_lin_matrix_A.allFinite() || !eigen_lin_vector_A.allFinite()))
            GPERROR("Non-finite GPU trainmtp weighted design matrix");
        gpu_linear_report_.host_design_bytes = size_t(eigen_lin_matrix_A.size() + eigen_lin_vector_A.size()) * sizeof(double);
    }
}

double Trainer_MTP::DirectGpuLinearDataLoss(bool fast) {
    double local = 0, global = 0;
    Training::collective_try([&] {
        // These optional model caches are unnecessary for the bounded serial
        // EFS verification. Restore policy even when a later HVP throws.
        struct Restore {
            Potentials::PotMTP &p; bool symmetry, radial;
            ~Restore() { p.full_symmetry_reuse = symmetry; p.radial_local_cache = radial; }
        } restore{*p_mtp, p_mtp->full_symmetry_reuse, p_mtp->radial_local_cache};
        if (UsesGpuCurvature()) { p_mtp->full_symmetry_reuse = false; p_mtp->radial_local_cache = false; }
        int cursor = 0;
        for (size_t k = 0; k < training_set_.size(); ++k) {
            const auto &reference = training_set_[k];
            auto prediction = UsesGpuCurvature() ? Training::gpu_hvp_geometry_copy(reference) : reference;
            prediction.resetEFS();
            if ((weight_eng > 0 && reference.has_energy()) || (!linear_fit_eng_only &&
                ((weight_force > 0 && reference.has_forces()) || (weight_stress > 0 && reference.has_stresses()))))
                p_mtp->getEFS(prediction, trainint_set_nbhs_[k]);
            const double normalization = GpuLinearNormalization(reference, fast);
            double ew = energy_weight(reference) * normalization;
            double sw = stress_weight(reference) * normalization;
            std::vector<double> fw(reference.size(), force_weight(reference) * normalization);
            const double hw = hessian_weight(reference) * normalization;
            const double vw = hvp_fit.observation_weight(reference, normalization);
            int hessian_start = -1, hvp_start = -1;
            for (int i = 0; i < reference.size(); ++i)
                if (!reference.atomic_fitness.empty()) fw[i] *= reference.atomic_fitness[i];
            if (fast && !linear_fit_eng_only) {
                if (weight_force > 0 && reference.has_forces()) for (int i = 0; i < reference.size(); ++i) {
                    if (!reference.atomic_fitness.empty() && reference.atomic_fitness[i] <= 0) continue;
                    if (cursor + 3 > weight_list.size()) GPERROR("GPU trainmtp cached force row mismatch");
                    fw[i] = weight_list[cursor] * weight_list[cursor]; cursor += 3;
                }
                if (weight_stress > 0 && reference.has_stresses()) {
                    if (cursor + 9 > weight_list.size()) GPERROR("GPU trainmtp cached virial row mismatch");
                    sw = weight_list[cursor] * weight_list[cursor]; cursor += 9;
                }
                if (weight_hessian > 0 && reference.has_hessian()) {
                    const size_t count = upper_rows(curvature_dof(reference));
                    if (count > size_t(weight_list.size()) - size_t(cursor)) GPERROR("GPU trainmtp cached Hessian row mismatch");
                    hessian_start = cursor; cursor += static_cast<int>(count);
                }
                if (hvp_fit.weight > 0 && reference.has_hvp()) {
                    if (size_t(reference.hvp_directions_.size()) > size_t(weight_list.size()) - size_t(cursor))
                        GPERROR("GPU trainmtp cached HVP row mismatch");
                    hvp_start = cursor; cursor += static_cast<int>(reference.hvp_directions_.size());
                }
                if (weight_eng > 0 && reference.has_energy()) {
                    if (cursor >= weight_list.size()) GPERROR("GPU trainmtp cached energy row mismatch");
                    ew = weight_list[cursor] * weight_list[cursor]; ++cursor;
                }
            }
            if (weight_eng > 0 && reference.has_energy()) {
                const double error = prediction.energy_ - reference.energy_;
                local += ew * error * error;
            }
            if (!linear_fit_eng_only && weight_force > 0 && reference.has_forces()) {
                if (prediction.forces_.size() != reference.forces_.size()) GPERROR("Invalid CPU trainmtp force prediction count");
                for (int i = 0; i < reference.size(); ++i) {
                    const double weight = fw[i];
                    if (weight > 0) local += weight * (prediction.forces_[i] - reference.forces_[i]).squaredNorm();
                }
            }
            if (!linear_fit_eng_only && weight_stress > 0 && reference.has_stresses())
                local += sw * (prediction.stress_ - reference.stress_).squaredNorm();
            if (!linear_fit_eng_only && ((weight_hessian > 0 && reference.has_hessian()) ||
                                        (hvp_fit.weight > 0 && reference.has_hvp()))) {
                const size_t d = curvature_dof(reference);
                Eigen::VectorXd direction = Eigen::VectorXd::Zero(Eigen::Index(d)), product;
                auto accumulate = [&](bool hessian, int probe) {
                    // Independent native CPU model contraction; neither GPU
                    // basis rows nor finite-step training losses are reused.
                    p_mtp->applyHessianVectorProduct(prediction, trainint_set_nbhs_[k], direction, product);
                    if (product.size() != Eigen::Index(d) || !product.allFinite())
                        GPERROR("Non-finite CPU trainmtp analytic curvature prediction");
                    const size_t count = hessian ? size_t(probe) + 1 : d;
                    for (size_t r = 0; r < count; ++r) {
                        double weight = hessian ? hw : vw;
                        if (fast) {
                            const size_t offset = hessian ? r * d - (r ? r * (r - 1) / 2 : 0) + size_t(probe) - r
                                                         : size_t(probe) * d + r;
                            const size_t index = size_t(hessian ? hessian_start : hvp_start) + offset;
                            weight = weight_list[Eigen::Index(index)] * weight_list[Eigen::Index(index)];
                        }
                        const double target = hessian ? reference.hessian_(r, probe) : reference.hvp_targets_(r, probe);
                        const double error = product[Eigen::Index(r)] - target;
                        local += weight * error * error;
                    }
                };
                if (weight_hessian > 0 && reference.has_hessian()) {
                    direction.setZero();
                    for (int j = 0; j < int(d); ++j) {
                        direction[j] = 1.; accumulate(true, j); direction[j] = 0.;
                    }
                }
                if (hvp_fit.weight > 0 && reference.has_hvp())
                    for (int j = 0; j < reference.hvp_directions_.cols(); ++j) {
                        direction = reference.hvp_directions_.col(j); accumulate(false, j);
                    }
            }
        }
        if (fast && !linear_fit_eng_only && cursor != weight_list.size()) GPERROR("GPU trainmtp cached row count changed; reinitialize the trainer");
        if (!std::isfinite(local) || local < 0) GPERROR("Non-finite CPU trainmtp data objective");
    });
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    if (!std::isfinite(global)) GPERROR("Non-finite global CPU trainmtp data objective");
    return global / double(train_set_size_all);
}

void Trainer_MTP::RunGpuLinear(bool fast) {
    ReleaseGpuNonlinear();
    const auto start = std::chrono::steady_clock::now();
    auto seconds = [](auto begin) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(); };
    Eigen::VectorXd saved, anchor;
    std::vector<double> saved_l1, saved_l2;
    const bool saved_reg_init = reg_init;
    Training::collective_try([&] {
        if (fast != fast_lfit) GPERROR("GPU trainmtp direct solver entry differs from initialized lfit_fastalgo");
        p_mtp->get_coeffs(saved);
        saved_l1 = p_mtp->reg_L1_vector; saved_l2 = p_mtp->reg_L2_vector;
        gpu_linear_report_ = {};
        gpu_linear_report_.backend = gpu_linear_options_.device;
        gpu_linear_report_.algorithm = fast ? "fast/eigen_ldlt" : "ordinary/alglib";
        gpu_linear_report_.options = gpu_linear_options_;
        if (UsesGpuCurvature()) {
            gpu_linear_report_.curvature_host_cap_bytes = GpuCurvatureHostCap();
            gpu_linear_report_.curvature_host_peak_bytes = gpu_curvature_host_peak_;
        }
    });
    try {
        auto phase = std::chrono::steady_clock::now();
        Training::collective_try([&] {
            if (do_orthgonalization && !(gpu_nonlinear_running_ && fix_radial_coeffs)) p_mtp->Orthogonalize();
            p_mtp->get_coeffs(anchor);
            if (!anchor.allFinite()) GPERROR("Non-finite trainmtp coefficients after orthogonalization");
        });
        gpu_linear_report_.orthogonalization_seconds = seconds(phase);
        phase = std::chrono::steady_clock::now();
        gpu_linear_report_.data_before = DirectGpuLinearDataLoss(fast);
        gpu_linear_report_.verification_seconds += seconds(phase);
        gpu_linear_active_ = true;
        if (fast) TrainLinearFast(); else TrainLinearAlg();
        gpu_linear_active_ = false;
        // The CPU solvers adapt ridge only on rank zero. GPU acceptance and
        // subsequent fits must all use the same effective regularization.
        MPI_Bcast(p_mtp->reg_L2_vector.data(), p_mtp->lin_coeff_size, MPI_DOUBLE, 0, mpi.comm);
        int initialized = reg_init;
        MPI_Bcast(&initialized, 1, MPI_INT, 0, mpi.comm);
        reg_init = initialized;
        phase = std::chrono::steady_clock::now();
        gpu_linear_report_.data_after = DirectGpuLinearDataLoss(fast);
        double local_target_squared = static_cast<double>(gpu_linear_target_squared_);
        double target_squared = 0;
        MPI_Allreduce(&local_target_squared, &target_squared, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
        double quadratic[4]{};
        Training::collective_try([&] {
            if (mpi.rank != 0) return;
            const int offset = fix_species_coeffs ? p_mtp->species_count : 0;
            auto evaluate = [&](bool candidate, double reference, int position) {
                long double value = target_squared, magnitude = std::abs(target_squared);
                for (int i = 0; i < n_size_mat; ++i) {
                    const long double ci = candidate ? p_mtp->get_linear_coeff(offset + i) :
                        anchor[p_mtp->radial_coeff_size + offset + i];
                    const long double rhs = 2 * ci * gpu_linear_rhs_[i];
                    value -= rhs; magnitude += std::abs(rhs);
                    for (int j = 0; j < n_size_mat; ++j) {
                        const long double cj = candidate ? p_mtp->get_linear_coeff(offset + j) :
                            anchor[p_mtp->radial_coeff_size + offset + j];
                        const long double term = ci * gpu_linear_gram_(i, j) * cj;
                        value += term; magnitude += std::abs(term);
                    }
                }
                quadratic[position] = static_cast<double>(value / train_set_size_all);
                // The quadratic is evaluated in extended precision, but the legacy
                // Gram/RHS are FP64 sums. Bound their cancellation using the absolute
                // terms and row/column counts, separately from the direct-loss tolerance.
                quadratic[position + 2] = 1e-10 * std::max(1., std::abs(reference)) +
                    64 * std::numeric_limits<double>::epsilon() *
                    (double(n_lfit_data_all) + n_size_mat + 1) *
                    static_cast<double>(magnitude / train_set_size_all);
                if (!std::isfinite(quadratic[position]) || !std::isfinite(quadratic[position + 2]))
                    GPERROR("Non-finite GPU trainmtp quadratic verification");
            };
            evaluate(false, gpu_linear_report_.data_before, 0);
            evaluate(true, gpu_linear_report_.data_after, 1);
        });
        MPI_Bcast(quadratic, 4, MPI_DOUBLE, 0, mpi.comm);
        gpu_linear_report_.predicted_data_before = quadratic[0];
        gpu_linear_report_.predicted_data_after = quadratic[1];
        gpu_linear_report_.quadratic_tolerance_before = quadratic[2];
        gpu_linear_report_.quadratic_tolerance_after = quadratic[3];
        Training::collective_try([&] {
            if (std::abs(gpu_linear_report_.data_before - quadratic[0]) > quadratic[2] ||
                std::abs(gpu_linear_report_.data_after - quadratic[1]) > quadratic[3])
                GPERROR("GPU trainmtp observations failed independent CPU quadratic verification");
            const int offset = fix_species_coeffs ? p_mtp->species_count : 0;
            double before = gpu_linear_report_.data_before, after = gpu_linear_report_.data_after;
            for (int i = offset; i < p_mtp->lin_coeff_size; ++i) {
                const double ridge = p_mtp->reg_L2_vector[i];
                const double old = anchor[p_mtp->radial_coeff_size + i];
                const double current = p_mtp->get_linear_coeff(i);
                if (!std::isfinite(ridge) || ridge < 0 || !std::isfinite(current)) GPERROR("Invalid GPU trainmtp effective ridge or candidate");
                before += ridge * old * old;
                after += ridge * current * current;
            }
            gpu_linear_report_.objective_before = before;
            gpu_linear_report_.objective_after = after;
            gpu_linear_report_.effective_l2 = p_mtp->reg_L2_vector;
            if (!std::isfinite(before) || !std::isfinite(after) ||
                after > before + 1e-10 * std::max(1., std::abs(before)))
                GPERROR("GPU trainmtp candidate failed independent CPU effective-objective verification");
        });
        gpu_linear_report_.verification_seconds += seconds(phase);
        gpu_linear_report_.verified = true;
        gpu_linear_report_.total_seconds = seconds(start);
    } catch (...) {
        gpu_linear_active_ = false;
        p_mtp->set_coeffs(saved);
        p_mtp->reg_L1_vector = std::move(saved_l1);
        p_mtp->reg_L2_vector = std::move(saved_l2);
        reg_init = saved_reg_init;
        gpu_linear_report_.verified = false;
        gpu_linear_report_.total_seconds = seconds(start);
        throw;
    }
}

void Trainer_MTP::TrainLinear() {
    if (UsesGpu() && gpu_nonlinear_running_ && fix_linear_coeffs) return;
    if (PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true)) {
        RunGpuLinear(fast_lfit);
        return;
    }
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
    if (!gpu_linear_active_ && PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true)) {
        RunGpuLinear(false);
        return;
    }
    RequireCurrentParameterLayout();
    const auto assembly_start = std::chrono::steady_clock::now();
    auto assemble = [&] {
    // reg_init = false;
    if (!gpu_linear_active_ && do_orthgonalization)
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
    if (gpu_linear_active_) AssembleGpuLinear(false);
    else for (int i = 0; i < training_set_.size(); ++i) {
        if (linear_fit_eng_only)
            AddLinearE(i);
        else
            AddLinear(i);
        if (i%20 == 1 && mpi.rank == 0)
            std::cout <<  " on proc " + std::to_string(mpi.rank) << " linear add " << i << " / " << train_set_size << " added." << std::endl;
    }
    };
    if (gpu_linear_active_) Training::collective_try(assemble);
    else assemble();
    if (gpu_linear_active_)
        gpu_linear_report_.assembly_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - assembly_start).count();
    const auto reduction_start = std::chrono::steady_clock::now();

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
    if (gpu_linear_active_)
        gpu_linear_report_.reduction_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - reduction_start).count();
    const auto solve_start = std::chrono::steady_clock::now();
    // solve the matrix on one processor and broadcast to all others
    std::string solve_error;
    if (mpi.rank == 0) try {
        if (gpu_linear_active_) {
            gpu_linear_gram_.resize(n_size_mat, n_size_mat);
            gpu_linear_rhs_.resize(n_size_mat);
            for (int i = 0; i < n_size_mat; ++i) {
                gpu_linear_rhs_[i] = lin_vector_mpi[i];
                for (int j = 0; j < n_size_mat; ++j)
                    gpu_linear_gram_(i, j) =
                        lin_matrix_mpi(std::min(i, j), std::max(i, j));
            }
            if (!gpu_linear_gram_.allFinite() || !gpu_linear_rhs_.allFinite())
                GPERROR("Non-finite GPU trainmtp normal system");
        }
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
    if (gpu_linear_active_)
        gpu_linear_report_.solve_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
}

void Trainer_MTP::TrainLinearFast()
{
    if (!gpu_linear_active_ && PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true)) {
        RunGpuLinear(true);
        return;
    }
    RequireCurrentParameterLayout();
    const auto assembly_start = std::chrono::steady_clock::now();
    auto assemble = [&] {
    // reg_init = false;
    if (!gpu_linear_active_ && do_orthgonalization)
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
    if (gpu_linear_active_) AssembleGpuLinear(true);
    else for (int i = 0; i < training_set_.size(); ++i) {
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
    };
    if (gpu_linear_active_) Training::collective_try(assemble);
    else assemble();
    if (gpu_linear_active_)
        gpu_linear_report_.assembly_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - assembly_start).count();
    const auto reduction_start = std::chrono::steady_clock::now();

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
    if (gpu_linear_active_)
        gpu_linear_report_.reduction_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - reduction_start).count();
    const auto solve_start = std::chrono::steady_clock::now();
    // solve the matrix on one processor and broadcast to all others
    std::string solve_error;
    if (mpi.rank == 0) try {
        if (gpu_linear_active_) {
            // MPI reduction publishes only the lower triangle consumed by LDLT.
            // Independent objective verification needs its symmetric completion.
            gpu_linear_gram_ = eigen_lin_matrix_mpi.selfadjointView<Eigen::Lower>();
            gpu_linear_rhs_ = eigen_lin_vector_mpi;
            if (!gpu_linear_gram_.allFinite() || !gpu_linear_rhs_.allFinite())
                GPERROR("Non-finite GPU trainmtp normal system");
        }
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
    if (gpu_linear_active_)
        gpu_linear_report_.solve_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
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

            if (weight_hessian > 0 && atom.has_hessian()) {
                double weight_hessian_i = hessian_weight(atom) * cc;
                const int ndof = 3 * atom.size();
                for (int row = 0; row < ndof; ++row) {
                    for (int col = row; col < ndof; ++col) {
                        weight_list(position_ladd) = sqrt(weight_hessian_i);
                        ++position_ladd;
                    }
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
    ReleaseGpuNonlinear();
    const bool gpu_linear = PreflightGpuLinear(pot, atoms, nbhs, false);
    const bool gpu_curvature = gpu_linear && (UsesGpuCurvature() || std::any_of(atoms.begin(), atoms.end(), [](const Atoms &a) {
        return a.have_hessian || a.hvp_directions_.size() || a.hvp_targets_.size();
    }));
    std::vector<Atoms> staged_references, staged_predictions;
    std::vector<Neighborhoods> staged_neighbors;
    if (gpu_curvature) Training::collective_try([&] {
        // Fresh copies are staged after checked old+new ownership reservation,
        // before train_init or any model/accepted-state mutation. Preserve all
        // reference metadata; prediction owners need geometry and EFS only.
        staged_references = atoms;
        staged_neighbors = nbhs;
        staged_predictions = prediction_geometries(atoms);
    });
    auto initialize_model = [&] {

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
    // p_mtp->Save("DEBUG.mtp");
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
    if (gpu_curvature) {
        training_set_.swap(staged_references);
        efs_set_.swap(staged_predictions);
        trainint_set_nbhs_.swap(staged_neighbors);
    } else {
        training_set_ = atoms;
        efs_set_ = atoms;
        trainint_set_nbhs_ = nbhs;
    }
    train_set_size = training_set_.size();
    train_set_size_aligned = train_set_size;
    train_set_size_all = train_set_size;
    // n_coeffs = p_mtp->get_coeff_size();
    // n_radial = n_coeffs - p_mtp->get_lin_coeff_size();
    n_radial_coeffs = p_mtp->radial_coeff_size;
    n_relax_coeffs = p_mtp->get_coeff_size();
    if (fix_linear_coeffs) n_relax_coeffs -= p_mtp->lin_coeff_size;
    // if (!fix_linear_coeffs && fix_species_coeffs) n_relax_coeffs -= p_mtp->species_count;
    if (fix_radial_coeffs) n_relax_coeffs -= p_mtp->radial_coeff_size;
    if (fix_linear_coeffs && fix_radial_coeffs) {
        GPERROR("Nothing to be optimized");
    }
    };
    if (gpu_linear) Training::collective_try(initialize_model);
    else initialize_model();
    SynchronizeGpuCoefficients();
    #ifdef IMRMLP_MPI
    MPI_Allreduce(&train_set_size, &train_set_size_all, 1, MPI_UNSIGNED_LONG, MPI_SUM, mpi.comm);
    MPI_Allreduce(&train_set_size, &train_set_size_aligned, 1, MPI_UNSIGNED_LONG, MPI_MAX, mpi.comm);
    #endif
    auto initialize_rows = [&] {
    emonitor.t_size_all = train_set_size_all;
    train_set_inds_aligned.assign(train_set_size_aligned, 0);
    std::iota(train_set_inds_aligned.begin(), train_set_inds_aligned.begin() + train_set_size, 0);
    // regression_coeffs.setlength(n_coeffs);
    optimization_coeffs.setlength(n_relax_coeffs);
    p_mtp->ae_get_coeffs(optimization_coeffs, fix_radial_coeffs, fix_linear_coeffs);
    update_vector.setlength(n_relax_coeffs);
    update_vector_last.setlength(n_relax_coeffs);
    for (int i = 0; i < n_relax_coeffs; ++i) {
        update_vector[i] = 0;
        update_vector_last[i] = 0;
    }
    loss = 0;
    loss_f_prev = 9e99;
    loss_gard.setlength(n_relax_coeffs);
    loss_mpi = 0;
    ploss = 0;
    // ploss_mpi = 0;
    loss_gard_mpi.setlength(n_relax_coeffs);
    pgard.setlength(n_relax_coeffs);
    // pgard_mpi.setlength(n_coeffs);
    for (int i = 0; i < n_relax_coeffs; ++i) {
        loss_gard[i] = 0;
        loss_gard_mpi[i] = 0;
        pgard[i] = 0;
        // pgard_mpi[i] = 0;
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
        if (weight_hessian > 0 && atom.has_hessian()) {
            double wgt = hessian_weight(atom);
            if (wgt >= 0) {
                const size_t ndof = gpu_linear ? curvature_dof(atom) : size_t(3 * atom.size());
                n_lfit_data += static_cast<int>(upper_rows(ndof));
            }
        }
        if (hvp_fit.weight > 0 && atom.has_hvp())
            n_lfit_data += atom.hvp_directions_.size();
    }
    };
    if (gpu_linear) Training::collective_try(initialize_rows);
    else initialize_rows();
    MPI_Allreduce(&n_lfit_data, &n_lfit_data_all, 1, MPI_INT, MPI_SUM, mpi.comm);
    auto initialize_buffers = [&] {
    gpu_linear_allocation_signature_ = GpuLinearAllocationSignature();
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
    };
    if (gpu_linear) Training::collective_try(initialize_buffers);
    else initialize_buffers();
}

void Trainer_MTP::init_valid(std::vector<Atoms> &atoms, std::vector<Neighborhoods> &nbhs) {
    int requested = UsesGpuLinear(), any = requested, all = requested;
    MPI_Allreduce(&requested, &any, 1, MPI_INT, MPI_MAX, mpi.comm);
    MPI_Allreduce(&requested, &all, 1, MPI_INT, MPI_MIN, mpi.comm);
    if (any != all) GPERROR("All trainmtp ranks must select the same observation backend");
    if (any) {
        std::vector<Atoms> prepared, predictions;
        std::vector<Neighborhoods> prepared_neighbors;
        Training::collective_try([&] {
            RequireCurrentParameterLayout();
            if (atoms.size() != nbhs.size()) GPERROR("Invalid GPU trainmtp validation neighborhood count");
            for (const auto &a : atoms) {
                (void)curvature_dof(a); a.validate_hvp(); Training::validate_hessian_fit_label(a);
            }
            if (UsesGpuCurvature()) {
                size_t bytes = gpu_validation_checked_add(GpuCurvatureOwnedBytes(),
                    GpuCurvatureReserve(*p_mtp, training_set_, trainint_set_nbhs_, false));
                bytes = gpu_validation_checked_add(bytes, Training::gpu_validation_atoms_bytes(atoms, true));
                bytes = gpu_validation_checked_add(bytes, Training::gpu_validation_neighbors_bytes(nbhs, true));
                bytes = gpu_validation_checked_add(bytes, prediction_geometry_bytes(atoms));
                if (bytes > GpuCurvatureHostCap()) GPERROR("GPU trainmtp validation staging exceeds gpu_curvature_host_mib");
                gpu_curvature_host_peak_ = std::max(gpu_curvature_host_peak_, bytes);
            }
            prepared = atoms;
            predictions = UsesGpuCurvature() ? prediction_geometries(atoms) : atoms;
            prepared_neighbors = nbhs;
        });
        valid_set_.swap(prepared); efs_valid_set_.swap(predictions); valid_set_nbhs_.swap(prepared_neighbors);
        valid_set_size = valid_set_.size();
        valid_set_size_all = valid_set_size;
#ifdef IMRMLP_MPI
        MPI_Allreduce(&valid_set_size, &valid_set_size_all, 1, MPI_UNSIGNED_LONG, MPI_SUM, mpi.comm);
#endif
        return;
    }
    auto initialize_validation = [&] {
    RequireCurrentParameterLayout();
    for (const auto &a : atoms) a.validate_hvp();

    valid_set_ = atoms;
    efs_valid_set_ = atoms;
    valid_set_nbhs_ = nbhs;
    valid_set_size  = valid_set_.size();
    valid_set_size_all = valid_set_size;
    };
    initialize_validation();
    #ifdef IMRMLP_MPI
    MPI_Allreduce(&valid_set_size, &valid_set_size_all, 1, MPI_UNSIGNED_LONG, MPI_SUM, mpi.comm);
    #endif
}

void Trainer_MTP::AccumulateGard(double &f, alglib::real_1d_array &gard) {
    const bool gpu = PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true);
    RequireCurrentParameterLayout();
    loss = 0;
    f = 0;
    emonitor.init();
    emonitor_valid.init();
    memset(loss_gard.getcontent(), 0, n_relax_coeffs * sizeof(double));
    memset(gard.getcontent(), 0, n_relax_coeffs * sizeof(double));
    if (gpu) {
        Training::collective_try([&] {
            std::vector<int> ids(train_set_size);
            std::iota(ids.begin(), ids.end(), 0);
            AccumulateGpuNonlinear(ids, loss, true);
            InitPenaltyGard();
        });
    } else for (int i = 0; i < train_set_size; ++i)
        AddLossGard(i, loss);
    #ifdef IMRMLP_MPI
    MPI_Reduce(&loss, &f, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(loss_gard.getcontent(), gard.getcontent(), n_relax_coeffs, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    #else
    gard = loss_gard;
    f = loss;
    #endif
    if (!UsesGpu()) InitPenaltyGard();
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
    const bool gpu = PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true);
    RequireCurrentParameterLayout();
    loss = 0;
    f = 0;
    emonitor.init();
    emonitor_valid.init();
    if (gpu) {
        Training::collective_try([&] {
            std::vector<int> ids(train_set_size);
            std::iota(ids.begin(), ids.end(), 0);
            AccumulateGpuNonlinear(ids, loss, false);
            InitPenaltyGard();
        });
    } else for (int i = 0; i < train_set_size; ++i)
        AddLoss(i, loss);
    #ifdef IMRMLP_MPI
    MPI_Reduce(&loss, &f, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    #else
    f = loss;
    #endif
    if (!UsesGpu()) InitPenaltyGard();
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
    const bool gpu = PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true);
    RequireCurrentParameterLayout();
    double loss_val = 0;
    f = 0;
    if (gpu) {
        Training::collective_try([&] {
            std::vector<int> ids(valid_set_size);
            std::iota(ids.begin(), ids.end(), 0);
            AccumulateGpuNonlinear(ids, f, false, true);
        });
        // Legacy validation reports errors only, not a weighted objective.
        f = 0.;
        return;
    }
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
    ptrainer->SaveCheckpoint("_Current_potential_save_" + std::to_string(ptrainer->bfgs_iter) + ".mtp",
                             ptrainer->bfgs_iter % ptrainer->save_every_it == 0);
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
    if (!gpu_nonlinear_running_ &&
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true)) {
        RunGpuNonlinear([&] { TrainNonLinearBFGS(limit); });
        return;
    }
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
        trainer.SaveCheckpoint("_Current_potential_save_" + std::to_string(trainer.bfgs_iter) + ".mtp",
                               trainer.bfgs_iter % trainer.save_every_it == 0);
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
    if (!gpu_nonlinear_running_ &&
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true)) {
        RunGpuNonlinear([&] { TrainBFGS(); });
        return;
    }
    RequireCurrentParameterLayout();
    if (!p_mtp->inited()) {
        if (random_init) {
            p_mtp->random_init();
            SynchronizeGpuCoefficients();
        }
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
    if (UsesGpu()) SaveCheckpoint(potfilesave); else save_current(potfilesave);
}

void Trainer_MTP::TrainDF() {
    RejectGpuLinearAction("TrainDF");
    RequireCurrentParameterLayout();
    if (!p_mtp->inited()) {
        if (random_init) {
            p_mtp->random_init();
            SynchronizeGpuCoefficients();
        }
    }
    TrainNonLinearDF(maxiter);
    if (UsesGpu()) SaveCheckpoint(potfilesave); else save_current(potfilesave);
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
    RejectGpuLinearAction("TrainRandom");
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
    if (!gpu_nonlinear_running_ &&
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true)) {
        RunGpuNonlinear([&] { TrainSGD(); });
        return;
    }
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
            SynchronizeGpuCoefficients();
        }
        Message("Rescaling");
        Rescale();
        SaveCheckpoint(potfilesave);
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
            if (UsesGpu()) {
                PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true);
                Training::collective_try([&] {
                    std::vector<int> ids(train_set_inds_aligned.begin() + icurrent,
                                         train_set_inds_aligned.begin() + icurrent + this_batch);
                    AccumulateGpuNonlinear(ids, loss, true);
                });
                icurrent += this_batch;
            } else for (int i = 0; i < this_batch; ++i) {
                int id = train_set_inds_aligned[icurrent++];
                AddLossGard(id, loss);
            }
            Message(" loss: " + std::to_string(loss_mpi));
            SGDUpdate(this_batch);
            ++bfgs_iter;
        }
        SaveCheckpoint(potfilesave + std::to_string(iepoch), iepoch % save_every_it == 0);
    }

    // postprocessing
    if (!linear_fit_eng_only && bfgs_rescale_final)
        Rescale();
    SaveCheckpoint(potfilesave);
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
    if (UsesGpu()) {
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true);
        Training::collective_try([&] { AccumulateGpuNonlinear({id}, _loss, true); });
        return;
    }
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
    if (UsesGpu()) {
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true);
        Training::collective_try([&] { AccumulateGpuNonlinear({id}, _loss, false); });
        return;
    }
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
    auto update = [&] {
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
    if (UsesGpu() && mpi.rank == 0)
        for (int i = 0; i < optimization_coeffs.length(); ++i)
            if (!std::isfinite(optimization_coeffs[i]))
                GPERROR("Non-finite GPU trainmtp SGD update");
    };
    if (UsesGpu()) Training::collective_try(update); else update();
    // Broadcast
#ifdef IMRMLP_MPI
    MPI_Bcast(optimization_coeffs.getcontent(),
              n_relax_coeffs,
              MPI_DOUBLE,
              0,
              mpi.comm);
#endif
    p_mtp->ae_set_coeffs(optimization_coeffs, fix_radial_coeffs, fix_linear_coeffs);
    if (UsesGpu()) Training::collective_try([&] { InitPenaltyGard(); }); else InitPenaltyGard();
}

void Trainer_MTP::TrainCG() {
    if (!gpu_nonlinear_running_ &&
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true)) {
        RunGpuNonlinear([&] { TrainCG(); });
        return;
    }
    RequireCurrentParameterLayout();
    if (!p_mtp->inited()) {
        if (random_init) {
            p_mtp->random_init();
            SynchronizeGpuCoefficients();
        }
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
    if (UsesGpu()) SaveCheckpoint(potfilesave); else save_current(potfilesave);
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

        SaveCheckpoint("_Current_potential_save_" + std::to_string(bfgs_iter) + ".mtp",
                       bfgs_iter % save_every_it == 0);

    MPI_Barrier(mpi.comm);

    return loss_mpi;
}

void Trainer_MTP::TrainNonLinearLBFGS(int it_limit)
{
    if (!gpu_nonlinear_running_ &&
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true)) {
        RunGpuNonlinear([&] { TrainNonLinearLBFGS(it_limit); });
        return;
    }
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
    if (!gpu_nonlinear_running_ &&
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true)) {
        RunGpuNonlinear([&] { TrainLBFGS(); });
        return;
    }
    RequireCurrentParameterLayout();
    // set scaling
    scaling_coeffs.resize(n_relax_coeffs);
    scaling_coeffs.setOnes();

    if (!p_mtp->inited()) {
        if (random_init) {
            p_mtp->random_init();
            SynchronizeGpuCoefficients();
        }
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
    if (UsesGpu()) SaveCheckpoint(potfilesave); else save_current(potfilesave);
}

void Trainer_MTP::TrainRadialLBFGSThenLinear(int it_limit)
{
    if (!gpu_nonlinear_running_ &&
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true)) {
        RunGpuNonlinear([&] { TrainRadialLBFGSThenLinear(it_limit); });
        return;
    }
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
            if (UsesGpu()) throw;
            Warning("Radial alglib LBFGS failed, continue with linear fit: " + std::string(e.what()));
        } catch (...) {
            maxiter = old_maxiter;
            if (UsesGpu()) throw;
            Warning("Radial alglib LBFGS failed with unknown error, continue with linear fit");
        }
    }

    TrainLinear();
}

void Trainer_MTP::TrainNonLinearCG(int it_limit)
{
    if (!gpu_nonlinear_running_ &&
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true)) {
        RunGpuNonlinear([&] { TrainNonLinearCG(it_limit); });
        return;
    }
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
    alglib::mincgsetcond(cgstate, bfgs_grad_tol, bfgs_step_tol, bfgs_conv_tol, it_limit);
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
    if (UsesGpu()) Training::publish_conditional_linear(*p_mtp, fnm);
    else p_mtp->Save(fnm);
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
    if (!gpu_nonlinear_running_ &&
        PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true)) {
        RunGpuNonlinear([&] { check_grad(ind, nudge); });
        return;
    }
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
    if (UsesGpuLinear())
        GPERROR("GPU trainmtp row assembly requires TrainLinear, TrainLinearAlg or TrainLinearFast");
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
    if (UsesGpuLinear())
        GPERROR("GPU trainmtp row assembly requires TrainLinear, TrainLinearAlg or TrainLinearFast");
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

double Trainer_MTP::hessian_weight(const Atoms &atom) const {
    double wgt = weight_hessian / pow(atom.size(), weight_hessian_power);
    return wgt * atom.fitness;
}
