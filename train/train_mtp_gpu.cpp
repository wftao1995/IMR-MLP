// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "train_mtp.h"
#include "collective_error.h"
#include "conditional_linear_persistence.h"
#include "potentials/mtp_gpu_adapter.h"
#include <chrono>
#include <numeric>
#include <sstream>

struct Trainer_MTP::GpuNonlinearState {
#ifdef IMRMLP_GPU
    Potentials::MTPGpuEvaluator evaluator;
#endif
    std::string definition;
    int device = -1;
    size_t cap = 0;
};

void Trainer_MTP::ReleaseGpuNonlinear() { gpu_nonlinear_.reset(); }

void Trainer_MTP::ValidateGpuNonlinearSettings() const {
    ValidateGpuSettings();
    if (!UsesGpu()) return;
    if (UsesGpuCurvature()) GPERROR("Nonlinear GPU trainmtp does not support Hessian/HVP losses");
    if (save_every_it <= 0) GPERROR("fit:save_every must be positive");
}

void Trainer_MTP::SynchronizeGpuCoefficients() {
    if (!UsesGpu()) return;
    Eigen::VectorXd coefficients;
    Training::collective_try([&] { p_mtp->get_coeffs(coefficients); });
    MPI_Bcast(coefficients.data(), coefficients.size(), MPI_DOUBLE, 0, mpi.comm);
    p_mtp->set_coeffs(coefficients);
    bool flags[] = {p_mtp->empty, p_mtp->radial_empty, p_mtp->linear_empty};
    MPI_Bcast(flags, 3, MPI_C_BOOL, 0, mpi.comm);
    p_mtp->empty = flags[0];
    p_mtp->radial_empty = flags[1];
    p_mtp->linear_empty = flags[2];
}

// Called by every rank, even when the legacy checkpoint was root-only.
void Trainer_MTP::SaveCheckpoint(const std::string &path, bool enabled) {
    if (!enabled) return;
    if (UsesGpu()) Training::collective_try([&] {
        if (mpi.rank == 0) save_current(path);
    });
    else if (mpi.rank == 0) save_current(path);
}

void Trainer_MTP::RunGpuNonlinear(const std::function<void()> &operation) {
    PreflightGpuLinear(p_mtp, training_set_, trainint_set_nbhs_, true, true);
    Eigen::VectorXd coefficients;
    std::vector<double> l1, l2;
    alglib::real_1d_array update, previous, optimization;
    std::vector<double> scaling;
    std::vector<int> indices;
    const bool initialized = p_mtp->inited(), ridge_initialized = reg_init;
    const bool radial_empty = p_mtp->radial_empty, linear_empty = p_mtp->linear_empty;
    const int iteration = bfgs_iter, last_refit = bfgs_last_linear_fit_iter;
    const double previous_loss = loss_f_prev;
    const bool pretrained = sgd_pretrain;
    const auto rng = RNG.mtd;
    Training::collective_try([&] {
        if (p_mtp->radial_basis_split) {
            for (const auto *basis : p_mtp->p_radical_basis_pair) scaling.push_back(basis->scaling);
        } else scaling.push_back(p_mtp->p_radical_basis->scaling);
        indices = train_set_inds_aligned;
        p_mtp->get_coeffs(coefficients);
        l1 = p_mtp->reg_L1_vector;
        l2 = p_mtp->reg_L2_vector;
        update = update_vector;
        previous = update_vector_last;
        optimization = optimization_coeffs;
    });
    auto previous_scheduler = std::move(sgd_scheduler);
    gpu_nonlinear_running_ = true;
    try {
        operation();
        if (!sgd_scheduler) sgd_scheduler = std::move(previous_scheduler);
        gpu_nonlinear_running_ = false;
    } catch (...) {
        gpu_nonlinear_running_ = false;
        ReleaseGpuNonlinear();
        if (p_mtp->radial_basis_split) {
            for (size_t i = 0; i < scaling.size(); ++i)
                p_mtp->p_radical_basis_pair[i]->scaling = scaling[i];
        } else p_mtp->p_radical_basis->scaling = scaling.front();
        p_mtp->set_coeffs(coefficients);
        if (p_mtp->inited() != initialized) p_mtp->set_empty_flag();
        p_mtp->radial_empty = radial_empty;
        p_mtp->linear_empty = linear_empty;
        p_mtp->reg_L1_vector.swap(l1);
        p_mtp->reg_L2_vector.swap(l2);
        reg_init = ridge_initialized;
        bfgs_iter = iteration;
        bfgs_last_linear_fit_iter = last_refit;
        loss_f_prev = previous_loss;
        sgd_pretrain = pretrained;
        update_vector = update;
        update_vector_last = previous;
        optimization_coeffs = optimization;
        train_set_inds_aligned.swap(indices);
        RNG.mtd = rng;
        sgd_scheduler = std::move(previous_scheduler);
        InitPenaltyGard();
        throw;
    }
}

void Trainer_MTP::AccumulateGpuNonlinear(const std::vector<int> &indices,
    double &result, bool gradient, bool validation) {
#ifndef IMRMLP_GPU
    GPERROR("GPU trainmtp requires a CUDA or HIP build");
#else
    const auto start = std::chrono::steady_clock::now();
    auto &references = validation ? valid_set_ : training_set_;
    auto &neighbors = validation ? valid_set_nbhs_ : trainint_set_nbhs_;
    auto &predictions = validation ? efs_valid_set_ : efs_set_;
    auto &monitor = validation ? emonitor_valid : emonitor;
    if (references.size() != neighbors.size() || predictions.size() != references.size())
        GPERROR("GPU trainmtp data storage changed; reinitialize the trainer");
    for (int id : indices) {
        if (id < 0 || size_t(id) >= references.size()) GPERROR("Invalid GPU trainmtp sample index");
        const auto &a = references[id];
        if (a.size() <= 0 || size_t(a.size()) != a.chemmap_.size() || neighbors[id].size() != a.size() ||
            !a.cell_.allFinite() || !std::isfinite(a.fitness) || a.fitness < 0. ||
            (!a.atomic_fitness.empty() && a.atomic_fitness.size() != size_t(a.size())) ||
            (a.has_energy() && !std::isfinite(a.energy_)) || (a.has_stresses() && !a.stress_.allFinite()) ||
            (a.has_forces() && a.forces_.size() != size_t(a.size())))
            GPERROR("Invalid GPU trainmtp EFS data or fitness");
        if (a.have_hessian || a.hvp_directions_.size() || a.hvp_targets_.size())
            GPERROR("Nonlinear GPU trainmtp does not support Hessian/HVP labels");
        for (double w : a.atomic_fitness) if (!std::isfinite(w) || w < 0.) GPERROR("Invalid GPU trainmtp atomic fitness");
        for (const auto &v : a.forces_) if (!v.allFinite()) GPERROR("Invalid GPU trainmtp force label");
    }
    const auto options = GpuLinearOptions();
    const size_t cap = options.gpu_memory_bytes + options.gpu_workspace_bytes;
    if (cap < options.gpu_memory_bytes) GPERROR("GPU trainmtp budget overflow");
    std::string error;
    // Serialization includes scaling, radial basis definitions and the moment
    // graph. Coefficient changes are handled by update_coefficients below.
    std::ostringstream serialized;
    p_mtp->write_header(serialized);
    p_mtp->write_radial_info(serialized);
    p_mtp->write_rules(serialized);
    if (p_mtp->radial_basis_split)
        for (const auto *basis : p_mtp->p_radical_basis_pair) serialized << ' ' << basis->scaling;
    std::string definition = serialized.str();
    // Preserve the owner across coefficient updates, but rebuild after any
    // change to the native basis definition, scaling or graph.
    if (!indices.empty() && (!gpu_nonlinear_ || gpu_nonlinear_->definition != definition ||
        gpu_nonlinear_->device != gpu_linear_options_.gpu_id || gpu_nonlinear_->cap != cap)) {
        ReleaseGpuNonlinear();
        auto pending = std::make_shared<GpuNonlinearState>();
        if (!pending->evaluator.initialize_training_bounded(*p_mtp, gpu_linear_options_.gpu_id, cap, &error))
            GPERROR("GPU trainmtp initialization failed: " + error);
        pending->definition = definition; pending->device = gpu_linear_options_.gpu_id; pending->cap = cap;
        gpu_nonlinear_ = std::move(pending);
    }
    if (!indices.empty() && !gpu_nonlinear_->evaluator.update_coefficients(*p_mtp, &error))
        GPERROR("GPU trainmtp coefficient update failed: " + error);
    Eigen::VectorXd full = Eigen::VectorXd::Zero(p_mtp->get_coeff_size());
    double pending_loss = 0;
    ErrorMonitor pending_monitor;
    size_t position = 0;
    while (position < indices.size()) {
        size_t end = position, atoms = 0;
        while (end < indices.size() && end - position < size_t(options.gpu_batch_structures)) {
            const int id = indices[end];
            if (id < 0 || size_t(id) >= references.size()) GPERROR("Invalid GPU trainmtp sample index");
            const size_t n = references[id].size();
            if (n > size_t(options.gpu_batch_atoms)) GPERROR("Single GPU trainmtp structure exceeds gpu_batch_atoms");
            if (atoms + n > size_t(options.gpu_batch_atoms)) break;
            atoms += n; ++end;
        }
        if (end == position) GPERROR("GPU trainmtp cannot form a complete-structure batch");
        // Allocation failures shrink a whole-structure batch. Pending results
        // are committed only after a complete successful attempt.
        for (;;) {
            std::vector<int> selection(indices.begin() + position, indices.begin() + end);
            Potentials::MTPGpuBatch batch;
            batch.configure_gradient_reduction(gpu_deterministic_reduction_, 128);
            bool ok = batch.build(references, neighbors, selection, &error);
            const size_t ns = selection.size(), na = batch.atom_count();
            std::vector<double> te(ns, 0.), ew(ns, 0.), tf(3*na, 0.), fw(na, 0.), ts(9*ns, 0.), sw(ns, 0.);
            bool repair_monitor = false;
            size_t offset = 0;
            int ne = 0, nf = 0, nv = 0;
            for (size_t s = 0; ok && s < ns; ++s) {
                const auto &a = references[selection[s]];
                double cc = 1.;
                if (weight_force_norm_f > 0 && a.has_forces()) {
                    double norm = 0.; for (const auto &f : a.forces_) norm += f.squaredNorm();
                    cc = weight_force_norm_d / (weight_force_norm_d +
                        std::pow(weight_force_norm_f * norm / a.size(), weight_force_norm_p));
                }
                if (weight_eng > 0 && a.has_energy()) {
                    te[s] = a.energy_; ew[s] = energy_weight(a)*cc; ++ne;
                    repair_monitor |= ew[s] == 0.;
                }
                if (weight_force > 0 && a.has_forces()) {
                    nf += a.size();
                    for (int i = 0; i < a.size(); ++i) {
                        fw[offset+i] = force_weight(a)*cc*(a.atomic_fitness.empty() ? 1. : a.atomic_fitness[i]);
                        repair_monitor |= fw[offset+i] == 0.;
                        for (int d = 0; d < 3; ++d) tf[3*(offset+i)+d] = a.forces_[i][d];
                    }
                }
                if (weight_stress > 0 && a.has_stresses()) {
                    sw[s] = stress_weight(a)*cc; ++nv;
                    repair_monitor |= sw[s] == 0.;
                    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) ts[9*s+3*i+j] = a.stress_(i,j);
                }
                for (double weight : {ew[s], sw[s]})
                    if (!std::isfinite(weight) || weight < 0.) GPERROR("Invalid GPU trainmtp observation weight");
                for (int i = 0; i < a.size(); ++i)
                    if (!std::isfinite(fw[offset+i]) || fw[offset+i] < 0.) GPERROR("Invalid GPU trainmtp force weight");
                offset += a.size();
            }
            std::array<double, MLIPCore::GpuDirect::MTPGpuLossRequest::metric_count> metrics{};
            Eigen::VectorXd chunk_gradient = Eigen::VectorXd::Zero(full.size());
            ErrorMonitor chunk_monitor;
            double chunk_loss = 0.;
            const bool device_loss = gpu_device_loss_ && !validation;
            if (ok && device_loss) {
                ok = batch.evaluate_loss(gpu_nonlinear_->evaluator, te, ew, tf, fw, ts, sw, metrics, true, true, &error);
                if (ok && gradient) ok = batch.accumulate_prepared_gradient(gpu_nonlinear_->evaluator,
                    chunk_gradient.data(), chunk_gradient.size(), true, true, &error);
                chunk_loss = metrics[0];
                chunk_monitor.loss_e = metrics[1]; chunk_monitor.loss_f = metrics[2]; chunk_monitor.loss_s = metrics[3];
                chunk_monitor.sq_energy_at_error = metrics[4]; chunk_monitor.sq_force_error = metrics[5];
                chunk_monitor.sq_stress_error = metrics[6];
                chunk_monitor.nenergy = ne; chunk_monitor.nforces = nf; chunk_monitor.nstresstensor = nv;
            }
            if (ok && (!device_loss || repair_monitor)) {
                ok = batch.evaluate(gpu_nonlinear_->evaluator, predictions, selection, true, &error);
                std::vector<double> de(ns, 0.), df(3*na, 0.), ds(9*ns, 0.);
                if (ok) {
                    if (device_loss) { chunk_monitor.sq_energy_at_error = 0.; chunk_monitor.sq_force_error = 0.; chunk_monitor.sq_stress_error = 0.; }
                    offset = 0;
                    for (size_t s = 0; s < ns; ++s) {
                        const auto &a = references[selection[s]]; const auto &p = predictions[selection[s]];
                        auto add = [&](double value, double weight, double &loss_channel) {
                            const double contribution = weight*value*value;
                            if (!device_loss) {chunk_loss += contribution; loss_channel += contribution;}
                        };
                        if (weight_eng > 0 && a.has_energy()) {
                            const double value = p.energy_ - a.energy_;
                            add(value, ew[s], chunk_monitor.loss_e); de[s] = 2*ew[s]*value;
                            chunk_monitor.sq_energy_at_error += value*value/(double(a.size())*a.size());
                        }
                        // Validation historically reports force/virial errors even
                        // when their training channel weight is zero.
                        if ((validation || weight_force > 0) && a.has_forces()) for (int i = 0; i < a.size(); ++i) {
                            const auto delta = (p.forces_[i]-a.forces_[i]).eval();
                            chunk_monitor.sq_force_error += delta.squaredNorm();
                            for (int d = 0; d < 3; ++d) {
                                add(delta[d], fw[offset+i], chunk_monitor.loss_f);
                                df[3*(offset+i)+d] = 2*fw[offset+i]*delta[d];
                            }
                        }
                        if ((validation || weight_stress > 0) && a.has_stresses()) for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
                            const double value = p.stress_(i,j)-a.stress_(i,j);
                            chunk_monitor.sq_stress_error += value*value;
                            add(value, sw[s], chunk_monitor.loss_s); ds[9*s+3*i+j] = 2*sw[s]*value;
                        }
                        offset += a.size();
                    }
                    if (!device_loss) {
                        chunk_monitor.nenergy = ne; chunk_monitor.nforces = nf; chunk_monitor.nstresstensor = nv;
                        if (validation) {
                            chunk_monitor.nforces = chunk_monitor.nstresstensor = 0;
                            for (int id : selection) {
                                if (references[id].has_forces()) chunk_monitor.nforces += references[id].size();
                                if (references[id].has_stresses()) ++chunk_monitor.nstresstensor;
                            }
                        }
                        if (gradient) ok = batch.accumulate_gradient_chunks(gpu_nonlinear_->evaluator, de, df, ds,
                            chunk_gradient.data(), chunk_gradient.size(), true, true, &error);
                    }
                }
            }
            const auto stats = gpu_nonlinear_->evaluator.context().memory_stats();
            gpu_nonlinear_report_.peak_device_bytes = std::max(gpu_nonlinear_report_.peak_device_bytes, stats.training_peak_device_bytes);
            if (!ok) {
                const bool budget = error.find("training_memory_budget_bytes") != std::string::npos;
                if (!budget || end-position == 1) GPERROR("GPU trainmtp batch failed: " + error);
                end = position + (end-position)/2;
                // Drop all retained high-water allocations before retrying.
                ReleaseGpuNonlinear();
                auto retry = std::make_shared<GpuNonlinearState>();
                if (!retry->evaluator.initialize_training_bounded(*p_mtp, gpu_linear_options_.gpu_id, cap, &error)) GPERROR(error);
                retry->definition = definition; retry->device = gpu_linear_options_.gpu_id; retry->cap = cap;
                gpu_nonlinear_ = std::move(retry);
                continue;
            }
            if (!std::isfinite(chunk_loss) || !chunk_gradient.allFinite()) GPERROR("Non-finite GPU trainmtp loss or gradient");
            pending_loss += chunk_loss; full += chunk_gradient;
            pending_monitor.loss_e += chunk_monitor.loss_e; pending_monitor.loss_f += chunk_monitor.loss_f; pending_monitor.loss_s += chunk_monitor.loss_s;
            pending_monitor.sq_energy_at_error += chunk_monitor.sq_energy_at_error;
            pending_monitor.sq_force_error += chunk_monitor.sq_force_error; pending_monitor.sq_stress_error += chunk_monitor.sq_stress_error;
            pending_monitor.nenergy += chunk_monitor.nenergy; pending_monitor.nforces += chunk_monitor.nforces;
            pending_monitor.nstresstensor += chunk_monitor.nstresstensor;
            ++gpu_nonlinear_report_.batches;
            position = end; break;
        }
    }
    result += pending_loss;
    if (gradient) {
        const int offset = fix_radial_coeffs ? n_radial_coeffs : 0;
        for (int i = 0; i < n_relax_coeffs; ++i) loss_gard[i] += full[offset+i];
        if (fix_species_coeffs && !fix_linear_coeffs) {
            const int species_offset = fix_radial_coeffs ? 0 : n_radial_coeffs;
            for (int i = 0; i < p_mtp->species_count; ++i) loss_gard[species_offset+i] = 0.;
        }
    }
    monitor.loss_e += pending_monitor.loss_e; monitor.loss_f += pending_monitor.loss_f; monitor.loss_s += pending_monitor.loss_s;
    monitor.sq_energy_at_error += pending_monitor.sq_energy_at_error;
    monitor.sq_force_error += pending_monitor.sq_force_error; monitor.sq_stress_error += pending_monitor.sq_stress_error;
    monitor.nenergy += pending_monitor.nenergy; monitor.nforces += pending_monitor.nforces; monitor.nstresstensor += pending_monitor.nstresstensor;
    ++gpu_nonlinear_report_.evaluations;
    if (validation) ++gpu_nonlinear_report_.validation_evaluations;
    if (gradient) ++gpu_nonlinear_report_.gradient_evaluations;
    gpu_nonlinear_report_.evaluation_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
#endif
}
