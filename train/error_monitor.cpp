// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "error_monitor.h"
#include "utils/gputils.h"

#include <cmath>

void ErrorMonitor::report()
{
    double hvp_error_all = 0, hvp_loss_all = 0;
    int hvp_count_all = 0;
    MPI_Reduce(&sq_hvp_error, &hvp_error_all, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&loss_hvp, &hvp_loss_all, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&nhvp, &hvp_count_all, 1, MPI_INT, MPI_SUM, 0, mpi.comm);
    if (mpi.rank == 0)
    {
        hvp_rms = hvp_count_all > 0 ? std::sqrt(hvp_error_all / hvp_count_all) : 0.;
        loss_hvp = t_size_all > 0 ? hvp_loss_all / t_size_all : 0.;
    }

    int neng_total = 0, nfce_total = 0, nstr_total = 0, nhess_total = 0;
    erms = 0;
    frms = 0;
    srms = 0;
    hrms = 0;
    MPI_Reduce(&nenergy, &neng_total, 1, MPI_INT, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&nforces, &nfce_total, 1, MPI_INT, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&nstresstensor, &nstr_total, 1, MPI_INT, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&nhessian, &nhess_total, 1, MPI_INT, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&sq_energy_at_error, &erms, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&sq_force_error, &frms, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&sq_stress_error, &srms, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&sq_hessian_error, &hrms, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    double le = 0, lf = 0, ls = 0, lh = 0;
    MPI_Reduce(&loss_e, &le, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&loss_f, &lf, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&loss_s, &ls, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&loss_h, &lh, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    // erms = 0; frms = 0; srms = 0;
    if (mpi.rank == 0)
    {
        erms = neng_total > 0 ? sqrt(erms / neng_total) : 0;
        frms = nfce_total > 0 ? sqrt(frms / nfce_total) : 0;
        srms = nstr_total > 0 ? sqrt(srms / nstr_total) : 0;
        hrms = nhess_total > 0 ? sqrt(hrms / nhess_total) : 0;
        loss_e = le / t_size_all;
        loss_f = lf / t_size_all;
        loss_s = ls / t_size_all;
        loss_h = lh / t_size_all;
    }
}

namespace Training
{
EFSFitErrors::EFSFitErrors(double we, double wf, double ws)
{
    weight_e = we;
    weight_f = wf;
    weight_v = ws;
    weights.energy = we;
    weights.force = wf;
    weights.stress = ws;
}

EFSFitErrors::EFSFitErrors(const EFSFitWeights &w) : EFSFitErrors(w.energy, w.force, w.stress)
{
    weights = w;
}

void EFSFitErrors::clear()
{
    positive_observations = 0;
    ncfgs = 0;
    e_sum = 0;
    f_sum = 0;
    v_sum = 0;
    loss = 0;
    e_c = 0;
    f_c = 0;
    v_c = 0;
    // e.clear();
    // f.clear();
    // v.clear();
}

void EFSFitErrors::add(Atoms &prd, Atoms &ref)
{
    if (prd.size() == 0 || prd.size() != ref.size())
        GPERROR("fastfit: paired structures must have matching, nonzero atom counts");
    if ((ref.has_forces() && ref.forces_.size() != size_t(ref.size())) ||
        (prd.has_forces() && prd.forces_.size() != size_t(prd.size())))
        GPERROR("fastfit: invalid force array length");
    if (!ref.atomic_fitness.empty() && ref.atomic_fitness.size() != size_t(ref.size()))
        GPERROR("fastfit: invalid atomic weight array length");
    const double normalization = weights.normalization(ref);
    const auto add_loss = [&](double squared, double weight)
    {
        if (!std::isfinite(weight) || weight < 0 || !std::isfinite(squared))
            GPERROR("fastfit: non-finite observation or invalid weight");
        loss += weight * squared;
        if (weight > 0)
            ++positive_observations;
    };
    ++ncfgs;
    if (ref.has_energy() && prd.has_energy())
    {
        double t = (prd.energy_ - ref.energy_) / prd.size();
        e_sum += t * t;
        add_loss(std::pow(prd.energy_ - ref.energy_, 2),
                 weights.energy_weight(ref) * normalization);
        ++e_c;
    }
    if (ref.has_forces() && prd.has_forces())
    {
        for (int i = 0; i < prd.size(); ++i)
        {
            double t = (prd.forces_[i] - ref.forces_[i]).squaredNorm();
            f_sum += t;
            add_loss(
                t, EFSFitWeights::atomic_weight(ref, i, weights.force_weight(ref) * normalization));
            ++f_c;
        }
    }
    if (ref.has_stresses() && prd.has_stresses())
    {
        double t = (prd.stress_ - ref.stress_).squaredNorm() / prd.size();
        add_loss((prd.stress_ - ref.stress_).squaredNorm(),
                 weights.stress_weight(ref) * normalization);
        v_sum += t;
        ++v_c;
    }
}

void EFSFitErrors::Reduce(double &ermse, double &frmse, double &vrmse, double &global_loss) const
{
    double local[9] = {e_sum,
                       f_sum,
                       v_sum,
                       loss,
                       static_cast<double>(e_c),
                       static_cast<double>(f_c),
                       static_cast<double>(v_c),
                       static_cast<double>(ncfgs),
                       positive_observations};
    double global[9] = {};
    MPI_Allreduce(local, global, 9, MPI_DOUBLE, MPI_SUM, mpi.comm);
    if (global[7] > 0 && global[8] == 0)
        GPERROR("fastfit: dataset has no positive-weight E/F/S observations");
    ermse = global[4] > 0 ? sqrt(global[0] / global[4]) : 0.0;
    frmse = global[5] > 0 ? sqrt(global[1] / global[5]) : 0.0;
    vrmse = global[6] > 0 ? sqrt(global[2] / global[6]) : 0.0;
    global_loss = global[3];
}
} // namespace Training
