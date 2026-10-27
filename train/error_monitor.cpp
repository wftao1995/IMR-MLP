// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "error_monitor.h"
#include "utils/gputils.h"

#include <cmath>

void ErrorMonitor::report() {
    double hvp_error_all = 0, hvp_loss_all = 0;
    int hvp_count_all = 0;
    MPI_Reduce(&sq_hvp_error, &hvp_error_all, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&loss_hvp, &hvp_loss_all, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
    MPI_Reduce(&nhvp, &hvp_count_all, 1, MPI_INT, MPI_SUM, 0, mpi.comm);
    if (mpi.rank == 0) {
        hvp_rms = hvp_count_all > 0 ? std::sqrt(hvp_error_all / hvp_count_all) : 0.;
        loss_hvp = t_size_all > 0 ? hvp_loss_all / t_size_all : 0.;
    }

    int neng_total = 0, nfce_total = 0, nstr_total = 0, nhess_total = 0;
    erms = 0; frms = 0; srms = 0; hrms = 0;
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
    if (mpi.rank == 0) {
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
