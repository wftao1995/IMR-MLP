// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "linear_efs_system.h"
#include "efs_square_root.h"
#include "utils/comm.h"
#include <limits>

namespace Training {
namespace { thread_local EFSFactorizationCounts *efs_counts=nullptr; }
void observe_efs_factorizations(EFSFactorizationCounts *counts) { efs_counts=counts; }
void count_efs_factorization(EFSFactorization kind) {
    if(efs_counts)++(*efs_counts)[static_cast<int>(kind)];
}
namespace {
constexpr int batch_size = 256;
using Rows = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

struct Accumulator {
    EFSQuadratic &out;
    Rows rows;
    Eigen::VectorXd residuals;
    int used = 0;
    bool root_mode;

    Accumulator(EFSQuadratic &result, int n,bool square_root)
        : out(result), rows(square_root ? n+1 : batch_size, n),
          residuals(rows.rows()),root_mode(square_root) {
        out.gram.setZero(n, n);
        out.residual_rhs.setZero(n);
        if(root_mode)out.square_root=Eigen::MatrixXd::Zero(n+1,n+1);
    }
    void flush() {
        if (!used) return;
        const auto batch = rows.topRows(used);
        if(root_mode) {
            Eigen::MatrixXd augmented(used,rows.cols()+1);
            augmented.leftCols(rows.cols())=batch;augmented.col(rows.cols())=residuals.head(used);
            out.square_root=merge_efs_roots(out.square_root,augmented,EFSFactorization::rows_qr);
            used=0;return;
        }
        out.gram.noalias() += batch.transpose() * batch;
        out.residual_rhs.noalias() += batch.transpose() * residuals.head(used);
        out.residual_norm += residuals.head(used).squaredNorm();
        used = 0;
    }
    void add(const double *row, double target, double weight,
             const Eigen::VectorXd &full_anchor) {
        if (!std::isfinite(weight) || weight < 0)
            GPERROR("Invalid EFS observation weight");
        if (weight == 0) return;
        Eigen::Map<const Eigen::VectorXd> full(row, full_anchor.size());
        if (!full.allFinite() || !std::isfinite(target))
            GPERROR("Non-finite EFS observation or basis component");
        const double scale = std::sqrt(weight);
        rows.row(used) = scale * full.tail(rows.cols()).transpose();
        residuals[used] = scale * (target - full.dot(full_anchor));
        ++out.observations;
        if (++used == rows.rows()) flush();
    }
};

void sum_buffer(double *data, Eigen::Index size) {
    // Keep MPI counts bounded without doubling the whole matrix's memory.
    constexpr Eigen::Index chunk_size = 65536;
    Eigen::VectorXd buffer(std::min(size, chunk_size));
    for (Eigen::Index pos = 0; pos < size; pos += chunk_size) {
        int count = static_cast<int>(std::min(size - pos, chunk_size));
        MPI_Allreduce(data + pos, buffer.data(), count, MPI_DOUBLE, MPI_SUM, mpi.comm);
        std::copy_n(buffer.data(), count, data + pos);
    }
}
}

double EFSQuadratic::loss(const Eigen::VectorXd &delta) const {
    if(square_root.size()) {
        const double value=(square_root.leftCols(delta.size())*delta-square_root.col(delta.size())).squaredNorm();
        if(!std::isfinite(value))GPERROR("Invalid square-root EFS loss");
        return value;
    }
    const double quadratic = delta.dot(gram * delta);
    const double linear = 2 * delta.dot(residual_rhs);
    const double value = quadratic - linear + residual_norm;
    const double tolerance = 128 * std::numeric_limits<double>::epsilon() *
        (std::abs(quadratic) + std::abs(linear) + residual_norm);
    if (!std::isfinite(value) || value < -tolerance)
        GPERROR("Invalid cached EFS loss");
    return std::max(0., value);
}

LinearEFSSystem::LinearEFSSystem(Potentials::PotMTP &pot, std::vector<Atoms> &atoms,
                               const EFSFitWeights &weights, int offset,bool square_root) {
    if (offset < 0 || offset >= pot.lin_coeff_size)
        GPERROR("Invalid linear coefficient offset");
    Eigen::VectorXd full_anchor = Eigen::Map<Eigen::VectorXd>(pot.linear_coeffs, pot.lin_coeff_size);
    anchor = full_anchor.tail(pot.lin_coeff_size - offset);
    Accumulator energy(channels[0], anchor.size(),square_root);
    Accumulator forces(channels[1], anchor.size(),square_root);
    Accumulator stress(channels[2], anchor.size(),square_root);
    std::vector<double> e;
    Eigen::Tensor<double, 3, Eigen::RowMajor> f, s;
    for (Atoms &a : atoms) {
        if (a.size() == 0 || (a.has_forces() && a.forces_.size() != size_t(a.size())) ||
            (!a.atomic_fitness.empty() && a.atomic_fitness.size() != size_t(a.size())))
            GPERROR("Invalid EFS training structure or atomic weights");
        const double norm = weights.normalization(a);
        Neighborhoods nbhs(a, pot.cutoff);
        if ((weights.force > 0 && a.has_forces()) || (weights.stress > 0 && a.has_stresses()))
            pot.CalcEFSCmpt_R(a, nbhs, e, f, s);
        else if (weights.energy > 0 && a.has_energy())
            pot.CalcECmpt(a, nbhs, e);
        else continue;
        if (a.has_energy())
            energy.add(e.data(), a.energy_, weights.energy_weight(a) * norm, full_anchor);
        if (weights.force > 0 && a.has_forces()) {
            for (int i = 0; i < a.size(); ++i)
                for (int d = 0; d < 3; ++d)
                    forces.add(&f(i, d, 0), a.forces_[i][d],
                        EFSFitWeights::atomic_weight(a, i, weights.force_weight(a) * norm), full_anchor);
        }
        if (weights.stress > 0 && a.has_stresses()) {
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
                stress.add(&s(i, j, 0), a.stress_(i, j),
                           weights.stress_weight(a) * norm, full_anchor);
        }
    }
    energy.flush(); forces.flush(); stress.flush();
}

void LinearEFSSystem::reduce() {
    for (auto &channel : channels) {
        if(channel.square_root.size()) {
            // Binary TSQR reduction, then the same bounded factor on all ranks.
#ifdef IMRMLP_MPI
            const int count=channel.square_root.size();
            for(int stride=1;stride<mpi.size;stride*=2) {
                if(mpi.rank%(2*stride)==stride) {
                    MPI_Send(channel.square_root.data(),count,MPI_DOUBLE,mpi.rank-stride,371,mpi.comm);break;
                }
                if(mpi.rank%(2*stride)==0 && mpi.rank+stride<mpi.size) {
                    Eigen::MatrixXd incoming(channel.square_root.rows(),channel.square_root.cols());
                    MPI_Recv(incoming.data(),count,MPI_DOUBLE,mpi.rank+stride,371,mpi.comm,MPI_STATUS_IGNORE);
                    channel.square_root=merge_efs_roots(channel.square_root,incoming,EFSFactorization::mpi_merge_qr);
                }
            }
#endif
            MPI_Bcast(channel.square_root.data(),channel.square_root.size(),MPI_DOUBLE,0,mpi.comm);
            const auto design=channel.square_root.leftCols(anchor.size());
            const auto residual=channel.square_root.col(anchor.size());
            channel.gram=design.transpose()*design;
            channel.residual_rhs=design.transpose()*residual;
            channel.residual_norm=residual.squaredNorm();
            double total=0;MPI_Allreduce(&channel.observations,&total,1,MPI_DOUBLE,MPI_SUM,mpi.comm);
            channel.observations=total;continue;
        }
        sum_buffer(channel.gram.data(), channel.gram.size());
        sum_buffer(channel.residual_rhs.data(), channel.residual_rhs.size());
        double local[2] = {channel.residual_norm, channel.observations}, global[2];
        MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, mpi.comm);
        channel.residual_norm = global[0]; channel.observations = global[1];
    }
}

Eigen::MatrixXd LinearEFSSystem::root_factor(double we,double wf,double ws) const {
    Eigen::MatrixXd result=Eigen::MatrixXd::Zero(anchor.size()+1,anchor.size()+1);
    const double weights[]={we,wf,ws};
    for(int i=0;i<3;++i)if(weights[i]>0) {
        if(!channels[i].square_root.size())GPERROR("Square-root assembly was not enabled");
        result=merge_efs_roots(result,std::sqrt(weights[i])*channels[i].square_root);
    }
    return result;
}

Eigen::MatrixXd LinearEFSSystem::matrix(double we, double wf, double ws) const {
    return we * channels[0].gram + wf * channels[1].gram + ws * channels[2].gram;
}
Eigen::VectorXd LinearEFSSystem::rhs(double we, double wf, double ws) const {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(anchor.size());
    const double weights[] = {we, wf, ws};
    for (int i = 0; i < 3; ++i)
        result += weights[i] * (channels[i].residual_rhs + channels[i].gram * anchor);
    return result;
}
double LinearEFSSystem::loss(const Eigen::VectorXd &coeffs, double we, double wf, double ws) const {
    if (coeffs.size() != anchor.size()) GPERROR("Invalid cached EFS coefficient count");
    const Eigen::VectorXd delta = coeffs - anchor;
    return we * channels[0].loss(delta) + wf * channels[1].loss(delta) + ws * channels[2].loss(delta);
}
void LinearEFSSystem::require_observations(double we, double wf, double ws) const {
    if (!(we * channels[0].observations + wf * channels[1].observations + ws * channels[2].observations > 0))
        GPERROR("Dataset has no positive-weight E/F/S observations");
}
} // namespace Training
