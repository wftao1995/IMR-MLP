// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "linear_refit.h"
#include "conditional_linear.h"
#include "potentials/potential_mtp_LES.h"
#include "utils/comm.h"
#include <Eigen/QR>
#include <Eigen/SVD>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <typeinfo>

namespace {
using namespace Potentials;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
void collective_error(const std::string& local) {
    int failed=local.empty() ? 0 : 1;
#ifdef IMRMLP_MPI
    int any=0; MPI_Allreduce(&failed,&any,1,MPI_INT,MPI_MAX,mpi.comm);
    failed=any;
#endif
    if(failed) GPERROR((local.empty() ? "Linear refit failed on another MPI rank" : local));
}
double global_sum(double value) {
#ifdef IMRMLP_MPI
    double sum=0.; MPI_Allreduce(&value,&sum,1,MPI_DOUBLE,MPI_SUM,mpi.comm); return sum;
#else
    return value;
#endif
}
Eigen::VectorXd observations(const Atoms& a) {
    Eigen::VectorXd y(1+3*a.size()+9); y[0]=a.energy_;
    for(int i=0;i<a.size();++i) for(int d=0;d<3;++d) y[1+3*i+d]=a.forces_[i][d];
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) y[1+3*a.size()+3*i+j]=a.stress_(i,j);
    return y;
}
// Read only labelled observations: absent arrays must never be dereferenced.
Eigen::VectorXd reference_values(const Atoms& a) {
    Eigen::VectorXd y=Eigen::VectorXd::Zero(1+3*a.size()+9);
    if(a.has_energy()) y[0]=a.energy_;
    if(a.has_forces()) for(int i=0;i<a.size();++i) for(int d=0;d<3;++d)
        y[1+3*i+d]=a.forces_[i][d];
    if(a.has_stresses()) for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        y[1+3*a.size()+3*i+j]=a.stress_(i,j);
    return y;
}
Eigen::VectorXd row_weights(const Atoms& a,const EFSFitWeights& w) {
    if(!a.atomic_fitness.empty() && a.atomic_fitness.size()!=size_t(a.size()))
        GPERROR("Invalid atomic_fitness length in linear refit");
    if(a.has_forces() && a.forces_.size()!=size_t(a.size())) GPERROR("Invalid reference forces");
    Eigen::VectorXd out=Eigen::VectorXd::Zero(1+3*a.size()+9);
    const double c=w.normalization(a);
    if(a.has_energy()) out[0]=w.energy_weight(a)*c;
    if(a.has_forces()) for(int i=0;i<a.size();++i)
        out.segment(1+3*i,3).setConstant(EFSFitWeights::atomic_weight(a,i,w.force_weight(a)*c));
    if(a.has_stresses()) out.tail(9).setConstant(w.stress_weight(a)*c);
    if(!out.allFinite() || out.minCoeff()<0.) GPERROR("Invalid observation weights in linear refit");
    return out;
}
Atoms matrix_probes(const Atoms &atom) {
    Atoms probes=atom;
    const int ndof=3*atom.size();
    probes.hvp_directions_=Eigen::MatrixXd::Identity(ndof,ndof);
    probes.hvp_targets_=atom.hessian_;
    return probes;
}
Eigen::Index curvature_rows(const Atoms &atom,const LinearRefitSettings &settings) {
    const Eigen::Index ndof=3*Eigen::Index(atom.size());
    return (settings.hvp.weight>0. && atom.has_hvp() ? ndof*atom.hvp_directions_.cols() : 0)
        +(settings.hessian.weight>0. && atom.has_hessian() ? ndof*ndof : 0)
        +(settings.charge_weight ? 1 : 0);
}
Eigen::VectorXd all_reference_values(const Atoms &atom,const LinearRefitSettings &settings) {
    const auto base=reference_values(atom);
    Eigen::VectorXd out(base.size()+curvature_rows(atom,settings));out.head(base.size())=base;
    Eigen::Index row=base.size();
    if(settings.hvp.weight>0. && atom.has_hvp()) {
        for(int k=0;k<atom.hvp_targets_.cols();++k) for(int i=0;i<atom.hvp_targets_.rows();++i)
            out[row++]=atom.hvp_targets_(i,k);
    }
    if(settings.hessian.weight>0. && atom.has_hessian()) {
        for(int k=0;k<atom.hessian_.cols();++k) for(int i=0;i<atom.hessian_.rows();++i)
            out[row++]=atom.hessian_(i,k);
    }
    if(settings.charge_target) out[row++]=settings.charge_target(atom);
    return out;
}
Eigen::VectorXd all_row_weights(const Atoms &atom,const EFSFitWeights &weights,const LinearRefitSettings &settings) {
    const auto base=row_weights(atom,weights);
    Eigen::VectorXd out(base.size()+curvature_rows(atom,settings));out.head(base.size())=base;
    Eigen::Index row=base.size();const double normalization=weights.normalization(atom);
    if(settings.hvp.weight>0. && atom.has_hvp()) {
        const int rows=atom.hvp_targets_.size();
        out.segment(row,rows).setConstant(settings.hvp.observation_weight(atom,normalization));row+=rows;
    }
    if(settings.hessian.weight>0. && atom.has_hessian()) {
        out.segment(row,atom.hessian_.size()).setConstant(settings.hessian.weight*atom.fitness*normalization
            /std::pow(atom.size(),settings.hessian.power));row+=atom.hessian_.size();
    }
    if(settings.charge_weight) out[row++]=settings.charge_weight(atom);
    if(!out.allFinite() || out.minCoeff()<0.) GPERROR("Invalid curvature observation weights in linear refit");
    return out;
}
Eigen::VectorXd all_observations(AnyPotential &potential,const Atoms &atom,const LinearRefitSettings &settings) {
    const auto base=observations(atom);
    Eigen::VectorXd out(base.size()+curvature_rows(atom,settings));out.head(base.size())=base;
    Eigen::Index row=base.size();
    Training::ForceObservation forces=[&](Atoms &scratch,Neighborhoods &neighbors) {
        scratch.resetEFS();potential.getEFS(scratch,neighbors);
        if(scratch.forces_.size()!=size_t(scratch.size())) GPERROR("Invalid refit force prediction count");
        Eigen::MatrixXd force(3*scratch.size(),1);
        for(int i=0;i<scratch.size();++i) force.col(0).segment<3>(3*i)=scratch.forces_[i].transpose();
        return force;
    };
    if(settings.hvp.weight>0. && atom.has_hvp()) {
        const auto values=Training::finite_difference_hvp_observations(atom,settings.hvp.step,potential.cutoff,forces);
        out.segment(row,values.rows())=values.col(0);row+=values.rows();
    }
    if(settings.hessian.weight>0. && atom.has_hessian()) {
        const auto values=Training::finite_difference_hvp_observations(matrix_probes(atom),settings.hessian.step,potential.cutoff,forces);
        out.segment(row,values.rows())=values.col(0);row+=values.rows();
    }
    if(settings.charge_weight) {
        // Curvature probes change model caches. Restore the undisplaced forward
        // state before asking for this model's total charge.
        auto copy=atom;Neighborhoods neighbors(copy,potential.cutoff);
        copy.resetEFS();potential.getEFS(copy,neighbors);
        out[row++]=potential.predicted_total_charge(copy);
    }
    return out;
}
// The model supplies explicit affine indices; the established QR/SVD solve
// below is shared by every accepted backend.
struct Adapter {
    AnyPotential& potential;
    Training::ConditionalLinearAdapter affine;
    PotMTP* sr=nullptr;
    PotMTP_LES* sog=nullptr;
    int n, composition_count;
    bool legacy_basis=false;
    Adapter(AnyPotential& p,const std::string& mode)
        :potential(p),affine(p,Training::conditional_linear_selection(p,mode),0.),
         n(affine.size()),composition_count(affine.composition_count) {
        if(typeid(p)==typeid(PotMTP) || typeid(p)==typeid(PotMTP_LES)) sr=static_cast<PotMTP*>(&p);
        if(typeid(p)==typeid(PotMTP_LES)) sog=static_cast<PotMTP_LES*>(&p);
        if(sog && !sog->lr_les) GPERROR("Linear refit requires an active LES/SOG model");
        legacy_basis=sr && (mode=="sr" || mode=="sr_sog");
        if(!p.inited()) GPERROR("Linear refit requires initialized model coefficients");
    }
    Eigen::VectorXd get() const {return affine.get();}
    void set(const Eigen::VectorXd& x) {affine.set(x);}
    bool selected(int i) const {return affine.selected(i);}
    Eigen::VectorXd regularization(double penalty_weight) const {
        // Penalty coefficients can depend on the radial-penalty setting (SOG).
        Training::ConditionalLinearAdapter current(potential,affine.blocks,penalty_weight);
        if((current.l1.array()!=0.).any()) GPERROR("Linear refit does not support nonzero selected L1 regularization");
        if(!current.l2.allFinite() || current.l2.minCoeff()<0.) GPERROR("Invalid linear refit L2 regularization");
        return current.l2;
    }
    Eigen::MatrixXd basis(Atoms& a,Neighborhoods& nbhs) {
        if(!legacy_basis) return affine.basis(a);
        std::vector<double> e; Eigen::Tensor<double,3> f,v;
        sr->CalcEFSCmpt(a,nbhs,e,f,v);
        const int nsr=sr->get_lin_coeff_size();
        Eigen::MatrixXd out(1+3*a.size()+9,n);
        for(int m=0;m<nsr;++m) {
            out(0,m)=e[m];
            for(int i=0;i<a.size();++i) for(int d=0;d<3;++d) out(1+3*i+d,m)=f(i,m,d);
            for(int i=0;i<3;++i) for(int j=0;j<3;++j) out(1+3*a.size()+3*i+j,m)=v(m,i,j);
        }
        if(n>nsr) out.rightCols(n-nsr)=sog->linear_sog_components(a,nbhs);
        return out;
    }
    Eigen::MatrixXd observation_basis(Atoms &atom,Neighborhoods &nbhs,const LinearRefitSettings &settings) {
        const auto base=basis(atom,nbhs);
        Eigen::MatrixXd out(base.rows()+curvature_rows(atom,settings),n);out.topRows(base.rows())=base;
        Eigen::Index row=base.rows();
        Training::ForceObservation forces=[&](Atoms &scratch,Neighborhoods &neighbors) {
            return basis(scratch,neighbors).middleRows(1,3*scratch.size()).eval();
        };
        if(settings.hvp.weight>0. && atom.has_hvp()) {
            const auto values=Training::finite_difference_hvp_observations(atom,settings.hvp.step,potential.cutoff,forces);
            out.middleRows(row,values.rows())=values;row+=values.rows();
        }
        if(settings.hessian.weight>0. && atom.has_hessian()) {
            const auto values=Training::finite_difference_hvp_observations(matrix_probes(atom),settings.hessian.step,potential.cutoff,forces);
            out.middleRows(row,values.rows())=values;row+=values.rows();
        }
        if(settings.charge_weight) {
            auto copy=atom;copy.resetEFS();potential.getEFS(copy,nbhs);
            Eigen::VectorXd gradient=Eigen::VectorXd::Zero(potential.get_NL_coeffs_count());
            potential.AccumulateTotalChargeGard(copy,nbhs,1.,gradient.data());
            for(int j=0;j<n;++j) out(row,j)=gradient[affine.indices[j]];
            if(out.row(row).cwiseAbs().maxCoeff()!=0.)
                GPERROR("Conditional refit requires selected parameters independent of total charge");
        }
        return out;
    }
};
// QR of augmented [A,b] retains the residual constant as its final column.
// Storage is bounded by one configuration plus a square compressed factor.
void compress(Eigen::MatrixXd& factor,const Eigen::MatrixXd& rows) {
    if(rows.rows()==0) return;
    const int p=factor.cols();
    Eigen::MatrixXd stacked(p+rows.rows(),p);
    stacked.topRows(p)=factor; stacked.bottomRows(rows.rows())=rows;
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(stacked);
    factor=qr.matrixQR().topRows(p).template triangularView<Eigen::Upper>();
}
int rank(const Eigen::VectorXd& s,double relative,double absolute) {
    if(s.size()==0) return 0;
    return (s.array()>std::max(absolute,relative*s[0])).count();
}
void json_vector(std::ostream& out,const Eigen::VectorXd& v) {
    out<<'['; for(int i=0;i<v.size();++i) {if(i) out<<','; out<<v[i];} out<<']';
}
void report(const LinearRefitSettings& s,const LinearRefitResult& r,double configs,double rows) {
    if(s.report.empty()) return;
    std::ofstream out(s.report); if(!out) GPERROR("Cannot write linear refit report: "+s.report);
    out<<std::setprecision(17)<<"{\n\"status\":\"accepted\",\n\"mode\":\""<<s.mode<<"\",\n"
       <<"\"configurations\":"<<configs<<",\n\"observations\":"<<rows
       <<",\n\"parameters\":"<<r.parameters<<",\n\"composition_rank\":"<<r.composition_rank
       <<",\n\"data_rank\":"<<r.data_rank<<",\n\"solve_rank\":"<<r.solve_rank
       <<",\n\"rtol\":"<<s.rtol<<",\n\"atol\":"<<s.atol
       << ",\"hvp_weight\":"<<s.hvp.weight<<",\"hvp_power\":"<<s.hvp.power<<",\"hvp_fd_step\":"<<s.hvp.step
       << ",\"hessian_weight\":"<<s.hessian.weight<<",\"hessian_power\":"<<s.hessian.power<<",\"hessian_fd_step\":"<<s.hessian.step
       <<",\n\"before\":"<<r.before<<",\n\"after\":"<<r.after<<",\n\"predicted\":"<<r.predicted
       <<",\n\"data_before\":"<<r.data_before<<",\n\"data_after\":"<<r.data_after
       <<",\n\"penalty_before\":"<<r.penalty_before<<",\n\"penalty_after\":"<<r.penalty_after
       <<",\n\"optimality\":"<<r.optimality<<",\n\"assembly_seconds\":"<<r.assembly_seconds
       <<",\n\"solve_seconds\":"<<r.solve_seconds<<",\n\"verification_seconds\":"<<r.verification_seconds;
    out<<",\n\"charge_before\":"<<r.charge_before<<",\n\"charge_after\":"<<r.charge_after;
    out<<",\n\"blocks\":[";
    for(size_t i=0;i<r.blocks.size();++i) {if(i) out<<',';out<<std::quoted(r.blocks[i]);}out<<']';
    out<<",\n\"parameter_indices\":[";
    for(size_t i=0;i<r.parameter_indices.size();++i) {if(i) out<<',';out<<r.parameter_indices[i];}out<<']';
    out<<",\n\"delta\":";json_vector(out,r.delta);
    out<<",\n\"column_scales\":";json_vector(out,r.scales);
    out<<",\n\"singular_values\":";json_vector(out,r.singular_values);
    out<<",\n\"l2\":";json_vector(out,r.l2);out<<"\n}\n";
    if(!out) GPERROR("Failed writing linear refit report");
}
}

LinearRefitResult linear_refit(Potentials::AnyPotential& p,std::vector<Atoms>& data,
    std::vector<Neighborhoods>& nbhs,const EFSFitWeights& weights,double penalty_weight,
    const LinearRefitSettings& settings) {
    std::string error;
    std::unique_ptr<Adapter> adapter;
    try {
        if(settings.solver!="qr_svd") GPERROR("Unknown linear refit solver");
        if(!std::isfinite(settings.rtol) || settings.rtol<0. || settings.rtol>=1.
            || !std::isfinite(settings.atol) || settings.atol<0.
            || !std::isfinite(settings.acceptance_tolerance) || settings.acceptance_tolerance<=0.)
            GPERROR("Invalid linear refit tolerances");
        if(!std::isfinite(penalty_weight) || penalty_weight<0.) GPERROR("Invalid penalty weight");
        for(double v:{weights.energy,weights.force,weights.stress})
            if(!std::isfinite(v) || v<0.) GPERROR("Invalid EFS fit weight");
        settings.hvp.validate();settings.hessian.validate();
        if(bool(settings.charge_weight)!=bool(settings.charge_target))
            GPERROR("Charge refit observations require both weight and target callbacks");
        if(settings.charge_weight && !p.supports_total_charge_training())
            GPERROR("Charge refit observations require analytic total-charge training support");
        for(const auto &atom:data) {
            atom.validate_hvp();
            if(atom.have_hessian && (atom.size()==0 || !atom.has_hessian() || !atom.hessian_.allFinite()))
                GPERROR("Linear refit requires finite 3N by 3N Hessian labels");
        }
        if(data.size()!=nbhs.size()) GPERROR("Linear refit neighborhood count mismatch");
        adapter=std::make_unique<Adapter>(p,settings.mode);
    } catch(const std::exception& e) {error=e.what();}
    collective_error(error);
    auto& a=*adapter; const int n=a.n;
    const auto original=a.get();
    Eigen::VectorXd full_original(p.get_NL_coeffs_count());p.nl_get_coeffs(full_original.data());
    Eigen::VectorXd widths;
    if(a.sog && a.sog->uses_sog())
        widths=Eigen::Map<Eigen::VectorXd>(a.sog->model_les.sog_log_bandwidths.data(),a.sog->kernel_count());
    LinearRefitResult result;result.parameters=n;result.delta=Eigen::VectorXd::Zero(n);
    result.parameter_indices=a.affine.indices;result.blocks=a.affine.blocks;
    auto start=Clock::now();
    const double configs=global_sum(data.size());
    if(configs==0.) GPERROR("Linear refit has no configurations");
    Eigen::MatrixXd factor=Eigen::MatrixXd::Zero(n+1,n+1);
    double rows=0.;
    auto penalty=[&](){
        double value=0.; Eigen::VectorXd g=Eigen::VectorXd::Zero(p.get_NL_coeffs_count());
        p.AddPenaltyGard(penalty_weight,value,g.data());return value;
    };
    try {
        result.l2=a.regularization(penalty_weight);
        result.penalty_before=penalty();
        for(size_t k=0;k<data.size();++k) {
            if(data[k].size()==0) continue;
            const auto w=all_row_weights(data[k],weights,settings);
            rows+=(w.array()>0.).count();
            auto prediction=data[k];p.getEFS(prediction,nbhs[k]);
            const auto residual=(all_reference_values(data[k],settings)-all_observations(p,prediction,settings)).eval();
            Eigen::MatrixXd block(residual.size(),n+1);
            block.leftCols(n)=a.observation_basis(data[k],nbhs[k],settings);block.col(n)=residual;
            if(!block.allFinite()) GPERROR("Non-finite linear refit basis or reference");
            for(int i=0;i<w.size();++i) block.row(i)*=std::sqrt(w[i]/configs);
            result.data_before+=block.col(n).squaredNorm();
            if(settings.charge_weight) result.charge_before+=std::pow(block(block.rows()-1,n),2);
            compress(factor,block);
        }
        if(!factor.allFinite() || !std::isfinite(result.penalty_before)) GPERROR("Non-finite refit assembly");
    } catch(const std::exception& e) {error=e.what();}
    collective_error(error);
    result.data_before=global_sum(result.data_before);result.charge_before=global_sum(result.charge_before);rows=global_sum(rows);
    if(rows==0.) GPERROR("Linear refit has no weighted labelled observations");
    result.before=result.data_before+result.penalty_before;
#ifdef IMRMLP_MPI
    std::vector<double> gathered(mpi.rank==0 ? mpi.size*factor.size() : 0);
    MPI_Gather(factor.data(),factor.size(),MPI_DOUBLE,gathered.data(),factor.size(),MPI_DOUBLE,0,mpi.comm);
    if(mpi.rank==0) {
        factor.setZero();
        for(int i=0;i<mpi.size;++i)
            compress(factor,Eigen::Map<Eigen::MatrixXd>(gathered.data()+i*factor.size(),n+1,n+1));
    }
#endif
    result.assembly_seconds=elapsed(start);start=Clock::now();
    if(mpi.rank==0) try {
        const Eigen::MatrixXd design=factor.leftCols(n);
        const Eigen::VectorXd rhs=factor.col(n);
        const int species=a.composition_count;
        // Materialize blocks: some Eigen LAPACKE versions pass the original
        // outer stride together with a packed temporary to LAPACK.
        Eigen::MatrixXd composition_vectors;
        if(species>0) {
            // L2 can identify a species gauge that composition data alone do
            // not identify. Freeze only the remaining true nullspace.
            Eigen::MatrixXd composition_matrix=Eigen::MatrixXd::Zero(design.rows()+species,species);
            composition_matrix.topRows(design.rows())=design.leftCols(species);
            composition_matrix.bottomRows(species)=result.l2.head(species).cwiseSqrt().asDiagonal();
            Eigen::JacobiSVD<Eigen::MatrixXd> composition(composition_matrix,Eigen::ComputeFullV);
            if(composition.info()!=Eigen::Success) GPERROR("Composition SVD failed");
            result.composition_rank=rank(composition.singularValues(),settings.rtol,settings.atol);
            composition_vectors=composition.matrixV().leftCols(result.composition_rank);
        }
        // Freeze the physical composition nullspace, before column scaling.
        const int reduced=n-species+result.composition_rank;
        if(reduced==0) GPERROR("Linear refit has no identifiable parameters");
        Eigen::MatrixXd transform=Eigen::MatrixXd::Zero(n,reduced);
        if(species>0) transform.topLeftCorner(species,result.composition_rank)=composition_vectors;
        transform.bottomRightCorner(n-species,n-species).setIdentity();
        Eigen::MatrixXd matrix(factor.rows()+n,reduced);
        matrix.topRows(factor.rows())=design*transform;
        matrix.bottomRows(n)=result.l2.cwiseSqrt().asDiagonal()*transform;
        Eigen::VectorXd target(factor.rows()+n);
        target.head(factor.rows())=rhs;
        target.tail(n)=-result.l2.cwiseSqrt().cwiseProduct(original);
        result.scales=matrix.colwise().norm().transpose();
        const double floor=std::max(settings.atol,1e-300);
        for(int i=0;i<reduced;++i) {
            if(result.scales[i]<=floor) result.scales[i]=1.;
            matrix.col(i)/=result.scales[i];
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> data_svd(matrix.topRows(factor.rows()).eval());
        if(data_svd.info()!=Eigen::Success) GPERROR("Data SVD failed");
        result.data_rank=rank(data_svd.singularValues(),settings.rtol,settings.atol);
        Eigen::JacobiSVD<Eigen::MatrixXd> solve(matrix,Eigen::ComputeThinU|Eigen::ComputeThinV);
        if(solve.info()!=Eigen::Success) GPERROR("Linear refit SVD failed");
        result.singular_values=solve.singularValues();
        result.solve_rank=rank(result.singular_values,settings.rtol,settings.atol);
        Eigen::VectorXd z=Eigen::VectorXd::Zero(reduced);
        for(int i=0;i<result.solve_rank;++i)
            z+=solve.matrixV().col(i)*(solve.matrixU().col(i).dot(target)/result.singular_values[i]);
        result.delta=transform*(z.cwiseQuotient(result.scales));
        result.optimality=(solve.matrixV().leftCols(result.solve_rank).transpose()*matrix.transpose()*(matrix*z-target)).norm()
            /std::max(1.,matrix.norm()*target.norm());
        result.predicted=(design*result.delta-rhs).squaredNorm()
            +(original+result.delta).cwiseProduct(result.l2).dot(original+result.delta)
            +result.penalty_before-original.cwiseProduct(result.l2).dot(original);
        if(!result.delta.allFinite() || !std::isfinite(result.predicted) || result.optimality>1e-8)
            GPERROR("Linear refit failed first-order optimality check");
    } catch(const std::exception& e) {error=e.what();}
    collective_error(error);
#ifdef IMRMLP_MPI
    MPI_Bcast(result.delta.data(),n,MPI_DOUBLE,0,mpi.comm);
    MPI_Bcast(&result.predicted,1,MPI_DOUBLE,0,mpi.comm);
#endif
    result.solve_seconds=elapsed(start); start=Clock::now();
    // The complete prediction is the independent acceptance test. Roll back
    // even when failure occurs only on one MPI rank or during serialization.
    try {
        a.set(original+result.delta);
        Eigen::VectorXd current(full_original.size());p.nl_get_coeffs(current.data());
        for(int i=0;i<current.size();++i)
            if(!a.selected(i) && current[i]!=full_original[i])
                GPERROR("Linear refit changed a frozen parameter");
        if(widths.size()>0 && !(widths.array()==Eigen::Map<Eigen::VectorXd>(a.sog->model_les.sog_log_bandwidths.data(),widths.size()).array()).all())
            GPERROR("Linear refit changed a frozen bandwidth");
        std::unique_ptr<PotMTP> reloaded;
        // Retain the established in-memory persistence check for legacy MTP.
        // Other backends have native bundle checks at final publication.
        if(a.sr) {
            std::ostringstream saved;p.Save(saved);
            if(a.sog) reloaded=std::make_unique<PotMTP_LES>(); else reloaded=std::make_unique<PotMTP>();
            std::istringstream input(saved.str());reloaded->Load(input);
        }
        for(size_t k=0;k<data.size();++k) {
            if(data[k].size()==0) continue;
            auto prediction=data[k];p.getEFS(prediction,nbhs[k]);
            const auto values=all_observations(p,prediction,settings);
            if(!values.allFinite()) GPERROR("Non-finite refitted prediction");
            const auto residual=(all_reference_values(data[k],settings)-values).eval();
            const auto observation_weights=all_row_weights(data[k],weights,settings);
            result.data_after+=residual.array().square().matrix().dot(observation_weights)/configs;
            if(settings.charge_weight) result.charge_after+=std::pow(residual[residual.size()-1],2)
                *observation_weights[observation_weights.size()-1]/configs;
            if(k==0 && reloaded) {
                auto restored=data[k];reloaded->getEFS(restored,nbhs[k]);
                if((all_observations(*reloaded,restored,settings)-values).norm()>1e-10*std::max(1.,values.norm()))
                    GPERROR("Linear refit model roundtrip mismatch");
            }
        }
        result.penalty_after=penalty();
    } catch(const std::exception& e) {error=e.what();}
    try {collective_error(error);} catch(...) {a.set(original);throw;}
    result.data_after=global_sum(result.data_after);result.charge_after=global_sum(result.charge_after);
    result.after=result.data_after+result.penalty_after;
    const double tolerance=settings.acceptance_tolerance*std::max(1.,std::abs(result.before));
    if(!std::isfinite(result.after) || result.after>result.before+tolerance || std::abs(result.after-result.predicted)>tolerance) {
        a.set(original);GPERROR("Linear refit rejected: full objective increased or disagrees with design matrix");
    }
    result.verification_seconds=elapsed(start);
    if(mpi.rank==0) try {report(settings,result,configs,rows);} catch(const std::exception& e) {error=e.what();}
    try {collective_error(error);} catch(...) {a.set(original);throw;}
    if(mpi.rank==0) Message("Linear refit accepted: "+std::to_string(result.before)+" -> "+std::to_string(result.after));
    return result;
}
