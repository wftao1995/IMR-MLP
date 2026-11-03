// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "runtime/mtp/gpu-direct/gpu_backend.h"

#if defined(IMRMLP_MPI) && defined(IMRMLP_GPU)
#include "fixtures/gpu_linear_curvature.h"
#include "potentials/mtp_gpu_adapter.h"
#include "train/collective_error.h"
#include "utils/comm.h"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>

int main(int argc,char **argv) {
    int provided=0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed=20261029;
    session.configData().runOrder=Catch::TestRunOrder::Declared;
    const int result=session.run(argc,argv);
    MPI_Finalize();return result;
}

namespace {
namespace C=GpuLinearCurvatureFixture;
namespace G=MLIPCore::GpuDirect;
using Potentials::PotMTP;
void all_require(bool value) {
    int local=value,all=0;
    MPI_Allreduce(&local,&all,1,MPI_INT,MPI_MIN,mpi.comm);
    REQUIRE(all==1);
}
bool available() {
    int devices=0,all=0;std::string error;
    int local=G::MTPGpuDirect::available_device_count(&devices,&error)&&devices>0;
    MPI_Allreduce(&local,&all,1,MPI_INT,MPI_MIN,mpi.comm);
    if(std::getenv("IMRMLP_REQUIRE_GPU")) {INFO(error);REQUIRE(all==1);}
    return all;
}
#define REQUIRE_CURVATURE_MPI() \
    if(std::string(G::compiled_backend())!="cuda") SKIP("Analytic linear curvature has no qualified HIP route"); \
    if(mpi.size!=2) SKIP("Requires exactly two MPI ranks"); \
    if(!available()) SKIP("No CUDA device on an MPI rank")

struct Capture {
    std::ostringstream output;
    std::ostream *previous=SetStreamForOutput(&output);
    ~Capture() {SetStreamForOutput(previous);}
};
struct Shard {
    std::vector<Atoms> data;
    std::vector<Neighborhoods> neighbors;
    explicit Shard(const C::Sample &sample) {
        for(size_t k=size_t(mpi.rank);k<sample.data.size();k+=size_t(mpi.size)) {
            data.push_back(sample.data[k]);neighbors.push_back(sample.neighbors[k]);
        }
    }
};
GPSettings settings(const C::Options &options,bool gpu) {
    auto result=options.settings(gpu);
    // Sharing remains an explicit runner opt-in; never silently oversubscribe.
    result.Modify("gpu_id",gpu&&std::getenv("IMRMLP_TEST_SHARED_GPU")?"0":"-1");
    return result;
}
bool close(double actual,double reference,double atol=2e-8,double rtol=3e-9) {
    return std::isfinite(actual)&&std::isfinite(reference)&&
        std::abs(actual-reference)<=atol+rtol*std::abs(reference);
}
bool vector_close(const Eigen::VectorXd &actual,const Eigen::VectorXd &reference,
                  double scale=1.) {
    if(actual.size()!=reference.size()||!actual.allFinite()||!reference.allFinite())return false;
    for(Eigen::Index i=0;i<actual.size();++i)
        if(!close(actual[i]/scale,reference[i]/scale,C::gpu_atol,C::gpu_rtol))return false;
    return true;
}
void physical_predictions(PotMTP &actual,PotMTP &reference,const C::Sample &sample) {
    bool good=true;
    Training::collective_try([&] {
        for(size_t k=0;k<sample.data.size();++k) {
            auto left=sample.data[k],right=sample.data[k];auto nbh=sample.neighbors[k];
            actual.getEFS(left,nbh);reference.getEFS(right,nbh);
            good=good&&close(left.energy_,right.energy_,2e-7,2e-8);
            for(size_t i=0;i<left.size();++i)for(int a=0;a<3;++a)
                good=good&&close(left.forces_[i][a],right.forces_[i][a],2e-7,2e-8);
            for(int a=0;a<3;++a)for(int b=0;b<3;++b)
                good=good&&close(left.stress_(a,b),right.stress_(a,b),2e-7,2e-8);
            if(left.has_hessian())for(int j=0;j<3*left.size();++j) {
                Eigen::VectorXd v=Eigen::VectorXd::Zero(3*left.size()),x,y;v[j]=1.;
                actual.applyHessianVectorProduct(left,nbh,v,x);
                reference.applyHessianVectorProduct(right,nbh,v,y);
                // All components are physical predictions, independent of the
                // selected upper-triangle training rows and their target noise.
                good=good&&vector_close(x,y);
            }
            if(left.has_hvp())for(Eigen::Index j=0;j<left.hvp_directions_.cols();++j) {
                const Eigen::VectorXd v=left.hvp_directions_.col(j);Eigen::VectorXd x,y;
                actual.applyHessianVectorProduct(left,nbh,v,x);
                reference.applyHessianVectorProduct(right,nbh,v,y);
                good=good&&vector_close(x,y,C::F::maximum_atom_norm(v));
            }
        }
    });
    all_require(good);
}
void root_ridge(PotMTP &model) {
    MPI_Bcast(model.reg_L2_vector.data(),model.lin_coeff_size,MPI_DOUBLE,0,mpi.comm);
}
void check_fit(Trainer_MTP &gpu,PotMTP &actual,PotMTP &expected,PotMTP &anchor,
               const C::Sample &sample,const C::Options &options,bool empty) {
    root_ridge(expected);
    const auto &report=gpu.GetGpuLinearReport();
    bool good=report.verified&&report.backend=="cuda"&&
        report.algorithm==(options.fast?"fast/eigen_ldlt":"ordinary/alglib")&&
        report.effective_l2.size()==size_t(actual.lin_coeff_size);
    if(good)for(int i=0;i<actual.lin_coeff_size;++i)
        good=good&&close(report.effective_l2[i],expected.reg_L2_vector[i],1e-12,2e-8)&&
            close(actual.reg_L2_vector[i],expected.reg_L2_vector[i],1e-12,2e-8);
    all_require(good);
    anchor.reg_L2_vector=report.effective_l2;
    double before=0.,after=0.,objective_before=0.,objective_after=0.,cpu_objective=0.;
    Training::collective_try([&] {
        before=C::data_loss(anchor,sample.data,sample.neighbors,options);
        after=C::data_loss(actual,sample.data,sample.neighbors,options);
        objective_before=C::objective(anchor,sample.data,sample.neighbors,options);
        objective_after=C::objective(actual,sample.data,sample.neighbors,options);
        cpu_objective=C::objective(expected,sample.data,sample.neighbors,options);
    });
    all_require(close(report.data_before,before)&&close(report.data_after,after));
    all_require(close(report.objective_before,objective_before)&&close(report.objective_after,objective_after)&&
        close(report.objective_after,cpu_objective));
    all_require(close(report.predicted_data_before,report.data_before)&&
        close(report.predicted_data_after,report.data_after));
    all_require(report.objective_after<=report.objective_before+1e-10*std::max(1.,std::abs(report.objective_before)));
    all_require(empty ? report.curvature_stats.blocks==0 : report.curvature_stats.blocks>0);
    all_require(report.curvature_stats.tracked_device_bytes<=report.options.gpu_memory_bytes+report.options.gpu_workspace_bytes&&
        report.curvature_stats.output_workspace_bytes<=report.options.gpu_workspace_bytes&&
        report.curvature_host_peak_bytes<=report.curvature_host_cap_bytes);
    physical_predictions(actual,expected,sample);
}
}

TEST_CASE("MPI analytic curvature fits match CPU with empty and uneven shards", "[gpu][mpi][linear-curvature][training]") {
    REQUIRE_CURVATURE_MPI();const int count=GENERATE(1,5);const bool fast=GENERATE(false,true);
    CAPTURE(count,fast);Capture capture;C::Sample sample(2,1,count);Shard shard(sample);
    C::Options options;options.fast=fast;options.diagonal=fast;options.fixed_species=count==1;
    auto cpu_model=sample.initial(),gpu_model=C::copy(*cpu_model);const auto initial=C::coefficients(*gpu_model);
    Trainer_MTP cpu(settings(options,false)),gpu(settings(options,true));
    cpu.init(shard.data,shard.neighbors,cpu_model.get());gpu.init(shard.data,shard.neighbors,gpu_model.get());
    auto anchor=C::copy(*gpu_model);
    all_require(shard.data.size()==size_t((count+1-mpi.rank)/2));
    cpu.TrainLinear();gpu.TrainLinear();
    check_fit(gpu,*gpu_model,*cpu_model,*anchor,sample,options,shard.data.empty());
    const auto final=C::coefficients(*gpu_model);
    all_require((final.head(gpu_model->radial_coeff_size).array()==initial.head(gpu_model->radial_coeff_size).array()).all());
    if(options.fixed_species)all_require((final.segment(gpu_model->radial_coeff_size,gpu_model->species_count).array()==
        initial.segment(gpu_model->radial_coeff_size,gpu_model->species_count).array()).all());
}

TEST_CASE("MPI curvature initialization rejects a rank-local invalid request before ownership and retries", "[gpu][mpi][linear-curvature][rejection]") {
    REQUIRE_CURVATURE_MPI();const int failure=GENERATE(0,1,2,3);CAPTURE(failure);
    Capture capture;C::Sample sample(2,1,5);Shard shard(sample);C::Options options;options.fast=true;
    auto model=sample.initial();const auto initial=C::coefficients(*model);
    const auto l1=model->reg_L1_vector,l2=model->reg_L2_vector;
    auto bad=settings(options,true);
    if(mpi.rank==1) {
        if(failure==0)shard.data.front().hvp_targets_(0,0)=std::numeric_limits<double>::quiet_NaN();
        if(failure==1)model->hc_correction=true;
        if(failure==2)bad.Modify("device","cpu");
        if(failure==3) {
            bad.Modify("gpu_curvature_host_mib","1");
            shard.data.front().features_["g4c2-memory-canary"]=std::string(1100*1024,'x');
        }
    }
    Trainer_MTP trainer(bad);std::string error;
    try {trainer.init(shard.data,shard.neighbors,model.get());}catch(const std::exception &e){error=e.what();}
    INFO(error);all_require(!error.empty());
    all_require((C::coefficients(*model).array()==initial.array()).all()&&
        model->reg_L1_vector==l1&&model->reg_L2_vector==l2);
    all_require(trainer.p_mtp==nullptr&&trainer.training_set_.empty()&&trainer.trainint_set_nbhs_.empty());
    if(mpi.rank==1) {
        if(failure==0)shard.data.front().hvp_targets_(0,0)=sample.data[1].hvp_targets_(0,0);
        if(failure==1)model->hc_correction=false;
        if(failure==3)shard.data.front().features_.erase("g4c2-memory-canary");
    }
    trainer.ApplySettings(settings(options,true));trainer.init(shard.data,shard.neighbors,model.get());
    trainer.TrainLinear();all_require(trainer.GetGpuLinearReport().verified);
    // The post-repair fit must also retain the independent physical objective.
    double objective=0.;Training::collective_try([&] {objective=C::objective(*model,sample.data,sample.neighbors,options);});
    all_require(close(trainer.GetGpuLinearReport().objective_after,objective));
}

TEST_CASE("MPI curvature preflight protects an initialized trainer from rank-local changes", "[gpu][mpi][linear-curvature][retry]") {
    REQUIRE_CURVATURE_MPI();const bool fast=GENERATE(false,true);CAPTURE(fast);
    Capture capture;C::Sample sample(2,1,5);Shard shard(sample);C::Options options;options.fast=fast;
    auto model=sample.initial();Trainer_MTP trainer(settings(options,true));
    trainer.init(shard.data,shard.neighbors,model.get());const auto initial=C::coefficients(*model);
    const auto l1=model->reg_L1_vector,l2=model->reg_L2_vector;
    for(int failure=0;failure<2;++failure) {
        if(mpi.rank==1) {
            if(!failure)trainer.training_set_.front().hvp_targets_(0,0)=std::numeric_limits<double>::quiet_NaN();
            else {GPSettings changed;changed.Modify("hvp_weight",".74");trainer.ApplySettings(changed);}
        }
        std::string error;try {trainer.TrainLinear();}catch(const std::exception &e){error=e.what();}
        INFO(error);all_require(!error.empty()&&!trainer.GetGpuLinearReport().verified);
        all_require((C::coefficients(*model).array()==initial.array()).all()&&
            model->reg_L1_vector==l1&&model->reg_L2_vector==l2);
        if(mpi.rank==1&&!failure)trainer.training_set_.front().hvp_targets_(0,0)=sample.data[1].hvp_targets_(0,0);
        trainer.ApplySettings(settings(options,true));
    }
    auto anchor=C::copy(*model),expected=sample.initial();Trainer_MTP cpu(settings(options,false));
    cpu.init(shard.data,shard.neighbors,expected.get());cpu.TrainLinear();trainer.TrainLinear();
    check_fit(trainer,*model,*expected,*anchor,sample,options,false);
}
#endif
