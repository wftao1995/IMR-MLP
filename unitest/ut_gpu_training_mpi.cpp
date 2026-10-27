// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "runtime/mtp/gpu-direct/gpu_backend.h"
#if defined(IMRMLP_MPI) && defined(IMRMLP_GPU)
#include "train/train_nonlinear_potential.h"
#include "train/collective_error.h"
#include "potentials/potential_mtp.h"
#include "potentials/potential_mtp_LES.h"
#include "unitest/les_curvature_fixture.h"
#include <cstdlib>
#include <filesystem>
#include <unistd.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed = 20261001;
    session.configData().runOrder = Catch::TestRunOrder::Declared;
    const int result = session.run(argc, argv);
    MPI_Finalize();
    return result;
}
namespace {
// Each CPU reference runs independently, but failures belong to the world
// test phase. Restore the communicator before collective_try agrees on them.
template<class F> void collective_self_baseline(F&& operation) {
    Training::collective_try([&] {
        struct RestoreWorld {
            ~RestoreWorld() {mpi.InitComm(MPI_COMM_WORLD);}
        } restore;
        mpi.InitComm(MPI_COMM_SELF);
        std::forward<F>(operation)();
    });
}
void all_require(bool value) {
    int local = value, all = 0;
    MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, mpi.comm);
    REQUIRE(all == 1);
}
bool available() {
    int count=0, ok=0; std::string error;
    const bool local=MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count,&error) && count>0;
    int good=local;
    MPI_Allreduce(&good,&ok,1,MPI_INT,MPI_MIN,mpi.comm);
    if(std::getenv("IMRMLP_REQUIRE_GPU")) {INFO(error); REQUIRE(ok);}
    return ok;
}
template<class Potential=Potentials::PotMTP> struct Sample {
    Potential potential{"unitest/test-data/portable-mtp/les.mtp"};
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> neighbors;
    Sample(int count, bool distributed) {
        if constexpr (std::is_same_v<Potential,Potentials::PotMTP_LES>) {
            potential.les_kernel="sog";potential.sog_channels=1;potential.sog_terms=2;
            potential.train_init();
            potential.les_species_coeffs[0]=.2;potential.les_species_coeffs[1]=-.15;
            std::fill_n(potential.les_moment_coeffs,potential.alpha_scalar_moments,0.);
            potential.les_moment_coeffs[0]=.3;
        }
        potential.train_init(); potential.set_inited();
        for(int i=0;i<count;++i) {
            Atoms a; a.cell_=12.*Eigen::Matrix3d::Identity();
            a.pos_={{2.,2.,2.},{4.1+.02*i,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
            a.chemmap_={0,1,0,1}; a.fitness=1.; a.id=i;
            potential.getEFS(a); a.energy_+=.03;
            if(!distributed || i%mpi.size==mpi.rank) {
                atoms.push_back(a); neighbors.emplace_back(a,potential.cutoff);
            }
        }
        potential.radial_coeffs[0]+=.001;
    }
};
GPSettings settings(bool gpu) {
    GPSettings s; s.Modify("energy_weight","1"); s.Modify("force_weight",".2");
    s.Modify("stress_weight",".1"); s.Modify("penalty_weight","0");
    s.Modify("parameter_scaling","none"); s.Modify("gpu_batch_structures","1");
    if(gpu) {
        s.Modify("device",MLIPCore::GpuDirect::MTPGpuDirect::backend_name());
        if(std::getenv("IMRMLP_TEST_SHARED_GPU")) s.Modify("gpu_id","0");
    }
    return s;
}
}
TEST_CASE("MPI CPU baseline failures restore world and propagate collectively", "[gpu][mpi][failure][baseline]") {
    const int world_rank=mpi.rank,world_size=mpi.size;
    REQUIRE_THROWS_WITH(collective_self_baseline([&] {
        if(world_rank==world_size-1) throw std::runtime_error("injected CPU baseline failure");
    }),Catch::Matchers::ContainsSubstring("injected CPU baseline failure"));
    int relation=MPI_UNEQUAL;MPI_Comm_compare(mpi.comm,MPI_COMM_WORLD,&relation);
    all_require(relation==MPI_IDENT && mpi.rank==world_rank && mpi.size==world_size);
    collective_self_baseline([] {});
    all_require(mpi.rank==world_rank && mpi.size==world_size);
}

TEMPLATE_TEST_CASE("MPI GPU callbacks match CPU with uneven and empty partitions", "[gpu][mpi]", Potentials::PotMTP, Potentials::PotMTP_LES) {
    if(!available()) SKIP("No GPU");
    const int count=GENERATE(1,7);
    const bool device_loss=GENERATE(false,true);
    Sample<TestType> whole(count,false), local(count,true);
    double lc=0.;Eigen::VectorXd gc(whole.potential.get_NL_coeffs_count());
    collective_self_baseline([&] {
        TrainNonlinear cpu(settings(false));cpu.init(whole.atoms,whole.neighbors,&whole.potential);
        cpu.AccumulateGard(lc,gc);
    });
    auto s=settings(true);s.Modify("gpu_device_loss",device_loss?"true":"false");
    TrainNonlinear gpu(s); gpu.init(local.atoms,local.neighbors,&local.potential);
    Sample<TestType> validation(1,true); gpu.init_valid(validation.atoms,validation.neighbors);
    double lg=0.;Eigen::VectorXd gg(gc.size());
    for(int repeat=0;repeat<2;++repeat) {
        gpu.AccumulateGard(lg,gg);
        all_require(std::abs(lg-lc)<1e-10+1e-9*std::abs(lc));
        all_require((gg-gc).norm()<1e-9+1e-8*gc.norm());
    }
}
TEST_CASE("MPI GPU rank-local failures exit collectively and allow a fresh run", "[gpu][mpi][failure]") {
    if(!available()) SKIP("No GPU");
    const int failure=GENERATE(0,1,2,3,4,5);
    Sample<> local(mpi.size*2,true);
    auto s=settings(true);
    s.Modify("epoch_limit","1");
    if(failure==0 && mpi.rank==mpi.size-1) s.Modify("gpu_id","999999");
    if(failure==1 && mpi.rank==mpi.size-1) s.Modify("gpu_memory_fraction","1e-30");
    if(failure==2 && mpi.rank==mpi.size-1) s.Modify("gpu_memory_fraction","-1");
    std::string message;
    try {
        TrainNonlinear gpu(s);gpu.init(local.atoms,local.neighbors,&local.potential);
        if((failure==3 || failure==5) && mpi.rank==mpi.size-1)
            gpu.training_set_[0].chemmap_[0]=999999; // local forward rejection
        if(failure==4) {
            Sample<> validation(mpi.size,true);
            gpu.init_valid(validation.atoms,validation.neighbors);
            if(mpi.rank==mpi.size-1) gpu.valid_set_[0].chemmap_[0]=999999;
        }
        double loss=0.;Eigen::VectorXd g(local.potential.get_NL_coeffs_count());
        if(failure==5) gpu.TrainSGD();
        else gpu.AccumulateGard(loss,g);
    } catch(const std::exception &e) { message=e.what(); }
    all_require(!message.empty());
    // Every rank must receive the first failure's message, before any next reduction.
    int length=message.size();MPI_Bcast(&length,1,MPI_INT,0,mpi.comm);
    std::string root=mpi.rank==0?message:std::string(length,'\0');
    MPI_Bcast(root.data(),length,MPI_CHAR,0,mpi.comm);all_require(message==root);
    TrainNonlinear recovered(settings(true));
    recovered.init(local.atoms,local.neighbors,&local.potential);
    double loss=0.;Eigen::VectorXd g(local.potential.get_NL_coeffs_count());
    recovered.AccumulateGard(loss,g);all_require(std::isfinite(loss)&&g.allFinite());
}

TEMPLATE_TEST_CASE("MPI GPU CG accepts EFS steps with empty shards and SOG stages", "[gpu][mpi][cg][stages]",
    Potentials::PotMTP,Potentials::PotMTP_LES) {
    if(!available()) SKIP("No GPU");
    const int count=GENERATE(1,7);
    Sample<TestType> local(count,true);
    long root_pid=mpi.rank==0 ? getpid() : 0;MPI_Bcast(&root_pid,1,MPI_LONG,0,mpi.comm);
    const auto directory=std::filesystem::temp_directory_path()/("atoms-agent-mpi-gpu-cg-"+std::to_string(root_pid));
    Training::collective_try([&] {if(mpi.rank==0) std::filesystem::create_directories(directory);});
    const auto path=(directory/"result.mtp").string();
    auto s=settings(true);s.Modify("save_to",path);s.Modify("bfgs_maxiteration","2");
    s.Modify("bfgs_grad_tol","0");s.Modify("bfgs_conv_tol","0");s.Modify("bfgs_step_tol","0");s.Modify("save_every","100");
    if constexpr(std::is_same_v<TestType,Potentials::PotMTP_LES>) {
        s.Modify("training_schedule","mtp_sog_from_scratch");
        s.Modify("sr_stage_iterations","1");s.Modify("les_stage_iterations","1");s.Modify("joint_stage_iterations","1");
    }
    TrainNonlinear gpu(s);gpu.init(local.atoms,local.neighbors,&local.potential);
    Sample<TestType> valid(1,true);gpu.init_valid(valid.atoms,valid.neighbors);
    double before=0.;Eigen::VectorXd gradient(local.potential.get_NL_coeffs_count());gpu.AccumulateGard(before,gradient);
    gpu.TrainCG();all_require(gpu.cg_termination_type>0 && gpu.bfgs_accepted_iter>0);
    double after=0.;gpu.AccumulateGard(after,gradient);all_require(std::isfinite(after) && gradient.allFinite());
    if constexpr(std::is_same_v<TestType,Potentials::PotMTP>) all_require(after<before);
    all_require(gpu.validation_evaluations>0);
    double root_loss=after;MPI_Bcast(&root_loss,1,MPI_DOUBLE,0,mpi.comm);all_require(std::abs(after-root_loss)<1e-14);
    int saved=mpi.rank==0 ? std::filesystem::exists(path) : 1;all_require(saved);
    if constexpr(std::is_same_v<TestType,Potentials::PotMTP_LES>) {
        saved=mpi.rank==0 ? std::filesystem::exists(path+".sr.snapshot") && std::filesystem::exists(path+".sog.snapshot") : 1;
        all_require(saved);
    }
    MPI_Barrier(mpi.comm);if(mpi.rank==0) std::filesystem::remove_all(directory);
}

namespace {
using NN2SOG=Potentials::PotMTP_M<Potentials::m_model_nn2>;
struct NN2Sample {
    NN2SOG potential;
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> neighbors;
    NN2Sample(int count,bool distributed) {
        LESCurvature::initialize(potential,true);potential.set_inited();
        for(int i=0;i<count;++i) {
            auto a=LESCurvature::geometry();a.pos_[1][0]+=.02*i;a.id=i;
            potential.getEFS(a);a.energy_+=.03;
            if(!distributed || i%mpi.size==mpi.rank) {
                atoms.push_back(a);neighbors.emplace_back(a,potential.cutoff);
            }
        }
        potential.radial_coeffs[0]+=.001;
    }
};
void mpi_run_optimizer(TrainNonlinear& trainer,const std::string& optimizer) {
    if(optimizer=="cg") trainer.TrainCG();
    else if(optimizer=="bfgs") trainer.TrainBFGS();
    else if(optimizer=="sgd") trainer.TrainSGD();
    else trainer.TrainShLBFGS();
}
std::vector<double> mpi_coefficients(const Potentials::AnyPotential& p) {
    std::vector<double> c(p.get_NL_coeffs_count());p.nl_get_coeffs(c.data());return c;
}
}
TEST_CASE("MPI GPU NN2 SOG stages match whole-data CPU accepted results", "[gpu][mpi][cg][stages][nn2]") {
    if(!available()) SKIP("No GPU");
    const int count=GENERATE(1,7);
    const std::string schedule=GENERATE("mtp_sog_from_scratch","mtp_sog_resume");
    const std::string optimizer=GENERATE("cg","bfgs","sgd","shlbfgs");
    INFO("optimizer="<<optimizer<<", schedule="<<schedule<<", count="<<count);
    const int world_rank=mpi.rank;
    NN2Sample whole(count,false),local(count,true),whole_valid(1,false),local_valid(1,true);
    long root_pid=mpi.rank==0 ? getpid() : 0;MPI_Bcast(&root_pid,1,MPI_LONG,0,mpi.comm);
    const auto directory=std::filesystem::temp_directory_path()/("atoms-agent-nn2-mpi-cg-"+std::to_string(root_pid));
    Training::collective_try([&] {if(mpi.rank==0) std::filesystem::create_directories(directory);});
    auto controls=settings(false);controls.Modify("training_schedule",schedule);
    controls.Modify("sr_stage_iterations","1");controls.Modify("les_stage_iterations","1");controls.Modify("joint_stage_iterations","1");
    controls.Modify("learning_rate","1e-5");controls.Modify("momentum","0");controls.Modify("batch_size","100000");
    controls.Modify("bfgs_maxiteration","1");controls.Modify("save_every","100");
    controls.Modify("bfgs_grad_tol","0");controls.Modify("bfgs_conv_tol","0");controls.Modify("bfgs_step_tol","0");
    auto cpu_settings=controls;cpu_settings.Modify("save_to",(directory/("cpu-"+std::to_string(world_rank)+".mtp")).string());
    const int n=whole.potential.get_NL_coeffs_count();Eigen::VectorXd initial_gradient(n),final_gradient(n);
    double initial_loss=0.,cpu_loss=0.;std::vector<double> initial_parameters,cpu_final;int cpu_accepted=0;
    collective_self_baseline([&] {
        TrainNonlinear cpu(cpu_settings);cpu.init(whole.atoms,whole.neighbors,&whole.potential);
        cpu.init_valid(whole_valid.atoms,whole_valid.neighbors);
        cpu.AccumulateGard(initial_loss,initial_gradient);
        initial_parameters=mpi_coefficients(whole.potential);
        mpi_run_optimizer(cpu,optimizer);cpu.AccumulateGard(cpu_loss,final_gradient);
        cpu_final=mpi_coefficients(whole.potential);
        cpu_accepted=(optimizer=="cg" || optimizer=="bfgs") ? cpu.bfgs_accepted_iter : cpu.bfgs_iter;
    });
    auto gpu_settings=controls;gpu_settings.Modify("device",MLIPCore::GpuDirect::MTPGpuDirect::backend_name());
    if(std::getenv("IMRMLP_TEST_SHARED_GPU")) gpu_settings.Modify("gpu_id","0");
    gpu_settings.Modify("save_to",(directory/"gpu.mtp").string());
    TrainNonlinear gpu(gpu_settings);gpu.init(local.atoms,local.neighbors,&local.potential);
    gpu.init_valid(local_valid.atoms,local_valid.neighbors);
    double gpu_loss=0.;Eigen::VectorXd gpu_gradient(n);gpu.AccumulateGard(gpu_loss,gpu_gradient);
    all_require(std::abs(gpu_loss-initial_loss)<1e-9*std::max(1.,std::abs(initial_loss)));
    all_require((gpu_gradient-initial_gradient).norm()<1e-8*std::max(1.,initial_gradient.norm()));
    mpi_run_optimizer(gpu,optimizer);gpu.AccumulateGard(gpu_loss,gpu_gradient);
    const int gpu_accepted=(optimizer=="cg" || optimizer=="bfgs") ? gpu.bfgs_accepted_iter : gpu.bfgs_iter;
    if(optimizer=="cg") all_require(gpu.cg_termination_type>0);
    all_require(gpu_accepted==cpu_accepted && cpu_accepted>0);
    all_require(std::abs(gpu_loss-cpu_loss)<2e-7*std::max(1.,std::abs(cpu_loss)));
    const auto gpu_final=mpi_coefficients(local.potential);
    const Eigen::Map<const Eigen::VectorXd> cpu_parameters(cpu_final.data(),n),gpu_parameters(gpu_final.data(),n);
    all_require((cpu_parameters-gpu_parameters).norm()<2e-6*std::max(1.,cpu_parameters.norm()));
    if(optimizer=="sgd") {
        const Eigen::Map<const Eigen::VectorXd> start_parameters(initial_parameters.data(),n);
        const double movement=(cpu_parameters-start_parameters).norm();all_require(movement>1e-12);
        all_require((cpu_parameters-gpu_parameters).norm()<1e-6*movement);
    }
    all_require((gpu_gradient-final_gradient).norm()<2e-6*std::max(1.,final_gradient.norm()));
    all_require(gpu.validation_evaluations>0);
    MPI_Barrier(mpi.comm);if(mpi.rank==0) std::filesystem::remove_all(directory);
}
#endif
