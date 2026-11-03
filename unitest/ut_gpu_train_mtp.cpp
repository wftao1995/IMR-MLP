// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "fixtures/mtp_linear_efs.h"
#include "potentials/mtp_gpu_adapter.h"
#include "train/train_mtp.h"
#include "utils/comm.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <tuple>

#ifdef IMRMLP_MPI
int main(int argc,char **argv) {
    int provided=0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed=20261028;
    session.configData().runOrder=Catch::TestRunOrder::Declared;
    const int status=session.run(argc,argv);
    MPI_Finalize();return status;
}
#endif

namespace {
using Potentials::PotMTP;
struct SelfCommunicator {
#ifdef IMRMLP_MPI
    MPI_Comm previous=mpi.comm;
    SelfCommunicator() {mpi.InitComm(MPI_COMM_SELF);}
    ~SelfCommunicator() {mpi.InitComm(previous);}
#endif
};
struct Capture {
    std::ostringstream stream;
    std::ostream *previous=SetStreamForOutput(&stream);
    ~Capture() {SetStreamForOutput(previous);}
};
Eigen::VectorXd coefficients(const PotMTP &model) {
    Eigen::VectorXd result(model.get_NL_coeffs_count());
    model.nl_get_coeffs(result.data());return result;
}
std::unique_ptr<PotMTP> copy_model(PotMTP &source) {
    std::ostringstream saved;source.Save(saved);
    auto model=std::make_unique<PotMTP>();
    std::istringstream input(saved.str());model->Load(input);
    model->train_init();model->set_inited();
    model->reg_L1_vector=source.reg_L1_vector;
    model->reg_L2_vector=source.reg_L2_vector;
    return model;
}
void close(double actual,double expected,double atol=2e-8,double rtol=3e-9) {
    CAPTURE(actual,expected);
    REQUIRE(std::isfinite(actual));REQUIRE(std::isfinite(expected));
    REQUIRE(std::abs(actual-expected)<=atol+rtol*std::abs(expected));
}
void same_predictions(PotMTP &actual,PotMTP &expected,const std::vector<Atoms> &data) {
    for(const auto &reference:data) {
        auto a=reference,b=reference;actual.getEFS(a);expected.getEFS(b);
        CAPTURE(reference.id,reference.size());
        close(a.energy_,b.energy_,2e-7,2e-8);
        double force_error=0.,force_scale=0.;
        for(size_t i=0;i<a.size();++i) {
            force_error=std::max(force_error,(a.forces_[i]-b.forces_[i]).cwiseAbs().maxCoeff());
            force_scale=std::max(force_scale,b.forces_[i].cwiseAbs().maxCoeff());
        }
        CAPTURE(force_error,force_scale);
        REQUIRE(force_error<=2e-7+2e-8*force_scale);
        const double virial_error=(a.stress_-b.stress_).cwiseAbs().maxCoeff();
        CAPTURE(virial_error);
        REQUIRE(virial_error<=2e-7+2e-8*b.stress_.cwiseAbs().maxCoeff());
    }
}
struct Options {
    bool fast=true,energy_only=false,fixed_species=false,orthogonal=false,diagonal=false;
    double energy=2.3,force=.17,virial=.031;
    double energy_power=1.3,force_power=.2,virial_power=.7;
    double norm_d=.3,norm_f=.8,norm_p=1.4;
    GPSettings settings(bool gpu=false) const {
        GPSettings s;
        for(const auto &[name,value]:std::vector<std::pair<std::string,std::string>>{
            {"device",gpu?MLIPCore::GpuDirect::compiled_backend():"cpu"},
            {"gpu_id","0"},{"gpu_batch_atoms","10"},{"gpu_batch_structures","2"},
            {"gpu_memory_mib","256"},{"gpu_workspace_mib","1"},
            {"weight_L1","0"},{"weight_L2",".013"},{"penalty_weight","0"},
            {"weight_rad_w0","0"},{"weight_rad_w1","0"},{"weight_rad_w2","0"},
            {"rescale_mode","fixed"},{"random_init","false"},
            {"force_randomlinear","false"},{"force_randomradial","false"}})s.Modify(name,value);
        for(const auto &[name,value]:std::vector<std::pair<std::string,bool>>{
            {"lfit_fastalgo",fast},{"lfit_eng_only",energy_only},
            {"fix_species_coeffs",fixed_species},{"do_orthogonalization",orthogonal}})
            s.Modify(name,value?"true":"false");
        for(const auto &[name,value]:std::vector<std::pair<std::string,double>>{
            {"energy_weight",energy},{"force_weight",force},{"stress_weight",virial},
            {"energy_pow",energy_power},{"force_pow",force_power},{"stress_pow",virial_power},
            {"force_wgt_d",norm_d},{"force_wgt_f",norm_f},{"force_wgt_p",norm_p}})
            s.Modify(name,fmt::format("{:.17g}",value));
        s.Modify("linear_precondition",diagonal?"diagonal":"none");
        return s;
    }
};
struct Fixture {
    PotMTP model;
    std::vector<Atoms> data;
    std::vector<Neighborhoods> neighbors;
    explicit Fixture(int variant=1,bool missing=true,int count=12,int side=0) {
        MTPLinearEFSFixture::load(model,variant);
        model.train_init();model.set_inited();
        for(int k=0;k<count;++k) {
            Atoms a;
            if(side) a=MTPLinearEnergyFixture::frame(k,model.species_count,side);
            else {
                a.cell_<<10.,0.,0.,.8,10.5,0.,-.3,.6,11.;
                a.pos_={{2.,2.,2.},{4.1+.12*std::sin(k),2.2,2.},
                    {2.3,4.2+.11*std::cos(k),2.4},{5.,4.8,3.1+.07*std::sin(2*k)}};
                a.chemmap_={0,model.species_count-1,k%model.species_count,(k/2)%model.species_count};
                if(k%2) {a.pos_.push_back({3.6,3.3,4.6+.03*k});a.chemmap_.push_back(model.species_count-1);}
            }
            a.id=k;a.fitness=.4+.09*k;
            a.atomic_fitness.assign(a.size(),.6+.04*k);
            if(!side)a.atomic_fitness[0]=0.;
            model.getEFS(a);
            a.energy_+=.013*std::cos(.6*k);a.forces_[1][0]+=.011;
            a.stress_(0,1)+=.017;a.stress_(1,0)-=.009;
            if(missing && k==1)a.have_energy=false;
            if(missing && k==2)a.forces_.clear();
            if(missing && k==3)a.have_stress=false;
            if(missing && k==4)a.fitness=0.;
            // Dedicated trainmtp retains its legacy normalization: this
            // generic-refit fixed-core attribute must not change its weights.
            if(k==5)a.global_attribs_d_["__gpml_fixed_core_force_norm2"]=1e6;
            data.push_back(a);neighbors.emplace_back(data.back(),model.cutoff);
        }
        for(int i=0;i<model.radial_coeff_size;++i)model.radial_coeffs[i]+=.01*std::cos(i+.2);
        for(int i=0;i<model.lin_coeff_size;++i)model.linear_coeffs[i]+=.06*std::sin(i+1.);
    }
};
// Independent direct predictions, retaining Trainer_MTP's legacy normalization
// (including its no-force-label fast/full branch), rather than the G2b weights.
double data_loss(PotMTP &model,const std::vector<Atoms> &data,const Options &o) {
    double result=0.;
    for(const auto &a:data) {
        auto predicted=a;model.getEFS(predicted);
        double cc=1.;
        if(o.norm_f>0. && !(o.fast && !o.energy_only && !a.has_forces())) {
            double force2=0.;
            if(a.has_forces())for(const auto &f:a.forces_)force2+=f.squaredNorm();
            cc=o.norm_d/(o.norm_d+std::pow(o.norm_f*force2/a.size(),o.norm_p));
        }
        if(a.has_energy() && o.energy>0.)
            result+=cc*o.energy*a.fitness/std::pow(a.size(),o.energy_power)*
                std::pow(predicted.energy_-a.energy_,2);
        if(!o.energy_only && a.has_forces() && o.force>0.)
            for(size_t i=0;i<a.size();++i)result+=cc*o.force/std::pow(a.size(),o.force_power)*
                (a.atomic_fitness.empty()?1.:a.atomic_fitness[i])*(predicted.forces_[i]-a.forces_[i]).squaredNorm();
        if(!o.energy_only && a.has_stresses() && o.virial>0.)
            result+=cc*o.virial/std::pow(a.size(),o.virial_power)*(predicted.stress_-a.stress_).squaredNorm();
    }
    return result/data.size();
}
double objective(PotMTP &model,const std::vector<Atoms> &data,const Options &o,
    const std::vector<double> &lambda) {
    double result=data_loss(model,data,o);
    for(int i=o.fixed_species?model.species_count:0;i<model.lin_coeff_size;++i)
        result+=lambda.at(i)*std::pow(model.linear_coeffs[i],2);
    return result;
}
void check_report(const Trainer_MTP &gpu,PotMTP &model,PotMTP &before,
    PotMTP &cpu,const std::vector<Atoms> &data,const Options &o) {
    const auto &r=gpu.GetGpuLinearReport();
    REQUIRE(r.verified);REQUIRE(r.backend==MLIPCore::GpuDirect::compiled_backend());
    REQUIRE(r.algorithm==(o.fast?"fast/eigen_ldlt":"ordinary/alglib"));
    REQUIRE(r.effective_l2.size()==size_t(model.lin_coeff_size));
    for(int i=0;i<model.lin_coeff_size;++i) {
        close(r.effective_l2[i],cpu.reg_L2_vector[i],1e-12,2e-8);
        close(r.effective_l2[i],model.reg_L2_vector[i],1e-12,2e-8);
    }
    close(r.data_before,data_loss(before,data,o));
    close(r.data_after,data_loss(model,data,o));
    const bool large=std::any_of(data.begin(),data.end(),[](const Atoms &a){return a.size()>=1000;});
    for(const auto &[predicted,direct,tolerance]:std::vector<std::tuple<double,double,double>>{
        {r.predicted_data_before,r.data_before,r.quadratic_tolerance_before},
        {r.predicted_data_after,r.data_after,r.quadratic_tolerance_after}}) {
        REQUIRE(std::isfinite(predicted));REQUIRE(std::isfinite(tolerance));REQUIRE(tolerance>=0.);
        // Large extensive targets can cancel in x^T G x - 2 x^T b + y^T y.
        // The physical CPU comparisons below retain their independent tight gates.
        if(large)REQUIRE(std::abs(predicted-direct)<=tolerance);
        else close(predicted,direct);
    }
    close(r.objective_before,objective(before,data,o,r.effective_l2));
    close(r.objective_after,objective(model,data,o,r.effective_l2));
    close(r.objective_after,objective(cpu,data,o,cpu.reg_L2_vector));
    REQUIRE(r.objective_after<=r.objective_before+1e-10*std::max(1.,std::abs(r.objective_before)));
    REQUIRE(r.gpu_stats.batches>0);REQUIRE(r.gpu_stats.observation_blocks>0);
    REQUIRE(r.gpu_stats.download_bytes>0);
    REQUIRE(r.gpu_stats.observation_workspace_bytes<=r.options.gpu_workspace_bytes);
    REQUIRE(r.gpu_stats.tracked_device_bytes<=r.options.gpu_memory_bytes+r.options.gpu_workspace_bytes);
    for(double t:{r.orthogonalization_seconds,r.assembly_seconds,r.reduction_seconds,
        r.solve_seconds,r.verification_seconds,r.total_seconds}) {
        REQUIRE(std::isfinite(t));REQUIRE(t>=0.);
    }
    REQUIRE(r.assembly_seconds>0.);REQUIRE(r.verification_seconds>0.);
}
}

TEST_CASE("Dedicated MTP linear fitting defaults to CPU and rejects unavailable backends",
    "[training][trainmtp][gpu-options]") {
    SelfCommunicator self;Capture capture;Fixture fixture;
    Options options;
    auto cpu_settings=options.settings();
    // GPU-only limits do not change the established CPU path.
    cpu_settings.Modify("gpu_workspace_mib","0");
    Trainer_MTP cpu(cpu_settings);
    REQUIRE_FALSE(cpu.UsesGpuLinear());REQUIRE_NOTHROW(cpu.ValidateGpuLinearSettings());
    cpu.init(fixture.data,fixture.neighbors,&fixture.model);
    cpu.TrainLinear();
    REQUIRE(coefficients(fixture.model).allFinite());REQUIRE_FALSE(cpu.GetGpuLinearReport().verified);
    auto invalid=options.settings();invalid.Modify("device","gpu");
    Trainer_MTP unknown(invalid);REQUIRE_THROWS(unknown.ValidateGpuLinearSettings());
#ifndef IMRMLP_GPU
    invalid.Modify("device","cuda");Trainer_MTP absent(invalid);
    REQUIRE_THROWS(absent.ValidateGpuLinearSettings());
    const auto original=coefficients(fixture.model);
    REQUIRE_THROWS(absent.init(fixture.data,fixture.neighbors,&fixture.model));
    REQUIRE((coefficients(fixture.model).array()==original.array()).all());
#endif
}

#ifdef IMRMLP_GPU
#define REQUIRE_GPU() do { \
    int count=0;std::string error; \
    if(!MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count,&error) || count==0) { \
        if(std::getenv("IMRMLP_REQUIRE_GPU"))FAIL("Required GPU unavailable: "<<error); \
        SKIP("No GPU device: "<<error); \
    } \
} while(false)

TEST_CASE("Dedicated GPU MTP fits preserve both CPU solvers and weighted objectives",
    "[gpu][training][trainmtp][linear]") {
    SelfCommunicator self;REQUIRE_GPU();Capture capture;
    const bool fast=GENERATE(false,true);
    const int scenario=GENERATE(0,1,2,3,4);
    CAPTURE(fast,scenario);
    Options options;options.fast=fast;
    options.fixed_species=scenario==1;options.diagonal=scenario==1 || scenario==3;
    options.energy_only=scenario==2;options.orthogonal=scenario==3;
    if(scenario==1 || scenario==2)options.norm_p=0.;
    if(scenario==4)options.energy=options.force=0.;
    Fixture sample(scenario==3?4:scenario==4?0:scenario+1,scenario!=3);
    auto cpu_model=copy_model(sample.model),gpu_model=copy_model(sample.model);
    auto cpu_settings=options.settings(),gpu_settings=options.settings(true);
    // Also exercise the legacy ALGLIB SPD branch on the virial-only fixture.
    if(scenario==4) {cpu_settings.Modify("safe_fit","false");gpu_settings.Modify("safe_fit","false");}
    Trainer_MTP cpu(cpu_settings),gpu(gpu_settings);
    cpu.init(sample.data,sample.neighbors,cpu_model.get());
    gpu.init(sample.data,sample.neighbors,gpu_model.get());
    REQUIRE(gpu.UsesGpuLinear());
    const auto original=coefficients(*gpu_model);
    auto before=copy_model(*gpu_model);
    if(options.orthogonal)before->Orthogonalize();
    cpu.TrainLinear();gpu.TrainLinear();
    same_predictions(*gpu_model,*cpu_model,sample.data);
    check_report(gpu,*gpu_model,*before,*cpu_model,sample.data,options);
    if(!options.orthogonal)
        REQUIRE((coefficients(*gpu_model).head(gpu_model->radial_coeff_size).array()==
            original.head(gpu_model->radial_coeff_size).array()).all());
    else close((coefficients(*gpu_model).head(gpu_model->radial_coeff_size)-
        coefficients(*cpu_model).head(cpu_model->radial_coeff_size)).norm(),0.,1e-12);
    if(options.fixed_species)
        for(int i=0;i<gpu_model->species_count;++i)
            REQUIRE(gpu_model->species_coeffs[i]==original[gpu_model->radial_coeff_size+i]);
    if(options.energy_only)
        REQUIRE(gpu.GetGpuLinearReport().gpu_stats.observation_blocks==gpu.GetGpuLinearReport().gpu_stats.batches);
}

TEST_CASE("Dedicated GPU fits refresh radial observations and survive repeated native reloads",
    "[gpu][training][trainmtp][reload]") {
    SelfCommunicator self;REQUIRE_GPU();Capture capture;
    const bool fast=GENERATE(false,true);
    Fixture sample(2);Options options;options.fast=fast;options.diagonal=true;
    auto cpu_model=copy_model(sample.model),gpu_model=copy_model(sample.model);
    Trainer_MTP cpu(options.settings()),gpu(options.settings(true));
    cpu.init(sample.data,sample.neighbors,cpu_model.get());gpu.init(sample.data,sample.neighbors,gpu_model.get());
    for(int round=0;round<3;++round) {
        CAPTURE(round,fast);
        if(round==2)for(int i=0;i<gpu_model->radial_coeff_size;++i) {
            const double change=.03*std::cos(i+.4);
            cpu_model->radial_coeffs[i]+=change;gpu_model->radial_coeffs[i]+=change;
        }
        auto before=copy_model(*gpu_model);
        cpu.TrainLinear();gpu.TrainLinear();
        same_predictions(*gpu_model,*cpu_model,sample.data);
        check_report(gpu,*gpu_model,*before,*cpu_model,sample.data,options);
        auto restored=copy_model(*gpu_model);same_predictions(*restored,*gpu_model,sample.data);
    }
}

TEST_CASE("Dedicated GPU observations obey a one MiB force workspace with both solvers",
    "[gpu][training][trainmtp][workspace]") {
    SelfCommunicator self;REQUIRE_GPU();Capture capture;
    const bool fast=GENERATE(false,true);
    Fixture sample(4,false,2,10);Options options;options.fast=fast;options.diagonal=true;
    auto cpu_model=copy_model(sample.model),gpu_model=copy_model(sample.model);
    auto gpu_settings=options.settings(true);
    gpu_settings.Modify("gpu_batch_atoms","1000");gpu_settings.Modify("gpu_batch_structures","1");
    Trainer_MTP cpu(options.settings()),gpu(gpu_settings);
    cpu.init(sample.data,sample.neighbors,cpu_model.get());gpu.init(sample.data,sample.neighbors,gpu_model.get());
    auto before=copy_model(*gpu_model);
    cpu.TrainLinear();gpu.TrainLinear();
    same_predictions(*gpu_model,*cpu_model,sample.data);
    check_report(gpu,*gpu_model,*before,*cpu_model,sample.data,options);
    const auto &r=gpu.GetGpuLinearReport();
    REQUIRE(gpu_model->lin_coeff_size==31);
    REQUIRE(r.options.gpu_workspace_bytes==(size_t(1)<<20));
    REQUIRE(r.gpu_stats.batches==2);REQUIRE(r.gpu_stats.observation_blocks>3*r.gpu_stats.batches);
    REQUIRE(r.gpu_stats.observation_workspace_bytes<=(size_t(1)<<20));
    REQUIRE(r.gpu_stats.observation_workspace_bytes>0);
    if(fast)REQUIRE(r.host_design_bytes>0);
    else REQUIRE(r.host_design_bytes==0);
}

TEST_CASE("Dedicated GPU direct solver failures restore radial coefficients and effective ridge",
    "[gpu][training][trainmtp][rollback]") {
    SelfCommunicator self;REQUIRE_GPU();Capture capture;
    const bool fast=GENERATE(false,true);
    Options options;options.fast=fast;options.orthogonal=true;options.diagonal=true;
    Fixture sample(4,false);
    auto cpu_model=copy_model(sample.model),gpu_model=copy_model(sample.model);
    Trainer_MTP cpu(options.settings()),gpu(options.settings(true));
    cpu.init(sample.data,sample.neighbors,cpu_model.get());gpu.init(sample.data,sample.neighbors,gpu_model.get());
    const auto original=coefficients(*gpu_model);
    const auto original_l1=gpu_model->reg_L1_vector,original_l2=gpu_model->reg_L2_vector;
    GPSettings invalid;invalid.Modify("gpu_batch_atoms","1");gpu.ApplySettings(invalid);
    if(fast)REQUIRE_THROWS(gpu.TrainLinearFast());else REQUIRE_THROWS(gpu.TrainLinearAlg());
    REQUIRE_FALSE(gpu.GetGpuLinearReport().verified);
    REQUIRE((coefficients(*gpu_model).array()==original.array()).all());
    REQUIRE(gpu_model->reg_L1_vector==original_l1);REQUIRE(gpu_model->reg_L2_vector==original_l2);
    GPSettings repair;repair.Modify("gpu_batch_atoms","10");gpu.ApplySettings(repair);
    auto before=copy_model(*gpu_model);before->Orthogonalize();
    cpu.TrainLinear();gpu.TrainLinear();
    same_predictions(*gpu_model,*cpu_model,sample.data);
    check_report(gpu,*gpu_model,*before,*cpu_model,sample.data,options);
    const auto accepted=coefficients(*gpu_model);
    const auto accepted_l2=gpu_model->reg_L2_vector;
    invalid.Modify("gpu_batch_atoms","10");invalid.Modify("gpu_id","2147483647");gpu.ApplySettings(invalid);
    REQUIRE_THROWS(gpu.TrainLinear());
    REQUIRE_FALSE(gpu.GetGpuLinearReport().verified);
    REQUIRE((coefficients(*gpu_model).array()==accepted.array()).all());
    REQUIRE(gpu_model->reg_L2_vector==accepted_l2);
    repair.Modify("gpu_id","0");gpu.ApplySettings(repair);
    before=copy_model(*gpu_model);before->Orthogonalize();
    cpu.TrainLinear();gpu.TrainLinear();
    same_predictions(*gpu_model,*cpu_model,sample.data);
    check_report(gpu,*gpu_model,*before,*cpu_model,sample.data,options);
    const auto retry_coefficients=coefficients(*gpu_model);
    const auto retry_l2=gpu_model->reg_L2_vector;
    for(const auto &[name,value]:std::vector<std::pair<std::string,std::string>>{
        {"force_weight",".34"},{"fix_species_coeffs","true"},
        {"lfit_fastalgo",fast?"false":"true"}}) {
        CAPTURE(name,value);
        GPSettings changed;changed.Modify(name,value);gpu.ApplySettings(changed);
        REQUIRE_THROWS_WITH(gpu.TrainLinear(),Catch::Matchers::ContainsSubstring("reinitialize"));
        REQUIRE_FALSE(gpu.GetGpuLinearReport().verified);
        REQUIRE((coefficients(*gpu_model).array()==retry_coefficients.array()).all());
        REQUIRE(gpu_model->reg_L2_vector==retry_l2);
        gpu.ApplySettings(options.settings(true));
    }
    before=copy_model(*gpu_model);before->Orthogonalize();
    cpu.TrainLinear();gpu.TrainLinear();
    same_predictions(*gpu_model,*cpu_model,sample.data);
    check_report(gpu,*gpu_model,*before,*cpu_model,sample.data,options);
}

TEST_CASE("Dedicated GPU fitting rejects unsupported models and observations before mutation",
    "[gpu][training][trainmtp][preflight]") {
    SelfCommunicator self;REQUIRE_GPU();Capture capture;
    Options options;
    for(const auto &[name,value]:std::vector<std::pair<std::string,std::string>>{
        {"hvp_weight","-.1"},{"hessian_weight","-.1"},{"radial_reg_type","1"},
        {"weight_L1","-.1"},{"gpu_workspace_mib","0"},{"gpu_memory_mib","0"},
        {"gpu_batch_atoms","0"},{"gpu_batch_structures","0"},{"do_ladder","true"}}) {
        CAPTURE(name,value);Fixture sample;
        const auto original=coefficients(sample.model);
        auto invalid=options.settings(true);invalid.Modify(name,value);
        Trainer_MTP gpu(invalid);
        REQUIRE_THROWS(gpu.init(sample.data,sample.neighbors,&sample.model));
        REQUIRE((coefficients(sample.model).array()==original.array()).all());
        REQUIRE_FALSE(gpu.GetGpuLinearReport().verified);
    }
    struct Derived : PotMTP {};
    Derived derived;MTPLinearEFSFixture::load(derived,1);derived.train_init();derived.set_inited();
    Fixture sample;const auto original=coefficients(derived);
    Trainer_MTP gpu(options.settings(true));
    REQUIRE_THROWS(gpu.init(sample.data,sample.neighbors,&derived));
    REQUIRE((coefficients(derived).array()==original.array()).all());
    // A settings default must not silently convert an unsupported loaded model.
    sample.model.radial_mode=1;
    const auto radial_original=coefficients(sample.model);
    Trainer_MTP radial(options.settings(true));
    REQUIRE_THROWS(radial.init(sample.data,sample.neighbors,&sample.model));
    REQUIRE(sample.model.radial_mode==1);
    REQUIRE((coefficients(sample.model).array()==radial_original.array()).all());
}
#undef REQUIRE_GPU
#endif
