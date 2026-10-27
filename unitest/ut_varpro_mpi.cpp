// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#ifdef IMRMLP_MPI
#include "train/varpro.h"
#include "train/train_nonlinear_potential.h"
#include "potentials/potential_mtp.h"
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed=20260929;
    session.configData().runOrder=Catch::TestRunOrder::Declared;
    const int result=session.run(argc,argv);
    MPI_Finalize();return result;
}
namespace {
using Potentials::PotMTP;
struct Sample {
    PotMTP p{"unitest/test-data/portable-mtp/les.mtp"};
    std::vector<Atoms> data;std::vector<Neighborhoods> nbhs;
    Sample(int count,bool distributed) {
        p.train_init();p.set_inited();
        for(int k=0;k<count;++k) {
            Atoms a;a.id=k;a.cell_=12.*Eigen::Matrix3d::Identity();
            a.pos_={{2.,2.,2.},{4.1+.1*std::sin(k),2.,2.},{2.,4.2+.1*std::cos(k),2.4},{5.,4.8,3.}};
            a.chemmap_={0,1,0,1};a.fitness=.6+.1*k;a.atomic_fitness={0.,.5,2.,3.};
            p.getEFS(a);a.energy_+=.002*std::cos(k);
            if(!distributed || k%mpi.size==mpi.rank) {data.push_back(a);nbhs.emplace_back(a,p.cutoff);}
        }
        for(int i=0;i<p.radial_coeff_size;++i) p.radial_coeffs[i]+=.01*std::cos(i);
    }
};
GPSettings settings(const std::string& prefix="") {
    GPSettings s;s.Modify("force_weight",".05");s.Modify("stress_weight",".001");
    s.Modify("weight_L2",".0001");s.Modify("penalty_weight",".00001");
    s.Modify("bfgs_grad_tol","0");s.Modify("bfgs_conv_tol","0");s.Modify("bfgs_step_tol","0");
    s.Modify("bfgs_maxiteration","2");s.Modify("save_every","1");
    s.Modify("save_to",prefix+"model.mtp");s.Modify("varpro_report",prefix+"report.jsonl");return s;
}
Eigen::VectorXd coeffs(PotMTP& p) {
    Eigen::VectorXd c(p.get_NL_coeffs_count());p.nl_get_coeffs(c.data());return c;
}
VarProEvaluation evaluate(Sample& sample,bool inject=false) {
    TrainNonlinear trainer(settings());trainer.init(sample.data,sample.nbhs,&sample.p);
    EFSFitWeights w;w.force=.05;w.stress=.001;
    MTPProfiledObjective objective(sample.p,sample.data,sample.nbhs,w,1e-5,{},
        [&](double& f,Eigen::VectorXd& g){
            trainer.AccumulateGard(f,g);
            if(inject && mpi.rank==mpi.size-1) throw std::runtime_error("injected worker failure");
        });
    return objective.evaluate(coeffs(sample.p).head(sample.p.radial_coeff_size));
}
struct Output {
    std::string dir;
    Output() {
        static int id=0;dir="vp-mpi-test-"+std::to_string(++id)+"/";
        if(mpi.rank==0) std::filesystem::create_directory(dir);
        MPI_Barrier(mpi.comm);
    }
    ~Output() {MPI_Barrier(mpi.comm);if(mpi.rank==0) std::filesystem::remove_all(dir);}
    std::string report() {
        std::ifstream in(dir+"report.jsonl");return {std::istreambuf_iterator<char>(in),{}};
    }
};
double field(const std::string& text,const std::string& name) {
    std::smatch m;
    REQUIRE(std::regex_search(text,m,std::regex("\\\""+name+"\\\":([-+0-9.eE]+)")));
    return std::stod(m[1]);
}
}

TEST_CASE("Distributed VP matches self-communicator reference including empty training ranks", "[varpro][mpi]") {
    const int count=GENERATE(1,7);
    Sample whole(count,false);
    mpi.InitComm(MPI_COMM_SELF);
    const auto reference=evaluate(whole);
    mpi.InitComm(MPI_COMM_WORLD);
    Sample local(count,true);
    const auto before=coeffs(local.p);
    const auto result=evaluate(local);
    REQUIRE(std::abs(result.objective-reference.objective)<1e-11);
    REQUIRE((result.gradient-reference.gradient).norm()<1e-9);
    REQUIRE((result.coefficients-reference.coefficients).norm()<1e-8);
    REQUIRE((coeffs(local.p).array()==before.array()).all());
    // A failure on just the last rank must throw on all ranks and restore all parameters.
    REQUIRE_THROWS(evaluate(local,true));
    REQUIRE((coeffs(local.p).array()==before.array()).all());
}

TEST_CASE("MPI VP validation selection handles root with no validation labels", "[varpro][mpi]") {
    const bool scaled=GENERATE(false,true);
    Sample local(7,true),validation(1,false);Output output;
    if(mpi.rank!=mpi.size-1) {validation.data.clear();validation.nbhs.clear();}
    auto s=settings(output.dir);
    if(scaled) {
        if(mpi.rank==0) {
            std::ofstream file(output.dir+"scales.txt");
            for(int i=0;i<local.p.radial_coeff_size;++i) file << std::pow(4.,std::sin(i+1)) << '\n';
        }
        MPI_Barrier(mpi.comm);
        // Only root reads scales; workers receive the validated vector.
        s.Modify("varpro_radial_scale_file",output.dir+(mpi.rank==0 ? "scales.txt" : "not-visible-on-worker"));
    }
    TrainNonlinear trainer(s);
    trainer.init(local.data,local.nbhs,&local.p);trainer.init_valid(validation.data,validation.nbhs);
    trainer.TrainVarPro();MPI_Barrier(mpi.comm);
    REQUIRE(trainer.bfgs_accepted_iter==2);
    const auto report=output.report();
    REQUIRE(field(report,"validation_configurations")==1);
    REQUIRE(field(report,"training_configurations")==7);
    REQUIRE(field(report,"mpi_processes")==mpi.size);
    REQUIRE(report.find("\"event\":\"failed\"")==std::string::npos);
    const auto pos=report.find("\"event\":\"final\"");REQUIRE(pos!=std::string::npos);
    const auto valid_pos=report.find("\"validation\":{",pos);REQUIRE(valid_pos!=std::string::npos);
    Sample full_validation(1,false);auto predicted=full_validation.data[0];local.p.getEFS(predicted);
    double f2=0.;for(int i=0;i<predicted.size();++i)
        f2+=(predicted.forces_[i]-full_validation.data[0].forces_[i]).squaredNorm();
    REQUIRE(field(report.substr(valid_pos),"force_eV_A")==Catch::Approx(std::sqrt(f2/(3*predicted.size()))).margin(1e-10));
    PotMTP saved(output.dir+"model.mtp");
    REQUIRE((coeffs(saved)-coeffs(local.p)).norm()<1e-10);
    REQUIRE(std::filesystem::exists(output.dir+"model.mtp.best-validation.mtp"));
    double min_data=1e100;
    std::istringstream lines(report);std::string line;
    while(std::getline(lines,line)) if(line.find("\"loss\":")!=std::string::npos)
        min_data=std::min(min_data,field(line,"data"));
    PotMTP data_best(output.dir+"model.mtp.best-data.mtp");
    s.Modify("penalty_weight","0");s.Modify("weight_L2","0");
    TrainNonlinear check(s);check.init(local.data,local.nbhs,&data_best);double loss;check.AccumulateLoss(loss);
    REQUIRE(loss==Catch::Approx(min_data).margin(1e-10));
}

TEST_CASE("MPI VP propagates rank-local input and root output failures without hanging", "[varpro][mpi]") {
    const int failure=GENERATE(0,1,2,3,4,5);
    Sample local(7,true);Output output;auto s=settings(output.dir);
    if(failure==0 && mpi.rank==mpi.size-1) local.data[0].atomic_fitness[1]=-1.;
    if(failure==1) s.Modify("varpro_report",output.dir+"missing/report.jsonl");
    if(failure==2 && mpi.rank==0)
        std::filesystem::create_directory(output.dir+"model.mtp.iter-000001.mtp");
    if(failure==3) {
        // A local 0/0 force normalization must be rejected collectively by refit.
        s.Modify("force_wgt_f","1e300");s.Modify("force_wgt_d","0");
        if(mpi.rank==mpi.size-1) for(auto& a:local.data) for(auto& f:a.forces_) f.setZero();
    }
    if(failure==4) s.Modify("varpro_radial_scale_file",output.dir+"missing-scales");
    // A missing parent passes fresh-path validation, then makes native model
    // staging fail on root during the initial accepted projection's save.
    if(failure==5) s.Modify("save_to",output.dir+"missing/model.mtp");
    MPI_Barrier(mpi.comm);
    Eigen::VectorXd projected;
    if(failure==5) projected=evaluate(local).coefficients;
    TrainNonlinear trainer(s);trainer.init(local.data,local.nbhs,&local.p);
    const auto original=coeffs(local.p);
    REQUIRE_THROWS(trainer.TrainVarPro());
    if(failure==5) {
        REQUIRE(output.report().find("\"event\":\"failed\"")!=std::string::npos);
        REQUIRE((projected-coeffs(local.p)).norm()<1e-10);
        REQUIRE_FALSE(std::filesystem::exists(output.dir+"missing/model.mtp"));
    }
    else if(failure!=2) REQUIRE((coeffs(local.p).array()==original.array()).all());
    else {
        REQUIRE(output.report().find("\"event\":\"failed\"")!=std::string::npos);
        PotMTP saved(output.dir+"model.mtp");REQUIRE((coeffs(saved)-coeffs(local.p)).norm()<1e-10);
    }
}

TEST_CASE("MPI VP takes collective refinement branches and rolls back exhausted probes", "[varpro][mpi]") {
    const bool persistent=GENERATE(false,true);
    Sample local(1,true);TrainNonlinear trainer(settings());trainer.init(local.data,local.nbhs,&local.p);
    EFSFitWeights w;w.force=.05;w.stress=.001;
    const auto before=coeffs(local.p);int calls=0;
    MTPProfiledObjective objective(local.p,local.data,local.nbhs,w,1e-5,{},
        [&](double& f,Eigen::VectorXd& g){
            ++calls;trainer.AccumulateGard(f,g);
            if(mpi.rank==mpi.size-1 && (persistent || calls==1)) g[g.size()-1]=1.;
        },1e-7,2);
    if(persistent) REQUIRE_THROWS(objective.evaluate(before.head(local.p.radial_coeff_size)));
    else {
        const auto result=objective.evaluate(before.head(local.p.radial_coeff_size));
        REQUIRE(result.linear_gradient_history.size()==2);
        REQUIRE(result.linear_gradient_history.front()==1.);
        REQUIRE(result.linear_gradient_inf<=1e-7);
    }
    REQUIRE(calls==(persistent ? 3 : 2));
    REQUIRE((coeffs(local.p).array()==before.array()).all());
}
#endif
