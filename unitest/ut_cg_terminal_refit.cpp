#include <catch2/catch_all.hpp>
#include "train/train_nonlinear_potential.h"
#include "train/conditional_linear_persistence.h"
#include "train/conditional_linear.h"
#include "potentials/potential_mtp.h"
#include "apps/apps.h"
#include "io/gpio.h"
#include "utils/comm.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

#ifdef IMRMLP_MPI
int main(int argc,char **argv) {
    MPI_Init(&argc,&argv);mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed=20261023;session.configData().runOrder=Catch::TestRunOrder::Declared;
    const int result=session.run(argc,argv);MPI_Finalize();return result;
}
#endif

namespace {
namespace fs=std::filesystem;
using Potentials::PotMTP;
struct Case {
    fs::path directory;
    PotMTP potential{"unitest/test-data/portable-mtp/les.mtp"};
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> neighbors;
    explicit Case(bool missing=false) {
        char path[128]="/tmp/imrmlp_cg_terminal_refit_XXXXXX";
        int ready=1;
        if(mpi.rank==0 && !mkdtemp(path)) ready=0;
        MPI_Bcast(&ready,1,MPI_INT,0,mpi.comm);
        if(!ready) throw std::runtime_error("Cannot create CG/refit fixture");
#ifdef IMRMLP_MPI
        MPI_Bcast(path,128,MPI_CHAR,0,mpi.comm);
#endif
        directory=path;
        potential.train_init();
        for(int k=0;k<6;++k) {
            Atoms a;a.fitness=.7+.1*k;a.cell_=Eigen::Matrix3d::Identity()*12.;
            a.pos_={{2.,2.,2.},{4.1+.04*k,2.2,2.},{2.3,4.2+.03*k,2.4},{5.,4.8,3.1+.02*k}};
            a.chemmap_={0,1,0,1};potential.getEFS(a);
            if(missing && k%2) {a.have_energy=false;a.have_stress=false;}
            // Explicitly exercise an empty local training/validation shard.
            if(mpi.rank==0) {atoms.push_back(a);neighbors.emplace_back(atoms.back(),potential.cutoff);}
        }
        for(int i=0;i<potential.get_lin_coeff_size();++i) potential.linear_coeffs[i]+=.03*std::cos(i+.2);
    }
    ~Case() {MPI_Barrier(mpi.comm);if(mpi.rank==0) {std::error_code e;fs::remove_all(directory,e);}}
    GPSettings settings(bool refit=true) const {
        GPSettings s;
        for(const auto &[k,v]:std::vector<std::pair<std::string,std::string>>{
            {"energy_weight",".7"},{"force_weight",".1"},{"stress_weight",".002"},
            {"penalty_weight",".00001"},{"weight_L1","0"},{"weight_L2",".0002"},
            {"parameter_scaling","none"},{"bfgs_maxiteration","1"},{"bfgs_grad_tol","0"},
            {"bfgs_conv_tol","0"},{"bfgs_step_tol","0"},{"save_every","1"}}) s.Modify(k,v);
        s.Modify("linear_refit",refit ? "sr" : "off");
        s.Modify("linear_refit_report",(directory/"report.json").string());
        s.Modify("save_to",(directory/"fitted.mtp").string());return s;
    }
    Eigen::VectorXd coefficients() const {
        Eigen::VectorXd x(potential.get_NL_coeffs_count());potential.nl_get_coeffs(x.data());return x;
    }
};
struct Capture {
    std::ostringstream stream;
    std::ostream *previous=SetStreamForOutput(&stream);
    ~Capture(){SetStreamForOutput(previous);}
    size_t count(const std::string &word) const {
        auto text=stream.str();size_t n=0,pos=0;
        while((pos=text.find(word,pos))!=std::string::npos) {++n;pos+=word.size();}return n;
    }
};
std::string contents(const fs::path &path) {
    std::ifstream in(path);return {std::istreambuf_iterator<char>(in),{}};
}
}

TEST_CASE("CPU CG terminal refit matches the manual chain with final validation and one solve", "[training][cg-terminal-refit]") {
    const bool missing=GENERATE(false,true);Case combined(missing),manual(missing);
    TrainNonlinear first(combined.settings()),second(manual.settings(false));
    first.init(combined.atoms,combined.neighbors,&combined.potential);
    second.init(manual.atoms,manual.neighbors,&manual.potential);
    first.init_valid(combined.atoms,combined.neighbors);second.init_valid(manual.atoms,manual.neighbors);
    Capture log;
    REQUIRE_NOTHROW(second.TrainCG());
    const auto cg=manual.coefficients();const int code=second.cg_termination_type,steps=second.bfgs_accepted_iter;
    const auto validations=second.validation_evaluations;
    GPSettings enable;enable.Modify("linear_refit","sr");second.ApplySettings(enable);
    REQUIRE_NOTHROW(second.TrainLinearRefit());
    REQUIRE_NOTHROW(first.TrainCG());
    REQUIRE(first.cg_termination_type==code);REQUIRE(first.bfgs_accepted_iter==steps);
    REQUIRE(first.validation_evaluations==validations+1);
    REQUIRE((combined.coefficients()-manual.coefficients()).norm()<2e-10);
    REQUIRE((combined.coefficients().head(combined.potential.radial_coeff_size).array()
        ==cg.head(combined.potential.radial_coeff_size).array()).all());
    REQUIRE_THROWS(first.SaveCGResult((combined.directory/"invalid-cg.mtp").string()));
    REQUIRE_THROWS(second.SaveCGResult((manual.directory/"invalid-cg.mtp").string()));
    REQUIRE_NOTHROW(first.SaveConditionalResult((combined.directory/"published.mtp").string()));
    PotMTP snapshot((combined.directory/"fitted.mtp.cg.snapshot").string());
    Eigen::VectorXd checkpoint(cg.size());snapshot.nl_get_coeffs(checkpoint.data());
    REQUIRE((checkpoint-cg).norm()<2e-12);
    REQUIRE(log.count("Linear refit accepted:")==(mpi.rank==0 ? 2 : 0));
    // Re-evaluated metrics describe the terminal model, not the CG checkpoint.
    const auto validation=Training::evaluate_dataset_metrics(combined.potential,combined.atoms);
    REQUIRE(validation.energy.has_value());
    // ErrorMonitor::report uses MPI_Reduce: only root owns global RMSE.
    if(mpi.rank==0) REQUIRE(first.emonitor_valid.erms==Catch::Approx(*validation.energy).margin(2e-10));
    PotMTP published((combined.directory/"published.mtp").string());
    for(auto a:combined.atoms) {auto b=a;combined.potential.getEFS(a);published.getEFS(b);
        REQUIRE((Training::ConditionalLinearAdapter::efs_values(a)-Training::ConditionalLinearAdapter::efs_values(b)).norm()<2e-10);}
}

TEST_CASE("CG terminal refit preflights settings and frozen affine indices before optimization", "[training][cg-terminal-refit]") {
    const auto [key,value]=GENERATE(Catch::Generators::table<std::string,std::string>({
        {"linear_refit_solver","invalid"},{"linear_refit_rtol","1"},{"linear_refit_atol","-1"},
        {"linear_refit_tolerance","0"},{"weight_L1",".1"},{"trainable_groups","radial"},
        {"training_schedule","mtp_les_resume"},{"device","cuda"}}));
    Case c;auto initial=c.settings();
    if(key=="weight_L1") initial.Modify(key,value);
    TrainNonlinear trainer(initial);trainer.init(c.atoms,c.neighbors,&c.potential);
    const auto original=c.coefficients();GPSettings bad;bad.Modify(key,value);trainer.ApplySettings(bad);
    REQUIRE_THROWS(trainer.TrainCG());REQUIRE(c.coefficients()==original);
    REQUIRE(trainer.cg_termination_type==0);
    REQUIRE_FALSE(fs::exists(c.directory/"fitted.mtp.cg.snapshot"));
    REQUIRE_FALSE(fs::exists(c.directory/"fitted.mtp.pre-refit"));
    REQUIRE_FALSE(fs::exists(c.directory/"report.json"));
}

TEST_CASE("Terminal refit failure restores the accepted CG endpoint and its save token", "[training][cg-terminal-refit]") {
    const bool publication=GENERATE(false,true);Case c;
    auto settings=c.settings();
    if(publication) {if(mpi.rank==0) fs::create_directory(c.directory/"fitted.mtp");}
    else settings.Modify("linear_refit_report",(c.directory/"missing/report.json").string());
    MPI_Barrier(mpi.comm);
    TrainNonlinear trainer(settings);trainer.init(c.atoms,c.neighbors,&c.potential);
    trainer.init_valid(c.atoms,c.neighbors);
    REQUIRE_THROWS(trainer.TrainCG());REQUIRE(trainer.cg_termination_type!=0);
    PotMTP snapshot((c.directory/"fitted.mtp.cg.snapshot").string());
    Eigen::VectorXd expected(c.potential.get_NL_coeffs_count());snapshot.nl_get_coeffs(expected.data());
    REQUIRE((c.coefficients()-expected).norm()<2e-12);
    REQUIRE_NOTHROW(trainer.SaveCGResult((c.directory/"recovered-cg.mtp").string()));
    REQUIRE(trainer.CurrentSettings()["linear_refit"]=="sr");
    REQUIRE(trainer.CurrentSettings()["save_to"]==(c.directory/"fitted.mtp").string());
    if(publication) REQUIRE(fs::is_directory(c.directory/"fitted.mtp"));
    else REQUIRE_FALSE(fs::exists(c.directory/"fitted.mtp"));
}

TEST_CASE("Direct low level CG remains pure and refuses an implicit terminal refit", "[training][cg-terminal-refit]") {
    Case c;TrainNonlinear trainer(c.settings());trainer.init(c.atoms,c.neighbors,&c.potential);
    const auto original=c.coefficients();REQUIRE_THROWS(trainer.TrainNonLinearCG(1));
    REQUIRE(c.coefficients()==original);REQUIRE_FALSE(fs::exists(c.directory/"report.json"));
}

TEST_CASE("CG agrees the original terminal refit mode before entering either rank branch", "[training][cg-terminal-refit][mpi]") {
    if(mpi.size<2) {SUCCEED("Requires at least two MPI ranks");return;}
    const bool off=GENERATE(false,true);Case c;
    TrainNonlinear trainer(c.settings());trainer.init(c.atoms,c.neighbors,&c.potential);
    GPSettings local;local.Modify("linear_refit",mpi.rank==0 ? "sr" : (off ? "off" : "all"));
    trainer.ApplySettings(local);const auto original=c.coefficients();
    REQUIRE_THROWS_WITH(trainer.TrainCG(),Catch::Matchers::ContainsSubstring("refit modes differ between ranks"));
    REQUIRE(c.coefficients()==original);REQUIRE(trainer.cg_termination_type==0);
    REQUIRE_FALSE(fs::exists(c.directory/"fitted.mtp.cg.snapshot"));
}

TEST_CASE("Real CG CLI runs one terminal refit and publishes the final conditional model", "[apps][cg-terminal-refit][cg-terminal-refit-cli]") {
    Case c;const auto input=(c.directory/"input.mtp").string();
    const auto settings=(c.directory/"fit.ini").string();
    if(mpi.rank==0) {
        c.potential.Save(input);GPIO io;REQUIRE(io.write(c.atoms,(c.directory/"train.bin").string(),"bin"));
        std::ofstream(c.directory/"index.map")<<"Al\nNi\n";
        std::ofstream out(settings);out<<"pot:pottype mtp\npot:potfile "<<input<<'\n';
        for(const auto &[key,value]:c.settings().settings)out<<"fit:"<<key<<' '<<value<<'\n';
    }
    MPI_Barrier(mpi.comm);
    std::vector<std::string> args={"trainnlpot","--do-cg","--settings="+settings,
        "--trainset="+(c.directory/"train.bin").string(),"--trainset-format=bin",
        "--index="+(c.directory/"index.map").string(),"--output="+(c.directory/"published.mtp").string()};
    CMDOption::global_cmdopts.reset();std::vector<char*> argv;for(auto &arg:args)argv.push_back(arg.data());
    Capture log;REQUIRE(trainnlpot(argv.size(),argv.data())==0);CMDOption::global_cmdopts.reset();
    REQUIRE(log.count("Linear refit accepted:")==(mpi.rank==0 ? 1 : 0));
    REQUIRE(fs::exists(c.directory/"fitted.mtp.cg.snapshot"));
    REQUIRE(contents(c.directory/"fitted.mtp.pre-refit")==contents(c.directory/"fitted.mtp.cg.snapshot"));
    PotMTP fitted((c.directory/"fitted.mtp").string()),published((c.directory/"published.mtp").string());
    for(auto a:c.atoms) {auto b=a;fitted.getEFS(a);published.getEFS(b);
        REQUIRE((Training::ConditionalLinearAdapter::efs_values(a)-Training::ConditionalLinearAdapter::efs_values(b)).norm()<2e-10);}
}
