// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "runtime/mtp/gpu-direct/gpu_backend.h"

#if defined(IMRMLP_MPI) && defined(IMRMLP_GPU)
#include "apps/apps.h"
#include "fixtures/mtp_linear_efs.h"
#include "io/gpio.h"
#include "potentials/mtp_gpu_adapter.h"
#include "train/collective_error.h"
#include "train/conditional_linear.h"
#include "train/train_mtp.h"
#include "train/training_data.h"
#include "utils/comm.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <unistd.h>

int main(int argc,char **argv) {
    int provided=0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed=20261028;
    session.configData().runOrder=Catch::TestRunOrder::Declared;
    const int result=session.run(argc,argv);
    MPI_Finalize();
    return result;
}

namespace {
namespace fs=std::filesystem;
using Potentials::PotMTP;
void all_require(bool value) {
    int local=value,all=0;
    MPI_Allreduce(&local,&all,1,MPI_INT,MPI_MIN,mpi.comm);
    REQUIRE(all==1);
}
bool available() {
    int devices=0,all=0;
    std::string error;
    int local=MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&devices,&error) && devices>0;
    MPI_Allreduce(&local,&all,1,MPI_INT,MPI_MIN,mpi.comm);
    if(std::getenv("IMRMLP_REQUIRE_GPU")) {INFO(error);REQUIRE(all==1);}
    return all;
}
struct Workspace {
    fs::path path;
    Workspace() {
        std::string directory;
        Training::collective_try([&] {
            if(mpi.rank!=0) return;
            char name[]="/tmp/gpml_gpu_train_mtp_mpi_XXXXXX";
            const char *created=mkdtemp(name);
            if(!created) throw std::runtime_error("Cannot create shared MPI trainmtp directory");
            directory=created;
        });
        int length=directory.size();
        MPI_Bcast(&length,1,MPI_INT,0,mpi.comm);
        directory.resize(length);MPI_Bcast(directory.data(),length,MPI_CHAR,0,mpi.comm);
        path=directory;
    }
    ~Workspace() {
        MPI_Barrier(mpi.comm);
        if(mpi.rank==0) {std::error_code error;fs::remove_all(path,error);}
    }
};
struct Capture {
    std::ostringstream output;
    std::ostream *previous=SetStreamForOutput(&output);
    ~Capture() {SetStreamForOutput(previous);}
};
struct Fixture {
    PotMTP model;
    std::vector<Atoms> whole,local;
    std::vector<Neighborhoods> neighbors;
    explicit Fixture(int count) {
        MTPLinearEFSFixture::load(model,4); model.train_init();model.set_inited();
        for(int k=0;k<count;++k) {
            Atoms a;a.id=k;a.cell_=12.*Eigen::Matrix3d::Identity();
            a.pos_={{2.,2.,2.},{4.1+.04*k,2.2,2.},{2.3,4.2+.03*k,2.4},{5.,4.8,3.1+.02*k}};
            a.chemmap_={0,1,0,1};
            if(k%2) {a.pos_.push_back({3.6,3.3,4.6+.03*k});a.chemmap_.push_back(1);}
            model.getEFS(a);
            a.energy_+=.013*std::cos(.6*k);a.forces_[1][0]+=.011;
            a.stress_(0,1)+=.017;a.stress_(1,0)-=.009;
            a.fitness=.7+.1*k;a.atomic_fitness.assign(a.size(),.8+.03*k);a.atomic_fitness[0]=0.;
            a.global_attribs_d_["fitness"]=a.fitness;
            a.local_attribs_d_["atomic_fitness"]=Eigen::Map<const Eigen::VectorXd>(a.atomic_fitness.data(),a.size());
            if(k==1) a.have_energy=false;
            if(k==2) a.forces_.clear();
            if(k==3) a.have_stress=false;
            whole.push_back(std::move(a));
        }
        for(int i=0;i<model.lin_coeff_size;++i) model.linear_coeffs[i]+=.03*std::cos(i+.2);
        local=whole;Training::distribute_training_data(local,false);
        neighbors=Training::training_neighborhoods(local,model.cutoff);
    }
};
GPSettings settings(bool gpu,bool fast,bool fixed_species=false,bool energy_only=false) {
    GPSettings result;
    for(const auto &[key,value]:std::vector<std::pair<std::string,std::string>>{
        {"energy_weight",".7"},{"force_weight",".1"},{"stress_weight",".002"},
        {"energy_pow","1.3"},{"force_pow",".2"},{"stress_pow",".7"},
        {"force_wgt_d",".3"},{"force_wgt_f",".8"},{"force_wgt_p","1.4"},
        {"weight_L1","0"},{"weight_L2",".003"},{"rescale_mode","fixed"},
        {"do_orthogonalization","false"},{"linear_precondition","diagonal"},
        {"gpu_batch_atoms","10"},{"gpu_batch_structures","2"},
        {"gpu_memory_mib","16"},{"gpu_workspace_mib","1"}})
        result.Modify(key,value);
    result.Modify("lfit_fastalgo",fast?"true":"false");
    result.Modify("lfit_eng_only",energy_only?"true":"false");
    result.Modify("fix_species_coeffs",fixed_species?"true":"false");
    result.Modify("device",gpu?MLIPCore::GpuDirect::compiled_backend():"cpu");
    result.Modify("gpu_id",gpu && std::getenv("IMRMLP_TEST_SHARED_GPU")?"0":"-1");
    return result;
}
void initialize(Trainer_MTP &trainer,Fixture &fixture) {
    trainer.init(fixture.local,fixture.neighbors,&fixture.model);
    trainer.init_valid(fixture.local,fixture.neighbors);
}
Eigen::VectorXd coefficients(const PotMTP &model) {
    Eigen::VectorXd values(model.get_NL_coeffs_count());model.nl_get_coeffs(values.data());return values;
}
bool close(double a,double b,double atol=2e-8,double rtol=1e-9) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=atol+rtol*std::abs(b);
}
bool predictions_close(PotMTP &actual,PotMTP &expected,const std::vector<Atoms> &data) {
    for(const auto &reference:data) {
        auto a=reference,b=reference;actual.getEFS(a);expected.getEFS(b);
        const auto av=Training::ConditionalLinearAdapter::efs_values(a);
        const auto bv=Training::ConditionalLinearAdapter::efs_values(b);
        if(av.size()!=bv.size()) return false;
        for(Eigen::Index i=0;i<av.size();++i) if(!close(av[i],bv[i],2e-7,1e-8)) return false;
    }
    return true;
}
void compare_predictions(PotMTP &actual,PotMTP &expected,const std::vector<Atoms> &data) {
    bool same=false;
    Training::collective_try([&] {same=predictions_close(actual,expected,data);});
    all_require(same);
}
// Independent legacy fitting objective: all nine raw virial entries, fitness
// only on energy, atomic fitness only on forces, and normalization by frames.
// AccumulateLoss has a different species-ridge contract and is not this oracle.
double data_loss(PotMTP &model,const std::vector<Atoms> &whole,bool energy_only) {
    double value=0.;
    for(const auto &reference:whole) {
        auto predicted=reference;model.getEFS(predicted);
        double mean_force=0.;
        if(reference.has_forces()) {
            for(const auto &f:reference.forces_) mean_force+=f.squaredNorm();
            mean_force/=reference.size();
        }
        const double cc=.3/(.3+std::pow(.8*mean_force,1.4));
        if(reference.has_energy()) value+=cc*.7*reference.fitness/std::pow(reference.size(),1.3)*
            std::pow(predicted.energy_-reference.energy_,2);
        if(energy_only) continue;
        if(reference.has_forces())
            for(int i=0;i<reference.size();++i) {
                const double atomic=reference.atomic_fitness.empty()?1.:reference.atomic_fitness[i];
                if(atomic>0.) value+=cc*.1/std::pow(reference.size(),.2)*atomic*
                    (predicted.forces_[i]-reference.forces_[i]).squaredNorm();
            }
        if(reference.has_stresses()) value+=cc*.002/std::pow(reference.size(),.7)*
            (predicted.stress_-reference.stress_).squaredNorm();
    }
    return value/whole.size();
}
double ridge(PotMTP &model,const std::vector<double> &lambda,bool fixed_species) {
    double value=0.;
    for(int i=fixed_species?model.species_count:0;i<model.lin_coeff_size;++i)
        value+=lambda[i]*model.linear_coeffs[i]*model.linear_coeffs[i];
    return value;
}
void check_lambda(Trainer_MTP &gpu,PotMTP &actual,PotMTP &expected) {
    auto lambda=expected.reg_L2_vector;
    MPI_Bcast(lambda.data(),lambda.size(),MPI_DOUBLE,0,mpi.comm);
    const auto &reported=gpu.GetGpuLinearReport().effective_l2;
    bool same=reported.size()==lambda.size() && actual.reg_L2_vector.size()==lambda.size();
    if(same) for(size_t i=0;i<lambda.size();++i)
        same=same && close(reported[i],lambda[i],1e-12,1e-9) &&
            close(actual.reg_L2_vector[i],lambda[i],1e-12,1e-9);
    all_require(same);
}
std::string contents(const fs::path &path) {
    std::ifstream input(path);
    if(!input) throw std::runtime_error("Cannot read MPI trainmtp output: "+path.string());
    return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}
void write_inputs(const Workspace &work,Fixture &fixture) {
    Training::collective_try([&] {
        if(mpi.rank!=0) return;
        fixture.model.Save((work.path/"input.mtp").string());
        GPIO io;
        if(!io.write(fixture.whole,(work.path/"train.bin").string(),"bin"))
            throw std::runtime_error("Cannot write MPI trainmtp input");
        std::ofstream index(work.path/"index.map");index<<"Al\nNi\n";index.close();
        if(!index) throw std::runtime_error("Cannot write MPI trainmtp species map");
    });
}
int run_cli(const Workspace &work,const GPSettings &fit,const std::string &name,
            int missing_rank=-1,int malformed_rank=-1) {
    const auto ini=work.path/(name+".ini"),malformed_ini=work.path/(name+"-malformed.ini");
    Training::collective_try([&] {
        if(mpi.rank!=0) return;
        std::ofstream out(ini);
        for(const auto &[key,value]:fit.settings) out<<"fit:"<<key<<' '<<value<<'\n';
        out.close();if(!out) throw std::runtime_error("Cannot write MPI trainmtp settings");
        if(malformed_rank>=0) {
            std::ofstream invalid(malformed_ini);invalid<<"fit:device\n";invalid.close();
            if(!invalid) throw std::runtime_error("Cannot write malformed MPI trainmtp settings");
        }
    });
    struct OptionsReset {
        OptionsReset() {CMDOption::global_cmdopts.reset();}
        ~OptionsReset() {CMDOption::global_cmdopts.reset();}
    } reset;
    struct CurrentDirectory {
        fs::path previous=fs::current_path();
        explicit CurrentDirectory(const fs::path &next) {fs::current_path(next);}
        ~CurrentDirectory() {fs::current_path(previous);}
    } directory(work.path);
    std::vector<std::string> args{"trainmtp","--linear-fit",
        "--settings="+(mpi.rank==malformed_rank?malformed_ini:ini).string(),
        "--potfile="+(work.path/(mpi.rank==missing_rank?"missing.mtp":"input.mtp")).string(),
        "--trainset="+(work.path/"train.bin").string(),
        "--trainset-format=bin","--validset="+(work.path/"train.bin").string(),
        "--validset-format=bin","--index="+(work.path/"index.map").string()};
    std::vector<char*> argv;for(auto &arg:args) argv.push_back(arg.data());
    return trainmtp(argv.size(),argv.data());
}
void check_saved(const fs::path &path,PotMTP &expected,const std::vector<Atoms> &data) {
    bool same=false;
    Training::collective_try([&] {PotMTP restored(path.string());same=predictions_close(restored,expected,data);});
    all_require(same);
}
}

TEST_CASE("MPI GPU trainmtp preserves both legacy solvers and effective ridge on empty and uneven shards",
    "[gpu][mpi][trainmtp]") {
    if(!available()) SKIP("No GPU on an MPI rank");
    if(mpi.size!=2) SKIP("Requires exactly two MPI ranks");
    const int count=GENERATE(1,7);
    const bool fast=GENERATE(false,true);
    const bool fixed_species=GENERATE(false,true);
    const bool energy_only=GENERATE(false,true);
    CAPTURE(count,fast,fixed_species,energy_only);
    Capture capture;Fixture cpu_case(count),gpu_case(count);
    all_require(gpu_case.local.size()==size_t((count+1-mpi.rank)/2));
    Trainer_MTP cpu(settings(false,fast,fixed_species,energy_only));
    Trainer_MTP gpu(settings(true,fast,fixed_species,energy_only));
    initialize(cpu,cpu_case);initialize(gpu,gpu_case);
    // Exercise adaptive regularization, whose effective values originate on root.
    std::fill(cpu_case.model.reg_L2_vector.begin(),cpu_case.model.reg_L2_vector.end(),1e-30);
    std::fill(gpu_case.model.reg_L2_vector.begin(),gpu_case.model.reg_L2_vector.end(),1e-30);
    const auto original=coefficients(gpu_case.model);
    const double data_before=data_loss(gpu_case.model,gpu_case.whole,energy_only);
    cpu.TrainLinear();gpu.TrainLinear();
    compare_predictions(gpu_case.model,cpu_case.model,gpu_case.whole);
    const auto &report=gpu.GetGpuLinearReport();
    all_require(report.verified && report.backend==MLIPCore::GpuDirect::compiled_backend());
    check_lambda(gpu,gpu_case.model,cpu_case.model);
    all_require((coefficients(gpu_case.model).head(gpu_case.model.radial_coeff_size).array()==
        original.head(gpu_case.model.radial_coeff_size).array()).all());
    if(fixed_species) all_require((coefficients(gpu_case.model).segment(gpu_case.model.radial_coeff_size,
        gpu_case.model.species_count).array()==original.segment(gpu_case.model.radial_coeff_size,
        gpu_case.model.species_count).array()).all());
    const auto fitted=coefficients(gpu_case.model);
    const double data_after=data_loss(gpu_case.model,gpu_case.whole,energy_only);
    const double objective_after=data_after+ridge(gpu_case.model,report.effective_l2,fixed_species);
    gpu_case.model.nl_set_coeffs(original.data());
    const double objective_before=data_before+ridge(gpu_case.model,report.effective_l2,fixed_species);
    gpu_case.model.nl_set_coeffs(fitted.data());
    all_require(close(report.data_before,data_before) && close(report.data_after,data_after));
    all_require(close(report.objective_before,objective_before) && close(report.objective_after,objective_after));
    all_require(report.objective_after<=report.objective_before+1e-8*std::max(1.,report.objective_before));
    all_require((gpu_case.local.empty() && report.gpu_stats.batches==0) ||
        (!gpu_case.local.empty() && report.gpu_stats.batches>0));
    all_require(report.gpu_stats.observation_workspace_bytes<=report.options.gpu_workspace_bytes);
    all_require(report.gpu_stats.tracked_device_bytes<=report.options.gpu_memory_bytes+report.options.gpu_workspace_bytes);
}

TEST_CASE("MPI GPU trainmtp synchronizes local failures and restores physical coefficients and ridge before retry",
    "[gpu][mpi][trainmtp][rollback]") {
    if(!available()) SKIP("No GPU on an MPI rank");
    if(mpi.size!=2) SKIP("Requires exactly two MPI ranks");
    const int count=GENERATE(1,7);
    const bool fast=GENERATE(false,true);
    const int failure=GENERATE(0,1,2,3,4,5);
    CAPTURE(count,fast,failure);
    Capture capture;Fixture cpu_case(count),gpu_case(count);
    auto cpu_settings=settings(false,fast),gpu_settings=settings(true,fast);
    if(failure==2 || failure==4) {cpu_settings.Modify("do_orthogonalization","true");gpu_settings.Modify("do_orthogonalization","true");}
    Trainer_MTP cpu(cpu_settings),gpu(gpu_settings);
    initialize(cpu,cpu_case);initialize(gpu,gpu_case);
    const auto original=coefficients(gpu_case.model);
    const auto l1=gpu_case.model.reg_L1_vector,l2=gpu_case.model.reg_L2_vector;
    const auto data=gpu.training_set_;
    auto validation=gpu_case.local;
    const int bad_rank=(failure==0 || failure==4 || failure==5)?(count==1?0:1):1;
    if(mpi.rank==bad_rank) {
        GPSettings invalid;
        if(failure==0) {
            gpu.training_set_.front().have_energy=true;
            gpu.training_set_.front().energy_=std::numeric_limits<double>::quiet_NaN();
        } else if(failure==1) invalid.Modify("force_weight","nan");
        else if(failure==2) invalid.Modify("gpu_id","999999");
        else if(failure==3) invalid.Modify("gpu_workspace_mib","0");
        else if(failure==4) invalid.Modify("gpu_batch_atoms","1"); // Visitor failure after orthogonalization.
        else validation.front().hvp_directions_=Eigen::MatrixXd::Ones(1,1); // Invalid validation shape.
        gpu.ApplySettings(invalid);
    }
    bool threw=false;
    try {
        if(failure==5) gpu.init_valid(validation,gpu_case.neighbors);
        else gpu.TrainLinear();
    } catch(const std::exception &) {threw=true;}
    all_require(threw);
    all_require((coefficients(gpu_case.model).array()==original.array()).all());
    all_require(gpu_case.model.reg_L1_vector==l1 && gpu_case.model.reg_L2_vector==l2);
    all_require(!gpu.GetGpuLinearReport().verified);
    gpu.training_set_=data;gpu.ApplySettings(gpu_settings);
    gpu.init_valid(gpu_case.local,gpu_case.neighbors);
    cpu.TrainLinear();gpu.TrainLinear();
    all_require(gpu.GetGpuLinearReport().verified);
    compare_predictions(gpu_case.model,cpu_case.model,gpu_case.whole);
    check_lambda(gpu,gpu_case.model,cpu_case.model);
}

TEST_CASE("MPI GPU trainmtp rejects late mixed backends collectively and permits repaired reuse",
    "[gpu][mpi][trainmtp][backend]") {
    if(!available()) SKIP("No GPU on an MPI rank");
    if(mpi.size!=2) SKIP("Requires exactly two MPI ranks");
    const int count=GENERATE(1,7),switched_rank=GENERATE(0,1);
    const bool fast=GENERATE(false,true);
    CAPTURE(count,switched_rank,fast);
    Capture capture;Fixture cpu_case(count),gpu_case(count);
    auto opts=settings(true,fast);
    // Rank-specific resource limits are legal; only the execution backend must agree.
    opts.Modify("gpu_batch_atoms",mpi.rank==0?"10":"6");
    opts.Modify("gpu_batch_structures",mpi.rank==0?"2":"1");
    Trainer_MTP cpu(settings(false,fast)),gpu(opts);
    initialize(cpu,cpu_case);initialize(gpu,gpu_case);
    cpu.TrainLinear();gpu.TrainLinear();
    all_require(gpu.GetGpuLinearReport().verified);
    compare_predictions(gpu_case.model,cpu_case.model,gpu_case.whole);
    const auto original=coefficients(gpu_case.model);
    const auto lambda=gpu_case.model.reg_L2_vector;
    if(mpi.rank==switched_rank) {
        GPSettings change;change.Modify("device","cpu");
        // A CPU-local error must not precede the common backend collective.
        if(switched_rank==0) change.Modify("linear_precondition","invalid");
        gpu.ApplySettings(change);
    }
    bool threw=false;
    try {gpu.TrainLinear();} catch(const std::exception &) {threw=true;}
    all_require(threw);
    all_require((coefficients(gpu_case.model).array()==original.array()).all() && gpu_case.model.reg_L2_vector==lambda);
    all_require(!gpu.GetGpuLinearReport().verified);
    gpu.ApplySettings(opts);cpu.TrainLinear();gpu.TrainLinear();
    all_require(gpu.GetGpuLinearReport().verified);
    compare_predictions(gpu_case.model,cpu_case.model,gpu_case.whole);
}

TEST_CASE("MPI trainmtp CLI publishes and reloads both GPU solvers and recovers from root output failure",
    "[gpu][mpi][trainmtp][cli]") {
    if(!available()) SKIP("No GPU on an MPI rank");
    if(mpi.size!=2) SKIP("Requires exactly two MPI ranks");
    const int count=GENERATE(1,7);
    const bool fast=GENERATE(false,true);
    CAPTURE(count,fast);
    Workspace work;Capture capture;Fixture fixture(count);
    write_inputs(work,fixture);
    const auto published=work.path/"_Trained_Potential_.mtp";
    all_require(run_cli(work,settings(false,fast),"cpu")==0);
    Training::collective_try([&] {if(mpi.rank==0) fs::rename(published,work.path/"cpu.mtp");});
    PotMTP expected((work.path/"cpu.mtp").string());
    for(int missing_rank:{0,1}) {
        bool failed=false;
        try {run_cli(work,settings(true,fast),"missing",missing_rank);}
        catch(const std::exception &) {failed=true;}
        all_require(failed);
        all_require(!fs::exists(published));
        failed=false;
        try {run_cli(work,settings(true,fast),"malformed",-1,missing_rank);}
        catch(const std::exception &) {failed=true;}
        all_require(failed);
        all_require(!fs::exists(published));
    }
    capture.output.str("");capture.output.clear();
    all_require(run_cli(work,settings(true,fast),"gpu")==0);
    check_saved(published,expected,fixture.whole);
    bool valid_report=true;
    Training::collective_try([&] {
        if(mpi.rank!=0) return;
        const std::string text=capture.output.str(),prefix="trainmtp GPU linear report ";
        const auto position=text.find(prefix);
        if(position==std::string::npos) {valid_report=false;return;}
        const auto begin=position+prefix.size(),end=text.find('\n',begin);
        picojson::value report;
        if(!picojson::parse(report,text.substr(begin,end-begin)).empty()) {valid_report=false;return;}
        valid_report=report.get("verified").get<bool>() &&
            report.get("observation_backend").get<std::string>()==MLIPCore::GpuDirect::compiled_backend() &&
            report.get("acceptance_backend").get<std::string>()=="cpu/direct" &&
            report.get("solve_backend").get<std::string>().starts_with("cpu/") &&
            report.get("effective_l2").get<picojson::array>().size()==size_t(expected.lin_coeff_size) &&
            report.get("objective_after").get<double>()<=report.get("objective_before").get<double>()+1e-8;
    });
    all_require(valid_report);
    std::string before;
    Training::collective_try([&] {before=contents(published);});
    Training::collective_try([&] {
        if(mpi.rank!=0) return;
        fs::rename(published,work.path/"gpu.mtp");fs::create_directory(published);
        std::ofstream keep(published/"keep");keep<<"keep existing output directory\n";
    });
    bool threw=false;
    try {run_cli(work,settings(true,fast),"failed");} catch(const std::exception &) {threw=true;}
    all_require(threw);
    bool intact=false;
    Training::collective_try([&] {
        intact=contents(work.path/"gpu.mtp")==before &&
            contents(published/"keep")=="keep existing output directory\n";
    });
    all_require(intact);
    Training::collective_try([&] {if(mpi.rank==0) fs::remove_all(published);});
    all_require(run_cli(work,settings(true,fast),"retry")==0);
    check_saved(published,expected,fixture.whole);
}
#endif
