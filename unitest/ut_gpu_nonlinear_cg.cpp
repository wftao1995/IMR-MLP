#include <catch2/catch_all.hpp>
#include "runtime/mtp/gpu-direct/gpu_backend.h"
#if defined(IMRMLP_GPU) && !defined(IMRMLP_MPI)
#include "train/train_nonlinear_potential.h"
#include "unitest/les_curvature_fixture.h"
#include "runtime/mtp/gpu-direct/mtp_gpu_direct.h"
#include <filesystem>
#include <unistd.h>

using namespace Potentials;
namespace {
bool cg_gpu_available() {
    int count=0;std::string error;
    const bool available=MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count,&error) && count>0;
    if(std::getenv("IMRMLP_REQUIRE_GPU")) {INFO(error);REQUIRE(available);}
    return available;
}
struct CGFiles {
    std::filesystem::path directory=std::filesystem::temp_directory_path()/
        ("atoms-agent-gpu-cg-"+std::to_string(getpid()));
    CGFiles() {std::filesystem::create_directories(directory);}
    ~CGFiles() {std::error_code error;std::filesystem::remove_all(directory,error);}
};
GPSettings cg_gpu_settings(const CGFiles& files,bool gpu) {
    GPSettings s;s.Modify("device",gpu ? MLIPCore::GpuDirect::MTPGpuDirect::backend_name() : "cpu");
    s.Modify("energy_weight","1");s.Modify("force_weight",".2");s.Modify("stress_weight",".1");
    s.Modify("penalty_weight","0");s.Modify("weight_L1","0");s.Modify("weight_L2","0");
    s.Modify("bfgs_maxiteration","2");s.Modify("bfgs_grad_tol","0");s.Modify("bfgs_step_tol","0");s.Modify("bfgs_conv_tol","0");
    s.Modify("save_every","100");s.Modify("gpu_batch_structures","1");
    s.Modify("save_to",(files.directory/"result.mtp").string());return s;
}
void load_cg_model(AnyPotential& model,const std::string& path) {model.Load(path);}
template<class Model> void setup_cg_model(Model& model) {
    LESCurvature::initialize(model,true);
    REQUIRE(static_cast<const PotMTP&>(model).inited());
    REQUIRE(static_cast<AnyPotential&>(model).inited());
    REQUIRE(Eigen::Map<const Eigen::VectorXd>(model.regression_coeffs,model.reg_coeff_size).allFinite());
    REQUIRE(model.model.coeffs==model.model_coeffs);
}
template<> void setup_cg_model(PotMTP& model) {load_cg_model(model,"unitest/test-data/portable-mtp/les.mtp");model.train_init();}
template<> void setup_cg_model(PotMTP_LES& model) {load_cg_model(model,"unitest/test-data/portable-mtp/les.mtp");LESCurvature::initialize(model,true);}
void run_staged_optimizer(TrainNonlinear& trainer,const std::string& optimizer) {
    if(optimizer=="cg") trainer.TrainCG();
    else if(optimizer=="bfgs") trainer.TrainBFGS();
    else if(optimizer=="sgd") trainer.TrainSGD();
    else trainer.TrainShLBFGS();
}
std::vector<double> cg_coefficients(const AnyPotential& model) {
    std::vector<double> values(model.get_NL_coeffs_count());model.nl_get_coeffs(values.data());return values;
}
}

TEMPLATE_TEST_CASE("GPU CG uses the existing adapter objective and accepted coefficients",
    "[gpu][training][cg]",PotMTP,PotMTP_LES,PotMTP_M<m_model_mtpq2>,
    PotMTP_M<m_model_mtpfs>,PotMTP_M<m_model_nn1>,PotMTP_M<m_model_nn2>) {
    if(!cg_gpu_available()) SKIP("No GPU");
    const bool device_loss=GENERATE(false,true);
    CGFiles files;TestType model;setup_cg_model(model);model.set_inited();
    auto target=LESCurvature::geometry();model.getEFS(target);target.energy_+=.03;
    target.fitness=.7;target.atomic_fitness={.8,1.1,.9,1.2};
    std::vector<Atoms> atoms{target};std::vector<Neighborhoods> nbhs{Neighborhoods(target,model.cutoff)};
    model.radial_coeffs[0]+=.001;
    auto cpu_settings=cg_gpu_settings(files,false);TrainNonlinear cpu(cpu_settings);cpu.init(atoms,nbhs,&model);
    const int n=model.get_NL_coeffs_count();Eigen::VectorXd gc(n),gg(n);double lc=0.,lg=0.;cpu.AccumulateGard(lc,gc);
    auto settings=cg_gpu_settings(files,true);settings.Modify("gpu_device_loss",device_loss ? "true" : "false");
    TrainNonlinear gpu(settings);gpu.init(atoms,nbhs,&model);gpu.init_valid(atoms,nbhs);gpu.AccumulateGard(lg,gg);
    REQUIRE(lg==Catch::Approx(lc).margin(1e-9*std::max(1.,std::abs(lc))));
    REQUIRE((gc-gg).norm()<1e-8*std::max(1.,gc.norm()));
    const auto original=cg_coefficients(model);Eigen::VectorXd direction(n);
    for(int i=0;i<n;++i) direction[i]=std::sin(.71*(i+1.));direction.normalize();
    for(double step:{1e-5,5e-6}) {
        auto shifted=original;for(int i=0;i<n;++i) shifted[i]+=step*direction[i];model.nl_set_coeffs(shifted.data());
        double plus=0.;gpu.AccumulateLoss(plus);
        for(int i=0;i<n;++i) shifted[i]=original[i]-step*direction[i];model.nl_set_coeffs(shifted.data());
        double minus=0.;gpu.AccumulateLoss(minus);model.nl_set_coeffs(original.data());
        REQUIRE(gg.dot(direction)==Catch::Approx((plus-minus)/(2*step)).margin(5e-6*std::max(1.,gg.norm())));
    }
    cpu.TrainCG();REQUIRE(cpu.cg_termination_type>0);
    double cpu_final_loss=0.;cpu.AccumulateLoss(cpu_final_loss);
    const auto cpu_final=cg_coefficients(model);
    model.nl_set_coeffs(original.data());
    gpu.TrainCG();REQUIRE(gpu.cg_termination_type>0);REQUIRE(gpu.bfgs_accepted_iter>0);
    double final_loss=0.;gpu.AccumulateLoss(final_loss);REQUIRE(final_loss<lg);
    REQUIRE(gpu.bfgs_accepted_iter==cpu.bfgs_accepted_iter);
    REQUIRE(final_loss==Catch::Approx(cpu_final_loss).margin(5e-8*std::max(1.,std::abs(cpu_final_loss))));
    const auto gpu_final=cg_coefficients(model);
    const Eigen::Map<const Eigen::VectorXd> cpu_parameters(cpu_final.data(),n),gpu_parameters(gpu_final.data(),n);
    REQUIRE((gpu_parameters-cpu_parameters).norm()<5e-7*std::max(1.,cpu_parameters.norm()));
    REQUIRE(gpu.validation_evaluations>0);
    gpu.SaveCGResult((files.directory/"accepted.mtp").string());
    TestType restored;setup_cg_model(restored);load_cg_model(restored,(files.directory/"accepted.mtp").string());
    REQUIRE(static_cast<const PotMTP&>(restored).inited());
    REQUIRE(static_cast<AnyPotential&>(restored).inited());
    auto a=target,b=target;model.getEFS(a);restored.getEFS(b);
    REQUIRE(a.energy_==Catch::Approx(b.energy_).margin(1e-10));
    for(int i=0;i<a.size();++i) REQUIRE((a.forces_[i]-b.forces_[i]).norm()<1e-9);
}

TEMPLATE_TEST_CASE("GPU SOG optimizer stages preserve warmup masks and save the final model",
    "[gpu][training][cg][stages]",PotMTP_LES,PotMTP_M<m_model_mtpq2>,
    PotMTP_M<m_model_mtpfs>,PotMTP_M<m_model_nn1>,PotMTP_M<m_model_nn2>) {
    if(!cg_gpu_available()) SKIP("No GPU");
    const std::string schedule=GENERATE("mtp_sog_from_scratch","mtp_sog_resume");
    const std::string optimizer=GENERATE("cg","bfgs","sgd","shlbfgs");
    INFO("optimizer="<<optimizer<<", schedule="<<schedule);
    CGFiles files;TestType model;setup_cg_model(model);model.set_inited();
    auto target=LESCurvature::geometry();model.getEFS(target);target.energy_+=.1;
    std::vector<Atoms> atoms{target};std::vector<Neighborhoods> nbhs{Neighborhoods(target,model.cutoff)};
    const auto initial_widths=model.model_les.sog_log_bandwidths;
    const auto initial=cg_coefficients(model);
    auto settings=cg_gpu_settings(files,true);settings.Modify("training_schedule",schedule);
    settings.Modify("learning_rate","1e-5");settings.Modify("momentum","0");settings.Modify("batch_size","100000");
    settings.Modify("sr_stage_iterations","1");settings.Modify("les_stage_iterations","1");settings.Modify("joint_stage_iterations","1");
    TrainNonlinear gpu(settings);gpu.init(atoms,nbhs,&model);gpu.init_valid(atoms,nbhs);
    TestType cpu_model;setup_cg_model(cpu_model);cpu_model.nl_set_coeffs(initial.data());
    auto cpu_settings=settings;cpu_settings.Modify("device","cpu");
    cpu_settings.Modify("save_to",(files.directory/"cpu-result.mtp").string());
    TrainNonlinear cpu(cpu_settings);cpu.init(atoms,nbhs,&cpu_model);cpu.init_valid(atoms,nbhs);
    run_staged_optimizer(cpu,optimizer);double cpu_loss=0.;cpu.AccumulateLoss(cpu_loss);const auto cpu_final=cg_coefficients(cpu_model);
    run_staged_optimizer(gpu,optimizer);
    if(optimizer=="cg") REQUIRE(gpu.cg_termination_type>0);
    double gpu_loss=0.;gpu.AccumulateLoss(gpu_loss);
    if(optimizer=="cg" || optimizer=="bfgs") {
        REQUIRE(gpu.bfgs_accepted_iter==cpu.bfgs_accepted_iter);REQUIRE(cpu.bfgs_accepted_iter>0);
    } else {REQUIRE(gpu.bfgs_iter==cpu.bfgs_iter);REQUIRE(cpu.bfgs_iter>0);}
    REQUIRE(gpu_loss==Catch::Approx(cpu_loss).margin(2e-7*std::max(1.,std::abs(cpu_loss))));
    const auto gpu_final=cg_coefficients(model);const int n=gpu_final.size();
    const Eigen::Map<const Eigen::VectorXd> cpu_parameters(cpu_final.data(),n),gpu_parameters(gpu_final.data(),n);
    REQUIRE((gpu_parameters-cpu_parameters).norm()<2e-6*std::max(1.,cpu_parameters.norm()));
    if(optimizer=="sgd") {
        const Eigen::Map<const Eigen::VectorXd> initial_parameters(initial.data(),n);
        const double movement=(cpu_parameters-initial_parameters).norm();REQUIRE(movement>1e-12);
        REQUIRE((gpu_parameters-cpu_parameters).norm()<1e-6*movement);
    }
    const auto path=(files.directory/"result.mtp").string();
    REQUIRE(std::filesystem::exists(path));REQUIRE(std::filesystem::exists(path+".sog.snapshot"));
    TestType warmup;setup_cg_model(warmup);load_cg_model(warmup,path+".sog.snapshot");
    for(size_t i=0;i<initial_widths.size();++i)
        REQUIRE(warmup.model_les.sog_log_bandwidths[i]==Catch::Approx(initial_widths[i]).margin(1e-12));
    if(schedule.ends_with("from_scratch")) {
        REQUIRE(std::filesystem::exists(path+".sr.snapshot"));TestType sr;setup_cg_model(sr);load_cg_model(sr,path+".sr.snapshot");
        const auto parameters=cg_coefficients(sr);
        for(const auto& group:sr.nl_parameter_groups())
            if(group.name=="les_species" || group.name=="les_moment")
                for(int i=group.begin;i<group.end;++i) REQUIRE(parameters[i]==0.);
    }
    auto sr_parameters=initial;
    if(schedule.ends_with("from_scratch")) {
        TestType sr;setup_cg_model(sr);load_cg_model(sr,path+".sr.snapshot");sr_parameters=cg_coefficients(sr);
    }
    const auto warm_parameters=cg_coefficients(warmup);
    for(const auto& group:warmup.nl_parameter_groups()) if(group.name.starts_with("sr_"))
        for(int i=group.begin;i<group.end;++i)
            REQUIRE(warm_parameters[i]==Catch::Approx(sr_parameters[i]).margin(1e-12));
    TestType saved;setup_cg_model(saved);load_cg_model(saved,path);auto a=target,b=target;model.getEFS(a);saved.getEFS(b);
    REQUIRE(a.energy_==Catch::Approx(b.energy_).margin(1e-10));
}
TEMPLATE_TEST_CASE("GPU SOG stage failures restore every supported head's physical parameters",
    "[gpu][training][cg][stages][rollback]",PotMTP_LES,PotMTP_M<m_model_mtpq2>,
    PotMTP_M<m_model_mtpfs>,PotMTP_M<m_model_nn1>,PotMTP_M<m_model_nn2>) {
    if(!cg_gpu_available()) SKIP("No GPU");
    CGFiles files;TestType model;setup_cg_model(model);model.set_inited();
    auto target=LESCurvature::geometry();model.getEFS(target);target.energy_+=.1;
    std::vector<Atoms> atoms{target};std::vector<Neighborhoods> nbhs{Neighborhoods(target,model.cutoff)};
    auto settings=cg_gpu_settings(files,true);settings.Modify("training_schedule","mtp_sog_from_scratch");
    settings.Modify("sr_stage_iterations","1");settings.Modify("les_stage_iterations","1");settings.Modify("joint_stage_iterations","1");
    TrainNonlinear gpu(settings);gpu.init(atoms,nbhs,&model);
    const auto initial=cg_coefficients(model),widths=model.model_les.sog_log_bandwidths,weights=model.model_les.sog_weights;
    const auto before=gpu.CurrentSettings().settings;
    std::filesystem::create_directory((files.directory/"result.mtp").string()+".sog.snapshot");
    REQUIRE_THROWS_WITH(gpu.TrainCG(),Catch::Matchers::ContainsSubstring("regular file"));
    REQUIRE(cg_coefficients(model)==initial);REQUIRE(model.lr_les);
    REQUIRE(model.model_les.sog_log_bandwidths==widths);REQUIRE(model.model_les.sog_weights==weights);
    REQUIRE(gpu.CurrentSettings().settings==before);REQUIRE(gpu.cg_termination_type==0);
}
#endif
