// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "fixtures/mtp_nonlinear_gpu.h"
#ifdef IMRMLP_MPI
int main(int argc,char **argv) {
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided); mpi.InitComm(MPI_COMM_SELF);
    const int result=Catch::Session().run(argc,argv); MPI_Finalize(); return result;
}
#endif
namespace F=MTPNonlinearGpuFixture;

TEST_CASE("trainmtp GPU settings distinguish nonlinear from linear curvature", "[training][trainmtp-nonlinear]") {
    Trainer_MTP defaults;
    REQUIRE(std::stod(defaults.CurrentSettings().settings.find("lbfgspp:epsgmax")->second)==0.);
    GPSettings explicit_option; explicit_option.Modify("lbfgspp:epsgmax",".001"); defaults.ApplySettings(explicit_option);
    REQUIRE(std::stod(defaults.CurrentSettings().settings.find("lbfgspp:epsgmax")->second)==.001);
    auto settings=F::settings(false); settings.Modify("device","cuda");
    // CPU-only builds reject the explicit device without creating any owner.
#ifndef IMRMLP_GPU
    Trainer_MTP t(settings); REQUIRE_THROWS(t.ValidateGpuSettings());
#else
    settings.Modify("device",F::G::compiled_backend());
    settings.Modify("random_init","true"); settings.Modify("apply_disp","true");
    settings.Modify("linear_fit_bfgs","true");
    Trainer_MTP t(settings); REQUIRE_NOTHROW(t.ValidateGpuNonlinearSettings());
    settings.Modify("hvp_weight","1"); t.ApplySettings(settings);
    REQUIRE_THROWS(t.ValidateGpuNonlinearSettings());
    settings.Modify("hvp_weight","0"); settings.Modify("do_ladder","true"); t.ApplySettings(settings);
    REQUIRE_THROWS(t.ValidateGpuSettings());
#endif
}
#ifdef IMRMLP_GPU
TEST_CASE("trainmtp GPU nonlinear weighted objective gradient and metrics match CPU", "[gpu][training][trainmtp-nonlinear]") {
    if (!F::available()) SKIP("No GPU");
    const bool device_loss=GENERATE(false,true), deterministic=GENERATE(false,true);
    const int frozen=GENERATE(0,1,2,3);
    F::Capture capture; F::Fixture f; auto cpu_model=F::copy(f.model);
    auto s=F::settings(); s.Modify("gpu_device_loss",device_loss?"true":"false");
    s.Modify("gpu_deterministic_reduction",deterministic?"true":"false");
    s.Modify("fix_radial_coeffs",frozen==1?"true":"false");
    s.Modify("fix_linear_coeffs",frozen==2?"true":"false");
    s.Modify("fix_species_coeffs",frozen==3?"true":"false");
    Trainer_MTP gpu(s); gpu.init(f.data,f.neighbors,&f.model); gpu.init_valid(f.data,f.neighbors);
    s.Modify("device","cpu"); Trainer_MTP cpu(s); cpu.init(f.data,f.neighbors,cpu_model.get()); cpu.init_valid(f.data,f.neighbors);
    const int n=gpu.get_relaxed_coeffs_count(); alglib::real_1d_array cg,gg; cg.setlength(n); gg.setlength(n);
    double cl=0.,gl=0.; cpu.AccumulateGard(cl,cg); gpu.AccumulateGard(gl,gg);
    REQUIRE(gl==Catch::Approx(cl).margin(2e-9));
    for (int i=0;i<n;++i) REQUIRE(gg[i]==Catch::Approx(cg[i]).margin(2e-8+1e-8*std::abs(cg[i])));
    REQUIRE(gpu.emonitor.erms==Catch::Approx(cpu.emonitor.erms).margin(2e-9));
    REQUIRE(gpu.emonitor.frms==Catch::Approx(cpu.emonitor.frms).margin(2e-9));
    REQUIRE(gpu.emonitor_valid.srms==Catch::Approx(cpu.emonitor_valid.srms).margin(2e-9));
    auto original=F::coefficients(f.model);
    Eigen::VectorXd direction=Eigen::VectorXd::NullaryExpr(n,[](Eigen::Index i){return std::sin(.71*(i+1));});
    if (frozen==3) direction.segment(f.model.radial_coeff_size,f.model.species_count).setZero();
    direction.normalize();
    alglib::real_1d_array x; x.setlength(n); f.model.ae_get_coeffs(x,gpu.fix_radial_coeffs,gpu.fix_linear_coeffs);
    double plus=0.,minus=0.; const double step=1e-6;
    for (int i=0;i<n;++i) x[i]+=step*direction[i]; f.model.ae_set_coeffs(x,gpu.fix_radial_coeffs,gpu.fix_linear_coeffs); gpu.AccumulateLoss(plus);
    for (int i=0;i<n;++i) x[i]-=2*step*direction[i]; f.model.ae_set_coeffs(x,gpu.fix_radial_coeffs,gpu.fix_linear_coeffs); gpu.AccumulateLoss(minus);
    double dot=0.; for (int i=0;i<n;++i) dot+=gg[i]*direction[i];
    REQUIRE(dot==Catch::Approx((plus-minus)/(2*step)).margin(3e-7*std::max(1.,std::abs(dot))));
    f.model.set_coeffs(original); F::rescale(f.model,1.07*F::scale(f.model)); F::rescale(*cpu_model,F::scale(f.model));
    cpu.AccumulateGard(cl,cg); gpu.AccumulateGard(gl,gg); REQUIRE(gl==Catch::Approx(cl).margin(2e-9));
    for (int i=0;i<n;++i) REQUIRE(gg[i]==Catch::Approx(cg[i]).margin(2e-8+1e-8*std::abs(cg[i])));
    if (f.model.radial_basis_split) {
        f.model.p_radical_basis_pair.back()->scaling *= 1.13;
        cpu_model->p_radical_basis_pair.back()->scaling *= 1.13;
        cpu.AccumulateGard(cl,cg); gpu.AccumulateGard(gl,gg);
        REQUIRE(gl==Catch::Approx(cl).margin(2e-9));
        for (int i=0;i<n;++i) REQUIRE(gg[i]==Catch::Approx(cg[i]).margin(2e-8+1e-8*std::abs(cg[i])));
    }
    REQUIRE(gpu.GetGpuNonlinearReport().validation_evaluations>0);
    REQUIRE(gpu.GetGpuNonlinearReport().peak_device_bytes<=9*1024*1024);
}

TEST_CASE("trainmtp GPU retains zero-weight virial-only error metrics", "[gpu][training][trainmtp-nonlinear]") {
    if (!F::available()) SKIP("No GPU");
    const bool device_loss=GENERATE(false,true), deterministic=GENERATE(false,true);
    F::Capture capture; F::Fixture f(1); auto cpu_model=F::copy(f.model);
    auto s=F::settings(); s.Modify("energy_weight","0"); s.Modify("force_weight","0");
    // A finite normalization exponent can underflow the effective weight.
    // Legacy raw error metrics still include the labeled structure.
    s.Modify("stress_pow","1024");
    s.Modify("gpu_device_loss",device_loss?"true":"false");
    s.Modify("gpu_deterministic_reduction",deterministic?"true":"false");
    Trainer_MTP gpu(s); gpu.init(f.data,f.neighbors,&f.model);
    s.Modify("device","cpu"); Trainer_MTP cpu(s); cpu.init(f.data,f.neighbors,cpu_model.get());
    double cl=0.,gl=0.; cpu.AccumulateLoss(cl); gpu.AccumulateLoss(gl);
    REQUIRE(gl==Catch::Approx(cl).margin(2e-9));
    REQUIRE(cpu.emonitor.sq_stress_error>0.);
    REQUIRE(gpu.emonitor.sq_stress_error==Catch::Approx(cpu.emonitor.sq_stress_error).margin(2e-9));
    REQUIRE(gpu.emonitor.nstresstensor==cpu.emonitor.nstresstensor);
}

TEST_CASE("trainmtp GPU optimizers retain mixed linear phases and publish reloadable models", "[gpu][training][trainmtp-nonlinear][optimizers]") {
    if (!F::available()) SKIP("No GPU");
    const std::string method=GENERATE("lbfgs","cg","sgd");
    const bool fast=GENERATE(false,true);
    F::Capture capture; F::Workspace files; F::Fixture f(3); auto cpu_model=F::copy(f.model);
    auto s=F::settings(); s.Modify("save_to",(files.directory/"gpu.mtp").string());
    s.Modify("lfit_fastalgo",fast?"true":"false"); s.Modify("skip_preinit","false");
    s.Modify("pre_init_iter","1"); s.Modify("linear_fit_bfgs","true");
    s.Modify("bfgs_rescale","true");
    Trainer_MTP gpu(s); gpu.init(f.data,f.neighbors,&f.model); gpu.init_valid(f.data,f.neighbors);
    auto before=F::coefficients(f.model); double initial=0.; gpu.AccumulateLoss(initial);
    s.Modify("device","cpu"); s.Modify("save_to",(files.directory/"cpu.mtp").string());
    Trainer_MTP cpu(s); cpu.init(f.data,f.neighbors,cpu_model.get()); cpu.init_valid(f.data,f.neighbors);
    REQUIRE_NOTHROW(F::optimize(cpu,method)); REQUIRE_NOTHROW(F::optimize(gpu,method));
    REQUIRE((F::coefficients(f.model)-before).norm()>1e-10);
    double cl=0.,gl=0.; cpu.AccumulateLoss(cl); gpu.AccumulateLoss(gl);
    REQUIRE(gl==Catch::Approx(cl).margin(2e-7*std::max(1.,std::abs(cl))));
    REQUIRE(gl<initial);
    F::PotMTP saved((files.directory/"gpu.mtp").string());
    REQUIRE((F::coefficients(saved)-F::coefficients(f.model)).norm()<1e-12);
    REQUIRE(gpu.GetGpuNonlinearReport().gradient_evaluations>0);
}

TEST_CASE("trainmtp GPU nonlinear failure restores model and permits same-trainer retry", "[gpu][training][trainmtp-nonlinear][recovery]") {
    if (!F::available()) SKIP("No GPU");
    const std::string method=GENERATE("lbfgs","cg","sgd");
    F::Capture capture; F::Workspace files; F::Fixture f(3);
    auto s=F::settings(); const auto path=files.directory/"blocked"; std::filesystem::create_directory(path);
    s.Modify("save_to",path.string());
    Trainer_MTP gpu(s); gpu.init(f.data,f.neighbors,&f.model);
    const auto before=F::coefficients(f.model); const auto ridge=f.model.reg_L2_vector; const double scaling=F::scale(f.model);
    REQUIRE_THROWS(F::optimize(gpu,method));
    REQUIRE((F::coefficients(f.model)-before).norm()==0.); REQUIRE(f.model.reg_L2_vector==ridge); REQUIRE(F::scale(f.model)==scaling);
    std::filesystem::remove(path); REQUIRE_NOTHROW(F::optimize(gpu,method)); REQUIRE(std::filesystem::is_regular_file(path));
    s.Modify("gpu_id","99999"); gpu.ApplySettings(s);
    REQUIRE_THROWS(F::optimize(gpu,method));
    s.Modify("gpu_id","0"); gpu.ApplySettings(s); REQUIRE_NOTHROW(F::optimize(gpu,method));
}

TEST_CASE("trainmtp nonlinear GPU shrinks complete batches under a tracked memory cap", "[gpu][training][trainmtp-nonlinear][memory]") {
    if (!F::available()) SKIP("No GPU");
    F::Capture capture; F::Fixture f(8,4);
    for (int i=8;i<256;++i) { f.data.push_back(f.data[i%8]); f.neighbors.push_back(f.neighbors[i%8]); }
    auto s=F::settings(); s.Modify("gpu_memory_mib","1"); s.Modify("gpu_workspace_mib","1");
    s.Modify("gpu_batch_atoms","2048"); s.Modify("gpu_batch_structures","512");
    Trainer_MTP gpu(s); gpu.init(f.data,f.neighbors,&f.model);
    alglib::real_1d_array gradient; gradient.setlength(gpu.get_relaxed_coeffs_count());
    double loss=0.; REQUIRE_NOTHROW(gpu.AccumulateGard(loss,gradient)); REQUIRE(std::isfinite(loss));
    REQUIRE(gpu.GetGpuNonlinearReport().batches>1);
    REQUIRE(gpu.GetGpuNonlinearReport().peak_device_bytes<=2*1024*1024);
    const auto before=F::coefficients(f.model);
    s.Modify("gpu_batch_atoms","3"); gpu.ApplySettings(s);
    REQUIRE_THROWS(gpu.AccumulateGard(loss,gradient)); REQUIRE((F::coefficients(f.model)-before).norm()==0.);
    s.Modify("gpu_batch_atoms","2048"); gpu.ApplySettings(s); REQUIRE_NOTHROW(gpu.AccumulateGard(loss,gradient));
}
#endif
