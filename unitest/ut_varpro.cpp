// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "train/varpro.h"
#include "train/train_nonlinear_potential.h"
#include "potentials/potential_mtp_LES.h"
#include "tools/constants.h"
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

using Catch::Approx;
using Potentials::PotMTP;
namespace {
struct Data {
    PotMTP p{"unitest/test-data/portable-mtp/les.mtp"};
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> nbhs;
    Data() {
        p.train_init();p.set_inited();
        for(int k=0;k<16;++k) {
            Atoms a;a.id=k;a.cell_=12.*Eigen::Matrix3d::Identity();
            a.pos_={{2.,2.,2.},{4.1+.2*std::sin(k),2.2,2.},
                    {2.3,4.2+.2*std::cos(k),2.4},{5.,4.8,3.1+.15*std::sin(2*k)}};
            a.chemmap_={0,1,0,1};a.fitness=.5+.1*k;a.atomic_fitness={0.,.3,2.,4.};
            p.getEFS(a);a.energy_+=.001*std::cos(k);
            atoms.push_back(a);nbhs.emplace_back(a,p.cutoff);
        }
        for(int i=0;i<p.radial_coeff_size;++i) p.radial_coeffs[i]+=.01*std::cos(i);
        for(int i=0;i<p.lin_coeff_size;++i) p.linear_coeffs[i]+=.03*std::sin(i+1);
    }
};
GPSettings settings() {
    GPSettings s;
    s.Modify("energy_weight","1");s.Modify("force_weight",".05");s.Modify("stress_weight",".001");
    s.Modify("weight_L2",".0001");s.Modify("penalty_weight",".00001");
    s.Modify("bfgs_grad_tol","0");s.Modify("bfgs_conv_tol","0");s.Modify("bfgs_step_tol","0");
    s.Modify("bfgs_maxiteration","2");s.Modify("save_every","1");
    return s;
}
Eigen::VectorXd coeffs(PotMTP& p) {
    Eigen::VectorXd c(p.get_NL_coeffs_count());p.nl_get_coeffs(c.data());return c;
}
struct Output {
    std::filesystem::path dir;
    Output() {
        static int n=0;dir="varpro-test-"+std::to_string(++n);
        REQUIRE(std::filesystem::create_directory(dir));
    }
    ~Output() {std::filesystem::remove_all(dir);}
    void configure(GPSettings& s) {
        s.Modify("save_to",(dir/"model.mtp").string());s.Modify("varpro_report",(dir/"report.jsonl").string());
    }
    std::string report() {
        std::ifstream f(dir/"report.jsonl");return {std::istreambuf_iterator<char>(f),{}};
    }
};
double field(const std::string& text,const std::string& name) {
    std::smatch m;
    REQUIRE(std::regex_search(text,m,std::regex("\\\""+name+"\\\":([-+0-9.eE]+)")));
    return std::stod(m[1]);
}
double force_error(PotMTP& p,const std::vector<Atoms>& data) {
    double squared=0.;int count=0;
    for(auto a:data) {auto b=a;p.getEFS(b);for(int i=0;i<a.size();++i) {
        squared+=(a.forces_[i]-b.forces_[i]).squaredNorm();count+=3;
    }}
    return std::sqrt(squared/count);
}
}

TEST_CASE("Profiled MTP probes agree with separate refit and gradient and restore all coefficients", "[varpro]") {
    Data d;TrainNonlinear trainer(settings());trainer.init(d.atoms,d.nbhs,&d.p);
    EFSFitWeights weights;weights.force=.05;weights.stress=.001;
    LinearRefitSettings ls;ls.mode="sr";ls.report="";
    MTPProfiledObjective objective(d.p,d.atoms,d.nbhs,weights,1e-5,ls,
        [&](double& f,Eigen::VectorXd& g){trainer.AccumulateGard(f,g);});
    const auto original=coeffs(d.p);
    Eigen::VectorXd r=original.head(d.p.radial_coeff_size);r[0]+=.02;
    const auto result=objective.evaluate(r);
    REQUIRE((coeffs(d.p).array()==original.array()).all());
    REQUIRE(result.linear_gradient_inf<1e-7);
    REQUIRE(result.objective==Approx(result.linear.data_after+result.radial_penalty+result.linear_penalty).margin(1e-10));
    auto trial=original.eval();trial.head(r.size())=r;d.p.nl_set_coeffs(trial.data());
    auto reference=linear_refit(d.p,d.atoms,d.nbhs,weights,1e-5,ls);
    Eigen::VectorXd gradient(original.size());double f;trainer.AccumulateGard(f,gradient);
    REQUIRE(f==Approx(result.objective).margin(1e-12));
    REQUIRE((gradient.head(r.size())-result.gradient).norm()<1e-10);
    REQUIRE((coeffs(d.p)-result.coefficients).norm()<1e-10);
    // A rejected probe after an accepted point restores both radial and linear blocks.
    const auto accepted=coeffs(d.p);r[1]+=.3;objective.evaluate(r);
    REQUIRE((coeffs(d.p).array()==accepted.array()).all());
    MTPProfiledObjective failure(d.p,d.atoms,d.nbhs,weights,1e-5,ls,
        [](double&,Eigen::VectorXd&){throw std::runtime_error("injected gradient failure");});
    REQUIRE_THROWS(failure.evaluate(r));
    REQUIRE((coeffs(d.p).array()==accepted.array()).all());
}

TEST_CASE("Profiled radial gradient matches finite differences with a fresh inner solve", "[varpro]") {
    Data d;TrainNonlinear trainer(settings());trainer.init(d.atoms,d.nbhs,&d.p);
    EFSFitWeights weights;weights.force=.05;weights.stress=.001;
    LinearRefitSettings ls;ls.mode="sr";ls.report="";
    MTPProfiledObjective objective(d.p,d.atoms,d.nbhs,weights,1e-5,ls,
        [&](double& f,Eigen::VectorXd& g){trainer.AccumulateGard(f,g);});
    const Eigen::VectorXd r=coeffs(d.p).head(d.p.radial_coeff_size);
    const auto result=objective.evaluate(r);
    for(int k=0;k<3;++k) {
        Eigen::VectorXd v(r.size());
        for(int i=0;i<v.size();++i) v[i]=std::sin((i+1)*(k+1));v.normalize();
        constexpr double h=2e-5;
        const double numeric=(objective.evaluate(r+h*v).objective-objective.evaluate(r-h*v).objective)/(2*h);
        REQUIRE(numeric==Approx(result.gradient.dot(v)).margin(2e-7));
    }
}

TEST_CASE("VP checkpoints, metrics and coefficient restarts refer to accepted models", "[varpro][training]") {
    Data d;Output output;auto s=settings();output.configure(s);
    auto validation=d.atoms;
    // Validation has different labels to exercise selection separately from training.
    for(auto& a:validation) for(auto& f:a.forces_) f*=1.05;
    TrainNonlinear trainer(s);trainer.init(d.atoms,d.nbhs,&d.p);trainer.init_valid(validation,d.nbhs);
    trainer.TrainVarPro();
    REQUIRE(trainer.bfgs_accepted_iter==2);
    const auto report=output.report();
    REQUIRE(report.find("\"event\":\"failed\"")==std::string::npos);
    REQUIRE(report.find("\"type\":5")!=std::string::npos);
    std::istringstream lines(report);std::string line;
    double minloss=1e100,minvalid=1e100,initial=0.;
    while(std::getline(lines,line)) if(line.find("\"loss\":")!=std::string::npos) {
        const double total=field(line,"total");
        REQUIRE(total==Approx(field(line,"data")+field(line,"radial_penalty")+field(line,"linear_penalty")).margin(1e-10));
        minloss=std::min(minloss,total);
        auto pos=line.find("\"validation\":{");REQUIRE(pos!=std::string::npos);
        minvalid=std::min(minvalid,field(line.substr(pos),"force_eV_A"));
        if(line.find("\"event\":\"initial\"")!=std::string::npos) initial=total;
    }
    PotMTP saved((output.dir/"model.mtp").string());
    REQUIRE((coeffs(saved)-coeffs(d.p)).norm()<1e-10);
    double stress_squared=0.;
    for(auto a:validation) {
        auto b=a;saved.getEFS(b);
        const double factor=160.21766208/std::abs(a.cell_.determinant());
        stress_squared+=(b.stress_-a.stress_).squaredNorm()*factor*factor;
    }
    const auto final_pos=report.find("\"event\":\"final\"");
    REQUIRE(final_pos!=std::string::npos);
    REQUIRE(field(report.substr(final_pos),"stress_GPa")
        ==Approx(std::sqrt(stress_squared/(9*validation.size()))).margin(1e-10));
    PotMTP best((output.dir/"model.mtp.best-validation.mtp").string());
    REQUIRE(force_error(best,validation)==Approx(minvalid).margin(1e-10));
    PotMTP trainbest((output.dir/"model.mtp.best-training.mtp").string());
    TrainNonlinear check(s);check.init(d.atoms,d.nbhs,&trainbest);double f;check.AccumulateLoss(f);
    REQUIRE(f==Approx(minloss).margin(1e-10));REQUIRE(f<initial);
    REQUIRE(std::filesystem::exists(output.dir/"model.mtp.iter-000002.mtp"));
    Output resumed;resumed.configure(s);s.Modify("bfgs_iteration_offset","2");s.Modify("bfgs_maxiteration","1");
    TrainNonlinear continuation(s);continuation.init(d.atoms,d.nbhs,&saved);continuation.TrainVarPro();
    REQUIRE(std::filesystem::exists(resumed.dir/"model.mtp.iter-000003.mtp"));
    REQUIRE(resumed.report().find("coefficients_only_history_reset")!=std::string::npos);
}

TEST_CASE("VP stress validation and checkpoint eligibility are rotation invariant", "[varpro][stress]") {
    const double cap=GENERATE(.9,1.1);
    for(const double angle:{0.,M_PI/4.}) {
        Data d;Output output;auto s=settings();output.configure(s);
        s.Modify("bfgs_maxiteration","0");
        s.Modify("weight_L2","0");s.Modify("penalty_weight","0");
        s.Modify("validation_metric","stress");
        s.Modify("validation_max_stress",std::to_string(cap));
        // Exact training labels keep the initial projection at this model.
        for(auto& a:d.atoms) d.p.getEFS(a);
        Atoms validation=d.atoms.front();
        validation.stress_(0,0)+=3*std::abs(validation.cell_.determinant())/eVA3_to_GPa;
        const Eigen::Matrix3d rotation=Eigen::AngleAxisd(angle,Eigen::Vector3d::UnitZ()).toRotationMatrix();
        validation.cell_=(validation.cell_*rotation.transpose()).eval();
        for(auto& r:validation.pos_) r=(r*rotation.transpose()).eval();
        for(auto& f:validation.forces_) f=(rotation*f).eval();
        validation.stress_=(rotation*validation.stress_*rotation.transpose()).eval();
        std::vector<Atoms> valid{validation};
        std::vector<Neighborhoods> valid_nbhs;
        valid_nbhs.emplace_back(valid[0],d.p.cutoff);
        TrainNonlinear trainer(s);trainer.init(d.atoms,d.nbhs,&d.p);
        trainer.init_valid(valid,valid_nbhs);trainer.TrainVarPro();
        const auto report=output.report();
        REQUIRE(field(report,"schema_version")==3);
        REQUIRE(report.find("\"stress_metric\":\"full_tensor_rmse_9\"")!=std::string::npos);
        const auto pos=report.find("\"validation\":{");REQUIRE(pos!=std::string::npos);
        REQUIRE(field(report.substr(pos),"stress_GPa")==Approx(1.).margin(1e-10));
        REQUIRE(std::filesystem::exists(output.dir/"model.mtp.best-validation.mtp")== (cap>1.));
    }
}

TEST_CASE("VP rejects unsupported objectives and does not select missing or ineligible validation metrics", "[varpro]") {
    Data d;Output output;auto s=settings();output.configure(s);s.Modify("bfgs_maxiteration","0");
    SECTION("L1") {
        s.Modify("weight_L1",".1");TrainNonlinear trainer(s);trainer.init(d.atoms,d.nbhs,&d.p);
        const auto before=coeffs(d.p);REQUIRE_THROWS(trainer.TrainVarPro());
        REQUIRE((coeffs(d.p).array()==before.array()).all());
    }
    SECTION("charge preflight") {
        TrainNonlinear trainer(s);trainer.init(d.atoms,d.nbhs,&d.p);
        GPSettings charge;charge.Modify("charge_weight","1");trainer.ApplySettings(charge);
        const auto before=coeffs(d.p);
        REQUIRE_THROWS_WITH(trainer.TrainVarPro(),Catch::Matchers::ContainsSubstring("without a charge loss"));
        REQUIRE((coeffs(d.p).array()==before.array()).all());
        REQUIRE(std::filesystem::is_empty(output.dir));
    }
    SECTION("unproved derived potential") {
        struct Unproved : Potentials::PotMTP { using PotMTP::PotMTP; };
        Unproved derived("unitest/test-data/portable-mtp/les.mtp");derived.train_init();
        REQUIRE_THROWS(MTPProfiledObjective(derived,d.atoms,d.nbhs,EFSFitWeights{},0.,{},
            [](double&,Eigen::VectorXd&){}));
    }
    SECTION("missing labels") {
        auto v=d.atoms;for(auto& a:v) a.forces_.clear();
        TrainNonlinear trainer(s);trainer.init(d.atoms,d.nbhs,&d.p);trainer.init_valid(v,d.nbhs);
        REQUIRE_THROWS(trainer.TrainVarPro());
        REQUIRE_FALSE(std::filesystem::exists(output.dir/"model.mtp.best-validation.mtp"));
    }
    SECTION("validation cap") {
        s.Modify("validation_max_force","0");TrainNonlinear trainer(s);trainer.init(d.atoms,d.nbhs,&d.p);
        trainer.init_valid(d.atoms,d.nbhs);trainer.TrainVarPro();
        REQUIRE_FALSE(std::filesystem::exists(output.dir/"model.mtp.best-validation.mtp"));
        REQUIRE(output.report().find("\"best_validation_iteration\":-1")!=std::string::npos);
        REQUIRE_THROWS(trainer.TrainVarPro()); // never silently overwrite a report
    }
}

TEST_CASE("VP rejects inconsistent envelope gradients and restores coefficients", "[varpro]") {
    const int error=GENERATE(0,1,2);
    Data d;TrainNonlinear trainer(settings());trainer.init(d.atoms,d.nbhs,&d.p);
    const auto original=coeffs(d.p);
    EFSFitWeights weights;weights.force=.05;weights.stress=.001;
    MTPProfiledObjective objective(d.p,d.atoms,d.nbhs,weights,1e-5,{},
        [&](double& f,Eigen::VectorXd& g){
            trainer.AccumulateGard(f,g);
            if(error==0) f+=1.;
            if(error==1) g[g.size()-1]=1.;
            if(error==2) g[0]=std::numeric_limits<double>::quiet_NaN();
        });
    REQUIRE_THROWS(objective.evaluate(original.head(d.p.radial_coeff_size)));
    REQUIRE((coeffs(d.p).array()==original.array()).all());
}

TEST_CASE("VP rejects invalid solver settings and conflicting paths before creating outputs", "[varpro]") {
    const int error=GENERATE(0,1,2,3,4);
    Data d;Output output;auto s=settings();output.configure(s);
    if(error==0) s.Modify("linear_refit_rtol","nan");
    if(error==1) s.Modify("linear_refit_solver","unknown");
    if(error==2) s.Modify("save_to",(output.dir/"report.jsonl").string());
    if(error==3) s.Modify("varpro_report",(output.dir/"model.mtp.best-data.mtp.tmp").string());
    if(error==4) {std::ofstream existing(output.dir/"model.mtp.best-data.mtp");existing<<"preserve";}
    TrainNonlinear trainer(s);trainer.init(d.atoms,d.nbhs,&d.p);
    REQUIRE_THROWS(trainer.TrainVarPro());
    REQUIRE_FALSE(std::filesystem::exists(output.dir/"report.jsonl"));
}

TEST_CASE("VP saves minimum data loss independently of penalties and snapshot interval", "[varpro][training]") {
    Data d;Output output;auto s=settings();output.configure(s);
    s.Modify("penalty_weight","100");s.Modify("bfgs_maxiteration","8");
    s.Modify("save_every","100");
    TrainNonlinear trainer(s);trainer.init(d.atoms,d.nbhs,&d.p);trainer.TrainVarPro();
    std::istringstream lines(output.report());std::string line;
    double min_data=1e100,final_data=0.,min_total=1e100;int best_data_iteration=-1,best_total_iteration=-1;
    while(std::getline(lines,line)) if(line.find("\"loss\":")!=std::string::npos) {
        const auto iteration=int(field(line,"iteration"));
        const double data=field(line,"data"),total=field(line,"total");
        if(data<min_data) {min_data=data;best_data_iteration=iteration;}
        if(total<min_total) {min_total=total;best_total_iteration=iteration;}
        REQUIRE(field(line,"best_data_iteration")==best_data_iteration);
        final_data=data;
    }
    // This deliberately strong radial penalty drives down total loss while
    // sacrificing the unpenalized data fit; selecting by total would be wrong.
    REQUIRE(best_data_iteration!=best_total_iteration);
    REQUIRE(min_data<final_data);
    REQUIRE_FALSE(std::filesystem::exists(output.dir/"model.mtp.iter-000001.mtp"));
    PotMTP saved((output.dir/"model.mtp.best-data.mtp").string());
    s.Modify("penalty_weight","0");s.Modify("weight_L2","0");
    TrainNonlinear check(s);check.init(d.atoms,d.nbhs,&saved);double data;check.AccumulateLoss(data);
    REQUIRE(data==Approx(min_data).margin(1e-10));
    const auto report=output.report();const auto pos=report.find("\"event\":\"termination\"");
    REQUIRE(field(report.substr(pos),"best_data_iteration")==best_data_iteration);
}

TEST_CASE("Scaled VP coordinates preserve the objective and obey the envelope chain rule", "[varpro]") {
    Data d;TrainNonlinear trainer(settings());trainer.init(d.atoms,d.nbhs,&d.p);
    EFSFitWeights weights;weights.force=.05;weights.stress=.001;
    MTPProfiledObjective objective(d.p,d.atoms,d.nbhs,weights,1e-5,{},
        [&](double& f,Eigen::VectorXd& g){trainer.AccumulateGard(f,g);});
    Eigen::VectorXd scales(d.p.radial_coeff_size);
    for(int i=0;i<scales.size();++i) scales[i]=std::pow(4.,std::sin(i+1));
    VarProRadialScaling coordinates(scales);
    const Eigen::VectorXd r=coeffs(d.p).head(scales.size()),z=coordinates.optimizer(r);
    REQUIRE((coordinates.physical(z)-r).norm()<1e-13);
    const auto physical=objective.evaluate(r),scaled=objective.evaluate(coordinates.physical(z));
    REQUIRE(scaled.objective==Approx(physical.objective).margin(1e-12));
    REQUIRE((scaled.coefficients-physical.coefficients).norm()<1e-9);
    const auto gradient=coordinates.gradient(scaled.gradient);
    for(int k=0;k<3;++k) {
        Eigen::VectorXd direction(z.size());
        for(int i=0;i<z.size();++i) direction[i]=std::sin((i+1)*(k+1));direction.normalize();
        for(double h:{2e-5,1e-5}) {
            const double numeric=(objective.evaluate(coordinates.physical(z+h*direction)).objective
                -objective.evaluate(coordinates.physical(z-h*direction)).objective)/(2*h);
            REQUIRE(numeric==Approx(gradient.dot(direction)).margin(2e-7));
        }
    }
    scales[0]=0.;REQUIRE_THROWS(VarProRadialScaling(scales));
    scales[0]=std::numeric_limits<double>::quiet_NaN();REQUIRE_THROWS(VarProRadialScaling(scales));
    REQUIRE_THROWS(coordinates.physical(Eigen::VectorXd::Ones(z.size()+1)));
}

TEST_CASE("VP loads exact radial scales and rejects malformed files before creating reports", "[varpro][training]") {
    const int mode=GENERATE(0,1,2,3,4,5);
    Data d;Output output;auto s=settings();output.configure(s);
    const auto path=output.dir/"scales.txt";s.Modify("varpro_radial_scale_file",path.string());
    if(mode!=1) {
        std::ofstream out(path);
        for(int i=0;i<d.p.radial_coeff_size-(mode==2);++i) out << ((mode==3 && i==0) ? -1.:1.) << '\n';
        if(mode==4) out << "1\n";
        if(mode==5) out << "garbage\n";
    }
    TrainNonlinear trainer(s);trainer.init(d.atoms,d.nbhs,&d.p);
    if(mode!=0) {
        REQUIRE_THROWS(trainer.TrainVarPro());
        REQUIRE_FALSE(std::filesystem::exists(output.dir/"report.jsonl"));
        return;
    }
    trainer.TrainVarPro();
    Data plain;Output reference;reference.configure(s);s.Modify("varpro_radial_scale_file","");
    TrainNonlinear unscaled(s);unscaled.init(plain.atoms,plain.nbhs,&plain.p);unscaled.TrainVarPro();
    REQUIRE((coeffs(d.p)-coeffs(plain.p)).norm()<1e-12);
    REQUIRE(output.report().find("physical_r=scale*z")!=std::string::npos);
}

TEST_CASE("VP residual refinement repairs cancellation without changing the stationarity threshold", "[varpro]") {
    Data d;TrainNonlinear trainer(settings());trainer.init(d.atoms,d.nbhs,&d.p);
    EFSFitWeights w;w.force=.05;w.stress=.001;
    const auto ordinary=coeffs(d.p);
    auto gradient=[&](double& f,Eigen::VectorXd& g){trainer.AccumulateGard(f,g);};
    const auto reference=MTPProfiledObjective(d.p,d.atoms,d.nbhs,w,1e-5,{},gradient)
        .evaluate(ordinary.head(d.p.radial_coeff_size));
    // Large starting moments cause subtraction of nearly equal coefficients in
    // the first incremental solve. Reference labels and species offsets stay fixed.
    for(int i=d.p.species_count;i<d.p.lin_coeff_size;++i)
        d.p.linear_coeffs[i]+=1e8*std::sin(i+1);
    const auto original=coeffs(d.p);
    const Eigen::VectorXd radial=original.head(d.p.radial_coeff_size);
    MTPProfiledObjective disabled(d.p,d.atoms,d.nbhs,w,1e-5,{},gradient,1e-7,0);
    REQUIRE_THROWS_WITH(disabled.evaluate(radial),Catch::Matchers::ContainsSubstring("not stationary enough"));
    REQUIRE((coeffs(d.p).array()==original.array()).all());
    const auto result=MTPProfiledObjective(d.p,d.atoms,d.nbhs,w,1e-5,{},gradient,1e-7,2).evaluate(radial);
    REQUIRE(result.linear_gradient_history.size()>1);
    REQUIRE(result.linear_gradient_history.size()<=3);
    REQUIRE(result.linear_gradient_history.front()>1e-7);
    REQUIRE(result.linear_gradient_inf<=1e-7);
    REQUIRE(result.objective==Approx(reference.objective).margin(1e-10));
    REQUIRE((result.gradient-reference.gradient).norm()<1e-7);
    REQUIRE((result.coefficients.head(radial.size()).array()==radial.array()).all());
    REQUIRE((coeffs(d.p).array()==original.array()).all());
    REQUIRE((result.coefficients.tail(d.p.lin_coeff_size)-original.tail(d.p.lin_coeff_size)-result.linear.delta).norm()<1e-6);
    // All structures have equal species counts: the species-offset difference
    // is the composition-null direction and must remain fixed.
    const int n=d.p.radial_coeff_size;
    REQUIRE(result.coefficients[n]-result.coefficients[n+1]==Approx(original[n]-original[n+1]).margin(1e-7));
}

TEST_CASE("VP limits residual refinement and rejects invalid objectives without retry", "[varpro]") {
    const int error=GENERATE(0,1,2);
    Data d;TrainNonlinear trainer(settings());trainer.init(d.atoms,d.nbhs,&d.p);
    EFSFitWeights w;w.force=.05;w.stress=.001;
    const auto original=coeffs(d.p);int calls=0;
    MTPProfiledObjective objective(d.p,d.atoms,d.nbhs,w,1e-5,{},
        [&](double& f,Eigen::VectorXd& g){
            ++calls;trainer.AccumulateGard(f,g);
            if(error==0) g[g.size()-1]=1.;
            if(error==1) f+=1.;
            if(error==2) g[0]=std::numeric_limits<double>::quiet_NaN();
        },1e-7,2);
    REQUIRE_THROWS(objective.evaluate(original.head(d.p.radial_coeff_size)));
    REQUIRE(calls==(error==0 ? 3 : 1));
    REQUIRE((coeffs(d.p).array()==original.array()).all());
    REQUIRE_THROWS(MTPProfiledObjective(d.p,d.atoms,d.nbhs,w,1e-5,{},
        [](double&,Eigen::VectorXd&){},1e-7,-1));
}
