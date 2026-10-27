// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtp_m.h"
#include "train/train_nonlinear_potential.h"
#include <sstream>
#include <fstream>
#include <filesystem>

using namespace Potentials;
using Catch::Approx;
namespace {
Atoms atoms() {
    Atoms a;
    a.cell_ << 10.,0,0,.8,10.5,0,-.3,.6,11.;
    a.pos_={{2.,2.,2.},{4.1,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
    a.chemmap_={0,1,0,1};a.fitness=1.;a.id=0;
    return a;
}
void initialize(PotMTP_LES& p) {
    p.les_kernel="sog";p.sog_terms=2;
    p.les_species_coeffs[0]=.23;p.les_species_coeffs[1]=-.17;
    std::fill_n(p.les_moment_coeffs,p.get_alpha_scalar_moments(),0.);
    p.les_moment_coeffs[0]=.35;p.les_moment_coeffs[1]=-.21;p.les_moment_coeffs[3]=.12;
    p.train_init();
    p.model_les.sog_weights={4.2,-1.3};
    p.model_les.sog_log_bandwidths={std::log(.7),std::log(1.2)};
    p.train_init();p.set_inited();
}
std::vector<double> coeffs(PotMTP_LES& p) {
    std::vector<double> x(p.get_NL_coeffs_count());p.nl_get_coeffs(x.data());return x;
}
}

TEST_CASE("MTP SOG full EFS gradients include radial and kernel response","[sog][training][grad]") {
    PotMTP_LES p("tests/case-mtp-les/les.mtp");initialize(p);
    const int mode=GENERATE(0,1,2,3);
    const double ew=(mode==0 || mode==3) ? .37 : 0.;
    Eigen::Matrix<double,-1,3> fw(4,3);fw<<.2,-.1,.3,-.5,.7,.1,.3,.2,-.4,.1,.2,.4;
    Eigen::Matrix3d sw;sw<<.1,.2,-.1,.3,-.2,.5,-.1,.2,.7;
    if(mode!=1 && mode!=3)fw.setZero();if(mode!=2 && mode!=3)sw.setZero();
    auto a=atoms();Neighborhoods nbhs(a,p.cutoff);
    auto x=coeffs(p);std::vector<double> g(x.size(),0.);
    Eigen::Matrix<double,-1,3,Eigen::RowMajor> unused;
    p.AccumulateEFSCombinationGard(a,nbhs,ew,fw,sw,unused,g.data());
    if(mode==0) {
        std::vector<double> only(x.size(),0.);
        p.AccumulateEFSCombinationGard(a,nbhs,ew,only.data());
        for(size_t i=0;i<g.size();++i)REQUIRE(g[i]==Approx(only[i]).margin(1e-10));
    }
    auto loss=[&](){auto b=atoms();p.getEFS(b);double l=ew*b.energy_+(sw.array()*b.stress_.array()).sum();for(int i=0;i<b.size();++i)l+=fw.row(i).dot(b.forces_[i]);return l;};
    for(size_t i=0;i<x.size();++i) {
        CAPTURE(mode,i);auto plus=x,minus=x;plus[i]+=1e-6;minus[i]-=1e-6;
        p.nl_set_coeffs(plus.data());double fp=loss();p.nl_set_coeffs(minus.data());double fm=loss();
        REQUIRE(g[i]==Approx((fp-fm)/2e-6).margin(3e-6));
    }
    p.nl_set_coeffs(x.data());
}

TEST_CASE("MTP SOG response forces and virial differentiate total energy","[sog][training]") {
    PotMTP_LES p("tests/case-mtp-les/les.mtp");initialize(p);
    auto a=atoms();p.getEFS(a);constexpr double h=1e-6;
    REQUIRE(a.charges_.empty());REQUIRE_FALSE(p.supports_total_charge_training());
    REQUIRE_THROWS_AS(p.compute_bec(a),GPException);
    for(int i=0;i<a.size();++i)for(int d=0;d<3;++d) {
        auto ap=atoms(),am=atoms();ap.pos_[i](d)+=h;am.pos_[i](d)-=h;p.getE(ap);p.getE(am);
        REQUIRE(a.forces_[i](d)==Approx(-(ap.energy_-am.energy_)/(2*h)).margin(2e-6));
    }
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
        auto ap=atoms(),am=atoms();Eigen::Matrix3d d=Eigen::Matrix3d::Zero();d(i,j)=h;
        ap.cell_=a.cell_*(Eigen::Matrix3d::Identity()+d).transpose();
        am.cell_=a.cell_*(Eigen::Matrix3d::Identity()-d).transpose();
        for(int n=0;n<a.size();++n){ap.pos_[n]=a.pos_[n]*(Eigen::Matrix3d::Identity()+d).transpose();am.pos_[n]=a.pos_[n]*(Eigen::Matrix3d::Identity()-d).transpose();}
        p.getE(ap);p.getE(am);
        REQUIRE(a.stress_(i,j)==Approx(-(ap.energy_-am.energy_)/(2*h)).margin(3e-6));
    }
}

TEST_CASE("SOG v3 roundtrip strict loading and old entry point rejection","[sog][load]") {
    PotMTP_LES p("tests/case-mtp-les/les.mtp");initialize(p);
    std::ostringstream out;p.Save(out);const auto source=out.str();
    REQUIRE(source.find("les_version = 3")!=std::string::npos);
    PotMTP_LES q;std::istringstream in(source);q.Load(in);
    auto a=atoms(),b=atoms();p.getEFS(a);q.getEFS(b);
    REQUIRE(a.energy_==Approx(b.energy_).margin(1e-12));
    auto x=coeffs(p),y=coeffs(q);for(size_t i=0;i<x.size();++i)REQUIRE(x[i]==Approx(y[i]).margin(1e-12));
    for(const auto& field:{"sog_weights","sog_log_bandwidths","sog_zero_mode","sog_self_mode","sog_log_bandwidth_min"}) {
        auto broken=source;auto start=broken.find(std::string(field)+" =");broken.erase(start,broken.find('\n',start)-start+1);
        std::istringstream missing(broken);REQUIRE_THROWS_AS(q.Load(missing),GPException);
    }
    for(const auto& extra:{"sog_terms = 2\n","sog_unknown = 1\n","sog_weights = {0,0}\n","les_version = 3\n",
                          "remove_self_interaction = 0\n","les_charge_treatment = uniform_background\n"}) {
        std::istringstream dup(source+extra);REQUIRE_THROWS_AS(q.Load(dup),GPException);
    }
    PotMTP_M<m_model_mtpq2> m;std::istringstream old(source);REQUIRE_THROWS_AS(m.Load(old),GPException);
    std::istringstream valid(source);q.Load(valid);
    PotMTP_LES coulomb("tests/case-mtp-les/les.mtp");std::ostringstream legacy;coulomb.Save(legacy);
    std::istringstream v2(legacy.str());q.Load(v2);REQUIRE_FALSE(q.uses_sog());REQUIRE(q.model_les.sog_weights.empty());
}

TEST_CASE("SOG trainer fits kernel weights while frozen groups remain unchanged","[sog][training]") {
    PotMTP_LES p("tests/case-mtp-les/les.mtp");initialize(p);
    std::vector<Atoms> data;
    for(int n=0;n<4;++n){auto a=atoms();a.pos_[1](0)+=.15*n;a.id=n;p.getEFS(a);data.push_back(a);}
    p.model_les.sog_weights={1.7,-.4};p.train_init();
    std::vector<Neighborhoods> nbhs;for(auto& a:data)nbhs.emplace_back(a,p.cutoff);
    GPSettings settings;settings.Modify("energy_weight","1");settings.Modify("force_weight","1");
    settings.Modify("stress_weight","0.1");settings.Modify("charge_weight","0");settings.Modify("penalty_weight","0");
    settings.Modify("parameter_scaling","none");settings.Modify("bfgs_grad_tol","1e-14");settings.Modify("bfgs_step_tol","0");settings.Modify("bfgs_conv_tol","0");
    TrainNonlinear trainer(settings);trainer.init(data,nbhs,&p);trainer.ConfigureTrainableGroups({"sog_weights"});
    double before=0.;Eigen::VectorXd gradient(p.get_NL_coeffs_count());trainer.AccumulateGard(before,gradient);
    const auto initial=coeffs(p);
    trainer.TrainNonLinearBFGS(20);
    double after=0.;trainer.AccumulateGard(after,gradient);
    // The fixture has an independent, frozen SR regularizer even when penalty_weight=0.
    double floor=0.;std::vector<double> penalty(initial.size(),0.);p.AddPenaltyGard(0.,floor,penalty.data());
    REQUIRE(before-floor>1e-10);REQUIRE(after-floor<(before-floor)*1e-4);
    const auto fitted=coeffs(p);
    for(const auto& group:p.nl_parameter_groups()) if(group.name!="sog_weights")
        for(int i=group.begin;i<group.end;++i)REQUIRE(initial[i]==Approx(fitted[i]).margin(1e-12));
    REQUIRE(std::abs(fitted[fitted.size()-4]-initial[initial.size()-4])>1e-3);
}

TEST_CASE("SOG staged schedules initialize zero heads and save resumable models","[sog][training]") {
    const std::string schedule=GENERATE(std::string("mtp_sog_from_scratch"),std::string("mtp_sog_resume"));
    PotMTP_LES p("tests/case-mtp-les/les.mtp");initialize(p);
    const int channels=GENERATE(1,3);p.sog_channels=channels;p.train_init();
    auto target=atoms();p.getEFS(target);
    std::fill_n(p.les_species_coeffs,2*channels,0.);
    std::fill_n(p.les_moment_coeffs,channels*p.get_alpha_scalar_moments(),0.);
    std::vector<Atoms> data={target};std::vector<Neighborhoods> nbhs;nbhs.emplace_back(data[0],p.cutoff);
    GPSettings settings;
    settings.Modify("training_schedule",schedule);settings.Modify("charge_weight","0");
    settings.Modify("sr_stage_iterations","1");settings.Modify("les_stage_iterations","1");settings.Modify("joint_stage_iterations","1");
    settings.Modify("sog_zero_init_range","0.1");settings.Modify("penalty_weight","0");
    settings.Modify("save_to",schedule+".mtp");
    TrainNonlinear trainer(settings);trainer.init(data,nbhs,&p);trainer.TrainBFGS();
    REQUIRE(std::abs(p.les_species_coeffs[0])+std::abs(p.les_species_coeffs[1])>1e-6);
    PotMTP_LES saved(schedule+".mtp");REQUIRE(saved.uses_sog());
    REQUIRE(saved.sog_channels==channels);
    for(int c=0;c<channels;++c)REQUIRE(std::abs(saved.les_species_coeffs[2*c])+std::abs(saved.les_species_coeffs[2*c+1])>1e-6);
    auto a=atoms(),b=atoms();p.getEFS(a);saved.getEFS(b);REQUIRE(a.energy_==Approx(b.energy_).margin(1e-12));
    const auto x=coeffs(p),y=coeffs(saved);
    for(size_t i=0;i<x.size();++i)REQUIRE(x[i]==Approx(y[i]).margin(1e-12));
    std::filesystem::remove(schedule+".mtp");std::filesystem::remove(schedule+".mtp.sr.snapshot");std::filesystem::remove(schedule+".mtp.sog.snapshot");
}
