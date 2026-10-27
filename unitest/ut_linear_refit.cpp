// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "train/linear_refit.h"
#include "potentials/potential_mtp_LES.h"
#include <sstream>

using namespace Potentials;
using Catch::Approx;
namespace {
Atoms frame(int k) {
    Atoms a; a.id=k; a.fitness=.5+.1*k;
    a.cell_<<10.,0,0,.8,10.5,0,-.3,.6,11.;
    a.pos_={{2.,2.,2.},{4.1+.03*k,2.2,2.},{2.3,4.2+.02*k,2.4},{5.,4.8,3.1+.04*k}};
    a.chemmap_={0,1,0,1}; a.atomic_fitness={0.,.3,2.,4.};return a;
}
void configure(PotMTP_LES& p,int channels=1) {
    p.les_kernel="sog";p.sog_terms=2;p.sog_channels=channels;p.train_init();
    for(int c=0;c<channels;++c) {
        p.les_species_coeffs[2*c]=.23+.03*c;p.les_species_coeffs[2*c+1]=-.17;
        std::fill_n(p.les_moment_coeffs+c*p.alpha_scalar_moments,p.alpha_scalar_moments,0.);
        p.les_moment_coeffs[c*p.alpha_scalar_moments]=.35;
        p.les_moment_coeffs[c*p.alpha_scalar_moments+1]=-.21;
        p.model_les.sog_weights[2*c]=4.2;p.model_les.sog_weights[2*c+1]=-1.3;
        p.model_les.sog_log_bandwidths[2*c]=std::log(.7+.1*c);
        p.model_les.sog_log_bandwidths[2*c+1]=std::log(1.2+.1*c);
    }
    p.train_init();p.set_inited();
}
Eigen::VectorXd efs(const Atoms& a) {
    Eigen::VectorXd out(1+3*a.size()+9);out[0]=a.energy_;
    for(int i=0;i<a.size();++i) for(int d=0;d<3;++d) out[1+3*i+d]=a.forces_[i][d];
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) out[1+3*a.size()+3*i+j]=a.stress_(i,j);
    return out;
}
}

TEST_CASE("SR refit preserves nonlinear and composition-null parameters", "[refit]") {
    const int model=GENERATE(0,1,2); // ordinary MTP, SOG, Coulomb LES
    std::unique_ptr<PotMTP> p;
    if(model) {
        auto s=std::make_unique<PotMTP_LES>("tests/case-mtp-les/les.mtp");
        if(model==1) configure(*s,3);
        else {
            s->lr_les=true;
            s->les_species_coeffs[0]=.23;s->les_species_coeffs[1]=-.17;
            std::fill_n(s->les_moment_coeffs,s->alpha_scalar_moments,0.);
            s->les_moment_coeffs[0]=.35;s->les_moment_coeffs[1]=-.21;
            s->train_init();s->set_inited();
        }
        p=std::move(s);
    }
    else {p=std::make_unique<PotMTP>("tests/case-mtp-les/les.mtp");p->train_init();}
    std::vector<Atoms> data;std::vector<Neighborhoods> nbhs;
    for(int k=0;k<12;++k) {auto a=frame(k);p->getEFS(a);data.push_back(a);nbhs.emplace_back(data.back(),p->cutoff);}
    for(int i=0;i<p->get_lin_coeff_size();++i) p->linear_coeffs[i]+=.03*std::cos(i);
    const double bias_difference=p->linear_coeffs[0]-p->linear_coeffs[1];
    const std::vector<double> radial(p->radial_coeffs,p->radial_coeffs+p->radial_coeff_size);
    Eigen::VectorXd before(p->get_NL_coeffs_count()),after(before.size());
    p->nl_get_coeffs(before.data());
    LinearRefitSettings settings;settings.mode="sr";settings.report="";
    EFSFitWeights w;w.norm_f=2.;
    auto result=linear_refit(*p,data,nbhs,w,0.,settings);
    REQUIRE(result.after<result.before*1e-5);
    REQUIRE(result.data_after<1e-12);
    REQUIRE(result.composition_rank==1);
    REQUIRE(p->linear_coeffs[0]-p->linear_coeffs[1]==Approx(bias_difference).margin(1e-12));
    REQUIRE(std::equal(radial.begin(),radial.end(),p->radial_coeffs));
    p->nl_get_coeffs(after.data());
    // The LES head is frozen, hence its energy, forces, and BEC stay fixed.
    REQUIRE((before.tail(before.size()-p->reg_coeff_size).array()
        ==after.tail(after.size()-p->reg_coeff_size).array()).all());
    if(model==2) {
        settings.mode="sr_sog";
        REQUIRE_THROWS(linear_refit(*p,data,nbhs,w,0.,settings));
    }
}

TEST_CASE("Refit accounts for L2 and rejects invalid inputs without changing coefficients", "[refit]") {
    PotMTP p("tests/case-mtp-les/les.mtp");p.train_init();
    auto a=frame(0);p.getEFS(a);
    std::vector<Atoms> data={a};std::vector<Neighborhoods> nbhs;nbhs.emplace_back(data[0],p.cutoff);
    LinearRefitSettings settings;settings.mode="sr";settings.report="";
    EFSFitWeights w;
    p.reg_L2_vector[p.species_count]=.2;
    auto result=linear_refit(p,data,nbhs,w,.01,settings);
    REQUIRE(result.after<=result.before+1e-10);
    REQUIRE(result.predicted==Approx(result.after).margin(1e-9));
    Eigen::VectorXd before(p.get_NL_coeffs_count()),after(before.size());p.nl_get_coeffs(before.data());
    SECTION("L1") {p.reg_L1_vector[p.species_count]=.1;REQUIRE_THROWS(linear_refit(p,data,nbhs,w,.01,settings));}
    SECTION("negative weight") {data[0].atomic_fitness[1]=-1.;REQUIRE_THROWS(linear_refit(p,data,nbhs,w,.01,settings));}
    SECTION("no active labels") {w.energy=w.force=w.stress=0.;REQUIRE_THROWS(linear_refit(p,data,nbhs,w,.01,settings));}
    SECTION("report failure rolls back") {settings.report="/missing-refit-directory/report.json";REQUIRE_THROWS(linear_refit(p,data,nbhs,w,.01,settings));}
    p.nl_get_coeffs(after.data());REQUIRE((before.array()==after.array()).all());
}

TEST_CASE("SOG refit columns match independent unit-weight complete EFS", "[refit][sog]") {
    const int channels=GENERATE(1,3);
    const bool zero=GENERATE(false,true),self=GENERATE(false,true);
    PotMTP_LES p("tests/case-mtp-les/les.mtp");configure(p,channels);
    p.remove_self_interaction=self;p.model_les.sog_include_zero=zero;p.train_init();
    auto a=frame(3);Neighborhoods nbhs(a,p.cutoff);
    const auto basis=p.linear_sog_components(a,nbhs);
    std::ostringstream saved;p.Save(saved);
    PotMTP_LES reference;std::istringstream input(saved.str());reference.Load(input);
    Eigen::VectorXd weights=Eigen::VectorXd::Zero(p.kernel_count());
    reference.set_sog_linear_weights(weights);auto offset=a;reference.getEFS(offset,nbhs);
    for(int m=0;m<p.kernel_count();++m) {
        CAPTURE(channels,zero,self,m);
        weights.setZero();weights[m]=1.;reference.set_sog_linear_weights(weights);
        auto prediction=a;reference.getEFS(prediction,nbhs);
        REQUIRE((efs(prediction)-efs(offset)-basis.col(m)).norm()<1e-9);
    }
    weights=Eigen::Map<Eigen::VectorXd>(p.model_les.sog_weights.data(),p.kernel_count());
    auto prediction=a;p.getEFS(prediction,nbhs);
    REQUIRE((efs(prediction)-efs(offset)-basis*weights).norm()<1e-9);
}

TEST_CASE("Joint SOG refit reduces the regularized objective with fixed heads and widths", "[refit][sog]") {
    const int channels=GENERATE(1,3);
    PotMTP_LES p("tests/case-mtp-les/les.mtp");configure(p,channels);
    // Include exactly redundant columns and zero/negative current weights.
    p.model_les.sog_log_bandwidths[1]=p.model_les.sog_log_bandwidths[0];
    std::vector<Atoms> data;std::vector<Neighborhoods> nbhs;
    for(int k=0;k<10;++k) {auto a=frame(k);p.getEFS(a);data.push_back(a);nbhs.emplace_back(data.back(),p.cutoff);}
    p.model_les.sog_weights[0]=0.;p.model_les.sog_weights[1]=-2.;
    for(int i=0;i<p.get_lin_coeff_size();++i) p.linear_coeffs[i]+=.01*std::sin(i+1);
    LinearRefitSettings settings;settings.mode="sr_sog";settings.report="";
    auto result=linear_refit(p,data,nbhs,EFSFitWeights{},1e-8,settings);
    REQUIRE(result.after<result.before);
    REQUIRE(result.predicted==Approx(result.after).margin(1e-9));
    REQUIRE(result.parameters==p.get_lin_coeff_size()+p.kernel_count());
    REQUIRE(result.data_rank<result.parameters);
    auto a=frame(4);p.getEFS(a);constexpr double h=1e-5;
    for(int d=0;d<3;++d) {
        auto plus=frame(4),minus=frame(4);plus.pos_[1][d]+=h;minus.pos_[1][d]-=h;
        p.getE(plus);p.getE(minus);
        REQUIRE(a.forces_[1][d]==Approx(-(plus.energy_-minus.energy_)/(2*h)).margin(3e-7));
    }
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        auto plus=frame(4),minus=frame(4);Eigen::Matrix3d d=Eigen::Matrix3d::Zero();d(i,j)=h;
        plus.cell_=a.cell_*(Eigen::Matrix3d::Identity()+d).transpose();
        minus.cell_=a.cell_*(Eigen::Matrix3d::Identity()-d).transpose();
        for(int k=0;k<a.size();++k) {
            plus.pos_[k]=a.pos_[k]*(Eigen::Matrix3d::Identity()+d).transpose();
            minus.pos_[k]=a.pos_[k]*(Eigen::Matrix3d::Identity()-d).transpose();
        }
        p.getE(plus);p.getE(minus);
        REQUIRE(a.stress_(i,j)==Approx(-(plus.energy_-minus.energy_)/(2*h)).margin(5e-7));
    }
}

TEST_CASE("Unobserved SOG weights follow L2 while force-only biases stay fixed", "[refit][sog]") {
    PotMTP_LES p("tests/case-mtp-les/les.mtp");configure(p);
    std::fill_n(p.les_species_coeffs,p.latent_species_count(),0.);
    std::fill_n(p.les_moment_coeffs,p.latent_moment_count(),0.);
    auto a=frame(0);p.getEFS(a);
    const double b0=p.linear_coeffs[0],b1=p.linear_coeffs[1];
    std::vector<Atoms> data={a};std::vector<Neighborhoods> nbhs;nbhs.emplace_back(data[0],p.cutoff);
    EFSFitWeights w;w.energy=0.;w.stress=0.;
    LinearRefitSettings settings;settings.mode="sr_sog";settings.report="";
    linear_refit(p,data,nbhs,w,.01,settings);
    REQUIRE(std::abs(p.model_les.sog_weights[0])+std::abs(p.model_les.sog_weights[1])<1e-12);
    REQUIRE(p.linear_coeffs[0]==b0);REQUIRE(p.linear_coeffs[1]==b1);
}

TEST_CASE("Refit uses only available labels", "[refit]") {
    const int label=GENERATE(0,1,2);
    PotMTP p("tests/case-mtp-les/les.mtp");p.train_init();
    std::vector<Atoms> data;std::vector<Neighborhoods> nbhs;
    for(int k=0;k<8;++k) {
        auto a=frame(k);p.getEFS(a);
        a.have_energy=label==0;a.have_stress=label==2;
        if(label!=1) a.forces_.clear();
        data.push_back(a);nbhs.emplace_back(data.back(),p.cutoff);
    }
    p.moment_coeffs[0]+=.2;
    LinearRefitSettings settings;settings.mode="sr";settings.report="";
    auto result=linear_refit(p,data,nbhs,EFSFitWeights{},0.,settings);
    REQUIRE(result.data_after<result.data_before*1e-6+1e-12);
}

TEST_CASE("Derived MTP heads must explicitly support conditional linear fitting", "[refit]") {
    struct OtherHead:PotMTP {using PotMTP::PotMTP;};
    OtherHead p("tests/case-mtp-les/les.mtp");p.train_init();
    std::vector<Atoms> data={frame(0)};std::vector<Neighborhoods> nbhs;nbhs.emplace_back(data[0],p.cutoff);
    LinearRefitSettings settings;settings.mode="sr";settings.report="";
    REQUIRE_THROWS(linear_refit(p,data,nbhs,EFSFitWeights{},0.,settings));
}
