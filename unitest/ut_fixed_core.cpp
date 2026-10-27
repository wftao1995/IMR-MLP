#include <catch2/catch_all.hpp>
#include "train/fixed_core.h"
#include "train/efs_fit_weights.h"
#include "potentials/potential_hybrid.h"
#include "apps/apps.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>

using namespace Potentials;
namespace {
Atoms geometry(int i=0) {
    Atoms a;a.cell_=12.*Eigen::Matrix3d::Identity();a.pbc_={false,false,false};
    a.pos_={{1.,1.,1.},{2.2+.04*i,1.3,1.2},{1.4,2.5-.02*i,1.6}};
    a.chemmap_={0,1,0};a.fitness=.8;return a;
}
std::unique_ptr<AnyPotential> core_model(bool mtp) {
    if(mtp)return std::make_unique<PotMTP>("unitest/test-data/portable-mtp/les.mtp");
    GPSettings s;s.Modify("z","13 28");s.Modify("r0","1");s.Modify("r1","4");
    return std::make_unique<PotentialZBL>(s);
}
void close_efs(const Atoms &a,const Atoms &b) {
    REQUIRE(a.energy_==Catch::Approx(b.energy_).margin(1e-9));
    REQUIRE((a.stress_-b.stress_).norm()<1e-8);
    for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-b.forces_[i]).norm()<1e-8);
}
struct Directory {
    std::filesystem::path original=std::filesystem::current_path(),path;
    Directory(){char name[]="/tmp/imrmlp-fixed-core-XXXXXX";path=mkdtemp(name);}
    ~Directory(){CMDOption::global_cmdopts.reset();std::filesystem::current_path(original);std::filesystem::remove_all(path);}
};
}

TEST_CASE("Fixed-core residual copies preserve labels, normalization and matching curvature operators", "[train][fixed-core]") {
    const bool mtp=GENERATE(false,true);auto core=core_model(mtp);
    PotMTP residual("unitest/test-data/portable-mtp/les.mtp");
    auto expected=geometry();residual.getEFS(expected);auto prediction=geometry();core->getEFS(prediction);
    auto reference=expected;reference.energy_+=prediction.energy_;reference.stress_+=prediction.stress_;
    for(int i=0;i<reference.size();++i){reference.forces_[i]+=prediction.forces_[i];reference.site_energies_[i]+=prediction.site_energies_[i];}
    reference.hvp_directions_=Eigen::MatrixXd::Random(9,2);reference.hvp_targets_=Eigen::MatrixXd::Zero(9,2);
    auto forces=[&](Atoms &a,Neighborhoods &)->Eigen::MatrixXd {
        residual.getEFS(a);auto b=a;core->getEFS(b);
        Eigen::MatrixXd f(9,1);for(int i=0;i<3;++i)f.block<3,1>(3*i,0)=(a.forces_[i]+b.forces_[i]).transpose();return f;
    };
    auto values=Training::finite_difference_hvp_observations(reference,1e-4,6.,forces);
    for(int k=0;k<2;++k)reference.hvp_targets_.col(k)=values.col(0).segment(9*k,9);
    auto full=reference;full.hvp_directions_=Eigen::MatrixXd::Identity(9,9);full.hvp_targets_=Eigen::MatrixXd::Zero(9,9);
    values=Training::finite_difference_hvp_observations(full,2e-4,6.,forces);
    reference.hessian_.resize(9,9);reference.have_hessian=true;
    for(int k=0;k<9;++k)reference.hessian_.col(k)=values.col(0).segment(9*k,9);
    auto missing=geometry();missing.clearEFS();
    const auto result=Training::fixed_core_residuals(*core,{reference,missing},1e-4,2e-4);
    close_efs(result[0],expected);
    for(int i=0;i<expected.size();++i) REQUIRE(result[0].site_energies_[i]==Catch::Approx(expected.site_energies_[i]).margin(1e-10));
    auto residual_forces=[&](Atoms &a,Neighborhoods &)->Eigen::MatrixXd {
        residual.getEFS(a);Eigen::MatrixXd f(9,1);for(int i=0;i<3;++i)f.block<3,1>(3*i,0)=a.forces_[i].transpose();return f;
    };
    auto probe=reference;values=Training::finite_difference_hvp_observations(probe,1e-4,6.,residual_forces);
    for(int k=0;k<2;++k)REQUIRE((result[0].hvp_targets_.col(k)-values.col(0).segment(9*k,9)).norm()<2e-8);
    values=Training::finite_difference_hvp_observations(full,2e-4,6.,residual_forces);
    for(int k=0;k<9;++k)REQUIRE((result[0].hessian_.col(k)-values.col(0).segment(9*k,9)).norm()<2e-8);
    EFSFitWeights weights;weights.norm_f=.4;
    REQUIRE(weights.normalization(result[0])==weights.normalization(reference));
    REQUIRE(reference.energy_==expected.energy_+prediction.energy_);
    REQUIRE(reference.global_attribs_d_.count("__imrmlp_fixed_core_force_norm2")==0);
    REQUIRE_FALSE(result[1].has_energy());REQUIRE_FALSE(result[1].has_forces());REQUIRE_FALSE(result[1].has_stresses());
    REQUIRE_FALSE(result[1].has_hvp());REQUIRE_FALSE(result[1].has_hessian());
    REQUIRE_FALSE(core->add_efs);
}

TEST_CASE("Fixed MTP core remains separate in a reloadable Hybrid manifest", "[fixed-core][hybrid]") {
    const bool mtp=GENERATE(false,true);auto core=core_model(mtp);PotMTP residual("unitest/test-data/portable-mtp/les.mtp");
    Directory directory;const auto destination=(directory.path/"combined").string();auto sample=geometry();
    auto a=sample,b=sample;residual.getEFS(a);core->getEFS(b);
    Training::save_fixed_core_hybrid(residual,*core,destination,&sample);
    GPSettings load;load.Modify("potfile",destination);PotentialHybrid combined(load);
    auto c=sample;combined.getEFS(c);a.energy_+=b.energy_;a.stress_+=b.stress_;
    for(int i=0;i<a.size();++i)a.forces_[i]+=b.forces_[i];close_efs(c,a);
    REQUIRE(combined.ntrains==1);REQUIRE(combined.train_ids==std::vector<int>{0});
    REQUIRE(combined.get_NL_coeffs_count()==residual.get_NL_coeffs_count());
    std::filesystem::create_directory(destination+".residual.mtp.blocked");
    const auto blocked=(directory.path/"blocked").string();std::filesystem::create_directory(blocked+".residual.mtp");
    REQUIRE_THROWS(Training::save_fixed_core_hybrid(residual,*core,blocked,&sample));
    REQUIRE_FALSE(std::filesystem::exists(blocked));
}

TEST_CASE("Real CLI fits copied residual labels and publishes the fixed-core combination", "[apps][fixed-core]") {
    PotMTP residual("unitest/test-data/portable-mtp/les.mtp");auto core=core_model(false);Directory directory;
    std::vector<Atoms> references;
    for(int k=0;k<3;++k){auto a=geometry(k),b=a;residual.getEFS(a);core->getEFS(b);
        a.energy_+=b.energy_;a.stress_+=b.stress_;for(int i=0;i<a.size();++i)a.forces_[i]+=b.forces_[i];references.push_back(a);}
    residual.species_coeffs[0]+=.2;residual.Save((directory.path/"base.mtp").string());core->Save((directory.path/"core").string());
    GPIO io;REQUIRE(io.write(references,(directory.path/"train.bin").string(),"bin"));
    std::filesystem::current_path(directory.path);std::ofstream("index.map")<<"Al\nNi\n";
    std::ofstream("settings")<<"pot:pottype mtp\npot:potfile base.mtp\nfit:save_to residual.mtp\nfit:bfgs_maxiteration 2\nfit:parameter_scaling none\nfit:penalty_weight 0\nfit:trainable_groups species\n";
    std::vector<std::string> args={"trainnlpot","--settings=settings","--do-lbfgs","--trainset=train.bin","--trainset-format=bin","--fixed-core=core","--output=complete"};
    std::vector<char*> argv;for(auto &s:args)argv.push_back(s.data());CMDOption::global_cmdopts.reset();
    REQUIRE(trainnlpot(argv.size(),argv.data())==0);
    PotMTP fitted("residual.mtp");GPSettings load;load.Modify("potfile","complete");PotentialHybrid full(load);
    auto a=geometry(),b=a,c=a;fitted.getEFS(a);core->getEFS(b);full.getEFS(c);
    a.energy_+=b.energy_;a.stress_+=b.stress_;for(int i=0;i<a.size();++i)a.forces_[i]+=b.forces_[i];close_efs(c,a);
}

TEST_CASE("Fixed core cannot alias its trainable residual", "[fixed-core][validation]") {
    PotMTP potential("unitest/test-data/portable-mtp/les.mtp");
    REQUIRE_THROWS_WITH(Training::validate_fixed_core(potential,potential),
        Catch::Matchers::ContainsSubstring("separate"));
}

TEST_CASE("Fixed-core save validation uses Cartesian work copies", "[fixed-core][hybrid]") {
    auto core=core_model(false);PotMTP residual("unitest/test-data/portable-mtp/les.mtp");
    Directory directory;auto sample=geometry();sample.to_fractional();
    const auto positions=sample.pos_;const auto destination=(directory.path/"fractional").string();
    Training::save_fixed_core_hybrid(residual,*core,destination,&sample);
    REQUIRE_FALSE(sample.cartesian);REQUIRE(sample.pos_==positions);
    GPSettings load;load.Modify("potfile",destination);PotentialHybrid full(load);
    auto a=sample,b=sample,c=sample;a.to_cartesian();b.to_cartesian();c.to_cartesian();
    residual.getEFS(a);core->getEFS(b);full.getEFS(c);
    a.energy_+=b.energy_;a.stress_+=b.stress_;
    for(int i=0;i<a.size();++i)a.forces_[i]+=b.forces_[i];close_efs(c,a);
    REQUIRE_THROWS(Training::save_fixed_core_hybrid(residual,*core,"",&sample));
}

TEST_CASE("An all-fixed Hybrid has no trainable species constants", "[fixed-core][hybrid]") {
    PotMTP source("unitest/test-data/portable-mtp/les.mtp");Directory directory;
    const auto path=(directory.path/"fixed.mtp").string();source.Save(path);
    GPSettings settings;settings.Modify("npot","1");settings.Modify("pot1:pottype","mtp");
    settings.Modify("pot1:potfile",path);settings.Modify("pot1:train","false");
    PotentialHybrid fixed(settings);REQUIRE(fixed.get_NL_coeffs_count()==0);
    REQUIRE(fixed.get_lin_coeff_size()==0);REQUIRE(fixed.nl_parameter_groups().empty());
    auto a=geometry(),b=a;source.getEFS(a);fixed.getEFS(b);close_efs(a,b);
    Neighborhoods neighbors(b,fixed.cutoff);std::vector<double> energy;
    Eigen::Tensor<double,3> force,stress;fixed.CalcEFSCmpt(b,neighbors,energy,force,stress);
    REQUIRE(energy.empty());REQUIRE(force.dimension(1)==0);REQUIRE(stress.dimension(0)==0);
    double sentinel=17.;fixed.AccumulateEFSCombinationGard(b,neighbors,2.,&sentinel);
    REQUIRE(sentinel==17.);
}
