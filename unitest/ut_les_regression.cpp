// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtp_m.h"

#include <fstream>
#include <sstream>

using namespace Potentials;
using Catch::Approx;

namespace {
Atoms les_atoms() {
    Atoms atom;
    atom.cell_ << 10.,0,0, .8,10.5,0, -.3,.6,11.;
    atom.pos_={{2.,2.,2.},{4.1,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
    atom.chemmap_={0,1,0,1};
    return atom;
}

void init_les(PotMTP_LES& pot) {
    pot.les_species_coeffs[0]=.23;
    pot.les_species_coeffs[1]=-.17;
    std::fill_n(pot.les_moment_coeffs,pot.get_alpha_scalar_moments(),0.);
    pot.les_moment_coeffs[0]=.35;
    pot.les_moment_coeffs[1]=-.21;
    pot.les_moment_coeffs[3]=.12;
    pot.train_init();
}

std::string saved_les() {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    init_les(pot);
    std::ostringstream out;
    pot.Save(out);
    return out.str();
}

using MTPM = PotMTP_M<m_model_mtpq2>;
void init_m_les(MTPM& pot) {
    // Reuse the fixture's radial basis and moment graph, with an MTP-M head.
    std::ifstream input("tests/case-mtp-les/les.mtp");
    REQUIRE(input.good());
    std::string source((std::istreambuf_iterator<char>(input)),{});
    source.resize(source.find("species_coeffs"));
    PotMTP reference("tests/case-mtp-les/les.mtp");
    const int n=reference.get_alpha_scalar_moments();
    source+="les_species_coeffs = {0.23, -0.17}\nles_moment_coeffs = {";
    for (int i=0;i<n;++i) {
        if (i) source+=", ";
        source+=std::to_string(i==0 ? .35 : i==1 ? -.21 : i==3 ? .12 : 0.);
    }
    source+="}\n";
    pot.lr_les=true;
    std::istringstream stream(source);
    pot.Load(stream);
    pot.model_coeffs[0]=.4;
    pot.model_coeffs[1]=.7;
    pot.model_coeffs[2]=.2;
    pot.model_coeffs[3]=.15;
    pot.model_coeffs[4]=.1;
    pot.train_init();
}
}

TEST_CASE("LES energy-only gradient uses the complete parameter layout", "[les][regression][grad]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    init_les(pot);
    const bool enabled=GENERATE(false,true);
    pot.lr_les=enabled;
    auto atom=les_atoms();
    Neighborhoods nbhs(atom,pot.cutoff);
    std::vector<double> coeffs(pot.get_NL_coeffs_count());
    pot.nl_get_coeffs(coeffs.data());
    std::vector<double> analytic(coeffs.size(),0.), full(coeffs.size(),0.);
    pot.AccumulateEFSCombinationGard(atom,nbhs,1.,analytic.data());
    Eigen::Matrix<double,-1,3> fw=Eigen::Matrix<double,-1,3>::Zero(atom.size(),3);
    Eigen::Matrix3d sw=Eigen::Matrix3d::Zero();
    Eigen::Matrix<double,-1,3,Eigen::RowMajor> unused;
    pot.AccumulateEFSCombinationGard(atom,nbhs,1.,fw,sw,unused,full.data());
    for (int i=0;i<static_cast<int>(coeffs.size());++i) {
        CAPTURE(enabled,i);
        REQUIRE(analytic[i]==Approx(full[i]).margin(1e-10));
        auto plus=coeffs,minus=coeffs;
        plus[i]+=1e-6;minus[i]-=1e-6;
        auto ap=les_atoms(),am=les_atoms();
        pot.nl_set_coeffs(plus.data());pot.getE(ap);
        pot.nl_set_coeffs(minus.data());pot.getE(am);
        REQUIRE(analytic[i]==Approx((ap.energy_-am.energy_)/2e-6).margin(2e-6));
    }
}



TEST_CASE("LES coefficient blocks require complete finite values and delimiters", "[les][regression][load]") {
    const std::string valid=saved_les();
    const std::string field=GENERATE(std::string("les_species_coeffs"),std::string("les_moment_coeffs"));
    const size_t start=valid.find(field);
    const size_t end=valid.find('}',start)+1;
    const std::string block=valid.substr(start,end-start);
    std::vector<std::string> malformed={
        field+" = {0.1,", field+" = {0.1}", field+" = {nan,}",
        field+" = {1e999,}", field+" {0.1,}",
        block.substr(0,block.size()-1),
        block.substr(0,block.size()-1)+"99, }", block+"\n"+block
    };
    for (const auto& replacement:malformed) {
        DYNAMIC_SECTION(field << ": malformed variant " << (&replacement-malformed.data())) {
            CAPTURE(replacement);
            auto text=valid;
            text.replace(start,end-start,replacement);
            PotMTP_LES pot;
            std::istringstream input(text);
            REQUIRE_THROWS_AS(pot.Load(input),GPException);
            REQUIRE_FALSE(pot.lr_les);
        }
    }
    for (bool trailing_comma:{false,true}) {
        auto text=valid;
        if (!trailing_comma) {
            size_t pos=0;
            while ((pos=text.find(", }",pos))!=std::string::npos)
                text.erase(pos,1);
        }
        PotMTP_LES pot;
        std::istringstream input(text);
        REQUIRE_NOTHROW(pot.Load(input));
        REQUIRE(pot.lr_les);
        auto atom=les_atoms();pot.getEFS(atom);
        REQUIRE(std::isfinite(atom.energy_));
    }
    PotMTP_LES pot;
    std::istringstream missing_scalar(valid+"\nles_sigma =");
    REQUIRE_THROWS_AS(pot.Load(missing_scalar),GPException);
}

TEST_CASE("LES reload clears activation coefficients and omitted model settings", "[les][regression][load]") {
    PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    init_les(pot);
    pot.les_cutoff_mode="reference_hard";
    pot.les_response_parity=-1;
    pot.remove_self_interaction=false;
    std::ifstream plain("tests/case-mtp-les/_Trained_Potential_.mtp");
    REQUIRE_NOTHROW(pot.Load(plain));
    REQUIRE_FALSE(pot.lr_les);
    REQUIRE(pot.les_response_parity==1);
    REQUIRE(pot.les_cutoff_mode=="smooth");
    REQUIRE(pot.remove_self_interaction);
    for (int i=0;i<pot.species_count;++i) REQUIRE(pot.les_species_coeffs[i]==0.);
    for (int i=0;i<pot.get_alpha_scalar_moments();++i) REQUIRE(pot.les_moment_coeffs[i]==0.);
    PotMTP reference("tests/case-mtp-les/_Trained_Potential_.mtp");
    auto actual=les_atoms(),expected=les_atoms();
    pot.getEFS(actual);reference.getEFS(expected);
    REQUIRE(actual.energy_==Approx(expected.energy_).margin(1e-12));
    const int old_count=pot.get_alpha_scalar_moments();
    std::ifstream larger("unitest/test-data/potential.mtp");
    REQUIRE_NOTHROW(pot.Load(larger));
    REQUIRE(pot.get_alpha_scalar_moments()>old_count);
    for (int i=0;i<pot.get_alpha_scalar_moments();++i) REQUIRE(pot.les_moment_coeffs[i]==0.);
    std::istringstream active(saved_les());
    REQUIRE_NOTHROW(pot.Load(active));
    REQUIRE(pot.lr_les);
    pot.getEFS(actual);
    REQUIRE(std::isfinite(actual.energy_));
}

TEST_CASE("MTP-M LES forces and virial differentiate the full energy", "[les][mtp_m][regression]") {
    MTPM pot;init_m_les(pot);
    auto atom=les_atoms();pot.getEFS(atom);
    const double delta=1e-5;
    for (int i=0;i<atom.size();++i) for (int d=0;d<3;++d) {
        auto plus=les_atoms(),minus=les_atoms();
        plus.pos_[i](d)+=delta;minus.pos_[i](d)-=delta;
        pot.getE(plus);pot.getE(minus);
        CAPTURE(i,d);
        REQUIRE(atom.forces_[i](d)==Approx(-(plus.energy_-minus.energy_)/(2*delta)).margin(2e-7));
    }
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        auto plus=les_atoms(),minus=les_atoms();
        Eigen::Matrix3d eps=Eigen::Matrix3d::Identity();eps(i,j)+=delta;
        plus.cell_=atom.cell_*eps.transpose();
        for (int k=0;k<atom.size();++k) plus.pos_[k]=atom.pos_[k]*eps.transpose();
        eps(i,j)-=2*delta;minus.cell_=atom.cell_*eps.transpose();
        for (int k=0;k<atom.size();++k) minus.pos_[k]=atom.pos_[k]*eps.transpose();
        pot.getE(plus);pot.getE(minus);
        CAPTURE(i,j);
        REQUIRE(atom.stress_(i,j)==Approx(-(plus.energy_-minus.energy_)/(2*delta)).margin(2e-7));
    }
    for (int mode=0;mode<5;++mode) {
        auto trial=les_atoms();
        if (mode==0) pot.getE(trial);
        if (mode==1) pot.getEOMP(trial);
        if (mode==2) pot.getEFSOMP(trial);
        if (mode==3) pot.getEFSOMPthread(trial);
        if (mode==4) {Neighborhoods nbhs(trial,pot.cutoff);pot.getEFS(trial,nbhs);}
        REQUIRE(trial.energy_==Approx(atom.energy_).margin(1e-12));
        double site_sum=0.;for (double e:trial.site_energies_) site_sum+=e;
        REQUIRE(site_sum==Approx(atom.energy_).margin(1e-12));
        if (mode>=2) for (int i=0;i<atom.size();++i)
            REQUIRE((trial.forces_[i]-atom.forces_[i]).norm()<1e-12);
    }
    std::vector<Atoms> parallel(4,les_atoms());
#pragma omp parallel for num_threads(2)
    for (int i=0;i<4;++i) pot.getEFSOMPthread(parallel[i]);
    for (const auto& trial:parallel) {
        REQUIRE(trial.energy_==Approx(atom.energy_).margin(1e-12));
        REQUIRE((trial.stress_-atom.stress_).norm()<1e-12);
    }
}



TEST_CASE("MTP-M LES full parameter gradients include the descriptor response", "[les][mtp_m][regression][grad]") {
    MTPM pot;init_m_les(pot);
    const int mode=GENERATE(0,1,2,3);
    if (mode==2) pot.req_les=false; // Frozen charge head still depends on fitted radial parameters.
    if (mode==3) {pot.req_radical=false;pot.req_species=false;pot.req_model=false;}
    auto atom=les_atoms();Neighborhoods nbhs(atom,pot.cutoff);
    Eigen::Matrix<double,-1,3> fw(atom.size(),3);
    fw<<.12,-.07,.03, -.04,.08,-.02, .06,.01,-.09, -.03,-.02,.08;
    Eigen::Matrix3d sw;
    sw<<.03,-.01,.005, -.01,-.02,.004, .005,.004,.01;
    if (mode==0) {fw.setZero();sw.setZero();}
    const double ew=.37;
    std::vector<double> coeffs(pot.get_NL_coeffs_count()),analytic(coeffs.size(),0.);
    pot.nl_get_coeffs(coeffs.data());
    Eigen::Matrix<double,-1,3,Eigen::RowMajor> unused;
    if (mode==0) pot.AccumulateEFSCombinationGard(atom,nbhs,ew,analytic.data());
    else pot.AccumulateEFSCombinationGard(atom,nbhs,ew,fw,sw,unused,analytic.data());
    const auto loss=[&](const std::vector<double>& x) {
        pot.nl_set_coeffs(x.data());auto trial=les_atoms();pot.getEFS(trial);
        double value=ew*trial.energy_+(sw.array()*trial.stress_.array()).sum();
        for (int i=0;i<trial.size();++i) value+=fw.row(i).dot(trial.forces_[i]);
        return value;
    };
    for (int i=0;i<static_cast<int>(coeffs.size());++i) {
        auto plus=coeffs,minus=coeffs;plus[i]+=1e-6;minus[i]-=1e-6;
        const double finite=(loss(plus)-loss(minus))/2e-6;
        CAPTURE(mode,i);
        REQUIRE(analytic[i]==Approx(finite).margin(3e-6));
    }
}

TEST_CASE("LES rejects removed fields in model files and settings", "[les][regression][errors]") {
    const std::string field = GENERATE("lr_k_cutoff", "les_output_scaling",
        "les_stress_fd_nudge", "enable_indirect_force", "remove_net_charge");
    auto message = Catch::Matchers::ContainsSubstring("Unsupported LES field: " + field);
    std::istringstream input(saved_les() + field + " = 1\n");
    PotMTP_LES model;
    REQUIRE_THROWS_WITH(model.Load(input), message);
    std::istringstream headed_input(saved_les() + field + " = 1\n");
    MTPM headed_model;
    REQUIRE_THROWS_WITH(headed_model.Load(headed_input), message);
    GPSettings settings;
    settings.Modify(field, "1");
    REQUIRE_THROWS_WITH(PotMTP_LES("tests/case-mtp-les/les.mtp", settings), message);
    REQUIRE_THROWS_WITH(MTPM("tests/case-mtp-les/les.mtp", settings), message);
}
