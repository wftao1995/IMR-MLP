// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp_m.h"
#include "potentials/potential_mtp_LES.h"
#include <fstream>
#include <filesystem>
#include <sstream>
using namespace Potentials;
using Catch::Approx;
using MTPM=PotMTP_M<m_model_mtpq2>;
namespace {
Atoms atoms() {
    Atoms a;a.cell_<<10.,0,0,.8,10.5,0,-.3,.6,11.;
    a.pos_={{2.,2.,2.},{4.1,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};a.chemmap_={0,1,0,1};return a;
}
std::string template_source(int channels=1) {
    std::ifstream f("tests/case-mtp-les/les.mtp");
    std::string source((std::istreambuf_iterator<char>(f)),{});source.resize(source.find("species_coeffs"));
    PotMTP ref("tests/case-mtp-les/les.mtp");int n=ref.get_alpha_scalar_moments();
    source+="les_species_coeffs = {";
    for(int c=0;c<channels;++c){if(c)source+=", ";source+="0.23, -0.17";}
    source+="}\nles_moment_coeffs = {";
    for(int i=0;i<channels*n;++i){if(i)source+=", ";source+=std::to_string(i%n==0?.35:i%n==1?-.21:i%n==3?.12:0.);}
    source+="}\n";
    return source;
}
void initialize(MTPM& p) {
    auto source=template_source();
    SOGModel sog;sog.weights={4.2,-1.3};sog.log_bandwidths={std::log(.7),std::log(1.2)};
    std::ostringstream metadata;metadata<<std::setprecision(17);sog.write(metadata);source+=metadata.str();
    std::istringstream in(source);p.Load(in);
    p.model_coeffs[0]=.4;p.model_coeffs[1]=.7;p.model_coeffs[2]=.2;p.model_coeffs[3]=.15;p.model_coeffs[4]=.1;
    p.train_init();
    std::vector<double> x(p.get_NL_coeffs_count());p.nl_get_coeffs(x.data());
    for(const auto& g:p.nl_parameter_groups())if(g.name=="sr_species" || g.name=="sr_moment")
        for(int i=g.begin;i<g.end;++i)x[i]=.01*std::sin(i+1.);
    p.nl_set_coeffs(x.data());
}
}
TEST_CASE("Explicit MTP-M SOG definitions survive loading coefficient templates","[sog][mtp_m][load]") {
    const int channels=GENERATE(1,3);
    SOGModel definition;definition.channels=channels;definition.dl=1.7;definition.taper=.2;
    definition.include_zero=false;definition.onsite=false;
    for(int c=0;c<channels;++c){
        definition.weights.push_back(4.2+c);definition.weights.push_back(-1.3-c);
        definition.log_bandwidths.push_back(-.4);definition.log_bandwidths.push_back(.2);
    }
    MTPM explicit_model;
    explicit_model.set_sog_definition(definition);
    std::istringstream input(template_source(channels));
    REQUIRE_NOTHROW(explicit_model.Load(input));
    REQUIRE(explicit_model.uses_sog());
    REQUIRE(explicit_model.les_channels()==channels);
    REQUIRE(explicit_model.req_les);
    std::ostringstream metadata;metadata<<std::setprecision(17);definition.write(metadata);
    MTPM file_model;
    std::istringstream file_input(template_source(channels)+metadata.str());file_model.Load(file_input);
    auto actual=atoms(),expected=atoms();
    explicit_model.getEFS(actual);file_model.getEFS(expected);
    REQUIRE(actual.energy_==Approx(expected.energy_).margin(1e-12));
    for(int i=0;i<actual.size();++i)REQUIRE((actual.forces_[i]-expected.forces_[i]).norm()<1e-11);
    REQUIRE((actual.stress_-expected.stress_).norm()<1e-11);
    auto symmetric=atoms();PotMTP& api=explicit_model;api.getEFSSymmOMP(symmetric);
    REQUIRE(symmetric.energy_==Approx(expected.energy_).margin(1e-12));
    for(int i=0;i<actual.size();++i)REQUIRE((symmetric.forces_[i]-expected.forces_[i]).norm()<1e-11);
    REQUIRE((symmetric.stress_-expected.stress_).norm()<1e-11);

    // Saved definitions are self-contained; explicit configuration is only needed for a template.
    std::ostringstream saved;explicit_model.Save(saved);
    MTPM restored;std::istringstream saved_input(saved.str());restored.Load(saved_input);
    REQUIRE(restored.sog_definition().weights==definition.weights);
    REQUIRE(restored.sog_definition().log_bandwidths==definition.log_bandwidths);

    // A file definition takes precedence over a pending explicit definition.
    MTPM overridden;overridden.set_sog_definition(definition);
    initialize(overridden);
    REQUIRE(overridden.les_channels()==1);
    REQUIRE(overridden.sog_definition().include_zero);

    // A later metadata-free model must not inherit the previously loaded SOG kernel.
    auto plain_source=template_source();plain_source.resize(plain_source.find("les_species_coeffs"));
    std::istringstream plain(plain_source);explicit_model.Load(plain);
    REQUIRE_FALSE(explicit_model.lr_les);
    REQUIRE_FALSE(explicit_model.uses_sog());
    REQUIRE_NOTHROW(explicit_model.getEFS(actual));
}
TEST_CASE("MTP-M SOG parameters include complete force response with frozen groups","[sog][mtp_m]") {
    MTPM p;initialize(p);
    const int freeze=GENERATE(0,1,2,3);
    if(freeze==1)p.req_les=false;
    if(freeze==2)p.req_sog=false;
    if(freeze==3){p.req_radical=false;p.req_species=false;p.req_moment=false;p.req_model=false;p.req_les=false;}
    p.train_init();
    auto a=atoms();Neighborhoods nbhs(a,p.cutoff);
    std::vector<double> x(p.get_NL_coeffs_count()),g(x.size(),0.);p.nl_get_coeffs(x.data());
    Eigen::Matrix<double,-1,3> fw(4,3);fw<<.2,-.1,.3,-.5,.7,.1,.3,.2,-.4,.1,.2,.4;
    Eigen::Matrix3d sw;sw<<.1,.2,-.1,.3,-.2,.5,-.1,.2,.7;
    Eigen::Matrix<double,-1,3,Eigen::RowMajor> unused;
    const bool energy_only=GENERATE(false,true);
    if (energy_only) {
        fw.setZero();sw.setZero();
        p.AccumulateEFSCombinationGard(a,nbhs,.37,g.data());
    } else p.AccumulateEFSCombinationGard(a,nbhs,.37,fw,sw,unused,g.data());
    auto loss=[&](){auto b=atoms();p.getEFS(b);double l=.37*b.energy_+(sw.array()*b.stress_.array()).sum();for(int i=0;i<b.size();++i)l+=fw.row(i).dot(b.forces_[i]);return l;};
    for(size_t i=0;i<x.size();++i){CAPTURE(freeze,i);auto plus=x,minus=x;plus[i]+=1e-6;minus[i]-=1e-6;p.nl_set_coeffs(plus.data());double ep=loss();p.nl_set_coeffs(minus.data());double em=loss();REQUIRE(g[i]==Approx((ep-em)/2e-6).margin(3e-6));}
    p.nl_set_coeffs(x.data());
}
TEST_CASE("MTP-M and MTP use identical SOG EFS and support model roundtrips","[sog][mtp_m]") {
    MTPM m;initialize(m);auto a=atoms();m.getEFS(a);REQUIRE(a.charges_.empty());
    m.lr_les=false;auto sr=atoms();m.getEFS(sr);m.lr_les=true;
    PotMTP_LES p("tests/case-mtp-les/les.mtp");p.les_kernel="sog";p.sog_terms=2;p.train_init();p.model_les=m.model_les;
    std::copy_n(m.les_species,2,p.les_species_coeffs);std::copy_n(m.les_moment,p.get_alpha_scalar_moments(),p.les_moment_coeffs);
    auto b=atoms();p.getEFS(b);p.lr_les=false;auto bsr=atoms();p.getEFS(bsr);
    REQUIRE(a.energy_-sr.energy_==Approx(b.energy_-bsr.energy_).margin(1e-11));
    for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-sr.forces_[i]-b.forces_[i]+bsr.forces_[i]).norm()<1e-10);
    REQUIRE((a.stress_-sr.stress_-b.stress_+bsr.stress_).norm()<1e-10);
    std::ostringstream out;m.Save(out);std::istringstream in(out.str());MTPM loaded;loaded.Load(in);loaded.train_init();auto c=atoms();loaded.getEFS(c);
    REQUIRE(loaded.uses_sog());REQUIRE(a.energy_==Approx(c.energy_).margin(1e-11));
    for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-c.forces_[i]).norm()<1e-10);
    REQUIRE((a.stress_-c.stress_).norm()<1e-10);
    std::vector<double> x(m.get_NL_coeffs_count()),y(loaded.get_NL_coeffs_count());
    m.nl_get_coeffs(x.data());loaded.nl_get_coeffs(y.data());
    REQUIRE(x.size()==y.size());
    for(size_t i=0;i<x.size();++i)REQUIRE(x[i]==Approx(y[i]).margin(1e-12));
}
TEST_CASE("Portable SOG metadata rejects invalid and conflicting fields","[sog][mtp_m][load]") {
    SOGModel p;p.weights={4.2,-1.3};p.log_bandwidths={-.4,.2};
    std::ostringstream out;out<<std::setprecision(17);p.write(out);
    const auto source=out.str();
    for(const auto& extra:{"les_version = 3\n","sog_channels = 2\n","sog_unknown = 1\n",
            "remove_self_interaction = 1\n","les_charge_treatment = uniform_background\n"}) {
        std::istringstream in(source+extra);REQUIRE_THROWS_AS(SOGModel::read(in),std::runtime_error);
    }
    for(const auto& replacement:{"sog_weights = {4.2}","sog_weights = {4.2, -1.3, 8}",
            "sog_weights = {nan, -1.3}","sog_weights = {4.2; -1.3}"}) {
        auto broken=source;const auto start=broken.find("sog_weights");
        broken.replace(start,broken.find('\n',start)-start,replacement);
        std::istringstream in(broken);REQUIRE_THROWS_AS(SOGModel::read(in),std::runtime_error);
    }
    MTPM m;initialize(m);std::ostringstream saved;m.Save(saved);
    auto broken=saved.str();const auto start=broken.find("les_species_coeffs");
    broken.replace(start,broken.find('\n',start)-start,"les_species_coeffs = {0.1}");
    MTPM invalid;std::istringstream in(broken);REQUIRE_THROWS_AS(invalid.Load(in),GPException);
}
TEST_CASE("MTP-M SOG file activation preserves requested trainable head","[sog][mtp_m][load]") {
    MTPM p;initialize(p);
    const auto file=std::filesystem::temp_directory_path()/"imrmlp-ut-sog-mtp-m.mtp";
    {std::ofstream out(file);p.Save(out);}
    MTPM loaded(file.string());
    REQUIRE(loaded.uses_sog());REQUIRE(loaded.req_les);
    REQUIRE(loaded.get_NL_coeffs_count()==p.get_NL_coeffs_count());
    GPSettings settings;settings.Modify("req_les","false");
    MTPM frozen(file.string(),settings);
    REQUIRE(frozen.uses_sog());REQUIRE_FALSE(frozen.req_les);
    settings.Modify("potfile",file.string());settings.Modify("req_les","true");
    MTPM configured(settings);
    REQUIRE(configured.uses_sog());REQUIRE(configured.req_les);
    std::filesystem::remove(file);
}

TEST_CASE("MTP-M Coulomb model files restore their definition and reject conflicting settings", "[les][mtp_m][load]") {
    MTPM original;
    original.lr_les = true;
    original.model_les.lr_sigma = .85;
    original.model_les.dl = 1.7;
    original.model_les.remove_self_interaction = false;
    std::istringstream input(template_source());
    original.Load(input);
    std::ostringstream saved;
    original.Save(saved);
    MTPM restored;
    std::istringstream serialized(saved.str());
    restored.Load(serialized);
    REQUIRE(restored.lr_les);
    REQUIRE_FALSE(restored.uses_sog());
    REQUIRE(restored.model_les.lr_sigma == Approx(.85));
    REQUIRE(restored.model_les.dl == Approx(1.7));
    REQUIRE_FALSE(restored.model_les.remove_self_interaction);
    auto expected = atoms(), actual = expected;
    original.getEFS(expected);
    restored.getEFS(actual);
    REQUIRE(actual.energy_ == Approx(expected.energy_).margin(1e-12));
    for (int i = 0; i < actual.size(); ++i)
        REQUIRE(actual.forces_[i].isApprox(expected.forces_[i], 1e-10));
    const auto file = std::filesystem::temp_directory_path() / "imrmlp-ut-coulomb-definition.mtp";
    { std::ofstream out(file); out << saved.str(); }
    GPSettings settings;
    settings.Modify("lr_sigma", ".85");
    REQUIRE_NOTHROW(MTPM(file.string(), settings));
    settings.Modify("lr_sigma", "1.2");
    REQUIRE_THROWS_AS(MTPM(file.string(), settings), GPException);
    std::filesystem::remove(file);
}
