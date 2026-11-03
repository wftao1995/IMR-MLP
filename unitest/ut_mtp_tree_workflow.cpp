#include <catch2/catch_all.hpp>
#include "apps/mtp_workflow.h"
#include "apps/mtp_tree_workflow.h"
#include "gpm_fixture.h"
#if __has_include("potentials/potential_mtp_edqrd.h")
#include "potentials/potential_mtp_edqrd.h"
#define MTP_WORKFLOW_EDQ
#endif
using namespace Potentials;
namespace {
void compare(AnyPotential &model,Atoms atoms,const Atoms &expected) {
    model.getEFS(atoms);
    REQUIRE(atoms.energy_==Catch::Approx(expected.energy_).margin(1e-8));
    REQUIRE((atoms.stress_-expected.stress_).norm()<1e-7);
    for(size_t i=0;i<atoms.size();++i)REQUIRE((atoms.forces_[i]-expected.forces_[i]).norm()<1e-8);
}
}
TEST_CASE("MTP tree adapters migrate split bases and full heads", "[mtp][workflow][tree]") {
    const auto kind=GENERATE("mtp_les","mtp_m","mtpnn1",
#ifdef MTP_WORKFLOW_EDQ
    "mtp_edq","mtp_edqrd",
#endif
    "hybrid","hybrid_scale","hmtp","descriptor_linear");
    MTPWorkflow::Stage stage;auto file=[&](const char *name){return (stage.path/name).string();};
    PotMTP source;load_swap_fixture(source,2,true);source.set_element_map({{0,"Al"},{1,"Ni"}});source.Save(file("source.mtp"));
    GPSettings config;config.Modify("pot:pottype",kind);
    config.Modify("pot:extmodel","mtp_m_n2");config.Modify("pot:n_input","2");config.Modify("pot:n_neuron","3");config.Modify("pot:n_neuron2","2");
    config.Modify("pot:nnlay","2");config.Modify("pot:nnhidden","3");config.Modify("pot:nnhidden2","2");
    if(std::string(kind)=="mtp_m"||std::string(kind)=="mtp_les") {
        config.Modify("pot:lr_les","true");config.Modify("pot:les_kernel","sog");config.Modify("pot:sog_channels","2");config.Modify("pot:sog_weights","1 2 1.5 .5");config.Modify("pot:sog_log_bandwidths","-.5 .2 -.2 .3");
    }
    config.Modify("pot:sr_potfile",file("source.mtp"));config.Modify("pot:charge_potfile",file("source.mtp"));
    config.Modify("pot:npot","2");config.Modify("pot:feature","min_distance");config.Modify("pot:boundary1","1 4");
    for(int i=1;i<=2;++i){auto prefix="pot:pot"+std::to_string(i)+":";config.Modify(prefix+"pottype","mtp");config.Modify(prefix+"potfile",file("source.mtp"));}
    MTPWorkflow::initialize_model(config,file("source.mtp"),file("tree.gpm"),"",{},false,true,.01,.02,"","","",false,false,1.,false,"");
    auto bundle=ModelBundle::load(file("tree.gpm"));auto &model=bundle->potential();
    if(std::string(kind)=="mtp_edqrd") {std::vector<double> coeff(model.get_NL_coeffs_count());model.nl_get_coeffs(coeff.data());coeff.back()=.7;model.nl_set_coeffs(coeff.data());}
    model.reg_L1_vector.assign(model.get_NL_coeffs_count(),.3);model.reg_L2_vector.assign(model.get_NL_coeffs_count(),.4);
    if(std::string(kind)=="mtp_edqrd"){model.reg_L1_vector.back()=0.;model.reg_L2_vector.back()=0.;}
    auto atoms=swap_lattice(2,2),expected=atoms;model.getEFS(expected);
    MTPWorkflow::Adjustment op;op.appended={"Cu"};op.sources={1};op.initialization="clone";op.order={2,0,1};op.basis="level-group";
    auto transformed=MTPWorkflow::adjust_tree(model,op);
    for(int &id:atoms.chemmap_)id=id==0?1:2;
    compare(*transformed,atoms,expected);
    REQUIRE(transformed->gpm_elements==std::map<int,std::string>{{0,"Cu"},{1,"Al"},{2,"Ni"}});
    REQUIRE(transformed->reg_L1_vector.size()==size_t(transformed->get_NL_coeffs_count()));
    for(double penalty:transformed->reg_L1_vector)REQUIRE((penalty==0.||penalty==.3));
    MTPWorkflow::save(*transformed,file("adjusted.gpm"));auto reloaded=ModelBundle::load(file("adjusted.gpm"));compare(reloaded->potential(),atoms,expected);
#ifdef MTP_WORKFLOW_EDQ
    if(std::string(kind)=="mtp_edqrd") {
        auto &edq=dynamic_cast<PotMTP_EDQRd&>(*transformed);std::vector<double> coeff(edq.get_NL_coeffs_count());edq.nl_get_coeffs(coeff.data());
        REQUIRE(coeff.back()==Catch::Approx(.7));REQUIRE(coeff[coeff.size()-2]==Catch::Approx(0.));

    }
#endif
}

#ifdef MTP_WORKFLOW_EDQ
TEST_CASE("EDQRd refuses inseparable penalties when changing reference gauge", "[mtp][workflow][tree]") {
    MTPWorkflow::Stage stage;auto file=[&](const char *name){return (stage.path/name).string();};
    PotMTP source;load_swap_fixture(source,3,false);source.set_element_map({{0,"Al"},{1,"Ni"},{2,"Cu"}});source.Save(file("source.mtp"));
    GPSettings config;config.Modify("pot:pottype","mtp_edqrd");config.Modify("pot:sr_potfile",file("source.mtp"));config.Modify("pot:charge_potfile",file("source.mtp"));
    MTPWorkflow::initialize_model(config,file("source.mtp"),file("tree.gpm"),"",{},false,true,.01,.02,"","","",false,false,1.,false,"");
    auto bundle=ModelBundle::load(file("tree.gpm"));auto &model=bundle->potential();
    model.reg_L1_vector.assign(model.get_NL_coeffs_count(),.3);auto before=ModelBundle::fingerprint(model);
    MTPWorkflow::Adjustment op;op.order={1,2,0};
    REQUIRE_THROWS_WITH(MTPWorkflow::adjust_tree(model,op),Catch::Matchers::ContainsSubstring("reference gauge"));
    REQUIRE(ModelBundle::fingerprint(model)==before);
}

#endif
