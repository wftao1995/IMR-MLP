#include <catch2/catch_all.hpp>
#include "apps/mtp_workflow.h"
#include "apps/mtp_tree_workflow.h"
#include "gpm_fixture.h"
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
    "hybrid","hybrid_scale","hmtp","descriptor_linear");
    MTPWorkflow::Stage stage;auto file=[&](const char *name){return (stage.path/name).string();};
    PotMTP source;load_swap_fixture(source,2,true);source.set_element_map({{0,"Al"},{1,"Ni"}});source.Save(file("source.mtp"));
    GPSettings config;config.Modify("pot:pottype",kind);
    config.Modify("pot:extmodel","mtp_m_n2");config.Modify("pot:n_input","2");config.Modify("pot:n_neuron","3");config.Modify("pot:n_neuron2","2");
    config.Modify("pot:nnlay","2");config.Modify("pot:nnhidden","3");config.Modify("pot:nnhidden2","2");
    if(std::string(kind)=="mtp_m"||std::string(kind)=="mtp_les") {
        config.Modify("pot:lr_les","true");config.Modify("pot:les_kernel","sog");config.Modify("pot:sog_channels","2");config.Modify("pot:sog_weights","1 2 1.5 .5");config.Modify("pot:sog_log_bandwidths","-.5 .2 -.2 .3");
    }
    config.Modify("pot:npot","2");config.Modify("pot:feature","min_distance");config.Modify("pot:boundary1","1 4");
    for(int i=1;i<=2;++i){auto prefix="pot:pot"+std::to_string(i)+":";config.Modify(prefix+"pottype","mtp");config.Modify(prefix+"potfile",file("source.mtp"));}
    MTPWorkflow::initialize_model(config,file("source.mtp"),file("tree.gpm"),"",{},false,true,.01,.02,"","","",false,false,1.,false,"");
    auto bundle=ModelBundle::load(file("tree.gpm"));auto &model=bundle->potential();
    model.reg_L1_vector.assign(model.get_NL_coeffs_count(),.3);model.reg_L2_vector.assign(model.get_NL_coeffs_count(),.4);
    auto atoms=swap_lattice(2,2),expected=atoms;model.getEFS(expected);
    MTPWorkflow::Adjustment op;op.appended={"Cu"};op.sources={1};op.initialization="clone";op.order={2,0,1};op.basis="level-group";
    auto transformed=MTPWorkflow::adjust_tree(model,op);
    for(int &id:atoms.chemmap_)id=id==0?1:2;
    compare(*transformed,atoms,expected);
    REQUIRE(transformed->gpm_elements==std::map<int,std::string>{{0,"Cu"},{1,"Al"},{2,"Ni"}});
    REQUIRE(transformed->reg_L1_vector.size()==size_t(transformed->get_NL_coeffs_count()));
    for(double penalty:transformed->reg_L1_vector)REQUIRE((penalty==0.||penalty==.3));
    MTPWorkflow::save(*transformed,file("adjusted.gpm"));auto reloaded=ModelBundle::load(file("adjusted.gpm"));compare(reloaded->potential(),atoms,expected);
}

TEST_CASE("Grouped basis sorting preserves interleaved accumulation targets", "[mtp][workflow][tree]") {
    PotMTP source;load_swap_fixture(source,2,false);
    source.set_element_map({{0,"Al"},{1,"Ni"}});
    source.alpha_index_times << 0,0,1,4, 2,2,1,4, 1,1,1,5, 3,3,1,5;
    auto atoms=swap_lattice(2,2),expected=atoms;source.getEFS(expected);
    MTPWorkflow::Adjustment op;op.basis="level-group";
    auto adjusted=MTPWorkflow::adjust_tree(source,op);
    MTPWorkflow::validate_graph(dynamic_cast<PotMTP&>(*adjusted));
    compare(*adjusted,atoms,expected);
    MTPWorkflow::Stage stage;const auto output=(stage.path/"grouped.gpm").string();
    MTPWorkflow::save(*adjusted,output);auto reloaded=ModelBundle::load(output);
    compare(reloaded->potential(),atoms,expected);
}
