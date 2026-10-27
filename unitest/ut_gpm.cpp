#include <catch2/catch_all.hpp>
#include "potentials/model_bundle.h"
#include "potentials/gpm_codec.h"
#include "potentials/potential_wrapper.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/potential_mtpnn1.h"
#include "potentials/potential_hybmtp.h"
#include "potentials/potential_hybrid.h"
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_eam_zhou_2004.h"
#include "potentials/potential_openkim.h"
#include "potentials/potential_shell.h"
#include "io/gpio.h"
#include "active_learn/al_potential.h"
#include "maxvol/active_selector.h"
#include "runtime/mtp/mlip.h"
#include "runtime/mtp/cpu/mtp_cpu.h"
#include "gpm_fixture.h"
#include "LAMMPS/USER-MLIP/standalone/tests/radial_fixture.h"
#include "potentials/native_inference.h"
#include <filesystem>
#include <spawn.h>
#include <sys/wait.h>

using namespace Potentials;
namespace {
struct Directory {
    std::filesystem::path path;
    Directory(){std::string name=(std::filesystem::temp_directory_path()/"gpml-test-gpm-XXXXXX").string();REQUIRE(mkdtemp(name.data()));path=name;}
    ~Directory(){std::error_code ec;std::filesystem::remove_all(path,ec);}
    std::string file(std::string name="model.gpm")const{return (path/name).string();}
};
void validate_fresh(const std::string &file) {
    const auto executable=std::filesystem::read_symlink("/proc/self/exe").string();
    std::string assignment="IMRMLP_GPM_CHILD_MODEL="+file;
    std::vector<char*> environment;for(char **p=environ;*p;++p)if(std::string(*p).rfind("IMRMLP_GPM_CHILD_MODEL=",0)!=0)environment.push_back(*p);
    environment.push_back(assignment.data());environment.push_back(nullptr);
    char *arguments[]={const_cast<char*>(executable.c_str()),const_cast<char*>("GPM child validation"),const_cast<char*>("--reporter"),const_cast<char*>("compact"),nullptr};
    pid_t pid;REQUIRE(posix_spawn(&pid,executable.c_str(),nullptr,nullptr,arguments,environment.data())==0);
    int status;REQUIRE(waitpid(pid,&status,0)==pid);REQUIRE(WIFEXITED(status));REQUIRE(WEXITSTATUS(status)==0);
}
void equivalent(AnyPotential &a,AnyPotential&b,Atoms atoms=swap_lattice(2,2)) {
    auto other=atoms;a.getEFS(atoms);b.getEFS(other);
    REQUIRE(other.energy_==Catch::Approx(atoms.energy_).margin(1e-11));
    REQUIRE((other.stress_-atoms.stress_).norm()<1e-9*(1+atoms.stress_.norm()));
    REQUIRE(other.forces_.size()==atoms.forces_.size());
    for(size_t i=0;i<atoms.forces_.size();++i)REQUIRE((other.forces_[i]-atoms.forces_[i]).norm()<1e-9*(1+atoms.forces_[i].norm()));
}
void roundtrip(AnyPotential&p,Atoms atoms=swap_lattice(2,2)) {
    Directory dir;p.Save(dir.file());auto b=ModelBundle::load(dir.file());
    equivalent(p,b->potential(),atoms);ModelBundle::validate(dir.file());validate_fresh(dir.file());
    REQUIRE(ModelBundle::fingerprint(p)==ModelBundle::fingerprint(b->potential()));
    const int count=p.get_NL_coeffs_count();REQUIRE(count==b->potential().get_NL_coeffs_count());
    if(count>0){std::vector<double> before(count),after(count);p.nl_get_coeffs(before.data());b->potential().nl_get_coeffs(after.data());REQUIRE(before==after);}
    if(p.supports_analytical_hessian_vector_product()){
        Eigen::VectorXd direction=Eigen::VectorXd::LinSpaced(3*atoms.size(),-.2,.3),before,after;
        p.applyHessianVectorProduct(atoms,direction,before);b->potential().applyHessianVectorProduct(atoms,direction,after);REQUIRE((before-after).norm()<1e-9*(1+before.norm()));
    }

}

void standalone_equivalent(PotMTP &p,const std::string &path) {
    int mapping[]={-1,0,1},types[]={1,2,1};double cutoff;
    MLIPCore::Cpu::Model model(MLIPCore::setup(path.c_str(),mapping,&cutoff,3));
    Atoms atoms;atoms.cell_=20*Eigen::Matrix3d::Identity();atoms.pbc_={false,false,false};
    atoms.pos_={{1,1,1},{2.2,1.4,1.2},{3.7,1.3,1.1}};atoms.chemmap_={0,1,0};p.getEFS(atoms);
    const double *positions[]={atoms.pos_[0].data(),atoms.pos_[1].data(),atoms.pos_[2].data()};
    MLIPCore::Cpu::Executor executor(*model);MLIPCore::Cpu::Workspace work(*model);
    work.reserve(*model,2,MLIPCore::Cpu::DerivativeMode::Cached);
    Eigen::Matrix<double,3,3> force=Eigen::Matrix3d::Zero(),virial=Eigen::Matrix3d::Zero();double energy=0;
    for(int i=0;i<3;++i){int neighbors[2],k=0;for(int j=0;j<3;++j)if(i!=j)neighbors[k++]=j;
        energy+=executor.evaluate_site(i,2,neighbors,types,positions,mapping,true,true,work);
        for(int e=0;e<work.edge_count();++e)for(int a=0;a<3;++a){double grad=work.gradient(e)[a];force(i,a)+=grad;force(work.neighbor(e),a)-=grad;
            for(int b=0;b<3;++b)virial(a,b)-=grad*work.displacement(e)[b];}}
    REQUIRE(energy==Catch::Approx(atoms.energy_).margin(1e-10).epsilon(1e-9));
    for(int i=0;i<3;++i)REQUIRE((force.row(i).transpose()-atoms.forces_[i]).norm()<1e-9*(1+atoms.forces_[i].norm()));
    REQUIRE((virial-atoms.stress_).norm()<1e-9*(1+atoms.stress_.norm()));
}

TEST_CASE("GPM child validation", "[.gpm-child]") {
    const auto *file=std::getenv("IMRMLP_GPM_CHILD_MODEL");REQUIRE(file);REQUIRE_NOTHROW(ModelBundle::validate(file));
    auto manifest=ModelBundle::inspect(file);const auto &root=GPM::object(GPM::at(GPM::object(GPM::at(manifest,"models")),GPM::string(GPM::at(manifest,"root_model"))));
    if(GPM::string(GPM::at(root,"type"))!="eam_zhou2004_parameters") {
        auto bundle=ModelBundle::load(file);auto atom=swap_lattice(GPM::string(GPM::at(root,"type"))=="shellfile"?1:2,std::max(1,bundle->potential().species_count));
        bundle->potential().getEFS(atom);REQUIRE(std::isfinite(atom.energy_));
        for(const auto &force:atom.forces_)REQUIRE(force.allFinite());
    }
}
}

TEST_CASE("GPM SHA and binary contract", "[gpm]") {
    REQUIRE(GPM::hash(std::string("abc"))=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    REQUIRE(GPM::hash(std::string())=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    REQUIRE_THROWS(GPM::parse("{\"x\":1,\"x\":2}"));
    REQUIRE_THROWS(GPM::parse(std::string("\"\xc0\xaf\"")));
    REQUIRE_THROWS(GPM::decimal(GPM::Value("18446744073709551616")));
    Directory dir;GPM::Writer writer;std::vector<double> v{1.,-0.,1e-200,1e200};
    writer.add("values","f64",{4},GPM::encode_f64(v.data(),v.size()));writer.save(dir.file());
    GPM::Reader reader(dir.file());REQUIRE(reader.info("values").offset%64==0);
    REQUIRE(GPM::decode_f64(reader,"values")==v);
    std::fstream file(dir.file(),std::ios::in|std::ios::out|std::ios::binary);file.seekp(64);file.put('X');file.close();
    REQUIRE_THROWS(GPM::Reader(dir.file()));
}

TEST_CASE("GPM MTP is closed and preserves metadata parameters and derivatives", "[gpm]") {
    PotMTP p;load_swap_fixture(p,2,GENERATE(false,true));p.set_element_map({{0,"Al"},{1,"Ni"}});
    if(p.radial_basis_split)for(size_t i=0;i<p.p_radical_basis_pair.size();++i)p.p_radical_basis_pair[i]->scaling=.7+.13*i;
    Directory dir;p.Save(dir.file());potential_wrapper wrapper(dir.file(),"auto");equivalent(p,*wrapper.pot);
    REQUIRE_THROWS(potential_wrapper(dir.file(),"lj612"));
    PotMTP typed;typed.Load(dir.file());equivalent(p,typed);
    REQUIRE(typed.potential_tag.elements()==p.potential_tag.elements());
    auto atoms=swap_lattice(2,2);HessianMatrix h1,h2;p.getEFSHessian(atoms,h1);typed.getEFSHessian(atoms,h2);REQUIRE(h1.hessian.size()==h2.hessian.size());for(size_t i=0;i<h1.hessian.size();++i)REQUIRE(h1.hessian[i].isApprox(h2.hessian[i],1e-11));
    ModelBundle::validate(dir.file());validate_fresh(dir.file());
    REQUIRE(MLIPCore::read_model_text(dir.file().c_str()).find("MTP\n")==0);
    standalone_equivalent(p,dir.file());
}

TEST_CASE("GPM MTP-M heads roundtrip", "[gpm]") {
    auto head=GENERATE("li","q2","fs","n1","n2");Directory dir;
    auto body=StandaloneRadialFixture::model({StandaloneRadialFixture::block("RBChebyshev")},4,head);
    std::ofstream(dir.file("template.mtp"))<<body;
    GPSettings s;s.Modify("pottype","mtp_m");s.Modify("potfile",dir.file("template.mtp"));s.Modify("extmodel",std::string("mtp_m_")+head);s.Modify("n_input","2");s.Modify("n_neuron","3");s.Modify("n_neuron2","2");s.Modify("typemap","");
    potential_wrapper wrapper(s);CAPTURE(head);roundtrip(*wrapper.pot);
    wrapper.pot->Save(dir.file());standalone_equivalent(dynamic_cast<PotMTP&>(*wrapper.pot),dir.file());
}

TEST_CASE("GPM built-in effective settings roundtrip", "[gpm]") {
    auto type=GENERATE("lj612","zbl","mindist");
    GPSettings s;s.Modify("pottype",type);s.Modify("typemap","");
    if(std::string(type)=="lj612")for(auto key:{"0-0","0-1","1-1"}){s.Modify(key,".1 1.2");s.Modify(std::string("cut:")+key,"4");}
    if(std::string(type)=="zbl")s.Modify("z","13 28");
    potential_wrapper wrapper(s);CAPTURE(type);roundtrip(*wrapper.pot);
}

TEST_CASE("GPM MaxVol resumes grading selection and candidate collection", "[gpm][al]") {
    Directory dir;PotMTP p;load_swap_fixture(p,2,false);p.Save(dir.file());auto b=ModelBundle::load(dir.file());
    ActiveSelector selector(new PotEFSBuilder(b->potential()));selector.select_mode=GENERATE(std::string("online"),std::string("full"),std::string("stream"));
    CAPTURE(selector.select_mode);
    selector.init_mode=GENERATE(std::string("data"),std::string("mlip"));selector.init_value=.01;selector.max_swaps=1;
    CAPTURE(selector.init_mode);
    auto a=swap_lattice(2,2);a.features_["custom"]="retained";a.global_attribs_d_["test"]=.125;
    selector.add_candidate(a);auto alternative=a;alternative.pos_[0][0]+=.12;selector.add_candidate(alternative);alternative.pos_[1][1]-=.23;selector.add_candidate(alternative);
    selector.select();selector.prepare_grading();double grade=selector.grade(a);
    b->attach_selector(selector);b->save(dir.file("resume.gpm"),ModelBundle::Profile::Resume);
    auto restored=ModelBundle::load(dir.file("resume.gpm"));GPSettings options;options.Modify("threshold","0");options.Modify("threshold_break","1e10");
    auto al=restored->create_al(options);REQUIRE(al->evaluate(a).uncertainty==Catch::Approx(grade).epsilon(1e-9));
    REQUIRE(al->maxvol_selector()->log_determinant()==Catch::Approx(selector.log_determinant()).margin(1e-8));
    al->check(a);REQUIRE(al->pool_size()==1);al->save(dir.file("pool.gpm"),restored->potential());
    auto continued=ModelBundle::load(dir.file("pool.gpm"));auto session=continued->create_al(GPSettings());
    REQUIRE(session->pool_size()==1);REQUIRE(session->pool()[0].features_.at("custom")=="retained");REQUIRE(session->pool()[0].global_attribs_d_.at("test")==.125);
    a.pos_[0][0]+=.17;selector.add_candidate(a);selector.select();auto next=al->maxvol_selector();next->add_candidate(a);next->select();
    REQUIRE(next->grade(a)==Catch::Approx(selector.grade(a)).epsilon(1e-8));
    REQUIRE(next->selected_indices()==selector.selected_indices());
    REQUIRE(next->converged()==selector.converged());
    auto &mtp=dynamic_cast<PotMTP&>(restored->potential());mtp.radial_coeffs[0]+=.1;
    REQUIRE_THROWS(restored->create_al(GPSettings()));REQUIRE_THROWS(al->evaluate(a));REQUIRE_THROWS(restored->save(dir.file("stale.gpm")));
    REQUIRE_NOTHROW(restored->save(dir.file("inference.gpm"),ModelBundle::Profile::Inference));
}

TEST_CASE("GPM NN removes all descriptor network and species side-file dependencies", "[gpm]") {
    Directory dir;PotMTP plain;load_swap_fixture(plain,2,false);plain.Save(dir.file("original.mtp"));
    GPSettings s;s.Modify("mtpcoeff",dir.file("original.mtp"));s.Modify("nncoeff",dir.file("absent.nn"));s.Modify("spcoeff",dir.file("absent.sc"));
    s.Modify("nnlay",GENERATE("1","2"));s.Modify("nnhidden","4");s.Modify("nnhidden2","3");
    PotMTPNN1 p(s);std::vector<double> weights(p.nn_coeff_size);for(size_t i=0;i<weights.size();++i)weights[i]=.02*std::sin(i+1);
    if(p.nnlay==1)p.p_nn->set_coeffs(weights.data());else p.p_nn2->set_coeffs(weights.data());p.Save(dir.file());std::filesystem::remove(dir.file("original.mtp"));
    auto bundle=ModelBundle::load(dir.file());equivalent(p,bundle->potential());ModelBundle::validate(dir.file());validate_fresh(dir.file());
}

TEST_CASE("GPM local and long-range composite models are closed", "[gpm]") {
    auto type=GENERATE("hmtp","hybrid","hybrid_scale","mtp_les");
    CAPTURE(type);Directory dir;PotMTP p;load_swap_fixture(p,2,false);p.Save(dir.file("a.mtp"));p.moment_coeffs[0]*=.7;p.Save(dir.file("b.mtp"));
    GPSettings s;s.Modify("pottype",type);s.Modify("typemap","");s.Modify("potfile",dir.file("a.mtp"));
    s.Modify("npot","2");s.Modify("pot1:pottype","mtp");s.Modify("pot1:potfile",dir.file("a.mtp"));s.Modify("pot2:pottype","mtp");s.Modify("pot2:potfile",dir.file("b.mtp"));
    s.Modify("boundary1","1.5 3.5");s.Modify("feature","min_distance");
    if(std::string(type)=="hmtp") {
        PotMTP first,second;load_swap_fixture(first,2,false);load_swap_fixture(second,2,false);PotMTPHyb3 h(&first,&second,nullptr);roundtrip(h);return;
    }
    if(std::string(type)=="hybrid"||std::string(type)=="hybrid_scale")s.Modify("potfile","");
    potential_wrapper wrapper(s);
    if(auto *les=dynamic_cast<PotMTP_LES*>(wrapper.pot)){les->les_kernel=GENERATE("coulomb","sog");les->sog_terms=2;les->sog_channels=les->les_kernel=="sog"?2:1;les->train_init();}
    wrapper.pot->Save(dir.file());std::filesystem::remove(dir.file("a.mtp"));std::filesystem::remove(dir.file("b.mtp"));
    auto bundle=ModelBundle::load(dir.file());equivalent(*wrapper.pot,bundle->potential());ModelBundle::validate(dir.file());validate_fresh(dir.file());
    if(std::string(type)!="mtp_les")REQUIRE_THROWS(MLIPCore::read_model_text(dir.file().c_str()));
}

TEST_CASE("GPM hybrid ZBL preserves node element labels and AL binding", "[gpm][al]") {
    Directory dir;PotMTP mtp;load_swap_fixture(mtp,2,false);
    mtp.set_element_map({{0,"Al"},{1,"Ni"}});mtp.Save(dir.file("short.mtp"));
    GPSettings s;s.Modify("pottype","hybrid");s.Modify("potfile","");s.Modify("typemap","");s.Modify("npot","2");
    s.Modify("pot1:pottype","mtp");s.Modify("pot1:potfile",dir.file("short.mtp"));s.Modify("pot1:train","true");
    s.Modify("pot2:pottype","zbl");s.Modify("pot2:z","13 28");s.Modify("pot2:train","false");
    potential_wrapper wrapper(s);
    const bool labeled=GENERATE(false,true);CAPTURE(labeled);
    if(labeled)wrapper.pot->set_element_map({{0,"Al"},{1,"Ni"}});
    const auto fingerprint=ModelBundle::fingerprint(*wrapper.pot);
    wrapper.pot->Save(dir.file());std::filesystem::remove(dir.file("short.mtp"));
    auto bundle=ModelBundle::load(dir.file());
    const auto &hybrid=dynamic_cast<PotentialHybrid&>(bundle->potential());
    REQUIRE(hybrid.children()[1]->gpm_elements.empty()==!labeled);
    REQUIRE(ModelBundle::fingerprint(bundle->potential())==fingerprint);
    equivalent(*wrapper.pot,bundle->potential());
    ModelBundle::validate(dir.file());validate_fresh(dir.file());
    bundle->save(dir.file("resaved.gpm"),ModelBundle::Profile::Inference);
    REQUIRE(ModelBundle::inspect(dir.file()).at("models")==ModelBundle::inspect(dir.file("resaved.gpm")).at("models"));
    ModelBundle::validate(dir.file("resaved.gpm"));

    ActiveSelector selector(new PotEFSBuilder(bundle->potential()));
    selector.select_mode="online";selector.init_mode="mlip";selector.init_value=.01;
    auto atoms=swap_lattice(2,2);selector.add_candidate(atoms);selector.select();selector.prepare_grading();
    const auto grade=selector.grade(atoms);
    const auto bound_fingerprint=ModelBundle::fingerprint(bundle->potential());
    bundle->attach_selector(selector);bundle->save(dir.file("resume.gpm"),ModelBundle::Profile::Resume);
    auto resumed=ModelBundle::load(dir.file("resume.gpm"));auto al=resumed->create_al(GPSettings());
    REQUIRE(al->evaluate(atoms).uncertainty==Catch::Approx(grade).epsilon(1e-9));
    resumed->save(dir.file("preserved.gpm"));
    ModelBundle::validate(dir.file("preserved.gpm"));validate_fresh(dir.file("preserved.gpm"));
    REQUIRE(ModelBundle::fingerprint(resumed->potential())==bound_fingerprint);

    GPM::Writer conflicting;conflicting.preserve(std::make_shared<GPM::Reader>(dir.file()));
    auto &zbl=conflicting.manifest.at("models").get<GPM::Object>().at("root/pot2").get<GPM::Object>();
    zbl.at("species").get<GPM::Array>()[0].get<GPM::Object>()["symbol"]=GPM::Value("Cu");
    conflicting.save(dir.file("conflicting.gpm"));
    REQUIRE_THROWS_WITH(ModelBundle::load(dir.file("conflicting.gpm")),Catch::Matchers::ContainsSubstring("child element order conflicts"));
}

TEST_CASE("GPM tables EAM and empirical variants roundtrip", "[gpm]") {
    auto type=GENERATE("sw","tersoff","pair_table","threebody_table","eampoly","eamalloy","eamfs");
    CAPTURE(type);Directory dir;auto atoms=swap_lattice(2,1);GPSettings s;s.Modify("pottype",type);s.Modify("typemap","");
    auto path=dir.file("source.pot");s.Modify("potfile",path);
    if(std::string(type)=="sw")std::ofstream(path)<<"H H 2body 1 1 2 0.5 4 0 3 3\nH H H 1 1 3 2 1 -0.3333333333333333 3\n";
    if(std::string(type)=="tersoff"||std::string(type)=="brenner"){
        std::ofstream(path)<<"mode="<<type<<"\nH H A=10 B=3 lambda1=2 lambda2=1 lambda3=0.7 n=1 beta=1 D0=0.8123456789012345 S=2 r0=1.4 gamma=0.7 R=3 D=0.3 c=1 d=1 h=-0.3\n";s.Modify("pottype","tersoff");
    }
    if(std::string(type)=="pair_table"){std::ofstream out(path);out<<"H\nN 4\n";for(int i=0;i<4;++i)out<<i<<' '<<.5+i<<' '<<(.5+i)*(.5+i)<<'\n';}
    if(std::string(type)=="threebody_table"){
        std::ofstream out(path);out<<"H\nnr 5 ntheta 5\nrmin 0.5 rmax 4.5\ntheta_min 0 theta_max 3.141592653589793\n";
        for(int i=0;i<25;++i)out<<.01*i<<' ';
    }
    if(std::string(type)=="eampoly"){
        s.Modify("potfile","");for(auto &[k,v]:std::map<std::string,std::string>{{"nelems","1"},{"npoly_rho","2"},{"npoly_f","1"},{"npoly_pair","2"},{"rmax_pair_0_0","6"},{"rmax_rho_0_0","6"},{"coeffs","0 0 0 0 0 1"}})s.Modify(k,v);
    }
    if(std::string(type)=="eamalloy"||std::string(type)=="eamfs"){
        // One-species setfl: nonzero embedding, density and pair tables.
        // The one-species alloy and FS layouts are identical.
        std::ofstream out(path);
        out << "GPM fixture\nsynthetic\nnot a materials potential\n1 H\n9 1 6 1 5\n"
            << "1 1 1 fcc\n";
        for(int i=0;i<9;++i)out << .01*i*i << ' ';
        out << '\n';
        for(int i=0;i<6;++i)out << .02*(5-i)*(5-i) << ' ';
        out << '\n';
        for(int i=0;i<6;++i)out << .03*i*(5-i)*(5-i) << ' ';
        out << '\n';
    }
    potential_wrapper wrapper(s);wrapper.pot->Save(dir.file());std::filesystem::remove(path);
    auto loaded=ModelBundle::load(dir.file());equivalent(*wrapper.pot,loaded->potential(),atoms);ModelBundle::validate(dir.file());validate_fresh(dir.file());
}

TEST_CASE("GPM descriptor linear embeds its MTM dependency", "[gpm]") {
    Directory dir;PotMTP p;load_swap_fixture(p,2,false);p.Save(dir.file("descriptor.mtp"));
    GPSettings s;s.Modify("pottype","descriptor_linear");s.Modify("typemap","");s.Modify("descriptor_linear:descriptor:dtype","mtm");s.Modify("descriptor_linear:descriptor:mtm:potfile",dir.file("descriptor.mtp"));
    potential_wrapper wrapper(s);std::vector<double> c(wrapper.pot->get_NL_coeffs_count(),.1);wrapper.pot->nl_set_coeffs(c.data());wrapper.pot->Save(dir.file());std::filesystem::remove(dir.file("descriptor.mtp"));
    auto loaded=ModelBundle::load(dir.file());equivalent(*wrapper.pot,loaded->potential());ModelBundle::validate(dir.file());validate_fresh(dir.file());
}

TEST_CASE("GPM validates lazy payloads and preserves optional extensions atomically", "[gpm]") {
    Directory dir;PotMTP p;load_swap_fixture(p,2,false);p.Save(dir.file());auto reader=std::make_shared<GPM::Reader>(dir.file());
    GPM::Writer w;w.preserve(reader);w.manifest["extensions"]=GPM::Value(GPM::Object{{"future",GPM::Value("preserve me")}});
    w.add("optional","u8",{3},GPM::Bytes{1,2,3});w.save(dir.file("extra.gpm"));
    auto bundle=ModelBundle::load(dir.file("extra.gpm"));bundle->save(dir.file("copy.gpm"));
    REQUIRE(ModelBundle::inspect(dir.file("copy.gpm")).at("extensions")==w.manifest.at("extensions"));
    GPM::Reader extra(dir.file("extra.gpm"));auto offset=extra.info("optional").offset;
    std::fstream file(dir.file("extra.gpm"),std::ios::in|std::ios::out|std::ios::binary);unsigned char header[64];file.read(reinterpret_cast<char*>(header),64);file.seekp(GPM::get_le(header+24,8)+offset);file.put('X');file.close();
    REQUIRE_NOTHROW(ModelBundle::load(dir.file("extra.gpm")));REQUIRE_THROWS(ModelBundle::validate(dir.file("extra.gpm")));
    GPM::Writer broken;broken.preserve(std::make_shared<GPM::Reader>(dir.file("extra.gpm")));
    REQUIRE_THROWS(broken.save(dir.file("copy.gpm")));REQUIRE_NOTHROW(ModelBundle::validate(dir.file("copy.gpm")));
    auto altered=[&](const std::function<void(GPM::Object&)> &change){GPM::Writer bad;bad.preserve(reader);change(bad.manifest);REQUIRE_THROWS([&]{bad.save(dir.file("bad.gpm"));ModelBundle::load(dir.file("bad.gpm"));}());};
    altered([](auto&o){o["root_model"]=GPM::Value("missing");});
    altered([](auto&o){o["required_features"]=GPM::Value(GPM::Array{GPM::Value("future-required")});});
    altered([](auto&o){o["models"].template get<GPM::Object>().at("root").template get<GPM::Object>()["version"]=GPM::Value(2.);});
    altered([](auto&o){o["species"]=GPM::Value(GPM::Array{});});
    altered([](auto&o){auto &root=o["models"].template get<GPM::Object>().at("root").template get<GPM::Object>();root["type"]=GPM::Value("committee");root["children"]=GPM::Value(GPM::Object{{"0",GPM::Value("root")}});});
    std::filesystem::copy_file(dir.file(),dir.file("short.gpm"));std::filesystem::resize_file(dir.file("short.gpm"),63);REQUIRE_THROWS(ModelBundle::load(dir.file("short.gpm")));
#ifdef USE_ZSTD
    GPM::Writer compressed;compressed.add("array","u8",{4096},GPM::Bytes(4096,7),"zstd");compressed.save(dir.file("compressed.gpm"));GPM::Reader cr(dir.file("compressed.gpm"));REQUIRE(cr.read("array")==GPM::Bytes(4096,7));
#endif
}

TEST_CASE("GPM radial definitions and native inference retain effective configuration", "[gpm][native]") {
    using namespace StandaloneRadialFixture;
    auto name=GENERATE("RBChebyshev","RBChebyCos","RBChebyshevPoly4","RBChebyshevPoly5","RBPolyB","RBPolyBCos","RBChebyshevConfigurable");
    std::istringstream input(model({block(name,4,11,4,true)}));PotMTP p;p.Load(input);CAPTURE(name);roundtrip(p);
    Directory dir;p.Save(dir.file());NativeInference native(dir.file(),{});auto atoms=swap_lattice(2,2);auto expected=atoms;p.getEFS(expected);auto actual=native.evaluate(atoms);
    REQUIRE(actual.energy==Catch::Approx(expected.energy_).margin(1e-11));
    NativeInferenceOptions conflict;conflict.pottype="mtp_m";REQUIRE_THROWS(NativeInference(dir.file(),conflict));
}

TEST_CASE("GPM Zhou parameter objects retain active elements and functions", "[gpm]") {
    Directory dir;std::ofstream(dir.file("parameters.txt"))<<"Al 2.86 1.4 20 20 6 3 0.4 0.4 0.3 0.7 -2 0 1 0 -2 0 1 0 0.8 -2 13 27 0 3 0.7 0.85 1.15\n";
    EAMZhou2004 original;original.LoadParameterSet(dir.file("parameters.txt"));original.add_element("Al");original.init();original.SaveParameterSet(dir.file());
    std::filesystem::remove(dir.file("parameters.txt"));EAMZhou2004 restored;restored.LoadParameterSet(dir.file());
    REQUIRE(original.frho(13,2.3)==restored.frho(13,2.3));REQUIRE(original.phi(13,13,2.3)==restored.phi(13,13,2.3));REQUIRE(original.femb(13,20)==restored.femb(13,20));ModelBundle::validate(dir.file());validate_fresh(dir.file());
}

TEST_CASE("GPM explicitly restores embedded AL configuration", "[gpm][al]") {
    Directory dir;PotMTP p;load_swap_fixture(p,2,false);ActiveSelector selector(new PotEFSBuilder(p));auto atom=swap_lattice(2,2);selector.add_candidate(atom);selector.select();selector.save_state(dir.file());
    potential_wrapper model(dir.file(),"auto");GPSettings s;s.Modify("source","embedded");auto al=AL::ALPotential::create(s);REQUIRE(al->active());REQUIRE_THROWS(al->evaluate(atom));
    al->init_builder(*model.pot);REQUIRE(al->evaluate(atom).uncertainty==Catch::Approx(selector.grade(atom)).epsilon(1e-8));
    s.Modify("state_file","old.sel");REQUIRE_THROWS(AL::ALPotential::create(s));
    validate_fresh(dir.file());
}

TEST_CASE("GPM external adapters retain inputs and report dependencies", "[gpm][external]") {
    SECTION("shell resources") {
        Directory dir;auto old=std::filesystem::current_path();struct Restore {std::filesystem::path path;~Restore(){std::filesystem::current_path(path);}} restore{old};std::filesystem::current_path(dir.path);
        std::ofstream("index.map")<<"H\n";std::ofstream("run.sh")<<"cp \"$1/reference.cfg\" \"$1/output.cfg\"\n";
        Atoms atom=swap_lattice(1,1);atom.energy_=-3;atom.have_energy=true;GPIO io;io.setmap({"H"});REQUIRE(io.write_one(atom,"reference.cfg","mlpcfg"));
        GPSettings s;s.Modify("runscript","run.sh");s.Modify("index","index.map");s.Modify("write_posformat","mlpcfg");s.Modify("write_posfile","input.cfg");s.Modify("read_outformat","mlpcfg");s.Modify("read_outfile","output.cfg");s.Modify("dependencies","bash cp");s.Modify("resource:reference.cfg","reference.cfg");
        PotShellCalc shell(s);shell.Save(dir.file());for(auto file:{"index.map","run.sh","reference.cfg"})std::filesystem::remove(file);
        auto loaded=ModelBundle::load(dir.file());loaded->potential().getEFS(atom);REQUIRE(atom.energy_==-3);ModelBundle::validate(dir.file());validate_fresh(dir.file());
    }
    SECTION("OpenKIM") {
#ifdef USE_KIM
        const std::string name="ex_model_Ar_P_LJ";KIM::Model *available=nullptr;int accepted=0;
        int error=KIM::Model::Create(KIM::NUMBERING::zeroBased,KIM::LENGTH_UNIT::A,KIM::ENERGY_UNIT::eV,KIM::CHARGE_UNIT::e,KIM::TEMPERATURE_UNIT::K,KIM::TIME_UNIT::ps,name,&accepted,&available);
        if(available)KIM::Model::Destroy(&available);if(error||!accepted)SKIP("Optional KIM example model is not installed");
        PotKIM p(name,{{0,"Ar"}});GPSettings overrides;overrides.Modify("kim_parameter:epsilon",".02");overrides.Modify("kim_parameter:sigma","3");overrides.Modify("kim_parameter:cutoff","6");p.ApplyParameterSettings(overrides);p.omp_init();roundtrip(p,swap_lattice(2,1));
#else
        SKIP("OpenKIM is not compiled");
#endif
    }
    SECTION("unavailable runtime error") {
#ifndef WITH_ACELMP
        Directory dir;GPM::Writer w;w.manifest["models"]=GPM::Value(GPM::Object{{"root",GPM::Value(GPM::Object{{"type",GPM::Value("acelmp")},{"version",GPM::Value(1.)}})}});w.save(dir.file());REQUIRE_THROWS_WITH(ModelBundle::load(dir.file()),Catch::Matchers::ContainsSubstring("ACE runtime support"));
#else
        SKIP("ACE adapter needs an installed PACE model fixture");
#endif
    }
}

TEST_CASE("GPM legacy MaxVol import verifies model rows", "[gpm][al]") {
    Directory dir;PotMTP p;load_swap_fixture(p,2,false);ActiveSelector selector(new PotEFSBuilder(p));
    selector.max_swaps=1;auto atom=swap_lattice(2,2);selector.add_candidate(atom);selector.select();selector.save_state(dir.file("legacy.sel"));
    p.Save(dir.file("source.gpm"));auto bundle=ModelBundle::load(dir.file("source.gpm"));
    ActiveSelector legacy;legacy.load_state(dir.file("legacy.sel"));legacy.init_builder(bundle->potential());
    bundle->attach_selector(legacy);bundle->save(dir.file("resume.gpm"),ModelBundle::Profile::Resume);
    auto restored=ModelBundle::load(dir.file("resume.gpm"))->create_al(GPSettings());
    REQUIRE(restored->evaluate(atom).uncertainty==Catch::Approx(selector.grade(atom)).epsilon(1e-8));
    REQUIRE(restored->maxvol_selector()->converged()==selector.converged());
    REQUIRE(restored->maxvol_selector()->termination_reason()==selector.termination_reason());
    auto &other=dynamic_cast<PotMTP&>(bundle->potential());other.radial_coeffs[0]+=.4;
    REQUIRE_THROWS_WITH(bundle->attach_selector(legacy),Catch::Matchers::ContainsSubstring("rows do not match"));
}
