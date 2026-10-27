#include <catch2/catch_all.hpp>
#include "basis/basis.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/standalone_export.h"
#include "LAMMPS/USER-MLIP/opt/radial_basis_opt.h"
#include "runtime/mtp/cpu/mtp_cpu.h"
#include "LAMMPS/USER-MLIP/standalone/tests/radial_fixture.h"
#include <memory>

namespace {
namespace R=MLIPCore::Radial;
namespace F=StandaloneRadialFixture;
using namespace Potentials;
MLIPCore::Cpu::Model load(const std::string &text) {
    int mapping[]={-1,0,1};double cutoff;
    return MLIPCore::Cpu::Model(MLIPCore::setup_from_text(text,mapping,&cutoff,3));
}
std::unique_ptr<PotMTP> native_model(const std::string &text,const std::string &head) {
    std::unique_ptr<PotMTP> p;
    if(head=="classic")p=std::make_unique<PotMTP>();
    else if(head=="li")p=std::make_unique<PotMTP_M<m_model_mtpli>>();
    else if(head=="q2")p=std::make_unique<PotMTP_M<m_model_mtpq2>>();
    else if(head=="fs")p=std::make_unique<PotMTP_M<m_model_mtpfs>>();
    else if(head=="n1")p=std::make_unique<PotMTP_M<m_model_nn1>>();
    else p=std::make_unique<PotMTP_M<m_model_nn2>>();
    GPSettings settings;settings.Modify("n_input","2");settings.Modify("n_neuron","3");settings.Modify("n_neuron2","2");
    if(auto *nn=dynamic_cast<PotMTP_M<m_model_nn1> *>(p.get()))nn->model.init(settings);
    if(auto *nn=dynamic_cast<PotMTP_M<m_model_nn2> *>(p.get()))nn->model.init(settings);
    std::istringstream in(text);p->Load(in);return p;
}
void compare_formula(const std::string &block,int size) {
    std::istringstream in(block);std::string key,eq,name;in>>key>>eq>>name;
    std::unique_ptr<AnyBasis> native(create_basis_by_name(name,in,.65));
    std::istringstream parsed(block.substr(block.find('=')+1)+"radial_funcs_count");
    // read_radial_basis expects the '=' token.
    parsed.str(block.substr(block.find('='))+"radial_funcs_count");
    auto b=MLIPOpt::read_radial_basis(parsed,.65).basis;
    if(R::is_poly(b)) {
        const Eigen::MatrixXd *matrix;
        const std::vector<double> *normalizations;
        if(auto *poly=dynamic_cast<RBasis_PolyB *>(native.get())) {matrix=&poly->coeffs;normalizations=&poly->invNalpha;}
        else {auto *cos=dynamic_cast<RBasis_PolyBCos *>(native.get());matrix=&cos->coeffs;normalizations=&cos->invNalpha;}
        b.polynomial_matrix.assign(matrix->data(),matrix->data()+size*size);b.polynomial_normalizations=*normalizations;
    }
    for(double r:{.3,.5,.500001,1.4,2.1,3.65,4.999999,5.,5.2}) {
        std::vector<double> v(size),d(size),nv(size),nd(size);
        // Exercise the classic loop's already-checked entry as well as clipping.
        if(b.excludes(r)) b.evaluate(r,v.data(),d.data());
        else b.evaluate_inside(r,v.data(),d.data());
        native->calc_der(r,nv,nd);
        if(r>=5) {std::fill(nv.begin(),nv.end(),0);std::fill(nd.begin(),nd.end(),0);}
        for(int i=0;i<size;++i) {
            CAPTURE(name,size,r,i);
            REQUIRE(v[i]==Catch::Approx(nv[i]).epsilon(2e-9).margin(1e-10));
            REQUIRE(d[i]==Catch::Approx(nd[i]).epsilon(2e-9).margin(1e-10));
        }
    }
}
void oracle(const std::string &text,const std::string &head) {
    auto native=native_model(text,head);
    auto model=load(Potentials::export_standalone_radial(text));
    Atoms atoms;atoms.cell_=20*Eigen::Matrix3d::Identity();
    atoms.pos_={{1,1,1},{1.3,1.1,1},{2.1,1.4,1.2},{4.7,1.3,1.1},{1.2,3.1,2.4}};
    atoms.chemmap_={0,1,0,1,0};atoms.resetEFS();Neighborhoods nbh(atoms,native->cutoff);native->getEFS(atoms,nbh);
    MLIPCore::Cpu::Executor executor(*model);
    for(auto mode:{MLIPCore::Cpu::DerivativeMode::Cached,MLIPCore::Cpu::DerivativeMode::Recompute}) {
        MLIPCore::Cpu::Workspace workspace(*model);workspace.reserve(*model,4,mode);
        int types[]={1,2,1,2,1},map[]={-1,0,1};
        const double *positions[5];for(int i=0;i<5;++i)positions[i]=atoms.pos_[i].data();
        Eigen::Matrix<double,5,3> force=Eigen::Matrix<double,5,3>::Zero();
        Eigen::Matrix3d virial=Eigen::Matrix3d::Zero();double total=0;
        for(int i=0;i<5;++i) {
            int neighbors[4],k=0;for(int j=0;j<5;++j)if(i!=j)neighbors[k++]=j;
            const double site=executor.evaluate_site(i,4,neighbors,types,positions,map,true,true,workspace);
            REQUIRE(site==Catch::Approx(atoms.site_energies_[i]).margin(1e-9).epsilon(1e-9));total+=site;
            for(int e=0;e<workspace.edge_count();++e) {
                const int j=workspace.neighbor(e);
                for(int a=0;a<3;++a) {
                    const double grad=workspace.gradient(e)[a];force(i,a)+=grad;force(j,a)-=grad;
                    for(int b=0;b<3;++b)virial(a,b)-=grad*workspace.displacement(e)[b];
                }
            }
        }
        REQUIRE(total==Catch::Approx(atoms.energy_).margin(1e-9).epsilon(1e-9));
        for(int i=0;i<5;++i)for(int a=0;a<3;++a)REQUIRE(force(i,a)==Catch::Approx(atoms.forces_[i][a]).margin(1e-9).epsilon(1e-9));
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)REQUIRE(virial(a,b)==Catch::Approx(atoms.stress_(a,b)).margin(1e-9).epsilon(1e-9));
    }
}
}

TEST_CASE("Standalone radial formulas cover six native bases and all configurable combinations","[standalone][radial]") {
    for(int size:{1,4}) {
        for(const auto *name:{"RBChebyshev","RBChebyCos","RBChebyshevPoly4","RBChebyshevPoly5","RBPolyB","RBPolyBCos"})
            compare_formula(F::block(name,size),size);
        for(int cut=0;cut<12;++cut)for(int shift=0;shift<5;++shift)for(bool flip:{false,true})
            compare_formula(F::block("RBChebyshevConfigurable",size,cut,shift,flip),size);
    }
    for(double lambda:{0.,-.7})for(double scale:{0.,.8})for(int shift:{1,2,4})
        compare_formula(F::block("RBChebyshevConfigurable",4,11,shift,true,.5,5,lambda,scale),4);
    for(int size:{2,6,8})for(const auto *name:{"RBPolyB","RBPolyBCos"})compare_formula(F::block(name,size),size);
    // The envelope-weighted recurrence must also agree at higher orders and
    // below the lower bound for the families that retain extrapolation.
    for(int size:{8,32}) {
        for(const auto *name:{"RBChebyshev","RBChebyCos","RBChebyshevPoly4","RBChebyshevPoly5"})
            compare_formula(F::block(name,size),size);
        for(int cut=0;cut<12;++cut)for(int shift=0;shift<5;++shift)for(bool flip:{false,true})
            compare_formula(F::block("RBChebyshevConfigurable",size,cut,shift,flip),size);
    }
}
TEST_CASE("Standalone fixed PolyB cache roundtrips rejects stale data and preserves MVS prefix","[standalone][radial][export]") {
    const auto original=F::model({F::block("RBPolyB",8)},8);
    const auto deployed=export_standalone_radial(original+"#MVS_v1.1\n\0binary");
    REQUIRE(deployed.substr(0,original.size())==original);
    REQUIRE(deployed.find("#MVS")==std::string::npos);
    REQUIRE(export_standalone_radial(deployed)==deployed);
    const auto parsed=load(deployed);REQUIRE(parsed->pair_radial_basis_count==4);
    auto native=native_model(original,"classic");auto *poly=dynamic_cast<RBasis_PolyB *>(native->p_radical_basis);
    const auto &b=parsed->pair_radial_bases[0];
    REQUIRE(b.polynomial_matrix==std::vector<double>(poly->coeffs.data(),poly->coeffs.data()+64));
    REQUIRE(b.polynomial_normalizations==poly->invNalpha);
    auto stale=deployed;stale.replace(stale.find("min_dist = 0.5"),14,"min_dist = 0.6");REQUIRE_THROWS(load(stale));
    REQUIRE_THROWS(load(deployed+R::cache_footer({const_cast<MLIPCore::mlip::radial_basis_t *>(&b)},false,2)));
    bool present;auto cache=R::read_cache(deployed,present);auto &root=cache.get<picojson::object>();
    auto malformed=[&] {return original+"# standalone_radial_v1 = "+cache.serialize()+"\n";};
    auto &pairs=root["pairs"].get<picojson::array>();pairs.pop_back();REQUIRE_THROWS(load(malformed()));
    pairs[0].get<picojson::object>()["matrix"]=picojson::value(picojson::array{});REQUIRE_THROWS(load(malformed()));
    REQUIRE_THROWS(load(original+"# standalone_radial_v2 = {}\n"));
}
TEST_CASE("Standalone loaded classic and five heads agree with native EFS for every radial representation","[standalone][radial][oracle]") {
    std::vector<std::vector<std::string>> cases;
    for(const auto *name:{"RBChebyshev","RBChebyCos","RBChebyshevPoly4","RBChebyshevPoly5","RBPolyB","RBPolyBCos"})
        cases.push_back({F::block(name)});
    for(int cut=0;cut<12;++cut)for(int shift=0;shift<5;++shift)for(bool flip:{false,true})
        cases.push_back({F::block("RBChebyshevConfigurable",4,cut,shift,flip)});
    cases.push_back({F::block("RBPolyB",4,0,0,false,.4,4.1),F::block("RBPolyBCos",4,0,0,false,.6,4.4),
        F::block("RBChebyshevConfigurable",4,11,4,true,.8,4.7),F::block("RBChebyshevPoly4")});
    for(const auto &blocks:cases)for(const std::string head:{"classic","li","q2","fs","n1","n2"}) {
        CAPTURE(head,blocks[0]);oracle(F::model(blocks,4,head),head);
    }
}
