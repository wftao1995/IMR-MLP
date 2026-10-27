#include <catch2/catch_all.hpp>
#include "potentials/native_inference.h"
#include "potentials/mtp_gpu_adapter.h"
#include "unitest/maxvol_mtp_fixture.h"
#include <chrono>
#include <cmath>
#include <limits>

using namespace Potentials;
namespace {
struct Candidates {
    std::vector<int> offsets,neighbors;
    NativeCandidateNeighborhoodView view;
    explicit Candidates(const Atoms &atoms,int owned=-1) {
        if(owned<0)owned=atoms.size();
        offsets.push_back(0);
        for(int i=0;i<owned;++i) {
            for(int j=0;j<atoms.size();++j)if(i!=j)neighbors.push_back(j);
            offsets.push_back(neighbors.size());
        }
        view={owned,atoms.size(),static_cast<int>(neighbors.size()),offsets.data(),neighbors.data()};
    }
};
std::string device(bool gpu) {
    if(!gpu)return "cpu";
#ifdef IMRMLP_GPU
    int count=0;std::string error;
    const bool ok=MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count,&error)&&count>0;
    if(std::getenv("IMRMLP_REQUIRE_GPU")){INFO(error);REQUIRE(ok);}
    if(!ok)SKIP("No GPU available for candidate deployment acceptance");
    return MLIPCore::GpuDirect::MTPGpuDirect::backend_name();
#else
    return "cpu";
#endif
}
struct Fixture {
    std::filesystem::path directory=std::filesystem::temp_directory_path()/
        ("native-candidates-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::vector<MaxVolMTPCase> cases=make_maxvol_mtp_models(directory);
    ~Fixture(){std::filesystem::remove_all(directory);}
    NativeInferenceOptions options(const MaxVolMTPCase &entry,const std::string &backend) {
        NativeInferenceOptions out;out.pottype=entry.pottype;out.device=backend;
        out.settings_file=(directory/(entry.name+".ini")).string();
        std::ofstream file(out.settings_file);
        for(const auto &[key,value]:entry.settings.settings)file<<"pot:"<<key<<' '<<value<<'\n';
        return out;
    }
};
void compare(const NativeInferenceResult &a,const NativeInferenceResult &b) {
    REQUIRE(a.energy==Catch::Approx(b.energy).margin(1e-8).epsilon(2e-9));
    const auto same=[](const auto &x,const auto &y,double absolute,double relative) {
        REQUIRE(x.rows()==y.rows());REQUIRE(x.cols()==y.cols());
        for(int i=0;i<x.rows();++i)for(int j=0;j<x.cols();++j)
            REQUIRE(std::abs(x(i,j)-y(i,j))<=absolute+relative*std::abs(y(i,j)));
    };
    same(a.forces,b.forces,1e-8,2e-9);same(a.virial,b.virial,1e-8,2e-9);
    same(a.site_energy,b.site_energy,1e-8,2e-9);same(a.site_virial,b.site_virial,1e-8,2e-9);
    same(a.charges,b.charges,1e-10,1e-9);same(a.latents,b.latents,1e-10,1e-9);
}
NativeInferenceRequest no_outputs() {
    NativeInferenceRequest r;r.energy=r.forces=r.site_energy=r.virial=r.site_virial=r.charges=r.latents=false;
    return r;
}
}

TEST_CASE("Native candidate builder preserves all families and requests", "[native][candidates][gpu][packed]") {
#ifdef IMRMLP_GPU
    const bool gpu=GENERATE(false,true);
#else
    const bool gpu=false;
#endif
    const auto backend=device(gpu);Fixture fixture;
    for(const auto &entry:fixture.cases) {
        DYNAMIC_SECTION(backend << "/" << entry.name) {
            NativeInference inference(entry.path,fixture.options(entry,backend));
            NativeInference reference(entry.path,fixture.options(entry,"cpu"));
            auto atoms=maxvol_mtp_atoms();atoms.cell_=32.*Eigen::Matrix3d::Identity();
            Candidates c(atoms);
            for(int field=-1;field<8;++field) {
                auto request=field<0?NativeInferenceRequest{}:no_outputs();
                switch(field) {case 0:request.forces=true;break;case 1:request.energy=true;break;
                    case 2:request.site_energy=true;break;case 3:request.virial=true;break;
                    case 4:request.site_virial=true;break;case 5:request.charges=true;break;
                    case 6:request.latents=true;break;}
                const auto expected=reference.evaluate(atoms,request);
                compare(inference.evaluate_distributed_candidates(atoms,c.view,{},request),expected);
            }
            const auto expected=reference.evaluate(atoms);
            bool mutated=false;NativeCollectives callbacks;
            callbacks.sum=[&](double *,std::size_t) {
                if(mutated)return;mutated=true;
                atoms.pos_[0][0]=std::numeric_limits<double>::quiet_NaN();
                atoms.chemmap_[0]=-1;atoms.cell_.setZero();
                atoms.features_.clear();atoms.global_attribs_d_.clear();
                if(!c.neighbors.empty())c.neighbors[0]=-1;
            };
            // The first reduction happens after preparation. Later numerical
            // stages must use owned arrays, never those mutated caller buffers.
            compare(inference.evaluate_distributed_candidates(atoms,c.view,callbacks),expected);
            REQUIRE(mutated);
        }
    }
}

TEST_CASE("Native candidate validation and topology recovery are transactional", "[native][candidates][gpu][packed]") {
#ifdef IMRMLP_GPU
    const bool gpu=GENERATE(false,true);
#else
    const bool gpu=false;
#endif
    const auto backend=device(gpu);Fixture fixture;const auto &entry=fixture.cases.front();
    NativeInference inference(entry.path,fixture.options(entry,backend));
    NativeInference reference(entry.path,fixture.options(entry,"cpu"));
    auto atoms=maxvol_mtp_atoms();atoms.cell_=32.*Eigen::Matrix3d::Identity();Candidates c(atoms);
    const auto evaluate=[&]{return inference.evaluate_distributed_candidates(atoms,c.view,{});};
    compare(evaluate(),reference.evaluate(atoms));auto revision=inference.statistics().candidate_topology_revision;
    REQUIRE(revision>0);
    atoms.pos_[0][0]+=.017;compare(evaluate(),reference.evaluate(atoms));
    REQUIRE(inference.statistics().candidate_topology_revision==revision);
    std::swap(atoms.chemmap_[0],atoms.chemmap_[1]);compare(evaluate(),reference.evaluate(atoms));
    if(atoms.chemmap_[0]!=atoms.chemmap_[1])REQUIRE(inference.statistics().candidate_topology_revision>revision);
    revision=inference.statistics().candidate_topology_revision;
    const int neighbor=c.neighbors[0];c.neighbors[0]=atoms.size();
    REQUIRE_THROWS(evaluate());REQUIRE(inference.statistics().candidate_topology_revision==0);
    c.neighbors[0]=neighbor;compare(evaluate(),reference.evaluate(atoms));
    REQUIRE(inference.statistics().candidate_topology_revision>revision);
    revision=inference.statistics().candidate_topology_revision;
    NativeCollectives peer_failure;peer_failure.sum=[](double *value,std::size_t){*value=1.;};
    REQUIRE_THROWS(inference.evaluate_distributed_candidates(atoms,c.view,peer_failure));
    REQUIRE(inference.statistics().candidate_topology_revision==0);
    compare(evaluate(),reference.evaluate(atoms));REQUIRE(inference.statistics().candidate_topology_revision>revision);
    revision=inference.statistics().candidate_topology_revision;
    auto provider=atoms;compare(inference.evaluate_distributed_candidates(provider,c.view,{}),reference.evaluate(provider));
    REQUIRE(inference.statistics().candidate_topology_revision>revision);
    const int end=c.offsets.back();c.offsets[1]=end+1;
    REQUIRE_THROWS(evaluate());c.offsets[1]=atoms.size()-1;
    auto invalid=c.view;invalid.candidate_count=-1;REQUIRE_THROWS(inference.evaluate_distributed_candidates(atoms,invalid,{}));
    invalid=c.view;invalid.owned_count=atoms.size()+1;REQUIRE_THROWS(inference.evaluate_distributed_candidates(atoms,invalid,{}));
    invalid=c.view;invalid.total_count=std::numeric_limits<int>::max();REQUIRE_THROWS(inference.evaluate_distributed_candidates(atoms,invalid,{}));
    invalid=c.view;invalid.neighbors=nullptr;REQUIRE_THROWS(inference.evaluate_distributed_candidates(atoms,invalid,{}));
    // All ghosts are validated, including an empty rank with no candidate edge.
    Candidates empty(atoms,0);auto good_empty=inference.evaluate_distributed_candidates(atoms,empty.view,{},no_outputs());
    REQUIRE(good_empty.forces.rows()==0);REQUIRE(inference.statistics().neighbor_edges==0);
    const auto coordinate=atoms.pos_.back();atoms.pos_.back()[0]=std::numeric_limits<double>::infinity();
    REQUIRE_THROWS(inference.evaluate_distributed_candidates(atoms,empty.view,{},no_outputs()));
    atoms.pos_.back()=coordinate;const int type=atoms.chemmap_.back();atoms.chemmap_.back()=-1;
    REQUIRE_THROWS(inference.evaluate_distributed_candidates(atoms,empty.view,{},no_outputs()));atoms.chemmap_.back()=type;
    const auto saved=atoms.pos_[neighbor];atoms.pos_[neighbor]=atoms.pos_[0];REQUIRE_THROWS(evaluate());atoms.pos_[neighbor]=saved;
    compare(evaluate(),reference.evaluate(atoms));
    Atoms zero;zero.cell_=32.*Eigen::Matrix3d::Identity();Candidates no_atoms(zero);
    const auto result=inference.evaluate_distributed_candidates(zero,no_atoms.view,{},no_outputs());
    REQUIRE(result.forces.rows()==0);REQUIRE(inference.statistics().neighbor_edges==0);
}
