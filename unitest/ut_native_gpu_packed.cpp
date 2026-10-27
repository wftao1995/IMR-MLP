#include <catch2/catch_all.hpp>
#include "potentials/native_inference.h"
#include "potentials/mtp_gpu_adapter.h"
#include "unitest/maxvol_mtp_fixture.h"
#include <chrono>
#include <numeric>
#include <limits>

#ifdef IMRMLP_GPU
using namespace Potentials;
namespace {
bool available() {
    int count=0;std::string error;
    const bool ok=MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count,&error)&&count>0;
    if(std::getenv("IMRMLP_REQUIRE_GPU")){INFO(error);REQUIRE(ok);}
    return ok;
}
struct CSR {
    std::vector<int> centers,offsets,neighbors;
    std::vector<double> positions,vectors,distances;
    NativePackedNeighborhoodView v;
    void refresh(Atoms &a,double cutoff,bool topology=true) {
        Neighborhoods nbhs(a,cutoff);
        centers.resize(a.size());std::iota(centers.begin(),centers.end(),0);
        offsets.assign(a.size()+1,0);neighbors.clear();vectors.clear();distances.clear();
        positions.resize(3*a.size());
        for(int i=0;i<a.size();++i) {
            for(int d=0;d<3;++d)positions[3*i+d]=a.pos_[i][d];
            for(int j=0;j<nbhs[i].nbhcount_;++j){neighbors.push_back(nbhs[i].indices_[j]);
                distances.push_back(nbhs[i].distances_[j]);
                for(int d=0;d<3;++d)vectors.push_back(nbhs[i].pos_[j][d]);}
            offsets[i+1]=neighbors.size();
        }
        v.owned_count=v.total_count=a.size();v.edge_count=neighbors.size();
        v.centers=centers.data();v.offsets=offsets.data();v.neighbors=neighbors.data();v.types=a.chemmap_.data();
        v.positions=positions.data();v.vectors=nullptr;v.distances=distances.data();
        v.topology_owner=reinterpret_cast<std::uintptr_t>(this);
        if(topology)++v.topology_revision;
    }
};
void compare(const NativeInferenceResult &a,const NativeInferenceResult &b,const NativeInferenceRequest &r) {
    if(r.forces)REQUIRE((a.forces-b.forces).norm()<2e-8);else REQUIRE(a.forces.rows()==0);
    if(r.energy)REQUIRE(a.energy==Catch::Approx(b.energy).margin(2e-8));else REQUIRE(a.energy==0.);
    if(r.virial)REQUIRE((a.virial-b.virial).norm()<2e-8);else REQUIRE(a.virial.norm()==0.);
    if(r.site_energy)REQUIRE((a.site_energy-b.site_energy).norm()<2e-8);else REQUIRE(a.site_energy.size()==0);
    if(r.site_virial)REQUIRE((a.site_virial-b.site_virial).norm()<2e-8);else REQUIRE(a.site_virial.rows()==0);
    if(r.charges)REQUIRE((a.charges-b.charges).norm()<2e-8);else REQUIRE(a.charges.size()==0);
    if(r.latents)REQUIRE((a.latents-b.latents).norm()<2e-8);else REQUIRE(a.latents.size()==0);
}
}
TEST_CASE("Native packed GPU entry preserves outputs across request and topology changes", "[native][gpu][packed]") {
    if(!available())SKIP("No GPU device");
    const auto dir=std::filesystem::temp_directory_path()/
        ("native-gpu-packed-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::filesystem::remove_all(path);}} cleanup{dir};
    const auto cases=make_maxvol_mtp_models(dir);
    for(const auto &entry:cases) {
      DYNAMIC_SECTION(entry.name) {
        NativeInferenceOptions options;options.pottype=entry.pottype;
        options.settings_file=(dir/(entry.name+".ini")).string();
        {std::ofstream out(options.settings_file);for(const auto &[k,v]:entry.settings.settings)out<<"pot:"<<k<<' '<<v<<'\n';}
        options.device=MLIPCore::GpuDirect::MTPGpuDirect::backend_name();
        NativeInference gpu(entry.path,options);options.device="cpu";NativeInference cpu(entry.path,options);
        auto a=maxvol_mtp_atoms();a.cell_=32.*Eigen::Matrix3d::Identity();
        CSR csr;csr.refresh(a,gpu.cutoff());
        for(int step=0;step<4;++step) {
            if(step==1){a.pos_[0][0]+=.013;csr.refresh(a,gpu.cutoff(),false);}
            if(step==2){a.pos_[2][0]+=12.;csr.refresh(a,gpu.cutoff());}
            if(step==3){a.pos_[2][0]-=12.;std::swap(a.chemmap_[0],a.chemmap_[1]);csr.refresh(a,gpu.cutoff());}
            const auto expected=cpu.evaluate(a);
            for(int field=-1;field<7;++field) {
                NativeInferenceRequest r;
                if(field>=0){r.forces=r.energy=r.site_energy=r.virial=r.site_virial=r.charges=r.latents=false;
                    switch(field){case 0:r.forces=true;break;case 1:r.energy=true;break;case 2:r.site_energy=true;break;
                        case 3:r.virial=true;break;case 4:r.site_virial=true;break;case 5:r.charges=true;break;case 6:r.latents=true;}}
                CAPTURE(step,field);
                compare(gpu.evaluate_distributed_packed(a,csr.v,NativeCollectives{},r),expected,r);
            }
            // A compatibility call must not leave a token that revives stale CSR.
            compare(gpu.evaluate(a),expected,{});
            compare(gpu.evaluate_distributed_packed(a,csr.v,NativeCollectives{}),expected,{});
        }
        // Independent provider with coincident revision, then explicitly disabled caching.
        CSR other;other.refresh(a,gpu.cutoff());other.v.topology_revision=csr.v.topology_revision;
        compare(gpu.evaluate_distributed_packed(a,other.v,NativeCollectives{}),cpu.evaluate(a),{});
        other.v.topology_owner=0;other.v.vectors=other.vectors.data();
        compare(gpu.evaluate_distributed_packed(a,other.v,NativeCollectives{}),cpu.evaluate(a),{});
        const auto good=cpu.evaluate(a);
        const auto check_recovery=[&]{compare(gpu.evaluate_distributed_packed(a,other.v,NativeCollectives{}),good,{});};
        const double distance=other.distances[0];other.distances[0]=0.;
        REQUIRE_THROWS(gpu.evaluate_distributed_packed(a,other.v,NativeCollectives{}));
        other.distances[0]=distance;check_recovery();
        const double component=other.vectors[0];other.vectors[0]=std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS(gpu.evaluate_distributed_packed(a,other.v,NativeCollectives{}));
        other.vectors[0]=component;check_recovery();
        other.v.vectors=nullptr;
        const int neighbor=other.neighbors[0];double saved[3];
        for(int d=0;d<3;++d){saved[d]=other.positions[3*neighbor+d];other.positions[3*neighbor+d]=other.positions[d];}
        // Supplied positive distance cannot hide an actual coordinate overlap.
        REQUIRE_THROWS(gpu.evaluate_distributed_packed(a,other.v,NativeCollectives{}));
        for(int d=0;d<3;++d)other.positions[3*neighbor+d]=saved[d];check_recovery();
        const double x=other.positions[0];other.positions[0]=std::numeric_limits<double>::infinity();
        REQUIRE_THROWS(gpu.evaluate_distributed_packed(a,other.v,NativeCollectives{}));
        other.positions[0]=x;check_recovery();
        const int type=a.chemmap_[0];a.chemmap_[0]=gpu.species_count();
        REQUIRE_THROWS(gpu.evaluate_distributed_packed(a,other.v,NativeCollectives{}));
        a.chemmap_[0]=type;check_recovery();
      }
    }
}
TEST_CASE("GPU force-only requests reuse CSR but refresh geometry and reduce global virial", "[gpu][packed][outputs]") {
    if(!available())SKIP("No GPU device");
    PotMTP model;load_swap_fixture(model,2,false);
    auto a=maxvol_mtp_atoms();a.cell_=32.*Eigen::Matrix3d::Identity();CSR csr;csr.refresh(a,model.cutoff);
    MTPGpuEvaluator evaluator;std::string error;REQUIRE(evaluator.initialize_short_range(model,0,&error));
    using namespace MLIPCore::GpuDirect;
    std::vector<double> f(3*a.size()),site(6*a.size());double energy=0.,global[6]{};
    MTPGpuBatchRequest r;
    const auto refresh=[&]{r.centers=csr.v.centers;r.offsets=csr.v.offsets;r.neighbors=csr.v.neighbors;
        r.center_count=r.nlocal=a.size();r.ntotal=a.size();r.type=a.chemmap_.data();r.x=csr.v.positions;
        r.topology_token={csr.v.topology_owner,csr.v.topology_revision};};
    refresh();r.force=f.data();r.total_energy=&energy;r.accumulate_force=false;
    r.compute_global_energy=false;r.compute_site_energy=false;r.reuse_topology=true;r.topology_includes_types=true;
    REQUIRE(evaluator.context().compute_batch(r,&error));
    const auto first=evaluator.context().runtime_stats();
    REQUIRE(first.topology_uploads==1);REQUIRE(first.topology_reuses==0);
    REQUIRE(evaluator.context().memory_stats().output_bytes==f.size()*sizeof(double));
    a.pos_[0][0]+=.017;csr.refresh(a,model.cutoff,false);refresh();
    REQUIRE(evaluator.context().compute_batch(r,&error));
    auto expected=a;model.getEFS(expected);
    for(int i=0;i<a.size();++i)for(int d=0;d<3;++d)REQUIRE(f[3*i+d]==Catch::Approx(expected.forces_[i][d]).margin(2e-9));
    const auto moved=evaluator.context().runtime_stats();
    REQUIRE(moved.topology_uploads==1);REQUIRE(moved.topology_reuses==1);REQUIRE(moved.coordinate_uploads==2);
    REQUIRE(moved.type_uploads==1);REQUIRE(moved.type_upload_bytes==a.size()*sizeof(int));
    r.global_virial=global;r.compute_global_energy=true;
    REQUIRE(evaluator.context().compute_batch(r,&error));
    r.site_virial=site.data();REQUIRE(evaluator.context().compute_batch(r,&error));
    for(int d=0;d<6;++d){double sum=0.;for(int i=0;i<a.size();++i)sum+=site[6*i+d];
        REQUIRE(global[d]==Catch::Approx(sum).margin(2e-9));}
    REQUIRE(energy==Catch::Approx(expected.energy_).margin(2e-9));
    r.copy_forces_to_host=false;r.force=nullptr;r.site_virial=nullptr;
    REQUIRE(evaluator.context().compute_batch(r,&error));
    for(int d=0;d<6;++d){double sum=0.;for(int i=0;i<a.size();++i)sum+=site[6*i+d];
        REQUIRE(global[d]==Catch::Approx(sum).margin(2e-9));}
    r.copy_forces_to_host=true;r.force=f.data();r.site_virial=site.data();
    // Changing providers, revision and failed topology each invalidate residency.
    ++r.topology_token.owner;REQUIRE(evaluator.context().compute_batch(r,&error));
    REQUIRE(evaluator.context().runtime_stats().topology_uploads==2);
    ++r.topology_token.revision;REQUIRE(evaluator.context().compute_batch(r,&error));
    REQUIRE(evaluator.context().runtime_stats().topology_uploads==3);
    const auto saved=csr.neighbors[0];csr.neighbors[0]=a.size();++r.topology_token.revision;
    REQUIRE_FALSE(evaluator.context().compute_batch(r,&error));csr.neighbors[0]=saved;
    REQUIRE(evaluator.context().compute_batch(r,&error));
    REQUIRE(evaluator.context().runtime_stats().topology_uploads==4);
    // A changed index with identical CSR dimensions must upload new topology.
    csr.neighbors[0]=2;++r.topology_token.revision;
    REQUIRE(evaluator.context().compute_batch(r,&error));
    REQUIRE(evaluator.context().runtime_stats().topology_uploads==5);
    MTPGpuEvaluator fresh;REQUIRE(fresh.initialize_short_range(model,0,&error));
    auto reference_request=r;std::vector<double> fresh_f(f.size()),fresh_w(site.size());double fresh_e=0.,fresh_global[6]{};
    reference_request.force=fresh_f.data();reference_request.site_virial=fresh_w.data();
    reference_request.total_energy=&fresh_e;reference_request.global_virial=fresh_global;
    REQUIRE(fresh.context().compute_batch(reference_request,&error));
    for(size_t i=0;i<f.size();++i)REQUIRE(f[i]==Catch::Approx(fresh_f[i]).margin(2e-9));
    for(int d=0;d<6;++d)REQUIRE(global[d]==Catch::Approx(fresh_global[d]).margin(2e-9));
    REQUIRE(energy==Catch::Approx(fresh_e).margin(2e-9));
    csr.neighbors[0]=saved;++r.topology_token.revision;
    r.global_virial=nullptr;r.site_virial=nullptr;r.compute_global_energy=false;
    REQUIRE(evaluator.context().compute_batch(r,&error));
    for(int i=0;i<a.size();++i)for(int d=0;d<3;++d)REQUIRE(f[3*i+d]==Catch::Approx(expected.forces_[i][d]).margin(2e-9));
    std::swap(a.chemmap_[0],a.chemmap_[1]);++r.topology_token.revision;
    REQUIRE(evaluator.context().compute_batch(r,&error));
    expected=a;model.getEFS(expected);
    for(int i=0;i<a.size();++i)for(int d=0;d<3;++d)REQUIRE(f[3*i+d]==Catch::Approx(expected.forces_[i][d]).margin(2e-9));
    const auto typed=evaluator.context().runtime_stats();
    // Legacy index-only residency still refreshes chemistry even if opted in
    // with an invalid token, and geometry remains refreshed on every call.
    r.topology_token={};std::swap(a.chemmap_[0],a.chemmap_[1]);
    REQUIRE(evaluator.context().compute_batch(r,&error));
    expected=a;model.getEFS(expected);
    for(int i=0;i<a.size();++i)for(int d=0;d<3;++d)REQUIRE(f[3*i+d]==Catch::Approx(expected.forces_[i][d]).margin(2e-9));
    const auto untagged=evaluator.context().runtime_stats();
    REQUIRE(untagged.type_uploads==typed.type_uploads+1);
    REQUIRE(untagged.type_upload_bytes==typed.type_upload_bytes+a.size()*sizeof(int));
    REQUIRE(untagged.coordinate_uploads==typed.coordinate_uploads+1);
    r.topology_includes_types=false;r.topology_token={81,1};r.input_token={81,1};
    REQUIRE(evaluator.context().compute_batch(r,&error));
    const auto legacy=evaluator.context().runtime_stats();
    std::swap(a.chemmap_[0],a.chemmap_[1]);++r.input_token.revision;
    REQUIRE(evaluator.context().compute_batch(r,&error));
    expected=a;model.getEFS(expected);
    for(int i=0;i<a.size();++i)for(int d=0;d<3;++d)REQUIRE(f[3*i+d]==Catch::Approx(expected.forces_[i][d]).margin(2e-9));
    REQUIRE(evaluator.context().runtime_stats().topology_uploads==legacy.topology_uploads);
    REQUIRE(evaluator.context().runtime_stats().type_uploads==legacy.type_uploads+1);
}
#else
TEST_CASE("Native GPU requests reject a CPU-only library instead of falling back", "[native][gpu][packed]") {
    const auto path=std::filesystem::temp_directory_path()/
        ("native-no-gpu-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".mtp");
    struct Cleanup{std::filesystem::path path;~Cleanup(){std::filesystem::remove(path);}} cleanup{path};
    Potentials::PotMTP model;load_swap_fixture(model,2,false);model.Save(path.string());
    Potentials::NativeInferenceOptions options;options.pottype="mtp";
    for(const auto *backend:{"cuda","hip"}) {
        options.device=backend;
        REQUIRE_THROWS_WITH(Potentials::NativeInference(path.string(),options),
            Catch::Matchers::ContainsSubstring("no GPU backend; CPU fallback is disabled"));
    }
}
#endif
