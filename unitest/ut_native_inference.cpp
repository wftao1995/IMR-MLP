#include <catch2/catch_all.hpp>
#include "potentials/native_inference.h"
#include "potentials/mtp_gpu_adapter.h"
#include "unitest/maxvol_mtp_fixture.h"
#include <chrono>

using namespace Potentials;
using Catch::Approx;

namespace {
struct Fixture {
    std::filesystem::path directory=std::filesystem::temp_directory_path()/
        ("native-inference-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::vector<MaxVolMTPCase> cases=make_maxvol_mtp_models(directory);
    ~Fixture(){std::filesystem::remove_all(directory);}
    std::string settings(const MaxVolMTPCase &entry) const {
        const auto file=(directory/(entry.name+".ini")).string();
        std::ofstream out(file);
        for(const auto &[key,value]:entry.settings.settings) out<<"pot:"<<key<<" "<<value<<'\n';
        return file;
    }
};

struct PackedNeighborhoods {
    std::vector<int> centers,offsets,neighbors;
    std::vector<double> positions,vectors,distances;
    NativePackedNeighborhoodView view;
    PackedNeighborhoods(const Atoms &atoms,const AnyNeighborhoods &nbhs) {
        for(int i=0;i<atoms.size();++i)
            for(int axis=0;axis<3;++axis)positions.push_back(atoms.pos_[i][axis]);
        offsets.push_back(0);
        for(int i=0;i<nbhs.size();++i) {
            centers.push_back(i);
            for(int j=0;j<nbhs[i].nbhcount_;++j) {
                neighbors.push_back(nbhs[i].indices_[j]);
                distances.push_back(nbhs[i].distances_[j]);
                for(int axis=0;axis<3;++axis)vectors.push_back(nbhs[i].pos_[j][axis]);
            }
            offsets.push_back(neighbors.size());
        }
        view.owned_count=nbhs.size();view.total_count=atoms.size();view.edge_count=neighbors.size();
        view.centers=centers.data();view.offsets=offsets.data();view.neighbors=neighbors.data();
        view.types=atoms.chemmap_.data();view.positions=positions.data();
        view.vectors=vectors.data();view.distances=distances.data();
        view.topology_owner=reinterpret_cast<std::uintptr_t>(this);view.topology_revision=1;
    }
};

void compare(const NativeInferenceResult &got,const Atoms &expected) {
    REQUIRE(got.energy==Approx(expected.energy_).margin(2e-8).epsilon(2e-9));
    REQUIRE(got.site_energy.sum()==Approx(got.energy).margin(2e-8));
    for(int i=0;i<expected.size();++i) {
        REQUIRE(got.site_energy[i]==Approx(expected.site_energies_[i]).margin(2e-8));
        for(int a=0;a<3;++a) REQUIRE(got.forces(i,a)==Approx(expected.forces_[i][a]).margin(2e-8));
    }
    REQUIRE((got.virial-expected.stress_).norm()<2e-8);
    const auto w=got.site_virial.colwise().sum().eval();
    REQUIRE(w[0]==Approx(got.virial(0,0)).margin(2e-8));
    REQUIRE(w[1]==Approx(got.virial(1,1)).margin(2e-8));
    REQUIRE(w[2]==Approx(got.virial(2,2)).margin(2e-8));
    REQUIRE(w[3]==Approx(.5*(got.virial(0,1)+got.virial(1,0))).margin(2e-8));
    REQUIRE(w[4]==Approx(.5*(got.virial(0,2)+got.virial(2,0))).margin(2e-8));
    REQUIRE(w[5]==Approx(.5*(got.virial(1,2)+got.virial(2,1))).margin(2e-8));
}
}

TEST_CASE("Native deployment evaluates every original trainer MTP family", "[native][deployment]") {
    std::string device="cpu";
#ifdef IMRMLP_GPU
    const bool gpu=GENERATE(false,true);
    if(gpu) {
        int count=0;std::string error;
        const bool available=MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count,&error)&&count>0;
        if(std::getenv("IMRMLP_REQUIRE_GPU")){INFO(error);REQUIRE(available);}
        if(!available)SKIP("No GPU available for native deployment acceptance");
        device=MLIPCore::GpuDirect::MTPGpuDirect::backend_name();
    }
#endif
    Fixture fixture;
    for(const auto &entry:fixture.cases) {
      DYNAMIC_SECTION(device << "/" << entry.name) {
        INFO(entry.name);
        NativeInferenceOptions options;options.pottype=entry.pottype;options.settings_file=fixture.settings(entry);options.device=device;
        NativeInference inference(entry.path,options);
        auto atoms=maxvol_mtp_atoms();
        atoms.cell_ << 9.0,0,0, .4,9.1,0, .2,.5,9.2;
        potential_wrapper wrapper(entry.settings);
        auto expected=atoms;wrapper.pot->getEFS(expected);
        const auto got=inference.evaluate(atoms);
        compare(got,expected);
        Neighborhoods neighborhoods(atoms,inference.cutoff());
        const auto distributed=inference.evaluate_distributed(atoms,neighborhoods,NativeCollectives{});
        compare(distributed,expected);
        REQUIRE((distributed.site_virial-got.site_virial).norm()<2e-8);
        REQUIRE((distributed.charges-got.charges).norm()<1e-10);
        REQUIRE((distributed.latents-got.latents).norm()<1e-10);
        // Explicit displacements retain periodic images whose indices point
        // at the same owned atom. Exercise the borrowed API for every family.
        PackedNeighborhoods packed(atoms,neighborhoods);
        const auto borrowed=inference.evaluate_distributed_packed(atoms,packed.view,NativeCollectives{});
        compare(borrowed,expected);
        REQUIRE((borrowed.site_virial-got.site_virial).norm()<2e-8);
        REQUIRE((borrowed.charges-got.charges).norm()<1e-10);
        REQUIRE((borrowed.latents-got.latents).norm()<1e-10);
        if (device=="cpu") {
            auto invalid_types=atoms.chemmap_;
            invalid_types[0]=inference.species_count();
            auto invalid_view=packed.view;invalid_view.types=invalid_types.data();
            REQUIRE_THROWS_WITH(inference.evaluate_distributed_packed(atoms,invalid_view,NativeCollectives{}),
                Catch::Matchers::ContainsSubstring("packed atom type outside the model species range"));
            if (packed.view.edge_count) {
                auto invalid_vectors=packed.vectors;
                std::fill_n(invalid_vectors.data(),3,0.);
                invalid_view=packed.view;invalid_view.vectors=invalid_vectors.data();
                REQUIRE_THROWS_WITH(inference.evaluate_distributed_packed(atoms,invalid_view,NativeCollectives{}),
                    Catch::Matchers::ContainsSubstring("invalid packed neighbor geometry"));
            }
        }
        if(inference.has_charges()) {
            REQUIRE(got.charges.size()==atoms.size());
            for(int i=0;i<atoms.size();++i)
                REQUIRE(got.charges[i]==Approx(expected.charges_[i]).margin(1e-10));
        }
        if(inference.latent_channels()) {
            REQUIRE(got.charges.size()==0);
            REQUIRE(got.latents.rows()==atoms.size());
            REQUIRE(got.latents.cols()==inference.latent_channels());
        }
        NativeInferenceRequest request;
        request.energy=false;request.site_energy=false;request.virial=false;request.site_virial=false;
        const auto force_only=inference.evaluate(atoms,request);
        REQUIRE(force_only.energy==0.);
        REQUIRE(force_only.site_energy.size()==0);
        REQUIRE(force_only.site_virial.rows()==0);
        REQUIRE((force_only.forces-got.forces).norm()<1e-10);
        for(int field=-1;field<7;++field) {
            NativeInferenceRequest one;
            one.forces=field==0;one.energy=field==1;one.site_energy=field==2;
            one.virial=field==3;one.site_virial=field==4;one.charges=field==5;one.latents=field==6;
            const auto actual=inference.evaluate_distributed(atoms,neighborhoods,NativeCollectives{},one);
            REQUIRE(actual.forces.rows()==(one.forces?atoms.size():0));
            if(one.forces)REQUIRE((actual.forces-got.forces).norm()<2e-8);
            REQUIRE(actual.energy==Approx(one.energy?got.energy:0.).margin(2e-8));
            REQUIRE(actual.site_energy.size()==(one.site_energy?atoms.size():0));
            if(one.site_energy)REQUIRE((actual.site_energy-got.site_energy).norm()<2e-8);
            REQUIRE((actual.virial-(one.virial?got.virial:Eigen::Matrix3d::Zero().eval())).norm()<2e-8);
            REQUIRE(actual.site_virial.rows()==(one.site_virial?atoms.size():0));
            if(one.site_virial)REQUIRE((actual.site_virial-got.site_virial).norm()<2e-8);
            REQUIRE(actual.charges.size()==(one.charges?got.charges.size():0));
            REQUIRE(actual.latents.size()==(one.latents?got.latents.size():0));
            // Alternate uncached and cached callers and output requests on
            // one evaluator; previous force/energy buffers must not leak.
            const auto borrowed_one=inference.evaluate_distributed_packed(atoms,packed.view,NativeCollectives{},one);
            REQUIRE(borrowed_one.forces.rows()==actual.forces.rows());
            REQUIRE(borrowed_one.site_energy.size()==actual.site_energy.size());
            REQUIRE(borrowed_one.site_virial.rows()==actual.site_virial.rows());
            REQUIRE(borrowed_one.charges.size()==actual.charges.size());
            REQUIRE(borrowed_one.latents.rows()==actual.latents.rows());
            REQUIRE(borrowed_one.latents.cols()==actual.latents.cols());
            REQUIRE(borrowed_one.energy==Approx(actual.energy).margin(2e-8));
            REQUIRE((borrowed_one.forces-actual.forces).norm()<2e-8);
            REQUIRE((borrowed_one.virial-actual.virial).norm()<2e-8);
            REQUIRE((borrowed_one.site_energy-actual.site_energy).norm()<2e-8);
            REQUIRE((borrowed_one.site_virial-actual.site_virial).norm()<2e-8);
            REQUIRE((borrowed_one.charges-actual.charges).norm()<1e-10);
            REQUIRE((borrowed_one.latents-actual.latents).norm()<1e-10);
        }
      }
    }
}

TEST_CASE("Self describing MTP-M deployment and loading failures", "[native][metadata]") {
    Fixture fixture;
    for(const auto &entry:fixture.cases) if(entry.pottype=="mtp_m") {
        INFO(entry.name);
        NativeInferenceOptions options;options.pottype="mtp_m";
        NativeInference inference(entry.path,options);
        auto atoms=maxvol_mtp_atoms();auto expected=atoms;
        potential_wrapper wrapper(entry.settings);wrapper.pot->getEFS(expected);
        compare(inference.evaluate(atoms),expected);
        options.pottype="mtp";
        REQUIRE_THROWS(NativeInference(entry.path,options));
    }
    NativeInferenceOptions invalid;invalid.pottype="unknown";
    REQUIRE_THROWS(NativeInference("missing",invalid));
#ifndef IMRMLP_GPU
    invalid.pottype="mtp";invalid.device="cuda";
    REQUIRE_THROWS_WITH(NativeInference((fixture.directory/"base.mtp").string(),invalid),
        Catch::Matchers::ContainsSubstring("CPU fallback is disabled"));
#endif
    for(const auto &entry:fixture.cases)if(entry.name=="sog") {
        const auto conflict=fixture.directory/"conflict.ini";
        std::ofstream(conflict)<<"pot:les_kernel coulomb\n";
        NativeInferenceOptions options;options.pottype=entry.pottype;options.settings_file=conflict.string();
        REQUIRE_THROWS_WITH(NativeInference(entry.path,options),Catch::Matchers::ContainsSubstring("conflicts with model metadata"));
        std::ofstream(conflict)<<"pot:sog_channels 7\n";
        REQUIRE_THROWS_WITH(NativeInference(entry.path,options),Catch::Matchers::ContainsSubstring("conflicts with model metadata"));
        std::ofstream(conflict)<<"pot:lr_les false\n";
        REQUIRE_THROWS_WITH(NativeInference(entry.path,options),Catch::Matchers::ContainsSubstring("conflicts with model metadata"));
    }
}
