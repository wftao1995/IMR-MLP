#include <catch2/catch_all.hpp>
#include "potentials/native_inference.h"
#include "potentials/mtp_gpu_adapter.h"
#include "unitest/maxvol_mtp_fixture.h"
#include <chrono>

using namespace Potentials;

namespace {
struct Models {
    std::filesystem::path directory=std::filesystem::temp_directory_path()/
        ("native-fd-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::vector<MaxVolMTPCase> entries=make_maxvol_mtp_models(directory);
    ~Models(){std::filesystem::remove_all(directory);}
};
}

TEST_CASE("Native family forces and cell virials converge under finite differences",
          "[native][finite-difference]") {
    std::string device="cpu";
#ifdef IMRMLP_GPU
    if(GENERATE(false,true)) {
        int count=0;std::string error;
        const bool available=MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count,&error)&&count>0;
        if(std::getenv("IMRMLP_REQUIRE_GPU")){INFO(error);REQUIRE(available);}
        if(!available)SKIP("GPU finite-difference acceptance requires a device");
        device=MLIPCore::GpuDirect::MTPGpuDirect::backend_name();
    }
#endif
    Models models;
    for(const auto &entry:models.entries)DYNAMIC_SECTION(device<<"/"<<entry.name) {
        const auto settings=models.directory/(entry.name+".ini");
        {std::ofstream out(settings);for(const auto &[key,value]:entry.settings.settings)out<<"pot:"<<key<<" "<<value<<'\n';}
        NativeInferenceOptions options;options.pottype=entry.pottype;options.device=device;options.settings_file=settings.string();
        NativeInference model(entry.path,options);
        auto atoms=maxvol_mtp_atoms();atoms.cell_<<9.,0.,0.,.4,9.1,0.,.2,.5,9.2;
        const auto reference=model.evaluate(atoms);
        NativeInferenceRequest energy_only;energy_only.forces=false;energy_only.site_energy=false;
        energy_only.virial=false;energy_only.site_virial=false;energy_only.charges=false;energy_only.latents=false;
        for(int component=0;component<9;++component) {
            INFO("coordinate/strain component="<<component);
            double previous_error=0.;
            for(double step:{1e-3,3e-4,1e-4}) {
                auto plus=atoms,minus=atoms;double expected;
                if(component<3) {
                    plus.pos_[0](component)+=step;minus.pos_[0](component)-=step;
                    expected=reference.forces(0,component);
                } else {
                    const int axes[6][2]={{0,0},{1,1},{2,2},{0,1},{0,2},{1,2}};
                    const int a=axes[component-3][0],b=axes[component-3][1];
                    Eigen::Matrix3d strain=Eigen::Matrix3d::Zero();
                    strain(a,b)=a==b?1.:.5;if(a!=b)strain(b,a)=.5;
                    const Eigen::Matrix3d p=Eigen::Matrix3d::Identity()+step*strain;
                    const Eigen::Matrix3d m=Eigen::Matrix3d::Identity()-step*strain;
                    plus.cell_=atoms.cell_*p;minus.cell_=atoms.cell_*m;
                    for(int i=0;i<atoms.size();++i){plus.pos_[i]=atoms.pos_[i]*p;minus.pos_[i]=atoms.pos_[i]*m;}
                    expected=.5*(reference.virial(a,b)+reference.virial(b,a));
                }
                const double actual=-(model.evaluate(plus,energy_only).energy-model.evaluate(minus,energy_only).energy)/(2.*step);
                const double error=std::abs(actual-expected);
                CAPTURE(step,actual,expected,error);
                if(step<1e-3) {
                    REQUIRE(error<=2e-6+2e-5*std::abs(expected));
                    // Central-difference truncation decreases quadratically;
                    // allow a roundoff floor when the derivative is tiny.
                    REQUIRE(error<=.35*previous_error+2e-7);
                }
                previous_error=error;
            }
        }
    }
}
