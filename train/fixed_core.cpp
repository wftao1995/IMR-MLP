#include "fixed_core.h"
#include "potentials/potential_hybrid.h"
#include "potentials/potential_settings.h"
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace Training {
using namespace Potentials;
void validate_fixed_core(AnyPotential &residual, AnyPotential &core) {
    if (&residual==&core) GPERROR("Fixed core must be separate from the trainable residual");
    if (typeid(residual)!=typeid(PotMTP)) GPERROR("Fixed-core fitting currently requires an ordinary MTP residual");
    if (typeid(core)!=typeid(PotMTP) && typeid(core)!=typeid(PotentialZBL))
        GPERROR("Fixed core must be an existing ordinary MTP or ZBL potential");
    if (!core.inited() || residual.species_count!=core.species_count)
        GPERROR("Fixed core must be initialized with the same species count");
    if (typeid(core)==typeid(PotMTP)) {
        const auto left=static_cast<PotMTP&>(residual).potential_tag.elements();
        const auto right=static_cast<PotMTP&>(core).potential_tag.elements();
        if(!left.empty() && !right.empty() && left!=right)
            GPERROR("Fixed core and residual element mappings differ");
    }
}

std::vector<Atoms> fixed_core_residuals(AnyPotential &core,
    const std::vector<Atoms> &references, double hvp_step, double hessian_step) {
    struct Restore {AnyPotential &p;bool add;~Restore(){p.add_efs=add;}} restore{core,core.add_efs};
    core.add_efs=false;
    auto residuals=references;
    for (auto &a:residuals) {
        for(const int species:a.chemmap_)
            if(species<0 || species>=core.species_count) GPERROR("Fixed-core reference has an unmapped species");
        if (!a.cartesian) a.to_cartesian();
        auto prediction=a; prediction.resetEFS(); core.getEFS(prediction);
        if (a.has_energy()) a.energy_-=prediction.energy_;
        if (a.has_forces()) {
            double mean=0.;for(const auto &f:a.forces_) mean+=f.squaredNorm();
            a.global_attribs_d_["__imrmlp_fixed_core_force_norm2"]=mean/a.size();
            for(int i=0;i<a.size();++i) a.forces_[i]-=prediction.forces_[i];
        }
        if (a.has_stresses()) a.stress_-=prediction.stress_;
        if (a.has_site_energies()) {
            if (prediction.site_energies_.size()!=a.site_energies_.size())
                GPERROR("Fixed core did not provide the required site energies");
            for(int i=0;i<a.size();++i) a.site_energies_[i]-=prediction.site_energies_[i];
        }
        auto forces=[&](Atoms &frame,Neighborhoods &neighbors)->Eigen::MatrixXd {
            frame.resetEFS();core.getEFS(frame,neighbors);
            Eigen::MatrixXd f(3*frame.size(),1);
            for(int i=0;i<frame.size();++i) f.block<3,1>(3*i,0)=frame.forces_[i].transpose();
            return f;
        };
        if(a.has_hvp()) {
            a.validate_hvp();
            auto values=finite_difference_hvp_observations(a,hvp_step,core.cutoff,forces);
            for(int k=0;k<a.hvp_targets_.cols();++k)
                a.hvp_targets_.col(k)-=values.col(0).segment(3*a.size()*k,3*a.size());
        }
        if(a.has_hessian()) {
            auto probes=a;probes.hvp_directions_=Eigen::MatrixXd::Identity(3*a.size(),3*a.size());
            probes.hvp_targets_=a.hessian_;
            auto values=finite_difference_hvp_observations(probes,hessian_step,core.cutoff,forces);
            for(int k=0;k<a.hessian_.cols();++k)
                a.hessian_.col(k)-=values.col(0).segment(3*a.size()*k,3*a.size());
        }
    }
    return residuals;
}

void save_fixed_core_hybrid(AnyPotential &residual,AnyPotential &core,
    const std::string &path,const Atoms *reference) {
    validate_fixed_core(residual,core);
    if(path.empty()) GPERROR("Fixed-core hybrid output must name a file");
    namespace fs=std::filesystem;
    const auto destination=fs::absolute(path).lexically_normal();
    std::string pattern=(destination.parent_path()/".imrmlp-core-XXXXXX").string();
    std::vector<char> chars(pattern.begin(),pattern.end());chars.push_back('\0');
    if(!::mkdtemp(chars.data())) GPERROR("Cannot stage fixed-core hybrid output: "+path);
    struct Stage {fs::path p;~Stage(){std::error_code ec;fs::remove_all(p,ec);}} stage{chars.data()};
    const auto manifest=stage.p/destination.filename();
    const auto residual_name=destination.filename().string()+".residual.mtp";
    const auto core_name=destination.filename().string()+".core.mtp";
    residual.Save((stage.p/residual_name).string());
    GPSettings settings;settings.Modify("pottype","hybrid");settings.Modify("npot","2");
    settings.Modify("pot1:pottype","mtp");settings.Modify("pot1:potfile",residual_name);
    settings.Modify("pot1:train","true");settings.Modify("pot2:train","false");
    if(typeid(core)==typeid(PotMTP)) {
        core.Save((stage.p/core_name).string());
        settings.Modify("pot2:pottype","mtp");settings.Modify("pot2:potfile",core_name);
    } else {
        std::ostringstream stream;core.Save(stream);std::istringstream input(stream.str());
        auto values=read_potential_settings(input);
        for(const auto &[key,value]:values.settings)settings.Modify("pot2:"+key,value);
    }
    {std::ofstream out(manifest);for(const auto &[key,value]:settings.settings)out<<key<<' '<<value<<'\n';
        if(!out)GPERROR("Cannot write fixed-core hybrid manifest");}
    GPSettings load;load.Modify("potfile",manifest.string());PotentialHybrid restored(load);
    if(reference && reference->size()) {
        auto sample=*reference;if(!sample.cartesian) sample.to_cartesian();
        auto a=sample,b=sample,c=sample;
        a.resetEFS();b.resetEFS();c.resetEFS();
        struct Flags {AnyPotential &a,&b;bool x,y;~Flags(){a.add_efs=x;b.add_efs=y;}} flags{residual,core,residual.add_efs,core.add_efs};
        residual.add_efs=false;core.add_efs=false;residual.getEFS(a);core.getEFS(b);restored.getEFS(c);
        if(std::abs(c.energy_-a.energy_-b.energy_)>1e-9*(1.+std::abs(c.energy_)))
            GPERROR("Fixed-core hybrid reload changed total energy");
        for(int i=0;i<c.size();++i)
            if((c.forces_[i]-a.forces_[i]-b.forces_[i]).norm()>1e-8*(1.+c.forces_[i].norm()))
                GPERROR("Fixed-core hybrid reload changed forces");
        if((c.stress_-a.stress_-b.stress_).norm()>1e-8*(1.+c.stress_.norm()))
            GPERROR("Fixed-core hybrid reload changed virial");
    }
    for(const auto &entry:fs::directory_iterator(stage.p)) {
        auto final=destination.parent_path()/entry.path().filename();
        if(fs::exists(final) && !fs::is_regular_file(final)) GPERROR("Fixed-core output must be a regular file or new path");
    }
    for(const auto &entry:fs::directory_iterator(stage.p))
        if(entry.path()!=manifest) fs::rename(entry.path(),destination.parent_path()/entry.path().filename());
    fs::rename(manifest,destination);
}
}
