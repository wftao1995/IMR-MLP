#pragma once

#include "conditional_linear.h"
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/potential_mtpnn1.h"
#include "potentials/potential_descriptor_linear.h"
#include "potentials/potential_hybrid_scale.h"
#include <filesystem>
#include <unistd.h>

namespace Training {
namespace ConditionalPersistence {
template<class Model>
std::unique_ptr<Potentials::AnyPotential> reload_mtpm(Potentials::AnyPotential &source,
                                                    const std::string &path) {
    using Potential=Potentials::PotMTP_M<Model>;
    const auto &p=static_cast<const Potential&>(source);
    GPSettings settings;
    if constexpr (std::is_same_v<Model,Potentials::m_model_nn1> ||
                  std::is_same_v<Model,Potentials::m_model_nn2>) {
        settings.Modify("n_input",std::to_string(p.model.n_input));
        settings.Modify("n_neuron",std::to_string(p.model.n_neuron));
        if constexpr (std::is_same_v<Model,Potentials::m_model_nn1>)
            settings.Modify("activation_function",Potential::serialized_activation_name(p.model.acfunc_enum));
        else {
            settings.Modify("n_neuron2",std::to_string(p.model.n_neuron2));
            settings.Modify("activation_function",Potential::serialized_activation_name(p.model.nn.acfunc));
        }
    }
    return std::make_unique<Potential>(path,settings);
}
inline std::unique_ptr<Potentials::AnyPotential> reload(Potentials::AnyPotential &source,
                                                       const std::string &path) {
    using namespace Potentials;
    if(typeid(source)==typeid(PotMTP)) return std::make_unique<PotMTP>(path);
    if(typeid(source)==typeid(PotMTP_LES)) return std::make_unique<PotMTP_LES>(path);
#define IMRMLP_CONDITIONAL_RELOAD_MODEL(Model) \
    if(typeid(source)==typeid(PotMTP_M<Model>)) return reload_mtpm<Model>(source,path);
    IMRMLP_CONDITIONAL_RELOAD_MODEL(m_model_mtpli)
    IMRMLP_CONDITIONAL_RELOAD_MODEL(m_model_mtpq2)
    IMRMLP_CONDITIONAL_RELOAD_MODEL(m_model_mtpfs)
    IMRMLP_CONDITIONAL_RELOAD_MODEL(m_model_nn1)
    IMRMLP_CONDITIONAL_RELOAD_MODEL(m_model_nn2)
#undef IMRMLP_CONDITIONAL_RELOAD_MODEL
    if(typeid(source)==typeid(PotMTPNN1)) return std::make_unique<PotMTPNN1>(path);
    if(typeid(source)==typeid(PotentialDescriptorLinear)) return std::make_unique<PotentialDescriptorLinear>(path);
    if(typeid(source)==typeid(PotentialHybridScale)) {
        GPSettings settings;settings.Modify("potfile",path);
        return std::make_unique<PotentialHybridScale>(settings);
    }
    GPERROR("Conditional refit publication requires an explicitly supported native serializer");
}
}

inline std::vector<std::string> conditional_linear_files(Potentials::AnyPotential &potential,
                                                        const std::string &path) {
    using namespace Potentials;
    if(typeid(potential)==typeid(PotMTPNN1)) return {path+".mtp",path+".nn",path+".sc"};
    std::vector<std::string> files{path};
    if(typeid(potential)==typeid(PotentialHybridScale))
        for(int i=0;i<static_cast<PotentialHybridScale&>(potential).npot;++i)
            files.push_back(path+".pot"+std::to_string(i+1));
    return files;
}

// Final/checkpoint publication uses each model's native serializer. All sidecar
// files are staged and reloaded before any destination file is replaced. Inner
// least-squares probes never perform this disk roundtrip.
inline void publish_conditional_linear(Potentials::AnyPotential &potential,
                                      const std::string &path, const Atoms *reference=nullptr) {
    namespace fs=std::filesystem;
    const fs::path destination=fs::absolute(path).lexically_normal();
    if(path.empty() || destination.filename().empty()) GPERROR("Conditional model output must name a file or native prefix");
    std::string pattern=(destination.parent_path()/".imrmlp-affine-save-XXXXXX").string();
    std::vector<char> name(pattern.begin(),pattern.end());name.push_back('\0');
    if(!::mkdtemp(name.data())) GPERROR("Cannot create conditional model staging directory: "+path);
    struct Stage {fs::path directory;~Stage(){std::error_code error;fs::remove_all(directory,error);}} stage{fs::path(name.data())};
    const auto temporary=stage.directory/destination.filename();
    potential.Save(temporary.string());
    std::vector<fs::path> files;
    for(const auto &entry:fs::directory_iterator(stage.directory)) {
        if(!fs::is_regular_file(entry.symlink_status()) || fs::file_size(entry.path())==0)
            GPERROR("Conditional native serializer did not produce a nonempty regular file");
        files.push_back(entry.path());
    }
    if(files.empty()) GPERROR("Conditional native serializer did not produce model files");
    const auto expected=conditional_linear_files(potential,temporary.string());
    if(files.size()!=expected.size()) GPERROR("Conditional native serializer produced an incomplete model bundle");
    for(const auto &file:expected)
        if(std::find(files.begin(),files.end(),fs::path(file))==files.end())
            GPERROR("Conditional native serializer omitted a model bundle member");
    auto restored=ConditionalPersistence::reload(potential,temporary.string());
    if(reference && reference->size()) {
        auto original=*reference,copy=*reference;
        Neighborhoods original_neighbors(original,potential.cutoff),copy_neighbors(copy,restored->cutoff);
        original.resetEFS();copy.resetEFS();potential.getEFS(original,original_neighbors);restored->getEFS(copy,copy_neighbors);
        const auto before=ConditionalLinearAdapter::efs_values(original);
        const auto after=ConditionalLinearAdapter::efs_values(copy);
        if(!before.allFinite() || !after.allFinite() || (before-after).norm()>1e-10*std::max(1.,before.norm()))
            GPERROR("Conditional native model roundtrip prediction mismatch");
    }
    // Check every destination before publishing the first member of a bundle.
    for(const auto &file:files) {
        const auto target=destination.parent_path()/file.filename();std::error_code error;
        const auto status=fs::symlink_status(target,error);
        if(error && error!=std::errc::no_such_file_or_directory) GPERROR("Cannot inspect conditional model output: "+target.string());
        if(status.type()!=fs::file_type::not_found && !fs::is_regular_file(status))
            GPERROR("Conditional model output must be a regular file or new path: "+target.string());
    }
    // Children precede a manifest when there is one. A reader of the manifest
    // then sees a complete, verified native bundle.
    std::stable_sort(files.begin(),files.end(),[&](const fs::path &a,const fs::path &b){
        return (a.filename()!=destination.filename())>(b.filename()!=destination.filename());
    });
    for(const auto &file:files) fs::rename(file,destination.parent_path()/file.filename());
}
} // namespace Training
