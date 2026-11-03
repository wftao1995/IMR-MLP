#pragma once

#include "potentials/model_bundle.h"
#include "potentials/potential.h"
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace Training {
// Training invalidates any AL binding carried by the input model. Ordinary
// Potential::Save keeps its stricter preserve semantics outside this workflow.
inline void save_trained_model(Potentials::AnyPotential &model,const std::string &path) {
    if(GPM::extension(path))
        Potentials::ModelBundle::save_potential(model,path,Potentials::ModelBundle::Profile::Inference);
    else model.Save(path);
}

inline std::string model_output_suffix(const std::string &path,const std::string &suffix) {
    if(!GPM::extension(path))return path+suffix;
    auto file=std::filesystem::path(path);
    return (file.parent_path()/(file.stem().string()+suffix+file.extension().string())).string();
}

inline std::string model_checkpoint(const std::string &base,int iteration,const std::string &legacy) {
    if(!GPM::extension(base))return legacy;
    std::ostringstream suffix;suffix<<".iter-"<<std::setw(6)<<std::setfill('0')<<iteration;
    return model_output_suffix(base,suffix.str());
}
}
