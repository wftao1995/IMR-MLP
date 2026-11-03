#pragma once
#include "potentials/potential.h"
namespace MTPWorkflow {
struct Adjustment {
    std::vector<std::string> appended,elements;
    std::vector<int> sources,order;
    std::string initialization,basis="keep";
};
std::unique_ptr<Potentials::AnyPotential> adjust_tree(Potentials::AnyPotential&,const Adjustment&);
void initialize_model(GPSettings,std::string input,std::string output,std::string type,
    const std::vector<std::string>& elements,bool reset,bool randomize,double lower,double upper,
    std::string dataset,const std::string& format,const std::string& index,bool fit,bool normalize,
    double scale,bool explicit_index,const std::string& offset_target);
}
