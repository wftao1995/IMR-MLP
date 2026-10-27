#include "standalone_export.h"
#include "potential_mtp.h"
#include "runtime/mtp/radial_cache.h"
#include <memory>

namespace Potentials {
std::string export_standalone_radial(const std::string &text) {
    namespace R=MLIPCore::Radial;
    std::istringstream lines(text);std::string line,body;
    while(std::getline(lines,line)) {
        auto content=R::trim(line);
        if(content.rfind("#MVS_v1.1",0)==0) break;
        if(!content.empty() && content[0]=='#' && R::trim(content.substr(1)).rfind("standalone_radial_",0)==0) continue;
        body+=line+'\n';
    }
    // Validate all head dimensions, long-range restrictions and existing cache.
    double cutoff=0;int mapping[1]={-1};
    std::unique_ptr<MLIPCore::mlip,decltype(&MLIPCore::destroy_mlip)> checked(
        MLIPCore::setup_from_text(text,mapping,&cutoff,1),MLIPCore::destroy_mlip);
    // Base parsing owns the radial coefficients; head coefficients stay in the
    // original text. The derived type permits MTP_M metadata in this radial view.
    struct RadialView : PotMTP {
        std::string metadata_model_type() const override { return "MTP_M"; }
    } native;
    native.coeffs_handled_external=true;
    native.read_flag_immediate_return=true;
    std::istringstream input(body);native.Load(input);
    const bool split=native.radial_basis_split;
    std::vector<MLIPCore::mlip::radial_basis_t> bases(split?native.species_count*native.species_count:1);
    std::vector<MLIPCore::mlip::radial_basis_t *> pointers;
    for(std::size_t i=0;i<bases.size();++i) {
        auto &b=bases[i];
        const AnyBasis *source=split?native.p_radical_basis_pair[i]:native.p_radical_basis;
        b.size=source->basis_size;b.min_dist=source->min_val;b.max_dist=source->max_val;b.scaling=source->scaling;
        if(const auto *poly=dynamic_cast<const RBasis_PolyB *>(source)) {
            b.rb=MLIPCore::mlip::RB_POLYB;
            b.polynomial_matrix.assign(poly->coeffs.data(),poly->coeffs.data()+b.size*b.size);
            b.polynomial_normalizations=poly->invNalpha;
        } else if(const auto *poly=dynamic_cast<const RBasis_PolyBCos *>(source)) {
            b.rb=MLIPCore::mlip::RB_POLYBCOS;
            b.polynomial_matrix.assign(poly->coeffs.data(),poly->coeffs.data()+b.size*b.size);
            b.polynomial_normalizations=poly->invNalpha;
        }
        pointers.push_back(&b);
    }
    bool present=false;R::read_cache(text,present);
    if(present) {
        for(auto &b:bases) {b.polynomial_matrix.clear();b.polynomial_normalizations.clear();}
        R::apply_cache(text,pointers,split,native.species_count);
    }
    return body+R::cache_footer(pointers,split,native.species_count);
}
}
