#pragma once

#include "radial_basis.h"
#if __has_include("mtp_metadata.h")
#include "mtp_metadata.h"
#else
#include "../../tools/mtp_metadata.h"
#endif
#include <sstream>

namespace MLIPCore::Radial {
inline constexpr const char *cache_marker = "standalone_radial_v1";

inline std::string trim(const std::string &s) {
    const auto a=s.find_first_not_of(" \r\n\t");
    return a==std::string::npos ? "" : s.substr(a,s.find_last_not_of(" \r\n\t")-a+1);
}
inline picojson::value read_cache(const std::string &text, bool &present) {
    present=false;
    picojson::value result;
    std::istringstream input(text); std::string line;
    while(std::getline(input,line)) {
        line=trim(line);
        if(line.rfind("#MVS_v1.1",0)==0) break;
        if(line.empty() || line[0]!='#') continue;
        line=trim(line.substr(1));
        if(line.rfind("standalone_radial_",0)!=0) continue;
        const auto equals=line.find('=');
        if(equals==std::string::npos || trim(line.substr(0,equals))!=cache_marker)
            radial_parse_error("unsupported standalone radial cache version");
        if(present) radial_parse_error("duplicate standalone radial cache");
        present=true;
        const auto error=picojson::parse(result,line.substr(equals+1));
        if(!error.empty() || !result.is<picojson::object>()) radial_parse_error("invalid standalone radial cache JSON");
    }
    return result;
}
inline const picojson::value &field(const picojson::object &o,const char *key) {
    const auto it=o.find(key);
    if(it==o.end()) radial_parse_error(std::string("radial cache missing ")+key);
    return it->second;
}
inline double number(const picojson::object &o,const char *key) {
    const auto &v=field(o,key);
    if(!v.is<double>() || !std::isfinite(v.get<double>())) radial_parse_error(std::string("invalid radial cache ")+key);
    return v.get<double>();
}
inline void match(const picojson::object &o,const char *key,double expected) {
    if(number(o,key)!=expected) radial_parse_error(std::string("radial cache definition mismatch: ")+key);
}
inline std::vector<double> array(const picojson::object &o,const char *key,std::size_t size) {
    const auto &v=field(o,key);
    if(!v.is<picojson::array>() || v.get<picojson::array>().size()!=size)
        radial_parse_error(std::string("radial cache dimension mismatch: ")+key);
    std::vector<double> result;result.reserve(size);
    for(const auto &x:v.get<picojson::array>()) {
        if(!x.is<double>() || !std::isfinite(x.get<double>())) radial_parse_error("non-finite radial cache data");
        result.push_back(x.get<double>());
    }
    return result;
}
// Pointers refer to one non-split block or species_count^2 ordered blocks.
// A present cache must describe every PolyB pair, including non-split replicas.
inline bool apply_cache(const std::string &text,const std::vector<mlip::radial_basis_t *> &bases,
                        bool split,int species_count) {
    bool present=false;
    const auto cache=read_cache(text,present);
    if(!present) return false;
    const auto &root=cache.get<picojson::object>();
    match(root,"species_count",species_count);match(root,"split",split?1:0);
    const auto &entries=field(root,"pairs");
    if(!entries.is<picojson::array>()) radial_parse_error("invalid radial cache pairs");
    const std::size_t pairs=std::size_t(species_count)*species_count;
    std::vector<bool> seen(pairs,false);
    for(const auto &entry:entries.get<picojson::array>()) {
        if(!entry.is<picojson::object>()) radial_parse_error("invalid radial cache pair");
        const auto &o=entry.get<picojson::object>();
        const double c=number(o,"center"),n=number(o,"neighbor");
        if(c<0 || n<0 || c>=species_count || n>=species_count || c!=std::floor(c) || n!=std::floor(n))
            radial_parse_error("invalid radial cache species pair");
        const auto pair=std::size_t(c)*species_count+std::size_t(n);
        if(seen[pair]) radial_parse_error("duplicate radial cache species pair");
        seen[pair]=true;
        auto &b=*bases[split?pair:0];
        if(!is_poly(b)) radial_parse_error("radial cache attached to a non-PolyB basis");
        match(o,"type",b.rb);match(o,"size",b.size);match(o,"min_dist",b.min_dist);
        match(o,"max_dist",b.max_dist);match(o,"scaling",b.scaling);
        const auto size=std::size_t(b.size);
        if(size>std::size_t(100000000)/size) radial_parse_error("PolyB matrix exceeds allocation limit");
        auto matrix=array(o,"matrix",size*size),normalizations=array(o,"normalizations",size);
        for(double x:normalizations) if(!(x>0)) radial_parse_error("invalid PolyB normalization");
        if(!split && !b.polynomial_matrix.empty() &&
           (b.polynomial_matrix!=matrix || b.polynomial_normalizations!=normalizations))
            radial_parse_error("conflicting non-split PolyB cache replicas");
        b.polynomial_matrix=std::move(matrix);b.polynomial_normalizations=std::move(normalizations);
    }
    for(std::size_t pair=0;pair<pairs;++pair)
        if(is_poly(*bases[split?pair:0])!=seen[pair]) radial_parse_error("incomplete PolyB radial cache");
    return true;
}
inline void finish_bases(const std::string &text,const std::vector<mlip::radial_basis_t *> &bases,
                         bool split,int species_count) {
    apply_cache(text,bases,split,species_count);
    for(auto *b:bases) if(is_poly(*b) && b->polynomial_matrix.empty()) initialize_polynomial(*b);
}
inline picojson::value doubles(const std::vector<double> &values) {
    picojson::array result;
    for(double x:values) {
        if(!std::isfinite(x)) radial_parse_error("cannot export non-finite PolyB data");
        result.emplace_back(x);
    }
    return picojson::value(result);
}
inline std::string cache_footer(const std::vector<mlip::radial_basis_t *> &bases,bool split,int species_count) {
    picojson::array entries;
    for(int i=0;i<species_count;++i) for(int j=0;j<species_count;++j) {
        const auto &b=*bases[split?std::size_t(i)*species_count+j:0];
        if(!is_poly(b)) continue;
        picojson::object o;
        o["center"]=picojson::value(double(i));o["neighbor"]=picojson::value(double(j));
        o["type"]=picojson::value(double(b.rb));o["size"]=picojson::value(double(b.size));
        o["min_dist"]=picojson::value(b.min_dist);o["max_dist"]=picojson::value(b.max_dist);
        o["scaling"]=picojson::value(b.scaling);
        o["matrix"]=doubles(b.polynomial_matrix);o["normalizations"]=doubles(b.polynomial_normalizations);
        entries.emplace_back(o);
    }
    picojson::object root;
    root["species_count"]=picojson::value(double(species_count));
    root["split"]=picojson::value(double(split));root["pairs"]=picojson::value(entries);
    return std::string("# ")+cache_marker+" = "+picojson::value(root).serialize()+"\n";
}
} // namespace MLIPCore::Radial
