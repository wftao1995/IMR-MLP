#pragma once
#include "gpm_container.h"

namespace GPM {
inline std::string trim(std::string s) {
    auto a=s.find_first_not_of(" \t\r\n");if(a==std::string::npos)return {};
    return s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);
}
inline Object named_settings(const std::string &text) {
    Object out;std::istringstream in(text);std::string line;
    while(std::getline(in,line)) {
        line=trim(line);if(line.empty()||line[0]=='#')continue;
        auto pos=line.find_first_of("= \t");require(pos!=std::string::npos,"invalid named setting");
        auto key=line.substr(0,pos),value=trim(line.substr(pos+1));
        if(!value.empty()&&value[0]=='=')value=trim(value.substr(1));
        require(out.emplace(key,Value(value)).second,"duplicate setting "+key);
    }
    return out;
}
inline Value vector_block(Writer&w,const std::string &id,const std::vector<double>&v,std::vector<uint64_t> shape={}) {
    if(shape.empty())shape={v.size()};w.add(id,"f64",shape,encode_f64(v.data(),v.size()));return Value(Object{{"block",Value(id)}});
}
inline std::vector<double> values(const Reader&r,const Value &reference) {return decode_f64(r,string(at(object(reference),"block")));}
inline std::vector<int32_t> indices(const Reader&r,const Value &reference) {
    auto id=string(at(object(reference),"block"));require(r.info(id).dtype=="i32","expected index block");auto b=r.read(id);
    std::vector<int32_t> result(b.size()/4);for(size_t i=0;i<result.size();++i){uint32_t x=get_le(b.data()+i*4,4);std::memcpy(&result[i],&x,4);}return result;
}
inline Value index_block(Writer&w,const std::string&id,const std::vector<int32_t>&v,std::vector<uint64_t> shape) {
    Bytes b(v.size()*4);for(size_t i=0;i<v.size();++i)put_le(b.data()+4*i,static_cast<uint32_t>(v[i]),4);
    w.add(id,"i32",shape,std::move(b));return Value(Object{{"block",Value(id)}});
}
inline void print_basis(std::ostream &out,const Object &basis) {
    for(const auto *name:{"radial_basis_type","min_dist","max_dist","radial_basis_size"})out<<name<<" = "<<string(at(basis,name))<<'\n';
    // The configurable-basis parser accepts named cutoff/shift fields until the
    // next radial section. Preserve the dependency order of function selectors.
    std::set<std::string> done{"radial_basis_type","min_dist","max_dist","radial_basis_size","scaling"};
    for(const auto *name:{"rb_b","basis_function_type","cutoff_type","cutoff_power","cutoff_coefficients","cutoff_b","cutoff_r1","cutoff_r2","shift_type","shift_lambda"}) {
        auto it=basis.find(name);if(it!=basis.end()){out<<name<<" = "<<string(it->second)<<'\n';done.insert(name);}}
    for(const auto &p:basis)if(!done.count(p.first))out<<p.first<<" = "<<string(p.second)<<'\n';
}

// Adapter to the existing audited parser, also used by the standalone reader.
// The file itself contains named MTP fields and typed arrays, not a text model.
inline std::string mtp_text(const Reader &reader,const Object &node,bool standalone=false) {
    const auto &c=object(at(node,"configuration")), &p=object(at(node,"parameters"));
    int ns=integer(at(c,"species_count")), nf=integer(at(c,"radial_functions")), nb=integer(at(c,"radial_size"));
    int nm=integer(at(c,"moments_count"));require(ns>0&&nf>0&&nb>0&&nm>0,"invalid MTP dimensions");
    const auto &bases=array(at(c,"radial_bases"));require(bases.size()==1 || bases.size()==checked_mul(ns,ns),"invalid radial bases");
    auto radial=values(reader,at(p,"radial")),species=values(reader,at(p,"species")),moments=values(reader,at(p,"moments"));
    auto basic=indices(reader,at(p,"basic_indices")),times=indices(reader,at(p,"product_indices")),mapping=indices(reader,at(p,"scalar_mapping"));
    require(radial.size()==checked_mul(checked_mul(ns,ns),checked_mul(nf,nb)) && basic.size()%4==0 && times.size()%4==0,"invalid MTP array sizes");
    require(!mapping.empty()&&species.size()%ns==0&&moments.size()%mapping.size()==0,"invalid MTP coefficient dimensions");
    require(basic.size()/4<=size_t(nm),"too many basic moments");
    for(size_t i=0;i<basic.size();i+=4){require(basic[i]>=0&&basic[i]<nf,"radial index out of bounds");
        for(int k=1;k<4;++k)require(basic[i+k]>=0&&basic[i+k]<=128,"invalid angular degree");}
    for(size_t i=0;i<times.size();i+=4){require(times[i]>=0&&times[i]<nm&&times[i+1]>=0&&times[i+1]<nm&&times[i+3]>=int(basic.size()/4)&&times[i+3]<nm,"invalid product rule");
        require(times[i]<times[i+3]&&times[i+1]<times[i+3],"cyclic moment rule");}
    for(auto x:mapping)require(x>=0&&x<nm,"invalid scalar index");
    auto metadata=object(at(c,"metadata"));
    if(standalone) {
        metadata["gpm_format"]=Value(1.0);
        // The standalone text executor has one global scale. Fold each
        // ordered pair's scale into its radial coefficients in this adapter.
        for(int pair=0;pair<ns*ns;++pair){double scale=std::stod(string(at(object(bases[bases.size()==1?0:pair]),"scaling")));require(std::isfinite(scale),"nonfinite radial scale");
            for(int k=0;k<nf*nb;++k){auto &value=radial[size_t(pair)*nf*nb+k];value*=scale;require(std::isfinite(value),"radial scale overflow");}}
    }
    std::ostringstream out;out.imbue(std::locale::classic());out<<std::setprecision(17)<<"MTP\nversion = 1.1.0\npotential_name = GPM\nscaling = "
       <<(standalone?"1":string(at(object(bases[0]),"scaling")))<<"\nspecies_count = "<<ns<<"\npotential_tag = "<<Value(metadata).serialize()<<'\n';
    if(bases.size()>1)out<<"radial_basis_split = 1\n";
    for(size_t i=0;i<bases.size();++i){if(bases.size()>1)out<<"pair "<<i/ns<<' '<<i%ns<<'\n';print_basis(out,object(bases[i]));}
    out<<"radial_funcs_count = "<<nf<<"\nradial_coeffs\n";size_t j=0;
    for(int a=0;a<ns;++a)for(int b=0;b<ns;++b){out<<a<<'-'<<b<<'\n';for(int f=0;f<nf;++f){out<<'{';for(int k=0;k<nb;++k)out<<(k?", ":"")<<radial[j++];out<<"}\n";}}
    out<<"alpha_moments_count = "<<nm<<"\nalpha_index_basic_count = "<<basic.size()/4<<"\nalpha_index_basic = {";
    auto rules=[&](const std::vector<int32_t>&v){for(size_t i=0;i<v.size();i+=4){if(i)out<<", ";out<<'{';for(int k=0;k<4;++k)out<<(k?", ":"")<<v[i+k];out<<'}';}out<<"}\n";};
    rules(basic);out<<"alpha_index_times_count = "<<times.size()/4<<"\nalpha_index_times = {";rules(times);
    out<<"alpha_scalar_moments = "<<mapping.size()<<"\nalpha_moment_mapping = {";
    for(size_t i=0;i<mapping.size();++i)out<<(i?", ":"")<<mapping[i];out<<"}\n";
    auto coefficients=[&](const std::string&name,const std::vector<double>&v){out<<name<<" = {";for(size_t i=0;i<v.size();++i){require(std::isfinite(v[i]),"nonfinite MTP coefficient");out<<(i?", ":"")<<v[i];}out<<"}\n";};
    coefficients("species_coeffs",species);coefficients("moment_coeffs",moments);
    for(const auto &entry:array(at(c,"head_fields"))) {
        const auto &field=object(entry);auto name=string(at(field,"name"));
        if(field.count("array"))coefficients(name,values(reader,field.at("array")));
        else out<<name<<" = "<<string(at(field,"value"))<<'\n';
    }
    return out.str();
}
} // namespace GPM
