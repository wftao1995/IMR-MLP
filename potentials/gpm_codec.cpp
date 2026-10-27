#include "gpm_codec.h"
#include "tools/gpm_mtp.h"
#include "potential_mtp.h"
#include "potential_mtp_m.h"
#include "potential_mtp_LES.h"
#include "potential_mtpnn1.h"
#include "potential_lj.h"
#include "potential_zbl.h"
#include "potential_mindist.h"
#include "potential_sw.h"
#include "potential_tersoff.h"
#include "potential_pair_table.h"
#include "potential_threebody_table.h"
#include "potential_eamalloy_spline.h"
#include "potential_eamfs_spline.h"
#include "potential_eam_poly.h"
#include "potential_hybrid.h"
#include "potential_hybrid_scale.h"
#include "potential_hybmtp.h"
#include "potential_descriptor_linear.h"
#include "potential_settings.h"
#include "potential_shell.h"
#include "potential_openkim.h"
#include "ace_lmp.h"
#include "descriptors/descriptor_MTM.h"
#include <typeindex>

namespace Potentials {
using namespace GPM;
namespace {
struct TemporaryResources {
    std::filesystem::path directory;
    TemporaryResources(){std::string p=(std::filesystem::temp_directory_path()/"gpml-gpm-XXXXXX").string();require(::mkdtemp(p.data()),"cannot create resource directory");directory=p;}
    ~TemporaryResources(){std::error_code ec;std::filesystem::remove_all(directory,ec);}
    std::string write(const std::string&name,const std::string&bytes){auto path=directory/name;std::ofstream out(path,std::ios::binary);out.write(bytes.data(),bytes.size());out.close();require(bool(out),"cannot extract resource");return path.string();}
};
void common(State &a,AnyPotential &p) {
    a.field("species_count",p.species_count);a.field("cutoff",p.cutoff);
    a.field("cutoffs",p.cutoffs);a.field("r_mins",p.r_mins);a.field("add_efs",p.add_efs);
    a.field("reg_L1",p.reg_L1_vector);a.field("reg_L2",p.reg_L2_vector);
    if(a.reading()) {
        require(p.species_count>=0&&p.species_count<=100000&&std::isfinite(p.cutoff)&&p.cutoff>=0,"invalid potential dimensions");
        p.cutoffsq=p.cutoff*p.cutoff;p.cutoffsqs=p.cutoffs.array().square().matrix();
    }
}
Object settings_object(const GPSettings &s) {Object o;for(const auto &p:s.settings)o[p.first]=Value(p.second);return o;}
GPSettings settings_from(const Object&o) {GPSettings s;for(const auto&p:o)s.Modify(p.first,string(p.second));return s;}
std::string real_text(double x){std::ostringstream s;s.imbue(std::locale::classic());s<<std::setprecision(17)<<x;return s.str();}


Object encode_settings(Writer&w,AnyPotential&p,const std::string&id) {
    std::ostringstream out;out<<std::setprecision(17);p.Save(out);auto fields=named_settings(out.str());Object result;
    for(const auto &entry:fields){auto key=entry.first;if(key.rfind("pot:",0)==0)key.erase(0,4);auto text=string(entry.second);
        std::istringstream in(text);std::vector<double> numbers;double x;while(in>>x)numbers.push_back(x);
        if(!numbers.empty()&&in.eof()){for(double n:numbers)require(std::isfinite(n),"nonfinite setting");result[key]=vector_block(w,id+"/settings/"+key,numbers);}
        else result[key]=Value(text);
    }return result;
}
GPSettings decode_settings(const Reader&r,const Object&o) {
    GPSettings result;for(const auto &entry:o){if(entry.second.is<std::string>())result.Modify(entry.first,string(entry.second));
        else {auto v=values(r,entry.second);std::ostringstream out;out<<std::setprecision(17);for(auto x:v)out<<x<<' ';result.Modify(entry.first,trim(out.str()));}}
    return result;
}
template<class NN> void nn_fields(State&a,NN&net) {
    const auto input_rows=net.w_input.rows(),input_cols=net.w_input.cols();

    a.field("activation",net.acfunc);a.field("input_weights",net.w_input);a.field("mean",net.norm_mean);a.field("inverse_std",net.norm_irstd);
    if constexpr(std::is_same_v<NN,IMRMLPNN::NN1lay>){a.field("hidden_bias",net.b_hidden);a.field("output_weights",net.w_hidden);}
    else {a.field("hidden1_bias",net.b_hidden1);a.field("hidden_weights",net.w_hidden1);a.field("hidden2_bias",net.b_hidden2);a.field("output_weights",net.w_hidden2);}
    require(net.w_input.rows()==input_rows&&net.w_input.cols()==input_cols&&net.norm_mean.size()==input_rows&&net.norm_irstd.size()==input_rows,"invalid network input dimensions");
    if constexpr(std::is_same_v<NN,IMRMLPNN::NN1lay>)require(net.b_hidden.size()==net.n_neuron&&net.w_hidden.size()==net.n_neuron,"invalid NN1 dimensions");
    else require(net.b_hidden1.size()==net.n_neuron1&&net.w_hidden1.rows()==net.n_neuron1&&net.w_hidden1.cols()==net.n_neuron2&&net.b_hidden2.size()==net.n_neuron2&&net.w_hidden2.size()==net.n_neuron2,"invalid NN2 dimensions");
    require(net.acfunc>=IMRMLPNN::TANH&&net.acfunc<=IMRMLPNN::SIGMOID,"invalid NN activation");
}

void encode_mtp(Writer&w,PotMTP &p,Object &node,const std::string&id) {
    require(p.inited(),"cannot save an uninitialized MTP");
    Object c{{"species_count",Value(double(p.species_count))},{"radial_functions",Value(double(p.radial_func_count))},
        {"radial_mode",Value(double(p.radial_mode))},{"radial_size",Value(double(p.rb_size))},{"moments_count",Value(double(p.alpha_moments_count))},
        {"metadata",Value(p.potential_tag.output(p.model_metadata(),p.species_count))}};
    Array bases;auto add_basis=[&](AnyBasis &b){std::ostringstream out;out<<std::setprecision(17);b.SaveToStream(out);auto config=named_settings(out.str());config["scaling"]=Value(real_text(b.scaling));bases.emplace_back(config);};
    if(p.radial_basis_split)for(auto *b:p.p_radical_basis_pair)add_basis(*b);else add_basis(*p.p_radical_basis);
    c["radial_bases"]=Value(bases);
    Object params;auto coeff=[&](const char*name,const double*values,int n){require(n>=0&&(n==0||values),"missing MTP coefficients");
        std::vector<double> v;if(n)v.assign(values,values+n);params[name]=vector_block(w,id+"/"+name,v);};
    coeff("radial",p.radial_coeffs,p.radial_coeff_size);coeff("species",p.species_coeffs,p.species_coeff_size);coeff("moments",p.moment_coeffs,p.moment_coeff_size);
    std::vector<int32_t> basic,times,mapping;for(int i=0;i<p.alpha_index_basic.rows();++i)for(int j=0;j<4;++j)basic.push_back(p.alpha_index_basic(i,j));
    for(int i=0;i<p.alpha_index_times.rows();++i)for(int j=0;j<4;++j)times.push_back(p.alpha_index_times(i,j));
    for(int i=0;i<p.alpha_scalar_moments;++i)mapping.push_back(p.alpha_moment_mapping[i]);
    params["basic_indices"]=index_block(w,id+"/basic_indices",basic,{basic.size()/4,4});
    params["product_indices"]=index_block(w,id+"/product_indices",times,{times.size()/4,4});
    params["scalar_mapping"]=index_block(w,id+"/scalar_mapping",mapping,{mapping.size()});
    // The existing family writers provide the authoritative head/kernel field
    // vocabulary. Convert those named fields to native values and FP64 blocks.
    std::ostringstream model;p.Save(model);std::istringstream in(model.str());std::string line;bool head=false;Array entries;
    while(std::getline(in,line)) {
        line=trim(line);if(line.rfind("moment_coeffs =",0)==0){head=true;continue;}if(!head||line.empty()||line[0]=='#')continue;
        auto equal=line.find('=');require(equal!=std::string::npos,"invalid MTP head field");auto name=trim(line.substr(0,equal)),value=trim(line.substr(equal+1));
        Object entry{{"name",Value(name)}};
        if(!value.empty()&&value.front()=='{') {
            for(char &ch:value)if(ch=='{'||ch=='}'||ch==',')ch=' ';
            std::istringstream numbers(value);std::vector<double> v;double x;while(numbers>>x){require(std::isfinite(x),"nonfinite head coefficient");v.push_back(x);}
            require(numbers.eof(),"invalid head array");entry["array"]=vector_block(w,id+"/head/"+name,v);
        }else entry["value"]=Value(value);
        entries.emplace_back(entry);
    }
    c["head_fields"]=Value(entries);node["configuration"]=Value(c);node["parameters"]=Value(params);
}
template<class Head> std::unique_ptr<AnyPotential> read_mtp_m(const Reader&r,const Object&node) {
    auto p=std::make_unique<PotMTP_M<Head>>();p->InitSettings();GPSettings settings;
    const auto &meta=object(at(object(at(node,"configuration")),"metadata"));
    if(meta.count("q"))settings.Modify("n_input",std::to_string(integer(meta.at("q"))));
    if(meta.count("nn")){const auto &nn=object(meta.at("nn"));const auto &layers=array(at(nn,"hidden_layers"));
        require(layers.size()==1||layers.size()==2,"invalid network layers");settings.Modify("n_neuron",std::to_string(integer(layers[0])));
        if(layers.size()>1)settings.Modify("n_neuron2",std::to_string(integer(layers[1])));settings.Modify("activation_function",string(at(nn,"activation")));}
    p->ApplySettings(settings);p->model.init(settings);std::istringstream text(mtp_text(r,node));p->Load(text);p->model.postinit();return p;
}
std::unique_ptr<AnyPotential> read_mtp(const Reader&r,const Object&node,const std::string&type) {
    std::unique_ptr<AnyPotential> p;
    if(type=="mtp_m") {
        auto model=string(at(object(at(object(at(node,"configuration")),"metadata")),"model"));
        if(model=="mtp_m_li")p=read_mtp_m<m_model_mtpli>(r,node);
        else if(model=="mtp_m_q2")p=read_mtp_m<m_model_mtpq2>(r,node);
        else if(model=="mtp_m_fs")p=read_mtp_m<m_model_mtpfs>(r,node);
        else if(model=="mtp_m_n1")p=read_mtp_m<m_model_nn1>(r,node);
        else if(model=="mtp_m_n2")p=read_mtp_m<m_model_nn2>(r,node);
        else require(false,"unsupported MTP head "+model);
    } else {
        if(type=="mtp_les")p=std::make_unique<PotMTP_LES>();else p=std::make_unique<PotMTP>();
        std::istringstream text(mtp_text(r,node));p->Load(text);
    }
    auto &mtp=dynamic_cast<PotMTP&>(*p);mtp.radial_mode=integer(at(object(at(node,"configuration")),"radial_mode"));require(mtp.radial_mode<=1,"unsupported radial mode");const auto &bases=array(at(object(at(node,"configuration")),"radial_bases"));
    for(size_t i=0;i<bases.size();++i){double scale=std::stod(string(at(object(bases[i]),"scaling")));require(std::isfinite(scale),"nonfinite basis scale");
        (mtp.radial_basis_split?mtp.p_radical_basis_pair[i]:mtp.p_radical_basis)->scaling=scale;}
    return p;
}
}

#define FIELD(x) a.field(#x,p.x)
template<> void GPMCodecAccess::fields(State&a,PairTable&p) {
    FIELD(nelements);FIELD(npoints);FIELD(rmin);FIELD(rmax);FIELD(dr);FIELD(inv_dr);FIELD(element_names);FIELD(pair_splines);
    require(p.nelements>0&&p.npoints>=2&&p.pair_splines.size()==size_t(p.nelements)*(p.nelements+1)/2,"invalid pair table");
    require(std::isfinite(p.dr)&&p.dr>0&&p.rmax>p.rmin&&p.inv_dr==1/p.dr,"invalid pair grid");
    for(const auto &m:p.pair_splines)require(m.rows()==p.npoints+2&&m.cols()==7&&m.allFinite(),"invalid pair spline");
}
template<> void GPMCodecAccess::fields(State&a,ThreebodyTable&p) {
    FIELD(nelements);FIELD(nr_points);FIELD(ntheta_points);FIELD(rmin);FIELD(rmax);FIELD(dr);FIELD(inv_dr);
    FIELD(theta_min);FIELD(theta_max);FIELD(dtheta);FIELD(inv_dtheta);FIELD(element_names);FIELD(tables);
    require(p.nelements>0&&p.nr_points>=2&&p.ntheta_points>=2,"invalid threebody table");
    require(p.tables.size()==size_t(p.nelements)*(p.nelements+1)*(p.nelements+2)/6&&p.dr>0&&p.dtheta>0,"invalid triplet grid");
    for(const auto &m:p.tables)require(m.rows()==p.nr_points&&m.cols()==p.ntheta_points&&m.allFinite(),"invalid triplet table");
}
template<> void GPMCodecAccess::fields(State&a,EAMAlloySpline&p) {
    FIELD(nelements);FIELD(Nrho);FIELD(Nr);FIELD(drho);FIELD(dr);FIELD(rdr);FIELD(rdrho);FIELD(rmax);FIELD(rhomin);FIELD(rhomax);FIELD(is_he_format);
    FIELD(frho_spline);FIELD(rhor_spline);FIELD(z2r_spline);
    require(p.nelements>0&&p.Nr>=2&&p.Nrho>=2&&p.frho_spline.size()==size_t(p.nelements)&&p.rhor_spline.size()==size_t(p.nelements),"invalid EAM alloy tables");
    require(p.z2r_spline.size()==size_t(p.nelements)*(p.nelements+1)/2&&p.dr>0&&p.drho>0,"invalid EAM grid");
    for(const auto &m:p.frho_spline)require(m.rows()==p.Nrho+1&&m.cols()==7&&m.allFinite(),"invalid embedding table");
    for(const auto *list:{&p.rhor_spline,&p.z2r_spline})for(const auto &m:*list)require(m.rows()==p.Nr+1&&m.cols()==7&&m.allFinite(),"invalid radial table");
}
template<> void GPMCodecAccess::fields(State&a,EAMFsSpline&p) {
    FIELD(nelements);FIELD(Nrho);FIELD(Nr);FIELD(drho);FIELD(dr);FIELD(rdr);FIELD(rdrho);FIELD(rmax);FIELD(rhomin);FIELD(rhomax);FIELD(is_he_format);
    FIELD(frho_spline);FIELD(rhor_spline);FIELD(z2r_spline);
    require(p.nelements>0&&p.Nr>=2&&p.Nrho>=2&&p.frho_spline.size()==size_t(p.nelements)&&p.rhor_spline.size()==size_t(p.nelements),"invalid EAM FS tables");
    require(p.z2r_spline.size()==size_t(p.nelements)*(p.nelements+1)/2&&p.dr>0&&p.drho>0,"invalid EAM grid");
    for(const auto &m:p.frho_spline)require(m.rows()==p.Nrho+1&&m.cols()==7&&m.allFinite(),"invalid embedding table");
    for(const auto &row:p.rhor_spline){require(row.size()==size_t(p.nelements),"invalid FS density count");for(const auto &m:row)require(m.rows()==p.Nr+1&&m.cols()==7&&m.allFinite(),"invalid density table");}
    for(const auto &m:p.z2r_spline)require(m.rows()==p.Nr+1&&m.cols()==7&&m.allFinite(),"invalid pair table");
}
template<> void GPMCodecAccess::fields(State&a,SW&p) {
    FIELD(nelements);FIELD(element_names_);require(p.nelements>0&&p.nelements<=128,"invalid SW elements");
    if(a.reading()){p.pair_params.assign(p.nelements,std::vector<SW::PairParam>(p.nelements));p.trip_params.assign(p.nelements,std::vector<std::vector<SW::TripletParam>>(p.nelements,std::vector<SW::TripletParam>(p.nelements)));}
    auto pair=[&](const char*key,double SW::PairParam::*member){std::vector<double> values;for(auto &row:p.pair_params)for(auto &v:row)values.push_back(v.*member);a.field(key,values);require(values.size()==size_t(p.nelements)*p.nelements,"invalid SW pair count");size_t n=0;for(auto &row:p.pair_params)for(auto &v:row)v.*member=values[n++];};
    auto trip=[&](const char*key,double SW::TripletParam::*member){std::vector<double> values;for(auto &plane:p.trip_params)for(auto &row:plane)for(auto &v:row)values.push_back(v.*member);a.field(key,values);require(values.size()==size_t(p.nelements)*p.nelements*p.nelements,"invalid SW triplet count");size_t n=0;for(auto &plane:p.trip_params)for(auto &row:plane)for(auto &v:row)v.*member=values[n++];};
#define PAIR(f) pair("pair_" #f,&SW::PairParam::f)
#define TRIP(f) trip("triplet_" #f,&SW::TripletParam::f)
    PAIR(epsilon);PAIR(sigma);PAIR(A);PAIR(B);PAIR(p);PAIR(q);PAIR(a);PAIR(cutoff);
    TRIP(epsilon);TRIP(sigma);TRIP(a);TRIP(lambda);TRIP(gamma);TRIP(cos_theta0);TRIP(cutoff);
#undef PAIR
#undef TRIP
    if(a.reading()){p.build_param_map();p.nl_inited=true;}
}
template<> void GPMCodecAccess::fields(State&a,Tersoff&p) {
    FIELD(nelements);FIELD(element_names_);FIELD(mode_);FIELD(cut_global);
    require(p.nelements>0&&p.nelements<=128&&(p.mode_==Tersoff::Mode::Tersoff||p.mode_==Tersoff::Mode::Brenner),"invalid Tersoff configuration");
    if(a.reading())p.params.assign(p.nelements,std::vector<Tersoff::Param>(p.nelements));
    auto field=[&](const char*key,double Tersoff::Param::*member){std::vector<double> values;for(auto &row:p.params)for(auto &v:row)values.push_back(v.*member);a.field(key,values);require(values.size()==size_t(p.nelements)*p.nelements,"invalid Tersoff pairs");size_t n=0;for(auto &row:p.params)for(auto &v:row)v.*member=values[n++];};
#define PARAM(f) field(#f,&Tersoff::Param::f)
    PARAM(R);PARAM(D);PARAM(c);PARAM(d);PARAM(h);PARAM(A);PARAM(B);PARAM(lambda1);PARAM(lambda2);PARAM(lambda3);PARAM(m);PARAM(n);PARAM(beta);PARAM(D0);PARAM(S);PARAM(r0);PARAM(gamma);
#undef PARAM
    if(a.reading()){p.build_param_map();p.nl_inited=true;}
}

#undef FIELD

std::string GPMCodecAccess::type(const AnyPotential &p) {
#define TYPE(T,n) if(typeid(p)==typeid(T))return n
    TYPE(PotMTP,"mtp");TYPE(PotMTP_LES,"mtp_les");TYPE(PotMTP_M<m_model_mtpli>,"mtp_m");TYPE(PotMTP_M<m_model_mtpq2>,"mtp_m");TYPE(PotMTP_M<m_model_mtpfs>,"mtp_m");TYPE(PotMTP_M<m_model_nn1>,"mtp_m");TYPE(PotMTP_M<m_model_nn2>,"mtp_m");
    TYPE(PotMTPNN1,"mtpnn1");TYPE(PotMTPHyb3,"hmtp");TYPE(PotentialHybrid,"hybrid");TYPE(PotentialHybridScale,"hybrid_scale");
    TYPE(LJ612,"lj612");TYPE(PotentialZBL,"zbl");TYPE(PotMDIST,"mindist");TYPE(SW,"sw");TYPE(Tersoff,"tersoff");TYPE(PairTable,"pair_table");TYPE(ThreebodyTable,"threebody_table");
    TYPE(EAMAlloySpline,"eamalloy");TYPE(EAMFsSpline,"eamfs");TYPE(PotentialEAMPoly,"eampoly");TYPE(PotentialDescriptorLinear,"descriptor_linear");TYPE(PotShellCalc,"shellfile");
#ifdef USE_KIM
    TYPE(PotKIM,"openkim");
#endif
#ifdef WITH_ACELMP
    TYPE(PotentialACELmp,"acelmp");
#endif
#undef TYPE
    throw std::runtime_error("GPM: no codec registered for this potential");
}

std::string GPMCodecAccess::encode(Writer&w,AnyPotential &p,const std::string&id) {
    return registry().at(type(p)).encode(w,p,id);
}

const std::map<std::string,GPMCodecAccess::Codec> &GPMCodecAccess::registry() {
    static const auto codecs=[] {
        std::map<std::string,Codec> entries;
        for(const char *name:{"mtp","mtp_m","mtp_les","mtpnn1","hmtp","hybrid","hybrid_scale","lj612","zbl","mindist","sw","tersoff","pair_table","threebody_table","eamalloy","eamfs","eampoly","descriptor_linear","shellfile","openkim","acelmp"})
            entries.emplace(name,Codec{1,&encode_builtin,&decode_builtin});
        return entries;
    }();
    return codecs;
}

std::string GPMCodecAccess::encode_builtin(Writer&w,AnyPotential &p,const std::string&id) {
    auto name=type(p);Object node{{"type",Value(name)},{"version",Value(1.0)},{"children",Value(Object{})}};
    State shared(w,id+"/common");common(shared,p);node["common"]=Value(shared.fields());
    auto elements=p.gpm_elements;
    if(auto *mtp=dynamic_cast<PotMTP*>(&p)){auto names=mtp->potential_tag.elements();for(size_t i=0;i<names.size();++i)elements[int(i)]=names[i];}
    auto named=[&](const std::vector<std::string>&names){for(size_t i=0;i<names.size();++i)if(!names[i].empty())elements[int(i)]=names[i];};
    if(name=="sw")named(static_cast<SW&>(p).element_names_);
    if(name=="tersoff")named(static_cast<Tersoff&>(p).element_names_);
    if(name=="pair_table")named(static_cast<PairTable&>(p).element_names);
    if(name=="threebody_table")named(static_cast<ThreebodyTable&>(p).element_names);
    if(name=="eampoly")named(static_cast<PotentialEAMPoly&>(p).type_name);
    Array species;for(int index=0;index<p.species_count;++index){Object entry{{"id",Value(double(index))}};if(elements.count(index))entry["symbol"]=Value(elements.at(index));species.emplace_back(entry);}
    node["species_mode"]=Value(name=="mindist"?"type_independent":"indexed");
    node["species"]=Value(species);
    if(auto *mtp=dynamic_cast<PotMTP*>(&p))encode_mtp(w,*mtp,node,id);
    else if(name=="shellfile") {
        auto &shell=static_cast<PotShellCalc&>(p);State a(w,id+"/state");
        Object resources;auto capture=[&](const char *group,const std::map<std::string,std::string>&files){Object refs;for(const auto &[name,contents]:files){auto block=id+"/resources/"+group+"/"+name;w.add(block,"u8",{contents.size()},Bytes(contents.begin(),contents.end()));refs[name]=Value(Object{{"block",Value(block)}});}resources[group]=Value(refs);};
        capture("scripts",shell.gpm_scripts_);capture("inputs",shell.gpm_inputs_);node["resources"]=Value(resources);
        require(!shell.gpm_dependencies_.empty(),"declare shellfile:dependencies before exporting a shell model");a.field("dependencies",shell.gpm_dependencies_);
        require(shell.runscript.empty()||shell.gpm_scripts_.count("run"),"shell run script was not captured; reload with a readable script");
        require(shell.prescript.empty()||shell.gpm_scripts_.count("pre"),"shell pre script was not captured");require(shell.postscript.empty()||shell.gpm_scripts_.count("post"),"shell post script was not captured");
        std::vector<int> map_ids;std::vector<std::string> map_names;for(const auto &[key,name]:shell.iobj.getmap()){map_ids.push_back(key);map_names.push_back(name);}a.field("map_ids",map_ids);a.field("map_names",map_names);
        a.field("read_outfile",shell.read_outfile);a.field("read_outformat",shell.read_outformat);a.field("write_posfile",shell.write_posfile);a.field("write_posformat",shell.write_posformat);a.field("read_energy_sep",shell.read_energy_sep);a.field("energy_file",shell.energy_file);
        node["state"]=Value(a.fields());Array deps{Value("bash")};for(const auto &dep:shell.gpm_dependencies_)deps.emplace_back(dep);node["requires"]=Value(deps);
#ifdef USE_KIM
    } else if(name=="openkim") {
        auto &kim=static_cast<PotKIM&>(p);Object config{{"model",Value(kim.kim_modelname)}};Object params;
        for(auto &[key,value]:kim.model_parameters){State a(w,id+"/kim/"+key);a.field("integers",value.integers);a.field("doubles",value.doubles);params[key]=Value(a.fields());}
        config["parameters"]=Value(params);node["configuration"]=Value(config);node["requires"]=Value(Array{Value("kim-api"),Value(kim.kim_modelname)});
#endif
#ifdef WITH_ACELMP
    } else if(name=="acelmp") {
        auto &ace=static_cast<PotentialACELmp&>(p);require(!ace.serialized_basis_.empty(),"ACE model resource unavailable");State a(w,id+"/state");a.field("elements",ace.elenames);a.field("recursive",ace.recursive);
        Bytes bytes(ace.serialized_basis_.begin(),ace.serialized_basis_.end());const auto size=bytes.size();w.add(id+"/ace_basis","u8",{size},std::move(bytes));node["resource"]=Value(Object{{"block",Value(id+"/ace_basis")}});node["state"]=Value(a.fields());node["requires"]=Value(Array{Value("pace")});
#endif
    } else if(name=="lj612"||name=="zbl"||name=="mindist") {
        node["configuration"]=Value(encode_settings(w,p,id));
    } else if(name=="eampoly") {
        auto &poly=static_cast<PotentialEAMPoly&>(p);State a(w,id+"/state");
        a.field("eam_fs",poly.eam_fs);a.field("type_names",poly.type_name);a.field("nelems",poly.nelems);a.field("npoly_rho",poly.npoly_rho);a.field("npoly_pair",poly.npoly_pair);a.field("npoly_f",poly.npoly_f);a.field("basis_poly",poly.b);
        a.field("basis_type",poly.basis_poly);a.field("rmax_pair",poly.rmax_pair);a.field("rmax_rho",poly.rmax_rho);a.field("rho_min",poly.rho_min);a.field("rho_max",poly.rho_max);a.field("coefficients",poly.regression_coeffs);
        node["state"]=Value(a.fields());
    } else if(name=="mtpnn1") {
        auto &nn=static_cast<PotMTPNN1&>(p);node["children"]=Value(Object{{"descriptor",Value(encode(w,*nn.p_mtp,id+"/descriptor"))}});
        State a(w,id+"/state");a.field("layers",nn.nnlay);a.field("species_coefficients",nn.species_coeffs);a.field("train_nn_only",nn.train_nn_only);
        int input=nn.nnlay==1?nn.p_nn->n_input:nn.p_nn2->n_input,hidden=nn.nnlay==1?nn.p_nn->n_neuron:nn.p_nn2->n_neuron1,hidden2=nn.nnlay==1?0:nn.p_nn2->n_neuron2;
        a.field("input",input);a.field("hidden",hidden);a.field("hidden2",hidden2);
        if(nn.nnlay==1)nn_fields(a,*nn.p_nn);else nn_fields(a,*nn.p_nn2);node["state"]=Value(a.fields());
    } else if(name=="hmtp"||name=="hybrid"||name=="hybrid_scale") {
        Object children,config;std::vector<const AnyPotential*> sub;
        if(name=="hmtp")for(auto *v:static_cast<PotMTPHyb3&>(p).children())sub.push_back(v);
        else if(name=="hybrid")sub=static_cast<PotentialHybrid&>(p).children();
        else sub=static_cast<PotentialHybridScale&>(p).children();
        config["npot"]=Value(std::to_string(sub.size()));
        for(size_t i=0;i<sub.size();++i){auto key="pot"+std::to_string(i+1);children[key]=Value(encode(w,*const_cast<AnyPotential*>(sub[i]),id+"/"+key));config[key+":pottype"]=Value(type(*sub[i]));
            if(name=="hybrid"){const auto &train=static_cast<PotentialHybrid&>(p).train_ids;config[key+":train"]=Value(std::find(train.begin(),train.end(),int(i))!=train.end()?"true":"false");}}
        if(name=="hybrid_scale"){auto &h=static_cast<PotentialHybridScale&>(p);config["feature"]=Value(h.feature);config["transition"]=Value(h.transition);
            config["feature_r0"]=Value(real_text(h.feature_r0));config["feature_r1"]=Value(real_text(h.feature_r1));
            for(size_t i=0;i<h.boundaries_.size();++i)config["boundary"+std::to_string(i+1)]=Value(real_text(h.boundaries_[i].left)+" "+real_text(h.boundaries_[i].right));}
        node["children"]=Value(children);node["configuration"]=Value(config);
    } else if(name=="descriptor_linear") {
        auto &d=static_cast<PotentialDescriptorLinear&>(p);auto config=settings_object(d.descriptor_settings_);config.erase("mtm:potfile");node["configuration"]=Value(config);
        if(auto *mtm=dynamic_cast<DescriptorMTM*>(d.desc_wrapper_->desc.get()))node["children"]=Value(Object{{"descriptor",Value(encode(w,mtm->pot(),id+"/descriptor"))}});
        State a(w,id+"/state");a.field("coefficients",d.coeffs_);node["state"]=Value(a.fields());
    } else {
        State state(w,id+"/state");
#define SNAP(T,n) if(name==n)fields(state,static_cast<T&>(p));else
        SNAP(SW,"sw") SNAP(Tersoff,"tersoff") SNAP(PairTable,"pair_table") SNAP(ThreebodyTable,"threebody_table") SNAP(EAMAlloySpline,"eamalloy") SNAP(EAMFsSpline,"eamfs")
        { require(false,"codec state not implemented: "+name); }
#undef SNAP
        node["state"]=Value(state.fields());
    }
    auto &nodes=w.manifest.at("models").get<Object>();
    for(const auto &[role,reference]:object(node.at("children")))for(const auto &entry:array(at(object(at(nodes,string(reference))),"species"))) {
        const auto &child=object(entry);if(!child.count("symbol"))continue;int index=integer(at(child,"id"));
        require(index<p.species_count,"child species exceed parent dimensions");const auto symbol=string(at(child,"symbol"));
        require(!elements.count(index)||elements.at(index)==symbol,"child element order conflicts with parent");elements[index]=symbol;
    }
    auto &declared=node.at("species").get<Array>();for(auto &entry:declared){auto &fields=entry.get<Object>();int index=integer(at(fields,"id"));if(elements.count(index))fields["symbol"]=Value(elements.at(index));}
    require(!nodes.count(id),"duplicate model id");nodes[id]=Value(node);
    return id;
}
std::unique_ptr<AnyPotential> GPMCodecAccess::decode(const Reader&r,const std::string&id) {std::set<std::string> stack;return decode_node(r,id,stack);}
std::unique_ptr<AnyPotential> GPMCodecAccess::decode_node(const Reader&r,const std::string&id,std::set<std::string>&stack) {
    const auto &node=object(at(object(at(r.manifest,"models")),id));
    const auto name=string(at(node,"type"));auto codec=registry().find(name);
    require(codec!=registry().end(),"unsupported model type "+name);
    require(integer(at(node,"version"))==codec->second.version,"unsupported model version");
    return codec->second.decode(r,id,stack);
}
std::unique_ptr<AnyPotential> GPMCodecAccess::decode_builtin(const Reader&r,const std::string&id,std::set<std::string>&stack) {
    require(stack.size()<64&&stack.insert(id).second,"cyclic or excessive model nesting");
    const auto &node=object(at(object(at(r.manifest,"models")),id));if(node.count("required_features"))require(array(node.at("required_features")).empty(),"unsupported model capability");require(integer(at(node,"version"))==1,"unsupported model version");auto name=string(at(node,"type"));
    std::unique_ptr<AnyPotential> result;
    if(name=="mtp"||name=="mtp_m"||name=="mtp_les")result=read_mtp(r,node,name);
    else if(name=="shellfile") {
        auto p=std::make_unique<PotShellCalc>();p->InitSettings();State a(r,object(at(node,"state")));a.field("dependencies",p->gpm_dependencies_);
        const auto &files=object(at(node,"resources"));auto restore=[&](const char *group,std::map<std::string,std::string>&dest){for(const auto &[name,reference]:object(at(files,group))){auto bytes=r.read(string(at(object(reference),"block")));dest[name]=std::string(bytes.begin(),bytes.end());}};
        restore("scripts",p->gpm_scripts_);restore("inputs",p->gpm_inputs_);
        std::vector<int> map_ids;std::vector<std::string> map_names;a.field("map_ids",map_ids);a.field("map_names",map_names);require(map_ids.size()==map_names.size(),"invalid shell map");std::map<int,std::string> mapping;for(size_t i=0;i<map_ids.size();++i)require(mapping.emplace(map_ids[i],map_names[i]).second,"duplicate shell map id");p->iobj.setmap(mapping);
        a.field("read_outfile",p->read_outfile);a.field("read_outformat",p->read_outformat);a.field("write_posfile",p->write_posfile);a.field("write_posformat",p->write_posformat);a.field("read_energy_sep",p->read_energy_sep);a.field("energy_file",p->energy_file);
        auto resources=std::make_shared<TemporaryResources>();for(const auto &[key,text]:p->gpm_scripts_){auto path=resources->write(key+".sh",text);if(key=="pre")p->prescript=path;else if(key=="run")p->runscript=path;else if(key=="post")p->postscript=path;else require(false,"unknown shell script role");}
        for(const auto &[name,text]:p->gpm_inputs_){std::filesystem::path path(name);require(!path.empty()&&!path.is_absolute(),"invalid shell resource path");for(const auto &part:path)require(part!="..","shell resource traversal");}
        p->gpm_resources=resources;result=std::move(p);
    } else if(name=="openkim") {
#ifdef USE_KIM
        const auto &config=object(at(node,"configuration"));std::map<int,std::string> elements;
        for(const auto &v:array(at(node,"species"))){const auto&o=object(v);if(o.count("symbol"))elements[integer(at(o,"id"))]=string(at(o,"symbol"));}
        auto p=std::make_unique<PotKIM>(string(at(config,"model")),elements);GPSettings settings;
        for(const auto &[key,value]:object(at(config,"parameters"))){State a(r,object(value));std::vector<int> integers;std::vector<double> doubles;a.field("integers",integers);a.field("doubles",doubles);std::ostringstream out;out<<std::setprecision(17);for(auto x:integers)out<<x<<' ';for(auto x:doubles)out<<x<<' ';settings.Modify("kim_parameter:"+key,trim(out.str()));}
        p->ApplyParameterSettings(settings);result=std::move(p);
#else
        require(false,"OpenKIM runtime support is not enabled");
#endif
    } else if(name=="acelmp") {
#ifdef WITH_ACELMP
        State a(r,object(at(node,"state")));std::vector<std::string> elements;bool recursive;a.field("elements",elements);a.field("recursive",recursive);auto bytes=r.read(string(at(object(at(node,"resource")),"block")));auto resources=std::make_shared<TemporaryResources>();auto path=resources->write("model.ace",std::string(bytes.begin(),bytes.end()));auto p=std::make_unique<PotentialACELmp>(path,elements,recursive);p->gpm_resources=resources;result=std::move(p);
#else
        require(false,"ACE runtime support is not enabled");
#endif
    } else if(name=="lj612"||name=="zbl"||name=="mindist") {
        auto config=decode_settings(r,object(at(node,"configuration")));
        if(name=="lj612")result=std::make_unique<LJ612>(config);
        else if(name=="zbl")result=std::make_unique<PotentialZBL>(config);
        else if(name=="mindist")result=std::make_unique<PotMDIST>(config);

    } else if(name=="eampoly") {
        State a(r,object(at(node,"state")));int ne,nr,np,nf,bt;double b,rmin,rmax;Eigen::MatrixXd pair,rho;std::vector<double> coeffs;
        a.field("nelems",ne);a.field("npoly_rho",nr);a.field("npoly_pair",np);a.field("npoly_f",nf);a.field("basis_poly",b);a.field("basis_type",bt);
        a.field("rmax_pair",pair);a.field("rmax_rho",rho);a.field("rho_min",rmin);a.field("rho_max",rmax);a.field("coefficients",coeffs);
        require(ne>0&&nr>0&&np>0&&nf>0&&pair.rows()==ne&&pair.cols()==ne&&rho.rows()==ne&&rho.cols()==ne,"invalid EAM polynomial dimensions");
        auto p=std::make_unique<PotentialEAMPoly>(nr,nf,np,ne,pair,rho,bt,b);require(coeffs.size()==size_t(p->n_coeffs),"invalid EAM coefficients");p->nl_set_coeffs(coeffs.data());p->rho_min=rmin;p->rho_max=rmax;a.field("eam_fs",p->eam_fs);a.field("type_names",p->type_name);result=std::move(p);
    } else if(name=="mtpnn1") {
        auto p=std::unique_ptr<PotMTPNN1>(new PotMTPNN1);p->InitSettings();State a(r,object(at(node,"state")));
        auto child=decode_node(r,string(at(object(at(node,"children")),"descriptor")),stack);require(typeid(*child)==typeid(PotMTP),"NN descriptor requires ordinary MTP");p->p_mtp=static_cast<PotMTP*>(child.release());
        a.field("layers",p->nnlay);a.field("species_coefficients",p->species_coeffs);a.field("train_nn_only",p->train_nn_only);int input,hidden,hidden2;a.field("input",input);a.field("hidden",hidden);a.field("hidden2",hidden2);
        require((p->nnlay==1||p->nnlay==2)&&input==p->p_mtp->alpha_scalar_moments+1&&hidden>0&&hidden<=100000&&hidden2>=0&&hidden2<=100000,"invalid NN dimensions");
        if(p->nnlay==1){p->p_nn=new IMRMLPNN::NN1lay;p->p_nn->setup(input,hidden);nn_fields(a,*p->p_nn);p->nn_coeff_size=p->p_nn->n_coeffs;}
        else{require(hidden2>0,"invalid NN second layer");p->p_nn2=new IMRMLPNN::NN2lay;p->p_nn2->setup(input,hidden,hidden2);nn_fields(a,*p->p_nn2);p->nn_coeff_size=p->p_nn2->n_coeffs;}
        require(p->species_coeffs.size()==size_t(p->p_mtp->species_count),"NN species count mismatch");p->nl_inited=true;result=std::move(p);
    } else if(name=="hmtp"||name=="hybrid"||name=="hybrid_scale") {
        auto config=settings_from(object(at(node,"configuration")));const auto &references=object(at(node,"children"));std::vector<std::unique_ptr<AnyPotential>> children;
        int n=Parameters::parse_value<int>(config["npot"]);require(n>0&&n<=100000&&references.size()==size_t(n),"invalid composite children");
        for(int i=0;i<n;++i)children.push_back(decode_node(r,string(at(references,"pot"+std::to_string(i+1))),stack));
        if(name=="hybrid")result=std::make_unique<PotentialHybrid>(config,std::move(children));
        else if(name=="hybrid_scale")result=std::make_unique<PotentialHybridScale>(config,std::move(children));
        else{require(n==2||n==3,"HMTP requires 2 or 3 children");for(auto &p:children)require(typeid(*p)==typeid(PotMTP),"HMTP requires ordinary MTP");
            auto h=std::make_unique<PotMTPHyb3>(static_cast<PotMTP*>(children[0].get()),static_cast<PotMTP*>(children[1].get()),n==3?static_cast<PotMTP*>(children[2].get()):nullptr);
            for(auto &p:children)p.release();h->needfree=true;result=std::move(h);}
    } else if(name=="descriptor_linear") {
        GPSettings config;for(const auto &entry:object(at(node,"configuration")))config.Modify("descriptor:"+entry.first,string(entry.second));
        std::shared_ptr<AnyPotential> dep;std::unique_ptr<Descriptors> desc;const auto &refs=object(at(node,"children"));
        if(refs.count("descriptor")){dep=decode_node(r,string(refs.at("descriptor")),stack);auto *mtp=dynamic_cast<PotMTP*>(dep.get());require(mtp,"invalid MTM dependency");desc=std::make_unique<DescriptorMTM>(*mtp);}
        auto p=std::make_unique<PotentialDescriptorLinear>(config,std::move(desc));State a(r,object(at(node,"state")));a.field("coefficients",p->coeffs_);require(p->coeffs_.size()==p->n_features_,"descriptor coefficient dimensions mismatch");p->gpm_resources=dep;result=std::move(p);
    } else {
        State state(r,object(at(node,"state")));
#define SNAP(T,n) if(name==n){auto p=std::make_unique<T>();fields(state,*p);result=std::move(p);}else
        SNAP(SW,"sw") SNAP(Tersoff,"tersoff") SNAP(PairTable,"pair_table") SNAP(ThreebodyTable,"threebody_table") SNAP(EAMAlloySpline,"eamalloy") SNAP(EAMFsSpline,"eamfs")
        { require(false,"unsupported model type "+name); }
#undef SNAP
    }
    const int constructed_species=result->species_count;
    State shared(r,object(at(node,"common")));common(shared,*result);
    require(constructed_species==0||constructed_species==result->species_count,"species dimensions conflict with model state");
    std::set<int> species_ids;std::map<int,std::string> declared_elements;
    for(const auto &entry:array(at(node,"species"))){const auto &o=object(entry);int index=integer(at(o,"id"));require(index<result->species_count,"invalid species id");require(species_ids.insert(index).second,"duplicate species id");if(o.count("symbol"))declared_elements[index]=string(at(o,"symbol"));}
    require(species_ids.size()==size_t(result->species_count),"incomplete species table");
    for(const auto &[index,symbol]:result->gpm_elements)require(declared_elements.count(index)&&declared_elements.at(index)==symbol,"species order conflicts with model state");
    const auto &children=object(at(node,"children"));
    for(const auto &[role,reference]:children)
        for(const auto &entry:array(at(object(at(object(at(r.manifest,"models")),string(reference))),"species"))) {
            const auto &child=object(entry);const int index=integer(at(child,"id"));
            if(child.count("symbol"))require(declared_elements.count(index)&&declared_elements.at(index)==string(at(child,"symbol")),"child element order conflicts with parent");
        }
    result->gpm_elements=std::move(declared_elements);
    // Each child has already restored its own effective mapping. Calling a
    // composite setter here would label previously unnamed children and change
    // the model fingerprint (including bindings of existing AL state).
    if(children.empty())result->set_element_map(result->gpm_elements);
    result->omp_init();stack.erase(id);return result;
}
} // namespace Potentials
