#include "model_bundle.h"
#include "gpm_codec.h"
#include "tools/gpm_mtp.h"
#include "potential_mtp.h"
#include "potential_eam_zhou_2004.h"
#include "potential_wrapper.h"
#include "active_learn/gpm_al.h"
#include "active_learn/al_potential.h"
#include "utils/mpi_stubs.h"

namespace Potentials {
using namespace GPM;
namespace {
void digest_fields(SHA256 &hash,const Value &value,const std::function<std::string(const std::string&)> &block_hash) {
    if(value.is<Object>()) {
        const auto &o=value.get<Object>();auto block=o.find("block");
        if(block!=o.end()) {auto h=block_hash(string(block->second));hash.update(h.data(),h.size());}
        for(const auto &p:o)digest_fields(hash,p.second,block_hash);
    } else if(value.is<Array>())for(const auto &v:value.get<Array>())digest_fields(hash,v,block_hash);
}
std::string fingerprint_writer(const Writer&w) {
    const auto &models=w.manifest.at("models");std::string json=models.serialize();SHA256 h;h.update(json.data(),json.size());
    digest_fields(h,models,[&](const std::string&id){return w.block_hash(id);});return hex(h.finish());
}
void publish(Writer&w,const std::string&path) {
    w.save(path);
}
Writer encode(AnyPotential&p) {
    Writer w;GPMCodecAccess::encode(w,p);w.manifest["species"]=w.manifest["models"].get<Object>().at("root").get<Object>().at("species");
    w.manifest["model_fingerprint"]=Value(fingerprint_writer(w));return w;
}
}
ModelBundle::~ModelBundle()=default;
std::shared_ptr<ModelBundle> ModelBundle::from_settings(GPSettings settings) {
    auto bundle=std::shared_ptr<ModelBundle>(new ModelBundle);
    potential_wrapper wrapper(settings);bundle->potential_.reset(wrapper.pot);wrapper.pot=nullptr;
    bundle->source_=bundle->potential_->gpm_source;
    if(!bundle->source_)bundle->pending_=std::make_unique<Writer>(encode(bundle->potential()));
    return bundle;
}
std::shared_ptr<ModelBundle> ModelBundle::load(const std::string &path) {
    auto b=std::shared_ptr<ModelBundle>(new ModelBundle);b->source_=std::make_shared<Reader>(path);
    const auto &units=object(at(b->source_->manifest,"units"));
    require(string(at(units,"length"))=="angstrom"&&string(at(units,"energy"))=="eV"&&string(at(units,"charge"))=="e"&&string(at(units,"stress"))=="virial_eV","unsupported unit convention");
    b->potential_=GPMCodecAccess::decode(*b->source_,string(at(b->source_->manifest,"root_model")));
    const auto &root=object(at(object(at(b->source_->manifest,"models")),string(at(b->source_->manifest,"root_model"))));
    require(at(b->source_->manifest,"species")==at(root,"species"),"top-level species table conflicts with root model");
    b->potential_->gpm_source=b->source_;return b;
}
Object ModelBundle::inspect(const std::string &path) {return Reader(path).manifest;}
void ModelBundle::validate(const std::string &path) {
    auto header=inspect(path);const auto &node=object(at(object(at(header,"models")),string(at(header,"root_model"))));
    if(string(at(node,"type"))=="eam_zhou2004_parameters"){EAMZhou2004 parameters;parameters.LoadParameterSet(path);return;}
    auto bundle=load(path);bundle->source_->validate();
    for(const auto &[id,node]:object(at(bundle->source_->manifest,"models")))
        if(id!=string(at(bundle->source_->manifest,"root_model")))GPMCodecAccess::decode(*bundle->source_,id);
    require(fingerprint(bundle->potential())==string(at(bundle->source_->manifest,"model_fingerprint")),"model fingerprint mismatch");
    if(bundle->source_->manifest.count("active_learning")){GPSettings options;bundle->create_al(options);}
}
AnyPotential &ModelBundle::potential()const {require(bool(potential_),"bundle no longer owns a potential");return *potential_;}
std::unique_ptr<AnyPotential> ModelBundle::release_potential(){return std::move(potential_);}
std::string ModelBundle::type(const AnyPotential&p){return GPMCodecAccess::type(p);}
std::string ModelBundle::fingerprint(AnyPotential&p){auto w=encode(p);return string(w.manifest.at("model_fingerprint"));}
void ModelBundle::save_potential(AnyPotential&p,const std::string&path,Profile profile) {
    auto w=encode(p);
    if(p.gpm_source && profile!=Profile::Inference) {
        const auto &old=p.gpm_source->manifest;
        bool same=string(at(old,"model_fingerprint"))==string(w.manifest.at("model_fingerprint"));
        require(same||!old.count("active_learning"),"model changed; rebuild AL state or explicitly save inference profile");
        if(same){require(profile!=Profile::Resume||old.count("active_learning"),"resume profile requires AL state");Writer preserved;preserved.preserve(p.gpm_source);publish(preserved,path);return;}
        for(const char*key:{"provenance","extensions"})if(old.count(key))w.manifest[key]=old.at(key);
    }
    require(profile!=Profile::Resume||p.gpm_source&&p.gpm_source->manifest.count("active_learning"),"resume profile requires AL state");
    publish(w,path);
}
void ModelBundle::save(const std::string&path,Profile profile) {
    if(pending_&&pending_->manifest.count("active_learning")&&profile!=Profile::Inference) {
        require(profile!=Profile::Resume||pending_->manifest.count("active_learning"),"resume profile requires AL state");
        require(fingerprint(potential())==string(at(pending_->manifest,"model_fingerprint")),"model changed after binding AL state");publish(*pending_,path);
    } else save_potential(potential(),path,profile);
}
std::unique_ptr<AL::ALPotential> ModelBundle::create_al(const GPSettings&overrides) {
    require(bool(source_),"no embedded AL source");
    require(fingerprint(potential())==string(at(source_->manifest,"model_fingerprint")),"model differs from embedded AL target");
    auto al=AL::GPMALAccess::decode(*source_,potential(),overrides);
    al->bundle_owner_=shared_from_this();return al;
}
void ModelBundle::attach_al(AL::ALPotential&al) {pending_=std::make_unique<Writer>(encode(potential()));AL::GPMALAccess::encode(*pending_,al,potential());}
void ModelBundle::attach_selector(ActiveSelector&selector) {pending_=std::make_unique<Writer>(encode(potential()));AL::GPMALAccess::encode_selector(*pending_,selector,potential());}
bool load_gpm_if_requested(AnyPotential&p,const std::string&path) {
    if(!GPM::is_file(path)&&!GPM::extension(path))return false;
    auto source=std::make_shared<Reader>(path);const auto &node=object(at(object(at(source->manifest,"models")),string(at(source->manifest,"root_model"))));
    require(GPMCodecAccess::type(p)==string(at(node,"type")),"GPM type conflicts with destination potential");
    require(integer(at(node,"version"))==1,"unsupported model version");
    auto *mtp=dynamic_cast<PotMTP*>(&p);require(mtp,"use ModelBundle to load this GPM family");
    std::istringstream stream(mtp_text(*source,node));p.Load(stream);
    const auto &bases=array(at(object(at(node,"configuration")),"radial_bases"));
    if(mtp->radial_basis_split)for(size_t i=0;i<bases.size();++i)mtp->p_radical_basis_pair[i]->scaling=std::stod(string(at(object(bases[i]),"scaling")));
    else mtp->p_radical_basis->scaling=std::stod(string(at(object(bases[0]),"scaling")));
    auto restored=GPMCodecAccess::decode(*source,string(at(source->manifest,"root_model")));
    mtp->radial_mode=dynamic_cast<PotMTP&>(*restored).radial_mode;
    p.add_efs=restored->add_efs;p.reg_L1_vector=restored->reg_L1_vector;p.reg_L2_vector=restored->reg_L2_vector;
    p.gpm_elements=restored->gpm_elements;p.gpm_source=source;return true;
}
bool save_gpm_if_requested(AnyPotential&p,const std::string&path) {if(!GPM::extension(path))return false;ModelBundle::save_potential(p,path);return true;}
} // namespace Potentials
