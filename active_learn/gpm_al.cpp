#include "gpm_al.h"
#include "al_potential.h"
#include "al_scorer.h"
#include "maxvol/active_selector.h"
#include "potentials/gpm_codec.h"
#include "potentials/gpm_state.h"
#include "potentials/model_bundle.h"
#include "potentials/potential_mtp.h"
#include <numeric>

namespace AL {
using namespace GPM;
namespace {
void atom_fields(State&a,Atoms&p) {
#define F(x) a.field(#x,p.x)
    F(cell_);F(pos_);F(chemmap_);F(pbc_);F(cartesian);F(ids_);
    F(have_ghost);F(nlocal_);F(nghost_);F(map_ghost_);F(molid_);F(molsize_);F(molnum_);F(mol_types_);F(bonds_);F(angles_);
    F(forces_);F(velocities_);F(have_energy);F(have_fenergy);F(have_stress);F(have_hessian);F(have_eff_chg);
    if(p.have_energy)F(energy_);if(p.have_fenergy)F(fenergy_);if(p.have_stress)F(stress_);
    F(hessian_);F(hvp_directions_);F(hvp_targets_);F(site_energies_);F(charges_);F(magmom_scalar_);F(magmoms_);F(born_eff_chgs_);F(collinear_);
    F(fitting_factor);F(fitness);F(atomic_fitness);F(masks);F(group_count);F(sel_dyncell_);F(sel_dynpos_);
    F(global_attribs_d_);F(local_attribs_d_);F(global_attribs_v_);F(local_attribs_v_);F(global_attribs_i_);F(global_attribs_vi_);F(local_attribs_i_);F(local_attribs_vi_);F(features_);F(mvgrade_);F(mindist_);F(maxdist_);F(id);
#undef F
    require(p.pos_.size()==p.chemmap_.size()&&p.cell_.allFinite(),"invalid AL structure");
    for(auto &v:p.pos_)require(v.allFinite(),"nonfinite AL coordinates");
}
Value structures(Writer&w,const std::string&id,std::vector<Atoms>&cfgs) {
    Array records;for(size_t i=0;i<cfgs.size();++i){State s(w,id+"/"+std::to_string(i));atom_fields(s,cfgs[i]);records.emplace_back(s.fields());}
    // Keep the potentially large configuration directory out of the file header.
    std::string json=Value(records).serialize();Bytes bytes(json.begin(),json.end());const auto size=bytes.size();w.add(id,"u8",{size},std::move(bytes));return Value(Object{{"block",Value(id)},{"version",Value(1.0)}});
}
std::vector<Atoms> structures(const Reader&r,const Value&v) {
    auto id=string(at(object(v),"block"));require(integer(at(object(v),"version"))==1,"unsupported structure version");auto bytes=r.read(id);auto records=parse(std::string(bytes.begin(),bytes.end()));
    const auto &list=array(records);std::vector<Atoms> result(list.size());for(size_t i=0;i<list.size();++i){State s(r,object(list[i]));atom_fields(s,result[i]);}return result;
}
void builder_fields(State&a,BuilderParams&p) {
#define F(x) a.field(#x,p.x)
    F(dtype);F(mode);F(w_energy);F(w_site);F(w_force);F(w_stress);F(linear_only);F(row_scaling);F(valid);
#undef F
    std::map<std::string,std::string> settings;for(const auto &entry:p.settings.settings)settings[entry.first]=entry.second;
    settings.erase("mtm:potfile");a.field("settings",settings);if(a.reading()){p.settings.settings.clear();for(const auto &entry:settings)p.settings.Modify(entry.first,entry.second);}
}
void row_fields(State&a,std::vector<BRowRef>&refs) {
    Eigen::Matrix<int,-1,6,Eigen::RowMajor> rows(refs.size(),6);
    if(!a.reading())for(size_t i=0;i<refs.size();++i){const auto &r=refs[i];rows.row(i)<<r.cfg_index,r.local_row,r.global_row,int(r.kind),r.atom_index,r.component;}
    a.field("row_references",rows);if(a.reading()){refs.resize(rows.rows());for(size_t i=0;i<refs.size();++i){require(rows(i,3)>=0&&rows(i,3)<=int(BRowKind::Unknown),"invalid row kind");refs[i]={rows(i,0),rows(i,1),rows(i,2),BRowKind(rows(i,3)),rows(i,4),rows(i,5)};}}
}
template<class Get> std::string binding_hash(const std::string &model,const Value &config,Get get) {
    SHA256 hash;hash.update(model.data(),model.size());auto text=config.serialize();hash.update(text.data(),text.size());
    std::function<void(const Value&)> walk=[&](const Value &v){if(v.is<Object>()){auto &o=v.get<Object>();if(o.count("block")){auto sum=get(string(o.at("block")));hash.update(sum.data(),sum.size());}for(const auto &entry:o)walk(entry.second);}else if(v.is<Array>())for(const auto &entry:v.get<Array>())walk(entry);};
    walk(config);return hex(hash.finish());
}
Object binding(Writer&w,const std::string&mode) {
    return {{"version",Value(1.0)},{"method",Value(mode)},{"target_model",Value("root")},{"model_fingerprint",at(w.manifest,"model_fingerprint")}};
}
}

Object GPMALAccess::selector(Writer&w,ActiveSelector&p,Potentials::AnyPotential&pot) {
    BuilderParams pb=p.pending_builder_;
    if(auto *b=dynamic_cast<PotEFSBuilder*>(p.builder_)){pb.dtype="pot_efs";pb.mode="cfg";pb.w_energy=b->w_energy;pb.w_site=b->w_site;pb.w_force=b->w_force;pb.w_stress=b->w_stress;pb.linear_only=b->linear_only;pb.valid=true;}
    else if(auto *b=dynamic_cast<DescriptorBuilder*>(p.builder_)){pb.dtype=b->dtype_;pb.mode=b->mode==DescriptorBuilder::Nbh?"nbh":"cfg";pb.settings=b->settings_;pb.valid=true;}
    require(pb.valid,"MaxVol builder metadata is unavailable");if(p.builder_)pb.row_scaling=p.builder_->row_scaling;
    State builder(w,"al/maxvol/builder");builder_fields(builder,pb);
    OnlineRowMatrix C;std::vector<BRowRef> refs=p.selected_row_refs_;
    if(p.online_state_.valid()){C=p.online_state_.active_rows();refs=p.online_state_.selected_refs();}
    else {C.resize(p.selected_row_indices_.size(),p.n_coeff_);for(size_t i=0;i<p.selected_row_indices_.size();++i){int index=p.selected_row_indices_[i];require(index>=0&&index<p.B_full_.rows(),"invalid active row index");C.row(i)=p.B_full_.row(index);}}
    require(C.rows()>0&&C.cols()==p.n_coeff_&&C.allFinite(),"no valid active rows to save");
    if(refs.size()!=size_t(C.rows())) {
        refs.clear();for(int row:p.selected_row_indices_){BRowRef ref;ref.global_row=row;for(size_t i=1;i<p.cfg_row_start_.size();++i)if(row>=p.cfg_row_start_[i-1]&&row<p.cfg_row_start_[i]){ref.cfg_index=i-1;ref.local_row=row-p.cfg_row_start_[i-1];break;}refs.push_back(ref);}
    }
    // Bind legacy states only after checking their rows against the supplied
    // model. Dimension equality alone does not establish compatibility.
    ActiveSelector check;check.pending_builder_=pb;check.n_coeff_=p.n_coeff_;check.init_builder(pot);
    int previous=-1;OnlineRowMatrix rows;bool verified_source=false;
    for(size_t i=0;i<refs.size();++i){const auto &ref=refs[i];
        if(ref.cfg_index<0){int nonzero=0;for(int j=0;j<C.cols();++j)if(C(i,j)!=0){require(C(i,j)==p.init_value,"invalid virtual active row");++nonzero;}require(nonzero==1,"unverifiable virtual row");continue;}
        require(ref.cfg_index<int(p.cfgs_.size()),"active row has no source configuration");
        if(previous!=ref.cfg_index){Atoms cfg=p.cfgs_[ref.cfg_index];rows.resize(0,p.n_coeff_);check.builder_->build_rows(cfg,rows,p.weight,p.pow_scale);previous=ref.cfg_index;}
        require(ref.local_row>=0&&ref.local_row<rows.rows(),"active row has invalid local index");
        require((C.row(i)-rows.row(ref.local_row)).norm()<=1e-8*(1+C.row(i).norm()),"MaxVol rows do not match the supplied model; rebuild selection");
        verified_source=true;
    }
    require(!p.gpm_legacy_unverified_||verified_source,"legacy MaxVol state has no verifiable model rows; rebuild selection");
    State a(w,"al/maxvol/state");
#define F(x) a.field(#x,p.x)
    F(n_coeff_);F(threshold);F(weight);F(pow_scale);F(init_mode);F(init_value);F(max_swaps);F(select_mode);F(stream_mode);F(stream_chunk_size);F(stream_fast_passes);F(stream_fast_shortlist);
    F(converged_);F(max_remaining_grade_);F(termination_reason_);F(online_processed_cfgs_);F(cfg_row_start_);F(is_new_);F(selected_config_indices_);
#undef F
    a.field("active_matrix",C);row_fields(a,refs);bool online=p.online_state_.valid();a.field("online",online);
    return {{"builder",Value(builder.fields())},{"state",Value(a.fields())},{"history",structures(w,"al/maxvol/history",p.cfgs_)},{"selected",structures(w,"al/maxvol/selected",p.selected_cfgs_)}};
}
std::unique_ptr<ActiveSelector> GPMALAccess::selector(const Reader&r,const Object&o,Potentials::AnyPotential&pot) {
    auto owner=std::make_unique<ActiveSelector>();auto &p=*owner;State builder(r,object(at(o,"builder")));builder_fields(builder,p.pending_builder_);
    State a(r,object(at(o,"state")));
#define F(x) a.field(#x,p.x)
    F(n_coeff_);F(threshold);F(weight);F(pow_scale);F(init_mode);F(init_value);F(max_swaps);F(select_mode);F(stream_mode);F(stream_chunk_size);F(stream_fast_passes);F(stream_fast_shortlist);
    F(converged_);F(max_remaining_grade_);F(termination_reason_);F(online_processed_cfgs_);F(cfg_row_start_);F(is_new_);F(selected_config_indices_);
#undef F
    a.field("active_matrix",p.B_full_);row_fields(a,p.selected_row_refs_);bool online;a.field("online",online);
    p.cfgs_=structures(r,at(o,"history"));p.selected_cfgs_=structures(r,at(o,"selected"));
    require(p.n_coeff_>0&&p.B_full_.cols()==p.n_coeff_&&p.B_full_.rows()>0&&p.B_full_.allFinite()&&p.selected_row_refs_.size()==size_t(p.B_full_.rows()),"invalid MaxVol active matrix");
    require(p.cfg_row_start_.size()==p.cfgs_.size()+1&&!p.cfg_row_start_.empty()&&p.cfg_row_start_[0]==0&&std::is_sorted(p.cfg_row_start_.begin(),p.cfg_row_start_.end())&&p.online_processed_cfgs_<=p.cfgs_.size()&&p.is_new_.size()==p.cfgs_.size(),"invalid MaxVol processing state");
    for(const auto &ref:p.selected_row_refs_)require(ref.cfg_index<0 || (ref.cfg_index<int(p.cfgs_.size())&&ref.local_row>=0&&ref.global_row>=p.cfg_row_start_[ref.cfg_index]&&ref.global_row<p.cfg_row_start_[ref.cfg_index+1]),"invalid active row reference");
    p.selected_row_indices_.resize(p.B_full_.rows());std::iota(p.selected_row_indices_.begin(),p.selected_row_indices_.end(),0);p.has_full_B_=false;p.using_stream_state_=true;
    if(online)require(p.online_state_.initialize_from_rows(p.B_full_,p.selected_row_refs_),"cannot restore online factorization");
    p.gpm_rebuild_full_=p.select_mode=="full";p.init_builder(pot);require(p.prepare_grading(),"cannot restore MaxVol grading");return owner;
}
void GPMALAccess::encode_selector(Writer&w,ActiveSelector&s,Potentials::AnyPotential&p) {
    auto o=binding(w,"maxvol");o["maxvol"]=Value(selector(w,s,p));o["binding_fingerprint"]=Value(binding_hash(string(at(o,"model_fingerprint")),at(object(o["maxvol"]),"builder"),[&](const std::string&id){return w.block_hash(id);}));w.manifest["active_learning"]=Value(o);
}
void GPMALAccess::encode(Writer&w,ALPotential&p,Potentials::AnyPotential&pot) {
    require(p.scorer_!=nullptr,"no active AL scorer");
    if(p.bound_model_)require(Potentials::ModelBundle::fingerprint(pot)==p.bound_fingerprint_,"model changed since AL binding");
    if(p.mode_name_=="maxvol"){auto *s=dynamic_cast<MaxVolScorer*>(p.scorer_.get());require(s&&s->as_,"missing MaxVol state");encode_selector(w,*s->as_,pot);}
    else require(false,"unsupported AL method");
    auto &o=w.manifest.at("active_learning").get<Object>();State a(w,"al/policy");
#define F(x) a.field(#x,p.x)
    F(threshold_);F(threshold_break_);F(should_abort_);F(sparsify_method_);F(sparsify_to_);F(check_interval_);F(step_counter_);
#undef F
    o["policy"]=Value(a.fields());o["candidate_pool"]=structures(w,"al/candidates",p.pool_);
}
std::unique_ptr<ALPotential> GPMALAccess::decode(const Reader&r,Potentials::AnyPotential&pot,const GPSettings&overrides) {
    require(overrides["state_file"].empty()&&overrides["committee_prefix"].empty(),"embedded AL conflicts with external state");
    const auto &o=object(at(r.manifest,"active_learning"));require(integer(at(o,"version"))==1,"unsupported AL version");
    if(o.count("required_features"))require(array(at(o,"required_features")).empty(),"unsupported AL capability");
    require(string(at(o,"target_model"))==string(at(r.manifest,"root_model"))&&string(at(o,"model_fingerprint"))==Potentials::ModelBundle::fingerprint(pot),"AL model fingerprint mismatch");
    auto p=std::make_unique<ALPotential>();p->mode_name_=string(at(o,"method"));
    require(overrides["mode"].empty()||overrides["mode"]==p->mode_name_,"explicit AL method conflicts with embedded state");
    if(p->mode_name_=="maxvol"){
        require(string(at(o,"binding_fingerprint"))==binding_hash(string(at(o,"model_fingerprint")),at(object(at(o,"maxvol")),"builder"),[&](const std::string&id){return r.info(id).sha256;}),"AL builder binding mismatch");
        auto selected=selector(r,object(at(o,"maxvol")),pot);auto s=std::make_unique<MaxVolScorer>();s->as_=selected.release();s->own_selector_=true;s->loaded_=true;s->builder_initialized_=true;s->cutoff_=pot.cutoff;p->scorer_=std::move(s);}
    else require(false,"unsupported AL method");
    if(o.count("policy")){State a(r,object(o.at("policy")));
#define F(x) a.field(#x,p->x)
        F(threshold_);F(threshold_break_);F(should_abort_);F(sparsify_method_);F(sparsify_to_);F(check_interval_);F(step_counter_);
#undef F
    }
    if(o.count("candidate_pool"))p->pool_=structures(r,o.at("candidate_pool"));
    auto override_value=[&](const char *key,auto &value){if(!overrides[key].empty())Parameters::bind(value,key)->apply_setting(overrides[key],"embedded AL");};
    override_value("threshold",p->threshold_);override_value("threshold_break",p->threshold_break_);override_value("interval",p->check_interval_);override_value("sparsify_to",p->sparsify_to_);override_value("sparsify_method",p->sparsify_method_);override_value("export_file",p->export_file_);
    require(!std::isnan(p->threshold_)&&!std::isnan(p->threshold_break_)&&p->check_interval_>0&&p->step_counter_>=0,"invalid AL policy");
    require(p->sparsify_method_=="fps"||p->sparsify_method_=="random","unsupported sparsification method");p->bound_model_=&pot;p->bound_fingerprint_=Potentials::ModelBundle::fingerprint(pot);return p;
}
} // namespace AL
