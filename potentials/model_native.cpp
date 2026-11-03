// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "model_bundle.h"
#include "gpm_codec.h"
#include "potential_wrapper.h"
#include "potential_hybrid.h"
#include "potential_hybrid_scale.h"
#include "active_learn/gpm_al.h"
#include "maxvol/active_selector.h"
#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <set>
#include <unistd.h>

namespace Potentials {
namespace {
namespace fs=std::filesystem;
using namespace GPM;

// Native writers are optional; the base implementation may write nothing.
void check_native(const Object &models,const std::string &id) {
    const auto &node=object(at(models,id));const auto type=string(at(node,"type"));
    static const std::set<std::string> supported={"mtp","mtp_m","mtp_les","mtpnn1","mtp_edq","mtp_edqrd",
        "hmtp","hybrid","hybrid_scale","eam_trainable","eampoly","lj612","zbl","mindist",
        "sw","tersoff","pair_table","threebody_table","descriptor_linear"};
    require(supported.count(type),"native export is not supported for model type: "+type);
    for(const auto &[key,child]:object(at(node,"children")))check_native(models,string(child));
}

void write_model(AnyPotential &model,const fs::path &path,const Object &models,const std::string &id,const Reader &reader) {
    const auto &node=object(at(models,id));const auto type=string(at(node,"type"));
    if(type=="descriptor_linear"&&object(at(node,"children")).count("descriptor")) {
        const auto child_id=string(at(object(at(node,"children")),"descriptor"));
        auto child=GPMCodecAccess::decode(reader,child_id);
        const auto name=path.filename().string()+".descriptor.mtp";
        write_model(*child,path.parent_path()/name,models,child_id,reader);
        std::ostringstream text;model.Save(text);std::istringstream lines(text.str());std::ofstream out(path);std::string line;
        while(std::getline(lines,line)) {
            if(line.starts_with("descriptor:mtm:potfile"))line="descriptor:mtm:potfile "+name;
            out<<line<<'\n';
        }
        out.close();require(bool(out),"failed writing native descriptor model: "+path.string());return;
    }
    if(type!="hybrid"&&type!="hybrid_scale") {model.Save(path.string());return;}
    // Hybrid::Save historically writes only trainable MTPs. Export the entire graph.
    auto children=type=="hybrid"?static_cast<PotentialHybrid&>(model).children():static_cast<PotentialHybridScale&>(model).children();
    std::ofstream out(path);require(bool(out),"cannot write native manifest: "+path.string());
    out<<"pottype "<<type<<'\n'<<"spcount "<<model.species_count<<'\n';
    for(const auto &[key,value]:object(at(node,"configuration")))out<<key<<' '<<string(value)<<'\n';
    for(size_t i=0;i<children.size();++i) {
        const auto key="pot"+std::to_string(i+1),name=path.filename().string()+"."+key;
        const auto child=string(at(object(at(node,"children")),key));
        write_model(*const_cast<AnyPotential*>(children[i]),path.parent_path()/name,models,child,reader);
        out<<key<<":potfile "<<name<<'\n';
    }
    out.close();require(bool(out),"failed writing native manifest: "+path.string());
}

void publish(const fs::path &stage,const fs::path &destination,const std::string &output) {
    std::vector<fs::path> names,backups,installed;
    for(const auto &entry:fs::directory_iterator(stage))names.push_back(entry.path().filename());
    for(const auto &suffix:{".sel",".index.map"}) {
        const fs::path name=output+suffix;
        if(!fs::exists(stage/name)&&(fs::exists(destination/name)||fs::is_symlink(destination/name)))names.push_back(name);
    }
    std::sort(names.begin(),names.end());
    std::string backup=(stage/".backup-XXXXXX").string();
    require(::mkdtemp(backup.data()),"cannot stage native output backups");
    // Reject directories before moving any old output.
    for(const auto &name:names)require(!fs::is_directory(destination/name),"native output is a directory: "+(destination/name).string());
    try {
        for(const auto &name:names) {
            const auto target=destination/name;
            if(fs::exists(target)||fs::is_symlink(target)){fs::rename(target,fs::path(backup)/name);backups.push_back(name);}
            if(fs::exists(stage/name)){fs::rename(stage/name,target);installed.push_back(name);}
        }
    } catch(...) {
        const auto error=std::current_exception();
        for(auto it=installed.rbegin();it!=installed.rend();++it)fs::remove(destination / *it);
        for(auto it=backups.rbegin();it!=backups.rend();++it)fs::rename(fs::path(backup) / *it,destination / *it);
        std::rethrow_exception(error);
    }
}
}

void ModelBundle::save_native(const std::string &path,Profile profile) {
    using namespace GPM;namespace fs=std::filesystem;
    require(!extension(path),"--legacy requires a native output path, not .gpm");
    require(bool(source_),"native conversion requires a GPM input");
    source_->validate();
    const auto &header=source_->manifest;const auto &models=object(at(header,"models"));
    const auto root=string(at(header,"root_model"));check_native(models,root);
    const bool has_al=header.count("active_learning");
    require(profile!=Profile::Resume||has_al,"resume profile requires AL state");
    std::unique_ptr<ActiveSelector> selector;
    if(has_al&&profile!=Profile::Inference)selector=AL::GPMALAccess::decode_selector(*source_,&potential());
    const auto output=fs::absolute(path).lexically_normal();
    require(!output.filename().empty(),"native output must name a file");
    std::string pattern=(output.parent_path()/".native-export-XXXXXX").string();
    require(::mkdtemp(pattern.data()),"cannot stage native output: "+path);
    struct Stage {fs::path p;~Stage(){std::error_code ec;fs::remove_all(p,ec);}} stage{pattern};
    const auto temporary=stage.p/output.filename();
    write_model(potential(),temporary,models,root,*source_);
    auto species=array(at(header,"species"));bool complete=!species.empty();
    for(auto &entry:species) {
        auto &s=entry.get<Object>();const auto found=potential().gpm_elements.find(integer(at(s,"id")));
        if(found!=potential().gpm_elements.end())s["symbol"]=Value(found->second);
    }
    for(const auto &entry:species)complete=complete&&object(entry).count("symbol");
    std::string map;
    if(complete) {
        map=temporary.string()+".index.map";std::ofstream out(map);
        for(const auto &entry:species){const auto &s=object(entry);out<<string(at(s,"symbol"))<<' '<<integer(at(s,"id"))<<'\n';}
        out.close();require(bool(out),"failed writing native element map");
    }
    GPSettings settings;
    settings.Modify("potfile",temporary.string());settings.Modify("pottype",type(potential()));settings.Modify("typemap",map);
    if(type(potential())=="hmtp")settings.Modify("potlevel",std::to_string(object(at(object(at(models,root)),"children")).size()));
    if(type(potential())=="mtp_m") {
        const auto &metadata=object(at(object(at(object(at(models,root)),"configuration")),"metadata"));
        settings.Modify("extmodel",string(at(metadata,"model")));
        if(metadata.count("q"))settings.Modify("n_input",std::to_string(integer(metadata.at("q"))));
        if(metadata.count("nn")) {
            const auto &nn=object(metadata.at("nn"));const auto &layers=array(at(nn,"hidden_layers"));
            settings.Modify("n_neuron",std::to_string(integer(layers.at(0))));
            if(layers.size()>1)settings.Modify("n_neuron2",std::to_string(integer(layers.at(1))));
            settings.Modify("activation_function",string(at(nn,"activation")));
        }
    }
    // The companion settings file records factory options needed by prefix-based families.
    std::ofstream config(temporary.string()+".ini");
    if(!map.empty())config<<"index "<<fs::path(map).filename().string()<<'\n';
    for(const auto &[key,value]:settings.settings) {
        if(value.empty())continue;
        config<<"pot:"<<key<<' '<<((key=="potfile"||key=="typemap")?fs::path(value).filename().string():value)<<'\n';
    }
    if(selector) {
        const auto &pb=selector->builder_parameters();BMatrixBuilderSettings builder;
        builder.dtype=pb.dtype;builder.mode=pb.mode;builder.potfile=output.filename().string();builder.pottype=type(potential());
        builder.all_coeffs=!pb.linear_only;builder.row_scaling=to_string(pb.row_scaling);
        builder.w_energy=pb.w_energy;builder.w_site=pb.w_site;builder.w_force=pb.w_force;builder.w_stress=pb.w_stress;
        config<<"state-file "<<output.filename().string()<<".sel\n";
        for(const auto &[key,value]:builder.CurrentSettings().settings)if(!value.empty())config<<"builder:"<<key<<' '<<value<<'\n';
        for(const auto &[key,value]:selector->CurrentSettings().settings)if(!value.empty())config<<"selector:"<<key<<' '<<value<<'\n';
        for(const auto &[key,value]:pb.settings.settings)if(key.starts_with(pb.dtype+":")&&!value.empty())config<<key<<' '<<value<<'\n';
    }
    config.close();require(bool(config),"failed writing native reload settings");
    potential_wrapper reloaded(settings);
    require(type(*reloaded.pot)==type(potential())&&reloaded.pot->species_count==potential().species_count,"native reload model mismatch");
    auto inference_signature=[](AnyPotential &model) {
        Writer writer;GPMCodecAccess::encode(writer,model);
        for(auto &[id,node]:writer.manifest.at("models").get<Object>()) {
            // Native wrappers propagate the root element map to their children.
            for(auto &entry:node.get<Object>().at("species").get<Array>()) {
                auto &s=entry.get<Object>();const auto element=model.gpm_elements.find(integer(at(s,"id")));
                if(!s.count("symbol")&&element!=model.gpm_elements.end())s["symbol"]=Value(element->second);
            }
            auto &common=node.get<Object>().at("common").get<Object>();
            for(const auto *key:{"reg_L1","reg_L2","add_efs"})common.erase(key);
        }
        return model_signature(writer);
    };
    require(inference_signature(potential())==inference_signature(*reloaded.pot),
            "native format cannot preserve this model's complete inference data");
    if(selector) {
        selector->save_state(temporary.string()+".sel",map);
        ActiveSelector check;check.load_state(temporary.string()+".sel",map);check.init_builder(*reloaded.pot);
    }
    publish(stage.p,output.parent_path(),output.filename().string());
}
}
