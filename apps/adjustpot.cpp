// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/potential_mtp.h"
#include "mtp_workflow.h"
#include "mtp_tree_workflow.h"
using namespace std;
int adjustmtp(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    string infile = "potential.mtp";
    string outfile = "output-adj.mtp";
    string settings_file,type="auto",basis_order,initialization;
    vector<string> appended;vector<int> sources;
    vector<string> element_names;
    vector<int> species_order; // reorder species also
    cmdopts.addop(settings_file,"settings","","Model factory configuration")
           .addop(type,"type","","Input model family (GPM infers it)")
           .addop(basis_order,"basis-order","","keep, level, or level-group")
           .addop(appended,"append-elements","","New element names",false,-1)
           .addop(sources,"clone-from","","Original species IDs for new species",false,-1)
           .addop(initialization,"new-species-init","","clone or random")
           .addop(infile, "input", "in", "input file", true, 1)
           .addop(outfile, "output", "out", "output file", true, 1)
           .addop(element_names, "elements", "", "Element symbols in output model species order", false, -1)
           .addop(species_order, "neworder", "", "new species order, i.e 2 1 0 means 0 1 2 -> 2 1 0", true, -1)
           .phrase_options(argc, argv);
    if(cmdopts.displayhelp){cmdopts.showhelp(cout);return 0;}
    const bool modern=GPM::is_file(infile)||GPM::extension(infile)||GPM::extension(outfile)||type!="auto"||!basis_order.empty()||!appended.empty()||!settings_file.empty();
    if(modern) {
        Training::collective_try([&]{
            if(mpi.rank!=0)return;
            MTPWorkflow::LocalComputation local;
            GPSettings all;if(!settings_file.empty())all.Load(settings_file);
            auto config=all.ExtractSubSettings("pot");config.Modify("potfile",infile);if(type!="auto"||config["pottype"].empty())config.Modify("pottype",type);
            std::string input_magic;if(MTPWorkflow::fs::exists(infile)){auto stream=open_ifile(infile);if(stream)*stream>>input_magic;}
            if(input_magic=="MTP"&&!GPM::is_file(infile)&&!GPM::extension(infile)&&(config["pottype"]=="auto"||config["pottype"]=="mtp_m")) {
                const auto header=MTPMetadata::read_header(infile);
                if(header.tag.text("type")=="MTP_LES")config.Modify("pottype","mtp_les");
                if(header.tag.text("type")=="MTP_M") {
                    config.Modify("pottype","mtp_m");
                    if(config["extmodel"].empty())config.Modify("extmodel",header.tag.text("model"));
                    if(config["n_input"].empty()&&header.tag.fields.count("q"))config.Modify("n_input",std::to_string(int(header.tag.fields.at("q").get<double>())));
                    if(header.tag.fields.count("nn")) {
                        const auto &nn=header.tag.fields.at("nn").get<MTPMetadata::Object>();const auto &layers=nn.at("hidden_layers").get<MTPMetadata::Array>();
                        for(size_t i=0;i<layers.size();++i){auto key=i?"n_neuron2":"n_neuron";if(config[key].empty())config.Modify(key,std::to_string(int(layers[i].get<double>())));}
                        if(config["activation_function"].empty())config.Modify("activation_function",nn.at("activation").get<std::string>());
                    }
                }
            }
            if(config["pottype"]=="auto"&&!GPM::is_file(infile)&&!GPM::extension(infile)) {
                auto stream=open_ifile(infile);std::string magic;if(stream)*stream>>magic;
                if(magic=="pottype"){std::string declared;*stream>>declared;config.Modify("pottype",declared);}
                else if(magic=="EDQRD")config.Modify("pottype","mtp_edqrd");
                else if(MTPWorkflow::fs::exists(infile+".nn"))config.Modify("pottype","mtpnn1");
                else if(MTPWorkflow::fs::exists(infile+"_sub1")){config.Modify("pottype","hmtp");config.Modify("potlevel",MTPWorkflow::fs::exists(infile+"_sub3")?"3":"2");}
            }
            Potentials::potential_wrapper wrapper(config);
            MTPWorkflow::Adjustment operation;operation.appended=appended;operation.elements=element_names;operation.sources=sources;operation.order=species_order;operation.initialization=initialization;operation.basis=basis_order.empty()?"keep":basis_order;
            auto adjusted=MTPWorkflow::adjust_tree(*wrapper.pot,operation);
            const auto output=cmdopts.was_provided("output")?outfile:"_Adjusted_Potential_.gpm";
            if(!cmdopts.was_provided("output")&&MTPWorkflow::fs::absolute(output).lexically_normal()==MTPWorkflow::fs::absolute(infile).lexically_normal())GPERROR("Explicit output is required to overwrite the input");
            MTPWorkflow::save(*adjusted,output);
        });return 0;
    }
    Training::collective_try([&]{
    if(mpi.rank!=0)return;
    MTPWorkflow::LocalComputation local;
    Potentials::PotMTP pot(infile);
    if (species_order.size() != 0) {
        if (species_order.size() > static_cast<size_t>(pot.species_count)) {
            cout << "Add new species into potential, this part of coefs will be initialized randomly" << endl;
        }
        const auto old_names = pot.potential_tag.elements();
        if (species_order.size() > static_cast<size_t>(pot.species_count) && !old_names.empty()) {
            if (element_names.size() != species_order.size())
                GPERROR("Tagged species expansion requires --elements for every output species");
            for (size_t i = 0; i < species_order.size(); ++i) {
                const int source = species_order[i];
                if (source >= 0 && source < pot.species_count && element_names[i] != old_names[source])
                    GPERROR("Output element order conflicts with existing species identities");
            }
            MTPMetadata::Tag checked;
            checked.set_elements(element_names, static_cast<int>(species_order.size()));
            pot.potential_tag.fields.erase("elements");
        }
        pot.adjust_species(species_order);
    }
    if (!element_names.empty()) {
        map<int, string> mapping;
        for (size_t i = 0; i < element_names.size(); ++i) mapping[static_cast<int>(i)] = element_names[i];
        pot.set_element_map(mapping);
    }
    MTPWorkflow::validate_graph(pot);
    pot.ReorderByLevel();
    const auto target=MTPWorkflow::fs::absolute(outfile).lexically_normal();
    if(!cmdopts.was_provided("output")&&target==MTPWorkflow::fs::absolute(infile).lexically_normal())GPERROR("Explicit output is required to overwrite the input");
    MTPWorkflow::Stage stage(target.parent_path());const auto temporary=(stage.path/target.filename()).string();
    pot.SaveOrdered(temporary);Potentials::PotMTP check(temporary);
    if(check.inited()!=pot.inited())GPERROR("Adjusted model failed native reload validation");
    MTPWorkflow::fs::rename(temporary,target);
    });
    return 0;
}
