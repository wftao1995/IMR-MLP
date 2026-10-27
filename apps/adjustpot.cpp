// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/potential_mtp.h"
using namespace std;
int adjustmtp(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    string infile = "potential.mtp";
    string outfile = "output-adj.mtp";
    vector<string> element_names;
    vector<int> species_order; // reorder species also
    cmdopts.addop(infile, "input", "in", "input file", true, 1)
           .addop(outfile, "output", "out", "output file", true, 1)
           .addop(element_names, "elements", "", "Element symbols in output model species order", false, -1)
           .addop(species_order, "neworder", "", "new species order, i.e 2 1 0 means 0 1 2 -> 2 1 0", true, -1)
           .phrase_options(argc, argv);
    Potentials::PotMTP pot(infile);
    if (species_order.size() != 0) {
        if (species_order.size() > static_cast<size_t>(pot.species_count)) {
            cout << "Add new species into potential, this part of coefs will be initialized randomly" << endl;
        }
        const auto old_names=pot.element_symbols();
        if(species_order.size()>static_cast<size_t>(pot.species_count)&&!old_names.empty()){
            if(element_names.size()!=species_order.size())GPERROR("Tagged species expansion requires --elements for every output species");
            for(size_t i=0;i<species_order.size();++i){int source=species_order[i];if(source>=0&&source<pot.species_count&&element_names[i]!=old_names[source])GPERROR("Output element order conflicts with existing species identities");}
            MTPMetadata::Tag checked;
            checked.set_elements(element_names, static_cast<int>(species_order.size()));
            pot.potential_tag.fields.erase("elements");
        }
        pot.adjust_species(species_order);
    }
    if(!element_names.empty())pot.set_element_symbols(element_names);
    pot.ReorderByLevel();
    pot.SaveOrdered(outfile);
    return 0;
}
