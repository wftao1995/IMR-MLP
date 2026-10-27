// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"

using namespace std;

static int subtract_pot(int argc, char *argv[]) {
    string inputfile1 = "input1.cfg";
    string inputfile2 = "input2.cfg";
    string outputfile = "output.cfg";
    string idfile = "index.map";
    GPIO iobj;
    vector<Atoms> atoms1, atoms2;
    bool showhelp = false;
    bool overwrite = false;
    string informat = "";
    string outformat = "";
    auto &cmdopts = CMDOption::global_cmdopts;
    cmdopts.addop(idfile, "index", "id", "Index map file")
           .addop(informat, "input-format", "if", "Input file format")
           .addop(overwrite, "overwrite", "ov", "Overwrite existing file")
           .addop(inputfile1, "input1", "", "First input file", true)
           .addop(inputfile2, "input2", "", "Second input file", true)
           .addop(outputfile, "output", "o", "Output file");
    cmdopts.phrase_options(argc, argv);
    if (GPIO::file_exist(outputfile) && !overwrite)
    {
        cerr << "Error: output file " << outputfile << " exists, used --overwrite to overwrite it" << endl;
        exit(128);
    }
    // iobj.setmap();
    if (!GPIO::file_exist(idfile))
    {
        iobj.setmap();
    }
    else
    {
        iobj.readmap(idfile);
    }

    if (!iobj.read(atoms1, inputfile1, informat))
    {
        cerr << "Failed to read cfg file" << endl;
        exit(128);
    }
    if (!iobj.read(atoms2, inputfile2, informat))
    {
        cerr << "Failed to read cfg file" << endl;
        exit(128);
    }

    if (atoms1.empty() || atoms1.size() != atoms2.size())
        GPERROR("Inputs must have equal nonzero frame counts");

    for (int i = 0; i < atoms1.size(); ++i) {
        auto &at1 = atoms1[i];
        auto &at2 = atoms2[i];
        const auto frame = " at frame " + std::to_string(i);
        if (at1.size() == 0 || at1.size() != at2.size() || at1.chemmap_ != at2.chemmap_)
            GPERROR("Inputs must have matching atoms and species" + frame);
        if (at1.has_energy() != at2.has_energy() ||
            at1.has_forces() != at2.has_forces() ||
            at1.has_stresses() != at2.has_stresses() ||
            at1.has_site_energies() != at2.has_site_energies())
            GPERROR("Inputs must have matching energy, force, stress and site-energy labels" + frame);
        if (at1.has_forces() &&
            (at1.forces_.size() != at1.size() || at2.forces_.size() != at1.size()))
            GPERROR("Force labels must contain one entry per atom" + frame);
        if (at1.has_site_energies() &&
            (at1.site_energies_.size() != at1.size() || at2.site_energies_.size() != at1.size()))
            GPERROR("Site-energy labels must contain one entry per atom" + frame);
        if (at1.has_energy()) at1.energy_ -= at2.energy_;
        if (at1.has_site_energies() && at2.has_site_energies()) {
            for (int j = 0; j < at1.size(); ++j) {
                at1.site_energies_[j] -= at2.site_energies_[j];
            }
        }
        if (at1.has_forces() && at2.has_forces()) {
            for (int j = 0; j < at1.size(); ++j) {
                at1.forces_[j] -= at2.forces_[j];
            }
        }
        if (at1.has_stresses() && at2.has_stresses()) {
            at1.stress_ -= at2.stress_;
        }
    }
    if (!iobj.write(atoms1, outputfile, outformat)) return 1;
    return 0;
}

int main(int argc, char *argv[])
{
    try {
        return subtract_pot(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
