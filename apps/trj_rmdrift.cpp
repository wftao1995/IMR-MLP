// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "utils/cmdopts.h"
#include "tools/chem.h"
#include "tools/tools.h"
#include "analyse/trajectory_transform.h"


using namespace std;
using namespace Eigen;

int main(int argc, char *argv[]) try {
    GPIO iobj;
    bool showhelp = false;
    string indexfile = "index.map";
    string informat = "";
    string outformat = "";
    string trjfile = "";
    string outfile = "";
    vector<Atoms> atoms;
    Atoms anatom;
    bool overwrite = false;

    auto &cmdopts = CMDOption::global_cmdopts;
    cmdopts.addop(indexfile, "index", "id", "Index file")
           .addop(iobj.absolute_index_read, "absolute-index-read", "abr", "Use atomic numbers when reading")
           .addop(iobj.absolute_index_write, "absolute-index-write", "abw", "Use atomic numbers when writing")
           .addop(informat, "input-format", "if", "Input file format")
           .addop(outformat, "output-format", "of", "Output file format")
           .addop(overwrite, "overwrite", "ov", "Overwrite existing file")
           .addop(trjfile, "input", "i", "Input trajectory file", true)
           .addop(outfile, "output", "o", "Output file", true);
    cmdopts.phrase_options(argc, argv);
    if (cmdopts.displayhelp) return 0;
    if (GPIO::file_exist(indexfile))
    {
        if (!iobj.readmap(indexfile))
            throw std::runtime_error("Failed to read index file: " + indexfile);
    }
    else
    {
        cerr << "ERROR: indexfile not found\n";
        exit(1);
    }
    if (!GPIO::file_exist(trjfile)) {
        cerr << "ERROR: inputfile " + trjfile + " not found\n";
        exit(1);
    }
    if (outfile == "") {
        cerr << "Warning: outputfile not specified\n";
        exit(1);
    }
    if (GPIO::file_exist(outfile) && !overwrite) {
        cerr << "ERROR: file exists, please use --overwrite to overwrite it\n";
        exit(1);
    }

    // iobj.set(atoms);
    // iobj.set(anatom);
    if (!iobj.read(atoms, trjfile, informat)) {
        std::cerr << "Erroring loading trajectories\n";
        exit(1);
    }
    if (atoms.empty()) throw std::invalid_argument("Trajectory contains no frames");
    for (const auto &frame : atoms)
        if (frame.size() == 0 || frame.size() != atoms.front().size() ||
            frame.chemmap_ != atoms.front().chemmap_ || frame.pbc_ != atoms.front().pbc_)
            throw std::invalid_argument("Trajectory atom count, type ordering or periodic axes change between frames");
    map<int, double> mass;
    auto spmap = iobj.getmap();
    for (auto &it : spmap) {
        if (iobj.atomic_vals[it.first].size() >= 2){
            mass[it.first] = iobj.atomic_vals[it.first][1];
        }
        else {
            const auto element = Chemicals::atomic_numbers.find(iobj.get_type(it.first));
            if (element == Chemicals::atomic_numbers.end())
                throw std::invalid_argument("Unknown element in index map");
            mass[it.first] = Chemicals::atomic_masses[element->second];
        }
        if (!std::isfinite(mass[it.first]) || mass[it.first] <= 0)
            throw std::invalid_argument("Atomic mass must be finite and positive");
    }
    for (auto &it : mass) {
        std::cout << "Mass of type " << it.first << " is " << it.second << std::endl;
    }
    unwarp(atoms);
    for (int i = 0; i < atoms.size(); ++i) {
        auto &atom = atoms[i];
        if (!atom.cartesian) atom.to_cartesian();
        RowVector3d cm = RowVector3d::Zero();
        double mass_total = 0;
        for (int iat = 0; iat < atom.size(); ++iat) {
            const auto value = mass.find(atom.chemmap_[iat]);
            if (value == mass.end()) throw std::invalid_argument("Missing atomic mass for trajectory atom");
            cm += atom.pos_[iat] * value->second;
            mass_total += value->second;
        }
        cm /= mass_total;
        for (int iat = 0; iat < atom.size(); ++iat) {
            atom.pos_[iat] -= cm;
        }
        // atom.wrap();
    }


    anatom = atoms[0];
    return iobj.write(atoms, outfile, outformat) ? 0 : 1;
}
catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
}
