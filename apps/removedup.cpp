// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"

#include <stdexcept>

using namespace std;

namespace {

Atoms::DuplicateAtomMode parse_duplicate_atom_mode(const string &mode)
{
    if (mode == "delete") return Atoms::DuplicateAtomMode::Delete;
    if (mode == "merge") return Atoms::DuplicateAtomMode::Merge;
    throw invalid_argument("remove_duplicate_atoms mode must be 'delete' or 'merge'");
}

} // namespace

int remove_duplicate_atoms(int argc, char *argv[]) {
    GPIO iobj;
    auto &cmdopts = CMDOption::global_cmdopts;
    string inputfile = "POSCAR";
    string outputfile = "POSCAR_REMOVED_DUPS";
    string informat = "POSCAR";
    string outformat = "POSCAR";
    string index = "index.map";
    string mode_str = "delete";
    double rtol = 0.5;
    cmdopts.addop(inputfile, "input", "in", "input file, default POSCAR", true)
           .addop(outputfile, "output", "o", "Output file, default POSCAR_REMOVED_DUPS", true)
           .addop(informat, "input-format", "if", "input format, default POSCAR")
           .addop(outformat, "output-format", "of", "output format, default POSCAR")
           .addop(index, "index", "id", "index file, default index.map")
           .addop(mode_str, "mode", "m", "duplicate handling mode: delete or merge, default delete")
           .addop(rtol, "rtol", "rt", "tolerance for removing duplicate atoms, default 0.5A");
    cmdopts.phrase_options(argc, argv);

    Atoms::DuplicateAtomMode mode;
    try {
        mode = parse_duplicate_atom_mode(mode_str);
        if (rtol <= 0.0) {
            throw invalid_argument("remove_duplicate_atoms rtol must be positive");
        }
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    if (!GPIO::file_exist(inputfile)) {
        cerr << "Error: input file not found: " << inputfile << endl;
        return 1;
    }
    if (!GPIO::file_exist(index)) {
        iobj.setmap();
    } else if (!iobj.readmap(index)) {
        return 1;
    }

    vector<Atoms> atoms;
    if (!iobj.read(atoms, inputfile, informat)) {
        cerr << "Error: failed to read input file: " << inputfile << endl;
        return 1;
    }
    if (atoms.empty()) {
        cerr << "Error: no atoms found in input file: " << inputfile << endl;
        return 1;
    }

    int total_removed = 0;
    for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
        auto stats = atoms[i].remove_duplicate_atoms(rtol, mode);
        total_removed += stats.removed_atoms;
        cout << "Frame " << i << ": " << stats.initial_atoms << " -> "
             << stats.final_atoms << " atoms, removed " << stats.removed_atoms
             << " in " << stats.duplicate_groups << " duplicate groups" << endl;
    }

    if (!iobj.write(atoms, outputfile, outformat)) {
        cerr << "Error: failed to write output file: " << outputfile << endl;
        return 1;
    }

    cout << "Removed " << total_removed << " duplicate atoms, wrote "
         << outputfile << endl;
    return 0;
}
