// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "analyse/trajectory_transform.h"
#include "analyse/compute_posavg.h"
#include "tools/chem.h"
using namespace std;
int calc_trjavg(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    string inputfile = "";
    string outputfile = "POSCAR_IDEAL";
    string indexfile = "index.map";
    string informat = "";
    string outformat = "";
    GPIO iobj;
    Atoms atom;
    vector<Atoms> atoms;
    int startfrom = 0; // from zero, phrase input from 1
    int end = INT_MAX; // from zero, phrase input from 1, include end
    int loadrate = 0;  // when loadrate is one, use keepone mode
    bool overwrite = false;
    bool appendto = false;
    bool wrap = true;
    string coordinates = "auto";
    string drift = "auto";

    cmdopts.addop(informat, "input-format", "if")
        .addop(outformat, "output-format", "of")
        .addop(indexfile, "index", "id")
        .addop(loadrate, "loadrate", "lr")
        .addop(startfrom, "start", "st")
        .addop(end, "end", "ed")
        .addop(overwrite, "overwrite", "ov", "", false, 0)
        .addop(appendto, "append", "a", "", false, 0)
        .addop(inputfile, "input-file", "", "", true)
        .addop(outputfile, "output-file", "", "", true)
        .addop(wrap, "wrap", "w", "wrap at PBC")
        .addop(coordinates, "average-coordinates", "", "auto (Cartesian for fixed cells, fractional otherwise), cartesian or fractional")
        .addop(drift, "drift", "", "auto (remove for fixed cells), remove or keep")
        .add_request("input-file")
        .phrase_options(argc, argv);

    startfrom--;
    if (end != INT_MAX)
        end--;
    if (!iobj.file_exist(inputfile))
    {
        cerr << "Error: inputfile " << inputfile << " not found" << endl;
        exit(128);
    }
    // if format not present, guess it
    // if (informat == "")
    //     informat = GPIO::guess_format(inputfile);
    // if (outformat == "")
    //     outformat = GPIO::guess_format(outputfile);

    if (!GPIO::file_exist(indexfile))
    {
        cerr << "Warning: no index file found" << endl;
        iobj.setmap();
    }
    else if (!iobj.readmap(indexfile))
    {
        cerr << "Warning: failed to read index file, use atom numbers instead" << endl;
        iobj.setmap();
    }
    if (overwrite && appendto)
    {
        cerr << "Conflict settings: append and overwrite" << endl;
        exit(128);
    }
    if (iobj.file_exist(outputfile) && (!overwrite && !appendto))
    {
        cerr << "Error: outputfile " << outputfile << " exist, please use --overwrite or --append" << endl;
        exit(128);
    }

    std::vector<Atoms> all_atoms;
    if (!iobj.read(all_atoms, inputfile, informat))
        throw std::runtime_error("Failed to read trajectory: " + inputfile);
    const int frame_stride = loadrate > 0 ? loadrate : 1;
    for (int iframe = std::max(0, startfrom);
         iframe < static_cast<int>(all_atoms.size()) && iframe <= end;
         iframe += frame_stride)
        atoms.push_back(std::move(all_atoms[iframe]));
    if (atoms.empty())
        throw std::invalid_argument("Selected trajectory range contains no frames");
    if (coordinates != "auto" && coordinates != "cartesian" && coordinates != "fractional")
        throw std::invalid_argument("average-coordinates must be auto, cartesian or fractional");
    if (drift != "auto" && drift != "remove" && drift != "keep")
        throw std::invalid_argument("drift must be auto, remove or keep");
    const double cell_scale = std::max(1.0, atoms.front().cell_.norm());
    const bool fixed_cell = std::all_of(atoms.begin(), atoms.end(), [&](const Atoms &frame) {
        return (frame.cell_ - atoms.front().cell_).norm() <= 1e-7 * cell_scale;
    });
    PositionAverageOptions options;
    options.coordinates = coordinates == "fractional" || (coordinates == "auto" && !fixed_cell)
        ? AverageCoordinates::Fractional : AverageCoordinates::Cartesian;
    options.remove_drift = drift == "remove" || (drift == "auto" && fixed_cell);
    options.geometry_only = true;
    options.wrap_result = wrap;
    if (options.remove_drift) {
        std::map<int, double> masses_by_type;
        for (const auto &[type, name] : iobj.getmap()) {
            const auto values = iobj.atomic_vals.find(type);
            if (values != iobj.atomic_vals.end() && values->second.size() >= 2) {
                masses_by_type[type] = values->second[1];
            } else {
                const auto found = Chemicals::atomic_numbers.find(name);
                if (found == Chemicals::atomic_numbers.end())
                    throw std::invalid_argument("Unknown element in index map: " + name);
                masses_by_type[type] = Chemicals::atomic_masses[found->second];
            }
        }
        for (int type : atoms.front().chemmap_) {
            const auto found = masses_by_type.find(type);
            if (found == masses_by_type.end() || !std::isfinite(found->second) || found->second <= 0.0)
                throw std::invalid_argument("Missing positive atomic mass for trajectory atom");
        }
        for (int type : atoms.front().chemmap_) options.masses.push_back(masses_by_type.at(type));
    }
    ComputePosavg average(std::move(options));
    average.init(atoms.front());
    for (const auto &frame : atoms) average.accumulate(frame);
    average.compute();
    atom = std::move(average.atomacc);
    iobj.append = appendto;
    if (!iobj.write_one(atom, outputfile, outformat))
        return 1;

    return 0;
}
