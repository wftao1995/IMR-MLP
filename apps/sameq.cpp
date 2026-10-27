// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file sameq.cpp
 * @author your name (you@domain.com)
 * @brief Check if two structures are the same, spglib-free version
 * @version 0.1
 * @date 2025-11-23
 *
 * @copyright Copyright (c) 2025
 * TODO: no-spglib version for large cell, spglib version for small cell
 */

#include "io/gpio.h"
#include "symmetry/structure_comparison.h"
#include <limits>
using namespace std;
int sameq(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    string inputfile1 = ""; // structure to compare I
    string inputfile2 = ""; // structure to compare II
    string indexfile = "index.map";
    string informat1 = "";
    string informat2 = "";
    GPIO iobj;
    Atoms atom1, atom2;
    double tol = 1e-4;
    bool scale_vol = false;
    bool allow_reflections = true;

    cmdopts.addop(allow_reflections, "allow-reflections", "", "include mirror images, default true")
        .addop(inputfile1, "input1", "in1", "input file 1", true)
           .addop(inputfile2, "input2", "in2", "input file 2", true)
           .addop(indexfile, "index", "id", "index file")
           .addop(informat1, "input-format1", "if1", "input format")
           .addop(informat2, "input-format2", "if2", "output format")
           .addop(scale_vol, "scale", "s", "scale to same volume", false, 0)
           .addop(tol, "tol", "tol", "tolerance for checking cell and positions, default 1e-4 A", true)
           .add_request("input1").add_request("input2")
           .phrase_options(argc, argv);

    if (GPIO::file_exist(indexfile))
        iobj.readmap(indexfile);
    else
        iobj.setmap();
    if (!GPIO::file_exist(inputfile1) || !GPIO::file_exist(inputfile2)) {
        cerr << "Error: input files not found" << endl;
        return 1;
    }
    if (!iobj.read_one(atom1, inputfile1, informat1) || !iobj.read_one(atom2, inputfile2, informat2))
        GPERROR("Failed to read comparison structures");
    Symmetry3D::StructureComparisonOptions options;
    options.symmetry_tolerance = tol;
    options.position_tolerance = tol;
    options.volume_tolerance = std::numeric_limits<double>::max();
    options.descriptor_tolerance = 14 * tol;
    options.anchor_descriptor_tolerance = tol;
    options.scale_volume = scale_vol;
    options.allow_reflections = allow_reflections;
    Symmetry3D::StructureComparator comparator(options);
    const auto first = comparator.prepare(std::move(atom1));
    const auto second = comparator.prepare(std::move(atom2));
    if (comparator.equivalent(first, second)) {
        std::cout << "Equivalent" << std::endl;
    } else {
        std::cout << "Not equivalent" << std::endl;
    }

    return 0;
}
