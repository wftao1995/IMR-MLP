// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/potential_mtp.h"

int mtp2spline(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    std::string infile = "potential.mtp";
    std::string outfile = "splinedpot.mtp";
    double deltar = 0.01;
    double rmin = -1;
    cmdopts.addop(deltar, "delta-r", "dr", "spline interval")
           .addop(rmin, "r-min", "min", "inner radius, below which will set to constant (negative: each pair's basis minimum)")
           .addop(infile, "input", "", "input potential file", true)
           .addop(outfile, "output", "", "output splined potential file", true);
    cmdopts.phrase_options(argc, argv);
    Potentials::PotMTP pot(infile);
    pot.SaveSpline(outfile, rmin, deltar);
    return 0;
}
