// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include <iomanip>
using namespace std;
int calc_ferr(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    vector<Atoms> atomsdft;
    vector<Atoms> atomsmlp;
    double temp = 300;
    string informat = "";
    string infile1 = "dft.cfg";
    string infile2 = "mlp.cfg";
    string indexfile = "index.map";
    GPIO iobj;
    constexpr double boltz = 8.61733326e-5; // eV/K
    cmdopts.addop(indexfile, "index", "id")
           .addop(infile1, "input", "in", "Reference dataset path (default: dft.cfg)")
           .addop(infile2, "prediction", "", "Prediction dataset path (default: mlp.cfg)")
           .addop(informat, "input-format", "if", "Input format for both datasets")
           .addop(temp, "temperature", "t")
           .phrase_options(argc, argv);
    if (GPIO::file_exist(indexfile)) {
        iobj.readmap(indexfile);
    } else {
        iobj.setmap();
    }

    if (!std::isfinite(temp) || temp <= 0)
        GPERROR("calc_ferr: temperature must be finite and positive");
    if (!iobj.read(atomsdft, infile1, informat) || atomsdft.empty())
        GPERROR("calc_ferr: failed to read nonempty reference dataset");
    if (!iobj.read(atomsmlp, infile2, informat) || atomsmlp.size() != atomsdft.size())
        GPERROR("calc_ferr: prediction dataset must match reference frame count");
    const int expected_atoms = atomsdft[0].size();
    std::vector<double> delta_energy;
    delta_energy.reserve(atomsdft.size());
    for (size_t i = 0; i < atomsdft.size(); ++i) {
        if (expected_atoms == 0 || atomsdft[i].size() != expected_atoms ||
            atomsmlp[i].size() != expected_atoms)
            GPERROR("calc_ferr: all paired structures must have the same nonzero atom count");
        if (!atomsdft[i].has_energy() || !atomsmlp[i].has_energy())
            GPERROR("calc_ferr: energy is required on both sides at frame " + std::to_string(i + 1));
        double delta = atomsdft[i].energy_ - atomsmlp[i].energy_;
        if (!std::isfinite(delta))
            GPERROR("calc_ferr: energy difference must be finite at frame " + std::to_string(i + 1));
        delta_energy.push_back(delta);
    }
    // Shift by the minimum energy difference so every exponential is <= 1.
    const double min_delta = *std::min_element(delta_energy.begin(), delta_energy.end());
    const double kT = temp * boltz;
    if (kT == 0)
        GPERROR("calc_ferr: temperature is too small");
    double accu = 0;
    for (double delta : delta_energy)
        accu += std::exp(-(delta - min_delta) / kT);
    double fdiff = min_delta - kT * std::log(accu / delta_energy.size());
    if (!std::isfinite(fdiff))
        GPERROR("calc_ferr: free-energy difference is not finite");
    std::cout << std::setprecision(17) << fdiff / expected_atoms << std::endl;
    return 0;
}
