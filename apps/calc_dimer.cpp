// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mindriver/dimer_new.h"
#include "potentials/potential_wrapper.h"
#include "tools/constants.h"
#include "deplibs/LBFGSpp/include/LBFGS.h"
#include "mindriver/fire.h"
#include "mindriver/sgd.h"
#include "mindriver/cg.h"

using namespace std;

class DimerAPPParam : public ConnectSettings {
public:
    string inputstrfile;
    string informat = "";
    string indexfile = "index.map";
    string output_structure = "dimer_final.vasp";

    void InitSettings() {
        CMDOption::CommandLineOptions &cmdopts = CMDOption::global_cmdopts;
        ConnectWithConsole(cmdopts, inputstrfile, "structure", "in",
                           "Input structure file (initial guess for transition state)", true);
        ConnectWithConsole(cmdopts, informat, "input-format", "if",
                           "Input structure format");
        ConnectWithConsole(cmdopts, indexfile, "index", "id",
                           "Index file mapping atom types to indices");
        ConnectWithConsole(cmdopts, output_structure, "output", "out",
                           "Output structure file for converged saddle point");
    }

    DimerAPPParam() {
        InitSettings();
    }
};

int calc_dimer(int argc, char *argv[]) {
    CMDOption::CommandLineOptions &cmdopts = CMDOption::global_cmdopts;
    GPSettings settings;
    string settingfile;
    GPIO iobj;
    iobj.packatoms = false;
    DimerAPPParam params;
    // Use the global CLI/settings binding rather than registering a duplicate.
    omp_num_threads = 1;

    cmdopts.addop(settingfile, "settings", "", "Input setting file.", true)
           .add_request("settings")
           .attach_settings("dimer:", "DIMER driver settings",
               [](std::ostream &os) { DimerDriver().ListSettings(os, "dimer:"); });
    cmdopts.phrase_options(argc, argv);

    if (GPIO::file_exist(settingfile)) {
        Message("Loading settings from " + settingfile);
        settings.Load(settingfile);
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    params.ApplySettings(settings);
    params.PrintSettings();

    // Set number of threads
    omp_set_num_threads(omp_num_threads);

    // Read index file
    if (GPIO::file_exist(params.indexfile)) {
        iobj.readmap(params.indexfile);
    }

    // Read input structure
    Atoms atom;
    {
        vector<Atoms> atoms;
        if (!iobj.read(atoms, params.inputstrfile, params.informat)) {
            std::cerr << "Failed to read DIMER input structure: " << params.inputstrfile << '\n';
            return 1;
        }
        if (atoms.empty()) {
            throw runtime_error("No atoms read from input file: " + params.inputstrfile);
        }
        atom = atoms[0];
        atom.wrap();
    }

    // Create DIMER driver
    DimerDriver dimerobj(settings.ExtractSubSettings("dimer"));

    // Create potential
    Potentials::potential_wrapper wrapper(settings.ExtractSubSettings("pot"));
    dimerobj.pot = wrapper.pot;
    dimerobj.iobj = &iobj;

    // Initialize
    dimerobj.init(atom);

    // Run
    Message("Starting DIMER saddle point search...");
    int status = dimerobj.run();

    if (status == 0) {
        Message("DIMER converged successfully!");
    } else {
        Message("DIMER reached maximum iterations without convergence.");
    }

    // Write summary
    if (!dimerobj.summary.empty()) dimerobj.write_summary(dimerobj.summary);

    // Write final structure
    atom = dimerobj.atom_center;
    if (!iobj.write_one(atom, params.output_structure))
        GPERROR("Failed to write DIMER final structure: " + params.output_structure);

    return status;
}
