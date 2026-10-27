// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/potential_wrapper.h"
#include "io/gpio.h"
#include "train/train_linear_potential.h"
#include "train/collective_error.h"
#include <memory>

using namespace std;

int train_linear(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;

    GPIO iobj;
    std::vector<Atoms> atoms, atoms_valid;
    string indexfile = "index.map";
    string infile = "train.cfg";
    string infilev = "";
    string informat = "mlpcfg";
    string poutfile = "_Trained_Potential_";
    string settingfile = "";
    GPSettings settings;
    bool debug = false;

    cmdopts.addop(debug, "debug", "", "Enable debug output stream", false, 0, 10)
        .addop(indexfile, "index", "id", "Index file mapping atom types to indices")
        .addop(infile, "trainset", "in", "Input training dataset file")
        .addop(infilev, "validset", "inv", "Validation dataset file")
        .addop(informat, "trainset-format", "if", "Training data format: mlpcfg | extxyz | vaspabn")
        .addop(poutfile, "output", "o", "Output potential parameter file")
        .addop(settingfile, "settings", "", "Main configuration settings file", true)
        .attach_settings("fit:", "Linear training settings",
            [](std::ostream &os) { TrainerLinearPotential().ListSettings(os, "fit:"); })
        .set_helponexit(false)
        .phrase_options(argc, argv);

    if (cmdopts.displayhelp) {
        if (mpi.rank == 0)
            cmdopts.showhelp(std::cout);
        return 0;
    }
    if (GPIO::file_exist(settingfile)) {
        Message("Loading settings");
        settings.Load(settingfile);
        settings.Print();
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    std::unique_ptr<ScopedStreamRedirect> debug_guard;
    if (debug) {
        auto debugfs = std::make_unique<std::ofstream>("DEBUG" + mpi.fnm_ending);
        if (*debugfs)
            debug_guard = std::make_unique<ScopedStreamRedirect>(global_debugstream, debugfs.release());
        else
            Warning("Failed to open debug output file: DEBUG" + mpi.fnm_ending);
    }
    if (GPIO::file_exist(indexfile)) {
        iobj.readmap(indexfile);
    } else {
        Warning("ERROR: indexfile not found\n");
    }
    Message("Reading trajectory file " + infile);
    if (!iobj.read(atoms, infile, informat))
        GPERROR("Failed to read complete training set: " + infile);
    if (!infilev.empty()) {
        if (!iobj.read(atoms_valid, infilev, informat))
            GPERROR("Failed to read complete validation set: " + infilev);
    }
    Message("Finish loading trajectory");

    for (int i = 0; i < atoms.size(); ++i) {
        if (atoms[i].local_attribs_d_.count("atomic_fitness")) {
            atoms[i].atomic_fitness.resize(atoms[i].size());
            memcpy(atoms[i].atomic_fitness.data(),
                   atoms[i].local_attribs_d_["atomic_fitness"].data(),
                   atoms[i].size() * sizeof(double));
        }
    }
    for (int i = 0; i < atoms.size(); ++i) {
        if (atoms[i].global_attribs_d_.count("fitness"))
            atoms[i].fitness = atoms[i].global_attribs_d_["fitness"];
        else
            atoms[i].fitness = 1.0;
    }

#ifdef IMRMLP_MPI
    Message("Balancing");
    {
        sort(atoms.begin(), atoms.end(),
             [](const Atoms &a, const Atoms &b) { return a.size() < b.size(); });
        auto itrm = remove_if(atoms.begin(), atoms.end(), [](Atoms &at) {
            return at.id % mpi.size == mpi.rank ? false : true;
        });
        atoms.erase(itrm, atoms.end());
        if (atoms_valid.size() > 0) {
            sort(atoms_valid.begin(), atoms_valid.end(),
                 [](const Atoms &a, const Atoms &b) { return a.size() < b.size(); });
            auto itrm2 = remove_if(atoms_valid.begin(), atoms_valid.end(), [](Atoms &at) {
                return at.id % mpi.size == mpi.rank ? false : true;
            });
            atoms_valid.erase(itrm2, atoms_valid.end());
        }
    }
#endif

    Message("Construct trainer");
    TrainerLinearPotential trainer(settings.ExtractSubSettings("fit"));
    if (atoms_valid.size() > 0)
        trainer.monit_valid_err = true;

    Message("Construct potential loader");
    Potentials::potential_wrapper wrapper(settings.ExtractSubSettings("pot"));
    Potentials::AnyPotential *p_pot = wrapper.pot;

    if (!p_pot->supports_linear_fit())
        GPERROR("Potential type does not support linear fitting");
    Message("Preparing neighborhoods");
    std::vector<Neighborhoods> nbhs, nbhs_valid;
    nbhs.resize(atoms.size());
    for (int i = 0; i < atoms.size(); ++i) {
        atoms[i].wrap();
        nbhs[i] = Neighborhoods(atoms[i], p_pot->cutoff);
    }
    Message("Init trainer");
    trainer.init(atoms, nbhs, p_pot);
    if (atoms_valid.size() > 0) {
        nbhs_valid.resize(atoms_valid.size());
        for (int i = 0; i < atoms_valid.size(); ++i) {
            atoms_valid[i].wrap();
            nbhs_valid[i] = Neighborhoods(atoms_valid[i], p_pot->cutoff);
        }
        trainer.init_valid(atoms_valid, nbhs_valid);
    }

    Message("Training run");
    trainer.Train();

    Training::collective_try([&] {
        if (mpi.rank == 0)
            p_pot->Save(poutfile);
    });

    Message("Finished");
    return 0;
}
