#include "train/training_data.h"
// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/potential_mtp.h"
#include "tools/tools.h"
#include "utils/gputils.h"
#include "io/gpio.h"
#include "train/train_mtp.h"
#include "train/collective_error.h"
#include "help_utils.h"
#include <iomanip>

// TODO: train valid test, using train and valid to fit potential, and test for testing

using namespace std;

int trainmtp(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;

    GPIO iobj;
    std::vector<Atoms> atoms, atoms_valid;
    string indexfile = "index.map"; // can use with formation energy, first is dft, second is mlp
    string pfile = "potential.mtp";
    string infile = "train.cfg";
    string infilev = "";
    string informat = "";
    string invformat = "";
    string poutfile = "_Trained_Potential_.mtp";
    string settingfile = "";
    GPSettings settings;
    bool do_linear_only = false;
    bool do_orth_only = false;
    bool do_resacle = false;
    bool do_spec_coeffs = false; // Generic named-group optimizer bridge.
    bool do_sgd = false;
    bool do_bfgs = false; // lbfgs by alglib
    bool do_cg = false;
    bool do_lbfgs = false; // lbfgspp
    bool do_df = false;
    bool do_matrixATA = false; // output design matrix A^T * A
    bool do_matrixA = false; // export full E/F/S design-matrix rows per MPI rank; output can be large
    bool do_calcgrad = false;
    bool debug = false;
    bool do_random = false;

    bool balance_atomic_fitness = false;

    // for test
    bool do_curv_test = false;
    int curv_test_which = 0;
    double curv_test_nudge = 0.001;
    bool check_grad = false;
    bool check_grad_all = false;

    cmdopts.addop(debug, "debug", "", "Enable debug output stream", false, 1, 10)
           .addop(indexfile, "index", "id", "Index file mapping atom types to indices")
           .addop(pfile, "potfile", "pot", "Potential file")
           .addop(do_spec_coeffs, "species-only", "", "Fit only species coefficients through the generic L-BFGS trainer", false, 0)
           .addop(do_linear_only, "linear-fit", "lf", "Perform only linear parameter fitting", false, 0)
           .addop(do_resacle, "do-rescale", "rs", "Use rescaling instead of a single linear fit (requires --linear-fit)", false, 0)
           .addop(balance_atomic_fitness, "balance-at", "bw", "Load balancing, when not all forces are fitted", false, 0)
           .addop(do_sgd, "do-sgd", "sgd", "Do Stochastic Gradient Descent optimization", false, 0)
           .addop(do_bfgs, "do-lbfgs", "lbf", "Do Limited-memory BFGS optimization", false, 0)
           .addop(do_df, "do-df", "df", "Do Differential Evolution iteration", false, 0)
           .addop(do_matrixATA, "do-matrix-ata", "ata", "Compute and output design matrix A^T*A for linear regression", false, 0)
           .addop(do_matrixA, "do-matrix-a", "a", "Compute and output full design matrix A (warning: may be memory intensive)", false, 0)
           .addop(do_cg, "do-cg", "cg", "Do Conjugate Gradient optimization", false, 0)
           .addop(do_random, "do-random", "random", "Do random radial coeffs search", false, 0)
           .addop(infile, "trainset", "train", "Input training dataset file")
           .addop(infilev, "validset", "valid", "Input validation dataset file")
           .addop(informat, "trainset-format", "if", "Training dataset file format")
           .addop(invformat, "validset-format", "vf", "Validation dataset file format")
           .addop(do_curv_test, "test-curv", "", "Test curvature calculation for debugging", false, 0)
           .addop(curv_test_which, "test-curv-which", "", "Specify which curvature component to test")
           .addop(curv_test_nudge, "test-curv-nudge", "", "Perturbation magnitude for curvature testing")
           .addop(do_calcgrad, "calc-grad", "g", "Calculate gradient without optimization", false, 0)
           .addop(check_grad, "check-grad", "", "Perform gradient verification check", false, 0)
           .addop(check_grad_all, "check-grad-all", "", "Comprehensive gradient verification for all parameters", false, 0)
           .addop(settingfile, "settings", "", "input setting file", true)
           .attach_settings("fit:", "MTP training settings",
               [](std::ostream &os) { Trainer_MTP().ListSettings(os, "fit:"); })
           .set_helponexit(false)
           .phrase_options(argc, argv);
    if (cmdopts.displayhelp) {
        if (mpi.rank == 0) cmdopts.showhelp(cout);
        return 0; // all ranks return to imrmlp's MPI_Finalize
    }
    if (GPIO::file_exist(settingfile)) {
        Message("Loading settings");
        settings.Load(settingfile);
        settings.Print();
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    const bool has_training_action = do_spec_coeffs || do_linear_only || do_sgd || do_bfgs || do_cg
        || do_lbfgs || do_df || do_random;
    const bool has_diagnostic_action = do_matrixATA || do_matrixA || do_calcgrad
        || do_curv_test || check_grad || check_grad_all;
    if (do_resacle && !do_linear_only)
        GPERROR("trainmtp: --do-rescale requires --linear-fit");
    if (!has_training_action && !has_diagnostic_action)
        GPERROR("trainmtp: select a training or diagnostic action (for example --linear-fit, --do-lbfgs, or --do-matrix-ata)");

    if (do_spec_coeffs && (do_linear_only || do_resacle || do_sgd || do_cg
        || do_lbfgs || do_df || do_random || has_diagnostic_action))
        GPERROR("trainmtp --species-only supports only the ordinary --do-lbfgs optimizer");
    if (do_spec_coeffs) {
        Trainer_MTP preflight(settings.ExtractSubSettings("fit"));
        preflight.ValidateSpeciesOnlySettings();
        const auto explicit_fit = settings.ExtractSubSettings("fit");
        for (const auto &[key, value] : explicit_fit.settings) {
            if (key.starts_with("lbfgspp:") || key.starts_with("lr_")
                || key == "learning_rate" || key == "momentum" || key == "batch_size"
                || key == "epoch_limit" || key == "clip_gradient" || key == "popsize"
                || key == "pre_init_iter" || key == "lfit_eng_only"
                || key == "lfit_fastalgo" || key == "rescale_mode"
                || key == "linear_precondition" || key.starts_with("shlbfgs_"))
                GPERROR("trainmtp --species-only does not support fit:" + key);
        }
    }

    // Validate the final settings/CLI choice before input or output side effects.
    if (debug) {
        std::ofstream *debugfs = new std::ofstream("DEBUG" + mpi.fnm_ending);
        SetStreamForDebug(debugfs);
    }
    if (GPIO::file_exist(indexfile)) {
        iobj.readmap(indexfile); // should be with atom radii
    } else {
        Warning("ERROR: indexfile not found\n");
    }
    // read atoms
    Message("Loading training set file " + infile);
    atoms = Training::load_training_data(iobj, infile, informat);
    if (!infilev.empty())
        atoms_valid = Training::load_training_data(iobj, infilev, invformat);
    Training::restore_training_fitness(atoms);
    Message("Finish loading training set, size: " + std::to_string(atoms.size()) + ", valid set size: " + std::to_string(atoms_valid.size()));
    Training::distribute_training_data(atoms, balance_atomic_fitness);
    Training::distribute_training_data(atoms_valid, false);
    // Init trainer
    Message("Construct trainer");
    Trainer_MTP trainer(settings.ExtractSubSettings("fit"));
    if (atoms_valid.size() > 0) {
        trainer.monit_valid_err = true;
    }
    MPI_Bcast(&trainer.monit_valid_err, 1, MPI_C_BOOL, 0, mpi.comm);
    Message("Construct potential loader");
    Potentials::PotMTP potential(pfile);
    const auto chemistry=iobj.getmap();std::vector<std::string> symbols;
    for(int i=0;i<potential.species_count;++i){auto it=chemistry.find(i);if(it==chemistry.end()||!MTPMetadata::element(it->second)){symbols.clear();break;}symbols.push_back(it->second);}
    if(!symbols.empty())potential.set_element_symbols(symbols);

    Message("Prepend to training");
    std::vector<Neighborhoods> nbhs, nbhs_valid;

    Message("Init nbhs");
    nbhs = Training::training_neighborhoods(atoms, potential.get_cutoff());
    Message("Init trainer");
    trainer.init(atoms, nbhs, &potential);
    nbhs_valid = Training::training_neighborhoods(atoms_valid, potential.get_cutoff());
    trainer.init_valid(atoms_valid, nbhs_valid);

    // for test
    if (do_curv_test) {
        Message("Testing curvature");
        trainer.test_curv(curv_test_which, curv_test_nudge);
        goto end;
    }
    if (check_grad) {
        Message("Checking grads");
        trainer.check_grad(curv_test_which, curv_test_nudge);
        goto end;
    }
    if (check_grad_all) {
        Message("Checking grads for all");
        int ncoeffs = trainer.get_relaxed_coeffs_count();
        for (int i = 0; i < ncoeffs; ++i) {
            trainer.check_grad(i, curv_test_nudge);
        }
        goto end;
    }
    if (do_matrixA) {
        Message("Computing design matrix only, this file may be HUGE");
        trainer.get_A();
        MPI_Barrier(mpi.comm);
        goto end;
    }
    if (do_calcgrad) {
        Message("Computing gradient info");
        double f;
        alglib::real_1d_array g;
        g.setlength(trainer.get_relaxed_coeffs_count());
        trainer.AccumulateGard(f, g);
        Training::collective_try([&] {
            if (mpi.rank != 0) return;
            ofstream ofs("output_grad_info");
            if (!ofs) GPERROR("Cannot open calc-grad output: output_grad_info");
            ofs << std::setprecision(17);
            for (int i = 0; i < g.length(); ++i) {
                ofs << g[i] << '\n';
            }
            if (!ofs) GPERROR("Failed writing calc-grad output: output_grad_info");
            ofs.close();
            if (!ofs) GPERROR("Failed closing calc-grad output: output_grad_info");
        });
        goto end;
    }

    if (do_spec_coeffs) {
        trainer.TrainSpeciesOnly();
    } else if (do_linear_only) {
        Message("Fit linear");
        if (do_resacle)
            trainer.Rescale();
        else
            trainer.TrainLinear();
    } else if (has_training_action) {
        Message("Training run");
        if (do_sgd)
            trainer.TrainSGD();
        if (do_bfgs)
            trainer.TrainBFGS();
        if (do_cg)
            trainer.TrainCG();
        if (do_lbfgs)
            trainer.TrainLBFGS();
        if (do_df)
            trainer.TrainDF();
        if (do_random) {
            trainer.TrainRandom();
            goto end;
        }
    }
    trainer.ladder_fin();
    if (has_training_action && mpi.rank == 0)
        potential.Save(poutfile);
    if (do_matrixATA) {
        Message("Computing energy, force and stress part of linearmatrix");
        trainer.get_ATA();
        MPI_Barrier(mpi.comm);
    }
end:
    Message("Finished");
    return 0;
}
