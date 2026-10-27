// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "train/training_data.h"
#include "potentials/potential_wrapper.h"
#include "potentials/potential_mtp.h"
#include "potentials/potential_zbl.h"
#include "tools/tools.h"
#include "io/gpio.h"
#include "train/train_nonlinear_potential.h"
#include "train/collective_error.h"
#include "train/fixed_core.h"
#include "potentials/potential_mtp.h"
#include "potentials/potential_zbl.h"

#include <iomanip>
#include <chrono>

using namespace std;

int trainnlpot(int argc, char *argv[]) {
    const auto application_start = std::chrono::steady_clock::now();
    const auto seconds_since = [](const auto &start) {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    };
    auto &cmdopts = CMDOption::global_cmdopts;
// #ifdef IMRMLP_MPI
//     MPI_Init(&argc, &argv);
//     mpi.InitComm(MPI_COMM_WORLD);
// #endif
    // auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    std::vector<Atoms> atoms, atoms_valid, atoms_test;
    bool showhelp = false;
    string indexfile = "index.map"; // can use with formation energy, first is dft, second is mlp
    string infile = "train.cfg";
    string infilev = "";
    string infiletest = "";
    string informat = "";
    string invformat = "";
    string intestformat = "";
    string poutfile = "_Trained_Potential_";
    string fixed_core_file, fixed_core_type = "zbl";
    std::vector<Potentials::HessianMatrix> fc2;
    string settingfile = "";
    GPSettings settings;
    bool do_sgd = false;
    bool do_varpro = false;
    bool do_bfgs = false; // lbfgs by alglib
    bool do_cg = false;
    bool do_calcgrad = false;
    bool linear_refit_only = false;
    bool debug = false;

    bool balance_atomic_fitness = false;

    // for test
    bool do_curv_test = false;
    int curv_test_which = 0;
    double curv_test_nudge = 0.001;
    bool check_grad = false;
    bool check_grad_all = false;

    cmdopts.addop(debug, "debug", "", "Enable debug output stream", false, 0, 10)
        .addop(indexfile, "index", "id", "Index file mapping atom types to indices")
        .addop(balance_atomic_fitness, "balance-atomic-fitness", "bw", "Load balancing, when not all forces are fitted", false, 0)
        .addop(do_sgd, "do-sgd", "sgd", "Do Stochastic Gradient Descent optimization", false, 0)
        .addop(do_bfgs, "do-lbfgs", "lbf", "Do Limited-memory BFGS optimization", false, 0)
        .addop(do_varpro, "do-varpro", "", "Do profiled radial L-BFGS with a linear refit at every trial (ordinary CPU MTP)", false, 0)
        .addop(do_cg, "do-cg", "cg", "Do Conjugate Gradient optimization", false, 0)
        .addop(do_calcgrad, "calc-grad", "g", "Calculate gradient without optimization", false, 0)
        .addop(linear_refit_only, "linear-refit-only", "", "Refit linear coefficients of an initialized model without nonlinear training", false, 0)
        .addop(check_grad, "check-grad", "", "Perform gradient verification check", false, 0)
        .addop(check_grad_all, "check-grad-all", "", "Comprehensive gradient verification for all parameters", false, 0)
        .addop(infile, "trainset", "train", "Input training dataset file")
        .addop(infilev, "validset", "inv", "Validation dataset file")
        .addop(infiletest, "testset", "", "Held-out dataset, evaluated only after final fitting")
        .addop(informat, "trainset-format", "if", "Training data format: mlpcfg | extxyz | vaspabn")
        .addop(invformat, "validset-format", "vf", "Validation dataset format")
        .addop(intestformat, "testset-format", "", "Held-out dataset format")
        .addop(poutfile, "output", "o", "Output potential parameter file")
        .addop(fixed_core_file, "fixed-core", "", "Existing fixed core; fit an ordinary MTP to residual labels and output a Hybrid manifest")
        .addop(fixed_core_type, "fixed-core-type", "", "Fixed core model type: zbl | mtp")
        .addop(do_curv_test, "test-curv", "", "Test curvature calculation for debugging", false, 0)
        .addop(curv_test_which, "test-curv-which", "", "Specify which curvature component to test")
        .addop(curv_test_nudge, "test-curv-nudge", "", "Perturbation magnitude for curvature testing")
        .addop(settingfile, "settings", "", "Main configuration settings file", true)
        .attach_settings("fit:", "Nonlinear training settings",
            [](std::ostream &os) { TrainNonlinear().ListSettings(os, "fit:"); })
        .set_helponexit(false)
        .phrase_options(argc, argv);

    if (cmdopts.displayhelp) {
        if (mpi.rank == 0) cmdopts.showhelp(std::cout);
        return 0; // all ranks return to gpmain's MPI_Finalize
    }
    if (GPIO::file_exist(settingfile)) {
        Message("Loading settings");
        settings.Load(settingfile);
        settings.Print();
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    // Reject unsupported requests before reading data, initializing a model,
    // constructing neighborhoods, or creating any training output.
    Message("Construct trainer");
    TrainNonlinear trainer(settings.ExtractSubSettings("fit"));
    if (do_cg) {
        trainer.ValidateCGSettings();
        if (do_sgd || do_varpro || linear_refit_only
            || do_calcgrad || check_grad || check_grad_all || do_curv_test)
            GPERROR("--do-cg supports a standalone run or a preceding --do-lbfgs run");
    }
    const bool linear_refit_enabled = trainer.CurrentSettings()["linear_refit"] != "off";
    if (linear_refit_only && !linear_refit_enabled)
        GPERROR("--linear-refit-only requires fit:linear_refit");
    if (!(do_sgd || do_bfgs || do_cg || do_varpro
          || do_calcgrad || check_grad || check_grad_all || do_curv_test
          || linear_refit_enabled))
        GPERROR("trainnlpot: select a training or diagnostic action, or enable fit:linear_refit");
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
    const auto load_start = std::chrono::steady_clock::now();
    // iobj.set(atoms);
    atoms = Training::load_training_data(iobj, infile, informat);
    if (!infilev.empty())
        atoms_valid = Training::load_training_data(iobj, infilev, invformat);
    if (!infiletest.empty())
        atoms_test = Training::load_training_data(iobj, infiletest, intestformat);
    Training::restore_training_fitness(atoms);
    Training::restore_training_fitness(atoms_valid);
    Message("Finish loading training set, size: " + std::to_string(atoms.size()) + ", valid set size: " + std::to_string(atoms_valid.size()));
    Message("trainnlpot timing load_s=" + std::to_string(seconds_since(load_start)));
    Training::distribute_training_data(atoms, balance_atomic_fitness);
    Training::distribute_training_data(atoms_valid, false);
    Training::distribute_training_data(atoms_test, false);
    if (atoms_valid.size() > 0) {
        trainer.monit_valid_err = true;
    }
    Message("Construct potential loader");
    Potentials::potential_wrapper wrapper(settings.ExtractSubSettings("pot"));
    Potentials::AnyPotential *p_pot = wrapper.pot;
    std::unique_ptr<Potentials::AnyPotential> fixed_core;
    if(!fixed_core_file.empty()) Training::collective_try([&] {
        if(fixed_core_type=="mtp") fixed_core=std::make_unique<Potentials::PotMTP>(fixed_core_file);
        else if(fixed_core_type=="zbl") {
            fixed_core=std::make_unique<Potentials::PotentialZBL>();fixed_core->Load(fixed_core_file);
        } else GPERROR("--fixed-core-type must be zbl or mtp");
        Training::validate_fixed_core(*p_pot,*fixed_core);
        atoms=Training::fixed_core_residuals(*fixed_core,atoms,trainer.hvp_fd_step(),trainer.hessian_fd_step());
        atoms_valid=Training::fixed_core_residuals(*fixed_core,atoms_valid,trainer.hvp_fd_step(),trainer.hessian_fd_step());
        atoms_test=Training::fixed_core_residuals(*fixed_core,atoms_test,trainer.hvp_fd_step(),trainer.hessian_fd_step());
    });
    if(do_varpro && (do_sgd || do_bfgs || do_cg || do_calcgrad
        || check_grad || check_grad_all || do_curv_test || linear_refit_only))
        GPERROR("--do-varpro cannot be combined with other optimizer or diagnostic options");
    if(linear_refit_only) {
        if(do_sgd || do_bfgs || do_cg || do_calcgrad || check_grad || check_grad_all || do_curv_test)
            GPERROR("--linear-refit-only cannot be combined with optimizer or gradient-check options");
        trainer.ValidateLinearRefitOnly(*p_pot);
    }
    Message("Prepend to training");
    // potential.train_init();
    std::vector<Neighborhoods> nbhs, nbhs_valid;

    Message("Init nbhs");
    const auto neighborhood_start = std::chrono::steady_clock::now();
    nbhs = Training::training_neighborhoods(atoms, p_pot->cutoff);
    Message("trainnlpot timing neighborhoods_s=" +
            std::to_string(seconds_since(neighborhood_start)));
    Message("Init trainer");
    const auto trainer_init_start = std::chrono::steady_clock::now();
    trainer.init(atoms, nbhs, p_pot);
    nbhs_valid = Training::training_neighborhoods(atoms_valid, p_pot->cutoff);
    trainer.init_valid(atoms_valid, nbhs_valid);
    Message("trainnlpot timing trainer_init_s=" +
            std::to_string(seconds_since(trainer_init_start)));

    // for test
    if (do_curv_test) {
        Message("Testing curvature");
        trainer.test_curv(curv_test_which, curv_test_nudge);
        // exit(0);
        goto end;
    }
    if (check_grad) {
        Message("Checking grads");
        trainer.check_grad(curv_test_which, curv_test_nudge);
        // exit(0);
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
    if (do_calcgrad) {
        Message("Computing gradient info");
        const auto gradient_start = std::chrono::steady_clock::now();
        double f;
        alglib::real_1d_array g;
        g.setlength(trainer.get_relaxed_coeffs_count());
        trainer.AccumulateGard(f, g);
        Training::collective_try([&] {
            if (mpi.rank != 0) return;
            ofstream loss_ofs("output_loss_info");
            if (!loss_ofs) GPERROR("Cannot open calc-grad output: output_loss_info");
            loss_ofs << std::setprecision(17) << f << '\n';
            if (!loss_ofs) GPERROR("Failed writing calc-grad output: output_loss_info");
            loss_ofs.close();
            if (!loss_ofs) GPERROR("Failed closing calc-grad output: output_loss_info");
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
        Message("trainnlpot timing gradient_and_output_s=" +
                std::to_string(seconds_since(gradient_start)));
        goto end;
    }

    Message("Tranining run");
    {
        auto options = trainer.workflow_options;
        options.validation_metric = trainer.CurrentSettings()["validation_metric"];
        auto result = Training::run_nonlinear_workflow(trainer, atoms_test, options,
            [&](TrainNonlinear &active) {
                if (do_sgd) active.TrainSGD();
                if (do_bfgs) active.TrainBFGS();
                if (do_cg) active.TrainCG();
                if (do_varpro) active.TrainVarPro();
                else if (!do_cg) active.TrainLinearRefit();
            });
        auto &active = result.final_trainer ? *result.final_trainer : trainer;
        if (fixed_core) Training::collective_try([&] {
            if(mpi.rank==0) Training::save_fixed_core_hybrid(*p_pot,*fixed_core,poutfile,
                atoms.empty()?nullptr:&atoms.front());
        });
        else if (do_varpro || linear_refit_enabled) active.SaveConditionalResult(poutfile);
        else if (do_cg) active.SaveCGResult(poutfile);
        else Training::collective_try([&] { if (mpi.rank == 0) p_pot->Save(poutfile); });
        if (mpi.rank == 0) {
            if (result.selected_multiplier)
                Message("Selected L2 multiplier: " + std::to_string(*result.selected_multiplier));
            const auto show = [](const char *name, const Training::DatasetMetrics &metrics) {
                if (metrics.energy) Message(std::string(name) + " energy_RMSE_eV_atom=" + std::to_string(*metrics.energy));
                if (metrics.force) Message(std::string(name) + " force_RMSE_eV_A=" + std::to_string(*metrics.force));
                if (metrics.stress) Message(std::string(name) + " stress_RMSE_GPa=" + std::to_string(*metrics.stress));
            };
            if (!infiletest.empty()) show("Final test", result.test);
        }
    }

end:
// #ifdef IMRMLP_MPI
//     MPI_Finalize();
// #endif
    Message("trainnlpot timing total_s=" +
            std::to_string(seconds_since(application_start)));
    Message("Finished");
    return 0;
}
