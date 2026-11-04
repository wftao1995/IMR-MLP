// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "training_output.h"
#include "persistence_mode.h"
#include "train/training_data.h"
#include "potentials/potential_mtp.h"
#include "potentials/model_elements.h"
#include "tools/tools.h"
#include "utils/gputils.h"
#include "io/gpio.h"
#include "species_map_utils.h"
#include "train/train_mtp.h"
#include "train/collective_error.h"
#include "train/conditional_linear_persistence.h"
#include <chrono>
#include <iomanip>

// TODO: train valid test, using train and valid to fit potential, and test for testing

using namespace std;

namespace
{
void print_gpu_linear_report(const GpuLinearReport &r)
{
    using picojson::value;
    using picojson::object;
    picojson::array lambda;
    for (double v : r.effective_l2)
        lambda.emplace_back(v);
    const auto &s = r.gpu_stats;
    const auto &c = r.curvature_stats;
    const object report{
        {"observation_backend", value(r.backend)},
        {"solve_backend", value("cpu/" + r.algorithm)},
        {"acceptance_backend", value("cpu/direct")},
        {"verified", value(r.verified)},
        {"data_before", value(r.data_before)},
        {"data_after", value(r.data_after)},
        {"predicted_data_before", value(r.predicted_data_before)},
        {"predicted_data_after", value(r.predicted_data_after)},
        {"quadratic_tolerance_before", value(r.quadratic_tolerance_before)},
        {"quadratic_tolerance_after", value(r.quadratic_tolerance_after)},
        {"objective_before", value(r.objective_before)},
        {"objective_after", value(r.objective_after)},
        {"effective_l2", value(lambda)},
        {"host_design_bytes", value(double(r.host_design_bytes))},
        {"curvature_host_cap_bytes", value(double(r.curvature_host_cap_bytes))},
        {"curvature_host_peak_bytes", value(double(r.curvature_host_peak_bytes))},
        {"curvature_observations",
         value(object{{"probes", value(double(c.probes))},
                      {"blocks", value(double(c.blocks))},
                      {"download_bytes", value(double(c.download_bytes))},
                      {"tracked_device_bytes", value(double(c.tracked_device_bytes))},
                      {"host_workspace_bytes", value(double(c.host_workspace_bytes))},
                      {"output_workspace_bytes", value(double(c.output_workspace_bytes))},
                      {"gpu_initialize_seconds", value(c.gpu_initialize_seconds)},
                      {"geometry_seconds", value(c.geometry_seconds)},
                      {"observation_seconds", value(c.observation_seconds)},
                      {"accumulation_seconds", value(c.accumulation_seconds)}})},
        {"options",
         value(object{{"gpu_id", value(double(r.options.gpu_id))},
                      {"batch_atoms", value(double(r.options.gpu_batch_atoms))},
                      {"batch_structures", value(double(r.options.gpu_batch_structures))},
                      {"memory_bytes", value(double(r.options.gpu_memory_bytes))},
                      {"workspace_bytes", value(double(r.options.gpu_workspace_bytes))}})},
        {"observations",
         value(object{{"batches", value(double(s.batches))},
                      {"blocks", value(double(s.observation_blocks))},
                      {"download_bytes", value(double(s.download_bytes))},
                      {"workspace_bytes", value(double(s.observation_workspace_bytes))},
                      {"tracked_device_bytes", value(double(s.tracked_device_bytes))}})},
        {"seconds", value(object{{"total", value(r.total_seconds)},
                                 {"orthogonalization", value(r.orthogonalization_seconds)},
                                 {"assembly", value(r.assembly_seconds)},
                                 {"reduction", value(r.reduction_seconds)},
                                 {"solve", value(r.solve_seconds)},
                                 {"verification", value(r.verification_seconds)},
                                 {"gpu_initialize", value(s.gpu_initialize_seconds)},
                                 {"neighborhoods", value(s.neighborhoods_seconds)},
                                 {"gpu_forward", value(s.gpu_forward_seconds)},
                                 {"gpu_observation", value(s.gpu_observation_seconds)},
                                 {"accumulation", value(s.accumulation_seconds)}})}};
    Message("trainmtp GPU linear report " + value(report).serialize());
}
}

int trainmtp(int argc, char *argv[])
{
    using Clock = std::chrono::steady_clock;
    const auto command_start = Clock::now();
    auto elapsed = [](auto start)
    {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    auto &cmdopts = CMDOption::global_cmdopts;
    //     MPI_Init(&argc, &argv);
    //     mpi.InitComm(MPI_COMM_WORLD);

    GPIO iobj;
    std::vector<Atoms> atoms, atoms_valid;
    string indexfile = "index.map"; // can use with formation energy, first is dft, second is mlp
    string pfile = "potential.mtp";
    string infile = "train.cfg";
    string infilev = "";
    string informat = "";
    string invformat = "";
    string poutfile = "_Trained_Potential_.gpm";
    bool save_maxvol_state = false;
    PersistenceMode persistence;
    persistence.RegisterConsole(cmdopts);
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
    bool do_matrixA =
        false; // export full E/F/S design-matrix rows per MPI rank; output can be large
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

    cmdopts.addop(poutfile, "output", "o",
                  "Final model output (default GPM; explicit native paths remain supported)");
    cmdopts.addop(save_maxvol_state, "save-maxvol-state", "",
                  "Rebuild MaxVol on the training set and embed it in the final GPM", false, 0);
    cmdopts.attach_settings("builder:", "Final GPM MaxVol builder",
                            [](std::ostream &out)
                            {
                                BMatrixBuilderSettings b;
                                b.dtype = "pot_efs";
                                b.ListSettings(out, "builder:");
                            });
    cmdopts.attach_settings("selector:", "Final GPM MaxVol selector",
                            [](std::ostream &out)
                            {
                                ActiveSelector().ListSettings(out, "selector:");
                            });
    cmdopts.addop(debug, "debug", "", "Enable debug output stream", false, 1, 10)
        .addop(indexfile, "index", "id", "Index file mapping atom types to indices")
        .addop(pfile, "potfile", "pot", "Potential file")
        .addop(do_spec_coeffs, "species-only", "",
               "Fit only species coefficients through the generic L-BFGS trainer", false, 0)
        .addop(do_linear_only, "linear-fit", "lf", "Perform only linear parameter fitting", false,
               0)
        .addop(do_resacle, "do-rescale", "rs",
               "Use rescaling instead of a single linear fit (requires --linear-fit)", false, 0)
        .addop(balance_atomic_fitness, "balance-at", "bw",
               "Load balancing, when not all forces are fitted", false, 0)
        .addop(do_sgd, "do-sgd", "sgd", "Do Stochastic Gradient Descent optimization", false, 0)
        .addop(do_bfgs, "do-lbfgs", "lbf", "Do Limited-memory BFGS optimization", false, 0)
        .addop(do_df, "do-df", "df", "Do Differential Evolution iteration", false, 0)
        .addop(do_matrixATA, "do-matrix-ata", "ata",
               "Compute and output design matrix A^T*A for linear regression", false, 0)
        .addop(do_matrixA, "do-matrix-a", "a",
               "Compute and output full design matrix A (warning: may be memory intensive)", false,
               0)
        .addop(do_cg, "do-cg", "cg", "Do Conjugate Gradient optimization", false, 0)
        .addop(do_random, "do-random", "random", "Do random radial coeffs search", false, 0)
        .addop(infile, "trainset", "train", "Input training dataset file")
        .addop(infilev, "validset", "valid", "Input validation dataset file")
        .addop(informat, "trainset-format", "if", "Training dataset file format")
        .addop(invformat, "validset-format", "vf", "Validation dataset file format")
        .addop(do_curv_test, "test-curv", "", "Test curvature calculation for debugging", false, 0)
        .addop(curv_test_which, "test-curv-which", "", "Specify which curvature component to test")
        .addop(curv_test_nudge, "test-curv-nudge", "",
               "Perturbation magnitude for curvature testing")
        .addop(do_calcgrad, "calc-grad", "g", "Calculate gradient without optimization", false, 0)
        .addop(check_grad, "check-grad", "", "Perform gradient verification check", false, 0)
        .addop(check_grad_all, "check-grad-all", "",
               "Comprehensive gradient verification for all parameters", false, 0)
        .addop(settingfile, "settings", "", "input setting file", true)
        .attach_settings("fit:", "MTP training settings",
                         [](std::ostream &os)
                         {
                             Trainer_MTP().ListSettings(os, "fit:");
                         })
        .set_helponexit(false)
        .phrase_options(argc, argv);
    if (cmdopts.displayhelp)
    {
        if (mpi.rank == 0)
            cmdopts.showhelp(cout);
        return 0; // all ranks return to imrmlp's MPI_Finalize
    }
    Training::collective_try(
        [&]
        {
            if (GPIO::file_exist(settingfile))
            {
                Message("Loading settings");
                settings.Load(settingfile);
                settings.Print();
            }
            apply_settings_to_cmdopts(cmdopts, settings);
            persistence.training_defaults(settings, poutfile, "_Trained_Potential_.mtp",
                                          "potential.mtp", "potential.gpm", save_maxvol_state);
        });
    Training::collective_try(
        [&]
        {
            Trainer_MTP preflight(settings.ExtractSubSettings("fit"));
            preflight.ValidateGpuSettings();
            if (preflight.UsesGpu())
            {
                if (do_spec_coeffs || do_df || do_random || do_matrixATA || do_matrixA ||
                    do_curv_test ||
                    (do_linear_only &&
                     (do_sgd || do_bfgs || do_cg || do_calcgrad || check_grad || check_grad_all)))
                    GPERROR(
                        "Unsupported GPU trainmtp action or conflicting linear/nonlinear actions");
                if (do_linear_only)
                    preflight.ValidateGpuLinearSettings();
                else
                    preflight.ValidateGpuNonlinearSettings();
            }
        });
    const bool has_training_action = do_spec_coeffs || do_linear_only || do_sgd || do_bfgs ||
                                     do_cg || do_lbfgs || do_df || do_random;
    const bool has_diagnostic_action =
        do_matrixATA || do_matrixA || do_calcgrad || do_curv_test || check_grad || check_grad_all;
    if (do_resacle && !do_linear_only)
        GPERROR("trainmtp: --do-rescale requires --linear-fit");
    if (!has_training_action && !has_diagnostic_action)
        GPERROR(
            "trainmtp: select a training or diagnostic action (for example --linear-fit, --do-lbfgs, or --do-matrix-ata)");

    if (do_spec_coeffs && (do_linear_only || do_resacle || do_sgd || do_cg || do_lbfgs || do_df ||
                           do_random || has_diagnostic_action))
        GPERROR("trainmtp --species-only supports only the ordinary --do-lbfgs optimizer");
    if (do_spec_coeffs)
    {
        Trainer_MTP preflight(settings.ExtractSubSettings("fit"));
        preflight.ValidateSpeciesOnlySettings();
        const auto explicit_fit = settings.ExtractSubSettings("fit");
        for (const auto &[key, value] : explicit_fit.settings)
        {
            if (key.starts_with("lbfgspp:") || key.starts_with("lr_") || key == "learning_rate" ||
                key == "momentum" || key == "batch_size" || key == "epoch_limit" ||
                key == "clip_gradient" || key == "popsize" || key == "pre_init_iter" ||
                key == "lfit_eng_only" || key == "lfit_fastalgo" || key == "rescale_mode" ||
                key == "linear_precondition" || key.starts_with("shlbfgs_"))
                GPERROR("trainmtp --species-only does not support fit:" + key);
        }
    }

    Training::validate_training_output(poutfile, save_maxvol_state);
    if (save_maxvol_state && !has_training_action)
        GPERROR("--save-maxvol-state requires a training action");
    // Validate the final settings/CLI choice before input or output side effects.
    if (debug)
    {
        std::ofstream *debugfs = new std::ofstream("DEBUG" + mpi.fnm_ending);
        SetStreamForDebug(debugfs);
    }
    Training::collective_try(
        [&]
        {
            const bool index_explicit = cmdopts.was_provided("index") || settings.count("index");
            iobj.install_species_map(
                indexfile, index_explicit,
                Potentials::probe_model_elements(make_potential_settings(settings, pfile)),
                /*require_if_absent=*/false);
        });
    // read atoms
    const auto load_start = Clock::now();
    Message("Loading training set file " + infile);
    Training::collective_try(
        [&]
        {
            atoms = Training::load_training_data(iobj, infile, informat);
            if (!infilev.empty())
                atoms_valid = Training::load_training_data(iobj, infilev, invformat);
            Training::restore_training_fitness(atoms);
        });
    Message("Finish loading training set, size: " + std::to_string(atoms.size()) +
            ", valid set size: " + std::to_string(atoms_valid.size()));
    Training::distribute_training_data(atoms, balance_atomic_fitness);
    Training::distribute_training_data(atoms_valid, false);
    // Init trainer
    Message("Construct trainer");
    Trainer_MTP trainer(settings.ExtractSubSettings("fit"));
    if (atoms_valid.size() > 0)
    {
        trainer.monit_valid_err = true;
    }
    MPI_Bcast(&trainer.monit_valid_err, 1, MPI_C_BOOL, 0, mpi.comm);
    Message("Construct potential loader");
    Potentials::PotMTP potential;
    Training::collective_try(
        [&]
        {
            potential.Load(pfile);
            const auto chemistry = iobj.getmap();
            std::vector<std::string> symbols;
            for (int i = 0; i < potential.species_count; ++i)
            {
                auto it = chemistry.find(i);
                if (it == chemistry.end() || !MTPMetadata::element(it->second))
                {
                    symbols.clear();
                    break;
                }
                symbols.push_back(it->second);
            }
            if (!symbols.empty())
                potential.set_element_symbols(symbols);
        });
    const double load_seconds = elapsed(load_start);
    Message("Prepend to training");
    const auto neighborhoods_start = Clock::now();
    std::vector<Neighborhoods> nbhs, nbhs_valid;
    Training::collective_try(
        [&]
        {
            nbhs = Training::training_neighborhoods(atoms, potential.get_cutoff());
            nbhs_valid = Training::training_neighborhoods(atoms_valid, potential.get_cutoff());
        });
    const double neighborhoods_seconds = elapsed(neighborhoods_start);
    const auto init_start = Clock::now();
    trainer.init(atoms, nbhs, &potential);
    trainer.init_valid(atoms_valid, nbhs_valid);
    const double init_seconds = elapsed(init_start);

    // for test
    if (do_curv_test)
    {
        Message("Testing curvature");
        trainer.test_curv(curv_test_which, curv_test_nudge);
        goto end;
    }
    if (check_grad)
    {
        Message("Checking grads");
        trainer.check_grad(curv_test_which, curv_test_nudge);
        goto end;
    }
    if (check_grad_all)
    {
        Message("Checking grads for all");
        int ncoeffs = trainer.get_relaxed_coeffs_count();
        for (int i = 0; i < ncoeffs; ++i)
        {
            trainer.check_grad(i, curv_test_nudge);
        }
        goto end;
    }
    if (do_matrixA)
    {
        Message("Computing design matrix only, this file may be HUGE");
        trainer.get_A();
        MPI_Barrier(mpi.comm);
        goto end;
    }
    if (do_calcgrad)
    {
        Message("Computing gradient info");
        double f;
        alglib::real_1d_array g;
        g.setlength(trainer.get_relaxed_coeffs_count());
        trainer.AccumulateGard(f, g);
        Training::collective_try(
            [&]
            {
                if (mpi.rank != 0)
                    return;
                ofstream ofs("output_grad_info");
                if (!ofs)
                    GPERROR("Cannot open calc-grad output: output_grad_info");
                ofs << std::setprecision(17);
                for (int i = 0; i < g.length(); ++i)
                {
                    ofs << g[i] << '\n';
                }
                if (!ofs)
                    GPERROR("Failed writing calc-grad output: output_grad_info");
                ofs.close();
                if (!ofs)
                    GPERROR("Failed closing calc-grad output: output_grad_info");
            });
        goto end;
    }

    if (do_spec_coeffs)
    {
        trainer.TrainSpeciesOnly();
    }
    else if (do_linear_only)
    {
        Message("Fit linear");
        if (do_resacle)
            trainer.Rescale();
        else
            trainer.TrainLinear();
    }
    else if (has_training_action)
    {
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
        if (do_random)
        {
            trainer.TrainRandom();
            if (!save_maxvol_state)
                goto end;
        }
    }
    trainer.ladder_fin();
    if (has_training_action)
    {
        Training::publish_training_result(
            poutfile, save_maxvol_state,
            [&](const std::string &output)
            {
                Training::collective_try(
                    [&]
                    {
                        if (mpi.rank != 0)
                            return;
                        if (trainer.UsesGpu())
                            Training::publish_conditional_linear(
                                potential, output, atoms.empty() ? nullptr : &atoms.front());
                        else
                            Training::save_trained_model(potential, output);
                    });
            },
            settings, iobj, infile, informat);
        if (trainer.UsesGpu() && mpi.rank == 0)
        {
            if (do_linear_only)
                print_gpu_linear_report(trainer.GetGpuLinearReport());
            else
            {
                const auto &r = trainer.GetGpuNonlinearReport();
                using picojson::value;
                Message(
                    "trainmtp GPU nonlinear report " +
                    value(picojson::object{
                              {"evaluations", value(double(r.evaluations))},
                              {"gradient_evaluations", value(double(r.gradient_evaluations))},
                              {"validation_evaluations", value(double(r.validation_evaluations))},
                              {"batches", value(double(r.batches))},
                              {"peak_device_bytes", value(double(r.peak_device_bytes))},
                              {"evaluation_seconds", value(r.evaluation_seconds)}})
                        .serialize());
            }
        }
    }
    if (do_matrixATA)
    {
        Message("Computing energy, force and stress part of linearmatrix");
        trainer.get_ATA();
        MPI_Barrier(mpi.comm);
    }
end:
    //     MPI_Finalize();
    Message("Finished");
    if (do_linear_only && !do_resacle && mpi.rank == 0)
    {
        Message("trainmtp timing load_s=" + std::to_string(load_seconds));
        Message("trainmtp timing neighborhoods_s=" + std::to_string(neighborhoods_seconds));
        Message("trainmtp timing trainer_init_s=" + std::to_string(init_seconds));
        Message("trainmtp timing total_s=" + std::to_string(elapsed(command_start)));
    }
    return 0;
}
