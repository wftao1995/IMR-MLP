// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "utils/gputils.h"
#include "utils/gputils_eigen.h"
#include "io/gpio.h"
#include "potentials/potential_mtp.h"
#include "libalglib/optimization.h"
#include "utils/mpi_stubs.h"
#include "basis_analysis/DescriptorCollector.h"
#include "basis_analysis/CorrelationAnalyzer.h"
#include "basis_analysis/DiagnosticReport.h"
#include "basis_analysis/strategies/ThresholdPruning.h"
#include "basis_analysis/strategies/ClusterRepresentative.h"
#include "basis_analysis/strategies/IterativeGreedyVIF.h"
#include "apps/fastfit_utils.h"
#include "apps/fastfit_errors.h"
#include "apps/fastfit_profile.h"
#include "apps/fastfit_cost.h"
#include "apps/fastfit_export.h"
#include "apps/fastfit_cache.h"
#include <iomanip>
#include "train/linear_efs_system.h"
#include "train/efs_objective.h"
#include "train/linear_equilibration.h"
#include "train/efs_square_root.h"
#include <map>

#include <omp.h>
namespace func_fastfit {
auto &cmdopt = CMDOption::global_cmdopts;
using namespace std;
using namespace Eigen;
using namespace Potentials;
class Params : public ConnectSettings {
public:
    string pfile = "potential.mtp";
    string mE = "MATATA_E";
    string mF = "MATATA_F";
    string mS = "MATATA_S";
    string vE = "VECATA_E";
    string vF = "VECATA_F";
    string vS = "VECATA_S";
    string ts = "train.cfg";
    string vs = "valid.cfg";
    string format = "mlpcfg";
    string ix = "index.map";
    string method = "llt";

    double weightE = 1;
    double weightF = 0.1;
    double weightS = 0.001;
    double weightL2 = 1e-8;
    string objective_convention = "legacy", l2_reference = "native", objective_report;

    EFSFitWeights normalization;
    bool use_matrices = false;
    bool profile_enabled = false;
    string profile_output;
    Profile profiler;
    string selection_objective = "legacy", cost_report = "fastfit-cost.tsv";
    string cost_archive_rule = "spread", cost_export_dir;
    double max_valid_loss = -1, cost_benchmark_seconds = .01;
    int cost_benchmark_topk = 8, cost_benchmark_configs = 16;
    int cost_archive_size = 128, cost_benchmark_rounds = 5;
    int seed = 42;
    int search_steps = -1;
    int population = 0, generations = 0;
    double mutation_rate = .05;
    double group_mutation_rate = 0.;
    string group_mutation_kind = "basic";
    string population_init = "half";
    double sa_temperature = 1., sa_cooling = .995;
    int mode = 0;
    bool cmp = false;

    // only for mode 4
    int discard_method = 0; // 0: select high; 1: discard low
    double discard_rate = 0.1;
    double discard_epslow = 1e-6;
    bool addback = false;
    bool sample_CUR = false;
    int nsearch_CUR = 10;
    // for mode 5

    // for mode 7: correlation analysis and de-correlation
    int corr_strategy = 0;          // 0: threshold, 1: clustering, 2: greedy VIF
    double corr_pearson_threshold = 0.95;  // Pearson threshold (strategies 0, 1)
    double corr_vif_threshold = 10.0;      // VIF threshold (strategies 0, 2)
    int corr_max_samples = 50000;          // max atoms for correlation matrix
    int corr_max_features = 0;             // max features for strategy 2 (0 = none)
    bool corr_addback = false;             // preserve DAG integrity
    bool corr_save_reduced = false;        // save __DECORR__.mtp after selection

    EFSFitWeights weights() const {
        auto w = normalization;
        w.energy = weightE; w.force = weightF; w.stress = weightS;
        return w;
    }
    Params() {
        ConnectWithConsole(cmdopt, objective_convention, "objective-convention", "", "legacy or shared mean-moments EFS/L2 objective");
        ConnectWithConsole(cmdopt, l2_reference, "l2-reference", "", "L2 input reference: native (default), mean or sum; explicit references require mean-moments");
        ConnectWithConsole(cmdopt, objective_report, "objective-report", "", "Optional TSV objective contract, including effective L2");
        ConnectWithConsole(cmdopt, selection_objective, "selection-objective", "", "legacy or error-cost (modes 5/6)");
        ConnectWithConsole(cmdopt, max_valid_loss, "max-valid-loss", "", "Required mean weighted validation loss ceiling for error-cost");
        ConnectWithConsole(cmdopt, cost_report, "cost-report", "", "Cost selection TSV report (default fastfit-cost.tsv)");
        ConnectWithConsole(cmdopt, cost_benchmark_topk, "cost-benchmark-topk", "", "Final measured candidates, excluding baseline (default 8)");
        ConnectWithConsole(cmdopt, cost_benchmark_configs, "cost-benchmark-configs", "", "Representative validation configurations (default 16)");
        ConnectWithConsole(cmdopt, cost_archive_size, "cost-archive-size", "", "Bounded cross-generation candidate archive (default 128)");
        ConnectWithConsole(cmdopt, cost_archive_rule, "cost-archive-rule", "", "spread (legacy default) or bounded pareto archive");
        ConnectWithConsole(cmdopt, cost_export_dir, "cost-export-dir", "", "Export verified models, both Pareto fronts, objective and timing rounds into a fresh directory");
        ConnectWithConsole(cmdopt, cost_benchmark_rounds, "cost-benchmark-rounds", "", "Repeated measurement rounds, at least 3 (default 5)");
        ConnectWithConsole(cmdopt, cost_benchmark_seconds, "cost-benchmark-seconds", "", "Minimum seconds per candidate/path/round (default .01)");
        ConnectWithConsole(cmdopt, profile_enabled, "profile", "", "Report exclusive stage times and process peak RSS", 0);
        ConnectWithConsole(cmdopt, profile_output, "profile-output", "", "Write profile TSV (also enables profiling)");
        ConnectWithConsole(cmdopt, use_matrices, "use-matrices", "", "Read and validate exported unweighted E/F/S blocks", 0);
        ConnectWithConsole(cmdopt, normalization.energy_power, "energy-power", "", "Energy squared-residual atom-count power (default 1)");
        ConnectWithConsole(cmdopt, normalization.force_power, "force-power", "", "Force squared-residual atom-count power (default 0)");
        ConnectWithConsole(cmdopt, normalization.stress_power, "stress-power", "", "Virial squared-residual atom-count power (default 1)");
        ConnectWithConsole(cmdopt, normalization.norm_d, "force-norm-d", "", "Force normalization denominator offset");
        ConnectWithConsole(cmdopt, normalization.norm_f, "force-norm-f", "", "Force normalization scale (0 disables)");
        ConnectWithConsole(cmdopt, normalization.norm_p, "force-norm-p", "", "Force normalization power");
        ConnectWithConsole(cmdopt, seed, "seed", "", "Reproducible random seed");
        ConnectWithConsole(cmdopt, search_steps, "search-steps", "", "SA/greedy steps (-1 selects automatic budget)");
        ConnectWithConsole(cmdopt, population, "population", "", "NSGA-II population (0 selects automatic size)");
        ConnectWithConsole(cmdopt, generations, "generations", "", "NSGA-II generations (0 selects automatic budget)");
        ConnectWithConsole(cmdopt, mutation_rate, "mutation-rate", "", "NSGA-II per-scalar flip probability: [0,1], -1 uses 1/nscalar (default .05)");
        ConnectWithConsole(cmdopt, population_init, "population-init", "", "NSGA-II initialization: half, stratified, or dependency (default half)");
        ConnectWithConsole(cmdopt, group_mutation_rate, "group-mutation-rate", "", "NSGA-II probability of dropping one dependency group after scalar mutation (default 0)");
        ConnectWithConsole(cmdopt, group_mutation_kind, "group-mutation-kind", "", "NSGA-II group family: basic, radial, mixed (equal family probability); initialization remains basic (default basic)");
        ConnectWithConsole(cmdopt, sa_temperature, "sa-temperature", "", "Initial temperature for mean per-configuration loss");
        ConnectWithConsole(cmdopt, sa_cooling, "sa-cooling", "", "SA cooling factor in (0,1]");
        ConnectWithConsole(cmdopt, corr_strategy, "corr-strategy", "cs",
            "Decorrel strategy for mode 7: 0=threshold, 1=clustering, 2=greedy VIF");
        ConnectWithConsole(cmdopt, corr_pearson_threshold, "corr-pearson", "cp",
            "Pearson threshold for decorrelation (default 0.95)");
        ConnectWithConsole(cmdopt, corr_vif_threshold, "corr-vif", "cv",
            "VIF threshold for decorrelation (default 10.0)");
        ConnectWithConsole(cmdopt, corr_max_samples, "corr-max-samples", "cms",
            "Max atoms sampled for correlation analysis (default 50000)");
        ConnectWithConsole(cmdopt, corr_max_features, "corr-max-features", "cmf",
            "Max features to keep (strategy 2, 0=no limit)");
        ConnectWithConsole(cmdopt, corr_addback, "corr-addback", "caddback",
            "Add back DAG-path scalars after decorrelation", 0);
        ConnectWithConsole(cmdopt, corr_save_reduced, "corr-save-reduced", "csave",
            "Save reduced potential after selection", 0);
        ConnectWithConsole(cmdopt, discard_method, "sparse-method", "dm", "only valid for mode = 4");
        ConnectWithConsole(cmdopt, discard_rate, "discard-rate", "drate", "rate to discard scalar, only valid for mode 4");
        ConnectWithConsole(cmdopt, discard_epslow, "epslow", "epslow", "like ML_EPS_LOW");
        ConnectWithConsole(cmdopt, addback, "addback", "addback", "add scalars on path back", 0);
        ConnectWithConsole(cmdopt, sample_CUR, "cur", "cur", "Do CUR sampling", 0);
        ConnectWithConsole(cmdopt, nsearch_CUR, "cur-nsearch", "ntry", "maximum search iteration of CUR");
        ConnectWithConsole(cmdopt, pfile, "potfile", "pot", "potential file");
        ConnectWithConsole(cmdopt, ts, "trainset", "ts", "training set");
        ConnectWithConsole(cmdopt, vs, "validset", "val", "validation set");
        ConnectWithConsole(cmdopt, format, "format", "fmt", "training set file format", 1, 2);
        ConnectWithConsole(cmdopt, ix, "index", "id", "index file", 1, 2);
        ConnectWithConsole(cmdopt, mE, "matE", "me", "Energy part of design matrix", 1, 2);
        ConnectWithConsole(cmdopt, mF, "matF", "mf", "Force part of design matrix", 1, 2);
        ConnectWithConsole(cmdopt, mS, "matS", "ms", "Stress part of design matrix", 1, 2);
        ConnectWithConsole(cmdopt, vE, "vecE", "ve", "Energy target", 1, 2);
        ConnectWithConsole(cmdopt, vF, "vecF", "vf", "Force target", 1, 2);
        ConnectWithConsole(cmdopt, vS, "vecS", "vs", "Stress target", 1, 2);
        ConnectWithConsole(cmdopt, weightE, "weight-energy", "we", "Energy weight");
        ConnectWithConsole(cmdopt, weightF, "weight-force", "wf", "Force weight");
        ConnectWithConsole(cmdopt, weightS, "weight-stress", "ws", "Stress weight");
        ConnectWithConsole(cmdopt, weightL2, "weight-l2", "ww", "L2 regression weight");
        ConnectWithConsole(cmdopt, mode, "mode", "m", "MODE: \n\t\t0: compute eig \n\t\t1: simple fit \n\t\t2: apply selection \n\t\t3: cut-one search \n\t\t4: CUR-based selection \n\t\t5: temper (SA/greedy swap) \n\t\t6: NSGA-II basis selection \n\t\t7: correlation analysis & decorrelation");
        ConnectWithConsole(cmdopt, method, "method", "", "Method for solving the linear system. Choose llt(very fast but might be less accurate sometimes), ldlt(fast, accurate in most cases) or svd(slow but reliable), default use llt");
        ConnectWithConsole(cmdopt, cmp, "cmp", "cmp", "Only for test", 0, 2);
    }
    void InitBySettings(GPSettings &settings) {
        ApplySettings(settings, cmdopt.provided_options());
    }
};

void solve(MatrixXd &A, VectorXd &b, VectorXd &x, string method, Profile &profile) {
    x.resize(b.size());
    int success = 1;
    if (mpi.rank == 0) {
      auto timing = profile.scope(Profile::Stage::Solve);
      try {
        if (A.rows() != b.size() || A.cols() != b.size() || !A.allFinite() || !b.allFinite())
            GPERROR("Invalid normal equations");
        // MTP columns can differ by many orders of magnitude. Solving raw
        // normal equations can then discard physical directions even with SVD.
        // Change coordinates after adding the physical regularizer, preserving
        // the objective and the caller's matrix for later reduced-basis solves.
        MatrixXd equilibrated = A;
        VectorXd rhs = b;
        const auto scales = Training::equilibrate_linear_system(
            equilibrated, rhs, b.size(), false);
        equilibrated = MatrixXd(equilibrated.selfadjointView<Eigen::Lower>());
        if (method == "ldlt") {
            Eigen::LDLT<MatrixXd> decomposition(equilibrated);
            success = decomposition.info() == Eigen::Success;
            if (success) x = decomposition.solve(rhs);
        }
        else if (method == "llt") {
            Eigen::LLT<MatrixXd> decomposition(equilibrated);
            success = decomposition.info() == Eigen::Success;
            if (success) x = decomposition.solve(rhs);
        }
        else if (method == "svd")
            x = equilibrated.bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(rhs);
        else if (method == "qr")
            x = equilibrated.colPivHouseholderQr().solve(rhs);
        else if (method == "cod")
            x = equilibrated.completeOrthogonalDecomposition().solve(rhs);
        else if (method == "alglib") {
            alglib::real_1d_array ae_x, ae_b;
            alglib::real_2d_array ae_mat;
            ae_x.setlength(b.size());
            ae_b.setlength(b.size());
            ae_mat.setlength(b.size(), b.size());
            for (int i = 0; i < b.size(); ++i) {
                ae_b[i] = rhs[i];
            }
            for (int i = 0; i < b.size(); ++i) {
                for (int j = 0; j < b.size(); ++j) {
                    ae_mat(i, j) = equilibrated(i, j);
                }
            }
            alglib::ae_int_t info;
            alglib::densesolverreport rep;
            alglib::rmatrixsolve(ae_mat, ae_b, ae_x, rep);
            x.resize(b.size());
            for (int i = 0; i < b.size(); ++i) {
                x[i] = ae_x[i];
            }
        } else if (method == "algfast") {
            alglib::real_1d_array ae_x, ae_b;
            alglib::real_2d_array ae_mat;
            ae_x.setlength(b.size());
            ae_b.setlength(b.size());
            ae_mat.setlength(b.size(), b.size());
            for (int i = 0; i < b.size(); ++i) {
                ae_b[i] = rhs[i];
            }
            for (int i = 0; i < b.size(); ++i) {
                for (int j = 0; j < b.size(); ++j) {
                    ae_mat(i, j) = equilibrated(i, j);
                }
            }
            alglib::ae_int_t info;
            alglib::densesolverreport rep;
            alglib::spdmatrixsolvefast(ae_mat, true, ae_b);
            x.resize(b.size());
            for (int i = 0; i < b.size(); ++i) {
                x[i] = ae_b[i];
            }
        } else success = 0;
        if (success && !x.allFinite()) success = 0;
        if (success) {
            const double residual = (equilibrated * x - rhs).stableNorm();
            const double scale = equilibrated.stableNorm() * x.stableNorm() + rhs.stableNorm();
            if (!std::isfinite(residual) || !std::isfinite(scale) || residual > 1e-8 * scale)
                success = 0;
        }
        if (success) {
            for (int i = 0; i < x.size(); ++i) x[i] *= scales[i];
            if (!x.allFinite()) success = 0;
        }
      } catch (...) { success = 0; }
    }
    profile.communicate([&] { MPI_Bcast(&success, 1, MPI_INT, 0, mpi.comm); });
    if (!success) GPERROR("Failed to solve normal equations with method '" + method + "'");
    profile.communicate([&] { MPI_Bcast(x.data(), x.size(), MPI_DOUBLE, 0, mpi.comm); });
}


void RemoveBasis(MatrixXd &mat, VectorXd &vec, int toberemoved) {
    MatrixXd reduced_mat;
    VectorXd reduced_vec;
    detail::RemoveOneBasis(mat, vec, toberemoved, reduced_mat, reduced_vec);
    mat = std::move(reduced_mat);
    vec = std::move(reduced_vec);
}

void Check(Params &params, VectorXd &x, PotMTP &pot, vector<Atoms> &atoms, vector<Atoms> &ref, double &ermse, double &frmse, double &srmse, double &loss, Profile::Stage stage = Profile::Stage::DirectEFS) {
    pot.set_linear_coeff(x.data());
    auto &profile = params.profiler;
    auto timing = profile.scope(stage);
    ErrorsReport errs(params.weights());
    pot.omp_init();
#pragma omp parallel
{
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < atoms.size(); ++i) {
        pot.getEFSOMPthread(atoms[i]);
    }
}
    errs.clear();
    for (int i = 0; i < atoms.size(); ++i) {
        errs.add(atoms[i], ref[i]);
    }
    profile.communicate([&] { errs.Reduce(ermse, frmse, srmse, loss); });
}

double Check(Params &params, PotMTP &pot, vector<Atoms> &atoms, vector<Atoms> &ref, double &ermse, double &frmse, double &srmse, double &loss, Profile::Stage stage = Profile::Stage::DirectEFS) {
    auto &profile = params.profiler;
    auto timing = profile.scope(stage);
    ErrorsReport errs(params.weights());
    pot.omp_init();
#pragma omp parallel
{
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < atoms.size(); ++i) {
        pot.getEFSOMPthread(atoms[i]);
    }
}
    errs.clear();
    for (int i = 0; i < atoms.size(); ++i) {
        errs.add(atoms[i], ref[i]);
    }
    profile.communicate([&] { errs.Reduce(ermse, frmse, srmse, loss); });
    return loss;
}

double Check(Params &params, PotMTP &pot, vector<Atoms> &atoms, vector<Atoms> &atomsref, vector<Atoms> &atomsv, vector<Atoms> &atomsv_ref, double &ermse, double &frmse, double &srmse, double &loss, double &ermse_v, double &frmse_v, double &srmse_v, double &loss_v, Profile::Stage stage = Profile::Stage::DirectEFS) {
    ermse = 0; frmse = 0; srmse = 0; loss = 0;
    ermse_v = 0; frmse_v = 0; srmse_v = 0; loss_v = 0;
    Check(params, pot, atoms, atomsref, ermse, frmse, srmse, loss, stage);
    Check(params, pot, atomsv, atomsv_ref, ermse_v, frmse_v, srmse_v, loss_v, stage);
    return loss + loss_v;
}

int fastfit(int argc, char *argv[], AssemblyCache *assembly_cache = nullptr, bool assemble_only = false,
            BenchmarkObserver *observer = nullptr, bool experimental_square_root = false) {
    const auto application_start = Profile::Clock::now();
    string settingfile = "settings.ini";
    Params params;
    GPSettings settings;
    cmdopt.addop(settingfile, "settings", "", "input setting file", true)
          .add_request("settings")
          .check();
    cmdopt.phrase_options(argc, argv);
    settings.Load(settingfile);
    params.InitBySettings(settings);
    auto &profile = params.profiler;
    profile.start(params.profile_enabled || !params.profile_output.empty(), application_start);
    auto input_timing = profile.scope(Profile::Stage::Input);
    const bool error_cost = params.selection_objective == "error-cost";
    if (params.selection_objective != "legacy" && !error_cost)
        GPERROR("selection-objective must be legacy or error-cost");
    if (error_cost && ((params.mode != 5 && params.mode != 6) ||
        !std::isfinite(params.max_valid_loss) || params.max_valid_loss < 0 ||
        params.cost_benchmark_topk < 1 || params.cost_benchmark_configs < 1 ||
        params.cost_archive_size < std::max(2, params.cost_benchmark_topk) ||
        params.cost_benchmark_rounds < 3 || params.cost_benchmark_rounds > 100 ||
        !std::isfinite(params.cost_benchmark_seconds) || params.cost_benchmark_seconds <= 0 ||
        params.cost_benchmark_seconds > 10 || params.cost_report.empty()))
        GPERROR("Invalid error-cost settings: modes 5/6 require max-valid-loss >= 0 and positive bounded benchmark budgets");
    // All ranks reach the same error boundary before any later collectives.
    auto collective_action = [&](auto &&action, bool root_only = false) {
        int failed = 0, any_failed = 0;
        std::string error;
        if (!root_only || mpi.rank == 0) {
            try { action(); } catch (const std::exception &e) { failed = 1; error = e.what(); }
        }
        profile.communicate([&] { MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, mpi.comm); });
        if (any_failed) GPERROR((error.empty() ? "Cost selection failed on another MPI rank" : error));
    };
    if (params.cost_archive_rule != "spread" && params.cost_archive_rule != "pareto")
        GPERROR("cost-archive-rule must be spread or pareto");
    if ((!params.cost_export_dir.empty() || params.cost_archive_rule != "spread") && !error_cost)
        GPERROR("Pareto archive/export requires selection-objective error-cost");
    collective_action([&] {
        if (!params.cost_export_dir.empty() && (std::filesystem::exists(params.cost_export_dir)
            || std::filesystem::exists(params.cost_export_dir + ".tmp")))
            GPERROR("cost-export-dir and its .tmp path must be fresh");
    }, true);

    for (double weight : {params.weightE, params.weightF, params.weightS, params.weightL2})
        if (!std::isfinite(weight) || weight < 0) GPERROR("fastfit weights must be finite and nonnegative");
    for (double value : {params.normalization.energy_power, params.normalization.force_power,
                         params.normalization.stress_power, params.normalization.norm_d,
                         params.normalization.norm_f, params.normalization.norm_p})
        if (!std::isfinite(value)) GPERROR("fastfit normalization parameters must be finite");
    if (params.normalization.norm_d <= 0 || params.normalization.norm_f < 0 || params.normalization.norm_p <= 0)
        GPERROR("Invalid force normalization parameters");
    if (params.search_steps < -1 || params.population < 0 || params.population == 1 || params.generations < 0 ||
        !std::isfinite(params.mutation_rate) ||
        (params.mutation_rate != -1. && (params.mutation_rate < 0 || params.mutation_rate > 1)) ||
        !std::isfinite(params.group_mutation_rate) || params.group_mutation_rate < 0 || params.group_mutation_rate > 1 ||
        (params.group_mutation_kind != "basic" && params.group_mutation_kind != "radial" && params.group_mutation_kind != "mixed") ||
        (params.population_init != "half" && params.population_init != "stratified" && params.population_init != "dependency") ||
        !std::isfinite(params.sa_temperature) || params.sa_temperature <= 0 ||
        !std::isfinite(params.sa_cooling) || params.sa_cooling <= 0 || params.sa_cooling > 1)
        GPERROR("Invalid fastfit search parameters");
    if (!std::isfinite(params.corr_pearson_threshold) || params.corr_pearson_threshold < 0 || params.corr_pearson_threshold > 1 ||
        !std::isfinite(params.corr_vif_threshold) || params.corr_vif_threshold < 1 || params.corr_max_features < 0)
        GPERROR("Invalid correlation thresholds or feature limit");
    if (params.mode < 0 || params.mode > 7) GPERROR("fastfit mode must be in [0, 7]");
    if (params.mode == 4 && params.discard_method != 0 && params.discard_method != 1) {
        GPERROR("sparse-method must be 0 (select high) or 1 (discard low)");
    }
    if (params.mode == 7 && (params.corr_strategy < 0 || params.corr_strategy > 2)) {
        GPERROR("corr-strategy must be 0 (threshold), 1 (clustering), or 2 (greedy VIF)");
    }
    if (params.mode == 4 && (!std::isfinite(params.discard_rate) || params.discard_rate < 0.0 || params.discard_rate > 1.0)) {
        GPERROR("discard-rate must be in [0, 1]");
    }
    if (params.mode == 7 && params.corr_max_samples <= 0) GPERROR("corr-max-samples must be positive");
    if (params.mode == 4 && params.sample_CUR && params.nsearch_CUR <= 0) {
        GPERROR("cur-nsearch must be positive when CUR sampling is enabled");
    }
    
    // MAIN
    GPIO iobj;
    if (GPIO::file_exist(params.ix))
        iobj.readmap(params.ix);
    else {
        Warning("index file not found");
        iobj.setmap();
    }
    if (!GPIO::file_exist(params.pfile)) {
        GPERROR("potential file not found: " + params.pfile);
    }
    auto model_bytes = [&] {
        std::ifstream in(params.pfile, std::ios::binary);
        if (!in) GPERROR("Cannot read potential for assembly cache");
        return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    };
    const std::string source_model = model_bytes();
    const std::string cached_model = assembly_cache ? source_model : std::string();
    PotMTP basepot(params.pfile);
    if (source_model != model_bytes()) GPERROR("Potential changed while loading inputs");
    vector<Atoms> atoms, atomsref, atoms_v, atomsref_v, benchmark_pool;
    {
        vector<Atoms> atmp;
        if (!iobj.read(atmp, params.ts, params.format)) GPERROR("Cannot read training set");
        for (int i = 0; i < atmp.size(); ++i) {
            if (i % mpi.size == mpi.rank) {
                atoms.push_back(atmp[i]);
            }
        }
        if (GPIO::file_exist(params.vs)) {
            atmp.clear();
            if (!iobj.read(atmp, params.vs, params.format)) GPERROR("Cannot read validation set");
            for (int i = 0; i < atmp.size(); ++i) {
                if (i % mpi.size == mpi.rank) {
                    atoms_v.push_back(atmp[i]);
                }
            }
            if (error_cost && mpi.rank == 0) benchmark_pool = std::move(atmp);
        }
    }
    atomsref = atoms;
    atomsref_v = atoms_v;
    input_timing.stop();
    int local_train_count = static_cast<int>(atoms.size());
    int local_valid_count = static_cast<int>(atoms_v.size());
    int global_train_count = 0;
    int global_valid_count = 0;
    profile.communicate([&] { MPI_Allreduce(&local_train_count, &global_train_count, 1, MPI_INT, MPI_SUM, mpi.comm); });
    profile.communicate([&] { MPI_Allreduce(&local_valid_count, &global_valid_count, 1, MPI_INT, MPI_SUM, mpi.comm); });
    const bool has_valid = global_valid_count > 0;
    if (error_cost && !has_valid) GPERROR("error-cost requires a nonempty labeled validation set");
    if (global_train_count == 0) GPERROR("training set contains no configurations");
    const auto objective = Training::EFSObjective::resolve(params.objective_convention,
        params.l2_reference, params.weightL2, global_train_count, true);
    if(experimental_square_root && (!objective.canonical() || params.mode==3 || params.cmp))
        GPERROR("Experimental square-root backend requires mean-moments and excludes mode3/cmp");
    if (objective.canonical() && basepot.radial_mode != 0)
        GPERROR("mean-moments currently supports ordinary MTP radial_mode=0");
    collective_action([&] {
        objective.write(cout, params.weights(), "# ");
        if (!params.objective_report.empty()) {
            if (std::filesystem::exists(params.objective_report)) GPERROR("objective-report must be a fresh path");
            ofstream out(params.objective_report);
            if (!out) GPERROR("Cannot open objective report: " + params.objective_report);
            objective.write(out, params.weights());
            out.close();
            if (!out) GPERROR("Cannot write objective report: " + params.objective_report);
        }
    }, true);
    const int nl = basepot.lin_coeff_size;
    const int nspecies = basepot.species_count;
    const int nscalar = basepot.alpha_scalar_moments;
    double ermse = 0, frmse = 0, srmse = 0, loss = 0;
    double ermse_v = 0, frmse_v = 0, srmse_v = 0, loss_v = 0;
    auto unit_weights = params.normalization;
    unit_weights.energy = unit_weights.force = unit_weights.stress = 1.;
    auto build_system = [&](vector<Atoms> &data) {
        auto timing = profile.scope(Profile::Stage::Assembly);
        std::shared_ptr<Training::LinearEFSSystem> result;
        std::string cache_key;
        if (assembly_cache) {
            collective_action([&] {
                cache_key = assembly_key(cached_model, data, unit_weights, mpi.rank, mpi.size);
                if(experimental_square_root)cache_key+="experimental-square-root-v1";
                cache_key.append(reinterpret_cast<const char *>(&mpi.comm), sizeof(mpi.comm));
                result = assembly_cache->find(cache_key);
            });
            int hit = bool(result), all_hit = 0;
            profile.communicate([&] { MPI_Allreduce(&hit, &all_hit, 1, MPI_INT, MPI_MIN, mpi.comm); });
            if (all_hit) {
                Message("Reusing verified in-process EFS assembly");
                return result;
            }
        }
        int failed = 0, any_failed = 0;
        std::string error;
        try { result = std::make_shared<Training::LinearEFSSystem>(basepot, data, unit_weights,0,experimental_square_root); }
        catch (const std::exception &e) { failed = 1; error = e.what(); }
        profile.communicate([&] { MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, mpi.comm); });
        if (any_failed) GPERROR("fastfit EFS assembly failed: " + error);
        profile.communicate([&] { result->reduce(); });
        if (assembly_cache) assembly_cache->put(std::move(cache_key), result);
        return result;
    };
    std::shared_ptr<Training::LinearEFSSystem> train_system, valid_system;
    MatrixXd matA;
    VectorXd vecA;
    if (params.mode != 7 || params.corr_save_reduced) {
        Message("Assembling EFS normal equations from current potential and data");
        train_system = build_system(atomsref);
        train_system->require_observations(params.weightE, params.weightF, params.weightS);
        profile.measure(Profile::Stage::Assembly, [&] {
            matA = train_system->matrix(params.weightE, params.weightF, params.weightS);
            vecA = train_system->rhs(params.weightE, params.weightF, params.weightS);
        });
        if (params.use_matrices) {
            auto timing = profile.scope(Profile::Stage::MatrixValidation);
            int valid = 1;
            std::string error;
            if (mpi.rank == 0) {
                try {
                    const string mats[] = {params.mE, params.mF, params.mS};
                    const string vecs[] = {params.vE, params.vF, params.vS};
                    for (int i = 0; i < 3; ++i) {
                        MatrixXd matrix; VectorXd rhs;
                        readMatrixFromBinFile(mats[i], matrix);
                        readVectorFromBinFile(vecs[i], rhs);
                        const auto &block = train_system->channels[i];
                        VectorXd expected = block.residual_rhs + block.gram * train_system->anchor;
                        if (matrix.rows() != nl || matrix.cols() != nl || rhs.size() != nl ||
                            !matrix.allFinite() || !rhs.allFinite() ||
                            !matrix.isApprox(block.gram, 1e-10) || !rhs.isApprox(expected, 1e-10))
                            GPERROR("External matrices do not match current data, potential, and normalization; regenerate with trainmtp --do-matrix-ata");
                    }
                } catch (const std::exception &e) { valid = 0; error = e.what(); }
            }
            profile.communicate([&] { MPI_Bcast(&valid, 1, MPI_INT, 0, mpi.comm); });
            if (!valid) GPERROR("fastfit external matrix validation failed: " + error);
            Message("External E/F/S blocks verified");
        }
        const int regularization_begin = objective.canonical() ? nspecies : 0;
        matA.diagonal().tail(nl - regularization_begin).array() += objective.native_l2;
        if ((params.mode == 5 || params.mode == 6) && has_valid) {
            valid_system = build_system(atomsref_v);
            valid_system->require_observations(params.weightE, params.weightF, params.weightS);
        }
    }
    if (observer && observer->systems)
        collective_action([&] { observer->systems(basepot, train_system, valid_system,
            matA, vecA, global_train_count, global_valid_count); }, true);
    if (assemble_only) {
        profile.report(cout, params.profile_output, params.mode, "");
        return 0;
    }
    std::unique_ptr<Training::EFSSquareRootFit> root_fit;
    if(experimental_square_root && train_system)collective_action([&] {
        auto timing=profile.scope(Profile::Stage::Assembly);
        root_fit=std::make_unique<Training::EFSSquareRootFit>(
            train_system->root_factor(params.weightE,params.weightF,params.weightS),
            train_system->anchor,nspecies,objective.native_l2);
    },true);
    auto root_solve=[&](const vector<int> &ids,VectorXd &out) {
        out.resize(nspecies+ids.size());
        collective_action([&] {
            auto timing=profile.scope(Profile::Stage::Solve);
            const VectorXd full=root_fit->fit(ids).coefficients;
            out.head(nspecies)=full.head(nspecies);
            for(int j=0;j<ids.size();++j)out[nspecies+j]=full[nspecies+ids[j]];
        },true);
        profile.communicate([&]{MPI_Bcast(out.data(),out.size(),MPI_DOUBLE,0,mpi.comm);});
    };
    cost::Workload cost_workload;
    double full_cost = 1.;
    cost::Archive cost_archive(params.cost_archive_size, params.max_valid_loss, params.cost_archive_rule == "pareto");
    struct Candidate {
        VectorXd coeffs;
        double train_loss = 0, valid_loss = 0, relative_cost = 0;
        BasisCostStats cost_stats;
    };
    auto cost_entry = [&](const vector<int> &ids, const Candidate &c) {
        return cost::Entry{ids, c.train_loss/global_train_count, c.valid_loss/global_valid_count,
                           c.relative_cost, c.cost_stats};
    };
    map<vector<int>, Candidate> candidates;
    int cache_hits = 0, cache_misses = 0;
    auto observe_candidate = [&](const vector<int> &ids, const Candidate &c, bool hit) {
        if (observer && observer->candidate)
            collective_action([&] { observer->candidate(ids, c.train_loss/global_train_count,
                has_valid ? c.valid_loss/global_valid_count : 0., hit); }, true);
    };
    auto fit_subset = [&](const vector<int> &scalar_ids, bool remember = true) {
        auto found = candidates.find(scalar_ids);
        if (found != candidates.end()) {
            ++cache_hits;
            if (error_cost && remember) cost_archive.add(cost_entry(scalar_ids, found->second));
            observe_candidate(scalar_ids, found->second, true);
            return found->second;
        }
        ++cache_misses;
        vector<int> indices;
        for (int i = 0; i < nspecies; ++i) indices.push_back(i);
        for (int i : scalar_ids) indices.push_back(nspecies + i);
        MatrixXd matrix; VectorXd rhs;
        detail::BuildReducedSystem(matA, vecA, indices, matrix, rhs);
        Candidate candidate;
        if(experimental_square_root)root_solve(scalar_ids,candidate.coeffs);
        else solve(matrix, rhs, candidate.coeffs, "ldlt", profile);
        VectorXd full = VectorXd::Zero(nl);
        for (int i = 0; i < indices.size(); ++i) full[indices[i]] = candidate.coeffs[i];
        double losses[2] = {};
        int valid_loss = 1;
        if (mpi.rank == 0) {
          auto timing = profile.scope(Profile::Stage::CachedScore);
          try {
            losses[0] = train_system->loss(full, params.weightE, params.weightF, params.weightS);
            if (valid_system) losses[1] = valid_system->loss(full, params.weightE, params.weightF, params.weightS);
          } catch (...) { valid_loss = 0; }
        }
        profile.communicate([&] { MPI_Bcast(&valid_loss, 1, MPI_INT, 0, mpi.comm); });
        if (!valid_loss) GPERROR("Invalid cached candidate loss");
        profile.communicate([&] { MPI_Bcast(losses, 2, MPI_DOUBLE, 0, mpi.comm); });
        candidate.train_loss = losses[0]; candidate.valid_loss = losses[1];
        if (error_cost) {
            auto timing = profile.scope(Profile::Stage::CostAnalysis);
            candidate.cost_stats = basepot.analyze_basis(scalar_ids).stats;
            candidate.relative_cost = candidate.cost_stats.efs_work(cost_workload.active,
                cost_workload.stored, basepot.rb_size) / full_cost;
            if (remember) cost_archive.add(cost_entry(scalar_ids, candidate));
        }
        // Bound coefficient-cache memory; repeated chromosomes/swaps need no solve.
        if (candidates.size() == 256) candidates.erase(candidates.begin());
        candidates.emplace(scalar_ids, candidate);
        observe_candidate(scalar_ids, candidate, false);
        return candidate;
    };

    auto selection_timing = profile.scope(Profile::Stage::Selection);
    std::string mode_error;
    try {
    if (error_cost) {
        auto timing = profile.scope(Profile::Stage::CostAnalysis);
        collective_action([&] {
            cost_workload = cost::sample_workload(basepot, benchmark_pool, params.cost_benchmark_configs, params.seed);
            vector<Atoms>().swap(benchmark_pool);
        }, true);
        double counts[] = {cost_workload.active, cost_workload.stored, cost_workload.max_neighbors};
        profile.communicate([&] { MPI_Bcast(counts, 3, MPI_DOUBLE, 0, mpi.comm); });
        cost_workload.active = counts[0]; cost_workload.stored = counts[1]; cost_workload.max_neighbors = counts[2];
        vector<int> all(nscalar); iota(all.begin(), all.end(), 0);
        full_cost = basepot.analyze_basis(all).stats.efs_work(counts[0], counts[1], basepot.rb_size);
        if (!(full_cost > 0)) GPERROR("Invalid full-basis EFS work estimate");
    }
    auto finish_cost_selection = [&](const string &output, const string &suffix) {
        vector<int> all(nscalar); iota(all.begin(), all.end(), 0);
        const auto baseline = fit_subset(all, false);
        vector<cost::Entry> entries{cost_entry(all, baseline)};
        for (auto &entry : cost_archive.shortlist(params.cost_benchmark_topk))
            if (entry.ids != all) entries.push_back(std::move(entry));
        vector<std::unique_ptr<PotMTP>> models;
        vector<cost::Measurement> measurements;
        vector<cost::Metrics> metrics;
        auto selectable = [&](const cost::Entry &e) {
            return cost::feasible(e.valid_loss, params.max_valid_loss) && (params.mode == 6 || e.ids != all);
        };
        auto write_report = [&](const string &status, int best) {
            auto timing = profile.scope(Profile::Stage::Output);
            collective_action([&] {
                ofstream out(params.cost_report);
                if (!out) GPERROR("Cannot open cost report: " + params.cost_report);
                objective.write(out, params.weights(), "# ");
                out << std::setprecision(17)
                    << "# status\t" << status << "\n# objective\terror-cost\n# estimator\tefs-ops-v1\n"
                    << "# loss_order_grid\t2^(frexp_exponent(max(1,abs(loss)))-42); feasibility uses raw loss\n"
                    << "# max_valid_loss\t" << params.max_valid_loss
                    << "\n# feasibility_tolerance\t" << cost::tolerance(params.max_valid_loss)
                    << "\n# seed\t" << params.seed << "\n# ranks\t" << mpi.size
                    << "\n# population_init\t" << params.population_init
                    << "\n# mutation_rate\t" << (params.mutation_rate == -1. ? 1./nscalar : params.mutation_rate)
                    << "\n# group_mutation_rate\t" << params.group_mutation_rate
                    << "\n# group_mutation_kind\t" << params.group_mutation_kind
                    << "\n# omp_threads\t" << omp_num_threads
                    << "\n# compiler\t" << __VERSION__
                    << "\n# backend\tPotMTP::getEFSOMP; rank0; total includes neighborhoods\n"
                    << "# tie_rule\tIQR overlap with fastest median, then validation loss, then IDs\n"
                    << "# archive_rule\t" << params.cost_archive_rule << "; bounded; retain cost/loss endpoints\n"
                    << "# archive_capacity\t" << params.cost_archive_size
                    << "\n# benchmark_rounds\t" << params.cost_benchmark_rounds
                    << "\n# benchmark_seconds_per_path_round\t" << params.cost_benchmark_seconds
                    << "\n# benchmark_topk\t" << params.cost_benchmark_topk
                    << "\n# active_neighbors_per_atom\t" << cost_workload.active
                    << "\n# stored_neighbors_per_atom\t" << cost_workload.stored
                    << "\n# input_mean_valid_loss\t" << valid_system->loss(valid_system->anchor,
                         params.weightE, params.weightF, params.weightS)/global_valid_count
                    << "\n# weights_E_F_V_L2\t" << params.weightE << ',' << params.weightF << ',' << params.weightS << ',' << params.weightL2
                    << "\n# powers_E_F_V\t" << params.normalization.energy_power << ',' << params.normalization.force_power << ',' << params.normalization.stress_power
                    << "\n# force_norm_d_f_p\t" << params.normalization.norm_d << ',' << params.normalization.norm_f << ',' << params.normalization.norm_p
                    << "\n# model\t" << params.pfile << "\n# trainset\t" << params.ts << "\n# validset\t" << params.vs
                    << "\n# selected_index\t" << best << "\n# sampled_global_validation_ids\t";
                for (auto id : cost_workload.global_ids) out << id << ',';
                out << '\n';
                for (auto name : {"OMP_PROC_BIND", "OMP_PLACES", "MKL_NUM_THREADS", "HOSTNAME"})
                    out << "# " << name << '\t' << (std::getenv(name) ? std::getenv(name) : "unset") << '\n';
                ifstream cpu("/proc/cpuinfo"); string line;
                while (getline(cpu, line)) if (line.starts_with("model name")) { out << "# cpu\t" << line << '\n'; break; }
                out << "index\tkind\tfeasible\tselectable\ttrain_mean_loss\tvalid_mean_loss\tcost_relative\tscalars\tbasics\tmoments\tproducts\tchannels\tmax_degree\tradial_bytes\tscratch_bytes_estimate\ttrain_E_RMSE\ttrain_F_RMSE\ttrain_V_RMSE\tvalid_E_RMSE\tvalid_F_RMSE\tvalid_V_RMSE\tkernel_ns_atom\ttotal_ns_atom\ttotal_q25\ttotal_q75\tspeedup_vs_baseline\toriginal_scalar_ids\n";
                for (size_t i = 0; i < entries.size(); ++i) {
                    const auto &e = entries[i]; const auto &s = e.stats;
                    out << i << '\t' << (i == 0 ? "baseline" : "candidate") << '\t'
                        << cost::feasible(e.valid_loss, params.max_valid_loss) << '\t' << selectable(e)
                        << '\t' << e.train_loss << '\t' << e.valid_loss << '\t' << e.relative
                        << '\t' << s.scalars << '\t' << s.basics << '\t' << s.moments << '\t' << s.products
                        << '\t' << s.channels << '\t' << s.max_degree << '\t' << s.radial_bytes
                        << '\t' << s.scratch_bytes(cost_workload.max_neighbors, basepot.rb_size, omp_num_threads);
                    if (i < metrics.size()) {
                        for (int k=0; k<3; ++k) out << '\t' << metrics[i].train[k];
                        for (int k=0; k<3; ++k) out << '\t' << metrics[i].valid[k];
                    } else out << "\tNA\tNA\tNA\tNA\tNA\tNA";
                    if (i < measurements.size()) {
                        const auto &t = measurements[i];
                        out << '\t' << t.kernel.median << '\t' << t.total.median << '\t' << t.total.low << '\t' << t.total.high
                            << '\t' << measurements[0].total.median/t.total.median;
                    } else out << "\tNA\tNA\tNA\tNA\tNA";
                    out << '\t'; for (int id : e.ids) out << id << ','; out << '\n';
                }
                out << "# archive\toriginal_ids\tmean_valid_loss\tcost_relative\n";
                for (const auto &e : cost_archive.entries()) {
                    out << "# archive\t"; for (int id : e.ids) out << id << ',';
                    out << '\t' << e.valid_loss << '\t' << e.relative << '\n';
                }
                out.close();
                if (!out) GPERROR("Cannot write cost report: " + params.cost_report);
            }, true);
        };
        if (none_of(entries.begin(), entries.end(), selectable)) {
            write_report("no_feasible_candidate", -1);
            GPERROR("No candidate meets max-valid-loss; see " + params.cost_report);
        }
        for (auto &entry : entries) {
            auto fit = fit_subset(entry.ids, false);
            std::unique_ptr<PotMTP> reloaded;
            collective_action([&] {
                PotMTP compact(basepot);
                set<int> kept(entry.ids.begin(), entry.ids.end()), removed;
                for (int i=0; i<nscalar; ++i) if (!kept.count(i)) removed.insert(i);
                compact.remove_scalar_by_id(removed);
                compact.set_linear_coeff(fit.coeffs.data());
                compact.remove_orphans();
                // Exercise exactly the .mtp serialization format without shared
                // temporary paths between MPI ranks or concurrent runs.
                stringstream serialized; compact.Save(serialized);
                if (!serialized) GPERROR("Cannot serialize compact candidate");
                reloaded = std::make_unique<PotMTP>(); reloaded->Load(serialized);
                vector<int> ids(reloaded->alpha_scalar_moments); iota(ids.begin(), ids.end(), 0);
                if (!(reloaded->analyze_basis(ids).stats == entry.stats)) GPERROR("Compact model cost statistics mismatch");
            });
            cost::Metrics m;
            auto direct = [&](vector<Atoms> &refs, std::array<double, 4> &result) {
                auto timing = profile.scope(Profile::Stage::FinalEFS);
                ErrorsReport report(params.weights());
                collective_action([&] {
                    reloaded->omp_init();
                    for (auto &ref : refs) { auto pred = ref; reloaded->getEFSOMP(pred); report.add(pred, ref); }
                });
                profile.communicate([&] { report.Reduce(result[0], result[1], result[2], result[3]); });
            };
            direct(atomsref, m.train); direct(atomsref_v, m.valid);
            const auto matches = [](double a, double b) { return std::abs(a-b) <= 1e-8 * std::max({1., std::abs(a), std::abs(b)}); };
            if (!matches(m.train[3]/global_train_count, entry.train_loss) ||
                !matches(m.valid[3]/global_valid_count, entry.valid_loss)) GPERROR("Direct EFS and cached candidate losses disagree");
            entry.train_loss = m.train[3]/global_train_count; entry.valid_loss = m.valid[3]/global_valid_count;
            metrics.push_back(m);
            if (mpi.rank == 0) models.push_back(std::move(reloaded));
        }
        if (none_of(entries.begin(), entries.end(), selectable)) {
            write_report("no_feasible_verified_candidate", -1);
            GPERROR("No serialized candidate passes the direct validation loss ceiling");
        }
        int best = -1;
        {
            auto timing = profile.scope(Profile::Stage::CostBenchmark);
            collective_action([&] {
                measurements = cost::benchmark(models, cost_workload, params.cost_benchmark_rounds,
                    params.cost_benchmark_seconds, params.seed);
                auto eligible = entries;
                if (params.mode == 5) eligible[0].valid_loss = INFINITY; // keep fixed scalar count
                best = cost::choose_measured(eligible, measurements, params.max_valid_loss);
            }, true);
            profile.communicate([&] { MPI_Bcast(&best, 1, MPI_INT, 0, mpi.comm); });
        }
        // Write diagnostics successfully before publishing a selected potential.
        write_report("ok", best);
        try {
            auto timing = profile.scope(Profile::Stage::Output);
            collective_action([&] {
                if (!params.cost_export_dir.empty())
                    cost::export_bundle(params.cost_export_dir,source_model,objective,params.weights(),
                        entries,models,measurements,metrics,cost_workload,params.max_valid_loss,params.mode,best,params.seed);
                ofstream out(output); if (!out) GPERROR("Cannot open selected potential: " + output);
                models[best]->Save(out); out.close(); if (!out) GPERROR("Cannot write selected potential: " + output);
                set<int> selected(entries[best].ids.begin(), entries[best].ids.end()), removed;
                for (int i=0; i<nscalar; ++i) if (!selected.count(i)) removed.insert(i);
                detail::WriteScalarIds("selected_ones_" + suffix + ".dat", selected);
                detail::WriteScalarIds("removed_" + suffix + ".dat", removed);
                fmt::print(cout, "COST BEST: mean_valid_loss={} cost_relative={} total_ns_atom={} speedup={} scalars={} saved={} report={}\n",
                    entries[best].valid_loss, entries[best].relative, measurements[best].total.median,
                    measurements[0].total.median/measurements[best].total.median, entries[best].ids.size(), output, params.cost_report);
            }, true);
        } catch (...) {
            write_report("output_failed", best);
            throw;
        }

    };

    if (params.mode == 0) {
        if (mpi.rank == 0) {
            Message("Computing condition number and singluar value");
            auto analysis_timing = profile.scope(Profile::Stage::Correlation);
            MatrixXd matr = matA.block(nspecies, nspecies, nl - nspecies, nl-nspecies);
            SelfAdjointEigenSolver<MatrixXd> eigsolver(matr, ComputeEigenvectors);
            VectorXd eigval = eigsolver.eigenvalues();
            MatrixXd eigvec = eigsolver.eigenvectors();
            analysis_timing.stop();
            auto output_timing = profile.scope(Profile::Stage::Output);
            ofstream ofs("EIGENSYS");
            for (int i = 0; i < eigval.size(); ++i)
                fmt::print(ofs, "{} {}\n", eigval[i], fmt::join(eigvec.col(i).begin(), eigvec.col(i).end(), " "));
            ofs.close();
            writeMatrixToBinFile("EIGVEC", eigvec);
            writeVectorToBinFile("EIGVAL", eigval);
        }
    } else if (params.mode == 1) {
        Message("Simple fitting");
        VectorXd x;
        x.resize(vecA.size());
        // we compare 
        // 1. LLT (fast but doc says low accuracy)
        // 2. LDLT 
        // 3. BDCSVD (note for ICC need -fp-model precise option, disable -ffast-math option)
        if (params.cmp) {
            VectorXd llt, ldlt, svd, alglib, algspd, qr, cod;
            Check(params, basepot, atoms, atomsref, ermse, frmse, srmse, loss);
            Check(params, basepot, atoms_v, atomsref_v, ermse_v, frmse_v, srmse_v, loss_v);
            if (mpi.rank == 0)
                fmt::print(cout, "{:6} : {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n", "BEG", ermse, frmse, srmse, loss, ermse_v, frmse_v, srmse_v, loss_v);
            
            auto run_one_test = [&](string method, VectorXd &xx) {
                solve(matA, vecA, xx, method, profile);
                basepot.set_linear_coeff(xx.data());
                Check(params, basepot, atoms, atomsref, ermse, frmse, srmse, loss);
                Check(params, basepot, atoms_v, atomsref_v, ermse_v, frmse_v, srmse_v, loss_v);
                if (mpi.rank == 0)
                    fmt::print(cout, "{:6} : {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n", method, ermse, frmse, srmse, loss, ermse_v, frmse_v, srmse_v, loss_v);
            };

            run_one_test("alglib", alglib);
            run_one_test("algfast", algspd);
            run_one_test("llt", llt);
            run_one_test("ldlt", ldlt);
            run_one_test("svd", svd);
            run_one_test("cod", cod);
            run_one_test("qr", qr);

        } else {
            if(experimental_square_root) {
                vector<int> all(nscalar);std::iota(all.begin(),all.end(),0);root_solve(all,x);
            } else solve(matA, vecA, x, params.method, profile);
            PotMTP pot(basepot);
            pot.set_linear_coeff(x.data());
            Check(params, pot, atoms, atomsref, ermse, frmse, srmse, loss, Profile::Stage::FinalEFS);
            Check(params, pot, atoms_v, atomsref_v, ermse_v, frmse_v, srmse_v, loss_v, Profile::Stage::FinalEFS);
            if (mpi.rank == 0) {
                fmt::print(cout, "{:6} : {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n", params.method, ermse, frmse, srmse, loss, ermse_v, frmse_v, srmse_v, loss_v);
                profile.measure(Profile::Stage::Output, [&] { pot.Save("__Trained__.mtp"); });
            }
        }
    } else if (params.mode == 2) {
        Message("Read selected_ones.dat and discard basis");
        VectorXd x;
        set<int> selected_scalars;
        try {
            selected_scalars = profile.measure(Profile::Stage::Input, [&] { return detail::ReadScalarIds("selected_ones.dat", nscalar); });
        } catch (const std::exception &error) {
            GPERROR(error.what());
        }
        vector<int> sel;
        sel.reserve(nspecies + selected_scalars.size());
        for (int i = 0; i < nspecies; ++i) sel.push_back(i);
        for (int id : selected_scalars) sel.push_back(nspecies + id);
        MatrixXd matr;
        VectorXd vecr;
        detail::BuildReducedSystem(matA, vecA, sel, matr, vecr);
        if (mpi.rank == 0) {
            auto analysis_timing = profile.scope(Profile::Stage::Correlation);
            SelfAdjointEigenSolver<MatrixXd> eigsolver(matr, ComputeEigenvectors);
            auto eigval = eigsolver.eigenvalues();
            MatrixXd eigvec = eigsolver.eigenvectors();
            analysis_timing.stop();
            auto output_timing = profile.scope(Profile::Stage::Output);
            ofstream ofs("EIGENSYS_REDUCED");
            for (int i = 0; i < eigval.size(); ++i)
                fmt::print(ofs, "{} {}\n", eigval[i], fmt::join(eigvec.col(i).begin(), eigvec.col(i).end(), " "));
            ofs.close();
        }
        if(experimental_square_root)root_solve(vector<int>(selected_scalars.begin(),selected_scalars.end()),x);
        else solve(matr, vecr, x, params.method, profile);
        PotMTP pot(basepot);
        set<int> removed;
        for (int i = 0; i < nl - nspecies; ++i) {
            if (selected_scalars.count(i) == 0) removed.insert(i);
        }
        pot.remove_scalar_by_id(removed);
        pot.set_linear_coeff(x.data());
        Check(params, pot, atoms, atomsref, atoms_v, atomsref_v, ermse, frmse, srmse, loss, ermse_v, frmse_v, srmse_v, loss_v);
        if (mpi.rank == 0)
            fmt::print(cout, "{:6} : {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n", "REDUCED", ermse, frmse, srmse, loss, ermse_v, frmse_v, srmse_v, loss_v);
        pot.remove_orphans();
        Check(params, pot, atoms, atomsref, atoms_v, atomsref_v, ermse, frmse, srmse, loss, ermse_v, frmse_v, srmse_v, loss_v, Profile::Stage::FinalEFS);
        if (mpi.rank == 0) profile.measure(Profile::Stage::Output, [&] { pot.Save("__REDUCED__.mtp"); });
        if (mpi.rank == 0)
            fmt::print(cout, "{:6} : {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n", "FINAL", ermse, frmse, srmse, loss, ermse_v, frmse_v, srmse_v, loss_v);
    } else if (params.mode == 3) {
        Message("Basis searching by cutone");
        VectorXd x;
        solve(matA, vecA, x, params.method, profile);
        Check(params, x, basepot, atoms, atomsref, ermse, frmse, srmse, loss);
        Check(params, x, basepot, atoms_v, atomsref_v, ermse_v, frmse_v, srmse_v, loss_v);
        if (mpi.rank == 0)
            fmt::print(cout, "{:6} : {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n", "BEG", ermse, frmse, srmse, loss, ermse_v, frmse_v, srmse_v, loss_v);
        for (int i = nspecies; i < nl; ++i) {
            MatrixXd nA = matA;
            VectorXd nV = vecA;
            RemoveBasis(nA, nV, i);
            solve(nA, nV, x, params.method, profile);
            PotMTP pot(basepot);
            set<int> rr;
            rr.insert(i-nspecies);
            pot.remove_scalar_by_id(rr);
            Check(params, x, pot, atoms, atomsref, ermse, frmse, srmse, loss);
            Check(params, x, pot, atoms_v, atomsref_v, ermse_v, frmse_v, srmse_v, loss_v);
            if (mpi.rank == 0) {
                fmt::print(cout, "{:6} : {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n", i-nspecies, ermse, frmse, srmse, loss, ermse_v, frmse_v, srmse_v, loss_v);
                cout.flush();
            }
        }
    } else if (params.mode == 4) {
        Message("Basis cutting by CUR");
        if (nscalar == 0) GPERROR("mode 4 requires at least one scalar moment");
        if (mpi.rank == 0) {
            MatrixXd eigvec;
            VectorXd eigval;
            profile.measure(Profile::Stage::Input, [&] { readMatrixFromBinFile("EIGVEC", eigvec); });
            profile.measure(Profile::Stage::Input, [&] { readVectorFromBinFile("EIGVAL", eigval); });
            if (eigvec.rows() != nscalar || eigvec.cols() != nscalar ||
                eigval.size() != nscalar || !eigvec.allFinite() || !eigval.allFinite()) {
                GPERROR("EIGVEC/EIGVAL dimensions or values are invalid for this potential");
            }
            vector<double> score;
            score.assign(nscalar, 0);
            int ndiscard = nscalar * params.discard_rate;
            int nselect = nscalar - ndiscard;
            // select high
            if (params.discard_method == 0) {
                double pref = 1.0 / std::max(1, nselect);
                for (int j = 0; j < nscalar; ++j) {
                    for (int m = 0; m < nselect; ++m) {
                        score[j] += pref * eigvec(j, nscalar - m - 1) * eigvec(j, nscalar - m - 1);
                    }
                }
            } // discard low 
            else if (params.discard_method == 1) {
                double pref = 1.0 / std::max(1, ndiscard);
                for (int j = 0; j < nscalar; ++j) {
                    for (int m = 0; m < nscalar; ++m) {
                        if (eigval(m) > params.discard_epslow) {
                            score[j] += pref * eigvec(j, m) * eigvec(j, m);
                        } else {
                            score[j] += 0;
                        }
                    }
                }
            } else {
                GPERROR("ERROR");
            }
            // now sort the score and select those high value ones
            vector<pair<int, double>> score_order;
            score_order.resize(score.size());
            for (int i = 0; i < score.size(); ++i) {
                score_order[i].first = i;
                score_order[i].second = score[i];
            }
            stable_sort(score_order.begin(), score_order.end(), [](const pair<int, double> &a, const pair<int, double> &b) {
                return a.second > b.second;
            });
            // select high
            auto score_output_timing = profile.scope(Profile::Stage::Output);
            ofstream ofs2("score.dat");
            for (int i = 0; i < nscalar; ++i) {
                ofs2 << i << " " << score[i] << '\n';
            }
            ofs2.close();
            score_output_timing.stop();
            //
            set<int> removed_ids;
            set<int> allowed_ids;
            if (params.discard_method == 0) {
                MatrixXd matM;
                matM = matA.block(nspecies, nspecies, nscalar, nscalar);
                if (params.sample_CUR && nselect > 0) {
                    auto dice_gen = uniform_real_distribution<double>(0, 1);
                    mt19937 mtd(static_cast<mt19937::result_type>(params.seed));
                    const double reconstruction_floor=experimental_square_root ? detail::CURRoundoffTolerance(matM) : 0.;
                    std::ofstream stability;
                    if(experimental_square_root) {
                        stability.open("cur-stability.tsv");
                        stability<<"sample\trange_rank\treconstruction\troundoff_floor\n"<<std::setprecision(17);
                    }
                    int sample_index=0;
                    auto sample_score = [&]() {
                        allowed_ids.clear();
                        removed_ids.clear();


                        vector<double> wheelprob = score;
                        for (int i = 1; i < wheelprob.size(); ++i) {
                            wheelprob[i] += wheelprob[i-1];
                        }
                        wheelprob.back() = 1.0;
                        vector<int> corrmap(nscalar, 0);
                        iota(corrmap.begin(), corrmap.end(), 0);
                        for (int i = 0; i < nselect; ++i) {
                            double diceval = dice_gen(mtd);
                            int selected;
                            for (int j = 0; j < corrmap.size(); ++j) {
                                if (diceval <= wheelprob[j]) {
                                    selected = corrmap[j];
                                    // pop out
                                    corrmap.erase(corrmap.begin() + j);
                                    wheelprob.erase(wheelprob.begin() + j);
                                    double csum = 0;
                                    for (int k = 0; k < corrmap.size(); ++k) {
                                        csum += score[corrmap[k]];
                                        wheelprob[k] = score[corrmap[k]];
                                    }
                                    if (csum <= 0.0) {
                                        const double uniform = corrmap.empty() ? 0.0 : 1.0 / corrmap.size();
                                        for (int k = 0; k < corrmap.size(); ++k) wheelprob[k] = uniform;
                                    } else {
                                        for (int k = 0; k < corrmap.size(); ++k) wheelprob[k] /= csum;
                                    }
                                    for (int k = 1; k < corrmap.size(); ++k) {
                                        wheelprob[k] += wheelprob[k-1];
                                    }
                                    if (!wheelprob.empty()) wheelprob.back() = 1.0;
                                    allowed_ids.insert(selected);
                                    break;
                                }
                            }
                        }
                        // compute 
                        int nselect_true = allowed_ids.size();
                        MatrixXd matC, matU, matR; // matR = matC.transpose()
                        matC.resize(nscalar, nselect_true);
                        matU.resize(nselect_true, nselect_true);
                        int ct = 0;
                        for (auto &t : allowed_ids) {
                            matC.col(ct) = matM.col(t);
                            ++ct;
                        }
                        matR = matC.transpose();
                        double reconstruction;
                        if(experimental_square_root) {
                            int rank=0;reconstruction=detail::StableCURResidual(matM,matC,rank);
                            stability<<sample_index++<<'\t'<<rank<<'\t'<<reconstruction<<'\t'<<reconstruction_floor<<'\n'<<std::flush;
                        } else {
                            MatrixXd inverse = matC.completeOrthogonalDecomposition().pseudoInverse();
                            matU = inverse * matM * inverse.transpose();
                            reconstruction = (matM - matC * matU * matR).norm();
                        }
                        if (observer && observer->cur_sample)
                            observer->cur_sample(vector<int>(allowed_ids.begin(), allowed_ids.end()), reconstruction);
                        return reconstruction;
                    };
                    // Keep the sampled basis set with the smallest CUR reconstruction error.
                    double scmin = 1e50;
                    set<int> allowed_best;
                    for (int i = 0; i < params.nsearch_CUR; ++i) {
                        double sc = sample_score();
                        Message("F" + to_string(i) + " = " + to_string(sc));
                        if (sc < scmin-reconstruction_floor) {
                            scmin = sc;
                            allowed_best = allowed_ids;
                        }
                    }
                    allowed_ids = allowed_best;
                    for (int i = 0; i < nscalar; ++i) {
                        if (allowed_ids.count(i) == 0) {
                            removed_ids.insert(i);
                        }
                    }
                }
                else {
                    for (int i = 0; i < nselect; ++i) {
                        allowed_ids.insert(score_order[i].first);
                    }
                    for (int i = nselect; i < nscalar; ++i) {
                        removed_ids.insert(score_order[i].first);
                    }
                }
            } else if (params.discard_method == 1) {
                for (int i = 0; i < nselect; ++i) {
                    allowed_ids.insert(score_order[i].first);
                }
                for (int i = nselect; i < nscalar; ++i) {
                    removed_ids.insert(score_order[i].first);
                }
            }
            // addback
            if (params.addback) {
                Message("ADD BACK");
                basepot.add_back_path_scalars_to_remove_id(removed_ids);
                allowed_ids.clear();
                for (int i = 0; i < basepot.alpha_scalar_moments; ++i) {
                    if (removed_ids.count(i) == 0) {
                        allowed_ids.insert(i);
                    }
                }
            }
            detail::ValidatePartition(allowed_ids, removed_ids, nscalar);
            profile.measure(Profile::Stage::Output, [&] { detail::WriteScalarIds("selected_ones.dat", allowed_ids); });
            profile.measure(Profile::Stage::Output, [&] { detail::WriteScalarIds("removed.dat", removed_ids); });
        }
    } else if (params.mode == 5) {
        Message("Basis cutting by temper (Simulated Annealing / Greedy swap)");

        // ── Read initial selection from mode 4 output ──
        set<int> best_selected, cur_selected, removed_set;
        try {
            best_selected = profile.measure(Profile::Stage::Input, [&] { return detail::ReadScalarIds("selected_ones.dat", nscalar); });
            removed_set = profile.measure(Profile::Stage::Input, [&] { return detail::ReadScalarIds("removed.dat", nscalar); });
            detail::ValidatePartition(best_selected, removed_set, nscalar);
        } catch (const std::exception &error) {
            GPERROR(error.what());
        }
        cur_selected = best_selected;
        const int nsel = static_cast<int>(best_selected.size());
        const int nrem = static_cast<int>(removed_set.size());

        // Use a bounded automatic search budget unless explicitly configured.
        const int nsteps = params.search_steps >= 0 ? params.search_steps :
            static_cast<int>(std::max(1LL, min<long long>(5000, static_cast<long long>(nsel) * nrem / 2)));
        const double T_init = params.sa_temperature;
        const double cooling = params.sa_cooling;
        bool is_greedy = (params.method == "greedy");

        {
            // ── Evaluation helper: build reduced ATA/ATy, solve, expand, check ──
            // Returns total loss (train + valid if present).
            auto eval_swap = [&](const set<int> &sel_ids, VectorXd &x_out,
                                 double &loss, double &loss_v) {
                auto candidate = fit_subset(vector<int>(sel_ids.begin(), sel_ids.end()));
                x_out = candidate.coeffs;
                loss = candidate.train_loss;
                loss_v = candidate.valid_loss;
                return candidate.relative_cost;
            };
            auto score = [&](double train, double valid, double relative) {
                if (error_cost) return cost::sa_score(valid/global_valid_count, relative, params.max_valid_loss);
                return train / global_train_count + (has_valid ? valid / global_valid_count : 0.);
            };

            // ── Initial baseline ──
            VectorXd x_best;
            double loss_best, loss_vb;
            double initial_cost = eval_swap(cur_selected, x_best, loss_best, loss_vb);
            double loss_cur = score(loss_best, loss_vb, initial_cost);
            double loss_bmin = loss_cur;

            if (mpi.rank == 0)
                fmt::print(cout, "INIT : train_loss={} valid_loss={} score={}\n", loss_best, loss_vb, loss_cur);

            // ── Random number generators ──
            int random_seed = params.seed;
            profile.communicate([&] { MPI_Bcast(&random_seed, 1, MPI_INT, 0, mpi.comm); });
            mt19937 rng(static_cast<mt19937::result_type>(random_seed));
            uniform_real_distribution<double> uni01(0.0, 1.0);

            vector<int> sel_vec(cur_selected.begin(), cur_selected.end());
            vector<int> rem_vec(removed_set.begin(), removed_set.end());

            if (sel_vec.empty() || rem_vec.empty()) {
                GPERROR("Empty selection or removal set -- cannot run mode 5. Check selected_ones.dat and removed.dat.");
            }

            double T = T_init;
            int    n_accept = 0;

            // ── Main swap loop ──
            for (int step = 0; step < nsteps; ++step) {
                // Pick one from each set
                int si = sel_vec[uniform_int_distribution<int>(0, (int)sel_vec.size() - 1)(rng)];
                int ri = rem_vec[uniform_int_distribution<int>(0, (int)rem_vec.size() - 1)(rng)];

                // Swap
                set<int> new_sel = cur_selected;
                new_sel.erase(si);
                new_sel.insert(ri);

                VectorXd x_new;
                double loss_n, loss_vn;
                double relative_cost = eval_swap(new_sel, x_new, loss_n, loss_vn);
                double loss_new = score(loss_n, loss_vn, relative_cost);

                // ── Accept / reject ──
                bool accept = false;
                if (loss_new <= loss_cur) {
                    accept = true;
                } else if (!is_greedy) {
                    double delta = loss_cur - loss_new; // negative
                    if (uni01(rng) < exp(delta / T))
                        accept = true;
                }

                if (accept) {
                    cur_selected = new_sel;
                    loss_cur = loss_new;
                    // Update vectors in-place
                    for (int k = 0; k < (int)sel_vec.size(); ++k)
                        if (sel_vec[k] == si) { sel_vec[k] = ri; break; }
                    for (int k = 0; k < (int)rem_vec.size(); ++k)
                        if (rem_vec[k] == ri) { rem_vec[k] = si; break; }
                    ++n_accept;

                    if (loss_new < loss_bmin) {
                        loss_bmin     = loss_new;
                        best_selected = new_sel;
                        x_best        = x_new;
                    }
                }

                if (!is_greedy) T = std::max(std::numeric_limits<double>::min(), T * cooling);

                // Periodic progress
                if ((step + 1) % max(1, nsteps / 10) == 0 || step == 0) {
                    if (mpi.rank == 0)
                        fmt::print(cout, "{:>6} : train_loss={} valid_loss={} score={} T={} acc={}/{}\n",
                                   step + 1, loss_n, loss_vn, loss_new, T, n_accept, step + 1);
                }
            }

            if (error_cost) finish_cost_selection("__TEMPERED__.mtp", "tempered");
            else {
            // ── Save best potential ──
            set<int> removed_best;
            for (int i = 0; i < nscalar; ++i)
                if (best_selected.count(i) == 0) removed_best.insert(i);

            PotMTP pot_best(basepot);
            pot_best.remove_scalar_by_id(removed_best);
            pot_best.set_linear_coeff(x_best.data());
            pot_best.remove_orphans();
            if (mpi.rank == 0) profile.measure(Profile::Stage::Output, [&] { pot_best.Save("__TEMPERED__.mtp"); });

            double ermse_b, frmse_b, srmse_b, ermse_vb, frmse_vb, srmse_vb;
            Check(params, pot_best, atoms, atomsref, atoms_v, atomsref_v,
                  ermse_b, frmse_b, srmse_b, loss_best,
                  ermse_vb, frmse_vb, srmse_vb, loss_vb, Profile::Stage::FinalEFS);
            if (mpi.rank == 0)
                fmt::print(cout, "{:>6} : {:15.10f} {:15.10f} {:15.10f} {:15.10f} | {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n",
                           "BEST", ermse_b, frmse_b, srmse_b, loss_best,
                           ermse_vb, frmse_vb, srmse_vb, loss_vb);
            }
        }
    } else if (params.mode == 6) {
        Message("Basis selection by NSGA-II (native implementation)");

        // Population and generation budgets are configurable.
        const int    pop_size      = params.population > 0 ? params.population : min(50, nscalar * 2);
        const int    n_generations = params.generations > 0 ? params.generations : min(100, nscalar);
        const double cross_rate    = 0.9;
        const double mut_rate      = params.mutation_rate == -1. ? 1./nscalar : params.mutation_rate;
        if (nscalar < 2) {
            GPERROR("Need at least 2 scalar moments for NSGA-II selection (have " +
                    to_string(nscalar) + ")");
        }
        if (mpi.rank == 0)
            fmt::print(cout, "NSGA-II initialization={} mutation_rate={} expected_flips={}\n",
                       params.population_init, mut_rate, mut_rate*nscalar);

        {
            // ═══════════════════════════════════════════════════════════════
            // Individual representation
            // ═══════════════════════════════════════════════════════════════
            struct Individual {
                vector<bool> chrom;      // length nscalar: true = selected
                double       obj[2];     // obj[0]=train_loss, obj[1]=valid_loss or nselected
                double       constraint_loss = 0; // unrounded loss for feasibility
                int          rank;       // Pareto front index
                double       crowd_dist; // crowding distance
                VectorXd     coeffs;     // full-length solution vector
            };

            // ── Evaluate one individual ──────────────────────────────────
            auto evaluate = [&](Individual &ind) {
                vector<int> sel_idxs;
                for (int i = 0; i < nscalar; ++i)
                    if (ind.chrom[i]) sel_idxs.push_back(i);
                if (sel_idxs.empty()) {
                    ind.chrom[0] = true;
                    sel_idxs.push_back(0);
                }

                auto candidate = fit_subset(sel_idxs);
                ind.coeffs = VectorXd::Zero(nl);
                ind.coeffs.head(nspecies) = candidate.coeffs.head(nspecies);
                for (int i = 0; i < sel_idxs.size(); ++i)
                    ind.coeffs(nspecies + sel_idxs[i]) = candidate.coeffs(nspecies + i);
                ind.constraint_loss = has_valid ? candidate.valid_loss/global_valid_count : 0.;
                ind.obj[0] = error_cost ? cost::order_loss(ind.constraint_loss) : candidate.train_loss;
                ind.obj[1] = error_cost ? candidate.relative_cost : (has_valid ? candidate.valid_loss : (double)sel_idxs.size());
            };

            // ═══════════════════════════════════════════════════════════════
            // Random generator
            // ═══════════════════════════════════════════════════════════════
            int random_seed = params.seed;
            profile.communicate([&] { MPI_Bcast(&random_seed, 1, MPI_INT, 0, mpi.comm); });
            mt19937 rng(static_cast<mt19937::result_type>(random_seed));
            uniform_real_distribution<double> uni01(0.0, 1.0);

            vector<vector<int>> dependency_groups;
            vector<vector<int>> radial_groups;
            vector<int> group_order;
            if (params.population_init == "dependency" || params.group_mutation_rate > 0) {
                auto timing = profile.scope(Profile::Stage::CostAnalysis);
                dependency_groups = cost::dependency_groups(basepot);
                group_order.resize(dependency_groups.size());
                iota(group_order.begin(), group_order.end(), 0);
                if (params.population_init == "dependency") shuffle(group_order.begin(), group_order.end(), rng);
                if (params.group_mutation_rate > 0 && params.group_mutation_kind != "basic")
                    radial_groups = cost::radial_dependency_groups(basepot);
                if (mpi.rank == 0) fmt::print(cout, "NSGA-II dependency_groups={} group_mutation_rate={} group_mutation_kind={} radial_groups={}\n",
                                             dependency_groups.size(), params.group_mutation_rate,
                                             params.group_mutation_kind, radial_groups.size());
            }
            auto drop_group = [&](vector<bool> &chrom) {
                const bool radial = params.group_mutation_kind == "radial" ||
                    (params.group_mutation_kind == "mixed" && uni01(rng) < .5);
                // An unavailable chosen family is a no-op; do not silently
                // reweight mixed mutation according to the current model.
                return cost::drop_dependency_group(chrom, radial ? radial_groups : dependency_groups, rng);
            };

            // ═══════════════════════════════════════════════════════════════
            // Initialise population
            // ═══════════════════════════════════════════════════════════════
            vector<Individual> pop(pop_size);

            // Seed individual 0 from mode-4 selection if available
            set<int> seed_set;
            if (GPIO::file_exist("selected_ones.dat")) {
                try {
                    seed_set = profile.measure(Profile::Stage::Input, [&] { return detail::ReadScalarIds("selected_ones.dat", nscalar); });
                } catch (const std::exception &error) {
                    GPERROR(error.what());
                }
            }
            for (int i = 0; i < pop_size; ++i) {
                pop[i].chrom.resize(nscalar);
                if ((error_cost || params.population_init == "dependency") && i == pop_size-1) {
                    std::fill(pop[i].chrom.begin(), pop[i].chrom.end(), true);
                } else if (i == 0 && !seed_set.empty()) {
                    for (int j = 0; j < nscalar; ++j)
                        pop[i].chrom[j] = (seed_set.count(j) > 0);
                } else if (params.population_init == "dependency" && !group_order.empty()) {
                    std::fill(pop[i].chrom.begin(), pop[i].chrom.end(), true);
                    for (int id : dependency_groups[group_order[i % group_order.size()]]) pop[i].chrom[id] = false;
                } else {
                    // Spread initial densities across the search space. A fixed
                    // 50% population concentrates large models near nscalar/2.
                    const double keep_probability = params.population_init == "stratified"
                        ? .05 + .9 * i / (pop_size-1.) : .5;
                    for (int j = 0; j < nscalar; ++j)
                        pop[i].chrom[j] = (uni01(rng) < keep_probability);
                }
                evaluate(pop[i]);
            }

            // ═══════════════════════════════════════════════════════════════
            // NSGA-II helpers (lambdas for access to outer scope)
            // ═══════════════════════════════════════════════════════════════

            // Check if individual a dominates b (better in at least one,
            // not worse in any).  Lower objective is better.
            auto dominates = [&](const Individual &a, const Individual &b) {
                if (error_cost) return cost::dominates(a.constraint_loss, a.obj[1], b.constraint_loss, b.obj[1], params.max_valid_loss);
                bool better = false;
                for (int o = 0; o < 2; ++o) {
                    if (a.obj[o] > b.obj[o]) return false; // worse
                    if (a.obj[o] < b.obj[o]) better = true;
                }
                return better;
            };

            auto crowding_sort = [](vector<Individual> &front) {
                int nf = (int)front.size();
                if (nf <= 2) {
                    for (auto &ind : front) ind.crowd_dist = INFINITY;
                    return;
                }
                for (auto &ind : front) ind.crowd_dist = 0.0;
                for (int o = 0; o < 2; ++o) {
                    sort(front.begin(), front.end(),
                         [o](const Individual &a, const Individual &b) {
                             return a.obj[o] < b.obj[o];
                         });
                    double range = front.back().obj[o] - front.front().obj[o];
                    if (range <= 0) range = 1.0;
                    front.front().crowd_dist = INFINITY;
                    front.back().crowd_dist  = INFINITY;
                    for (int i = 1; i < nf - 1; ++i)
                        front[i].crowd_dist += (front[i + 1].obj[o] - front[i - 1].obj[o]) / range;
                }
            };

            auto assign_crowding = [](vector<Individual> &individuals,
                                      const vector<int> &front_indices) {
                vector<pair<double, double>> objectives;
                objectives.reserve(front_indices.size());
                for (int idx : front_indices)
                    objectives.emplace_back(individuals[idx].obj[0], individuals[idx].obj[1]);
                const vector<double> distances = detail::CrowdingDistances(objectives);
                for (size_t i = 0; i < front_indices.size(); ++i)
                    individuals[front_indices[i]].crowd_dist = distances[i];
            };

            // Non-dominated sort: assigns ranks and returns fronts.
            auto nondominated_sort = [&](vector<Individual> &individuals) {
                int N = (int)individuals.size();
                vector<vector<int>> S(N);
                vector<int> n_dom(N, 0);
                vector<vector<int>> fronts(1);
                for (int p = 0; p < N; ++p) {
                    for (int q = 0; q < N; ++q) {
                        if (p == q) continue;
                        if (dominates(individuals[p], individuals[q]))
                            S[p].push_back(q);
                        else if (dominates(individuals[q], individuals[p]))
                            ++n_dom[p];
                    }
                    if (n_dom[p] == 0) {
                        individuals[p].rank = 0;
                        fronts[0].push_back(p);
                    }
                }
                int fi = 0;
                while (fi < (int)fronts.size() && !fronts[fi].empty()) {
                    vector<int> next_front;
                    for (int p : fronts[fi])
                        for (int q : S[p])
                            if (--n_dom[q] == 0) {
                                individuals[q].rank = fi + 1;
                                next_front.push_back(q);
                            }
                    ++fi;
                    if (!next_front.empty()) fronts.push_back(next_front);
                }
                return fronts;
            };

            // ═══════════════════════════════════════════════════════════════
            // Main generational loop
            // ═══════════════════════════════════════════════════════════════
            for (int gen = 0; gen < n_generations; ++gen) {
                // ── 1. Non-dominated sort ─────────────────────────────────
                auto fronts = nondominated_sort(pop);

                // ── 2. Crowding distance per front ─────────────────────────
                for (auto &fidxs : fronts) {
                    assign_crowding(pop, fidxs);
                }

                // ── 3. Create offspring ───────────────────────────────────
                const int N = (int)pop.size();
                vector<Individual> offspring(pop_size);
                for (int i = 0; i < pop_size; i += 2) {
                    // Binary tournament selection
                    auto tournament = [&]() -> int {
                        int a = uniform_int_distribution<int>(0, N - 1)(rng);
                        int b = uniform_int_distribution<int>(0, N - 1)(rng);
                        if (pop[a].rank != pop[b].rank)
                            return pop[a].rank < pop[b].rank ? a : b;
                        return pop[a].crowd_dist > pop[b].crowd_dist ? a : b;
                    };
                    int p1 = tournament();
                    int p2 = tournament();

                    // Single-point crossover
                    vector<bool> c1 = pop[p1].chrom;
                    vector<bool> c2 = pop[p2].chrom;
                    if (uni01(rng) < cross_rate && nscalar > 1) {
                        int cx = uniform_int_distribution<int>(1, nscalar - 1)(rng);
                        for (int j = cx; j < nscalar; ++j)
                            swap(c1[j], c2[j]);
                    }

                    // Mutation
                    for (int j = 0; j < nscalar; ++j) {
                        if (uni01(rng) < mut_rate) c1[j] = !c1[j];
                        if (uni01(rng) < mut_rate) c2[j] = !c2[j];
                    }
                    if (params.group_mutation_rate > 0) {
                        if (uni01(rng) < params.group_mutation_rate) drop_group(c1);
                        if (uni01(rng) < params.group_mutation_rate) drop_group(c2);
                    }

                    offspring[i].chrom = c1;
                    evaluate(offspring[i]);
                    if (i + 1 < pop_size) {
                        offspring[i + 1].chrom = c2;
                        evaluate(offspring[i + 1]);
                    }
                }

                // ── 4. Environmental selection (elitism) ─────────────────
                vector<Individual> combined;
                combined.reserve(2 * pop_size);
                for (auto &ind : pop)    combined.push_back(ind);
                for (auto &ind : offspring) combined.push_back(ind);

                // Re-rank combined population
                auto Cfronts = nondominated_sort(combined);

                // Fill new population front by front
                pop.clear();
                for (auto &fidxs : Cfronts) {
                    if (pop.size() + fidxs.size() <= (size_t)pop_size) {
                        for (int idx : fidxs) pop.push_back(combined[idx]);
                    } else {
                        // Sort last partial front by crowding distance
                        vector<Individual> last;
                        for (int idx : fidxs) last.push_back(combined[idx]);
                        crowding_sort(last);
                        sort(last.begin(), last.end(),
                             [](const Individual &a, const Individual &b) {
                                 return a.crowd_dist > b.crowd_dist;
                             });
                        int need = pop_size - (int)pop.size();
                        for (int k = 0; k < need; ++k)
                            pop.push_back(last[k]);
                        break;
                    }
                }

                // Progress report
                if ((gen + 1) % max(1, n_generations / 10) == 0 || gen == 0) {
                    // Find best train_loss in population
                    int best_i = 0;
                    for (int i = 1; i < (int)pop.size(); ++i)
                        if (pop[i].obj[0] < pop[best_i].obj[0]) best_i = i;
                    if (mpi.rank == 0)
                        fmt::print(cout, "gen {:>3d} : {}={:15.10f}  obj2={:15.10f}  nsel={:d}\n",
                                   gen + 1, error_cost ? "mean_valid_loss" : "train_loss", pop[best_i].obj[0], pop[best_i].obj[1],
                                   (int)count(pop[best_i].chrom.begin(), pop[best_i].chrom.end(), true));
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // Save a deterministic representative from the final Pareto front.
            // ═══════════════════════════════════════════════════════════════
            if (error_cost) finish_cost_selection("__NSGA2__.mtp", "nsga2");
            else {
            auto final_fronts = nondominated_sort(pop);
            const vector<int> &pareto = final_fronts.front();
            int best_i = pareto.front();
            if (has_valid) {
                auto better_validation_choice = [&](int lhs, int rhs) {
                    if (pop[lhs].obj[1] != pop[rhs].obj[1]) return pop[lhs].obj[1] < pop[rhs].obj[1];
                    if (pop[lhs].obj[0] != pop[rhs].obj[0]) return pop[lhs].obj[0] < pop[rhs].obj[0];
                    const int lhs_size = count(pop[lhs].chrom.begin(), pop[lhs].chrom.end(), true);
                    const int rhs_size = count(pop[rhs].chrom.begin(), pop[rhs].chrom.end(), true);
                    if (lhs_size != rhs_size) return lhs_size < rhs_size;
                    return pop[lhs].chrom < pop[rhs].chrom;
                };
                for (int idx : pareto) if (better_validation_choice(idx, best_i)) best_i = idx;
            } else {
                vector<pair<double, double>> objectives;
                objectives.reserve(pareto.size());
                for (int idx : pareto) objectives.emplace_back(pop[idx].obj[0], pop[idx].obj[1]);
                best_i = pareto[detail::SelectKneePoint(objectives)];
            }

            set<int> best_sel, best_rem;
            for (int i = 0; i < nscalar; ++i) {
                if (pop[best_i].chrom[i]) best_sel.insert(i);
                else                       best_rem.insert(i);
            }

            // Build compact coefficient vector (after remove_scalar_by_id,
            // the potential expects species_count + best_sel.size() coeffs)
            VectorXd x_compact(nspecies + best_sel.size());
            for (int i = 0; i < nspecies; ++i) x_compact(i) = pop[best_i].coeffs(i);
            int col = nspecies;
            for (int si : best_sel) x_compact(col++) = pop[best_i].coeffs(nspecies + si);

            PotMTP pot_best(basepot);
            pot_best.remove_scalar_by_id(best_rem);
            pot_best.set_linear_coeff(x_compact.data());
            pot_best.remove_orphans();
            if (mpi.rank == 0) profile.measure(Profile::Stage::Output, [&] { pot_best.Save("__NSGA2__.mtp"); });

            // Write selection to file for downstream use
            if (mpi.rank == 0) {
                profile.measure(Profile::Stage::Output, [&] { detail::WriteScalarIds("selected_ones_nsga2.dat", best_sel); });
                profile.measure(Profile::Stage::Output, [&] { detail::WriteScalarIds("removed_nsga2.dat", best_rem); });
            }

            double ermse, frmse, srmse, ermse_v, frmse_v, srmse_v, loss_v;
            Check(params, pot_best, atoms, atomsref, atoms_v, atomsref_v,
                  ermse, frmse, srmse, pop[best_i].obj[0],
                  ermse_v, frmse_v, srmse_v, loss_v, Profile::Stage::FinalEFS);
            if (mpi.rank == 0) {
                fmt::print(cout, "{:>6} : {:15.10f} {:15.10f} {:15.10f} {:15.10f} | {:15.10f} {:15.10f} {:15.10f} {:15.10f}\n",
                           "BEST", ermse, frmse, srmse, pop[best_i].obj[0],
                           ermse_v, frmse_v, srmse_v, loss_v);
                Message("Saved __NSGA2__.mtp");
            }
            }
        }
    } else if (params.mode == 7) {
        Message("Basis correlation analysis and de-correlation");
        int nscalar = nl - nspecies;
        if (nscalar < 2) {
            GPERROR("Need at least 2 scalar moments for correlation analysis");
        }

        // ── Phase 1: allocate a global sampling budget and collect locally ──
        int local_site_count = 0;
        for (const Atoms &configuration : atoms) local_site_count += configuration.size();
        vector<int> all_site_counts(mpi.size, 0);
        profile.communicate([&] { MPI_Allgather(&local_site_count, 1, MPI_INT,
                      all_site_counts.data(), 1, MPI_INT, mpi.comm); });
        vector<int> sample_quotas = detail::AllocateSampleQuotas(all_site_counts,
                                                                 params.corr_max_samples);
        BasisAnalysis::DescriptorCollector collector(basepot, sample_quotas[mpi.rank], static_cast<unsigned>(params.seed) + mpi.rank);
        Eigen::MatrixXd desc_local = profile.measure(Profile::Stage::Descriptors, [&] {
            return collector.collect(atoms);
        });
        int nsites_local = desc_local.cols();

        // ── Phase 2: gather all descriptors to rank 0 via MPI ──
        Eigen::MatrixXd desc_global;
        int nsites_global = 0;

        if (mpi.size > 1) {
            vector<int> all_nsites(mpi.size, 0);
            profile.communicate([&] { MPI_Allgather(&nsites_local, 1, MPI_INT,
                          all_nsites.data(), 1, MPI_INT, mpi.comm); });

            // Compute displacements in columns (for matrix assembly)
            std::vector<int> col_displs(mpi.size, 0);
            for (int r = 1; r < mpi.size; ++r) {
                col_displs[r] = col_displs[r - 1] + all_nsites[r - 1];
            }
            nsites_global = col_displs.back() + all_nsites.back();

            int gather_counts_valid = 1;
            const long long local_doubles = static_cast<long long>(nscalar) * nsites_local;
            if (local_doubles > std::numeric_limits<int>::max()) gather_counts_valid = 0;
            for (int r = 0; r < mpi.size; ++r) {
                const long long count = static_cast<long long>(nscalar) * all_nsites[r];
                const long long displacement = static_cast<long long>(nscalar) * col_displs[r];
                if (count > std::numeric_limits<int>::max() ||
                    displacement > std::numeric_limits<int>::max()) gather_counts_valid = 0;
            }
            int all_counts_valid = 0;
            profile.communicate([&] { MPI_Allreduce(&gather_counts_valid, &all_counts_valid, 1, MPI_INT, MPI_MIN, mpi.comm); });
            if (!all_counts_valid) GPERROR("correlation descriptor gather exceeds MPI int count limits");

            vector<int> recvcounts(mpi.size), dbl_displs(mpi.size);
            for (int r = 0; r < mpi.size; ++r) {
                recvcounts[r] = nscalar * all_nsites[r];
                dbl_displs[r] = nscalar * col_displs[r];
            }
            if (mpi.rank == 0) desc_global.resize(nscalar, nsites_global);
            profile.communicate([&] { MPI_Gatherv(desc_local.data(), nscalar * nsites_local, MPI_DOUBLE,
                        desc_global.data(), recvcounts.data(), dbl_displs.data(), MPI_DOUBLE, 0, mpi.comm); });
            desc_local.resize(0, 0);
        } else {
            nsites_global = nsites_local;
            desc_global = std::move(desc_local);
        }

        // ── Phase 3: analysis, selection, and output (rank 0 only) ──
        if (mpi.rank == 0 && nsites_global == 0) {
            Warning("no descriptor data gathered across ranks");
        }
        if (mpi.rank == 0 && nsites_global > 0) {

            // Build strategy
            std::unique_ptr<BasisAnalysis::Decorrelator> strategy;
            if (params.corr_strategy == 0) {
                strategy = std::make_unique<BasisAnalysis::ThresholdPruning>(
                    params.corr_pearson_threshold, params.corr_vif_threshold);
            } else if (params.corr_strategy == 1) {
                strategy = std::make_unique<BasisAnalysis::ClusterRepresentative>(
                    params.corr_pearson_threshold);
            } else if (params.corr_strategy == 2) {
                strategy = std::make_unique<BasisAnalysis::IterativeGreedyVIF>(
                    params.corr_vif_threshold, params.corr_max_features);
            } else {
                GPERROR("Unknown corr_strategy: must be 0 (threshold), 1 (clustering), or 2 (greedy VIF)");
            }
            Message("Strategy: " + strategy->name());

            // Run correlation analysis on the global descriptor matrix
            BasisAnalysis::CorrelationAnalyzer corr_analyzer(desc_global);
            auto corr_result = std::make_unique<BasisAnalysis::CorrelationResult>(
                profile.measure(Profile::Stage::Correlation, [&] { return corr_analyzer.compute_all(); }));

            // Diagnostic report
            BasisAnalysis::DiagnosticReport report;
            report.set_analysis(*corr_result);

            // Apply de-correlation strategy
            auto [selected, removed] = strategy->select(*corr_result, nscalar);

            // Optional DAG preservation
            if (params.corr_addback && !removed.empty()) {
                basepot.add_back_path_scalars_to_remove_id(removed);
                selected.clear();
                for (int i = 0; i < nscalar; ++i) {
                    if (removed.count(i) == 0) selected.insert(i);
                }
            }

            report.add_strategy_result(strategy->name(), selected, removed);

            Message("Total atoms sampled: " + to_string(nsites_global)
                    + " (gathered from " + to_string(mpi.size) + " rank(s))");
            Message("Selected: " + to_string(selected.size()) + " / " + to_string(nscalar) + " scalar moments");
            Message("Removed:  " + to_string(removed.size()));

            // Canonical format: unshifted scalar ids; species are implicit.
            detail::ValidatePartition(selected, removed, nscalar);
            profile.measure(Profile::Stage::Output, [&] { detail::WriteScalarIds("selected_ones.dat", selected); });
            profile.measure(Profile::Stage::Output, [&] { detail::WriteScalarIds("removed.dat", removed); });

            // Write diagnostic report
            profile.measure(Profile::Stage::Output, [&] { report.write_all("correlation", params.corr_addback); });
            Message("Diagnostic report written to correlation_*.dat");

            // Optionally save reduced potential
            if (params.corr_save_reduced) {
                PotMTP pot_reduced(basepot);
                pot_reduced.remove_scalar_by_id(removed);

                std::set<int> reduced_ids;
                for (int i = 0; i < nspecies; ++i) reduced_ids.insert(i);
                for (auto &sid : selected) reduced_ids.insert(nspecies + sid);
                std::vector<int> idxs(reduced_ids.begin(), reduced_ids.end());
                MatrixXd matr;
                VectorXd vecr;
                detail::BuildReducedSystem(matA, vecA, idxs, matr, vecr);
                VectorXd x = profile.measure(Profile::Stage::Solve, [&]() -> VectorXd {
                    const auto scales = Training::equilibrate_linear_system(
                        matr, vecr, vecr.size(), false);
                    Eigen::LDLT<MatrixXd> decomposition(matr.selfadjointView<Eigen::Lower>());
                    if (decomposition.info() != Eigen::Success) {
                        GPERROR("Failed to solve decorrelated normal equations with LDLT");
                    }
                    VectorXd x = decomposition.solve(vecr);
                    for (int i = 0; i < x.size(); ++i) x[i] *= scales[i];
                    if (!x.allFinite()) GPERROR("Decorrelated normal-equation solution is not finite");
                    return x;
                });
                pot_reduced.set_linear_coeff(x.data());

                pot_reduced.remove_orphans();
                profile.measure(Profile::Stage::Output, [&] { pot_reduced.Save("__DECORR__.mtp"); });
                Message("Saved __DECORR__.mtp with " + to_string(selected.size()) + " scalar moments");
            }
        }
    }
    } catch (const std::exception &error) {
        // Root-only modes can fail while other ranks are already finishing.
        // Let every rank reach the profiling collectives before propagating it.
        mode_error = error.what();
        if (!params.profile_enabled && params.profile_output.empty()) throw;
    }
    if (mpi.rank == 0 && (params.mode == 5 || params.mode == 6))
        fmt::print(cout, "Candidate cache: {} solves, {} hits\n", cache_misses, cache_hits);
    selection_timing.stop();
    profile.report(cout, params.profile_output, params.mode, mode_error);
    return 0;
}
int fastfit_cached(int argc, char *argv[], AssemblyCache &cache, bool assemble_only) {
    return fastfit(argc, argv, &cache, assemble_only);
}
int fastfit_observed(int argc, char *argv[], AssemblyCache &cache,
                    BenchmarkObserver &observer, bool assemble_only,bool experimental_square_root) {
    return fastfit(argc, argv, &cache, assemble_only, &observer,experimental_square_root);
}
}

int fastfit(int argc, char *argv[]) {
    return func_fastfit::fastfit(argc, argv);
}
