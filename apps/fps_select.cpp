// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file fps_select.cpp
 * @brief Diversity-based representative structure selection.
 *
 * Two methods via --method:
 *   fps  — Farthest Point Sampling (geometric diversity, default)
 *   cur  — CUR decomposition (SVD leverage-score sampling)
 *
 * Supports multiple descriptor types via --dtype:
 *   mtm          — MTM descriptor (MTP scalar basis)
 *   pot_efs      — Full coefficient gradient dE/dc
 *   behler       — ACSF / wACSF symmetry functions
 *   bondorder    — Steinhardt bond-order parameters
 */

#include "io/gpio.h"
#include "maxvol/fps_selector.h"
#include "maxvol/b_matrix_builder.h"
#include "maxvol/cur_selection.h"
#include "utils/gputils.h"
#include <fmt/format.h>
#include <memory>
#include <numeric>
#include <set>

using namespace std;
using namespace Eigen;

// ═══════════════════════════════════════════════════════════════════════════════
// diverse_select — unified diversity-based structure selection (FPS + CUR)
// ═══════════════════════════════════════════════════════════════════════════════
int diverse_select(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    GPSettings settings;
    GPIO iobj;
    BMatrixBuilderSettings builder_settings;

    string pool_file   = "pool.xyz";
    string index_file  = "index.map";
    string format      = "";
    string output_file = "selected.xyz";
    string settingfile = "diverse_select.ini";
    string score_file  = "";
    string method      = "fps";         // fps | cur
    string metric_str  = "euclidean";   // FPS only

    // Shared
    int    n_select     = 0;
    int    seed         = 42;
    bool   verbose      = false;
    builder_settings.mode = "cfg";        // structure-level descriptors

    // FPS-specific
    double dist_thr     = 0.0;
    int    first_idx    = -1;

    // CUR-specific
    bool   cur_deterministic = false;
    bool   compute_error = false;

    cmdopts.addop(pool_file,   "pool",  "", "input structure pool")
           .addop(index_file,  "index", "id", "index map file")
           .addop(format,      "format",     "f", "input format")
           .addop(output_file, "output",     "o", "output selected structures")
           .addop(score_file,  "score-file", "", "output per-structure scores")
           .addop(method,      "method",     "", "selection method: fps (default) | cur")
           .addop(n_select,    "n-select",   "n", "number of structures to select")
           // FPS-specific
           .addop(dist_thr,    "threshold-distance", "", "FPS: stop when min-distance drops below this")
           .addop(first_idx,   "first-idx",  "", "FPS: index of first selected structure (-1 = random)")
           .addop(metric_str,  "metric",     "", "FPS: distance metric — euclidean (default) | cosine")
           // CUR-specific
           .addop(cur_deterministic, "cur-deterministic", "", "CUR: use deterministic top-k selection")
           .addop(compute_error,     "compute-error",     "", "CUR: compute reconstruction error")
           // General
           .addop(seed,        "seed",       "", "random seed (default 42)")
           .addop(verbose,     "verbose",    "v", "verbose output", false, 0);
    builder_settings.RegisterConsole(cmdopts);
    cmdopts.addop(settingfile, "settings", "", "settings file", true)
           .phrase_options(argc, argv);

    if (GPIO::file_exist(settingfile)) {
        settings.Load(settingfile);
        apply_settings_to_cmdopts(cmdopts, settings);
        builder_settings.ApplySettings(settings.ExtractSubSettings("builder"));
    }
    if (GPIO::file_exist(index_file)) iobj.readmap(index_file);

    // ═══════════════════════════════════════════════════════════════════════
    // CUR path
    // ═══════════════════════════════════════════════════════════════════════
    if (method == "cur") {
        if (n_select <= 0) {
            Message("ERROR: --n-select is required for CUR");
            return 1;
        }

        BMatrixBuilderSettings cur_settings;
        cur_settings.dtype = builder_settings.dtype;
        cur_settings.potfile = builder_settings.potfile;
        cur_settings.pottype = builder_settings.pottype;
        cur_settings.all_coeffs = builder_settings.all_coeffs;
        cur_settings.row_scaling = builder_settings.row_scaling;
        cur_settings.w_energy = builder_settings.w_energy;
        cur_settings.w_site = builder_settings.w_site;
        cur_settings.w_force = builder_settings.w_force;
        cur_settings.w_stress = builder_settings.w_stress;
        cur_settings.mode = "cfg";
        auto cur_builder = make_builder(cur_settings, settings);
        if (!cur_builder) { Message("ERROR: failed to create builder"); return 1; }

        int ncols  = cur_builder->coeff_count();
        if (ncols <= 0)
            GPERROR(fmt::format("CUR requires at least one descriptor feature (got {})", ncols));
        double rcut = cur_builder->cutoff();

        Message(fmt::format("dtype={} ncols={} cutoff={:.3f} method=cur n_select={}",
                            cur_settings.dtype, ncols, rcut, n_select));
        Message(fmt::format("Reading pool from: {}", pool_file));
        vector<Atoms> pool;
        if (!iobj.read(pool, pool_file, format))
            GPERROR("Failed to read complete candidate pool: " + pool_file);
        Message(fmt::format("Pool size: {}", pool.size()));
        if (pool.empty()) { Message("ERROR: empty pool"); return 1; }

        Message("Building descriptor matrix...");
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> D(pool.size(), ncols);
        for (size_t i = 0; i < pool.size(); ++i) {
            Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_cfg;
            int nr = cur_builder->build_rows(pool[i], B_cfg, 1.0, 0.5);
            if (nr <= 0) {
                D.row(i).setZero();
            } else if (nr == 1) {
                D.row(i) = B_cfg.row(0);
            } else {
                D.row(i) = B_cfg.topRows(nr).colwise().mean();
            }
        }

        int ns = std::min(n_select, static_cast<int>(pool.size()));
        Message(fmt::format("Running CUR ({} × {}, selecting {})...",
                            D.rows(), D.cols(), ns));

        CURResult result = cur_select_rows(D, ns, cur_deterministic,
                                           seed, compute_error);

        if (!score_file.empty()) {
            ofstream sf(score_file);
            if (!sf) GPERROR("Cannot open score file: " + score_file);
            sf << "# idx  leverage_score  selected(1=yes)\n";
            std::set<int> ss(result.selected_indices.begin(),
                             result.selected_indices.end());
            for (size_t i = 0; i < pool.size(); ++i)
                sf << i << " " << result.scores[i] << " "
                   << (ss.count(static_cast<int>(i)) ? 1 : 0) << "\n";
            sf.close();
            if (!sf) GPERROR("Failed to write scores: " + score_file);
            Message(fmt::format("Scores written to {}", score_file));
        }

        if (compute_error && result.reconstruction_error >= 0)
            Message(fmt::format("Relative reconstruction error: {:.6f}",
                                result.reconstruction_error));

        double total_score = std::accumulate(result.scores.begin(),
                                             result.scores.end(), 0.0);
        double sel_score = 0;
        for (int idx : result.selected_indices)
            sel_score += result.scores[idx];
        Message(fmt::format("Selected {} / {}  leverage captured: {:.1f}%",
                            result.selected_indices.size(), pool.size(),
                            100.0 * sel_score / (total_score + 1e-16)));

        const auto &sel_idx = result.selected_indices;
        if (!sel_idx.empty()) {
            vector<Atoms> out;
            out.reserve(sel_idx.size());
            for (int idx : sel_idx)
                out.push_back(pool[idx]);
            if (!iobj.write(out, output_file, ""))
                GPERROR("Failed to write structures: " + output_file);
            Message(fmt::format("Written {} configs to {}", out.size(), output_file));
        }

        return 0;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FPS path (default)
    // ═══════════════════════════════════════════════════════════════════════
    if (n_select <= 0 && dist_thr <= 0.0) {
        Message("ERROR: specify --n-select or --threshold-distance (or both)");
        return 1;
    }

    auto builder = make_builder(builder_settings, settings);
    if (!builder) { Message("ERROR: failed to create builder"); return 1; }

    int ncols  = builder->coeff_count();
    double rcut = builder->cutoff();

    FPSSelector fps(builder.release());
    if (metric_str == "cosine") fps.metric = FPSSelector::Cosine;
    fps.verbose = verbose;
    fps.random_seed = seed;

    Message(fmt::format("dtype={} pottype={} ncols={} cutoff={:.3f} method=fps metric={}",
                        builder_settings.dtype, builder_settings.pottype,
                        ncols, rcut, metric_str));
    Message(fmt::format("Reading pool from: {}", pool_file));
    vector<Atoms> pool;
    if (!iobj.read(pool, pool_file, format))
        GPERROR("Failed to read complete candidate pool: " + pool_file);
    Message(fmt::format("Pool size: {}", pool.size()));
    if (pool.empty()) {
        Message("ERROR: empty pool — nothing to select from");
        return 1;
    }

    if (pool.empty()) { Message("ERROR: empty pool"); return 1; }

    Message("Building descriptor matrix...");
    fps.add_candidates(pool);

    int nsel = 0;
    if (dist_thr > 0.0) {
        nsel = fps.select_by_threshold(dist_thr, n_select > 0 ? n_select : 0, first_idx);
        Message(fmt::format("FPS threshold: threshold={:.4f}, selected {} / {}",
                            dist_thr, nsel, pool.size()));
    } else {
        int actual_n = min(n_select, static_cast<int>(pool.size()));
        nsel = fps.select(actual_n, first_idx);
        Message(fmt::format("FPS select: requested {}, selected {} / {}",
                            n_select, nsel, pool.size()));
    }

    if (!score_file.empty() && nsel > 0) {
        ofstream sf(score_file);
        if (!sf) GPERROR("Cannot open score file: " + score_file);
        sf << "# idx  FPS_order  FPS_score  selected(1=yes)\n";
        const auto &si = fps.selected_indices();
        const auto &scores = fps.selection_scores();
        const auto &order = fps.selection_order();

        map<int, pair<int, double>> sel_info;
        for (size_t k = 0; k < si.size(); ++k)
            sel_info[si[k]] = {order[k], scores[k]};

        set<int> ss(si.begin(), si.end());
        for (size_t i = 0; i < pool.size(); ++i) {
            int ord = -1;
            double sc = -1.0;
            auto it = sel_info.find(static_cast<int>(i));
            if (it != sel_info.end()) { ord = it->second.first; sc = it->second.second; }
            sf << i << " " << ord << " " << sc << " " << (ss.count(i) ? 1 : 0) << "\n";
        }
        sf.close();
        if (!sf) GPERROR("Failed to write scores: " + score_file);
        Message(fmt::format("Scores written to {}", score_file));
    }

    const auto &sel = fps.get_selected();
    if (!sel.empty()) {
        const auto &si = fps.selected_indices();
        const auto &scores = fps.selection_scores();
        const auto &order = fps.selection_order();

        vector<Atoms> out;
        for (size_t k = 0; k < sel.size(); ++k) {
            Atoms cfg = sel[k];
            cfg.features_["FPS_order"] = fmt::format("{}", order[k]);
            cfg.features_["FPS_score"] = fmt::format("{:.6f}", scores[k]);
            cfg.features_["FPS_seed"] = (k == 0) ? "1" : "0";
            out.push_back(cfg);
        }
        if (!iobj.write(out, output_file, ""))
            GPERROR("Failed to write structures: " + output_file);
        Message(fmt::format("Written {} configs to {}", out.size(), output_file));
    }

    return 0;
}
