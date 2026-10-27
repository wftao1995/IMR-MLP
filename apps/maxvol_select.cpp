// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file maxvol_select.cpp
 * @brief MaxVol active-learning commands: select, sample, select_add.
 *
 * Supports multiple descriptor types via --dtype:
 *   mtm          — MTM descriptor (MTP scalar basis, recommended)
 *   pot_efs      — Full coefficient gradient dE/dc (generic, any potential)
 *   behler       — ACSF / wACSF symmetry functions
 *   bondorder    — Steinhardt bond-order parameters
 */

#include "io/gpio.h"
#include "maxvol/active_selector.h"
#include "maxvol/b_matrix_builder.h"
#include "utils/gputils.h"
#include <fmt/format.h>
#include <memory>

using namespace std;
using namespace Eigen;

static void copy_selector_settings(ActiveSelector &dst, const ActiveSelector &src)
{
    dst.threshold = src.threshold;
    dst.weight = src.weight;
    dst.pow_scale = src.pow_scale;
    dst.init_mode = src.init_mode;
    dst.init_value = src.init_value;
    dst.max_swaps = src.max_swaps;
    dst.verbose = src.verbose;
    dst.select_mode = src.select_mode;
    dst.stream_mode = src.stream_mode;
    dst.stream_chunk_size = src.stream_chunk_size;
    dst.stream_fast_passes = src.stream_fast_passes;
    dst.stream_cache_mode = src.stream_cache_mode;
    dst.stream_memory_limit_mb = src.stream_memory_limit_mb;
    dst.stream_cache_dir = src.stream_cache_dir;
    dst.stream_fast_shortlist = src.stream_fast_shortlist;
}

// ═══════════════════════════════════════════════════════════════════════════
// select
// ═══════════════════════════════════════════════════════════════════════════
int maxvol_select(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    GPSettings settings;
    GPIO iobj;
    BMatrixBuilderSettings builder_settings;
    ActiveSelector selector_settings;

    string pool_file   = "pool.xyz";
    string index_file  = "index.map";
    string iformat     = "";
    string oformat     = "";
    string output_file = "selected.xyz";
    string settingfile = "select.ini";
    string score_file  = "";

    cmdopts.addop(pool_file,   "pool",  "", "input structure pool")
           .addop(index_file,  "index", "id", "index map file")
           .addop(iformat,     "input-format", "if", "input format")
           .addop(oformat,     "output-format", "of", "output format")
           .addop(output_file, "output",     "o", "output selected structures")
           .addop(score_file,  "score-file", "", "output per-structure MaxVol scores");
    builder_settings.RegisterConsole(cmdopts);
    selector_settings.RegisterConsole(cmdopts);
    cmdopts.addop(settingfile, "settings", "", "settings file", true)
           .phrase_options(argc, argv);

    if (GPIO::file_exist(settingfile)) {
        settings.Load(settingfile);
        apply_settings_to_cmdopts(cmdopts, settings);
        builder_settings.ApplySettings(settings.ExtractSubSettings("builder"));
        selector_settings.ApplySettings(settings.ExtractSubSettings("selector"));
    }
    if (GPIO::file_exist(index_file)) iobj.readmap(index_file);

    auto builder = make_builder(builder_settings, settings);
    if (!builder) {
        Message("ERROR: failed to create B-matrix builder");
        return 1;
    }

    int ncols  = builder->coeff_count();
    double rcut = builder->cutoff();

    ActiveSelector as(builder.release());
    copy_selector_settings(as, selector_settings);

    Message(fmt::format("dtype={} pottype={} ncols={} cutoff={:.3f} mode={}",
                        builder_settings.dtype, builder_settings.pottype,
                        ncols, rcut, builder_settings.mode));
    Message(fmt::format("Reading pool from: {}", pool_file));
    vector<Atoms> pool;
    if (!iobj.read(pool, pool_file, iformat))
        GPERROR("Failed to read structures: " + pool_file);
    Message(fmt::format("Pool size: {}", pool.size()));
    if (pool.empty()) {
        Message("ERROR: empty pool — nothing to select from");
        return 1;
    }

    int nsel = 0;
    if (as.select_mode == "full") {
        Message("Building B matrix...");
        as.add_candidates_parallel(pool);
        Message(fmt::format("Running MaxVol full/reference (threshold={})", as.threshold));
        nsel = as.select();
    } else if (as.select_mode == "online") {
        Message(fmt::format("Running online MaxVol initialization (threshold={})",
                            as.threshold));
        as.add_candidates_unbuilt(pool);
        nsel = as.select_online();
    } else {
        Message(fmt::format("Running streaming MaxVol (threshold={}, mode={})",
                            as.threshold, as.stream_mode));
        if (mpi.size > 1) {
            as.add_candidates_unbuilt(pool);
            nsel = as.select_streaming_mpi(mpi.comm);
        } else {
            nsel = as.select_streaming(pool);
        }
    }
    Message(fmt::format("Selected {} / {}  log|det|={:.3f}",
                        nsel, pool.size(), as.log_determinant()));
    Message(fmt::format("Selection status: {}  max_grade={:.6g}  virtual_rows={}",
                        as.converged() ? "converged" : "WARNING: not converged",
                        as.max_remaining_grade(), as.virtual_row_count()));
    if (!as.converged())
        Message(fmt::format("WARNING: MaxVol state written with termination reason '{}'",
                            as.termination_reason()));
    if (as.verbose && as.select_mode == "stream") {
        const auto &stats = as.stream_stats();
        Message(fmt::format(
            "Stream stats: cache={} resident={} MiB mapped={} MiB "
            "threads={}/{}/{} builder_calls={} rows={} scans={} swaps={} "
            "build={:.3f}s init={:.3f}s score={:.3f}s total={:.3f}s",
            stats.cache_backend,
            stats.resident_bytes / (1024.0 * 1024.0),
            stats.mapped_bytes / (1024.0 * 1024.0),
            stats.configured_threads, stats.build_threads, stats.score_threads,
            stats.builder_calls, stats.rows_built,
            stats.score_passes, stats.swaps,
            stats.build_seconds, stats.initialization_seconds,
            stats.score_seconds, stats.total_seconds));
    }

    if (mpi.rank == 0 && !score_file.empty()) {
        ofstream sf(score_file);
        sf << "# idx  grade  selected(1=yes)\n";
        vector<double> grades;
        as.compute_grades(grades);
        const auto &si = as.selected_indices();
        set<int> ss(si.begin(), si.end());
        for (size_t i = 0; i < pool.size(); ++i)
            sf << i << " " << grades[i] << " " << (ss.count(i)?1:0) << "\n";
    }

    // Save state so maxvol_sample / maxvol_extend can reuse this active set
    string state_filename = builder_settings.potfile + ".sel";
    if (mpi.rank == 0)
        as.save_state(state_filename, index_file, builder_settings.potfile,
                      builder_settings.pottype);
    if (as.verbose) Message(fmt::format("Active-set state saved to {}", state_filename));

    const auto &sel = as.get_selected();
    if (mpi.rank == 0 && !sel.empty()) {
        vector<Atoms> out(sel.begin(), sel.end());
        if (!iobj.write(out, output_file, oformat))
            GPERROR("Failed to write structures: " + output_file);
        Message(fmt::format("Written {} configs to {}", out.size(), output_file));
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// sample
// ═══════════════════════════════════════════════════════════════════════════
int maxvol_sample(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    GPSettings settings; GPIO iobj;
    BMatrixBuilderSettings builder_settings;
    ActiveSelector selector_settings;

    string input_file = "candidates.xyz";
    string output_file = "sampled.xyz", state_file = "";
    string index_file = "index.map", iformat = "", oformat = "extxyz", settingfile = "sample.ini";

    double thr_save = 2.1, thr_break = 11.0;
    bool add_grade = true;
    bool dump_state = false, selected_only = false;
    string extract_file = "";

    cmdopts.addop(input_file,  "input",      "", "input configuration file")
           .addop(output_file, "output",     "o", "output sampled structures")
           .addop(state_file,  "state-file", "", "active-set state file")
           .addop(index_file,  "index", "id", "index map file")
           .addop(iformat,     "input-format", "if", "input format")
           .addop(oformat,     "output-format", "of", "output format")
           .addop(thr_save,    "threshold-save",  "", "grade above this → save")
           .addop(thr_break,   "threshold-break", "", "grade above this → abort")
           .addop(add_grade,   "add-grade",  "", "add MV_grade feature")
           .addop(dump_state,  "dump-state", "", "print state file summary (no grading)")
           .addop(extract_file,"extract-configs", "", "export configs from state file to given path")
           .addop(selected_only,"selected-only","","export only selected configs (with --extract-configs)");
    builder_settings.RegisterConsole(cmdopts);
    selector_settings.RegisterConsole(cmdopts);
    cmdopts.addop(settingfile, "settings", "", "settings file", true)
           .phrase_options(argc, argv);

    if (GPIO::file_exist(settingfile)) {
        settings.Load(settingfile);
        apply_settings_to_cmdopts(cmdopts, settings);
        builder_settings.ApplySettings(settings.ExtractSubSettings("builder"));
        selector_settings.ApplySettings(settings.ExtractSubSettings("selector"));
    }
    if (GPIO::file_exist(index_file)) iobj.readmap(index_file);
    if (state_file.empty()) state_file = builder_settings.potfile + ".sel";

    // ── Dump / extract mode (no builder or candidates needed) ──────────
    if (dump_state) {
        ActiveSelector::dump_state(state_file, std::cout);
        return 0;
    }
    if (!extract_file.empty()) {
        ActiveSelector::export_configs(state_file, extract_file, selected_only, index_file);
        Message(fmt::format("Configs exported to {}", extract_file));
        return 0;
    }

    // ── Read candidates early; skip expensive builder construction if empty ──
    vector<Atoms> cfgs;
    if (!iobj.read(cfgs, input_file, iformat))
        GPERROR("Failed to read structures: " + input_file);
    if (cfgs.empty()) {
        Message(fmt::format("No candidates found in {}", input_file));
        return 0;
    }
    Message(fmt::format("Candidates: {}", cfgs.size()));

    auto builder = make_builder(builder_settings, settings);
    if (!builder) {
        Message("ERROR: failed to create B-matrix builder");
        return 1;
    }
    ActiveSelector as(builder.release());
    copy_selector_settings(as, selector_settings);

    Message(fmt::format("Loading state from {}", state_file));
    as.load_state(state_file, index_file);
    if (!as.converged())
        Message(fmt::format("WARNING: grading with non-converged state "
                            "(reason={}, max_grade={:.6g})",
                            as.termination_reason(), as.max_remaining_grade()));

    if (!as.prepare_grading()) {
        Message("ERROR: failed to prepare MaxVol grading state");
        return 1;
    }

    // ── Grading ────────────────────────────────────────────────────────
    std::vector<double> grades(cfgs.size(), -1.0);
    as.grade_all(cfgs, grades);

    // grades[i] already corresponds to cfgs[i] — both in original order.
    // Parallel threads write grades[i] for distinct i ranges; no reordering needed.

    // ── Output ──────────────────────────────────────────────────────────
    int ns = 0;
    ofstream ofs(output_file, ios::binary);
    if (!ofs) GPERROR("Cannot open output file: " + output_file);
    for (size_t i = 0; i < cfgs.size(); ++i) {
        double g = grades[i];
        if (g > thr_break) {
            Message(fmt::format("BREAK: cfg {} grade {:.3f} > {} — stopping",
                                i, g, thr_break));
            break;
        }
        if (g > thr_save) {
            if (add_grade) cfgs[i].features_["MV_grade"] = fmt::format("{:.6f}", g);
            if (!iobj.write_one(cfgs[i], ofs, oformat))
                GPERROR("Failed to write structures: " + output_file);
            ++ns;
        }
    }
    ofs.close();
    if (!ofs) GPERROR("Failed to write structures: " + output_file);
    Message(fmt::format("Saved {} / {} (grade > {})", ns, cfgs.size(), thr_save));
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// select_add
// ═══════════════════════════════════════════════════════════════════════════
int maxvol_extend(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    GPSettings settings; GPIO iobj;
    BMatrixBuilderSettings builder_settings;
    ActiveSelector selector_settings;

    string train_file = "";
    string state_file = "", candidates_file = "candidates.xyz";
    string output_file = "new_selected.xyz";
    string index_file = "index.map";
    string iformat = "", oformat = "extxyz", settingfile = "select_add.ini";

    cmdopts.addop(state_file,     "state-file", "", "active-set state file (skip Stage 1)")
           .addop(train_file,     "train",      "", "training set file")
           .addop(candidates_file,"candidates", "", "candidate structures file")
           .addop(output_file,    "output",     "o", "output newly selected structures")
           .addop(index_file,     "index", "id", "index map file")
           .addop(iformat,        "input-format",    "if", "input format")
           .addop(oformat,        "output-format",   "of", "output format");
    builder_settings.RegisterConsole(cmdopts);
    selector_settings.RegisterConsole(cmdopts);
    cmdopts.addop(settingfile, "settings", "", "settings file", true)
           .phrase_options(argc, argv);

    if (GPIO::file_exist(settingfile)) {
        settings.Load(settingfile);
        apply_settings_to_cmdopts(cmdopts, settings);
        builder_settings.ApplySettings(settings.ExtractSubSettings("builder"));
        selector_settings.ApplySettings(settings.ExtractSubSettings("selector"));
    }
    if (GPIO::file_exist(index_file)) iobj.readmap(index_file);

    // ── Read candidates early; skip expensive builder construction if empty ──
    vector<Atoms> candidates;
    if (!iobj.read(candidates, candidates_file, iformat))
        GPERROR("Failed to read structures: " + candidates_file);
    if (candidates.empty()) {
        Message(fmt::format("No candidates found in {}", candidates_file));
        return 0;
    }
    for (auto &c : candidates) c.features_[ActiveSelector::kNewTag] = "1";

    auto builder = make_builder(builder_settings, settings);
    if (!builder) {
        Message("ERROR: failed to create B-matrix builder");
        return 1;
    }
    ActiveSelector as(builder.release());
    copy_selector_settings(as, selector_settings);

    if (!state_file.empty()) {
        // ── State-file branch: load existing active set, then select on
        //     training ∪ candidates in a single pass ──────────────────
        Message(fmt::format("Loading active set from {}", state_file));
        as.load_state(state_file, index_file);
        if (!as.converged())
            Message(fmt::format("WARNING: extending non-converged state "
                                "(reason={}, max_grade={:.6g})",
                                as.termination_reason(), as.max_remaining_grade()));

        if (!train_file.empty()) {
            Message(fmt::format("Adding extra training from {}", train_file));
            vector<Atoms> extra_train;
            if (!iobj.read(extra_train, train_file, iformat))
                GPERROR("Failed to read structures: " + train_file);
            if (as.select_mode == "full")
                as.add_candidates_parallel(extra_train);
            else
                as.add_candidates_unbuilt(extra_train);
        }

        const int candidate_start = as.num_candidates();
        Message(fmt::format("Adding {} candidates", candidates.size()));
        if (as.select_mode == "full") {
            as.rebuild_B_from_configs();
            as.add_candidates_parallel(candidates);
        } else {
            as.add_candidates_unbuilt(candidates);
        }

        Message(fmt::format("Running MaxVol (threshold={})", as.threshold));
        int n = 0;
        if (as.select_mode == "full")
            n = as.select(as.threshold);
        else if (as.select_mode == "online")
            n = as.select_online();
        else
            n = as.select_streaming();
        Message(fmt::format("Active set: {} configs  log|det|={:.3f}",
                            n, as.log_determinant()));

        vector<Atoms> new_sel;
        const auto &selected = as.get_selected();
        const auto &selected_indices = as.selected_indices();
        for (size_t i = 0; i < selected.size() && i < selected_indices.size(); ++i) {
            const int idx = selected_indices[i];
            if (idx >= candidate_start &&
                idx < candidate_start + static_cast<int>(candidates.size()) &&
                selected[i].size() > 0) {
                Atoms out = selected[i];
                out.features_.erase(ActiveSelector::kNewTag);
                new_sel.push_back(out);
            }
        }
        Message(fmt::format("Newly selected from candidates: {}", new_sel.size()));
        if (!new_sel.empty() && !iobj.write(new_sel, output_file, oformat))
            GPERROR("Failed to write structures: " + output_file);
        as.save_state(state_file, index_file, builder_settings.potfile,
                      builder_settings.pottype);

    } else {
        // ── Original 3-stage logic (no state file) ─────────────────────────
        if (train_file.empty()) {
            Message("ERROR: --train is required when --state-file is not given");
            return 1;
        }

        Message(fmt::format("Stage 1: building active set from {}", train_file));
        vector<Atoms> train;
        if (!iobj.read(train, train_file, iformat))
            GPERROR("Failed to read structures: " + train_file);

        auto train_builder = make_builder(builder_settings, settings);
        if (!train_builder) {
            Message("ERROR: failed to create training B-matrix builder");
            return 1;
        }
        ActiveSelector train_as(train_builder.release());
        copy_selector_settings(train_as, selector_settings);

        if (as.select_mode == "full") {
            train_as.select_mode = "full";
            train_as.add_candidates_parallel(train);
            int n1 = train_as.select(1.001);
            Message(fmt::format("Selected {} from training set", n1));

            Message(fmt::format("Stage 2: evaluating {}", candidates_file));
            as.add_candidates_parallel(candidates);
            int n2 = as.select(as.threshold);
            Message(fmt::format("Selected {} new configs", n2));

            Message("Stage 3: re-selecting training set");
            if (!as.append_built_candidates_from(train_as)) {
                Message("ERROR: failed to append prebuilt training B matrix");
                return 1;
            }
            as.select(1.001);
        } else if (as.select_mode == "online") {
            Message("Online selection on training plus candidates");
            vector<Atoms> combined = train;
            combined.insert(combined.end(), candidates.begin(), candidates.end());
            as.add_candidates_unbuilt(combined);
            as.select_online();
        } else {
            Message("Streaming selection on training plus candidates");
            vector<Atoms> combined = train;
            combined.insert(combined.end(), candidates.begin(), candidates.end());
            as.select_streaming(combined);
        }

        vector<Atoms> new_sel;
        for (const auto &cfg : as.get_selected()) {
            auto it = cfg.features_.find(ActiveSelector::kNewTag);
            if (it != cfg.features_.end() && it->second == "1" && cfg.size() > 0) {
                Atoms out = cfg;
                out.features_.erase(ActiveSelector::kNewTag);
                new_sel.push_back(out);
            }
        }
        Message(fmt::format("Newly selected: {} configs  log|det|={:.3f}",
                            new_sel.size(), as.log_determinant()));
        if (!new_sel.empty() && !iobj.write(new_sel, output_file, oformat))
            GPERROR("Failed to write structures: " + output_file);
    }
    return 0;
}
