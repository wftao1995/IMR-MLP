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
#include "persistence_mode.h"
#include "maxvol/active_selector.h"
#include "maxvol/b_matrix_builder.h"
#include "utils/gputils.h"
#include <fmt/format.h>
#include <memory>
#include "potentials/model_bundle.h"
#include "potentials/model_elements.h"
#include "potentials/potential_wrapper.h"
#include "train/collective_error.h"
#include "utils/compressed_stream.h"
#include <filesystem>
#include <unistd.h>

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
bool maxvol_gpm(const string &path)
{
    return !path.empty() && (GPM::is_file(path) || GPM::extension(path));
}

bool maxvol_model_builder(const string &dtype)
{
    return dtype == "mtm" || dtype == "mtp" || dtype == "mtp_scalar" || dtype == "pot_efs" ||
           dtype == "mtp_energy";
}

string maxvol_default_state(const BMatrixBuilderSettings &builder, bool legacy = false)
{
    return builder.potfile +
           (!legacy && maxvol_model_builder(builder.dtype) ? ".maxvol.gpm" : ".sel");
}

bool maxvol_has_embedded_state(const string &path)
{
    if (!maxvol_gpm(path))
        return false;
    const auto manifest = Potentials::ModelBundle::inspect(path);
    if (!manifest.count("active_learning"))
        return false;
    GPM::require(GPM::string(GPM::at(GPM::object(manifest.at("active_learning")), "method")) ==
                     "maxvol",
                 "GPM contains a different AL method, not MaxVol");
    return true;
}

bool maxvol_explicit(const GPSettings &settings, const string &key, const string &group = "builder")
{
    return CMDOption::global_cmdopts.was_provided(key) || settings.count(key) ||
           settings.count(group + ":" + key);
}

void maxvol_species(GPIO &io, const string &path, const string &index, const GPSettings &settings)
{
    GPSettings model;
    model.Modify("potfile", path);
    model.Modify("pottype", "auto");
    const auto names = Potentials::probe_model_elements(model);
    const bool explicit_index = maxvol_explicit(settings, "index", "");
    if (!names.empty())
    {
        GPIO declared;
        declared.setmap(names);
        if (explicit_index)
        {
            io.install_species_map(index, true, names, false);
            GPM::require(io.getmap() == declared.getmap(),
                         "index map conflicts with model species order");
        }
        else
            io.setmap(names);
    }
    else if (GPIO::file_exist(index))
        io.readmap(index);
}

unique_ptr<ActiveSelector> maxvol_load(const string &state, const BMatrixBuilderSettings &bs,
                                       const ActiveSelector &runtime, const GPSettings &settings,
                                       const string &index, string *identity = nullptr)
{
    unique_ptr<ActiveSelector> result;
    if (!GPIO::file_exist(state))
        GPERROR("No MaxVol state found; run maxvol_select or specify --state-file");
    if (!maxvol_gpm(state))
    {
        auto builder = make_builder(bs, const_cast<GPSettings &>(settings));
        if (!builder)
            GPERROR("Failed to create B-matrix builder");
        if (identity)
        {
            ostringstream value;
            builder->save_config(value);
            Potentials::AnyPotential *model = nullptr;
            if (auto *b = dynamic_cast<PotEFSBuilder *>(builder.get()))
                model = &b->potential();
            if (auto *b = dynamic_cast<DescriptorBuilder *>(builder.get()))
                model = b->target_potential();
            if (model)
                value << Potentials::ModelBundle::fingerprint(*model);
            *identity = value.str();
        }
        result = make_unique<ActiveSelector>(builder.release());
        copy_selector_settings(*result, runtime);
        result->load_state(state, index);
        return result;
    }
    result = make_unique<ActiveSelector>();
    result->load_state(state, maxvol_explicit(settings, "index", "") ? index : "");
    const auto &pb = result->builder_parameters();
    if (maxvol_explicit(settings, "potfile"))
    {
        auto options = settings.ExtractSubSettings("pot");
        options.Modify("potfile", bs.potfile);
        options.Modify("pottype", maxvol_explicit(settings, "pottype") ? bs.pottype : "auto");
        Potentials::potential_wrapper explicit_model(options);
        if (!maxvol_gpm(bs.potfile))
        {
            GPIO mapping;
            maxvol_species(mapping, state, index, settings);
            explicit_model.pot->set_element_map(mapping.getmap());
            if (pb.dtype == "pot_efs" || pb.dtype == "mtp_energy")
                explicit_model.pot->train_init();
        }
        GPM::require(Potentials::ModelBundle::fingerprint(*explicit_model.pot) ==
                         Potentials::ModelBundle::fingerprint(*result->state_potential()),
                     "explicit potential differs from GPM MaxVol target");
    }
    if (maxvol_explicit(settings, "pottype") && bs.pottype != "auto")
        GPM::require(bs.pottype == Potentials::ModelBundle::type(*result->state_potential()),
                     "pottype conflicts with GPM target");
    auto check = [&](const string &key, const auto &requested, const auto &stored,
                     const string &group = "builder")
    {
        if (maxvol_explicit(settings, key, group))
            GPM::require(requested == stored,
                         "explicit " + group + ":" + key +
                             " conflicts with embedded MaxVol state; rebuild selection");
    };
    check("dtype", bs.dtype, pb.dtype);
    if (pb.dtype == "mtm" || pb.dtype == "mtp" || pb.dtype == "mtp_scalar")
        check("mode", bs.mode, pb.mode);
    check("w-energy", bs.w_energy, pb.w_energy);
    check("w-site", bs.w_site, pb.w_site);
    check("w-force", bs.w_force, pb.w_force);
    check("w-stress", bs.w_stress, pb.w_stress);
    check("all-coeffs", bs.all_coeffs, !pb.linear_only);
    check("row-scaling", bs.row_scaling, string(to_string(pb.row_scaling)));
#define CHECK(field, key) check(key, runtime.field, result->field, "selector")
    CHECK(weight, "weight");
    CHECK(pow_scale, "pow-scale");
    CHECK(init_value, "init-value");
    CHECK(select_mode, "select-mode");
    CHECK(stream_mode, "stream-mode");
    CHECK(threshold, "threshold");
    CHECK(max_swaps, "max-swaps");
    CHECK(stream_chunk_size, "stream-chunk-size");
    CHECK(stream_fast_passes, "stream-fast-passes");
    CHECK(stream_fast_shortlist, "stream-fast-shortlist");
#undef CHECK
    if (maxvol_explicit(settings, "init-mode", "selector") && runtime.init_mode != "auto")
    {
        const string effective =
            result->init_mode == "auto"
                ? (pb.dtype == "pot_efs" || pb.dtype == "mtp_energy" ? "mlip" : "data")
                : result->init_mode;
        GPM::require(runtime.init_mode == effective,
                     "explicit init-mode conflicts with embedded MaxVol state");
    }
    result->verbose = runtime.verbose;
    if (maxvol_explicit(settings, "stream-memory-limit-mb", "selector"))
        result->stream_memory_limit_mb = runtime.stream_memory_limit_mb;
    if (maxvol_explicit(settings, "stream-cache-mode", "selector"))
        result->stream_cache_mode = runtime.stream_cache_mode;
    if (maxvol_explicit(settings, "stream-cache-dir", "selector"))
        result->stream_cache_dir = runtime.stream_cache_dir;
    return result;
}

struct MaxVolStagedOutput
{
    filesystem::path directory, temporary, destination;
    bool preserve_backup = false;

    explicit MaxVolStagedOutput(const string &path) : destination(path)
    {
        auto parent = destination.parent_path();
        if (parent.empty())
            parent = ".";
        if (destination.filename().empty())
            GPERROR("MaxVol output requires a file basename: " + path);
        string name = (parent / ".gpml-maxvol-output-XXXXXX").string();
        vector<char> writable(name.begin(), name.end());
        writable.push_back('\0');
        if (!mkdtemp(writable.data()))
            GPERROR("Cannot stage MaxVol output: " + path);
        directory = writable.data();
        try
        {
            filesystem::create_directory(directory / "new");
            filesystem::create_directory(directory / "backup");
            // Keep the complete basename, including compression suffixes, and
            // do not create it: NetCDF newFile requires an absent destination.
            temporary = directory / "new" / destination.filename();
        }
        catch (...)
        {
            error_code ignored;
            filesystem::remove_all(directory, ignored);
            throw;
        }
    }

    ~MaxVolStagedOutput()
    {
        if (!preserve_backup)
        {
            error_code ignored;
            filesystem::remove_all(directory, ignored);
        }
    }
};

void publish_maxvol_selection_outputs(vector<unique_ptr<MaxVolStagedOutput>> &outputs)
{
    struct File
    {
        filesystem::path source, destination, backup;
        MaxVolStagedOutput *owner;
        bool published = false, backed_up = false;
    };

    vector<File> files;
    set<filesystem::path> destinations;
    for (const auto &output : outputs)
    {
        auto parent = output->destination.parent_path();
        if (parent.empty())
            parent = ".";
        size_t generated = 0;
        for (const auto &entry : filesystem::directory_iterator(output->directory / "new"))
        {
            if (!entry.is_regular_file() || entry.is_symlink())
                GPERROR("Unexpected staged MaxVol output type");
            const auto destination = parent / entry.path().filename();
            const auto canonical_destination =
                filesystem::weakly_canonical(parent) / destination.filename();
            if (!destinations.insert(canonical_destination).second)
                GPERROR("Generated MaxVol output files overlap: " + destination.string());
            const auto status = filesystem::symlink_status(destination);
            if (filesystem::exists(status) && !filesystem::is_regular_file(status) &&
                !filesystem::is_symlink(status))
                GPERROR("MaxVol output destination is not a regular file: " + destination.string());
            files.push_back({entry.path(), destination,
                             output->directory / "backup" / entry.path().filename(), output.get()});
            ++generated;
        }
        if (!generated)
            GPERROR("MaxVol writer produced no output files: " + output->destination.string());
    }
    sort(files.begin(), files.end(),
         [](const File &a, const File &b)
         {
             return a.destination.string() < b.destination.string();
         });
    // From the first backup rename onward, a staging directory may contain
    // the only surviving copy of an old output. Keep every owner alive across
    // unexpected rollback exceptions, including allocation failures while
    // formatting diagnostics; only a completely successful path may clean it.
    for (auto &output : outputs)
        output->preserve_backup = true;
    try
    {
        for (auto &file : files)
            if (filesystem::exists(filesystem::symlink_status(file.destination)))
            {
                filesystem::rename(file.destination, file.backup);
                file.backed_up = true;
            }
        for (auto &file : files)
        {
            filesystem::rename(file.source, file.destination);
            file.published = true;
        }
    }
    catch (...)
    {
        const auto failure = current_exception();
        string restore_errors;
        for (size_t i = files.size(); i-- > 0;)
        {
            auto &file = files[i];
            error_code error;
            if (file.published)
                filesystem::remove(file.destination, error);
            if (file.backed_up)
                filesystem::rename(file.backup, file.destination, error);
            if (error)
            {
                file.owner->preserve_backup = true;
                restore_errors += " " + file.owner->directory.string() + ": " + error.message();
            }
        }
        if (!restore_errors.empty())
            GPERROR("Cannot restore all MaxVol outputs; recovery backups retained at" +
                    restore_errors);
        for (auto &output : outputs)
            output->preserve_backup = false;
        rethrow_exception(failure);
    }
    for (auto &output : outputs)
        output->preserve_backup = false;
}

// A valid MLIP active set may contain only virtual rows. Publish an empty
// selection explicitly: GPIO's filename overload otherwise leaves an old
// destination untouched and creates no file for transactional publication.
bool write_maxvol_selection(GPIO &io, vector<Atoms> &selected, const string &path,
                            const string &format)
{
    if (!selected.empty())
        return io.write(selected, path, format);
    const auto output_format =
        format.empty() ? GPIO::guess_format(path) : GPIO::format2enum(format);
    if (output_format == GPIO::UNKNOWN)
        GPERROR("Unknown MaxVol selection output format");
    auto empty = open_ofile(path, false, true);
    return empty && close_ofile(empty);
}

static void publish_maxvol_extension(ActiveSelector &as, GPIO &io, vector<Atoms> &selected,
                                     const string &output, const string &format,
                                     const string &state, const string &index,
                                     const BMatrixBuilderSettings &builder, bool legacy = false)
{
    Training::collective_try(
        [&]
        {
            if (mpi.rank != 0)
                return;
            if (legacy)
            {
                if (!selected.empty() && !io.write(selected, output, format))
                    GPERROR("Failed to write structures: " + output);
                if (!state.empty())
                    as.save_state(state, index, builder.potfile, builder.pottype);
                return;
            }
            if (filesystem::absolute(state).lexically_normal() ==
                filesystem::absolute(output).lexically_normal())
                GPERROR("MaxVol extension output and state paths must differ");
            vector<unique_ptr<MaxVolStagedOutput>> outputs;
            outputs.push_back(make_unique<MaxVolStagedOutput>(output));
            outputs.push_back(make_unique<MaxVolStagedOutput>(state));
            if (!write_maxvol_selection(io, selected, outputs[0]->temporary.string(), format))
                GPERROR("Cannot stage selected structures: " + output);
            as.save_state(outputs[1]->temporary.string(), index, builder.potfile, builder.pottype);
            publish_maxvol_selection_outputs(outputs);
        });
}

static void publish_legacy_maxvol_selection(ActiveSelector &as, GPIO &io, const string &output,
                                            const string &format, const string &state,
                                            const string &index,
                                            const BMatrixBuilderSettings &builder,
                                            const string &scores)
{
    if (!scores.empty())
    {
        ofstream stream(scores);
        vector<double> grades;
        as.compute_grades(grades);
        set<int> ids(as.selected_indices().begin(), as.selected_indices().end());
        stream << "# idx  grade  selected(1=yes)\n";
        for (size_t i = 0; i < grades.size(); ++i)
            stream << i << " " << grades[i] << " " << (ids.count(i) ? 1 : 0) << "\n";
        stream.close();
        if (!stream)
            GPERROR("Cannot write MaxVol scores: " + scores);
    }
    as.save_state(state, index, builder.potfile, builder.pottype);
    vector<Atoms> selected(as.get_selected().begin(), as.get_selected().end());
    if (!selected.empty() && !io.write(selected, output, format))
        GPERROR("Failed to write structures: " + output);
}

int maxvol_select(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    PersistenceMode persistence;
    persistence.RegisterConsole(cmdopts);
    GPSettings settings;
    GPIO iobj;
    BMatrixBuilderSettings builder_settings;
    ActiveSelector selector_settings;

    string pool_file = "pool.xyz";
    string index_file = "index.map";
    string iformat = "";
    string oformat = "";
    string output_file = "selected.xyz";
    string settingfile = "select.ini";
    string score_file = "", state_output;

    cmdopts.addop(pool_file, "pool", "", "input structure pool")
        .addop(index_file, "index", "id", "index map file")
        .addop(iformat, "input-format", "if", "input format")
        .addop(oformat, "output-format", "of", "output format")
        .addop(output_file, "output", "o", "output selected structures")
        .addop(score_file, "score-file", "", "output per-structure MaxVol scores");
    cmdopts.addop(state_output, "state-output", "", "Output MaxVol state (.gpm or legacy .sel)");
    builder_settings.RegisterConsole(cmdopts);
    selector_settings.RegisterConsole(cmdopts);
    cmdopts.addop(settingfile, "settings", "", "settings file", true).phrase_options(argc, argv);

    if (GPIO::file_exist(settingfile))
    {
        settings.Load(settingfile);
        apply_settings_to_cmdopts(cmdopts, settings);
        builder_settings.ApplySettings(settings.ExtractSubSettings("builder"));
        selector_settings.ApplySettings(settings.ExtractSubSettings("selector"));
    }
    if (state_output.empty())
        state_output = maxvol_default_state(builder_settings, persistence.legacy);
    persistence.validate_state(state_output);
    if (GPM::extension(state_output) && !maxvol_model_builder(builder_settings.dtype) &&
        !GPIO::file_exist(builder_settings.potfile))
        GPERROR(
            "GPM selector export requires a target potential; use .sel for independent descriptors");
    maxvol_species(iobj, builder_settings.potfile, index_file, settings);

    auto builder = make_builder(builder_settings, settings);
    if (!builder)
    {
        Message("ERROR: failed to create B-matrix builder");
        return 1;
    }

    if (!maxvol_gpm(builder_settings.potfile))
    {
        if (auto *b = dynamic_cast<PotEFSBuilder *>(builder.get()))
            b->potential().set_element_map(iobj.getmap());
        if (auto *b = dynamic_cast<DescriptorBuilder *>(builder.get()))
            if (auto *p = b->target_potential())
                p->set_element_map(iobj.getmap());
    }
    int ncols = builder->coeff_count();
    double rcut = builder->cutoff();

    ActiveSelector as(builder.release());
    copy_selector_settings(as, selector_settings);

    Message(fmt::format("dtype={} pottype={} ncols={} cutoff={:.3f} mode={}",
                        builder_settings.dtype, builder_settings.pottype, ncols, rcut,
                        builder_settings.mode));
    Message(fmt::format("Reading pool from: {}", pool_file));
    vector<Atoms> pool;
    if (!iobj.read(pool, pool_file, iformat))
        GPERROR("Failed to read structures: " + pool_file);
    Message(fmt::format("Pool size: {}", pool.size()));
    if (pool.empty())
    {
        Message("ERROR: empty pool — nothing to select from");
        return 1;
    }

    int nsel = 0;
    if (as.select_mode == "full")
    {
        Message("Building B matrix...");
        as.add_candidates_parallel(pool);
        Message(fmt::format("Running MaxVol full/reference (threshold={})", as.threshold));
        nsel = as.select();
    }
    else if (as.select_mode == "online")
    {
        Message(fmt::format("Running online MaxVol initialization (threshold={})", as.threshold));
        as.add_candidates_unbuilt(pool);
        nsel = as.select_online();
    }
    else
    {
        Message(fmt::format("Running streaming MaxVol (threshold={}, mode={})", as.threshold,
                            as.stream_mode));
        if (mpi.size > 1)
        {
            as.add_candidates_unbuilt(pool);
            nsel = as.select_streaming_mpi(mpi.comm);
        }
        else
        {
            nsel = as.select_streaming(pool);
        }
    }
    Message(
        fmt::format("Selected {} / {}  log|det|={:.3f}", nsel, pool.size(), as.log_determinant()));
    Message(fmt::format("Selection status: {}  max_grade={:.6g}  virtual_rows={}",
                        as.converged() ? "converged" : "WARNING: not converged",
                        as.max_remaining_grade(), as.virtual_row_count()));
    if (!as.converged())
        Message(fmt::format("WARNING: MaxVol state written with termination reason '{}'",
                            as.termination_reason()));
    if (as.verbose && as.select_mode == "stream")
    {
        const auto &stats = as.stream_stats();
        Message(fmt::format("Stream stats: cache={} resident={} MiB mapped={} MiB "
                            "threads={}/{}/{} builder_calls={} rows={} scans={} swaps={} "
                            "build={:.3f}s init={:.3f}s score={:.3f}s total={:.3f}s",
                            stats.cache_backend, stats.resident_bytes / (1024.0 * 1024.0),
                            stats.mapped_bytes / (1024.0 * 1024.0), stats.configured_threads,
                            stats.build_threads, stats.score_threads, stats.builder_calls,
                            stats.rows_built, stats.score_passes, stats.swaps, stats.build_seconds,
                            stats.initialization_seconds, stats.score_seconds,
                            stats.total_seconds));
        Message(fmt::format(
            "Stream factors: full={} ridge_updates={} factor={:.3f}s update={:.3f}s scan={:.3f}s",
            stats.full_factorizations, stats.ridge_updates, stats.factorization_seconds,
            stats.ridge_update_seconds, stats.scan_seconds));
    }

    Training::collective_try(
        [&]
        {
            if (mpi.rank != 0)
                return;
            if (persistence.legacy)
            {
                publish_legacy_maxvol_selection(as, iobj, output_file, oformat, state_output,
                                                index_file, builder_settings, score_file);
                return;
            }
            vector<string> paths{state_output, output_file};
            if (!score_file.empty())
                paths.push_back(score_file);
            set<filesystem::path> unique;
            vector<unique_ptr<MaxVolStagedOutput>> outputs;
            for (const auto &path : paths)
            {
                if (!unique.insert(filesystem::absolute(path).lexically_normal()).second)
                    GPERROR("MaxVol output destinations must differ");
                outputs.push_back(make_unique<MaxVolStagedOutput>(path));
            }
            as.save_state(outputs[0]->temporary.string(), index_file, builder_settings.potfile,
                          builder_settings.pottype);
            vector<Atoms> selected(as.get_selected().begin(), as.get_selected().end());
            if (!write_maxvol_selection(iobj, selected, outputs[1]->temporary.string(), oformat))
                GPERROR("Cannot stage selected structures: " + output_file);
            if (!score_file.empty())
            {
                ofstream stream(outputs[2]->temporary);
                vector<double> grades;
                as.compute_grades(grades);
                set<int> ids(as.selected_indices().begin(), as.selected_indices().end());
                stream << "# idx  grade  selected(1=yes)\n";
                for (size_t i = 0; i < pool.size(); ++i)
                    stream << i << " " << grades[i] << " " << (ids.count(i) ? 1 : 0) << "\n";
                stream.close();
                if (!stream)
                    GPERROR("Cannot write staged MaxVol scores: " + score_file);
            }
            publish_maxvol_selection_outputs(outputs);
            Message(fmt::format("Written {} configs to {}; state saved to {}", selected.size(),
                                output_file, state_output));
        });
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// sample
// ═══════════════════════════════════════════════════════════════════════════
int maxvol_sample(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    PersistenceMode persistence;
    persistence.RegisterConsole(cmdopts);
    GPSettings settings;
    GPIO iobj;
    BMatrixBuilderSettings builder_settings;
    ActiveSelector selector_settings;

    string input_file = "candidates.xyz";
    string output_file = "sampled.xyz", state_file = "";
    string index_file = "index.map", iformat = "", oformat = "extxyz", settingfile = "sample.ini";

    double thr_save = 2.1, thr_break = 11.0;
    bool add_grade = true;
    bool dump_state = false, selected_only = false;
    string extract_file = "";

    cmdopts.addop(input_file, "input", "", "input configuration file")
        .addop(output_file, "output", "o", "output sampled structures")
        .addop(state_file, "state-file", "", "active-set state file")
        .addop(index_file, "index", "id", "index map file")
        .addop(iformat, "input-format", "if", "input format")
        .addop(oformat, "output-format", "of", "output format")
        .addop(thr_save, "threshold-save", "", "grade above this → save")
        .addop(thr_break, "threshold-break", "", "grade above this → abort")
        .addop(add_grade, "add-grade", "", "add MV_grade feature")
        .addop(dump_state, "dump-state", "", "print state file summary (no grading)")
        .addop(extract_file, "extract-configs", "", "export configs from state file to given path")
        .addop(selected_only, "selected-only", "",
               "export only selected configs (with --extract-configs)");
    builder_settings.RegisterConsole(cmdopts);
    selector_settings.RegisterConsole(cmdopts);
    cmdopts.addop(settingfile, "settings", "", "settings file", true).phrase_options(argc, argv);

    if (GPIO::file_exist(settingfile))
    {
        settings.Load(settingfile);
        apply_settings_to_cmdopts(cmdopts, settings);
        builder_settings.ApplySettings(settings.ExtractSubSettings("builder"));
        selector_settings.ApplySettings(settings.ExtractSubSettings("selector"));
    }
    if (state_file.empty())
    {
        if (persistence.legacy)
            state_file = builder_settings.potfile + ".sel";
        else if (maxvol_has_embedded_state(builder_settings.potfile))
            state_file = builder_settings.potfile;
        else if (GPIO::file_exist(maxvol_default_state(builder_settings)))
            state_file = maxvol_default_state(builder_settings);
        else
            state_file = builder_settings.potfile + ".sel";
    }

    persistence.validate_state(state_file);

    // ── Dump / extract mode (no builder or candidates needed) ──────────
    if (dump_state)
    {
        ActiveSelector::dump_state(state_file, std::cout);
        return 0;
    }
    if (!extract_file.empty())
    {
        ActiveSelector::export_configs(
            state_file, extract_file, selected_only,
            maxvol_gpm(state_file) && !maxvol_explicit(settings, "index", "") ? "" : index_file);
        Message(fmt::format("Configs exported to {}", extract_file));
        return 0;
    }

    // ── Read candidates early; skip expensive builder construction if empty ──
    maxvol_species(iobj, maxvol_gpm(state_file) ? state_file : builder_settings.potfile, index_file,
                   settings);
    vector<Atoms> cfgs;
    if (!iobj.read(cfgs, input_file, iformat))
        GPERROR("Failed to read structures: " + input_file);
    if (cfgs.empty())
    {
        Message(fmt::format("No candidates found in {}", input_file));
        return 0;
    }
    Message(fmt::format("Candidates: {}", cfgs.size()));

    auto selector =
        maxvol_load(state_file, builder_settings, selector_settings, settings, index_file);
    auto &as = *selector;
    if (!as.converged())
        Message(fmt::format("WARNING: grading with non-converged state "
                            "(reason={}, max_grade={:.6g})",
                            as.termination_reason(), as.max_remaining_grade()));

    if (!as.prepare_grading())
    {
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
    if (!ofs)
        GPERROR("Cannot open output file: " + output_file);
    for (size_t i = 0; i < cfgs.size(); ++i)
    {
        double g = grades[i];
        if (g > thr_break)
        {
            Message(fmt::format("BREAK: cfg {} grade {:.3f} > {} — stopping", i, g, thr_break));
            break;
        }
        if (g > thr_save)
        {
            if (add_grade)
                cfgs[i].features_["MV_grade"] = fmt::format("{:.6f}", g);
            if (!iobj.write_one(cfgs[i], ofs, oformat))
                GPERROR("Failed to write structures: " + output_file);
            ++ns;
        }
    }
    ofs.close();
    if (!ofs)
        GPERROR("Failed to write structures: " + output_file);
    Message(fmt::format("Saved {} / {} (grade > {})", ns, cfgs.size(), thr_save));
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// select_add
// ═══════════════════════════════════════════════════════════════════════════
int maxvol_extend(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    PersistenceMode persistence;
    persistence.RegisterConsole(cmdopts);
    GPSettings settings;
    GPIO iobj;
    BMatrixBuilderSettings builder_settings;
    ActiveSelector selector_settings;

    string train_file = "";
    string state_file = "", candidates_file = "candidates.xyz";
    string output_file = "new_selected.xyz", state_output;
    string index_file = "index.map";
    string iformat = "", oformat = "extxyz", settingfile = "select_add.ini";

    cmdopts.addop(state_file, "state-file", "", "active-set state file (skip Stage 1)")
        .addop(train_file, "train", "", "training set file")
        .addop(candidates_file, "candidates", "", "candidate structures file")
        .addop(output_file, "output", "o", "output newly selected structures")
        .addop(index_file, "index", "id", "index map file")
        .addop(iformat, "input-format", "if", "input format")
        .addop(oformat, "output-format", "of", "output format");
    cmdopts.addop(state_output, "state-output", "", "Output MaxVol state (.gpm or legacy .sel)");
    builder_settings.RegisterConsole(cmdopts);
    selector_settings.RegisterConsole(cmdopts);
    cmdopts.addop(settingfile, "settings", "", "settings file", true).phrase_options(argc, argv);

    if (GPIO::file_exist(settingfile))
    {
        settings.Load(settingfile);
        apply_settings_to_cmdopts(cmdopts, settings);
        builder_settings.ApplySettings(settings.ExtractSubSettings("builder"));
        selector_settings.ApplySettings(settings.ExtractSubSettings("selector"));
    }
    if (!persistence.legacy && state_file.empty() &&
        maxvol_has_embedded_state(builder_settings.potfile))
        state_file = builder_settings.potfile;
    if (state_output.empty() && (!persistence.legacy || !state_file.empty()))
        state_output = state_file.empty() ? maxvol_default_state(builder_settings) : state_file;
    persistence.validate_state(state_file);
    persistence.validate_state(state_output);
    maxvol_species(iobj, maxvol_gpm(state_file) ? state_file : builder_settings.potfile, index_file,
                   settings);

    // ── Read candidates early; skip expensive builder construction if empty ──
    vector<Atoms> candidates;
    if (!iobj.read(candidates, candidates_file, iformat))
        GPERROR("Failed to read structures: " + candidates_file);
    if (candidates.empty())
    {
        Message(fmt::format("No candidates found in {}", candidates_file));
        return 0;
    }
    if (state_file.empty() && train_file.empty())
        GPERROR("Provide a GPM MaxVol state, --state-file, or --train to initialize selection");
    for (auto &c : candidates)
        c.features_[ActiveSelector::kNewTag] = "1";

    unique_ptr<ActiveSelector> selector;
    if (!state_file.empty())
        selector =
            maxvol_load(state_file, builder_settings, selector_settings, settings, index_file);
    else
    {
        auto builder = make_builder(builder_settings, settings);
        if (!builder)
            GPERROR("Failed to create B-matrix builder");
        selector = make_unique<ActiveSelector>(builder.release());
        copy_selector_settings(*selector, selector_settings);
    }
    auto &as = *selector;

    if (!state_file.empty())
    {
        // ── State-file branch: load existing active set, then select on
        //     training ∪ candidates in a single pass ──────────────────
        Message(fmt::format("Loading active set from {}", state_file));
        // State was loaded together with its target model above.
        if (!as.converged())
            Message(fmt::format("WARNING: extending non-converged state "
                                "(reason={}, max_grade={:.6g})",
                                as.termination_reason(), as.max_remaining_grade()));

        if (!train_file.empty())
        {
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
        if (as.select_mode == "full")
        {
            as.rebuild_B_from_configs();
            as.add_candidates_parallel(candidates);
        }
        else
        {
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
        Message(fmt::format("Active set: {} configs  log|det|={:.3f}", n, as.log_determinant()));

        vector<Atoms> new_sel;
        const auto &selected = as.get_selected();
        const auto &selected_indices = as.selected_indices();
        for (size_t i = 0; i < selected.size() && i < selected_indices.size(); ++i)
        {
            const int idx = selected_indices[i];
            if (idx >= candidate_start &&
                idx < candidate_start + static_cast<int>(candidates.size()) &&
                selected[i].size() > 0)
            {
                Atoms out = selected[i];
                out.features_.erase(ActiveSelector::kNewTag);
                new_sel.push_back(out);
            }
        }
        Message(fmt::format("Newly selected from candidates: {}", new_sel.size()));
        publish_maxvol_extension(as, iobj, new_sel, output_file, oformat, state_output, index_file,
                                 builder_settings, persistence.legacy);
    }
    else
    {
        // ── Original 3-stage logic (no state file) ─────────────────────────
        if (train_file.empty())
        {
            Message("ERROR: --train is required when --state-file is not given");
            return 1;
        }

        Message(fmt::format("Stage 1: building active set from {}", train_file));
        vector<Atoms> train;
        if (!iobj.read(train, train_file, iformat))
            GPERROR("Failed to read structures: " + train_file);

        auto train_builder = make_builder(builder_settings, settings);
        if (!train_builder)
        {
            Message("ERROR: failed to create training B-matrix builder");
            return 1;
        }
        ActiveSelector train_as(train_builder.release());
        copy_selector_settings(train_as, selector_settings);

        if (as.select_mode == "full")
        {
            train_as.select_mode = "full";
            train_as.add_candidates_parallel(train);
            int n1 = train_as.select(1.001);
            Message(fmt::format("Selected {} from training set", n1));

            Message(fmt::format("Stage 2: evaluating {}", candidates_file));
            as.add_candidates_parallel(candidates);
            int n2 = as.select(as.threshold);
            Message(fmt::format("Selected {} new configs", n2));

            Message("Stage 3: re-selecting training set");
            if (!as.append_built_candidates_from(train_as))
            {
                Message("ERROR: failed to append prebuilt training B matrix");
                return 1;
            }
            as.select(1.001);
        }
        else if (as.select_mode == "online")
        {
            Message("Online selection on training plus candidates");
            vector<Atoms> combined = train;
            combined.insert(combined.end(), candidates.begin(), candidates.end());
            as.add_candidates_unbuilt(combined);
            as.select_online();
        }
        else
        {
            Message("Streaming selection on training plus candidates");
            vector<Atoms> combined = train;
            combined.insert(combined.end(), candidates.begin(), candidates.end());
            as.select_streaming(combined);
        }

        vector<Atoms> new_sel;
        for (const auto &cfg : as.get_selected())
        {
            auto it = cfg.features_.find(ActiveSelector::kNewTag);
            if (it != cfg.features_.end() && it->second == "1" && cfg.size() > 0)
            {
                Atoms out = cfg;
                out.features_.erase(ActiveSelector::kNewTag);
                new_sel.push_back(out);
            }
        }
        Message(fmt::format("Newly selected: {} configs  log|det|={:.3f}", new_sel.size(),
                            as.log_determinant()));
        publish_maxvol_extension(as, iobj, new_sel, output_file, oformat, state_output, index_file,
                                 builder_settings, persistence.legacy);
    }
    return 0;
}
