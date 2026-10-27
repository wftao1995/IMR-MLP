// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file calc_crowding.cpp
 * @brief Compute bounded-memory per-atom descriptor crowding weights.
 */

#include "io/gpio.h"
#include "maxvol/b_matrix_builder.h"
#include "maxvol/crowding.h"
#include "utils/gputils.h"
#include "utils/compressed_stream.h"
#include <fmt/format.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace {

string sequential_format(const string &filename, const string &requested)
{
    if (!requested.empty()) return requested;
    switch (GPIO::guess_format(filename)) {
    case GPIO::EXTXYZ: return "extxyz";
    case GPIO::MLPCFG: return "mlpcfg";
    case GPIO::MLIPBIN: return "mlipbin";
    case GPIO::LMPTRJ: return "lmptrj";
    case GPIO::LMPYAML: return "lmpyaml";
    default:
        throw runtime_error(
            "calc_crowding: input format cannot be inferred for sequential reading");
    }
}

bool descriptor_dtype(const string &dtype)
{
    return dtype == "mtm" || dtype == "mtp" || dtype == "mtp_scalar" ||
           dtype == "behler" || dtype == "bondorder" || dtype == "biso4";
}

bool attribute_output_format(GPIO::supported_format format)
{
    return format == GPIO::EXTXYZ || format == GPIO::MLPCFG ||
           format == GPIO::MLIPBIN;
}

} // namespace

int calc_crowding(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    GPSettings settings;
    BMatrixBuilderSettings builder_settings;

    string infile = "pool.xyz";
    string outfile = "pool-crowded.xyz";
    string iformat;
    string oformat;
    string settingfile = "crowding.ini";
    string indexfile = "index.map";
    CrowdingParams crowding;
    bool bucket_by_element = true;
    bool apply_to_energy = false;

    cmdopts.addop(infile, "input", "in", "input structures")
           .addop(outfile, "output", "out", "output structures with atomic_fitness")
           .addop(iformat, "input-format", "if", "input format")
           .addop(oformat, "output-format", "of", "output format")
           .addop(settingfile, "settings", "", "settings file", true)
           .addop(indexfile, "index", "id", "index map file")
           .addop(crowding.kernel, "kernel", "", "kernel: cosine or gaussian")
           .addop(crowding.sigma, "sigma", "", "gaussian bandwidth (<=0: automatic)")
           .addop(crowding.standardize, "standardize", "",
                  "scale descriptor components before the kernel", false, 0)
           .addop(crowding.alpha, "alpha", "", "inverse-density exponent")
           .addop(crowding.clip_min, "clip-min", "", "lower weight clip")
           .addop(crowding.clip_max, "clip-max", "", "upper weight clip")
           .addop(crowding.max_density_points, "sample", "",
                  "reference rows per element (-1: exact small-data mode)")
           .addop(crowding.seed, "seed", "", "deterministic sampling seed")
           .addop(crowding.batch_rows, "batch-rows", "", "density query batch rows")
           .addop(bucket_by_element, "bucket", "",
                  "bucket density by chemical element", false, 0)
           .addop(apply_to_energy, "apply-to-energy", "",
                  "write config fitness as mean atomic weight", false, 0);
    builder_settings.RegisterConsole(cmdopts);
    cmdopts.phrase_options(argc, argv);

    if (GPIO::file_exist(settingfile)) {
        settings.Load(settingfile);
        apply_settings_to_cmdopts(cmdopts, settings);
        builder_settings.ApplySettings(settings.ExtractSubSettings("builder"),
                                       cmdopts.provided_options());
    }

    try {
        if (!descriptor_dtype(builder_settings.dtype))
            throw invalid_argument(fmt::format(
                "calc_crowding: dtype '{}' is not a per-atom descriptor builder",
                builder_settings.dtype));
        if (!GPIO::file_exist(indexfile))
            throw runtime_error(fmt::format("index map file '{}' not found", indexfile));
        if (!GPIO::file_exist(infile))
            throw runtime_error(fmt::format("input file '{}' not found", infile));
        if (infile == outfile)
            throw invalid_argument("calc_crowding: input and output must differ");

        // The model constructor performs the complete crowding-parameter check.
        CrowdingModel model_probe(1, crowding);
        (void)model_probe;

        builder_settings.mode = "nbh";
        auto builder = make_builder(builder_settings, settings);
        if (!builder) throw runtime_error("failed to create descriptor builder");
        const int ncols = builder->coeff_count();
        if (ncols <= 0) throw runtime_error("descriptor builder has no columns");

        const string seq_iformat = sequential_format(infile, iformat);
        const auto output_kind = oformat.empty()
            ? GPIO::guess_format(outfile) : GPIO::format2enum(oformat);
        if (!attribute_output_format(output_kind))
            throw invalid_argument(
                "calc_crowding: output format must be extxyz, mlpcfg, or mlipbin");

        Message(fmt::format(
            "dtype={} ncols={} cutoff={:.3f} kernel={} sample={} seed={} batch_rows={}",
            builder_settings.dtype, ncols, builder->cutoff(), crowding.kernel,
            crowding.max_density_points, crowding.seed, crowding.batch_rows));

        auto for_each_config = [&](auto &&callback) {
            GPIO reader;
            if (!reader.readmap(indexfile))
                throw runtime_error("failed to read index map");
            auto input = open_ifile(infile);
            if (!input) throw runtime_error("failed to open input file");
            if (!reader.seq_read_init(*input, seq_iformat))
                throw runtime_error("failed to initialize sequential input");
            size_t cfg_index = 0;
            while (true) {
                Atoms cfg;
                if (!reader.seq_read(*input, cfg)) {
                    if (!input->eof())
                        throw runtime_error(fmt::format(
                            "failed while reading configuration {}", cfg_index));
                    break;
                }
                callback(cfg, cfg_index++);
            }
            return cfg_index;
        };

        auto build_descriptor_block = [&](Atoms &cfg, uint64_t first_row,
                                          BMatrixBuilder::RowMatrix &B,
                                          vector<int> &buckets,
                                          vector<uint64_t> &row_ids) {
            B.resize(0, ncols);
            vector<BRowRef> refs;
            const int nr = builder->build_rows(cfg, B, 1.0, 0.0, &refs);
            if (nr != cfg.size() || B.rows() != nr ||
                refs.size() != static_cast<size_t>(nr))
                throw runtime_error(
                    "descriptor builder must produce exactly one row per atom");
            buckets.resize(static_cast<size_t>(nr));
            row_ids.resize(static_cast<size_t>(nr));
            for (int r = 0; r < nr; ++r) {
                if (refs[r].kind != BRowKind::Descriptor ||
                    refs[r].atom_index < 0 || refs[r].atom_index >= cfg.size())
                    throw runtime_error("descriptor builder returned invalid row metadata");
                buckets[r] = bucket_by_element
                    ? cfg.chemmap_[refs[r].atom_index] : 0;
                row_ids[r] = first_row + static_cast<uint64_t>(r);
            }
        };

        // Pass 1: statistics and bounded reference sampling.
        CrowdingModel model(ncols, crowding);
        uint64_t observed_rows = 0;
        const size_t config_count = for_each_config(
            [&](Atoms &cfg, size_t cfg_index) {
                BMatrixBuilder::RowMatrix B;
                vector<int> buckets;
                vector<uint64_t> ids;
                build_descriptor_block(cfg, observed_rows, B, buckets, ids);
                model.observe(B, buckets, ids);
                observed_rows += static_cast<uint64_t>(B.rows());
                if ((cfg_index + 1) % 500 == 0)
                    Message(fmt::format("Reference pass: {} configurations", cfg_index + 1));
            });
        if (config_count == 0 || observed_rows == 0)
            throw runtime_error("no descriptor rows were generated");
        model.finalize();
        Message(fmt::format("Reference pass complete: {} configurations, {} atoms",
                            config_count, observed_rows));

        // Pass 2: batch density evaluation. Only scalar density and bucket IDs
        // are retained, not descriptors or configurations.
        vector<double> densities;
        vector<int> all_buckets;
        densities.reserve(static_cast<size_t>(observed_rows));
        all_buckets.reserve(static_cast<size_t>(observed_rows));
        uint64_t evaluated_rows = 0;
        const size_t second_configs = for_each_config(
            [&](Atoms &cfg, size_t cfg_index) {
                BMatrixBuilder::RowMatrix B;
                vector<int> buckets;
                vector<uint64_t> ids;
                build_descriptor_block(cfg, evaluated_rows, B, buckets, ids);
                for (int begin = 0; begin < B.rows(); begin += crowding.batch_rows) {
                    const int count = min(crowding.batch_rows,
                                          static_cast<int>(B.rows()) - begin);
                    BMatrixBuilder::RowMatrix block = B.middleRows(begin, count);
                    vector<int> block_buckets(buckets.begin() + begin,
                                              buckets.begin() + begin + count);
                    vector<uint64_t> block_ids(ids.begin() + begin,
                                               ids.begin() + begin + count);
                    auto rho = model.density(block, block_buckets, block_ids);
                    densities.insert(densities.end(), rho.begin(), rho.end());
                    all_buckets.insert(all_buckets.end(), block_buckets.begin(),
                                       block_buckets.end());
                }
                evaluated_rows += static_cast<uint64_t>(B.rows());
                if ((cfg_index + 1) % 500 == 0)
                    Message(fmt::format("Density pass: {} configurations", cfg_index + 1));
            });
        if (second_configs != config_count || evaluated_rows != observed_rows)
            throw runtime_error("input changed between crowding passes");
        vector<double> weights = CrowdingWeights::weights_from_density(
            densities, all_buckets, crowding);
        densities.clear();
        densities.shrink_to_fit();
        all_buckets.clear();
        all_buckets.shrink_to_fit();

        // Pass 3: read original data once more, attach weights, and write.
        GPIO reader, writer;
        if (!reader.readmap(indexfile) || !writer.readmap(indexfile))
            throw runtime_error("failed to read index map for output pass");
        auto input = open_ifile(infile);
        if (!input || !reader.seq_read_init(*input, seq_iformat))
            throw runtime_error("failed to initialize output input pass");
        auto output = open_ofile(outfile, false, output_kind == GPIO::MLIPBIN);
        if (!output || !writer.seq_write_init(*output, oformat, outfile))
            throw runtime_error("failed to initialize sequential output");

        uint64_t written_rows = 0;
        size_t written_configs = 0;
        while (true) {
            Atoms cfg;
            if (!reader.seq_read(*input, cfg)) {
                if (!input->eof())
                    throw runtime_error("failed while reading output pass");
                break;
            }
            if (written_rows + static_cast<uint64_t>(cfg.size()) > weights.size())
                throw runtime_error("weight/configuration size mismatch");
            cfg.atomic_fitness.resize(static_cast<size_t>(cfg.size()));
            Eigen::VectorXd attr(cfg.size());
            double mean = 0.0;
            for (int a = 0; a < cfg.size(); ++a) {
                const double w = weights[written_rows + static_cast<uint64_t>(a)];
                cfg.atomic_fitness[a] = w;
                attr[a] = w;
                mean += w;
            }
            cfg.local_attribs_d_["atomic_fitness"] = std::move(attr);
            if (apply_to_energy && cfg.size() > 0) {
                cfg.fitness = mean / static_cast<double>(cfg.size());
                cfg.global_attribs_d_["fitness"] = cfg.fitness;
                cfg.features_.erase("Fitness");
            }
            if (!writer.seq_write(*output, cfg))
                throw runtime_error("failed while writing crowded structures");
            written_rows += static_cast<uint64_t>(cfg.size());
            ++written_configs;
        }
        if (written_rows != observed_rows || written_configs != config_count)
            throw runtime_error("output pass did not consume every weight");

        const auto [wmin, wmax] = minmax_element(weights.begin(), weights.end());
        vector<double> sorted = weights;
        nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        Message(fmt::format("Weight range: [{:.6g}, {:.6g}] median={:.6g}",
                            *wmin, *wmax, sorted[sorted.size() / 2]));
        Message(fmt::format("Wrote {} configurations and {} atomic weights to {}",
                            written_configs, written_rows, outfile));
        Message("Complete!");
        return 0;
    } catch (const exception &e) {
        Message(fmt::format("ERROR: {}", e.what()));
        return 1;
    }
}
