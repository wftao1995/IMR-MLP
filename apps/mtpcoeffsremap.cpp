// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "basis_analysis/MTPBasisEquivalence.h"
#include "potentials/potential_mtp.h"
#include "utils/compressed_stream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace {

using BasisAnalysis::MTPBasisComparison;
using BasisAnalysis::MTPBasisMode;
using BasisAnalysis::MTPBasisRelation;

struct SidecarResult {
    bool provided = false;
    bool scalar_ids_match = false;
    optional<vector<int>> preferred_mapping;
};

struct PotentialSections {
    bool radial_coeffs = false;
    bool species_coeffs = false;
    bool moment_coeffs = false;
    bool unsupported = false;
};

using CompatibilityResult = BasisAnalysis::MTPadialCompatibility;

string trim(string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

string json_escape(const string &value) {
    ostringstream output;
    for (unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20) {
                output << "\\u" << hex << setw(4) << setfill('0')
                       << static_cast<int>(character) << dec;
            } else {
                output << character;
            }
        }
    }
    return output.str();
}

string quoted_json(const string &value) {
    return '"' + json_escape(value) + '"';
}

filesystem::path normalized_path(const filesystem::path &path) {
    error_code error;
    auto normalized = filesystem::weakly_canonical(path, error);
    if (!error) return normalized;
    return filesystem::absolute(path).lexically_normal();
}

bool same_path(const filesystem::path &left, const filesystem::path &right) {
    return normalized_path(left) == normalized_path(right);
}

filesystem::path temporary_sibling(const filesystem::path &output) {
    const auto stamp = chrono::steady_clock::now().time_since_epoch().count();
    return output.parent_path() /
           (output.filename().string() + ".tmp-" + to_string(stamp));
}

void atomic_write_text(const filesystem::path &path, const string &text,
                       bool overwrite) {
    if (filesystem::exists(path) && !overwrite)
        throw runtime_error("output already exists; use --overwrite: " +
                            path.string());
    if (!path.parent_path().empty() && !filesystem::exists(path.parent_path()))
        throw runtime_error("output directory does not exist: " +
                            path.parent_path().string());
    const auto temporary = temporary_sibling(path);
    try {
        ofstream output(temporary);
        if (!output) throw runtime_error("cannot open output: " + temporary.string());
        output << text;
        output.close();
        if (!output) throw runtime_error("failed to write output: " + temporary.string());
        error_code error;
        filesystem::rename(temporary, path, error);
        if (error)
            throw runtime_error("cannot atomically replace " + path.string() +
                                ": " + error.message());
    } catch (...) {
        error_code ignored;
        filesystem::remove(temporary, ignored);
        throw;
    }
}

map<int, int> read_tensor_mapping(const string &filename, int moments_count) {
    auto input = open_ifile(filename);
    if (!input) throw runtime_error("cannot open mapping file: " + filename);
    map<int, int> result;
    string line;
    int line_number = 0;
    while (getline(*input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != string::npos) line.erase(comment);
        line = trim(line);
        if (line.empty()) continue;
        istringstream parser(line);
        int moment = -1;
        int tensor = -1;
        string extra;
        if (!(parser >> moment >> tensor) || (parser >> extra))
            throw runtime_error(filename + ":" + to_string(line_number) +
                                ": expected 'moment_id tensor_id'");
        if (moment < 0 || moment >= moments_count || tensor < 0)
            throw runtime_error(filename + ":" + to_string(line_number) +
                                ": mapping id is out of range");
        if (!result.emplace(moment, tensor).second)
            throw runtime_error(filename + ":" + to_string(line_number) +
                                ": duplicate moment id");
    }
    if (result.empty()) throw runtime_error("mapping file is empty: " + filename);
    return result;
}

SidecarResult prepare_sidecar_mapping(
    const BasisAnalysis::MTPBasisDefinition &source,
    const BasisAnalysis::MTPBasisDefinition &target,
    const string &source_file, const string &target_file) {
    SidecarResult result;
    result.provided = true;
    const auto source_mapping =
        read_tensor_mapping(source_file, source.moments_count);
    const auto target_mapping =
        read_tensor_mapping(target_file, target.moments_count);

    vector<int> source_tensor_ids;
    source_tensor_ids.reserve(source.scalars.size());
    for (int scalar : source.scalars) {
        const auto found = source_mapping.find(scalar);
        if (found == source_mapping.end())
            throw runtime_error("source mapping is missing scalar moment " +
                                to_string(scalar));
        source_tensor_ids.push_back(found->second);
    }
    map<int, deque<int>> target_positions;
    for (int i = 0; i < static_cast<int>(target.scalars.size()); ++i) {
        const auto found = target_mapping.find(target.scalars[i]);
        if (found == target_mapping.end())
            throw runtime_error("target mapping is missing scalar moment " +
                                to_string(target.scalars[i]));
        target_positions[found->second].push_back(i);
    }

    if (source.scalars.size() != target.scalars.size()) return result;
    vector<int> preferred(source.scalars.size(), -1);
    for (int i = 0; i < static_cast<int>(source_tensor_ids.size()); ++i) {
        auto found = target_positions.find(source_tensor_ids[i]);
        if (found == target_positions.end() || found->second.empty()) return result;
        preferred[i] = found->second.front();
        found->second.pop_front();
    }
    if (any_of(target_positions.begin(), target_positions.end(),
               [](const auto &item) { return !item.second.empty(); }))
        return result;
    result.scalar_ids_match = true;
    result.preferred_mapping = std::move(preferred);
    return result;
}

vector<string> read_index_map(const string &filename) {
    auto input = open_ifile(filename);
    if (!input) throw runtime_error("cannot open index map: " + filename);
    vector<string> species;
    string line;
    while (getline(*input, line)) {
        const auto comment = line.find('#');
        if (comment != string::npos) line.erase(comment);
        istringstream parser(line);
        string symbol;
        if (parser >> symbol) species.push_back(symbol);
    }
    if (species.empty()) throw runtime_error("index map is empty: " + filename);
    return species;
}

PotentialSections inspect_potential_sections(const string &filename) {
    auto input = open_ifile(filename);
    if (!input) throw runtime_error("cannot inspect potential: " + filename);
    PotentialSections result;
    string line;
    while (getline(*input, line)) {
        line = trim(line);
        if (line == "radial_coeffs") result.radial_coeffs = true;
        if (line.starts_with("species_coeffs")) result.species_coeffs = true;
        if (line.starts_with("moment_coeffs")) result.moment_coeffs = true;
        if (line.starts_with("les_") ||
            line.starts_with("remove_self_interaction") ||
            line.starts_with("model_coeffs") ||
            line.starts_with("nn_coeffs"))
            result.unsupported = true;
    }
    return result;
}

void save_potential_atomically(Potentials::PotMTP &potential,
                               const filesystem::path &output,
                               bool overwrite) {
    const string extension = output.extension().string();
    if (extension == ".gz" || extension == ".zst")
        throw runtime_error("compressed mapped outputs are not supported");
    if (filesystem::exists(output) && !overwrite)
        throw runtime_error("output already exists; use --overwrite: " +
                            output.string());
    if (!output.parent_path().empty() &&
        !filesystem::exists(output.parent_path()))
        throw runtime_error("output directory does not exist: " +
                            output.parent_path().string());

    const auto temporary = temporary_sibling(output);
    try {
        potential.Save(temporary.string());
        if (!filesystem::exists(temporary) || filesystem::file_size(temporary) == 0)
            throw runtime_error("mapped potential was not written");
        error_code error;
        filesystem::rename(temporary, output, error);
        if (error)
            throw runtime_error("cannot atomically replace " + output.string() +
                                ": " + error.message());
    } catch (...) {
        error_code ignored;
        filesystem::remove(temporary, ignored);
        throw;
    }
}

string diagnostics_json(const BasisAnalysis::MTPBasisDiagnostics &diagnostics) {
    ostringstream output;
    output << "{\n"
           << "      \"moments\": " << diagnostics.moments_count << ",\n"
           << "      \"basic\": " << diagnostics.basic_count << ",\n"
           << "      \"rules\": " << diagnostics.rules_count << ",\n"
           << "      \"scalars\": " << diagnostics.scalar_count << ",\n"
           << "      \"active_moments\": " << diagnostics.active_moments << ",\n"
           << "      \"dead_moments\": " << diagnostics.dead_moments << ",\n"
           << "      \"dead_rules\": " << diagnostics.dead_rules << ",\n"
           << "      \"duplicate_scalars\": " << diagnostics.duplicate_scalars << ",\n"
           << "      \"polynomial_terms\": " << diagnostics.polynomial_terms << ",\n"
           << "      \"max_polynomial_terms\": "
           << diagnostics.max_polynomial_terms << "\n"
           << "    }";
    return output.str();
}

string comparison_json(const string &source_file, const string &target_file,
                       const string &mode, const MTPBasisComparison &comparison,
                       const CompatibilityResult &radial,
                       bool index_checked, bool index_compatible,
                       const SidecarResult &sidecar, bool equivalent,
                       const string &mapped_output,
                       const vector<string> &warnings) {
    ostringstream output;
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"source\": " << quoted_json(source_file) << ",\n"
           << "  \"target\": " << quoted_json(target_file) << ",\n"
           << "  \"requested_mode\": " << quoted_json(mode) << ",\n"
           << "  \"relation\": "
           << quoted_json(BasisAnalysis::mtp_basis_relation_name(comparison.relation))
           << ",\n"
           << "  \"equivalent\": " << (equivalent ? "true" : "false") << ",\n"
           << "  \"radial_compatible\": "
           << (radial.compatible ? "true" : "false") << ",\n"
           << "  \"radial_reason\": " << quoted_json(radial.reason) << ",\n"
           << "  \"index_maps_checked\": "
           << (index_checked ? "true" : "false") << ",\n"
           << "  \"index_maps_compatible\": "
           << (index_compatible ? "true" : "false") << ",\n"
           << "  \"sidecars_provided\": "
           << (sidecar.provided ? "true" : "false") << ",\n"
           << "  \"sidecar_scalar_ids_match\": "
           << (sidecar.scalar_ids_match ? "true" : "false") << ",\n"
           << "  \"mapped_output\": " << quoted_json(mapped_output) << ",\n"
           << "  \"ranks\": {\"source\": " << comparison.source_rank
           << ", \"target\": " << comparison.target_rank
           << ", \"combined\": " << comparison.combined_rank << "},\n"
           << "  \"diagnostics\": {\n"
           << "    \"source\": " << diagnostics_json(comparison.source) << ",\n"
           << "    \"target\": " << diagnostics_json(comparison.target) << "\n"
           << "  },\n"
           << "  \"permutation\": [";
    for (int i = 0; i < static_cast<int>(comparison.source_to_target.size()); ++i) {
        if (i) output << ", ";
        output << comparison.source_to_target[i];
    }
    output << "],\n  \"transform\": [";
    for (int i = 0; i < static_cast<int>(comparison.transform.size()); ++i) {
        const auto &entry = comparison.transform[i];
        if (i) output << ',';
        output << "\n    {\"source\": " << entry.source
               << ", \"target\": " << entry.target
               << ", \"numerator\": " << quoted_json(entry.numerator)
               << ", \"denominator\": " << quoted_json(entry.denominator)
               << '}';
    }
    if (!comparison.transform.empty()) output << '\n';
    output << "  ],\n  \"warnings\": [";
    for (int i = 0; i < static_cast<int>(warnings.size()); ++i) {
        if (i) output << ", ";
        output << quoted_json(warnings[i]);
    }
    output << "]\n}\n";
    return output.str();
}

MTPBasisMode parse_mode(const string &mode) {
    if (mode == "auto") return MTPBasisMode::Auto;
    if (mode == "item") return MTPBasisMode::Item;
    if (mode == "span") return MTPBasisMode::Span;
    throw runtime_error("--equivalence must be auto, item, or span");
}

int run_mtpbasis(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    string source_file;
    string target_file;
    string source_mapping_file;
    string target_mapping_file;
    string source_index_file;
    string target_index_file;
    string output_file;
    string report_file;
    string equivalence_mode = "auto";
    bool overwrite = false;

    try {
        cmdopts
            .set_usage("imrmlp mtpbasis --source SOURCE.mtp --target TARGET.mtp [options]\n")
            .addop(source_file, "source", "s", "source MTP basis/potential")
            .addop(target_file, "target", "t", "target MTP basis/potential")
            .addop(source_mapping_file, "source-map", "sm", "source mid_tid_mapping file")
            .addop(target_mapping_file, "target-map", "tm", "target mid_tid_mapping file")
            .addop(source_index_file, "source-index", "si", "source index.map")
            .addop(target_index_file, "target-index", "ti", "target index.map")
            .addop(equivalence_mode, "equivalence", "eq", "auto, item, or span")
            .addop(output_file, "output", "o", "write target topology with mapped source parameters")
            .addop(report_file, "report", "", "write a machine-readable JSON report")
            .addop(overwrite, "overwrite", "ov", "overwrite output/report files", false, 0)
            .phrase_options(argc, argv);

        if (source_file.empty() || target_file.empty())
            throw runtime_error("both --source and --target are required");
        if ((source_mapping_file.empty()) != (target_mapping_file.empty()))
            throw runtime_error("--source-map and --target-map must be provided together");
        if ((source_index_file.empty()) != (target_index_file.empty()))
            throw runtime_error("--source-index and --target-index must be provided together");
        if (!output_file.empty() &&
            (same_path(output_file, source_file) || same_path(output_file, target_file)))
            throw runtime_error("--output must not overwrite either input potential");
        if (!report_file.empty() &&
            (same_path(report_file, source_file) || same_path(report_file, target_file) ||
             (!output_file.empty() && same_path(report_file, output_file))))
            throw runtime_error("--report must be separate from all potential files");

        Potentials::PotMTP source(source_file);
        Potentials::PotMTP target(target_file);
        const auto source_definition =
            BasisAnalysis::mtp_basis_definition(source);
        const auto target_definition =
            BasisAnalysis::mtp_basis_definition(target);

        SidecarResult sidecar;
        if (!source_mapping_file.empty())
            sidecar = prepare_sidecar_mapping(
                source_definition, target_definition,
                source_mapping_file, target_mapping_file);

        const auto comparison = BasisAnalysis::compare_mtp_bases(
            source_definition, target_definition,
            parse_mode(equivalence_mode),
            sidecar.preferred_mapping ? &*sidecar.preferred_mapping : nullptr);
        const auto radial =
            BasisAnalysis::compare_mtp_radial_layouts(source, target);
        const bool index_checked = !source_index_file.empty();
        const bool index_compatible =
            !index_checked ||
            read_index_map(source_index_file) == read_index_map(target_index_file);
        const bool topology_equivalent =
            comparison.relation != MTPBasisRelation::NotEquivalent;
        const bool equivalent =
            topology_equivalent && radial.compatible && index_compatible;

        vector<string> warnings;
        if (!index_checked)
            warnings.emplace_back(
                "species order was not verified; provide both index.map files");
        else if (!index_compatible)
            warnings.emplace_back("source and target index.map species orders differ");
        if (sidecar.provided && !sidecar.scalar_ids_match)
            warnings.emplace_back(
                "sidecar scalar tensor-id sets differ; exact topology comparison was used");
        if (comparison.source.dead_moments || comparison.target.dead_moments)
            warnings.emplace_back("one or both bases contain dead moments");

        string mapped_output;
        if (!output_file.empty()) {
            if (!equivalent)
                throw runtime_error("parameters cannot be mapped between non-equivalent bases");
            const auto source_sections =
                inspect_potential_sections(source_file);
            const auto target_sections =
                inspect_potential_sections(target_file);
            if (source_sections.unsupported || target_sections.unsupported)
                throw runtime_error(
                    "parameter mapping supports classic linear MTP files only");
            if (!source_sections.radial_coeffs ||
                !source_sections.species_coeffs ||
                !source_sections.moment_coeffs)
                throw runtime_error(
                    "source potential must contain radial_coeffs, "
                    "species_coeffs, and moment_coeffs");
            BasisAnalysis::map_mtp_linear_parameters(source, target, comparison);
            save_potential_atomically(target, output_file, overwrite);
            mapped_output = output_file;
        }

        cout << "MTP basis relation: "
             << BasisAnalysis::mtp_basis_relation_name(comparison.relation) << '\n';
        cout << "Ranks (source/target/combined): " << comparison.source_rank
             << '/' << comparison.target_rank << '/' << comparison.combined_rank << '\n';
        cout << "Source moments/rules/scalars/dead: "
             << comparison.source.moments_count << '/'
             << comparison.source.rules_count << '/'
             << comparison.source.scalar_count << '/'
             << comparison.source.dead_moments << '\n';
        cout << "Target moments/rules/scalars/dead: "
             << comparison.target.moments_count << '/'
             << comparison.target.rules_count << '/'
             << comparison.target.scalar_count << '/'
             << comparison.target.dead_moments << '\n';
        cout << "Radial compatibility: "
             << (radial.compatible ? "yes" : "no")
             << (radial.reason.empty() ? "" : " (" + radial.reason + ")") << '\n';
        cout << "Species-order compatibility: "
             << (index_checked ? (index_compatible ? "yes" : "no")
                               : "not checked")
             << '\n';
        cout << "Overall equivalent: " << (equivalent ? "yes" : "no") << '\n';
        if (!mapped_output.empty())
            cout << "Mapped potential: " << mapped_output << '\n';
        for (const auto &warning : warnings) cerr << "Warning: " << warning << '\n';

        if (!report_file.empty()) {
            atomic_write_text(
                report_file,
                comparison_json(source_file, target_file, equivalence_mode,
                                comparison, radial, index_checked,
                                index_compatible, sidecar, equivalent,
                                mapped_output, warnings),
                overwrite);
        }
        return equivalent ? 0 : 1;
    } catch (const exception &error) {
        cerr << "mtpbasis error: " << error.what() << '\n';
        return 2;
    }
}

} // namespace

int mtpbasis(int argc, char *argv[]) {
    return run_mtpbasis(argc, argv);
}
