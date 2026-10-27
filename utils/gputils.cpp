// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "gputils.h"
#include "utils/compressed_stream.h"
#include <fstream>
#include <iostream>
#include <set>
#include "comm.h"
#include "tools/procstr.h"
#include <cstring>

const std::string GPSettings::EMPTY_STRING = "";
std::ostream *global_ostream = nullptr;
std::ostream *global_debugstream = &std::cerr;
int output_ostream_printinfo_level = 1;
int output_debug_printinfo_level = 1;

void GPSettings::Load(const std::string &filename)
{
    auto stream = open_ifile(filename);
    if (!stream) GPERROR("Cannot open file: " + filename);
    Message("Loading settings from file " + filename);
    std::string line, section_prefix;
    std::size_t line_number = 0;
    while (std::getline(*stream, line)) {
        ++line_number;
        trim_all(line);
        const bool blank = line.empty();
        trim_comment(line, "#!");
        trim_all(line);
        if (line.empty()) {
            if (blank) section_prefix.clear();
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            section_prefix = line.substr(1, line.size() - 2);
            if (!section_prefix.empty()) section_prefix += ':';
            continue;
        }
        const auto first_line = line_number;
        while (!line.empty() && line.back() == '\\' &&
               (line.size() == 1 || std::isspace(static_cast<unsigned char>(line[line.size() - 2])))) {
            line.pop_back();
            std::string continuation;
            if (!std::getline(*stream, continuation))
                GPERROR(filename + ":" + std::to_string(first_line) + ": unfinished continuation");
            ++line_number;
            trim_comment(continuation, "#!");
            trim_all(continuation);
            line += " " + continuation;
            trim_all(line);
        }
        const auto separator = line.find_first_of(" \t");
        if (separator == std::string::npos)
            GPERROR(filename + ":" + std::to_string(first_line) + ": missing setting value: " + line);
        const auto key = section_prefix + line.substr(0, separator);
        auto value = line.substr(separator + 1);
        trim_all(value);
        if (value.empty()) GPERROR(filename + ":" + std::to_string(first_line) + ": missing value for " + key);
        if (!settings.count(key)) locations[key] = {filename + ":" + std::to_string(first_line) + " [" + key + "]"};
        settings.emplace(key, std::move(value));
    }
    if (stream->bad()) GPERROR("Failed to read settings file: " + filename);
}

void GPSettings::Print()
{
    for (auto &stng : settings)
        if (!stng.second.empty())
            Message('\t' + stng.first + " = " + stng.second);
}

GPSettings GPSettings::ExtractSubSettings(std::string key) const
{
    key += ":";
    GPSettings result;
    for (const auto &[name, value] : settings) {
        if (name.starts_with(key)) {
            const auto local_name = name.substr(key.size());
            result.settings.emplace(local_name, value);
            auto location = locations.find(name);
            if (location != locations.end()) result.locations.emplace(local_name, location->second);
        }
    }
    return result;
}

void ConnectSettings::ApplySettings(const GPSettings &settings, const std::set<std::string> &skip)
{
    for (const auto &[name, binding] : parameters) {
        // Existing programmatic callers use an empty setting as "unspecified".
        if (skip.count(name) || !settings.count(name) ||
            (settings[name].empty() && settings.SourceOf(name) != Parameters::Source::CommandLine)) continue;
        try {
            binding->apply_setting(settings[name], settings.Location(name), settings.SourceOf(name));
        } catch (const Parameters::Error &error) {
            GPERROR(error.what());
        }
    }
}

void ConnectSettings::PrintSettings()
{
    for (const auto &[name, binding] : parameters) {
        if (binding->level > options_list_global_level) continue;
        const auto value = binding->format();
        if (!value.empty()) Message('\t' + name + " = " + value);
    }
}

GPSettings ConnectSettings::CurrentSettings() const
{
    GPSettings settings;
    for (const auto &[name, binding] : parameters) {
        const auto value = binding->format();
        if (!value.empty()) settings.Modify(name, value);
    }
    return settings;
}

void ConnectSettings::ListSettings()
{
    for (const auto &setting : GetSettings())
        Message("\t" + setting.name + ": [" + setting.type + "] : " + setting.help);
}

std::vector<ConnectSettings::SettingInfo> ConnectSettings::GetSettings() const
{
    std::vector<SettingInfo> result;
    for (const auto &[name, binding] : parameters) {
        if (binding->level <= options_list_global_level)
            result.push_back({name, binding->type, binding->help, binding->nargs, binding->level});
    }
    return result;
}

void ConnectSettings::ListSettings(std::ostream &os, const std::string &ns) const
{
    for (const auto &setting : GetSettings()) {
        const auto key = ns + setting.name;
        os << "    " << key;
        if (key.size() < 24) os << std::string(24 - key.size(), ' ');
        else os << "  ";
        os << '[' << setting.type << ']';
        if (!setting.help.empty()) os << "  " << setting.help;
        os << '\n';
    }
}

void DebugPutStr(const std::string &str)
{
    if (global_debugstream != nullptr)
        *global_debugstream << str << std::flush;
}

std::ostream *SetStreamForOutput(std::ostream *_p_os)
{
    std::ostream *tmp = global_ostream;
    global_ostream = _p_os;
    return tmp;
}

std::ostream *SetStreamForDebug(std::ostream *_p_os)
{
    std::ostream *tmp = global_debugstream;
    global_debugstream = _p_os;
    return tmp;
}

#ifdef IMRMLP_MPI
/***
 * Merge maps on all mpi ranks
 * TODO: improve this
 * ***/
#define MAX_BUFFER 1000
void mpi_merge_map(std::map<std::string, int> &target)
{
    char pkey[MAX_BUFFER];
    std::map<std::string, int> ret;

    for (int i = 0; i < mpi.size; ++i)
    {
        int gsize = target.size();
        int lsize = target.size();
        if (mpi.rank == i)
            MPI_Bcast(&gsize, 1, MPI_INT, i, mpi.comm);
        auto it = target.begin();
        for (int j = 0; j < gsize; ++j)
        {
            if (mpi.rank == i)
            {
                std::strcpy(pkey, (it->first).c_str());
                MPI_Bcast(pkey, MAX_BUFFER, MPI_CHAR, i, mpi.comm);
                ++it;
            }

            if (ret[pkey] == 0)
            {
                int buffer = target[pkey];
                int sum;
                MPI_Allreduce(&buffer, &sum, 1, MPI_INT, MPI_SUM, mpi.comm);
                ret[pkey] = sum;
            }
        }
    }
    target = ret;
}

void mpi_merge_map(std::map<int, int> &target)
{
    int pkey;
    std::map<int, int> ret;

    for (int i = 0; i < mpi.size; ++i)
    {
        int gsize = target.size();
        int lsize = target.size();
        if (mpi.rank == i)
            MPI_Bcast(&gsize, 1, MPI_INT, i, mpi.comm);
        auto it = target.begin();
        for (int j = 0; j < gsize; ++j)
        {
            if (mpi.rank == i)
            {
                pkey = it->first;
                MPI_Bcast(&pkey, 1, MPI_INT, i, mpi.comm);
                ++it;
            }

            if (ret[pkey] == 0)
            {
                int buffer = target[pkey];
                int sum;
                MPI_Allreduce(&buffer, &sum, 1, MPI_INT, MPI_SUM, mpi.comm);
                ret[pkey] = sum;
            }
        }
    }
    target = ret;
}
#endif

thread_local global_rng RNG;

void readMatrixFromFile(const std::string& filename, Eigen::MatrixXd &matrix) {
    auto file = open_ifile(filename);
    if (!file) return;
    std::string line;

    std::vector<std::vector<double>> matrixData;

    while (std::getline(*file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        matrixData.push_back(row);
    }

    int rows = matrixData.size();
    int cols = matrixData.front().size();
    matrix.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = matrixData[i][j];
        }
    }
}

bool safe_isclose(double a, double b, double rel_tol, double abs_tol)
{
    if (std::isnan(a) || std::isnan(b))
        return false; // NaN
    if (a == b)
        return true; // inf
    return std::fabs(a - b) <= std::max(rel_tol * std::max(std::fabs(a), std::fabs(b)), abs_tol);
}

void apply_settings_to_cmdopts(CMDOption::CommandLineOptions &opts, const GPSettings &settings) {
    opts.apply_settings(settings);
}
