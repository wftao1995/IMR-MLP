// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "cmdopts.h"
#include "gputils.h"
#include "config/version.h"
#include "comm.h"
#include <stdexcept>

#ifdef USE_OMP
#include <omp.h>
#endif

void set_omp_num_threads(int n) {
    if (n < 1) throw std::invalid_argument("OpenMP thread count must be positive");
    omp_num_threads = n;
#ifdef USE_OMP
    omp_set_num_threads(n);
#endif
}

int get_omp_max_threads() {
#ifdef USE_OMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void global_omp_init() {
    set_omp_num_threads(omp_num_threads);
}

int options_list_global_level = 1;
int omp_num_threads = 4;
CMDOption::CommandLineOptions CMDOption::global_cmdopts;

CMDOption::CommandLineOptions::CommandLineOptions() {
    addop(options_list_global_level, "showhelp", "showhelp", "specify display level", false, 1, 10);
    addop(displayhelp, "help", "h", "Display help info", false, 0, 0);
    addop(displayversion, "version", "ver", "Display version info", false, 0, 10);
    addop(omp_num_threads, "num_threads", "np",
          "Number of openmp threads, note that this is a global parameter, not all program support it");
}

CMDOption::CommandLineOptions::CommandLineOptions(const CommandLineOptions &other) {
    *this = other;
}

CMDOption::CommandLineOptions &CMDOption::CommandLineOptions::operator=(const CommandLineOptions &other) {
    if (this == &other) return *this;
    ops = other.ops;
    maps2l = other.maps2l;
    positional_ops = other.positional_ops;
    requested_ops = other.requested_ops;
    processed_ops = other.processed_ops;
    settings_groups = other.settings_groups;
    displayhelp = other.displayhelp;
    displayversion = other.displayversion;
    unknow_options_raise_error = other.unknow_options_raise_error;
    help_on_exit = other.help_on_exit;
    usagestr = other.usagestr;
    rebind_flags();
    return *this;
}

void CMDOption::CommandLineOptions::rebind_flags() {
    for (const auto &[name, target] : std::map<std::string, bool *>{{"help", &displayhelp}, {"version", &displayversion}}) {
        auto &old = ops.at(name).binding;
        auto replacement = Parameters::bind(*target, old->name, old->help, old->nargs, old->level);
        replacement->source = old->source;
        old = std::move(replacement);
    }
}

CMDOption::CommandLineOptions &CMDOption::CommandLineOptions::add_binding(
    std::shared_ptr<Parameters::Binding> binding, std::string name, std::string shortname, bool positional) {
    if (ops.count(name) || (!shortname.empty() && maps2l.count(shortname))) {
        if (options_list_global_level > duplicate_option_warning_level)
            std::cerr << "Warning: Option " << (ops.count(name) ? name : shortname)
                      << " has been used before, the latter one won't be registered" << std::endl;
        return *this;
    }
    if (!shortname.empty()) maps2l[shortname] = name;
    ops.emplace(name, Option{std::move(binding), std::move(shortname), positional});
    if (positional) positional_ops.push_back(name);
    return *this;
}

CMDOption::CommandLineOptions &CMDOption::CommandLineOptions::add_dummy(std::string name, std::string shortname) {
    return add_binding(nullptr, std::move(name), std::move(shortname));
}

CMDOption::CommandLineOptions &CMDOption::CommandLineOptions::add_request(const std::string &name) {
    if (!ops.count(name)) throw ParseError("ERROR: Option " + name + " not found");
    requested_ops.insert(name);
    return *this;
}

CMDOption::CommandLineOptions &CMDOption::CommandLineOptions::check() {
    bool optional = false;
    for (const auto &name : positional_ops) {
        if (!requested_ops.count(name)) optional = true;
        else if (optional)
            throw ParseError("ERROR: required positional options must precede optional positional options");
    }
    return *this;
}

CMDOption::CommandLineOptions &CMDOption::CommandLineOptions::attach_settings(
    std::string ns, std::string title, std::function<void(std::ostream &)> printer) {
    settings_groups.push_back({std::move(ns), std::move(title), std::move(printer)});
    return *this;
}

void CMDOption::CommandLineOptions::modify_level(const std::string &name, int level) {
    auto it = ops.find(name);
    if (it == ops.end() || !it->second.binding) throw ParseError("Unknown option " + name);
    it->second.binding->level = level;
}

CMDOption::CommandLineOptions &CMDOption::CommandLineOptions::map_setting(
    const std::string &name, std::string setting_key) {
    auto it = ops.find(name);
    if (it == ops.end() || !it->second.binding) throw ParseError("Unknown option " + name);
    it->second.setting_key = std::move(setting_key);
    return *this;
}

namespace {
bool option_token(const std::string &value) {
    if (value.size() < 2 || value[0] != '-') return false;
    // Negative numbers, including -.5, remain values.
    return !std::isdigit(static_cast<unsigned char>(value[1])) && value[1] != '.';
}
}

void CMDOption::CommandLineOptions::phrase_options(int argc, char *argv[], int argoffset) {
    using Parameters::Values;
    std::map<std::string, Parameters::Occurrences> pending;
    std::map<const Parameters::Binding *, std::string> identities;
    std::set<std::string> supplied;
    std::vector<std::string> order, positional;
    auto collect = [&](const std::string &name, Values values) {
        const auto *identity = ops.at(name).binding.get();
        auto [entry, first] = identities.emplace(identity, name);
        const auto &canonical = entry->second;
        if (first) order.push_back(canonical);
        pending[canonical].push_back(std::move(values));
        supplied.insert(name);
    };
    bool positional_only = false;
    for (int i = argoffset; i < argc; ++i) {
        std::string token = argv[i];
        if (token == "--" && !positional_only) { positional_only = true; continue; }
        if (positional_only || !option_token(token)) { positional.push_back(token); continue; }
        std::string name, inline_value;
        if (token.starts_with("--")) {
            const auto equals = token.find('=');
            name = token.substr(2, equals == std::string::npos ? equals : equals - 2);
            if (equals != std::string::npos) inline_value = token.substr(equals + 1);
        } else {
            const auto shortname = token.substr(1);
            auto it = maps2l.find(shortname);
            if (it != maps2l.end()) name = it->second;
        }
        auto found = ops.find(name);
        if (found == ops.end()) {
            if (unknow_options_raise_error) throw ParseError("Unknown option " + token);
            continue;
        }
        const auto &binding = found->second.binding;
        if (!binding) throw ParseError("Reserved option " + token + " cannot be used");
        Values values;
        if (!inline_value.empty()) values.push_back(std::move(inline_value));
        const int count = binding->nargs;
        if (count == 0) {
            if (!binding->boolean) throw ParseError("Option " + name + " expects a boolean flag");
        } else if (count > 0) {
            while (values.size() < static_cast<std::size_t>(count) && i + 1 < argc) {
                if (option_token(argv[i + 1])) break;
                values.emplace_back(argv[++i]);
            }
            if (values.size() != static_cast<std::size_t>(count))
                throw ParseError("Missing value for option --" + name + ": expected " + std::to_string(count) +
                                 ", received " + std::to_string(values.size()));
        } else if (count == -1) {
            while (i + 1 < argc && !option_token(argv[i + 1])) {
                if (std::string(argv[i + 1]) == "-") { ++i; break; }
                values.emplace_back(argv[++i]);
            }
        } else throw ParseError("Invalid argument count for option " + name);
        collect(name, std::move(values));
    }

    std::size_t cursor = 0;
    for (std::size_t p = 0; p < positional_ops.size(); ++p) {
        const auto &name = positional_ops[p];
        if (identities.count(ops.at(name).binding.get())) continue;
        while (cursor < positional.size() && positional[cursor] == "-") ++cursor;
        if (cursor == positional.size()) break;
        const auto &binding = ops.at(name).binding;
        std::size_t count = 0;
        if (binding->nargs > 0) count = binding->nargs;
        else if (binding->nargs == -1) {
            auto separator = std::find(positional.begin() + cursor, positional.end(), "-");
            if (separator != positional.end()) count = separator - (positional.begin() + cursor);
            else {
                std::size_t reserved = 0;
                for (std::size_t j = p + 1; j < positional_ops.size(); ++j) {
                    if (!identities.count(ops.at(positional_ops[j]).binding.get()))
                        reserved += std::max(1, ops.at(positional_ops[j]).binding->nargs);
                }
                if (positional.size() - cursor > reserved) count = positional.size() - cursor - reserved;
            }
        } else throw ParseError("Invalid argument count for positional option " + name);
        if (count == 0 || count > positional.size() - cursor)
            throw ParseError("Missing value for positional option " + name);
        collect(name, Values(positional.begin() + cursor, positional.begin() + cursor + count));
        cursor += count;
    }
    while (cursor < positional.size() && positional[cursor] == "-") ++cursor;
    if (cursor != positional.size()) throw ParseError("ERROR: extra positional options given");
    if (!pending.count("help") && !pending.count("version")) {
        for (const auto &name : requested_ops)
            if (!identities.count(ops.at(name).binding.get()))
                throw ParseError("ERROR: must give " + name + " in command-line options");
    }

    // Prepare complete parameter values before any writes. A rejected scalar or
    // container leaves its target unchanged, including repeated CLI occurrences.
    std::vector<Parameters::Commit> commits;
    for (const auto &name : order)
        commits.push_back(ops.at(name).binding->prepare(pending.at(name), Parameters::Source::CommandLine,
                                                      "command line --" + name));
    for (auto &[name, option] : ops)
        if (option.binding) option.binding->source = Parameters::Source::Default;
    for (auto &commit : commits) if (commit) commit();
    processed_ops = std::move(supplied);

    if (help_on_exit && was_provided("help")) {
        if (mpi.rank == 0) showhelp(std::cout);
        std::exit(0);
    }
    if (was_provided("version")) {
        if (mpi.rank == 0) showversion(std::cout);
        std::exit(0);
    }
    global_omp_init();
}

void CMDOption::CommandLineOptions::apply_settings(const GPSettings &settings) {
    for (auto &[name, option] : ops) {
        if (!option.binding || was_provided(name)) continue;
        const auto &canonical = option.setting_key.empty() ? option.binding->name : option.setting_key;
        const auto &key = (!settings[canonical].empty() ||
                          settings.SourceOf(canonical) == Parameters::Source::CommandLine) ? canonical : name;
        if (!settings[key].empty() || settings.SourceOf(key) == Parameters::Source::CommandLine)
            option.binding->apply_setting(settings[key], settings.Location(key), settings.SourceOf(key));
    }
    global_omp_init();
}

void CMDOption::CommandLineOptions::apply_overrides(GPSettings &settings) const {
    for (const auto &[name, option] : ops) {
        if (option.setting_key.empty()) continue;
        // The flat settings spelling remains a fallback for existing input files.
        if (was_provided(name))
            settings.OverrideFromCommandLine(option.setting_key, option.binding->format());
        else if (option.binding->source == Parameters::Source::Settings &&
                 settings[option.setting_key].empty() && !settings[name].empty())
            settings.Modify(option.setting_key, option.binding->format());
    }
}

std::ostream &CMDOption::CommandLineOptions::showhelp(std::ostream &os) {
    os << "Usage: \nInput format: commands [positional args] -short_name [var] --long_name [var]\n"
          "Options: longname shortname vars\n";
    if (!usagestr.empty()) os << usagestr;
    os << "Command line options:\n";
    for (const auto &[name, option] : ops) {
        const auto &binding = option.binding;
        if (!binding || option.positional || binding->level > options_list_global_level) continue;
        os << "\t--" << name << " ";
        if (!option.shortname.empty()) os << "-" << option.shortname;
        os << " [";
        if (binding->nargs == 0) os << "noarg";
        else {
            os << binding->type;
            if (binding->nargs > 1) os << ' ' << binding->nargs << " args";
            else if (binding->nargs == -1) os << " n args";
        }
        os << "]: " << binding->help << '\n';
    }
    os << "Positional args, use '-' to separate if narg is arbitrary:\n";
    for (const auto &name : positional_ops) {
        const auto &option = ops.at(name);
        const auto &binding = option.binding;
        os << '\t' << name << ' ';
        if (!option.shortname.empty()) os << '-' << option.shortname;
        os << " [" << binding->type;
        if (binding->nargs > 1) os << ' ' << binding->nargs << " args";
        else if (binding->nargs == -1) os << " n args";
        os << "]: " << binding->help << '\n';
    }
    if (!settings_groups.empty()) {
        os << "\nSettings file options:\n"
              "  (These keys can be used in a settings file with the shown namespace prefix;\n"
              "   CLI equivalents without the prefix are listed in 'Command line options' above)\n";
        for (const auto &group : settings_groups) {
            os << "\n  ══ " << group.title;
            if (!group.ns.empty()) os << " (" << group.ns << ")";
            os << " ══\n";
            if (group.print_fn) group.print_fn(os);
        }
    }
    return os;
}

std::ostream& CMDOption::CommandLineOptions::showversion(std::ostream& os)
{
    os << "Version:    " << PROJECT_VERSION << "." << GIT_COMMIT_ID << std::endl;
    os << "Target:     " << BUILD_TARGET    << std::endl;
    os << "Date:       " << BUILD_DATE      << std::endl;
    std::string compiler;
    #if defined(__clang__)
    compiler = "Clang " + std::string(__clang_version__);
    #elif defined(__GNUC__)
    compiler = "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
    #elif defined(_MSC_VER)
    compiler = "MSVC " + std::to_string(_MSC_VER);
    #else
    compiler = "Unknown Compiler";
    #endif
    os << "Compiler:   " << compiler << std::endl;
    std::string mpi_enables = "No\n";
    std::string omp_enables = "No\n";

    #ifdef IMRMLP_MPI
    char version_str[MPI_MAX_LIBRARY_VERSION_STRING];
    int version_len;
    MPI_Get_library_version(version_str, &version_len);
    mpi_enables = std::string(version_str, version_len);
    #endif
    #ifdef USE_OMP
    omp_enables = "Yes\n";
    #endif
    os << "MPI:        " << mpi_enables;
    os << "OpenMP:     " << omp_enables << std::endl;
    return os;
}
