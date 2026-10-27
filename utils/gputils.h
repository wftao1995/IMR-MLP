// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

/**
 * Whitespace-separated settings with colon-delimited namespaces.
 * Comments start with a hash or exclamation mark. A blank line ends a bracket
 * section; duplicate keys retain their first value. Values may continue on the
 * next line with a trailing backslash. See GPSettings::Load for parsing rules.
 */

#include "eigen_core.h"
#include "tools/procstr.h"
#include "cmdopts.h"
#include "comm.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <random>

class GPSettings
{
    struct Origin {
        std::string location;
        Parameters::Source source = Parameters::Source::Settings;
    };
    std::map<std::string, Origin> locations;
public:
    std::multimap<std::string, std::string> settings;
    static const std::string EMPTY_STRING;

public:
    GPSettings(){};
    GPSettings(std::multimap<std::string, std::string> &&_settings)
        : settings(std::move(_settings)){};
    GPSettings(const std::string filename)
    {
        Load(filename);
    }
    ~GPSettings(){};
    const std::string &operator[](const std::string &key) const
    {
        auto pos = settings.find(key);
        if (pos == settings.end())
            return EMPTY_STRING;
        return pos->second;
    }

    int count(const std::string &key) const {
        return settings.count(key);
    }

    bool empty() const {
        return settings.empty();
    }

    std::string Location(const std::string &key) const {
        auto it = locations.find(key);
        return it == locations.end() ? "settings" : it->second.location;
    }

    Parameters::Source SourceOf(const std::string &key) const {
        auto it = locations.find(key);
        return it == locations.end() ? Parameters::Source::Settings : it->second.source;
    }

    void Modify(const std::string &key, const std::string &val) {
        locations.erase(key);
        auto pos = settings.find(key);
        if (pos == settings.end()) {
            settings.insert(std::make_pair(key, val));
        } else pos->second = val;
    }

    void OverrideFromCommandLine(const std::string &key, const std::string &value) {
        Modify(key, value);
        locations[key] = {"command line", Parameters::Source::CommandLine};
    }

    const std::pair<std::multimap<std::string, std::string>::const_iterator, std::multimap<std::string, std::string>::const_iterator>
    equal_range(const std::string &key) const
    {
        return settings.equal_range(key);
    }

    GPSettings ExtractSubSettings(std::string key) const;
    void Load(const std::string &filename);
    void Print();
};

// Non-owning views of registered parameters. Copying this object shares bindings,
// not the target variables; an owning derived class must register its own members
// again if it copies them. Connect() replaces an existing registration for that use.
class ConnectSettings {
    std::map<std::string, std::shared_ptr<Parameters::Binding>> parameters;
public:
    template<class T>
    std::shared_ptr<Parameters::Binding> Connect(T &target, const std::string &name,
                                                const std::string &help = "", int narg = 1, int lev = 0) {
        auto binding = Parameters::bind(target, name, help, narg, lev);
        parameters[name] = binding;
        return binding;
    }

    template<class T>
    void ConnectWithConsole(CMDOption::CommandLineOptions &cmdopts, T &target, const std::string &name,
                            const std::string &shortname = "", const std::string &help = "",
                            int narg = 1, int lev = 0) {
        auto binding = Connect(target, name, help, narg, lev);
        cmdopts.add_binding(std::move(binding), name, shortname);
    }

    // Expose an existing setting under an optional CLI alias. Both names share
    // source tracking, so a later settings application cannot overwrite the CLI.
    void BindConsole(CMDOption::CommandLineOptions &cmdopts, const std::string &setting,
                     const std::string &name = "", const std::string &shortname = "") {
        cmdopts.add_binding(parameters.at(setting), name.empty() ? setting : name, shortname);
    }

    struct SettingInfo {
        std::string name, type, help;
        int nargs, level;
    };
    std::vector<SettingInfo> GetSettings() const;
    /// Snapshot current bound values, including defaults and hidden settings.
    GPSettings CurrentSettings() const;
    bool HasSettings() const { return !parameters.empty(); }
    void ListSettings(std::ostream &os, const std::string &ns = "") const;
    void ApplySettings(const GPSettings &settings, const std::set<std::string> &skip = {});
    void PrintSettings();
    void ListSettings();
};

// #define IMRMLP_DEBUG_LEVEL 0

class GPException : public std::exception
{
public:
    std::string message;
    GPException(const std::string &_message) : message(_message){};

    virtual const char *What() const
    {
        return message.c_str();
    }

    const char *what() const noexcept override
    {
        return message.c_str();
    }
};

class GPError {
    ;
public:
    GPError(){};
};

#define GPERROR(errstr) throw GPException((std::string) "ERROR: " + errstr +            \
                                          " \n    Thrown by function " + __FUNCTION__ + \
                                          " \n    In source file: " + __FILE__ +        \
                                          " \n    Line: " + std::to_string(static_cast<long long>(__LINE__)) + "\n")

#define ARGERROR(errstr) throw std::invalid_argument((std::string) "ERROR: " + errstr +            \
                                                     " \n    Thrown by function " + __FUNCTION__ + \
                                                     " \n    In source file: " + __FILE__ +        \
                                                     " \n    Line: " + std::to_string(static_cast<long long>(__LINE__)) + "\n")

extern std::ostream *global_ostream;
extern std::ostream *global_debugstream;
extern int output_ostream_printinfo_level;
extern int output_debug_printinfo_level;

/// RAII guard that replaces a global stream pointer and restores it on destruction.
/// The replacement stream is deleted on destruction (ownership is taken).
class ScopedStreamRedirect {
    std::ostream *&target_;
    std::ostream *saved_;
    std::ostream *replacement_;
public:
    ScopedStreamRedirect(std::ostream *&target, std::ostream *replacement)
        : target_(target), saved_(target), replacement_(replacement)
    {
        target_ = replacement;
    }
    ~ScopedStreamRedirect() {
        target_ = saved_;
        if (replacement_ && replacement_ != &std::cerr && replacement_ != &std::cout)
            delete replacement_;
    }
    ScopedStreamRedirect(const ScopedStreamRedirect &) = delete;
    ScopedStreamRedirect &operator=(const ScopedStreamRedirect &) = delete;
};

template<typename T>
void ScreenMessage(const T &str) {
    std::cout << str << std::endl;
}

template<typename T>
void Message(const T &str) {
    if (global_ostream == nullptr && mpi.rank == 0) {
        std::cout << str << std::endl;
    } else if (global_ostream != nullptr) {
        *global_ostream << str << std::endl;
    }
}

template<typename T>
void LogMessage(const T &str) {
    if (global_ostream == nullptr && mpi.rank == 0) {
        std::cout << str << '\n';
    } else if (global_ostream != nullptr) {
        *global_ostream << str << '\n';
    }
}

template<typename T>
void Warning(const T &str) {
    if (global_debugstream == nullptr && mpi.rank == 0) {
        std::cerr << str << std::endl;
    } else if (global_debugstream != nullptr) {
        *global_debugstream << str << std::endl;
    }
}

template<typename G>
void DebugWrite(const G &str) {
    if (global_debugstream != nullptr)
        *global_debugstream << str << std::endl;
}
template<typename G, typename F, typename L>
void DebugWrite(const G &str, const F &file, const L &line) {
    if (global_debugstream != nullptr)
        *global_debugstream << file << "  " << line << " : " << str << std::endl;
}
void DebugPutStr(const std::string &str);
// Changes output stream for Message() and Warning(). If os==nullptr all warnings and messages will be suppressed. Returns pointer to previos stream
std::ostream *SetStreamForOutput(std::ostream *_p_os);
std::ostream *SetStreamForDebug(std::ostream *_p_os);

void apply_settings_to_cmdopts(CMDOption::CommandLineOptions &opts, const GPSettings &settings);

struct global_rng
{
    std::random_device rd;
    std::mt19937 mtd;
    std::default_random_engine fastrng;

    global_rng()
    {
        mtd.seed(rd());
        fastrng.seed(rd());
    }

    void reinit()
    {
        mtd.seed(rd());
        fastrng.seed(rd());
    }

    void seed(unsigned int seeder)
    {
        mtd.seed(seeder);
        fastrng.seed(seeder);
    }
};

extern thread_local global_rng RNG;

#ifdef IMRMLP_MPI
void mpi_merge_map(std::map<std::string, int> &target);
void mpi_merge_map(std::map<int, int> &target);
#endif

bool safe_isclose(double a, double b, double rel_tol = 1e-9, double abs_tol = 0.0);
