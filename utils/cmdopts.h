// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "parameter.h"
#include "tools/tools.h"
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

class GPSettings;

extern int options_list_global_level;
extern int omp_num_threads;
// Set the CLI preference and the calling task's OpenMP team-size request.
// Models must not reset that request when preparing their scratch buffers.
void set_omp_num_threads(int n);
int get_omp_max_threads();
void global_omp_init();

namespace CMDOption {
inline constexpr int duplicate_option_warning_level = 100;
using ParseError = Parameters::Error;

// Non-owning parameter views. External targets must outlive the parser.
// Copies keep external bindings shared, but rebind the parser's own help/version flags.
class CommandLineOptions {
    struct Option {
        std::shared_ptr<Parameters::Binding> binding;
        std::string shortname;
        bool positional = false;
        std::string setting_key;
    };
    std::map<std::string, Option> ops;
    std::map<std::string, std::string> maps2l;
    std::vector<std::string> positional_ops;
    std::set<std::string> requested_ops, processed_ops;
    void rebind_flags();

public:
    struct SettingsGroup {
        std::string ns, title;
        std::function<void(std::ostream &)> print_fn;
    };
    std::vector<SettingsGroup> settings_groups;
    bool displayhelp = false;
    bool displayversion = false;
    bool unknow_options_raise_error = true;
    bool help_on_exit = true;
    std::string usagestr;

    CommandLineOptions();
    CommandLineOptions(const CommandLineOptions &other);
    CommandLineOptions &operator=(const CommandLineOptions &other);
    void reset() { *this = CommandLineOptions(); }

    template<class T>
    CommandLineOptions &addop(T &target, std::string name, std::string shortname = "",
                             std::string help = "", bool positional = false,
                             int nargs = 1, int level = 0) {
        auto binding = Parameters::bind(target, name, std::move(help), nargs, level);
        return add_binding(std::move(binding), std::move(name), std::move(shortname), positional);
    }
    CommandLineOptions &add_binding(std::shared_ptr<Parameters::Binding> binding,
                                    std::string name, std::string shortname = "",
                                    bool positional = false);
    CommandLineOptions &add_dummy(std::string name = "dummy", std::string shortname = "");
    CommandLineOptions &add_request(const std::string &name);
    CommandLineOptions &check();
    CommandLineOptions &set_helponexit(bool state) { help_on_exit = state; return *this; }
    CommandLineOptions &set_unknownexit(bool state) { unknow_options_raise_error = state; return *this; }
    CommandLineOptions &set_usage(std::string value) { usagestr = std::move(value); return *this; }
    CommandLineOptions &attach_settings(std::string ns, std::string title,
                                        std::function<void(std::ostream &)> printer);
    void modify_level(const std::string &name, int level);
    CommandLineOptions &map_setting(const std::string &name, std::string setting_key);
    bool was_provided(const std::string &name) const { return processed_ops.count(name) != 0; }
    const std::set<std::string> &provided_options() const { return processed_ops; }

    void phrase_options(int argc, char *argv[], int argoffset = 1);
    void apply_settings(const GPSettings &settings);
    // Forward explicit CLI shortcuts to modules constructed after parsing.
    void apply_overrides(GPSettings &settings) const;
    std::ostream &showhelp(std::ostream &os);
    std::ostream &showversion(std::ostream &os);
};
extern CommandLineOptions global_cmdopts;
} // namespace CMDOption
