// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "utils/gputils.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>

namespace {
void command_line(CMDOption::CommandLineOptions &options, std::vector<std::string> args) {
    args.insert(args.begin(), "parameter-test");
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    options.set_helponexit(false);
    options.phrase_options(static_cast<int>(argv.size()), argv.data());
}

template<class T>
void apply(T &value, const std::string &text, int route, int count = 1) {
    GPSettings settings;
    settings.Modify("value", text);
    if (route == 2) {
        ConnectSettings connected;
        connected.Connect(value, "value", "", count);
        connected.ApplySettings(settings);
    } else {
        CMDOption::CommandLineOptions options;
        options.addop(value, "value", "", "", false, count);
        if (route == 1) apply_settings_to_cmdopts(options, settings);
        else {
            std::vector<std::string> args{"--value"};
            if constexpr (Parameters::Container<T>::value) {
                std::istringstream stream(text);
                for (std::string token; stream >> token;) args.push_back(token);
            } else args.push_back(text);
            command_line(options, args);
        }
    }
}

struct SettingsFile {
    std::string path;
    explicit SettingsFile(const std::string &content) {
        char name[] = "/tmp/imrmlp_parameters_XXXXXX";
        const int fd = mkstemp(name);
        if (fd < 0) throw std::runtime_error("cannot create temporary settings");
        close(fd);
        path = name;
        std::ofstream(path) << content;
    }
    ~SettingsFile() { std::filesystem::remove(path); }
};
}

TEST_CASE("boolean grammar agrees across all parameter entry points", "[parameters]") {
    const int route = GENERATE(0, 1, 2);
    const auto text = GENERATE("1", "true", "TRUE", "yes", "Y", "0", "false", "FALSE", "no", "N");
    CAPTURE(route, text);
    const bool expected = std::string(text) == "1" || std::string(text) == "true" ||
                          std::string(text) == "TRUE" || std::string(text) == "yes" || std::string(text) == "Y";
    bool value = !expected;
    apply(value, text, route);
    REQUIRE(value == expected);
}

TEST_CASE("settings snapshots include effective defaults and hidden bound values", "[parameters]") {
    ConnectSettings connected;
    double value = 0.12345678901234567;
    std::vector<double> weights{0.32109876543210987, 2.0};
    std::string empty;
    connected.Connect(value, "value", "", 1, 100);
    connected.Connect(weights, "weights", "", -1);
    connected.Connect(empty, "empty");
    const auto snapshot = connected.CurrentSettings();
    value = 4;
    weights.clear();
    connected.ApplySettings(snapshot);
    REQUIRE(value == 0.12345678901234567);
    REQUIRE(weights == std::vector<double>{0.32109876543210987, 2.0});
    REQUIRE(snapshot.count("empty") == 0);
}

TEST_CASE("invalid boolean words cannot silently disable a parameter", "[parameters]") {
    const int route = GENERATE(0, 1, 2);
    const auto text = GENERATE("garbage", "tomato", "yesplease", "falsehood", "10");
    CAPTURE(route, text);
    bool value = true;
    REQUIRE_THROWS(apply(value, text, route));
    REQUIRE(value);
}

TEST_CASE("unsigned values are checked against the destination range", "[parameters]") {
    const int route = GENERATE(0, 1, 2);
    const auto text = GENERATE("-1", " -1", "4294967296", "18446744073709551616");
    CAPTURE(route, text);
    unsigned int value = 17;
    REQUIRE_THROWS(apply(value, text, route));
    REQUIRE(value == 17);
}

TEST_CASE("scalar conversion failure preserves the bound value", "[parameters]") {
    const int route = GENERATE(0, 1, 2);
    const auto text = GENERATE("12junk", "1.5", "2147483648", "-2147483649", "word");
    CAPTURE(route, text);
    int value = 7;
    REQUIRE_THROWS(apply(value, text, route));
    REQUIRE(value == 7);
}

TEST_CASE("container conversion failure preserves the complete previous value", "[parameters]") {
    const int route = GENERATE(0, 1, 2);
    const auto text = GENERATE("1 2junk 3", "1 word 3", "1 2147483648 3");
    CAPTURE(route, text);
    std::vector<int> values{7, 8};
    REQUIRE_THROWS(apply(values, text, route, -1));
    REQUIRE(values == std::vector<int>{7, 8});
}

TEST_CASE("fixed container sizes are enforced through every entry point", "[parameters]") {
    const int route = GENERATE(0, 1, 2);
    const auto text = GENERATE("1 2", "1 2 3 4");
    CAPTURE(route, text);
    std::vector<int> values{7, 8, 9};
    REQUIRE_THROWS(apply(values, text, route, 3));
    REQUIRE(values == std::vector<int>{7, 8, 9});
}

TEST_CASE("containers use the same element grammar and replace defaults", "[parameters]") {
    const int route = GENERATE(0, 1, 2);
    std::vector<bool> flags{false};
    apply(flags, "true 0 Yes", route, 3);
    REQUIRE(flags == std::vector<bool>{true, false, true});
    std::vector<float> floats{9};
    apply(floats, "2.5f -1.25 0x1p3", route, 3);
    REQUIRE(floats == std::vector<float>{2.5f, -1.25f, 8.f});
    std::set<unsigned long> ids{9};
    apply(ids, "2 3 2", route, 3);
    REQUIRE(ids == std::set<unsigned long>{2, 3});
}

TEST_CASE("repeated CLI containers commit only after all occurrences validate", "[parameters]") {
    CMDOption::CommandLineOptions options;
    std::vector<int> values{9};
    options.addop(values, "values", "", "", false, 2);
    REQUIRE_THROWS(command_line(options, {"--values", "1", "2", "--values", "3", "bad"}));
    REQUIRE(values == std::vector<int>{9});
    command_line(options, {"--values", "1", "2", "--values", "3", "4"});
    REQUIRE(values == std::vector<int>{1, 2, 3, 4});
}

TEST_CASE("shared bindings preserve CLI precedence across repeated settings applications", "[parameters]") {
    const bool settings_first = GENERATE(false, true);
    int value = 1;
    ConnectSettings connected;
    CMDOption::CommandLineOptions options;
    connected.ConnectWithConsole(options, value, "count");
    GPSettings settings;
    settings.Modify("count", "3");
    if (settings_first) connected.ApplySettings(settings);
    command_line(options, {"--count", "7"});
    apply_settings_to_cmdopts(options, settings);
    connected.ApplySettings(settings);
    REQUIRE(value == 7);
    // Starting a new parse resets the previous invocation's source precedence.
    command_line(options, {});
    connected.ApplySettings(settings);
    REQUIRE(value == 3);
}

TEST_CASE("CLI aliases share the original setting identity", "[parameters]") {
    double tolerance = 1;
    ConnectSettings connected;
    connected.Connect(tolerance, "gtol");
    CMDOption::CommandLineOptions options;
    connected.BindConsole(options, "gtol", "tolerance", "t");
    GPSettings settings;
    settings.Modify("relax:gtol", "0.01");
    auto local = settings.ExtractSubSettings("relax");
    apply_settings_to_cmdopts(options, local);
    REQUIRE(tolerance == Catch::Approx(0.01));
    command_line(options, {"-t", "0.02"});
    connected.ApplySettings(local);
    apply_settings_to_cmdopts(options, local);
    REQUIRE(tolerance == Catch::Approx(0.02));
}

TEST_CASE("copied and reset parsers rebind their own flags", "[parameters]") {
    CMDOption::CommandLineOptions original;
    int value = 1;
    original.addop(value, "value");
    auto copy = original;
    command_line(copy, {"--help", "--value", "7"});
    REQUIRE(copy.displayhelp);
    REQUIRE_FALSE(original.displayhelp);
    REQUIRE(value == 7);
    copy.reset();
    command_line(copy, {"--help"});
    REQUIRE(copy.displayhelp);
    REQUIRE_FALSE(original.displayversion);
    copy.reset();
    REQUIRE_FALSE(copy.displayhelp);
}

TEST_CASE("multiple CLI aliases apply occurrences to one parameter in input order", "[parameters]") {
    bool enabled = false;
    ConnectSettings connected;
    connected.Connect(enabled, "enabled", "", 0);
    CMDOption::CommandLineOptions options;
    connected.BindConsole(options, "enabled", "first");
    connected.BindConsole(options, "enabled", "second");
    command_line(options, {"--first", "--second"});
    REQUIRE_FALSE(enabled);
    REQUIRE(options.was_provided("first"));
    REQUIRE(options.was_provided("second"));
    command_line(options, {"--second=false", "--first"});
    REQUIRE(enabled);
}

TEST_CASE("deferred module shortcuts preserve explicit false and zero values", "[parameters]") {
    bool enabled = true;
    double pressure = -1;
    CMDOption::CommandLineOptions options;
    options.addop(enabled, "enabled").map_setting("enabled", "relax:enabled")
           .addop(pressure, "pressure").map_setting("pressure", "relax:pressure");
    GPSettings settings;
    settings.Modify("relax:enabled", "true");
    settings.Modify("relax:pressure", "3");
    command_line(options, {"--enabled=false", "--pressure=0"});
    options.apply_settings(settings);
    options.apply_overrides(settings);
    bool module_enabled = true;
    double module_pressure = -1;
    ConnectSettings module;
    module.Connect(module_enabled, "enabled");
    module.Connect(module_pressure, "pressure");
    module.ApplySettings(settings.ExtractSubSettings("relax"));
    REQUIRE_FALSE(module_enabled);
    REQUIRE(module_pressure == 0);
    GPSettings later_settings;
    later_settings.Modify("enabled", "true");
    later_settings.Modify("pressure", "9");
    module.ApplySettings(later_settings);
    REQUIRE_FALSE(module_enabled);
    REQUIRE(module_pressure == 0);
}

TEST_CASE("explicit empty CLI strings survive deferred configuration", "[parameters]") {
    std::string value = "initial";
    CMDOption::CommandLineOptions options;
    options.addop(value, "value").map_setting("value", "module:value");
    command_line(options, {"--value", ""});
    GPSettings settings;
    options.apply_overrides(settings);
    std::string module_value = "default";
    ConnectSettings module;
    module.Connect(module_value, "value");
    module.ApplySettings(settings.ExtractSubSettings("module"));
    REQUIRE(module_value.empty());
    std::string alias_value = "default";
    CMDOption::CommandLineOptions alias_options;
    alias_options.addop(alias_value, "alias").map_setting("alias", "module:value");
    alias_options.apply_settings(settings);
    REQUIRE(alias_value.empty());
}

TEST_CASE("deferred mappings keep canonical settings and flat fallback spellings", "[parameters]") {
    const bool canonical = GENERATE(false, true);
    double value = -1;
    CMDOption::CommandLineOptions options;
    options.addop(value, "gtol").map_setting("gtol", "relax:gtol");
    GPSettings settings;
    settings.Modify("gtol", "0.1");
    if (canonical) settings.Modify("relax:gtol", "0.2");
    options.apply_settings(settings);
    options.apply_overrides(settings);
    REQUIRE(value == Catch::Approx(canonical ? 0.2 : 0.1));
    REQUIRE(settings["relax:gtol"] == (canonical ? "0.2" : Parameters::format_value(0.1)));
}

TEST_CASE("empty programmatic settings retain existing defaults", "[parameters]") {
    const int route = GENERATE(1, 2);
    std::string value = "mtp";
    apply(value, "", route);
    REQUIRE(value == "mtp");
}

TEST_CASE("settings retain duplicate entries and error locations through extraction", "[parameters]") {
    SettingsFile file("repeat first\nrepeat second\n[relax]\nmaxiter 12junk\n");
    GPSettings settings(file.path);
    REQUIRE(settings["repeat"] == "first");
    const auto [begin, end] = settings.equal_range("repeat");
    REQUIRE(std::distance(begin, end) == 2);
    int value = 7;
    ConnectSettings connected;
    connected.Connect(value, "maxiter");
    REQUIRE_THROWS_WITH(connected.ApplySettings(settings.ExtractSubSettings("relax")),
                        Catch::Matchers::ContainsSubstring(file.path + ":4 [relax:maxiter]"));
    REQUIRE(value == 7);
}

TEST_CASE("settings preserve whitespace continuation and blank section boundaries", "[parameters]") {
    SettingsFile file("[group]\nvalues 1 \\\n 2 3\n\nname\tvalue with spaces\n");
    GPSettings settings(file.path);
    std::vector<int> values;
    ConnectSettings connected;
    connected.Connect(values, "values", "", 3);
    connected.ApplySettings(settings.ExtractSubSettings("group"));
    REQUIRE(values == std::vector<int>{1, 2, 3});
    REQUIRE(settings["name"] == "value with spaces");
    REQUIRE_FALSE(settings.count("group:name"));
}

TEST_CASE("positional lists and negative values retain unambiguous boundaries", "[parameters]") {
    CMDOption::CommandLineOptions options;
    std::vector<double> values;
    std::string output;
    options.addop(values, "values", "", "", true, -1).addop(output, "output", "", "", true);
    command_line(options, {"-1", "-.5", "-", "result"});
    REQUIRE(values == std::vector<double>{-1, -0.5});
    REQUIRE(output == "result");
    options.reset();
    options.addop(output, "output", "", "", true);
    command_line(options, {"--", "-filename"});
    REQUIRE(output == "-filename");
}
