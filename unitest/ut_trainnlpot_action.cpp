#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "utils/gputils.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

namespace {
struct Workspace {
    std::filesystem::path original = std::filesystem::current_path();
    std::filesystem::path directory;
    std::ostringstream debug;
    std::ostream *previous_debug;

    Workspace() {
        char name[] = "/tmp/imrmlp_trainnlpot_action_XXXXXX";
        const char *created = mkdtemp(name);
        if (!created) throw std::runtime_error("Cannot create trainnlpot test directory");
        directory = created;
        std::filesystem::current_path(directory);
        previous_debug = SetStreamForDebug(&debug);
    }
    ~Workspace() {
        SetStreamForDebug(previous_debug);
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(directory);
    }
    int run(std::vector<std::string> arguments) {
        CMDOption::global_cmdopts.reset();
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        return command_maps.at("trainnlpot").first(argv.size(), argv.data());
    }
    std::map<std::string, std::string> files() const {
        std::map<std::string, std::string> result;
        for (const auto &entry : std::filesystem::directory_iterator(directory)) {
            std::ifstream input(entry.path());
            result[entry.path().filename().string()] = std::string(
                std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        }
        return result;
    }
};
}

TEST_CASE("trainnlpot rejects an empty action before input or output", "[apps][trainnlpot]") {
    const std::string action = GENERATE("", "do-sgd", "do-lbfgs", "do-cg",
         "do-varpro", "calc-grad", "check-grad", "check-grad-all", "test-curv");
    const bool existing_outputs = GENERATE(false, true);
    Workspace work;
    for (const auto *name : {"base.mtp", "train.cfg", "index.map"})
        std::ofstream(name) << "invalid input " << name << '\n';
    if (existing_outputs)
        for (const auto *name : {"final.mtp", "fitted.mtp", "fitted.mtp.pre-refit",
                "DEBUG", "DEBUG_0", "objective.json", "output_loss_info", "output_grad_info"})
            std::ofstream(name) << "previous output " << name << '\n';
    {
        std::ofstream settings("settings.ini");
        settings << "debug true\nfit:debug true\npot:potfile base.mtp\n"
                    "trainset train.cfg\noutput final.mtp\nfit:save_to fitted.mtp\n"
                    "fit:random_init true\nfit:objective-report objective.json\n";
        if (!action.empty()) settings << action << " true\n";
    }
    const auto before = work.files();
    std::vector<std::string> arguments = {"trainnlpot", "settings.ini"};
    if (!action.empty()) arguments.push_back("--" + action + "=false");
    REQUIRE_THROWS_WITH(work.run(arguments),
        Catch::Matchers::ContainsSubstring("select a training or diagnostic action"));
    REQUIRE(work.files() == before);
}

TEST_CASE("trainnlpot refit-only rejects disabled refitting before input", "[apps][trainnlpot]") {
    Workspace work;
    std::ofstream("settings.ini") << "debug true\nfit:linear_refit off\n";
    const auto before = work.files();
    REQUIRE_THROWS_WITH(work.run({"trainnlpot", "settings.ini", "--linear-refit-only"}),
        Catch::Matchers::ContainsSubstring("--linear-refit-only requires fit:linear_refit"));
    REQUIRE(work.files() == before);
}

TEST_CASE("trainnlpot help does not require an action or settings file", "[apps][trainnlpot]") {
    Workspace work;
    REQUIRE(work.run({"trainnlpot", "--help"}) == 0);
    REQUIRE(work.files().empty());
}
