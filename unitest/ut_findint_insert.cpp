#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include "tools/chem.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
const std::string host_text = "1\nLattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3\nH 0 0 0\n";

std::string contents(const std::string &path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

struct InsertWorkspace {
    std::filesystem::path original = std::filesystem::current_path();
    std::filesystem::path directory;
    InsertWorkspace() {
        char name[] = "/tmp/imrmlp_findint_insert_XXXXXX";
        const char *created = mkdtemp(name);
        if (!created) throw std::runtime_error("Cannot create findint test directory");
        directory = created;
        std::filesystem::current_path(directory);
        std::ofstream("host.xyz") << host_text;
    }
    ~InsertWorkspace() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(directory);
    }
    int run(const std::string &element, bool sites_only) {
        CMDOption::global_cmdopts.reset();
        std::vector<std::string> args{"findint", "--input=host.xyz", "--input-format=extxyz",
            "--output=sites.xyz", "--output-format=extxyz", "--top=1"};
        if (!element.empty()) args.push_back("--insert=" + element);
        if (sites_only) args.push_back("--sites-only");
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return command_maps.at("findint").first(argv.size(), argv.data());
    }
};
}

TEST_CASE("Findint rejects unknown insertion elements before opening output", "[apps][findint][insert]") {
    const bool sites_only = GENERATE(false, true);
    const bool existing_target = GENERATE(false, true);
    InsertWorkspace work;
    const std::string sentinel = "existing target sentinel\n";
    if (existing_target) std::ofstream("sites.xyz") << sentinel;
    const auto map_size = Chemicals::atomic_numbers.size();
    REQUIRE(work.run("NotAnElement", sites_only) != 0);
    REQUIRE(Chemicals::atomic_numbers.size() == map_size);
    REQUIRE_FALSE(Chemicals::atomic_numbers.contains("NotAnElement"));
    if (existing_target) REQUIRE(contents("sites.xyz") == sentinel);
    else REQUIRE_FALSE(std::filesystem::exists("sites.xyz"));
    REQUIRE(contents("host.xyz") == host_text);
}

TEST_CASE("Findint resolves valid and unspecified insertion species in both outputs", "[apps][findint][insert]") {
    const bool sites_only = GENERATE(false, true);
    const std::string element = GENERATE(std::string("C"), std::string("Og"), std::string(""));
    const int expected = element == "C" ? 6 : element == "Og" ? 118 : 1;
    InsertWorkspace work;
    REQUIRE(work.run(element, sites_only) == 0);
    GPIO io;
    Atoms result;
    REQUIRE(io.read_one(result, "sites.xyz", "extxyz"));
    REQUIRE(result.size() == (sites_only ? 1 : 2));
    REQUIRE(result.chemmap_.back() == expected);
    REQUIRE(result.cell_.isApprox(4 * Eigen::Matrix3d::Identity()));
    REQUIRE(result.pos_.back().isApprox(Eigen::RowVector3d(2., 2., 2.)));
    if (!sites_only) {
        REQUIRE(result.chemmap_.front() == 1);
        REQUIRE(result.pos_.front().isZero());
    }
    REQUIRE(contents("host.xyz") == host_text);
}
