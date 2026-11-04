// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
#include <catch2/catch_all.hpp>

#include "potentials/model_elements.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <unistd.h>

namespace
{

std::string write_model(const std::filesystem::path &dir, bool tagged)
{
    std::ifstream input("unitest/test-data/potential.mtp");
    REQUIRE(input.good());
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto begin = text.find("potential_tag =");
    REQUIRE(begin != std::string::npos);
    const auto end = text.find('\n', begin);
    REQUIRE(end != std::string::npos);
    const std::string line =
        tagged ? "potential_tag = {\"schema\":\"gpml.mtp.metadata\",\"version\":1,"
                 "\"type\":\"MTP\",\"model\":\"linear\",\"elements\":[\"Al\",\"Ni\"]}"
               : "potential_tag = ";
    text.replace(begin, end - begin, line);
    const auto path = dir / (tagged ? "tagged.mtp" : "legacy.mtp");
    std::ofstream output(path);
    REQUIRE(output.good());
    output << text;
    return path.string();
}

} // namespace

TEST_CASE("probe_model_elements reads MTP declarations", "[potentials][model-elements]")
{
    using namespace Potentials;
    const auto dir = std::filesystem::temp_directory_path() /
                     ("imrmlp-model-elements-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    const auto tagged = write_model(dir, true);
    const auto legacy = write_model(dir, false);

    GPSettings settings;
    settings.Modify("pottype", "mtp");
    settings.Modify("potfile", tagged);
    REQUIRE(probe_model_elements(settings) == std::vector<std::string>{"Al", "Ni"});

    settings.Modify("potfile", legacy);
    REQUIRE(probe_model_elements(settings).empty());

    std::filesystem::remove_all(dir);
}

TEST_CASE("probe_model_elements follows hybrid children", "[potentials][model-elements]")
{
    using namespace Potentials;
    const auto dir = std::filesystem::temp_directory_path() /
                     ("imrmlp-model-elements-hybrid-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    const auto tagged = write_model(dir, true);

    GPSettings settings;
    settings.Modify("pottype", "hybrid");
    settings.Modify("pot1:pottype", "mtp");
    settings.Modify("pot1:potfile", tagged);
    settings.Modify("pot2:pottype", "zbl");
    REQUIRE(probe_model_elements(settings) == std::vector<std::string>{"Al", "Ni"});

    std::filesystem::remove_all(dir);
}
