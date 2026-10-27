// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "potentials/potential_mtp.h"
#include "utils/gputils_eigen.h"
#include <filesystem>
#include <fstream>
#include <sstream>

TEST_CASE("MTP exports report open and write failures", "[mtp][io][errors]") {
    Potentials::PotMTP pot("unitest/test-data/potential.mtp");
    std::ofstream("error_parent_file") << "not a directory";
    const std::string missing = "error_parent_file/output.mtp";
    REQUIRE_THROWS_WITH(pot.Save(missing), Catch::Matchers::ContainsSubstring(missing));
    REQUIRE_THROWS_WITH(pot.SaveSpline(missing, 1.0, 1.0), Catch::Matchers::ContainsSubstring(missing));
    REQUIRE_THROWS_WITH(pot.SaveOrdered(missing), Catch::Matchers::ContainsSubstring(missing));

    std::ostringstream failed;
    failed.setstate(std::ios::badbit);
    REQUIRE_THROWS(pot.Save(failed));
    if (std::filesystem::exists("/dev/full")) {
        REQUIRE_THROWS_WITH(pot.Save("/dev/full"), Catch::Matchers::ContainsSubstring("/dev/full"));
        REQUIRE_THROWS_WITH(pot.SaveSpline("/dev/full", 1.0, 1.0), Catch::Matchers::ContainsSubstring("/dev/full"));
    }
    pot.Save("error_roundtrip.mtp");
    Potentials::PotMTP restored("error_roundtrip.mtp");
    REQUIRE(restored.species_count == pot.species_count);
    REQUIRE(restored.alpha_scalar_moments == pot.alpha_scalar_moments);

    // A single ladder level containing the complete model.
    pot.max_level = 1;
    pot.lev_scalar_nodes = {0, pot.alpha_moments_count};
    pot.lev_scalar_count = {pot.alpha_scalar_moments, 0};
    pot.lev_times_count = {pot.alpha_index_times_count, 0};
    if (std::filesystem::exists("/dev/full"))
        REQUIRE_THROWS_WITH(pot.SaveOrdered("/dev/full"), Catch::Matchers::ContainsSubstring("/dev/full"));
    std::filesystem::create_directory("error_ordered.mtp.ladder2");
    REQUIRE_THROWS_WITH(pot.SaveOrdered("error_ordered.mtp"),
                        Catch::Matchers::ContainsSubstring("error_ordered.mtp.ladder2"));
}

TEST_CASE("Matrix and vector exports propagate failures", "[io][errors]") {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd vector = Eigen::VectorXd::Ones(2);
    std::ofstream("error_matrix_parent") << "not a directory";
    std::vector<std::string> paths{"error_matrix_parent/out"};
    if (std::filesystem::exists("/dev/full")) paths.push_back("/dev/full");
    for (const auto &path : paths) {
        REQUIRE_THROWS_WITH(writeMatrixToFile(path, matrix), Catch::Matchers::ContainsSubstring(path));
        REQUIRE_THROWS_WITH(writeMatrixToBinFile(path, matrix), Catch::Matchers::ContainsSubstring(path));
        REQUIRE_THROWS_WITH(writeVectorToBinFile(path, vector), Catch::Matchers::ContainsSubstring(path));
    }
    writeVectorToBinFile("error_vector_roundtrip.bin", vector);
    Eigen::VectorXd restored;
    readVectorFromBinFile("error_vector_roundtrip.bin", restored);
    REQUIRE(restored == vector);
}

TEST_CASE("Species maps accept whitespace and replace previous values", "[gpio][io]") {
    GPIO io;
    std::ofstream("error_space.map") << "H -2.0 3.0\nHe -4.0\n";
    REQUIRE(io.readmap("error_space.map"));
    const auto names = io.getmap();
    const auto values = io.atomic_vals;
    std::ofstream("error_tab.map") << "\tH\t-2.0\t3.0\r\nHe\t-4.0\r\n";
    REQUIRE(io.readmap("error_tab.map"));
    REQUIRE(io.getmap() == names);
    REQUIRE(io.getrevmap().at("He") == 1);
    REQUIRE(io.atomic_vals == values);
    REQUIRE(io.atomradii.at(0) == -2.0);
    std::ofstream("error_symbols.map") << "He\r\n";
    REQUIRE(io.readmap("error_symbols.map"));
    REQUIRE(io.getmap().at(0) == "He");
    REQUIRE(io.atomradii.empty());
    REQUIRE(io.atomic_vals.empty());
    std::ofstream("error_empty.map");
    REQUIRE_FALSE(io.readmap("error_empty.map"));
    REQUIRE_FALSE(io.readmap("error_matrix_parent/missing.map"));
}

TEST_CASE("Debug output may be disabled", "[utils][errors]") {
    auto *previous = SetStreamForDebug(nullptr);
    DebugPutStr("ignored");
    std::ostringstream output;
    SetStreamForDebug(&output);
    DebugPutStr("visible");
    SetStreamForDebug(previous);
    REQUIRE(output.str() == "visible");
}
