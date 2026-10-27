// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "LAMMPS/USER-MLIP/mlip_legacy_reader.h"
#include <sstream>

TEST_CASE("Legacy LAMMPS header parser reads normal tracked MTP metadata", "[mtp][reader]") {
    // Header copied from the tracked test-only fixture
    // LAMMPS/USER-MLIP/transport/tests/simple_analytic.mtp.
    // Keep this regression independent of private test data and the test cwd.
    std::istringstream input(R"mtp(MTP
version = 1.1.0
potential_name = analytic_interface_test
species_count = 1
potential_tag = test-only
radial_basis_type = RBChebyshev
min_dist = 0.0
max_dist = 5.0
radial_basis_size = 2
)mtp");
    std::string token;
    while (input >> token && token != "species_count") {}
    REQUIRE(token == "species_count");
    std::string assignment;
    int species = 0;
    REQUIRE(bool(input >> assignment >> species));
    REQUIRE(assignment == "=");
    REQUIRE(species == 1);
    REQUIRE(MLIPLegacyReader::read_radial_basis_type(input) == "RBChebyshev");
    double value = 0;
    REQUIRE(bool(input >> token >> assignment >> value));
    REQUIRE(token == "min_dist");
    REQUIRE(value == 0.0);
    REQUIRE(bool(input >> token >> assignment >> value));
    REQUIRE(token == "max_dist");
    REQUIRE(value == 5.0);
}

TEST_CASE("Legacy LAMMPS MTP headers retain the radial basis and following fields", "[mtp][reader]") {
    for (const std::string tag : {std::string{}, std::string{"potential_tag = reader-test\n"},
                                 std::string{"potential_tag = {\"type\": \"MTP\", \"elements\": [\"Al\"]}\n"}}) {
      for (const std::string separator : {" = ", " =", "\t=\t"}) {
        INFO("Optional tag: " << tag);
        INFO("Assignment separator: " << separator);
        std::istringstream input(tag +
            "radial_basis_type" + separator + "RBChebyshev\nmin_dist = 1.6\nmax_dist = 5.0\n");
        REQUIRE(MLIPLegacyReader::read_radial_basis_type(input) == "RBChebyshev");
        std::string field, assignment;
        double value = 0;
        REQUIRE(bool(input >> field >> assignment >> value));
        REQUIRE(field == "min_dist");
        REQUIRE(assignment == "=");
        REQUIRE(value == Catch::Approx(1.6));
        REQUIRE(bool(input >> field >> assignment >> value));
        REQUIRE(field == "max_dist");
        REQUIRE(assignment == "=");
        REQUIRE(value == 5.0);
      }
    }
}

TEST_CASE("Legacy LAMMPS readers reject removed Yukawa sections before consuming their payload", "[mtp][reader][input]") {
    for (const std::string section : {"yuk_coeffs", "yuk3_coeffs"}) {
        for (const std::string tag : {std::string{}, std::string{"potential_tag = old-model\n"}}) {
            INFO("Removed section: " << section << "; tag: " << tag);
            std::istringstream input(tag + section +
                "\n0 0 1.0 0.8 5.0\nradial_basis_type = RBChebyshev\n");
            REQUIRE_THROWS_WITH(MLIPLegacyReader::read_radial_basis_type(input),
                               Catch::Matchers::ContainsSubstring("have been removed and are unsupported"));
            std::string next;
            REQUIRE(bool(input >> next));
            REQUIRE(next == "0");
        }
    }
}

TEST_CASE("Legacy LAMMPS header parser reports truncated or unsupported headers", "[mtp][reader][input]") {
    for (const std::string text : {"", "potential_tag = sample\n", "unknown = 1\n",
                                  "radial_basis_type", "radial_basis_type =", "radial_basis_type : RBChebyshev"}) {
        INFO("Invalid header: " << text);
        std::istringstream input(text);
        REQUIRE_THROWS_AS(MLIPLegacyReader::read_radial_basis_type(input), std::runtime_error);
    }
    std::istringstream input("residual_cosplus\n");
    REQUIRE_THROWS_WITH(MLIPLegacyReader::read_radial_basis_type(input),
                       Catch::Matchers::ContainsSubstring("does not support residuals"));
}
