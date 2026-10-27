// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "apps/mlpphdisp_utils.h"

#include <sstream>

TEST_CASE("phonopyfc finds required YAML fields", "[phonopyfc]")
{
    std::istringstream input(
        "phonopy:\n"
        "  version: test\n"
        "supercell:\n");
    std::string line;

    phonopyfc_detail::find_required_yaml_field(
        input, line, "supercell:", "input.yaml");

    REQUIRE(line == "supercell:");
}

TEST_CASE("phonopyfc rejects a missing required YAML field", "[phonopyfc]")
{
    std::istringstream input(
        "phonopy:\n"
        "  version: test\n");
    std::string line;

    REQUIRE_THROWS_WITH(
        phonopyfc_detail::find_required_yaml_field(
            input, line, "supercell:", "input.yaml"),
        Catch::Matchers::ContainsSubstring(
            "required field 'supercell:' not found"));
}

TEST_CASE("phonopyfc rejects EOF while waiting for a YAML record", "[phonopyfc]")
{
    std::istringstream input;
    std::string line;

    REQUIRE_THROWS_WITH(
        phonopyfc_detail::read_required_yaml_line(
            input, line, "displacement_ids:", "input.yaml"),
        Catch::Matchers::ContainsSubstring(
            "expected 'displacement_ids:' before end"));
}
