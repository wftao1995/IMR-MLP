// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "active_learn/al_potential.h"
#include <filesystem>
#include <set>

static Atoms candidate(int id, bool identical)
{
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * (10 + (identical ? 0 : id));
    atom.pos_ = {{1, 1, 1}};
    atom.chemmap_ = {0};
    atom.features_["audit_id"] = std::to_string(id);
    return atom;
}

TEST_CASE("AL FPS selects distinct candidates including zero-distance ties", "[al][fps]")
{
    const bool identical = GENERATE(false, true);
    AL::ALPotential al;
    for (int i = 0; i < 4; ++i) al.add_to_pool(candidate(i, identical));
    al.finalize(3, "al-fps.xyz");
    REQUIRE(al.pool_size() == 3);
    std::set<std::string> ids;
    for (const auto &atom : al.pool()) ids.insert(atom.features_.at("audit_id"));
    REQUIRE(ids.size() == 3);
    REQUIRE(al.pool()[0].features_.at("audit_id") == "0");
    if (!identical) REQUIRE(al.pool()[1].features_.at("audit_id") == "3");
    std::filesystem::remove("al-fps.xyz");
}

TEST_CASE("AL export failures throw", "[al][errors]")
{
    AL::ALPotential al;
    al.add_to_pool(candidate(0, true));
    SECTION("missing directory") {
        REQUIRE_THROWS(al.finalize(1, "al-missing-directory/out.xyz"));
    }
    SECTION("write failure") {
        if (std::filesystem::exists("/dev/full"))
            REQUIRE_THROWS(al.finalize(1, "/dev/full"));
    }
}

TEST_CASE("AL disabled sparsification clears the pool", "[al]")
{
    AL::ALPotential al;
    al.add_to_pool(candidate(0, true));
    REQUIRE_NOTHROW(al.finalize(-1, "unused.xyz"));
    REQUIRE(al.pool_size() == 0);
}
