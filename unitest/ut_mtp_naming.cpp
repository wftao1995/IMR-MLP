// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp.h"
#include "potentials/potential_wrapper.h"
#include "train/train_mtp.h"
#include "base/neighborhood.h"
#include <type_traits>
#include <fstream>
#include <sstream>

static_assert(std::is_same_v<Potentials::PotMTP, Potentials::PotMTP>);
static_assert(std::is_same_v<Trainer_MTP, Trainer_MTP>);

TEST_CASE("MTP level metadata errors are catchable", "[mtp][errors]") {
    std::ifstream input("unitest/test-data/potential.mtp");
    REQUIRE(input.good());
    std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const auto coefficients = text.find("moment_coeffs");
    REQUIRE(coefficients != std::string::npos);
    text.resize(text.find('\n', coefficients) + 1);
    text += "max_level = 0\n";
    std::string missing;
    SECTION("missing node counts") { missing = "lev_scalar_nodes"; }
    SECTION("missing scalar counts") {
        text += "lev_scalar_nodes = {0}\n";
        missing = "lev_scalar_count";
    }
    text += "invalid_tag\n";
    std::istringstream malformed(text);
    Potentials::PotMTP potential;
    REQUIRE_THROWS_WITH(potential.Load(malformed), Catch::Matchers::ContainsSubstring(missing));
}

TEST_CASE("Default and explicit MTP types evaluate identical EFS", "[mtp]") {
    using namespace Potentials;
    const std::string path = "unitest/test-data/potential.mtp";
    potential_wrapper reference_pot(path, "mtp");
    GPSettings settings;
    settings.Modify("potfile", path);
    potential_wrapper default_pot(settings);
    REQUIRE(default_pot.pottype == "mtp");
    const std::string type = GENERATE("mtp", "");
    potential_wrapper named(path, type);
    REQUIRE(dynamic_cast<PotMTP *>(named.pot) != nullptr);
    REQUIRE(named.pot->cutoff == reference_pot.pot->cutoff);
    Atoms reference;
    reference.cell_ = Eigen::Matrix3d::Identity() * 20.0;
    reference.pos_ = {{0, 0, 0}, {2.3, 0.2, 0.1}};
    reference.chemmap_ = {0, 1};
    Atoms actual = reference;
    reference_pot.pot->omp_init();
    named.pot->omp_init();
    Neighborhoods nbhs(reference, reference_pot.pot->cutoff);
    reference_pot.pot->getEFS(reference, nbhs);
    named.pot->getEFS(actual, nbhs);
    REQUIRE(actual.energy_ == Catch::Approx(reference.energy_));
    for (int i = 0; i < 2; ++i)
        for (int a = 0; a < 3; ++a)
            REQUIRE(actual.forces_[i][a] == Catch::Approx(reference.forces_[i][a]));
    REQUIRE(actual.stress_.isApprox(reference.stress_));
}

TEST_CASE("Unknown potential type is rejected", "[mtp][errors]") {
    const std::string type = "unknown_potential";
    REQUIRE_THROWS_WITH(Potentials::potential_wrapper("unitest/test-data/potential.mtp", type),
                        Catch::Matchers::ContainsSubstring("Unknown format"));
}

TEST_CASE("Ordinary MTP exposes physical parameter groups", "[mtp][training]") {
    Potentials::PotMTP model("unitest/test-data/potential.mtp");
    const auto groups = model.nl_parameter_groups();
    REQUIRE(groups.size() == 4);
    REQUIRE(groups[0].name == "all");
    REQUIRE(groups[0].end == model.reg_coeff_size);
    REQUIRE(groups[1].name == "radial");
    REQUIRE(groups[1].begin == 0);
    REQUIRE(groups[1].end == model.radial_coeff_size);
    REQUIRE(groups[2].name == "species");
    REQUIRE(groups[2].begin == groups[1].end);
    REQUIRE(groups[2].end == model.radial_coeff_size + model.species_count);
    REQUIRE(groups[3].name == "moment");
    REQUIRE(groups[3].begin == groups[2].end);
    REQUIRE(groups[3].end == model.reg_coeff_size);
}
