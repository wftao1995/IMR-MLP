// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "basis_analysis/MTPBasisEquivalence.h"
#include "base/structures.h"
#include "potentials/potential_mtp.h"

#include <map>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace {

using BasisAnalysis::MTPBasisComparison;
using BasisAnalysis::MTPBasisDefinition;
using BasisAnalysis::MTPBasisMode;
using BasisAnalysis::MTPBasisRelation;
using Catch::Approx;
using std::map;
using std::pair;
using std::string;
using std::vector;

MTPBasisDefinition item_source() {
    return {
        4,
        {{{0, 0, 0, 0}}, {{1, 0, 0, 0}}},
        {{{0, 0, 1, 2}}, {{0, 1, 1, 3}}},
        {2, 3},
    };
}

MTPBasisDefinition item_target() {
    return {
        4,
        {{{1, 0, 0, 0}}, {{0, 0, 0, 0}}},
        {{{0, 1, 1, 2}}, {{1, 1, 1, 3}}},
        {2, 3},
    };
}

MTPBasisDefinition span_target() {
    return {
        4,
        {{{0, 0, 0, 0}}, {{1, 0, 0, 0}}},
        {{{0, 0, 1, 2}}, {{1, 1, 1, 2}},
         {{0, 0, 1, 3}}, {{1, 1, 3, 3}}},
        {2, 3},
    };
}

map<pair<int, int>, string> transform_coefficients(
    const MTPBasisComparison &comparison) {
    map<pair<int, int>, string> result;
    for (const auto &entry : comparison.transform) {
        result[{entry.source, entry.target}] =
            entry.numerator + "/" + entry.denominator;
    }
    return result;
}

string small_potential_text(bool span_basis, const string &moment_coeffs) {
    std::ostringstream output;
    output
        << "MTP\n"
        << "version = 1.1.0\n"
        << "potential_name = MTP1m\n"
        << "scaling = 1\n"
        << "species_count = 1\n"
        << "potential_tag =\n"
        << "radial_basis_type = RBChebyshev\n"
        << "  min_dist = 1\n"
        << "  max_dist = 5\n"
        << "  radial_basis_size = 2\n"
        << "radial_funcs_count = 2\n"
        << "radial_coeffs\n"
        << "  0-0\n"
        << "    {1, 0}\n"
        << "    {0, 1}\n"
        << "alpha_moments_count = 4\n"
        << "alpha_index_basic_count = 2\n"
        << "alpha_index_basic = {{0, 0, 0, 0}, {1, 0, 0, 0}}\n";
    if (span_basis) {
        output
            << "alpha_index_times_count = 4\n"
            << "alpha_index_times = {{0, 0, 1, 2}, {1, 1, 1, 2}, "
               "{0, 0, 1, 3}, {1, 1, 3, 3}}\n";
    } else {
        output
            << "alpha_index_times_count = 2\n"
            << "alpha_index_times = {{0, 0, 1, 2}, {1, 1, 1, 3}}\n";
    }
    output
        << "alpha_scalar_moments = 2\n"
        << "alpha_moment_mapping = {2, 3}\n"
        << "species_coeffs = {0.75}\n"
        << "moment_coeffs = " << moment_coeffs << "\n";
    return output.str();
}

} // namespace

TEST_CASE("MTP basis comparison ignores moment numbering and contraction order",
          "[mtp-basis][item]") {
    const auto comparison = BasisAnalysis::compare_mtp_bases(
        item_source(), item_target(), MTPBasisMode::Auto);
    const vector<int> expected_mapping{1, 0};
    const map<pair<int, int>, string> expected_transform{
        {{0, 1}, "1/1"}, {{1, 0}, "1/1"}};
    REQUIRE(comparison.relation == MTPBasisRelation::ItemEquivalent);
    REQUIRE(comparison.source_to_target == expected_mapping);
    REQUIRE(comparison.source_rank == 2);
    REQUIRE(comparison.target_rank == 2);
    REQUIRE(comparison.combined_rank == 2);
    REQUIRE(transform_coefficients(comparison) == expected_transform);
}

TEST_CASE("MTP basis comparison verifies a sidecar-proposed scalar mapping",
          "[mtp-basis][mapping]") {
    const vector<int> correct{1, 0};
    const auto comparison = BasisAnalysis::compare_mtp_bases(
        item_source(), item_target(), MTPBasisMode::Auto, &correct);
    REQUIRE(comparison.source_to_target == correct);

    const vector<int> wrong{0, 1};
    REQUIRE_THROWS_WITH(
        BasisAnalysis::compare_mtp_bases(
            item_source(), item_target(), MTPBasisMode::Auto, &wrong),
        Catch::Matchers::ContainsSubstring("disagrees with exact"));
}

TEST_CASE("MTP basis span equivalence produces an exact rational transform",
          "[mtp-basis][span]") {
    const MTPBasisDefinition source{
        4,
        {{{0, 0, 0, 0}}, {{1, 0, 0, 0}}},
        {{{0, 0, 1, 2}}, {{1, 1, 1, 3}}},
        {2, 3},
    };
    const auto comparison = BasisAnalysis::compare_mtp_bases(
        source, span_target(), MTPBasisMode::Auto);
    const map<pair<int, int>, string> expected_transform{
        {{0, 0}, "3/2"}, {{0, 1}, "-1/2"},
        {{1, 0}, "-1/2"}, {{1, 1}, "1/2"}};
    REQUIRE(comparison.relation == MTPBasisRelation::SpanEquivalent);
    REQUIRE(comparison.source_rank == 2);
    REQUIRE(comparison.target_rank == 2);
    REQUIRE(comparison.combined_rank == 2);
    REQUIRE(transform_coefficients(comparison) == expected_transform);

    const auto item_only = BasisAnalysis::compare_mtp_bases(
        source, span_target(), MTPBasisMode::Item);
    REQUIRE(item_only.relation == MTPBasisRelation::NotEquivalent);
}

TEST_CASE("MTP span parameter mapping preserves energy forces and stress",
          "[mtp-basis][span][parameters]") {
    std::istringstream source_stream(
        small_potential_text(false, "{3, 5}"));
    std::istringstream target_stream(
        small_potential_text(true, "{0, 0}"));
    Potentials::PotMTP source;
    Potentials::PotMTP target;
    source.Load(source_stream);
    target.Load(target_stream);

    const auto comparison = BasisAnalysis::compare_mtp_bases(
        BasisAnalysis::mtp_basis_definition(source),
        BasisAnalysis::mtp_basis_definition(target));
    REQUIRE(comparison.relation == MTPBasisRelation::SpanEquivalent);
    REQUIRE(BasisAnalysis::compare_mtp_radial_layouts(source, target).compatible);
    BasisAnalysis::map_mtp_linear_parameters(source, target, comparison);
    REQUIRE(target.species_coeffs[0] == Approx(0.75));
    REQUIRE(target.moment_coeffs[0] == Approx(2.0));
    REQUIRE(target.moment_coeffs[1] == Approx(1.0));
    for (int i = 0; i < source.radial_coeff_size; ++i)
        REQUIRE(target.radial_coeffs[i] == source.radial_coeffs[i]);

    Atoms source_atoms;
    source_atoms.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    source_atoms.pos_ = {{1.0, 1.0, 1.0},
                         {3.1, 1.2, 1.1},
                         {2.0, 3.4, 1.6}};
    source_atoms.chemmap_ = {0, 0, 0};
    Atoms target_atoms = source_atoms;
    source_atoms.resetEFS();
    target_atoms.resetEFS();
    Neighborhoods source_neighborhoods(source_atoms, source.cutoff);
    Neighborhoods target_neighborhoods(target_atoms, target.cutoff);
    source.getEFS(source_atoms, source_neighborhoods);
    target.getEFS(target_atoms, target_neighborhoods);

    REQUIRE(target_atoms.energy_ == Approx(source_atoms.energy_).margin(1e-11));
    for (int i = 0; i < source_atoms.size(); ++i)
        for (int axis = 0; axis < 3; ++axis)
            REQUIRE(target_atoms.forces_[i][axis] ==
                    Approx(source_atoms.forces_[i][axis]).margin(1e-11));
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            REQUIRE(target_atoms.stress_(row, column) ==
                    Approx(source_atoms.stress_(row, column)).margin(1e-11));
}

TEST_CASE("MTP parameter mapping rejects absent coefficients and radial mismatch",
          "[mtp-basis][parameters][validation]") {
    string basis_only = small_potential_text(false, "{3, 5}");
    basis_only.erase(basis_only.find("species_coeffs"));
    std::istringstream basis_only_stream(basis_only);
    std::istringstream target_stream(
        small_potential_text(false, "{0, 0}"));
    Potentials::PotMTP source;
    Potentials::PotMTP target;
    source.Load(basis_only_stream);
    target.Load(target_stream);
    const auto comparison = BasisAnalysis::compare_mtp_bases(
        BasisAnalysis::mtp_basis_definition(source),
        BasisAnalysis::mtp_basis_definition(target));
    REQUIRE(source.empty);
    REQUIRE_THROWS_WITH(
        BasisAnalysis::map_mtp_linear_parameters(source, target, comparison),
        Catch::Matchers::ContainsSubstring("no complete"));

    std::istringstream parameterized_source_stream(
        small_potential_text(false, "{3, 5}"));
    Potentials::PotMTP parameterized_source;
    parameterized_source.Load(parameterized_source_stream);
    target.p_radical_basis->max_val -= 0.5;
    const auto radial = BasisAnalysis::compare_mtp_radial_layouts(
        parameterized_source, target);
    REQUIRE_FALSE(radial.compatible);
    REQUIRE(radial.reason == "radial basis definition differs");
    REQUIRE_THROWS_WITH(
        BasisAnalysis::map_mtp_linear_parameters(
            parameterized_source, target, comparison),
        Catch::Matchers::ContainsSubstring("radial basis definition differs"));
}

TEST_CASE("MTP basis comparison distinguishes equal counts from equal spaces",
          "[mtp-basis][span]") {
    const MTPBasisDefinition source{
        3,
        {{{0, 0, 0, 0}}, {{1, 0, 0, 0}}},
        {{{0, 0, 1, 2}}},
        {2},
    };
    const MTPBasisDefinition target{
        3,
        {{{0, 0, 0, 0}}, {{1, 0, 0, 0}}},
        {{{1, 1, 1, 2}}},
        {2},
    };
    const auto comparison = BasisAnalysis::compare_mtp_bases(source, target);
    REQUIRE(comparison.relation == MTPBasisRelation::NotEquivalent);
    REQUIRE(comparison.source_rank == 1);
    REQUIRE(comparison.target_rank == 1);
    REQUIRE(comparison.combined_rank == 2);
}

TEST_CASE("MTP basis diagnostics report dead moments and duplicate scalars",
          "[mtp-basis][diagnostics]") {
    const MTPBasisDefinition definition{
        5,
        {{{0, 0, 0, 0}}, {{1, 0, 0, 0}}},
        {{{0, 0, 1, 2}}, {{1, 1, 1, 3}}, {{0, 1, 1, 4}}},
        {2, 2},
    };
    const auto comparison = BasisAnalysis::compare_mtp_bases(definition, definition);
    REQUIRE(comparison.relation == MTPBasisRelation::ItemEquivalent);
    REQUIRE(comparison.source.dead_moments == 3);
    REQUIRE(comparison.source.dead_rules == 2);
    REQUIRE(comparison.source.duplicate_scalars == 1);
    REQUIRE(comparison.source_rank == 1);
}

TEST_CASE("MTP basis comparison rejects invalid schedules",
          "[mtp-basis][topology]") {
    auto invalid = item_source();
    invalid.rules[0] = {0, 5, 1, 2};
    REQUIRE_THROWS_WITH(
        BasisAnalysis::compare_mtp_bases(invalid, item_target()),
        Catch::Matchers::ContainsSubstring("out-of-range"));

    invalid = item_source();
    invalid.rules[0][2] = 0;
    REQUIRE_THROWS_WITH(
        BasisAnalysis::compare_mtp_bases(invalid, item_target()),
        Catch::Matchers::ContainsSubstring("non-positive"));
}

TEST_CASE("MTP basis comparison handles a production-sized fixture",
          "[mtp-basis][fixture]") {
    Potentials::PotMTP potential("unitest/test-data/pot4b.mtp");
    const auto definition = BasisAnalysis::mtp_basis_definition(potential);
    const auto comparison = BasisAnalysis::compare_mtp_bases(definition, definition);
    REQUIRE(comparison.relation == MTPBasisRelation::ItemEquivalent);
    REQUIRE(comparison.source.scalar_count == 756);
    REQUIRE(comparison.source_rank == 756);
    REQUIRE(comparison.source.dead_moments == 0);
}
