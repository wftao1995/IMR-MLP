// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace Potentials {
class PotMTP;
}

namespace BasisAnalysis {

enum class MTPBasisMode {
    Auto,
    Item,
    Span,
};

enum class MTPBasisRelation {
    NotEquivalent,
    ItemEquivalent,
    SpanEquivalent,
};

struct MTPBasisDefinition {
    int moments_count = 0;
    std::vector<std::array<int, 4>> basic;
    std::vector<std::array<int, 4>> rules;
    std::vector<int> scalars;
};

struct MTPBasisDiagnostics {
    int moments_count = 0;
    int basic_count = 0;
    int rules_count = 0;
    int scalar_count = 0;
    int active_moments = 0;
    int dead_moments = 0;
    int dead_rules = 0;
    int duplicate_scalars = 0;
    std::size_t polynomial_terms = 0;
    std::size_t max_polynomial_terms = 0;
};

struct MTPBasisTransformEntry {
    int source = -1;
    int target = -1;
    std::string numerator;
    std::string denominator;
};

struct MTPBasisComparison {
    MTPBasisRelation relation = MTPBasisRelation::NotEquivalent;
    MTPBasisDiagnostics source;
    MTPBasisDiagnostics target;
    int source_rank = 0;
    int target_rank = 0;
    int combined_rank = 0;

    // Filled only for item equivalence. source_to_target[i] is the unique,
    // deterministic target scalar selected for source scalar i.
    std::vector<int> source_to_target;

    // Exact coefficients R(i,j) in source_basis[i] = sum_j R(i,j)
    // target_basis[j].  It is populated for both item and span equivalence.
    std::vector<MTPBasisTransformEntry> transform;
};

struct MTPadialCompatibility {
    bool compatible = true;
    std::string reason;
};

MTPBasisDefinition mtp_basis_definition(const Potentials::PotMTP &potential);

MTPBasisComparison compare_mtp_bases(
    const MTPBasisDefinition &source,
    const MTPBasisDefinition &target,
    MTPBasisMode mode = MTPBasisMode::Auto,
    const std::vector<int> *preferred_item_mapping = nullptr);

double mtp_basis_transform_value(const MTPBasisTransformEntry &entry);

MTPadialCompatibility compare_mtp_radial_layouts(
    const Potentials::PotMTP &source,
    const Potentials::PotMTP &target);

void map_mtp_linear_parameters(
    const Potentials::PotMTP &source,
    Potentials::PotMTP &target,
    const MTPBasisComparison &comparison);

const char *mtp_basis_relation_name(MTPBasisRelation relation);

} // namespace BasisAnalysis
