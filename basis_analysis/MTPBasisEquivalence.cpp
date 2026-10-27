// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "MTPBasisEquivalence.h"

#include "potentials/potential_mtp.h"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace BasisAnalysis {
namespace {

using Integer = boost::multiprecision::cpp_int;
using Rational = boost::rational<Integer>;
using BasicKey = std::array<int, 4>;
using Monomial = std::vector<BasicKey>;
using Polynomial = std::map<Monomial, Integer>;
using RationalPolynomial = std::map<Monomial, Rational>;
using Combination = std::map<int, Rational>;

void erase_zero_terms(Polynomial &polynomial) {
    for (auto it = polynomial.begin(); it != polynomial.end();) {
        if (it->second == 0)
            it = polynomial.erase(it);
        else
            ++it;
    }
}

void erase_zero_terms(RationalPolynomial &polynomial) {
    for (auto it = polynomial.begin(); it != polynomial.end();) {
        if (it->second == 0)
            it = polynomial.erase(it);
        else
            ++it;
    }
}

void erase_zero_terms(Combination &combination) {
    for (auto it = combination.begin(); it != combination.end();) {
        if (it->second == 0)
            it = combination.erase(it);
        else
            ++it;
    }
}

Monomial multiply_monomials(const Monomial &left, const Monomial &right) {
    Monomial product;
    product.reserve(left.size() + right.size());
    std::merge(left.begin(), left.end(), right.begin(), right.end(),
               std::back_inserter(product));
    return product;
}

Polynomial multiply_polynomials(const Polynomial &left,
                                const Polynomial &right,
                                int coefficient) {
    Polynomial result;
    for (const auto &[left_monomial, left_coefficient] : left) {
        for (const auto &[right_monomial, right_coefficient] : right) {
            result[multiply_monomials(left_monomial, right_monomial)] +=
                left_coefficient * right_coefficient * coefficient;
        }
    }
    erase_zero_terms(result);
    return result;
}

void add_polynomial(Polynomial &destination, Polynomial source) {
    for (auto &[monomial, coefficient] : source)
        destination[std::move(monomial)] += coefficient;
    erase_zero_terms(destination);
}

struct ExpandedBasis {
    std::vector<Polynomial> scalars;
    MTPBasisDiagnostics diagnostics;
};

void validate_definition(const MTPBasisDefinition &definition) {
    const int basic_count = static_cast<int>(definition.basic.size());
    if (definition.moments_count < basic_count || definition.moments_count <= 0)
        throw std::runtime_error(
            "alpha_moments_count is smaller than alpha_index_basic_count");
    if (definition.scalars.empty())
        throw std::runtime_error("alpha_moment_mapping is empty");

    std::vector<char> ready(definition.moments_count, 0);
    std::fill(ready.begin(), ready.begin() + basic_count, 1);
    int previous_target = basic_count - 1;
    for (const auto &rule : definition.rules) {
        for (int field : {0, 1, 3}) {
            if (rule[field] < 0 || rule[field] >= definition.moments_count)
                throw std::runtime_error(
                    "alpha_index_times contains an out-of-range moment id");
        }
        if (rule[2] <= 0)
            throw std::runtime_error(
                "alpha_index_times contains a non-positive coefficient");
        if (!ready[rule[0]] || !ready[rule[1]])
            throw std::runtime_error(
                "alpha_index_times is not topologically ordered");
        if (rule[3] < basic_count || rule[0] >= rule[3] ||
            rule[1] >= rule[3] || rule[3] < previous_target)
            throw std::runtime_error(
                "alpha_index_times has an invalid or decreasing target");
        ready[rule[3]] = 1;
        previous_target = rule[3];
    }
    for (int moment = basic_count; moment < definition.moments_count; ++moment) {
        if (!ready[moment])
            throw std::runtime_error(
                "alpha_moments_count contains an undefined non-basic moment");
    }
    for (int moment : definition.scalars) {
        if (moment < 0 || moment >= definition.moments_count || !ready[moment])
            throw std::runtime_error(
                "alpha_moment_mapping references an invalid moment");
    }
}

std::vector<char> find_reachable_moments(
    const MTPBasisDefinition &definition) {
    std::vector<std::vector<std::pair<int, int>>> inputs(
        definition.moments_count);
    for (const auto &rule : definition.rules)
        inputs[rule[3]].emplace_back(rule[0], rule[1]);

    std::vector<char> reachable(definition.moments_count, 0);
    std::vector<int> stack(definition.scalars.begin(), definition.scalars.end());
    while (!stack.empty()) {
        const int moment = stack.back();
        stack.pop_back();
        if (reachable[moment]) continue;
        reachable[moment] = 1;
        for (const auto &[left, right] : inputs[moment]) {
            stack.push_back(left);
            stack.push_back(right);
        }
    }
    return reachable;
}

ExpandedBasis expand_basis(const MTPBasisDefinition &definition) {
    validate_definition(definition);
    const int basic_count = static_cast<int>(definition.basic.size());
    const auto reachable = find_reachable_moments(definition);

    ExpandedBasis result;
    result.diagnostics.moments_count = definition.moments_count;
    result.diagnostics.basic_count = basic_count;
    result.diagnostics.rules_count = static_cast<int>(definition.rules.size());
    result.diagnostics.scalar_count = static_cast<int>(definition.scalars.size());
    result.diagnostics.active_moments =
        static_cast<int>(std::count(reachable.begin(), reachable.end(), 1));
    result.diagnostics.dead_moments =
        definition.moments_count - result.diagnostics.active_moments;
    result.diagnostics.dead_rules = static_cast<int>(std::count_if(
        definition.rules.begin(), definition.rules.end(),
        [&](const auto &rule) { return !reachable[rule[3]]; }));

    std::vector<std::vector<int>> scalar_positions(definition.moments_count);
    for (int i = 0; i < static_cast<int>(definition.scalars.size()); ++i)
        scalar_positions[definition.scalars[i]].push_back(i);

    std::vector<int> remaining_uses(definition.moments_count, 0);
    for (const auto &rule : definition.rules) {
        if (!reachable[rule[3]]) continue;
        ++remaining_uses[rule[0]];
        ++remaining_uses[rule[1]];
    }
    for (int scalar : definition.scalars) ++remaining_uses[scalar];

    std::vector<std::unique_ptr<Polynomial>> moments(definition.moments_count);
    for (int i = 0; i < basic_count; ++i) {
        if (!reachable[i]) continue;
        moments[i] = std::make_unique<Polynomial>();
        (*moments[i])[Monomial{definition.basic[i]}] = 1;
    }
    result.scalars.resize(definition.scalars.size());

    auto consume = [&](int moment) {
        if (--remaining_uses[moment] == 0) moments[moment].reset();
    };
    auto capture_scalars = [&](int moment) {
        for (int scalar_position : scalar_positions[moment]) {
            if (!moments[moment])
                throw std::runtime_error(
                    "internal error while expanding an MTP scalar moment");
            result.scalars[scalar_position] = *moments[moment];
            consume(moment);
        }
    };

    for (int i = 0; i < basic_count; ++i) {
        if (reachable[i]) capture_scalars(i);
    }

    int open_target = -1;
    auto finish_target = [&]() {
        if (open_target >= 0) capture_scalars(open_target);
    };
    for (const auto &rule : definition.rules) {
        if (!reachable[rule[3]]) continue;
        if (rule[3] != open_target) {
            finish_target();
            open_target = rule[3];
            moments[open_target] = std::make_unique<Polynomial>();
        }
        if (!moments[rule[0]] || !moments[rule[1]])
            throw std::runtime_error(
                "internal error: released MTP moment still has a consumer");
        add_polynomial(*moments[open_target],
                       multiply_polynomials(*moments[rule[0]],
                                            *moments[rule[1]], rule[2]));
        consume(rule[0]);
        consume(rule[1]);
    }
    finish_target();

    std::set<Polynomial> unique_scalars;
    for (const auto &scalar : result.scalars) {
        if (scalar.empty())
            throw std::runtime_error("an expanded MTP scalar is identically zero");
        result.diagnostics.polynomial_terms += scalar.size();
        result.diagnostics.max_polynomial_terms =
            std::max(result.diagnostics.max_polynomial_terms, scalar.size());
        unique_scalars.insert(scalar);
    }
    result.diagnostics.duplicate_scalars =
        static_cast<int>(result.scalars.size() - unique_scalars.size());
    return result;
}

void subtract_scaled(RationalPolynomial &destination,
                     const RationalPolynomial &source,
                     const Rational &scale) {
    for (const auto &[monomial, coefficient] : source)
        destination[monomial] -= scale * coefficient;
    erase_zero_terms(destination);
}

void add_scaled(Combination &destination, const Combination &source,
                const Rational &scale) {
    for (const auto &[index, coefficient] : source)
        destination[index] += scale * coefficient;
    erase_zero_terms(destination);
}

RationalPolynomial to_rational(const Polynomial &source) {
    RationalPolynomial result;
    for (const auto &[monomial, coefficient] : source)
        result.emplace(monomial, Rational(coefficient));
    return result;
}

class ExactSpanBasis {
    struct Row {
        RationalPolynomial polynomial;
        Combination combination;
    };
    std::map<Monomial, Row> rows_;

public:
    bool reduce(const Polynomial &polynomial, Combination *combination) const {
        RationalPolynomial remainder = to_rational(polynomial);
        Combination representation;
        while (!remainder.empty()) {
            const auto pivot = remainder.begin();
            const auto row = rows_.find(pivot->first);
            if (row == rows_.end()) break;
            const Rational scale = pivot->second;
            subtract_scaled(remainder, row->second.polynomial, scale);
            add_scaled(representation, row->second.combination, scale);
        }
        if (!remainder.empty()) return false;
        if (combination) *combination = std::move(representation);
        return true;
    }

    bool insert(const Polynomial &polynomial, int original_index) {
        RationalPolynomial remainder = to_rational(polynomial);
        Combination representation;
        representation[original_index] = Rational(1);
        while (!remainder.empty()) {
            const auto pivot = remainder.begin();
            const auto row = rows_.find(pivot->first);
            if (row == rows_.end()) break;
            const Rational scale = pivot->second;
            subtract_scaled(remainder, row->second.polynomial, scale);
            add_scaled(representation, row->second.combination, -scale);
        }
        if (remainder.empty()) return false;

        const Monomial pivot_key = remainder.begin()->first;
        const Rational pivot_value = remainder.begin()->second;
        for (auto &[_, coefficient] : remainder) coefficient /= pivot_value;
        for (auto &[_, coefficient] : representation)
            coefficient /= pivot_value;
        rows_.emplace(pivot_key,
                      Row{std::move(remainder), std::move(representation)});
        return true;
    }

    int rank() const { return static_cast<int>(rows_.size()); }
};

std::vector<int> find_item_mapping(
    const std::vector<Polynomial> &source,
    const std::vector<Polynomial> &target,
    const std::vector<int> *preferred_mapping) {
    if (source.size() != target.size()) return {};
    if (preferred_mapping) {
        if (preferred_mapping->size() != source.size())
            throw std::runtime_error(
                "sidecar tensor mapping does not cover every source scalar");
        std::vector<char> used(target.size(), 0);
        for (int i = 0; i < static_cast<int>(source.size()); ++i) {
            const int target_index = (*preferred_mapping)[i];
            if (target_index < 0 ||
                target_index >= static_cast<int>(target.size()) ||
                used[target_index])
                throw std::runtime_error(
                    "sidecar tensor mapping is not a scalar bijection");
            if (source[i] != target[target_index])
                throw std::runtime_error(
                    "sidecar tensor mapping disagrees with exact scalar expressions");
            used[target_index] = 1;
        }
        return *preferred_mapping;
    }
    std::map<Polynomial, std::deque<int>> target_positions;
    for (int i = 0; i < static_cast<int>(target.size()); ++i)
        target_positions[target[i]].push_back(i);

    std::vector<int> mapping(source.size(), -1);
    for (int i = 0; i < static_cast<int>(source.size()); ++i) {
        auto found = target_positions.find(source[i]);
        if (found == target_positions.end() || found->second.empty()) return {};
        mapping[i] = found->second.front();
        found->second.pop_front();
    }
    return mapping;
}

MTPBasisTransformEntry transform_entry(int source, int target,
                                       const Rational &coefficient) {
    return {source, target, coefficient.numerator().str(),
            coefficient.denominator().str()};
}

} // namespace

MTPBasisDefinition mtp_basis_definition(
    const Potentials::PotMTP &potential) {
    MTPBasisDefinition result;
    result.moments_count = potential.alpha_moments_count;
    result.basic.resize(potential.alpha_index_basic_count);
    for (int i = 0; i < potential.alpha_index_basic_count; ++i)
        for (int j = 0; j < 4; ++j)
            result.basic[i][j] = potential.alpha_index_basic(i, j);
    result.rules.resize(potential.alpha_index_times_count);
    for (int i = 0; i < potential.alpha_index_times_count; ++i)
        for (int j = 0; j < 4; ++j)
            result.rules[i][j] = potential.alpha_index_times(i, j);
    result.scalars.assign(potential.alpha_moment_mapping,
                          potential.alpha_moment_mapping +
                              potential.alpha_scalar_moments);
    return result;
}

MTPBasisComparison compare_mtp_bases(const MTPBasisDefinition &source,
                                     const MTPBasisDefinition &target,
                                     MTPBasisMode mode,
                                     const std::vector<int> *preferred_item_mapping) {
    const ExpandedBasis source_expanded = expand_basis(source);
    const ExpandedBasis target_expanded = expand_basis(target);

    MTPBasisComparison result;
    result.source = source_expanded.diagnostics;
    result.target = target_expanded.diagnostics;
    result.source_to_target = find_item_mapping(
        source_expanded.scalars, target_expanded.scalars,
        preferred_item_mapping);
    if (!result.source_to_target.empty()) {
        result.relation = MTPBasisRelation::ItemEquivalent;
        ExactSpanBasis rank_basis;
        for (int i = 0; i < static_cast<int>(target_expanded.scalars.size()); ++i)
            rank_basis.insert(target_expanded.scalars[i], i);
        result.source_rank = result.target_rank = result.combined_rank =
            rank_basis.rank();
        for (int i = 0; i < static_cast<int>(result.source_to_target.size()); ++i)
            result.transform.push_back(
                transform_entry(i, result.source_to_target[i], Rational(1)));
        return result;
    }

    ExactSpanBasis source_basis;
    for (int i = 0; i < static_cast<int>(source_expanded.scalars.size()); ++i)
        source_basis.insert(source_expanded.scalars[i], i);
    ExactSpanBasis target_basis;
    for (int i = 0; i < static_cast<int>(target_expanded.scalars.size()); ++i)
        target_basis.insert(target_expanded.scalars[i], i);
    result.source_rank = source_basis.rank();
    result.target_rank = target_basis.rank();

    ExactSpanBasis combined_basis;
    for (int i = 0; i < static_cast<int>(target_expanded.scalars.size()); ++i)
        combined_basis.insert(target_expanded.scalars[i], i);
    for (int i = 0; i < static_cast<int>(source_expanded.scalars.size()); ++i)
        combined_basis.insert(source_expanded.scalars[i],
                              static_cast<int>(target_expanded.scalars.size()) + i);
    result.combined_rank = combined_basis.rank();

    if (mode == MTPBasisMode::Item ||
        result.source_rank != result.target_rank ||
        result.combined_rank != result.target_rank) {
        result.relation = MTPBasisRelation::NotEquivalent;
        return result;
    }

    std::vector<MTPBasisTransformEntry> transform;
    for (int i = 0; i < static_cast<int>(source_expanded.scalars.size()); ++i) {
        Combination representation;
        if (!target_basis.reduce(source_expanded.scalars[i], &representation)) {
            result.relation = MTPBasisRelation::NotEquivalent;
            return result;
        }
        for (const auto &[target_index, coefficient] : representation)
            transform.push_back(transform_entry(i, target_index, coefficient));
    }
    result.transform = std::move(transform);
    result.relation = MTPBasisRelation::SpanEquivalent;
    return result;
}

double mtp_basis_transform_value(const MTPBasisTransformEntry &entry) {
    using Decimal = boost::multiprecision::cpp_dec_float_100;
    const Decimal numerator(entry.numerator);
    const Decimal denominator(entry.denominator);
    return (numerator / denominator).convert_to<double>();
}

MTPadialCompatibility compare_mtp_radial_layouts(
    const Potentials::PotMTP &source,
    const Potentials::PotMTP &target) {
    if (source.species_count != target.species_count)
        return {false, "species_count differs"};
    if (source.radial_basis_split != target.radial_basis_split)
        return {false, "radial split mode differs"};
    if (source.radial_func_count != target.radial_func_count)
        return {false, "radial_funcs_count differs"};
    if (source.rb_size != target.rb_size)
        return {false, "radial_basis_size differs"};
    if (source.radial_coeff_size != target.radial_coeff_size)
        return {false, "radial coefficient layout differs"};

    auto signature = [](const Potentials::PotMTP &potential) {
        std::ostringstream output;
        output << std::scientific
               << std::setprecision(std::numeric_limits<double>::max_digits10);
        output << potential.species_count << ' '
               << potential.radial_basis_split << ' '
               << potential.radial_func_count << ' ' << potential.rb_size
               << '\n';
        if (potential.radial_basis_split) {
            const std::size_t expected =
                static_cast<std::size_t>(potential.species_count) *
                potential.species_count;
            if (potential.p_radical_basis_pair.size() != expected)
                throw std::runtime_error(
                    "split radial basis has the wrong pair count");
            for (auto *basis : potential.p_radical_basis_pair) {
                if (!basis)
                    throw std::runtime_error(
                        "split radial basis contains a null pair");
                output << "scaling = " << basis->scaling << '\n';
                basis->SaveToStream(output);
            }
        } else {
            if (!potential.p_radical_basis)
                throw std::runtime_error("potential has no radial basis");
            output << "scaling = " << potential.p_radical_basis->scaling
                   << '\n';
            potential.p_radical_basis->SaveToStream(output);
        }
        return output.str();
    };
    if (signature(source) != signature(target))
        return {false, "radial basis definition differs"};
    return {};
}

void map_mtp_linear_parameters(
    const Potentials::PotMTP &source,
    Potentials::PotMTP &target,
    const MTPBasisComparison &comparison) {
    if (comparison.relation == MTPBasisRelation::NotEquivalent)
        throw std::runtime_error(
            "parameters cannot be mapped between non-equivalent bases");
    if (source.empty)
        throw std::runtime_error(
            "source potential has no complete species/moment parameter block");
    const auto radial = compare_mtp_radial_layouts(source, target);
    if (!radial.compatible)
        throw std::runtime_error("parameter layouts are incompatible: " +
                                 radial.reason);

    std::copy(source.radial_coeffs,
              source.radial_coeffs + source.radial_coeff_size,
              target.radial_coeffs);
    std::copy(source.species_coeffs,
              source.species_coeffs + source.species_count,
              target.species_coeffs);
    std::fill(target.moment_coeffs,
              target.moment_coeffs + target.alpha_scalar_moments, 0.0);
    for (const auto &entry : comparison.transform) {
        if (entry.source < 0 || entry.source >= source.alpha_scalar_moments ||
            entry.target < 0 || entry.target >= target.alpha_scalar_moments)
            throw std::runtime_error("basis transform coefficient is out of range");
        const double factor = mtp_basis_transform_value(entry);
        const double contribution = source.moment_coeffs[entry.source] * factor;
        if (!std::isfinite(factor) || !std::isfinite(contribution))
            throw std::runtime_error(
                "basis transform is not representable as double");
        target.moment_coeffs[entry.target] += contribution;
        if (!std::isfinite(target.moment_coeffs[entry.target]))
            throw std::runtime_error(
                "mapped moment coefficient overflowed double");
    }
    target.empty = false;
}

const char *mtp_basis_relation_name(MTPBasisRelation relation) {
    switch (relation) {
    case MTPBasisRelation::ItemEquivalent:
        return "item";
    case MTPBasisRelation::SpanEquivalent:
        return "span";
    default:
        return "not-equivalent";
    }
}

} // namespace BasisAnalysis
