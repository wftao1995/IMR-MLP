// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_rdf.h"
#include "base/neighborhood.h"
#include <cmath>
#include <limits>
#include <numeric>

TEST_CASE("RDF includes the last rounded bin and excludes neighbors beyond cutoff", "[rdf]")
{
    const bool kd = GENERATE(false, true);
    const double delta = GENERATE(0.1, 0.3, 2.0, 7.0);
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 40.0;
    atom.pos_ = {{0, 0, 0}, {4.99, 0, 0}, {12, 0, 0}};
    atom.chemmap_ = {0, 0, 0};
    ComputeRDF rdf(delta, 5.0);
    rdf.ntypes = 1;
    rdf_count_type total;
    rdf_pcount_type partial;
    rdf.initret(total);
    rdf.initret(partial);
    const size_t bin = std::round(4.99 / delta);
    REQUIRE(bin < total.size());
    unsigned long count = 0;
    std::vector<unsigned long> type_count(1, 0);
    if (kd) {
        NeighborhoodKD nbhs(&atom, 13.0);
        rdf.count_pcf(nbhs, total, partial, count, type_count);
    } else {
        Neighborhoods nbhs(atom, 13.0);
        rdf.count_pcf(nbhs, total, partial, count, type_count);
    }
    REQUIRE(count == 3);
    REQUIRE(type_count[0] == 3);
    REQUIRE(total[bin] == 2);
    REQUIRE(partial.at(rdf_partial_type(0, 0)) == total);
    REQUIRE(std::accumulate(total.begin(), total.end(), 0UL) == 2);
}

TEST_CASE("RDF rejects invalid histogram dimensions", "[rdf]")
{
    const double value = GENERATE(0.0, -0.1, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN());
    REQUIRE_THROWS_AS(ComputeRDF(value, 5.0), std::invalid_argument);
    REQUIRE_THROWS_AS(ComputeRDF(0.1, value), std::invalid_argument);
    REQUIRE_THROWS_AS(ComputeRDF(1e-300, 5.0), std::invalid_argument);
}

TEST_CASE("RDF cutoff search stays within short histograms", "[rdf]")
{
    const double delta = GENERATE(3.0, 7.0);
    ComputeRDF rdf(delta, 5.0);
    rdf_result_type result(rdf.r_dists.size());
    std::iota(result.begin(), result.end(), 0.0);
    REQUIRE(rdf.get_rdfcut(result) == Catch::Approx(rdf.r_dists.back()));
}

namespace {
Atoms rdf_pair(double distance) {
    Atoms atom;
    atom.cell_ = 20.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1, 1, 1}, {1 + distance, 1, 1}};
    atom.chemmap_ = {0, 0};
    return atom;
}
}

TEST_CASE("RDF normalizes rounded bins by their sampled spherical volumes", "[rdf][normalization]")
{
    const double distance = GENERATE(1.0, 4.9);
    auto atom = rdf_pair(distance);
    ComputeRDF rdf(3.0, 5.0);
    rdf.ntypes = 1;
    REQUIRE(rdf.r_dists == std::vector<double>{0.0, 3.0, 6.0});
    // Independent volumes: first sphere, unchanged interior shell, clipped
    // last shell. The last output center remains beyond the radial limit.
    const std::vector<double> expected_volume{
        4.0 * M_PI / 3.0 * std::pow(1.5, 3),
        4.0 * M_PI / 3.0 * (std::pow(4.5, 3) - std::pow(1.5, 3)),
        4.0 * M_PI / 3.0 * (125.0 - std::pow(4.5, 3))};
    for (size_t i = 0; i < expected_volume.size(); ++i)
        REQUIRE(rdf.shell_vol[i] == Catch::Approx(expected_volume[i]).margin(1e-12));

    rdf_count_type total;
    rdf_pcount_type partial;
    rdf.initret(total); rdf.initret(partial);
    unsigned long count = 0;
    std::vector<unsigned long> type_count(1, 0);
    Neighborhoods nbhs(atom, 5.0);
    rdf.count_pcf(nbhs, total, partial, count, type_count);
    const size_t bin = distance == 1.0 ? 0 : 2;
    REQUIRE(count == 2); REQUIRE(type_count[0] == 2);
    REQUIRE(total[bin] == 2);
    REQUIRE(std::accumulate(total.begin(), total.end(), 0UL) == 2);
    double rho = 2.0 / 8000.0;
    rdf_result_type result, same_species;
    rdf.compute_pcf(total, count, rho, result);
    rdf.compute_pcf(0, 0, type_count[0], rho, partial, same_species);
    const double expected = 2.0 / (rho * expected_volume[bin] * 2.0);
    REQUIRE(result[bin] == Catch::Approx(expected).epsilon(1e-12));
    for (size_t i = 0; i < result.size(); ++i) {
        REQUIRE(std::isfinite(result[i]));
        REQUIRE(same_species[i] == Catch::Approx(result[i]).margin(1e-12));
        if (i != bin) REQUIRE(result[i] == 0.0);
    }
}

TEST_CASE("RDF retains zero-support tail bins with finite zero density", "[rdf][normalization]")
{
    auto atom = rdf_pair(4.9);
    ComputeRDF rdf(2.0, 5.0);
    rdf.ntypes = 1;
    REQUIRE(rdf.r_dists == std::vector<double>{0.0, 2.0, 4.0, 6.0});
    REQUIRE(rdf.shell_vol.back() == 0.0);
    rdf_count_type total;
    rdf_pcount_type partial;
    rdf.initret(total); rdf.initret(partial);
    unsigned long count = 0;
    std::vector<unsigned long> type_count(1, 0);
    Neighborhoods nbhs(atom, 5.0);
    rdf.count_pcf(nbhs, total, partial, count, type_count);
    REQUIRE(total[2] == 2); REQUIRE(total.back() == 0);
    double rho = 2.0 / 8000.0;
    rdf_result_type result, same_species;
    rdf.compute_pcf(total, count, rho, result);
    rdf.compute_pcf(0, 0, type_count[0], rho, partial, same_species);
    REQUIRE(result.size() == 4); REQUIRE(same_species.size() == 4);
    REQUIRE(result.back() == 0.0); REQUIRE(same_species.back() == 0.0);
    REQUIRE(result[2] > 0.0);
    for (double value : result) REQUIRE(std::isfinite(value));
    for (double value : same_species) REQUIRE(std::isfinite(value));
}

TEST_CASE("RDF empty species and empty statistics use finite zero placeholders", "[rdf][normalization]")
{
    auto atom = rdf_pair(1.0);
    ComputeRDF rdf(1.0, 5.0);
    rdf.ntypes = 2;
    rdf_count_type total;
    rdf_pcount_type partial;
    rdf.initret(total); rdf.initret(partial);
    unsigned long count = 0;
    std::vector<unsigned long> type_count(2, 0);
    Neighborhoods nbhs(atom, 5.0);
    rdf.count_pcf(nbhs, total, partial, count, type_count);
    REQUIRE(type_count == std::vector<unsigned long>{2, 0});
    double rho = 2.0 / 8000.0;
    rdf_result_type result, same_species;
    rdf.compute_pcf(total, count, rho, result);
    rdf.compute_pcf(0, 0, type_count[0], rho, partial, same_species);
    REQUIRE(result[1] > 0.0);
    REQUIRE(same_species == result);
    for (const auto [center, neighbor] : {std::pair{0, 1}, std::pair{1, 0}, std::pair{1, 1}}) {
        const double neighbor_rho = neighbor == 0 ? rho : 0.0;
        rdf_result_type empty(1, 99.0);
        rdf.compute_pcf(center, neighbor, type_count[center], neighbor_rho, partial, empty);
        REQUIRE(empty.size() == rdf.r_dists.size());
        for (double value : empty) { REQUIRE(std::isfinite(value)); REQUIRE(value == 0.0); }
    }
    rdf_count_type no_counts;
    rdf.initret(no_counts);
    unsigned long no_centers = 0;
    for (double density : {rho, 0.0}) {
        rdf.compute_pcf(no_counts, no_centers, density, result);
        REQUIRE(result.size() == rdf.r_dists.size());
        for (double value : result) { REQUIRE(std::isfinite(value)); REQUIRE(value == 0.0); }
    }
}

TEST_CASE("RDF rejects nonfinite or negative density with populated statistics", "[rdf][normalization]")
{
    ComputeRDF rdf(1.0, 2.0);
    rdf.ntypes = 1;
    rdf_count_type total;
    rdf_pcount_type partial;
    rdf.initret(total); rdf.initret(partial);
    total[1] = partial[rdf_partial_type(0, 0)][1] = 2;
    unsigned long centers = 2;
    rdf_result_type result;
    double density = GENERATE(-1.0, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN());
    REQUIRE_THROWS_AS(rdf.compute_pcf(total, centers, density, result), std::invalid_argument);
    REQUIRE_THROWS_AS(rdf.compute_pcf(0, 0, centers, density, partial, result), std::invalid_argument);
    // An empty center population is still represented by finite zero placeholders.
    centers = 0;
    REQUIRE_NOTHROW(rdf.compute_pcf(total, centers, density, result));
    for (double value : result) REQUIRE(value == 0.0);
    REQUIRE_NOTHROW(rdf.compute_pcf(0, 0, centers, density, partial, result));
    for (double value : result) REQUIRE(value == 0.0);
}
