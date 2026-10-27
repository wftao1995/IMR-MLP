// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "mtmbasis/mtm.h"
#include "potentials/potential_mtp.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace {

bool has_edges(const std::vector<alpha_index_type> &alphas,
               const std::vector<std::pair<int, int>> &edges)
{
    for (const auto &alpha : alphas) {
        bool match = true;
        for (int i = 0; i < alpha.second.rows(); ++i) {
            for (int j = i + 1; j < alpha.second.cols(); ++j) {
                const bool expected = std::ranges::find(edges, std::pair{i, j}) != edges.end();
                if (alpha.second(i, j) != static_cast<int>(expected)) {
                    match = false;
                    break;
                }
            }
            if (!match) break;
        }
        if (match) return true;
    }
    return false;
}

void write_text(const std::string &path, const std::string &text)
{
    std::ofstream output(path);
    REQUIRE(output.good());
    output << text;
}

} // namespace

TEST_CASE("mtmbasis exports loadable linear MTP metadata", "[mtmbasis][metadata]")
{
    PickTensors pick;
    pick.radial_funcs_count = 1;
    pick.alpha_index_basic_count = 1;
    pick.alpha_index_basic = {{0, 0, 0, 0}};
    std::vector<std::array<int, 4>> products;
    std::vector<int> scalars{0};
    const std::string path = "ut_mtmbasis_metadata.mtp" + mpi.fnm_ending;
    struct Cleanup {
        std::string path;
        ~Cleanup() { std::remove(path.c_str()); }
    } cleanup{path};
    pick.save_potential("ut_mtmbasis_metadata.mtp", 1, 0, products, scalars);
    const auto header = MTPMetadata::read_header(path);
    REQUIRE(header.tag.structured());
    REQUIRE(header.tag.text("type") == "MTP");
    REQUIRE(header.tag.text("model") == "linear");
    Potentials::PotMTP model(path);
    REQUIRE(model.alpha_scalar_moments == 1);
    REQUIRE(model.potential_tag.text("model") == "linear");
}

TEST_CASE("mtmbasis enumerates complete deterministic scalar contractions",
          "[mtmbasis][enumeration]")
{
    SECTION("dimension one") {
        REQUIRE(mtmbasis_detail::enumerate_scalar_alpha_indices({0}, {0}).size() == 1);
        REQUIRE(mtmbasis_detail::enumerate_scalar_alpha_indices({0}, {1}).empty());
    }

    SECTION("four labeled rank-one tensors") {
        const auto alphas = mtmbasis_detail::enumerate_scalar_alpha_indices(
            {0, 1, 2, 3}, {1, 1, 1, 1});
        REQUIRE(alphas.size() == 3);
    }

    SECTION("six-dimensional historical missed solution") {
        const auto alphas = mtmbasis_detail::enumerate_scalar_alpha_indices(
            {0, 1, 2, 3, 4, 5}, {1, 1, 1, 1, 1, 1});
        REQUIRE(alphas.size() == 15);
        REQUIRE(has_edges(alphas, {{0, 3}, {1, 5}, {2, 4}}));

        const auto repeated = mtmbasis_detail::enumerate_scalar_alpha_indices(
            {0, 1, 2, 3, 4, 5}, {1, 1, 1, 1, 1, 1});
        REQUIRE(repeated.size() == alphas.size());
        for (size_t i = 0; i < alphas.size(); ++i) {
            REQUIRE(repeated[i].first == alphas[i].first);
            REQUIRE(repeated[i].second == alphas[i].second);
        }
    }

    SECTION("seven-dimensional historical missed solution") {
        const auto alphas = mtmbasis_detail::enumerate_scalar_alpha_indices(
            {0, 1, 2, 3, 4, 5, 6}, {0, 1, 1, 1, 1, 1, 1});
        REQUIRE(alphas.size() == 15);
        REQUIRE(has_edges(alphas, {{1, 3}, {2, 4}, {5, 6}}));
    }
}

TEST_CASE("mtmbasis tensor identity is exact under vertex permutation",
          "[mtmbasis][identity]")
{
    alpha_index_type left(3);
    left.first << 0, 0, 0;
    left.second << 0, 0, 1,
                   0, 1, 0,
                   1, 0, 0;
    left.solve();

    alpha_index_type permuted = left;
    permuted.bswap(0, 1);
    permuted.solve();
    REQUIRE(left == permuted);

    alpha_index_type different = left;
    different.second(0, 2) = 0;
    different.second(2, 0) = 0;
    different.second(0, 1) = 1;
    different.second(1, 0) = 1;
    different.solve();
    REQUIRE_FALSE(left == different);
}

TEST_CASE("mtmbasis component identity follows tensor vertex permutations",
          "[mtmbasis][identity]")
{
    alpha_index_type shape(3);
    shape.first << 0, 1, 2;
    shape.second << 1, 1, 0,
                    1, 2, 2,
                    0, 2, 3;

    const std::vector<int> mu{0, 1, 2};
    const std::vector<int> nv{1, 2, 3};
    const std::vector<int> components{
        1, 0, 0,
        0, 2, 0,
        0, 1, 2};
    const non_ele_id_type original(23, shape, mu, nv, components);

    auto permuted_shape = shape;
    permuted_shape.bswap(0, 2);
    const std::vector<int> permuted_mu{2, 1, 0};
    const std::vector<int> permuted_nv{3, 2, 1};
    const std::vector<int> permuted_components{
        0, 1, 2,
        0, 2, 0,
        1, 0, 0};
    const non_ele_id_type permuted(
        23, permuted_shape, permuted_mu, permuted_nv, permuted_components);

    REQUIRE_FALSE(original < permuted);
    REQUIRE_FALSE(permuted < original);

    auto different_components = components;
    different_components[0] = 0;
    different_components[1] = 1;
    const non_ele_id_type different(23, shape, mu, nv, different_components);
    REQUIRE((original < different || different < original));
}

TEST_CASE("mtmbasis component memoization includes the ordered tensor shape",
          "[mtmbasis][identity][cache]")
{
    alpha_index_type fully_symmetric(5);
    fully_symmetric.first.setZero();
    fully_symmetric.second.setZero();
    fully_symmetric.solve();

    alpha_index_type partitioned(5);
    partitioned.first.setZero();
    partitioned.second.setZero();
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            partitioned.second(i, j) = 1;
            partitioned.second(j, i) = 1;
        }
    }
    partitioned.second(3, 4) = 1;
    partitioned.second(4, 3) = 1;
    partitioned.solve();

    REQUIRE(fully_symmetric.canonical_orders.size() == 120);
    REQUIRE(partitioned.canonical_orders.size() == 12);

    const std::vector<int> mu(5, 0);
    const std::vector<int> nv(5, 0);
    const std::vector<int> components{
        4, 0, 0,
        3, 0, 0,
        2, 0, 0,
        1, 0, 0,
        0, 0, 0};

    non_ele_id_canonical_cache_type cache;
    const non_ele_id_type symmetric_cached(
        42, fully_symmetric, mu, nv, components, &cache);
    const non_ele_id_type partitioned_cached(
        42, partitioned, mu, nv, components, &cache);
    const non_ele_id_type partitioned_direct(
        42, partitioned, mu, nv, components);
    const non_ele_id_type partitioned_cached_again(
        42, partitioned, mu, nv, components, &cache);

    REQUIRE(symmetric_cached.canonical != partitioned_direct.canonical);
    REQUIRE(partitioned_cached.canonical == partitioned_direct.canonical);
    REQUIRE(partitioned_cached_again.canonical == partitioned_direct.canonical);
    REQUIRE(cache.size() == 2);
}

TEST_CASE("mtmbasis restores parent order using both mu and contractions",
          "[mtmbasis][restore]")
{
    PickTensors picker;
    alpha_index_type parent(3);
    parent.tensor_id = 17;
    parent.first << 20, 10, 30;
    parent.second << 0, 0, 1,
                     0, 0, 2,
                     1, 2, 0;
    parent.solve();
    const auto original_first = parent.first;
    const auto original_second = parent.second;

    alpha_index_type expected = parent;
    expected.bswap(0, 1);
    alpha_index_type child_left;
    alpha_index_type child_right;
    expected.split_k(1, child_left, child_right);
    child_left.solve();
    child_right.solve();

    auto restored = picker.restore_b(parent, child_left, child_right);
    REQUIRE(picker.restore_b_test(restored, child_left, child_right));
    REQUIRE(restored.first(0) == 10);
    REQUIRE(parent.first == original_first);
    REQUIRE(parent.second == original_second);

    child_left.first(0) = 99;
    REQUIRE_THROWS_AS(picker.restore_b(parent, child_left, child_right), GPException);
}

TEST_CASE("mtmbasis decomposition selection is local to one tensor",
          "[mtmbasis][kselect]")
{
    REQUIRE(mtmbasis_detail::select_low_trace_decompositions(
        {{0, 3}, {1, 1}, {2, 2}, {3, 1}}, 2) == std::vector<int>{1, 3, 2});
    REQUIRE(mtmbasis_detail::select_low_trace_decompositions(
        {{7, 9}, {8, 4}}, 1) == std::vector<int>{8});
    REQUIRE_THROWS_AS(
        mtmbasis_detail::select_low_trace_decompositions({{0, 1}}, 0),
        GPException);
}

TEST_CASE("mtmbasis validates rule ids separately from coefficients",
          "[mtmbasis][topology]")
{
    REQUIRE_NOTHROW(mtmbasis_detail::validate_alpha_topology(
        2, 3, {{0, 1, 1000, 2}}, {2}));
    REQUIRE_THROWS_AS(mtmbasis_detail::validate_alpha_topology(
        2, 3, {{0, 1, 0, 2}}, {2}), GPException);
    REQUIRE_THROWS_AS(mtmbasis_detail::validate_alpha_topology(
        2, 3, {{0, 4, 1, 2}}, {2}), GPException);
    REQUIRE_THROWS_AS(mtmbasis_detail::validate_alpha_topology(
        2, 4, {{2, 1, 1, 3}}, {3}), GPException);
    REQUIRE_THROWS_AS(mtmbasis_detail::validate_alpha_topology(
        2, 4, {{0, 1, 1, 2}}, {3}), GPException);
    REQUIRE_THROWS_AS(mtmbasis_detail::validate_alpha_topology(
        2, 4, {{0, 1, 1, 2}}, {2}), GPException);
    REQUIRE_THROWS_AS(mtmbasis_detail::validate_alpha_topology(
        2, 3, {{0, 1, 1, 1}, {0, 1, 1, 2}}, {2}), GPException);
}

TEST_CASE("mtmbasis GA selection CDF is complete and robust",
          "[mtmbasis][search]")
{
    REQUIRE(mtmbasis_detail::normalized_selection_cdf({1, 2, 1}) ==
            std::vector<double>{0.0, 0.25, 0.75, 1.0});
    REQUIRE(mtmbasis_detail::normalized_selection_cdf({0, 0}) ==
            std::vector<double>{0.0, 0.5, 1.0});
    REQUIRE_THROWS_AS(mtmbasis_detail::normalized_selection_cdf({1, -1}),
                      GPException);
}

TEST_CASE("mtmbasis weighted fitness prioritizes rules on ties",
          "[mtmbasis][search]")
{
    REQUIRE(mtmbasis_detail::weighted_fitness_better(
        {104, 9}, {100, 10}, 1.0, 4.0));
    REQUIRE_FALSE(mtmbasis_detail::weighted_fitness_better(
        {105, 9}, {100, 10}, 1.0, 4.0));
    REQUIRE_FALSE(mtmbasis_detail::weighted_fitness_better(
        {100, 9}, {104, 8}, 1.0, 4.0));
    REQUIRE_FALSE(mtmbasis_detail::weighted_fitness_better(
        {100, 10}, {100, 10}, 1.0, 4.0));
}

TEST_CASE("mtmbasis search progress is rate limited",
          "[mtmbasis][search]")
{
    std::vector<int> reports;
    for (int completed = 1; completed <= 100; ++completed) {
        if (mtmbasis_detail::progress_report_due(completed, 100))
            reports.push_back(completed);
    }
    REQUIRE(reports ==
            std::vector<int>{10, 20, 30, 40, 50, 60, 70, 80, 90, 100});

    reports.clear();
    for (int completed = 1; completed <= 3; ++completed) {
        if (mtmbasis_detail::progress_report_due(completed, 3))
            reports.push_back(completed);
    }
    REQUIRE(reports == std::vector<int>{1, 2, 3});
    REQUIRE_FALSE(mtmbasis_detail::progress_report_due(0, 100));
    REQUIRE_FALSE(mtmbasis_detail::progress_report_due(101, 100));
    REQUIRE_FALSE(mtmbasis_detail::progress_report_due(1, 0));
    REQUIRE_FALSE(mtmbasis_detail::progress_report_due(1, 100, 0));
}

TEST_CASE("mtmbasis ignores the removed exact count cache setting",
          "[mtmbasis][settings]")
{
    GPSettings legacy_settings;
    legacy_settings.Modify("exact_count_cache", "no");
    REQUIRE_NOTHROW(PickTensors(legacy_settings));

    std::ostringstream listed_settings;
    PickTensors().ListSettings(listed_settings, "");
    REQUIRE(listed_settings.str().find("exact_count_cache") ==
            std::string::npos);
}

TEST_CASE("mtmbasis alpha pool input is validated and round trips",
          "[mtmbasis][input]")
{
    PickTensors writer;
    writer.alphas_gen = mtmbasis_detail::enumerate_scalar_alpha_indices(
        {0, 1, 2, 3}, {1, 1, 1, 1});
    writer.save_pool("ut_mtmbasis_pool.dat", writer.alphas_gen);

    PickTensors reader;
    REQUIRE_NOTHROW(reader.load_alpha_pool_from_file("ut_mtmbasis_pool.dat"));
    REQUIRE(reader.alphas_gen.size() == writer.alphas_gen.size());
    for (size_t i = 0; i < reader.alphas_gen.size(); ++i)
        REQUIRE(reader.alphas_gen[i] == writer.alphas_gen[i]);

    REQUIRE_THROWS_AS(reader.load_alpha_pool_from_file("missing-alpha-pool.dat"),
                      GPException);

    write_text("ut_mtmbasis_bad_pool.dat",
        "Bid: 0\n"
        "mus: 0 1\n"
        "nvs: 1 1\n"
        "level: 0 dim 2 maxpoly 2\n"
        "0 1\n"
        "0 1\n"
        "0 0\n");
    REQUIRE_THROWS_AS(reader.load_alpha_pool_from_file("ut_mtmbasis_bad_pool.dat"),
                      GPException);
}

TEST_CASE("mtmbasis moments file rejects malformed lists", "[mtmbasis][input]")
{
    PickTensors picker;
    picker.max_dim = 1;
    picker.max_lev = 100;

    write_text("ut_mtmbasis_duplicate_moments.dat", "2\n0 0\n0 0\n");
    REQUIRE_THROWS_AS(picker.gen_alpha_pool_by_shapeev_level(
        "ut_mtmbasis_duplicate_moments.dat"), GPException);

    write_text("ut_mtmbasis_truncated_moments.dat", "2\n0 0\n");
    REQUIRE_THROWS_AS(picker.gen_alpha_pool_by_shapeev_level(
        "ut_mtmbasis_truncated_moments.dat"), GPException);

    write_text("ut_mtmbasis_extra_moments.dat", "1\n0 0\nextra\n");
    REQUIRE_THROWS_AS(picker.gen_alpha_pool_by_shapeev_level(
        "ut_mtmbasis_extra_moments.dat"), GPException);
}
