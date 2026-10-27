// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <chrono>
#include "runtime/mtp/mlip.cpp"

TEST_CASE("Standalone MTP reader preserves the default radial scaling", "[mtp][reader]") {
    const std::string model =
        "MTP\nversion = 1.1.0\nspecies_count = 1\npotential_tag = reader-test\n"
        "radial_basis_type = RBChebyshev\nmin_dist = 0\nmax_dist = 5\n"
        "radial_basis_size = 2\nradial_funcs_count = 1\nradial_coeffs\n0-0\n{0.25, 0.125}\n"
        "alpha_moments_count = 1\nalpha_index_basic_count = 1\nalpha_index_basic = {{0, 0, 0, 0}}\n"
        "alpha_index_times_count = 0\nalpha_index_times = {}\nalpha_scalar_moments = 1\n"
        "alpha_moment_mapping = {0}\nspecies_coeffs = {0.125}\nmoment_coeffs = {0.75}\n";
    const auto file = std::filesystem::temp_directory_path() /
        ("imrmlp-reader-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".mtp");
    struct Cleanup { std::filesystem::path file; ~Cleanup() { std::filesystem::remove(file); } } cleanup{file};
    int mapping[MLIPCore::mlip_g_s]{};
    for (double expected : {1.0, 0.5}) {
        auto input = model;
        if (expected == 0.5) input.insert(input.find("species_count"), "scaling = 0.5\n");
        { std::ofstream output(file); output << input; }
        auto name = file.string(); double cutoff = 0;
        auto *potential = MLIPCore::setup(name.data(), mapping, &cutoff);
        REQUIRE(potential->scaling == expected);
        REQUIRE(cutoff == 5.0);
        // This value is passed to the GPU radial evaluator, unlike the old zero default.
        REQUIRE(potential->radial_coeffs[0] == 0.25);
        MLIPCore::destroy_mlip(potential);
    }
}

namespace {
std::string standalone_reader_fixture()
{
    return "MTP\nversion = 1.1.0\nspecies_count = 1\npotential_tag = reader-test\n"
        "radial_basis_type = RBChebyshev\nmin_dist = 0\nmax_dist = 5\n"
        "radial_basis_size = 2\nradial_funcs_count = 1\nradial_coeffs\n0-0\n{0.25, 0.125}\n"
        "alpha_moments_count = 1\nalpha_index_basic_count = 1\nalpha_index_basic = {{0, 0, 0, 0}}\n"
        "alpha_index_times_count = 0\nalpha_index_times = {}\nalpha_scalar_moments = 1\n"
        "alpha_moment_mapping = {0}\nspecies_coeffs = {0.125}\nmoment_coeffs = {0.75}\n";
}
void replace_reader_text(std::string &text, const std::string &from, const std::string &to)
{
    const auto pos = text.find(from);
    REQUIRE(pos != std::string::npos);
    text.replace(pos, from.size(), to);
}
}

TEST_CASE("Standalone reader validates dimensions graph and long-range payload", "[mtp][reader][standalone]")
{
    std::vector<int> map(33, 0); map[0] = -1;
    double cutoff = 0;
    const auto read = [&](const std::string &text) {
        return std::unique_ptr<MLIPCore::mlip, void(*)(MLIPCore::mlip*)>(
            MLIPCore::setup_from_text(text, map.data(), &cutoff, map.size()), MLIPCore::destroy_mlip);
    };
    SECTION("mapping size is independent of the legacy fixed array") {
        auto model = read(standalone_reader_fixture());
        REQUIRE(model->mapping_size() == 33);
        REQUIRE(model->mapping_data()[32] == 0);
    }
    SECTION("sparse declared index range is not a count of serialized entries") {
        auto text = standalone_reader_fixture();
        replace_reader_text(text, "alpha_moments_count = 1", "alpha_moments_count = 1000000");
        auto model = read(text);
        REQUIRE(model->alpha_moments_count == 1000000);
    }
    SECTION("duplicate scalar mappings remain valid additive entries") {
        auto text = standalone_reader_fixture();
        replace_reader_text(text, "alpha_scalar_moments = 1", "alpha_scalar_moments = 2");
        replace_reader_text(text, "alpha_moment_mapping = {0}", "alpha_moment_mapping = {0, 0}");
        replace_reader_text(text, "moment_coeffs = {0.75}", "moment_coeffs = {0.75, -0.25}");
        REQUIRE(read(text)->alpha_scalar_moments == 2);
    }
    SECTION("bad counts and coefficients throw instead of terminating the process") {
        for (const auto &replacement : std::vector<std::pair<std::string, std::string>>{
            {"radial_basis_size = 2", "radial_basis_size = -2"},
            {"species_count = 1", "species_count = 2147483647"},
            {"species_coeffs = {0.125}", "species_coeffs = {nan}"},
            {"moment_coeffs = {0.75}", "moment_coeffs = {}"},
            {"alpha_moment_mapping = {0}", "alpha_moment_mapping = {1}"},
            {"alpha_index_basic = {{0, 0, 0, 0}}", "alpha_index_basic = {{0, -1, 0, 0}}"}}) {
            auto text = standalone_reader_fixture();
            replace_reader_text(text, replacement.first, replacement.second);
            REQUIRE_THROWS_AS(read(text), std::runtime_error);
        }
    }
    SECTION("legacy long-range coefficients cannot be ignored") {
        REQUIRE_THROWS_AS(read(standalone_reader_fixture() + "les_species_coeffs = {1}\n"), std::runtime_error);
        REQUIRE_THROWS_AS(read(standalone_reader_fixture() + "sog_version = 4\n"), std::runtime_error);
    }
    SECTION("undefined graph inputs are rejected before execution") {
        auto text = standalone_reader_fixture();
        replace_reader_text(text, "alpha_moments_count = 1", "alpha_moments_count = 2");
        replace_reader_text(text, "alpha_index_times_count = 0", "alpha_index_times_count = 1");
        replace_reader_text(text, "alpha_index_times = {}", "alpha_index_times = {{0, 1, 1, 1}}");
        REQUIRE_THROWS_AS(read(text), std::runtime_error);
    }
    SECTION("zero-level order metadata remains valid") {
        auto text = standalone_reader_fixture() +
            "max_level = 0\nlev_scalar_nodes = {0}\nlev_scalar_count = {1}\nlev_times_count = {0}\n";
        auto model = read(text);
        REQUIRE(model->have_order_info);
        REQUIRE(model->lev_scalar_accum[0] == 0);
        REQUIRE(model->lev_times_nodes[0] == 0);
    }
    SECTION("metadata integers reject trailing characters") {
        auto text = standalone_reader_fixture() + "# mtp_descriptor_count = 2junk\n";
        REQUIRE_THROWS_AS(read(text), std::runtime_error);
    }
    SECTION("unused neural dimensions cannot request invalid workspaces") {
        for (const auto &suffix : {"# mtp_hidden1 = -1\n", "# mtp_hidden2 = 2147483647\n",
                                  "# mtp_hidden1 = 1\n", "# mtp_hidden2 = 1\n"})
            REQUIRE_THROWS_AS(read(standalone_reader_fixture() + suffix), std::runtime_error);
    }
}

TEST_CASE("Standalone reader accepts complete MTP_M coefficient blocks", "[mtp][reader][standalone]")
{
    int mapping[MLIPCore::mlip_g_s]{};
    double cutoff = 0;
    for (const auto &head : {std::string("mtp_m_li"), std::string("mtp_m_q2"), std::string("mtp_m_fs")}) {
        auto text = standalone_reader_fixture();
        const int q = head == "mtp_m_li" ? 1 : 2;
        const int size = head == "mtp_m_li" ? 0 : (head == "mtp_m_q2" ? 5 : 1);
        replace_reader_text(text, "species_coeffs = {0.125}",
            q == 1 ? "species_coeffs = {0.125, 0.5}" : "species_coeffs = {0.125, 0.5, -0.25}");
        replace_reader_text(text, "moment_coeffs = {0.75}",
            q == 1 ? "moment_coeffs = {0.75}" : "moment_coeffs = {0.75, 0.25}");
        text += head == "mtp_m_li" ? "model_coeffs = {}\n" :
            (head == "mtp_m_q2" ? "model_coeffs = {1, 2, 3, 4, 5}\n" : "model_coeffs = {0.5}\n");
        text += "# mtp_model = " + head + "\n# mtp_descriptor_count = " + std::to_string(q) +
            "\n# mtp_model_coeffs_count = " + std::to_string(size) + "\n";
        std::unique_ptr<MLIPCore::mlip, void(*)(MLIPCore::mlip*)> model(
            MLIPCore::setup_from_text(text, mapping, &cutoff), MLIPCore::destroy_mlip);
        REQUIRE(model->descriptor_count == q);
        REQUIRE(model->model_coeffs_count == size);
        REQUIRE(model->descriptor_species_coeffs[0] == 0.5);
        REQUIRE(model->species_coeffs[0] == 0.125);
        for (const auto &suffix : {"# mtp_hidden1 = 1\n", "# mtp_hidden2 = 1\n"})
            REQUIRE_THROWS_AS(MLIPCore::setup_from_text(text + suffix, mapping, &cutoff), std::runtime_error);
    }
}
