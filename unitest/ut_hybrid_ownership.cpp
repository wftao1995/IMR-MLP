// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_hybrid.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>

using Catch::Approx;
using Potentials::PotentialHybrid;
using Potentials::PotMTP;

namespace {
struct ModelFile {
    std::filesystem::path directory, path;
    ModelFile() {
        auto pattern = (std::filesystem::temp_directory_path() / "imrmlp-hybrid-owner-XXXXXX").string();
        const char *created = mkdtemp(pattern.data());
        if (!created) throw std::runtime_error("Cannot create Hybrid ownership fixture");
        directory = created;
        path = directory / "model.mtp";
        try {
            // Exact tracked LAMMPS/USER-MLIP/transport/tests/simple_analytic.mtp,
            // embedded to keep this regression independent of test CWD/data.
            std::ofstream output(path);
            output << R"mtp(MTP
version = 1.1.0
potential_name = analytic_interface_test
species_count = 1
potential_tag = test-only
radial_basis_type = RBChebyshev
min_dist = 0.0
max_dist = 5.0
radial_basis_size = 2
radial_funcs_count = 1
radial_coeffs
0-0
{0.25, 0.125}
alpha_moments_count = 1
alpha_index_basic_count = 1
alpha_index_basic = {{0, 0, 0, 0}}
alpha_index_times_count = 0
alpha_index_times = {}
alpha_scalar_moments = 1
alpha_moment_mapping = {0}
species_coeffs = {0.125}
moment_coeffs = {0.75}
)mtp";
            output.close();
            if (!output) throw std::runtime_error("Cannot write Hybrid ownership fixture");
        } catch (...) {
            std::error_code error;
            std::filesystem::remove_all(directory, error);
            throw;
        }
    }
    ~ModelFile() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

GPSettings settings_for(const ModelFile &file, bool invalid_second)
{
    GPSettings settings;
    settings.settings.emplace("npot", "2");
    settings.settings.emplace("spcount", "1");
    settings.settings.emplace("pot1:pottype", "mtp");
    settings.settings.emplace("pot1:potfile", file.path.string());
    settings.settings.emplace("pot2:pottype", invalid_second ? "invalid_ownership_probe" : "mtp");
    settings.settings.emplace("pot2:potfile", file.path.string());
    return settings;
}
}

TEST_CASE("Hybrid construction failure releases an already loaded MTP buffer", "[hybrid][ownership]")
{
    ModelFile file;
    // The first MTP is fully loaded and its OMP scratch initialized before
    // the unsupported second type throws. LSan must see no abandoned first
    // coefficient allocation, and the original configuration error remains.
    for (int attempt = 0; attempt < 3; ++attempt)
    {
        CAPTURE(attempt);
        try {
            PotentialHybrid invalid(settings_for(file, true));
            FAIL("Hybrid unexpectedly accepted an unsupported second type");
        } catch (const GPException &error) {
            REQUIRE_THAT(std::string(error.What()), Catch::Matchers::ContainsSubstring(
                "hybrid: unsupported sub-potential type: invalid_ownership_probe"));
        }
    }
}

TEST_CASE("Hybrid successful construction transfers both MTP buffers once", "[hybrid][ownership]")
{
    ModelFile file;
    PotMTP reference(file.path.string());
    PotentialHybrid hybrid(settings_for(file, false));
    REQUIRE(hybrid.ppots.size() == 2);
    for (int i = 0; i < 2; ++i)
    {
        auto *model = dynamic_cast<PotMTP *>(hybrid.ppots[i].get());
        REQUIRE(model != nullptr);
        REQUIRE(model->coeffs_handled_external);
        REQUIRE(model->regression_coeffs == nullptr);
        REQUIRE(model->radial_coeffs == hybrid.radial_coeffs + i * reference.radial_coeff_size);
        REQUIRE(model->moment_coeffs == hybrid.moment_coeffs + i * reference.alpha_scalar_moments);
        REQUIRE(model->species_coeffs[0] == (i == 0 ? reference.species_coeffs[0] : 0.));
    }
    Atoms expected;
    expected.cell_ = 20. * Eigen::Matrix3d::Identity();
    expected.pbc_ = {false, false, false};
    expected.pos_ = {{0, 0, 0}, {2.1, 0.2, 0.1}};
    expected.chemmap_ = {0, 0};
    auto actual = expected;
    reference.getEFS(expected);
    hybrid.getEFS(actual);
    REQUIRE(actual.energy_ == Approx(2 * expected.energy_ - expected.size() * reference.species_coeffs[0]).margin(1e-12));
    REQUIRE(actual.forces_.size() == expected.forces_.size());
    for (int i = 0; i < expected.size(); ++i)
        REQUIRE((actual.forces_[i] - 2 * expected.forces_[i]).norm() < 1e-12);
    REQUIRE((actual.stress_ - 2 * expected.stress_).norm() < 1e-12);
}
