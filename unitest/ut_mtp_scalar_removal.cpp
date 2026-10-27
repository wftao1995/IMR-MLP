// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp.h"
#include "utils/cmdopts.h"
#include <cmath>
#include <set>
#include <sstream>
#include <vector>
#ifdef USE_OMP
#include <omp.h>
#endif

using Catch::Approx;
using Potentials::PotMTP;

namespace {
struct ThreadSettings {
    int preference = omp_num_threads;
#ifdef USE_OMP
    int requested = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
#endif
    ThreadSettings() { set_omp_num_threads(2); }
    ~ThreadSettings() {
        omp_num_threads = preference;
#ifdef USE_OMP
        omp_set_num_threads(requested);
        omp_set_dynamic(dynamic);
#endif
    }
};

void load_model(PotMTP &model, bool species_only = false)
{
    // The tracked mixedprec-poly5.mtp model, embedded so this regression
    // also runs in tests.mk's empty temporary working directory.
    std::string text = R"mtp(MTP
version = 1.1.0
potential_name = MTP1m
scaling = 1
species_count = 2
potential_tag =
radial_basis_type = RBChebyshevPoly5
  min_dist = 0.4
  max_dist = 4.5
  radial_basis_size = 2
  rb_b = -0.7
radial_funcs_count = 1
radial_coeffs
  0-0
    {0.70, -0.20}
  0-1
    {0.45, 0.10}
  1-0
    {-0.35, 0.25}
  1-1
    {0.80, -0.15}
alpha_moments_count = 5
alpha_index_basic_count = 4
alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}
alpha_index_times_count = 3
alpha_index_times = {{1, 1, 1, 4}, {2, 2, 1, 4}, {3, 3, 1, 4}}
)mtp";
    text += species_only ? "alpha_scalar_moments = 0\nalpha_moment_mapping = {}\n"
                         : "alpha_scalar_moments = 2\nalpha_moment_mapping = {0, 4}\n";
    text += "species_coeffs = {0.3, -0.1}\n";
    text += species_only ? "moment_coeffs = {}\n" : "moment_coeffs = {1.2, -0.4}\n";
    std::istringstream input(text);
    model.Load(input);
}

Atoms geometry()
{
    Atoms atoms;
    atoms.cell_ = 20. * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{0, 0, 0}, {1.5, 0.2, 0.1}, {0.4, 1.8, 0.3}, {1.8, 1.5, 1.2}};
    atoms.chemmap_ = {0, 1, 0, 1};
    return atoms;
}

void equal_efs(const Atoms &actual, const Atoms &expected)
{
    REQUIRE(std::isfinite(actual.energy_));
    REQUIRE(actual.energy_ == Approx(expected.energy_).epsilon(1e-11).margin(1e-11));
    REQUIRE(actual.forces_.size() == expected.forces_.size());
    REQUIRE(actual.site_energies_.size() == expected.site_energies_.size());
    for (int i = 0; i < expected.size(); ++i)
    {
        REQUIRE(std::isfinite(actual.site_energies_[i]));
        REQUIRE(actual.site_energies_[i] == Approx(expected.site_energies_[i]).epsilon(1e-11).margin(1e-11));
        for (int a = 0; a < 3; ++a)
        {
            REQUIRE(std::isfinite(actual.forces_[i][a]));
            REQUIRE(actual.forces_[i][a] == Approx(expected.forces_[i][a]).epsilon(1e-11).margin(1e-11));
        }
    }
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            REQUIRE(std::isfinite(actual.stress_(a, b)));
            REQUIRE(actual.stress_(a, b) == Approx(expected.stress_(a, b)).epsilon(1e-11).margin(1e-11));
        }
}

void check_evaluation(PotMTP &model, const Atoms &expected)
{
    auto serial = geometry();
    model.getEFS(serial);
    equal_efs(serial, expected);
    auto parallel = geometry();
    model.omp_init();
    model.getEFSOMP(parallel);
    equal_efs(parallel, expected);
    auto outer = geometry();
    model.getEFSOMPthread(outer);
    equal_efs(outer, expected);
}

void check_species_only(const Atoms &atoms)
{
    REQUIRE(atoms.energy_ == Approx(0.4).margin(1e-12));
    for (int i = 0; i < atoms.size(); ++i)
    {
        REQUIRE(atoms.site_energies_[i] == Approx(atoms.chemmap_[i] == 0 ? 0.3 : -0.1).margin(1e-12));
        REQUIRE(atoms.forces_[i].norm() < 1e-12);
    }
    REQUIRE(atoms.stress_.norm() < 1e-12);
}

void check_reload(PotMTP &model, const Atoms &expected)
{
    std::ostringstream saved;
    model.Save(saved);
    const std::string serialized = saved.str();
    if (model.alpha_scalar_moments == 0)
    {
        // Reload skips the empty mapping's contents, so inspect the output
        // independently to catch an out-of-bounds value written inside {}.
        REQUIRE(serialized.find("\nalpha_moment_mapping = {}\n") != std::string::npos);
    }
    PotMTP reloaded;
    std::istringstream input(serialized);
    reloaded.Load(input);
    REQUIRE(reloaded.alpha_scalar_moments == model.alpha_scalar_moments);
    check_evaluation(reloaded, expected);
}
}

TEST_CASE("MTP scalar removal preserves the retained model and species-only limit", "[mtp][scalar-removal]")
{
    ThreadSettings threads;
    const int count = GENERATE(0, 1, 2);
    CAPTURE(count);
    PotMTP model, reference;
    load_model(model);
    load_model(reference);
    REQUIRE(model.alpha_scalar_moments == 2);
    const std::vector<double> radial(model.radial_coeffs, model.radial_coeffs + model.radial_coeff_size);
    const std::vector<int> mapping(model.alpha_moment_mapping, model.alpha_moment_mapping + 2);
    const std::vector<double> moments(model.moment_coeffs, model.moment_coeffs + 2);
    std::set<int> removed;
    for (int i = 0; i < count; ++i)
    {
        removed.insert(i);
        reference.moment_coeffs[i] = 0;
    }
    auto expected = geometry();
    reference.getEFS(expected);
    if (count == 2) check_species_only(expected);
    else REQUIRE(expected.forces_[0].norm() > 1e-6);

    // Exercise deletion after scratch has already been used, not just a
    // freshly loaded model. Empty removal must preserve the buffer itself.
    auto warm = geometry();
    model.getEFSOMP(warm);
    const auto old_coefficients = model.regression_coeffs;
    model.remove_scalar_by_id(removed);
    if (count == 0) REQUIRE(model.regression_coeffs == old_coefficients);
    REQUIRE(model.alpha_scalar_moments == 2 - count);
    REQUIRE(model.moment_coeff_size == 2 - count);
    REQUIRE(model.lin_coeff_size == model.species_count + 2 - count);
    REQUIRE(model.reg_coeff_size == model.radial_coeff_size + model.lin_coeff_size);
    REQUIRE(model.radial_coeffs == model.regression_coeffs);
    REQUIRE(model.species_coeffs == model.regression_coeffs + model.radial_coeff_size);
    REQUIRE(model.moment_coeffs == model.species_coeffs + model.species_count);
    for (int i = 0; i < model.radial_coeff_size; ++i) REQUIRE(model.radial_coeffs[i] == radial[i]);
    REQUIRE(model.species_coeffs[0] == 0.3);
    REQUIRE(model.species_coeffs[1] == -0.1);
    for (int i = 0; i < model.alpha_scalar_moments; ++i)
    {
        REQUIRE(model.alpha_moment_mapping[i] == mapping[i + count]);
        REQUIRE(model.moment_coeffs[i] == moments[i + count]);
    }
    check_evaluation(model, expected);
    check_reload(model, expected);
    model.remove_orphans();
    check_evaluation(model, expected);
    check_reload(model, expected);
    std::set<int> none;
    REQUIRE_NOTHROW(model.remove_scalar_by_id(none));
    check_evaluation(model, expected);
}

TEST_CASE("MTP initially zero-scalar models allow empty removal and reload", "[mtp][scalar-removal]")
{
    ThreadSettings threads;
    PotMTP model;
    load_model(model, true);
    REQUIRE(model.alpha_scalar_moments == 0);
    auto expected = geometry();
    model.getEFS(expected);
    check_species_only(expected);
    model.omp_init();
    std::set<int> none;
    REQUIRE_NOTHROW(model.remove_scalar_by_id(none));
    check_evaluation(model, expected);
    check_reload(model, expected);
}
