// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp.h"
#include "potentials/potential_mtpnn1.h"
#include "potentials/potential_wrapper.h"
#include "utils/cmdopts.h"
#include <cmath>
#include <filesystem>
#include <numeric>
#ifdef USE_OMP
#include <omp.h>
#endif

using namespace Potentials;
using Catch::Approx;

namespace {
struct ThreadSettings {
    int preference = omp_num_threads;
#ifdef USE_OMP
    int requested = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
    int levels = omp_get_max_active_levels();
#endif
    ~ThreadSettings() {
        omp_num_threads = preference;
#ifdef USE_OMP
        omp_set_num_threads(requested);
        omp_set_dynamic(dynamic);
        omp_set_max_active_levels(levels);
#endif
    }
};

std::string model_path(const std::string &name = "pot4b.mtp") {
    for (const auto &prefix : {"unitest/test-data/", "../../unitest/test-data/",
                               "../../../unitest/test-data/"}) {
        auto path = std::string(prefix) + name;
        if (std::filesystem::exists(path)) return path;
    }
    throw std::runtime_error("MTP thread regression fixture not found: " + name);
}

Atoms geometry(int count = 8) {
    Atoms atom;
    atom.cell_ = 20.0 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {false, false, false};
    for (int i = 0; i < count; ++i) {
        const int x = i / 4, y = (i / 2) % 2, z = i % 2;
        atom.pos_.emplace_back(2.6*x + .023*y, 2.6*y + .017*z, 2.6*z + .019*x);
        atom.chemmap_.push_back(0);
    }
    return atom;
}

void compare(const Atoms &actual, const Atoms &expected, double factor = 1., bool energy_only = false) {
    REQUIRE(actual.energy_ == Approx(factor * expected.energy_).epsilon(1e-11).margin(1e-11));
    REQUIRE(actual.site_energies_.size() == expected.site_energies_.size());
    for (int i = 0; i < expected.size(); ++i)
        REQUIRE(actual.site_energies_[i] == Approx(factor * expected.site_energies_[i]).epsilon(1e-11).margin(1e-11));
    REQUIRE(actual.energy_ == Approx(std::accumulate(actual.site_energies_.begin(),
        actual.site_energies_.end(), 0.)).epsilon(1e-11).margin(1e-11));
    if (energy_only) return;
    REQUIRE(actual.forces_.size() == expected.forces_.size());
    for (int i = 0; i < expected.size(); ++i)
        for (int a = 0; a < 3; ++a)
            REQUIRE(actual.forces_[i][a] == Approx(factor * expected.forces_[i][a]).epsilon(1e-11).margin(1e-11));
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            REQUIRE(actual.stress_(a,b) == Approx(factor * expected.stress_(a,b)).epsilon(1e-11).margin(1e-11));
}

void check_efs_and_energy(AnyPotential &pot, const Atoms &input) {
    auto expected = input;
    pot.getEFS(expected);
    auto actual = input;
    pot.getEFSOMP(actual);
    compare(actual, expected);
    pot.getEOMP(actual);
    compare(actual, expected, 1., true);
}
}

TEST_CASE("OpenMP setter validates requests and model construction preserves them", "[openmp][threads]") {
    ThreadSettings restore;
    for (int requested : {1, 2, 4}) {
        set_omp_num_threads(requested);
        REQUIRE(omp_num_threads == requested);
        const int before = get_omp_max_threads();
        potential_wrapper wrapper(model_path(), "mtp");
        REQUIRE(get_omp_max_threads() == before);
        for (int invalid : {0, -1}) {
            REQUIRE_THROWS_AS(set_omp_num_threads(invalid), std::invalid_argument);
            REQUIRE(omp_num_threads == requested);
            REQUIRE(get_omp_max_threads() == before);
        }
    }
}

TEST_CASE("Warm and cold MTP models preserve full EFS when thread requests change", "[openmp][threads][mtp]") {
    ThreadSettings restore;
    set_omp_num_threads(4);
    PotMTP first(model_path()), second(model_path());
    first.omp_init();
    second.omp_init();
    // The first call is cold at fewer threads; subsequent calls grow and shrink
    // both the team and geometry, with two independently owned buffer pools.
    for (int requested : {1, 4, 1, 2, 4}) {
        CAPTURE(requested);
        set_omp_num_threads(requested);
        for (int count : {8, 2}) {
            CAPTURE(count);
            check_efs_and_energy(first, geometry(count));
            check_efs_and_energy(second, geometry(count));
        }
        auto expected = geometry();
        first.getEFS(expected);
        auto added = expected;
        Neighborhoods nbhs(added, first.cutoff);
        first.getEFSOMPadd(added, nbhs, .37);
        compare(added, expected, 1.37);
        first.getEFSOMPadd(added, nbhs);
        compare(added, expected, 2.37);
    }
#ifdef USE_OMP
    // Native callers can set the OpenMP ICV without modifying the CLI option.
    omp_set_num_threads(1);
    REQUIRE(omp_num_threads == 4);
    check_efs_and_energy(first, geometry());
    REQUIRE(omp_get_max_threads() == 1);
#endif
}

TEST_CASE("MTP reductions use the actual dynamic or nested team", "[openmp][threads][mtp]") {
    ThreadSettings restore;
    set_omp_num_threads(4);
    PotMTP pot(model_path());
    auto reference = geometry();
    pot.getEFS(reference);
    auto warm = geometry();
    pot.getEFSOMP(warm);
    compare(warm, reference);
#ifdef USE_OMP
    omp_set_dynamic(1);
    check_efs_and_energy(pot, geometry());
    omp_set_dynamic(0);
    omp_set_max_active_levels(1);
    Atoms nested = geometry(), nested_energy = geometry(), nested_added = reference;
    int observed_inner_team = 0;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            omp_set_num_threads(4);
#pragma omp parallel num_threads(4)
            {
#pragma omp single
                observed_inner_team = omp_get_num_threads();
            }
            pot.getEFSOMP(nested);
            Neighborhoods nbhs(nested_energy, pot.cutoff);
            pot.getEOMP(nested_energy, nbhs);
            pot.getEFSOMPadd(nested_added, nbhs, .37);
        }
    }
    REQUIRE(observed_inner_team == 1);
    compare(nested, reference);
    compare(nested_energy, reference, 1., true);
    compare(nested_added, reference, 1.37);
#endif
}

TEST_CASE("Outer teams can evaluate one MTP with call-local scratch", "[openmp][threads][mtp]") {
    ThreadSettings restore;
    set_omp_num_threads(1);
    PotMTP pot(model_path());
    pot.omp_init();
    auto reference = geometry();
    pot.getEFS(reference);
    std::vector<Atoms> actual(8, geometry());
#ifdef USE_OMP
#pragma omp parallel for num_threads(4)
#endif
    for (int i = 0; i < static_cast<int>(actual.size()); ++i)
        pot.getEFSOMPthread(actual[i]);
    for (const auto &atom : actual) compare(atom, reference);
}

TEST_CASE("MTP descriptor and curvature scratch follows runtime changes and reloads", "[openmp][threads][mtp]") {
    ThreadSettings restore;
    set_omp_num_threads(1);
    PotMTP pot(model_path());
    auto atom = geometry(2);
    Eigen::MatrixXd descriptors;
    pot.CalcLocalDescriptors(atom, descriptors); // works without an earlier omp_init
    HessianMatrix hessian;
    pot.getEFSHessian(atom, hessian);
    const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(6, -.4, .6);
    Eigen::VectorXd expected = Eigen::VectorXd::Zero(6);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            expected.segment<3>(3*i) += hessian(i,j) * direction.segment<3>(3*j);
    for (int requested : {4, 1, 2, 4}) {
        set_omp_num_threads(requested);
        Eigen::MatrixXd actual_desc;
        pot.CalcLocalDescriptors(atom, actual_desc);
        REQUIRE((actual_desc - descriptors).norm() <= 1e-11 * std::max(1., descriptors.norm()));
        Eigen::VectorXd product;
        pot.applyHessianVectorProduct(atom, direction, product);
        REQUIRE((product - expected).norm() <= 1e-10 * std::max(1., expected.norm()));
        HessianMatrix actual_hessian;
        Neighborhoods nbhs(atom, pot.cutoff);
        pot.getEFSHessianOMP(atom, nbhs, actual_hessian);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                REQUIRE((actual_hessian(i,j) - hessian(i,j)).norm() <= 1e-10 * std::max(1., hessian(i,j).norm()));
    }
    pot.Load(model_path("potential.mtp"));
    check_efs_and_energy(pot, geometry());
}

TEST_CASE("Neural MTP wrappers resize both model and network thread scratch", "[openmp][threads][mtpnn]") {
    ThreadSettings restore;
    const int layers = GENERATE(1, 2);
    set_omp_num_threads(1);
    GPSettings settings;
    settings.Modify("mtpcoeff", model_path());
    settings.Modify("nncoeff", "");
    settings.Modify("spcoeff", "");
    settings.Modify("nnlay", std::to_string(layers));
    settings.Modify("nnhidden", "4");
    settings.Modify("nnhidden2", "3");
    PotMTPNN1 pot(settings);
    std::vector<double> coeffs(pot.nn_coeff_size);
    for (int i = 0; i < static_cast<int>(coeffs.size()); ++i) coeffs[i] = .05 * std::sin(i + 1.);
    if (layers == 1) pot.p_nn->set_coeffs(coeffs.data());
    else pot.p_nn2->set_coeffs(coeffs.data());
    for (int requested : {1, 4, 1, 2, 4}) {
        set_omp_num_threads(requested);
        check_efs_and_energy(pot, geometry());
        check_efs_and_energy(pot, geometry(2));
    }
#ifdef USE_OMP
    omp_set_dynamic(1);
    check_efs_and_energy(pot, geometry());
    omp_set_dynamic(0);
    omp_set_max_active_levels(1);
    AnyPotential &api = pot;
    auto expected = geometry(), actual = geometry();
    api.getEFS(expected);
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            omp_set_num_threads(4);
            api.getEFSOMP(actual);
        }
    }
    compare(actual, expected);
#endif
}
