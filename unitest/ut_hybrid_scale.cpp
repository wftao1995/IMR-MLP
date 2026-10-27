// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "base/neighborhood.h"
#include "potentials/potential_hybrid_scale.h"

#include <filesystem>
#include <numeric>
#include <unistd.h>

using Catch::Approx;
using namespace Potentials;

static void add_setting(GPSettings &settings, const std::string &key, const std::string &value) {
    settings.settings.emplace(key, value);
}

static Atoms make_two_atom_structure(double distance) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 20.0;
    atom.pos_ = {
        {0.0, 0.0, 0.0},
        {distance, 0.0, 0.0}
    };
    atom.chemmap_ = {0, 0};
    return atom;
}

static double quintic_smoothstep(double x0, double x1, double x) {
    if (x <= x0) return 0.0;
    if (x >= x1) return 1.0;
    const double t = (x - x0) / (x1 - x0);
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
}

static void add_constant_eam(GPSettings &settings, const std::string &prefix,
                             double site_energy) {
    add_setting(settings, prefix + ":pottype", "eampoly");
    add_setting(settings, prefix + ":nelems", "1");
    add_setting(settings, prefix + ":npoly_rho", "2");
    add_setting(settings, prefix + ":npoly_f", "1");
    add_setting(settings, prefix + ":npoly_pair", "2");
    add_setting(settings, prefix + ":rmax_pair_0_0", "6.0");
    add_setting(settings, prefix + ":rmax_rho_0_0", "6.0");
    add_setting(settings, prefix + ":coeffs",
                "0 0 0 0 0 " + std::to_string(site_energy));
}

static GPSettings make_constant_hybrid_settings(const std::string &feature) {
    GPSettings settings;
    add_setting(settings, "pottype", "hybrid_scale");
    add_setting(settings, "npot", "2");
    add_setting(settings, "feature", feature);
    add_setting(settings, "transition", "smoothstep");
    add_setting(settings, "feature_r0", "0.0");
    add_setting(settings, "feature_r1", "5.0");
    add_setting(settings, "boundary1", feature == "min_distance" ? "1.5 2.5" : "0.2 1.2");

    add_constant_eam(settings, "pot1", 1.0);
    add_constant_eam(settings, "pot2", 3.0);

    return settings;
}

static double efs_objective(PotentialHybridScale &pot, Atoms &atom,
                            Neighborhoods &nbhs, double dLdE,
                            const Eigen::Matrix<double, -1, 3> &dLdF,
                            const Eigen::Matrix3d &dLdS) {
    atom.resetEFS();
    pot.getEFS(atom, nbhs);
    double value = dLdE * atom.energy_;
    for (int i = 0; i < atom.size(); ++i) {
        value += dLdF.row(i).dot(atom.forces_[i]);
    }
    value += (dLdS.array() * atom.stress_.array()).sum();
    return value;
}

TEST_CASE("PotentialHybridScale min_distance selects endpoint potentials", "[hybrid_scale]") {
    PotentialHybridScale pot(make_constant_hybrid_settings("min_distance"));

    auto near_atom = make_two_atom_structure(1.0);
    Neighborhoods near_nbhs(near_atom, pot.cutoff);
    near_atom.resetEFS();
    pot.getEFS(near_atom, near_nbhs);
    REQUIRE(near_atom.energy_ == Approx(2.0).epsilon(1e-12));

    auto far_atom = make_two_atom_structure(3.0);
    Neighborhoods far_nbhs(far_atom, pot.cutoff);
    far_atom.resetEFS();
    pot.getEFS(far_atom, far_nbhs);
    REQUIRE(far_atom.energy_ == Approx(6.0).epsilon(1e-12));
}

TEST_CASE("PotentialHybridScale smooth_coordination gives expected weighted energy", "[hybrid_scale]") {
    PotentialHybridScale pot(make_constant_hybrid_settings("smooth_coordination"));
    auto atom = make_two_atom_structure(2.0);
    Neighborhoods nbhs(atom, pot.cutoff);

    atom.resetEFS();
    pot.getEFS(atom, nbhs);

    const double q = 1.0 - quintic_smoothstep(0.0, 5.0, 2.0);
    const double w2 = quintic_smoothstep(0.2, 1.2, q);
    const double expected_site = (1.0 - w2) * 1.0 + w2 * 3.0;
    REQUIRE(atom.energy_ == Approx(2.0 * expected_site).epsilon(1e-12));
}

TEST_CASE("PotentialHybridScale force matches finite difference for feature derivative", "[hybrid_scale]") {
    PotentialHybridScale pot(make_constant_hybrid_settings("smooth_coordination"));
    auto atom = make_two_atom_structure(2.0);
    Neighborhoods nbhs(atom, pot.cutoff);

    atom.resetEFS();
    pot.getEFS(atom, nbhs);
    const auto analytical_forces = atom.forces_;

    const double dx = 1e-6;
    for (int i = 0; i < atom.size(); ++i) {
        for (int a = 0; a < 3; ++a) {
            atom.pos_[i][a] += dx;
            nbhs.init(atom, pot.cutoff);
            atom.resetEFS();
            pot.getEFS(atom, nbhs);
            const double e_plus = atom.energy_;

            atom.pos_[i][a] -= 2.0 * dx;
            nbhs.init(atom, pot.cutoff);
            atom.resetEFS();
            pot.getEFS(atom, nbhs);
            const double e_minus = atom.energy_;

            atom.pos_[i][a] += dx;
            const double fd_force = -(e_plus - e_minus) / (2.0 * dx);
            INFO("atom=" << i << " component=" << a);
            REQUIRE(analytical_forces[i][a] == Approx(fd_force).margin(1e-6));
        }
    }
}

TEST_CASE("PotentialHybridScale stick-breaking weights normalize with overlapping boundaries",
          "[hybrid_scale]") {
    GPSettings settings = make_constant_hybrid_settings("smooth_coordination");
    settings.Modify("npot", "3");
    settings.Modify("boundary1", "0.2 1.0");
    add_setting(settings, "boundary2", "0.4 1.2");
    add_constant_eam(settings, "pot3", 5.0);

    PotentialHybridScale pot(settings);
    auto atom = make_two_atom_structure(2.0);
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getE(atom, nbhs);

    const double q = 1.0 - quintic_smoothstep(0.0, 5.0, 2.0);
    const double s0 = quintic_smoothstep(0.2, 1.0, q);
    const double s1 = quintic_smoothstep(0.4, 1.2, q);
    const double expected_site = (1.0 - s0) + s0 * (1.0 - s1) * 3.0
                               + s0 * s1 * 5.0;
    REQUIRE(atom.energy_ == Approx(2.0 * expected_site).epsilon(1e-12));
}

TEST_CASE("PotentialHybridScale energy-only path preserves forces and stress",
          "[hybrid_scale]") {
    PotentialHybridScale pot(make_constant_hybrid_settings("smooth_coordination"));
    auto atom = make_two_atom_structure(2.0);
    atom.forces_.assign(atom.size(), Eigen::RowVector3d(1.0, 2.0, 3.0));
    atom.stress_.setConstant(4.0);
    const auto forces_before = atom.forces_;
    const auto stress_before = atom.stress_;
    Neighborhoods nbhs(atom, pot.cutoff);

    pot.getE(atom, nbhs);

    REQUIRE(atom.forces_ == forces_before);
    REQUIRE(atom.stress_ == stress_before);
    REQUIRE(atom.have_energy);
    REQUIRE_FALSE(atom.have_stress);
}

TEST_CASE("PotentialHybridScale exposes concatenated parameter groups",
          "[hybrid_scale]") {
    PotentialHybridScale pot(make_constant_hybrid_settings("smooth_coordination"));
    REQUIRE(pot.get_NL_coeffs_count() == 12);
    const auto groups = pot.nl_parameter_groups();
    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].name == "pot1/all");
    REQUIRE(groups[0].begin == 0);
    REQUIRE(groups[0].end == 6);
    REQUIRE(groups[1].name == "pot2/all");
    REQUIRE(groups[1].begin == 6);
    REQUIRE(groups[1].end == 12);

    std::vector<double> before(12), after(12);
    pot.nl_get_coeffs(before.data());
    before[2] += 0.25;
    before[9] -= 0.5;
    pot.nl_set_coeffs(before.data());
    pot.nl_get_coeffs(after.data());
    REQUIRE(after == before);
}

TEST_CASE("PotentialHybridScale analytical EFS parameter gradient matches finite difference",
          "[hybrid_scale]") {
    PotentialHybridScale pot(make_constant_hybrid_settings("smooth_coordination"));
    auto atom = make_two_atom_structure(2.0);
    Neighborhoods nbhs(atom, pot.cutoff);
    const int ncoeff = pot.get_NL_coeffs_count();

    Eigen::Matrix<double, -1, 3> dLdF(atom.size(), 3);
    dLdF << 0.3, -0.2, 0.1,
           -0.4, 0.5, -0.6;
    Eigen::Matrix3d dLdS;
    dLdS << 0.2, -0.1, 0.05,
            0.03, 0.1, -0.04,
            -0.02, 0.06, -0.08;
    const double dLdE = 0.7;
    std::vector<double> analytical(ncoeff, 0.0);
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buffer;
    pot.AccumulateEFSCombinationGard(atom, nbhs, dLdE, dLdF, dLdS,
                                     buffer, analytical.data());

    std::vector<double> coeffs(ncoeff);
    pot.nl_get_coeffs(coeffs.data());
    const double h = 1e-6;
    for (int i = 0; i < ncoeff; ++i) {
        auto trial = coeffs;
        trial[i] += h;
        pot.nl_set_coeffs(trial.data());
        const double plus = efs_objective(pot, atom, nbhs, dLdE, dLdF, dLdS);
        trial[i] -= 2.0 * h;
        pot.nl_set_coeffs(trial.data());
        const double minus = efs_objective(pot, atom, nbhs, dLdE, dLdF, dLdS);
        INFO("coefficient=" << i);
        REQUIRE(analytical[i] == Approx((plus - minus) / (2.0 * h)).margin(2e-5));
    }
    pot.nl_set_coeffs(coeffs.data());
}

TEST_CASE("PotentialHybridScale energy-only parameter gradient matches finite difference",
          "[hybrid_scale]") {
    PotentialHybridScale pot(make_constant_hybrid_settings("min_distance"));
    auto atom = make_two_atom_structure(2.0);
    Neighborhoods nbhs(atom, pot.cutoff);
    const int ncoeff = pot.get_NL_coeffs_count();
    std::vector<double> analytical(ncoeff, 0.0);
    constexpr double dLdE = -0.4;
    pot.AccumulateEFSCombinationGard(atom, nbhs, dLdE, analytical.data());

    std::vector<double> coeffs(ncoeff);
    pot.nl_get_coeffs(coeffs.data());
    const double h = 1e-6;
    for (int i = 0; i < ncoeff; ++i) {
        auto trial = coeffs;
        trial[i] += h;
        pot.nl_set_coeffs(trial.data());
        pot.getE(atom, nbhs);
        const double plus = dLdE * atom.energy_;
        trial[i] -= 2.0 * h;
        pot.nl_set_coeffs(trial.data());
        pot.getE(atom, nbhs);
        const double minus = dLdE * atom.energy_;
        INFO("coefficient=" << i);
        REQUIRE(analytical[i] == Approx((plus - minus) / (2.0 * h)).margin(2e-6));
    }
    pot.nl_set_coeffs(coeffs.data());
}

TEST_CASE("PotentialHybridScale honors additive EFS mode", "[hybrid_scale]") {
    PotentialHybridScale pot(make_constant_hybrid_settings("smooth_coordination"));
    auto atom = make_two_atom_structure(2.0);
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);
    const double energy_once = atom.energy_;
    const auto forces_once = atom.forces_;

    pot.add_efs = true;
    pot.getEFS(atom, nbhs);

    REQUIRE(atom.energy_ == Approx(2.0 * energy_once).epsilon(1e-12));
    for (int i = 0; i < atom.size(); ++i) {
        REQUIRE(atom.forces_[i].isApprox(2.0 * forces_once[i], 1e-12));
    }
}

TEST_CASE("PotentialHybridScale manifest roundtrip preserves predictions",
          "[hybrid_scale]") {
    PotentialHybridScale original(make_constant_hybrid_settings("smooth_coordination"));
    auto atom = make_two_atom_structure(2.0);
    Neighborhoods nbhs(atom, original.cutoff);
    original.getEFS(atom, nbhs);
    const double expected_energy = atom.energy_;
    const auto expected_forces = atom.forces_;

    const std::filesystem::path manifest = std::filesystem::path("/tmp")
        / ("imrmlp_hybrid_scale_" + std::to_string(getpid()) + ".settings");
    original.Save(manifest.string());
    GPSettings reload_settings;
    add_setting(reload_settings, "potfile", manifest.string());
    PotentialHybridScale reloaded(reload_settings);
    auto reloaded_atom = make_two_atom_structure(2.0);
    Neighborhoods reloaded_nbhs(reloaded_atom, reloaded.cutoff);
    reloaded.getEFS(reloaded_atom, reloaded_nbhs);

    REQUIRE(reloaded_atom.energy_ == Approx(expected_energy).epsilon(1e-7));
    for (int i = 0; i < atom.size(); ++i) {
        REQUIRE(reloaded_atom.forces_[i].isApprox(expected_forces[i], 1e-7));
    }
    std::filesystem::remove(manifest);
    std::filesystem::remove(manifest.string() + ".pot1");
    std::filesystem::remove(manifest.string() + ".pot2");
}

TEST_CASE("PotentialHybridScale rejects malformed and unordered boundaries",
          "[hybrid_scale]") {
    auto malformed = make_constant_hybrid_settings("smooth_coordination");
    malformed.Modify("boundary1", "0.2 1.2 trailing");
    REQUIRE_THROWS_AS(PotentialHybridScale(malformed), GPException);

    auto unordered = make_constant_hybrid_settings("smooth_coordination");
    unordered.Modify("npot", "3");
    unordered.Modify("boundary1", "0.5 1.5");
    add_setting(unordered, "boundary2", "0.4 1.6");
    add_constant_eam(unordered, "pot3", 5.0);
    REQUIRE_THROWS_AS(PotentialHybridScale(unordered), GPException);

    auto nonfinite = make_constant_hybrid_settings("smooth_coordination");
    nonfinite.Modify("feature_r1", "nan");
    REQUIRE_THROWS_AS(PotentialHybridScale(nonfinite), GPException);

    auto mismatched_species = make_constant_hybrid_settings("smooth_coordination");
    mismatched_species.Modify("pot2:nelems", "2");
    REQUIRE_THROWS_AS(PotentialHybridScale(mismatched_species), GPException);
}

TEST_CASE("Hybrid scale neighborhoods include the feature range", "[hybrid_scale]") {
    const std::string feature = GENERATE(std::string("smooth_coordination"), std::string("min_distance"));
    auto settings = make_constant_hybrid_settings(feature);
    settings.Modify("feature_r1", "12");
    settings.Modify("boundary1", feature == "min_distance" ? "4 10" : "0 1");
    PotentialHybridScale pot(settings);
    REQUIRE(pot.cutoff == Approx(feature == "min_distance" ? 10 : 12));
    REQUIRE(pot.cutoffs(0, 0) == Approx(pot.cutoff));
    REQUIRE(pot.cutoffsq == Approx(pot.cutoff * pot.cutoff));

    auto atom = make_two_atom_structure(7);
    atom.pbc_ = {false, false, false};
    pot.getEFS(atom);
    const double q = feature == "min_distance" ? 7 : 1 - quintic_smoothstep(0, 12, 7);
    const double w = feature == "min_distance" ? quintic_smoothstep(4, 10, q)
                                               : quintic_smoothstep(0, 1, q);
    REQUIRE(atom.energy_ == Approx(2 + 4 * w));
    const double force = atom.forces_[1][0];
    const double h = 1e-6;
    atom.pos_[1][0] += h;
    pot.getE(atom);
    const double plus = atom.energy_;
    atom.pos_[1][0] -= 2 * h;
    pot.getE(atom);
    REQUIRE(force == Approx(-(plus - atom.energy_) / (2 * h)).margin(1e-8));

    atom.pos_[1][0] = 6 - h;
    pot.getE(atom);
    const double below = atom.energy_;
    atom.pos_[1][0] = 6 + h;
    pot.getE(atom);
    REQUIRE(std::abs(below - atom.energy_) < 1e-5);
}
