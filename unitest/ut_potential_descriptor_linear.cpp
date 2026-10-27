// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_descriptor_linear.h"
#include "potentials/potential_pair_table.h"
#include "potentials/potential_wrapper.h"
#include <cmath>
#include <filesystem>
#include <Eigen/Geometry>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unistd.h>

using Catch::Approx;
using Potentials::PotentialDescriptorLinear;

namespace {
GPSettings behler_settings(double radius = 5) {
    GPSettings s;
    s.Modify("descriptor:dtype", "behler");
    s.Modify("descriptor:behler:dispcount", "2");
    s.Modify("descriptor:behler:dispselect", "1 2");
    s.Modify("descriptor:behler:eta", "0.3");
    s.Modify("descriptor:behler:cutoff", std::to_string(radius));
    return s;
}
Atoms cluster() {
    Atoms at;
    at.cell_ = 24 * Eigen::Matrix3d::Identity();
    at.pos_ = {{0, 0, 0}, {1.7, 0.2, 0.3}, {-0.4, 1.6, 0.5}};
    at.chemmap_ = {0, 0, 0};
    return at;
}
void evaluate(PotentialDescriptorLinear &pot, Atoms &at) {
    static_cast<Potentials::AnyPotential &>(pot).getEFS(at);
}
struct ModelFile {
    std::string path;
    ModelFile() {
        char name[] = "/tmp/imrmlp_descriptor_XXXXXX";
        int fd = mkstemp(name);
        if (fd < 0) throw std::runtime_error("cannot create temporary model");
        close(fd);
        path = name;
    }
    ~ModelFile() { std::filesystem::remove(path); }
};
}

TEST_CASE("Linear descriptor models preserve predictions through the potential factory", "[potential][descriptor_linear]") {
    const std::string dtype = GENERATE("behler", "bondorder");
    CAPTURE(dtype);
    auto settings = behler_settings();
    settings.Modify("descriptor:dtype", dtype);
    settings.Modify("descriptor:soap:ntypes", "1");
    settings.Modify("descriptor:soap:nmax", "2");
    settings.Modify("descriptor:soap:lmax", "1");
    settings.Modify("descriptor:bondorder:cutoff", "5");
    settings.Modify("descriptor:biso3_lmax", "2");
    settings.Modify("descriptor:biso3_rsize", "3");
    settings.Modify("descriptor:biso4:twojmax", "3");
    settings.Modify("species_count", "1");
    PotentialDescriptorLinear original(settings);
    Eigen::VectorXd coeffs = Eigen::VectorXd::LinSpaced(original.get_NL_coeffs_count(), -0.173, 0.612);
    original.nl_set_coeffs(coeffs.data());
    auto reference = cluster();
    evaluate(original, reference);
    ModelFile file;
    original.Save(file.path);
    REQUIRE(std::filesystem::file_size(file.path) > 0);
    // Saved configuration takes precedence over unrelated caller defaults.
    GPSettings load;
    load.Modify("pottype", "descriptor_linear");
    load.Modify("potfile", file.path);
    load.Modify("descriptor_linear:descriptor:dtype", "unknown");
    Potentials::potential_wrapper wrapper(load);
    auto &restored = *wrapper.pot;
    REQUIRE(restored.species_count == 1);
    REQUIRE(restored.cutoff == original.cutoff);
    Eigen::VectorXd saved(coeffs.size());
    restored.nl_get_coeffs(saved.data());
    REQUIRE(saved == coeffs);
    auto predicted = cluster();
    restored.getEFS(predicted);
    REQUIRE(predicted.energy_ == Approx(reference.energy_).margin(1e-12));
    REQUIRE((predicted.stress_ - reference.stress_).norm() < 1e-11);
    for (int i = 0; i < predicted.size(); ++i)
        REQUIRE((predicted.forces_[i] - reference.forces_[i]).norm() < 1e-11);
}

TEST_CASE("Linear descriptor loading rejects incomplete and corrupt model files", "[potential][descriptor_linear]") {
    PotentialDescriptorLinear pot(behler_settings());
    std::ostringstream stream;
    pot.Save(stream);
    std::string model = stream.str();
    const int failure = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
    CAPTURE(failure);
    auto replace_line = [&](const std::string &key, const std::string &value) {
        auto start = model.find(key + ' ');
        REQUIRE(start != std::string::npos);
        model.replace(start, model.find('\n', start) - start, key + ' ' + value);
    };
    if (failure == 0) model.clear();
    if (failure == 1) replace_line("descriptor_linear_version", "2");
    if (failure == 2) replace_line("feature_count", "3");
    if (failure == 3) replace_line("coefficients", "0.1");
    if (failure == 4) replace_line("coefficients", "0.1 0.2 garbage");
    if (failure == 5) replace_line("coefficients", "0.1 nan");
    if (failure == 6) model += "feature_count 2\n";
    if (failure == 7) replace_line("cutoff", "4.0");
    ModelFile file;
    std::ofstream(file.path) << model;
    REQUIRE_THROWS(PotentialDescriptorLinear(file.path));
}

TEST_CASE("Unsupported native descriptor reloads fail and preserve predictions", "[potential][descriptor_linear][load-contract]") {
    const bool from_file = GENERATE(false, true);
    CAPTURE(from_file);
    PotentialDescriptorLinear original(behler_settings()), replacement(behler_settings());
    const auto weights = Eigen::VectorXd::LinSpaced(original.get_NL_coeffs_count(), .4, .9).eval();
    const auto other_weights = Eigen::VectorXd::LinSpaced(replacement.get_NL_coeffs_count(), -.8, -.2).eval();
    original.nl_set_coeffs(weights.data());
    replacement.nl_set_coeffs(other_weights.data());
    auto reference = cluster(), other = cluster();
    evaluate(original, reference);
    evaluate(replacement, other);
    REQUIRE(std::abs(other.energy_ - reference.energy_) > 1e-6);
    const double cutoff = original.cutoff;
    const int species_count = original.species_count;

    std::ostringstream saved;
    replacement.Save(saved);
    Potentials::AnyPotential &api = original;
    if (from_file) {
        ModelFile file;
        { std::ofstream output(file.path); output << saved.str(); REQUIRE(output.good()); }
        REQUIRE_THROWS_WITH(api.Load(file.path), Catch::Matchers::ContainsSubstring("Load(istream)")
                            && Catch::Matchers::ContainsSubstring("not supported"));
    } else {
        std::istringstream input(saved.str());
        REQUIRE_THROWS_WITH(api.Load(input), Catch::Matchers::ContainsSubstring("Load(istream)")
                            && Catch::Matchers::ContainsSubstring("not supported"));
    }
    Eigen::VectorXd actual_weights(weights.size());
    api.nl_get_coeffs(actual_weights.data());
    REQUIRE(actual_weights == weights);
    REQUIRE(api.cutoff == cutoff);
    REQUIRE(api.species_count == species_count);
    auto actual = cluster();
    api.getEFS(actual);
    REQUIRE(actual.energy_ == Approx(reference.energy_).margin(1e-12));
    REQUIRE(actual.site_energies_ == reference.site_energies_);
    REQUIRE((actual.stress_ - reference.stress_).norm() < 1e-11);
    for (int i = 0; i < actual.size(); ++i)
        REQUIRE((actual.forces_[i] - reference.forces_[i]).norm() < 1e-11);
}

TEST_CASE("Supported native table reloads retain stream and filename dispatch", "[potential][descriptor_linear][load-contract]") {
    const bool from_file = GENERATE(false, true);
    CAPTURE(from_file);
    Potentials::PairTable potential;
    Potentials::AnyPotential &api = potential;
    std::istringstream initial("H\nN 4\n0 .5 .25\n1 1.5 2.25\n2 2.5 6.25\n3 3.5 12.25\n");
    api.Load(initial);
    auto reference = cluster();
    api.getEFS(reference);
    REQUIRE(std::abs(reference.energy_) > 1e-6);
    const std::string replacement = "H\nN 4\n0 .5 .5\n1 1.5 4.5\n2 2.5 12.5\n3 3.5 24.5\n";
    if (from_file) {
        ModelFile file;
        { std::ofstream output(file.path); output << replacement; REQUIRE(output.good()); }
        api.Load(file.path);
    } else {
        std::istringstream input(replacement);
        api.Load(input);
    }
    REQUIRE(api.species_count == 1);
    REQUIRE(api.cutoff == Approx(3.5));
    auto actual = cluster();
    api.getEFS(actual);
    REQUIRE(actual.energy_ == Approx(2 * reference.energy_).margin(1e-11));
    REQUIRE((actual.stress_ - 2 * reference.stress_).norm() < 1e-10);
    for (int i = 0; i < actual.size(); ++i) {
        REQUIRE(actual.site_energies_[i] == Approx(2 * reference.site_energies_[i]).margin(1e-11));
        REQUIRE((actual.forces_[i] - 2 * reference.forces_[i]).norm() < 1e-10);
    }
}

TEST_CASE("Linear descriptor force and stress parameter gradients match finite differences", "[potential][descriptor_linear]") {
    const int mode = GENERATE(0, 1, 2, 3);
    PotentialDescriptorLinear pot(behler_settings());
    double coeffs[] = {0.7, -0.2};
    pot.nl_set_coeffs(coeffs);
    auto at = cluster();
    Neighborhoods nbhs(at, pot.cutoff);
    const double we = (mode == 0 || mode == 3) ? 0.8 : 0;
    Eigen::Matrix<double, -1, 3> wf = Eigen::Matrix<double, -1, 3>::Zero(at.size(), 3);
    Eigen::Matrix3d ws = Eigen::Matrix3d::Zero();
    if (mode == 1 || mode == 3) wf << 0.2, 0.1, -0.3, -0.4, 0.2, 0.5, 0.3, -0.2, 0.1;
    if (mode == 2 || mode == 3) ws << 0.5, -0.1, 0.2, -0.1, 0.3, 0.4, 0.2, 0.4, -0.2;
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buf;
    double grad[] = {0, 0};
    pot.AccumulateEFSCombinationGard(at, nbhs, we, wf, ws, buf, grad);
    auto objective = [&] {
        pot.nl_set_coeffs(coeffs);
        pot.getEFS(at, nbhs);
        double value = we * at.energy_ + (ws.array() * at.stress_.array()).sum();
        for (int i = 0; i < at.size(); ++i) value += wf.row(i).dot(at.forces_[i]);
        return value;
    };
    for (int f = 0; f < 2; ++f) {
        CAPTURE(mode, f);
        const double original = coeffs[f], h = 1e-6;
        coeffs[f] = original + h;
        const double plus = objective();
        coeffs[f] = original - h;
        const double minus = objective();
        coeffs[f] = original;
        REQUIRE(grad[f] == Approx((plus - minus) / (2*h)).margin(1e-8));
    }
}

TEST_CASE("Linear descriptor site energies sum to total including additive evaluation", "[potential][descriptor_linear]") {
    PotentialDescriptorLinear pot(behler_settings());
    double coeffs[] = {0.7, -0.2};
    pot.nl_set_coeffs(coeffs);
    auto at = cluster();
    evaluate(pot, at);
    const auto sites = at.site_energies_;
    const double energy = at.energy_;
    REQUIRE(energy != 0);
    REQUIRE(std::accumulate(sites.begin(), sites.end(), 0.) == Approx(energy));
    REQUIRE(sites[0] != Approx(sites[1]));
    pot.add_efs = true;
    evaluate(pot, at);
    REQUIRE(at.energy_ == Approx(2 * energy));
    for (int i = 0; i < at.size(); ++i) REQUIRE(at.site_energies_[i] == Approx(2 * sites[i]));
}

TEST_CASE("Linear descriptor neighbor radius follows the descriptor with its configured cutoff", "[potential][descriptor_linear]") {
    GPSettings s = behler_settings(8);
    PotentialDescriptorLinear pot(s);
    descriptor_wrapper desc(s.ExtractSubSettings("descriptor"));
    REQUIRE(pot.cutoff == Approx(desc.desc->cutoff_));
    REQUIRE(pot.cutoffsq == Approx(pot.cutoff * pot.cutoff));
    REQUIRE(pot.cutoff > 6);
    Eigen::VectorXd coeffs = Eigen::VectorXd::LinSpaced(pot.get_NL_coeffs_count(), 0.2, 0.6);
    pot.nl_set_coeffs(coeffs.data());
    auto at = cluster();
    at.pos_ = {{0, 0, 0}, {6, 0, 0}};
    at.chemmap_ = {0, 0};
    Neighborhoods reference(at, desc.desc->cutoff_);
    double expected = 0;
    for (int i = 0; i < reference.size(); ++i) {
        Eigen::VectorXd values;
        desc.compute(reference[i], values);
        expected += coeffs.dot(values);
    }
    evaluate(pot, at);
    REQUIRE(expected > 0);
    REQUIRE(at.energy_ == Approx(expected).margin(1e-12));
    Neighborhoods truncated(at, 5);
    pot.getEFS(at, truncated);
    REQUIRE(std::abs(at.energy_ - expected) > 1e-6);
}

TEST_CASE("Linear descriptor explicit cutoffs must agree with the descriptor", "[potential][descriptor_linear]") {
    auto s = behler_settings(8);
    const std::string key = GENERATE("cutoff", "descriptor:cutoff");
    s.Modify(key, GENERATE("4", "9", "0", "-1", "nan"));
    REQUIRE_THROWS_WITH(PotentialDescriptorLinear(s), Catch::Matchers::ContainsSubstring("must match"));
    s.Modify(key, "8");
    PotentialDescriptorLinear pot(s);
    REQUIRE(pot.cutoff == 8);
}

TEST_CASE("Linear descriptor legacy cutoff initializes descriptors without a radius default", "[potential][descriptor_linear]") {
    const std::string type = GENERATE("behler", "bondorder");
    const std::string key = GENERATE("cutoff", "descriptor:cutoff", "");
    GPSettings s;
    s.Modify("descriptor:dtype", type);
    s.Modify("descriptor:behler:dispcount", "1");
    if (!key.empty()) s.Modify(key, "6");
    PotentialDescriptorLinear pot(s);
    REQUIRE(pot.cutoff == (key.empty() ? 5 : 6));
    auto at = cluster();
    evaluate(pot, at);
    REQUIRE(std::isfinite(at.energy_));
}
