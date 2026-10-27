// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "NNBase/nn1hidden.h"
#include "NNBase/nn2hidden.h"
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>
#include <type_traits>

using Catch::Approx;

// Write the original layout independently of the current serializer.
template<class NN>
static std::string legacy_bytes(const NN &nn) {
    std::ostringstream out(std::ios::binary);
    auto write = [&](const auto &value) {
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    };
    auto array = [&](const auto &values) {
        out.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(double));
    };
    if constexpr (std::is_same_v<NN, IMRMLPNN::NN1lay>) {
        write(nn.n_neuron); write(nn.n_input);
        array(nn.w_input); array(nn.w_hidden); array(nn.b_hidden);
    } else {
        write(nn.n_neuron1); write(nn.n_neuron2); write(nn.n_input);
        array(nn.w_input); array(nn.b_hidden1); array(nn.w_hidden1);
        array(nn.b_hidden2); array(nn.w_hidden2);
    }
    array(nn.norm_mean); array(nn.norm_irstd);
    return out.str();
}

static void write_bytes(const std::string &file, const std::string &bytes) {
    std::ofstream out(file, std::ios::binary);
    out.write(bytes.data(), bytes.size());
}

template<class NN>
static NN network() {
    NN nn;
    if constexpr (std::is_same_v<NN, IMRMLPNN::NN1lay>) nn.setup(3, 4);
    else nn.setup(3, 4, 2);
    std::vector<double> coefficients(nn.get_coeff_size());
    for (size_t i = 0; i < coefficients.size(); ++i) coefficients[i] = 0.1 * std::sin(i + 1);
    nn.set_coeffs(coefficients.data());
    nn.norm_mean.setConstant(0.2);
    nn.norm_irstd.setConstant(0.7);
    return nn;
}

TEMPLATE_TEST_CASE("NN binary roundtrip preserves activation and normalization", "[nn][io]",
                   IMRMLPNN::NN1lay, IMRMLPNN::NN2lay) {
    const auto activation = GENERATE(IMRMLPNN::TANH, IMRMLPNN::SIGMOID, IMRMLPNN::RELU);
    auto original = network<TestType>();
    original.acfunc = activation;
    original.dump_coeffs("roundtrip.nn");
    TestType loaded("roundtrip.nn");
    REQUIRE(loaded.acfunc == activation);
    REQUIRE(loaded.n_input == original.n_input);
    REQUIRE(loaded.norm_mean == original.norm_mean);
    REQUIRE(loaded.norm_irstd == original.norm_irstd);
    const Eigen::VectorXd input = Eigen::VectorXd::Ones(3);
    Eigen::VectorXd derivative(3), restored_derivative(3);
    double energy, restored_energy;
    int tag = 0;
    original.propagate(input, energy, derivative, tag, 0);
    loaded.propagate(input, restored_energy, restored_derivative, tag, 0);
    REQUIRE(restored_energy == Approx(energy));
    REQUIRE(restored_derivative.isApprox(derivative));
}

TEMPLATE_TEST_CASE("NN loader accepts legacy files and rejects corrupt data", "[nn][io]",
                   IMRMLPNN::NN1lay, IMRMLPNN::NN2lay) {
    auto original = network<TestType>();
    const std::string legacy = legacy_bytes(original);
    write_bytes("legacy.nn", legacy);
    TestType loaded("legacy.nn");
    REQUIRE(loaded.acfunc == IMRMLPNN::TANH);
    std::vector<double> expected(original.get_coeff_size()), restored(loaded.get_coeff_size());
    original.get_coeffs(expected.data()); loaded.get_coeffs(restored.data());
    REQUIRE(restored == expected);

    original.dump_coeffs("versioned.nn");
    std::ifstream input("versioned.nn", std::ios::binary);
    const std::string versioned{std::istreambuf_iterator<char>(input), {}};
    for (const auto &bytes : {legacy, versioned}) {
        for (size_t length : {size_t(0), size_t(3), size_t(8), size_t(16), bytes.size() - 1}) {
            write_bytes("broken.nn", bytes.substr(0, length));
            REQUIRE_THROWS_WITH(loaded.load_coeffs("broken.nn"), Catch::Matchers::ContainsSubstring("broken.nn"));
        }
    }
    for (int dimension : {0, -1, std::numeric_limits<int>::max()}) {
        auto bytes = legacy;
        std::memcpy(bytes.data(), &dimension, sizeof(dimension));
        write_bytes("broken.nn", bytes);
        REQUIRE_THROWS(loaded.load_coeffs("broken.nn"));
    }
    for (int field : {1, 2, 3}) { // version, layer count, activation
        auto bytes = versioned;
        const int invalid = 99;
        std::memcpy(bytes.data() + field * sizeof(int), &invalid, sizeof(invalid));
        write_bytes("broken.nn", bytes);
        REQUIRE_THROWS(loaded.load_coeffs("broken.nn"));
    }
    auto bytes = legacy;
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    std::memcpy(bytes.data() + bytes.size() - sizeof(double), &invalid, sizeof(invalid));
    write_bytes("broken.nn", bytes);
    REQUIRE_THROWS(loaded.load_coeffs("broken.nn"));
    loaded.get_coeffs(restored.data());
    REQUIRE(restored == expected); // A failed reload leaves the usable model intact.

    const auto weights_only = legacy.substr(0, legacy.size() - 2 * original.n_input * sizeof(double));
    write_bytes("weights-only.nn", weights_only);
    TestType old("weights-only.nn");
    old.get_coeffs(restored.data());
    REQUIRE(restored == expected);
    REQUIRE(old.norm_mean.isZero());
    REQUIRE(old.norm_irstd.isOnes());
    // Only the complete legacy weights-only variant may omit normalization.
    write_bytes("broken.nn", legacy.substr(0, weights_only.size() - 1));
    REQUIRE_THROWS(loaded.load_coeffs("broken.nn"));
    write_bytes("broken.nn", legacy.substr(0, weights_only.size() + sizeof(double)));
    REQUIRE_THROWS(loaded.load_coeffs("broken.nn"));
    write_bytes("broken.nn", versioned.substr(0, versioned.size() - 2 * original.n_input * sizeof(double)));
    REQUIRE_THROWS(loaded.load_coeffs("broken.nn"));
}

TEMPLATE_TEST_CASE("NN serializers report open and write failures", "[nn][io]",
                   IMRMLPNN::NN1lay, IMRMLPNN::NN2lay) {
    auto nn = network<TestType>();
    REQUIRE_THROWS_WITH(nn.dump_coeffs("missing-directory/model.nn"), Catch::Matchers::ContainsSubstring("model.nn"));
    REQUIRE_THROWS_WITH(nn.dump_coeffs_txt("missing-directory/model.txt"), Catch::Matchers::ContainsSubstring("model.txt"));
    if (!std::filesystem::exists("/dev/full")) SKIP("/dev/full unavailable");
    REQUIRE_THROWS_WITH(nn.dump_coeffs("/dev/full"), Catch::Matchers::ContainsSubstring("/dev/full"));
    REQUIRE_THROWS_WITH(nn.dump_coeffs_txt("/dev/full"), Catch::Matchers::ContainsSubstring("/dev/full"));
}
