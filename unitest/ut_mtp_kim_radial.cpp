// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "basis/basis.h"
#include "mtp_kim/mtp_kim_radial.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct BasisCase {
    std::string name;
    std::string extra;
};

std::string parameters(const BasisCase &test_case)
{
    return "min_dist = 0.5\n"
           "max_dist = 5.0\n"
           "radial_basis_size = 6\n" + test_case.extra;
}

} // namespace

TEST_CASE("MTP KIM radial bases match the IMRMLP implementations",
          "[mtp_kim][radial]")
{
    const std::vector<BasisCase> cases = {
        {"RBChebyshev", ""},
        {"RBChebyshevPoly4", "rb_b = -0.7\n"},
        {"RBChebyshevPoly5", "rb_b = -0.7\n"},
        {"RBPolyB", ""},
        {"RBPolyBCos", ""},
        {"RBChebyCos", ""},
        {"RBChebyshevConfigurable",
         "cutoff_type = quadratic\nshift_type = linear\n"},
        {"RBChebyshevConfigurable",
         "cutoff_type = cosine\nshift_type = exponential\nshift_lambda = 1.2\n"},
        {"RBChebyshevConfigurable",
         "cutoff_type = poly5\ncutoff_b = -0.7\n"
         "shift_type = exponential\nshift_lambda = 1.2\n"},
        {"RBChebyshevConfigurable",
         "basis_function_type = one_minus_chebyshev\n"
         "cutoff_type = normalized_quadratic\n"
         "shift_type = i2\n"},
        {"RBChebyshevConfigurable",
         "basis_function_type = chebyshev\n"
         "cutoff_type = cosine_left\n"
         "shift_type = exponential_left\nshift_lambda = 1.2\n"},
        {"RBChebyshevConfigurable",
         "basis_function_type = chebyshev\n"
         "cutoff_type = sine_squared\n"
         "shift_type = exponential_tanh\n"
         "shift_lambda = 1.1\nshift_scale_lambda = 0.55\n"},
        {"RBChebyshevConfigurable",
         "basis_function_type = chebyshev\n"
         "cutoff_type = poly3\ncutoff_b = -0.7\n"
         "shift_type = linear\n"},
        {"RBChebyshevConfigurable",
         "basis_function_type = chebyshev\n"
         "cutoff_type = smoothstep5\nshift_type = linear\n"},
        {"RBChebyshevConfigurable",
         "basis_function_type = chebyshev\n"
         "cutoff_type = smoothstep7\n"
         "cutoff_r1 = 1.5\ncutoff_r2 = 3.8\n"
         "shift_type = linear\n"},
    };
    constexpr double scaling = 0.037;
    const std::vector<double> radii = {0.7, 1.5, 2.4, 3.8, 4.7};

    for (const auto &test_case : cases) {
        DYNAMIC_SECTION(test_case.name << " / " << test_case.extra) {
            const std::string text = parameters(test_case);
            std::istringstream native_input(text);
            std::unique_ptr<AnyBasis> native(
                create_basis_by_name(test_case.name, native_input, scaling));
            std::istringstream kim_input(text);
            const auto kim = mtp_kim::RadialBasis::load(
                test_case.name, kim_input, scaling);

            REQUIRE(kim->basis_size == native->basis_size);
            std::vector<double> kim_values(kim->basis_size);
            std::vector<double> kim_derivatives(kim->basis_size);
            std::vector<double> native_values(kim->basis_size);
            std::vector<double> native_derivatives(kim->basis_size);
            // RBPolyB uses a deliberately ill-conditioned overlap matrix.
            // The portable Jacobi solver and Eigen's eigensolver can differ
            // slightly while representing the same inverse square root.
            const bool orthogonal_polynomial =
                test_case.name == "RBPolyB" || test_case.name == "RBPolyBCos";
            const double relative_tolerance = orthogonal_polynomial ? 2e-6 : 2e-10;
            const double absolute_tolerance = orthogonal_polynomial ? 5e-9 : 2e-11;
            for (double radius : radii) {
                kim->evaluate(radius, kim_values.data(), kim_derivatives.data());
                native->calc_der(radius, native_values, native_derivatives);
                for (int i = 0; i < kim->basis_size; ++i) {
                    CAPTURE(radius, i, kim_values[i], native_values[i],
                            kim_derivatives[i], native_derivatives[i]);
                    CHECK_THAT(kim_values[i],
                        Catch::Matchers::WithinRel(native_values[i], relative_tolerance) ||
                        Catch::Matchers::WithinAbs(native_values[i], absolute_tolerance));
                    CHECK_THAT(kim_derivatives[i],
                        Catch::Matchers::WithinRel(native_derivatives[i], relative_tolerance) ||
                        Catch::Matchers::WithinAbs(native_derivatives[i], absolute_tolerance));
                }
            }
        }
    }
}

TEST_CASE("MTP KIM radial parser consumes exactly one split basis",
          "[mtp_kim][radial][parser]")
{
    std::istringstream input(
        "min_dist = 0.5\n"
        "max_dist = 5.0\n"
        "radial_basis_size = 4\n"
        "pair 1 0\n");
    const auto basis = mtp_kim::RadialBasis::load("RBChebyCos", input, 0.1);
    std::string next;
    input >> next;
    CHECK(next == "pair");
}

TEST_CASE("MTP KIM supports the constant radial basis",
          "[mtp_kim][radial]")
{
    std::istringstream input(parameters({"RBConstant", ""}));
    const auto basis = mtp_kim::RadialBasis::load("RBConstant", input, 0.037);
    std::vector<double> values(1), derivatives(1);
    for (double radius : {0.7, 2.4, 4.7}) {
        basis->evaluate(radius, values.data(), derivatives.data());
        const double delta = radius - 5.0;
        CHECK(values[0] == delta * delta);
        CHECK(derivatives[0] == 2.0 * delta);
    }
}

TEST_CASE("MTP KIM orthogonal polynomial bases work at practical sizes",
          "[mtp_kim][radial][polyb]")
{
    constexpr double scaling = 0.037;
    for (const std::string name : {"RBPolyB", "RBPolyBCos"}) {
        for (int size : {1, 3, 6, 8}) {
            DYNAMIC_SECTION(name << " size=" << size) {
                const std::string text =
                    "min_dist = 0.5\nmax_dist = 5.0\nradial_basis_size = " +
                    std::to_string(size) + "\n";
                std::istringstream native_input(text);
                std::unique_ptr<AnyBasis> native(
                    create_basis_by_name(name, native_input, scaling));
                std::istringstream kim_input(text);
                const auto kim = mtp_kim::RadialBasis::load(
                    name, kim_input, scaling);
                std::vector<double> values(size), derivatives(size);
                std::vector<double> native_values(size), native_derivatives(size);
                // At size 8 the monomial overlap matrix is sufficiently
                // ill-conditioned that independent double- and extended-
                // precision eigensolvers no longer agree to machine precision.
                const double relative_tolerance = size < 8 ? 2e-6 : 2e-3;
                const double absolute_tolerance = size < 8 ? 5e-9 : 1e-7;
                for (double radius : {0.7, 2.4, 4.7}) {
                    kim->evaluate(radius, values.data(), derivatives.data());
                    native->calc_der(radius, native_values, native_derivatives);
                    for (int i = 0; i < size; ++i) {
                        CAPTURE(name, size, radius, i, values[i], native_values[i],
                                derivatives[i], native_derivatives[i]);
                        CHECK_THAT(values[i],
                            Catch::Matchers::WithinRel(native_values[i], relative_tolerance) ||
                            Catch::Matchers::WithinAbs(native_values[i], absolute_tolerance));
                        CHECK_THAT(derivatives[i],
                            Catch::Matchers::WithinRel(native_derivatives[i], relative_tolerance) ||
                            Catch::Matchers::WithinAbs(native_derivatives[i], absolute_tolerance));
                    }
                }
            }
        }
    }
}

TEST_CASE("MTP KIM radial parser rejects invalid definitions",
          "[mtp_kim][radial][parser]")
{
    SECTION("unknown basis") {
        std::istringstream input("min_dist = 0.5\nmax_dist = 5\nradial_basis_size = 4\n");
        CHECK_THROWS_WITH(mtp_kim::RadialBasis::load("RBDoesNotExist", input, 1.0),
                          "unsupported basis type RBDoesNotExist");
    }
    SECTION("invalid interval") {
        std::istringstream input("min_dist = 5\nmax_dist = 0.5\nradial_basis_size = 4\n");
        CHECK_THROWS_WITH(mtp_kim::RadialBasis::load("RBChebyshev", input, 1.0),
                          "radial basis requires min_dist < max_dist");
    }
    SECTION("missing parameter") {
        std::istringstream input("min_dist = 0.5\nmax_dist = 5\nradial_basis_size = 4\n");
        CHECK_THROWS_WITH(mtp_kim::RadialBasis::load("RBChebyshevPoly5", input, 1.0),
                          "expected rb_b = <value>");
    }
}
