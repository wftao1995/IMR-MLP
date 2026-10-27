// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "basis/basis.h"
#include <cmath>
#include <memory>
#include <vector>

using Catch::Approx;

// ── Numerical differentiation helpers ────────────────────────────────────

/// Central-difference first derivative with adaptive step near boundaries.
/// When x is within h of a boundary in [lo, hi], shifts the stencil
/// inward so it stays inside the domain.
static double num_der_safe(const std::function<double(double)> &f,
                           double x, double h = 1e-6,
                           double lo = -1e300, double hi = 1e300)
{
    // If stencil would leave domain, use one‑sided difference
    if (x - h < lo) {
        // forward difference
        return (-f(x + 2*h) + 4*f(x + h) - 3*f(x)) / (2.0 * h);
    }
    if (x + h > hi) {
        // backward difference
        return (f(x - 2*h) - 4*f(x - h) + 3*f(x)) / (2.0 * h);
    }
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

/// Central-difference second derivative with adaptive step near boundaries.
static double num_der2_safe(const std::function<double(double)> &f,
                            double x, double h = 1e-5,
                            double lo = -1e300, double hi = 1e300)
{
    if (x - h < lo || x + h > hi) {
        // Forward/backward stencil for second derivative
        // f''(x) ≈ (2f(x) - 5f(x+h) + 4f(x+2h) - f(x+3h)) / h^2   (forward)
        // f''(x) ≈ (2f(x) - 5f(x-h) + 4f(x-2h) - f(x-3h)) / h^2   (backward)
        if (x + 3*h > hi) {
            // backward
            return (2*f(x) - 5*f(x-h) + 4*f(x-2*h) - f(x-3*h)) / (h*h);
        } else {
            // forward
            return (2*f(x) - 5*f(x+h) + 4*f(x+2*h) - f(x+3*h)) / (h*h);
        }
    }
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
}

// ── Generic basis validator ─────────────────────────────────────────────

/// Validate calc_val, calc_der, calc_derder for a given AnyBasis at test point r.
/// Checks:
///   1. calc_der.val  == calc_val.val     (cross-method value consistency)
///   2. calc_derder.val == calc_val.val   (cross-method value consistency)
///   3. calc_derder.der == calc_der.der   (cross-method derivative consistency)
///   4. der[i] ≈ num_der(val)            (analytical first derivative)
///   5. derder[i] ≈ num_der2(val)        (analytical second derivative)
///   6. derder[i] ≈ num_der(der)         (second derivative from first derivative)
/// Uses one‑sided stencils when the test point is near domain boundaries.
static void validate_basis(AnyBasis &basis, double r,
                           double lo = -1e300, double hi = 1e300,
                           double tol_der = 1e-6, double tol_derder = 1e-4)
{
    int n = basis.basis_size;
    REQUIRE(n > 0);

    // Capture each method's output independently for cross-method comparison
    std::vector<double> val_ref(n);
    basis.calc_val(r, val_ref);

    std::vector<double> val_der(n), der_der(n);
    basis.calc_der(r, val_der, der_der);

    std::vector<double> val_derder(n), der_derder(n), derder_derder(n);
    basis.calc_derder(r, val_derder, der_derder, derder_derder);

    // Cross-method consistency checks
    for (int i = 0; i < n; ++i) {
        INFO("basis=" << basis.name << " i=" << i << " r=" << r
             << " cross-method val: calc_val=" << val_ref[i]
             << " calc_der=" << val_der[i]);
        REQUIRE(val_der[i] == Approx(val_ref[i]).margin(tol_der));

        INFO("basis=" << basis.name << " i=" << i << " r=" << r
             << " cross-method val: calc_val=" << val_ref[i]
             << " calc_derder=" << val_derder[i]);
        REQUIRE(val_derder[i] == Approx(val_ref[i]).margin(tol_der));

        INFO("basis=" << basis.name << " i=" << i << " r=" << r
             << " cross-method der: calc_der=" << der_der[i]
             << " calc_derder=" << der_derder[i]);
        REQUIRE(der_derder[i] == Approx(der_der[i]).margin(tol_der));
    }

    // Numerical derivative checks (use calc_der's val/der as reference since
    // cross-method check above already verified they're equivalent to calc_val)
    for (int i = 0; i < n; ++i) {
        // Numerical first derivative from calc_val
        auto fval = [&](double x) {
            std::vector<double> v(n);
            basis.calc_val(x, v);
            return v[i];
        };
        double num_d = num_der_safe(fval, r, 1e-6, lo, hi);
        INFO("basis=" << basis.name << " i=" << i << " r=" << r
             << " analytical der=" << der_der[i] << " numerical=" << num_d);
        REQUIRE(der_der[i] == Approx(num_d).margin(tol_der));

        // Numerical second derivative from calc_val
        double num_d2 = num_der2_safe(fval, r, 1e-4, lo, hi);
        INFO("basis=" << basis.name << " i=" << i << " r=" << r
             << " analytical derder=" << derder_derder[i]
             << " numerical(val)=" << num_d2);
        REQUIRE(derder_derder[i] == Approx(num_d2).margin(tol_derder));

        // Numerical second derivative from calc_der
        auto fder = [&](double x) {
            std::vector<double> v(n), d(n);
            basis.calc_der(x, v, d);
            return d[i];
        };
        double num_d2_from_der = num_der_safe(fder, r, 1e-6, lo, hi);
        INFO("basis=" << basis.name << " i=" << i << " r=" << r
             << " analytical derder=" << derder_derder[i]
             << " numerical(der)=" << num_d2_from_der);
        REQUIRE(derder_derder[i] == Approx(num_d2_from_der).margin(tol_derder));
    }
}

/// Wrapper with automatic lo/hi from basis bounds.
static void validate_basis_bounded(AnyBasis &basis, double r,
                                   double tol_der = 1e-6, double tol_derder = 1e-4)
{
    validate_basis(basis, r, basis.min_val, basis.max_val, tol_der, tol_derder);
}

/// Validate at multiple test points
static void validate_basis_multi(AnyBasis &basis,
                                 const std::vector<double> &test_points,
                                 double lo = -1e300, double hi = 1e300,
                                 double tol_der = 1e-6, double tol_derder = 1e-4)
{
    for (double r : test_points) {
        INFO("test_point=" << r);
        validate_basis(basis, r, lo, hi, tol_der, tol_derder);
    }
}

/// Wrapper that auto‑detects domain from basis bounds.
static void validate_basis_multi_bounded(AnyBasis &basis,
                                         const std::vector<double> &test_points,
                                         double tol_der = 1e-6, double tol_derder = 1e-4)
{
    validate_basis_multi(basis, test_points, basis.min_val, basis.max_val,
                         tol_der, tol_derder);
}

static std::unique_ptr<RBasis_ChebyshevConfigurable> make_configurable(
    Cutoff::CutoffType cutoff_type,
    XTransForm::ShiftType shift_type = XTransForm::ShiftType::Linear,
    const Cutoff::Parameters &cutoff_parameters = {},
    const XTransForm::Parameters &shift_parameters = {},
    BasisFunctionType basis_function_type = BasisFunctionType::Chebyshev,
    double min_val = 0.5, double max_val = 5.0, int size = 6)
{
    return std::make_unique<RBasis_ChebyshevConfigurable>(
        min_val, max_val, size,
        Cutoff::make_cutoff(
            cutoff_type, min_val, max_val, cutoff_parameters),
        XTransForm::make_shift(
            shift_type, min_val, max_val, shift_parameters),
        1.0, basis_function_type);
}

TEST_CASE("Retained radial bases have consistent derivatives",
          "[basis][grad]")
{
    RBasis_Chebyshev chebyshev(0.5, 5.0, 8);
    validate_basis_multi(
        chebyshev, {0.8, 1.5, 2.5, 3.5, 4.5});

    RBasis_ChebyCos cosine(0.5, 5.0, 6);
    validate_basis_multi(
        cosine, {0.8, 1.5, 2.5, 3.5, 4.5});

    RBasis_ChebyshevPoly4 poly4(0.5, 5.0, 6, -0.5);
    validate_basis_multi_bounded(
        poly4, {0.5 + 1e-10, 1.5, 2.5, 3.5, 4.5});

    RBasis_ChebyshevPoly5 poly5(0.5, 5.0, 6, -0.3);
    validate_basis_multi_bounded(
        poly5, {0.5 + 1e-10, 1.5, 2.5, 3.5, 4.5});

    RBasis_PolyB polyb(0.5, 5.0, 6);
    validate_basis_multi(
        polyb, {0.8, 1.5, 2.5, 3.5, 4.5},
        -1e300, 1e300, 1e-6, 3e-2);

    RBasis_PolyBCos polyb_cos(0.0, 5.0, 6);
    validate_basis_multi(
        polyb_cos, {0.5, 1.5, 2.5, 3.5, 4.5},
        -1e300, 1e300, 1e-6, 2e-2);
}

TEST_CASE("Configurable cutoff families have consistent derivatives",
          "[basis][grad][configurable][cutoff]")
{
    Cutoff::Parameters polynomial;
    polynomial.b = -0.5;
    Cutoff::Parameters plateau;
    plateau.r1 = 1.5;
    plateau.r2 = 3.5;

    for (Cutoff::CutoffType type : {
             Cutoff::CutoffType::Quadratic,
             Cutoff::CutoffType::NormalizedQuadratic,
             Cutoff::CutoffType::Cosine,
             Cutoff::CutoffType::CosineLeft,
             Cutoff::CutoffType::CosineUnscaled,
             Cutoff::CutoffType::CosineSquared,
             Cutoff::CutoffType::SineSquared,
             Cutoff::CutoffType::Poly3,
             Cutoff::CutoffType::Poly4,
             Cutoff::CutoffType::Poly5,
             Cutoff::CutoffType::Smoothstep5,
             Cutoff::CutoffType::Smoothstep7}) {
        const Cutoff::Parameters &parameters =
            type == Cutoff::CutoffType::Smoothstep7
                ? plateau
                : polynomial;
        auto basis = make_configurable(
            type, XTransForm::ShiftType::Linear, parameters);
        validate_basis_multi_bounded(
            *basis, {0.8, 1.7, 2.5, 3.8, 4.5});
    }
}

TEST_CASE("Configurable transform families have consistent derivatives",
          "[basis][grad][configurable][transform]")
{
    XTransForm::Parameters parameters;
    parameters.lambda = 1.2;
    parameters.scale_lambda = 1.0;

    for (XTransForm::ShiftType type : {
             XTransForm::ShiftType::Linear,
             XTransForm::ShiftType::Exponential,
             XTransForm::ShiftType::ExponentialLeft,
             XTransForm::ShiftType::I2,
             XTransForm::ShiftType::ExponentialTanh}) {
        auto basis = make_configurable(
            Cutoff::CutoffType::SineSquared, type, {}, parameters);
        validate_basis_multi_bounded(
            *basis, {0.8, 1.5, 2.5, 3.5, 4.5});
    }
}

TEST_CASE("Configurable one-minus Chebyshev form has consistent derivatives",
          "[basis][grad][configurable][flip]")
{
    XTransForm::Parameters parameters;
    parameters.lambda = 1.0;
    for (XTransForm::ShiftType type : {
             XTransForm::ShiftType::Linear,
             XTransForm::ShiftType::Exponential}) {
        auto basis = make_configurable(
            Cutoff::CutoffType::Quadratic, type, {}, parameters,
            BasisFunctionType::OneMinusChebyshev);
        validate_basis_multi_bounded(
            *basis, {0.8, 1.5, 2.5, 3.5, 4.5});
    }
}

TEST_CASE("Configurable smoothstep7 includes both plateau boundaries",
          "[basis][grad][configurable][boundary]")
{
    Cutoff::Parameters parameters;
    parameters.r1 = 1.5;
    parameters.r2 = 3.5;
    auto basis = make_configurable(
        Cutoff::CutoffType::Smoothstep7,
        XTransForm::ShiftType::Linear, parameters);

    std::vector<double> val;
    basis->calc_val(parameters.r1, val);
    CHECK(val[0] == Approx(1.0));
    basis->calc_val(parameters.r2, val);
    CHECK(val[0] == Approx(1.0));

    basis->calc_val(0.4, val);
    for (double value : val) CHECK(value == 0.0);
    basis->calc_val(5.1, val);
    for (double value : val) CHECK(value == 0.0);
}

TEST_CASE("Basis size consistency", "[basis][grad][consistency]")
{
    RBasis_Chebyshev size_two(0.5, 5.0, 2);
    RBasis_Chebyshev size_four(0.5, 5.0, 4);
    std::vector<double> values_two(size_two.basis_size);
    std::vector<double> values_four(size_four.basis_size);
    size_two.calc_val(2.5, values_two);
    size_four.calc_val(2.5, values_four);
    CHECK(values_four[0] == Approx(values_two[0]).margin(1e-15));
    CHECK(values_four[1] == Approx(values_two[1]).margin(1e-15));
}
