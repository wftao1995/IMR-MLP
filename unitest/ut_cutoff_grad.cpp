// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "descriptors/CutoffFuncs.h"
#include <cmath>

using Catch::Approx;

// ── Numerical differentiation helpers ────────────────────────────────────

static double num_der(const std::function<double(double)> &f,
                      double x, double h = 1e-6)
{
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

/// Validate fcut and dfcut for a CutoffFunctions instance
static void validate_cutoff(CutoffFunctions &cf, double r,
                            double tol_der = 1e-6)
{
    double fc_ana = cf.fcut(r);
    double dfc_ana = cf.dfcut(r);

    // Numerical first derivative from fcut
    auto f_fc = [&](double x) { return cf.fcut(x); };
    double dfc_num = num_der(f_fc, r);

    INFO("r=" << r << " analytical=" << dfc_ana << " numerical=" << dfc_num);
    REQUIRE(dfc_ana == Approx(dfc_num).margin(tol_der));

    // Check fdfcut returns consistent values
    double fc_fdf, dfc_fdf;
    cf.fdfcut(r, fc_fdf, dfc_fdf);
    REQUIRE(fc_fdf == Approx(fc_ana).margin(1e-15));
    REQUIRE(dfc_fdf == Approx(dfc_ana).margin(tol_der));
}

static void validate_cutoff_multi(CutoffFunctions &cf,
                                  const std::vector<double> &test_points,
                                  double tol_der = 1e-6)
{
    for (double r : test_points) {
        INFO("test_point=" << r);
        validate_cutoff(cf, r, tol_der);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CT_HARD
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("CutoffFunctions CT_HARD val/der", "[cutoff][grad][hard]")
{
    CutoffFunctions cf;
    cf.setCutoffType(CutoffFunctions::CT_HARD);
    cf.setCutoffRadius(5.0);

    // Always 1, derivative always 0
    for (double r : {0.0, 1.0, 2.5, 4.9}) {
        REQUIRE(cf.fcut(r) == Approx(1.0).margin(1e-15));
        REQUIRE(cf.dfcut(r) == Approx(0.0).margin(1e-15));
    }
    // At cutoff: should be 0
    REQUIRE(cf.fcut(5.0) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.fcut(6.0) == Approx(0.0).margin(1e-15));
}

// ═══════════════════════════════════════════════════════════════════════════
// CT_COS_BASIC
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("CutoffFunctions CT_COS_BASIC val/der", "[cutoff][grad][cosbasic]")
{
    CutoffFunctions cf;
    cf.setCutoffType(CutoffFunctions::CT_COS_BASIC);
    cf.setCutoffRadius(5.0);

    validate_cutoff_multi(cf, {0.5, 1.0, 2.0, 3.0, 4.0, 4.9});
    // At exactly r=0, f=1, derivative=0
    REQUIRE(cf.fcut(0.0) == Approx(1.0).margin(1e-15));
    REQUIRE(cf.dfcut(0.0) == Approx(0.0).margin(1e-10));
    // At exactly cutoff: f=0, derivative=0
    REQUIRE(cf.fcut(5.0) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.dfcut(5.0) == Approx(0.0).margin(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════════
// CT_COS (with inner cutoff)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("CutoffFunctions CT_COS val/der", "[cutoff][grad][cos]")
{
    CutoffFunctions cf;
    cf.setCutoffType(CutoffFunctions::CT_COS);
    cf.setCutoffRadius(5.0);
    cf.setCutoffParameter(0.2); // inner cutoff at 20% = 1.0

    // Inside inner cutoff: f=1, df=0
    REQUIRE(cf.fcut(0.0) == Approx(1.0).margin(1e-15));
    REQUIRE(cf.dfcut(0.0) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.fcut(0.5) == Approx(1.0).margin(1e-15));
    REQUIRE(cf.dfcut(0.5) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.fcut(1.0) == Approx(1.0).margin(1e-15));
    REQUIRE(cf.dfcut(1.0) == Approx(0.0).margin(1e-10));

    // Between inner and outer cutoff
    validate_cutoff_multi(cf, {1.5, 2.0, 3.0, 4.0, 4.5});

    // At cutoff: f=0, df=0
    REQUIRE(cf.fcut(5.0) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.dfcut(5.0) == Approx(0.0).margin(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════════
// CT_TANHU
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("CutoffFunctions CT_TANHU val/der", "[cutoff][grad][tanhu]")
{
    CutoffFunctions cf;
    cf.setCutoffType(CutoffFunctions::CT_TANHU);
    cf.setCutoffRadius(5.0);

    validate_cutoff_multi(cf, {0.5, 1.0, 2.0, 3.0, 4.0, 4.9});
    // At cutoff: exactly 0
    REQUIRE(cf.fcut(5.0) == Approx(0.0).margin(1e-15));
}

// ═══════════════════════════════════════════════════════════════════════════
// CT_TANH (normalized version)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("CutoffFunctions CT_TANH val/der", "[cutoff][grad][tanh]")
{
    CutoffFunctions cf;
    cf.setCutoffType(CutoffFunctions::CT_TANH);
    cf.setCutoffRadius(5.0);

    // f(0) should be 1.0 (normalized)
    REQUIRE(cf.fcut(0.0) == Approx(1.0).margin(1e-10));

    validate_cutoff_multi(cf, {0.5, 1.0, 2.0, 3.0, 4.0, 4.9});
    // At cutoff: exactly 0
    REQUIRE(cf.fcut(5.0) == Approx(0.0).margin(1e-15));
}

// ═══════════════════════════════════════════════════════════════════════════
// CT_EXP
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("CutoffFunctions CT_EXP val/der", "[cutoff][grad][exp]")
{
    CutoffFunctions cf;
    cf.setCutoffType(CutoffFunctions::CT_EXP);
    cf.setCutoffRadius(5.0);
    cf.setCutoffParameter(0.2); // inner cutoff at 1.0

    // Inside inner cutoff: f=1, df=0
    REQUIRE(cf.fcut(0.0) == Approx(1.0).margin(1e-15));
    REQUIRE(cf.dfcut(0.0) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.fcut(1.0) == Approx(1.0).margin(1e-10));

    // Between inner and outer
    validate_cutoff_multi(cf, {1.5, 2.0, 3.0, 4.0, 4.5});
    // At cutoff: 0
    REQUIRE(cf.fcut(5.0) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.dfcut(5.0) == Approx(0.0).margin(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════════
// CT_POLY1 through CT_POLY4
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("CutoffFunctions CT_POLY1 val/der", "[cutoff][grad][poly1]")
{
    CutoffFunctions cf;
    cf.setCutoffType(CutoffFunctions::CT_POLY1);
    cf.setCutoffRadius(5.0);
    cf.setCutoffParameter(0.2);

    // Inside inner cutoff
    REQUIRE(cf.fcut(0.5) == Approx(1.0).margin(1e-15));
    REQUIRE(cf.dfcut(0.5) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.fcut(1.0) == Approx(1.0).margin(1e-10));
    REQUIRE(cf.dfcut(1.0) == Approx(0.0).margin(1e-10));

    validate_cutoff_multi(cf, {1.5, 2.0, 3.0, 4.0, 4.5});
    REQUIRE(cf.fcut(5.0) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.dfcut(5.0) == Approx(0.0).margin(1e-10));
}

TEST_CASE("CutoffFunctions CT_POLY2 val/der", "[cutoff][grad][poly2]")
{
    CutoffFunctions cf;
    cf.setCutoffType(CutoffFunctions::CT_POLY2);
    cf.setCutoffRadius(5.0);
    cf.setCutoffParameter(0.2);

    REQUIRE(cf.fcut(0.5) == Approx(1.0).margin(1e-15));
    REQUIRE(cf.dfcut(0.5) == Approx(0.0).margin(1e-15));

    validate_cutoff_multi(cf, {1.5, 2.0, 3.0, 4.0, 4.5});
    REQUIRE(cf.fcut(5.0) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.dfcut(5.0) == Approx(0.0).margin(1e-10));
}

TEST_CASE("CutoffFunctions CT_POLY3 val/der", "[cutoff][grad][poly3]")
{
    CutoffFunctions cf;
    cf.setCutoffType(CutoffFunctions::CT_POLY3);
    cf.setCutoffRadius(5.0);
    cf.setCutoffParameter(0.2);

    REQUIRE(cf.fcut(0.5) == Approx(1.0).margin(1e-15));
    REQUIRE(cf.dfcut(0.5) == Approx(0.0).margin(1e-15));

    validate_cutoff_multi(cf, {1.5, 2.0, 3.0, 4.0, 4.5});
    REQUIRE(cf.fcut(5.0) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.dfcut(5.0) == Approx(0.0).margin(1e-10));
}

TEST_CASE("CutoffFunctions CT_POLY4 val/der", "[cutoff][grad][poly4]")
{
    CutoffFunctions cf;
    cf.setCutoffType(CutoffFunctions::CT_POLY4);
    cf.setCutoffRadius(5.0);
    cf.setCutoffParameter(0.2);

    REQUIRE(cf.fcut(0.5) == Approx(1.0).margin(1e-15));
    REQUIRE(cf.dfcut(0.5) == Approx(0.0).margin(1e-15));

    validate_cutoff_multi(cf, {1.5, 2.0, 3.0, 4.0, 4.5});
    REQUIRE(cf.fcut(5.0) == Approx(0.0).margin(1e-15));
    REQUIRE(cf.dfcut(5.0) == Approx(0.0).margin(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-validation: fdfcut consistency
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("CutoffFunctions fdfcut matches separate fcut+dfcut", "[cutoff][consistency]")
{
    auto check = [](CutoffFunctions &cf, double r) {
        double fc1 = cf.fcut(r);
        double dfc1 = cf.dfcut(r);
        double fc2, dfc2;
        cf.fdfcut(r, fc2, dfc2);
        REQUIRE(fc1 == Approx(fc2).margin(1e-15));
        REQUIRE(dfc1 == Approx(dfc2).margin(1e-15));
    };

    CutoffFunctions cf;

    cf.setCutoffType(CutoffFunctions::CT_COS_BASIC);
    cf.setCutoffRadius(5.0);
    for (double r : {0.5, 1.5, 3.0, 4.5}) check(cf, r);

    cf.setCutoffType(CutoffFunctions::CT_COS);
    cf.setCutoffParameter(0.3);
    for (double r : {0.5, 1.5, 3.0, 4.5}) check(cf, r);

    cf.setCutoffType(CutoffFunctions::CT_TANH);
    for (double r : {0.5, 1.5, 3.0, 4.5}) check(cf, r);

    cf.setCutoffType(CutoffFunctions::CT_EXP);
    cf.setCutoffParameter(0.1);
    for (double r : {0.5, 1.5, 3.0, 4.5}) check(cf, r);
}
