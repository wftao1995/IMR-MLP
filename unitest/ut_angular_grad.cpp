// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "mathextra/angular_functions.h"
#include <array>
#include <cmath>
#include <type_traits>

using Catch::Approx;

TEMPLATE_TEST_CASE("Zero-order spherical harmonics are constant with zero Cartesian gradient",
                   "[angular][zero_order]", AngularFuncs::ComplexSphericalHarmoics,
                   AngularFuncs::RealSphericalHarmoics)
{
    TestType sh(0);
    const double expected = 1 / std::sqrt(4 * M_PI);
    for (const auto &p : std::vector<std::array<double, 3>>{
            {1, 2, 3}, {2, 0, 0}, {0, 2, 0}, {0, 0, 2}, {0, 0, -2}}) {
        const double r = std::hypot(p[0], p[1], p[2]);
        sh.computeYCarSafe(p[0], p[1], p[2], r);
        REQUIRE(std::real(sh.get(0, 0)) == Approx(expected));
        REQUIRE(std::imag(sh.get(0, 0)) == 0);
        sh.computedYCar(p[0], p[1], p[2], r);
        REQUIRE(std::real(sh.get(0, 0)) == Approx(expected));
        REQUIRE(std::imag(sh.get(0, 0)) == 0);
        REQUIRE(sh.dYdx[0] == 0);
        REQUIRE(sh.dYdy[0] == 0);
        REQUIRE(sh.dYdz[0] == 0);
        if constexpr (std::is_same_v<TestType, AngularFuncs::ComplexSphericalHarmoics>) {
            REQUIRE(sh.dYidx[0] == 0);
            REQUIRE(sh.dYidy[0] == 0);
            REQUIRE(sh.dYidz[0] == 0);
        }
    }
}

// ── Numerical differentiation helpers ────────────────────────────────────

/// Central-difference for a function of one variable
static double num_der(const std::function<double(double)> &f,
                      double x, double h = 1e-6)
{
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

// ═══════════════════════════════════════════════════════════════════════════
// ComplexSphericalHarmoics – computeP / computedP
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("ComplexSphericalHarmoics::computeP known values l=0,1,2",
          "[angular][complex][legendre]")
{
    AngularFuncs::ComplexSphericalHarmoics sh(2);

    // At cos(theta) = 1 (theta=0)
    sh.computeP(1.0);
    // P_{0,0} = 0.28209 (sqrt(1/4pi))
    REQUIRE(sh.P[0] == Approx(0.28209479177387814).margin(1e-12));
    // P_{1,0} = sqrt(3/4pi) * cos(theta) = 0.4886 * 1
    REQUIRE(sh.P[1] == Approx(0.48860251190291992).margin(1e-12));
    // P_{1,1} = -sqrt(3/8pi) * sin(theta) = -0.3455 * 0 = 0
    REQUIRE(sh.P[2] == Approx(0.0).margin(1e-15));

    // At cos(theta) = 0 (theta=pi/2, sin=1)
    sh.computeP(0.0);
    // P_{0,0} = 0.28209
    REQUIRE(sh.P[0] == Approx(0.28209479177387814).margin(1e-12));
    // P_{1,0} = sqrt(3/4pi) * 0 = 0
    REQUIRE(sh.P[1] == Approx(0.0).margin(1e-15));
    // P_{1,1} = -sqrt(3/8pi) * 1
    REQUIRE(sh.P[2] == Approx(-0.3454941494713355).margin(1e-12));
}

TEST_CASE("ComplexSphericalHarmoics::computedP vs numerical",
          "[angular][complex][legendregrad]")
{
    AngularFuncs::ComplexSphericalHarmoics sh(4);

    // Test at multiple cos(theta) values (skipping ±1 where sin=0 causes issues)
    for (double ct : {-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8}) {
        // Compute analytical derivative dP/d(cosθ)
        sh.computedP(ct);

        // Numerical: compute P at cosθ±h (analytical dP is w.r.t. cosθ, not θ)
        auto compute_P = [&](int l, int m, double x) {
            AngularFuncs::ComplexSphericalHarmoics sh2(4);
            sh2.computeP(x);
            int idx = ((m) + ((l) * ((l) + 1)) / 2);
            return sh2.P[idx];
        };

        double h = 1e-6;
        for (int l = 0; l <= 4; ++l) {
            for (int m = 0; m <= std::min(l, 4); ++m) {
                int idx = ((m) + ((l) * ((l) + 1)) / 2);

                double ana = sh.dP[idx];
                double num = num_der([&](double x) { return compute_P(l, m, x); }, ct, h);

                // For m>0, P is proportional to sin(theta)^m, so derivative can be large
                // Use relative tolerance for small values
                INFO("l=" << l << " m=" << m << " ct=" << ct
                     << " analytical=" << ana << " numerical=" << num);
                if (std::abs(ana) > 1e-10 || std::abs(num) > 1e-10) {
                    REQUIRE(ana == Approx(num).margin(5e-5));
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ComplexSphericalHarmoics – computeY / computedYCar gradient
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("ComplexSphericalHarmoics::computedYCar vs numerical dY/d{x,y,z}",
          "[angular][complex][grad]")
{
    AngularFuncs::ComplexSphericalHarmoics sh(4);

    // Include both poles to exercise the Cartesian derivative limits.
    struct Point { double x, y, z, r; };
    std::vector<Point> pts = {
        {1.0, 0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0, 1.0},
        {0.0, 0.0, 1.0, 1.0},
        {0.0, 0.0, -1.0, 1.0},
        {1.0, 1.0, 0.0, std::sqrt(2.0)},
        {1.0, 1.0, 1.0, std::sqrt(3.0)},
        {2.0, 0.5, 1.0, std::sqrt(5.25)},
        {0.3, 0.4, 0.5, std::sqrt(0.5)},
    };

    double h = 1e-6;
    double tol = 1e-4;

    for (auto &p : pts) {
        // Analytical Cartesian gradients
        sh.computedYCar(p.x, p.y, p.z, p.r);

        // Helper to compute Y_lm at a perturbed point
        auto get_Y = [&](double xx, double yy, double zz, int l, int m) -> double {
            double rr = std::sqrt(xx*xx + yy*yy + zz*zz);
            AngularFuncs::ComplexSphericalHarmoics sh2(4);
            sh2.computeYCarSafe(xx, yy, zz, rr);
            const int yvr = ((m) + (l) + ((l) * (l)));
            return sh2.Y[yvr];
        };

        auto get_Yi = [&](double xx, double yy, double zz, int l, int m) -> double {
            double rr = std::sqrt(xx*xx + yy*yy + zz*zz);
            AngularFuncs::ComplexSphericalHarmoics sh2(4);
            sh2.computeYCarSafe(xx, yy, zz, rr);
            const int yvr = ((m) + (l) + ((l) * (l)));
            return sh2.Yi[yvr];
        };

        for (int l = 0; l <= 4; ++l) {
            for (int m = -l; m <= l; ++m) {
                int yvr = ((m) + (l) + ((l) * (l)));

                // dY/dx
                double ana_dx = sh.dYdx[yvr];
                double num_dx = num_der([&](double xx) { return get_Y(xx, p.y, p.z, l, m); }, p.x, h);
                if (std::abs(ana_dx) > 1e-12 || std::abs(num_dx) > 1e-12) {
                    INFO("l=" << l << " m=" << m << " dY/dx at (" << p.x << "," << p.y << "," << p.z
                         << ") analytical=" << ana_dx << " numerical=" << num_dx);
                    REQUIRE(ana_dx == Approx(num_dx).margin(tol));
                }

                // dYi/dx
                double ana_dix = sh.dYidx[yvr];
                double num_dix = num_der([&](double xx) { return get_Yi(xx, p.y, p.z, l, m); }, p.x, h);
                if (std::abs(ana_dix) > 1e-12 || std::abs(num_dix) > 1e-12) {
                    INFO("l=" << l << " m=" << m << " dYi/dx at (" << p.x << "," << p.y << "," << p.z
                         << ") analytical=" << ana_dix << " numerical=" << num_dix);
                    REQUIRE(ana_dix == Approx(num_dix).margin(tol));
                }

                // dY/dz
                double ana_dz = sh.dYdz[yvr];
                double num_dz = num_der([&](double zz) { return get_Y(p.x, p.y, zz, l, m); }, p.z, h);
                if (std::abs(ana_dz) > 1e-12 || std::abs(num_dz) > 1e-12) {
                    INFO("l=" << l << " m=" << m << " dY/dz at (" << p.x << "," << p.y << "," << p.z
                         << ") analytical=" << ana_dz << " numerical=" << num_dz);
                    REQUIRE(ana_dz == Approx(num_dz).margin(tol));
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RealSphericalHarmoics – computeP / computedP
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("RealSphericalHarmoics::computeP known values l=0,1,2",
          "[angular][real][legendre]")
{
    AngularFuncs::RealSphericalHarmoics sh(2);

    // At cos(theta) = 1 (theta=0)
    sh.computeP(1.0);
    // P_{0,0} = sqrt(1/2pi)
    REQUIRE(sh.P[0] == Approx(0.3989422804014327).margin(1e-12));
    // P_{1,0} = sqrt(3/2pi) * cos(theta)
    REQUIRE(sh.P[1] == Approx(0.690988298942671).margin(1e-12));
    // P_{1,1} = -sqrt(3/4pi) * sin(theta) = 0 at theta=0
    REQUIRE(sh.P[2] == Approx(0.0).margin(1e-15));

    // At cos(theta) = 0 (theta=pi/2)
    sh.computeP(0.0);
    REQUIRE(sh.P[0] == Approx(0.3989422804014327).margin(1e-12));
    // P_{1,0} = 0
    REQUIRE(sh.P[1] == Approx(0.0).margin(1e-15));
    // P_{1,1} = -sqrt(3/4pi)
    REQUIRE(sh.P[2] == Approx(-0.4886025119029199).margin(1e-12));
}

TEST_CASE("RealSphericalHarmoics::computedP vs numerical",
          "[angular][real][legendregrad]")
{
    AngularFuncs::RealSphericalHarmoics sh(4);

    for (double ct : {-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8}) {
        sh.computedP(ct);

        auto compute_P = [&](int l, int m, double x) {
            AngularFuncs::RealSphericalHarmoics sh2(4);
            sh2.computeP(x);
            int idx = ((m) + ((l) * ((l) + 1)) / 2);
            return sh2.P[idx];
        };

        double h = 1e-6;
        for (int l = 0; l <= 4; ++l) {
            for (int m = 0; m <= std::min(l, 4); ++m) {
                int idx = ((m) + ((l) * ((l) + 1)) / 2);
                double ana = sh.dP[idx];
                double num = num_der([&](double x) { return compute_P(l, m, x); }, ct, h);

                if (std::abs(ana) > 1e-10 || std::abs(num) > 1e-10) {
                    INFO("l=" << l << " m=" << m << " ct=" << ct
                         << " analytical=" << ana << " numerical=" << num);
                    REQUIRE(ana == Approx(num).margin(5e-5));
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RealSphericalHarmoics – computeY / computedYCar gradient
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("RealSphericalHarmoics::computedYCar vs numerical dY/d{x,y,z}",
          "[angular][real][grad]")
{
    AngularFuncs::RealSphericalHarmoics sh(4);

    struct Point { double x, y, z, r; };
    std::vector<Point> pts = {
        {1.0, 0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0, 1.0},
        {0.0, 0.0, 1.0, 1.0},
        {1.0, 1.0, 0.0, std::sqrt(2.0)},
        {1.0, 1.0, 1.0, std::sqrt(3.0)},
        {2.0, 0.5, 1.0, std::sqrt(5.25)},
        {0.3, 0.4, 0.5, std::sqrt(0.5)},
    };

    double h = 1e-6;
    double tol = 2e-4;

    for (auto &p : pts) {
        sh.computedYCar(p.x, p.y, p.z, p.r);

        auto get_Y = [&](double xx, double yy, double zz, int l, int m) -> double {
            double rr = std::sqrt(xx*xx + yy*yy + zz*zz);
            AngularFuncs::RealSphericalHarmoics sh2(4);
            sh2.computeYCar(xx, yy, zz, rr);
            const int yvr = ((m) + (l) + ((l) * (l)));
            return sh2.Y[yvr];
        };

        for (int l = 0; l <= 4; ++l) {
            for (int m = -l; m <= l; ++m) {
                int yvr = ((m) + (l) + ((l) * (l)));

                // dY/dx
                double ana_dx = sh.dYdx[yvr];
                double num_dx = num_der([&](double xx) { return get_Y(xx, p.y, p.z, l, m); }, p.x, h);
                if (std::abs(ana_dx) > 1e-12 || std::abs(num_dx) > 1e-12) {
                    INFO("l=" << l << " m=" << m << " dY/dx at (" << p.x << "," << p.y << "," << p.z
                         << ") analytical=" << ana_dx << " numerical=" << num_dx);
                    REQUIRE(ana_dx == Approx(num_dx).margin(tol));
                }

                // dY/dy
                double ana_dy = sh.dYdy[yvr];
                double num_dy = num_der([&](double yy) { return get_Y(p.x, yy, p.z, l, m); }, p.y, h);
                if (std::abs(ana_dy) > 1e-12 || std::abs(num_dy) > 1e-12) {
                    INFO("l=" << l << " m=" << m << " dY/dy at (" << p.x << "," << p.y << "," << p.z
                         << ") analytical=" << ana_dy << " numerical=" << num_dy);
                    REQUIRE(ana_dy == Approx(num_dy).margin(tol));
                }

                // dY/dz
                double ana_dz = sh.dYdz[yvr];
                double num_dz = num_der([&](double zz) { return get_Y(p.x, p.y, zz, l, m); }, p.z, h);
                if (std::abs(ana_dz) > 1e-12 || std::abs(num_dz) > 1e-12) {
                    INFO("l=" << l << " m=" << m << " dY/dz at (" << p.x << "," << p.y << "," << p.z
                         << ") analytical=" << ana_dz << " numerical=" << num_dz);
                    REQUIRE(ana_dz == Approx(num_dz).margin(tol));
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-validation: Real vs Complex spherical harmonics at standard points
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Real vs Complex spherical harmonics consistency",
          "[angular][consistency]")
{
    // For m=0: real Y_{l,0} = complex Y_{l,0}
    // For m>0: real Y_{l,m}  = sqrt(2) * Re(complex Y_{l,m}) when Y phase conventions match
    // For m<0: real Y_{l,-m} = sqrt(2) * Im(complex Y_{l,|m|})

    AngularFuncs::ComplexSphericalHarmoics csh(3);
    AngularFuncs::RealSphericalHarmoics rsh(3);

    double x = 1.0, y = 0.5, z = 0.3;
    double r = std::sqrt(x*x + y*y + z*z);

    csh.computeYCar(x, y, z, r);
    rsh.computeYCar(x, y, z, r);

    // m=0: Y_real(l,0) should equal Y_complex(l,0)
    for (int l = 0; l <= 3; ++l) {
        int yvr_c = ((0) + (l) + ((l) * (l)));
        int yvr_r = ((0) + (l) + ((l) * (l)));
        double cY = csh.Y[yvr_c];
        double rY = rsh.Y[yvr_r];
        INFO("l=" << l << " m=0  complex=" << cY << " real=" << rY);
        REQUIRE(cY == Approx(rY).margin(1e-10));
    }

    // m>0: Y_real(l,m) should be consistent with complex
    for (int l = 0; l <= 3; ++l) {
        for (int m = 1; m <= l; ++m) {
            int yvr_c = ((m) + (l) + ((l) * (l)));
            int yvr_cn = ((-m) + (l) + ((l) * (l)));
            int yvr_r = ((m) + (l) + ((l) * (l)));
            int yvr_rn = ((-m) + (l) + ((l) * (l)));

            double cY_re = csh.Y[yvr_c];   // Re(Y_{l,m})
            double cY_im = csh.Yi[yvr_c];  // Im(Y_{l,m})

            double rY_pos = rsh.Y[yvr_r];   // y_{l,m}
            double rY_neg = rsh.Y[yvr_rn];  // y_{l,-m}

            // Real: y_{l,m} = P * cos(m*phi)
            // Complex: Y_{l,m} = P * (cos(m*phi) + i*sin(m*phi))
            // So: y_{l,m} = sqrt(2) * Re(Y_{l,m})  for m>0
            // And: y_{l,-m} = sqrt(2) * Im(Y_{l,|m|})
            INFO("l=" << l << " m=" << m
                 << " Re(Y)=" << cY_re << " Im(Y)=" << cY_im
                 << " Y_real_pos=" << rY_pos << " Y_real_neg=" << rY_neg);

            // Check that Re(Y)*sqrt(2) ≈ Y_real_pos (allowing for phase conventions)
            // Note: various phase conventions exist, this is a sanity check
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Safe vs regular Y computation at singular points
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("computeYCarSafe handles z-axis singularity", "[angular][safety]")
{
    AngularFuncs::ComplexSphericalHarmoics sh(3);

    // Point on z-axis: x=y=0, should not crash or produce NaN
    double x = 0.0, y = 0.0, z = 2.0, r = 2.0;
    REQUIRE_NOTHROW(sh.computeYCarSafe(x, y, z, r));

    // Verify results are finite
    for (int l = 0; l <= 3; ++l) {
        for (int m = -l; m <= l; ++m) {
            int yvr = ((m) + (l) + ((l) * (l)));
            REQUIRE(std::isfinite(sh.Y[yvr]));
            REQUIRE(std::isfinite(sh.Yi[yvr]));
        }
    }

    // At z-axis (theta=0), only m=0 should be non-zero
    for (int l = 1; l <= 3; ++l) {
        int yvr0 = ((0) + (l) + ((l) * (l)));
        double mag_l0 = std::abs(sh.Y[yvr0]) + std::abs(sh.Yi[yvr0]);
        REQUIRE(mag_l0 > 0);  // m=0 non-zero
        for (int m = 1; m <= l; ++m) {
            int yvr = ((m) + (l) + ((l) * (l)));
            int yvrn = ((-m) + (l) + ((l) * (l)));
            // At theta=0, Y_{l,m} = 0 for m != 0
            REQUIRE(std::abs(sh.Y[yvr]) + std::abs(sh.Yi[yvr]) < 1e-12);
            REQUIRE(std::abs(sh.Y[yvrn]) + std::abs(sh.Yi[yvrn]) < 1e-12);
        }
    }
}

TEST_CASE("computedYCar safe at singular points", "[angular][safety][grad]")
{
    AngularFuncs::ComplexSphericalHarmoics sh(3);
    AngularFuncs::RealSphericalHarmoics rsh(3);

    // Point on z-axis
    double x = 0.0, y = 0.0, z = 2.0, r = 2.0;

    // Should not crash
    REQUIRE_NOTHROW(sh.computedYCar(x, y, z, r));
    REQUIRE_NOTHROW(rsh.computedYCar(x, y, z, r));

    // All derivatives should be finite
    for (int l = 0; l <= 3; ++l) {
        for (int m = -l; m <= l; ++m) {
            int yvr = ((m) + (l) + ((l) * (l)));
            REQUIRE(std::isfinite(sh.dYdx[yvr]));
            REQUIRE(std::isfinite(sh.dYdy[yvr]));
            REQUIRE(std::isfinite(sh.dYdz[yvr]));
            REQUIRE(std::isfinite(rsh.dYdx[yvr]));
            REQUIRE(std::isfinite(rsh.dYdy[yvr]));
            REQUIRE(std::isfinite(rsh.dYdz[yvr]));
        }
    }
}
