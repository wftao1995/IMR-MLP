// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_Behler.h"
#include "descriptors/descriptor_BondOrder.h"
#include <cmath>
#include <type_traits>

using Catch::Approx;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Build a simple 2-atom Neighborhood at distance d along x.
static Neighborhood make_dimer_nbh(double d = 2.0, double cutoff = 6.0)
{
    Neighborhood nbh;
    nbh.nbhcount_ = 1;
    nbh.distances_ = {d};
    nbh.pos_ = {{d, 0.0, 0.0}};
    nbh.chemmap_ = {0};
    nbh.centertype = 0;
    return nbh;
}

/// Build a 4-atom square Neighborhood in the xy plane.
static Neighborhood make_square_nbh(double side = 2.0, double cutoff = 6.0)
{
    Neighborhood nbh;
    nbh.nbhcount_ = 4;
    nbh.distances_ = {side, side, side * std::sqrt(2.0), side * std::sqrt(2.0)};
    nbh.pos_ = {
        { side,  0.0,   0.0},
        { 0.0,   side,  0.0},
        { side,  side,  0.0},
        {-side,  side,  0.0}
    };
    nbh.chemmap_ = {0, 0, 0, 0};
    nbh.centertype = 0;
    return nbh;
}

// ═══════════════════════════════════════════════════════════════════════════
// DescriptorBehler  (ACSF)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("DescriptorBehler G1", "[descriptors][behler]")
{
    DescriptorBehler d;
    int sel[] = {1};
    double params[] = {6.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // rcut, eta, rs, kppa, zeta, lambda
    d.set_parameters(1, sel, params);
    d.init();

    auto nbh = make_dimer_nbh(2.0, 6.0);
    Eigen::VectorXd out;
    d.compute(nbh, out);

    REQUIRE(d.feature_size() == 1);
    REQUIRE(out.size() == 1);
    // G1 = sum_j fcut(r_ij). For a single neighbor at r=2, cutoff=6:
    // fcut = 0.5*(cos(pi*2/6) + 1) = 0.5*(cos(pi/3) + 1) = 0.5*(0.5+1) = 0.75
    REQUIRE(out[0] == Approx(0.75).margin(1e-6));
}

TEST_CASE("DescriptorBehler G2", "[descriptors][behler]")
{
    DescriptorBehler d;
    int sel[] = {2};
    double params[] = {6.0, 0.5, 1.0, 0.0, 1.0, 0.0}; // rcut=6, eta=0.5, rs=1.0
    d.set_parameters(1, sel, params);
    d.init();

    auto nbh = make_dimer_nbh(2.0, 6.0);
    Eigen::VectorXd out;
    d.compute(nbh, out);

    REQUIRE(out.size() == 1);
    // G2 = exp(-eta*(r-rs)^2) * fcut(r)
    //    = exp(-0.5*(1)^2) * 0.75 = exp(-0.5) * 0.75 ≈ 0.4549
    double expected = std::exp(-0.5) * 0.75;
    REQUIRE(out[0] == Approx(expected).margin(1e-6));
}

TEST_CASE("DescriptorBehler G4 and G5", "[descriptors][behler]")
{
    DescriptorBehler d;
    int sel[] = {4, 5};
    // rcut, eta, rs, kppa, zeta, lambda
    double params[] = {6.0, 0.1, 0.0, 0.0, 2.0, 1.0};
    d.set_parameters(2, sel, params);
    d.init();

    auto nbh = make_square_nbh(2.0, 6.0);
    Eigen::VectorXd out;
    d.compute(nbh, out);

    REQUIRE(d.feature_size() == 2);
    REQUIRE(out.size() == 2);
    // Both G4 and G5 should be non-negative for lambda=1, cos>= -1
    REQUIRE(out[0] >= 0.0);
    REQUIRE(out[1] >= 0.0);
    // G4 should differ from G5 (G4 includes rij distances in neighbor-pair terms)
    REQUIRE(out[0] != Approx(out[1]).margin(1e-10));
}

TEST_CASE("DescriptorBehler G3 angular", "[descriptors][behler]")
{
    DescriptorBehler d;
    int sel[] = {3};
    // G3: cos angular term — rcut, eta, rs, kppa, zeta, lambda
    double params[] = {6.0, 0.3, 0.0, 0.0, 1.0, 1.0};
    d.set_parameters(1, sel, params);
    d.init();

    auto nbh = make_square_nbh(2.0, 6.0);
    Eigen::VectorXd out;
    d.compute(nbh, out);

    REQUIRE(d.feature_size() == 1);
    REQUIRE(out.size() == 1);
    // G3 = sum_jk fcut(r_ij)*fcut(r_ik)*fcut(r_jk) * (1+lambda*cos(theta))^zeta * exp(-eta*(...))
    // For a square the cos values differ → expect non-zero
    REQUIRE(std::isfinite(out[0]));
    REQUIRE(out[0] >= 0.0);
}

TEST_CASE("DescriptorBehler all G types together", "[descriptors][behler]")
{
    DescriptorBehler d;
    int sel[] = {1, 2, 3, 4, 5};
    double params[] = {6.0, 0.2, 0.0, 0.0, 1.0, 1.0};
    d.set_parameters(5, sel, params);
    d.init();

    REQUIRE(d.feature_size() == 5);

    auto nbh = make_square_nbh(2.0, 6.0);
    Eigen::VectorXd out;
    d.compute(nbh, out);

    REQUIRE(out.size() == 5);
    for (int i = 0; i < 5; ++i) {
        REQUIRE(std::isfinite(out[i]));
        REQUIRE(out[i] >= 0.0);  // all G with lambda=1 are non-negative
    }
}

TEST_CASE("DescriptorBehler virtual interface", "[descriptors][behler]")
{
    DescriptorBehler d;
    int sel[] = {1, 2, 3};
    double params[] = {5.0, 0.3, 0.5, 1.0, 2.0, 0.0};
    d.set_parameters(3, sel, params);
    d.init();

    REQUIRE(d.feature_size() == 3);

    auto nbh = make_dimer_nbh(2.5, 5.0);
    Eigen::VectorXd out;
    d.compute(nbh, out);  // single-nbh override
    REQUIRE(out.size() == 3);
}

TEST_CASE("Behler validates symmetry function selections", "[descriptors][behler]")
{
    GPSettings s;
    s.Modify("cutoff", "5");
    SECTION("missing selection only supports G1 through G5") {
        s.Modify("dispcount", GENERATE("-1", "0", "6"));
        REQUIRE_THROWS_WITH(DescriptorBehler(s), Catch::Matchers::ContainsSubstring("dispcount"));
    }
    SECTION("selection length must match count") {
        s.Modify("dispcount", "2");
        s.Modify("dispselect", GENERATE("1", "1 2 3"));
        REQUIRE_THROWS_WITH(DescriptorBehler(s), Catch::Matchers::ContainsSubstring("dispselect length"));
    }
    SECTION("each selection must be a supported Gi") {
        s.Modify("dispcount", "1");
        s.Modify("dispselect", GENERATE("0", "6", "-1"));
        REQUIRE_THROWS_WITH(DescriptorBehler(s), Catch::Matchers::ContainsSubstring("between 1 and 5"));
    }
    SECTION("repeated valid selections may exceed five features") {
        s.Modify("dispcount", "6");
        s.Modify("dispselect", "1 2 3 4 5 1");
        DescriptorBehler desc(s);
        auto nbh = make_dimer_nbh();
        Eigen::VectorXd out;
        desc.compute(nbh, out);
        REQUIRE(out.size() == 6);
        REQUIRE(out[0] == out[5]);
    }
    SECTION("array setter rejects unsupported Gi") {
        DescriptorBehler desc;
        int selection[] = {6};
        double parameters[] = {5, 0, 0, 0, 1, 1};
        REQUIRE_THROWS_WITH(desc.set_parameters(1, selection, parameters),
                           Catch::Matchers::ContainsSubstring("between 1 and 5"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DescriptorBondOrder
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("DescriptorBondOrder compute", "[descriptors][bondorder]")
{
    DescriptorBondOrder bo(4);  // maxl = 4 → output: Q_2,Q_4,W_2,W_4 = 4
    REQUIRE(bo.feature_size() == 4);

    auto nbh = make_square_nbh(2.0);
    Eigen::VectorXd out;
    bo.compute(nbh, out);

    REQUIRE(out.size() == 4);
    // Ql should be non-negative (sum of squared spherical harmonics)
    int n_q = bo.feature_size() / 2;
    for (int i = 0; i < n_q; ++i)
        REQUIRE(out[i] >= 0.0);
}

TEST_CASE("DescriptorBondOrder dimer analytical", "[descriptors][bondorder]")
{
    DescriptorBondOrder bo(4);  // maxl=4 → Q2,Q4,W2,W4
    auto nbh = make_square_nbh(2.0);

    // Raw compute gives Ql and Wl vectors
    std::vector<double> Ql, Wl;
    bo.compute(nbh, Ql, Wl);

    // maxl=4 → internal Ql includes l=0,2,4 → 3 values
    // feature_size() strips l=0 → 2 Q + 2 W = 4
    REQUIRE(Ql.size() == 3);  // includes Q0
    REQUIRE(Wl.size() == 3);  // includes W0

    // Q0 = 1 (trivial invariant), Q2,Q4 should be valid
    REQUIRE(Ql[0] == Approx(1.0).margin(1e-10));
    for (size_t i = 1; i < Ql.size(); ++i)
        REQUIRE(Ql[i] >= 0.0);
    // At least one non-trivial Ql should be positive
    REQUIRE(Ql[1] + Ql[2] > 0.0);

    // Eigen compute strips Q0,W0: output = [Q2, Q4, W2, W4] = 4
    Eigen::VectorXd out;
    bo.compute(nbh, out);
    REQUIRE(out.size() == 4);

    for (size_t i = 0; i < 2; ++i)
        REQUIRE(out[i] == Approx(Ql[i + 1]).margin(1e-10));
    for (size_t i = 0; i < 2; ++i)
        REQUIRE(out[i + 2] == Approx(Wl[i + 1]).margin(1e-10));
}

TEST_CASE("DescriptorBondOrder virtual interface", "[descriptors][bondorder]")
{
    DescriptorBondOrder bo(6);  // maxl = 6 → output: Q_2,Q_4,Q_6,W_2,W_4,W_6 = 6
    REQUIRE(bo.feature_size() == 6);

    // Test polymorphic call through base class pointer
    Descriptors *base = &bo;
    REQUIRE(base->feature_size() == 6);

    auto nbh = make_dimer_nbh(2.0);
    Eigen::VectorXd out;
    base->compute(nbh, out);
    REQUIRE(out.size() == 6);
}

TEST_CASE("Bond order handles empty and single-neighbor environments consistently", "[descriptors][bondorder]")
{
    const int count = GENERATE(0, 1);
    const bool outer = GENERATE(false, true);
    GPSettings settings;
    settings.Modify("cutoff", "5");
    settings.Modify("maxl", "4");
    settings.Modify("normalize_w_outer", outer ? "true" : "false");
    DescriptorBondOrder desc(settings);
    auto nbh = make_dimer_nbh();
    nbh.nbhcount_ = count;
    Eigen::VectorXd basic, with_options;
    desc.compute(nbh, basic);
    DescriptorGradient gradient;
    gradient.grad.resize(1, 2, 3);
    gradient.grad.setConstant(123);
    ComputeOptions opts;
    opts.compute_derivative = true;
    opts.gradient = &gradient;
    desc.compute(nbh, with_options, opts);
    REQUIRE(basic.allFinite());
    REQUIRE(with_options.isApprox(basic));
    REQUIRE(basic[0] == Approx(count));
    REQUIRE(basic[1] == Approx(count));
    REQUIRE(gradient.grad.dimension(0) == 4);
    REQUIRE(gradient.grad.dimension(1) == count);
    REQUIRE(gradient.grad.dimension(2) == 3);
    for (int i = 0; i < gradient.grad.size(); ++i)
        REQUIRE(std::isfinite(gradient.grad.data()[i]));
    std::vector<double> q, w;
    desc.compute_ref(nbh, q, w);
    REQUIRE(q[1] == Approx(count));
    REQUIRE(q[2] == Approx(count));
    for (double value : w) REQUIRE(std::isfinite(value));
    desc.compute_ref_pairs(nbh, 5, q, w);
    REQUIRE(q[1] == Approx(count));
    REQUIRE(q[2] == Approx(count));
    for (double value : w) REQUIRE(std::isfinite(value));
}

TEST_CASE("Bond order rejects termwise normalization that breaks rotational invariance", "[descriptors][bondorder]")
{
    GPSettings s;
    s.Modify("normalize_w_inner", "true");
    REQUIRE_THROWS_WITH(DescriptorBondOrder(s), Catch::Matchers::ContainsSubstring("not rotationally invariant"));
}

TEST_CASE("Bond order validates angular order before allocating buffers", "[descriptors][bondorder]")
{
    const int maxl = GENERATE(-2, -1, 0, 1, 3, 5);
    const auto message = Catch::Matchers::ContainsSubstring("maxl must be even and at least 2");
    REQUIRE_THROWS_WITH(DescriptorBondOrder(maxl), message);
    GPSettings settings;
    settings.Modify("maxl", std::to_string(maxl));
    REQUIRE_THROWS_WITH(DescriptorBondOrder(settings), message);
    DescriptorBondOrder desc(4);
    REQUIRE_THROWS_WITH(desc.init(maxl), message);
    auto nbh = make_dimer_nbh();
    Eigen::VectorXd out;
    desc.compute(nbh, out);
    REQUIRE(out.size() == desc.feature_size());
    REQUIRE(out.allFinite());
}

TEST_CASE("Bond order clones own their buffers and settings", "[descriptors][bondorder][clone]")
{
    GPSettings settings;
    settings.Modify("maxl", GENERATE("2", "6"));
    settings.Modify("cutoff", "4.25");
    settings.Modify("normalize_w_outer", GENERATE("false", "true"));
    auto source = std::make_unique<DescriptorBondOrder>(settings);
    auto nbh = make_square_nbh();
    Eigen::VectorXd expected, actual;
    source->compute(nbh, expected);
    {
        auto copy = source->clone();
        REQUIRE(copy->feature_size() == source->feature_size());
        REQUIRE(copy->cutoff_ == source->cutoff_);
        copy->compute(nbh, actual);
        REQUIRE(actual.isApprox(expected));

        GPSettings changed;
        changed.Modify("cutoff", "3.5");
        dynamic_cast<DescriptorBondOrder &>(*copy).ApplySettings(changed);
        REQUIRE(copy->cutoff_ == 3.5);
        REQUIRE(source->cutoff_ == 4.25);
        // Reinitializing the copy must not free the source's harmonic buffers.
        dynamic_cast<DescriptorBondOrder &>(*copy).init(4);
    }
    source->compute(nbh, actual);
    REQUIRE(actual.isApprox(expected));

    auto survivor = source->clone();
    source.reset();
    survivor->compute(nbh, actual);
    REQUIRE(actual.isApprox(expected));
}

TEST_CASE("Bond order pair normalization counts only included bonds", "[descriptors][bondorder]")
{
    GPSettings settings;
    settings.Modify("normalize_w_outer", GENERATE("false", "true"));
    DescriptorBondOrder desc(settings);
    Neighborhood nbh;
    nbh.centertype = 0;
    nbh.nbhcount_ = 2;
    nbh.pos_ = {{0, 0, 2}, {0, 0, -2}};
    nbh.distances_ = {2, 2};
    nbh.chemmap_ = {0, 0};
    std::vector<double> q, w, expected_q, expected_w;
    desc.compute_ref(nbh, expected_q, expected_w);
    // The neighbor-neighbor bond has length 4; all bonds are collinear.
    desc.compute_ref_pairs(nbh, GENERATE(3.0, 4.0, 5.0), q, w);
    for (int i = 0; i < q.size(); ++i) {
        REQUIRE(q[i] == Approx(1).margin(1e-12));
        REQUIRE(q[i] == Approx(expected_q[i]).margin(1e-12));
        REQUIRE(w[i] == Approx(expected_w[i]).margin(1e-12));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Descriptors polymorphic dispatch
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Polymorphic compute via Descriptors*", "[descriptors][polymorphic]")
{
    DescriptorBehler behler;
    int sel[] = {1, 2};
    double params[] = {5.0, 0.3, 0.0, 0.0, 1.0, 0.0};
    behler.set_parameters(2, sel, params);
    behler.init();

    DescriptorBondOrder bo(4);

    Descriptors *d1 = &behler;
    Descriptors *d2 = &bo;

    auto nbh = make_dimer_nbh(2.0);

    Eigen::VectorXd out1, out2;
    d1->compute(nbh, out1);
    d2->compute(nbh, out2);

    REQUIRE(out1.size() == 2);
    REQUIRE(out2.size() == 4);
}

TEMPLATE_TEST_CASE("Descriptor clone settings belong to the clone", "[descriptors][clone]",
                   DescriptorBehler)
{
    GPSettings settings;
    settings.Modify("cutoff", "5");
    settings.Modify("rcut", "5");
    settings.Modify("ntypes", "1");
    settings.Modify("dispcount", "1");
    settings.Modify("twojmax", "2");
    auto source = std::make_unique<TestType>(settings);
    auto nbh = make_square_nbh();
    Eigen::VectorXd expected, actual;
    source->compute(nbh, expected);
    auto copy = source->clone();
    auto &typed_copy = dynamic_cast<TestType &>(*copy);
    copy->compute(nbh, actual);
    REQUIRE(actual.isApprox(expected));

    GPSettings changed;
    changed.Modify("cutoff", "3.5");
    changed.Modify("rcut", "3.5");
    typed_copy.ApplySettings(changed);
    REQUIRE(source->cutoff_ == 5);
    REQUIRE(typed_copy.cutoff_ == 3.5);

    source.reset();
    changed.Modify("cutoff", "4.5");
    changed.Modify("rcut", "4.5");
    typed_copy.ApplySettings(changed);
    REQUIRE(typed_copy.cutoff_ == 4.5);
}
