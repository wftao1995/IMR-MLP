// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "descriptors/descriptor_Behler.h"
#include "descriptors/descriptor_BondOrder.h"
#include "descriptors/descriptor_MTM.h"
#include "descriptors/descriptors.h"
#include "potentials/potential_mtp.h"
#include "base/neighborhood.h"
#include "io/gpio.h"
#include <cmath>
#include <Eigen/Core>

using Catch::Approx;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Dimer: 1 neighbor at distance d along x-axis.
static Neighborhood make_dimer_nbh(double d = 2.0)
{
    Neighborhood nbh;
    nbh.nbhcount_  = 1;
    nbh.distances_ = {d};
    nbh.pos_       = {{d, 0.0, 0.0}};
    nbh.chemmap_   = {0};
    nbh.centertype = 0;
    return nbh;
}

/// Small cluster: 3 neighbors forming a triangle + 1 center atom.
static Neighborhood make_cluster_nbh()
{
    Neighborhood nbh;
    nbh.nbhcount_  = 4;
    nbh.distances_ = {2.0, 2.0, 2.0, std::sqrt(8.0)};
    nbh.pos_       = {
        { 2.0,  0.0,  0.0},
        { 0.0,  2.0,  0.0},
        { 0.0,  0.0,  2.0},
        { 2.0,  2.0,  0.0}
    };
    nbh.chemmap_   = {0, 0, 1, 0};
    nbh.centertype = 0;
    return nbh;
}

/// Compare analytical gradient with central-difference numerical gradient.
/// Tests a sampled subset of features to keep runtime reasonable.
static void
validate_gradient(Descriptors &desc, Neighborhood &nbh,
                  double dx = 1e-5, double tol = 5e-5, bool exhaustive = false)
{
    const int nfeat = desc.feature_size();
    const int N     = nbh.nbhcount_;
    REQUIRE(nfeat > 0);
    REQUIRE(N > 0);

    // ── Analytical gradient ────────────────────────────────────────────
    ComputeOptions opts;
    opts.compute_derivative = true;
    DescriptorGradient grad;
    opts.gradient = &grad;
    Eigen::VectorXd out_ref;
    desc.compute(nbh, out_ref, opts);

    REQUIRE(grad.grad.dimension(0) == nfeat);
    REQUIRE(grad.grad.dimension(1) == N);
    REQUIRE(grad.grad.dimension(2) == 3);

    // ── Numerical gradient (central difference) ────────────────────────
    int feat_step = exhaustive ? 1 : std::max(1, nfeat / 5);
    int neig_step = exhaustive ? 1 : std::max(1, N / 2);

    for (int f = 0; f < nfeat; f += feat_step) {
        for (int j = 0; j < N; j += neig_step) {
            for (int a = 0; a < 3; ++a) {
                double analytical = grad.grad(f, j, a);

                // Perturb neighbor position (and distance)
                double orig_pos = nbh.pos_[j][a];
                double orig_dist = nbh.distances_[j];

                nbh.pos_[j][a] = orig_pos + dx;
                nbh.distances_[j] = nbh.pos_[j].norm();
                Eigen::VectorXd fplus(nfeat);
                desc.compute(nbh, fplus);

                nbh.pos_[j][a] = orig_pos - dx;
                nbh.distances_[j] = nbh.pos_[j].norm();
                Eigen::VectorXd fminus(nfeat);
                desc.compute(nbh, fminus);

                nbh.pos_[j][a] = orig_pos;     // restore
                nbh.distances_[j] = orig_dist;

                double numerical = (fplus[f] - fminus[f]) / (2.0 * dx);

                INFO("feature=" << f << " neighbor=" << j
                     << " comp=" << a << " analytical=" << analytical
                     << " numerical=" << numerical);
                REQUIRE(analytical == Approx(numerical).margin(tol));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Behler (ACSF)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Behler gradient vs finite-difference", "[descriptors][grad][behler]")
{
    GPSettings settings;
    settings.Modify("dispcount",  "3");
    settings.Modify("dispselect", "1 2 3");
    settings.Modify("cutoff",     "5.0");
    settings.Modify("eta",        "0.3");
    settings.Modify("rshift",     "0.0");
    settings.Modify("kppa",       "2.0");
    settings.Modify("zeta",       "1.0");
    settings.Modify("lambda",     "1.0");
    DescriptorBehler desc(settings);

    auto nbh = make_dimer_nbh(2.0);
    validate_gradient(desc, nbh, 1e-5, 1e-5);
}

TEST_CASE("Behler angular gradients match values", "[descriptors][grad][behler]")
{
    GPSettings s;
    s.Modify("dispcount", "2");
    s.Modify("dispselect", "4 5");
    s.Modify("cutoff", "5");
    s.Modify("eta", "0.3");
    s.Modify("zeta", "2");
    s.Modify("lambda", "1");
    DescriptorBehler desc(s);
    auto dimer = make_dimer_nbh();
    Eigen::VectorXd out;
    desc.compute(dimer, out);
    REQUIRE(out.isZero()); // Angular terms need two distinct neighbors.
    validate_gradient(desc, dimer);
    auto cluster = make_cluster_nbh();
    validate_gradient(desc, cluster);
}

// ═══════════════════════════════════════════════════════════════════════════
// BondOrder
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("BondOrder gradient vs finite-difference", "[descriptors][grad][bondorder]")
{
    GPSettings settings;
    settings.Modify("cutoff", "5.0");
    settings.Modify("maxl",   "4");
    DescriptorBondOrder desc(settings);

    auto nbh = make_cluster_nbh();
    validate_gradient(desc, nbh, 1e-5, 2e-5);
}

// ═══════════════════════════════════════════════════════════════════════════
// MTM  (Moment Tensor Model)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("MTM gradient vs finite-difference", "[descriptors][grad][mtm]")
{
    // Load potential.mtp from test data directory
    std::string pot_path = "tests/case-descriptors/potential.mtp";
    if (!GPIO::file_exist(pot_path)) {
        WARN("potential.mtp not found at " << pot_path << " — skipping MTM gradient test");
        return;
    }
    Potentials::PotMTP pot(pot_path);
    DescriptorMTM desc(pot);

    auto nbh = make_dimer_nbh(2.5);
    validate_gradient(desc, nbh, 1e-5, 1e-4);
}

TEST_CASE("Bond order Cartesian gradients remain correct on both poles", "[descriptors][grad][bondorder]")
{
    GPSettings s;
    s.Modify("cutoff", "5");
    s.Modify("maxl", "6");
    s.Modify("normalize_w_outer", GENERATE("false", "true"));
    DescriptorBondOrder desc(s);
    auto nbh = make_cluster_nbh();
    nbh.nbhcount_ = 3;
    nbh.pos_ = {{0, 0, GENERATE(-2.0, 2.0)}, {1.7, .2, .3}, {-.4, 1.6, .5}};
    nbh.chemmap_ = {0, 0, 0};
    nbh.distances_.clear();
    for (const auto &p : nbh.pos_) nbh.distances_.push_back(p.norm());
    validate_gradient(desc, nbh, 1e-5, 2e-6, true);
}
