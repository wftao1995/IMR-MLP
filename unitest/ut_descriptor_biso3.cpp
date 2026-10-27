#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_BispectrumSO3.h"
#include "descriptors/descriptors.h"
#include <cmath>

using Catch::Approx;

// ── Helpers ────────────────────────────────────────────────────────────────

static Neighborhood make_dimer_nbh(double d = 2.0)
{
    Neighborhood nbh;
    nbh.nbhcount_ = 1;
    nbh.distances_ = {d};
    nbh.pos_ = {{d, 0.0, 0.0}};
    nbh.chemmap_ = {0};
    nbh.centertype = 0;
    return nbh;
}

static Neighborhood make_square_nbh(double side = 2.0)
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
// Construction
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Descriptor_BispectrumSO3 construction", "[biso3]")
{
    Descriptor_BispectrumSO3 biso3(2, 0.0, 5.0, 3);

    REQUIRE(biso3.feature_size() > 0);
    REQUIRE(biso3.cutoff_ == Approx(5.0));
}

TEST_CASE("Descriptor_BispectrumSO3 feature_size scales with lmax", "[biso3]")
{
    Descriptor_BispectrumSO3 small(1, 0.0, 5.0, 2);
    Descriptor_BispectrumSO3 large(3, 0.0, 5.0, 2);

    // Larger lmax should give larger feature vector
    REQUIRE(large.feature_size() > small.feature_size());
}

TEST_CASE("Descriptor_BispectrumSO3 feature_size scales with rsize", "[biso3]")
{
    Descriptor_BispectrumSO3 small(2, 0.0, 5.0, 2);
    Descriptor_BispectrumSO3 large(2, 0.0, 5.0, 4);

    // More radial functions should give larger feature vector
    REQUIRE(large.feature_size() > small.feature_size());
}

// ═══════════════════════════════════════════════════════════════════════════
// Compute
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Descriptor_BispectrumSO3 compute on dimer", "[biso3]")
{
    Descriptor_BispectrumSO3 biso3(2, 0.0, 5.0, 3);
    auto nbh = make_dimer_nbh(2.0);

    Eigen::VectorXd out;
    biso3.compute(nbh, out);

    REQUIRE(out.size() == biso3.feature_size());
    for (int i = 0; i < out.size(); ++i)
        REQUIRE(std::isfinite(out[i]));

    // At least some components should be non-zero
    double sum_abs = out.array().abs().sum();
    REQUIRE(sum_abs > 0.0);
}

TEST_CASE("Descriptor_BispectrumSO3 compute on square", "[biso3]")
{
    Descriptor_BispectrumSO3 biso3(2, 0.0, 5.0, 3);
    auto nbh = make_square_nbh(2.0);

    Eigen::VectorXd out;
    biso3.compute(nbh, out);

    REQUIRE(out.size() == biso3.feature_size());
    for (int i = 0; i < out.size(); ++i)
        REQUIRE(std::isfinite(out[i]));
}

TEST_CASE("Descriptor_BispectrumSO3 compute twice gives same result", "[biso3]")
{
    Descriptor_BispectrumSO3 biso3(2, 0.0, 5.0, 3);
    auto nbh = make_dimer_nbh(1.8);

    Eigen::VectorXd out1, out2;
    biso3.compute(nbh, out1);
    biso3.compute(nbh, out2);

    REQUIRE(out1.size() == out2.size());
    for (int i = 0; i < out1.size(); ++i)
        REQUIRE(out1[i] == Approx(out2[i]).margin(1e-12));
}

TEST_CASE("Descriptor_BispectrumSO3 different distances differ", "[biso3]")
{
    Descriptor_BispectrumSO3 biso3(2, 0.0, 6.0, 3);
    auto nbh1 = make_dimer_nbh(1.5);
    auto nbh2 = make_dimer_nbh(3.0);

    Eigen::VectorXd out1, out2;
    biso3.compute(nbh1, out1);
    biso3.compute(nbh2, out2);

    double diff = (out1 - out2).norm();
    REQUIRE(diff > 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════════
// Virtual interface
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Descriptor_BispectrumSO3 through Descriptors base", "[biso3]")
{
    Descriptor_BispectrumSO3 biso3(2, 0.0, 5.0, 3);
    Descriptors *base = &biso3;

    REQUIRE(base->feature_size() == biso3.feature_size());

    auto nbh = make_dimer_nbh(2.0);
    Eigen::VectorXd out;
    base->compute(nbh, out);

    REQUIRE(out.size() == biso3.feature_size());
}

// ═══════════════════════════════════════════════════════════════════════════
// get_bi_ind
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("Descriptor_BispectrumSO3 get_bi_ind returns valid indices", "[biso3]")
{
    Descriptor_BispectrumSO3 biso3(2, 0.0, 5.0, 3);

    for (int i = 0; i < std::min(10, biso3.feature_size()); ++i) {
        auto indices = biso3.get_bi_ind(i);
        REQUIRE(indices.size() == 4);
        for (int idx : indices)
            REQUIRE(idx >= 0);
    }
}
