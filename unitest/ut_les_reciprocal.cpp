// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/les.h"
#include <catch2/catch_all.hpp>
using namespace Potentials;
namespace {
Atoms cell()
{
    Atoms a;
    a.cell_ << 10., 0, 0, 3.7, 9., 0, -2.1, 1.3, 11.;
    for (int i = 0; i < 7; ++i) {
        a.pos_.push_back(Eigen::Vector3d(.7 + i, .3 + .5 * i, 2. + .2 * i));
        a.chemmap_.push_back(0);
    }
    return a;
}
void compare(LES& cached, const Atoms& a, const Eigen::VectorXd& q)
{
    LES direct = cached;
    direct.use_recurrence = false;
    direct.use_cache = false;
    const auto expected = direct.evaluate(a, q, true), actual = cached.evaluate(a, q, true);
    REQUIRE(std::abs(actual.energy - expected.energy) < 1e-11);
    REQUIRE((actual.forces - expected.forces).norm() < 1e-10);
    REQUIRE((actual.virial - expected.virial).norm() < 1e-10);
    REQUIRE((actual.dEdq - expected.dEdq).norm() < 1e-10);
    REQUIRE((actual.charge_hessian - expected.charge_hessian).norm() < 1e-10);
    Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(a.size(), -.3, .9);
    REQUIRE((cached.apply_charge_hessian(a, v) - expected.charge_hessian * v).norm() < 1e-10);
}
}
TEST_CASE("Reciprocal caches invalidate for cell, cutoff and in-place kernel updates",
    "[les][sog][cache]")
{
    LES les;
    les.kernel = LES::Kernel::SOG;
    les.sog_weights = { 2.3, -.7 };
    les.sog_log_bandwidths = { -.8, .1 };
    auto a = cell();
    Eigen::VectorXd q = Eigen::VectorXd::LinSpaced(a.size(), -.2, .7);
    compare(les, a, q);
    compare(les, a, q);
    a.cell_(1, 0) += .8;
    compare(les, a, q); // same volume, different reciprocal geometry
    a.pos_[2](1) += .4;
    compare(les, a, q);
    std::swap(a.pos_[0], a.pos_[3]);
    compare(les, a, q);
    q(0) += .3;
    compare(les, a, q);
    les.sog_weights[0] *= .8;
    compare(les, a, q);
    les.sog_log_bandwidths[1] -= .3;
    compare(les, a, q);
    les.dl = 1.7;
    compare(les, a, q);
    les.taper_fraction = .2;
    compare(les, a, q);
    les.cutoff_mode = LES::CutoffMode::ReferenceHard;
    compare(les, a, q);
    les.sog_include_zero = false;
    compare(les, a, q);
    les.remove_self_interaction = false;
    compare(les, a, q);
    les.kernel = LES::Kernel::Coulomb;
    compare(les, a, q);
    les.lr_sigma = .8;
    compare(les, a, q);
    les.remove_self_interaction = true;
    compare(les, a, q);
    les.sog_weights.clear();
    les.sog_log_bandwidths.clear();
    compare(les, a, q);
    les.kvecs.clear();
    les.k_kernel.clear();
    compare(les, a, q);
}
TEST_CASE("High harmonic recurrence and memory fallback agree with direct phases", "[les][phase]")
{
    const std::array<std::array<double, 3>, 3> b
        = { { { .7, .1, -.2 }, { 0., .5, .1 }, { .2, 0., .9 } } };
    const auto position = [](int i, int d) { return .37 * (i + 1) * (d + 1) - .2; };
    ReciprocalPhases direct, recursive;
    direct.prepare(5, { 513, 77, 65 }, b, position, false);
    recursive.prepare(5, { 513, 77, 65 }, b, position, true);
    REQUIRE(recursive.uses_recurrence());
    for (const auto& index :
        { std::array<int, 3> { 513, -77, 65 }, { -32, 33, -64 }, { 0, 0, 0 }, { 1, -1, 1 } }) {
        std::array<double, 3> k { };
        for (int a = 0; a < 3; ++a)
            for (int d = 0; d < 3; ++d)
                k[d] += index[a] * b[a][d];
        double c[5], s[5], cr[5], sr[5];
        direct.evaluate(index, k, c, s);
        recursive.evaluate(index, k, cr, sr);
        for (int i = 0; i < 5; ++i) {
            REQUIRE(std::abs(c[i] - cr[i]) < 2e-12);
            REQUIRE(std::abs(s[i] - sr[i]) < 2e-12);
        }
    }
    recursive.prepare(5, { 1000000, 1000000, 1000000 }, b, position, true);
    REQUIRE_FALSE(recursive.uses_recurrence());
}
TEST_CASE("Recurrence preserves mixed kernel and force VJPs", "[sog][phase][gradient]")
{
    auto a = cell();
    LES fast;
    fast.kernel = LES::Kernel::SOG;
    fast.sog_weights = { 3., -2. };
    fast.sog_log_bandwidths = { -.2, -.2001 };
    Eigen::VectorXd q = Eigen::VectorXd::LinSpaced(a.size(), -.2, .7);
    Eigen::Matrix<double, -1, 3> fw(a.size(), 3);
    for (int i = 0; i < a.size(); ++i)
        fw.row(i) << .1 * i, -.2, .3;
    Eigen::Matrix3d sw;
    sw << .1, .2, -.3, .2, .4, .1, -.3, .1, -.2;
    LES direct = fast;
    direct.use_recurrence = false;
    direct.use_cache = false;
    Eigen::VectorXd f, d;
    fast.compute_les_dLdq_from_forces(a, q, fw, sw, f);
    direct.compute_les_dLdq_from_forces(a, q, fw, sw, d);
    REQUIRE((f - d).norm() < 1e-11);
    double gf[4] = { }, gd[4] = { };
    fast.accumulate_kernel_vjp(a, q, .7, fw, sw, q, gf);
    direct.accumulate_kernel_vjp(a, q, .7, fw, sw, q, gd);
    for (int i = 0; i < 4; ++i)
        REQUIRE(std::abs(gf[i] - gd[i]) < 1e-11);
}
