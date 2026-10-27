// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_BondOrder.h"
#include "descriptors/descriptor_MTM.h"
#include "potentials/potential_mtp.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {
Neighborhood environment() {
    Neighborhood n;
    n.centertype = 0;
    n.pos_ = {{0, 0, 2}, {1.7, .2, .3}, {-.4, 1.6, .5}};
    n.nbhcount_ = n.pos_.size();
    n.chemmap_ = {0, 1, 0};
    for (const auto &p : n.pos_) n.distances_.push_back(p.norm());
    return n;
}
std::vector<Eigen::Matrix3d> rotations() {
    std::vector<Eigen::Matrix3d> result = {
        Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix(),
        Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()).toRotationMatrix()
    };
    std::mt19937 gen(20261008);
    std::uniform_real_distribution<double> sample(-1, 1);
    for (int i = 0; i < 6; ++i) {
        Eigen::Vector3d axis(sample(gen), sample(gen), sample(gen));
        result.push_back(Eigen::AngleAxisd(M_PI * sample(gen), axis.normalized()).toRotationMatrix());
    }
    return result;
}
void check_rotation(Descriptors &desc, Neighborhood n, bool gradient) {
    Eigen::VectorXd ref;
    DescriptorGradient dg;
    ComputeOptions opts;
    opts.compute_derivative = gradient;
    opts.gradient = &dg;
    desc.compute(n, ref, opts);
    REQUIRE(ref.allFinite());
    for (const auto &R : rotations()) {
        auto rotated = n;
        for (auto &p : rotated.pos_) p = p * R.transpose();
        Eigen::VectorXd actual;
        DescriptorGradient dr;
        opts.gradient = &dr;
        desc.compute(rotated, actual, opts);
        REQUIRE(actual.allFinite());
        REQUIRE((actual-ref).norm() < 1e-10 * (1 + ref.norm()));
        if (gradient) {
            for (int f = 0; f < ref.size(); ++f)
                for (int j = 0; j < n.nbhcount_; ++j) {
                    Eigen::Vector3d g, gr;
                    for (int a = 0; a < 3; ++a) {g[a] = dg.grad(f,j,a); gr[a] = dr.grad(f,j,a);}
                    CAPTURE(f, j);
                    REQUIRE(gr.allFinite());
                    REQUIRE((gr - R*g).norm() < 1e-9 * (1 + g.norm()));
                }
        }
    }
}
}

TEST_CASE("Bond order values and gradients respect rotations", "[rotation][bondorder]") {
    GPSettings s;
    s.Modify("cutoff", "5");
    s.Modify("maxl", "6");
    s.Modify("normalize_w_outer", GENERATE("false", "true"));
    DescriptorBondOrder desc(s);
    check_rotation(desc, environment(), true);
    auto dimer = environment();
    dimer.nbhcount_ = 1; dimer.pos_.resize(1); dimer.distances_.resize(1); dimer.chemmap_.resize(1);
    check_rotation(desc, dimer, true);
    Eigen::VectorXd values;
    desc.compute(dimer, values);
    // A single bond has Q_l=1 and normalized W_l=(l l l; 0 0 0).
    for (int i = 0; i < 3; ++i) REQUIRE(values[i] == Catch::Approx(1).margin(1e-12));
    const bool raw = s["normalize_w_outer"] == "true";
    REQUIRE(values[3] == Catch::Approx(-std::sqrt(2.0/35) *
            (raw ? std::pow(5.0/(4*M_PI), 1.5) : 1)).margin(1e-12));
    REQUIRE(values[4] == Catch::Approx(std::sqrt(18.0/1001) *
            (raw ? std::pow(9.0/(4*M_PI), 1.5) : 1)).margin(1e-12));
}

TEST_CASE("MTP scalar basis and EFS respect rotations", "[rotation][mtp]") {
    const std::string model = GENERATE("unitest/test-data/pot4b.mtp", "tests/case-mtp-les/les.mtp");
    const bool periodic = GENERATE(false, true);
    const bool parallel = GENERATE(false, true);
    CAPTURE(model, periodic, parallel);
    Potentials::PotMTP pot(model);
    pot.omp_init();
    DescriptorMTM desc(pot);
    Atoms ref;
    ref.cell_ << 5.7, .4, -.2, .3, 6.1, .5, -.2, .6, 5.9;
    ref.pbc_ = {periodic, periodic, periodic};
    ref.pos_ = {{.3,.4,.5}, {2.2,.6,.8}, {.5,2.5,1.1}, {2.7,2.6,3.2}};
    ref.chemmap_ = {0,1,0,1};
    auto evaluate = [&](Atoms &a) {
        if (parallel) pot.getEFSOMP(a);
        else pot.getEFS(a);
    };
    evaluate(ref);
    REQUIRE(std::isfinite(ref.energy_));
    REQUIRE(ref.stress_.allFinite());
    Neighborhoods original_nbhs(ref, pot.cutoff);
    std::vector<Eigen::VectorXd> basis(ref.size());
    for (int i = 0; i < ref.size(); ++i) desc.compute(original_nbhs[i], basis[i]);
    double max_energy = 0, max_force = 0, max_stress = 0, max_basis = 0;
    for (const auto &R : rotations()) {
        auto actual = ref;
        actual.cell_ = actual.cell_ * R.transpose();
        for (auto &p : actual.pos_) p = p * R.transpose();
        evaluate(actual);
        const double de = std::abs(actual.energy_ - ref.energy_);
        const double ds = (actual.stress_ - R*ref.stress_*R.transpose()).norm();
        max_energy = std::max(max_energy, de); max_stress = std::max(max_stress, ds);
        REQUIRE(de < 1e-10 * (1 + std::abs(ref.energy_)));
        REQUIRE(ds < 1e-10 * (1 + ref.stress_.norm()));
        Neighborhoods nbhs(actual, pot.cutoff);
        for (int i = 0; i < actual.size(); ++i) {
            const double df = (actual.forces_[i] - R*ref.forces_[i]).norm();
            max_force = std::max(max_force, df);
            REQUIRE(df < 1e-10 * (1 + ref.forces_[i].norm()));
            REQUIRE(std::abs(actual.site_energies_[i] - ref.site_energies_[i]) <
                    1e-10 * (1 + std::abs(ref.site_energies_[i])));
            Eigen::VectorXd b;
            desc.compute(nbhs[i], b);
            max_basis = std::max(max_basis, (b-basis[i]).norm());
            REQUIRE((b-basis[i]).norm() < 1e-10 * (1 + basis[i].norm()));
        }
    }
    std::cout << "MTP rotation: " << model << " periodic=" << periodic << " omp=" << parallel
              << " max errors E=" << max_energy << " F=" << max_force << " S=" << max_stress
              << " basis=" << max_basis << '\n';
}
