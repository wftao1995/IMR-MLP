// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_fftfilter.h"
#include <limits>
#ifdef USE_OMP
#include <omp.h>
#endif

static std::vector<Atoms> trajectory() {
    std::vector<Atoms> atoms(5);
    for (int i = 0; i < 5; ++i) {
        atoms[i].cell_ = Eigen::Matrix3d::Identity() * 10;
        atoms[i].pos_ = {{1.0 + i * 0.1, 1, 1}};
        atoms[i].chemmap_ = {0};
    }
    return atoms;
}

TEST_CASE("FFT smoothing rejects invalid input before indexing frames", "[fft][errors]") {
    const bool mirrored = GENERATE(false, true);
    ComputeSmoothFFT smoother;
    auto atoms = trajectory();
    double dt = 1;
    SECTION("empty input") { atoms.clear(); }
    SECTION("single frame") { atoms.resize(1); }
    SECTION("empty frames") { atoms.assign(2, Atoms{}); }
    SECTION("unequal atom counts") { atoms.back().pos_.clear(); }
    SECTION("zero timestep") { dt = 0; }
    SECTION("negative timestep") { dt = -1; }
    SECTION("nonfinite timestep") { dt = std::numeric_limits<double>::infinity(); }
    SECTION("negative cutoff") { smoother.params.freq_cutoff = -1; }
    SECTION("nonfinite cutoff") { smoother.params.freq_cutoff = std::numeric_limits<double>::quiet_NaN(); }
    if (mirrored) REQUIRE_THROWS(smoother.compute2(atoms, dt, ComputeSmoothFFT::BLOCK));
    else REQUIRE_THROWS(smoother.compute(atoms, dt, ComputeSmoothFFT::BLOCK));
}

TEST_CASE("FFT cutoff boundaries preserve finite positions and the mean", "[fft]") {
    const bool mirrored = GENERATE(false, true);
    const int window = GENERATE(ComputeSmoothFFT::BLOCK, ComputeSmoothFFT::SUPERGAUSSIAN);
    const double cutoff = GENERATE(0.0, 1e-13, 1e300);
    ComputeSmoothFFT smoother;
    smoother.params.freq_cutoff = cutoff;
    auto atoms = trajectory();
    if (mirrored) smoother.compute2(atoms, 1, window);
    else smoother.compute(atoms, 1, window);
    double mean_x = 0;
    for (const auto &atom : atoms) {
        REQUIRE(atom.pos_[0].allFinite());
        REQUIRE(atom.cell_.allFinite());
        REQUIRE(atom.cell_.determinant() == Catch::Approx(1000));
        mean_x += atom.pos_[0](0);
        if (cutoff < 1) REQUIRE(atom.pos_[0](0) == Catch::Approx(1.2));
    }
    REQUIRE(mean_x / atoms.size() == Catch::Approx(1.2));
    if (cutoff > 1e100) {
        for (int i = 0; i < atoms.size(); ++i)
            REQUIRE(atoms[i].pos_[0](0) == Catch::Approx(1.0 + i * 0.1));
    }
}

TEST_CASE("Mirrored FFT handles two frames with changing cells", "[fft]") {
    ComputeSmoothFFT smoother;
    smoother.even_seq = GENERATE(false, true);
    smoother.params.freq_cutoff = 1e6;
    for (int repeat = 0; repeat < 10; ++repeat) {
        auto atoms = trajectory();
        atoms.resize(2);
        atoms[1].cell_ *= 1.2;
        const auto expected = atoms;
        smoother.compute2(atoms, 1, ComputeSmoothFFT::BLOCK);
        for (int i = 0; i < 2; ++i) {
            REQUIRE(atoms[i].cell_.isApprox(expected[i].cell_));
            REQUIRE(atoms[i].pos_[0].isApprox(expected[i].pos_[0]));
        }
    }
}

TEST_CASE("Rebuilt velocities are Cartesian derivatives in distance per fs", "[fft]") {
    const int representation = GENERATE(0, 1, 2); // Cartesian, fractional, mixed
    ComputeSmoothFFT smoother;
    smoother.even_seq = GENERATE(false, true);
    smoother.rebuild_velo = true;
    smoother.params.freq_cutoff = 1e6;
    const int count = 16;
    const double dt = 2.0;
    const int length = smoother.even_seq ? 2 * count : 2 * count - 1;
    const double omega = 2 * M_PI / (length * dt);
    std::vector<Atoms> atoms(count);
    for (int i = 0; i < count; ++i) {
        double phase = omega * dt * (i + (smoother.even_seq ? 0.5 : 0.0));
        // Fixed fractional position in a moving cell tests the cell contribution.
        atoms[i].cell_ = Eigen::Matrix3d::Identity() * (10 + std::cos(phase));
        atoms[i].pos_ = {{0.2 * atoms[i].cell_(0, 0), 0, 0}};
        atoms[i].chemmap_ = {0};
        if (representation == 1 || (representation == 2 && i % 2))
            atoms[i].to_fractional();
    }
    smoother.compute2(atoms, dt, ComputeSmoothFFT::BLOCK);
    for (int i = 0; i < count; ++i) {
        double phase = omega * dt * (i + (smoother.even_seq ? 0.5 : 0.0));
        const Eigen::Matrix3d cell = Eigen::Matrix3d::Identity() * (10 + std::cos(phase));
        REQUIRE(atoms[i].cartesian);
        REQUIRE(atoms[i].cell_.isApprox(cell, 1e-12));
        REQUIRE(atoms[i].pos_[0].isApprox(Eigen::RowVector3d(.2 * (10 + std::cos(phase)), 0, 0), 1e-12));
        REQUIRE(atoms[i].velocities_.size() == 1);
        REQUIRE(atoms[i].velocities_[0](0) == Catch::Approx(-0.2 * omega * std::sin(phase)).margin(1e-10));
        REQUIRE(atoms[i].velocities_[0].tail<2>().norm() < 1e-10);
    }
}

TEST_CASE("Legacy FFT mean uses Cartesian geometry for fractional and mixed frames", "[fft]") {
    const bool mixed = GENERATE(false, true);
    std::vector<Atoms> atoms(4);
    const Eigen::Matrix3d cell = Eigen::Vector3d(4, 5, 6).asDiagonal();
    for (int i = 0; i < 4; ++i) {
        atoms[i].cell_ = cell;
        atoms[i].pbc_ = {false, false, false};
        atoms[i].pos_ = {{1 + .4 * i, 1, 1}};
        atoms[i].chemmap_ = {0};
        if (!mixed || i % 2) atoms[i].to_fractional();
    }
    ComputeSmoothFFT smoother;
    smoother.params.freq_cutoff = 0;
    smoother.compute(atoms, 2, ComputeSmoothFFT::BLOCK);
    REQUIRE(atoms.size() == 4);
    for (const auto &atom : atoms) {
        REQUIRE(atom.cartesian);
        REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(1.6, 1, 1), 1e-12));
        REQUIRE(atom.cell_.isApprox(cell, 1e-12));
        REQUIRE(atom.velocities_.empty());
    }
}

TEST_CASE("FFT worker errors propagate to the caller", "[fft][errors]") {
#ifdef USE_OMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(2);
#endif
    auto atoms = trajectory();
    for (auto &atom : atoms) atom.forces_ = {{0, 0, 0}};
    atoms.back().forces_[0](0) = std::numeric_limits<double>::quiet_NaN();
    ComputeSmoothFFT smoother;
    REQUIRE_THROWS(smoother.compute2(atoms, 1, ComputeSmoothFFT::BLOCK));
#ifdef USE_OMP
    omp_set_num_threads(previous);
#endif
}

TEST_CASE("Direct FFT rejects unsupported velocity reconstruction before mutation", "[fft][errors]") {
    ComputeSmoothFFT smoother;
    smoother.rebuild_velo = true;
    auto atoms = trajectory();
    const auto original = atoms;
    REQUIRE_THROWS_WITH(smoother.compute(atoms, 1, ComputeSmoothFFT::BLOCK),
                        Catch::Matchers::ContainsSubstring("compute2"));
    for (int i = 0; i < atoms.size(); ++i) {
        REQUIRE(atoms[i].pos_[0].isApprox(original[i].pos_[0]));
        REQUIRE(atoms[i].cell_.isApprox(original[i].cell_));
        REQUIRE_FALSE(atoms[i].has_velo());
    }
}
