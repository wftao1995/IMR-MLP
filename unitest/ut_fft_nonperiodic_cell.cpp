#include <catch2/catch_all.hpp>
#include "analyse/compute_fftfilter.h"
#include <cmath>

TEST_CASE("Both FFT entry points smooth cell-free molecular coordinates directly", "[fft][nonperiodic]")
{
    const bool mirrored = GENERATE(false, true);
    const int cells = GENERATE(0, 1, 2); // all zero, all invertible baseline, mixed
    const bool high_cutoff = GENERATE(false, true);
    const std::vector<Eigen::RowVector3d> signal{{1, 2, -2}, {3, 2, 4}, {-1, 2, 0}, {5, 2, 6}};
    std::vector<Atoms> atoms(signal.size());
    for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
        atoms[i].pbc_ = {false, false, false};
        atoms[i].cell_.setZero();
        if (cells == 1 || (cells == 2 && i > 0))
            atoms[i].cell_ = 10 * Eigen::Matrix3d::Identity();
        atoms[i].pos_ = {signal[i]}; atoms[i].chemmap_ = {0};
    }
    ComputeSmoothFFT smoother;
    smoother.params.freq_cutoff = high_cutoff ? 1e6 : 0;
    if (mirrored) smoother.compute2(atoms, 1, ComputeSmoothFFT::BLOCK);
    else smoother.compute(atoms, 1, ComputeSmoothFFT::BLOCK);
    for (int i = 0; i < static_cast<int>(atoms.size()); ++i) {
        const Eigen::RowVector3d expected = high_cutoff ? signal[i] : Eigen::RowVector3d(2, 2, 2);
        REQUIRE(atoms[i].cartesian);
        REQUIRE(atoms[i].pos_[0].allFinite());
        REQUIRE(atoms[i].pos_[0].isApprox(expected, 1e-12));
        REQUIRE(atoms[i].is_nonperiodic());
        REQUIRE(atoms[i].velocities_.empty());
        REQUIRE(atoms[i].cell_.allFinite());
        if (cells == 0) REQUIRE(atoms[i].cell_.isZero(1e-12));
        if (cells == 1) REQUIRE(atoms[i].cell_.isApprox(10 * Eigen::Matrix3d::Identity(), 1e-12));
    }
}

TEST_CASE("Cell-free molecular FFT reconstructs Cartesian velocity in Angstrom per fs", "[fft][nonperiodic][velocity]")
{
    const bool full_cell = GENERATE(false, true);
    const bool even_sequence = GENERATE(false, true);
    constexpr int count = 16;
    constexpr double dt = 2, amplitude = .2;
    const int length = even_sequence ? 2 * count : 2 * count - 1;
    const double omega = 2 * std::acos(-1.) / (length * dt);
    std::vector<Atoms> atoms(count);
    for (int i = 0; i < count; ++i) {
        const double phase = omega * dt * (i + (even_sequence ? .5 : 0.));
        atoms[i].pbc_ = {false, false, false};
        atoms[i].cell_.setZero();
        if (full_cell) atoms[i].cell_ = 10 * Eigen::Matrix3d::Identity();
        atoms[i].pos_ = {{1 + amplitude * std::cos(phase), 2, 3}};
        atoms[i].chemmap_ = {0};
    }
    ComputeSmoothFFT smoother; smoother.even_seq = even_sequence;
    smoother.rebuild_velo = true; smoother.params.freq_cutoff = 1e6;
    smoother.compute2(atoms, dt, ComputeSmoothFFT::BLOCK);
    for (int i = 0; i < count; ++i) {
        const double phase = omega * dt * (i + (even_sequence ? .5 : 0.));
        REQUIRE(atoms[i].pos_[0](0) == Catch::Approx(1 + amplitude * std::cos(phase)).margin(1e-12));
        REQUIRE(atoms[i].velocities_.size() == 1);
        REQUIRE(atoms[i].velocities_[0](0) == Catch::Approx(-amplitude * omega * std::sin(phase)).margin(1e-11));
        REQUIRE(atoms[i].velocities_[0].tail<2>().norm() < 1e-11);
        REQUIRE(atoms[i].cartesian);
    }
}
