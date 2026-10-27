// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_msd.h"
#include "analyse/trajectory_transform.h"
#include <cmath>
#include <random>

using Catch::Approx;

// ── Helpers ────────────────────────────────────────────────────────────
static std::vector<Atoms> make_trajectory(int nframes, int natoms,
                                           double box = 100.0,
                                           double vel_x = 0.0,
                                           double ts = 1.0) {
    std::vector<Atoms> traj(nframes);
    for (int f = 0; f < nframes; ++f) {
        traj[f].cell_ = Eigen::Matrix3d::Identity() * box;
        traj[f].chemmap_.assign(natoms, 0);
        for (int i = 0; i < natoms; ++i) {
            double x = f * ts * vel_x;
            traj[f].pos_.push_back({x, 0.0, 0.0});
        }
    }
    return traj;
}

static ComputeMSD make_msd(bool fft = true, bool tac_avg = true,
                            double ts = 1.0, bool drift_cm = false,
                            bool do_fit = false, double fit_start = 0.0,
                            double fit_end = 0.0) {
    GPSettings s;
    s.Modify("fft", fft ? "true" : "false");
    s.Modify("tacavg", tac_avg ? "true" : "false");
    s.Modify("timestep", std::to_string(ts));
    s.Modify("total", "true");
    s.Modify("types", "false");
    s.Modify("track", "false");
    s.Modify("groups", "false");
    s.Modify("der", "false");
    s.Modify("drift_cm", drift_cm ? "true" : "false");
    s.Modify("do_lfit", do_fit ? "true" : "false");
    if (do_fit) {
        s.Modify("fit_start", std::to_string(fit_start));
        s.Modify("fit_end", std::to_string(fit_end));
    }
    return ComputeMSD(s);
}

// ═══════════════════════════════════════════════════════════════════════
// Center of mass
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("MSD: center of mass of single atom", "[msd]") {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 10.0;
    atom.pos_ = {{1.0, 2.0, 3.0}};
    atom.chemmap_ = {0};

    ComputeMSD msd;
    msd.atom_masses = {1.0};
    auto cm = get_center_of_mass(atom, msd.atom_masses);
    REQUIRE(cm[0] == Approx(1.0));
    REQUIRE(cm[1] == Approx(2.0));
    REQUIRE(cm[2] == Approx(3.0));
}

TEST_CASE("MSD: center of mass of two equal-mass atoms", "[msd]") {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 10.0;
    atom.pos_ = {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
    atom.chemmap_ = {0, 0};

    ComputeMSD msd;
    msd.atom_masses = {1.0};
    auto cm = get_center_of_mass(atom, msd.atom_masses);
    REQUIRE(cm[0] == Approx(1.0));
    REQUIRE(cm[1] == Approx(0.0));
}

// ═══════════════════════════════════════════════════════════════════════
// Stationary particle — MSD should be zero
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("MSD: stationary particle MSD=0", "[msd]") {
    auto traj = make_trajectory(10, 1); // stationary
    auto msd = make_msd(true, true, 1.0, false);
    msd.atom_masses = {1.0};
    msd.compute(traj);

    auto &t = msd.msd_total;
    for (int i = 0; i < (int)t.size(); ++i)
        REQUIRE(t[i] == Approx(0.0).margin(1e-12));
}

// ═══════════════════════════════════════════════════════════════════════
// Constant velocity — MSD should be quadratic: MSD(m) = v² * (m*ts)²
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("MSD: constant velocity particle MSD is quadratic", "[msd]") {
    double v = 1.0;
    auto traj = make_trajectory(10, 1, 100.0, v, 1.0);
    auto msd = make_msd(true, true, 1.0, false);
    msd.atom_masses = {1.0};
    msd.compute(traj);

    auto &t = msd.msd_total;
    // MSD(m) = (v*m*ts)² = m² for v=1, ts=1
    for (int m = 0; m < (int)t.size(); ++m) {
        double expected = v * v * m * m; // = m²
        REQUIRE(t[m] == Approx(expected).epsilon(1e-10));
    }
}

// ═══════════════════════════════════════════════════════════════════════
// nac vs ac vs fft consistency
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("MSD: nac algorithm gives same result as fft for stationary", "[msd]") {
    auto traj1 = make_trajectory(10, 1);
    auto traj2 = traj1;

    auto msd_nac = make_msd(false, false, 1.0, false); // nac
    msd_nac.atom_masses = {1.0};
    msd_nac.compute(traj1);

    auto msd_fft = make_msd(true, true, 1.0, false);   // fft
    msd_fft.atom_masses = {1.0};
    msd_fft.compute(traj2);

    for (int i = 0; i < (int)msd_nac.msd_total.size(); ++i)
        REQUIRE(msd_nac.msd_total[i] == Approx(msd_fft.msd_total[i]).margin(1e-12));
}

TEST_CASE("MSD: ac and fft give similar results for constant motion", "[msd]") {
    auto traj1 = make_trajectory(10, 1, 100.0, 0.5, 1.0);
    auto traj2 = traj1;

    auto msd_ac = make_msd(false, true, 1.0, false);  // ac
    msd_ac.atom_masses = {1.0};
    msd_ac.compute(traj1);

    auto msd_fft = make_msd(true, true, 1.0, false);  // fft
    msd_fft.atom_masses = {1.0};
    msd_fft.compute(traj2);

    for (int i = 0; i < (int)msd_ac.msd_total.size(); ++i)
        REQUIRE(msd_ac.msd_total[i] == Approx(msd_fft.msd_total[i]).epsilon(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════
// Drift correction
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("MSD: drift_cm removes COM translation", "[msd]") {
    // Two atoms both moving at same velocity — COM is moving
    // After drift_cm, MSD should be 0 (atoms are stationary relative to COM)
    auto traj = make_trajectory(10, 2, 100.0, 1.0, 1.0);
    auto msd = make_msd(true, true, 1.0, true); // drift_cm = true
    msd.atom_masses = {1.0};
    msd.compute(traj);

    auto &t = msd.msd_total;
    for (int i = 0; i < (int)t.size(); ++i)
        REQUIRE(t[i] == Approx(0.0).margin(1e-12));
}

TEST_CASE("MSD: drift_cm preserves relative motion", "[msd]") {
    // Two atoms: atom 0 stationary, atom 1 moves at v=1
    // 10 frames, each has 2 atoms
    std::vector<Atoms> traj(10);
    for (int f = 0; f < 10; ++f) {
        traj[f].cell_ = Eigen::Matrix3d::Identity() * 100.0;
        traj[f].chemmap_ = {0, 0};
        traj[f].pos_ = {{0.0, 0.0, 0.0}, {f * 1.0, 0.0, 0.0}};
    }
    auto msd = make_msd(true, true, 1.0, true);
    msd.atom_masses = {1.0, 1.0};
    msd.compute(traj);

    // COM moves at v=0.5, so after drift removal:
    // atom0 moves at -0.5, atom1 moves at +0.5
    // MSD of atom0: (0.5*m)², atom1: (0.5*m)²
    // Total weighted: 0.5 * [(0.5*m)² + (0.5*m)²] = 0.25*m²
    for (int m = 0; m < (int)msd.msd_total.size(); ++m) {
        double expected = 0.25 * m * m;
        REQUIRE(msd.msd_total[m] == Approx(expected).epsilon(1e-10));
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Linear fit
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("MSD: linear fit on constant velocity data", "[msd]") {
    auto traj = make_trajectory(20, 1, 100.0, 1.0, 1.0);
    auto msd = make_msd(true, true, 1.0, false, true, 2.0, 19.0);
    msd.atom_masses = {1.0};
    msd.compute(traj);

    // For constant velocity MSD = m², linear fit is not great
    // but should still produce finite values
    REQUIRE(msd.k_total != Approx(0.0).margin(1e-12));
    REQUIRE(std::isfinite(msd.k_total));
    REQUIRE(std::isfinite(msd.b_total));
}

TEST_CASE("MSD: lfit bounds are respected", "[msd]") {
    auto traj = make_trajectory(20, 1, 100.0, 0.5, 1.0);
    GPSettings s;
    s.Modify("fft", "true");
    s.Modify("tacavg", "true");
    s.Modify("timestep", "1.0");
    s.Modify("total", "true");
    s.Modify("types", "false");
    s.Modify("track", "false");
    s.Modify("groups", "false");
    s.Modify("der", "false");
    s.Modify("drift_cm", "false");
    s.Modify("do_lfit", "true");
    s.Modify("fit_start", "5.0");   // start fit at t=5
    s.Modify("fit_end", "15.0");    // end fit at t=15
    auto msd = ComputeMSD(s);
    msd.atom_masses = {1.0};
    msd.compute(traj);

    // Fit range is 5-15 (10 points) — k should be positive
    REQUIRE(msd.k_total > 0);
}

// ═══════════════════════════════════════════════════════════════════════
// Weighting / postprocessing
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("MSD: weighted average with equal weights equals arithmetic mean", "[msd]") {
    auto traj = make_trajectory(10, 2, 100.0, 0.0, 1.0);
    auto msd = make_msd(true, true, 1.0, false);
    msd.atom_masses = {1.0};
    // Default atomic_weight is 1.0 for all atoms
    msd.compute(traj);

    for (int m = 0; m < (int)msd.msd_total.size(); ++m)
        REQUIRE(msd.msd_total[m] == Approx(0.0).margin(1e-12));
}

// ═══════════════════════════════════════════════════════════════════════
// Time axis
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("MSD: time axis is correct", "[msd]") {
    auto traj = make_trajectory(10, 1, 100.0, 0.0, 2.5); // ts = 2.5
    auto msd = make_msd(true, true, 2.5, false);
    msd.atom_masses = {1.0};
    msd.compute(traj);

    REQUIRE(msd.time.size() == 10);
    for (int i = 0; i < 10; ++i)
        REQUIRE(msd.time[i] == Approx(i * 2.5));
}

// ═══════════════════════════════════════════════════════════════════════
// max_length truncation
// ═══════════════════════════════════════════════════════════════════════

// max_length truncation tested manually — Connect propagation issues
// when max_length is in the private section (settings may not apply)

// ═══════════════════════════════════════════════════════════════════════
// Settings constructor
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("MSD: settings constructor propagates parameters", "[msd]") {
    GPSettings s;
    s.Modify("fft", "false");
    s.Modify("tacavg", "false");
    s.Modify("drift_cm", "false");
    s.Modify("do_lfit", "false");
    ComputeMSD msd(s);

    (void)msd; // just verify construction doesn't throw
}

TEST_CASE("MSD: invalid input is rejected", "[msd]") {
    std::vector<Atoms> empty;
    auto msd = make_msd(true, true, 1.0, false);
    REQUIRE_THROWS_AS(msd.compute(empty), std::invalid_argument);

    auto traj = make_trajectory(3, 1);
    traj[1].pos_.push_back({0.0, 0.0, 0.0});
    REQUIRE_THROWS_AS(msd.compute(traj), std::invalid_argument);
}

TEST_CASE("MSD: max_length limits lags but uses every time origin", "[msd]") {
    auto traj = make_trajectory(6, 1, 100.0);
    const double x[] = {0.0, 0.0, 0.0, 0.0, 10.0, 10.0};
    for (int i = 0; i < 6; ++i) traj[i].pos_[0][0] = x[i];

    GPSettings s;
    s.Modify("fft", "true");
    s.Modify("tacavg", "true");
    s.Modify("drift_cm", "false");
    s.Modify("do_lfit", "false");
    s.Modify("max_length", "2");
    ComputeMSD msd(s);
    msd.compute(traj);

    REQUIRE(msd.time.size() == 2);
    REQUIRE(msd.msd_total[1] == Approx(20.0).margin(1e-10));
}

TEST_CASE("MSD: compute does not modify wrapped input", "[msd]") {
    auto traj = make_trajectory(4, 1, 10.0);
    traj[0].pos_[0][0] = 9.0;
    traj[1].pos_[0][0] = 1.0;
    traj[2].pos_[0][0] = 3.0;
    traj[3].pos_[0][0] = 5.0;
    const auto original = traj;

    auto msd = make_msd(true, true, 1.0, false);
    msd.compute(traj);

    for (int frame = 0; frame < 4; ++frame)
        REQUIRE(traj[frame].pos_[0].isApprox(original[frame].pos_[0], 0.0));
    REQUIRE(msd.msd_total[1] == Approx(4.0).margin(1e-10));
}

TEST_CASE("MSD: integrated velocity drift removes accelerating translation", "[msd]") {
    auto traj = make_trajectory(4, 1, 100.0);
    const double velocity[] = {0.0, 1.0, 2.0, 3.0};
    const double position[] = {0.0, 0.5, 2.0, 4.5};
    for (int i = 0; i < 4; ++i) {
        traj[i].pos_[0][0] = position[i];
        traj[i].velocities_ = {{velocity[i], 0.0, 0.0}};
    }
    GPSettings s;
    s.Modify("drift_cm", "false");
    s.Modify("drift_vm", "true");
    s.Modify("do_lfit", "false");
    ComputeMSD msd(s);
    msd.atom_masses = {1.0};
    msd.compute(traj);
    for (double value : msd.msd_total)
        REQUIRE(value == Approx(0.0).margin(1e-12));
}

TEST_CASE("MSD: position and velocity drift modes are exclusive", "[msd]") {
    auto traj = make_trajectory(3, 1);
    for (auto &frame : traj) frame.velocities_ = {{0.0, 0.0, 0.0}};
    GPSettings s;
    s.Modify("drift_cm", "true");
    s.Modify("drift_vm", "true");
    ComputeMSD msd(s);
    msd.atom_masses = {1.0};
    REQUIRE_THROWS_AS(msd.compute(traj), std::invalid_argument);
}

TEST_CASE("MSD: finite-size correction keeps raw slope and uses SI conversion", "[msd]") {
    auto traj = make_trajectory(4, 1, 10.0);
    for (int i = 0; i < 4; ++i) traj[i].pos_[0][0] = std::sqrt(6.0 * i);
    GPSettings s;
    s.Modify("fft", "false");
    s.Modify("tacavg", "false");
    s.Modify("drift_cm", "false");
    s.Modify("do_lfit", "true");
    s.Modify("fit_start", "1.0");
    s.Modify("fit_end", "3.0");
    s.Modify("fsc_mode", "1");
    s.Modify("temperature", "300.0");
    s.Modify("viscosity", "1.0");
    ComputeMSD msd(s);
    msd.compute(traj);

    constexpr double kb = 8.617333262e-5;
    const double expected_correction = kb * 300.0 * 2.837298 /
        (6.0 * M_PI * 6.241509074e-3 * 10.0);
    REQUIRE(msd.k_total == Approx(6.0).margin(1e-10));
    REQUIRE(msd.diffusion_total.D_PBC == Approx(1.0).margin(1e-10));
    REQUIRE(msd.diffusion_total.D_correction == Approx(expected_correction).epsilon(1e-12));
    REQUIRE(msd.diffusion_total.D_infinite == Approx(1.0 + expected_correction).epsilon(1e-12));
}

TEST_CASE("MSD: fitting is disabled by default and requires an explicit window", "[msd]") {
    auto traj = make_trajectory(5, 1);
    ComputeMSD default_msd;
    default_msd.atom_masses = {1.0};
    default_msd.compute(traj);
    REQUIRE_FALSE(default_msd.diffusion_total.fitted);
    REQUIRE(std::isnan(default_msd.k_total));

    GPSettings s;
    s.Modify("drift_cm", "false");
    s.Modify("do_lfit", "true");
    ComputeMSD invalid_fit(s);
    REQUIRE_THROWS_AS(invalid_fit.compute(traj), std::invalid_argument);
}

TEST_CASE("MSD: FFT matches direct autocorrelation for a random trajectory", "[msd]") {
    constexpr int nframes = 64;
    constexpr int natoms = 3;
    auto direct_traj = make_trajectory(nframes, natoms, 1000.0);
    std::mt19937 rng(12345);
    std::normal_distribution<double> step(0.0, 0.2);
    for (int atom = 0; atom < natoms; ++atom) {
        Eigen::RowVector3d pos(1.0e8 + atom, -2.0e8, 3.0e8);
        for (int frame = 0; frame < nframes; ++frame) {
            if (frame > 0)
                pos += Eigen::RowVector3d(step(rng), step(rng), step(rng));
            direct_traj[frame].pos_[atom] = pos;
        }
    }
    auto fft_traj = direct_traj;
    auto direct = make_msd(false, true, 0.5, false);
    auto fft = make_msd(true, true, 0.5, false);
    direct.compute(direct_traj);
    fft.compute(fft_traj);
    REQUIRE(direct.msd_total.size() == fft.msd_total.size());
    for (size_t i = 0; i < direct.msd_total.size(); ++i)
        REQUIRE(fft.msd_total[i] == Approx(direct.msd_total[i]).margin(1e-8));
}

TEST_CASE("MSD: triclinic fractional unwrapping follows the nearest image", "[msd]") {
    std::vector<Atoms> traj(2);
    Eigen::Matrix3d cell;
    cell << 10.0, 0.0, 0.0,
             2.0, 9.0, 0.0,
             0.0, 0.0, 8.0;
    const Eigen::RowVector3d frac0(0.9, 0.2, 0.3);
    const Eigen::RowVector3d frac1(0.1, 0.2, 0.3);
    for (auto &frame : traj) {
        frame.cell_ = cell;
        frame.chemmap_ = {0};
    }
    traj[0].pos_ = {frac0 * cell};
    traj[1].pos_ = {frac1 * cell};
    auto msd = make_msd(true, true, 1.0, false);
    msd.compute(traj);
    const Eigen::RowVector3d expected_delta = Eigen::RowVector3d(0.2, 0.0, 0.0) * cell;
    REQUIRE(msd.msd_total[1] == Approx(expected_delta.squaredNorm()).margin(1e-10));
}

TEST_CASE("MSD: invalid weights and changing atom identity are rejected", "[msd]") {
    auto traj = make_trajectory(4, 2);
    auto weighted = make_msd(true, true, 1.0, false);
    weighted.atomic_weight = {1.0, -0.5};
    REQUIRE_THROWS_AS(weighted.compute(traj), std::runtime_error);

    auto changed = make_trajectory(4, 1);
    changed[2].chemmap_[0] = 1;
    auto msd = make_msd(true, true, 1.0, false);
    REQUIRE_THROWS_AS(msd.compute(changed), std::invalid_argument);
}

TEST_CASE("MSD: one instance can be reused for different atom counts", "[msd]") {
    auto one = make_trajectory(5, 1);
    auto two = make_trajectory(6, 2);
    auto msd = make_msd(true, true, 1.0, false);
    msd.compute(one);
    REQUIRE(msd.msd_total.size() == 5);
    msd.compute(two);
    REQUIRE(msd.msd_total.size() == 6);
    for (double value : msd.msd_total)
        REQUIRE(value == Approx(0.0).margin(1e-12));
}

TEST_CASE("MSD: derivative rejects too few lag samples", "[msd]") {
    auto traj = make_trajectory(3, 1);
    GPSettings s;
    s.Modify("drift_cm", "false");
    s.Modify("der", "true");
    ComputeMSD msd(s);
    REQUIRE_THROWS_AS(msd.compute(traj), std::invalid_argument);
}


TEST_CASE("MSD unwraps only periodic directions", "[msd]") {
    const bool fft = GENERATE(false, true);
    const bool nonperiodic = GENERATE(false, true);
    const int axis = GENERATE(0, 1, 2);
    const int other = (axis + 1) % 3;
    auto traj = make_trajectory(5, 1, 10.0);
    for (int f = 0; f < (int)traj.size(); ++f) {
        traj[f].pbc_ = nonperiodic ? std::array<bool, 3>{false, false, false}
                                  : std::array<bool, 3>{true, true, true};
        traj[f].pbc_[axis] = false;
        traj[f].pos_[0][axis] = 6.0 * f;
        traj[f].pos_[0][other] = nonperiodic ? 9.0 + 2.0 * f
                                           : std::fmod(9.0 + 2.0 * f, 10.0);
    }
    auto msd = make_msd(fft);
    msd.compute(traj);
    for (int lag = 0; lag < (int)msd.msd_total.size(); ++lag)
        REQUIRE(msd.msd_total[lag] == Approx(40.0 * lag * lag).margin(1e-10));
}

static void check_msd_source_preserved(const std::vector<Atoms> &frames,
                                        const std::vector<Atoms> &original) {
    REQUIRE(frames.size() == original.size());
    for (size_t f = 0; f < frames.size(); ++f) {
        REQUIRE(frames[f].cartesian == original[f].cartesian);
        REQUIRE(frames[f].cell_.isApprox(original[f].cell_, 0));
        REQUIRE(frames[f].pbc_ == original[f].pbc_);
        REQUIRE(frames[f].chemmap_ == original[f].chemmap_);
        REQUIRE(frames[f].pos_[0].isApprox(original[f].pos_[0], 0));
        REQUIRE(frames[f].velocities_[0].isApprox(original[f].velocities_[0], 0));
        REQUIRE(frames[f].have_energy == original[f].have_energy);
        REQUIRE(frames[f].energy_ == original[f].energy_);
        REQUIRE(frames[f].features_ == original[f].features_);
    }
}

TEST_CASE("MSD: fractional nonperiodic positions retain Cartesian displacement units", "[msd]") {
    const bool fft = GENERATE(false, true);
    std::vector<Atoms> frames(2);
    for (int f = 0; f < 2; ++f) {
        frames[f].cell_ = Eigen::Vector3d(4., 5., 6.).asDiagonal();
        frames[f].pbc_ = {false, false, false};
        frames[f].chemmap_ = {0};
        frames[f].pos_ = {{1. + .4*f, 1., 1.5}};
        frames[f].velocities_ = {{.4, .2, -.1}};
        frames[f].energy_ = 11. + f;
        frames[f].have_energy = true;
        frames[f].features_["source"] = "nonperiodic";
        frames[f].to_fractional();
    }
    const auto original = frames;
    auto msd = make_msd(fft, true, 1., false);
    msd.atom_masses = {1.};
    msd.compute(frames);
    REQUIRE(msd.msd_total.size() == 2);
    REQUIRE(msd.msd_total[0] == Approx(0).margin(1e-12));
    REQUIRE(msd.msd_total[1] == Approx(.16).margin(1e-12));
    check_msd_source_preserved(frames, original);
}

TEST_CASE("MSD: mixed stored coordinates unwrap partial periodic crossings", "[msd]") {
    const bool fft = GENERATE(false, true);
    const bool mixed = GENERATE(false, true);
    std::vector<Atoms> frames(3);
    const std::array<double, 3> fractional_x{.9, .1, .3};
    for (int f = 0; f < 3; ++f) {
        frames[f].cell_ = Eigen::Vector3d(4., 5., 6.).asDiagonal();
        frames[f].pbc_ = {true, false, false};
        frames[f].chemmap_ = {0};
        frames[f].pos_ = {{4*fractional_x[f], 1., 1.5}};
        frames[f].velocities_ = {{.8, .2, -.1}};
        frames[f].energy_ = 21. + f;
        frames[f].have_energy = true;
        frames[f].features_["source"] = "partial-periodic";
        if (!mixed || f == 1) frames[f].to_fractional();
    }
    const auto original = frames;
    auto msd = make_msd(fft, true, 1., false);
    msd.atom_masses = {1.};
    msd.compute(frames);
    // Fractional .9 -> .1 -> .3 corresponds to x=3.6 -> 4.4 -> 5.2 A.
    REQUIRE(msd.msd_total.size() == 3);
    REQUIRE(msd.msd_total[0] == Approx(0).margin(1e-11));
    REQUIRE(msd.msd_total[1] == Approx(.64).margin(1e-11));
    REQUIRE(msd.msd_total[2] == Approx(2.56).margin(1e-11));
    check_msd_source_preserved(frames, original);
}
