// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_vacf.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>

using Catch::Approx;

static std::vector<Atoms> make_vacf_trajectory(int nframes, int natoms,
                                                const Eigen::RowVector3d &velo)
{
    std::vector<Atoms> traj(nframes);
    for (int f = 0; f < nframes; ++f) {
        traj[f].cell_ = Eigen::Matrix3d::Identity() * 50.0;
        traj[f].chemmap_.assign(natoms, 0);
        traj[f].pos_.resize(natoms);
        traj[f].velocities_.resize(natoms);
        for (int i = 0; i < natoms; ++i) {
            traj[f].pos_[i] = Eigen::RowVector3d(f * velo[0], f * velo[1], f * velo[2]);
            traj[f].velocities_[i] = velo;
        }
    }
    return traj;
}

static ComputeVACF make_vacf(int algo, bool normalize = false, int ndim = 3)
{
    GPSettings s;
    s.Modify("algo", std::to_string(algo));
    s.Modify("timestep", "1.0");
    s.Modify("total", "true");
    s.Modify("types", "false");
    s.Modify("track", "false");
    s.Modify("groups", "false");
    s.Modify("vdos", "false");
    s.Modify("normalize", normalize ? "true" : "false");
    s.Modify("ndim", std::to_string(ndim));
    return ComputeVACF(s);
}

TEST_CASE("VACF: fft and direct algorithms agree for constant velocity", "[vacf]")
{
    auto traj_fft = make_vacf_trajectory(8, 2, {1.0, 2.0, 0.0});
    auto traj_ac = traj_fft;

    auto vacf_fft = make_vacf(1, false, 2);
    vacf_fft.compute(traj_fft);

    auto vacf_ac = make_vacf(2, false, 2);
    vacf_ac.compute(traj_ac);

    REQUIRE(vacf_fft.vacf_isotropic.total.size() == vacf_ac.vacf_isotropic.total.size());
    for (int m = 0; m < (int)vacf_fft.vacf_isotropic.total.size(); ++m) {
        REQUIRE(vacf_fft.vacf_isotropic.total[m][0] == Approx(1.0).margin(1e-12));
        REQUIRE(vacf_fft.vacf_isotropic.total[m][1] == Approx(4.0).margin(1e-12));
        REQUIRE(vacf_fft.vacf_isotropic.total[m][3] == Approx(2.5).margin(1e-12));
        REQUIRE(vacf_ac.vacf_isotropic.total[m][0] == Approx(vacf_fft.vacf_isotropic.total[m][0]).margin(1e-12));
        REQUIRE(vacf_ac.vacf_isotropic.total[m][1] == Approx(vacf_fft.vacf_isotropic.total[m][1]).margin(1e-12));
        REQUIRE(vacf_ac.vacf_isotropic.total[m][3] == Approx(vacf_fft.vacf_isotropic.total[m][3]).margin(1e-12));
    }
}

TEST_CASE("VACF: normalization handles zero components without NaN", "[vacf]")
{
    auto traj = make_vacf_trajectory(6, 1, {1.0, 0.0, 0.0});
    auto vacf = make_vacf(1, true, 3);
    vacf.compute(traj);

    for (auto &row : vacf.vacf_isotropic.total) {
        for (double value : row) {
            REQUIRE(std::isfinite(value));
        }
    }
    REQUIRE(vacf.vacf_isotropic.total[0][0] == Approx(1.0));
    REQUIRE(vacf.vacf_isotropic.total[0][1] == Approx(0.0));
}

TEST_CASE("VACF: projected scalar correlation uses time-major projection buffers", "[vacf]")
{
    std::ofstream ofs("PROJDIR");
    ofs << "1\n1 0 0\n";
    ofs.close();

    GPSettings s;
    s.Modify("algo", "1");
    s.Modify("projection", "true");
    s.Modify("noiso", "true");
    s.Modify("normalize", "false");
    s.Modify("total", "true");
    s.Modify("vdos", "false");
    s.Modify("projdir", "PROJDIR");
    ComputeVACF vacf(s);

    auto traj = make_vacf_trajectory(5, 1, {3.0, 4.0, 0.0});
    vacf.compute(traj);

    REQUIRE(vacf.vacf_projection.size() == 1);
    REQUIRE(vacf.vacf_projection[0].total.size() == 5);
    for (auto &row : vacf.vacf_projection[0].total) {
        REQUIRE(row[0] == Approx(9.0).margin(1e-12));
    }
}

TEST_CASE("VACF: AR algorithm reproduces zero-lag variance", "[vacf]")
{
    auto traj = make_vacf_trajectory(8, 1, {1.0, 0.5, 0.0});
    auto vacf = make_vacf(3, false, 2);
    vacf.compute(traj);

    REQUIRE(vacf.vacf_isotropic.total.size() == 8);
    REQUIRE(vacf.vacf_isotropic.total[0][0] == Approx(1.0).margin(1e-12));
    REQUIRE(vacf.vacf_isotropic.total[0][1] == Approx(0.25).margin(1e-12));
    REQUIRE(vacf.vacf_isotropic.total[0][3] == Approx(0.625).margin(1e-12));
    for (auto &row : vacf.vacf_isotropic.total) {
        REQUIRE(std::isfinite(row[0]));
        REQUIRE(std::isfinite(row[1]));
        REQUIRE(std::isfinite(row[3]));
    }
}

static std::vector<Atoms> make_ar1_trajectory(int nframes, double phi, double sigma)
{
    // single atom, x-velocity follows v[t] = phi * v[t-1] + sigma * eps[t]
    std::vector<Atoms> traj(nframes);
    std::mt19937 rng(42);
    std::normal_distribution<double> norm(0.0, sigma);
    double v = norm(rng) / std::sqrt(1.0 - phi * phi);
    for (int f = 0; f < nframes; ++f) {
        traj[f].cell_ = Eigen::Matrix3d::Identity() * 50.0;
        traj[f].chemmap_.assign(1, 0);
        traj[f].pos_.resize(1);
        traj[f].velocities_.resize(1);
        if (f > 0)
            v = phi * v + norm(rng);
        traj[f].velocities_[0] = Eigen::RowVector3d(v, 0.0, 0.0);
    }
    return traj;
}

TEST_CASE("VACF: AR auto-selection matches fixed order one and exponential decay", "[vacf]")
{
    auto traj_auto = make_ar1_trajectory(20000, 0.9, 1.0);
    auto traj_fixed = traj_auto;

    auto vacf_auto = make_vacf(3, false, 1);
    vacf_auto.compute(traj_auto);

    GPSettings s;
    s.Modify("algo", "3");
    s.Modify("ar_order", "1");
    s.Modify("timestep", "1.0");
    s.Modify("total", "true");
    s.Modify("types", "false");
    s.Modify("track", "false");
    s.Modify("groups", "false");
    s.Modify("vdos", "false");
    s.Modify("normalize", "false");
    s.Modify("ndim", "1");
    ComputeVACF vacf_fixed(s);
    vacf_fixed.compute(traj_fixed);

    const auto &a = vacf_auto.vacf_isotropic.total;
    const auto &b = vacf_fixed.vacf_isotropic.total;
    REQUIRE(a.size() == 20000);
    for (int m = 0; m < 100; ++m)
        REQUIRE(a[m][0] == Approx(b[m][0]).margin(1e-12));

    const double g0 = a[0][0];
    REQUIRE(g0 == Approx(1.0 / (1.0 - 0.9 * 0.9)).epsilon(0.1));
    for (int m = 1; m <= 8; ++m)
        REQUIRE(a[m][0] / g0 == Approx(std::pow(0.9, m)).epsilon(0.1));
}

TEST_CASE("VACF: fixed AR order is not replaced by BIC selection", "[vacf]")
{
    auto traj = make_ar1_trajectory(20000, 0.9, 1.0);
    std::vector<double> v(traj.size());
    for (size_t i = 0; i < traj.size(); ++i)
        v[i] = traj[i].velocities_[0][0];

    double r0 = 0.0, r1 = 0.0, r2 = 0.0;
    for (size_t t = 0; t < v.size(); ++t) r0 += v[t] * v[t];
    for (size_t t = 1; t < v.size(); ++t) r1 += v[t] * v[t - 1];
    for (size_t t = 2; t < v.size(); ++t) r2 += v[t] * v[t - 2];
    r0 /= v.size();
    r1 /= v.size();
    r2 /= v.size();
    const double a1_order1 = r1 / r0;
    const double sigma1 = r0 * (1.0 - a1_order1 * a1_order1);
    const double a2 = (r2 - a1_order1 * r1) / sigma1;
    const double a1 = a1_order1 * (1.0 - a2);
    const double expected_g2 = a1 * r1 + a2 * r0;

    GPSettings s;
    s.Modify("algo", "3");
    s.Modify("ar_order", "2");
    s.Modify("timestep", "1.0");
    s.Modify("total", "true");
    s.Modify("types", "false");
    s.Modify("track", "false");
    s.Modify("groups", "false");
    s.Modify("vdos", "false");
    s.Modify("normalize", "false");
    s.Modify("ndim", "1");
    ComputeVACF vacf(s);
    vacf.compute(traj);

    REQUIRE(vacf.vacf_isotropic.total[2][0] == Approx(expected_g2).margin(1e-12));
}

TEST_CASE("VACF: track-only selection initializes filters", "[vacf]")
{
    std::ofstream ofs("TRACKID");
    ofs << "1\n1\n";
    ofs.close();

    GPSettings s;
    s.Modify("algo", "1");
    s.Modify("total", "false");
    s.Modify("track", "true");
    s.Modify("trackid", "TRACKID");
    s.Modify("normalize", "false");
    ComputeVACF vacf(s);

    auto traj = make_vacf_trajectory(5, 2, {2.0, 0.0, 0.0});
    vacf.compute(traj);

    REQUIRE(vacf.vacf_isotropic.track.size() == 1);
    REQUIRE(vacf.vacf_isotropic.track[0].size() == 5);
    for (auto &row : vacf.vacf_isotropic.track[0])
        REQUIRE(row[0] == Approx(4.0).margin(1e-12));
}

TEST_CASE("VACF: type output supports non-contiguous chemical type ids", "[vacf]")
{
    auto traj = make_vacf_trajectory(4, 1, {1.0, 0.0, 0.0});
    for (auto &frame : traj)
        frame.chemmap_[0] = 5;

    GPSettings s;
    s.Modify("algo", "1");
    s.Modify("total", "false");
    s.Modify("types", "true");
    s.Modify("normalize", "false");
    s.Modify("output_header", "VACF_TYPE_TEST");
    ComputeVACF vacf(s);
    vacf.compute(traj);
    REQUIRE_NOTHROW(vacf.write_results());
}

TEST_CASE("VACF: invalid input is rejected", "[vacf]")
{
    std::vector<Atoms> empty;
    auto vacf = make_vacf(1);
    REQUIRE_THROWS_AS(vacf.compute(empty), std::invalid_argument);

    auto traj = make_vacf_trajectory(3, 1, {1.0, 0.0, 0.0});
    traj[1].chemmap_.push_back(0);
    REQUIRE_THROWS_AS(vacf.compute(traj), std::invalid_argument);

    auto short_traj = make_vacf_trajectory(4, 1, {1.0, 0.0, 0.0});
    GPSettings s;
    s.Modify("algo", "3");
    s.Modify("ar_order", "2");
    s.Modify("vdos", "false");
    ComputeVACF fixed_too_high(s);
    REQUIRE_THROWS_AS(fixed_too_high.compute(short_traj), std::invalid_argument);
}

TEST_CASE("VACF: fft and direct algorithms agree for non-constant velocity", "[vacf]")
{
    auto traj_fft = make_vacf_trajectory(32, 2, {0.0, 0.0, 0.0});
    for (int f = 0; f < (int)traj_fft.size(); ++f) {
        traj_fft[f].velocities_[0] = Eigen::Vector3d(std::sin(0.3 * f), std::cos(0.2 * f), 0.1 * f);
        traj_fft[f].velocities_[1] = Eigen::Vector3d(std::cos(0.17 * f), -0.2 * f, std::sin(0.11 * f));
    }
    auto traj_direct = traj_fft;
    auto fft = make_vacf(1, false, 3);
    auto direct = make_vacf(2, false, 3);
    fft.compute(traj_fft);
    direct.compute(traj_direct);

    for (int m = 0; m < (int)traj_fft.size(); ++m)
        for (int a = 0; a < 4; ++a)
            REQUIRE(fft.vacf_isotropic.total[m][a] ==
                    Approx(direct.vacf_isotropic.total[m][a]).margin(2e-11));
}

TEST_CASE("VACF: normalization is applied after weighted aggregation", "[vacf]")
{
    auto traj = make_vacf_trajectory(2, 2, {0.0, 0.0, 0.0});
    traj[0].velocities_[0] = Eigen::Vector3d(1.0, 0.0, 0.0);
    traj[1].velocities_[0] = Eigen::Vector3d(1.0, 0.0, 0.0);
    traj[0].velocities_[1] = Eigen::Vector3d(10.0, 0.0, 0.0);
    traj[1].velocities_[1] = Eigen::Vector3d(0.0, 0.0, 0.0);

    auto vacf = make_vacf(2, true, 1);
    vacf.compute(traj);
    REQUIRE(vacf.vacf_isotropic.total[0][0] == Approx(1.0));
    REQUIRE(vacf.vacf_isotropic.total[1][0] == Approx(1.0 / 51.0).margin(1e-14));
}

TEST_CASE("VACF: VDOS clamps Nyquist and keeps zero channels finite", "[vacf]")
{
    GPSettings s;
    s.Modify("algo", "1");
    s.Modify("timestep", "1.0");
    s.Modify("total", "true");
    s.Modify("vdos", "true");
    s.Modify("normalize", "true");
    s.Modify("dfreq", "10.0");
    s.Modify("fmax", "50.0");
    ComputeVACF vacf(s);
    auto traj = make_vacf_trajectory(6, 1, {1.0, 0.0, 0.0});
    vacf.compute(traj);

    REQUIRE(vacf.frquency.size() == 6);
    REQUIRE(vacf.frquency.back() == Approx(0.5));
    for (const auto &row : vacf.vacf_isotropic.vdos_total)
        for (double value : row)
            REQUIRE(std::isfinite(value));
    for (const auto &row : vacf.vacf_isotropic.vdos_total) {
        REQUIRE(row[1] == Approx(0.0));
        REQUIRE(row[2] == Approx(0.0));
    }
}

TEST_CASE("VACF: projected VDOS normalizes direction and scalar channels", "[vacf]")
{
    std::ofstream ofs("PROJDIR_VDOS");
    ofs << "1\n2 0 0\n";
    ofs.close();
    GPSettings s;
    s.Modify("algo", "1");
    s.Modify("projection", "true");
    s.Modify("noiso", "true");
    s.Modify("projdir", "PROJDIR_VDOS");
    s.Modify("total", "true");
    s.Modify("vdos", "true");
    s.Modify("normalize", "true");
    s.Modify("timestep", "0.1");
    s.Modify("dfreq", "0.5");
    s.Modify("fmax", "5.0");
    ComputeVACF vacf(s);
    auto traj = make_vacf_trajectory(8, 1, {3.0, 4.0, 0.0});
    vacf.compute(traj);

    REQUIRE(vacf.vacf_projection[0].total[0][0] == Approx(1.0));
    REQUIRE(vacf.vacf_projection[0].total[0][3] == Approx(1.0));
    for (const auto &row : vacf.vacf_projection[0].vdos_total)
        for (double value : row)
            REQUIRE(std::isfinite(value));
}

TEST_CASE("VACF: missing velocities use non-mutating periodic central differences", "[vacf]")
{
    auto traj = make_vacf_trajectory(3, 1, {0.0, 0.0, 0.0});
    for (auto &frame : traj) {
        frame.cell_ = Eigen::Matrix3d::Identity() * 10.0;
        frame.velocities_.clear();
    }
    traj[0].pos_[0] = Eigen::RowVector3d(9.8, 0.0, 0.0);
    traj[1].pos_[0] = Eigen::RowVector3d(0.0, 0.0, 0.0);
    traj[2].pos_[0] = Eigen::RowVector3d(0.2, 0.0, 0.0);

    auto vacf = make_vacf(2, false, 1);
    vacf.compute(traj);
    REQUIRE(vacf.vacf_isotropic.total.size() == 1);
    REQUIRE(vacf.vacf_isotropic.total[0][0] == Approx(0.04).margin(1e-14));
    for (const auto &frame : traj) REQUIRE(frame.velocities_.empty());

    traj[0].velocities_.resize(1, Eigen::Vector3d::Zero());
    REQUIRE_THROWS_AS(vacf.compute(traj), std::invalid_argument);
}

TEST_CASE("VACF: mass weighting and optional center-of-mass drift are explicit", "[vacf]")
{
    auto traj = make_vacf_trajectory(4, 2, {0.0, 0.0, 0.0});
    for (auto &frame : traj) {
        frame.chemmap_ = {0, 1};
        frame.velocities_[0] = Eigen::Vector3d(1.0, 0.0, 0.0);
        frame.velocities_[1] = Eigen::Vector3d(3.0, 0.0, 0.0);
    }
    GPSettings mass_settings;
    mass_settings.Modify("algo", "1");
    mass_settings.Modify("total", "true");
    mass_settings.Modify("normalize", "false");
    mass_settings.Modify("mass_weighted", "true");
    ComputeVACF mass_vacf(mass_settings);
    mass_vacf.atom_masses = {1.0, 3.0};
    mass_vacf.compute(traj);
    REQUIRE(mass_vacf.vacf_isotropic.total[0][0] == Approx(7.0));

    for (auto &frame : traj) {
        frame.chemmap_ = {0, 0};
        frame.velocities_[0] = Eigen::Vector3d(2.0, 0.0, 0.0);
        frame.velocities_[1] = Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    GPSettings drift_settings;
    drift_settings.Modify("algo", "1");
    drift_settings.Modify("total", "true");
    drift_settings.Modify("normalize", "false");
    drift_settings.Modify("remove_drift", "true");
    ComputeVACF drift_vacf(drift_settings);
    drift_vacf.atom_masses = {1.0};
    drift_vacf.compute(traj);
    REQUIRE(drift_vacf.vacf_isotropic.total[0][0] == Approx(1.0));
}

TEST_CASE("VACF: one object can process trajectories repeatedly", "[vacf]")
{
    GPSettings s;
    s.Modify("algo", "1");
    s.Modify("total", "false");
    s.Modify("types", "true");
    s.Modify("normalize", "false");
    ComputeVACF vacf(s);

    auto first = make_vacf_trajectory(4, 1, {1.0, 0.0, 0.0});
    vacf.compute(first);
    REQUIRE(vacf.vacf_isotropic.type.size() == 1);
    auto second = make_vacf_trajectory(7, 1, {2.0, 0.0, 0.0});
    for (auto &frame : second) frame.chemmap_[0] = 5;
    vacf.compute(second);
    REQUIRE(vacf.vacf_isotropic.type.size() == 1);
    REQUIRE(vacf.vacf_isotropic.type[0].size() == 7);
    REQUIRE(vacf.vacf_isotropic.type[0][0][0] == Approx(4.0));
}


TEST_CASE("VACF derived velocities wrap only periodic directions", "[vacf]")
{
    const int algo = GENERATE(1, 2);
    const bool nonperiodic = GENERATE(false, true);
    const int axis = GENERATE(0, 1, 2);
    const int other = (axis + 1) % 3;
    auto traj = make_vacf_trajectory(5, 1, {0, 0, 0});
    for (int f = 0; f < (int)traj.size(); ++f) {
        traj[f].cell_ = Eigen::Matrix3d::Identity() * 10.0;
        traj[f].pbc_ = nonperiodic ? std::array<bool, 3>{false, false, false}
                                  : std::array<bool, 3>{true, true, true};
        traj[f].pbc_[axis] = false;
        traj[f].velocities_.clear();
        traj[f].pos_[0][axis] = 6.0 * f;
        traj[f].pos_[0][other] = nonperiodic ? 9.0 + 2.0 * f
                                           : std::fmod(9.0 + 2.0 * f, 10.0);
    }
    auto vacf = make_vacf(algo, false);
    vacf.compute(traj);
    for (const auto &row : vacf.vacf_isotropic.total) {
        REQUIRE(row[axis] == Approx(36.0));
        REQUIRE(row[other] == Approx(4.0));
    }
}

TEST_CASE("VACF derives Cartesian velocities from fractional and mixed position storage", "[vacf]")
{
    const bool mixed = GENERATE(false, true);
    auto trajectory = make_vacf_trajectory(5, 1, {0, 0, 0});
    for (int f = 0; f < 5; ++f) {
        auto &frame = trajectory[f];
        frame.cell_ = Eigen::Vector3d(4, 5, 6).asDiagonal();
        frame.pbc_ = {false, false, false};
        frame.velocities_.clear();
        frame.pos_[0] = {1 + .4 * f, 1 + .2 * f, 1};
        if (!mixed || f == 2) frame.to_fractional();
    }
    const auto original = trajectory;
    auto vacf = make_vacf(2, false);
    vacf.compute(trajectory);
    REQUIRE(vacf.vacf_isotropic.total.size() == 3);
    // Physical central-difference velocity is (.4,.2,0) Angstrom/ps.
    for (const auto &row : vacf.vacf_isotropic.total) {
        CHECK(row[0] == Approx(.16).margin(1e-12));
        CHECK(row[1] == Approx(.04).margin(1e-12));
        CHECK(row[2] == Approx(0).margin(1e-12));
        CHECK(row[3] == Approx(.20 / 3).margin(1e-12));
    }
    for (int f = 0; f < 5; ++f) {
        CHECK(trajectory[f].pos_[0].isApprox(original[f].pos_[0], 0));
        CHECK(trajectory[f].cell_.isApprox(original[f].cell_, 0));
        CHECK(trajectory[f].pbc_ == original[f].pbc_);
        CHECK(trajectory[f].chemmap_ == original[f].chemmap_);
        CHECK(trajectory[f].cartesian == original[f].cartesian);
        CHECK(trajectory[f].velocities_.empty());
    }
}

TEST_CASE("VACF derived velocities use exact skew periodic images", "[vacf]")
{
    const bool fully_periodic = GENERATE(false, true);
    auto trajectory = make_vacf_trajectory(5, 1, {0, 0, 0});
    for (int f = 0; f < 5; ++f) {
        auto &frame = trajectory[f];
        frame.cell_ << 1, 0, 0, 10, 1, 0, 0, 0, 10;
        frame.pbc_ = {true, fully_periodic, fully_periodic};
        frame.velocities_.clear();
        frame.pos_[0] = {5.39 * f, .49 * f, 0};
    }
    const auto original = trajectory;
    auto vacf = make_vacf(1, false);
    vacf.compute(trajectory);
    REQUIRE(vacf.vacf_isotropic.total.size() == 3);
    // Subtracting 5a gives (.39,.49,0), with squared length .3922.
    // For TTT the next b branch gives (.39,-.51,0), which is longer.
    for (const auto &row : vacf.vacf_isotropic.total) {
        CHECK(row[0] == Approx(.1521).margin(1e-12));
        CHECK(row[1] == Approx(.2401).margin(1e-12));
        CHECK(row[2] == Approx(0).margin(1e-12));
        CHECK(row[3] == Approx(.3922 / 3).margin(1e-12));
    }
    for (int f = 0; f < 5; ++f) {
        CHECK(trajectory[f].pos_[0].isApprox(original[f].pos_[0], 0));
        CHECK(trajectory[f].cell_.isApprox(original[f].cell_, 0));
        CHECK(trajectory[f].pbc_ == original[f].pbc_);
        CHECK(trajectory[f].chemmap_ == original[f].chemmap_);
        CHECK(trajectory[f].cartesian == original[f].cartesian);
        CHECK(trajectory[f].velocities_.empty());
    }
}

TEST_CASE("VACF supplied Cartesian velocities remain independent of fractional positions", "[vacf]")
{
    auto trajectory = make_vacf_trajectory(5, 1, {1, 2, 0});
    for (int f = 0; f < 5; ++f) {
        auto &frame = trajectory[f];
        frame.cell_ << 1, 0, 0, 10, 1, 0, 0, 0, 10;
        frame.pbc_ = {true, false, false};
        frame.pos_[0] = {5.39 * f, .49 * f, 0};
        frame.to_fractional();
    }
    const auto original = trajectory;
    auto vacf = make_vacf(2, false);
    vacf.compute(trajectory);
    REQUIRE(vacf.vacf_isotropic.total.size() == 5);
    for (const auto &row : vacf.vacf_isotropic.total) {
        CHECK(row[0] == Approx(1).margin(1e-12));
        CHECK(row[1] == Approx(4).margin(1e-12));
        CHECK(row[2] == Approx(0).margin(1e-12));
        CHECK(row[3] == Approx(5. / 3).margin(1e-12));
    }
    for (int f = 0; f < 5; ++f) {
        CHECK(trajectory[f].pos_[0].isApprox(original[f].pos_[0], 0));
        CHECK(trajectory[f].cell_.isApprox(original[f].cell_, 0));
        CHECK(trajectory[f].pbc_ == original[f].pbc_);
        CHECK(trajectory[f].chemmap_ == original[f].chemmap_);
        CHECK(trajectory[f].cartesian == original[f].cartesian);
        CHECK(trajectory[f].velocities_[0].isApprox(original[f].velocities_[0], 0));
    }
}

TEST_CASE("VACF checks small total and VDOS output failures on close", "[vacf][output]")
{
    if (!std::filesystem::exists("/dev/full")) SKIP("/dev/full is unavailable");
    const bool fail_vdos = GENERATE(false, true);
    struct Workspace {
        std::filesystem::path path;
        Workspace() {
            char name[] = "/tmp/imrmlp_vacf_output_XXXXXX";
            const char *directory = mkdtemp(name);
            if (!directory) throw std::runtime_error("Cannot create VACF output workspace");
            path = directory;
        }
        ~Workspace() { std::filesystem::remove_all(path); }
    } work;
    GPSettings s;
    s.Modify("algo", "1"); s.Modify("total", "true");
    s.Modify("normalize", "false"); s.Modify("vdos", "true");
    s.Modify("timestep", "0.1"); s.Modify("dfreq", "1"); s.Modify("fmax", "1");
    ComputeVACF vacf(s);
    auto trajectory = make_vacf_trajectory(4, 1, {1, 0, 0});
    vacf.compute(trajectory);
    vacf.f_header = (work.path / "control").string();
    REQUIRE_NOTHROW(vacf.write_results());
    REQUIRE(std::filesystem::file_size(vacf.f_header + "_total.dat") > 0);
    REQUIRE(std::filesystem::file_size(vacf.f_header + "_vdos_total.dat") > 0);

    vacf.f_header = (work.path / "failed").string();
    const std::string failed_path = vacf.f_header + (fail_vdos ? "_vdos_total.dat" : "_total.dat");
    std::filesystem::create_symlink("/dev/full", failed_path);
    REQUIRE_THROWS_WITH(vacf.write_results(), Catch::Matchers::ContainsSubstring(failed_path));
}
