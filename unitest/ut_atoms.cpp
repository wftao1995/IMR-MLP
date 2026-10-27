// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "base/structures.h"
#include <cmath>

using Catch::Approx;

// ── Test helpers ───────────────────────────────────────────────────────
static Atoms make_dimer(double a = 12.0, double d = 2.0) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * a;
    atom.pos_ = {{0, 0, 0}, {d, 0, 0}};
    atom.chemmap_ = {0, 0};
    return atom;
}

static Atoms make_sc_cell(double a = 5.0, int n = 2) {
    // n×n×n simple cubic supercell
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * a * n;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                atom.pos_.push_back({i * a, j * a, k * a});
    atom.chemmap_.assign(atom.pos_.size(), 0);
    return atom;
}

// ═══════════════════════════════════════════════════════════════════════
// Construction / Basic
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms default construction", "[atoms]") {
    Atoms atom;
    REQUIRE(atom.size() == 0);
    REQUIRE(atom.id == 0);
    REQUIRE(atom.cartesian == true);
    REQUIRE(atom.have_energy == false);
    REQUIRE(atom.have_stress == false);
    REQUIRE(atom.have_fenergy == false);
    REQUIRE(atom.cell_.isApprox(Eigen::Matrix3d::Zero()));
}

TEST_CASE("Atoms basic dimer construction", "[atoms]") {
    auto atom = make_dimer(10.0, 2.5);
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.chemmap_.size() == 2);
    REQUIRE(atom.chemmap_[0] == 0);
    REQUIRE(atom.chemmap_[1] == 0);
}

TEST_CASE("Atoms clear empties all data", "[atoms]") {
    auto atom = make_dimer();
    atom.clear();
    REQUIRE(atom.size() == 0);
    REQUIRE(atom.chemmap_.empty());
    REQUIRE(atom.forces_.empty());
}

// ═══════════════════════════════════════════════════════════════════════
// Cell operations
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms get_volume", "[atoms]") {
    auto atom = make_dimer(5.0);
    // cell = 5*I, volume = 5^3 = 125
    REQUIRE(atom.get_volume() == Approx(125.0));
}

TEST_CASE("Atoms get_volume oblique cell", "[atoms]") {
    Atoms atom;
    atom.cell_ << 3, 0, 0,
                  1, 4, 0,
                  1, 1, 5;
    // det = 3*4*5 = 60
    REQUIRE(atom.get_volume() == Approx(60.0));
}

TEST_CASE("Atoms get_metric", "[atoms]") {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 3.0;
    Eigen::Matrix3d G = atom.get_metric();
    REQUIRE(G.isApprox(Eigen::Matrix3d::Identity() * 9.0));
}

TEST_CASE("Atoms get_rec / get_recip", "[atoms]") {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 2.0;
    Eigen::Matrix3d rec = atom.get_rec();
    Eigen::Matrix3d recip = atom.get_recip();
    // rec = (2I)^{-T} = 0.5 * I
    REQUIRE(rec.isApprox(Eigen::Matrix3d::Identity() * 0.5));
    // recip = 2*pi*rec
    REQUIRE(recip.isApprox(Eigen::Matrix3d::Identity() * M_PI));
}

TEST_CASE("Atoms is_right_hand positive volume", "[atoms]") {
    auto atom = make_dimer(5.0);
    REQUIRE(atom.is_right_hand());
}

TEST_CASE("Atoms is_right_hand negative volume", "[atoms]") {
    auto atom = make_dimer(5.0);
    atom.cell_ = -atom.cell_;
    REQUIRE(!atom.is_right_hand());
}

TEST_CASE("Atoms get_handedness positive", "[atoms]") {
    auto atom = make_dimer(5.0);
    REQUIRE(atom.get_handedness() == 1);
}

TEST_CASE("Atoms get_handedness negative", "[atoms]") {
    auto atom = make_dimer(5.0);
    atom.cell_ = -atom.cell_;
    REQUIRE(atom.get_handedness() == -1);
}

TEST_CASE("Atoms isorth", "[atoms]") {
    auto atom = make_dimer(5.0);
    REQUIRE(atom.isorth());
    atom.cell_(0, 1) = 1.0;
    REQUIRE(!atom.isorth());
}

TEST_CASE("Atoms to_righthand", "[atoms]") {
    auto atom = make_dimer(5.0);
    atom.cell_ = -atom.cell_;
    REQUIRE(atom.get_handedness() == -1);
    atom.to_righthand();
    REQUIRE(atom.get_handedness() == 1);
}

TEST_CASE("Atoms set_cell 6 params (radians)", "[atoms]") {
    Atoms atom;
    double a90 = M_PI / 2.0;
    atom.set_cell(3, 4, 5, a90, a90, a90);
    REQUIRE(atom.cell_(0, 0) == Approx(3.0));
    REQUIRE(atom.cell_(1, 1) == Approx(4.0));
    REQUIRE(atom.cell_(2, 2) == Approx(5.0));
    REQUIRE(atom.cell_(0, 1) == Approx(0.0));
    REQUIRE(atom.cell_(0, 2) == Approx(0.0));
    REQUIRE(atom.cell_(1, 2) == Approx(0.0));
}

TEST_CASE("Atoms set_cell matrix", "[atoms]") {
    Atoms atom;
    Eigen::Matrix3d c = Eigen::Matrix3d::Identity() * 6.0;
    atom.set_cell(c);
    REQUIRE(atom.cell_.isApprox(c));
}

TEST_CASE("Atoms standard_form lower", "[atoms]") {
    Atoms atom;
    atom.cell_ << 6, 0, 0,
                  0, 6, 0,
                  0, 0, 6;
    double vol0 = atom.get_volume();
    auto Q = atom.standard_form(false); // returns orthogonal factor Q
    // cell_ is now the lower-triangular L (QR decomposition)
    REQUIRE(atom.cell_(0, 1) == Approx(0.0));
    REQUIRE(atom.cell_(0, 2) == Approx(0.0));
    REQUIRE(atom.cell_(1, 2) == Approx(0.0));
    REQUIRE(atom.cell_.determinant() > 0);
    // Volume conserved in L: |det(L)| = |det(original)|
    REQUIRE(atom.get_volume() == Approx(vol0).epsilon(1e-10));
    // Q is orthogonal: det(Q) = ±1
    REQUIRE(std::abs(Q.determinant()) == Approx(1.0));
}

TEST_CASE("Atoms standard_form upper", "[atoms]") {
    Atoms atom;
    atom.cell_ << 5, 0, 0,
                  0, 5, 0,
                  0, 0, 5;
    auto Q = atom.standard_form(true);
    REQUIRE(atom.cell_(1, 0) == Approx(0.0));
    REQUIRE(atom.cell_(2, 0) == Approx(0.0));
    REQUIRE(atom.cell_(2, 1) == Approx(0.0));
    REQUIRE(atom.cell_.determinant() > 0);
    REQUIRE(std::abs(Q.determinant()) == Approx(1.0));
}

// ═══════════════════════════════════════════════════════════════════════
// Coordinate transformations
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms dir2car / car2dir roundtrip", "[atoms]") {
    auto atom = make_dimer(5.0, 2.0);
    Eigen::RowVector3d dir_pos = {0.3, 0.4, 0.5};
    auto car_pos = atom.dir2car(dir_pos);
    auto dir_back = atom.car2dir(car_pos);
    REQUIRE(dir_back.isApprox(dir_pos, 1e-10));
}

TEST_CASE("Atoms to_fractional and to_cartesian", "[atoms]") {
    Atoms atom;
    atom.cell_ << 5, 0, 0,
                  0, 5, 0,
                  0, 0, 5;
    atom.pos_ = {{1.25, 2.5, 3.75}}; // Cartesian: 0.25, 0.5, 0.75 in fractional
    atom.chemmap_ = {0};

    atom.to_fractional();
    REQUIRE(atom.pos_[0][0] == Approx(0.25));
    REQUIRE(atom.pos_[0][1] == Approx(0.5));
    REQUIRE(atom.pos_[0][2] == Approx(0.75));

    atom.to_cartesian();
    REQUIRE(atom.pos_[0][0] == Approx(1.25));
    REQUIRE(atom.pos_[0][1] == Approx(2.5));
    REQUIRE(atom.pos_[0][2] == Approx(3.75));
}

TEST_CASE("Atoms get_dir_pos", "[atoms]") {
    Atoms atom;
    atom.cell_ << 4, 0, 0,
                  0, 4, 0,
                  0, 0, 4;
    atom.pos_ = {{2, 2, 2}, {1, 1, 1}};
    atom.chemmap_ = {0, 0};

    std::vector<Eigen::RowVector3d> dirpos;
    atom.get_dir_pos(dirpos);
    REQUIRE(dirpos.size() == 2);
    REQUIRE(dirpos[0][0] == Approx(0.5));
    REQUIRE(dirpos[1][0] == Approx(0.25));
}

TEST_CASE("Atoms set_dir_pos", "[atoms]") {
    Atoms atom;
    atom.cell_ << 3, 0, 0,
                  0, 3, 0,
                  0, 0, 3;
    atom.pos_ = {{1, 1, 1}}; // dummy
    atom.chemmap_ = {0};

    std::vector<Eigen::RowVector3d> dirpos = {{0.5, 1.0 / 3.0, 2.0 / 3.0}};
    atom.set_dir_pos(dirpos);
    REQUIRE(atom.pos_[0][0] == Approx(1.5));
    REQUIRE(atom.pos_[0][1] == Approx(1.0));
    REQUIRE(atom.pos_[0][2] == Approx(2.0));
}

// ═══════════════════════════════════════════════════════════════════════
// Distance / position
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms get_distance", "[atoms]") {
    auto atom = make_dimer(10.0, 3.0);
    REQUIRE(atom.get_distance(0, 1) == Approx(3.0));
}

TEST_CASE("Atoms get_rab vector", "[atoms]") {
    auto atom = make_dimer(10.0, 4.0);
    auto rab = atom.get_rab(0, 1);
    REQUIRE(rab[0] == Approx(4.0));
    REQUIRE(rab[1] == Approx(0.0));
    REQUIRE(rab[2] == Approx(0.0));
}

// Independent finite integer enumeration for the small, bounded fixtures below.
static Eigen::RowVector3d reference_pair_image(const Atoms &atom,
                                              const Eigen::RowVector3d &delta) {
    Eigen::RowVector3d best = delta;
    const int nx = atom.pbc_[0] ? 8 : 0;
    const int ny = atom.pbc_[1] ? 8 : 0;
    const int nz = atom.pbc_[2] ? 8 : 0;
    for (int x = -nx; x <= nx; ++x)
        for (int y = -ny; y <= ny; ++y)
            for (int z = -nz; z <= nz; ++z) {
                const Eigen::RowVector3d candidate = delta - x * atom.cell_.row(0)
                    - y * atom.cell_.row(1) - z * atom.cell_.row(2);
                if (candidate.squaredNorm() < best.squaredNorm()) best = candidate;
            }
    return best;
}

TEST_CASE("Native pair displacement needs no cell for open boundaries", "[atoms][mic]") {
    Atoms atom;
    atom.pbc_ = {false, false, false};
    atom.pos_ = {{1, 2, 3}, {8, -6, 5}};
    atom.chemmap_ = {0, 0};
    REQUIRE(atom.cell_.isZero());
    const Eigen::RowVector3d expected(7, -8, 2);
    REQUIRE(atom.get_rab(0, 1).isApprox(expected, 1e-12));
    REQUIRE(atom.get_rab(1, 0).isApprox(-expected, 1e-12));
    REQUIRE(atom.get_distance(0, 1) == Approx(std::sqrt(117.)));
    REQUIRE(atom.get_rab(0, 0).isZero());
    REQUIRE(atom.get_distance(1, 1) == 0);
}

TEST_CASE("Native pair images respect periodic sublattices and skew geometry", "[atoms][mic]") {
    const int rank = GENERATE(1, 2, 3);
    CAPTURE(rank);
    Atoms atom;
    atom.cell_.setZero();
    Eigen::RowVector3d delta, expected;
    if (rank == 1) {
        atom.pbc_ = {false, true, false};
        atom.cell_.row(1) << 10, 0, 0;
        delta << 29, 2, 0;
        expected << -1, 2, 0;
    } else if (rank == 2) {
        atom.pbc_ = {true, false, true};
        atom.cell_.row(0) << 4, 0, 0;
        atom.cell_.row(2) << 3, 2, 0;
        delta = Eigen::RowVector3d(3.43, .98, 7) + 3 * atom.cell_.row(0)
            - 2 * atom.cell_.row(2);
        expected << .43, -1.02, 7;
    } else {
        atom.pbc_ = {true, true, true};
        atom.cell_ << 4, 0, 0, 3, 2, 0, 1.2, .7, 3;
        delta = Eigen::RowVector3d(3.43, .98, 1.31) + 4 * atom.cell_.row(0)
            - 3 * atom.cell_.row(1) + 2 * atom.cell_.row(2);
        expected << .43, -1.02, 1.31;
    }
    atom.pos_ = {Eigen::RowVector3d(.2, .3, .4), Eigen::RowVector3d(.2, .3, .4) + delta};
    atom.chemmap_ = {0, 0};
    atom.forces_ = {{1, 2, 3}, {4, 5, 6}};
    const auto source = atom;
    const auto enumerated = reference_pair_image(atom, delta);
    REQUIRE(enumerated.isApprox(expected, 1e-12));
    REQUIRE(atom.get_rab(0, 1).isApprox(enumerated, 1e-12));
    REQUIRE(atom.get_rab(1, 0).isApprox(-enumerated, 1e-12));
    REQUIRE(atom.get_distance(0, 1) == Approx(enumerated.norm()).margin(1e-12));
    REQUIRE(atom.get_rab(0, 0).isZero());
    REQUIRE(atom.get_distance(1, 1) == 0);
    REQUIRE(atom.pos_ == source.pos_);
    REQUIRE(atom.cell_ == source.cell_);
    REQUIRE(atom.pbc_ == source.pbc_);
    REQUIRE(atom.cartesian == source.cartesian);
    REQUIRE(atom.forces_ == source.forces_);
}

TEST_CASE("Native pair images convert fractional deltas without changing the source", "[atoms][mic]") {
    const bool partial = GENERATE(false, true);
    CAPTURE(partial);
    Atoms atom;
    atom.cell_ << 4, 0, 0, 3, 2, 0, 1.2, .7, 3;
    atom.pbc_ = {true, true, !partial};
    const Eigen::RowVector3d delta = Eigen::RowVector3d(3.43, .98, 1.31)
        + 4 * atom.cell_.row(0) - 3 * atom.cell_.row(1) + 2 * atom.cell_.row(2);
    atom.pos_ = {Eigen::RowVector3d(1.3, -.4, .2), Eigen::RowVector3d(1.3, -.4, .2) + delta};
    atom.chemmap_ = {0, 0};
    const auto expected = reference_pair_image(atom, delta);
    REQUIRE(atom.get_rab(0, 1).isApprox(expected, 1e-12));
    atom.to_fractional();
    const auto source = atom;
    REQUIRE(atom.get_rab(0, 1).isApprox(expected, 1e-12));
    REQUIRE(atom.get_rab(1, 0).isApprox(-expected, 1e-12));
    REQUIRE(atom.get_distance(0, 1) == Approx(expected.norm()).margin(1e-12));
    REQUIRE(atom.get_rab(0, 0).isZero());
    REQUIRE(atom.get_distance(1, 1) == 0);
    REQUIRE_FALSE(atom.cartesian);
    REQUIRE(atom.pos_ == source.pos_);
    REQUIRE(atom.cell_ == source.cell_);
    REQUIRE(atom.pbc_ == source.pbc_);
}

TEST_CASE("Atoms get_cm", "[atoms]") {
    Atoms atom;
    atom.pos_ = {{0, 0, 0}, {2, 0, 0}, {1, 1, 1}};
    atom.chemmap_ = {0, 0, 0};
    auto cm = atom.get_cm();
    REQUIRE(cm[0] == Approx(1.0));
    REQUIRE(cm[1] == Approx(1.0 / 3.0));
    REQUIRE(cm[2] == Approx(1.0 / 3.0));
}

// ═══════════════════════════════════════════════════════════════════════
// EFS management
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms resetEFS", "[atoms]") {
    auto atom = make_dimer();
    atom.resetEFS();
    REQUIRE(atom.energy_ == Approx(0.0));
    REQUIRE(atom.stress_.isZero());
    REQUIRE(atom.forces_.size() == atom.size());
    for (int i = 0; i < atom.size(); ++i)
        REQUIRE(atom.forces_[i].isZero());
    REQUIRE(atom.have_energy == true);
    REQUIRE(atom.have_stress == true);
}

TEST_CASE("Atoms resetE", "[atoms]") {
    auto atom = make_dimer();
    atom.energy_ = 5.0;
    atom.resetE();
    REQUIRE(atom.energy_ == Approx(0.0));
    REQUIRE(atom.have_energy == true);
    REQUIRE(atom.have_stress == false);
    // resetE does NOT resize forces (unlike resetEFS)
}

TEST_CASE("Atoms clearEFS", "[atoms]") {
    auto atom = make_dimer();
    atom.resetEFS();                  // sets have_energy=true, have_stress=true
    REQUIRE(atom.have_energy == true);
    atom.clearEFS();
    REQUIRE_FALSE(atom.has_forces());
    REQUIRE_FALSE(atom.has_site_energies());
    REQUIRE(atom.have_energy == false);
    REQUIRE(atom.have_stress == false);
}

TEST_CASE("Atoms setEFSflag / setEflag", "[atoms]") {
    Atoms atom;
    atom.setEFSflag();
    REQUIRE(atom.have_energy == true);
    REQUIRE(atom.have_stress == true);

    atom.have_energy = false;
    atom.have_stress = false;
    atom.setEflag();
    REQUIRE(atom.have_energy == true);
    REQUIRE(atom.have_stress == false);
}

// ═══════════════════════════════════════════════════════════════════════
// Translate / rotate / wrap
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms translate", "[atoms]") {
    auto atom = make_dimer(10.0, 2.0);
    atom.translate({1.0, 2.0, 3.0});
    REQUIRE(atom.pos_[0][0] == Approx(1.0));
    REQUIRE(atom.pos_[0][1] == Approx(2.0));
    REQUIRE(atom.pos_[0][2] == Approx(3.0));
    REQUIRE(atom.pos_[1][0] == Approx(3.0));
    REQUIRE(atom.pos_[1][1] == Approx(2.0));
}

TEST_CASE("Atoms rotate around axis preserves distance", "[atoms]") {
    Atoms atom;
    atom.pos_ = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    atom.chemmap_ = {0, 0, 0};
    atom.rotate(Eigen::Vector3d::UnitZ(), M_PI / 2.0);
    // (1,0,0) -> (0,1,0), (0,1,0) -> (-1,0,0), (0,0,1) -> (0,0,1)
    REQUIRE(atom.pos_[0][0] == Approx(0.0).margin(1e-10));
    REQUIRE(atom.pos_[0][1] == Approx(1.0).margin(1e-10));
    REQUIRE(atom.pos_[2][2] == Approx(1.0).margin(1e-10));
}

TEST_CASE("Atoms rotate lattice vector", "[atoms]") {
    auto atom = make_dimer(5.0, 2.0);
    auto cell0 = atom.cell_;
    atom.rotate<true, false>(1, 0, 0, M_PI); // no cell rotation
    REQUIRE(atom.cell_.isApprox(cell0)); // cell unchanged
}

TEST_CASE("Atoms wrap brings atoms inside cell", "[atoms]") {
    Atoms atom;
    atom.cell_ << 10, 0, 0,
                   0, 10, 0,
                   0,  0, 10;
    atom.pos_ = {{-2, 5, 12}, {0, 0, 0}};
    atom.chemmap_ = {0, 0};
    atom.wrap();
    // -2 should become 8 (with periodic wrapping)
    // 12 should become 2
    for (int a = 0; a < 3; ++a) {
        REQUIRE(atom.pos_[0][a] >= -1e-10);
        REQUIRE(atom.pos_[0][a] < 10.0 + 1e-10);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Cell deformation
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms scale_by_factor", "[atoms]") {
    auto atom = make_dimer(4.0, 2.0);
    double v0 = atom.get_volume();
    atom.scale_by_factor(1.5);
    // volume scales by factor^3, each cell vector by factor
    REQUIRE(atom.get_volume() == Approx(v0 * 1.5 * 1.5 * 1.5));
    REQUIRE(atom.cell_(0, 0) == Approx(6.0)); // 4*1.5
}

TEST_CASE("Atoms scale_to_volume", "[atoms]") {
    auto atom = make_dimer(4.0, 2.0);
    atom.scale_to_volume(1000.0);
    REQUIRE(atom.get_volume() == Approx(1000.0));
}

TEST_CASE("Atoms deformR right-multiply", "[atoms]") {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 3.0;
    atom.pos_ = {{1, 0, 0}};
    atom.chemmap_ = {0};
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0, 0) = 2.0;
    atom.deformR(F);
    // cell_ = cell_ * F = 3I * diag(2,1,1) = diag(6,3,3)
    REQUIRE(atom.cell_(0, 0) == Approx(6.0));
    REQUIRE(atom.cell_(1, 1) == Approx(3.0));
    // pos_ = pos_ * F, so (1,0,0) -> (2,0,0)
    REQUIRE(atom.pos_[0][0] == Approx(2.0));
}

TEST_CASE("Atoms deformL left-multiply", "[atoms]") {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 3.0;
    atom.pos_ = {{1, 0, 0}};
    atom.chemmap_ = {0};
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0, 0) = 2.0;
    atom.deformL(F);
    // cell_ = F * cell_ = diag(2,1,1) * 3I = diag(6,3,3)
    REQUIRE(atom.cell_(0, 0) == Approx(6.0));
    // fractional coords preserved: (1/3,0,0) * diag(6,3,3) = (2,0,0)
    REQUIRE(atom.pos_[0][0] == Approx(2.0));
}

// ═══════════════════════════════════════════════════════════════════════
// Resize / reserve
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms resize", "[atoms]") {
    Atoms atom = make_dimer();
    REQUIRE(atom.size() == 2);
    atom.resize(5);
    REQUIRE(atom.size() == 5);
    REQUIRE(atom.chemmap_.size() == 5);
}

TEST_CASE("Atoms reserve", "[atoms]") {
    Atoms atom;
    atom.reserve(100);
    REQUIRE(atom.pos_.capacity() >= 100);
}

// ═══════════════════════════════════════════════════════════════════════
// Type operations
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms get_types returns unique sorted types", "[atoms]") {
    Atoms atom;
    atom.pos_ = {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}};
    atom.chemmap_ = {0, 1, 1, 0, 2, 2, 2};
    auto types = atom.get_types();
    // get_types() returns unique, sorted types
    REQUIRE(types == std::vector<int>({0, 1, 2}));
}

TEST_CASE("Atoms get_types_set", "[atoms]") {
    Atoms atom;
    atom.pos_ = {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}};
    atom.chemmap_ = {0, 1, 1, 0, 2, 2, 2};
    auto tset = atom.get_types_set();
    REQUIRE(tset.size() == 3);
    REQUIRE(tset.count(0));
    REQUIRE(tset.count(1));
    REQUIRE(tset.count(2));
}

TEST_CASE("Atoms types_count_map", "[atoms]") {
    Atoms atom;
    atom.chemmap_ = {0, 0, 1, 0, 1, 2};
    auto tcm = atom.types_count_map();
    REQUIRE(tcm[0] == 3);
    REQUIRE(tcm[1] == 2);
    REQUIRE(tcm[2] == 1);
}

// ═══════════════════════════════════════════════════════════════════════
// Has flags
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms has_* flags", "[atoms]") {
    Atoms atom;
    REQUIRE(!atom.has_forces());
    REQUIRE(!atom.has_energy());
    REQUIRE(!atom.has_stresses());
    REQUIRE(!atom.has_velo());
    REQUIRE(!atom.has_charges());
    REQUIRE(!atom.has_ids());
    REQUIRE(!atom.has_molecule());
    REQUIRE(!atom.has_bonds());
    REQUIRE(!atom.has_angle());

    atom.forces_.resize(1);
    REQUIRE(atom.has_forces());

    atom.have_energy = true;
    REQUIRE(atom.has_energy());

    atom.have_stress = true;
    REQUIRE(atom.has_stresses());

    atom.velocities_.resize(1);
    REQUIRE(atom.has_velo());
}

TEST_CASE("Atoms has_site_energies has_eff_charges has_magmom", "[atoms]") {
    Atoms atom;
    REQUIRE(!atom.has_site_energies());
    REQUIRE(!atom.has_eff_charges());
    REQUIRE(!atom.has_magmom());

    atom.site_energies_ = {1.0};
    REQUIRE(atom.has_site_energies());

    atom.born_eff_chgs_.resize(1);
    REQUIRE(atom.has_eff_charges());
}

// ═══════════════════════════════════════════════════════════════════════
// get_voigt_stress / report_maxforce
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms get_voigt_stress", "[atoms]") {
    Atoms atom;
    atom.stress_ << 1, 4, 5,
                     4, 2, 6,
                     5, 6, 3;
    auto voigt = atom.get_voigt_stress();
    REQUIRE(voigt.size() == 6);
    REQUIRE(voigt[0] == Approx(1.0));  // xx
    REQUIRE(voigt[1] == Approx(2.0));  // yy
    REQUIRE(voigt[2] == Approx(3.0));  // zz
    REQUIRE(voigt[3] == Approx(6.0));  // yz
    REQUIRE(voigt[4] == Approx(5.0));  // xz
    REQUIRE(voigt[5] == Approx(4.0));  // xy
}

TEST_CASE("Atoms report_maxforce", "[atoms]") {
    Atoms atom;
    atom.forces_ = {{1, 0, 0}, {0, 2, 0}, {0, 0, 0}};
    REQUIRE(atom.report_maxforce() == Approx(2.0));
}

// ═══════════════════════════════════════════════════════════════════════
// get_strain
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms get_strain identity", "[atoms]") {
    Atoms ref = make_dimer(5.0, 2.0);
    Atoms def = ref; // identical
    auto strain = def.get_strain(ref);
    REQUIRE(strain.isApprox(Eigen::Matrix3d::Zero(), 1e-10));
}

TEST_CASE("Atoms get_strain hydrostatic expansion", "[atoms]") {
    Atoms ref = make_dimer(5.0, 2.0);
    Atoms def = ref;
    def.cell_ *= 1.02;
    auto strain = def.get_strain(ref);
    for (int i = 0; i < 3; ++i)
        REQUIRE(strain(i, i) == Approx(0.02).epsilon(1e-8));
}

// ═══════════════════════════════════════════════════════════════════════
// get_cell_parameters / get_lower_cell / get_norm_cell
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms get_cell_parameters cubic", "[atoms]") {
    auto atom = make_dimer(5.0);
    auto params = atom.get_cell_parameters();
    // get_cell_parameters returns angles in radians
    REQUIRE(params.size() == 6);
    REQUIRE(params[0] == Approx(5.0));        // a
    REQUIRE(params[1] == Approx(5.0));        // b
    REQUIRE(params[2] == Approx(5.0));        // c
    REQUIRE(params[3] == Approx(M_PI / 2.0)); // alpha
    REQUIRE(params[4] == Approx(M_PI / 2.0)); // beta
    REQUIRE(params[5] == Approx(M_PI / 2.0)); // gamma
}

TEST_CASE("Atoms get_lower_cell", "[atoms]") {
    auto atom = make_dimer(5.0);
    auto lower = atom.get_lower_cell();
    // lower triangular
    REQUIRE(lower(0, 1) == Approx(0.0));
    REQUIRE(lower(0, 2) == Approx(0.0));
    REQUIRE(lower(1, 2) == Approx(0.0));
    REQUIRE(lower.determinant() > 0);
}

TEST_CASE("Atoms get_norm_cell", "[atoms]") {
    auto atom = make_dimer(5.0);
    auto norm = atom.get_norm_cell();
    // normalized cell vectors should all have unit length for cubic
    for (int i = 0; i < 3; ++i)
        REQUIRE(norm.row(i).norm() == Approx(1.0));
}

// ═══════════════════════════════════════════════════════════════════════
// alloc helpers
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms alloc_ld / alloc_li / alloc_lvd / alloc_lvi", "[atoms]") {
    Atoms atom = make_dimer();
    atom.alloc_ld("test_double");
    REQUIRE(atom.local_attribs_d_["test_double"].size() == 2);

    atom.alloc_li("test_int");
    REQUIRE(atom.local_attribs_i_["test_int"].size() == 2);

    atom.alloc_lvd("test_vec", 4);
    REQUIRE(atom.local_attribs_v_["test_vec"].rows() == 4);
    REQUIRE(atom.local_attribs_v_["test_vec"].cols() == 2);

    atom.alloc_lvi("test_veci", 3);
    REQUIRE(atom.local_attribs_vi_["test_veci"].rows() == 3);
    REQUIRE(atom.local_attribs_vi_["test_veci"].cols() == 2);
}

// ═══════════════════════════════════════════════════════════════════════
// swap_axis
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms swap_axis", "[atoms]") {
    Atoms atom;
    atom.cell_ << 3, 0, 0,
                  0, 4, 0,
                  0, 0, 5;
    atom.pos_ = {{1, 2, 3}};
    atom.chemmap_ = {0};

    double vol0 = atom.get_volume();
    atom.swap_axis(132); // xyz -> xzy: swap y and z axes
    REQUIRE(atom.size() == 1);
    // Volume preserved
    REQUIRE(atom.get_volume() == Approx(std::abs(vol0)).epsilon(1e-10));
    // Cell is valid
    REQUIRE(atom.get_handedness() == 1);
}

// ═══════════════════════════════════════════════════════════════════════
// Masks
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Atoms has_mask", "[atoms]") {
    Atoms atom;
    REQUIRE(!atom.has_mask());
    atom.masks = {0, 0, 1};
    REQUIRE(atom.has_mask());
}


TEST_CASE("Minimum distance respects free directions and absent nonperiodic cell rows", "[atoms][mindist]") {
    auto atom = make_dimer(10, 8);
    atom.pbc_ = {false, false, false};
    atom.cell_.setZero();
    REQUIRE(atom.report_mindist() == Approx(8));
    atom.pos_[1] = {2e10, 0, 0};
    REQUIRE(atom.report_mindist() == Approx(2e10));

    SECTION("One periodic direction needs only one cell row") {
        atom.pbc_ = {false, true, false};
        atom.cell_.row(1) << 10, 0, 0;
        atom.pos_[1] = {29, 2, 0};
        REQUIRE(atom.report_mindist() == Approx(std::sqrt(5.0)));
        atom.pos_[1] = {0, 8, 0};
        REQUIRE(atom.report_mindist() == Approx(8));
    }
    SECTION("Two periodic directions leave the normal displacement free") {
        atom.pbc_ = {true, false, true};
        atom.cell_.row(0) << 10, 0, 0;
        atom.cell_.row(2) << 3, 10, 0;
        atom.pos_[1] = {0, 0, 8};
        REQUIRE(atom.report_mindist() == Approx(8));
    }
}

TEST_CASE("Minimum distance finds skew images and preserves coordinate representation", "[atoms][mindist]") {
    auto atom = make_dimer();
    atom.cell_ << 4, 0, 0, 3, 2, 0, 0, 0, 10;
    atom.pos_[1] = {3.43, .98, 0};
    const double expected = std::sqrt(1.2253); // delta - second cell row
    REQUIRE(atom.report_mindist() == Approx(expected));
    atom.pos_[1] += 7 * atom.cell_.row(0) - 4 * atom.cell_.row(1)
        + 3 * atom.cell_.row(2);
    REQUIRE(atom.report_mindist() == Approx(expected));
    atom.to_fractional();
    const auto original_positions = atom.pos_;
    const auto original_cell = atom.cell_;
    atom.forces_ = {{1, 2, 3}, {4, 5, 6}};
    REQUIRE(atom.report_mindist() == Approx(expected));
    REQUIRE_FALSE(atom.cartesian);
    REQUIRE(atom.cell_.isApprox(original_cell));
    for (size_t i = 0; i < atom.pos_.size(); ++i)
        REQUIRE(atom.pos_[i].isApprox(original_positions[i]));
    REQUIRE(atom.forces_[1].isApprox(Eigen::Vector3d(4, 5, 6)));
}

TEST_CASE("Minimum distance includes periodic self images and distinct coincident atoms", "[atoms][mindist]") {
    Atoms atom;
    REQUIRE(atom.report_mindist() == Approx(1e10));
    atom.pos_ = {{0, 0, 0}};
    atom.chemmap_ = {0};
    atom.cell_ << 4, 0, 0, 3, 2, 0, 0, 0, 10;
    // The shortest lattice vector is B-A, not an original cell row.
    REQUIRE(atom.report_mindist() == Approx(std::sqrt(5.0)));
    atom.pbc_ = {false, false, false};
    REQUIRE(atom.report_mindist() == Approx(1e10));
    atom.pos_.push_back({0, 0, 0});
    atom.chemmap_.push_back(0);
    REQUIRE(atom.report_mindist() == 0);
    atom.pbc_ = {true, true, true};
    atom.pos_[1] = atom.cell_.row(1);
    REQUIRE(atom.report_mindist() == Approx(0).margin(1e-12));
}

TEST_CASE("Minimum distance includes distant periodic images", "[atoms][mindist]") {
    auto atom = make_dimer(100, 60);
    atom.pos_[0] = {20, 50, 50};
    atom.pos_[1] = {80, 50, 50};
    REQUIRE(atom.report_mindist() == Approx(40));
    atom.pos_[1] = {331, 50, 50};
    REQUIRE(atom.report_mindist() == Approx(11));
}

namespace {
Atoms coordinate_mutator_fixture(bool fractional, bool left_handed = false) {
    Atoms atom;
    atom.cell_ << 4, 0, 0, 0, 5, 0, 0, 0, left_handed ? -6 : 6;
    atom.pos_ = fractional ? std::vector<Eigen::RowVector3d>{{.25, .4, .5}} :
        std::vector<Eigen::RowVector3d>{{1, 2, left_handed ? -3. : 3.}};
    atom.chemmap_ = {0};
    atom.cartesian = !fractional;
    return atom;
}

Eigen::RowVector3d physical_position(const Atoms &atom) {
    if (atom.cartesian) return atom.pos_[0];
    return atom.pos_[0] * atom.cell_;
}
}

TEST_CASE("Direct position getter respects both stored representations", "[atoms][coordinates]") {
    const bool fractional = GENERATE(false, true);
    auto atom = coordinate_mutator_fixture(fractional);
    atom.cell_(0, 1) = 1;
    if (!fractional) atom.pos_[0] = {1, 2.25, 3};
    const auto original_position = atom.pos_[0];
    const auto original_cell = atom.cell_;
    std::vector<Eigen::RowVector3d> direct;
    atom.get_dir_pos(direct);
    REQUIRE(direct.size() == 1);
    REQUIRE(direct[0].isApprox(Eigen::RowVector3d(.25, .4, .5)));
    REQUIRE(atom.pos_[0] == original_position);
    REQUIRE(atom.cell_ == original_cell);
    REQUIRE(atom.cartesian == !fractional);
}

TEST_CASE("Scaled cell setters preserve the stored representation", "[atoms][coordinates]") {
    const bool fractional = GENERATE(false, true);
    const bool matrix_setter = GENERATE(false, true);
    auto atom = coordinate_mutator_fixture(fractional);
    Eigen::Matrix3d expected;
    expected << 8, 0, 0, 0, 10, 0, 0, 0, 12;
    if (matrix_setter) atom.set_cell(expected);
    else atom.set_cell(8, 10, 12, M_PI / 2, M_PI / 2, M_PI / 2, true);
    REQUIRE(atom.cell_.isApprox(expected));
    REQUIRE(atom.cartesian == !fractional);
    REQUIRE(physical_position(atom).isApprox(Eigen::RowVector3d(2, 4, 6)));
    if (fractional) REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(.25, .4, .5)));
}

TEST_CASE("Scaled matrix cell setter retains its handedness choice", "[atoms][coordinates]") {
    const bool fractional = GENERATE(false, true);
    auto atom = coordinate_mutator_fixture(fractional);
    Eigen::Matrix3d next;
    next << 8, 0, 0, 0, 10, 0, 0, 0, -12;
    atom.set_cell(next);
    REQUIRE(atom.cell_.isApprox(next));
    REQUIRE(atom.cartesian == !fractional);
    // Existing Cartesian semantics flip the direct z coordinate when handedness changes.
    REQUIRE(physical_position(atom).isApprox(Eigen::RowVector3d(2, 4, 6)));
    if (fractional) REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(.25, .4, -.5)));
}

TEST_CASE("Unscaled cell replacement preserves physical positions across handedness", "[atoms][coordinates]") {
    const bool fractional = GENERATE(false, true);
    auto atom = coordinate_mutator_fixture(fractional, true);
    atom.set_cell(8, 10, 12, M_PI / 2, M_PI / 2, M_PI / 2, false);
    Eigen::Matrix3d expected;
    expected << 8, 0, 0, 0, 10, 0, 0, 0, 12;
    REQUIRE(atom.cell_.isApprox(expected));
    REQUIRE(atom.cartesian == !fractional);
    REQUIRE(physical_position(atom).isApprox(Eigen::RowVector3d(1, 2, -3)));
    if (fractional) REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(.125, .2, -.25)));
}

TEST_CASE("Left and right deformation preserve direct positions without double conversion", "[atoms][coordinates]") {
    const bool fractional = GENERATE(false, true);
    const bool left = GENERATE(false, true);
    auto atom = coordinate_mutator_fixture(fractional);
    atom.cell_(0, 1) = 1;
    if (!fractional) atom.pos_[0] = {1, 2.25, 3};
    Eigen::Matrix3d deformation = Eigen::Matrix3d::Identity();
    deformation(0, 1) = .5;
    if (left) atom.deformL(deformation);
    else atom.deformR(deformation);
    Eigen::Matrix3d expected;
    expected << 4, left ? 3.5 : 3., 0, 0, 5, 0, 0, 0, 6;
    REQUIRE(atom.cell_.isApprox(expected));
    REQUIRE(atom.cartesian == !fractional);
    REQUIRE(physical_position(atom).isApprox(Eigen::RowVector3d(1, left ? 2.875 : 2.75, 3)));
    if (fractional) REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(.25, .4, .5)));
}

TEST_CASE("Factor and volume scaling preserve fractional components", "[atoms][coordinates]") {
    const bool fractional = GENERATE(false, true);
    const bool by_volume = GENERATE(false, true);
    auto atom = coordinate_mutator_fixture(fractional);
    if (by_volume) atom.scale_to_volume(960);
    else atom.scale_by_factor(2);
    Eigen::Matrix3d expected;
    expected << 8, 0, 0, 0, 10, 0, 0, 0, 12;
    REQUIRE(atom.cell_.isApprox(expected));
    REQUIRE(atom.get_volume() == Approx(960));
    REQUIRE(atom.cartesian == !fractional);
    REQUIRE(physical_position(atom).isApprox(Eigen::RowVector3d(2, 4, 6)));
    if (fractional) REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(.25, .4, .5)));
}

TEST_CASE("Translation takes a Cartesian displacement for either stored representation", "[atoms][coordinates]") {
    const bool fractional = GENERATE(false, true);
    auto atom = coordinate_mutator_fixture(fractional);
    const auto original_cell = atom.cell_;
    atom.translate(Eigen::Vector3d(1, 0, 0));
    REQUIRE(atom.cell_ == original_cell);
    REQUIRE(atom.cartesian == !fractional);
    REQUIRE(physical_position(atom).isApprox(Eigen::RowVector3d(2, 2, 3)));
    if (fractional) REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(.5, .4, .5)));
}

TEST_CASE("Direct position setter explicitly stores Cartesian coordinates", "[atoms][coordinates]") {
    const bool fractional = GENERATE(false, true);
    auto atom = coordinate_mutator_fixture(fractional);
    atom.set_dir_pos({Eigen::RowVector3d(.5, .2, .25)});
    REQUIRE(atom.cartesian);
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(2, 1, 1.5)));
}

TEST_CASE("Standard form applies the same cell replacement to fractional inputs", "[atoms][coordinates]") {
    const bool fractional = GENERATE(false, true);
    Atoms atom;
    atom.cell_ << 4, 0, 0, 0, 5, 0, 0, 0, 6;
    atom.pos_ = fractional ? std::vector<Eigen::RowVector3d>{{.25, .4, .5}} :
        std::vector<Eigen::RowVector3d>{{1, 2, 3}};
    atom.chemmap_ = {0};
    atom.cartesian = !fractional;
    const auto rotation = atom.standard_form(false);
    // A diagonal cell isolates representation handling from the QR basis algorithm.
    Eigen::Matrix3d expected;
    expected << 4, 0, 0, 0, 5, 0, 0, 0, 6;
    REQUIRE(atom.cell_.isApprox(expected));
    REQUIRE(rotation.isApprox(Eigen::Matrix3d::Identity()));
    REQUIRE(atom.cartesian == !fractional);
    REQUIRE(physical_position(atom).isApprox(Eigen::RowVector3d(1, 2, 3)));
    if (fractional) REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(.25, .4, .5)));
}


TEST_CASE("Standard form lower preserves the row-cell metric with an independent rotation", "[atoms][standard_form]") {
    const bool fractional = GENERATE(false, true);
    Atoms atom;
    atom.cell_ << 3, 4, 0, 0, 5, 0, 0, 0, 6;
    atom.pos_ = {{.75, 3, 3}};
    atom.chemmap_ = {0};
    const Eigen::Matrix3d original_cell = atom.cell_;
    if (fractional) atom.to_fractional();
    Eigen::Matrix3d expected_cell, expected_rotation;
    expected_cell << 5, 0, 0, 4, 3, 0, 0, 0, 6;
    expected_rotation << .6, .8, 0, -.8, .6, 0, 0, 0, 1;
    const Eigen::Matrix3d rotation = atom.standard_form(false);
    CHECK(atom.cell_.isApprox(expected_cell, 1e-12));
    CHECK(rotation.isApprox(expected_rotation, 1e-12));
    CHECK((atom.cell_ * rotation).isApprox(original_cell, 1e-12));
    CHECK((atom.cell_ * atom.cell_.transpose()).isApprox(
        original_cell * original_cell.transpose(), 1e-12));
    CHECK((rotation * rotation.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    CHECK(rotation.determinant() == Approx(1));
    CHECK(atom.cell_.determinant() == Approx(90));
    REQUIRE(atom.cartesian == !fractional);
    const Eigen::RowVector3d position = fractional ?
        Eigen::RowVector3d(atom.pos_[0] * atom.cell_) : atom.pos_[0];
    CHECK(position.isApprox(Eigen::RowVector3d(2.85, 1.2, 3), 1e-12));
    if (fractional)
        CHECK(atom.pos_[0].isApprox(Eigen::RowVector3d(.25, .4, .5), 1e-12));
}

TEST_CASE("Standard form upper and left-handed cells obey the row-cell factorization", "[atoms][standard_form]") {
    Atoms atom;
    Eigen::Matrix3d expected_cell, expected_rotation;
    bool upper = false;
    SECTION("Upper form requires a nontrivial proper rotation") {
        upper = true;
        atom.cell_ << 5, 0, 0, 4, 3, 0, 0, 0, 6;
        expected_cell << 3, 4, 0, 0, 5, 0, 0, 0, 6;
        expected_rotation << .6, -.8, 0, .8, .6, 0, 0, 0, 1;
    }
    SECTION("Left-handed lower form retains negative diagonal signs") {
        atom.cell_ << 3, 4, 0, 0, 5, 0, 0, 0, -6;
        expected_cell << -5, 0, 0, -4, -3, 0, 0, 0, -6;
        expected_rotation << -.6, -.8, 0, .8, -.6, 0, 0, 0, 1;
    }
    const Eigen::Matrix3d original_cell = atom.cell_;
    const Eigen::RowVector3d direct(.25, .4, .5);
    atom.pos_ = {direct * original_cell};
    atom.chemmap_ = {0};
    const Eigen::Matrix3d rotation = atom.standard_form(upper);
    CHECK(atom.cell_.isApprox(expected_cell, 1e-12));
    CHECK(rotation.isApprox(expected_rotation, 1e-12));
    CHECK((atom.cell_ * rotation).isApprox(original_cell, 1e-12));
    CHECK((atom.cell_ * atom.cell_.transpose()).isApprox(
        original_cell * original_cell.transpose(), 1e-12));
    CHECK((rotation * rotation.transpose()).isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    CHECK(rotation.determinant() == Approx(1));
    CHECK(atom.cell_.determinant() == Approx(original_cell.determinant()));
    CHECK(atom.pos_[0].isApprox(direct * expected_cell, 1e-12));
    REQUIRE(atom.cartesian);
}

TEST_CASE("Standard form rotates established physical labels with Cartesian positions", "[atoms][standard_form]") {
    const bool fractional = GENERATE(false, true);
    Atoms atom;
    atom.cell_ << 3, 4, 0, 0, 5, 0, 0, 0, 6;
    atom.pos_ = {{.75, 3, 3}, {1, 2, 1}};
    atom.chemmap_ = {0, 1};
    atom.forces_ = {{1, 2, 3}, {-.4, .2, 1}};
    atom.velocities_ = {{.3, .2, .1}, {-.1, .4, .2}};
    atom.magmoms_ = {{1, 2, 0}, {0, 1, 2}};
    atom.born_eff_chgs_ = {Eigen::Vector3d(1, 2, 3).asDiagonal(), Eigen::Matrix3d::Identity()};
    atom.have_eff_chg = true;
    atom.stress_ << 1, .2, .3, .2, 2, .4, .3, .4, 3;
    atom.have_stress = true;
    atom.energy_ = -7;
    atom.have_energy = true;
    atom.charges_ = {.1, -.2};
    atom.hessian_ = Eigen::VectorXd::LinSpaced(6, 1, 6).asDiagonal();
    atom.hessian_(0, 4) = atom.hessian_(4, 0) = .3;
    atom.have_hessian = true;
    atom.hvp_directions_.resize(6, 2);
    atom.hvp_directions_.col(0) = Eigen::VectorXd::LinSpaced(6, .1, .6);
    atom.hvp_directions_.col(1) = Eigen::VectorXd::LinSpaced(6, -.2, .3);
    atom.hvp_targets_ = atom.hessian_ * atom.hvp_directions_;
    const Atoms original = atom;
    if (fractional) atom.to_fractional();
    Eigen::Matrix3d expected_rotation;
    expected_rotation << .6, .8, 0, -.8, .6, 0, 0, 0, 1;
    const Eigen::Matrix3d rotation = atom.standard_form(false);
    CHECK(rotation.isApprox(expected_rotation, 1e-12));
    REQUIRE(atom.cartesian == !fractional);
    for (int i = 0; i < atom.size(); ++i) {
        const Eigen::RowVector3d position = fractional ?
            Eigen::RowVector3d(atom.pos_[i] * atom.cell_) : atom.pos_[i];
        CHECK(position.isApprox(original.pos_[i] * expected_rotation.transpose(), 1e-12));
        CHECK(atom.forces_[i].isApprox(expected_rotation * original.forces_[i], 1e-12));
        CHECK(atom.velocities_[i].isApprox(expected_rotation * original.velocities_[i], 1e-12));
        CHECK(atom.magmoms_[i].isApprox(expected_rotation * original.magmoms_[i], 1e-12));
        CHECK(atom.born_eff_chgs_[i].isApprox(
            expected_rotation * original.born_eff_chgs_[i] * expected_rotation.transpose(), 1e-12));
        CHECK(atom.hvp_directions_.middleRows(3 * i, 3).isApprox(
            expected_rotation * original.hvp_directions_.middleRows(3 * i, 3), 1e-12));
        CHECK(atom.hvp_targets_.middleRows(3 * i, 3).isApprox(
            expected_rotation * original.hvp_targets_.middleRows(3 * i, 3), 1e-12));
        for (int j = 0; j < atom.size(); ++j)
            CHECK((atom.hessian_.block<3, 3>(3 * i, 3 * j).isApprox(
                expected_rotation * original.hessian_.block<3, 3>(3 * i, 3 * j)
                * expected_rotation.transpose(), 1e-12)));
    }
    CHECK(atom.stress_.isApprox(
        expected_rotation * original.stress_ * expected_rotation.transpose(), 1e-12));
    CHECK((atom.hessian_ * atom.hvp_directions_).isApprox(atom.hvp_targets_, 1e-12));
    CHECK(atom.chemmap_ == original.chemmap_);
    CHECK(atom.energy_ == original.energy_);
    CHECK(atom.charges_ == original.charges_);
}
