// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "base/structures.h"
#include "base/neighborhoodKD.h"
#include "io/gpio.h"
#include <sstream>
#include <array>
#include <numbers>
#include <stdexcept>

using Catch::Approx;

static Atoms labeled_atoms()
{
    Atoms a;
    a.cell_ << 10, 0, 0, 2, 11, 0, 1, 3, 12;
    a.pos_ = {{1, 2, 3}, {4, 5, 6}};
    a.chemmap_ = {1, 2};
    a.forces_ = {{1, 2, 3}, {4, 5, 6}};
    a.velocities_ = {{3, 2, 1}, {6, 5, 4}};
    a.site_energies_ = {3, 4};
    a.charges_ = {0.3, -0.3};
    a.global_attribs_d_["total_charge"] = .8; // independent of the atomic charge sum
    a.magmom_scalar_ = {1, -1};
    a.magmoms_ = {{1, 2, 0}, {0, 1, 2}};
    a.born_eff_chgs_ = {Eigen::Vector3d(1, 2, 3).asDiagonal(), Eigen::Matrix3d::Identity()};
    a.have_eff_chg = true;
    a.energy_ = 7; a.have_energy = true;
    a.fenergy_ = 6; a.have_fenergy = true;
    a.stress_ = Eigen::Vector3d(2, 3, 4).asDiagonal(); a.have_stress = true;
    a.atomic_fitness = {0.1, 0.2};
    a.masks = {1, 2};
    a.sel_dynpos_ = {{1, 0, 1}, {0, 1, 0}};
    a.fitting_factor = {9, 1, 2, 8};
    a.local_attribs_d_["scalar"] = Eigen::Vector2d(1, 2);
    a.local_attribs_i_["integer"] = Eigen::Vector2i(3, 4);
    a.local_attribs_v_["vector"] = Eigen::MatrixXd::Ones(3, 2);
    a.local_attribs_vi_["ivec"] = Eigen::MatrixXi::Ones(2, 2);
    a.ids_ = {10, 20};
    a.bonds_ = {{{0, 1}}};
    a.molid_ = {0, 0}; a.molsize_ = {2}; a.molnum_ = 1;
    a.hessian_ = Eigen::VectorXd::LinSpaced(6, 1, 6).asDiagonal();
    a.hessian_(0, 3) = a.hessian_(3, 0) = .4;
    a.hessian_(1, 5) = a.hessian_(5, 1) = -.2;
    a.have_hessian = true;
    a.hvp_directions_.resize(6, 2);
    a.hvp_directions_.col(0) = Eigen::VectorXd::LinSpaced(6, .1, .6);
    a.hvp_directions_.col(1) = Eigen::VectorXd::LinSpaced(6, -.3, .2);
    a.hvp_targets_ = a.hessian_ * a.hvp_directions_;
    return a;
}

TEST_CASE("Supercell builders preserve complete physical labels", "[atoms][supercell]")
{
    const int builder = GENERATE(0, 1, 2, 3);
    const auto original = labeled_atoms();
    auto a = original;
    Eigen::Matrix3i matrix = Eigen::Matrix3i::Identity(); matrix(0, 0) = 2;
    if (builder == 0) a.mksuperL(matrix);
    if (builder == 1) a.mksuper({2, 1, 1});
    if (builder == 2) a.mksuperL_hnf(matrix);
    if (builder == 3) a.createSuperL(matrix.cast<double>());
    REQUIRE(a.size() == 4);
    REQUIRE(a.forces_.size() == 4);
    REQUIRE(a.velocities_.size() == 4);
    REQUIRE(a.site_energies_.size() == 4);
    for (int i = 0; i < 4; ++i) {
        const int src = i % 2;
        REQUIRE(a.forces_[i].isApprox(original.forces_[src]));
        REQUIRE(a.velocities_[i].isApprox(original.velocities_[src]));
        REQUIRE(a.site_energies_[i] == original.site_energies_[src]);
        REQUIRE(a.charges_[i] == original.charges_[src]);
        REQUIRE(a.magmoms_[i].isApprox(original.magmoms_[src]));
        REQUIRE(a.magmom_scalar_[i] == original.magmom_scalar_[src]);
        REQUIRE(a.born_eff_chgs_[i].isApprox(original.born_eff_chgs_[src]));
        REQUIRE(a.atomic_fitness[i] == original.atomic_fitness[src]);
        REQUIRE(a.masks[i] == original.masks[src]);
        REQUIRE(a.sel_dynpos_[i].isApprox(original.sel_dynpos_[src]));
        REQUIRE(a.local_attribs_d_.at("scalar")[i] == src + 1);
        REQUIRE(a.local_attribs_i_.at("integer")[i] == src + 3);
    }
    REQUIRE(a.local_attribs_v_.at("vector").cols() == 4);
    REQUIRE(a.local_attribs_vi_.at("ivec").cols() == 4);
    REQUIRE(a.fitting_factor == std::vector<double>{9, 1, 2, 1, 2, 8});
    REQUIRE(a.energy_ == Approx(14));
    REQUIRE(a.fenergy_ == Approx(12));
    REQUIRE(a.stress_.isApprox(2 * original.stress_));
    REQUIRE(a.global_attribs_d_.at("total_charge") == Approx(1.6));
    REQUIRE(a.ids_.empty());
    REQUIRE(a.bonds_.empty());
    REQUIRE(a.molid_.empty());
    REQUIRE_FALSE(a.have_hessian);
    REQUIRE(a.hvp_directions_.size() == 0);
    REQUIRE(a.hvp_targets_.size() == 0);
    GPIO io; io.setmap();
    std::stringstream output;
    // This fixture deliberately contains both magnetic representations; BIN
    // can retain both, while EXTXYZ has only one canonical magmoms column.
    REQUIRE(io.write_one(a, output, "bin"));
    Atoms restored;
    REQUIRE(io.read_one(restored, output, "bin"));
    REQUIRE(restored.size() == 4);
    REQUIRE(restored.forces_.size() == 4);
    REQUIRE(restored.magmoms_ == a.magmoms_);
    REQUIRE(restored.magmom_scalar_ == a.magmom_scalar_);
    REQUIRE(restored.energy_ == Approx(14));
    REQUIRE(restored.global_attribs_d_.at("total_charge") == Approx(1.6));
}

TEST_CASE("Single-copy supercell builders preserve Cartesian HVP labels", "[atoms][supercell][hvp]")
{
    const int builder = GENERATE(0, 1, 2, 3);
    const auto original = labeled_atoms();
    auto a = original;
    const Eigen::Matrix3i identity = Eigen::Matrix3i::Identity();
    if (builder == 0) a.mksuperL(identity);
    if (builder == 1) a.mksuper({1, 1, 1});
    if (builder == 2) a.mksuperL_hnf(identity);
    if (builder == 3) a.createSuperL(identity.cast<double>());
    REQUIRE(a.size() == original.size());
    REQUIRE(a.cell_.isApprox(original.cell_));
    REQUIRE(a.pos_ == original.pos_);
    REQUIRE(a.hessian_.isApprox(original.hessian_));
    REQUIRE(a.hvp_directions_.isApprox(original.hvp_directions_));
    REQUIRE(a.hvp_targets_.isApprox(original.hvp_targets_));
    REQUIRE((a.hessian_ * a.hvp_directions_).isApprox(a.hvp_targets_));
    REQUIRE_NOTHROW(a.validate_hvp());
    REQUIRE(a.global_attribs_d_.at("total_charge") == original.global_attribs_d_.at("total_charge"));
}

TEST_CASE("Unimodular basis changes preserve Cartesian HVP labels", "[atoms][cell][hvp]")
{
    auto a = labeled_atoms();
    const auto original = a;
    SECTION("integer shear") {
        Eigen::Matrix3i matrix = Eigen::Matrix3i::Identity();
        matrix(0, 1) = 1;
        a.mksuperL(matrix);
        REQUIRE(a.cell_.isApprox(matrix.cast<double>() * original.cell_));
    }
    SECTION("signed axis swap") {
        a.pbc_ = {true, false, true};
        a.swap_axis(132);
        REQUIRE(a.pbc_ == std::array<bool, 3>{true, true, false});
        REQUIRE(a.get_handedness() == original.get_handedness());
    }
    REQUIRE(a.size() == original.size());
    REQUIRE(a.pos_ == original.pos_);
    REQUIRE(a.hessian_.isApprox(original.hessian_));
    REQUIRE(a.hvp_directions_.isApprox(original.hvp_directions_));
    REQUIRE(a.hvp_targets_.isApprox(original.hvp_targets_));
    REQUIRE((a.hessian_ * a.hvp_directions_).isApprox(a.hvp_targets_));
    REQUIRE_NOTHROW(a.validate_hvp());
}

TEST_CASE("Supercell expansion does not infer a missing total charge", "[atoms][supercell][charge]")
{
    auto a = labeled_atoms();
    a.global_attribs_d_.erase("total_charge");
    a.charges_ = {.2, .3};
    a.mksuper({2, 1, 1});
    REQUIRE(a.charges_ == std::vector<double>{.2, .3, .2, .3});
    REQUIRE_FALSE(a.global_attribs_d_.contains("total_charge"));
}

TEST_CASE("Cell standardization rotates physical vectors and tensors", "[atoms][rotation]")
{
    const bool legacy_name = GENERATE(false, true);
    const bool left_handed = GENERATE(false, true);
    const auto reference = labeled_atoms();
    auto a = reference;
    const Eigen::Matrix3d q = Eigen::AngleAxisd(0.63, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    a.cell_ = reference.cell_ * q.transpose();
    if (left_handed) a.cell_.row(2) *= -1;
    for (int i = 0; i < a.size(); ++i) {
        a.pos_[i] = reference.pos_[i] * q.transpose();
        a.forces_[i] = q * reference.forces_[i];
        a.velocities_[i] = q * reference.velocities_[i];
        a.magmoms_[i] = q * reference.magmoms_[i];
        a.born_eff_chgs_[i] = q * reference.born_eff_chgs_[i] * q.transpose();
        a.hvp_directions_.middleRows(3 * i, 3) = q * reference.hvp_directions_.middleRows(3 * i, 3);
        a.hvp_targets_.middleRows(3 * i, 3) = q * reference.hvp_targets_.middleRows(3 * i, 3);
        for (int j = 0; j < a.size(); ++j)
            a.hessian_.block<3, 3>(3 * i, 3 * j) = q * reference.hessian_.block<3, 3>(3 * i, 3 * j) * q.transpose();
    }
    a.stress_ = q * reference.stress_ * q.transpose();
    if (legacy_name) a.mkLower(); else a.to_lower_standard();
    REQUIRE(a.cell_.isApprox(reference.cell_));
    for (int i = 0; i < a.size(); ++i) {
        REQUIRE(a.pos_[i].isApprox(reference.pos_[i]));
        REQUIRE(a.forces_[i].isApprox(reference.forces_[i]));
        REQUIRE(a.velocities_[i].isApprox(reference.velocities_[i]));
        REQUIRE(a.magmoms_[i].isApprox(reference.magmoms_[i]));
        REQUIRE(a.born_eff_chgs_[i].isApprox(reference.born_eff_chgs_[i]));
    }
    REQUIRE(a.stress_.isApprox(reference.stress_));
    REQUIRE(a.hessian_.isApprox(reference.hessian_));
    REQUIRE(a.hvp_directions_.isApprox(reference.hvp_directions_));
    REQUIRE(a.hvp_targets_.isApprox(reference.hvp_targets_));
    REQUIRE(a.energy_ == reference.energy_);
}

TEST_CASE("Changing cell handedness preserves Cartesian positions", "[atoms][cell]")
{
    auto a = labeled_atoms();
    a.cell_.row(2) *= -1;
    const auto original = a;
    const bool fractional = GENERATE(false, true);
    if (fractional) a.to_fractional();
    a.to_righthand();
    REQUIRE(a.get_handedness() == 1);
    REQUIRE(a.cartesian == !fractional);
    if (fractional) a.to_cartesian();
    for (int i = 0; i < a.size(); ++i)
        REQUIRE(a.pos_[i].isApprox(original.pos_[i]));
}

TEST_CASE("Axis permutations reorder periodic flags", "[atoms][pbc]")
{
    const int mode = GENERATE(123, 132, 213, 231, 312, 321);
    const bool keephand = GENERATE(false, true);
    auto a = labeled_atoms(); a.pbc_ = {true, false, true};
    const auto old = a;
    a.swap_axis(mode, keephand);
    const std::array<int, 3> axes{mode / 100 - 1, mode / 10 % 10 - 1, mode % 10 - 1};
    for (int i = 0; i < 3; ++i) REQUIRE(a.pbc_[i] == old.pbc_[axes[i]]);
    REQUIRE(a.size() == old.size());
    REQUIRE(a.energy_ == old.energy_);
    REQUIRE(a.forces_[0].isApprox(old.forces_[0]));
    if (keephand) REQUIRE(a.get_handedness() == old.get_handedness());
}

TEST_CASE("Wrapping only changes periodic fractional coordinates", "[atoms][pbc]")
{
    const int free_axis = GENERATE(0, 1, 2);
    const int method = GENERATE(0, 1, 2, 3);
    auto a = labeled_atoms(); a.pbc_[free_axis] = false;
    Eigen::RowVector3d f(-0.2, 1.3, 1.8), expected = f;
    for (int i = 0; i < 3; ++i) if (a.pbc_[i]) expected[i] -= std::floor(expected[i]);
    a.pos_[0] = f * a.cell_;
    if (method == 0) a.wrap();
    if (method == 1) a.wrap(a.pos_[0]);
    if (method == 2) a.hwrap();
    if (method == 3) { a.to_fractional(); a.wrap(); a.to_cartesian(); }
    REQUIRE(a.pos_[0].isApprox(expected * a.cell_));
}

TEST_CASE("KD images only extend periodic lattice directions", "[neighborhood][pbc]")
{
    const int axis = GENERATE(0, 1, 2);
    const bool periodic = GENERATE(false, true);
    Atoms a; a.cell_ = Eigen::Matrix3d::Identity() * 10;
    a.pos_ = {{5, 5, 5}, {5, 5, 5}}; a.chemmap_ = {0, 0};
    a.pos_[0][axis] = 0.2; a.pos_[1][axis] = 9.8;
    a.pbc_ = {false, false, false}; a.pbc_[axis] = periodic;
    // The free coordinates need not be inside the cell.
    a.pos_[0][(axis + 1) % 3] = a.pos_[1][(axis + 1) % 3] = 100;
    NeighborhoodKD kd(&a, 5.0);
    REQUIRE(kd.query(0, 1).distances_[0] == Approx(periodic ? 0.4 : 9.6));
}

TEST_CASE("Nonperiodic geometries need no cell for wrapping or KD search", "[atoms][pbc]")
{
    Atoms a; a.pbc_ = {false, false, false};
    a.pos_ = {{-1, 0, 0}, {2, 0, 0}}; a.chemmap_ = {0, 0};
    a.wrap(); a.hwrap(); a.wrap(a.pos_[0]);
    REQUIRE(a.pos_[0][0] == -1);
    NeighborhoodKD kd(&a, 5.0);
    REQUIRE(kd.query(0, 1).distances_[0] == Approx(3));
}


TEST_CASE("Geometry-only supercells normalize positions and preserve PBC and parent mapping", "[atoms][supercell]")
{
    const bool fractional = GENERATE(false, true);
    const int handedness = GENERATE(1, -1);
    Atoms source;
    source.cell_ << 4, 0, 0, 0, 5, 0, 0, 0, 6;
    source.pos_ = {{.5, 1, 1.8}, {2.5, 2, 4.2}};
    source.chemmap_ = {1, 8};
    source.pbc_ = {true, true, false};
    source.energy_ = -7;
    source.have_energy = true;
    source.forces_ = {{1, 2, 3}, {4, 5, 6}};
    if (fractional) source.to_fractional();
    const Atoms original = source;
    Eigen::Matrix3i matrix;
    matrix << 2 * handedness, 0, 0, 1, 1, 0, 0, 0, 1;
    std::map<int, int> mapping{{99, 42}};
    const Atoms result = source.create_supercell(matrix, mapping);
    REQUIRE(result.size() == 4);
    REQUIRE(result.cartesian);
    CHECK(result.pbc_ == source.pbc_);
    const Eigen::Matrix3d expected_cell = matrix.cast<double>() * source.cell_;
    CHECK(result.cell_.isApprox(expected_cell));
    CHECK(result.cell_.determinant() == Approx(handedness * 240));
    REQUIRE(mapping.size() == 4);
    // For this matrix, the image offsets are {0, A} or {-A, 0}.
    const int first_image = handedness > 0 ? 0 : -1;
    const std::array<Eigen::RowVector3d, 2> parents{
        Eigen::RowVector3d(.5, 1, 1.8), Eigen::RowVector3d(2.5, 2, 4.2)};
    for (int i = 0; i < result.size(); ++i) {
        REQUIRE(mapping.at(i) == i % 2);
        CHECK(result.chemmap_[i] == source.chemmap_[i % 2]);
        const Eigen::RowVector3d expected = parents[i % 2]
            + (first_image + i / 2) * source.cell_.row(0);
        CHECK(result.pos_[i].isApprox(expected, 1e-12));
    }
    CHECK_FALSE(result.have_energy);
    CHECK(result.forces_.empty());
    CHECK(source.cartesian == original.cartesian);
    CHECK(source.pbc_ == original.pbc_);
    CHECK(source.cell_.isApprox(original.cell_));
    CHECK(source.chemmap_ == original.chemmap_);
    CHECK(source.have_energy == original.have_energy);
    CHECK(source.energy_ == original.energy_);
    for (int i = 0; i < source.size(); ++i) {
        CHECK(source.pos_[i].isApprox(original.pos_[i]));
        CHECK(source.forces_[i].isApprox(original.forces_[i]));
    }
}

TEST_CASE("Geometry-only supercell rejects a singular integer matrix before replacing the map", "[atoms][supercell]")
{
    Atoms source;
    source.cell_ = Eigen::Matrix3d::Identity() * 4;
    source.pos_ = {{1, 2, 3}};
    source.chemmap_ = {1};
    std::map<int, int> mapping{{99, 42}};
    const auto original_mapping = mapping;
    const auto original_positions = source.pos_;
    Eigen::Matrix3i matrix = Eigen::Matrix3i::Identity();
    matrix.row(1).setZero();
    REQUIRE_THROWS_AS(source.create_supercell(matrix, mapping), std::invalid_argument);
    CHECK(mapping == original_mapping);
    CHECK(source.pos_[0].isApprox(original_positions[0]));
}

namespace {
Atoms coordinate_transform_fixture(bool fractional)
{
    Atoms atoms;
    atoms.cell_ = Eigen::Vector3d(4, 5, 6).asDiagonal();
    atoms.pos_ = {{1, 2, 3}};
    atoms.chemmap_ = {1};
    atoms.pbc_ = {true, false, true};
    if (fractional) atoms.to_fractional();
    return atoms;
}

Eigen::RowVector3d physical_position(const Atoms& atoms, int index)
{
    if (atoms.cartesian) return atoms.pos_[index];
    return atoms.pos_[index] * atoms.cell_;
}
}

TEST_CASE("In-place supercell builders preserve coordinate representation and physical images", "[atoms][supercell][coordinates]")
{
    const int builder = GENERATE(0, 1, 2, 3);
    const bool fractional = GENERATE(false, true);
    auto atoms = coordinate_transform_fixture(fractional);
    Eigen::Matrix3i matrix = Eigen::Matrix3i::Identity();
    matrix(0, 0) = 2;
    if (builder == 0) atoms.mksuperL(matrix);
    if (builder == 1) atoms.mksuper({2, 1, 1});
    if (builder == 2) atoms.mksuperL_hnf(matrix);
    if (builder == 3) atoms.createSuperL(matrix.cast<double>());
    REQUIRE(atoms.size() == 2);
    CHECK(atoms.cartesian == !fractional);
    CHECK(atoms.pbc_ == std::array<bool, 3>{true, false, true});
    const Eigen::Matrix3d expected_cell = Eigen::Vector3d(8, 5, 6).asDiagonal();
    CHECK(atoms.cell_.isApprox(expected_cell, 1e-12));
    for (int i = 0; i < atoms.size(); ++i) {
        // Independent images of (1,2,3) translated by the old A=(4,0,0).
        const Eigen::RowVector3d expected(1 + 4 * i, 2, 3);
        CHECK(physical_position(atoms, i).isApprox(expected, 1e-12));
        CHECK(atoms.chemmap_[i] == 1);
        if (fractional) {
            const Eigen::RowVector3d expected_direct(.125 + .5 * i, .4, .5);
            CHECK(atoms.pos_[i].isApprox(expected_direct, 1e-12));
        }
    }
}

TEST_CASE("Axis permutation preserves fractional geometry and permutes partial PBC", "[atoms][pbc][coordinates]")
{
    const bool fractional = GENERATE(false, true);
    auto atoms = coordinate_transform_fixture(fractional);
    atoms.swap_axis(231);
    REQUIRE(atoms.size() == 1);
    CHECK(atoms.cartesian == !fractional);
    CHECK(atoms.pbc_ == std::array<bool, 3>{false, true, true});
    Eigen::Matrix3d expected_cell;
    expected_cell << 0, 5, 0, 0, 0, 6, 4, 0, 0;
    CHECK(atoms.cell_.isApprox(expected_cell, 1e-12));
    CHECK(physical_position(atoms, 0).isApprox(Eigen::RowVector3d(1, 2, 3), 1e-12));
    if (fractional)
        CHECK(atoms.pos_[0].isApprox(Eigen::RowVector3d(.4, .5, .25), 1e-12));
}

TEST_CASE("Cartesian-axis rotation preserves stored representation in all action modes", "[atoms][rotation][coordinates]")
{
    const int action = GENERATE(0, 1, 2);
    const bool fractional = GENERATE(false, true);
    auto atoms = coordinate_transform_fixture(fractional);
    const auto original_pbc = atoms.pbc_;
    const double angle = std::numbers::pi / 2;
    if (action == 0) atoms.rotate<true, true>(Eigen::Vector3d::UnitZ(), angle);
    if (action == 1) atoms.rotate<true, false>(Eigen::Vector3d::UnitZ(), angle);
    if (action == 2) atoms.rotate<false, true>(Eigen::Vector3d::UnitZ(), angle);
    CHECK(atoms.cartesian == !fractional);
    CHECK(atoms.pbc_ == original_pbc);
    Eigen::Matrix3d rotated_cell;
    rotated_cell << 0, 4, 0, -5, 0, 0, 0, 0, 6;
    const Eigen::Matrix3d unchanged_cell = Eigen::Vector3d(4, 5, 6).asDiagonal();
    CHECK(atoms.cell_.isApprox(action == 1 ? unchanged_cell : rotated_cell, 1e-12));
    const Eigen::RowVector3d expected = action == 2
        ? Eigen::RowVector3d(1, 2, 3) : Eigen::RowVector3d(-2, 1, 3);
    CHECK(physical_position(atoms, 0).isApprox(expected, 1e-12));
    if (fractional) {
        Eigen::RowVector3d expected_direct(.25, .4, .5);
        if (action == 1) expected_direct = Eigen::RowVector3d(-.5, .2, .5);
        if (action == 2) expected_direct = Eigen::RowVector3d(.5, -.2, .5);
        CHECK(atoms.pos_[0].isApprox(expected_direct, 1e-12));
    }
}

TEST_CASE("Rotation with both actions disabled leaves fractional storage untouched", "[atoms][rotation][coordinates]")
{
    auto atoms = coordinate_transform_fixture(true);
    const auto original = atoms;
    atoms.rotate<false, false>(Eigen::Vector3d::UnitZ(), std::numbers::pi / 2);
    CHECK_FALSE(atoms.cartesian);
    CHECK(atoms.pos_ == original.pos_);
    CHECK(atoms.cell_ == original.cell_);
    CHECK(atoms.pbc_ == original.pbc_);
}

TEST_CASE("Rotation center remains Cartesian for fractional stored positions", "[atoms][rotation][coordinates]")
{
    for (bool fractional : {false, true}) {
        auto atoms = coordinate_transform_fixture(false);
        atoms.pos_[0] = Eigen::RowVector3d(2, 0, 0);
        if (fractional) atoms.to_fractional();
        atoms.rotate<true, false>(Eigen::Vector3d::UnitZ(), std::numbers::pi / 2,
                                 Eigen::Vector3d(1, 0, 0));
        CHECK(atoms.cartesian == !fractional);
        const Eigen::Matrix3d expected_cell = Eigen::Vector3d(4, 5, 6).asDiagonal();
        CHECK(atoms.cell_.isApprox(expected_cell, 1e-12));
        CHECK(physical_position(atoms, 0).isApprox(Eigen::RowVector3d(1, 1, 0), 1e-12));
        if (fractional)
            CHECK(atoms.pos_[0].isApprox(Eigen::RowVector3d(.25, .2, 0), 1e-12));
    }
}

TEST_CASE("Lattice-axis rotation reuses the fractional coordinate conversion", "[atoms][rotation][coordinates]")
{
    auto atoms = coordinate_transform_fixture(true);
    atoms.rotate(1, 0, 0, std::numbers::pi / 2);
    CHECK_FALSE(atoms.cartesian);
    Eigen::Matrix3d expected_cell;
    expected_cell << 4, 0, 0, 0, 0, 5, 0, -6, 0;
    CHECK(atoms.cell_.isApprox(expected_cell, 1e-12));
    CHECK(physical_position(atoms, 0).isApprox(Eigen::RowVector3d(1, -3, 2), 1e-12));
    CHECK(atoms.pos_[0].isApprox(Eigen::RowVector3d(.25, .4, .5), 1e-12));
}


TEST_CASE("Cell standardization rotates only the named Cartesian initial magnetic vectors", "[atoms][rotation][magmoms]")
{
    const std::string method = GENERATE("standard", "lower", "legacy");
    const bool scalar = GENERATE(false, true);
    Atoms original;
    original.cell_ << 0,4,0, -5,0,0, 0,0,6;
    original.pos_ = {{1,2,3},{2,3,4}};
    original.chemmap_ = {0,1};
    original.magmoms_ = {{0,3,0},{0,0,4}};
    if (scalar) {
        original.local_attribs_d_["initial_magmoms"] = Eigen::Vector2d(.25,-.75);
    } else {
        original.local_attribs_v_["initial_magmoms"].resize(3,2);
        original.local_attribs_v_["initial_magmoms"] << 1,0, 0,2, 0,1;
    }
    original.local_attribs_v_["unrelated_vectors"] = Eigen::MatrixXd::Ones(3,2);
    auto transformed = original;
    if (method == "standard") transformed.standard_form(false);
    else if (method == "lower") transformed.to_lower_standard();
    else transformed.mkLower();
    // This cell needs a clockwise quarter-turn about Cartesian z, independently
    // of the transformation routine's returned matrix or computed moments.
    const Eigen::Matrix3d expected_cell = Eigen::Vector3d(4,5,6).asDiagonal();
    REQUIRE(transformed.cell_.isApprox(expected_cell,1e-12));
    REQUIRE(transformed.pos_[0].isApprox(Eigen::RowVector3d(2,-1,3),1e-12));
    REQUIRE(transformed.pos_[1].isApprox(Eigen::RowVector3d(3,-2,4),1e-12));
    REQUIRE(transformed.magmoms_[0].isApprox(Eigen::Vector3d(3,0,0),1e-12));
    REQUIRE(transformed.magmoms_[1].isApprox(Eigen::Vector3d(0,0,4),1e-12));
    if (scalar) {
        REQUIRE(transformed.local_attribs_d_.at("initial_magmoms").isApprox(Eigen::Vector2d(.25,-.75)));
        REQUIRE_FALSE(transformed.local_attribs_v_.contains("initial_magmoms"));
    } else {
        Eigen::Matrix<double,3,2> expected;
        expected << 0,2, -1,0, 0,1;
        REQUIRE(transformed.local_attribs_v_.at("initial_magmoms").isApprox(expected,1e-12));
        REQUIRE_FALSE(transformed.local_attribs_d_.contains("initial_magmoms"));
        REQUIRE(original.local_attribs_v_.at("initial_magmoms")(0,0) == 1);
        REQUIRE(original.local_attribs_v_.at("initial_magmoms")(1,0) == 0);
    }
    REQUIRE(transformed.local_attribs_v_.at("unrelated_vectors").isApprox(
        original.local_attribs_v_.at("unrelated_vectors"),0));
    REQUIRE(original.pos_[0].isApprox(Eigen::RowVector3d(1,2,3),0));
    REQUIRE(original.magmoms_[0].isApprox(Eigen::Vector3d(0,3,0),0));
}
