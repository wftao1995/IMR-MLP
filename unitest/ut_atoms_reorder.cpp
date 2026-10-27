#include <catch2/catch_all.hpp>
#include "base/structures.h"
#include <array>
#include <limits>
#include <sstream>

namespace {
Atoms labeled_atoms()
{
    Atoms a;
    a.cell_ << 8, 0, 0, 1, 9, 0, 2, 3, 10;
    a.pos_ = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    a.chemmap_ = {0, 2, 1};
    a.ids_ = {101, 102, 103};
    a.pbc_ = {true, false, true};
    a.cartesian = false;
    a.nlocal_ = 3; a.nghost_ = 0; a.molnum_ = 0;
    a.id = 42; a.mvgrade_ = 2; a.mindist_ = 3; a.maxdist_ = 9;
    a.energy_ = 17; a.fenergy_ = 15;
    a.have_energy = a.have_fenergy = a.have_stress = true;
    a.stress_ << 1, 2, 3, 2, 4, 5, 3, 5, 6;
    a.sel_dyncell_ << 1, 0, 1, 0, 1, 0, 1, 0, 1;
    a.fitness = .7; a.group_count = 3; a.collinear_ = false;
    a.have_eff_chg = true;
    for (int i = 0; i < 3; ++i) {
        a.forces_.emplace_back(i + 1, i + 2, i + 3);
        a.velocities_.emplace_back(i + 4, i + 5, i + 6);
        a.site_energies_.push_back(10 + i);
        a.charges_.push_back(.2 + i);
        a.magmom_scalar_.push_back(.3 + i);
        a.magmoms_.emplace_back(i + 7, i + 8, i + 9);
        Eigen::Matrix3d born;
        born << 1+i, 2+i, 3+i, 4+i, 5+i, 6+i, 7+i, 8+i, 9+i;
        a.born_eff_chgs_.push_back(born);
        a.atomic_fitness.push_back(.4 + i);
        a.masks.push_back(5 + i);
        a.sel_dynpos_.emplace_back(i == 0, i == 1, i == 2);
    }
    a.fitting_factor = {7, 11, 12, 13, 19};
    a.local_attribs_d_["scalar"] = Eigen::Vector3d(3, 7, 11);
    a.local_attribs_i_["integer"] = Eigen::Vector3i(13, 17, 19);
    a.local_attribs_v_["vector"].resize(2, 3);
    a.local_attribs_v_["vector"] << 1, 2, 3, 4, 5, 6;
    a.local_attribs_vi_["integer_vector"].resize(4, 3);
    a.local_attribs_vi_["integer_vector"] << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    a.local_attribs_d_["empty"].resize(0);
    a.local_attribs_i_["empty"].resize(0);
    a.local_attribs_v_["empty"].resize(2, 0);
    a.local_attribs_vi_["empty"].resize(4, 0);
    a.global_attribs_d_["temperature"] = 300;
    a.global_attribs_i_["step"] = 100;
    a.global_attribs_v_["field"] = Eigen::Vector3d(4, 8, 16);
    a.global_attribs_vi_["counts"] = Eigen::Vector3i(5, 9, 17);
    a.features_["origin"] = "reorder test";
    Eigen::MatrixXd matrix(9, 9);
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j) matrix(i, j) = (i + 1) * (j + 2) + (i == j ? 3 : 0);
    a.hessian_ = matrix.transpose() * matrix;
    a.have_hessian = true;
    a.hvp_directions_.resize(9, 2);
    for (int i = 0; i < 9; ++i) {
        a.hvp_directions_(i, 0) = .25 * (i + 1);
        a.hvp_directions_(i, 1) = .5 * (i % 3) - 1;
    }
    a.hvp_targets_ = a.hessian_ * a.hvp_directions_;
    return a;
}

std::string snapshot(const Atoms &a)
{
    std::ostringstream output(std::ios::out | std::ios::binary);
    a.write_to_bin(output);
    // The legacy binary helper does not include derivative labels.
    output.write(reinterpret_cast<const char *>(&a.have_hessian), sizeof(a.have_hessian));
    for (const auto *matrix : {&a.hessian_, &a.hvp_directions_, &a.hvp_targets_}) {
        const Eigen::Index rows = matrix->rows(), cols = matrix->cols();
        output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
        output.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
        if (matrix->size())
            output.write(reinterpret_cast<const char *>(matrix->data()), sizeof(double) * matrix->size());
    }
    return output.str();
}
}

TEST_CASE("Atoms reorder preserves every supported atom field and structural metadata", "[atoms][reorder]")
{
    auto a = labeled_atoms();
    const auto original = a;
    const std::array<int, 3> order{2, 0, 1};
    a.reorder(std::span<const int>(order));
    const auto check = [&order](const auto &actual, const auto &before) {
        REQUIRE(actual.size() == before.size());
        for (size_t i = 0; i < order.size(); ++i) CHECK(actual[i] == before[order[i]]);
    };
    check(a.pos_, original.pos_);
    check(a.chemmap_, original.chemmap_);
    check(a.ids_, original.ids_);
    check(a.forces_, original.forces_);
    check(a.velocities_, original.velocities_);
    check(a.site_energies_, original.site_energies_);
    check(a.charges_, original.charges_);
    check(a.magmom_scalar_, original.magmom_scalar_);
    check(a.magmoms_, original.magmoms_);
    check(a.born_eff_chgs_, original.born_eff_chgs_);
    check(a.atomic_fitness, original.atomic_fitness);
    check(a.masks, original.masks);
    check(a.sel_dynpos_, original.sel_dynpos_);
    for (int i = 0; i < 3; ++i) {
        CHECK(a.local_attribs_d_.at("scalar")[i] == original.local_attribs_d_.at("scalar")[order[i]]);
        CHECK(a.local_attribs_i_.at("integer")[i] == original.local_attribs_i_.at("integer")[order[i]]);
        CHECK(a.local_attribs_v_.at("vector").col(i) == original.local_attribs_v_.at("vector").col(order[i]));
        CHECK(a.local_attribs_vi_.at("integer_vector").col(i) == original.local_attribs_vi_.at("integer_vector").col(order[i]));
    }
    CHECK(a.local_attribs_d_.at("empty").size() == 0);
    CHECK(a.local_attribs_i_.at("empty").size() == 0);
    CHECK(a.local_attribs_v_.at("empty").rows() == 2);
    CHECK(a.local_attribs_v_.at("empty").cols() == 0);
    CHECK(a.local_attribs_vi_.at("empty").rows() == 4);
    CHECK(a.local_attribs_vi_.at("empty").cols() == 0);
    CHECK(a.fitting_factor == std::vector<double>{7, 13, 11, 12, 19});
    CHECK(a.cell_ == original.cell_);
    CHECK(a.stress_ == original.stress_);
    CHECK(a.sel_dyncell_ == original.sel_dyncell_);
    CHECK(a.pbc_ == original.pbc_);
    CHECK(a.cartesian == original.cartesian);
    CHECK(a.energy_ == original.energy_);
    CHECK(a.fenergy_ == original.fenergy_);
    CHECK(a.fitness == original.fitness);
    CHECK(a.global_attribs_d_ == original.global_attribs_d_);
    CHECK(a.global_attribs_i_ == original.global_attribs_i_);
    CHECK(a.global_attribs_v_ == original.global_attribs_v_);
    CHECK(a.global_attribs_vi_ == original.global_attribs_vi_);
    CHECK(a.features_ == original.features_);
    CHECK(a.have_hessian == original.have_hessian);
    CHECK(a.have_eff_chg == original.have_eff_chg);
    // A non-self-inverse permutation catches either direction being reversed.
    const std::vector<int> inverse{1, 2, 0};
    a.reorder(inverse);
    CHECK(snapshot(a) == snapshot(original));
}

TEST_CASE("Atoms reorder transforms Hessian and HVP with the same Cartesian permutation", "[atoms][reorder][hvp]")
{
    auto a = labeled_atoms();
    const auto before = a;
    Eigen::Matrix<double, 9, 9> q = Eigen::Matrix<double, 9, 9>::Zero();
    q.block<3, 3>(0, 6).setIdentity();
    q.block<3, 3>(3, 0).setIdentity();
    q.block<3, 3>(6, 3).setIdentity();
    const std::vector<int> order{2, 0, 1};
    a.reorder(order);
    CHECK(a.hessian_.isApprox(q * before.hessian_ * q.transpose(), 1e-14));
    CHECK(a.hvp_directions_.isApprox(q * before.hvp_directions_, 1e-14));
    CHECK(a.hvp_targets_.isApprox(q * before.hvp_targets_, 1e-14));
    CHECK((a.hessian_ * a.hvp_directions_).isApprox(a.hvp_targets_, 1e-14));
    CHECK_NOTHROW(a.validate_hvp());
    // Stored matrices also follow atoms if the optional Hessian flag is off.
    auto unflagged = before;
    unflagged.have_hessian = false;
    unflagged.reorder(order);
    CHECK(unflagged.hessian_ == a.hessian_);
    CHECK_FALSE(unflagged.have_hessian);
}

TEST_CASE("Atoms reorder handles identity empty structures and the legacy pointer contract", "[atoms][reorder]")
{
    auto a = labeled_atoms();
    const auto original = snapshot(a);
    int identity[]{0, 1, 2};
    a.reorder(identity);
    CHECK(snapshot(a) == original);
    REQUIRE_THROWS_AS(a.reorder(static_cast<int *>(nullptr)), std::invalid_argument);
    CHECK(snapshot(a) == original);
    Atoms empty;
    REQUIRE_NOTHROW(empty.reorder(std::span<const int>{}));
    REQUIRE_NOTHROW(empty.reorder(static_cast<int *>(nullptr)));
    CHECK(empty.size() == 0);
    // No whole-Atoms copy or uninitialized molecule/ghost-count read is needed.
    Atoms geometry;
    geometry.pos_ = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    geometry.chemmap_ = {0, 1, 2};
    geometry.ids_ = {2, 0, 1};
    geometry.reorder(geometry.ids_.data()); // The permutation aliases an output field.
    CHECK(geometry.ids_ == std::vector<int>{1, 2, 0});
    CHECK(geometry.chemmap_ == std::vector<int>{2, 0, 1});
    CHECK(geometry.pos_[0] == Eigen::RowVector3d(7, 8, 9));
}

TEST_CASE("Atoms reorder preserves the structure terms in fitting factors", "[atoms][reorder]")
{
    const int layout = GENERATE(0, 1, 4, 5);
    auto a = labeled_atoms();
    const std::vector<double> values{7, 11, 12, 13, 19};
    a.fitting_factor.assign(values.begin(), values.begin() + layout);
    a.reorder(std::vector<int>{2, 0, 1});
    const std::vector<double> expected{7, 13, 11, 12, 19};
    CHECK(a.fitting_factor == std::vector<double>(expected.begin(), expected.begin() + layout));
}

TEST_CASE("Atoms reorder rejects invalid permutations before changing any field", "[atoms][reorder]")
{
    const int mode = GENERATE(0, 1, 2, 3, 4);
    auto a = labeled_atoms();
    const auto original = snapshot(a);
    const std::vector<std::vector<int>> orders{{0, 1}, {0, 1, 2, 3}, {0, -1, 2}, {0, 1, 3}, {1, 1, 2}};
    REQUIRE_THROWS_AS(a.reorder(orders[mode]), std::invalid_argument);
    CHECK(snapshot(a) == original);
}

TEST_CASE("Atoms reorder rejects incomplete atom fields transactionally", "[atoms][reorder]")
{
    const int field = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21);
    auto a = labeled_atoms();
    switch (field) {
    case 0: a.chemmap_.resize(2); break;
    case 1: a.ids_.resize(2); break;
    case 2: a.forces_.resize(2); break;
    case 3: a.velocities_.resize(2); break;
    case 4: a.site_energies_.resize(2); break;
    case 5: a.charges_.resize(2); break;
    case 6: a.magmom_scalar_.resize(2); break;
    case 7: a.magmoms_.resize(2); break;
    case 8: a.born_eff_chgs_.resize(2); break;
    case 9: a.atomic_fitness.resize(2); break;
    case 10: a.masks.resize(2); break;
    case 11: a.sel_dynpos_.resize(2); break;
    case 12: a.local_attribs_d_["scalar"].conservativeResize(2); break;
    case 13: a.local_attribs_i_["integer"].conservativeResize(2); break;
    case 14: a.local_attribs_v_["vector"].conservativeResize(2, 2); break;
    case 15: a.local_attribs_vi_["integer_vector"].conservativeResize(4, 2); break;
    case 16: a.fitting_factor.resize(3); break;
    case 17: a.hessian_.conservativeResize(9, 6); break;
    case 18: a.hvp_targets_.conservativeResize(9, 1); break;
    case 19: a.hvp_directions_.col(0).setZero(); break;
    case 20: a.hvp_targets_(0, 0) = std::numeric_limits<double>::quiet_NaN(); break;
    case 21: a.hvp_directions_.resize(0, 0); break;
    }
    const auto original = snapshot(a);
    REQUIRE_THROWS_AS(a.reorder(std::vector<int>{2, 0, 1}), std::invalid_argument);
    CHECK(snapshot(a) == original);
}

TEST_CASE("Atoms reorder rejects molecular and ghost layouts without changing them", "[atoms][reorder]")
{
    const int layout = GENERATE(0, 1, 2, 3, 4, 5, 6);
    auto a = labeled_atoms();
    switch (layout) {
    case 0: a.molid_ = {1, 1, 1}; break;
    case 1: a.molsize_ = {3}; break;
    case 2: a.mol_types_ = {2}; break;
    case 3: a.bonds_ = {{0, 2}}; break;
    case 4: a.angles_ = {{0, 1, 2}}; break;
    case 5: a.have_ghost = true; break;
    case 6: a.map_ghost_ = {1}; break;
    }
    const auto original = snapshot(a);
    REQUIRE_THROWS_WITH(a.reorder(std::vector<int>{2, 0, 1}),
                        Catch::Matchers::ContainsSubstring("molecular topology or ghost layouts"));
    CHECK(snapshot(a) == original);
}
