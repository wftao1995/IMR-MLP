#include <catch2/catch_all.hpp>
#include "symmetry/symmetry.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
Atoms primitive(bool tilted = false, bool left_handed = false, bool two_species = false)
{
    Atoms atoms;
    atoms.cell_ << 4, 0, 0, 0, 5, 0, 0, 0, 6;
    if (tilted) {
        atoms.cell_(1, 0) = .5;
        atoms.cell_(2, 0) = .25;
        atoms.cell_(2, 1) = .75;
    }
    if (left_handed) atoms.cell_.row(0) *= -1;
    atoms.resize(two_species ? 2 : 1);
    atoms.pos_[0] = Eigen::RowVector3d(.125, .25, .375) * atoms.cell_;
    atoms.chemmap_[0] = 1;
    if (two_species) {
        atoms.pos_[1] = Eigen::RowVector3d(.65, .6, .55) * atoms.cell_;
        atoms.chemmap_[1] = 8;
    }
    return atoms;
}

Atoms slab(bool repeat_x = true)
{
    const int copies = repeat_x ? 2 : 1;
    Atoms atoms;
    atoms.cell_ << 4 * copies, 0, 0, 0, 5, 0, 0, 0, 20;
    atoms.pbc_ = {true, true, false};
    atoms.resize(2 * copies);
    for (int layer = 0; layer < 2; ++layer) {
        for (int copy = 0; copy < copies; ++copy) {
            const int i = layer * copies + copy;
            atoms.pos_[i] = Eigen::RowVector3d(.5 + 4 * copy, 1, 10 * layer);
            atoms.chemmap_[i] = 1;
        }
    }
    return atoms;
}

int reduce(int api, Atoms& atoms, double tol = 1e-5, int direction = 0)
{
    switch (api) {
    case 0: return Symmetry3D::trans_reduce(atoms, tol);
    case 1: return Symmetry3D::trans_reduce_1d(atoms, direction, tol);
    case 2: return Symmetry3D::find_primitive(atoms, tol);
    default: return Symmetry3D::all_reduce(atoms, tol);
    }
}

void set_extensive_labels(Atoms& atoms)
{
    atoms.energy_ = -5;
    atoms.fenergy_ = -4.5;
    atoms.stress_ << 1, .25, .5, .25, 2, .75, .5, .75, 3;
    atoms.have_energy = atoms.have_fenergy = atoms.have_stress = true;
    atoms.global_attribs_d_["total_charge"] = .8;
    atoms.charges_.assign(atoms.size(), .15); // not the explicit frame charge
}

void require_extensive_labels(const Atoms& atoms, const Atoms& primitive, int copies = 1)
{
    REQUIRE(atoms.have_energy);
    REQUIRE(atoms.have_fenergy);
    REQUIRE(atoms.have_stress);
    REQUIRE(atoms.size() == copies * primitive.size());
    CHECK(atoms.energy_ == Catch::Approx(copies * primitive.energy_).margin(1e-12));
    CHECK(atoms.fenergy_ == Catch::Approx(copies * primitive.fenergy_).margin(1e-12));
    CHECK(atoms.stress_.isApprox(copies * primitive.stress_, 1e-12));
    CHECK(atoms.energy_ / atoms.size() == Catch::Approx(primitive.energy_ / primitive.size()).margin(1e-12));
    CHECK((atoms.stress_ / atoms.get_volume()).isApprox(primitive.stress_ / primitive.get_volume(), 1e-12));
    REQUIRE(atoms.global_attribs_d_.contains("total_charge"));
    CHECK(atoms.global_attribs_d_.at("total_charge") ==
          Catch::Approx(copies * primitive.global_attribs_d_.at("total_charge")).margin(1e-12));
}

std::string snapshot(const Atoms& atoms)
{
    // The legacy helper writes each vector separately, including malformed
    // position/species lengths. Byte comparison also preserves NaN payloads.
    std::ostringstream stream(std::ios::out | std::ios::binary);
    atoms.write_to_bin(stream);
    for (bool periodic : atoms.pbc_) stream.write(reinterpret_cast<const char*>(&periodic), sizeof(periodic));
    return stream.str();
}

void require_unchanged_failure(int api, Atoms& atoms, double tol, const std::string& message)
{
    const auto before = snapshot(atoms);
    REQUIRE_THROWS_AS(reduce(api, atoms, tol), std::invalid_argument);
    CHECK(snapshot(atoms) == before);
    REQUIRE_THROWS_WITH(reduce(api, atoms, tol), Catch::Matchers::ContainsSubstring(message));
    CHECK(snapshot(atoms) == before);
}
}

TEST_CASE("3D reduction rejects empty and mismatched structures before wrapping", "[symmetry][reduction]")
{
    const int api = GENERATE(0, 1, 2, 3);
    auto atoms = primitive();
    atoms.pos_[0] += 2 * atoms.cell_.row(0);
    atoms.id = 42;
    atoms.have_energy = true;
    atoms.energy_ = -7;
    atoms.forces_ = {{1, 2, 3}};
    atoms.features_["sentinel"] = "unchanged";
    SECTION("empty") {
        atoms.resize(0);
        require_unchanged_failure(api, atoms, 1e-5, "nonempty");
    }
    SECTION("missing species") {
        atoms.chemmap_.clear();
        require_unchanged_failure(api, atoms, 1e-5, "matching position and species counts");
    }
    SECTION("excess species") {
        atoms.chemmap_.push_back(8);
        require_unchanged_failure(api, atoms, 1e-5, "matching position and species counts");
    }
}

TEST_CASE("3D reduction rejects invalid tolerance before changing input", "[symmetry][reduction]")
{
    const int api = GENERATE(0, 1, 2, 3);
    const double tol = GENERATE(0., -1., std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::max());
    auto atoms = primitive();
    atoms.pos_[0] += 2 * atoms.cell_.row(0);
    require_unchanged_failure(api, atoms, tol, "positive finite tolerance");
}

TEST_CASE("3D reduction rejects invalid geometry before changing input", "[symmetry][reduction]")
{
    const int api = GENERATE(0, 1, 2, 3);
    auto atoms = primitive();
    atoms.pos_[0] += 2 * atoms.cell_.row(0);
    SECTION("zero cell") { atoms.cell_.setZero(); }
    SECTION("singular cell") { atoms.cell_.row(2) = 2 * atoms.cell_.row(0); }
    SECTION("nonfinite cell") { atoms.cell_(1, 2) = std::numeric_limits<double>::quiet_NaN(); }
    SECTION("infinite cell") { atoms.cell_(1, 2) = std::numeric_limits<double>::infinity(); }
    SECTION("unrepresentable determinant") { atoms.cell_ *= 1e200; }
    SECTION("unrepresentable inverse") { atoms.cell_ *= std::numeric_limits<double>::denorm_min(); }
    SECTION("nonfinite position") { atoms.pos_[0][0] = std::numeric_limits<double>::quiet_NaN(); }
    SECTION("infinite position") { atoms.pos_[0][0] = -std::numeric_limits<double>::infinity(); }
    SECTION("fractional coordinate overflow") {
        atoms.cell_ = .5 * Eigen::Matrix3d::Identity();
        atoms.pos_[0][0] = std::numeric_limits<double>::max();
    }
    require_unchanged_failure(api, atoms, 1e-5, "Cell reduction requires");
}

TEST_CASE("One direction reduction validates direction before cell access", "[symmetry][reduction]")
{
    const int direction = GENERATE(-1, 3, std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    auto atoms = primitive();
    atoms.pos_[0] += 2 * atoms.cell_.row(0);
    const auto before = snapshot(atoms);
    REQUIRE_THROWS_WITH(Symmetry3D::trans_reduce_1d(atoms, direction),
                        "Cell reduction direction must be 0, 1 or 2");
    CHECK(snapshot(atoms) == before);
}

TEST_CASE("Valid primitive structures retain successful no reduction behavior", "[symmetry][reduction]")
{
    const int api = GENERATE(0, 1, 2, 3);
    const bool tilted = GENERATE(false, true);
    const bool left_handed = GENERATE(false, true);
    const bool two_species = GENERATE(false, true);
    auto atoms = primitive(tilted, left_handed, two_species);
    const auto before = atoms;
    REQUIRE(reduce(api, atoms) == 1);
    REQUIRE(atoms.size() == before.size());
    CHECK(atoms.cell_.isApprox(before.cell_, 1e-12));
    CHECK(atoms.pbc_ == before.pbc_);
    CHECK(atoms.chemmap_ == before.chemmap_);
    for (int i = 0; i < atoms.size(); ++i) CHECK(atoms.pos_[i].isApprox(before.pos_[i], 1e-12));
}

TEST_CASE("3D reduction accepts small representable cells without absolute volume cutoff", "[symmetry][reduction]")
{
    const int api = GENERATE(0, 1, 2, 3);
    auto atoms = primitive();
    atoms.cell_ *= 1e-6;
    atoms.pos_[0] *= 1e-6;
    const double volume = atoms.cell_.determinant();
    REQUIRE(volume < 1e-14);
    CHECK(reduce(api, atoms, 1e-9) == 1);
    CHECK(atoms.cell_.determinant() == Catch::Approx(volume));
}

TEST_CASE("Existing supercell reductions retain species and lattice multiplicity", "[symmetry][reduction]")
{
    const int api = GENERATE(0, 1, 2, 3);
    const int direction = GENERATE(0, 1, 2);
    const int copies = GENERATE(2, 3);
    const bool tilted = GENERATE(false, true);
    const bool left_handed = GENERATE(false, true);
    const auto original = primitive(tilted, left_handed, true);
    Atoms atoms = original;
    for (int copy = 1; copy < copies; ++copy) {
        for (int i = 0; i < original.size(); ++i) {
            atoms.pos_.push_back(original.pos_[i] + copy * original.cell_.row(direction));
            atoms.chemmap_.push_back(original.chemmap_[i]);
        }
    }
    atoms.cell_.row(direction) *= copies;
    const Eigen::Matrix3d supercell = atoms.cell_;
    REQUIRE(reduce(api, atoms, 1e-5, direction) == copies);
    REQUIRE(atoms.size() == 2);
    CHECK(atoms.pbc_ == original.pbc_);
    CHECK(atoms.types_count_map() == original.types_count_map());
    CHECK(atoms.cell_.determinant() == Catch::Approx(original.cell_.determinant()).margin(1e-10));
    const Eigen::Matrix3d relation = supercell * atoms.cell_.inverse();
    CHECK((relation - relation.array().round().matrix()).norm() < 1e-10);
    for (int i = 0; i < atoms.size(); ++i) {
        const int reference = atoms.chemmap_[i] == 1 ? 0 : 1;
        const Eigen::RowVector3d displacement = (atoms.pos_[i] - original.pos_[reference]) * original.cell_.inverse();
        CHECK((displacement - displacement.array().round().matrix()).norm() < 1e-10);
    }
}

TEST_CASE("Reduction preserves frame id and named structure labels", "[symmetry][reduction]")
{
    const int api = GENERATE(0, 1, 2, 3);
    const auto original = primitive();
    Atoms atoms = original;
    atoms.resize(4);
    for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
            const int index = 2 * x + y;
            atoms.pos_[index] = original.pos_[0] + x * original.cell_.row(0) + y * original.cell_.row(1);
            atoms.chemmap_[index] = original.chemmap_[0];
        }
    }
    atoms.cell_.row(0) *= 2;
    atoms.cell_.row(1) *= 2;
    atoms.id = 73;
    atoms.features_["label"] = "bulk sweep 0007";
    atoms.features_["config_type"] = "strained_bulk";
    const auto labels = atoms.features_;
    REQUIRE(reduce(api, atoms) == (api == 1 ? 2 : 4));
    REQUIRE(atoms.id == 73);
    REQUIRE(atoms.features_ == labels);
    if (api == 1) {
        REQUIRE(reduce(api, atoms, 1e-5, 1) == 2);
        REQUIRE(atoms.id == 73);
        REQUIRE(atoms.features_ == labels);
    }
    REQUIRE(atoms.size() == 1);
}

TEST_CASE("Reduction does not create absent structure labels", "[symmetry][reduction]")
{
    const int api = GENERATE(0, 1, 2, 3);
    auto atoms = primitive();
    atoms.pos_.push_back(atoms.pos_[0] + atoms.cell_.row(0));
    atoms.chemmap_.push_back(atoms.chemmap_[0]);
    atoms.cell_.row(0) *= 2;
    REQUIRE(reduce(api, atoms) == 2);
    REQUIRE(atoms.id == 0);
    REQUIRE(atoms.features_.empty());
    REQUIRE_FALSE(atoms.have_energy);
    REQUIRE_FALSE(atoms.have_fenergy);
    REQUIRE_FALSE(atoms.have_stress);
    REQUIRE_FALSE(atoms.global_attribs_d_.contains("total_charge"));
}

TEST_CASE("Reduction preserves extensive labels across multiple periodic directions", "[symmetry][reduction][labels]")
{
    const int api = GENERATE(0, 1, 2, 3);
    auto original = primitive(true, false, true);
    set_extensive_labels(original);
    auto atoms = original;
    atoms.mksuper(Eigen::Vector3i(2, 2, 1));
    require_extensive_labels(atoms, original, 4);
    REQUIRE(reduce(api, atoms) == (api == 1 ? 2 : 4));
    if (api == 1) {
        require_extensive_labels(atoms, original, 2);
        REQUIRE(reduce(api, atoms, 1e-5, 1) == 2);
    }
    require_extensive_labels(atoms, original);
}

TEST_CASE("All reduction scales extensive labels once per actual stage", "[symmetry][reduction][labels]")
{
    auto original = primitive();
    set_extensive_labels(original);
    auto expanded = original;
    Eigen::Matrix3i supercell;
    supercell << 2, 2, 0, -1, 1, 0, 0, 0, 1;
    expanded.mksuperL(supercell);
    require_extensive_labels(expanded, original, 4);

    auto sequential = expanded;
    REQUIRE(Symmetry3D::trans_reduce(sequential) == 2);
    require_extensive_labels(sequential, original, 2);
    REQUIRE(Symmetry3D::find_primitive(sequential) == 2);
    require_extensive_labels(sequential, original);

    auto combined = expanded;
    REQUIRE(Symmetry3D::all_reduce(combined) == 4);
    require_extensive_labels(combined, original);
}

TEST_CASE("No reduction leaves extensive labels unchanged", "[symmetry][reduction][labels]")
{
    const int api = GENERATE(0, 1, 2, 3);
    auto atoms = primitive(true);
    set_extensive_labels(atoms);
    const auto original = atoms;
    REQUIRE(reduce(api, atoms) == 1);
    require_extensive_labels(atoms, original);
    CHECK(atoms.energy_ == original.energy_);
    CHECK(atoms.fenergy_ == original.fenergy_);
    CHECK((atoms.stress_.array() == original.stress_.array()).all());
    CHECK(atoms.global_attribs_d_.at("total_charge") == original.global_attribs_d_.at("total_charge"));
}

TEST_CASE("Reduction keeps a free-energy label without inventing other labels", "[symmetry][reduction][labels]")
{
    const int api = GENERATE(0, 1, 2, 3);
    auto atoms = primitive();
    atoms.mksuper(Eigen::Vector3i(2, 1, 1));
    atoms.have_fenergy = true;
    atoms.fenergy_ = -9;
    REQUIRE(reduce(api, atoms) == 2);
    REQUIRE(atoms.have_fenergy);
    CHECK(atoms.fenergy_ == Catch::Approx(-4.5));
    REQUIRE_FALSE(atoms.have_energy);
    REQUIRE_FALSE(atoms.have_stress);
    REQUIRE_FALSE(atoms.global_attribs_d_.contains("total_charge"));
}

TEST_CASE("Translational reduction preserves finite slab layers and periodicity", "[symmetry][reduction]")
{
    const int api = GENERATE(0, 1);
    auto atoms = slab();
    const auto before = atoms;
    REQUIRE(reduce(api, atoms) == 2);
    REQUIRE(atoms.size() == 2);
    CHECK(atoms.pbc_ == before.pbc_);
    Eigen::Matrix3d expected_cell = before.cell_;
    expected_cell.row(0) /= 2;
    CHECK(atoms.cell_.isApprox(expected_cell, 1e-12));
    // Both finite z layers remain, although their separation is half the cell.
    for (int layer = 0; layer < 2; ++layer) {
        const Eigen::RowVector3d expected(.5, 1, 10 * layer);
        CHECK(atoms.pos_[layer].isApprox(expected, 1e-12));
        CHECK(atoms.chemmap_[layer] == 1);
    }
}

TEST_CASE("Translational reduction does not treat finite repetition as a periodic axis", "[symmetry][reduction]")
{
    auto atoms = slab(false);
    const auto before = atoms;
    REQUIRE(Symmetry3D::trans_reduce(atoms) == 1);
    REQUIRE(atoms.size() == 2);
    CHECK(atoms.cell_.isApprox(before.cell_, 1e-12));
    CHECK(atoms.pbc_ == before.pbc_);
    for (int i = 0; i < atoms.size(); ++i)
        CHECK(atoms.pos_[i].isApprox(before.pos_[i], 1e-12));
}

TEST_CASE("One direction reduction leaves nonperiodic directions unchanged", "[symmetry][reduction]")
{
    auto atoms = slab();
    atoms.pos_[0] += 2 * atoms.cell_.row(0);
    atoms.id = 42;
    atoms.features_["label"] = "finite slab";
    const auto before = snapshot(atoms);
    REQUIRE(Symmetry3D::trans_reduce_1d(atoms, 2) == 1);
    CHECK(snapshot(atoms) == before);
}

TEST_CASE("3D primitive reduction rejects mixed periodicity before any reduction", "[symmetry][reduction]")
{
    const int api = GENERATE(2, 3);
    auto atoms = slab();
    // This cell can reduce along x, so all_reduce must reject before that stage.
    atoms.pos_[0] += 2 * atoms.cell_.row(0);
    atoms.features_["label"] = "finite slab";
    require_unchanged_failure(api, atoms, 1e-5, "full periodicity");
    REQUIRE_THROWS_WITH(reduce(api, atoms), Catch::Matchers::ContainsSubstring("trans_reduce"));
}


TEST_CASE("Fractional translation reduction finds the same physical primitive cell", "[symmetry][reduction][fractional]")
{
    Atoms cartesian;
    cartesian.cell_ << 8, 0, 0, 0, 5, 0, 0, 0, 6;
    cartesian.pos_ = {{.5, 1.25, 2.25}, {4.5, 1.25, 2.25}};
    cartesian.chemmap_ = {1, 1};
    auto fractional = cartesian;
    fractional.to_fractional();
    REQUIRE(Symmetry3D::trans_reduce(cartesian) == 2);
    REQUIRE(Symmetry3D::trans_reduce(fractional) == 2);
    REQUIRE(cartesian.cartesian);
    REQUIRE_FALSE(fractional.cartesian);
    REQUIRE(fractional.size() == 1);
    CHECK(fractional.cell_.isApprox(cartesian.cell_, 1e-12));
    CHECK((fractional.pos_[0] * fractional.cell_).isApprox(cartesian.pos_[0], 1e-12));
}

TEST_CASE("Fractional standalone cell correction preserves physical pair geometry", "[symmetry][reduction][fractional]")
{
    Atoms fractional;
    fractional.cell_ << 4, 0, 0, 3, 2, 0, 0, 0, 6;
    fractional.pos_ = {{.1, .2, .3}, {.2, .35, .4}};
    fractional.chemmap_ = {1, 8};
    fractional.cartesian = false;
    auto cartesian = fractional;
    cartesian.to_cartesian();
    const auto original_cell = cartesian.cell_;
    REQUIRE(Symmetry3D::correct_cell(cartesian) == 0);
    REQUIRE(Symmetry3D::correct_cell(fractional) == 0);
    REQUIRE(cartesian.cartesian);
    REQUIRE_FALSE(fractional.cartesian);
    CHECK_FALSE(cartesian.cell_.isApprox(original_cell));
    CHECK(fractional.cell_.isApprox(cartesian.cell_, 1e-12));
    // Independent finite enumeration for this small fixture, without the MIC kernel.
    const Eigen::RowVector3d delta =
        (fractional.pos_[1] - fractional.pos_[0]) * fractional.cell_;
    double pair_distance = 1e10;
    for (int i = -4; i <= 4; ++i)
        for (int j = -4; j <= 4; ++j)
            for (int k = -4; k <= 4; ++k)
                pair_distance = std::min(pair_distance,
                    (delta - Eigen::RowVector3d(i, j, k) * fractional.cell_).norm());
    CHECK(pair_distance == Catch::Approx(std::sqrt(1.1725)).margin(1e-12));
    for (int i = 0; i < fractional.size(); ++i) {
        const Eigen::RowVector3d shift =
            (fractional.pos_[i] * fractional.cell_ - cartesian.pos_[i]) * cartesian.cell_.inverse();
        CHECK((shift - shift.array().round().matrix()).norm() < 1e-12);
    }
}

TEST_CASE("Fractional reduction wrappers retain representation through nested stages", "[symmetry][reduction][fractional]")
{
    const int api = GENERATE(0, 1, 2, 3);
    auto original = primitive(true, false, true);
    set_extensive_labels(original);
    auto cartesian = original;
    cartesian.mksuper(Eigen::Vector3i(2, 1, 1));
    auto fractional = cartesian;
    fractional.to_fractional();
    REQUIRE(reduce(api, cartesian) == 2);
    REQUIRE(reduce(api, fractional) == 2);
    REQUIRE(cartesian.cartesian);
    REQUIRE_FALSE(fractional.cartesian);
    CHECK(fractional.cell_.isApprox(cartesian.cell_, 1e-12));
    CHECK(fractional.chemmap_ == cartesian.chemmap_);
    require_extensive_labels(fractional, original);
    for (int i = 0; i < fractional.size(); ++i) {
        const Eigen::RowVector3d shift =
            (fractional.pos_[i] * fractional.cell_ - cartesian.pos_[i]) * cartesian.cell_.inverse();
        CHECK((shift - shift.array().round().matrix()).norm() < 1e-10);
    }
    // A second call exercises the no-reduction return after representation conversion.
    REQUIRE(reduce(api, fractional) == 1);
    REQUIRE_FALSE(fractional.cartesian);
    require_extensive_labels(fractional, original);
}

TEST_CASE("Fractional one-direction reduction respects finite layers and free-direction early return", "[symmetry][reduction][fractional]")
{
    auto cartesian = slab();
    auto fractional = cartesian;
    fractional.to_fractional();
    const auto before = snapshot(fractional);
    REQUIRE(Symmetry3D::trans_reduce_1d(fractional, 2) == 1);
    REQUIRE_FALSE(fractional.cartesian);
    CHECK(snapshot(fractional) == before);
    REQUIRE(Symmetry3D::trans_reduce_1d(cartesian, 0) == 2);
    REQUIRE(Symmetry3D::trans_reduce_1d(fractional, 0) == 2);
    REQUIRE_FALSE(fractional.cartesian);
    CHECK(fractional.pbc_ == cartesian.pbc_);
    CHECK(fractional.cell_.isApprox(cartesian.cell_, 1e-12));
    REQUIRE(fractional.size() == cartesian.size());
    for (int i = 0; i < fractional.size(); ++i)
        CHECK((fractional.pos_[i] * fractional.cell_).isApprox(cartesian.pos_[i], 1e-12));
}
