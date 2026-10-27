#include <catch2/catch_all.hpp>
#include "base/structures.h"

namespace {
void apply_wrap(Atoms &atom, int method)
{
    if (method == 0) atom.wrap();
    if (method == 1) atom.wrap(atom.pos_[0]);
    if (method == 2) atom.hwrap();
}
}

TEST_CASE("Wrapping a 2D zero-open-row cell preserves its perpendicular coordinate", "[atoms][wrap][pbc]")
{
    const int method = GENERATE(0, 1, 2);
    const int first = GENERATE(0, 1, 2);
    Atoms atom;
    atom.cell_.setZero();
    atom.cell_.row(first) << 4, 0, 0;
    atom.cell_.row((first + 1) % 3) << 1, 3, 0;
    atom.pbc_ = {false, false, false};
    atom.pbc_[first] = atom.pbc_[(first + 1) % 3] = true;
    atom.pos_ = {{.5, 3.9, 7}}; atom.chemmap_ = {0};
    const Eigen::RowVector3d expected(3.5, .9, 7);
    apply_wrap(atom, method);
    REQUIRE(atom.pos_[0].allFinite());
    REQUIRE(atom.pos_[0].isApprox(expected));
    apply_wrap(atom, method);
    REQUIRE(atom.pos_[0].isApprox(expected));
}

TEST_CASE("Wrapping a 1D skew cell keeps both perpendicular components", "[atoms][wrap][pbc]")
{
    const int method = GENERATE(0, 1, 2);
    Atoms atom; atom.cell_.setZero(); atom.cell_.row(1) << 2, 2, 0;
    atom.pbc_ = {false, true, false};
    atom.pos_ = {{.6, -5.4, 5}}; atom.chemmap_ = {0};
    apply_wrap(atom, method);
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(4.6, -1.4, 5)));
}

TEST_CASE("Full skew cells keep wrapping in their original fractional convention", "[atoms][wrap][pbc]")
{
    const int method = GENERATE(0, 1, 2);
    Atoms atom; atom.cell_ << 4, 0, 0, 1, 3, 0, 2, 1, 5;
    atom.pbc_ = {true, true, false};
    atom.pos_ = {{4.1, 5.7, 9}}; atom.chemmap_ = {0};
    // Fractional [-.2, 1.3, 1.8], retaining the entire open lattice component.
    apply_wrap(atom, method);
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(7.1, 2.7, 9)));
}

TEST_CASE("Wrapping rejects dependent periodic rows and leaves nonperiodic zero cells alone", "[atoms][wrap][pbc]")
{
    const int method = GENERATE(0, 1, 2);
    Atoms atom; atom.cell_.setZero(); atom.pos_ = {{12, -1, 3}}; atom.chemmap_ = {0};
    atom.pbc_ = {false, false, false};
    apply_wrap(atom, method);
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(12, -1, 3)));
    atom.pbc_ = {true, true, false};
    atom.cell_.row(0) << 2, 0, 0; atom.cell_.row(1) << 4, 0, 0;
    REQUIRE_THROWS_WITH(apply_wrap(atom, method), Catch::Matchers::ContainsSubstring("independent periodic"));
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(12, -1, 3)));
}
