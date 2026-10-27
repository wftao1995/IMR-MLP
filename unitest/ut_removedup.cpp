// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "base/structures.h"

using Catch::Approx;

namespace {

Atoms make_line_atoms(const std::vector<double> &xs, int type = 0)
{
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 10.0;
    atom.chemmap_.assign(xs.size(), type);
    for (double x : xs) {
        atom.pos_.push_back({x, 0.0, 0.0});
    }
    return atom;
}

} // namespace

TEST_CASE("remove_duplicate_atoms deletes exact same-type overlap", "[removedup]")
{
    auto atom = make_line_atoms({0.0, 0.0, 3.0});

    auto stats = atom.remove_duplicate_atoms(0.2, Atoms::DuplicateAtomMode::Delete);

    REQUIRE(stats.initial_atoms == 3);
    REQUIRE(stats.final_atoms == 2);
    REQUIRE(stats.removed_atoms == 1);
    REQUIRE(stats.duplicate_groups == 1);
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.pos_[0](0) == Approx(0.0));
    REQUIRE(atom.pos_[1](0) == Approx(3.0));
}

TEST_CASE("remove_duplicate_atoms keeps representative nearest component mean", "[removedup]")
{
    auto atom = make_line_atoms({0.0, 0.1, 0.2});

    auto stats = atom.remove_duplicate_atoms(0.15, Atoms::DuplicateAtomMode::Delete);

    REQUIRE(stats.removed_atoms == 2);
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.pos_[0](0) == Approx(0.1));
}

TEST_CASE("remove_duplicate_atoms merge mode moves representative to mean", "[removedup]")
{
    auto atom = make_line_atoms({0.0, 0.1, 0.4});

    auto stats = atom.remove_duplicate_atoms(0.45, Atoms::DuplicateAtomMode::Merge);

    REQUIRE(stats.removed_atoms == 2);
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.pos_[0](0) == Approx((0.0 + 0.1 + 0.4) / 3.0));
}

TEST_CASE("remove_duplicate_atoms does not merge different atom types", "[removedup]")
{
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 10.0;
    atom.pos_ = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    atom.chemmap_ = {0, 1};

    auto stats = atom.remove_duplicate_atoms(0.2, Atoms::DuplicateAtomMode::Delete);

    REQUIRE(stats.removed_atoms == 0);
    REQUIRE(atom.size() == 2);
}

TEST_CASE("remove_duplicate_atoms merges periodic-boundary duplicates", "[removedup]")
{
    auto atom = make_line_atoms({0.1, 9.95});

    auto stats = atom.remove_duplicate_atoms(0.3, Atoms::DuplicateAtomMode::Merge);

    REQUIRE(stats.removed_atoms == 1);
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.pos_[0](0) == Approx(0.025));
}

TEST_CASE("remove_duplicate_atoms compacts per-atom properties", "[removedup]")
{
    auto atom = make_line_atoms({0.0, 0.0, 2.0});
    atom.ids_ = {10, 11, 12};
    atom.forces_ = {{1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}};
    atom.local_attribs_d_["weight"].resize(3);
    atom.local_attribs_d_["weight"] << 1.0, 2.0, 3.0;
    atom.local_attribs_v_["desc"].resize(2, 3);
    atom.local_attribs_v_["desc"] << 1.0, 2.0, 3.0,
                                      4.0, 5.0, 6.0;

    auto stats = atom.remove_duplicate_atoms(0.2, Atoms::DuplicateAtomMode::Delete);

    REQUIRE(stats.removed_atoms == 1);
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.ids_ == std::vector<int>{10, 12});
    REQUIRE(atom.forces_.size() == 2);
    REQUIRE(atom.local_attribs_d_.at("weight").size() == 2);
    REQUIRE(atom.local_attribs_d_.at("weight")(0) == Approx(1.0));
    REQUIRE(atom.local_attribs_d_.at("weight")(1) == Approx(3.0));
    REQUIRE(atom.local_attribs_v_.at("desc").cols() == 2);
    REQUIRE(atom.local_attribs_v_.at("desc")(0, 0) == Approx(1.0));
    REQUIRE(atom.local_attribs_v_.at("desc")(0, 1) == Approx(3.0));
}
