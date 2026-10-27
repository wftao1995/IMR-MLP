#include <catch2/catch_all.hpp>
#include "analyse/trajectory_transform.h"

namespace {
Atoms frame(const Eigen::Matrix3d &cell, std::array<bool, 3> pbc,
            const Eigen::RowVector3d &first, const Eigen::RowVector3d &second) {
    Atoms result;
    result.cell_ = cell;
    result.pbc_ = pbc;
    result.pos_ = {first, second};
    result.chemmap_ = {1, 1};
    return result;
}
void check_unwrap(const std::vector<Atoms> &input, const std::vector<Atoms> &expected,
                  bool reference) {
    std::vector<Atoms> actual = input;
    int offset = 0;
    if (reference) {
        actual.erase(actual.begin());
        unwarp(input.front(), actual);
        offset = 1;
    } else unwarp(actual);
    REQUIRE(actual.size() == expected.size() - offset);
    for (int i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i].cell_ == expected[i + offset].cell_);
        for (int j = 0; j < actual[i].size(); ++j) {
            REQUIRE(actual[i].pos_[j].allFinite());
            REQUIRE(actual[i].pos_[j].isApprox(expected[i + offset].pos_[j], 1e-12));
        }
    }
}
}

TEST_CASE("Unwrap skew 2D cells uses periodic nonprefix rows and preserves open displacement", "[trajectory][unwrap][partialcell]") {
    const auto reference = GENERATE(false, true);
    const auto full_rank_padding = GENERATE(false, true);
    const Eigen::RowVector3d a(4, 0, 1), b(1, 3, 0), normal(-3, 1, 12);
    Eigen::Matrix3d cell = Eigen::Matrix3d::Zero();
    cell.row(0) = a;
    cell.row(2) = b;
    if (full_rank_padding) cell.row(1) = normal;
    const std::array<bool, 3> pbc = {true, false, true};
    std::vector<Atoms> input, expected;
    for (int i = 0; i < 3; ++i) {
        const double wrapped = i == 0 ? .45 : -.55 + .1 * i;
        const double unwrapped = .45 + .1 * i;
        input.push_back(frame(cell, pbc, wrapped * (a + b) + i * normal,
                              -wrapped * (a + b) - 2 * i * normal));
        expected.push_back(frame(cell, pbc, unwrapped * (a + b) + i * normal,
                                 -unwrapped * (a + b) - 2 * i * normal));
    }
    check_unwrap(input, expected, reference);
}

TEST_CASE("Unwrap skew 1D changing cell keeps integer winding in the current cell", "[trajectory][unwrap][partialcell]") {
    const auto reference = GENERATE(false, true);
    const Eigen::RowVector3d normal(-1, 4, 7);
    const std::array<bool, 3> pbc = {false, false, true};
    std::vector<Atoms> input, expected;
    for (int i = 0; i < 3; ++i) {
        Eigen::Matrix3d cell = Eigen::Matrix3d::Zero();
        const Eigen::RowVector3d a(4 + i, 1 + .25 * i, 0);
        cell.row(2) = a;
        const double wrapped = i == 0 ? .45 : -.55 + .1 * i;
        const double unwrapped = .45 + .1 * i;
        input.push_back(frame(cell, pbc, wrapped * a + i * normal,
                              -wrapped * a - 2 * i * normal));
        expected.push_back(frame(cell, pbc, unwrapped * a + i * normal,
                                 -unwrapped * a - 2 * i * normal));
    }
    check_unwrap(input, expected, reference);
}

TEST_CASE("Unwrap nonperiodic zero cells leaves Cartesian positions unchanged", "[trajectory][unwrap][partialcell]") {
    const auto reference = GENERATE(false, true);
    std::vector<Atoms> input;
    for (int i = 0; i < 3; ++i)
        input.push_back(frame(Eigen::Matrix3d::Zero(), {false, false, false},
                              Eigen::RowVector3d(i * 17, -3 * i, 21 * i),
                              Eigen::RowVector3d(-4 * i, 50 * i, -i)));
    check_unwrap(input, input, reference);
}

TEST_CASE("Unwrap rejects dependent periodic cell rows in either overload", "[trajectory][unwrap][partialcell]") {
    const auto reference = GENERATE(false, true);
    const auto zero_vector = GENERATE(false, true);
    Eigen::Matrix3d cell = Eigen::Matrix3d::Zero();
    if (!zero_vector) {
        cell.row(0) << 4, 0, 1;
        cell.row(2) << 8, 0, 2;
    }
    Atoms first = frame(cell, {true, false, true}, Eigen::RowVector3d::Zero(),
                        Eigen::RowVector3d::Ones());
    std::vector<Atoms> input{first, first};
    if (reference) REQUIRE_THROWS_AS(unwarp(first, input), std::invalid_argument);
    else REQUIRE_THROWS_AS(unwarp(input), std::invalid_argument);
}
