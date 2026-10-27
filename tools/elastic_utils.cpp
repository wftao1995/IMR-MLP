// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "tools/elastic_utils.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ElasticUtils {
namespace {

constexpr std::array<std::array<int, 2>, 6> pairs{{
    {{0, 0}}, {{1, 1}}, {{2, 2}}, {{1, 2}}, {{0, 2}}, {{0, 1}}
}};

int voigt_index(int a, int b)
{
    if (a == b) return a;
    if ((a == 1 && b == 2) || (a == 2 && b == 1)) return 3;
    if ((a == 0 && b == 2) || (a == 2 && b == 0)) return 4;
    return 5;
}

} // namespace

Eigen::Matrix<double, 9, 9> voigt_to_tensor(const Eigen::Matrix<double, 6, 6> &voigt)
{
    Eigen::Matrix<double, 9, 9> tensor;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 3; ++c)
                for (int d = 0; d < 3; ++d)
                    tensor(3 * a + b, 3 * c + d) = voigt(voigt_index(a, b), voigt_index(c, d));
    return tensor;
}

Eigen::Matrix<double, 6, 6> tensor_to_voigt(const Eigen::Matrix<double, 9, 9> &tensor)
{
    Eigen::Matrix<double, 6, 6> voigt;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            voigt(i, j) = tensor(3 * pairs[i][0] + pairs[i][1],
                                  3 * pairs[j][0] + pairs[j][1]);
    return 0.5 * (voigt + voigt.transpose()).eval();
}

Eigen::Matrix<double, 6, 6> project_by_symmetry(
    const Eigen::Matrix<double, 6, 6> &stiffness,
    const Eigen::Matrix3d &cell,
    const std::vector<Eigen::Matrix3d> &fractional_rotations)
{
    if (fractional_rotations.empty())
        throw std::invalid_argument("Cannot project elastic tensor without symmetry rotations");
    const auto tensor = voigt_to_tensor(stiffness);
    Eigen::Matrix<double, 9, 9> projected = Eigen::Matrix<double, 9, 9>::Zero();
    for (const auto &fractional : fractional_rotations) {
        const Eigen::Matrix3d row_rotation = cell.inverse() * fractional * cell;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int c = 0; c < 3; ++c)
                    for (int d = 0; d < 3; ++d) {
                        double value = 0.0;
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j)
                                for (int k = 0; k < 3; ++k)
                                    for (int l = 0; l < 3; ++l)
                                        value += row_rotation(i, a) * row_rotation(j, b)
                                               * row_rotation(k, c) * row_rotation(l, d)
                                               * tensor(3 * i + j, 3 * k + l);
                        projected(3 * a + b, 3 * c + d) += value;
                    }
    }
    projected /= static_cast<double>(fractional_rotations.size());
    projected = 0.5 * (projected + projected.transpose()).eval();
    return tensor_to_voigt(projected);
}

Eigen::Matrix<double, 6, 6> born_stiffness(
    const Eigen::Matrix<double, 6, 6> &material_stiffness,
    double pressure)
{
    Eigen::Matrix<double, 6, 6> born = material_stiffness;
    for (int i = 0; i < 6; ++i)
        born(i, i) -= pressure;
    born(0, 1) += pressure; born(1, 0) += pressure;
    born(0, 2) += pressure; born(2, 0) += pressure;
    born(1, 2) += pressure; born(2, 1) += pressure;
    return born;
}

Eigen::Matrix<double, 6, 6> compliance(
    const Eigen::Matrix<double, 6, 6> &stiffness)
{
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(
        stiffness, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto singular = svd.singularValues();
    if (singular(0) == 0.0
        || singular(5) <= std::numeric_limits<double>::epsilon() * singular(0))
        throw std::runtime_error("Material stiffness is singular; compliance is undefined");
    return svd.solve(Eigen::Matrix<double, 6, 6>::Identity());
}

} // namespace ElasticUtils
