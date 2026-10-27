#include <catch2/catch_all.hpp>
#include "symmetry/intersitials.h"
#include <Eigen/Geometry>
#include <limits>

namespace {
std::vector<Eigen::RowVector3d> sites(Atoms &atom, bool vertices, bool edges, bool faces)
{
    std::vector<Eigen::RowVector3d> result;
    get_intersitial_sites(atom, result, {{0, 0.3}}, vertices, edges, faces,
                         0.6, 1.0, 0.1, 0.1);
    return result;
}
double periodic_error(Eigen::RowVector3d difference)
{
    // Comparison is modulo lattice translations, not a MIC distance oracle.
    for (int d = 0; d < 3; ++d) difference[d] -= std::round(difference[d]);
    return difference.norm();
}
Atoms host(const Eigen::Matrix3d &cell)
{
    Atoms atom;
    atom.cell_ = cell;
    atom.pbc_ = {true, true, true};
    atom.cartesian = true;
    atom.pos_ = {Eigen::RowVector3d(0.13, 0.21, 0.17) * cell};
    atom.chemmap_ = {0};
    atom.energy_ = 7.25;
    atom.forces_ = {Eigen::RowVector3d(0.2, 0.4, 0.7)};
    return atom;
}
void unchanged(const Atoms &atom, const Atoms &before)
{
    REQUIRE((atom.cell_-before.cell_).norm() == 0.0);
    REQUIRE(atom.pbc_ == before.pbc_);
    REQUIRE(atom.cartesian == before.cartesian);
    REQUIRE(atom.chemmap_ == before.chemmap_);
    REQUIRE((atom.pos_[0]-before.pos_[0]).norm() == 0.0);
    REQUIRE(atom.energy_ == before.energy_);
    REQUIRE((atom.forces_[0]-before.forces_[0]).norm() == 0.0);
}
}

TEST_CASE("Full-cell interstitial sites return in the original Cartesian frame", "[interstitial][frame]")
{
    // All nine rotation entries are nonzero: tests cannot pass by remapping
    // only an xy plane or assuming a diagonal original cell.
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.73, Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
    REQUIRE(rotation.determinant() == Catch::Approx(1.0));
    REQUIRE(rotation.cwiseAbs().minCoeff() > 0.01);
    const int mode = GENERATE(0, 1, 2, 3);
    const bool skew = GENERATE(false, true);
    const bool vertices = mode == 0 || mode == 3;
    const bool edges = mode == 1 || mode == 3;
    const bool faces = mode == 2 || mode == 3;
    Eigen::Matrix3d cell = 4.0 * Eigen::Matrix3d::Identity();
    if (skew) cell << 4,0,0, 0.7,4.4,0, 0.3,0.6,4.8;
    Atoms base = host(cell), rotated = host(cell * rotation);
    const Atoms base_before = base, rotated_before = rotated;
    auto reference = sites(base, vertices, edges, faces);
    auto actual = sites(rotated, vertices, edges, faces);
    unchanged(base, base_before);
    unchanged(rotated, rotated_before);
    REQUIRE_FALSE(reference.empty());
    REQUIRE(actual.size() == reference.size());
    const Eigen::Matrix3d inverse = rotated.cell_.inverse();
    for (const auto &expected : reference) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto &candidate : actual)
            best = std::min(best, periodic_error(((candidate - expected * rotation) * inverse).eval()));
        REQUIRE(best < 1e-8);
    }
    if (!skew) {
        // Independent one-host cubic Voronoi geometry, including translation.
        // Vertex .5,.5,.5; edge centres permute .5,.5,0; face centres .5,0,0.
        std::vector<Eigen::RowVector3d> expected;
        if (vertices) expected.emplace_back(0.5,0.5,0.5);
        if (edges) {
            expected.emplace_back(0,0.5,0.5); expected.emplace_back(0.5,0,0.5); expected.emplace_back(0.5,0.5,0);
        }
        if (faces) {
            expected.emplace_back(0.5,0,0); expected.emplace_back(0,0.5,0); expected.emplace_back(0,0,0.5);
        }
        REQUIRE(actual.size() == expected.size());
        for (auto delta : expected) {
            const Eigen::RowVector3d fractional = Eigen::RowVector3d(0.13,0.21,0.17) + delta;
            double best = std::numeric_limits<double>::infinity();
            for (const auto &candidate : actual)
                best = std::min(best, periodic_error((candidate * inverse - fractional).eval()));
            REQUIRE(best < 1e-8);
        }
    }
}
