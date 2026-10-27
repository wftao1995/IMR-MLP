#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp.h"
#include <Eigen/Geometry>

using namespace Potentials;

namespace {
Atoms tetrahedra(double cell_length) {
    Atoms a;
    a.cell_ = cell_length * Eigen::Matrix3d::Identity();
    a.pbc_ = {true, true, true};
    const Eigen::RowVector3d center = Eigen::RowVector3d::Constant(cell_length / 2.);
    for (int species = 0; species < 2; ++species) {
        const double radius = species ? 1.19 : .63;
        for (const auto &v : std::vector<Eigen::RowVector3d>{{1., 1., 1.}, {1., -1., -1.}, {-1., 1., -1.}, {-1., -1., 1.}}) {
            a.pos_.push_back(center + radius * v);
            a.chemmap_.push_back(species);
        }
    }
    return a;
}
void equivalent(const Atoms &a, const Atoms &b) {
    REQUIRE(a.energy_ == Catch::Approx(b.energy_).margin(2e-10));
    REQUIRE((a.stress_ - b.stress_).norm() < 2e-9);
    REQUIRE(a.site_energies_.size() == b.site_energies_.size());
    for (int i = 0; i < a.size(); ++i) {
        REQUIRE((a.forces_[i] - b.forces_[i]).norm() < 2e-9);
        REQUIRE(a.site_energies_[i] == Catch::Approx(b.site_energies_[i]).margin(2e-10));
    }
}
}

TEST_CASE("MTP full symmetry reuses rotated site orbits and periodic images",
          "[mtp][symmetry-reuse]") {
    PotMTP p("unitest/test-data/portable-mtp/les.mtp");
    REQUIRE_FALSE(p.full_symmetry_reuse);
    for (double length : {12., 3.8}) {
        auto input = tetrahedra(length);
        const Eigen::Matrix3d rotation = Eigen::AngleAxisd(.37, Eigen::Vector3d(1., 2., 3.).normalized()).toRotationMatrix();
        input.cell_ = input.cell_ * rotation;
        for (auto &r : input.pos_) r = r * rotation;
        p.full_symmetry_reuse = false;
        auto reference = input;
        p.getEFS(reference);
        p.full_symmetry_reuse = true;
        auto actual = input;
        p.getEFSOMP(actual);
        equivalent(actual, reference);
        REQUIRE(p.last_symmetry_site_evaluations == 2);
        p.getEFSSymmOMP(actual);
        equivalent(actual, reference);
        REQUIRE(p.last_symmetry_site_evaluations == 2);

        p.add_efs = true;
        p.getEFS(actual);
        reference.energy_ *= 2.;
        reference.stress_ *= 2.;
        for (auto &f : reference.forces_) f *= 2.;
        for (auto &e : reference.site_energies_) e *= 2.;
        equivalent(actual, reference);
        p.add_efs = false;
    }
}

TEST_CASE("MTP full symmetry falls back for inexact or nonperiodic maps",
          "[mtp][symmetry-reuse]") {
    PotMTP p("unitest/test-data/portable-mtp/les.mtp");
    auto input = tetrahedra(12.);
    SECTION("approximately symmetric geometry") { input.pos_[0][0] += 1e-5; }
    SECTION("open direction") { input.pbc_[2] = false; }
    auto reference = input;
    p.getEFS(reference);
    p.full_symmetry_reuse = true;
    auto actual = input;
    // A deliberately generous discovery tolerance must not modify geometry.
    p.getEFSSymmOMP(actual, .01);
    equivalent(actual, reference);
    REQUIRE(p.last_symmetry_site_evaluations == input.size());
}
