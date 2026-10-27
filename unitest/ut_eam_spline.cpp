// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_eamalloy_spline.h"
#include "potentials/potential_eamfs_spline.h"
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace Potentials;
using Catch::Approx;

namespace {
void write_pairs(const std::string &file, bool fs) {
    std::ofstream out(file);
    out << "audit\naudit\naudit\n3 H He Li\n5 1 5 1 4\n";
    for (int i = 0; i < 3; ++i) {
        out << i + 1 << " 1 1 fcc\n0 0 0 0 0\n";
        for (int j = 0; j < (fs ? 3 : 1); ++j) out << "0 0 0 0 0\n";
    }
    for (int k = 1; k <= 6; ++k) {
        for (int r = 0; r < 5; ++r) out << k * r * r << " ";
        out << "\n";
    }
}

std::unique_ptr<AnyPotential> load(bool fs, std::string file, std::map<std::string, int> map = {}) {
    if (fs) return std::make_unique<EAMFsSpline>(file, map);
    return std::make_unique<EAMAlloySpline>(file, map);
}

Atoms dimer(int i = 0, int j = 0) {
    Atoms atoms;
    atoms.cell_ = Eigen::Matrix3d::Identity() * 20;
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{0, 0, 0}, {2, 0, 0}};
    atoms.chemmap_ = {i, j};
    return atoms;
}

void evaluate(AnyPotential &pot, Atoms &atoms, bool omp) {
    Neighborhoods nbhs(atoms, pot.cutoff);
    if (omp) pot.getEFSOMP(atoms, nbhs);
    else pot.getEFS(atoms, nbhs);
}
}

TEST_CASE("EAM tables preserve pair identities under species remapping", "[eam][io]") {
    const bool fs = GENERATE(false, true);
    const bool omp = GENERATE(false, true);
    const int mapping = GENERATE(0, 1, 2);
    write_pairs("three-species.eam", fs);
    std::map<std::string, int> map;
    if (mapping == 1) map = {{"H", 2}, {"He", 1}, {"Li", 0}};
    if (mapping == 2) map = {{"H", 1}, {"Li", 0}};
    auto pot = load(fs, "three-species.eam", map);
    REQUIRE(pot->species_count == (mapping == 2 ? 2 : 3));
    const std::vector<std::string> names{"H", "He", "Li"};
    const double slopes[3][3] = {{1, 2, 4}, {2, 3, 5}, {4, 5, 6}};
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        if (!map.empty() && (!map.count(names[i]) || !map.count(names[j]))) continue;
        auto atoms = dimer(map.empty() ? i : map.at(names[i]), map.empty() ? j : map.at(names[j]));
        evaluate(*pot, atoms, omp);
        CAPTURE(fs, omp, mapping, i, j);
        REQUIRE(atoms.energy_ == Approx(2 * slopes[i][j]));
        REQUIRE(atoms.forces_[0](0) == Approx(slopes[i][j]));
        REQUIRE(atoms.forces_[1](0) == Approx(-slopes[i][j]));
        REQUIRE(atoms.stress_(0, 0) == Approx(-2 * slopes[i][j]));
        REQUIRE(atoms.site_energies_[0] == Approx(slopes[i][j]));
        pot->add_efs = true;
        evaluate(*pot, atoms, omp);
        REQUIRE(atoms.energy_ == Approx(4 * slopes[i][j]));
        REQUIRE(atoms.forces_[0](0) == Approx(2 * slopes[i][j]));
        REQUIRE(atoms.site_energies_[0] == Approx(2 * slopes[i][j]));
        REQUIRE(atoms.stress_(0, 0) == Approx(-4 * slopes[i][j]));
        pot->add_efs = false;
    }
}

TEST_CASE("EAM FS density rows distinguish hosts from neighbors", "[eam]") {
    const bool omp = GENERATE(false, true);
    const bool reverse = GENERATE(false, true);
    {
        std::ofstream out("asymmetric.fs");
        out << std::setprecision(17) << "a\nb\nc\n2 H He\n5 2 5 1 4\n";
        const double density[2][2] = {{0.2, 0.7}, {0.4, 0.3}}; // [host][neighbor]
        for (int neighbor = 0; neighbor < 2; ++neighbor) {
            out << neighbor + 1 << " 1 1 fcc\n";
            for (int rho = 0; rho <= 8; rho += 2) out << (neighbor + 1) * rho << " ";
            out << "\n";
            for (int host = 0; host < 2; ++host) {
                for (int r = 0; r < 5; ++r) out << density[host][neighbor] * r << " ";
                out << "\n";
            }
        }
        out << "0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0\n";
    }
    auto pot = load(true, "asymmetric.fs", reverse ? std::map<std::string, int>{{"H", 1}, {"He", 0}}
                                                   : std::map<std::string, int>{});
    auto atoms = dimer(reverse ? 1 : 0, reverse ? 0 : 1);
    evaluate(*pot, atoms, omp);
    REQUIRE(atoms.energy_ == Approx(3.0));
    REQUIRE(atoms.forces_[0](0) == Approx(1.5));
    REQUIRE(atoms.stress_(0, 0) == Approx(-3.0));
}

TEST_CASE("EAM HE embedding grid uses its declared density origin", "[eam][he]") {
    const bool fs = GENERATE(false, true);
    const bool omp = GENERATE(false, true);
    const double rho = GENERATE(-1.0, 0.0, 1.0, 2.5);
    {
        std::ofstream out("negative-density.eam");
        out << "a\nb\nc\n1 H\n5 1 5 1 4 2\n1 1 1 fcc\n4 1 0 1 4\n";
        for (int i = 0; i < 5; ++i) out << rho << " ";
        out << "\n0 0 0 0 0\n";
    }
    auto pot = load(fs, "negative-density.eam");
    auto atoms = dimer();
    evaluate(*pot, atoms, omp);
    // Above the last node use the endpoint slope (4 - 1) / 1 = 3.
    const double embedding = rho <= 2 ? rho * rho : 4 + 3 * (rho - 2);
    REQUIRE(atoms.energy_ == Approx(2 * embedding));
    REQUIRE(atoms.forces_[0].norm() == Approx(0.0).margin(1e-12));
}

TEST_CASE("EAM loading rejects missing malformed and incomplete tables", "[eam][io]") {
    const bool fs = GENERATE(false, true);
    REQUIRE_THROWS_WITH(load(fs, "nonexistent-directory/potential.eam"), Catch::Matchers::ContainsSubstring("potential.eam"));
    const std::vector<std::string> invalid{
        "", "a\nb\nc\n0\n", "a\nb\nc\n2 H H\n",
        "a\nb\nc\n1 H\n2 1 5 1 4\n",
        "a\nb\nc\n1 H\n5 0 5 1 4\n",
        "a\nb\nc\n1 H\n5 1 5 1 4 2 garbage\n",
        "a\nb\nc\n1 H\n5 1 5 1 4\n1 1 1 fcc\n0 0 0 0 0\n",
        "a\nb\nc\n1 H\n5 1 5 1 4\n1 1 1 fcc\n0 0 nan 0 0\n"
    };
    for (const auto &contents : invalid) {
        { std::ofstream out("invalid.eam"); out << contents; }
        CAPTURE(contents);
        REQUIRE_THROWS_WITH(load(fs, "invalid.eam"), Catch::Matchers::ContainsSubstring("invalid.eam"));
    }
    write_pairs("map-validation.eam", fs);
    for (const auto &map : std::vector<std::map<std::string, int>>{
            {{"H", 0}, {"He", 0}}, {{"H", 2}}, {{"O", 0}}, {{"H", -1}}}) {
        REQUIRE_THROWS_WITH(load(fs, "map-validation.eam", map), Catch::Matchers::ContainsSubstring("species map"));
    }
}
