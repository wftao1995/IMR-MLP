// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/trajectory_transform.h"
#include "apps/apps.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>

using Catch::Approx;
namespace {
struct Workspace {
    std::filesystem::path path;
    Workspace() {
        char name[] = "/tmp/imrmlp_trjavg_XXXXXX";
        const char *dir = mkdtemp(name);
        if (!dir) throw std::runtime_error("Cannot create trajectory average test directory");
        path = dir;
        std::ofstream(path / "index.map") << "Al\n";
    }
    ~Workspace() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::remove_all(path);
    }
    int run() {
        CMDOption::global_cmdopts.reset();
        std::vector<std::string> args = {"calc_trjavg", "--input-file=" + (path / "in.xyz").string(),
            "--output-file=" + (path / "out.bin").string(), "--input-format=extxyz", "--output-format=bin",
            "--index=" + (path / "index.map").string(), "--overwrite", "--wrap=false"};
        std::vector<char *> argv;
        for (auto &s : args) argv.push_back(s.data());
        return calc_trjavg(argv.size(), argv.data());
    }
};
Atoms labeled_frame() {
    Atoms a;
    a.cell_ = 20 * Eigen::Matrix3d::Identity();
    a.pos_ = {{1, 2, 3}, {3, 2, 3}};
    a.chemmap_ = {0, 0};
    a.pbc_ = {false, false, false};
    a.forces_ = {{1, 2, 3}, {4, 5, 6}};
    a.velocities_ = a.forces_;
    a.energy_ = 7; a.have_energy = true;
    a.fenergy_ = 6; a.have_fenergy = true;
    a.site_energies_ = {3, 4};
    a.stress_.setConstant(13); a.have_stress = true;
    a.hessian_ = Eigen::MatrixXd::Identity(6, 6); a.have_hessian = true;
    a.hvp_directions_ = Eigen::MatrixXd::Ones(6, 1);
    a.hvp_targets_ = a.hvp_directions_;
    a.features_["origin"] = "first_frame";
    return a;
}
void require_geometry_only(const Atoms &a) {
    REQUIRE_FALSE(a.has_energy());
    REQUIRE_FALSE(a.has_fenergy());
    REQUIRE_FALSE(a.has_stresses());
    REQUIRE_FALSE(a.have_hessian);
    REQUIRE(a.hessian_.size() == 0);
    REQUIRE(a.hvp_directions_.size() == 0);
    REQUIRE(a.hvp_targets_.size() == 0);
    REQUIRE(a.forces_.empty());
    REQUIRE(a.velocities_.empty());
    REQUIRE(a.site_energies_.empty());
    REQUIRE(a.features_.empty());
}
}

TEST_CASE("Fixed-cell centroid returns geometry without frame labels", "[trajectory][centroid]") {
    auto a = labeled_frame(), b = a;
    b.pos_[0][0] += .2;
    b.pos_[1][0] += .8;
    const auto centroid = fixed_cell_centroid({a, b}, {1, 1});
    REQUIRE(centroid.pos_[0][0] == Approx(.85));
    REQUIRE(centroid.pos_[1][0] == Approx(3.15));
    REQUIRE(centroid.cell_.isApprox(a.cell_));
    REQUIRE(centroid.chemmap_ == a.chemmap_);
    REQUIRE(centroid.pbc_ == a.pbc_);
    require_geometry_only(centroid);
}

TEST_CASE("Trajectory average outputs only geometry for fixed and variable cells", "[trajectory][command]") {
    const bool variable_cell = GENERATE(false, true);
    Workspace work;
    auto a = labeled_frame(), b = a;
    if (variable_cell) b.cell_ *= 1.1;
    b.pos_[0][0] += .2;
    b.pos_[1][0] += .8;
    GPIO io;
    REQUIRE(io.readmap((work.path / "index.map").string()));
    std::vector<Atoms> frames{a, b};
    REQUIRE(io.write(frames, (work.path / "in.xyz").string(), "extxyz"));
    Atoms input;
    REQUIRE(io.read_one(input, (work.path / "in.xyz").string(), "extxyz"));
    REQUIRE(input.has_hessian());
    REQUIRE(input.hessian_.isApprox(a.hessian_));
    REQUIRE(work.run() == 0);
    Atoms average;
    REQUIRE(io.read_one(average, (work.path / "out.bin").string(), "bin"));
    REQUIRE(average.size() == 2);
    REQUIRE(average.chemmap_ == a.chemmap_);
    REQUIRE(average.pbc_ == a.pbc_);
    REQUIRE(average.cell_.isApprox((a.cell_ + b.cell_) * .5));
    require_geometry_only(average);
}

TEST_CASE("Variable-cell trajectory averaging preserves nonperiodic displacements", "[trajectory][command]") {
    Workspace work;
    auto a = labeled_frame(), b = a;
    b.cell_ *= 1.1;
    for (auto &p : b.pos_) p[0] += 15.2;
    GPIO io;
    REQUIRE(io.readmap((work.path / "index.map").string()));
    std::vector<Atoms> frames{a, b};
    REQUIRE(io.write(frames, (work.path / "in.xyz").string(), "extxyz"));
    REQUIRE(work.run() == 0);
    Atoms average;
    REQUIRE(io.read_one(average, (work.path / "out.bin").string(), "bin"));
    REQUIRE(average.pos_[0][0] == Approx((1.0 / 20 + 16.2 / 22) * .5 * 21));
}

TEST_CASE("Trajectory average returns failure when the output cannot be written", "[trajectory][command]") {
    const bool full = GENERATE(false, true);
    Workspace work;
    GPIO io;
    REQUIRE(io.readmap((work.path / "index.map").string()));
    std::vector<Atoms> frames{labeled_frame()};
    REQUIRE(io.write(frames, (work.path / "in.xyz").string(), "extxyz"));
    if (full) std::filesystem::create_symlink("/dev/full", work.path / "out.bin");
    else std::filesystem::create_directory(work.path / "out.bin");
    REQUIRE(work.run() != 0);
}
