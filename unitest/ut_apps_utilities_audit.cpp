// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>

#define main trj_rmdrift_audit_main
#include "apps/trj_rmdrift.cpp"
#undef main

namespace {
int run(std::vector<std::string> args) {
    CMDOption::global_cmdopts.reset();
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    return command_maps.at(args[0]).first(argv.size(), argv.data());
}

struct Directory {
    std::filesystem::path previous = std::filesystem::current_path();
    std::filesystem::path path;
    Directory() {
        char pattern[] = "/tmp/imrmlp_apps_utilities_XXXXXX";
        const char *directory = mkdtemp(pattern);
        if (!directory) throw std::runtime_error("Cannot create audit directory");
        path = directory;
        std::filesystem::current_path(path);
    }
    ~Directory() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
};
}

TEST_CASE("Trajectory drift uses atomic masses rather than neighbor radii", "[apps][trajectory][audit]") {
    Directory directory;
    // Internal motion with exactly zero mass-weighted drift: 1*0.5 + 4*(-0.125)=0.
    std::ofstream("index.map") << "H 0.4 1\nHe 0.4 4\n";
    Atoms first;
    first.cell_ = 20 * Eigen::Matrix3d::Identity();
    first.pbc_ = {false, false, false};
    first.pos_ = {{1, 2, 3}, {3, 2, 3}};
    first.chemmap_ = {0, 1};
    Atoms second = first;
    second.pos_[0][0] += .5;
    second.pos_[1][0] -= .125;
    GPIO io;
    std::vector<Atoms> frames{first, second};
    REQUIRE(io.write(frames, "trajectory.bin", "bin"));
    REQUIRE(run({"calc_trjavg", "--input-file=trajectory.bin", "--input-format=bin",
                 "--output-file=average.bin", "--output-format=bin", "--index=index.map",
                 "--drift=remove", "--wrap=false"}) == 0);
    Atoms average;
    REQUIRE(io.read_one(average, "average.bin", "bin"));
    CHECK(average.pos_[0][0] == Catch::Approx(1.25).margin(1e-12));
    CHECK(average.pos_[1][0] == Catch::Approx(2.9375).margin(1e-12));
}

TEST_CASE("Standalone drift removal centers the physical mass distribution", "[apps][trajectory][audit]") {
    Directory directory;
    const bool explicit_mass = GENERATE(false, true);
    std::ofstream("index.map") << (explicit_mass ? "H 0.4 1\nHe 0.4 4\n" : "H 0.4\nHe 0.4\n");
    Atoms atom;
    atom.cell_ = 20 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {false, false, false};
    atom.pos_ = {{1, 2, 3}, {3, 4, 5}};
    atom.chemmap_ = {0, 1};
    GPIO io;
    REQUIRE(io.write_one(atom, "input.bin", "bin"));
    std::vector<std::string> args{"trj_rmdrift", "--input=input.bin", "--output=centered.bin",
        "--input-format=bin", "--output-format=bin"};
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    CMDOption::global_cmdopts.reset();
    REQUIRE(trj_rmdrift_audit_main(argv.size(), argv.data()) == 0);
    Atoms centered;
    REQUIRE(io.read_one(centered, "centered.bin", "bin"));
    const double h = explicit_mass ? 1 : Chemicals::atomic_masses[1];
    const double he = explicit_mass ? 4 : Chemicals::atomic_masses[2];
    CHECK((h*centered.pos_[0]+he*centered.pos_[1]).norm() < 1e-12);
    CHECK((centered.pos_[1]-centered.pos_[0]).isApprox(atom.pos_[1]-atom.pos_[0], 1e-12));
}
