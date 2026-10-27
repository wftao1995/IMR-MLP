// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include "utils/cmdopts.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>

namespace {
struct Workspace {
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    int threads = omp_num_threads;
    std::filesystem::path original = std::filesystem::current_path(), path;

    Workspace() {
        char name[] = "/tmp/imrmlp_split_set_XXXXXX";
        const char *directory = mkdtemp(name);
        if (!directory) throw std::runtime_error("Cannot create split_set test directory");
        path = directory;
        std::filesystem::current_path(path);
    }
    ~Workspace() {
        CMDOption::global_cmdopts = saved;
        omp_num_threads = threads;
        global_omp_init();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
};

int run(std::vector<std::string> arguments) {
    CMDOption::global_cmdopts.reset();
    CMDOption::global_cmdopts.set_helponexit(false);
    arguments.insert(arguments.begin(), "split_set");
    std::vector<char *> argv;
    for (auto &argument : arguments) argv.push_back(argument.data());
    return split_set(static_cast<int>(argv.size()), argv.data());
}

void write_input(size_t count) {
    Atoms atom;
    atom.cell_ = 12.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1., 1., 1.}, {3., 1., 1.}};
    atom.chemmap_ = {1, 2};
    atom.have_energy = true;
    std::vector<Atoms> frames;
    for (size_t i = 0; i < count; ++i) {
        atom.energy_ = static_cast<double>(i);
        atom.forces_ = {{static_cast<double>(i) + 0.25, 0., 0.}, {0., 0., 0.}};
        frames.push_back(atom);
    }
    GPIO io;
    REQUIRE(io.write(frames, "input.cfg", "mlpcfg"));
}

std::vector<int> read_subset(size_t index, const std::string &prefix = "split_") {
    const auto filename = prefix + std::to_string(index) + ".xyz";
    REQUIRE(std::filesystem::exists(filename));
    if (std::filesystem::file_size(filename) == 0) return {};
    GPIO io;
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, filename, "extxyz"));
    std::vector<int> ids;
    for (const auto &frame : frames) {
        REQUIRE(frame.has_energy());
        const auto id = static_cast<int>(frame.energy_);
        REQUIRE(frame.energy_ == static_cast<double>(id));
        REQUIRE(frame.size() == 2);
        REQUIRE(frame.chemmap_ == std::vector<int>{1, 2});
        REQUIRE(frame.cell_.isApprox(12.0 * Eigen::Matrix3d::Identity()));
        REQUIRE(frame.pos_[0].isApprox(Eigen::RowVector3d(1., 1., 1.)));
        REQUIRE(frame.has_forces());
        REQUIRE(frame.forces_[0](0) == static_cast<double>(id) + 0.25);
        ids.push_back(id);
    }
    return ids;
}

std::vector<int> sequence(int start, size_t count) {
    std::vector<int> ids(count);
    std::iota(ids.begin(), ids.end(), start);
    return ids;
}

void require_contiguous(const std::vector<size_t> &sizes) {
    int start = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        REQUIRE(read_subset(i) == sequence(start, sizes[i]));
        start += static_cast<int>(sizes[i]);
    }
    REQUIRE_FALSE(std::filesystem::exists("split_" + std::to_string(sizes.size()) + ".xyz"));
}
}

TEST_CASE("split_set normalizes relative weights and takes consecutive segments", "[cli][split_set]") {
    Workspace work;
    write_input(100);
    const auto ratios = GENERATE("8,1,1", "0.8,0.1,0.1", "80,10,10", " 8 , 1 , 1 ",
                                 "8e300,1e300,1e300", "8e-300,1e-300,1e-300");
    CAPTURE(ratios);
    REQUIRE(run({"input.cfg", "split_", "--ratios=" + std::string(ratios)}) == 0);
    require_contiguous({80, 10, 10});
}

TEST_CASE("split_set apportions leftover structures by largest remainder", "[cli][split_set]") {
    Workspace work;
    size_t count = 11;
    std::string ratios = "8,1,1";
    std::vector<size_t> sizes{9, 1, 1};
    SECTION("uneven split") {}
    SECTION("ties favor earlier subsets") {
        count = 10;
        ratios = "1,1,1";
        sizes = {4, 3, 3};
    }
    SECTION("larger remainder takes priority over index") {
        count = 7;
        ratios = "1,2,3";
        sizes = {1, 2, 4};
    }
    SECTION("equal remainders with unequal weights") {
        count = 10;
        ratios = "3,1";
        sizes = {8, 2};
    }
    SECTION("positive weights can yield empty subsets") {
        count = 2;
        ratios = "1,1,1";
        sizes = {1, 1, 0};
    }
    SECTION("single structure") {
        count = 1;
        ratios = "1,8,1";
        sizes = {0, 1, 0};
    }
    SECTION("zero weight remains empty") {
        count = 5;
        ratios = "0,2,0,3";
        sizes = {0, 2, 0, 3};
        std::ofstream("split_0.xyz") << "stale data\n";
    }
    write_input(count);
    REQUIRE(run({"input.cfg", "split_", "--ratios=" + ratios}) == 0);
    require_contiguous(sizes);
}

TEST_CASE("split_set shuffles while preserving counts and every structure", "[cli][split_set]") {
    Workspace work;
    write_input(100);
    REQUIRE(run({"input.cfg", "split_", "--ratios=8,1,1", "--shuffle", "yes"}) == 0);
    const std::vector<size_t> sizes{80, 10, 10};
    std::vector<int> ids;
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto subset = read_subset(i);
        REQUIRE(subset.size() == sizes[i]);
        ids.insert(ids.end(), subset.begin(), subset.end());
    }
    std::sort(ids.begin(), ids.end());
    REQUIRE(ids == sequence(0, 100));
}

TEST_CASE("split_set validates ratios before writing any subset", "[cli][split_set][errors]") {
    Workspace work;
    write_input(5);
    const auto ratios = GENERATE("1", "0,0", "-1,2", "nan,1", "1,inf", "1,-inf", "1,word",
                                 "1,2junk", "1,,2", ",1,2", "1,2,", " ,1", "1:2", "1,1e99999");
    CAPTURE(ratios);
    REQUIRE(run({"input.cfg", "split_", "--ratios=" + std::string(ratios)}) == 1);
    REQUIRE_FALSE(std::filesystem::exists("split_0.xyz"));
}

TEST_CASE("split_set rejects missing ratio values", "[cli][split_set][errors]") {
    Workspace work;
    write_input(5);
    const auto option = GENERATE("--ratios", "--ratios=");
    REQUIRE_THROWS_AS(run({"input.cfg", "split_", option}), CMDOption::ParseError);
    REQUIRE_FALSE(std::filesystem::exists("split_0.xyz"));
}

TEST_CASE("split_set reconciles explicit subset counts with ratio length", "[cli][split_set]") {
    Workspace work;
    write_input(10);
    const int count = GENERATE(-1, 0, 2, 3, 4);
    CAPTURE(count);
    REQUIRE(run({"--input=input.cfg", "--output-prefix=split_", "--ratios", "8,1,1",
                 "-n", std::to_string(count)}) == (count == 3 ? 0 : 1));
    if (count == 3) require_contiguous({8, 1, 1});
    else REQUIRE_FALSE(std::filesystem::exists("split_0.xyz"));
}

TEST_CASE("split_set preserves equal round-robin splitting", "[cli][split_set]") {
    Workspace work;
    SECTION("explicit subset count") {
        write_input(7);
        REQUIRE(run({"input.cfg", "split_", "--nsplit=3"}) == 0);
        REQUIRE(read_subset(0) == std::vector<int>{0, 3, 6});
        REQUIRE(read_subset(1) == std::vector<int>{1, 4});
        REQUIRE(read_subset(2) == std::vector<int>{2, 5});
    }
    SECTION("default is ten subsets and empty subsets are skipped") {
        write_input(2);
        REQUIRE(run({"input.cfg", "split_"}) == 0);
        REQUIRE(read_subset(0) == std::vector<int>{0});
        REQUIRE(read_subset(1) == std::vector<int>{1});
        REQUIRE_FALSE(std::filesystem::exists("split_2.xyz"));
    }
}

TEST_CASE("split_set reports output failures including empty subsets", "[cli][split_set][errors]") {
    Workspace work;
    write_input(5);
    const auto ratios = GENERATE("8,1,1", "0,1");
    REQUIRE(run({"input.cfg", "missing/split_", "--ratios=" + std::string(ratios)}) == 1);
    REQUIRE_FALSE(std::filesystem::exists("missing"));
}

TEST_CASE("split_set propagates write and close failures", "[cli][split_set][errors]") {
    Workspace work;
    write_input(5);
    std::filesystem::create_symlink("/dev/full", "split_0.xyz");
    REQUIRE(run({"input.cfg", "split_", "--ratios=1,1"}) == 1);
    REQUIRE_FALSE(std::filesystem::exists("split_1.xyz"));
}

TEST_CASE("split_set help describes ratio mode without requiring input", "[cli][split_set][help]") {
    Workspace work;
    REQUIRE(run({"--help"}) == 0);
    std::ostringstream help;
    CMDOption::global_cmdopts.showhelp(help);
    REQUIRE(help.str().find("--ratios") != std::string::npos);
    REQUIRE(help.str().find("consecutive segments") != std::string::npos);
    REQUIRE(std::filesystem::is_empty(work.path));
}
