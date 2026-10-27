// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
struct Workspace {
    std::filesystem::path previous = std::filesystem::current_path(), path;
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    int threads = omp_num_threads;
    Workspace() {
        char name[] = "/tmp/imrmlp_cna_output_XXXXXX";
        const char *directory = mkdtemp(name);
        if (!directory) throw std::runtime_error("Cannot create CNA workspace");
        path = directory;
        std::filesystem::current_path(path);
        omp_num_threads = 1;
        global_omp_init();
        std::ofstream("index.map") << "H 0.4 1\nHe 0.4 4\n";
    }
    ~Workspace() {
        CMDOption::global_cmdopts = saved;
        omp_num_threads = threads;
        global_omp_init();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
    void input(std::vector<Atoms> frames) {
        GPIO io;
        REQUIRE(io.write(frames, "input.bin", "bin"));
    }
    int run(std::vector<std::string> extra = {}, bool trajectory = true) {
        CMDOption::global_cmdopts.reset();
        std::vector<std::string> args{"calc_prop", "--c-cna", "--input=input.bin",
            "--input-format=bin", "--index=index.map", "--cutoff=1.1"};
        if (trajectory) args.push_back("--output-xyz=cna.xyz");
        args.insert(args.end(), extra.begin(), extra.end());
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return calc_prop(argv.size(), argv.data());
    }
    std::vector<Atoms> output() {
        GPIO io;
        REQUIRE(io.readmap("index.map"));
        std::vector<Atoms> frames;
        REQUIRE(io.read(frames, "cna.xyz", "extxyz"));
        return frames;
    }
};

Atoms cluster(std::vector<Eigen::RowVector3d> positions) {
    Atoms atom;
    atom.cell_ = 20 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {false, false, false};
    atom.pos_ = std::move(positions);
    atom.chemmap_.assign(atom.size(), 0);
    return atom;
}

struct CountRow {
    int frame, atom, type, neighbor_type, index, species_index, count;
};
std::vector<CountRow> read_counts() {
    std::ifstream input("output_cna.txt");
    REQUIRE(input.good());
    std::vector<CountRow> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        CountRow row;
        std::istringstream fields(line);
        REQUIRE(bool(fields >> row.frame >> row.atom >> row.type >> row.neighbor_type
                            >> row.index >> row.species_index >> row.count));
        rows.push_back(row);
    }
    return rows;
}
}

TEST_CASE("CNA exports all sites and frames with zero-filled histogram columns", "[apps][cna][output]") {
    Workspace work;
    auto triangle = cluster({{2, 2, 2}, {3, 2, 2}, {2.5, 2.866025403784, 2}});
    auto chain = cluster({{2, 2, 2}, {3, 2, 2}, {4, 2, 2}});
    auto isolated = cluster({{2, 2, 2}, {5, 2, 2}, {8, 2, 2}});
    // Re-analysis must remove bins left in the input by an earlier calculation.
    triangle.local_attribs_i_["cna_9999"] = Eigen::VectorXi::Ones(3);
    work.input({triangle, chain, isolated});
    REQUIRE(work.run() == 0);
    const auto frames = work.output();
    REQUIRE(frames.size() == 3);
    const std::array<std::array<int, 3>, 3> expected{{{2, 2, 2}, {1, 2, 1}, {0, 0, 0}}};
    for (int frame = 0; frame < 3; ++frame) {
        CHECK_FALSE(frames[frame].local_attribs_i_.contains("cna_9999"));
        for (int atom = 0; atom < 3; ++atom) {
            CHECK(frames[frame].local_attribs_i_.at("cna_total")[atom] == expected[frame][atom]);
            CHECK(frames[frame].local_attribs_i_.at("cna_1100")[atom] == (frame == 0 ? 2 : 0));
            CHECK(frames[frame].local_attribs_i_.at("cna_1000")[atom] == (frame == 1 ? expected[frame][atom] : 0));
        }
    }
    const auto rows = read_counts();
    REQUIRE(rows.size() == 9);
    for (const auto &row : rows) {
        REQUIRE(row.frame >= 0);
        REQUIRE(row.frame < 3);
        REQUIRE(row.atom >= 0);
        REQUIRE(row.atom < 3);
        CHECK(row.type == 0);
        CHECK(row.neighbor_type == -1);
        CHECK(row.species_index == -1);
        CHECK(row.count == expected[row.frame][row.atom]);
        CHECK(row.index == (row.frame == 0 ? 1100 : row.frame == 1 ? 1000 : 0));
    }
}

TEST_CASE("CNA crystal exports preserve standard indices in species-resolved counts", "[apps][cna][crystal]") {
    Workspace work;
    const bool bcc = GENERATE(false, true);
    const std::vector<Eigen::RowVector3d> basis = bcc
        ? std::vector<Eigen::RowVector3d>{{0, 0, 0}, {.5, .5, .5}}
        : std::vector<Eigen::RowVector3d>{{0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0}};
    Atoms crystal;
    crystal.cell_ = 4 * Eigen::Matrix3d::Identity();
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 4; ++z)
                for (size_t site = 0; site < basis.size(); ++site) {
                    crystal.pos_.push_back(basis[site] + Eigen::RowVector3d(x, y, z));
                    crystal.chemmap_.push_back(site % 2);
                }
    work.input({crystal});
    std::ofstream("cna.ini") << "cna:specie_info true\n";
    REQUIRE(work.run({"--settings=cna.ini", bcc ? "--cutoff=1.01" : "--cutoff=0.75"}) == 0);
    const auto frames = work.output();
    REQUIRE(frames.size() == 1);
    const auto &out = frames[0];
    std::map<int, int> expected = bcc ? std::map<int, int>{{1666, 8}, {1444, 6}}
                                      : std::map<int, int>{{1421, 12}};
    for (const auto &[index, count] : expected)
        for (int atom = 0; atom < out.size(); ++atom)
            REQUIRE(out.local_attribs_i_.at("cna_" + std::to_string(index))[atom] == count);
    std::vector<std::map<int, int>> species_counts(out.size());
    for (const auto &row : read_counts()) {
        REQUIRE(row.atom >= 0);
        REQUIRE(row.atom < out.size());
        REQUIRE(row.type == crystal.chemmap_[row.atom]);
        REQUIRE(expected.contains(row.index));
        if (row.neighbor_type >= 0) {
            species_counts[row.atom][row.index] += row.count;
            const auto name = "cna_sp_" + std::to_string(row.type) + "_" + std::to_string(row.neighbor_type)
                + "_" + std::to_string(row.index) + "_" + std::to_string(row.species_index);
            REQUIRE(out.local_attribs_i_.at(name)[row.atom] == row.count);
        }
    }
    for (const auto &site : species_counts) REQUIRE(site == expected);
}

TEST_CASE("CNA writes a default count file without an annotated trajectory", "[apps][cna][output]") {
    Workspace work;
    work.input({cluster({{2, 2, 2}, {3, 2, 2}})});
    REQUIRE(work.run({}, false) == 0);
    const auto rows = read_counts();
    REQUIRE(rows.size() == 2);
    for (const auto &row : rows) {
        CHECK(row.index == 1000);
        CHECK(row.count == 1);
    }
    CHECK_FALSE(std::filesystem::exists("cna.xyz"));
}

TEST_CASE("CNA reports output failures before publishing the annotated trajectory", "[apps][cna][errors]") {
    Workspace work;
    const bool full = GENERATE(false, true);
    work.input({cluster({{2, 2, 2}, {3, 2, 2}})});
    if (full) std::filesystem::create_symlink("/dev/full", "output_cna.txt");
    else std::filesystem::create_directory("output_cna.txt");
    CHECK_THROWS_AS(work.run(), GPException);
    CHECK_FALSE(std::filesystem::exists("cna.xyz"));
}

TEST_CASE("CNA keeps periodic image identities in a primitive cell", "[apps][cna][errors]") {
    Workspace work;
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity();
    atom.pos_ = {{0, 0, 0}};
    atom.chemmap_ = {0};
    work.input({atom});
    CHECK(work.run() == 0);
    CHECK(std::filesystem::exists("output_cna.txt"));
}

TEST_CASE("CNA count output cannot alias its annotated trajectory", "[apps][cna][errors]") {
    Workspace work;
    const bool hardlink = GENERATE(false, true);
    work.input({cluster({{2, 2, 2}, {3, 2, 2}})});
    if (hardlink) {
        std::ofstream("counts.txt") << "existing output\n";
        std::filesystem::create_hard_link("counts.txt", "cna.xyz");
    } else {
        // counts.txt does not exist yet, so the final link is dangling.
        std::filesystem::create_symlink("counts.txt", "cna.xyz");
    }
    CHECK_THROWS_WITH(work.run({"--output-cna=counts.txt"}),
                      Catch::Matchers::ContainsSubstring("distinct"));
    if (hardlink) {
        std::ifstream input("counts.txt");
        std::string line;
        REQUIRE(bool(std::getline(input, line)));
        CHECK(line == "existing output");
    } else CHECK_FALSE(std::filesystem::exists("counts.txt"));
}
