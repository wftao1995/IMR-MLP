// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "utils/cmdopts.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <omp.h>

namespace {
struct Workspace {
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    int threads = omp_num_threads;
    std::filesystem::path original = std::filesystem::current_path(), path;
    Workspace() {
        char name[] = "/tmp/imrmlp_cli_docs_XXXXXX";
        const char *dir = mkdtemp(name);
        if (!dir) throw std::runtime_error("Cannot create CLI test directory");
        path = dir;
        std::filesystem::current_path(path);
        auto &options = CMDOption::global_cmdopts;
        options = CMDOption::CommandLineOptions{};
        options.set_helponexit(false);
    }
    ~Workspace() {
        CMDOption::global_cmdopts = saved;
        omp_num_threads = threads;
        global_omp_init();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
};

int run(int (*command)(int, char **), std::vector<std::string> arguments) {
    std::vector<char *> argv;
    for (auto &argument : arguments) argv.push_back(argument.data());
    return command(argv.size(), argv.data());
}

void write_input() {
    std::ofstream("index.map") << "H\nHe\n";
    Atoms atom;
    atom.cell_ = 12.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1., 1., 1.}, {3.3, 1., 1.}};
    atom.chemmap_ = {0, 0};
    atom.energy_ = 12.;
    atom.have_energy = true;
    GPIO io;
    std::vector<Atoms> frames{atom, atom};
    REQUIRE(io.write(frames, "input.cfg", "mlpcfg"));
}
}

TEST_CASE("relax help succeeds without settings or generated files", "[cli][help]") {
    Workspace work;
    REQUIRE(run(relaxstrv2, {"relax", "--help"}) == 0);
    REQUIRE(CMDOption::global_cmdopts.displayhelp);
    REQUIRE(std::filesystem::is_empty(work.path));
}

TEST_CASE("convstr reports output failures in bulk and sequential modes", "[cli][errors]") {
    Workspace work;
    write_input();
    const std::string destination = GENERATE("missing/output.xyz", "/dev/full");
    const std::string read_rate = GENERATE("0", "1");
    const std::string write_rate = GENERATE("0", "1");
    INFO("destination=" << destination << " loadrate=" << read_rate << " outrate=" << write_rate);
    REQUIRE(run(convstr, {"convstr", "input.cfg", destination, "--output-format=extxyz",
                         "--input-format=mlpcfg", "--overwrite", "--loadrate=" + read_rate,
                         "--outrate=" + write_rate}) == 1);
}

TEST_CASE("split_set rejects nonpositive counts", "[cli][errors]") {
    Workspace work;
    write_input();
    REQUIRE(run(split_set, {"split_set", "input.cfg", "split", GENERATE("--nsplit=0", "--nsplit=-1")}) == 1);
    REQUIRE_FALSE(std::filesystem::exists("split0.xyz"));
}

TEST_CASE("calc_efs reports output failures", "[cli][errors]") {
    const auto potential = std::filesystem::absolute("unitest/test-data/potential.mtp").string();
    Workspace work;
    write_input();
    REQUIRE(run(calc_efs, {"calc_efs", "input.cfg", "missing/output.cfg", "--potfile=" + potential,
                          "--pottype=mtp", "--num_threads=1"}) == 1);
}

TEST_CASE("calc_efs propagates shell errors out of OpenMP workers", "[cli][errors][openmp]") {
    Workspace work;
    write_input();
    std::ofstream("run.sh") << "exit 9\n";
    std::ofstream("input.ini") << "pot:shellfile:runscript run.sh\n"
                                 "pot:shellfile:write_posfile input.cfg\n"
                                 "pot:shellfile:write_posformat mlpcfg\n";
    REQUIRE_THROWS_AS(run(calc_efs, {"calc_efs", "input.cfg", "output.cfg", "--pottype=shellfile",
                                    "--num_threads=2"}), std::runtime_error);
    REQUIRE_FALSE(std::filesystem::exists("output.cfg"));
}

TEST_CASE("relax reports output failures after convergence", "[cli][errors]") {
    const auto potential = std::filesystem::absolute("unitest/test-data/potential.mtp").string();
    Workspace work;
    write_input();
    std::ofstream("relax.ini") << "input input.cfg\noutput /dev/full\noutput-format mlpcfg\n"
                               << "potfile " << potential << "\npottype mtp\n"
                               << "relax:gtol 1e20\nrelax:write_traj false\n";
    std::vector<std::string> args{"relax", "relax.ini"};
    if (GENERATE(false, true)) args.push_back("--seq-write");
    REQUIRE(run(relaxstrv2, args) == 1);
}

TEST_CASE("phonopyfc reports a missing index without terminating the caller", "[cli][errors]") {
    const auto potential = std::filesystem::absolute("unitest/test-data/potential.mtp").string();
    Workspace work;
    REQUIRE_THROWS_WITH(run(phonopyfc, {"phonopyfc", potential, "unused.yaml", "--index-file=missing.map"}),
                        Catch::Matchers::ContainsSubstring("Cannot open index file"));
}

TEST_CASE("EOS fitting stops when the fit fails", "[cli][errors]") {
    Workspace work;
    std::ofstream("empty.dat");
    REQUIRE(run(calc_eos, {"calc_eos", "--fit-only", "--output=empty.dat"}) == 1);
    REQUIRE_FALSE(std::filesystem::exists("veph.dat"));
}

TEST_CASE("CUR selection honors row scaling from CLI and settings", "[cli][selection]") {
    const auto potential = std::filesystem::absolute("unitest/test-data/potential.mtp").string();
    Workspace work;
    std::ofstream("index.map") << "H\nHe\n";
    std::vector<Atoms> pool;
    for (int n : {2, 3, 4, 5}) {
        Atoms atoms;
        atoms.cell_ = Eigen::Matrix3d::Identity() * 30;
        atoms.pbc_ = {false, false, false};
        for (int j = 0; j < n; ++j) {
            atoms.pos_.push_back({1 + 1.7 * j, 1 + 0.15 * j * j, 1});
            atoms.chemmap_.push_back(0);
        }
        pool.push_back(atoms);
    }
    GPIO io;
    io.readmap("index.map");
    REQUIRE(io.write(pool, "pool.cfg", "mlpcfg"));
    const std::string scaling = GENERATE(std::string("uniform"), std::string("mlip"));
    std::vector<std::string> args{
        "diverse_select", "--method=cur", "--pool=pool.cfg", "--format=mlpcfg",
        "--dtype=pot_efs", "--potfile=" + potential, "--w-site=1",
        "--n-select=1", "--cur-deterministic=true", "--output=selected.xyz"};
    if (GENERATE(false, true))
        std::ofstream("diverse_select.ini") << "builder:row-scaling " << scaling << "\n";
    else
        args.push_back("--row-scaling=" + scaling);
    REQUIRE(run(diverse_select, args) == 0);
    std::vector<Atoms> selected;
    REQUIRE(io.read(selected, "selected.xyz", "extxyz"));
    REQUIRE(selected.size() == 1);
    // With this pool, uniform scaling selects the dimer, while MLIP scaling
    // gives greater weight to site rows and selects the four-atom structure.
    REQUIRE(selected[0].size() == (scaling == "uniform" ? 2 : 4));
}
