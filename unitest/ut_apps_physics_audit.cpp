// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include "tools/constants.h"
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <filesystem>
#include <fstream>

namespace {
struct PhysicsWorkspace {
    std::filesystem::path original = std::filesystem::current_path(), path;
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    int threads = omp_num_threads;
    PhysicsWorkspace() {
        char name[] = "/tmp/imrmlp_apps_physics_XXXXXX";
        const char *directory = mkdtemp(name);
        if (!directory) throw std::runtime_error("mkdtemp failed");
        path = directory;
        std::filesystem::current_path(path);
        omp_num_threads = 1;
        global_omp_init();
        std::ofstream("index.map") << "H 1 1\n";
        std::ofstream("lj.ini") << "ntypes 1\n0-0 0.7 2.5\ncut:0-0 6\n";
    }
    ~PhysicsWorkspace() {
        CMDOption::global_cmdopts = saved;
        omp_num_threads = threads;
        global_omp_init();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
    int run(std::vector<std::string> args) {
        CMDOption::global_cmdopts = CMDOption::CommandLineOptions{};
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return command_maps.at(args[0]).first(argv.size(), argv.data());
    }
    void write(Atoms atom) {
        GPIO io;
        REQUIRE(io.readmap("index.map"));
        REQUIRE(io.write_one(atom, "input.cfg", "mlpcfg"));
    }
};
} // namespace

TEST_CASE("EOS pressure-volume reference includes its measured pressure and enthalpy",
          "[apps][physics][eos]") {
    PhysicsWorkspace work;
    Atoms atom;
    atom.cell_ = 30 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{5, 5, 5}, {7.8, 5.4, 4.8}};
    atom.chemmap_ = {0, 0};
    work.write(atom);
    std::ofstream("settings.ini") << "pot:pottype lj612\npot:potfile lj.ini\n";
    REQUIRE(work.run({"calc_eos", "--curve=pv", "--skip-prelax",
        "--settings=settings.ini", "--input=input.cfg", "--input-format=mlpcfg",
        "--output=pv.dat", "--npos=0", "--nneg=0"}) == 0);
    // Independent LJ pair energy and virial: W = -r dE/dr.
    const double r = (atom.pos_[1] - atom.pos_[0]).norm();
    const double ratio6 = std::pow(2.5/r, 6);
    const double energy = 4*.7*(ratio6*ratio6 - ratio6);
    const double virial_trace = 24*.7*(2*ratio6*ratio6 - ratio6);
    const double pressure = virial_trace / (3*atom.get_volume());
    std::ifstream result("pv.dat");
    std::string header;
    REQUIRE(static_cast<bool>(std::getline(result, header)));
    double p, volume, e, h;
    REQUIRE(static_cast<bool>(result >> p >> volume >> e >> h));
    CHECK(std::abs(pressure * eVA3_to_GPa) > 1e-5);
    CHECK(p == Catch::Approx(pressure * eVA3_to_GPa).margin(5.1e-7));
    CHECK(volume == Catch::Approx(atom.get_volume()/2).margin(5.1e-7));
    CHECK(e == Catch::Approx(energy/2).margin(5.1e-7));
    CHECK(h == Catch::Approx((energy + virial_trace/3)/2).margin(5.1e-7));
}
