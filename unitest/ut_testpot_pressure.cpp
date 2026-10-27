// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "potentials/potential_lj.h"
#include "tools/constants.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace {
constexpr double epsilon = 0.7;
constexpr double sigma = 2.5;

// fmt::print writes to the stdout FILE rather than std::cout's stream buffer.
// Capture the descriptor so this regression checks the actual CLI diagnostic.
class StdoutCapture {
    FILE *file = nullptr;
    int saved = -1;
public:
    StdoutCapture() {
        std::cout.flush();
        std::fflush(stdout);
        file = std::tmpfile();
        if (!file) throw std::runtime_error("Cannot create stdout capture");
        saved = dup(STDOUT_FILENO);
        if (saved < 0 || dup2(fileno(file), STDOUT_FILENO) < 0) {
            if (saved >= 0) close(saved);
            std::fclose(file);
            throw std::runtime_error("Cannot redirect stdout");
        }
    }
    void restore() {
        if (saved < 0) return;
        std::cout.flush();
        std::fflush(stdout);
        dup2(saved, STDOUT_FILENO);
        close(saved);
        saved = -1;
    }
    std::string finish() {
        restore();
        if (std::fseek(file, 0, SEEK_SET) != 0)
            throw std::runtime_error("Cannot rewind captured stdout");
        std::string output;
        char buffer[4096];
        while (const auto count = std::fread(buffer, 1, sizeof(buffer), file))
            output.append(buffer, count);
        if (std::ferror(file)) throw std::runtime_error("Cannot read captured stdout");
        return output;
    }
    ~StdoutCapture() {
        restore();
        std::fclose(file);
    }
};

struct Workspace {
    std::filesystem::path previous = std::filesystem::current_path(), path;
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    int threads = omp_num_threads;
    Workspace() {
        char name[] = "/tmp/imrmlp_testpot_pressure_XXXXXX";
        const char *directory = mkdtemp(name);
        if (!directory) throw std::runtime_error("Cannot create pressure test directory");
        path = directory;
        std::filesystem::current_path(path);
        omp_num_threads = 1;
        global_omp_init();
        std::ofstream("index.map") << "Ar\n";
        std::ofstream("lj.ini") << "ntypes 1\n0-0 0.7 2.5\ncut:0-0 5\n";
        std::ofstream("settings.ini") << "pot:pottype lj612\npot:potfile lj.ini\n";
    }
    ~Workspace() {
        CMDOption::global_cmdopts = saved;
        omp_num_threads = threads;
        global_omp_init();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
    void write(const Atoms &atom) {
        std::ofstream output("input.xyz");
        output << std::setprecision(17) << "2\nLattice=\"";
        for (int i=0; i<3; ++i)
            for (int j=0; j<3; ++j)
                output << (i || j ? " " : "") << atom.cell_(i,j);
        output << "\" Properties=species:S:1:pos:R:3\n";
        for (const auto &position : atom.pos_)
            output << "Ar " << position(0) << ' ' << position(1) << ' ' << position(2) << '\n';
    }
    std::pair<int,std::string> run() {
        CMDOption::global_cmdopts = CMDOption::CommandLineOptions{};
        CMDOption::global_cmdopts.set_helponexit(false);
        std::vector<std::string> arguments{"testpot", "settings.ini", "--input=input.xyz",
            "--format=extxyz", "--test-stress=1", "--nugedvalue=0.00001", "--stress-nudge=0.00001"};
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        StdoutCapture capture;
        const int status = testpot(argv.size(), argv.data());
        return {status, capture.finish()};
    }
};

Atoms pair(double distance, bool skew) {
    Atoms atom;
    atom.cell_ = 12 * Eigen::Matrix3d::Identity();
    if (skew) {
        atom.cell_(1,0) = 0.7;
        atom.cell_(2,0) = -0.3;
        atom.cell_(2,1) = 0.4;
    }
    const Eigen::RowVector3d first(4,4,4);
    atom.pos_ = {first, first + distance * Eigen::RowVector3d(2,-1,1).normalized()};
    atom.chemmap_ = {0,0};
    return atom;
}

double pair_energy(double distance) {
    const double ratio6 = std::pow(sigma / distance, 6);
    return 4 * epsilon * (ratio6*ratio6 - ratio6);
}

double reported_pressure(const std::string &output) {
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        const auto marker = line.find("Pressure = ");
        if (marker != std::string::npos)
            return std::stod(line.substr(marker + 11));
    }
    throw std::runtime_error("testpot did not print its pressure diagnostic");
}
}

TEST_CASE("testpot pressure is positive under compression and agrees with minus dE dV",
          "[apps][testpot][pressure][numerics]") {
    Workspace work;
    const double distance = GENERATE(2.65, 3.2);
    const bool skew = GENERATE(false, true);
    const auto atom = pair(distance, skew);
    const double volume = std::abs(atom.cell_.determinant());
    work.write(atom);
    const auto [status, output] = work.run();
    REQUIRE(status == 0);
    const double printed = reported_pressure(output);

    // E(r)=4 eps[(sigma/r)^12-(sigma/r)^6]. Isotropic scaling at fixed
    // fractional coordinates gives p=-dE/dV=-r*E'(r)/(3V), positive for
    // the compressed pair and negative for the stretched pair.
    const double ratio6 = std::pow(sigma / distance, 6);
    const double exact = 24 * epsilon * (2*ratio6*ratio6-ratio6)
                       / (3*volume) * eVA3_to_GPa;
    CAPTURE(distance, skew, printed, exact);
    CHECK(printed * exact > 0);
    CHECK(printed == Catch::Approx(exact).margin(5.1e-7)); // six-decimal CLI output

    GPSettings settings;
    settings.Load("lj.ini");
    Potentials::LJ612 potential(settings);
    for (double relative_step : {1e-4, 1e-5}) {
        const double dv = relative_step * volume;
        auto plus = atom, minus = atom;
        const double scale_plus = std::cbrt(1 + relative_step);
        const double scale_minus = std::cbrt(1 - relative_step);
        plus.cell_ *= scale_plus;
        minus.cell_ *= scale_minus;
        for (auto &position : plus.pos_) position *= scale_plus;
        for (auto &position : minus.pos_) position *= scale_minus;
        potential.getE(plus);
        potential.getE(minus);
        CHECK(plus.energy_ == Catch::Approx(pair_energy(distance*scale_plus)).margin(1e-12));
        CHECK(minus.energy_ == Catch::Approx(pair_energy(distance*scale_minus)).margin(1e-12));
        const double fd = -(plus.energy_-minus.energy_) / (2*dv) * eVA3_to_GPa;
        CAPTURE(relative_step, fd);
        CHECK(fd == Catch::Approx(exact).margin(3e-8));
        CHECK(printed == Catch::Approx(fd).margin(5.1e-7));
    }
}
