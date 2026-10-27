// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "mindriver/dimer_new.h"
#include "potentials/potential_mtp.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cstdlib>

namespace {
struct Workspace {
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    std::filesystem::path original = std::filesystem::current_path(), path;
    std::string potential = std::filesystem::absolute("unitest/test-data/potential.mtp").string();
    int threads = omp_num_threads;
    Workspace() {
        char name[] = "/tmp/imrmlp_saddle_inputs_XXXXXX";
        const char *dir = mkdtemp(name);
        if (!dir) throw std::runtime_error("Cannot create saddle test directory");
        path = dir;
        std::filesystem::current_path(path);
        std::ofstream("index.map") << "H\nHe\n";
        omp_num_threads = 1;
        global_omp_init();
    }
    ~Workspace() {
        CMDOption::global_cmdopts = saved;
        omp_num_threads = threads;
        global_omp_init();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
    int run(int (*command)(int, char **), std::vector<std::string> arguments) {
        CMDOption::global_cmdopts = CMDOption::CommandLineOptions{};
        CMDOption::global_cmdopts.set_helponexit(false);
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        return command(argv.size(), argv.data());
    }
    int nr(std::string limit, std::string tolerance, std::string output = "final.cfg") {
        return run(nroptimize, {"nroptimize", "--potfile=" + potential, "--input=input.cfg",
                               "--output=" + output, "--no-cell", "--max-iteration=" + limit,
                               "--gtol=" + tolerance});
    }
};

Atoms structure() {
    Atoms atom;
    atom.cell_ << 20, 1, 2, 0, 25, 3, 0, 0, 30;
    atom.pos_ = {{1, 1, 1}, {3, 1, 1}};
    atom.chemmap_ = {0, 0};
    return atom;
}

void write_structure(Atoms atom) {
    GPIO io;
    REQUIRE(io.write_one(atom, "input.cfg", "mlpcfg"));
}
}

TEST_CASE("NR reports convergence including the initial and final allowed states", "[nr][cli]") {
    Workspace work;
    write_structure(structure());
    const std::string limit = GENERATE("0", "1");
    const bool converge = GENERATE(false, true);
    REQUIRE(work.nr(limit, converge ? "1e20" : "1e-30") == (converge ? 0 : 2));
    REQUIRE(std::filesystem::exists("final.cfg"));
    REQUIRE(std::filesystem::exists("str.iteration.0.cfg") == (!converge && limit == "1"));
}

TEST_CASE("NR accepts convergence on its last permitted step", "[nr][cli]") {
    Workspace work;
    auto initial = structure();
    write_structure(initial);
    REQUIRE(work.nr("1", "1e-30") == 2);
    GPIO io;
    Atoms final;
    REQUIRE(io.read_one(final, "final.cfg", "mlpcfg"));
    Potentials::PotMTP pot(work.potential);
    pot.getEFS(initial);
    pot.getEFS(final);
    const auto max_force = [](const Atoms &atom) {
        double result = 0;
        for (const auto &force : atom.forces_) result = std::max(result, force.cwiseAbs().maxCoeff());
        return result;
    };
    REQUIRE(max_force(final) < max_force(initial));
    const double tolerance = (max_force(initial) + max_force(final)) / 2;
    REQUIRE(work.nr("1", std::to_string(tolerance)) == 0);
}

TEST_CASE("NR retains useful steps until the force tolerance is reached", "[nr][cli][numerics]") {
    Workspace work;
    write_structure(structure());
    REQUIRE(work.nr("300", "0.01") == 0);
    GPIO io;
    Atoms final;
    REQUIRE(io.read_one(final, "final.cfg", "mlpcfg"));
    Potentials::PotMTP pot(work.potential);
    pot.getEFS(final);
    for (const auto &force : final.forces_)
        CHECK(force.cwiseAbs().maxCoeff() < 0.01);
}

TEST_CASE("NR rejects invalid limits and propagates output failures", "[nr][cli]") {
    Workspace work;
    write_structure(structure());
    SECTION("negative iteration limit") {
        REQUIRE_THROWS_WITH(work.nr("-1", "0.01"), Catch::Matchers::ContainsSubstring("max-iteration"));
    }
    SECTION("invalid tolerance") {
        const std::string tolerance = GENERATE("0", "-1", "nan", "inf");
        REQUIRE_THROWS(work.nr("1", tolerance));
    }
    SECTION("final write failure after convergence") {
        REQUIRE_THROWS(work.nr("0", "1e20", "/dev/full"));
    }
}

TEST_CASE("NR convergence considers only free degrees of freedom", "[nr][cli]") {
    Workspace work;
    write_structure(structure());
    REQUIRE(work.run(nroptimize, {"nroptimize", "--potfile=" + work.potential,
        "--input=input.cfg", "--output=final.cfg", "--no-cell", "--fixed-atoms", "0", "1",
        "--max-iteration=0", "--gtol=1e-30"}) == 0);
}

TEST_CASE("Mode following reports exhaustion and validates MODECAR output", "[modefollowing][cli]") {
    Workspace work;
    write_structure(structure());
    std::ofstream("settings.ini") << "relax:driver fire\nrelax:car_mode true\nrelax:relax_cell false\n";
    const std::vector<std::string> base{"modefollowing", "settings.ini", "--potfile=" + work.potential,
        "--input=input.cfg", "--input-format=mlpcfg", "--nocell", "--car-rep", "--imode=0"};
    auto args = base;
    SECTION("zero budget is not success") {
        args.insert(args.end(), {"--mode=2", "--max-loop=0"});
        REQUIRE(work.run(modefollowing, args) == 2);
        REQUIRE(std::filesystem::exists("POSCAR_SAD"));
    }
    SECTION("one iteration with a small structure is bounded") {
        std::ofstream("settings.ini", std::ios::app) << "relax:maxiteration 0\nrelax:gtol 1e-30\n";
        args.insert(args.end(), {"--mode=2", "--max-loop=1"});
        REQUIRE(work.run(modefollowing, args) == 2);
    }
    SECTION("invalid eigenvector index") {
        args.insert(args.end(), {"--mode=9", "--imode=1000"});
        REQUIRE_THROWS(work.run(modefollowing, args));
    }
    SECTION("MODECAR write failure") {
        args.insert(args.end(), {"--mode=9", "--modecar-out=/dev/full"});
        REQUIRE_THROWS(work.run(modefollowing, args));
    }
}

TEST_CASE("Dimer rejects invalid explicit MODECAR files before opening outputs", "[dimer][modecar]") {
    Workspace work;
    auto atom = structure();
    const std::string content = GENERATE("missing", "directory", "", "0 0 0 0 0 0",
        "1e-20 0 0 0 0 0", "1 0 0", "1 0 bad 0 0 0", "nan 0 0 0 0 0",
        "1 0 inf 0 0 0", "1e999 0 0 0 0 0", "1 0 0 0 0 0 7", "1 0 0 0 0 0 junk");
    INFO(content);
    if (content == "directory") std::filesystem::create_directory("MODECAR");
    else if (content != "missing") std::ofstream("MODECAR") << content;
    DimerDriver driver;
    driver.modecarf = "MODECAR";
    driver.write_traj = true;
    REQUIRE_THROWS_WITH(driver.init(atom), Catch::Matchers::ContainsSubstring("MODECAR"));
    REQUIRE_FALSE(std::filesystem::exists(driver.trajfile));
}

TEST_CASE("Dimer accepts exact legacy directions and preserves state on read errors", "[dimer][modecar]") {
    Workspace work;
    auto atom = structure();
    DimerDriver driver;
    driver.vc_flag = GENERATE(false, true);
    driver.symm_strain = GENERATE(false, true);
    driver.init(atom); // An omitted file still requests a random direction.
    REQUIRE(driver.modedir.norm() == Catch::Approx(1));
    Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(driver.nlen, 1, driver.nlen);
    {
        std::ofstream file("MODECAR");
        for (int i = 0; i < driver.nlen; ++i) file << expected(i) << '\n';
        file << " \t\n";
    }
    driver.read_modecar("MODECAR");
    REQUIRE(driver.modedir.isApprox(expected.normalized()));
    const Eigen::VectorXd before = driver.modedir;
    std::ofstream("MODECAR") << "1 0 0";
    REQUIRE_THROWS(driver.read_modecar("MODECAR"));
    REQUIRE(driver.modedir == before);
}

TEST_CASE("Versioned MODECAR round trips all Dimer coordinate conventions", "[dimer][modecar]") {
    Workspace work;
    auto atom = structure();
    DimerDriver driver;
    driver.vc_flag = GENERATE(false, true);
    driver.symm_strain = GENERATE(false, true);
    driver.restore_F_cholesky = GENERATE(false, true);
    driver.recast_atomic_force = GENERATE(false, true);
    driver.init(atom);
    const Eigen::VectorXd original = driver.modedir;
    driver.write_modecar("MODECAR");
    driver.modecarf = "MODECAR";
    driver.init(atom);
    REQUIRE(driver.modedir.isApprox(original, 1e-14));
    driver.natom += 1;
    REQUIRE_THROWS_WITH(driver.read_modecar("MODECAR"), Catch::Matchers::ContainsSubstring("atom count"));
    driver.natom -= 1;
    if (driver.vc_flag) {
        driver.refcell(0, 0) += 1;
        REQUIRE_THROWS_WITH(driver.read_modecar("MODECAR"), Catch::Matchers::ContainsSubstring("reference cell"));
        driver.refcell = atom.cell_;
        driver.recast_atomic_force = !driver.recast_atomic_force;
        REQUIRE_THROWS_WITH(driver.read_modecar("MODECAR"), Catch::Matchers::ContainsSubstring("coordinate convention"));
    }
}

TEST_CASE("modefollowing MODECAR preserves the physical tangent in a skew cell", "[modecar][cli]") {
    Workspace work;
    auto atom = structure();
    write_structure(atom);
    const bool fixed = GENERATE(false, true);
    const bool cartesian = GENERATE(false, true);
    const int offset = fixed ? 0 : (cartesian ? 9 : 6);
    const int size = offset + 3 * atom.size();
    Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(size, 1, size).normalized();
    // A known, unique lowest eigenvector avoids potential-specific mode ordering.
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity(size, size) - 2 * direction * direction.transpose();
    {
        std::ofstream file("hessian");
        file << std::setprecision(17) << hessian << '\n';
    }
    std::ofstream("settings.ini") << "\n";
    std::vector<std::string> args{"modefollowing", "settings.ini", "--potfile=" + work.potential,
        "--input=input.cfg", "--input-format=mlpcfg", "--hessian=hessian", "--mode=9", "--imode=0"};
    if (fixed) args.push_back("--nocell");
    if (cartesian) args.push_back("--car-rep");
    REQUIRE(work.run(modefollowing, args) == 0);

    DimerDriver driver;
    driver.vc_flag = !fixed;
    driver.symm_strain = !cartesian;
    driver.restore_F_cholesky = !cartesian;
    driver.modecarf = "MODECAR";
    driver.init(atom);
    Atoms plus = atom, minus = atom;
    constexpr double delta = 1e-5;
    driver.X2LatPos(driver.xx_center + delta * driver.modedir, plus);
    driver.X2LatPos(driver.xx_center - delta * driver.modedir, minus);

    Eigen::Matrix3d dcell = Eigen::Matrix3d::Zero();
    if (!fixed) {
        if (cartesian) {
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) dcell(a, b) = direction(3 * a + b);
        } else {
            Eigen::Matrix3d tangent = Eigen::Matrix3d::Zero();
            tangent.diagonal() = .5 * direction.head<3>();
            tangent(1, 0) = direction(5);
            tangent(2, 0) = direction(4);
            tangent(2, 1) = direction(3);
            dcell = atom.cell_ * tangent;
        }
    }
    Eigen::VectorXd expected(9 + 3 * atom.size()), actual(expected.size());
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            expected(3 * a + b) = dcell(a, b);
            actual(3 * a + b) = (plus.cell_(a, b) - minus.cell_(a, b)) / (2 * delta);
        }
    for (int i = 0; i < atom.size(); ++i) {
        Eigen::RowVector3d displacement = direction.segment<3>(offset + 3 * i).transpose();
        if (!cartesian)
            displacement = (displacement * atom.cell_ + atom.car2dir(atom.pos_[i]) * dcell).eval();
        expected.segment<3>(9 + 3 * i) = displacement.transpose();
        actual.segment<3>(9 + 3 * i) = ((plus.pos_[i] - minus.pos_[i]) / (2 * delta)).transpose();
    }
    expected.normalize();
    actual.normalize();
    // Eigenvectors are only defined up to a global sign.
    REQUIRE(std::min((actual - expected).norm(), (actual + expected).norm()) < 1e-8);
}
