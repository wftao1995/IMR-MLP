// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "io/gpio.h"
#include "potentials/potential_shell.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

class ShellTestDirectory {
public:
    explicit ShellTestDirectory(const std::string &name)
        : original_(fs::current_path()), path_(original_ / ("pot-shell-" + name))
    {
        fs::remove_all(path_);
        fs::create_directory(path_);
        fs::current_path(path_);
        write_file("index.map", "X\n");
    }

    ~ShellTestDirectory()
    {
        fs::current_path(original_);
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    static void write_file(const fs::path &path, const std::string &contents)
    {
        std::ofstream ofs(path);
        ofs << contents;
        REQUIRE(ofs.good());
    }

private:
    fs::path original_;
    fs::path path_;
};

Atoms make_atom(double energy)
{
    Atoms atom;
    atom.cell_ = 5.0 * Eigen::Matrix3d::Identity();
    atom.pos_.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));
    atom.chemmap_.push_back(0);
    atom.energy_ = energy;
    atom.have_energy = true;
    return atom;
}

GPSettings shell_settings()
{
    GPSettings settings;
    settings.Modify("index", "index.map");
    settings.Modify("prescript", "pre.sh");
    settings.Modify("runscript", "run.sh");
    settings.Modify("postscript", "post.sh");
    settings.Modify("write_posfile", "input.cfg");
    settings.Modify("write_posformat", "mlpcfg");
    return settings;
}

void write_cfg(const fs::path &path, Atoms atom)
{
    GPIO io;
    REQUIRE(io.write_one(atom, path.string(), "mlpcfg"));
}

} // namespace

TEST_CASE("PotShellCalc completes a successful shell calculation", "[potential][shell]")
{
    ShellTestDirectory directory("success");
    ShellTestDirectory::write_file("pre.sh", "touch \"$1/pre-ran\"\n");
    ShellTestDirectory::write_file("run.sh", "cp reference.cfg \"$1/output.cfg\"\n");
    ShellTestDirectory::write_file("post.sh", "touch \"$1/post-ran\"\n");

    Atoms expected = make_atom(-3.5);
    write_cfg("reference.cfg", expected);

    Potentials::PotShellCalc calculator(shell_settings());
    Atoms actual = make_atom(12.0);
    REQUIRE_NOTHROW(calculator.getEFS(actual));

    REQUIRE(actual.energy_ == Catch::Approx(-3.5));
    REQUIRE(fs::exists("run-job-omp-thread-0/pre-ran"));
    REQUIRE(fs::exists("run-job-omp-thread-0/post-ran"));
}

TEST_CASE("PotShellCalc preserves complete script and job paths", "[potential][shell]")
{
    ShellTestDirectory directory("path space's");
    fs::create_directory("scripts space's");
    auto settings = shell_settings();
    for (const auto &name : {"pre", "run", "post"}) {
        const auto script = fs::path("scripts space's") / (std::string(name) + ".sh");
        std::string contents = "[ \"$#\" -eq 1 ] || exit 31\n"
                               "[ -d \"$1\" ] || exit 32\n";
        contents += "printf '%s\\n' \"$1\" > \"$1/" + std::string(name) + "-arg.txt\"\n";
        if (std::string(name) == "run")
            contents += "cp reference.cfg \"$1/output.cfg\"\n";
        ShellTestDirectory::write_file(script, contents);
        settings.Modify(std::string(name) == "run" ? "runscript" :
                        std::string(name) == "pre" ? "prescript" : "postscript", script.string());
    }
    auto expected = make_atom(-3.5);
    expected.forces_ = {{.4, -.7, 1.2}};
    write_cfg("reference.cfg", expected);
    const auto cwd = fs::current_path();
    const auto jobdir = cwd / "run-job-omp-thread-0";
    Potentials::PotShellCalc calculator(settings);
    auto actual = make_atom(12.0);
    REQUIRE_NOTHROW(calculator.getEFS(actual));
    REQUIRE(fs::current_path() == cwd);
    REQUIRE(actual.has_energy());
    REQUIRE(actual.energy_ == Catch::Approx(-3.5));
    REQUIRE(actual.has_forces());
    REQUIRE(actual.forces_[0].isApprox(Eigen::Vector3d(.4, -.7, 1.2), 1e-12));
    REQUIRE(fs::exists(jobdir / "input.cfg"));
    for (const auto &name : {"pre", "run", "post"}) {
        std::ifstream input(jobdir / (std::string(name) + "-arg.txt"));
        std::string argument;
        REQUIRE(static_cast<bool>(std::getline(input, argument)));
        REQUIRE(argument == jobdir.string());
    }
}

TEST_CASE("PotShellCalc stops after a pre- or run-script failure", "[potential][shell]")
{
    bool fail_pre = false;
    SECTION("pre-script failure") { fail_pre = true; }
    SECTION("run-script failure") { fail_pre = false; }

    ShellTestDirectory directory(fail_pre ? "pre-failure" : "run-failure");
    ShellTestDirectory::write_file(
        "pre.sh", fail_pre ? "exit 7\n" : "touch \"$1/pre-ran\"\n");
    ShellTestDirectory::write_file(
        "run.sh", fail_pre ? "touch \"$1/run-ran\"\n"
                           : "touch \"$1/run-ran\"\nexit 9\n");
    ShellTestDirectory::write_file("post.sh", "touch \"$1/post-ran\"\n");

    fs::create_directory("run-job-omp-thread-0");
    Atoms stale = make_atom(-99.0);
    write_cfg("run-job-omp-thread-0/output.cfg", stale);

    Potentials::PotShellCalc calculator(shell_settings());
    Atoms actual = make_atom(12.0);
    REQUIRE_THROWS_AS(calculator.getEFS(actual), std::runtime_error);

    REQUIRE(actual.energy_ == Catch::Approx(12.0));
    REQUIRE_FALSE(fs::exists("run-job-omp-thread-0/post-ran"));
    if (fail_pre) {
        REQUIRE_FALSE(fs::exists("run-job-omp-thread-0/run-ran"));
    } else {
        REQUIRE(fs::exists("run-job-omp-thread-0/pre-ran"));
        REQUIRE(fs::exists("run-job-omp-thread-0/run-ran"));
    }
}

TEST_CASE("PotShellCalc reports post-script failures", "[potential][shell]")
{
    ShellTestDirectory directory("post-failure");
    ShellTestDirectory::write_file("post.sh", "exit 11\n");

    Potentials::PotShellCalc calculator(shell_settings());
    REQUIRE(calculator.exec_post("post.sh") != 0);
}

TEST_CASE("PotShellCalc rejects missing or unreadable output", "[potential][shell]")
{
    ShellTestDirectory directory("missing-output");
    GPSettings settings = shell_settings();
    Potentials::PotShellCalc calculator(settings);
    Atoms atom = make_atom(12.0);

    SECTION("missing output") {
        REQUIRE(calculator.read_output(atom, ".") != 0);
    }
    SECTION("empty output") {
        ShellTestDirectory::write_file("output.cfg", "");
        REQUIRE(calculator.read_output(atom, ".") != 0);
    }
    SECTION("malformed output") {
        ShellTestDirectory::write_file("output.cfg", "not a configuration\n");
        REQUIRE(calculator.read_output(atom, ".") != 0);
    }
    REQUIRE(atom.energy_ == Catch::Approx(12.0));
}

TEST_CASE("PotShellCalc rejects unreadable separate energy", "[potential][shell]")
{
    ShellTestDirectory directory("energy-failure");
    GPSettings settings = shell_settings();
    settings.Modify("read_energy", "true");
    settings.Modify("engfile", "eng.dat");
    Potentials::PotShellCalc calculator(settings);

    Atoms output = make_atom(-3.5);
    write_cfg("output.cfg", output);
    ShellTestDirectory::write_file("eng.dat", "not-an-energy\n");

    Atoms atom = make_atom(12.0);
    REQUIRE(calculator.read_output(atom, ".") != 0);
    REQUIRE(atom.energy_ == Catch::Approx(12.0));
}

TEST_CASE("PotShellCalc marks separately read energy as available", "[potential][shell]")
{
    ShellTestDirectory directory("separate-energy");
    auto settings = shell_settings();
    settings.Modify("read_energy", "true");
    settings.Modify("engfile", "eng.dat");
    Potentials::PotShellCalc calculator(settings);
    auto output = make_atom(0.);
    output.have_energy = false;
    output.forces_ = {{1., 2., 3.}};
    write_cfg("output.cfg", output);
    ShellTestDirectory::write_file("eng.dat", "-12.5\n");

    Atoms atom;
    REQUIRE(calculator.read_output(atom, ".") == 0);
    REQUIRE(atom.has_energy());
    REQUIRE(atom.energy_ == Catch::Approx(-12.5));
    REQUIRE(atom.has_forces());
    REQUIRE((atom.forces_[0] - output.forces_[0]).norm() == 0.);
}

TEST_CASE("PotShellCalc propagates input output and post-script errors", "[potential][shell]")
{
    ShellTestDirectory directory("io-failure");
    auto settings = shell_settings();
    ShellTestDirectory::write_file("pre.sh", "touch \"$1/pre-ran\"\n");
    ShellTestDirectory::write_file("run.sh", "cp reference.cfg \"$1/output.cfg\"\n");
    ShellTestDirectory::write_file("post.sh", "exit 0\n");
    write_cfg("reference.cfg", make_atom(-3.5));
    SECTION("input cannot be written") { settings.Modify("write_posfile", "missing/input.cfg"); }
    SECTION("output missing despite successful script") {
        ShellTestDirectory::write_file("run.sh", "exit 0\n");
    }
    SECTION("post-script fails") {
        ShellTestDirectory::write_file("post.sh", "exit 11\n");
    }
    Potentials::PotShellCalc calculator(settings);
    auto atom = make_atom(12.0);
    REQUIRE_THROWS_AS(calculator.getEFS(atom), std::runtime_error);
}
