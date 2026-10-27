#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
struct Workspace {
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    std::filesystem::path original = std::filesystem::current_path(), path;
    Workspace() {
        char name[] = "/tmp/imrmlp_directed_orth_XXXXXX";
        const char *directory = mkdtemp(name);
        if (!directory) throw std::runtime_error("mkdtemp failed");
        path = directory;
        std::filesystem::current_path(path);
        std::ofstream("index.map") << "H 1\nHe 1\n";
    }
    ~Workspace() {
        CMDOption::global_cmdopts = saved;
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
    int run(std::vector<std::string> extra, std::string &report) {
        CMDOption::global_cmdopts = CMDOption::CommandLineOptions{};
        std::vector<std::string> args{"mk_orth", "--input=input.xyz", "--input-format=extxyz",
            "--output=output.xyz", "--output-format=extxyz", "--overwrite"};
        args.insert(args.end(), extra.begin(), extra.end());
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        std::ostringstream output;
        struct RestoreStream {
            std::streambuf *saved;
            ~RestoreStream() { std::cout.rdbuf(saved); }
        } restore{std::cout.rdbuf(output.rdbuf())};
        const int status = mk_orth(argv.size(), argv.data());
        report = output.str();
        return status;
    }
};

Atoms source(bool left = false, bool rotated = false) {
    Atoms atom;
    atom.cell_ << 2, 0, 0, -1, std::sqrt(3.), 0, 0, 0, 3;
    if (left) atom.cell_.row(2) *= -1;
    atom.pos_ = {Eigen::RowVector3d(.13, .27, .19)*atom.cell_,
                 Eigen::RowVector3d(.63, .77, .69)*atom.cell_};
    atom.chemmap_ = {0, 1};
    if (rotated) atom.rotate(Eigen::Vector3d(1, 2, 3), .71);
    atom.forces_ = {{.1, .2, .3}, {-.4, .5, .6}};
    atom.velocities_ = {{1, 0, 0}, {0, 1, 1}};
    atom.have_energy = atom.have_stress = true;
    atom.energy_ = 7.;
    atom.stress_ << 1, .2, .3, .2, 2, .4, .3, .4, 3;
    return atom;
}

Eigen::Matrix3i reported_matrix(const std::string &report) {
    const auto position = report.find("Directed supercell matrix");
    REQUIRE(position != std::string::npos);
    const auto newline = report.find('\n', position);
    REQUIRE(newline != std::string::npos);
    std::istringstream input(report.substr(newline+1));
    Eigen::Matrix3i matrix;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) REQUIRE(input >> matrix(i, j));
    return matrix;
}
}

TEST_CASE("Directed orthogonal cell is an integer supercell and a proper rigid rotation", "[cell][orthogonal][direction]") {
    const int target = GENERATE(0, 1, 2);
    const bool left = GENERATE(false, true);
    const bool rotated = GENERATE(false, true);
    Workspace work;
    const auto before = source(left, rotated);
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    auto input = before;
    REQUIRE(io.write_one(input, "input.xyz", "extxyz"));
    const char *axes[] = {"x", "y", "z"};
    std::string report;
    REQUIRE(work.run({"--axis-choice", "2", "2", "0", "--axis-to", axes[target],
                      "--tol=1e-11", "--max-expand=2"}, report) == 0);
    Atoms after;
    REQUIRE(io.read_one(after, "output.xyz", "extxyz"));
    const Eigen::Matrix3i integer = reported_matrix(report);
    REQUIRE(integer.row(target) == Eigen::RowVector3i(1, 1, 0));
    const int copies = std::abs(integer.determinant());
    REQUIRE(copies == 2);
    REQUIRE(after.size() == copies*before.size());
    REQUIRE(after.cell_.determinant() > 0);
    const Eigen::Matrix3d unrotated = integer.cast<double>()*before.cell_;
    const Eigen::Matrix3d rotation = unrotated.inverse()*after.cell_;
    REQUIRE((rotation.transpose()*rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-11));
    REQUIRE(rotation.determinant() == Catch::Approx(1.).margin(1e-11));
    REQUIRE((after.cell_*after.cell_.transpose()).isApprox(unrotated*unrotated.transpose(), 1e-11));
    REQUIRE(after.cell_(target, target) > 0);
    for (int j = 0; j < 3; ++j)
        if (j != target) REQUIRE(std::abs(after.cell_(target, j)) < 1e-10);
    REQUIRE(after.energy_ == Catch::Approx(copies*before.energy_));
    REQUIRE(after.stress_.isApprox(rotation.transpose()*(copies*before.stress_)*rotation, 1e-10));
    for (int i = 0; i < after.size(); ++i) {
        const int type = after.chemmap_[i];
        const Eigen::RowVector3d fractional = after.pos_[i]*rotation.transpose()*before.cell_.inverse();
        Eigen::RowVector3d displacement = fractional-before.pos_[type]*before.cell_.inverse();
        displacement -= displacement.array().round().matrix();
        REQUIRE(displacement.norm() < 1e-10);
        REQUIRE(after.forces_[i].isApprox(rotation.transpose()*before.forces_[type], 1e-10));
        REQUIRE(after.velocities_[i].isApprox(rotation.transpose()*before.velocities_[type], 1e-10));
    }
}

TEST_CASE("Directed orthogonal search rejects impossible completion and invalid directions without output", "[cell][orthogonal][direction]") {
    const int problem = GENERATE(0, 1, 2, 3, 4);
    Workspace work;
    auto atom = source();
    std::vector<std::string> options{"--axis-choice", "1", "1", "0", "--max-expand=2", "--tol=1e-12"};
    if (problem == 0) {
        atom.cell_ << 1, 0, 0, std::sqrt(2.), 1, 0, std::sqrt(3.), std::sqrt(5.), 1;
        options = {"--axis-choice", "1", "0", "0", "--max-expand=2", "--tol=1e-12"};
    } else if (problem == 1) options = {"--axis-choice", "0", "0", "0"};
    else if (problem == 2) options = {"--axis-choice", "1", "0"};
    else if (problem == 3) options.push_back("--axis-to=invalid");
    else atom.pbc_[2] = false;
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    REQUIRE(io.write_one(atom, "input.xyz", "extxyz"));
    std::string report;
    REQUIRE_THROWS(work.run(options, report));
    REQUIRE_FALSE(std::filesystem::exists("output.xyz"));
}
