// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include "tools/constants.h"
#include "apps/fastfit_profile.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace func_fastfit {
void solve(Eigen::MatrixXd &, Eigen::VectorXd &, Eigen::VectorXd &, std::string, Profile &);
}

namespace {
struct ErrorAuditWorkspace {
    std::filesystem::path previous = std::filesystem::current_path(), directory;
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    ErrorAuditWorkspace() {
        char name[] = "/tmp/imrmlp_apps_training_audit_XXXXXX";
        const auto path = mkdtemp(name);
        if (!path) throw std::runtime_error("Cannot create audit workspace");
        directory = path;
        std::filesystem::current_path(directory);
        std::ofstream("index.map") << "H\nHe\n";
    }
    ~ErrorAuditWorkspace() {
        CMDOption::global_cmdopts = saved;
        SetStreamForOutput(&std::cout);
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(directory);
    }
};

void error_audit_frame(const std::string &path, const std::string &energy = "1",
                       const std::string &force = "2", const std::string &virial = "3",
                       const std::string &species = "H", double cell_x = 5) {
    std::ofstream out(path);
    out << "1\nLattice=\"" << cell_x << " 0 0 0 5 0 0 0 5\" pbc=\"F F F\" "
        << "Properties=species:S:1:pos:R:3:forces:R:3 energy=" << energy
        << " virial=\"" << virial << " 0 0 0 0 0 0 0 0\"\n"
        << species << " 1 1 1 " << force << " 0 0\n";
}

int error_audit_compare() {
    CMDOption::global_cmdopts = CMDOption::CommandLineOptions();
    std::vector<std::string> args{"calc_err", "--input", "ref.xyz", "--input-predict", "pred.xyz",
        "--input-format", "extxyz", "--output-summary", "summary.txt",
        "--output-virial-diff", "virial.dat"};
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    return calc_err(argv.size(), argv.data());
}
}

TEST_CASE("calc_err rejects nonfinite labels instead of reporting a zero maximum", "[apps][audit][errors]") {
    ErrorAuditWorkspace workspace;
    const bool reference = GENERATE(false, true);
    error_audit_frame("ref.xyz");
    error_audit_frame("pred.xyz");
    const std::string path = reference ? "ref.xyz" : "pred.xyz";
    SECTION("NaN energy") { error_audit_frame(path, "nan"); }
    SECTION("infinite force") { error_audit_frame(path, "1", "inf"); }
    SECTION("NaN virial") { error_audit_frame(path, "1", "2", "nan"); }
    REQUIRE_THROWS_WITH(error_audit_compare(), (Catch::Matchers::ContainsSubstring("finite") || Catch::Matchers::ContainsSubstring("failed to read nonempty")));
    REQUIRE_FALSE(std::filesystem::exists("summary.txt"));
}

TEST_CASE("calc_err requires matching atom species", "[apps][audit][errors]") {
    ErrorAuditWorkspace workspace;
    error_audit_frame("ref.xyz");
    error_audit_frame("pred.xyz", "1", "2", "3", "He");
    REQUIRE_THROWS_WITH(error_audit_compare(), Catch::Matchers::ContainsSubstring("species"));
}

TEST_CASE("calc_err stress conversion uses physical cell volume", "[apps][audit][stress]") {
    ErrorAuditWorkspace workspace;
    const double cell_x = GENERATE(0., 5., -5.);
    error_audit_frame("ref.xyz", "1", "2", "3", "H", cell_x);
    error_audit_frame("pred.xyz", "1", "2", "0", "H", cell_x);
    if (cell_x == 0) {
        REQUIRE_THROWS_WITH(error_audit_compare(), Catch::Matchers::ContainsSubstring("volume"));
    } else {
        REQUIRE(error_audit_compare() == 0);
        std::ifstream in("virial.dat");
        std::string header;
        std::getline(in, header);
        double reference, prediction;
        REQUIRE(bool(in >> reference >> prediction));
        REQUIRE(reference == Catch::Approx(3 * eVA3_to_GPa / 125));
        REQUIRE(prediction == 0);
    }
}

TEST_CASE("fastfit solvers retain directions with very different physical units", "[apps][audit][fastfit]") {
    const std::string method = GENERATE("llt", "ldlt", "svd", "qr", "cod", "alglib", "algfast");
    Eigen::Matrix3d correlation;
    correlation << 2., .3, -.2, .3, 3., .4, -.2, .4, 1.;
    const Eigen::Vector3d units(1e-12, 1., 1e12), expected(.7, -.4, .9);
    Eigen::MatrixXd h = units.asDiagonal() * correlation * units.asDiagonal();
    Eigen::VectorXd b = units.asDiagonal() * correlation * expected;
    const Eigen::MatrixXd original_h = h;
    const Eigen::VectorXd original_b = b;
    Eigen::VectorXd x;
    func_fastfit::Profile profile;
    func_fastfit::solve(h, b, x, method, profile);
    REQUIRE((units.cwiseProduct(x) - expected).norm() < 1e-12);
    REQUIRE((h - original_h).norm() == 0);
    REQUIRE((b - original_b).norm() == 0);
}

TEST_CASE("calc_ferr preserves the precision of its stable free energy estimate", "[apps][audit][free-energy]") {
    ErrorAuditWorkspace workspace;
    std::ofstream reference("ref.xyz"), prediction("mlp.cfg");
    for (const double energy : {-10000., -10000.1, -9999.9}) {
        reference << std::setprecision(17) << "1\nLattice=\"5 0 0 0 5 0 0 0 5\" "
            << "Properties=species:S:1:pos:R:3 energy=" << energy << "\nH 1 1 1\n";
        prediction << "1\nLattice=\"5 0 0 0 5 0 0 0 5\" "
            << "Properties=species:S:1:pos:R:3 energy=0\nH 1 1 1\n";
    }
    reference.close();
    prediction.close();
    CMDOption::global_cmdopts = CMDOption::CommandLineOptions();
    std::vector<std::string> args{"calc_ferr", "--input", "ref.xyz", "--input-format", "extxyz"};
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    std::ostringstream output;
    struct Capture {
        std::streambuf *buffer;
        std::streamsize precision;
        Capture(std::ostream &out) : buffer(std::cout.rdbuf(out.rdbuf())), precision(std::cout.precision()) {}
        ~Capture() { std::cout.rdbuf(buffer); std::cout.precision(precision); }
    };
    int status;
    {
        Capture capture(output);
        status = calc_ferr(argv.size(), argv.data());
    }
    REQUIRE(status == 0);
    // Independent Python shifted log-sum-exp at kB=8.61733326e-5 eV/K, T=300 K.
    REQUIRE(std::stod(output.str()) == Catch::Approx(-10000.072144380465).margin(2e-11).epsilon(0));
}
