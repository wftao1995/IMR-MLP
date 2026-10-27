// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include "potentials/potential_mtp.h"
#include "tools/constants.h"
#include "tools/elastic_utils.h"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
using Matrix6 = Eigen::Matrix<double, 6, 6>;

struct ElasticWorkspace {
    std::filesystem::path original = std::filesystem::current_path(), path;
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    int threads = omp_num_threads;
    std::ostringstream messages;
    std::ostream *old_debug;

    ElasticWorkspace() {
        char name[] = "/tmp/imrmlp_elastic_XXXXXX";
        const char *dir = mkdtemp(name);
        if (!dir) throw std::runtime_error("mkdtemp failed");
        path = dir;
        std::filesystem::current_path(path);
        omp_num_threads = 1;
        global_omp_init();
        old_debug = SetStreamForDebug(&messages);
        std::ofstream("index.map") << "Al\n";
        // Each site sums w(r)=(r-3.6)^2*(r-2.8)^2 over its neighbors.
        // Only the first FCC shell lies within the cutoff in these tests.
        std::ofstream("pair.mtp") << R"(MTP
version = 1.1.0
potential_name = elastic_pair
scaling = 1
species_count = 1
potential_tag =
radial_basis_type = RBChebyshev
min_dist = 0
max_dist = 3.6
radial_basis_size = 3
radial_funcs_count = 1
radial_coeffs
0-0
{2.62, -3.6, 1.62}
alpha_moments_count = 1
alpha_index_basic_count = 1
alpha_index_basic = {{0, 0, 0, 0}}
alpha_index_times_count = 0
alpha_index_times = {}
alpha_scalar_moments = 1
alpha_moment_mapping = {0}
species_coeffs = {0}
moment_coeffs = {1}
)";
        std::ofstream("relax.ini") << "relax:gtol 1e-10\nrelax:ftol 0\nrelax:xtol 0\n";
    }
    ~ElasticWorkspace() {
        SetStreamForDebug(old_debug);
        CMDOption::global_cmdopts = saved;
        omp_num_threads = threads;
        global_omp_init();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
    void write(Atoms atom) {
        GPIO io;
        io.readmap("index.map");
        REQUIRE(io.write_one(atom, "input.cfg", "mlpcfg"));
    }
    int run(int mode, double pressure = 0.0, double step = 1e-4,
            bool full_hessian = false) {
        CMDOption::global_cmdopts = CMDOption::CommandLineOptions{};
        messages.str("");
        messages.clear();
        std::vector<std::string> args{
            "mlpcalccellhess", "--potfile=pair.mtp", "--input=input.cfg",
            "--input-format=mlpcfg", "--index=index.map", "--settings=relax.ini",
            "--elastic-output=elastic.out", "--eigensys-output=eigensys.out",
            "--symprec=1e-8", "--pressure=" + fmt::format("{:.17g}", pressure),
            "--strain-magnitude=" + fmt::format("{:.17g}", step)};
        if (mode < 0) args.push_back("--method=analytical");
        else {
            args.push_back("--method=numerical");
            args.push_back("--mode=" + std::to_string(mode));
        }
        if (full_hessian) args.push_back("--compute-eigsys");
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return command_maps.at(args[0]).first(argv.size(), argv.data());
    }
    Matrix6 tensor(const std::string &label) {
        std::ifstream in("elastic.out");
        std::string line;
        while (std::getline(in, line) && line != label) {}
        REQUIRE(line == label);
        Matrix6 result;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) REQUIRE(static_cast<bool>(in >> result(i, j)));
        return result;
    }
};

Atoms fcc() {
    const double half = 2.8 / std::sqrt(2.0);
    Atoms atom;
    atom.cell_ << 0, half, half, half, 0, half, half, half, 0;
    atom.pos_ = {{0, 0, 0}};
    atom.chemmap_ = {0};
    return atom;
}

// Independent closed-form derivatives of the radial polynomial with respect
// to engineering Green strain, without using the potential's stress/Hessian.
Matrix6 pair_material(Atoms atom) {
    Neighborhoods nbhs(atom, 3.6);
    Matrix6 result = Matrix6::Zero();
    for (const auto &r : nbhs[0].pos_) {
        const double d = r.norm(), a = d - 3.6, b = d - 2.8;
        const double first = 2*a*b*(a+b);
        const double second = 2*((a+b)*(a+b) + 2*a*b);
        Eigen::Matrix<double, 6, 1> v;
        v << r[0]*r[0], r[1]*r[1], r[2]*r[2], r[1]*r[2], r[0]*r[2], r[0]*r[1];
        result += (second - first/d)/(d*d) * v * v.transpose();
    }
    return result * (eVA3_to_GPa / atom.get_volume());
}
} // namespace

TEST_CASE("Elastic command returns material stiffness at zero and finite prestress", "[elastic][apps]") {
    ElasticWorkspace work;
    const int geometry = GENERATE(0, 1, 2);
    Atoms atom = fcc();
    if (geometry == 1) atom.cell_ *= 0.97;
    if (geometry == 2) {
        Eigen::Matrix3d deform;
        deform << 0.98, 0.04, -0.02, 0.04, 1.01, 0.03, -0.02, 0.03, 0.99;
        atom.cell_ = (atom.cell_ * deform).eval();
    }
    work.write(atom);
    Potentials::PotMTP pot("pair.mtp");
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);
    const double pressure = atom.stress_.trace() * eVA3_to_GPa / (3*atom.get_volume());
    if (geometry == 0) REQUIRE(std::abs(pressure) < 1e-9);
    if (geometry == 1) REQUIRE(pressure > 1.0);
    if (geometry == 2) REQUIRE(std::abs(atom.stress_(0, 1)) > 1e-3);
    const Matrix6 expected = pair_material(atom);
    for (int mode : {-1, 0, 1}) {
        CAPTURE(geometry, mode);
        REQUIRE(work.run(mode, pressure) == 0);
        const Matrix6 material = work.tensor("Stiffness tensor (GPa)");
        CHECK((material - expected).cwiseAbs().maxCoeff() < 2e-3);
        const Matrix6 born = work.tensor("Born/Wallace stiffness tensor (GPa)");
        CHECK((born - ElasticUtils::born_stiffness(expected, pressure)).cwiseAbs().maxCoeff() < 2e-3);
        CHECK(work.messages.str().find("ionic relaxation unconverged") == std::string::npos);
        // External pressure selects the Born tensor, not the definition of C.
        REQUIRE(work.run(mode, 0.0) == 0);
        CHECK((work.tensor("Stiffness tensor (GPa)") - material).cwiseAbs().maxCoeff() < 1e-6);
    }
}

TEST_CASE("Elastic command warns and continues after unconverged ionic relaxation", "[elastic][apps]") {
    ElasticWorkspace work;
    const int mode = GENERATE(0, 1);
    Atoms atom;
    const double a = 2.8 * std::sqrt(2.0);
    atom.cell_ = Eigen::Matrix3d::Identity() * a;
    atom.pos_ = {{0,0,0}, {0,a/2,a/2}, {a/2,0,a/2}, {a/2,a/2,0}};
    atom.chemmap_ = {0,0,0,0};
    atom.pos_[1][0] += 0.05;
    work.write(atom);
    std::ofstream("relax.ini") << "relax:gtol 1e-12\nrelax:maxiteration 0\n";
    REQUIRE(work.run(mode) == 0);
    const auto warnings = work.messages.str();
    CHECK(warnings.find("ionic relaxation unconverged at Green-Lagrange strain") != std::string::npos);
    CHECK(warnings.find("max|gradient|=") != std::string::npos);
    CHECK(warnings.find("reason=iteration_limit") != std::string::npos);
    CHECK(warnings.find("Continuing with the last iterate") != std::string::npos);
    REQUIRE(work.tensor("Stiffness tensor (GPa)").allFinite());
    std::ifstream in("elastic.out");
    const std::string output((std::istreambuf_iterator<char>(in)), {});
    CHECK(output.find("WARNING: calc_elastic:") != std::string::npos);
    CHECK(output.find("strain evaluations had unconverged ionic relaxation") != std::string::npos);
}

TEST_CASE("Elastic command rejects inadmissible Green strain", "[elastic][apps]") {
    ElasticWorkspace work;
    work.write(fcc());
    const int mode = GENERATE(0, 1);
    REQUIRE_THROWS_AS(work.run(mode, 0.0, 0.6), std::invalid_argument);
}

TEST_CASE("Elastic command reports scientific output close failures", "[elastic][apps][errors]") {
    if (!std::filesystem::exists("/dev/full")) SKIP("/dev/full is unavailable");
    const std::string destination = GENERATE("numerical-elastic", "analytical-elastic",
                                              "eigensystem", "full-hessian");
    ElasticWorkspace work;
    work.write(fcc());
    const std::string filename = destination == "eigensystem" ? "eigensys.out"
        : destination == "full-hessian" ? "eigtotal.dat" : "elastic.out";
    std::filesystem::create_symlink("/dev/full", filename);
    const int mode = destination == "numerical-elastic" ? 0 : -1;
    REQUIRE_THROWS_WITH(work.run(mode, 0., 1e-4, destination == "full-hessian"),
                        Catch::Matchers::ContainsSubstring(filename));
}

TEST_CASE("Elastic command writes the optional combined Hessian eigensystem", "[elastic][apps]") {
    ElasticWorkspace work;
    const auto atom = fcc();
    work.write(atom);
    REQUIRE(work.run(-1, 0., 1e-4, true) == 0);
    std::ifstream input("eigtotal.dat");
    REQUIRE(input.good());
    std::string line;
    REQUIRE(static_cast<bool>(std::getline(input, line)));
    REQUIRE(line == "FULL HESSIAN");
    const int dimension = 3 * atom.size() + 9;
    for (int i = 0; i < dimension; ++i)
        for (int j = 0; j < dimension; ++j) {
            double value;
            REQUIRE(static_cast<bool>(input >> value));
            REQUIRE(std::isfinite(value));
        }
    std::getline(input, line);
    REQUIRE(static_cast<bool>(std::getline(input, line)));
    REQUIRE(line == "EIGEN SYSTEM");
    for (int i = 0; i < dimension; ++i)
        for (int j = 0; j < dimension + 1; ++j) {
            double value;
            REQUIRE(static_cast<bool>(input >> value));
            REQUIRE(std::isfinite(value));
        }
    REQUIRE((work.tensor("Stiffness tensor (GPa)") - pair_material(atom)).cwiseAbs().maxCoeff() < 2e-3);
}

TEST_CASE("Elastic command rejects an incomplete input before replacing scientific outputs", "[elastic][apps][errors][input]") {
    ElasticWorkspace work;
    work.write(fcc());
    std::ofstream("input.cfg", std::ios::app) << "BEGIN_CFG\n Size\n 1\n";
    const std::string sentinel = "previous scientific result\n";
    std::ofstream("elastic.out") << sentinel;
    std::ofstream("eigensys.out") << sentinel;
    const int mode = GENERATE(-1, 0);
    REQUIRE_THROWS_WITH(work.run(mode), Catch::Matchers::ContainsSubstring("Failed to read complete input")
                        && Catch::Matchers::ContainsSubstring("input.cfg"));
    for (const auto &filename : {"elastic.out", "eigensys.out"}) {
        std::ifstream input(filename);
        std::ostringstream contents;
        contents << input.rdbuf();
        REQUIRE(contents.str() == sentinel);
    }
    REQUIRE_FALSE(std::filesystem::exists("eigtotal.dat"));
}
