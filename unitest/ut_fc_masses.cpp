#include <catch2/catch_all.hpp>

#include "apps/apps.h"
#include "io/gpio.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {

struct Workspace {
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    std::filesystem::path original = std::filesystem::current_path(), path;
    int threads = omp_num_threads;

    Workspace() {
        char name[] = "/tmp/imrmlp_fc_masses_XXXXXX";
        const char *directory = mkdtemp(name);
        if (!directory) throw std::runtime_error("Cannot create FC mass test directory");
        path = directory;
        std::filesystem::current_path(path);
        omp_num_threads = 1;
        global_omp_init();
        // Identical pair interactions for both types: E(r) = (r-5)^2 (r-1).
        std::ofstream("pair.mtp") << R"(MTP
version = 1.1.0
potential_name = analytic_pair
scaling = 1
species_count = 2
potential_tag =
radial_basis_type = RBChebyshev
min_dist = 1
max_dist = 5
radial_basis_size = 2
radial_funcs_count = 1
radial_coeffs
0-0
{1, 1}
0-1
{1, 1}
1-0
{1, 1}
1-1
{1, 1}
alpha_moments_count = 1
alpha_index_basic_count = 1
alpha_index_basic = {{0, 0, 0, 0}}
alpha_index_times_count = 0
alpha_index_times = {}
alpha_scalar_moments = 1
alpha_moment_mapping = {0}
species_coeffs = {0, 0}
moment_coeffs = {1}
)";
        std::ofstream("input.xyz") <<
            "2\nLattice=\"20 0 0 0 20 0 0 0 20\" "
            "Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n"
            "H 2 2 2\nHe 4 2 2\n";
    }

    ~Workspace() {
        CMDOption::global_cmdopts = saved;
        omp_num_threads = threads;
        global_omp_init();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }

    int run(const std::string &index, const std::vector<std::string> &extra = {},
            bool lowest_modes = true) {
        std::ofstream("index.map") << index;
        CMDOption::global_cmdopts = CMDOption::CommandLineOptions{};
        std::vector<std::string> arguments{
            "calc_fc", "--potfile=pair.mtp", "--input=input.xyz",
            "--input-format=extxyz", "--index=index.map"};
        if (lowest_modes) arguments.push_back("--lowest-modes=3");
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        return calc_fc(argv.size(), argv.data());
    }

    Eigen::Matrix<double, 3, 4> modes(const std::string &index) {
        REQUIRE(run(index) == 0);
        Eigen::Matrix<double, 3, 4> result;
        std::ifstream input("EIGENVALS");
        std::string header;
        REQUIRE(std::getline(input, header));
        for (int row = 0; row < result.rows(); ++row)
            for (int column = 0; column < result.cols(); ++column)
                REQUIRE(input >> result(row, column));
        return result;
    }
};

void require_pair_modes(const Eigen::Matrix<double, 3, 4> &modes,
                        double first_mass, double second_mass) {
    // At r=2, E''=-10 and E'/r=1.5. Each internal eigenvalue is
    // the corresponding pair stiffness times (1/m1 + 1/m2).
    const double inverse_mass_sum = 1.0 / first_mass + 1.0 / second_mass;
    const Eigen::Vector3d expected = inverse_mass_sum * Eigen::Vector3d(-10, 1.5, 1.5);
    for (int mode = 0; mode < 3; ++mode) {
        CHECK(modes(mode, 0) == Catch::Approx(expected(mode)).margin(1e-10));
        const double frequency = std::copysign(15.633304237154926 *
            std::sqrt(std::abs(expected(mode))), expected(mode));
        CHECK(modes(mode, 1) == Catch::Approx(frequency).margin(1e-9));
    }
}

} // namespace

TEST_CASE("Lowest modes use the mass column independently of atomic radii",
          "[fc][mass][eigenmodes][cli]") {
    Workspace work;
    const auto baseline = work.modes("H 0.4 2\nHe 0.7 5\n");
    require_pair_modes(baseline, 2, 5);

    const auto scaled = work.modes("H 0.4 8\nHe 0.7 20\n");
    require_pair_modes(scaled, 8, 20);
    CHECK(scaled.col(0).isApprox(baseline.col(0) / 4, 1e-10));
    CHECK(scaled.col(1).isApprox(baseline.col(1) / 2, 1e-10));

    const auto new_radii = work.modes("H 1.6 2\nHe 2.8 5\n");
    CHECK(new_radii.isApprox(baseline, 1e-10));
}

TEST_CASE("Lowest modes fall back to element masses separately for each type",
          "[fc][mass][eigenmodes][cli]") {
    Workspace work;
    const std::string index = GENERATE("H\nHe\n", "H 0\nHe 0.7\n",
                                      "H 0.4 2\nHe 0.7\n", "H 0.4\nHe 0.7 5\n");
    const double first_mass = index.find("0.4 2") != std::string::npos ? 2 : 1.008;
    const double second_mass = index.find("0.7 5") != std::string::npos ? 5 : 4.002602;
    require_pair_modes(work.modes(index), first_mass, second_mass);
}

TEST_CASE("Lowest modes reject invalid explicit masses before writing results",
          "[fc][mass][eigenmodes][cli]") {
    Workspace work;
    const std::string mass = GENERATE("0", "-1", "nan", "inf", "1e999");
    const bool first_type = GENERATE(false, true);
    const std::string index = first_type ? "H 0.4 " + mass + "\nHe 0.7 5\n"
                                         : "H 0.4 2\nHe 0.7 " + mass + "\n";
    REQUIRE_THROWS_WITH(work.run(index), Catch::Matchers::ContainsSubstring("mass"));
    CHECK_FALSE(std::filesystem::exists("EIGENVALS"));
}

TEST_CASE("Lowest modes reject unknown elements when their mass is omitted",
          "[fc][mass][eigenmodes][cli]") {
    Workspace work;
    REQUIRE_THROWS_WITH(work.run("H 0.4 2\nUnknown 0.7\n"),
                        Catch::Matchers::ContainsSubstring("Unknown element"));
    CHECK_FALSE(std::filesystem::exists("EIGENVALS"));
}





TEST_CASE("FC reference read failures stop Gamma and lowest-mode calculations",
          "[fc][input][cli]") {
    Workspace work;
    const bool lowest_modes = GENERATE(false, true);
    CAPTURE(lowest_modes);
    std::ofstream("input.xyz", std::ios::app)
        << "2\nProperties=species:S:1:pos:R:3\nH 2 2 2\n";
    const std::string sentinel = "previous eigensystem\n";
    std::ofstream("EIGENVALS") << sentinel;
    const std::vector<std::string> arguments{
        "--write-force-constant=false",};
    REQUIRE_THROWS_WITH(work.run("H 0.4 2\nHe 0.7 5\n", arguments, lowest_modes),
        Catch::Matchers::ContainsSubstring("Failed to read complete input")
        && Catch::Matchers::ContainsSubstring("input.xyz"));
    std::ifstream preserved("EIGENVALS");
    std::string line;
    REQUIRE(std::getline(preserved, line));
    REQUIRE(line == "previous eigensystem");
    REQUIRE_FALSE(std::filesystem::exists("supercell_created.xsf"));
    REQUIRE_FALSE(std::filesystem::exists("sup_debug_efs.xyz"));
    REQUIRE_FALSE(std::filesystem::exists("FORCE_CONSTANTS"));
}
