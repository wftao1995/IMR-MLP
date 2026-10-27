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
struct WCSROOutputWorkspace {
    std::filesystem::path previous = std::filesystem::current_path(), directory;
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    int previous_threads = omp_num_threads;
    GPIO io;
    std::ostringstream messages;
    std::ostream *previous_output = nullptr, *previous_debug = nullptr;

    WCSROOutputWorkspace() {
        char pattern[] = "/tmp/imrmlp_wcsro_output_XXXXXX";
        const char *path = mkdtemp(pattern);
        if (!path) throw std::runtime_error("Cannot create WCSRO output workspace");
        directory = path;
        std::filesystem::current_path(directory);
        omp_num_threads = 1;
        global_omp_init();
        previous_output = SetStreamForOutput(&messages);
        previous_debug = SetStreamForDebug(&messages);
        std::ofstream("index.map") << "H\nHe\nLi\n";
        std::ofstream("settings.ini");
    }
    ~WCSROOutputWorkspace() {
        SetStreamForOutput(previous_output);
        SetStreamForDebug(previous_debug);
        CMDOption::global_cmdopts = saved;
        omp_num_threads = previous_threads;
        global_omp_init();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(directory);
    }
    void write(std::vector<Atoms> frames) {
        REQUIRE(io.readmap("index.map"));
        REQUIRE(io.write(frames, "input.bin", "bin"));
    }
    int run(std::vector<std::string> extra = {}, double cutoff = 1.1,
            const std::string &output = "wcsro.txt") {
        CMDOption::global_cmdopts = CMDOption::CommandLineOptions{};
        std::vector<std::string> args{"calc_prop", "--c-wcsro", "--input=input.bin",
            "--input-format=bin", "--index=index.map", "--settings=settings.ini",
            "--cutoff=" + fmt::format("{:.17g}", cutoff), "--output-sro=" + output};
        args.insert(args.end(), extra.begin(), extra.end());
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return calc_prop(argv.size(), argv.data());
    }
    std::string text() const {
        std::ifstream input("wcsro.txt");
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }
    std::vector<std::array<double, 6>> rows() const {
        std::ifstream input("wcsro.txt");
        REQUIRE(input.good());
        bool wcsro = false;
        std::string line;
        std::vector<std::array<double, 6>> result;
        while (std::getline(input, line)) {
            if (line.starts_with("# WCSRO")) wcsro = true;
            if (!wcsro || line.empty() || line.starts_with('#')) continue;
            std::istringstream stream(line);
            std::array<double, 6> row;
            for (double &value : row) REQUIRE(static_cast<bool>(stream >> value));
            std::string trailing;
            REQUIRE_FALSE(static_cast<bool>(stream >> trailing));
            result.push_back(row);
        }
        REQUIRE(wcsro);
        return result;
    }
};

Atoms binary_star(int first_type = 0, int second_type = 1) {
    Atoms atom;
    atom.cell_ = 20 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {false, false, false};
    atom.pos_ = {{5,5,5}, {4,5,5}, {6,5,5}, {5,6,5}};
    atom.chemmap_ = {first_type, second_type, second_type, second_type};
    return atom;
}

void check_star_rows(const std::vector<std::array<double, 6>> &rows,
                     int frames, bool pm, int first_type = 0, int second_type = 1) {
    REQUIRE(rows.size() == 4);
    const std::array<int, 2> types{first_type, second_type};
    const std::array<double, 2> composition{.25, .75};
    const std::array<double, 4> expected = pm ? std::array<double, 4>{-1./3, -1./3, -3, -3}
                                            : std::array<double, 4>{1, -1./3, -3, 1};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            const auto &row = rows[2*i+j];
            CHECK(row[0] == types[i]);
            CHECK(row[1] == types[j]);
            CHECK(row[2] == Catch::Approx(composition[i]).margin(1e-14));
            CHECK(row[3] == Catch::Approx(composition[j]).margin(1e-14));
            CHECK(row[4] == (i == j ? 0 : 3*frames));
            CHECK(row[5] == Catch::Approx(expected[2*i+j]).margin(1e-14));
        }
}
} // namespace

TEST_CASE("calc_prop exports conditional WCSRO and PM for every directed binary pair",
          "[apps][calc_prop][wcsro][output]") {
    WCSROOutputWorkspace workspace;
    const int frames = GENERATE(1, 2);
    const bool pm = GENERATE(false, true);
    workspace.write(std::vector<Atoms>(frames, binary_star()));
    std::ofstream("settings.ini") << "wcsro:sro_type " << (pm ? "PM" : "WC") << '\n';
    REQUIRE(workspace.run() == 0);
    check_star_rows(workspace.rows(), frames, pm);
    CHECK(workspace.text().find("pooled over " + std::to_string(frames) + " selected frames") != std::string::npos);
}

TEST_CASE("calc_prop preserves noncontiguous original WCSRO species IDs",
          "[apps][calc_prop][wcsro][output]") {
    WCSROOutputWorkspace workspace;
    std::ofstream("index.map") << "H\nHe\nLi\nBe\n";
    workspace.write({binary_star(1,3)});
    REQUIRE(workspace.run() == 0);
    check_star_rows(workspace.rows(), 1, false, 1, 3);
    CHECK(workspace.text().find("# types 1=He 3=Be") != std::string::npos);
}

TEST_CASE("calc_prop pools only the selected WCSRO trajectory frames",
          "[apps][calc_prop][wcsro][output]") {
    WCSROOutputWorkspace workspace;
    Atoms unselected = binary_star();
    unselected.chemmap_.assign(4,2);
    workspace.write({binary_star(), binary_star(), unselected});
    // The third frame would introduce a third species if frame selection were ignored.
    REQUIRE(workspace.run({"--frameend=1"}) == 0);
    check_star_rows(workspace.rows(), 2, false);
}

TEST_CASE("calc_prop WCSRO explicit radial shell takes precedence over global cutoffs",
          "[apps][calc_prop][wcsro][output]") {
    WCSROOutputWorkspace workspace;
    Atoms atom;
    atom.cell_ = 30 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {false, false, false};
    // Two isolated AB dimers of length 1 and 2. The requested shell selects only the latter.
    atom.pos_ = {{3,3,3}, {4,3,3}, {15,15,15}, {17,15,15}};
    atom.chemmap_ = {0,1,0,1};
    workspace.write({atom});
    std::ofstream("settings.ini") << "wcsro:r_inn 1.5\nwcsro:r_out 2.5\n";
    // A global radius of .5 would contain no bonds; explicit outer=2.5 must be used to build the list.
    REQUIRE(workspace.run({}, .5) == 0);
    const auto rows = workspace.rows();
    REQUIRE(rows.size() == 4);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            CHECK(rows[2*i+j][2] == Catch::Approx(.5));
            CHECK(rows[2*i+j][3] == Catch::Approx(.5));
            CHECK(rows[2*i+j][4] == (i == j ? 0 : 1));
            CHECK(rows[2*i+j][5] == Catch::Approx(i == j ? 1 : -1));
        }
}

TEST_CASE("calc_prop keeps SRO and WCSRO sections in their shared output file",
          "[apps][calc_prop][wcsro][output]") {
    WCSROOutputWorkspace workspace;
    workspace.write({binary_star()});
    REQUIRE(workspace.run({"--sro"}) == 0);
    const std::string output = workspace.text();
    CHECK(output.find("# SRO parameter of configurations") != std::string::npos);
    CHECK(output.find("Configuration 1") != std::string::npos);
    CHECK(output.find("# WCSRO pooled") != std::string::npos);
    check_star_rows(workspace.rows(), 1, false);
}

TEST_CASE("calc_prop rejects undefined WCSRO species or neighbor domains before output",
          "[apps][calc_prop][wcsro][errors]") {
    WCSROOutputWorkspace workspace;
    const int invalid = GENERATE(0,1,2,3);
    Atoms atom = binary_star();
    std::string expected;
    if (invalid == 0) { atom.chemmap_.assign(4,0); expected = "exactly two species"; }
    if (invalid == 1) { atom.chemmap_[3] = 2; expected = "exactly two species"; }
    if (invalid == 2) {
        for (auto &position : atom.pos_) position *= 3;
        expected = "neighbor pair";
    }
    if (invalid == 3) {
        atom.pos_ = {{12,12,12}, {4,5,5}, {5,5,5}, {5,6,5}};
        expected = "neighbors around both species";
    }
    workspace.write({atom});
    REQUIRE_THROWS_WITH(workspace.run(), Catch::Matchers::ContainsSubstring(expected));
    CHECK_FALSE(std::filesystem::exists("wcsro.txt"));
}

TEST_CASE("calc_prop reports failed WCSRO output writes", "[apps][calc_prop][wcsro][errors]") {
    WCSROOutputWorkspace workspace;
    workspace.write({binary_star()});
    const bool full_device = GENERATE(false, true);
    std::string output = "blocked";
    if (full_device) {
        if (!std::filesystem::exists("/dev/full")) SKIP("/dev/full is unavailable");
        output = "/dev/full";
    } else {
        std::filesystem::create_directory(output);
    }
    REQUIRE_THROWS_WITH(workspace.run({}, 1.1, output), Catch::Matchers::ContainsSubstring("WCSRO output"));
}

TEST_CASE("calc_prop rejects conflicting SRO and annotated trajectory paths",
          "[apps][calc_prop][wcsro][errors]") {
    WCSROOutputWorkspace workspace;
    workspace.write({binary_star()});
    const std::string trajectory = GENERATE("wcsro.txt", "./wcsro.txt");
    REQUIRE_THROWS(workspace.run({"--output-xyz=" + trajectory}));
    CHECK_FALSE(std::filesystem::exists("wcsro.txt"));
}

TEST_CASE("calc_prop rejects linked aliases of the WCSRO output", "[apps][calc_prop][wcsro][errors]") {
    WCSROOutputWorkspace workspace;
    workspace.write({binary_star()});
    const bool hardlink = GENERATE(false, true);
    if (hardlink) {
        std::ofstream("wcsro.txt") << "existing output\n";
        std::filesystem::create_hard_link("wcsro.txt", "annotated.xyz");
    } else {
        std::filesystem::create_symlink("wcsro.txt", "annotated.xyz");
    }
    REQUIRE_THROWS_WITH(workspace.run({"--output-xyz=annotated.xyz"}),
                        Catch::Matchers::ContainsSubstring("distinct"));
    if (hardlink) {
        std::ifstream input("wcsro.txt");
        std::string line;
        REQUIRE(bool(std::getline(input, line)));
        CHECK(line == "existing output");
    } else CHECK_FALSE(std::filesystem::exists("wcsro.txt"));
}
