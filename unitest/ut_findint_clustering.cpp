#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>
#include <limits>

namespace {
struct FindintWorkspace {
    std::filesystem::path old = std::filesystem::current_path(), path;
    FindintWorkspace() {
        char name[] = "/tmp/imrmlp_findint_cluster_XXXXXX";
        const char *dir = mkdtemp(name);
        if (!dir) throw std::runtime_error("Cannot create findint test directory");
        path = dir;
        std::filesystem::current_path(path);
    }
    ~FindintWorkspace() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(old);
        std::filesystem::remove_all(path);
    }
    int run(std::vector<std::string> args) {
        CMDOption::global_cmdopts.reset();
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return command_maps.at("findint").first(argv.size(), argv.data());
    }
    void input(const std::string &file, double shift, double displacement = .12) {
        std::ofstream out(file);
        out << "4\nLattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n";
        for (const Eigen::RowVector3d p : {Eigen::RowVector3d{displacement,0,0},
                 Eigen::RowVector3d{0,2,2}, Eigen::RowVector3d{2,0,2}, Eigen::RowVector3d{2,2,0}})
            out << "Cu " << p[0]+shift << ' ' << p[1]+shift << ' ' << p[2]+shift << '\n';
    }
    Atoms read(const std::string &file) {
        GPIO io;
        io.setmap();
        std::vector<Atoms> result;
        REQUIRE(io.read(result, file, "extxyz"));
        REQUIRE(result.size() == 1);
        return result[0];
    }
};

// Independent finite image enumeration for these orthogonal 4 A fixtures.
double periodic_distance(const Eigen::RowVector3d &a, const Eigen::RowVector3d &b) {
    double best = std::numeric_limits<double>::infinity();
    for (int x=-1; x<=1; ++x) for (int y=-1; y<=1; ++y) for (int z=-1; z<=1; ++z)
        best = std::min(best, (a-b+Eigen::RowVector3d{4.*x,4.*y,4.*z}).norm());
    return best;
}
void same_sites(const Atoms &a, const Atoms &b, double shift = 0) {
    REQUIRE(a.size() == b.size());
    for (const auto &p : a.pos_) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto &q : b.pos_)
            best = std::min(best, periodic_distance(p+Eigen::RowVector3d::Constant(shift), q));
        CHECK(best < 1e-10);
    }
}
}

TEST_CASE("findint averages periodic clusters near their anchor", "[apps][findint]") {
    FindintWorkspace w;
    w.input("fcc.xyz", 0);
    w.input("shift.xyz", .37);
    auto args = [](std::string input, std::string output) {
        return std::vector<std::string>{"findint", input, "--sites-only", "--output-format=extxyz", "--output="+output};
    };
    REQUIRE(w.run(args("fcc.xyz", "sites.xyz")) == 0);
    REQUIRE(w.run(args("shift.xyz", "shift-sites.xyz")) == 0);
    const auto sites = w.read("sites.xyz"), shifted = w.read("shift-sites.xyz");
    REQUIRE(sites.size() == 12);
    same_sites(sites, shifted, .37);
    double boundary_distance = 10;
    for (const auto &p : sites.pos_)
        boundary_distance = std::min(boundary_distance, periodic_distance(p, {2.12,0,0}));
    CHECK(boundary_distance < 1e-10);

    auto raw_args = args("fcc.xyz", "raw.xyz");
    raw_args.push_back("--cluster-tol=0");
    REQUIRE(w.run(raw_args) == 0);
    auto raw = w.read("raw.xyz");
    REQUIRE(raw.size() == 17);
    int split_vertices = 0;
    for (const auto &p : raw.pos_)
        if (periodic_distance(p, {2.12,0,0}) < .006) ++split_vertices;
    CHECK(split_vertices == 4);

    auto slow_args = args("fcc.xyz", "slow.xyz");
    slow_args.push_back("--no-fast-cluster");
    REQUIRE(w.run(slow_args) == 0);
    same_sites(sites, w.read("slow.xyz"));
    slow_args = args("shift.xyz", "slow-shift.xyz");
    slow_args.push_back("--no-fast-cluster");
    REQUIRE(w.run(slow_args) == 0);
    same_sites(sites, w.read("slow-shift.xyz"), .37);

    auto top_args = args("fcc.xyz", "top.xyz");
    top_args.push_back("--top=1");
    REQUIRE(w.run(top_args) == 0);
    CHECK(w.read("top.xyz").size() == 1);
    REQUIRE(w.run({"findint", "fcc.xyz", "--insert=H", "--output-format=extxyz", "--output=append.xyz"}) == 0);
    const auto appended = w.read("append.xyz"), original = w.read("fcc.xyz");
    REQUIRE(appended.size() == original.size()+sites.size());
    for (int i=0; i<original.size(); ++i) {
        CHECK(appended.chemmap_[i] == original.chemmap_[i]);
        CHECK((appended.pos_[i]-original.pos_[i]).norm() < 1e-12);
    }

    // An undistorted crystal with all sites away from a wrap boundary remains healthy.
    w.input("healthy.xyz", .37, 0);
    REQUIRE(w.run(args("healthy.xyz", "healthy-sites.xyz")) == 0);
    const auto healthy = w.read("healthy-sites.xyz");
    CHECK(healthy.size() == 12);
    auto healthy_raw_args = args("healthy.xyz", "healthy-raw.xyz");
    healthy_raw_args.push_back("--cluster-tol=0");
    REQUIRE(w.run(healthy_raw_args) == 0);
    same_sites(healthy, w.read("healthy-raw.xyz"));
}
