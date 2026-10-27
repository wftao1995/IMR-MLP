#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace {
std::string bytes(const std::string &filename) {
    std::ifstream input(filename, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
struct LabelWorkspace {
    std::filesystem::path original = std::filesystem::current_path(), path;
    LabelWorkspace() {
        char name[] = "/tmp/imrmlp_findint_labels_XXXXXX";
        const char *created = mkdtemp(name);
        if (!created) throw std::runtime_error("Cannot create findint test directory");
        path = created;
        std::filesystem::current_path(path);
    }
    ~LabelWorkspace() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
    int run(const std::string &source, bool sites_only) {
        CMDOption::global_cmdopts.reset();
        std::vector<std::string> args{"findint", source, "--output=sites.xyz",
            "--output-format=extxyz", "--top=1", "--insert=C"};
        if (sites_only) args.push_back("--sites-only");
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return command_maps.at("findint").first(argv.size(), argv.data());
    }
};
}

TEST_CASE("Findint writes new geometry without host physical labels", "[apps][findint][labels]") {
    const bool sites_only = GENERATE(false, true);
    const int labels = GENERATE(0, 1, 2);
    LabelWorkspace work;
    Atoms host;
    host.cell_ = 4 * Eigen::Matrix3d::Identity();
    host.pos_ = {{0, 0, 0}};
    host.chemmap_ = {1};
    host.have_energy = host.have_fenergy = true;
    host.energy_ = -3;
    host.fenergy_ = -3.1;
    if (labels) {
        host.have_stress = true;
        host.stress_.setIdentity();
        host.forces_ = {{.1, .2, .3}};
        host.site_energies_ = {-3};
        host.charges_ = {.2};
        host.velocities_ = {{1, 2, 3}};
        host.magmoms_ = {{.4, .5, .6}};
        host.global_attribs_d_["total_charge"] = .2;
        host.local_attribs_d_["score"] = Eigen::VectorXd::Constant(1, .7);
        host.features_["label"] = "host-only";
    }
    if (labels == 2) {
        host.hvp_directions_ = Eigen::MatrixXd::Ones(3, 1);
        host.hvp_targets_ = 2 * host.hvp_directions_;
    }
    const std::string filename = labels == 2 ? "host.bin" : "host.xyz";
    GPIO io;
    REQUIRE(io.write_one(host, filename, labels == 2 ? "bin" : "extxyz"));
    const auto saved = bytes(filename);
    REQUIRE(work.run(filename, sites_only) == 0);
    REQUIRE(bytes(filename) == saved);
    Atoms output;
    REQUIRE(io.read_one(output, "sites.xyz", "extxyz"));
    REQUIRE(output.size() == (sites_only ? 1 : 2));
    REQUIRE(output.chemmap_.back() == 6);
    REQUIRE(output.pos_.back().isApprox(Eigen::RowVector3d(2, 2, 2)));
    REQUIRE(output.cell_.isApprox(host.cell_));
    REQUIRE(output.pbc_ == host.pbc_);
    if (!sites_only) {
        REQUIRE(output.chemmap_[0] == 1);
        REQUIRE(output.pos_[0].isZero());
    }
    REQUIRE_FALSE(output.have_energy);
    REQUIRE_FALSE(output.have_fenergy);
    REQUIRE_FALSE(output.have_stress);
    REQUIRE(output.forces_.empty());
    REQUIRE(output.site_energies_.empty());
    REQUIRE(output.charges_.empty());
    REQUIRE(output.velocities_.empty());
    REQUIRE(output.magmoms_.empty());
    REQUIRE_FALSE(output.have_hessian);
    REQUIRE_FALSE(output.has_hvp());
    REQUIRE(output.local_attribs_d_.empty());
    REQUIRE(output.global_attribs_d_.empty());
    REQUIRE(output.features_.empty());
}
