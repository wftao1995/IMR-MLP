#include <catch2/catch_all.hpp>
#include "apps/relax_masks.h"
#include "mindriver/relax_new.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>
#include <sstream>

int relaxstrv2(int argc, char *argv[]);

namespace {
struct Directory {
    std::filesystem::path previous = std::filesystem::current_path();
    std::filesystem::path path;
    Directory() {
        char pattern[] = "/tmp/imrmlp_position_mask_XXXXXX";
        const char *directory = mkdtemp(pattern);
        if (!directory) throw std::runtime_error("Cannot create position-mask test directory");
        path = directory;
        std::filesystem::current_path(path);
    }
    ~Directory() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
};

Atoms skew_atoms(int count = 2) {
    Atoms atom;
    atom.cell_ << 10, 1, 0.5, 0, 11, 0.7, 0, 0, 12;
    atom.pos_ = {{2, 2, 2}, {3.4, 2.3, 2.2}};
    atom.pos_.resize(count);
    atom.chemmap_.assign(count, 0);
    atom.sel_dyncell_ = Eigen::Matrix3d::Identity();
    atom.sel_dynpos_.assign(count, Eigen::RowVector3d::Zero());
    return atom;
}

void require_masks_equal(const std::vector<Atoms> &actual, const std::vector<Atoms> &expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i].sel_dyncell_.isApprox(expected[i].sel_dyncell_));
        REQUIRE(actual[i].sel_dynpos_.size() == expected[i].sel_dynpos_.size());
        for (size_t j = 0; j < actual[i].sel_dynpos_.size(); ++j)
            REQUIRE(actual[i].sel_dynpos_[j].isApprox(expected[i].sel_dynpos_[j]));
    }
}

struct ZeroPotential : Potentials::AnyPotential {
    ZeroPotential() {
        cutoff = 2; cutoffsq = 4; species_count = 1;
        cutoffs = Eigen::MatrixXd::Constant(1, 1, 2);
        cutoffsqs = Eigen::MatrixXd::Constant(1, 1, 4);
        r_mins = Eigen::MatrixXd::Zero(1, 1);
    }
    void getEFS(Atoms &atom, AnyNeighborhoods &) override {
        atom.energy_ = 0;
        atom.forces_.assign(atom.size(), Eigen::RowVector3d::Zero());
        atom.stress_.setZero();
    }
};

void write_cli_settings(const std::string &extra) {
    std::ofstream("relax.ini") <<
        "input input.xyz\ninput-format extxyz\noutput output.xyz\noutput-format extxyz\n"
        "pot:pottype lj612\npot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 4\n"
        "relax:gtol 1e-5\nrelax:maxiteration 300\nrelax:write_traj False\n" << extra;
}

int run_cli(std::vector<std::string> extra = {}) {
    std::vector<std::string> args{"relax", "relax.ini", "--num_threads", "1"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    CMDOption::global_cmdopts.reset();
    return relaxstrv2(argv.size(), argv.data());
}

std::vector<Atoms> write_cli_atoms() {
    std::ofstream("index.map") << "H 0\n";
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> atoms{skew_atoms(), skew_atoms()};
    atoms[1].pos_[1](0) += 0.1;
    REQUIRE(io.write(atoms, "input.xyz", "extxyz"));
    return atoms;
}

std::vector<Atoms> read_cli_output() {
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> atoms;
    REQUIRE(io.read(atoms, "output.xyz", "extxyz"));
    return atoms;
}
}

TEST_CASE("Global position mask overrides sparse seldyn positions in every frame only", "[position_mask][seldyn]") {
    std::vector<Atoms> atoms{skew_atoms(2), skew_atoms(1)};
    std::istringstream input("2\n1 2\n1 0 0\n0 0 0\n0 0 1\n0 0 1 0\n"
                             "2 1\n0 1 0\n1 0 0\n0 0 0\n");
    AppsRelax::read_masks(input, atoms);
    const auto before = atoms;
    RelaxationDrvV2 driver;
    driver.control.seldyn_file.clear();
    AppsRelax::apply_position_mask("1.0 0e0 +1", atoms, driver);
    REQUIRE(driver.flags.selective_dynamics);
    for (size_t i = 0; i < atoms.size(); ++i) {
        REQUIRE(atoms[i].sel_dyncell_.isApprox(before[i].sel_dyncell_));
        for (size_t j = 0; j < atoms[i].pos_.size(); ++j) {
            REQUIRE(atoms[i].sel_dynpos_[j].isApprox(Eigen::RowVector3d(1, 0, 1)));
            REQUIRE(atoms[i].pos_[j].isApprox(before[i].pos_[j]));
        }
    }
}

TEST_CASE("Global position mask rejects malformed values before modifying state", "[position_mask][errors]") {
    const std::string text = GENERATE("", " ", "1", "1 0", "1 0 1 0", "1 2 0", "-1 0 1",
                                      "1 .5 0", "1 nan 0", "1 inf 0", "1 0 1x", "True False True", "1,0,1");
    std::vector<Atoms> atoms{skew_atoms()};
    const auto before = atoms;
    RelaxationDrvV2 driver;
    REQUIRE_THROWS_WITH(AppsRelax::apply_position_mask(text, atoms, driver),
                        Catch::Matchers::ContainsSubstring("position-mask"));
    require_masks_equal(atoms, before);
    REQUIRE_FALSE(driver.flags.selective_dynamics);
}

TEST_CASE("Global position mask rejects inactive atomic relaxation and coordinate conflicts", "[position_mask][errors]") {
    const std::string conflict = GENERATE("isif5", "isif6", "isif7", "relax_atom", "keep_symm", "vib", "vib2", "lc", "lf");
    std::vector<Atoms> atoms{skew_atoms()};
    const auto before = atoms;
    RelaxationDrvV2 driver;
    if (conflict.starts_with("isif")) driver.set_isif(std::stoi(conflict.substr(4)));
    if (conflict == "relax_atom") driver.flags.relax_atom = false;
    if (conflict == "keep_symm") driver.flags.keep_symm = true;
    if (conflict == "vib") driver.flags.vibffilter = true;
    if (conflict == "vib2") driver.flags.constraint_vib_modes = true;
    if (conflict == "lc") driver.flags.linear_constraint = true;
    if (conflict == "lf") driver.flags.linear_transform = true;
    REQUIRE_THROWS_WITH(AppsRelax::apply_position_mask("1 0 0", atoms, driver),
                        Catch::Matchers::ContainsSubstring("position-mask"));
    require_masks_equal(atoms, before);
    REQUIRE_FALSE(driver.flags.selective_dynamics);
}

TEST_CASE("Global position mask follows existing optimizer coordinates in a skew cell", "[position_mask][coordinates]") {
    const int mode = GENERATE(0, 1, 2); // Default reference Cartesian, current Cartesian, fractional.
    std::vector<Atoms> atoms{skew_atoms(1)};
    RelaxationDrvV2 driver;
    driver.control.seldyn_file.clear();
    driver.flags.cartesian = mode == 1;
    driver.flags.atomic_force_recast = mode != 2;
    AppsRelax::apply_position_mask("1 0 0", atoms, driver);
    ZeroPotential potential;
    driver.init(&potential, atoms[0]);
    driver.apply_modifiers(driver.modifiers_list);
    REQUIRE(driver.x.size() == 1);
    driver.x[0] += 0.01;
    driver.eval();
    Eigen::RowVector3d expected(0.01, 0, 0);
    if (mode == 2) expected = expected * atoms[0].cell_;
    REQUIRE((driver.atom.pos_[0] - atoms[0].pos_[0]).isApprox(expected, 1e-10));
    REQUIRE(driver.atom.cell_.isApprox(atoms[0].cell_));
}

TEST_CASE("Global position mask preserves active lattice masks", "[position_mask][coordinates]") {
    std::vector<Atoms> atoms{skew_atoms(1)};
    atoms[0].sel_dyncell_.setZero();
    atoms[0].sel_dyncell_(1, 1) = 1;
    RelaxationDrvV2 driver;
    driver.control.seldyn_file.clear();
    driver.flags.relax_cell = true;
    driver.flags.relax_cell_use_strain = false;
    AppsRelax::apply_position_mask("0 1 0", atoms, driver);
    ZeroPotential potential;
    driver.init(&potential, atoms[0]);
    REQUIRE(driver.atom.sel_dyncell_.isApprox(atoms[0].sel_dyncell_));
    REQUIRE(driver.modifiers_list.size() == 1);
    auto *filter = dynamic_cast<FilterSelDyn *>(driver.modifiers_list[0]);
    REQUIRE(filter != nullptr);
    REQUIRE(filter->ids_pass == std::vector<int>{4, 10});
}

TEST_CASE("Position mask CLI overrides settings and freezes default Cartesian components in all frames", "[position_mask][cli]") {
    Directory directory;
    const auto initial = write_cli_atoms();
    write_cli_settings("isif 2\nposition-mask 0 0 0\nrelax:selective_dynamics False\n");
    REQUIRE(run_cli() == 0);
    const auto frozen = read_cli_output();
    REQUIRE(frozen.size() == initial.size());
    for (size_t i = 0; i < initial.size(); ++i)
        for (size_t j = 0; j < initial[i].pos_.size(); ++j)
            REQUIRE(frozen[i].pos_[j].isApprox(initial[i].pos_[j], 1e-10));

    REQUIRE(run_cli({"--position-mask", "1 0 0", "--selective_dynamics", "False",
                     "--tags", "selective_dynamics", "--vals", "False"}) == 0);
    const auto relaxed = read_cli_output();
    REQUIRE(relaxed.size() == initial.size());
    for (size_t i = 0; i < initial.size(); ++i) {
        REQUIRE(relaxed[i].cell_.isApprox(initial[i].cell_));
        for (size_t j = 0; j < initial[i].pos_.size(); ++j) {
            REQUIRE(std::abs(relaxed[i].pos_[j](0) - initial[i].pos_[j](0)) > 1e-3);
            REQUIRE(relaxed[i].pos_[j](1) == Catch::Approx(initial[i].pos_[j](1)).margin(1e-10));
            REQUIRE(relaxed[i].pos_[j](2) == Catch::Approx(initial[i].pos_[j](2)).margin(1e-10));
        }
    }
}

TEST_CASE("Position mask CLI rejects settings conflicts before writing output", "[position_mask][cli][errors]") {
    Directory directory;
    write_cli_atoms();
    const std::string extra = GENERATE("isif 5\n", "isif 6\n", "isif 7\n", "relax:relax_atom False\n",
                                      "relax:keep_symm True\n", "relax:vibffilter True\n", "position-mask 1 0\n");
    write_cli_settings(extra);
    const std::vector<std::string> args = extra.starts_with("position-mask") ?
        std::vector<std::string>{} : std::vector<std::string>{"--position-mask", "1 0 0"};
    REQUIRE_THROWS_WITH(run_cli(args), Catch::Matchers::ContainsSubstring("position-mask"));
    REQUIRE_FALSE(std::filesystem::exists("output.xyz"));
}

TEST_CASE("Position mask CLI rejects active coordinate-mask files and allows disabling them", "[position_mask][cli][errors]") {
    Directory directory;
    write_cli_atoms();
    std::ofstream("SELECTIVE_DYNAMICS") << "6 0 0 0 0 0 0\n";
    write_cli_settings("isif 2\nposition-mask 0 0 0\n");
    REQUIRE_THROWS_WITH(run_cli(), Catch::Matchers::ContainsSubstring("relax:seldyn_file"));
    REQUIRE_FALSE(std::filesystem::exists("output.xyz"));
    write_cli_settings("isif 2\nposition-mask 0 0 0\nrelax:seldyn_file NONE\n");
    REQUIRE(run_cli() == 0);
}

TEST_CASE("No position mask preserves unconstrained CLI relaxation", "[position_mask][cli]") {
    Directory directory;
    const auto initial = write_cli_atoms();
    write_cli_settings("isif 2\n");
    REQUIRE(run_cli() == 0);
    const auto relaxed = read_cli_output();
    REQUIRE(relaxed.size() == initial.size());
    for (size_t i = 0; i < initial.size(); ++i) {
        REQUIRE(std::abs(relaxed[i].pos_[0](1) - initial[i].pos_[0](1)) > 1e-3);
        REQUIRE(std::abs(relaxed[i].pos_[0](2) - initial[i].pos_[0](2)) > 1e-3);
    }
}
