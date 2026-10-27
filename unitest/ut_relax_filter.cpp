#include <catch2/catch_all.hpp>
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
        char pattern[] = "/tmp/imrmlp_relax_filter_XXXXXX";
        const char *directory = mkdtemp(pattern);
        if (!directory) throw std::runtime_error("Cannot create filter test directory");
        path = directory;
        std::filesystem::current_path(path);
    }
    ~Directory() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
};

Atoms make_atoms(int count = 2) {
    Atoms atom;
    atom.cell_ << 10, 1, 0.5, 0, 11, 0.7, 0, 0, 12;
    atom.pos_ = {{2, 2, 2}, {3.4, 2.3, 2.2}};
    atom.pos_.resize(count);
    atom.chemmap_.assign(count, 0);
    atom.sel_dyncell_.setOnes();
    return atom;
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

std::string mask_text(int size, const std::vector<int> &selected) {
    std::ostringstream output;
    output << size;
    for (int i = 0; i < size; ++i)
        output << ' ' << (std::find(selected.begin(), selected.end(), i) != selected.end() ? 1 : 0);
    return output.str();
}

FilterSelDyn *filter(RelaxationDrvV2 &driver) {
    REQUIRE(driver.modifiers_list.size() == 1);
    auto *selected = dynamic_cast<FilterSelDyn *>(driver.modifiers_list[0]);
    REQUIRE(selected != nullptr);
    return selected;
}

void write_settings(const std::string &extra = "") {
    std::ofstream("relax.ini") <<
        "input input.xyz\ninput-format extxyz\noutput output.xyz\noutput-format extxyz\n"
        "pot:pottype lj612\npot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 4\n"
        "relax:gtol 1e-5\nrelax:maxiteration 300\nrelax:write_traj False\nisif 2\n" << extra;
}

int run_cli(std::vector<std::string> extra = {}) {
    std::vector<std::string> args{"relax", "relax.ini", "--num_threads", "1"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    CMDOption::global_cmdopts.reset();
    return relaxstrv2(argv.size(), argv.data());
}

std::vector<Atoms> write_atoms(bool varying_count = false) {
    std::ofstream("index.map") << "H 0\n";
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> atoms{make_atoms(), make_atoms(varying_count ? 1 : 2)};
    atoms[1].pos_[0](0) += 0.1;
    REQUIRE(io.write(atoms, "input.xyz", "extxyz"));
    return atoms;
}

std::vector<Atoms> read_output() {
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> atoms;
    REQUIRE(io.read(atoms, "output.xyz", "extxyz"));
    return atoms;
}
}

TEST_CASE("Full coordinate mask reads exact binary vectors including an empty selection", "[relax_filter][parser]") {
    std::istringstream input(" 6\n1 0\t0 1 0 1\n\n");
    REQUIRE(FilterSelDyn::read_mask(input, 6) == std::vector<int>{0, 3, 5});
    std::istringstream frozen("3 0 0 0");
    REQUIRE(FilterSelDyn::read_mask(frozen, 3).empty());
    std::istringstream empty("0\n");
    REQUIRE(FilterSelDyn::read_mask(empty, 0).empty());
}

TEST_CASE("Full coordinate mask rejects malformed and partial files", "[relax_filter][parser][errors]") {
    const std::string text = GENERATE("", " ", "x", "-1", "2 1 0", "4 1 0 1 0", "3.0 1 0 0",
        "9999999999999999999999999", "3", "3 1 0", "3 1 0 2", "3 1 0 -1", "3 1 0 nan",
        "3 1 0 inf", "3 1 0 0.5", "3 1 0 1x", "3 1 0 1 0", "3 1 0 1 garbage", "3 1 0 1 # comment");
    std::istringstream input(text);
    REQUIRE_THROWS_WITH(FilterSelDyn::read_mask(input, 3),
                        Catch::Matchers::ContainsSubstring("optimization-coordinate mask"));
}

TEST_CASE("Selective dynamics commits validated indices once and preserves them on failure", "[relax_filter][errors]") {
    RelaxationDrvV2 driver;
    driver.x.setZero(3);
    FilterSelDyn selected;
    std::vector<int> first{0, 2};
    selected.init(&driver, &first);
    selected.init(&driver, &first);
    REQUIRE(selected.ids_pass == first);
    const std::vector<int> invalid = GENERATE(std::vector<int>{-1}, std::vector<int>{3}, std::vector<int>{0, 0});
    auto value = invalid;
    REQUIRE_THROWS(selected.init(&driver, &value));
    REQUIRE(selected.ids_pass == first);
}

TEST_CASE("Full coordinate masks use the actual cell and atomic optimizer dimensions", "[relax_filter][driver]") {
    Directory directory;
    const int offset = GENERATE(0, 1, 6, 9);
    const bool relax_atom = GENERATE(false, true);
    RelaxationDrvV2 driver;
    driver.flags.selective_dynamics = true;
    driver.flags.fix_shape = offset == 1;
    driver.flags.relax_cell = offset == 6 || offset == 9;
    driver.flags.relax_cell_use_strain = offset != 9;
    driver.flags.relax_atom = relax_atom;
    auto atom = make_atoms();
    ZeroPotential potential;
    const int size = offset + (relax_atom ? 6 : 0);
    const std::vector<int> selected = size ? std::vector<int>{size - 1} : std::vector<int>{};
    std::ofstream("SELECTIVE_DYNAMICS") << mask_text(size, selected);
    driver.init(&potential, atom);
    REQUIRE(driver.x.size() == size);
    REQUIRE(driver.xposoffset == offset);
    REQUIRE(filter(driver)->ids_pass == selected);
    driver.apply_modifiers(driver.modifiers_list);
    REQUIRE(driver.x.size() == selected.size());
    REQUIRE(driver.prec_scaling.size() == selected.size());
}

TEST_CASE("Full coordinate masks preserve the optimizer position-coordinate semantics", "[relax_filter][coordinates]") {
    Directory directory;
    const int mode = GENERATE(0, 1, 2); // reference Cartesian, current Cartesian, fractional
    auto atom = make_atoms(1);
    RelaxationDrvV2 driver;
    driver.flags.selective_dynamics = true;
    driver.flags.cartesian = mode == 1;
    driver.flags.atomic_force_recast = mode != 2;
    std::ofstream("SELECTIVE_DYNAMICS") << "3 1 0 0";
    ZeroPotential potential;
    driver.init(&potential, atom);
    driver.apply_modifiers(driver.modifiers_list);
    REQUIRE(driver.x.size() == 1);
    driver.x[0] += 0.01;
    driver.eval();
    Eigen::RowVector3d expected(0.01, 0, 0);
    if (mode == 2) expected = expected * atom.cell_;
    REQUIRE((driver.atom.pos_[0] - atom.pos_[0]).isApprox(expected, 1e-10));
    REQUIRE(driver.atom.cell_.isApprox(atom.cell_));
}

TEST_CASE("A fixed shape scale can be masked without touching reduced atomic coordinates", "[relax_filter][driver]") {
    Directory directory;
    const bool relax_atom = GENERATE(false, true);
    const bool movable_atom = GENERATE(false, true);
    const bool displace = GENERATE(false, true);
    auto atom = make_atoms(1);
    RelaxationDrvV2 driver;
    driver.flags.selective_dynamics = true;
    driver.flags.fix_shape = true;
    driver.flags.relax_atom = relax_atom;
    driver.flags.apply_disp = displace;
    const std::vector<int> selected = relax_atom && movable_atom ? std::vector<int>{1} : std::vector<int>{};
    std::ofstream("SELECTIVE_DYNAMICS") << mask_text(relax_atom ? 4 : 1, selected);
    ZeroPotential potential;
    driver.init(&potential, atom);
    REQUIRE_NOTHROW(driver.run());
    REQUIRE(driver.state_vars.converged);
    REQUIRE(driver.atom.cell_.isApprox(atom.cell_, 1e-12));
    if (selected.empty() || !displace)
        REQUIRE(driver.atom.pos_[0].isApprox(atom.pos_[0], 1e-12));
    REQUIRE(driver.atom.pos_[0].allFinite());
}

TEST_CASE("Driver validates default and configured mask files before replacing state", "[relax_filter][driver][errors]") {
    Directory directory;
    const std::string path = GENERATE("SELECTIVE_DYNAMICS", "custom.mask");
    GPSettings settings;
    settings.Modify("selective_dynamics", "True");
    if (path != "SELECTIVE_DYNAMICS") settings.Modify("seldyn_file", path);
    RelaxationDrvV2 driver(settings);
    auto atom = make_atoms(1);
    ZeroPotential potential;
    std::ofstream(path) << "3 1 0 0";
    driver.init(&potential, atom);
    const auto old_x = driver.x;
    const auto *old_filter = filter(driver);
    driver.state_vars.converged = true;
    driver.state_vars.curr_iter = 12;
    driver.has_checkpoint = true;
    const std::string invalid = GENERATE("4 1 0 0 0", "3 1 0", "3 1 0 2", "3 1 0 0 extra");
    std::ofstream(path) << invalid;
    auto other = atom;
    other.pos_[0](0) += 2;
    REQUIRE_THROWS_WITH(driver.init(&potential, other), Catch::Matchers::ContainsSubstring("optimization-coordinate mask"));
    REQUIRE_FALSE(driver.state_vars.converged);
    REQUIRE_FALSE(driver.has_checkpoint);
    REQUIRE(driver.state_vars.curr_iter == 0);
    REQUIRE(driver.x.isApprox(old_x));
    REQUIRE(filter(driver) == old_filter);
    REQUIRE(filter(driver)->ids_pass == std::vector<int>{0});
    REQUIRE(driver.atom.pos_[0].isApprox(atom.pos_[0]));
}

TEST_CASE("File masks reject conflicting coordinate modifiers before consuming their files", "[relax_filter][driver][errors]") {
    Directory directory;
    const std::string conflict = GENERATE("keep_symm", "vib", "vib2", "lc", "lf");
    RelaxationDrvV2 driver;
    driver.flags.selective_dynamics = true;
    driver.flags.keep_symm = conflict == "keep_symm";
    driver.flags.vibffilter = conflict == "vib";
    driver.flags.constraint_vib_modes = conflict == "vib2";
    driver.flags.linear_constraint = conflict == "lc";
    driver.flags.linear_transform = conflict == "lf";
    std::ofstream("SELECTIVE_DYNAMICS") << "3 1 0 0";
    auto atom = make_atoms(1);
    ZeroPotential potential;
    REQUIRE_THROWS_WITH(driver.init(&potential, atom), Catch::Matchers::ContainsSubstring("incompatible"));
    REQUIRE(driver.modifiers_list.empty());
}

TEST_CASE("Filter CLI enables SD and wins over settings and tags in every frame", "[relax_filter][cli]") {
    Directory directory;
    const auto initial = write_atoms();
    std::ofstream("frozen.mask") << "6 0 0 0 0 0 0";
    std::ofstream("moving.mask") << "6 1 0 0 1 0 0";
    write_settings("filter frozen.mask\nrelax:selective_dynamics False\nrelax:seldyn_file missing.mask\n");
    REQUIRE(run_cli() == 0);
    const auto frozen = read_output();
    REQUIRE(frozen.size() == initial.size());
    for (size_t i = 0; i < initial.size(); ++i)
        for (size_t j = 0; j < initial[i].size(); ++j)
            REQUIRE(frozen[i].pos_[j].isApprox(initial[i].pos_[j], 1e-10));
    REQUIRE(run_cli({"--filter", "moving.mask", "--selective_dynamics", "False",
                     "--tags", "seldyn_file", "selective_dynamics", "--vals", "missing-again.mask", "False"}) == 0);
    const auto relaxed = read_output();
    REQUIRE(relaxed.size() == initial.size());
    for (size_t i = 0; i < initial.size(); ++i)
        for (size_t j = 0; j < initial[i].size(); ++j) {
            REQUIRE(std::abs(relaxed[i].pos_[j](0) - initial[i].pos_[j](0)) > 1e-3);
            REQUIRE(relaxed[i].pos_[j](1) == Catch::Approx(initial[i].pos_[j](1)).margin(1e-10));
            REQUIRE(relaxed[i].pos_[j](2) == Catch::Approx(initial[i].pos_[j](2)).margin(1e-10));
        }
}

TEST_CASE("Filter CLI rejects missing files and conflicting masks before writing results", "[relax_filter][cli][errors]") {
    Directory directory;
    write_atoms();
    std::ofstream("valid.mask") << "6 0 0 0 0 0 0";
    const std::string extra = GENERATE("filter missing.mask\n", "filter \n", "position-mask 1 0 0\n",
        "seldyn missing.seldyn\n", "cell-mask 1 1 1 1 1 1\n", "relax:strain_mask 1 1 1 1 1 1\n",
        "relax:lattice_mask 1 1 1 1 1 1\n", "relax:keep_symm True\n", "relax:vibffilter True\n",
        "relax:with_constraint_vib True\n", "relax:with_linear_constraint True\n", "relax:with_linear_transform True\n");
    write_settings(extra);
    const auto args = extra.starts_with("filter") ? std::vector<std::string>{} :
        std::vector<std::string>{"--filter", "valid.mask"};
    REQUIRE_THROWS_WITH(run_cli(args), Catch::Matchers::ContainsSubstring("filter"));
    REQUIRE_FALSE(std::filesystem::exists("output.xyz"));
}

TEST_CASE("Filter CLI rejects embedded frozen atom flags", "[relax_filter][cli][errors]") {
    Directory directory;
    write_atoms();
    auto atom = make_atoms();
    atom.sel_dynpos_.assign(atom.size(), Eigen::RowVector3d(1, 0, 1));
    GPIO io;
    io.force_write_all_to_one = true;
    REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> atoms{atom};
    REQUIRE(io.write(atoms, "POSCAR", "poscar"));
    std::ofstream("valid.mask") << "6 1 0 0 1 0 0";
    write_settings("filter valid.mask\n");
    REQUIRE_THROWS_WITH(run_cli({"--input", "POSCAR", "--input-format", "poscar"}),
                        Catch::Matchers::ContainsSubstring("embedded"));
    REQUIRE_FALSE(std::filesystem::exists("output.xyz"));
}

TEST_CASE("Filter CLI revalidates a mask against every frame size", "[relax_filter][cli][errors]") {
    Directory directory;
    write_atoms(true);
    std::ofstream("valid.mask") << "6 0 0 0 0 0 0";
    write_settings("filter valid.mask\n");
    REQUIRE(run_cli() == 1);
    const auto completed = read_output();
    REQUIRE(completed.size() == 1); // Existing CLI policy keeps successful frames, with failure status.
    REQUIRE(completed[0].pos_.size() == 2);
}

TEST_CASE("Legacy default and named mask files retain their behavior with strict validation", "[relax_filter][cli]") {
    Directory directory;
    const auto initial = write_atoms();
    const std::string path = GENERATE("SELECTIVE_DYNAMICS", "custom.mask");
    std::ofstream(path) << "6 0 0 0 0 0 0";
    write_settings("relax:selective_dynamics True\nrelax:seldyn_file " + path + "\n");
    REQUIRE(run_cli() == 0);
    const auto frozen = read_output();
    REQUIRE(frozen.size() == initial.size());
    REQUIRE(frozen[0].pos_[0].isApprox(initial[0].pos_[0], 1e-10));
    std::filesystem::remove("output.xyz");
    std::ofstream(path) << "6 0 0 0 0 0 0 extra";
    REQUIRE(run_cli() == 1);
    REQUIRE_FALSE(std::filesystem::exists("output.xyz"));
    std::filesystem::remove(path);
    REQUIRE(run_cli() == 0); // Existing missing-file fallback is deliberately retained.
    REQUIRE(std::abs(read_output()[0].pos_[0](1) - initial[0].pos_[0](1)) > 1e-3);
}
