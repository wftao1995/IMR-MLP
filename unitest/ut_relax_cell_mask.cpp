#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>

int relaxstrv2(int argc, char *argv[]);

namespace {
struct Directory {
    std::filesystem::path old = std::filesystem::current_path(), path;
    Directory() {
        char name[] = "/tmp/imrmlp_cell_mask_XXXXXX";
        auto *created = mkdtemp(name);
        if (!created) throw std::runtime_error("mkdtemp failed");
        path = created;
        std::filesystem::current_path(path);
    }
    ~Directory() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(old);
        std::filesystem::remove_all(path);
    }
};
Atoms atoms() {
    Atoms a;
    a.cell_ = 10 * Eigen::Matrix3d::Identity();
    a.pos_ = {{1, 1, 1}, {2.4, 1.2, 1.3}};
    a.chemmap_ = {0, 0};
    a.sel_dyncell_.setOnes();
    return a;
}
struct VolumePotential : Potentials::AnyPotential {
    int calls = 0;
    VolumePotential() {
        cutoff = 2; cutoffsq = 4; species_count = 1;
        cutoffs = Eigen::MatrixXd::Constant(1, 1, 2);
        cutoffsqs = Eigen::MatrixXd::Constant(1, 1, 4);
        r_mins = Eigen::MatrixXd::Zero(1, 1);
    }
    void evaluate(Atoms &a) {
        ++calls;
        a.energy_ = a.get_volume();
        a.forces_.assign(a.size(), Eigen::Vector3d::Zero());
        a.stress_ = -a.energy_ * Eigen::Matrix3d::Identity();
    }
    void getEFS(Atoms &a, AnyNeighborhoods &) override { evaluate(a); }
    void getEFSOMP(Atoms &a) override { evaluate(a); }
    void getEFSOMP(Atoms &a, AnyNeighborhoods &) override { evaluate(a); }
};
GPSettings settings(const std::string &mask = "") {
    GPSettings s;
    s.Modify("relax_cell", "True"); s.Modify("relax_atom", "False");
    s.Modify("selective_dynamics", "True"); s.Modify("write_traj", "False");
    s.Modify("driver", "gd"); s.Modify("maxiteration", "1");
    s.Modify("seldyn_file", "");
    if (!mask.empty()) s.Modify("strain_mask", mask);
    return s;
}
std::vector<int> selected(RelaxationDrvV2 &d) {
    for (auto *modifier : d.modifiers_list)
        if (auto *filter = dynamic_cast<FilterSelDyn *>(modifier)) return filter->ids_pass;
    throw std::runtime_error("missing selective dynamics filter");
}
void write_input() {
    std::ofstream("index.map") << "H 0\n";
    GPIO io; REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> a{atoms(), atoms()};
    REQUIRE(io.write(a, "input.cfg", "mlpcfg"));
}
void write_settings(const std::string &extra = "") {
    std::istringstream text(
        "input input.cfg\ninput-format mlpcfg\noutput output.cfg\noutput-format mlpcfg\n"
        "pot:pottype lj612\npot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 4\n"
        "isif 6\nrelax:gtol 1e-30\nrelax:maxiteration 1\nrelax:write_traj False\n" + extra);
    std::map<std::string, std::string> values;
    for (std::string line; std::getline(text, line); ) {
        const auto i = line.find(' ');
        if (i != std::string::npos) values[line.substr(0, i)] = line.substr(i + 1);
    }
    std::ofstream out("relax.ini");
    for (const auto &[key, value] : values) out << key << ' ' << value << '\n';
}
int run(std::vector<std::string> extra = {}) {
    std::vector<std::string> args{"relax", "relax.ini", "--num_threads", "1"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<char *> argv;
    for (auto &a : args) argv.push_back(a.data());
    CMDOption::global_cmdopts.reset();
    return relaxstrv2(argv.size(), argv.data());
}
std::string bytes(const std::string &path) {
    std::ifstream in(path); return {std::istreambuf_iterator<char>(in), {}};
}
}

TEST_CASE("Default symmetric strain cell mask deterministically frees all six coordinates", "[cell_mask]") {
    Directory directory;
    auto a = atoms(); VolumePotential p;
    RelaxationDrvV2 d(settings());
    for (auto v : d.symm_strain_mask) REQUIRE(v == 1.0);
    d.init(&p, a);
    REQUIRE(selected(d) == std::vector<int>{0, 1, 2, 3, 4, 5});
    REQUIRE(d.x.size() == 6);
    REQUIRE(d.run() != 0);
    REQUIRE(p.calls > 0);
    REQUIRE(d.atom.get_volume() < a.get_volume());
    // Reinitialization does not carry a previous frame's reduced mask state.
    d.init(&p, a);
    REQUIRE(selected(d).size() == 6);
}

TEST_CASE("Cell masks preserve positive selection semantics and map each coordinate layout", "[cell_mask]") {
    Directory directory;
    const int mode = GENERATE(0, 2, 3); // symmetric, raw lattice, full strain
    auto s = settings(mode < 2 ? "0 -2 0 0 0 +0.25" : "0 -2 0 0 +0.25 0 0 0 0");
    if (mode == 1) { s.Modify("symm_strain", "False"); s.Modify("symm_strain_oldstyle", "True"); }
    if (mode == 2) s.Modify("cell_strain", "False");
    if (mode == 3) s.Modify("symm_strain", "False");
    RelaxationDrvV2 d(s); VolumePotential p; auto a = atoms();
    d.init(&p, a);
    REQUIRE(selected(d) == std::vector<int>{mode < 2 ? 1 : 4});
    d.run();
    REQUIRE(d.x.size() == 1);
    if (mode < 2) REQUIRE(d.symm_strain_mask[1] == 1);
    else REQUIRE(d.cellmask(1, 1) == 1);
}

TEST_CASE("Cell-mask parsing rejects malformed values without changing previous masks", "[cell_mask][errors]") {
    RelaxationDrvV2 d;
    d.configure_cell_mask("1 0 0 0 0 0");
    const auto previous = d.symm_strain_mask;
    const auto text = d.control.lattice_mask_str;
    const std::string invalid = GENERATE("", "1 0", "1 0 0 0 0 0 0", "1 0 0 0 0 0 0 0 0 0",
        "nan 0 0 0 0 0", "1 inf 0 0 0 0", "1 -inf 0 0 0 0", "1 0junk 0 0 0 0",
        "1 1e999 0 0 0 0", "1 x 0 0 0 0");
    REQUIRE_THROWS_AS(d.configure_cell_mask(invalid), std::invalid_argument);
    REQUIRE(d.symm_strain_mask == previous);
    REQUIRE(d.control.lattice_mask_str == text);
}

TEST_CASE("Cell masks validate the final ISIF mode before replacing geometry", "[cell_mask][errors]") {
    const int isif = GENERATE(2, 7, 8, 13);
    VolumePotential p; auto a = atoms();
    RelaxationDrvV2 d(settings("1 0 0 0 0 0"));
    d.atom = a; d.atom.pos_[0](0) = 5;
    d.set_isif(isif);
    REQUIRE_THROWS_WITH(d.init(&p, a), Catch::Matchers::ContainsSubstring("Cell mask"));
    REQUIRE(p.calls == 0);
    REQUIRE(d.atom.pos_[0](0) == 5);
    REQUIRE_FALSE(d.state_vars.converged);
}

TEST_CASE("Cell mask rejects incompatible transform and full-coordinate file", "[cell_mask][errors]") {
    Directory directory;
    const std::string extra = GENERATE("vib", "lf", "symmetry", "file");
    RelaxationDrvV2 d(settings("1 0 0 0 0 0"));
    if (extra == "vib") d.flags.constraint_vib_modes = true;
    if (extra == "lf") d.flags.linear_transform = true;
    if (extra == "symmetry") d.flags.keep_symm = true;
    if (extra == "file") {
        std::ofstream("mask") << "6 1 1 1 1 1 1\n";
        d.control.seldyn_file = "mask";
    }
    VolumePotential p; auto a = atoms();
    REQUIRE_THROWS_WITH(d.init(&p, a), Catch::Matchers::ContainsSubstring("Cell mask"));
    REQUIRE(p.calls == 0);
}



TEST_CASE("V2 FIRE rejects before initialization evaluation or early convergence", "[cell_mask][fire][errors]") {
    const int integration = GENERATE(-1, 1, 2);
    const bool zero_dof = GENERATE(false, true);
    auto s = settings(); s.Modify("driver", "fire");
    RelaxationDrvV2 d(s); VolumePotential p; auto a = atoms();
    if (zero_dof) { d.flags.relax_cell = false; d.flags.relax_atom = false; }
    d.buffer_fire.mdint = integration;
    d.flags.apply_disp = true; d.control.max_iter = 0;
    d.atom = a;
    REQUIRE_THROWS_WITH(d.init(&p, a), Catch::Matchers::ContainsSubstring("fire:mdalgo=0"));
    REQUIRE_THROWS_WITH(d.run(), Catch::Matchers::ContainsSubstring("fire:mdalgo=0"));
    REQUIRE_THROWS_AS(d.run_fire(), std::invalid_argument);
    REQUIRE(p.calls == 0);
    REQUIRE(d.atom.pos_[0].isApprox(a.pos_[0]));
    REQUIRE_FALSE(d.state_vars.converged);
}

TEST_CASE("Default FIRE still performs a cell update", "[cell_mask][fire]") {
    Directory directory;
    auto s = settings(); s.Modify("driver", "fire");
    RelaxationDrvV2 d(s); VolumePotential p; auto a = atoms();
    d.init(&p, a); d.run();
    REQUIRE(d.state_vars.curr_iter == 1);
    REQUIRE(d.atom.get_volume() < a.get_volume());
}

TEST_CASE("V2 missing constraint file fails before input or output access", "[cell_mask][cli][errors]") {
    Directory directory;
    const bool from_settings = GENERATE(false, true);
    write_settings(from_settings ? "file-constraints missing.json\n" : "");
    std::ofstream("output.cfg") << "preserve";
    REQUIRE_THROWS_WITH(run(from_settings ? std::vector<std::string>{} :
        std::vector<std::string>{"--file-constraints", "missing.json"}),
        Catch::Matchers::ContainsSubstring("cannot open"));
    REQUIRE(bytes("output.cfg") == "preserve");
}

TEST_CASE("V2 cell mask CLI validates after ISIF and tags and preserves output on errors", "[cell_mask][cli][errors]") {
    Directory directory; write_input();
    const std::string extra = GENERATE("cell-mask 1 0 0 0 0 0 0 0 0\n",
        "cell-mask 1 nan 0 0 0 0\n", "cell-mask 1 0junk 0 0 0 0\n",
        "cell-mask 1 0 0 0 0 0\nisif 2\n", "cell-mask 1 0 0 0 0 0\nisif 7\n",
        "relax:driver fire\nrelax:fire:mdalgo 1\n");
    write_settings(extra);
    std::ofstream("output.cfg") << "preserve";
    REQUIRE_THROWS(run());
    REQUIRE(bytes("output.cfg") == "preserve");
}

TEST_CASE("V2 cell-mask selects real cell motion in batch and stream modes", "[cell_mask][cli]") {
    Directory directory; write_input();
    const int mode = GENERATE(0, 1, 2); // symmetric6/raw9/strain9
    const bool sequential = GENERATE(false, true);
    const std::string mask = mode == 0 ? "0.2 0 0 0 0 0" : "0.2 0 0 0 0 0 0 0 0";
    write_settings("cell-mask " + mask + "\nrelax:pressure 10\n" +
        (mode == 0 ? std::string{} : mode == 1 ? "relax:cell_strain False\n" : "relax:symm_strain False\n"));
    std::vector<std::string> args;
    if (sequential) args.push_back("--seq-read");
    REQUIRE(run(args) == 2);
    GPIO io; REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> result; REQUIRE(io.read(result, "output.cfg", "mlpcfg"));
    REQUIRE(result.size() == 2);
    for (const auto &a : result) {
        REQUIRE(std::abs(a.cell_(0, 0) - 10) > 1e-8);
        REQUIRE(a.cell_(1, 1) == Catch::Approx(10).margin(1e-8));
        REQUIRE(a.cell_(2, 2) == Catch::Approx(10).margin(1e-8));
        REQUIRE(a.cell_(0, 1) == Catch::Approx(0).margin(1e-8));
    }
}

TEST_CASE("V2 global cell mask wins over driver aliases and checks final tag overrides", "[cell_mask][cli]") {
    Directory directory; write_input();
    write_settings("relax:relax_cell False\nrelax:strain_mask broken\nrelax:lattice_mask broken\n");
    REQUIRE(run({"--cell-mask", "0 0 0 0 0 0"}) == 0); // ISIF 6 wins over relax_cell
    const auto before = bytes("output.cfg");
    REQUIRE_THROWS_WITH(run({"--cell-mask", "0 0 0 0 0 0", "--tags", "symm_strain", "--vals", "False"}),
                        Catch::Matchers::ContainsSubstring("9 components"));
    REQUIRE(bytes("output.cfg") == before);
}
