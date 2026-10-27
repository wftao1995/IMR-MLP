#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>

int relaxstrv2(int argc, char *argv[]);

namespace {
struct TemporaryDirectory {
    std::filesystem::path previous = std::filesystem::current_path(), path;
    TemporaryDirectory() {
        char name[] = "/tmp/imrmlp_v2_options_XXXXXX";
        const char *created = mkdtemp(name);
        if (!created) throw std::runtime_error("mkdtemp failed");
        path = created;
        std::filesystem::current_path(path);
    }
    ~TemporaryDirectory() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
};
struct CountingPotential : Potentials::AnyPotential {
    int calls = 0;
    CountingPotential() {
        cutoff = 2; cutoffsq = 4; species_count = 1;
        cutoffs = Eigen::MatrixXd::Constant(1, 1, 2);
        cutoffsqs = Eigen::MatrixXd::Constant(1, 1, 4);
    }
    void getEFS(Atoms &, AnyNeighborhoods &) override { ++calls; }
    void getEFSOMP(Atoms &) override { ++calls; }
    void getEFSOMP(Atoms &, AnyNeighborhoods &) override { ++calls; }
};
Atoms initial_atoms() {
    Atoms a;
    a.cell_ = 10 * Eigen::Matrix3d::Identity();
    a.pos_ = {{1, 1, 1}, {2.4, 1.2, 1.3}};
    a.chemmap_ = {0, 0};
    return a;
}
std::string contents(const std::string &file) {
    std::ifstream in(file);
    return {std::istreambuf_iterator<char>(in), {}};
}
int invoke(std::vector<std::string> extra = {}) {
    std::vector<std::string> args{"relax", "settings.ini", "--num_threads", "1"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    CMDOption::global_cmdopts.reset();
    return relaxstrv2(argv.size(), argv.data());
}
void write_inputs(const std::string &extra, int max_iterations = 0) {
    std::ofstream("index.map") << "H 0\n";
    GPIO io; REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> a{initial_atoms()};
    REQUIRE(io.write(a, "input.cfg", "mlpcfg"));
    std::ofstream("settings.ini") <<
        "input input.cfg\ninput-format mlpcfg\noutput output.cfg\noutput-format mlpcfg\n"
        "pot:pottype lj612\npot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 4\n"
        "isif 2\nrelax:maxiteration " << max_iterations << "\nrelax:write_traj False\n" << extra;
}
}

TEST_CASE("V2 constrained solver rejects nonconservative gradients before geometry changes", "[relax_v2_options]") {
    const std::string name = GENERATE("with_constraint_sqp", "with_constraint_nlc",
        "with_constraint_lc");
    const bool zero_dof = GENERATE(false, true);
    GPSettings s;
    s.Modify("driver", "fire");
    s.Modify("write_traj", "False");
    s.Modify(name, "True");
    RelaxationDrvV2 d(s);
    REQUIRE_NOTHROW(d.validate_options());
    d.flags.freeze_small_grad = true;
    auto a = initial_atoms(); CountingPotential potential;
    d.flags.apply_disp = true;
    d.control.max_iter = 0;
    if (zero_dof) { d.flags.relax_cell = false; d.flags.relax_atom = false; }
    d.atom = a;
    REQUIRE_THROWS_AS(d.init(&potential, a), std::invalid_argument);
    REQUIRE_THROWS_AS(d.run(), std::invalid_argument);
    REQUIRE(potential.calls == 0);
    REQUIRE(d.atom.cell_.isApprox(a.cell_));
    REQUIRE(d.atom.pos_[0].isApprox(a.pos_[0]));
    REQUIRE_FALSE(d.state_vars.converged);
}

TEST_CASE("V2 constrained solver direct entry accepts an empty coordinate subspace", "[relax_v2_options]") {
    RelaxationDrvV2 d;
    REQUIRE(d.run_alg() == 0);
    REQUIRE(d.state_vars.converged);
}

TEST_CASE("V2 supported defaults and independent coordinate filter remain valid", "[relax_v2_options]") {
    RelaxationDrvV2 d;
    REQUIRE(d.control.precond_file.empty());
    REQUIRE_NOTHROW(d.validate_options());
    d.flags.with_constraint_aul = true;
    REQUIRE_NOTHROW(d.validate_options());
    d.flags.with_constraint_aul = false;
    d.flags.linear_constraint = true;
    REQUIRE_NOTHROW(d.validate_options());
    d.flags.linear_constraint = false;
    d.control.driver = "fire";
    REQUIRE_NOTHROW(d.validate_options());
    d.buffer_fire.downhill_check = true;
    REQUIRE_NOTHROW(d.validate_options());
}

TEST_CASE("V2 missing preconditioner preserves existing CLI output even at zero iteration budget", "[relax_v2_options][cli]") {
    TemporaryDirectory directory;
    write_inputs("relax:precond_file missing.precondition\n");
    std::ofstream("output.cfg") << "preserve old result";
    REQUIRE_THROWS_AS(invoke(), std::invalid_argument);
    REQUIRE(contents("output.cfg") == "preserve old result");
    REQUIRE_FALSE(std::filesystem::exists("reltrj.xyz"));
}

TEST_CASE("Published V2 settings example uses effective V2 keys", "[relax_v2_options]") {
    GPSettings s;
    auto example = std::filesystem::path("example-inputs/relax.ini");
    if (!std::filesystem::exists(example))
        example = std::filesystem::canonical("unitest/test-data").parent_path().parent_path() / example;
    s.Load(example.string());
    auto r = s.ExtractSubSettings("relax");
    REQUIRE_FALSE(r.count("ciival"));
    REQUIRE_FALSE(r.count("symm_strain_old"));
    REQUIRE_FALSE(r.count("constraints_loops"));
    REQUIRE_FALSE(r.count("precond_file"));
    RelaxationDrvV2 d(r);
    REQUIRE_NOTHROW(d.validate_options());
    REQUIRE(d.control.constraint_iter_limit == 5);
    // Non-default overrides through the documented names must reach the driver.
    r.Modify("stiffness", "10 11 12 13 14 15");
    r.Modify("cell_factor", "2.5");
    r.Modify("constraint_loops", "7");
    RelaxationDrvV2 changed(r);
    REQUIRE(changed.control.stiffness_str == "10 11 12 13 14 15");
    REQUIRE(changed.control.cell_factor == 2.5);
    REQUIRE(changed.control.constraint_iter_limit == 7);
}

TEST_CASE("Generated V2 example contains recognized public input keys", "[relax_v2_options][cli]") {
    TemporaryDirectory directory;
    REQUIRE(invoke({"--create-input"}) == 0);
    GPSettings settings;
    settings.Load("relax_simple_example.ini");
    REQUIRE(settings["input"] == "POSCAR");
    REQUIRE(settings["output"] == "CONTCAR");
    REQUIRE(settings["index"] == "index.map");
    REQUIRE(settings["relax:index"] == "index.map");
    REQUIRE_FALSE(settings.count("inut-file"));
}

TEST_CASE("V2 example generation reports output failures", "[relax_v2_options][cli]") {
    TemporaryDirectory directory;
    SECTION("output cannot be opened") {
        std::filesystem::create_directory("relax_simple_example.ini");
        REQUIRE(invoke({"--create-input"}) == 1);
        REQUIRE(std::filesystem::is_directory("relax_simple_example.ini"));
    }
    SECTION("output cannot be written") {
        if (!std::filesystem::exists("/dev/full")) SKIP("/dev/full is unavailable");
        std::filesystem::create_symlink("/dev/full", "relax_simple_example.ini");
        REQUIRE(invoke({"--create-input"}) == 1);
    }
}

TEST_CASE("V2 tags apply before external seldyn masks and override direct options", "[relax_v2_options][cli][seldyn]") {
    TemporaryDirectory directory;
    write_inputs("seldyn frozen.mask\nrelax:gtol 1e-5\n", 300);
    std::ofstream("frozen.mask") <<
        "1\n1 2\n1 1 1\n1 1 1\n1 1 1\n0 0 0 0\n1 0 0 0\n";
    const auto initial = initial_atoms();
    auto read_output = [] {
        GPIO io;
        REQUIRE(io.readmap("index.map"));
        std::vector<Atoms> result;
        REQUIRE(io.read(result, "output.cfg", "mlpcfg"));
        REQUIRE(result.size() == 1);
        return result[0];
    };

    REQUIRE(invoke({"--selective_dynamics", "True"}) == 0);
    const auto direct = read_output();
    REQUIRE(invoke({"--tags", "selective_dynamics", "--vals", "True"}) == 0);
    const auto tagged = read_output();
    REQUIRE(direct.size() == initial.size());
    REQUIRE(tagged.size() == initial.size());
    for (int i = 0; i < initial.size(); ++i) {
        REQUIRE(direct.pos_[i].isApprox(initial.pos_[i], 1e-9));
        REQUIRE(tagged.pos_[i].isApprox(direct.pos_[i], 1e-9));
    }

    REQUIRE(invoke({"--selective_dynamics", "True", "--tags", "selective_dynamics", "--vals", "False"}) == 0);
    const auto released = read_output();
    REQUIRE(released.size() == initial.size());
    REQUIRE(released.cell_.isApprox(initial.cell_, 1e-9));
    for (int i = 0; i < initial.size(); ++i)
        REQUIRE((released.pos_[i] - initial.pos_[i]).norm() > 1e-4);
}

TEST_CASE("V2 diagonal file works through relax CLI and mismatches preserve output", "[relax_v2_options][cli][relax_precond_file]") {
    TemporaryDirectory directory;
    std::ofstream("diagonal.txt") << "6 2 2 2 2 2 2\n";
    write_inputs("relax:driver fire\nrelax:precond_file diagonal.txt\nrelax:fire:ts_init 0.01\n", 3);
    // A finite improving three-step trajectory reports its unconverged budget.
    REQUIRE(invoke() == 2);
    GPIO io; REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> result;
    REQUIRE(io.read(result, "output.cfg", "mlpcfg"));
    REQUIRE(result.size() == 1);
    const auto initial = initial_atoms();
    const double radius = (initial.pos_[1] - initial.pos_[0]).norm();
    const double inv6 = std::pow(radius, -6);
    REQUIRE(result[0].energy_ < 4 * (inv6 * inv6 - inv6));
    const auto accepted = contents("output.cfg");
    std::ofstream("diagonal.txt") << "3 1 1 1\n";
    REQUIRE_THROWS_AS(invoke(), std::invalid_argument);
    REQUIRE(contents("output.cfg") == accepted);
}
