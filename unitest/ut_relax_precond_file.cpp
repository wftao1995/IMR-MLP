#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include <filesystem>
#include <fstream>

namespace {
struct Directory {
    std::filesystem::path previous = std::filesystem::current_path(), path;
    Directory() {
        char name[] = "/tmp/imrmlp_precond_file_XXXXXX";
        const char *created = mkdtemp(name);
        if (!created) throw std::runtime_error("mkdtemp failed");
        path = created;
        std::filesystem::current_path(path);
    }
    ~Directory() {
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
};
Atoms initial() {
    Atoms a;
    a.cell_ = 10 * Eigen::Matrix3d::Identity();
    a.pbc_ = {false, false, false};
    a.pos_ = {{1, 2, 3}};
    a.chemmap_ = {0};
    return a;
}
struct Harmonic : Potentials::AnyPotential {
    int calls = 0;
    void evaluate(Atoms &a) {
        ++calls;
        a.energy_ = 0.5 * a.pos_[0].squaredNorm();
        a.forces_ = {-a.pos_[0]};
        a.stress_.setZero();
    }
    void getEFSOMP(Atoms &a) override { evaluate(a); }
    void getEFS(Atoms &a, AnyNeighborhoods &) override { evaluate(a); }
};
GPSettings settings() {
    GPSettings s;
    s.Modify("driver", "fire");
    s.Modify("precond_file", "diagonal.txt");
    s.Modify("write_traj", "False");
    s.Modify("cartesian", "True");
    s.Modify("atomic_force_recast", "False");
    s.Modify("relax_cell", "False");
    s.Modify("maxiteration", "1");
    s.Modify("stepmax", "100");
    s.Modify("fire:ts_init", "0.1");
    return s;
}
}

TEST_CASE("V2 file diagonal controls actual FIRE displacement and reloads on init", "[relax_precond_file]") {
    Directory directory;
    std::ofstream("diagonal.txt") << "3\n1 2 4\n";
    auto a = initial(); Harmonic p;
    RelaxationDrvV2 d(settings());
    d.init(&p, a);
    REQUIRE(d.prec_scaling.isApprox(Eigen::Vector3d(1, 2, 4)));
    d.run();
    // One zero-velocity Euler kick: x_new = x - dt^2 * gradient / scaling.
    REQUIRE(d.atom.pos_[0][0] == Catch::Approx(0.99));
    REQUIRE(d.atom.pos_[0][1] == Catch::Approx(1.99));
    REQUIRE(d.atom.pos_[0][2] == Catch::Approx(2.9925));
    REQUIRE(d.f == Catch::Approx(0.5 * d.atom.pos_[0].squaredNorm()));
    std::ofstream("diagonal.txt") << "3 2 4 8\n";
    d.init(&p, a);
    d.run();
    REQUIRE(d.atom.pos_[0][0] == Catch::Approx(0.995));
    REQUIRE(d.atom.pos_[0][1] == Catch::Approx(1.995));
    REQUIRE(d.atom.pos_[0][2] == Catch::Approx(2.99625));
}

TEST_CASE("V2 file scaling follows masks and linear coordinate transforms", "[relax_precond_file]") {
    Directory directory;
    std::ofstream("diagonal.txt") << "3 2 8 18\n";
    auto a = initial(); Harmonic p;
    auto s = settings();
    SECTION("selection consumes full count before projecting") {
        s.Modify("selective_dynamics", "True");
        a.sel_dynpos_ = {{1, 0, 1}};
        RelaxationDrvV2 d(s); d.init(&p, a);
        REQUIRE(d.prec_scaling.isApprox(Eigen::Vector2d(2, 18)));
        d.run();
        REQUIRE(d.atom.pos_[0][1] == 2);
        REQUIRE(d.atom.pos_[0][0] == Catch::Approx(0.995));
        REQUIRE(d.atom.pos_[0][2] == Catch::Approx(3 - 0.01 * 3 / 18));
    }
    SECTION("LF uses diagonal of inverse-transpose metric") {
        // y=A*x; H_y=A^-T diag(d) A^-1. Off-diagonal coupling is deliberately discarded.
        std::ofstream("lf.txt") << "3 3\n2 1 0\n0 1 0\n0 0 3\n0 0 0\n";
        s.Modify("with_linear_transform", "True"); s.Modify("lf_file", "lf.txt");
        RelaxationDrvV2 d(s); d.init(&p, a);
        REQUIRE(d.prec_scaling.isApprox(Eigen::Vector3d(0.5, 8.5, 2)));
    }
    SECTION("LC uses diagonal of nullspace projected metric") {
        std::ofstream("lc.txt") << "1 3\n1 1 0\n3\n";
        s.Modify("with_linear_constraint", "True"); s.Modify("lc_file", "lc.txt");
        RelaxationDrvV2 d(s); d.init(&p, a);
        REQUIRE(d.prec_scaling.size() == 2);
        REQUIRE(d.prec_scaling[0] == Catch::Approx(5));
        REQUIRE(d.prec_scaling[1] == Catch::Approx(18));
    }
}

TEST_CASE("V2 rejects invalid diagonal files before geometry replacement or evaluation", "[relax_precond_file]") {
    Directory directory;
    const std::string text = GENERATE("", "-1", "2 1 1", "3 1 1", "3 1 0 1",
        "3 1 -1 1", "3 1 nan 1", "3 1 inf 1", "3 1 1 1 extra", "3 1 1 1 2");
    std::ofstream("diagonal.txt") << text;
    auto a = initial(); Harmonic p;
    RelaxationDrvV2 d(settings());
    d.atom = a; d.atom.pos_[0][0] = 99;
    REQUIRE_THROWS_AS(d.init(&p, a), std::invalid_argument);
    REQUIRE(p.calls == 0);
    REQUIRE(d.atom.pos_[0][0] == 99);
    REQUIRE_THROWS_AS(d.run(), std::logic_error);
}

TEST_CASE("V2 rejects file scaling combinations that overwrite or ignore it", "[relax_precond_file]") {
    Directory directory;
    std::ofstream("diagonal.txt") << "3 1 1 1\n";
    const std::string option = GENERATE("keep_symm", "with_vib_filter", "with_constraint_vib_modes",
        "force_switchoff_scaling", "lbfgspp", "external");
    auto s = settings();
    RelaxationDrvV2 d(s);
    if (option == "keep_symm") d.flags.keep_symm = true;
    else if (option == "with_vib_filter") d.flags.vibffilter = true;
    else if (option == "with_constraint_vib_modes") d.flags.constraint_vib_modes = true;
    else if (option == "force_switchoff_scaling") d.flags.force_switchoff_scaling = true;
    else if (option == "lbfgspp") d.control.driver = option;
    else d.inplace_modifiers_init = false;
    auto a = initial(); Harmonic p;
    REQUIRE_THROWS_AS(d.init(&p, a), std::invalid_argument);
    REQUIRE(p.calls == 0);
}

TEST_CASE("V2 file scaling covers cell prefix and overrides built in scaling", "[relax_precond_file]") {
    Directory directory;
    auto a = initial(); a.pbc_ = {true, true, true}; Harmonic p;
    auto s = settings();
    s.Modify("relax_cell", "True"); s.Modify("cartesian", "False");
    s.Modify("stiffness", "1 2 3 4 5 6");
    std::ofstream("diagonal.txt") << "9 1 2 3 4 5 6 7 8 9\n";
    RelaxationDrvV2 d(s); d.init(&p, a);
    REQUIRE(d.use_dprecond);
    for (int i = 0; i < 9; ++i) REQUIRE(d.prec_scaling[i] == i + 1);
}

TEST_CASE("V2 diagonal file is accepted by existing ALGLIB optimizer scaling", "[relax_precond_file]") {
    Directory directory;
    std::ofstream("diagonal.txt") << "3 1 2 4\n";
    const std::string optimizer = GENERATE("lbfgs", "cg");
    auto s = settings(); s.Modify("driver", optimizer); s.Modify("maxiteration", "100");
    s.Modify("gtol", "1e-8");
    auto a = initial(); Harmonic p;
    RelaxationDrvV2 d(s); d.init(&p, a);
    REQUIRE(d.run() == 0);
    REQUIRE(d.atom.pos_[0].norm() < 1e-7);
    REQUIRE(d.f < 1e-14);
    REQUIRE(d.prec_scaling.isApprox(Eigen::Vector3d(1, 2, 4)));
}

TEST_CASE("V2 diagonal file controls actual normalized GD step", "[relax_precond_file]") {
    Directory directory;
    std::ofstream("diagonal.txt") << "3 1 2 4\n";
    auto s = settings(); s.Modify("driver", "gd");
    auto a = initial(); Harmonic p;
    RelaxationDrvV2 d(s); d.init(&p, a);
    d.run();
    // GD's initial step cap is .1 and max gradient is 3.
    REQUIRE(d.atom.pos_[0][0] == Catch::Approx(1 - 0.1 / 3));
    REQUIRE(d.atom.pos_[0][1] == Catch::Approx(2 - 0.1 / 3));
    REQUIRE(d.atom.pos_[0][2] == Catch::Approx(3 - 0.1 / 4));
}
