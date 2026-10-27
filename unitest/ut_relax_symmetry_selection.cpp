#include <catch2/catch_all.hpp>
#include "mindriver/relax_symmetry_selection.h"
#include "mindriver/relax_new.h"
#include "potentials/potential_lj.h"
#include <sstream>

namespace {
Symmetry3D::symdataset fourfold(bool translations = false) {
    Symmetry3D::symdataset data{};
    data.n_operations = 4;
    data.n_atoms = translations ? 4 : 5;
    data.symop_mapping.resize(4, data.n_atoms);
    Eigen::Matrix3d quarter;
    quarter << 0, 1, 0, -1, 0, 0, 0, 0, 1;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    for (int op = 0; op < 4; ++op) {
        data.symm_ops.emplace_back(translations ? Eigen::Matrix3d::Identity().eval() : rotation,
            translations ? Eigen::RowVector3d(.25*op, 0, 0) : Eigen::RowVector3d::Zero().eval());
        for (int atom = 0; atom < 4; ++atom) data.symop_mapping(op, atom) = (atom+op)%4;
        if (!translations) data.symop_mapping(op, 4) = 4;
        rotation = (rotation*quarter).eval();
    }
    return data;
}

Atoms mirror_sites() {
    Atoms atom;
    atom.cell_ = Eigen::Vector3d(4., 5., 6.).asDiagonal();
    for (const auto &fractional : {Eigen::RowVector3d(0, .17, .23), Eigen::RowVector3d(0, .83, .23),
                                   Eigen::RowVector3d(.21, 0, .49), Eigen::RowVector3d(.79, 0, .49)})
        atom.pos_.push_back(fractional*atom.cell_);
    atom.chemmap_ = {0, 0, 1, 1};
    return atom;
}

int operation(Atoms atom, const Eigen::Matrix3d &rotation,
              const Eigen::RowVector3d &translation = Eigen::RowVector3d::Zero()) {
    const auto dataset = Symmetry3D::get_sym_dataset(atom, 1e-5, true);
    for (int i = 0; i < dataset.n_operations; ++i) {
        const auto difference = dataset.symm_ops[i].second-translation;
        if (dataset.symm_ops[i].first.isApprox(rotation, 1e-12) &&
            (difference.array()-difference.array().round()).matrix().norm() < 1e-8)
            return i;
    }
    throw std::runtime_error("Fixture initial symmetry operation is missing");
}

GPSettings lj_settings() {
    GPSettings settings;
    settings.Modify("ntypes", "2");
    for (const char *pair : {"0-0", "0-1", "1-1"}) {
        settings.Modify(pair, ".5 .7");
        settings.Modify(std::string("cut:")+pair, "2.8");
    }
    return settings;
}

FilterSymmCoord &filter(RelaxationDrvV2 &driver) {
    auto *result = dynamic_cast<FilterSymmCoord *>(driver.modifiers_list.front());
    REQUIRE(result != nullptr);
    return *result;
}

void configure(RelaxationDrvV2 &driver, const std::string &operations,
               const std::string &sites, bool recast = true) {
    driver.flags.keep_symm = true;
    driver.flags.relax_cell = true;
    driver.flags.relax_atom = true;
    driver.flags.atomic_force_recast = recast;
    driver.flags.write_traj = false;
    driver.control.symmetry_operations = operations;
    driver.control.symmetry_sites = sites;
}

struct ZeroPotential : Potentials::AnyPotential {
    ZeroPotential() {
        cutoff = 2.8; cutoffsq = cutoff*cutoff; species_count = 2;
        cutoffs = Eigen::MatrixXd::Constant(2, 2, cutoff);
        cutoffsqs = Eigen::MatrixXd::Constant(2, 2, cutoffsq);
        r_mins = Eigen::MatrixXd::Zero(2, 2);
    }
    void getEFS(Atoms &atom, AnyNeighborhoods &) override { zero(atom); }
    void getEFSOMP(Atoms &atom, AnyNeighborhoods &) override { zero(atom); }
    static void zero(Atoms &atom) {
        atom.resetEFS(); atom.energy_ = 0.; atom.stress_.setZero();
        atom.forces_.assign(atom.size(), Eigen::Vector3d::Zero());
    }
};
}

TEST_CASE("Initial symmetry generators close with identity and preserve subgroup mappings", "[symmetry-selection][selection-helper]") {
    const bool translations = GENERATE(false, true);
    const auto data = fourfold(translations);
    const auto generated = RelaxSymmetry::select_initial(data, "1", "");
    REQUIRE(generated.operations == std::vector<int>{0, 1, 2, 3});
    REQUIRE(generated.positions.symop_mapping == data.symop_mapping);
    REQUIRE(generated.positions.equivalent_atoms[3] == 0);
    const auto half = RelaxSymmetry::select_initial(data, " 2,2 ", "");
    REQUIRE(half.operations == std::vector<int>{0, 2});
    REQUIRE(half.positions.equivalent_atoms[2] == 0);
    REQUIRE(half.positions.equivalent_atoms[3] == 1);
    const auto unit = RelaxSymmetry::select_initial(data, "0", "");
    REQUIRE(unit.operations == std::vector<int>{0});
    REQUIRE(unit.positions.equivalent_atoms[3] == 3);
    const auto defaults = RelaxSymmetry::select_initial(data, "", "");
    REQUIRE(defaults.operations == generated.operations);
}

TEST_CASE("Site-only selection uses actual stabilizers without imposing unused global operations", "[symmetry-selection][selection-helper]") {
    const auto data = fourfold();
    const auto moving_site = RelaxSymmetry::select_initial(data, "1", "0");
    REQUIRE(moving_site.site_only);
    REQUIRE(moving_site.cell_operations == std::vector<int>{0});
    REQUIRE(moving_site.positions.equivalent_atoms == std::vector<int>{0, 1, 2, 3, 4});
    const auto fixed_site = RelaxSymmetry::select_initial(data, "1", "4");
    REQUIRE(fixed_site.cell_operations == std::vector<int>{0, 1, 2, 3});
    REQUIRE(fixed_site.sites == std::vector<int>{4});
    REQUIRE(RelaxSymmetry::select_initial(data, "1", "0,4").cell_operations == fixed_site.cell_operations);
}

TEST_CASE("Initial operation and site selection rejects malformed ids and nonclosing datasets", "[symmetry-selection][selection-helper]") {
    auto data = fourfold();
    for (const char *bad : {"-1", "4", "1,,2", "1,", ",1", "1.5", "999999999999999999999999"})
        REQUIRE_THROWS(RelaxSymmetry::select_initial(data, bad, ""));
    for (const char *bad : {"-1", "5", "0,,1", "1,"})
        REQUIRE_THROWS(RelaxSymmetry::select_initial(data, "1", bad));
    data.symm_ops.resize(2);
    data.n_operations = 2;
    data.symop_mapping = data.symop_mapping.topRows(2).eval();
    REQUIRE_THROWS(RelaxSymmetry::select_initial(data, "1", ""));
    data = fourfold();
    data.symop_mapping(0, 1) = 0;
    REQUIRE_THROWS(RelaxSymmetry::select_initial(data, "1", ""));
}

TEST_CASE("Selected initial mirror groups give exact global and site-only degrees of freedom", "[symmetry-selection][relax]") {
    const int mode = GENERATE(0, 1, 2, 3, 4, 5);
    auto atom = mirror_sites();
    const Eigen::Matrix3d mx = Eigen::Vector3d(-1, 1, 1).asDiagonal();
    const Eigen::Matrix3d my = Eigen::Vector3d(1, -1, 1).asDiagonal();
    const auto x = std::to_string(operation(atom, mx)), y = std::to_string(operation(atom, my));
    const auto identity = std::to_string(operation(atom, Eigen::Matrix3d::Identity()));
    const std::string operations[] = {"", identity, x, x, y, x+","+y};
    const std::string sites[] = {"", "", "", "0", "0", "0,2"};
    const int cells[] = {3, 6, 4, 4, 6, 3}, positions[] = {4, 12, 7, 11, 12, 10};
    ZeroPotential potential;
    RelaxationDrvV2 driver;
    configure(driver, operations[mode], sites[mode]);
    driver.init(&potential, atom);
    auto &selected = filter(driver);
    REQUIRE(selected.n_red_cell == cells[mode]);
    REQUIRE(selected.n_red_pos == positions[mode]);
    if (mode == 3) REQUIRE(selected.cell_basis.row(4).norm() > .5); // yz remains free under mx
    driver.LatPos2X();
    driver.apply_modifiers(driver.modifiers_list);
    const Eigen::VectorXd q = .003*Eigen::VectorXd::LinSpaced(driver.x.size(), -.3, .7);
    driver.x = q;
    driver.reverse_modifiers(driver.modifiers_list);
    const auto full = driver.x;
    driver.apply_modifiers(driver.modifiers_list);
    REQUIRE(driver.x.isApprox(q, 1e-11));
    driver.reverse_modifiers(driver.modifiers_list);
    REQUIRE(driver.x.isApprox(full, 1e-11));
}

TEST_CASE("Site subgroup projection and exact cell pullback match independent energy differences", "[symmetry-selection][relax][gradient]") {
    const bool recast = GENERATE(false, true);
    auto atom = mirror_sites();
    const auto mx = std::to_string(operation(atom, Eigen::Vector3d(-1, 1, 1).asDiagonal()));
    const auto my = std::to_string(operation(atom, Eigen::Vector3d(1, -1, 1).asDiagonal()));
    Potentials::LJ612 potential(lj_settings());
    RelaxationDrvV2 driver;
    configure(driver, mx+","+my, "0,2", recast);
    driver.init(&potential, atom);
    driver.LatPos2X(); driver.apply_modifiers(driver.modifiers_list);
    const Eigen::VectorXd q = .002*Eigen::VectorXd::LinSpaced(driver.x.size(), -.3, .4);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(q.size());
    driver(q, gradient);
    REQUIRE(gradient.allFinite()); REQUIRE(gradient.norm() > 1e-6);
    for (double step : {1e-6, 5e-7}) {
        for (int i = 0; i < q.size(); ++i) {
            Eigen::VectorXd plus = q, minus = q, temporary = Eigen::VectorXd::Zero(q.size());
            plus[i] += step; minus[i] -= step;
            const double fd = (driver(plus, temporary)-driver(minus, temporary))/(2*step);
            REQUIRE(gradient[i] == Catch::Approx(fd).margin(4e-6));
        }
    }
}

TEST_CASE("Site-only rerun retains the initial selected mapping when free atoms break global symmetry", "[symmetry-selection][relax][restart]") {
    auto atom = mirror_sites();
    const auto mx = std::to_string(operation(atom, Eigen::Vector3d(-1, 1, 1).asDiagonal()));
    ZeroPotential potential;
    RelaxationDrvV2 driver;
    configure(driver, mx, "0");
    driver.control.max_iter = 0;
    driver.init(&potential, atom);
    REQUIRE(driver.run() == 0);
    driver.atom.pos_[1][0] += .071; // this unlisted site is intentionally free
    REQUIRE(driver.run() == 0);
    REQUIRE(filter(driver).n_red_pos == 11);
    REQUIRE(filter(driver).n_red_cell == 4);
    REQUIRE(driver.atom.pos_[1][0] == Catch::Approx(.071).margin(1e-10));
    REQUIRE(std::abs(driver.atom.pos_[0][0]) < 1e-10);
    driver.control.symmetry_sites = "0,1";
    REQUIRE_THROWS(driver.run());
}

TEST_CASE("Initial glide images remain fixed across wrapping and repeated runs", "[symmetry-selection][relax][restart]") {
    const bool recast = GENERATE(false, true);
    Atoms atom;
    atom.cell_ = Eigen::Vector3d(4., 5., 6.).asDiagonal();
    atom.pos_ = {Eigen::RowVector3d(.2, .1, .3)*atom.cell_, Eigen::RowVector3d(.8, .6, .3)*atom.cell_};
    atom.chemmap_ = {0, 0};
    const auto glide = std::to_string(operation(atom, Eigen::Vector3d(-1, 1, 1).asDiagonal(), Eigen::RowVector3d(0, .5, 0)));
    ZeroPotential potential;
    RelaxationDrvV2 driver;
    configure(driver, glide, "", recast);
    driver.control.max_iter = 0;
    driver.init(&potential, atom);
    REQUIRE(driver.run() == 0);
    driver.reverse_modifiers(driver.modifiers_list);
    for (int i = 0; i < atom.size(); ++i) driver.x[driver.xposoffset+3*i+1] += recast ? 5.5 : 1.1;
    driver.apply_modifiers(driver.modifiers_list);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(driver.x.size());
    const Eigen::VectorXd q = driver.x;
    driver(q, gradient);
    driver.accept_checkpoint(false);
    const auto wrapped = driver.atom.pos_;
    REQUIRE(driver.run() == 0);
    REQUIRE(filter(driver).n_red_cell == 4);
    REQUIRE(filter(driver).n_red_pos == 3);
    for (int i = 0; i < atom.size(); ++i) REQUIRE(driver.atom.pos_[i].isApprox(wrapped[i], 1e-10));
}
