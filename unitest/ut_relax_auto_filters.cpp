#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>

namespace {
struct AutoFilterFiles {
    std::filesystem::path dir = std::filesystem::temp_directory_path() /
        ("imrmlp-auto-filters-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    AutoFilterFiles() { std::filesystem::create_directory(dir); }
    ~AutoFilterFiles() { std::error_code error; std::filesystem::remove_all(dir, error); }
    std::string path(const char *name) const { return (dir / name).string(); }
    void affine(const char *name, const Eigen::MatrixXd &matrix, const Eigen::VectorXd &offset) const {
        std::ofstream out(path(name));
        out << matrix.rows() << ' ' << matrix.cols() << '\n' << std::setprecision(17);
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) out << matrix(i, j) << ' ';
            out << '\n';
        }
        for (double value : offset) out << value << ' ';
        REQUIRE(out.good());
    }
};
struct AutoFilterQuadratic : Potentials::AnyPotential {
    int calls = 0;
    AutoFilterQuadratic() { cutoff = 3; cutoffsq = 9; species_count = 1; }
    void getEFS(Atoms &atoms, AnyNeighborhoods &) override {
        ++calls; atoms.energy_ = 0.; atoms.stress_.setZero(); atoms.forces_.resize(atoms.size());
        for (int i = 0; i < atoms.size(); ++i) {
            atoms.energy_ += .5 * atoms.pos_[i].squaredNorm();
            atoms.forces_[i] = -atoms.pos_[i];
        }
    }
};
}

TEST_CASE("Automatic diagonal rebuild after displacement follows LF LC and their composition", "[relax][automatic][filters]") {
    const std::string mode = GENERATE("lf", "lc", "lf-lc");
    INFO(mode);
    AutoFilterFiles files; AutoFilterQuadratic potential;
    Atoms atoms; atoms.cell_ << 4, -4, 0, 4, 4, 0, 0, 0, 5;
    atoms.pbc_ = {true, true, true}; atoms.pos_ = {{1., .5, -.3}}; atoms.chemmap_ = {0};
    RelaxationDrvV2 driver;
    driver.flags.cartesian = false; driver.flags.atomic_force_recast = false;
    driver.flags.relax_cell = true; driver.flags.relax_atom = true;
    driver.flags.dprecond = true; driver.flags.write_traj = false;
    driver.control.driver = "lbfgs"; driver.control.max_iter = 0;
    driver.control.seldyn_file = files.path("absent-mask");
    // Read the actual nine-coordinate chart before adding either filter.
    driver.init(&potential, atoms);
    const Eigen::VectorXd full_initial = driver.x;
    REQUIRE(full_initial.size() == 9);
    const double initial_volume = driver.vol0;
    Eigen::MatrixXd transform = Eigen::MatrixXd::Identity(9, 9);
    Eigen::VectorXd offset = Eigen::VectorXd::Zero(9);
    const bool lf = mode != "lc", lc = mode != "lf";
    if (lf) {
        transform(0, 1) = .2; transform(3, 4) = -.15;
        transform(6, 6) = 1.5; transform(6, 7) = .4; transform(7, 8) = -.3;
        offset.setConstant(.02);
        files.affine("lf", transform, offset);
        driver.flags.linear_transform = true; driver.control.lf_file = files.path("lf");
    }
    if (lc) {
        // Freeze the six cell coordinates in the actual LF output chart.
        Eigen::MatrixXd selector = Eigen::MatrixXd::Zero(6, 9);
        selector.leftCols(6).setIdentity();
        const Eigen::VectorXd target = selector * (transform * full_initial + offset);
        files.affine("lc", selector, target);
        driver.flags.linear_constraint = true; driver.control.lc_file = files.path("lc");
    }
    driver.init(&potential, atoms);
    FilterLC *constraint = nullptr;
    for (auto *modifier : driver.modifiers_list)
        if (auto *candidate = dynamic_cast<FilterLC *>(modifier)) constraint = candidate;
    REQUIRE((constraint != nullptr) == lc);
    const Eigen::MatrixXd inverse = transform.inverse();
    const auto expected_diagonal = [&]() -> Eigen::VectorXd {
        Eigen::VectorXd full(9);
        full.head(6).setConstant(160 / eVA3_to_GPa / driver.control.cell_factor * initial_volume);
        full.tail(3) = driver.refcell.rowwise().squaredNorm();
        // An explicit matrix pullback supplies an independent projection oracle.
        Eigen::MatrixXd projected = inverse.transpose() * full.asDiagonal() * inverse;
        if (constraint) projected = (constraint->Q2.transpose() * projected * constraint->Q2).eval();
        return projected.diagonal();
    };
    REQUIRE(driver.prec_scaling.size() == (lc ? 3 : 9));
    REQUIRE(driver.prec_scaling.isApprox(expected_diagonal(), 1e-12));
    driver.flags.apply_disp = true; driver.flags.disp_gaussian = true; driver.control.disp = .001;
    const int calls_before = potential.calls;
    REQUIRE_NOTHROW(driver.run());
    REQUIRE(potential.calls >= calls_before + 2); // displaced evaluation and reset-reference evaluation
    REQUIRE(driver.prec_scaling.size() == (lc ? 3 : 9));
    REQUIRE(driver.prec_scaling.allFinite()); REQUIRE((driver.prec_scaling.array() > 0.).all());
    REQUIRE(driver.prec_scaling.isApprox(expected_diagonal(), 1e-11));
    if (lc) REQUIRE(driver.atom.cell_.isApprox(atoms.cell_, 1e-12));
    else REQUIRE_FALSE(driver.refcell.isApprox(atoms.cell_, 1e-15));
}
