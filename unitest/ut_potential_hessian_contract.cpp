#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp_m.h"
#include <array>
#include <fstream>
#include <functional>
#include <sstream>

using namespace Potentials;

namespace {
class UnsupportedPotential : public AnyPotential {
public:
    int evaluations = 0;
    void getEFS(Atoms &, AnyNeighborhoods &) override { ++evaluations; }
};

class ParameterGradientPotential : public AnyPotential {
public:
    double parameter = 1.25;
    int evaluations = 0;
    ParameterGradientPotential() { cutoff = 4.; }
    void getEFS(Atoms &atom, AnyNeighborhoods &) override {
        ++evaluations;
        atom.resetEFS();
        for (int i = 0; i < atom.size(); ++i) {
            const double x = atom.pos_[i][0];
            const double energy = parameter*x*x;
            atom.energy_ += energy;
            atom.site_energies_[i] = energy;
            atom.forces_[i] = Eigen::Vector3d(-2*parameter*x,0,0);
            atom.stress_(0,0) -= 2*energy;
        }
    }
};

Atoms sentinel_atoms() {
    Atoms atom;
    atom.cell_ = 20. * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1., 2., 3.}, {3., 2., 1.}};
    atom.chemmap_ = {0, 0};
    atom.energy_ = 123.;
    atom.have_energy = true;
    atom.forces_ = {{1., 2., 3.}, {-4., -5., -6.}};
    atom.stress_ = 7. * Eigen::Matrix3d::Identity();
    atom.have_stress = true;
    atom.site_energies_ = {42., 81.};
    return atom;
}

void require_unchanged(const Atoms &actual, const Atoms &before) {
    REQUIRE(actual.pos_ == before.pos_);
    REQUIRE(actual.chemmap_ == before.chemmap_);
    REQUIRE(actual.cell_ == before.cell_);
    REQUIRE(actual.energy_ == before.energy_);
    REQUIRE(actual.have_energy == before.have_energy);
    REQUIRE(actual.forces_ == before.forces_);
    REQUIRE(actual.stress_ == before.stress_);
    REQUIRE(actual.have_stress == before.have_stress);
    REQUIRE(actual.site_energies_ == before.site_energies_);
}
}

TEST_CASE("Unsupported Hessian interfaces fail without changing results",
          "[hessian][potential-contract]") {
    UnsupportedPotential potential;
    AnyPotential &api = potential;
    auto atom = sentinel_atoms();
    const auto before = atom;
    Neighborhoods neighbors(atom, potential.cutoff);
    HessianMatrix dense;
    dense.resize(1);
    dense(0, 0).setConstant(19.);
    HessianSparse sparse;
    sparse.size = 2;
    sparse(0, 1).setConstant(-11.);
    Eigen::MatrixXd cell = Eigen::MatrixXd::Constant(2, 3, 31.);
    Eigen::MatrixXd mixed = Eigen::MatrixXd::Constant(3, 4, 37.);
    std::vector<double> diagonal{41., 43., 47.};

    const std::array<std::pair<const char *, std::function<void()>>, 12> calls{{
        {"getEFSHessian(Atoms&, HessianMatrix&)", [&] { api.getEFSHessian(atom, dense); }},
        {"getEFSHessian(Atoms&, AnyNeighborhoods&, HessianMatrix&)", [&] { api.getEFSHessian(atom, neighbors, dense); }},
        {"getEFSHessianCell(Atoms&, AnyNeighborhoods&", [&] { api.getEFSHessianCell(atom, neighbors, dense, cell, mixed); }},
        {"getEFSHessianCell(Atoms&, HessianMatrix&", [&] { api.getEFSHessianCell(atom, dense, cell, mixed); }},
        {"getEFSHessianSymm", [&] { api.getEFSHessianSymm(atom, dense); }},
        {"getEFSHessianCellSymm", [&] { api.getEFSHessianCellSymm(atom, dense, cell, mixed); }},
        {"getEFSHessianCellSymmOMP", [&] { api.getEFSHessianCellSymmOMP(atom, dense, cell, mixed); }},
        {"getEFSHessianSymmDiaOMP", [&] { api.getEFSHessianSymmDiaOMP(atom, diagonal); }},
        {"getHessian", [&] { api.getHessian(atom, neighbors, {{0, 1}}, sparse); }},
        {"getEFSHessianOMP", [&] { api.getEFSHessianOMP(atom, neighbors, dense); }},
        {"getEFSHessianCellOMP", [&] { api.getEFSHessianCellOMP(atom, neighbors, dense, cell, mixed); }},
        // This existing delegate must report the missing serial Symm implementation.
        {"getEFSHessianSymm", [&] { api.getEFSHessianSymmOMP(atom, dense); }}
    }};
    const int index = GENERATE(range(0, 12));
    CAPTURE(index, calls[index].first);
    REQUIRE_THROWS_WITH(calls[index].second(),
                        Catch::Matchers::ContainsSubstring(std::string(calls[index].first).substr(0, std::string(calls[index].first).find('(')))
                        && Catch::Matchers::ContainsSubstring("support"));
    require_unchanged(atom, before);
    REQUIRE(potential.evaluations == 0);
    REQUIRE_FALSE(api.supports_analytical_hessian());
    REQUIRE_FALSE(api.supports_analytical_cell_hessian());
    REQUIRE(dense.size == 1);
    REQUIRE(dense.hessian.size() == 1);
    REQUIRE(dense(0, 0) == Eigen::Matrix3d::Constant(19.));
    REQUIRE(sparse.size == 2);
    REQUIRE(sparse.hessian.size() == 1);
    REQUIRE(sparse.hessian.contains({0, 1}));
    REQUIRE(sparse(0, 1) == Eigen::Matrix3d::Constant(-11.));
    REQUIRE(cell == Eigen::MatrixXd::Constant(2, 3, 31.));
    REQUIRE(mixed == Eigen::MatrixXd::Constant(3, 4, 37.));
    REQUIRE(diagonal == std::vector<double>{41., 43., 47.});
}

TEST_CASE("Numerical parameter-gradient helpers restore EFS values and presence on normal return",
          "[potential-contract][numerical-gradient]") {
    const bool energy_present = GENERATE(false,true);
    for (const bool energy_only : {false,true}) {
        CAPTURE(energy_present,energy_only);
        ParameterGradientPotential potential;
        AnyPotential &api = potential;
        auto atom = sentinel_atoms();
        atom.have_energy = energy_present;
        atom.have_stress = !energy_present;
        // One initial state has no force labels at all; helpers must restore it.
        if (!energy_present) atom.forces_.clear();
        const auto source = atom;
        const double parameter = potential.parameter;
        Neighborhoods neighbors(atom,potential.cutoff);
        Eigen::Matrix<double,-1,3> force_weights = Eigen::Matrix<double,-1,3>::Zero(2,3);
        force_weights(0,0) = .2; force_weights(1,0) = .4;
        Eigen::Matrix3d stress_weights = Eigen::Matrix3d::Zero();
        stress_weights(0,0) = .3;
        double accumulator = 5.;
        int updates = 0;
        auto perturb = [&](int index,double delta) {
            REQUIRE(index == 0);
            ++updates;
            potential.parameter += delta;
        };
        if (energy_only)
            api.numerical_gradient_eonly(atom,neighbors,.7,&accumulator,1,perturb);
        else
            api.numerical_gradient_efs(atom,neighbors,.7,force_weights,stress_weights,
                                       &accumulator,1,perturb);
        // x=1,3: dE/dp=10, dFx/dp=(-2,-6), dWxx/dp=-20.
        // The complete combination is 7-.4-2.4-6=-1.8; energy-only is 7.
        REQUIRE(accumulator == Catch::Approx(5+(energy_only ? 7. : -1.8)).margin(1e-8));
        REQUIRE(potential.parameter == Catch::Approx(parameter).margin(1e-14));
        REQUIRE(potential.evaluations == 2);
        REQUIRE(updates == 3);
        require_unchanged(atom,source);
        REQUIRE(atom.pbc_ == source.pbc_);
        REQUIRE(atom.cartesian == source.cartesian);
    }
}

TEST_CASE("Inherited Hessian SymmOMP dispatches to the implemented Symm method",
          "[hessian][potential-contract]") {
    class SymmetricPotential : public UnsupportedPotential {
    public:
        int calls = 0;
        double tolerance = 0.;
        void getEFSHessianSymm(Atoms &atom, HessianMatrix &result, double symprec) override {
            ++calls;
            tolerance = symprec;
            result.resize(atom.size());
            result(0, 1).setConstant(17.);
        }
    } potential;
    AnyPotential &api = potential;
    auto atom = sentinel_atoms();
    HessianMatrix result;
    api.getEFSHessianSymmOMP(atom, result, 2e-5);
    REQUIRE(potential.calls == 1);
    REQUIRE(potential.tolerance == 2e-5);
    REQUIRE(result.size == atom.size());
    REQUIRE(result(0, 1) == Eigen::Matrix3d::Constant(17.));
}

TEST_CASE("MTP diagonal queries honor implemented and unsupported model contracts",
          "[hessian][potential-contract]") {
    const bool model_head = GENERATE(false, true);
    std::unique_ptr<AnyPotential> potential;
    if (model_head) {
        // The existing fixture's radial/moment graph is also a valid MTP-M template.
        std::ifstream input("tests/case-mtp-les/les.mtp");
        REQUIRE(input.good());
        std::string source((std::istreambuf_iterator<char>(input)), {});
        const auto coefficients = source.find("species_coeffs");
        REQUIRE(coefficients != std::string::npos);
        source.resize(coefficients);
        auto mtpm = std::make_unique<PotMTP_M<m_model_mtpq2>>();
        std::istringstream model(source);
        mtpm->Load(model);
        REQUIRE_FALSE(mtpm->lr_les);
        for (int i = 0; i < mtpm->moment_coeff_size; ++i)
            mtpm->moment_coeffs[i] = .005 * (i + 1);
        for (int i = 0; i < mtpm->model_coeff_size; ++i)
            mtpm->model_coeffs[i] = .1 * (i + 1);
        mtpm->model.coeffs = mtpm->model_coeffs;
        potential = std::move(mtpm);
    } else {
        potential = std::make_unique<PotMTP>("tests/case-mtp-les/les.mtp");
    }
    auto atom = sentinel_atoms();
    const auto before = atom;
    std::vector<double> diagonal{2., 3., 5.};
    if (model_head) {
        auto fresh = before;
        potential->getEFS(fresh);
        auto dense_state = before;
        HessianMatrix dense;
        potential->getEFSHessian(dense_state, dense);
        potential->getEFSHessianSymmDiaOMP(atom, diagonal);
        REQUIRE(diagonal.size() == 3 * atom.size());
        double norm = 0.;
        for (int i = 0; i < atom.size(); ++i)
            for (int d = 0; d < 3; ++d) {
                REQUIRE(std::isfinite(diagonal[3*i+d]));
                REQUIRE(diagonal[3*i+d] == Catch::Approx(dense(i,d,i,d)).margin(1e-10));
                norm += std::abs(diagonal[3*i+d]);
            }
        REQUIRE(norm > 1e-10);
        REQUIRE(atom.energy_ == Catch::Approx(fresh.energy_).margin(1e-10));
        REQUIRE((atom.stress_ - fresh.stress_).norm() < 1e-10);
        REQUIRE(atom.site_energies_.size() == fresh.site_energies_.size());
        for (int i = 0; i < atom.size(); ++i) {
            REQUIRE((atom.forces_[i] - fresh.forces_[i]).norm() < 1e-10);
            REQUIRE(atom.site_energies_[i] == Catch::Approx(fresh.site_energies_[i]).margin(1e-10));
        }
    } else {
        REQUIRE_THROWS_WITH(potential->getEFSHessianSymmDiaOMP(atom, diagonal),
                            Catch::Matchers::ContainsSubstring("getEFSHessianSymmDiaOMP"));
        REQUIRE(diagonal == std::vector<double>{2., 3., 5.});
        require_unchanged(atom, before);
    }
}
