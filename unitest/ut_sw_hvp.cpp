#include <catch2/catch_all.hpp>
#include "potentials/potential_sw.h"
#include "analyse/compute_hessian.h"
#include <limits>
#include <map>
#include <sstream>

namespace {
std::string parameters(bool mixed, int terms) {
    std::ostringstream out;
    const int types = mixed ? 2 : 1;
    const char *names[] = {"H", "He"};
    for (int i = 0; i < types; ++i) {
        for (int j = i; j < types; ++j)
            out << names[i] << ' ' << names[j] << " 2body " << (terms == 2 ? 0. : 1.1 + .1*i + .2*j)
                << " .8 2.3 .6 4.3 1.2 4.5 3.4\n";
        for (int j = 0; j < types; ++j)
            for (int k = j; k < types; ++k)
                out << names[i] << ' ' << names[j] << ' ' << names[k] << ' '
                    << (terms == 1 ? 0. : .7 + .2*i + .1*j + .3*k)
                    << " .75 4.8 1.3 .9 -.2 3.25\n";
    }
    return out.str();
}

Atoms configuration(bool mixed, int geometry) {
    Atoms atom;
    atom.cell_ << 4.4, .3, .1, .1, 4.7, .2, .2, .1, 4.9;
    atom.pos_ = {{.5, .7, .8}, {1.8, .9, 1.}, {.9, 2., 1.4}, {1.9, 2.1, 2.}};
    atom.chemmap_ = mixed ? std::vector<int>{0, 1, 0, 1} : std::vector<int>{0, 0, 0, 0};
    atom.pbc_ = {geometry == 1, geometry == 1, geometry == 1};
    if (geometry == 2) {
        atom.cell_ = 2.5 * Eigen::Matrix3d::Identity();
        atom.pos_ = {{.2, .3, .4}};
        atom.chemmap_ = {0};
        atom.pbc_ = {true, true, true};
    }
    return atom;
}

Eigen::VectorXd force_fd(Potentials::AnyPotential &pot, const Atoms &atom,
                        const Eigen::VectorXd &direction, double step, double cutoff) {
    auto plus = atom, minus = atom;
    for (int i = 0; i < atom.size(); ++i) {
        plus.pos_[i] += step * direction.segment<3>(3*i).transpose();
        minus.pos_[i] -= step * direction.segment<3>(3*i).transpose();
    }
    Neighborhoods np(plus, cutoff), nm(minus, cutoff);
    pot.getEFS(plus, np);
    pot.getEFS(minus, nm);
    Eigen::VectorXd result(direction.size());
    for (int i = 0; i < atom.size(); ++i)
        result.segment<3>(3*i) = -(plus.forces_[i]-minus.forces_[i]).transpose() / (2*step);
    return result;
}

void check_state(const Atoms &atom, const Atoms &before) {
    REQUIRE(atom.pos_ == before.pos_);
    REQUIRE(atom.cell_ == before.cell_);
    REQUIRE(atom.energy_ == before.energy_);
    REQUIRE(atom.forces_ == before.forces_);
    REQUIRE(atom.stress_ == before.stress_);
    REQUIRE(atom.site_energies_ == before.site_energies_);
    REQUIRE(atom.have_energy == before.have_energy);
    REQUIRE(atom.have_stress == before.have_stress);
    REQUIRE(atom.have_hessian == before.have_hessian);
    REQUIRE(atom.hessian_ == before.hessian_);
    REQUIRE(atom.hvp_directions_ == before.hvp_directions_);
    REQUIRE(atom.hvp_targets_ == before.hvp_targets_);
    REQUIRE(atom.features_ == before.features_);
    REQUIRE(atom.local_attribs_d_ == before.local_attribs_d_);
}
}

TEST_CASE("SW native HVP differentiates pair and triplet forces without changing source state", "[sw][hvp]") {
    const bool mixed = GENERATE(false, true);
    const int terms = GENERATE(0, 1, 2); // both, pair only, triplet only
    const int geometry = GENERATE(0, 1, 2); // cluster, repeated periodic sources, N=1
    Potentials::SW model;
    std::istringstream input(parameters(mixed, terms));
    model.Load(input);
    Potentials::AnyPotential &pot = model;
    REQUIRE(pot.supports_analytical_hessian_vector_product());
    REQUIRE(pot.supports_analytical_hessian());
    REQUIRE(pot.supports_analytical_cell_hessian());
    auto atom = configuration(mixed, geometry);
    Neighborhoods nbhs(atom, pot.cutoff);
    if (geometry == 1) {
        bool repeated = false;
        for (int i = 0; i < atom.size(); ++i) {
            std::map<int, int> counts;
            for (int j : nbhs[i].indices_) repeated |= ++counts[j] > 1;
        }
        REQUIRE(repeated);
    }
    if (geometry == 2) REQUIRE(nbhs[0].nbhcount_ > 1);
    pot.getEFS(atom, nbhs);
    atom.features_["label"] = "retained";
    atom.local_attribs_d_["marker"] = Eigen::VectorXd::Ones(atom.size());
    atom.have_hessian = true;
    atom.hessian_ = Eigen::MatrixXd::Identity(3*atom.size(), 3*atom.size());
    atom.hvp_directions_ = Eigen::MatrixXd::Ones(3*atom.size(), 1);
    atom.hvp_targets_ = 2 * atom.hvp_directions_;
    const auto before = atom;
    for (int probe = 0; probe < 3; ++probe) {
        Eigen::VectorXd direction(3*atom.size());
        for (int i = 0; i < direction.size(); ++i) direction[i] = std::sin(.71*(i+1)*(probe+1));
        Eigen::VectorXd result = Eigen::VectorXd::Constant(2, 123.);
        pot.applyHessianVectorProduct(atom, nbhs, direction, result);
        REQUIRE(result.size() == direction.size());
        for (double step : {1e-5, 5e-6}) {
            const auto fd = force_fd(pot, atom, direction, step, pot.cutoff);
            INFO("mixed=" << mixed << " terms=" << terms << " geometry=" << geometry << " step=" << step);
            REQUIRE((result-fd).norm() < 2e-7 * std::max(1., fd.norm()));
        }
        if (geometry != 2) REQUIRE(result.norm() > 1e-6);
        else REQUIRE(result.isZero(1e-12));
        auto alias = direction;
        pot.applyHessianVectorProduct(atom, nbhs, alias, alias);
        REQUIRE(alias.isApprox(result, 1e-12));
        pot.add_efs = true;
        Eigen::VectorXd additive;
        pot.applyHessianVectorProduct(atom, nbhs, direction, additive);
        pot.add_efs = false;
        REQUIRE(additive == result);
        ComputeHessian consumer;
        consumer.method = ComputeHessian::HessianMethod::Analytic;
        Eigen::VectorXd consumed;
        consumer.compute_hvp(atom, direction, consumed, &pot);
        REQUIRE(consumed.isApprox(result, 1e-12));
        consumer.method = ComputeHessian::HessianMethod::Auto;
        consumer.compute_hvp(atom, direction, consumed, &pot);
        REQUIRE(consumed.isApprox(result, 1e-12));
        auto fractional = atom;
        fractional.to_fractional();
        consumer.compute_hvp(fractional, direction, consumed, &pot);
        REQUIRE((consumed-result).norm() < 1e-11*std::max(1., result.norm()));
        consumer.fractional_pos = true;
        REQUIRE_THROWS(consumer.compute_hvp(atom, direction, consumed, &pot));
    }
    const auto translation = Eigen::VectorXd::Ones(3*atom.size()).eval();
    Eigen::VectorXd zero;
    pot.applyHessianVectorProduct(atom, nbhs, translation, zero);
    REQUIRE(zero.isZero(1e-12));
    check_state(atom, before);
}

TEST_CASE("SW HVP preserves cutoff and collinear branches and validates query vectors", "[sw][hvp]") {
    Potentials::SW model;
    std::istringstream input(parameters(false, 0));
    model.Load(input);
    Potentials::AnyPotential &pot = model;
    REQUIRE(pot.supports_analytical_hessian_vector_product());
    for (double radius : {1.3, 3.3, 3.5}) {
        auto atom = configuration(false, 0);
        atom.pos_ = {{0, 0, 0}, {radius, 0, 0}, {-1.1, 0, 0}};
        atom.chemmap_ = {0, 0, 0};
        for (double cutoff : {2., pot.cutoff}) {
            Neighborhoods nbhs(atom, cutoff);
            Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(9, -.4, .7), result;
            pot.applyHessianVectorProduct(atom, nbhs, direction, result);
            for (double step : {1e-5, 5e-6})
                REQUIRE((result-force_fd(pot, atom, direction, step, cutoff)).norm() < 1e-7*std::max(1., result.norm()));
        }
    }
    auto atom = configuration(false, 0);
    Neighborhoods nbhs(atom, pot.cutoff);
    Eigen::VectorXd product = Eigen::VectorXd::Constant(3, 123.);
    const auto sentinel = product;
    Eigen::VectorXd bad = Eigen::VectorXd::Zero(3*atom.size()+1);
    REQUIRE_THROWS(pot.applyHessianVectorProduct(atom, nbhs, bad, product));
    REQUIRE(product == sentinel);
    bad = Eigen::VectorXd::Zero(3*atom.size());
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(pot.applyHessianVectorProduct(atom, nbhs, bad, product));
    REQUIRE(product == sentinel);
    auto other = atom;
    other.pos_.pop_back(); other.chemmap_.pop_back();
    Neighborhoods wrong(other, pot.cutoff);
    REQUIRE_THROWS(pot.applyHessianVectorProduct(atom, wrong, Eigen::VectorXd::Ones(3*atom.size()), product));
    REQUIRE(product == sentinel);
}
