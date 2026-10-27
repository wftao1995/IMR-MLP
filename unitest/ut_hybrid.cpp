// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "potentials/potential_hybrid.h"
#include "potentials/potential_wrapper.h"
#include "apps/apps.h"
#include "analyse/compute_hessian.h"
#include "tools/hessian_utils.h"
#include "io/gpio.h"
#include "unitest/many_body_mtp_fixture.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>

using Catch::Approx;
using namespace Potentials;

namespace {

void add_setting(GPSettings &settings, const std::string &key,
                 const std::string &value) {
    settings.settings.emplace(key, value);
}

Atoms make_structure(const std::vector<int> &types) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 30.0;
    for (int i = 0; i < static_cast<int>(types.size()); ++i) {
        atom.pos_.push_back({2.0 * i, 0.0, 0.0});
    }
    atom.chemmap_ = types;
    return atom;
}

GPSettings make_zbl_hybrid_settings(const std::string &type) {
    GPSettings settings;
    add_setting(settings, "pottype", type);
    add_setting(settings, "npot", "1");
    add_setting(settings, "spcount", "2");
    add_setting(settings, "pot1:pottype", "zbl");
    add_setting(settings, "pot1:z", "13 29");
    add_setting(settings, "pot1:r0", "1.0");
    add_setting(settings, "pot1:r1", "3.0");
    return settings;
}

struct HVPWorkspace {
    CMDOption::CommandLineOptions options = CMDOption::global_cmdopts;
    std::filesystem::path original = std::filesystem::current_path(), path;
    int threads = omp_num_threads;
    HVPWorkspace() {
        char name[] = "/tmp/imrmlp_hybrid_hvp_XXXXXX";
        const auto directory = mkdtemp(name);
        if (!directory) throw std::runtime_error("Cannot create hybrid HVP test directory");
        path = directory;
        std::filesystem::current_path(path);
        omp_num_threads = 1;
        global_omp_init();
        PotMTP mtp;
        load_many_body_mtp_fixture(mtp, 2);
        mtp.Save("model.mtp");
    }
    ~HVPWorkspace() {
        CMDOption::global_cmdopts = options;
        omp_num_threads = threads;
        global_omp_init();
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
};

GPSettings mixed_hvp_settings() {
    GPSettings settings;
    settings.Modify("npot", "2");
    settings.Modify("spcount", "2");
    settings.Modify("pot1:pottype", "mtp");
    settings.Modify("pot1:potfile", "model.mtp");
    settings.Modify("pot2:pottype", "zbl");
    settings.Modify("pot2:z", "13 28");
    settings.Modify("pot2:r0", "1.1");
    settings.Modify("pot2:r1", "2.4");
    return settings;
}

Atoms mixed_hvp_atoms() {
    Atoms atom;
    atom.cell_ = 20 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {false, false, false};
    atom.pos_ = {{.4, .6, .5}, {1.9, .9, .7}, {3.9, 1.3, .9}, {2.3, 3.4, 1.1}};
    atom.chemmap_ = {0, 1, 0, 1};
    return atom;
}

Eigen::MatrixXd hybrid_force_hessian(PotentialHybrid &pot, const Atoms &source, double step) {
    const int dimension = 3 * source.size();
    Eigen::MatrixXd result(dimension, dimension);
    for (int column = 0; column < dimension; ++column) {
        auto plus = source, minus = source;
        plus.pos_[column / 3][column % 3] += step;
        minus.pos_[column / 3][column % 3] -= step;
        pot.getEFS(plus);
        pot.getEFS(minus);
        for (int atom = 0; atom < source.size(); ++atom)
            result.block<3, 1>(3 * atom, column) =
                -(plus.forces_[atom] - minus.forces_[atom]).transpose() / (2 * step);
    }
    return result;
}

} // namespace

TEST_CASE("Hybrid exposes its species count through the potential interface", "[hybrid][species]") {
    const bool mixed = GENERATE(false, true);
    HVPWorkspace workspace;
    PotentialHybrid pot(mixed ? mixed_hvp_settings() : make_zbl_hybrid_settings("hybrid"));
    const AnyPotential &api = pot;

    REQUIRE(pot.species_count == 2);
    REQUIRE(api.species_count == 2);
    for (const auto &child : pot.ppots)
        REQUIRE(child->species_count == api.species_count);
}

TEST_CASE("hybrid serial OMP and thread entry points agree", "[hybrid]") {
    PotentialHybrid pot(make_zbl_hybrid_settings("hybrid"));
    auto serial = make_structure({0, 1});
    auto omp = serial;
    auto thread = serial;

    pot.getEFS(serial);
    pot.getEFSOMP(omp);
    pot.getEFSOMPthread(thread);

    REQUIRE(omp.energy_ == Approx(serial.energy_).epsilon(1e-13));
    REQUIRE(thread.energy_ == Approx(serial.energy_).epsilon(1e-13));
    REQUIRE(omp.stress_.isApprox(serial.stress_, 1e-13));
    REQUIRE(thread.stress_.isApprox(serial.stress_, 1e-13));
    for (int i = 0; i < serial.size(); ++i) {
        REQUIRE(omp.forces_[i].isApprox(serial.forces_[i], 1e-13));
        REQUIRE(thread.forces_[i].isApprox(serial.forces_[i], 1e-13));
    }
}

TEST_CASE("hybrid energy-only evaluation preserves EFS arrays", "[hybrid]") {
    PotentialHybrid pot(make_zbl_hybrid_settings("hybrid"));
    auto atom = make_structure({0, 1});
    atom.forces_.assign(atom.size(), Eigen::RowVector3d(1.0, 2.0, 3.0));
    atom.stress_.setConstant(4.0);
    const auto forces_before = atom.forces_;
    const auto stress_before = atom.stress_;

    pot.getE(atom);

    REQUIRE(atom.forces_ == forces_before);
    REQUIRE(atom.stress_ == stress_before);
    REQUIRE(atom.have_energy);
    REQUIRE_FALSE(atom.have_stress);
}

TEST_CASE("hybrid shared species gradient counts atoms by type", "[hybrid]") {
    GPSettings settings;
    add_setting(settings, "pottype", "hybrid");
    add_setting(settings, "npot", "1");
    add_setting(settings, "spcount", "2");
    add_setting(settings, "pot1:pottype", "mtp");
    add_setting(settings, "pot1:potfile", "tests/case-train/potential.mtp");
    PotentialHybrid pot(settings);
    pot.train_init();
    auto atom = make_structure({0, 0, 1});
    Neighborhoods nbhs(atom, pot.cutoff);
    Eigen::Matrix<double, -1, 3> dLdF(atom.size(), 3);
    dLdF.setZero();
    Eigen::Matrix3d dLdS = Eigen::Matrix3d::Zero();
    std::vector<double> gradient(pot.nreg_coeffs, 0.0);
    constexpr double dLdE = 0.75;

    pot.AccumulateEFSCombinationGard(atom, nbhs, dLdE, dLdF, dLdS,
                                     gradient.data());

    REQUIRE(gradient[pot.nrad_coeffs] == Approx(2.0 * dLdE).margin(1e-12));
    REQUIRE(gradient[pot.nrad_coeffs + 1] == Approx(dLdE).margin(1e-12));

    REQUIRE(pot.get_NL_coeffs_count() == pot.nreg_coeffs);
    std::vector<double> coeffs(pot.nreg_coeffs);
    pot.nl_get_coeffs(coeffs.data());
    pot.nl_set_coeffs(coeffs.data());
    const auto groups = pot.nl_parameter_groups();
    REQUIRE(groups.size() == 3);
    REQUIRE(groups.back().end == pot.nreg_coeffs);

    AnyPotential *base = &pot;
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buffer;
    std::vector<double> polymorphic_gradient(pot.nreg_coeffs, 0.0);
    base->AccumulateEFSCombinationGard(atom, nbhs, dLdE, dLdF, dLdS,
                                       buffer, polymorphic_gradient.data());
    for (int i = 0; i < pot.nreg_coeffs; ++i) {
        REQUIRE(polymorphic_gradient[i] == Approx(gradient[i]).margin(1e-12));
    }

    std::vector<double> energy_gradient(pot.nreg_coeffs, 0.0);
    base->AccumulateEFSCombinationGard(atom, nbhs, dLdE,
                                       energy_gradient.data());
    for (int i = 0; i < pot.nreg_coeffs; ++i) {
        REQUIRE(energy_gradient[i] == Approx(gradient[i]).margin(1e-12));
    }
}

TEST_CASE("Hybrid analytic HVP sums mixed-cutoff MTP and ZBL without changing source EFS", "[hybrid][hvp]") {
    HVPWorkspace work;
    PotentialHybrid pot(mixed_hvp_settings());
    auto atom = mixed_hvp_atoms();
    pot.getEFS(atom);
    const auto source = atom;
    std::vector<double> coefficients(pot.get_NL_coeffs_count());
    pot.nl_get_coeffs(coefficients.data());
    Neighborhoods nbhs(atom, pot.cutoff);
    auto zbl_nbhs = nbhs.sub_neighborhoods(pot.ppots[1]->cutoff);
    REQUIRE(zbl_nbhs[0].nbhcount_ > 0);
    REQUIRE(zbl_nbhs[0].nbhcount_ < nbhs[0].nbhcount_);
    REQUIRE(pot.supports_analytical_hessian_vector_product());
    REQUIRE(pot.supports_analytical_hessian());
    Eigen::VectorXd u(3 * atom.size()), v(u.size());
    for (int i = 0; i < u.size(); ++i) {
        u[i] = std::sin(.3 + .7 * i);
        v[i] = std::cos(.6 + .4 * i);
    }
    Eigen::VectorXd hu = Eigen::VectorXd::Constant(1, 99.), hv;
    pot.applyHessianVectorProduct(atom, nbhs, u, hu);
    AnyPotential &api = pot;
    api.applyHessianVectorProduct(atom, v, hv);
    for (double step : {1e-5, 5e-6}) {
        const auto fd = hybrid_force_hessian(pot, atom, step);
        REQUIRE((hu - fd * u).norm() < 3e-6);
        REQUIRE((hv - fd * v).norm() < 3e-6);
    }
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(u.size());
    for (auto &child : pot.ppots) {
        auto child_nbhs = nbhs.sub_neighborhoods(child->cutoff);
        Eigen::VectorXd child_product;
        child->applyHessianVectorProduct(atom, child_nbhs, u, child_product);
        REQUIRE(child_product.norm() > 1e-7);
        sum += child_product;
    }
    REQUIRE((hu - sum).norm() < 1e-11);
    auto alias = u;
    pot.add_efs = true;
    pot.applyHessianVectorProduct(atom, nbhs, alias, alias);
    REQUIRE((alias - hu).norm() < 1e-11);
    Eigen::VectorXd combined;
    pot.applyHessianVectorProduct(atom, 2 * u - .3 * v, combined);
    REQUIRE((combined - (2 * hu - .3 * hv)).norm() < 1e-10);
    REQUIRE(u.dot(hv) == Approx(v.dot(hu)).margin(1e-10));
    Eigen::VectorXd translation = Eigen::VectorXd::Zero(u.size()), zero;
    for (int i = 0; i < atom.size(); ++i) translation.segment<3>(3 * i) << .2, -.4, .7;
    pot.applyHessianVectorProduct(atom, translation, zero);
    REQUIRE(zero.norm() < 1e-12);
    REQUIRE(atom.pos_ == source.pos_); REQUIRE(atom.cell_ == source.cell_);
    REQUIRE(atom.pbc_ == source.pbc_); REQUIRE(atom.chemmap_ == source.chemmap_);
    REQUIRE(atom.cartesian == source.cartesian);
    REQUIRE(atom.energy_ == source.energy_); REQUIRE(atom.forces_ == source.forces_);
    REQUIRE(atom.stress_ == source.stress_); REQUIRE(atom.site_energies_ == source.site_energies_);
    REQUIRE(atom.have_energy == source.have_energy); REQUIRE(atom.have_stress == source.have_stress);
    std::vector<double> after(coefficients.size()); pot.nl_get_coeffs(after.data());
    REQUIRE(after == coefficients);
}

TEST_CASE("Hybrid HVP capability follows every child and rejects unsupported combinations", "[hybrid][hvp]") {
    HVPWorkspace work;
    PotentialHybrid pot(mixed_hvp_settings());
    REQUIRE(pot.supports_analytical_hessian_vector_product());
    struct Unsupported : AnyPotential {
        void getEFS(Atoms &, AnyNeighborhoods &) override {
            GPERROR("Unsupported HVP test child must not be evaluated");
        }
    };
    auto supported = std::move(pot.ppots[1]);
    pot.ppots[1] = std::make_unique<Unsupported>();
    REQUIRE_FALSE(pot.supports_analytical_hessian_vector_product());
    auto atom = mixed_hvp_atoms();
    const auto source = atom;
    Neighborhoods nbhs(atom, pot.cutoff);
    const Eigen::VectorXd direction = Eigen::VectorXd::Ones(3 * atom.size());
    Eigen::VectorXd product = Eigen::VectorXd::Constant(2, 17.);
    REQUIRE_THROWS_WITH(pot.applyHessianVectorProduct(atom, nbhs, direction, product),
        Catch::Matchers::ContainsSubstring("every sub-potential"));
    REQUIRE(product == Eigen::VectorXd::Constant(2, 17.));
    REQUIRE(atom.pos_ == source.pos_); REQUIRE(atom.cell_ == source.cell_);
    pot.ppots[1] = std::move(supported);
    REQUIRE(pot.supports_analytical_hessian_vector_product());
}

TEST_CASE("Hybrid dense atomic Hessians sum mixed-cutoff children and preserve additive EFS",
          "[hybrid][hessian]") {
    HVPWorkspace work;
    const bool reverse = GENERATE(false, true);
    CAPTURE(reverse);
    GPSettings settings;
    for (const auto &[key, value] : mixed_hvp_settings().settings) {
        std::string target = key;
        if (reverse && key.starts_with("pot1:")) target = "pot2:" + key.substr(5);
        if (reverse && key.starts_with("pot2:")) target = "pot1:" + key.substr(5);
        settings.Modify(target, value);
    }
    PotentialHybrid pot(settings);
    REQUIRE(pot.supports_analytical_hessian());
    REQUIRE(pot.supports_analytical_cell_hessian());
    auto reference = mixed_hvp_atoms();
    pot.getEFS(reference);
    Neighborhoods nbhs(reference, pot.cutoff);
    auto short_nbhs = nbhs.sub_neighborhoods(pot.ppots[reverse ? 0 : 1]->cutoff);
    REQUIRE(short_nbhs[0].nbhcount_ > 0);
    REQUIRE(short_nbhs[0].nbhcount_ < nbhs[0].nbhcount_);
    HessianMatrix expected;
    expected.resize(reference.size());
    for (auto &child : pot.ppots) {
        auto child_atom = mixed_hvp_atoms();
        child_atom.resetEFS();
        auto child_nbhs = nbhs.sub_neighborhoods(child->cutoff);
        HessianMatrix block;
        child->getEFSHessian(child_atom, child_nbhs, block);
        for (int i = 0; i < reference.size(); ++i)
            for (int j = 0; j < reference.size(); ++j)
                expected(i, j) += block(i, j);
    }
    const Eigen::MatrixXd dense = HessianUtils::to_dense(expected);
    REQUIRE(dense.norm() > 1e-7);
    REQUIRE((dense - hybrid_force_hessian(pot, reference, 5e-6)).norm() < 3e-6);
    const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(dense.rows(), -.7, .9);
    Eigen::VectorXd hvp;
    pot.applyHessianVectorProduct(reference, nbhs, direction, hvp);
    REQUIRE((hvp - dense * direction).norm() < 1e-9);

    auto seed = mixed_hvp_atoms();
    seed.resetEFS();
    for (int i = 0; i < seed.size(); ++i) {
        seed.site_energies_[i] = .17 * (i + 1);
        seed.energy_ += seed.site_energies_[i];
        seed.forces_[i] << .2, -.3, .4;
    }
    seed.stress_.setConstant(.43);
    AnyPotential &api = pot;
    for (int entry = 0; entry < 5; ++entry) {
        CAPTURE(entry);
        auto atom = seed;
        HessianMatrix result;
        result.resize(1);
        result(0, 0).setConstant(71.);
        auto evaluate = [&] {
            if (entry == 0) api.getEFSHessian(atom, nbhs, result);
            if (entry == 1) api.getEFSHessian(atom, result);
            if (entry == 2) api.getEFSHessianOMP(atom, nbhs, result);
            if (entry == 3) api.getEFSHessianSymm(atom, result);
            if (entry == 4) api.getEFSHessianSymmOMP(atom, result);
        };
        auto check = [&](double seed_factor, double contribution_factor) {
            REQUIRE(atom.energy_ == Approx(seed_factor * seed.energy_
                + contribution_factor * reference.energy_).margin(1e-9));
            REQUIRE((atom.stress_ - seed_factor * seed.stress_
                - contribution_factor * reference.stress_).norm() < 1e-9);
            for (int i = 0; i < atom.size(); ++i) {
                REQUIRE(atom.site_energies_[i] == Approx(seed_factor * seed.site_energies_[i]
                    + contribution_factor * reference.site_energies_[i]).margin(1e-9));
                REQUIRE((atom.forces_[i] - seed_factor * seed.forces_[i]
                    - contribution_factor * reference.forces_[i]).norm() < 1e-9);
            }
            REQUIRE((HessianUtils::to_dense(result) - dense).norm() < 1e-9);
            REQUIRE(atom.pos_ == seed.pos_);
            REQUIRE(atom.cell_ == seed.cell_);
        };
        pot.add_efs = false;
        evaluate();
        check(0., 1.);
        atom = seed;
        pot.add_efs = true;
        evaluate();
        check(1., 1.);
        evaluate();
        check(1., 2.);
    }
    pot.add_efs = false;
    for (auto method : {ComputeHessian::HessianMethod::Auto,
                        ComputeHessian::HessianMethod::Analytic}) {
        ComputeHessian compute;
        compute.method = method;
        auto atom = seed;
        Eigen::MatrixXd calculated;
        compute.compute_fc(atom, calculated, &pot);
        REQUIRE((calculated - dense).norm() < 1e-9);
        HessianMatrix blocks;
        compute.compute_fc_blocks(atom, blocks, &pot);
        REQUIRE((HessianUtils::to_dense(blocks) - dense).norm() < 1e-9);
        REQUIRE(atom.energy_ == seed.energy_);
        REQUIRE(atom.forces_ == seed.forces_);
    }
}

TEST_CASE("Hybrid dense Hessians reject unsupported children before changing outputs",
          "[hybrid][hessian]") {
    HVPWorkspace work;
    PotentialHybrid pot(mixed_hvp_settings());
    struct Unsupported : AnyPotential {
        void getEFS(Atoms &, AnyNeighborhoods &) override {
            FAIL("Unsupported dense test child must not be evaluated");
        }
    };
    auto supported = std::move(pot.ppots[1]);
    pot.ppots[1] = std::make_unique<Unsupported>();
    REQUIRE_FALSE(pot.supports_analytical_hessian());
    AnyPotential &api = pot;
    auto atom = mixed_hvp_atoms();
    atom.resetEFS();
    atom.energy_ = 19.;
    atom.stress_.setConstant(23.);
    const auto source = atom;
    Neighborhoods nbhs(atom, pot.cutoff);
    HessianMatrix result;
    result.resize(1);
    result(0, 0).setConstant(29.);
    for (int entry = 0; entry < 5; ++entry) {
        CAPTURE(entry);
        auto evaluate = [&] {
            if (entry == 0) api.getEFSHessian(atom, nbhs, result);
            if (entry == 1) api.getEFSHessian(atom, result);
            if (entry == 2) api.getEFSHessianOMP(atom, nbhs, result);
            if (entry == 3) api.getEFSHessianSymm(atom, result);
            if (entry == 4) api.getEFSHessianSymmOMP(atom, result);
        };
        REQUIRE_THROWS_WITH(evaluate(), Catch::Matchers::ContainsSubstring("every sub-potential"));
        REQUIRE(atom.energy_ == source.energy_);
        REQUIRE(atom.forces_ == source.forces_);
        REQUIRE(atom.stress_ == source.stress_);
        REQUIRE(atom.site_energies_ == source.site_energies_);
        REQUIRE(result.size == 1);
        REQUIRE(result(0, 0) == Eigen::Matrix3d::Constant(29.));
    }
    pot.ppots[1] = std::move(supported);
    REQUIRE(pot.supports_analytical_hessian());
}

TEST_CASE("Hybrid MTP ZBL HVP drives calc_fc lowest modes", "[hybrid][hvp][cli]") {
    HVPWorkspace work;
    PotentialHybrid pot(mixed_hvp_settings());
    auto atom = mixed_hvp_atoms();
    const auto fd_raw = hybrid_force_hessian(pot, atom, 5e-6);
    REQUIRE((fd_raw - fd_raw.transpose()).norm() < 2e-6);
    const Eigen::MatrixXd fd = .5 * (fd_raw + fd_raw.transpose());
    const int dimension = 3 * atom.size();
    Eigen::VectorXd roots(dimension);
    Eigen::MatrixXd translations = Eigen::MatrixXd::Zero(dimension, 3);
    for (int i = 0; i < atom.size(); ++i) {
        const double root = std::sqrt(atom.chemmap_[i] == 0 ? 2. : 5.);
        roots.segment<3>(3 * i).setConstant(root);
        for (int axis = 0; axis < 3; ++axis) translations(3 * i + axis, axis) = root;
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(translations);
    const Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(dimension, dimension);
    const Eigen::MatrixXd internal = q.rightCols(dimension - 3);
    const Eigen::MatrixXd dynamical = roots.cwiseInverse().asDiagonal() * fd * roots.cwiseInverse().asDiagonal();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(internal.transpose() * dynamical * internal);
    REQUIRE(solver.info() == Eigen::Success);
    std::ofstream("index.map") << "Al 1.0 2\nNi 1.0 5\n";
    GPIO io; REQUIRE(io.readmap("index.map"));
    REQUIRE(io.write_one(atom, "input.xyz", "extxyz"));
    const auto settings = mixed_hvp_settings();
    {
        std::ofstream out("hybrid.ini");
        for (const auto &[key, value] : settings.settings) out << "pot:" << key << ' ' << value << '\n';
    }
    CMDOption::global_cmdopts = CMDOption::CommandLineOptions{};
    std::vector<std::string> args{"calc_fc", "--settings=hybrid.ini", "--potential-type=hybrid",
        "--potfile=model.mtp", "--input=input.xyz", "--input-format=extxyz", "--index=index.map",
        "--hessian-method=analytic", "--lowest-modes=2", "--lowest-mode-tol=1e-9"};
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    REQUIRE(calc_fc(argv.size(), argv.data()) == 0);
    REQUIRE_FALSE(std::filesystem::exists("FORCE_CONSTANTS"));
    std::ifstream output("EIGENVALS"); std::string header;
    REQUIRE(std::getline(output, header));
    for (int mode = 0; mode < 2; ++mode) {
        double eigenvalue, frequency, energy, wavenumber;
        REQUIRE(output >> eigenvalue >> frequency >> energy >> wavenumber);
        REQUIRE(eigenvalue == Approx(solver.eigenvalues()[mode]).margin(3e-6));
    }
}

TEST_CASE("Hybrid explicit children take priority over legacy top-level native potfile",
          "[hybrid][manifest][hvp]") {
    HVPWorkspace work;
    const bool fixed=GENERATE(false,true);
    auto settings=mixed_hvp_settings();settings.Modify("pot1:train",fixed ? "false" : "true");
    PotentialHybrid reference(settings);
    settings.Modify("potfile","model.mtp");
    PotentialHybrid explicit_model(settings);
    REQUIRE(explicit_model.npot==2);REQUIRE(explicit_model.ntrains==(fixed ? 0 : 1));
    auto expected=mixed_hvp_atoms(),actual=expected;
    reference.getEFS(expected);explicit_model.getEFS(actual);
    REQUIRE(actual.energy_==Approx(expected.energy_).margin(1e-12));
    REQUIRE((actual.stress_-expected.stress_).norm()<1e-11);
    for(int i=0;i<actual.size();++i)REQUIRE((actual.forces_[i]-expected.forces_[i]).norm()<1e-11);
    const Eigen::VectorXd direction=Eigen::VectorXd::LinSpaced(3*actual.size(),-.2,.3);
    Eigen::VectorXd expected_hvp,actual_hvp;
    reference.applyHessianVectorProduct(expected,direction,expected_hvp);
    explicit_model.applyHessianVectorProduct(actual,direction,actual_hvp);
    REQUIRE((actual_hvp-expected_hvp).norm()<1e-11);
    // A manifest with partial overrides still takes the native manifest path.
    {
        std::ofstream out("definition.hybrid");out<<"pottype hybrid\n";
        for(const auto &[key,value]:mixed_hvp_settings().settings)out<<key<<' '<<value<<'\n';
    }
    GPSettings manifest;manifest.Modify("potfile","definition.hybrid");manifest.Modify("npot","2");
    manifest.Modify("pot1:train",fixed ? "false" : "true");
    PotentialHybrid restored(manifest);actual=mixed_hvp_atoms();restored.getEFS(actual);
    REQUIRE(restored.ntrains==(fixed ? 0 : 1));
    REQUIRE(actual.energy_==Approx(expected.energy_).margin(1e-12));
    for(int i=0;i<actual.size();++i)REQUIRE((actual.forces_[i]-expected.forces_[i]).norm()<1e-11);
}

TEST_CASE("Hybrid exposes its initialized species count through the base potential", "[hybrid][species]") {
    PotentialHybrid potential(make_zbl_hybrid_settings("hybrid"));
    const AnyPotential &base = potential;
    REQUIRE(base.species_count == 2);
}
