// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "train/linear_equilibration.h"
#include "train/train_mtp.h"
#include <limits>
#include <sstream>

TEST_CASE("Linear equilibration solves an extreme diagonal congruence", "[training][scaling]") {
    const bool upper = GENERATE(false, true);
    Eigen::Matrix3d correlation;
    correlation << 2., .3, -.2, .3, 3., .4, -.2, .4, 1.;
    Eigen::Vector3d units(1e-100, 1., 1e100), expected(.7, -.4, .9);
    Eigen::MatrixXd h = units.asDiagonal() * correlation * units.asDiagonal();
    Eigen::VectorXd b = units.asDiagonal() * correlation * expected;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
        if (upper ? j < i : j > i) h(i,j) = std::numeric_limits<double>::quiet_NaN();
    const auto d = Training::equilibrate_linear_system(h, b, 3, upper);
    Eigen::MatrixXd full = upper ? Eigen::MatrixXd(h.selfadjointView<Eigen::Upper>())
                                : Eigen::MatrixXd(h.selfadjointView<Eigen::Lower>());
    Eigen::VectorXd x = full.ldlt().solve(b);
    for (int i = 0; i < 3; ++i) x[i] *= d[i] * units[i];
    REQUIRE((x - expected).norm() < 1e-13);
    REQUIRE((full.diagonal().array() - 1.).abs().maxCoeff() < 1e-15);
}

TEST_CASE("Equilibration preserves the regularized least squares objective", "[training][scaling]") {
    Eigen::Matrix<double, 5, 3> a;
    a << 1., 2e3, .003, 2., -1e3, .004, 3., 0., -.001,
         -1., 3e3, .006, 0., -2e3, .005;
    Eigen::VectorXd y(5); y << 1., -2., .5, .9, 3.;
    Eigen::Vector3d penalty(.4, 2e5, 1e-5);
    Eigen::MatrixXd augmented(8, 3); augmented.topRows(5) = a;
    augmented.bottomRows(3) = penalty.cwiseSqrt().asDiagonal();
    Eigen::VectorXd target = Eigen::VectorXd::Zero(8); target.head(5) = y;
    Eigen::VectorXd reference = augmented.colPivHouseholderQr().solve(target);
    Eigen::MatrixXd h = a.transpose() * a; h.diagonal() += penalty;
    Eigen::VectorXd b = a.transpose() * y;
    auto d = Training::equilibrate_linear_system(h, b, 3, false);
    Eigen::VectorXd x = h.selfadjointView<Eigen::Lower>().ldlt().solve(b);
    for (int i = 0; i < 3; ++i) x[i] *= d[i];
    REQUIRE((augmented * (x-reference)).norm() < 1e-10);
}

TEST_CASE("Equilibration does not invent information for zero columns", "[training][scaling]") {
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(2, 2);
    h(1,1) = 4.;
    Eigen::VectorXd b(2); b << 0., 2.;
    auto d = Training::equilibrate_linear_system(h, b, 2, true);
    REQUIRE(d[0] == 1.);
    REQUIRE(h(0,0) == 0.);
    h(0,0) = -1.;
    REQUIRE_THROWS(Training::equilibrate_linear_system(h, b, 2, true));
    h(0,0) = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS(Training::equilibrate_linear_system(h, b, 2, true));
}

TEST_CASE("MTP fixed rescale and diagonal solve preserve EFS fitting", "[training][scaling][mtp]") {
    const bool fast = GENERATE(false, true);
    const bool energy_only = GENERATE(false, true);
    const bool fixed_species = GENERATE(false, true);
    INFO("fast=" << fast << " energy_only=" << energy_only << " fixed_species=" << fixed_species);
    auto *log = new std::ostringstream;
    ScopedStreamRedirect redirect(global_ostream, log);
    Potentials::PotMTP truth("tests/case-mtp-les/les.mtp");
    truth.train_init();
    truth.set_scale(.2);
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> nbhs;
    for (int k=0; k<64; ++k) {
        Atoms a;
        a.cell_ = Eigen::Matrix3d::Identity() * 12.;
        a.pos_ = {{2.,2.,2.}, {4.1+.4*std::sin(k),2.2+.2*std::cos(2*k),2.},
                  {2.3,4.2+.4*std::cos(k),2.4}, {5.,4.8+.2*std::cos(3*k),3.1+.35*std::sin(2*k)}};
        a.chemmap_ = {0,1,k%2,(k/2)%2};
        truth.getEFS(a);
        a.energy_ += .002 * std::sin(3*k); // inconsistent observations + L2
        atoms.push_back(a); nbhs.emplace_back(a, truth.cutoff);
    }
    std::ostringstream saved; truth.Save(saved);
    std::vector<double> objectives;
    std::vector<std::vector<Atoms>> predictions;
    for (const char *method : {"none", "diagonal"}) {
        Potentials::PotMTP model;
        std::istringstream input(saved.str()); model.Load(input); model.set_inited();
        GPSettings s;
        s.Modify("do_orthogonalization", "false");
        s.Modify("lfit_fastalgo", fast ? "true" : "false");
        s.Modify("lfit_eng_only", energy_only ? "true" : "false");
        s.Modify("fix_species_coeffs", fixed_species ? "true" : "false");
        s.Modify("weight_L2", "1e-6");
        s.Modify("linear_precondition", method);
        s.Modify("rescale_mode", "fixed");
        Trainer_MTP trainer(s); trainer.init(atoms, nbhs, &model);
        const double radial_scaling = model.get_scale();
        Eigen::VectorXd radial = Eigen::Map<Eigen::VectorXd>(model.radial_coeffs, model.radial_coeff_size);
        const std::vector<double> species(model.species_coeffs, model.species_coeffs + model.species_count);
        trainer.Rescale();
        REQUIRE(model.get_scale() == radial_scaling);
        REQUIRE((radial-Eigen::Map<Eigen::VectorXd>(model.radial_coeffs, model.radial_coeff_size)).norm() == 0.);
        if (fixed_species) for(int i=0; i<model.species_count; ++i)
            REQUIRE(model.species_coeffs[i] == species[i]);
        double loss; trainer.AccumulateLoss(loss); objectives.push_back(loss);
        predictions.push_back(atoms);
        for (auto &a : predictions.back()) model.getEFS(a);
    }
    REQUIRE(objectives[0] == Catch::Approx(objectives[1]).epsilon(1e-7).margin(1e-10));
    for (size_t k=0; k<atoms.size(); ++k) {
        REQUIRE(predictions[0][k].energy_ == Catch::Approx(predictions[1][k].energy_).margin(1e-7));
        for(size_t i=0; i<atoms[k].size(); ++i)
            REQUIRE((predictions[0][k].forces_[i]-predictions[1][k].forces_[i]).norm() < 1e-6);
        REQUIRE((predictions[0][k].stress_-predictions[1][k].stress_).norm() < 1e-6);
    }
    REQUIRE(log->str().find("Rescaling to") == std::string::npos);
}
