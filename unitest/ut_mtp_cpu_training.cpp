// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "fixtures/mtp_linear_efs.h"
#include "unitest/les_curvature_fixture.h"
#include "train/train_nonlinear_potential.h"

using namespace Potentials;
using Catch::Approx;

TEST_CASE("MTP CPU site gradients retain subsets and offset layouts", "[mtp][cpu][gradient]")
{
    const int variant = GENERATE(2, 4);
    const bool derivatives = GENERATE(false, true);
    const bool isolated = GENERATE(false, true);
    PotMTP model;
    MTPLinearEFSFixture::load(model, variant);
    model.train_init();
    model.set_inited();
    auto atoms = MTPLinearEFSFixture::frames(2)[isolated ? 2 : 1];
    Neighborhoods neighbors(atoms, model.cutoff);
    auto &site = neighbors[0];
    SiteCoordinateMatrix weights(site.nbhcount_, 3);
    for (int j = 0; j < site.nbhcount_; ++j)
        weights.row(j) << .07 * (j + 1), -.13, .2;
    auto *force_weight = derivatives ? &weights : nullptr;
    constexpr double energy_weight = .3;
    const auto objective = [&]
    {
        double energy = 0.;
        SiteCoordinateMatrix gradient;
        if (derivatives)
        {
            model.CalcSiteEFS(site, energy, gradient);
            return energy_weight * energy + (gradient.array() * weights.array()).sum();
        }
        model.CalcSiteE(site, energy);
        return energy_weight * energy;
    };
    Eigen::VectorXd full = Eigen::VectorXd::Zero(model.reg_coeff_size);
    model.AccumulateFPGard(site, full.data(), energy_weight, force_weight);
    for (int i = 0; i < full.size(); ++i)
    {
        CAPTURE(variant, derivatives, isolated, i);
        double &coefficient = model.regression_coeffs[i];
        const double original = coefficient, step = 1e-6 * std::max(1., std::abs(original));
        coefficient = original + step;
        const double plus = objective();
        coefficient = original - step;
        const double minus = objective();
        coefficient = original;
        REQUIRE(full[i] == Approx((plus - minus) / (2 * step)).margin(3e-5).epsilon(3e-6));
    }
    const int radial = model.radial_coeff_size, moments = model.alpha_scalar_moments;
    Eigen::VectorXd selected = Eigen::VectorXd::Zero(full.size());
    model.AccumulateFPGardRonly(site, selected.data(), energy_weight, force_weight);
    model.AccumulateFPGardLonly(site, selected.data() + radial, energy_weight, force_weight);
    REQUIRE(selected.isApprox(full, 1e-13));
    Eigen::VectorXd offset = Eigen::VectorXd::Zero(radial + moments + 3);
    model.AccumulateFPGard(site, offset.data(), energy_weight, force_weight, 2, radial + 3);
    Eigen::VectorXd separate = Eigen::VectorXd::Zero(offset.size());
    model.AccumulateFPGardRonly(site, separate.data(), energy_weight, force_weight, 2);
    model.AccumulateFPGardLonly(site, separate.data(), energy_weight, force_weight, radial + 3);
    REQUIRE(offset.isApprox(separate, 1e-13));
    REQUIRE(offset.segment(2, radial).isApprox(full.head(radial), 1e-13));
    REQUIRE(offset.tail(moments).isApprox(full.segment(radial + model.species_count, moments),
                                          1e-13));
    REQUIRE(offset.head(2).isZero());
    REQUIRE(offset[radial + 2] == 0.);
}

TEST_CASE("MTP CPU parameter gradients retain cutoff-equality semantics", "[mtp][cpu][cutoff]")
{
    PotMTP model;
    MTPLinearEFSFixture::load(model, 2);
    std::ostringstream saved;
    model.Save(saved);
    auto text = saved.str();
    const std::string basis = "radial_basis_type = RBChebyshev";
    const auto begin = text.find(basis);
    REQUIRE(begin != std::string::npos);
    text.replace(begin, basis.size(), "radial_basis_type = RBChebyshevConfigurable");
    const auto size = text.find("radial_basis_size", begin);
    REQUIRE(size != std::string::npos);
    const auto end = text.find('\n', size);
    REQUIRE(end != std::string::npos);
    text.insert(end + 1, "cutoff_type = cosine_left\nshift_type = linear\n");
    std::istringstream input(text);
    model.Load(input);
    model.train_init();
    model.set_inited();
    REQUIRE(model.cutoffs(0, 0) == 4.);
    REQUIRE(model.cutoff > 4.);
    Atoms atoms;
    atoms.cell_ = 20. * Eigen::Matrix3d::Identity();
    atoms.pos_ = {{0., 1., 2.}, {4., 1., 2.}};
    atoms.chemmap_ = {0, 0};
    Neighborhoods neighbors(atoms, model.cutoff);
    auto &site = neighbors[0];
    REQUIRE(site.nbhcount_ == 1);
    REQUIRE(site.distances_[0] == model.cutoffs(0, 0));
    SiteCoordinateMatrix weights(1, 3);
    weights << .2, -.1, .3;
    constexpr double energy_weight = .4;
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(model.reg_coeff_size);
    model.AccumulateFPGard(site, gradient.data(), energy_weight, &weights);
    const auto objective = [&]
    {
        double energy = 0.;
        SiteCoordinateMatrix derivative;
        model.CalcSiteEFS(site, energy, derivative);
        return energy_weight * energy + (derivative.array() * weights.array()).sum();
    };
    for (int i = 0; i < gradient.size(); ++i)
    {
        const double original = model.regression_coeffs[i], step = 1e-6;
        model.regression_coeffs[i] = original + step;
        const double plus = objective();
        model.regression_coeffs[i] = original - step;
        const double minus = objective();
        model.regression_coeffs[i] = original;
        CAPTURE(i);
        REQUIRE(gradient[i] == Approx((plus - minus) / (2 * step)).margin(2e-7));
    }
}

TEMPLATE_TEST_CASE("MTP-M CPU energy-only training preserves full EFS pullbacks",
                   "[mtp_m][cpu][training]", m_model_mtpli, m_model_mtpq2, m_model_mtpfs,
                   m_model_nn1, m_model_nn2)
{
    const bool frozen_radial = GENERATE(false, true);
    const bool missing_labels = GENERATE(false, true);
    const bool isolated = GENERATE(false, true);
    PotMTP_M<TestType> model;
    LESCurvature::initialize(model, false);
    model.lr_les = false;
    model.req_radical = !frozen_radial;
    model.train_init();
    Atoms reference = LESCurvature::geometry();
    if (isolated)
    {
        reference.pos_.resize(1);
        reference.chemmap_.resize(1);
    }
    model.getEFS(reference);
    reference.energy_ += .3;
    if (missing_labels)
    {
        reference.forces_.clear();
        reference.have_stress = false;
    }
    std::vector<Atoms> data{reference};
    std::vector<Neighborhoods> nbhs{Neighborhoods(reference, model.cutoff)};
    if (isolated)
        REQUIRE(nbhs[0][0].nbhcount_ == 0);
    if (frozen_radial)
    {
        auto workspace = model.make_site_workspace();
        auto &site = nbhs[0][0];
        SiteCoordinateMatrix weights = SiteCoordinateMatrix::Constant(site.nbhcount_, 3, .2);
        Eigen::VectorXd selected = Eigen::VectorXd::Zero(model.get_NL_coeffs_count());
        model.AccumulateFPGard(site, selected.data(), .23, &weights, workspace);
        REQUIRE(workspace.moments.moment_jacobian_coeffs.size() == 0);
        REQUIRE(workspace.moments.moment_jacobian_coeff_coord.size() == 0);
    }
    GPSettings settings;
    settings.Modify("energy_weight", "1");
    settings.Modify("energy_pow", "0");
    settings.Modify("force_weight", missing_labels ? "1" : "0");
    settings.Modify("stress_weight", missing_labels ? "1" : "0");
    settings.Modify("weight_L2", "0");
    settings.Modify("penalty_weight", "0");
    TrainNonlinear trainer(settings);
    trainer.init(data, nbhs, &model);
    Eigen::VectorXd gradient(model.get_NL_coeffs_count());
    double loss = 0., loss_only = 0.;
    trainer.AccumulateGard(loss, gradient);
    trainer.AccumulateLoss(loss_only);

    Atoms prediction = reference;
    model.getEFS(prediction, nbhs[0]);
    const double error = prediction.energy_ - reference.energy_;
    Eigen::Matrix<double, -1, 3> force = Eigen::MatrixXd::Zero(reference.size(), 3);
    Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();
    SiteCoordinateMatrix workspace;
    Eigen::VectorXd expected = Eigen::VectorXd::Zero(gradient.size());
    model.AccumulateEFSCombinationGard(prediction, nbhs[0], 2. * reference.fitness * error,
                                        force, stress, workspace, expected.data());
    REQUIRE(loss == Approx(reference.fitness * error * error).margin(1e-11));
    REQUIRE(loss_only == Approx(loss).margin(1e-11));
    REQUIRE((gradient - expected).cwiseAbs().maxCoeff() < 2e-10);
    Eigen::VectorXd coefficients(gradient.size());
    model.nl_get_coeffs(coefficients.data());
    for (int i : {0, int(coefficients.size()) / 2, int(coefficients.size()) - 1})
    {
        const double original = coefficients[i], step = 1e-6;
        auto objective = [&](double value)
        {
            coefficients[i] = value;
            model.nl_set_coeffs(coefficients.data());
            Atoms sample = reference;
            model.getEFS(sample, nbhs[0]);
            const double delta = sample.energy_ - reference.energy_;
            return reference.fitness * delta * delta;
        };
        const double plus = objective(original + step);
        const double minus = objective(original - step);
        objective(original);
        CAPTURE(frozen_radial, missing_labels, isolated, i);
        REQUIRE(gradient[i] == Approx((plus - minus) / (2 * step)).margin(2e-7));
    }
}
