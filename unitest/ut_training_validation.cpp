// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp_LES.h"
#include "train/train_nonlinear_potential.h"
#include <cstdlib>

using Catch::Approx;
using Potentials::PotMTP_LES;

namespace {
void init_model(PotMTP_LES &p) {
    p.les_kernel = "sog";
    p.sog_terms = 2;
    p.sog_channels = 3;
    p.train_init();
    for (int c = 0; c < 3; ++c) {
        p.les_species_coeffs[2*c] = .23;
        p.les_species_coeffs[2*c+1] = -.17;
        std::fill_n(p.les_moment_coeffs+c*p.alpha_scalar_moments, p.alpha_scalar_moments, 0.);
        p.les_moment_coeffs[c*p.alpha_scalar_moments] = .35;
        p.model_les.sog_weights[2*c] = 4.2/(c+1);
        p.model_les.sog_weights[2*c+1] = -1.3;
        p.model_les.sog_log_bandwidths[2*c] = -.4;
        p.model_les.sog_log_bandwidths[2*c+1] = .2;
    }
    p.train_init();
    p.set_inited();
}

std::vector<Atoms> data(PotMTP_LES &p, int count) {
    std::vector<Atoms> result;
    for (int i = 0; i < count; ++i) {
        Atoms a;
        a.cell_ << 10.+i,0,0,.8,10.5,0,-.3,.6,11.;
        a.pos_ = {{2.,2.,2.},{4.1+.1*i,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
        a.chemmap_ = {0,1,0,1};
        a.fitness = 1.; a.id = i;
        p.getEFS(a);
        result.push_back(a);
    }
    return result;
}

std::vector<Neighborhoods> neighbors(std::vector<Atoms> &atoms, double cutoff) {
    std::vector<Neighborhoods> result;
    for (auto &a : atoms) result.emplace_back(a, cutoff);
    return result;
}

GPSettings settings() {
    GPSettings s;
    s.Modify("energy_weight", "1"); s.Modify("force_weight", "1");
    s.Modify("stress_weight", "0.1"); s.Modify("penalty_weight", "0");
    s.Modify("bfgs_grad_tol", "1e-20"); s.Modify("bfgs_step_tol", "0");
    s.Modify("bfgs_conv_tol", "0"); s.Modify("parameter_scaling", "none");
    s.Modify("save_every", "100");
    return s;
}

void check_validation(TrainNonlinear &trainer, PotMTP_LES &p) {
    double energy = 0., forces = 0., stress = 0.;
    int atoms = 0;
    for (size_t i = 0; i < trainer.valid_set_.size(); ++i) {
        const auto &target = trainer.valid_set_[i];
        auto ref = target;
        p.getEFS(ref);
        const auto &actual = trainer.efs_valid_set_[i];
        REQUIRE(actual.energy_ == Approx(ref.energy_).margin(1e-9));
        REQUIRE((actual.stress_-ref.stress_).norm() < 1e-8);
        for (int j = 0; j < ref.size(); ++j) {
            REQUIRE((actual.forces_[j]-ref.forces_[j]).norm() < 1e-8);
            forces += (ref.forces_[j]-target.forces_[j]).squaredNorm();
        }
        energy += std::pow((ref.energy_-target.energy_)/ref.size(), 2);
        stress += (ref.stress_-target.stress_).squaredNorm();
        atoms += ref.size();
    }
    REQUIRE(trainer.emonitor_valid.sq_energy_at_error == Approx(energy).margin(1e-12));
    REQUIRE(trainer.emonitor_valid.sq_force_error == Approx(forces).margin(1e-12));
    REQUIRE(trainer.emonitor_valid.sq_stress_error == Approx(stress).margin(1e-12));
    REQUIRE(trainer.emonitor_valid.nforces == atoms);
}
}

TEST_CASE("Validation cadence leaves L-BFGS coefficients unchanged and reports final parameters", "[training][validation]") {
    std::vector<double> previous;
    for (int interval : {1, 2, 10}) {
        PotMTP_LES p("tests/case-mtp-les/les.mtp"); init_model(p);
        auto train = data(p, 3), valid = data(p, 2);
        auto tn = neighbors(train, p.cutoff), vn = neighbors(valid, p.cutoff);
        p.model_les.sog_weights[0] += .7; p.train_init();
        auto s = settings(); s.Modify("validation_interval", std::to_string(interval));
        TrainNonlinear trainer(s); trainer.init(train, tn, &p); trainer.init_valid(valid, vn);
        trainer.ConfigureTrainableGroups({"sog_weights"});
        trainer.TrainNonLinearBFGS(3);
        REQUIRE(trainer.bfgs_accepted_iter == 3);
        REQUIRE(trainer.validation_evaluations == size_t(1+3/interval+(3%interval != 0)));
        check_validation(trainer, p);
        std::vector<double> coeff(p.get_NL_coeffs_count()); p.nl_get_coeffs(coeff.data());
        if (!previous.empty()) for (size_t i = 0; i < coeff.size(); ++i)
            REQUIRE(coeff[i] == Approx(previous[i]).margin(1e-12));
        previous = coeff;
        // A direct gradient call still evaluates validation after the optimizer scope exits.
        const auto count = trainer.validation_evaluations;
        double loss; Eigen::VectorXd gradient(coeff.size()); trainer.AccumulateGard(loss, gradient);
        REQUIRE(trainer.validation_evaluations == count+1);
    }
}

TEST_CASE("Validation interval rejects zero", "[training][validation]") {
    PotMTP_LES p("tests/case-mtp-les/les.mtp"); init_model(p);
    auto train = data(p, 1); auto tn = neighbors(train, p.cutoff);
    auto s = settings(); s.Modify("validation_interval", "0");
    TrainNonlinear trainer(s);
    REQUIRE_THROWS(trainer.init(train, tn, &p));
}

#ifdef IMRMLP_GPU
TEST_CASE("GPU validation preserves training gradients across cached chunks and coefficient changes", "[training][validation][gpu]") {
    int devices = 0; std::string error;
    const bool available = MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&devices, &error) && devices > 0;
    if (std::getenv("IMRMLP_REQUIRE_GPU")) { INFO(error); REQUIRE(available); }
    if (!available) SKIP("No GPU device");
    const bool cache = GENERATE(false, true);
    PotMTP_LES p("tests/case-mtp-les/les.mtp"); init_model(p);
    auto train = data(p, 3), valid = data(p, 2);
    auto tn = neighbors(train, p.cutoff), vn = neighbors(valid, p.cutoff);
    auto s = settings(); TrainNonlinear cpu(s); cpu.init(train, tn, &p); cpu.init_valid(valid, vn);
    s.Modify("device", MLIPCore::GpuDirect::MTPGpuDirect::backend_name());
    s.Modify("gpu_batch_structures", "1"); s.Modify("gpu_cache_batches", cache ? "true" : "false");
    TrainNonlinear gpu(s); gpu.init(train, tn, &p); gpu.init_valid(valid, vn);
    for (int step = 0; step < 3; ++step) {
        p.model_les.sog_weights[0] += .2;
        Eigen::VectorXd cg(p.get_NL_coeffs_count()), gg(cg.size()); double cl, gl;
        cpu.AccumulateGard(cl, cg); gpu.AccumulateGard(gl, gg);
        REQUIRE(gl == Approx(cl).margin(1e-9));
        REQUIRE((cg-gg).norm() < 1e-8);
        check_validation(gpu, p);
    }
    gpu.ConfigureTrainableGroups({"sog_weights"});
    gpu.TrainNonLinearBFGS(3);
    REQUIRE(gpu.validation_evaluations == 7);
    check_validation(gpu, p);
    // Replacing the validation set invalidates both predictions and host batch geometry.
    valid.resize(1); vn.resize(1); gpu.init_valid(valid, vn);
    double loss; Eigen::VectorXd gradient(p.get_NL_coeffs_count()); gpu.AccumulateGard(loss, gradient);
    check_validation(gpu, p);
}
#endif
