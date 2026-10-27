#include <catch2/catch_all.hpp>
#include "train/train_mtp.h"
#include "train/train_nonlinear_potential.h"
#include <filesystem>
#include <algorithm>
#include <type_traits>

namespace {
struct SGDData {
    Potentials::PotMTP potential{"unitest/test-data/portable-mtp/les.mtp"};
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> neighbors;
    SGDData() {
        potential.train_init(); potential.set_inited();
        Atoms atom; atom.cell_ = 12. * Eigen::Matrix3d::Identity();
        atom.pos_ = {{2.,2.,2.},{4.1,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
        atom.chemmap_ = {0,1,0,1};
        potential.getEFS(atom); atom.energy_ += .1;
        atoms.push_back(atom); neighbors.emplace_back(atom, potential.cutoff);
    }
};
GPSettings sgd_controls(const std::string &path) {
    GPSettings settings;
    settings.Modify("energy_weight", "1"); settings.Modify("force_weight", "0");
    settings.Modify("stress_weight", "0"); settings.Modify("penalty_weight", "0");
    settings.Modify("weight_L2", "0"); settings.Modify("bfgs_rescale", "false");
    settings.Modify("epoch_limit", "1"); settings.Modify("batch_size", "1");
    settings.Modify("learning_rate", ".0001"); settings.Modify("momentum", "0");
    settings.Modify("save_to", path); settings.Modify("save_every", "100");
    return settings;
}
Eigen::VectorXd sgd_coefficients(Potentials::PotMTP &potential) {
    Eigen::VectorXd result(potential.get_NL_coeffs_count());
    potential.nl_get_coeffs(result.data()); return result;
}
void remove_sgd_output(const std::string &path) {
    std::filesystem::remove(path); std::filesystem::remove(path + "0");
}
}

TEMPLATE_TEST_CASE("SGD switching a reused trainer to none restores a constant learning rate", "[sgd][training]",
    Trainer_MTP, TrainNonlinear) {
    SGDData reused, fresh;
    auto controls = sgd_controls("sgd-reused.mtp");
    controls.Modify("lr_scheduler", "step"); controls.Modify("lr_decay_steps", "1");
    controls.Modify("lr_decay_rate", ".2"); controls.Modify("epoch_limit", "2");
    TestType trainer(controls); trainer.init(reused.atoms, reused.neighbors, &reused.potential);
    trainer.TrainSGD();
    fresh.potential.nl_set_coeffs(sgd_coefficients(reused.potential).data());
    controls.Modify("lr_scheduler", "none"); controls.Modify("epoch_limit", "2");
    controls.Modify("learning_rate", ".0002"); trainer.ApplySettings(controls);
    auto fresh_controls = controls; fresh_controls.Modify("save_to", "sgd-fresh.mtp");
    TestType reference(fresh_controls); reference.init(fresh.atoms, fresh.neighbors, &fresh.potential);
    trainer.TrainSGD(); reference.TrainSGD();
    REQUIRE((sgd_coefficients(reused.potential) - sgd_coefficients(fresh.potential)).norm() < 1e-13);
    remove_sgd_output("sgd-reused.mtp"); remove_sgd_output("sgd-fresh.mtp");
}

TEMPLATE_TEST_CASE("SGD rejects invalid controls before changing the model or saving", "[sgd][training]",
    Trainer_MTP, TrainNonlinear) {
    const int invalid = GENERATE(0,1,2,3,4,5,6);
    SGDData data;
    auto controls = sgd_controls("sgd-invalid.mtp");
    TestType trainer(controls); trainer.init(data.atoms, data.neighbors, &data.potential);
    const auto before = sgd_coefficients(data.potential);
    if (invalid == 0) controls.Modify("lr_scheduler", "cosnie");
    if (invalid == 1) {controls.Modify("lr_scheduler", "step"); controls.Modify("lr_decay_steps", "0");}
    if (invalid == 2) controls.Modify("learning_rate", "-1");
    if (invalid == 3) controls.Modify("momentum", "-1");
    if (invalid == 4) controls.Modify("clip_gradient", "-1");
    if (invalid == 5) {controls.Modify("lr_scheduler", "cosine"); controls.Modify("lr_final", "-1");}
    if (invalid == 6) {controls.Modify("lr_scheduler", "step"); controls.Modify("lr_warmup_steps", "-1");}
    trainer.ApplySettings(controls);
    remove_sgd_output("sgd-invalid.mtp");
    REQUIRE_THROWS(trainer.TrainSGD());
    REQUIRE((sgd_coefficients(data.potential) - before).norm() == 0.);
    REQUIRE_FALSE(std::filesystem::exists("sgd-invalid.mtp"));
    REQUIRE_FALSE(std::filesystem::exists("sgd-invalid.mtp0"));
}

TEMPLATE_TEST_CASE("Zero-epoch SGD with cosine is an empty operation", "[sgd][training]", Trainer_MTP, TrainNonlinear) {
    SGDData data;
    auto controls = sgd_controls("sgd-zero.mtp");
    controls.Modify("epoch_limit", "0"); controls.Modify("lr_scheduler", "cosine");
    TestType trainer(controls); trainer.init(data.atoms, data.neighbors, &data.potential);
    const auto before = sgd_coefficients(data.potential);
    remove_sgd_output("sgd-zero.mtp");
    if constexpr (std::is_same_v<TestType, TrainNonlinear>) trainer.TrainNonLinearSGD(0);
    else trainer.TrainSGD();
    REQUIRE((sgd_coefficients(data.potential) - before).norm() == 0.);
    REQUIRE_FALSE(std::filesystem::exists("sgd-zero.mtp"));
    REQUIRE_FALSE(std::filesystem::exists("sgd-zero.mtp0"));
}

TEST_CASE("MTP force-only gradients and SGD updates match the generic trainer", "[sgd][training][mtp][force]") {
    const bool missing_energy = GENERATE(false, true);
    SGDData dedicated, generic;
    for (auto *data : {&dedicated, &generic}) {
        data->atoms[0].forces_[0][0] += .1;
        data->atoms[0].have_energy = !missing_energy;
    }
    auto controls = sgd_controls("sgd-force-only.mtp");
    controls.Modify("energy_weight", missing_energy ? "1" : "0");
    controls.Modify("force_weight", "1");
    Trainer_MTP trainer(controls);
    trainer.init(dedicated.atoms, dedicated.neighbors, &dedicated.potential);
    TrainNonlinear reference(controls);
    reference.init(generic.atoms, generic.neighbors, &generic.potential);
    double actual_loss = 0., expected_loss = 0.;
    alglib::real_1d_array actual_gradient;
    actual_gradient.setlength(trainer.get_relaxed_coeffs_count());
    Eigen::VectorXd expected_gradient(generic.potential.get_NL_coeffs_count());
    trainer.AccumulateGard(actual_loss, actual_gradient);
    reference.AccumulateGard(expected_loss, expected_gradient);
    const Eigen::Map<const Eigen::VectorXd> gradient(actual_gradient.getcontent(), actual_gradient.length());
    REQUIRE(actual_loss == Catch::Approx(expected_loss).margin(1e-14));
    REQUIRE(expected_gradient.norm() > 1e-8);
    REQUIRE((gradient - expected_gradient).norm() < 1e-11 * std::max(1., expected_gradient.norm()));
    const auto before = sgd_coefficients(dedicated.potential);
    trainer.TrainSGD();
    const auto after = sgd_coefficients(dedicated.potential);
    REQUIRE((after - before + .0001 * expected_gradient).norm() < 1e-13 * std::max(1., before.norm()));
    REQUIRE((after - before).norm() > 1e-12);
    remove_sgd_output("sgd-force-only.mtp");
}
