#include <catch2/catch_all.hpp>
#include "train/nonlinear_workflow.h"
#include "unitest/nonlinear_cg_fixture.h"
#include <set>

namespace {
class WorkflowPotential : public CGFixture::Potential {
public:
    int test_evaluations = 0;
    std::string published_path;
    Eigen::Vector2d published_coefficients;
    void Save(const std::string& path) override {
        published_path = path;
        published_coefficients = coefficients;
        ++saves;
    }
    void getEFS(Atoms& atom, AnyNeighborhoods& nbhs) override {
        if (atom.id == 99) ++test_evaluations;
        CGFixture::Potential::getEFS(atom, nbhs);
    }
};
Atoms label(int id, double energy = 2.) {
    Atoms a;
    a.id = id;
    a.cell_ = Eigen::Matrix3d::Identity() * 10.;
    a.pos_ = {{1., 1., 1.}};
    a.chemmap_ = {0};
    a.energy_ = energy;
    a.have_energy = true;
    a.forces_ = {{-1., 0., 0.}};
    a.stress_.setZero();
    a.stress_(0, 0) = -1.;
    a.have_stress = true;
    return a;
}
void initialize(TrainNonlinear& trainer, WorkflowPotential& potential) {
    std::vector<Atoms> training{label(10)}, validation{label(42)};
    std::vector<Neighborhoods> neighborhoods, valid_neighborhoods;
    neighborhoods.emplace_back(training[0], potential.cutoff);
    valid_neighborhoods.emplace_back(validation[0], potential.cutoff);
    trainer.init(training, neighborhoods, &potential);
    trainer.init_valid(validation, valid_neighborhoods);
}
GPSettings settings() {
    GPSettings s;
    s.Modify("weight_L2", "1");
    s.Modify("bfgs_maxiteration", "7");
    s.Modify("save_to", "unused-workflow-potential");
    return s;
}
}

TEST_CASE("Workflow metrics use per-atom energy component forces and full tensor GPa",
          "[nonlinear][workflow][metrics]") {
    WorkflowPotential potential;
    potential.coefficients << 2., .5;
    potential.add_efs = true;
    auto target = label(99);
    target.energy_ = -.5;
    target.pos_.push_back({2., 1., 1.});
    target.chemmap_.push_back(0);
    target.forces_.assign(2, Eigen::Vector3d::Zero());
    target.stress_.setZero();
    target.to_fractional();
    const auto metrics = Training::evaluate_dataset_metrics(potential, {target});
    REQUIRE(*metrics.energy == Catch::Approx(2.5));
    REQUIRE(*metrics.force == Catch::Approx(4. / std::sqrt(6.)));
    REQUIRE(*metrics.stress == Catch::Approx(4. * 160.21766208 / 1000. / 3.));
    REQUIRE(metrics.energy_count == 1);
    REQUIRE(metrics.force_count == 6);
    REQUIRE(metrics.stress_count == 9);
    REQUIRE(potential.add_efs);
    auto unlabeled = target;
    unlabeled.have_energy = false;
    unlabeled.have_stress = false;
    unlabeled.forces_.clear();
    const auto absent = Training::evaluate_dataset_metrics(potential, {unlabeled});
    REQUIRE_FALSE(absent.energy);
    REQUIRE_FALSE(absent.force);
    REQUIRE_FALSE(absent.stress);
    REQUIRE(potential.test_evaluations == 1);
}

TEST_CASE("Default nonlinear workflow preserves trainer and skips extra dataset predictions",
          "[nonlinear][workflow]") {
    WorkflowPotential potential;
    TrainNonlinear trainer(settings());
    initialize(trainer, potential);
    int calls = 0;
    const auto result = Training::run_nonlinear_workflow(trainer, {}, {}, [&](TrainNonlinear& current) {
        REQUIRE(&current == &trainer);
        ++calls;
        current.cg_termination_type = 4;
    });
    REQUIRE(calls == 1);
    REQUIRE_FALSE(result.final_trainer);
    REQUIRE(trainer.cg_termination_type == 4);
    REQUIRE(potential.evaluations == 0);
}

TEST_CASE("Regularization candidates share initialized snapshot and refit excludes test labels",
          "[nonlinear][workflow][selection]") {
    const bool refit = GENERATE(false, true);
    WorkflowPotential potential;
    TrainNonlinear trainer(settings());
    initialize(trainer, potential);
    const auto starting = potential.coefficients;
    Training::WorkflowOptions options;
    options.regularization_search = true;
    options.refit_train_valid = refit;
    int calls = 0;
    std::set<std::string> save_paths, report_paths;
    std::vector<unsigned int> draws;
    const auto result = Training::run_nonlinear_workflow(trainer, {label(99, 100000.)}, options,
        [&](TrainNonlinear& current) {
            REQUIRE(potential.test_evaluations == 0);
            REQUIRE(&current != &trainer);
            const auto snapshot = current.CurrentSettings();
            REQUIRE(snapshot["bfgs_maxiteration"] == "7");
            REQUIRE(snapshot["random_init"] == "false");
            REQUIRE(snapshot["force_random"] == "false");
            REQUIRE(snapshot["apply_disp"] == "false");
            save_paths.insert(snapshot["save_to"]);
            report_paths.insert(snapshot["varpro_report"]);
            if (calls < 3) {
                REQUIRE(current.training_set_.size() == 1);
                REQUIRE(current.valid_set_.size() == 1);
                REQUIRE((potential.coefficients - starting).norm() == 0.);
                draws.push_back(RNG.mtd());
                potential.coefficients << std::sqrt(std::stod(snapshot["weight_L2"])), .5;
            } else {
                REQUIRE(current.training_set_.size() == 2);
                REQUIRE(current.valid_set_.empty());
                REQUIRE(potential.coefficients[0] == Catch::Approx(1.));
                potential.coefficients[1] = 2.;
            }
            for (const auto& a : current.training_set_) REQUIRE(a.id != 99);
            for (const auto& a : current.valid_set_) REQUIRE(a.id != 99);
            ++calls;
        });
    REQUIRE(calls == (refit ? 4 : 3));
    REQUIRE(result.final_trainer);
    REQUIRE(result.candidates.size() == 3);
    REQUIRE(*result.selected_multiplier == Catch::Approx(1.));
    REQUIRE(*result.validation.force == Catch::Approx(0.));
    REQUIRE(potential.coefficients[0] == Catch::Approx(1.));
    REQUIRE(potential.coefficients[1] == Catch::Approx(refit ? 2. : .5));
    REQUIRE(potential.published_path == "unused-workflow-potential");
    REQUIRE((potential.published_coefficients - potential.coefficients).norm() == 0.);
    REQUIRE(save_paths.size() == calls);
    REQUIRE(report_paths.size() == calls);
    REQUIRE(draws[0] == draws[1]);
    REQUIRE(draws[1] == draws[2]);
    REQUIRE(potential.test_evaluations == 1);
    const double expected_test_energy = std::abs((1. * 1. + (refit ? 2. : .5)) - 100000.);
    REQUIRE(*result.test.energy == Catch::Approx(expected_test_energy));
}

TEST_CASE("Nonlinear workflow restores original coefficients on failure and rejects missing labels",
          "[nonlinear][workflow][failure]") {
    WorkflowPotential potential;
    TrainNonlinear trainer(settings());
    initialize(trainer, potential);
    const auto starting = potential.coefficients;
    Training::WorkflowOptions options;
    options.regularization_search = true;
    REQUIRE_THROWS(Training::run_nonlinear_workflow(trainer, {label(99)}, options,
        [&](TrainNonlinear&) {
            potential.coefficients.setZero();
            throw std::runtime_error("injected workflow failure");
        }));
    REQUIRE((potential.coefficients - starting).norm() == 0.);
    REQUIRE(potential.test_evaluations == 0);
    for (auto& a : trainer.valid_set_) a.forces_.clear();
    int calls = 0;
    REQUIRE_THROWS(Training::run_nonlinear_workflow(trainer, {}, options,
        [&](TrainNonlinear&) { ++calls; }));
    REQUIRE(calls == 0);
}

TEST_CASE("Refit alone starts from trained coefficients and disables initialization mutations",
          "[nonlinear][workflow][refit]") {
    WorkflowPotential potential;
    TrainNonlinear trainer(settings());
    initialize(trainer, potential);
    GPSettings initialization_flags;
    initialization_flags.Modify("random_init", "true");
    initialization_flags.Modify("force_random", "true");
    initialization_flags.Modify("apply_disp", "true");
    trainer.ApplySettings(initialization_flags);
    Training::WorkflowOptions options;
    options.refit_train_valid = true;
    int calls = 0;
    const auto result = Training::run_nonlinear_workflow(trainer, {label(99)}, options,
        [&](TrainNonlinear& current) {
            if (calls++ == 0) {
                REQUIRE(&current == &trainer);
                potential.coefficients << 1., 2.;
            } else {
                REQUIRE(&current != &trainer);
                REQUIRE(current.training_set_.size() == 2);
                REQUIRE(current.valid_set_.empty());
                REQUIRE(potential.coefficients[0] == Catch::Approx(1.));
                REQUIRE(potential.coefficients[1] == Catch::Approx(2.));
                REQUIRE(current.CurrentSettings()["force_random"] == "false");
                REQUIRE(current.CurrentSettings()["random_init"] == "false");
                REQUIRE(current.CurrentSettings()["apply_disp"] == "false");
            }
            REQUIRE(potential.test_evaluations == 0);
        });
    REQUIRE(calls == 2);
    REQUIRE(result.final_trainer);
    REQUIRE(potential.test_evaluations == 1);
    REQUIRE(result.candidates.empty());
}
