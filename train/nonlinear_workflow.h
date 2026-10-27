#pragma once

#include "base/structures.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class TrainNonlinear;
namespace Potentials { class AnyPotential; }

namespace Training {
struct WorkflowOptions {
    bool refit_train_valid = false;
    bool regularization_search = false;
    std::vector<double> regularization_multipliers{.1, 1., 10.};
    std::string validation_metric = "force";
};
struct DatasetMetrics {
    std::optional<double> energy, force, stress;
    // Number of labeled structures, force components and tensor components.
    size_t energy_count = 0, force_count = 0, stress_count = 0;
};
struct RegularizationCandidate {
    double multiplier = 1., score = 0.;
    DatasetMetrics validation;
};
struct WorkflowResult {
    // Null retains the original trainer, including its optimizer completion state.
    std::unique_ptr<TrainNonlinear> final_trainer;
    DatasetMetrics training, validation, test;
    std::vector<RegularizationCandidate> candidates;
    std::optional<double> selected_multiplier;
    WorkflowResult();
    ~WorkflowResult();
    WorkflowResult(WorkflowResult&&) noexcept;
    WorkflowResult& operator=(WorkflowResult&&) noexcept;
};
DatasetMetrics evaluate_dataset_metrics(Potentials::AnyPotential& potential,
                                       const std::vector<Atoms>& data);
WorkflowResult run_nonlinear_workflow(TrainNonlinear& initialized,
    const std::vector<Atoms>& test, const WorkflowOptions& options,
    const std::function<void(TrainNonlinear&)>& run);
}
