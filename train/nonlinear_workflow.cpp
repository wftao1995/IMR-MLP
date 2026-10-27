#include "nonlinear_workflow.h"
#include "train_nonlinear_potential.h"
#include "collective_error.h"
#include "utils/comm.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace Training {
WorkflowResult::WorkflowResult() = default;
WorkflowResult::~WorkflowResult() = default;
WorkflowResult::WorkflowResult(WorkflowResult&&) noexcept = default;
WorkflowResult& WorkflowResult::operator=(WorkflowResult&&) noexcept = default;

namespace {
std::array<double, 6> global_sums(std::array<double, 6> local) {
#ifdef IMRMLP_MPI
    std::array<double, 6> total{};
    MPI_Allreduce(local.data(), total.data(), 6, MPI_DOUBLE, MPI_SUM, mpi.comm);
    return total;
#else
    return local;
#endif
}
std::optional<double> metric(const DatasetMetrics& metrics, const std::string& name) {
    if (name == "energy") return metrics.energy;
    if (name == "force") return metrics.force;
    if (name == "stress") return metrics.stress;
    throw std::runtime_error("Workflow validation_metric must be energy, force or stress");
}
std::vector<double> coefficients(Potentials::AnyPotential& potential) {
    std::vector<double> values(potential.get_NL_coeffs_count());
    potential.nl_get_coeffs(values.data());
    return values;
}
std::string real(double value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}
std::string session_suffix() {
    unsigned long long stamp = std::chrono::system_clock::now().time_since_epoch().count();
#ifdef IMRMLP_MPI
    MPI_Bcast(&stamp, 1, MPI_UNSIGNED_LONG_LONG, 0, mpi.comm);
#endif
    return ".workflow-" + std::to_string(stamp);
}
GPSettings settings_for_run(const GPSettings& original, const std::string& suffix) {
    auto settings = original;
    settings.Modify("random_init", "false");
    settings.Modify("force_random", "false");
    settings.Modify("apply_disp", "false");
    // Reports and intermediate checkpoints must belong to this run alone.
    for (const std::string key : {"save_to", "varpro_report", "objective-report"})
        if (!settings[key].empty()) settings.Modify(key, settings[key] + suffix);
    return settings;
}
std::unique_ptr<TrainNonlinear> make_trainer(const GPSettings& settings,
    std::vector<Atoms> data, std::vector<Neighborhoods> neighborhoods,
    Potentials::AnyPotential& potential, const std::vector<Atoms>* valid,
    const std::vector<Neighborhoods>* valid_neighborhoods) {
    auto trainer = std::make_unique<TrainNonlinear>(settings);
    trainer->init(data, neighborhoods, &potential);
    if (valid) {
        auto labels = *valid;
        auto nbhs = *valid_neighborhoods;
        trainer->init_valid(labels, nbhs);
    }
    return trainer;
}
}

DatasetMetrics evaluate_dataset_metrics(Potentials::AnyPotential& potential,
                                       const std::vector<Atoms>& data) {
    std::array<double, 6> sums{};
    const bool previous_add = potential.add_efs;
    potential.add_efs = false;
    try {
        collective_try([&] {
            for (const auto& reference : data) {
                if (reference.size() == 0) continue;
                if (!reference.has_energy() && !reference.has_forces() && !reference.has_stresses()) continue;
                auto prediction = reference;
                if (!prediction.cartesian) prediction.to_cartesian();
                potential.getEFS(prediction);
                if (reference.has_energy()) {
                    const double error = (prediction.energy_ - reference.energy_) / reference.size();
                    sums[0] += error * error;
                    sums[3] += 1;
                }
                if (reference.has_forces()) {
                    if (prediction.forces_.size() != reference.forces_.size())
                        throw std::runtime_error("Workflow force prediction count mismatch");
                    for (int i = 0; i < reference.size(); ++i)
                        sums[1] += (prediction.forces_[i] - reference.forces_[i]).squaredNorm();
                    sums[4] += 3 * reference.size();
                }
                if (reference.has_stresses()) {
                    const double volume = std::abs(reference.cell_.determinant());
                    if (!(volume > 0.) || !std::isfinite(volume))
                        throw std::runtime_error("Workflow stress metrics require finite nonzero cell volume");
                    sums[2] += (prediction.stress_ - reference.stress_).squaredNorm()
                             * std::pow(160.21766208 / volume, 2);
                    sums[5] += 9;
                }
            }
            for (double value : sums)
                if (!std::isfinite(value)) throw std::runtime_error("Nonfinite workflow EFS metrics");
        });
    } catch (...) {
        potential.add_efs = previous_add;
        throw;
    }
    potential.add_efs = previous_add;
    sums = global_sums(sums);
    for (double value : sums)
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite global workflow EFS metrics");
    DatasetMetrics result;
    result.energy_count = static_cast<size_t>(sums[3]);
    result.force_count = static_cast<size_t>(sums[4]);
    result.stress_count = static_cast<size_t>(sums[5]);
    if (sums[3]) result.energy = std::sqrt(sums[0] / sums[3]);
    if (sums[4]) result.force = std::sqrt(sums[1] / sums[4]);
    if (sums[5]) result.stress = std::sqrt(sums[2] / sums[5]);
    return result;
}

WorkflowResult run_nonlinear_workflow(TrainNonlinear& initialized,
    const std::vector<Atoms>& test, const WorkflowOptions& options,
    const std::function<void(TrainNonlinear&)>& run) {
    if (!initialized.p_pot) throw std::runtime_error("Workflow requires an initialized trainer");
    auto& potential = *initialized.p_pot;
    const auto initial = coefficients(potential);
    const auto initial_rng = RNG.mtd;
    const auto initial_fast_rng = RNG.fastrng;
    WorkflowResult result;
    try {
        if (!options.regularization_search && !options.refit_train_valid) {
            run(initialized);
            result.test = evaluate_dataset_metrics(potential, test);
            return result;
        }
        const auto settings = initialized.CurrentSettings();
        const bool special = options.regularization_search || options.refit_train_valid;
        collective_try([&] {
            if (!run) throw std::runtime_error("Workflow requires a training callback");
            if (special && settings["training_schedule"] != "single")
                throw std::runtime_error("Search/refit currently requires training_schedule=single");
            if (special && settings["device"] != "cpu")
                throw std::runtime_error("Search/refit currently requires device=cpu");
            if (special && !settings["shlbfgs_randomperb"].empty()
                && std::stod(settings["shlbfgs_randomperb"]) != 0.)
                throw std::runtime_error("Search/refit does not support stochastic ShLBFGS perturbations");
            if (special && !potential.inited())
                throw std::runtime_error("Search/refit requires initialized potential coefficients");
            if (options.regularization_search) {
                metric({}, options.validation_metric);
                if (options.regularization_multipliers.empty())
                    throw std::runtime_error("Regularization search requires candidate multipliers");
                for (double multiplier : options.regularization_multipliers)
                    if (!(multiplier > 0.) || !std::isfinite(multiplier))
                        throw std::runtime_error("Regularization multipliers must be finite and positive");
            }
        });
        auto counts = global_sums({0, 0, 0, double(initialized.valid_set_.size()), 0, 0});
        if (special && !counts[3]) throw std::runtime_error("Search/refit requires a validation set");
        GPSettings selected_settings = settings;
        const auto suffix = special ? session_suffix() : std::string();
        if (options.regularization_search) {
            std::array<double, 6> labels{};
            for (const auto& a : initialized.valid_set_) {
                if (!a.size()) continue;
                labels[0] += a.has_energy();
                labels[1] += a.has_forces();
                labels[2] += a.has_stresses();
            }
            labels = global_sums(labels);
            const int label = options.validation_metric == "energy" ? 0 : options.validation_metric == "force" ? 1 : 2;
            if (!labels[label]) throw std::runtime_error("Validation set has no labels for workflow selection metric");
            const std::string baseline = !settings["weight_L2"].empty() ? settings["weight_L2"] : settings["weight_reg"];
            const double l2 = baseline.empty() ? 1e-10 : std::stod(baseline);
            if (!(l2 > 0.) || !std::isfinite(l2))
                throw std::runtime_error("Regularization search requires a finite positive base weight_L2");
            double best = std::numeric_limits<double>::infinity();
            std::vector<double> best_coefficients;
            for (size_t i = 0; i < options.regularization_multipliers.size(); ++i) {
                const double multiplier = options.regularization_multipliers[i];
                const double candidate_l2 = l2 * multiplier;
                if (!std::isfinite(candidate_l2)) throw std::runtime_error("Regularization candidate weight overflows");
                potential.nl_set_coeffs(initial.data());
                RNG.mtd = initial_rng;
                RNG.fastrng = initial_fast_rng;
                auto candidate_settings = settings_for_run(settings, suffix + ".candidate-" + std::to_string(i));
                candidate_settings.Modify("weight_L2", real(candidate_l2));
                candidate_settings.Modify("weight_reg", real(candidate_l2));
                auto candidate = make_trainer(candidate_settings, initialized.training_set_,
                    initialized.trainint_set_nbhs_, potential, &initialized.valid_set_, &initialized.valid_set_nbhs_);
                run(*candidate);
                auto validation = evaluate_dataset_metrics(potential, initialized.valid_set_);
                const double score = *metric(validation, options.validation_metric);
                result.candidates.push_back({multiplier, score, validation});
                if (score < best) {
                    best = score;
                    best_coefficients = coefficients(potential);
                    selected_settings = candidate_settings;
                    result.selected_multiplier = multiplier;
                    result.validation = validation;
                    result.final_trainer = std::move(candidate);
                }
            }
            potential.nl_set_coeffs(best_coefficients.data());
        } else {
            run(initialized);
            result.validation = evaluate_dataset_metrics(potential, initialized.valid_set_);
        }
        if (options.refit_train_valid) {
            auto combined = initialized.training_set_;
            combined.insert(combined.end(), initialized.valid_set_.begin(), initialized.valid_set_.end());
            auto neighborhoods = initialized.trainint_set_nbhs_;
            neighborhoods.insert(neighborhoods.end(), initialized.valid_set_nbhs_.begin(), initialized.valid_set_nbhs_.end());
            auto refit = make_trainer(settings_for_run(selected_settings, suffix + ".refit"),
                std::move(combined), std::move(neighborhoods), potential, nullptr, nullptr);
            run(*refit);
            result.final_trainer = std::move(refit);
        }
        const auto& final_training = result.final_trainer ? result.final_trainer->training_set_ : initialized.training_set_;
        result.training = evaluate_dataset_metrics(potential, final_training);
        result.test = evaluate_dataset_metrics(potential, test);
        if (!settings["save_to"].empty())
            collective_try([&] { result.final_trainer->save_current(settings["save_to"]); });
        return result;
    } catch (...) {
        potential.nl_set_coeffs(initial.data());
        RNG.mtd = initial_rng;
        RNG.fastrng = initial_fast_rng;
        throw;
    }
}
}
