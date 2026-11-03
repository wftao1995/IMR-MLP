// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "gpu_force_fd.h"

#include "runtime/mtp/gpu-direct/gpu_backend.h"
#include "base/neighborhood.h"
#include "potentials/mtp_gpu_adapter.h"
#include "potentials/potential_mtp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::size_t checked_product(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        GPERROR("GPU force_fd storage size overflow");
    return a * b;
}

std::size_t bytes_from_mib(int value)
{
    if (value <= 0)
        GPERROR("GPU force_fd memory budgets must be positive");
    return checked_product(static_cast<std::size_t>(value), std::size_t{1} << 20);
}

int checked_dof(std::size_t atoms)
{
    if (atoms > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 3)
        GPERROR("GPU force_fd atom count exceeds the Cartesian index range");
    return static_cast<int>(3 * atoms);
}

#ifdef IMRMLP_GPU
template <class Perturb, class Consume>
void evaluate_pairs(const Atoms &input, const Potentials::AnyPotential &potential,
                    std::size_t pairs, double denominator, double cutoff,
                    const HessianExecutionSettings &settings,
                    HessianExecutionReport &report,
                    Perturb perturb, Consume consume)
{
    if (pairs == 0)
        return;
    if (!std::isfinite(denominator) || denominator <= 0.0)
        GPERROR("GPU force_fd difference denominator must be finite and positive");
    Atoms reference = HessianGpu::geometry_copy(input);
    const std::size_t atoms = reference.size();
    // The adapter's Cartesian and virial flat-array offsets use int products.
    const auto atom_limit = std::min(static_cast<std::size_t>(settings.gpu_batch_atoms),
        static_cast<std::size_t>(std::numeric_limits<int>::max() / 6));
    const std::size_t batch_limit = std::min(
        static_cast<std::size_t>(settings.gpu_batch_structures),
        atom_limit / atoms);
    if (!batch_limit)
        GPERROR("GPU force_fd structure exceeds gpu_batch_atoms");
    const std::size_t frames = checked_product(2, pairs);
    const std::size_t budget = bytes_from_mib(settings.gpu_memory_mib);
    std::string error;
    const auto require = [&](bool ok) {
        if (!ok)
            GPERROR("GPU force_fd: " + error);
    };
    auto start = Clock::now();
    Potentials::MTPGpuEvaluator evaluator;
    require(evaluator.initialize_bounded(potential, settings.gpu_id, budget, &error));
    ++report.evaluator_initializations;
    report.setup_seconds += elapsed(start);
    report.tracked_device_bytes = evaluator.context().memory_stats().tracked_device_bytes;
    Potentials::MTPGpuBatch batch;
    std::vector<Eigen::Vector3d> positive;

    for (std::size_t begin = 0; begin < frames;) {
        const std::size_t count = std::min(batch_limit, frames - begin);
        std::vector<Atoms> candidates;
        std::vector<Neighborhoods> neighborhoods;
        std::vector<const Atoms *> candidate_ptrs;
        std::vector<const AnyNeighborhoods *> neighborhood_ptrs;
        candidates.reserve(count);
        neighborhoods.reserve(count);
        candidate_ptrs.reserve(count);
        neighborhood_ptrs.reserve(count);
        start = Clock::now();
        std::size_t neighbors = 0;
        for (std::size_t j = 0; j < count; ++j) {
            const std::size_t frame = begin + j;
            candidates.push_back(reference);
            auto &candidate = candidates.back();
            perturb(candidate, frame / 2, (frame % 2 == 0) ? 1.0 : -1.0);
            for (std::size_t i = 0; i < atoms; ++i) {
                if (!candidate.pos_[i].allFinite())
                    GPERROR("GPU force_fd perturbation produced nonfinite positions");
                report.maximum_atom_displacement = std::max(report.maximum_atom_displacement,
                    (candidate.pos_[i] - reference.pos_[i]).stableNorm());
            }
            // Each displaced structure can have a different cutoff/image graph.
            neighborhoods.emplace_back(candidate, cutoff);
            for (int s = 0; s < neighborhoods.back().size(); ++s) {
                const auto &site = neighborhoods.back()[s];
                if (site.nbhcount_ < 0 ||
                    static_cast<std::size_t>(site.nbhcount_) >
                        static_cast<std::size_t>(std::numeric_limits<int>::max() / 3) - neighbors)
                    GPERROR("GPU force_fd batch neighbor index overflow");
                neighbors += static_cast<std::size_t>(site.nbhcount_);
            }
            candidate_ptrs.push_back(&candidate);
            neighborhood_ptrs.push_back(&neighborhoods.back());
        }
        report.neighborhoods_seconds += elapsed(start);
        report.peak_batch_structures = std::max(report.peak_batch_structures, count);
        report.peak_batch_atoms = std::max(report.peak_batch_atoms, checked_product(count, atoms));
        report.peak_batch_neighbors = std::max(report.peak_batch_neighbors, neighbors);
        start = Clock::now();
        require(batch.build(candidate_ptrs, neighborhood_ptrs, &error));
        report.packing_seconds += elapsed(start);
        std::vector<Potentials::MTPGpuBatch::InferenceResult> predictions;
        const auto before = evaluator.context().runtime_stats();
        start = Clock::now();
        require(batch.evaluate_inference(evaluator, predictions, false, true, budget, &error));
        report.evaluation_seconds += elapsed(start);
        const auto after = evaluator.context().runtime_stats();
        report.geometry_uploads += after.coordinate_uploads - before.coordinate_uploads;
        report.geometry_reuses += after.input_reuses - before.input_reuses;
        report.download_bytes += after.output_download_bytes - before.output_download_bytes;
        report.tracked_device_bytes = std::max(report.tracked_device_bytes,
            evaluator.context().memory_stats().tracked_device_bytes);
        if (report.tracked_device_bytes > budget)
            GPERROR("GPU force_fd exceeded gpu_memory_mib");
        if (predictions.size() != count)
            GPERROR("GPU force_fd returned an invalid structure count");
        report.evaluated_structures += count;
        ++report.batches;
        start = Clock::now();
        for (std::size_t j = 0; j < count; ++j) {
            const std::size_t frame = begin + j;
            auto &forces = predictions[j].forces;
            if (forces.size() != atoms)
                GPERROR("GPU force_fd returned an invalid force count");
            if (frame % 2 == 0) {
                // At most one unmatched plus frame survives a batch boundary.
                positive = std::move(forces);
            } else {
                for (std::size_t i = 0; i < atoms; ++i) {
                    const Eigen::Vector3d value = -(positive[i] - forces[i]) / denominator;
                    if (!value.allFinite())
                        GPERROR("GPU force_fd produced nonfinite curvature");
                    consume(frame / 2, i, value);
                }
                positive.clear();
            }
        }
        report.reduction_seconds += elapsed(start);
        begin += count;
    }
}
#endif
} // namespace

namespace HessianGpu {

bool begin(const HessianExecutionSettings &settings, HessianExecutionReport &report)
{
    report = {};
    using namespace MLIPCore::GpuDirect;
    report.backend = canonical_device(settings.device);
    if (!valid_device(settings.device))
        GPERROR("Hessian device must be cpu, cuda, hip or dcu");
    if (!device_supported(settings.device))
        GPERROR("Hessian device=" + settings.device + " is unavailable in this " + compiled_backend() + " build");
    if (settings.gpu_id < -1 || settings.gpu_batch_atoms <= 0 || settings.gpu_batch_structures <= 0)
        GPERROR("Hessian GPU device and batch limits are invalid");
    (void)bytes_from_mib(settings.gpu_memory_mib);
    (void)bytes_from_mib(settings.host_memory_mib);
    return report.backend != "cpu";
}

void validate(const Atoms &atom, const Potentials::AnyPotential &potential,
              double cutoff, const HessianExecutionSettings &settings)
{
    (void)checked_dof(atom.size());
    if (typeid(potential) != typeid(Potentials::PotMTP))
        GPERROR("GPU force_fd requires an ordinary PotMTP potential");
    const auto &model = static_cast<const Potentials::PotMTP &>(potential);
    if (model.radial_mode != 0 || model.hc_correction || model.rR_correction || model.add_efs)
        GPERROR("GPU force_fd requires ordinary MTP radial_mode=0 without corrections or add_efs");
    if (model.empty || !model.nl_inited || model.species_count <= 0 ||
        model.get_NL_coeffs_count() <= 0 || !model.regression_coeffs ||
        !model.radial_coeffs || !model.species_coeffs || !model.moment_coeffs)
        GPERROR("GPU force_fd requires an initialized complete ordinary MTP");
    for (int i = 0; i < model.get_NL_coeffs_count(); ++i)
        if (!std::isfinite(model.regression_coeffs[i]))
            GPERROR("GPU force_fd requires finite MTP coefficients");
    if (!std::isfinite(cutoff) || cutoff <= 0.0)
        GPERROR("GPU force_fd requires a finite positive cutoff");
    if (atom.have_ghost || atom.nghost_ != 0 || !atom.map_ghost_.empty() ||
        (atom.nlocal_ != 0 && (atom.nlocal_ < 0 || static_cast<std::size_t>(atom.nlocal_) != atom.size())))
        GPERROR("GPU force_fd requires complete structures without ghosts");
    if (atom.chemmap_.size() != atom.size() || !atom.cell_.allFinite())
        GPERROR("GPU force_fd requires finite geometry and one species per atom");
    if (atom.size() && (!atom.is_nonperiodic() || !atom.cartesian)) {
        const double determinant = atom.cell_.determinant();
        if (!std::isfinite(determinant) || std::abs(determinant) <= 1e-14)
            GPERROR("GPU force_fd requires a nonsingular periodic or fractional cell");
    }
    for (std::size_t i = 0; i < atom.size(); ++i)
        if (!atom.pos_[i].allFinite() || atom.chemmap_[i] < 0 || atom.chemmap_[i] >= model.species_count)
            GPERROR("GPU force_fd positions or species are invalid");
    if (atom.size() > static_cast<std::size_t>(settings.gpu_batch_atoms))
        GPERROR("GPU force_fd structure exceeds gpu_batch_atoms; increase the explicit limit");
#ifdef IMRMLP_GPU
    // Validate even zero/empty queries without creating an evaluator or buffers.
    int devices = 0;
    std::string error;
    if (!MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&devices, &error))
        GPERROR("GPU force_fd device discovery failed: " + error);
    if (!devices || settings.gpu_id >= devices)
        GPERROR("GPU force_fd device index is outside the visible device count");
#endif
    if (atom.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 6))
        GPERROR("GPU force_fd structure exceeds the packed force/virial index range");
}

Atoms geometry_copy(const Atoms &atom)
{
    Atoms result;
    result.cell_ = atom.cell_;
    result.pos_ = atom.pos_;
    result.chemmap_ = atom.chemmap_;
    result.pbc_ = atom.pbc_;
    result.cartesian = atom.cartesian;
    if (!result.cartesian)
        result.to_cartesian();
    result.cartesian = true;
    for (const auto &position : result.pos_)
        if (!position.allFinite())
            GPERROR("GPU force_fd Cartesian conversion produced nonfinite positions");
    result.resetEFS();
    return result;
}

std::size_t check_hessian_storage(std::size_t atoms, bool blocks, bool projection,
                                 const HessianExecutionSettings &settings,
                                 HessianExecutionReport &report)
{
    const std::size_t dof = static_cast<std::size_t>(checked_dof(atoms));
    const auto elements = checked_product(dof, dof);
    if (elements > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()))
        GPERROR("GPU force_fd Hessian exceeds the Eigen index range");
    if (blocks && checked_product(atoms, atoms) >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()))
        GPERROR("GPU force_fd block Hessian exceeds the block index range");
    const auto bytes = blocks ? checked_product(checked_product(atoms, atoms), sizeof(Eigen::Matrix3d))
                              : checked_product(elements, sizeof(double));
    const auto owned = checked_product(bytes, projection ? 2 : 1);
    report.host_hessian_bytes = owned;
    if (owned > bytes_from_mib(settings.host_memory_mib))
        GPERROR("GPU force_fd requires " + std::to_string(owned) +
                " bytes of new Hessian/projection storage; exceeds host_memory_mib");
    return dof;
}

std::size_t check_vector_storage(std::size_t atoms, const HessianExecutionSettings &settings,
                                HessianExecutionReport &report)
{
    const auto dof = static_cast<std::size_t>(checked_dof(atoms));
    report.host_hessian_bytes = checked_product(dof, sizeof(double));
    if (report.host_hessian_bytes > bytes_from_mib(settings.host_memory_mib))
        GPERROR("GPU force_fd HVP output exceeds host_memory_mib");
    return dof;
}

void symmetrize(Eigen::MatrixXd &matrix)
{
    for (Eigen::Index j = 0; j < matrix.cols(); ++j)
        for (Eigen::Index i = 0; i < j; ++i) {
            const double average = 0.5 * matrix(i, j) + 0.5 * matrix(j, i);
            matrix(i, j) = matrix(j, i) = average;
        }
}

Eigen::MatrixXd dense(const Atoms &atom, const Potentials::AnyPotential &potential,
                      double step, double cutoff, bool projection,
                      const HessianExecutionSettings &settings, HessianExecutionReport &report)
{
    if (!std::isfinite(2.0 * step) || step <= 0.0)
        GPERROR("GPU force_fd difference denominator must be finite and positive");
    const auto dof = check_hessian_storage(atom.size(), false, projection, settings, report);
    Eigen::MatrixXd result(dof, dof);
#ifdef IMRMLP_GPU
    evaluate_pairs(atom, potential, dof, 2.0 * step, cutoff, settings, report,
        [step](Atoms &frame, std::size_t column, double sign) {
            frame.pos_[column / 3][column % 3] += sign * step;
        },
        [&](std::size_t column, std::size_t i, const Eigen::Vector3d &value) {
            result.block<3, 1>(3 * i, column) = value;
        });
    const auto start = Clock::now();
    symmetrize(result);
    report.symmetry_seconds += elapsed(start);
#else
    (void)potential; (void)step; (void)cutoff;
    GPERROR("GPU force_fd is unavailable in this CPU build");
#endif
    return result;
}

Potentials::HessianMatrix blocks(const Atoms &atom, const Potentials::AnyPotential &potential,
                                double step, double cutoff,
                                const HessianExecutionSettings &settings, HessianExecutionReport &report)
{
    if (!std::isfinite(2.0 * step) || step <= 0.0)
        GPERROR("GPU force_fd difference denominator must be finite and positive");
    const auto dof = check_hessian_storage(atom.size(), true, false, settings, report);
    Potentials::HessianMatrix result;
    result.resize(static_cast<int>(atom.size()));
#ifdef IMRMLP_GPU
    evaluate_pairs(atom, potential, dof, 2.0 * step, cutoff, settings, report,
        [step](Atoms &frame, std::size_t column, double sign) {
            frame.pos_[column / 3][column % 3] += sign * step;
        },
        [&](std::size_t column, std::size_t i, const Eigen::Vector3d &value) {
            result(static_cast<int>(i), static_cast<int>(column / 3)).col(column % 3) = value;
        });
    const auto start = Clock::now();
    for (int i = 0; i < result.size; ++i) {
        auto &diagonal = result(i, i);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < a; ++b) {
                const double average = 0.5 * diagonal(a, b) + 0.5 * diagonal(b, a);
                diagonal(a, b) = diagonal(b, a) = average;
            }
        for (int j = i + 1; j < result.size; ++j) {
            const Eigen::Matrix3d average = 0.5 * result(i, j) + 0.5 * result(j, i).transpose();
            result(i, j) = average;
            result(j, i) = average.transpose();
        }
    }
    report.symmetry_seconds += elapsed(start);
#else
    (void)dof; (void)potential; (void)step; (void)cutoff;
    GPERROR("GPU force_fd is unavailable in this CPU build");
#endif
    return result;
}

Eigen::VectorXd hvp(const Atoms &atom, const Eigen::VectorXd &direction,
                    const Potentials::AnyPotential &potential, double delta, double cutoff,
                    const HessianExecutionSettings &settings, HessianExecutionReport &report)
{
    if (!std::isfinite(2.0 * delta) || delta <= 0.0)
        GPERROR("GPU force_fd difference denominator must be finite and positive");
    const auto dof = check_vector_storage(atom.size(), settings, report);
    Eigen::VectorXd result(dof);
#ifdef IMRMLP_GPU
    evaluate_pairs(atom, potential, 1, 2.0 * delta, cutoff, settings, report,
        [&](Atoms &frame, std::size_t, double sign) {
            for (int i = 0; i < dof / 3; ++i)
                frame.pos_[i] += sign * delta * direction.segment<3>(3 * i).transpose();
        },
        [&](std::size_t, std::size_t i, const Eigen::Vector3d &value) {
            result.segment<3>(3 * i) = value;
        });
#else
    (void)direction; (void)potential; (void)delta; (void)cutoff;
    GPERROR("GPU force_fd is unavailable in this CPU build");
#endif
    return result;
}

} // namespace HessianGpu
