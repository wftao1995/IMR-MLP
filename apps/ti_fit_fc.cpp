#include "apps/apps.h"

#include "analyse/compute_hessian.h"
#include "analyse/trajectory_transform.h"
#include "apps/ti_reference_app.h"
#include "io/gpio.h"
#include "potentials/potential_wrapper.h"
#include "tools/chem.h"
#include "tools/hessian_io.h"
#include "tools/hessian_utils.h"
#include "tools/ti_harmonic_reference.h"
#include "tools/ti_reference_io.h"
#include "tools/tools.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

std::vector<double> atomic_masses(const Atoms &atoms, GPIO &io)
{
    auto masses_by_type = io.atomradii;
    if (masses_by_type.empty()) {
        for (const auto &[type, name] : io.getmap()) {
            const auto found = Chemicals::atomic_numbers.find(name);
            if (found == Chemicals::atomic_numbers.end())
                throw std::invalid_argument("Unknown element in index map: " + name);
            masses_by_type[type] = Chemicals::atomic_masses[found->second];
        }
    }
    std::vector<double> masses(atoms.size());
    for (int i = 0; i < atoms.size(); ++i) {
        const auto found = masses_by_type.find(atoms.chemmap_[i]);
        if (found == masses_by_type.end() || !std::isfinite(found->second) || found->second <= 0.0)
            throw std::invalid_argument("Missing positive mass for atom type "
                                        + std::to_string(atoms.chemmap_[i]));
        masses[i] = found->second;
    }
    return masses;
}

TIHarmonicReference::RootRepresentation parse_root(const std::string &value)
{
    if (value == "sqrt") return TIHarmonicReference::RootRepresentation::SquareRoot;
    if (value == "root4") return TIHarmonicReference::RootRepresentation::FourthRoot;
    if (value == "normal") return TIHarmonicReference::RootRepresentation::Normal;
    throw std::invalid_argument("--root-representation must be normal, sqrt, or root4");
}

TIHarmonicReference::StabilizationPolicy parse_stabilization(const std::string &value)
{
    if (value == "negative-floor")
        return TIHarmonicReference::StabilizationPolicy::NegativeFloor;
    if (value == "spectral-floor")
        return TIHarmonicReference::StabilizationPolicy::SpectralFloor;
    if (value == "abs")
        return TIHarmonicReference::StabilizationPolicy::AbsoluteValue;
    throw std::invalid_argument(
        "--initial-stabilization must be negative-floor, spectral-floor, or abs");
}

std::vector<TIHarmonicReference::ForceSample> evaluate_frames(
    std::vector<Atoms> frames,
    const Atoms &reference,
    const std::vector<double> &masses,
    Potentials::AnyPotential *potential,
    bool use_trajectory_forces)
{
    std::vector<TIHarmonicReference::ForceSample> samples;
    samples.reserve(frames.size());
    for (auto &frame : frames) {
        if (!use_trajectory_forces) potential->getEFSOMP(frame);
        if (frame.forces_.size() != static_cast<size_t>(frame.size()))
            throw std::invalid_argument("A trajectory frame does not contain forces");
        TIHarmonicReference::ForceSample sample;
        sample.displacement_cartesian = minimum_image_displacement(
            reference, frame, masses, true);
        sample.force_cartesian.resize(3 * frame.size());
        for (int i = 0; i < frame.size(); ++i)
            sample.force_cartesian.segment<3>(3 * i) = frame.forces_[i];
        if (frame.has_energy()) sample.energy = frame.energy_;
        samples.push_back(std::move(sample));
    }
    return samples;
}

double minimum_distance(Atoms &atoms)
{
    double result = std::numeric_limits<double>::infinity();
    for (int i = 0; i < atoms.size(); ++i)
        for (int j = i + 1; j < atoms.size(); ++j)
            result = std::min(result, atoms.get_rab(i, j).norm());
    return result;
}

struct EnsembleSafetyDiagnostics {
    int attempted_samples = 0;
    int rejected_samples = 0;
    int displacement_rejected_samples = 0;
    int distance_rejected_samples = 0;
    double maximum_attempted_displacement = 0.0;
    double minimum_attempted_distance = std::numeric_limits<double>::infinity();
};

struct EnsembleSafetyAssessment {
    double maximum_displacement = 0.0;
    double minimum_distance = std::numeric_limits<double>::infinity();
    bool displacement_rejected = false;
    bool distance_rejected = false;
};

EnsembleSafetyAssessment assess_ensemble_structure(
    const Atoms &reference,
    const TIHarmonicReference::EnsembleSample &sample,
    double maximum_displacement,
    double minimum_distance_ratio,
    double reference_minimum_distance)
{
    Atoms generated = reference;
    EnsembleSafetyAssessment assessment;
    for (int i = 0; i < generated.size(); ++i) {
        const Eigen::Vector3d displacement = sample.displacement_cartesian.segment<3>(3 * i);
        assessment.maximum_displacement = std::max(
            assessment.maximum_displacement, displacement.norm());
        generated.pos_[i] += displacement.transpose();
    }
    generated.wrap();
    assessment.minimum_distance = minimum_distance(generated);
    assessment.displacement_rejected = assessment.maximum_displacement > maximum_displacement;
    assessment.distance_rejected = assessment.minimum_distance
        < minimum_distance_ratio * reference_minimum_distance;
    return assessment;
}

bool accept_ensemble_group(
    const Atoms &reference,
    std::span<const TIHarmonicReference::EnsembleSample> candidates,
    double maximum_displacement,
    double minimum_distance_ratio,
    double reference_minimum_distance,
    EnsembleSafetyDiagnostics &diagnostics)
{
    bool displacement_rejected = false;
    bool distance_rejected = false;
    for (const auto &candidate : candidates) {
        const auto assessment = assess_ensemble_structure(
            reference, candidate, maximum_displacement,
            minimum_distance_ratio, reference_minimum_distance);
        diagnostics.maximum_attempted_displacement = std::max(
            diagnostics.maximum_attempted_displacement, assessment.maximum_displacement);
        diagnostics.minimum_attempted_distance = std::min(
            diagnostics.minimum_attempted_distance, assessment.minimum_distance);
        displacement_rejected = displacement_rejected || assessment.displacement_rejected;
        distance_rejected = distance_rejected || assessment.distance_rejected;
    }
    diagnostics.attempted_samples += static_cast<int>(candidates.size());
    if (!displacement_rejected && !distance_rejected) return true;
    diagnostics.rejected_samples += static_cast<int>(candidates.size());
    if (displacement_rejected)
        diagnostics.displacement_rejected_samples += static_cast<int>(candidates.size());
    if (distance_rejected)
        diagnostics.distance_rejected_samples += static_cast<int>(candidates.size());
    return false;
}

std::string sampling_failure_message(
    const std::string &message,
    const EnsembleSafetyDiagnostics &diagnostics,
    double maximum_displacement,
    double minimum_distance_ratio,
    double reference_minimum_distance)
{
    return fmt::format(
        "{}; rejected {} of {} attempted samples (displacement: {}, distance: {}), "
        "maximum displacement {:.8g} Angstrom (limit {:.8g}), minimum distance {:.8g} "
        "Angstrom (limit {:.8g} = {:.8g} x reference {:.8g})",
        message, diagnostics.rejected_samples, diagnostics.attempted_samples,
        diagnostics.displacement_rejected_samples, diagnostics.distance_rejected_samples,
        diagnostics.maximum_attempted_displacement, maximum_displacement,
        diagnostics.minimum_attempted_distance,
        minimum_distance_ratio * reference_minimum_distance,
        minimum_distance_ratio, reference_minimum_distance);
}

Atoms evaluate_ensemble_structure(
    const Atoms &reference,
    const TIHarmonicReference::EnsembleSample &sample,
    Potentials::AnyPotential *potential)
{
    Atoms generated = reference;
    for (int i = 0; i < generated.size(); ++i)
        generated.pos_[i] += sample.displacement_cartesian.segment<3>(3 * i).transpose();
    generated.wrap();
    potential->getEFSOMP(generated);
    if (!generated.has_energy() || !std::isfinite(generated.energy_)
        || generated.forces_.size() != static_cast<size_t>(generated.size())
        || !std::all_of(generated.forces_.begin(), generated.forces_.end(),
                        [](const Eigen::RowVector3d &force) { return force.allFinite(); }))
        throw std::runtime_error("Potential returned non-finite energy or forces for an ensemble sample");
    return generated;
}

void write_iteration_report(const std::string &prefix,
                            const std::string &workflow,
                            const std::string &initial_stabilization,
                            const TIHarmonicReference::Diagnostics &initial_diagnostics,
                            double temperature,
                            const TIHarmonicReference::ReferenceResult &reference,
                            const TIHarmonicReference::SelfConsistentResult *iteration,
                            double training_rmse,
                            double validation_force_rmse,
                            double final_md_force_rmse,
                            double harmonic_validation_force_rmse,
                            int trajectory_frames,
                            const std::string &sparse_source,
                            const TIReferenceIO::SparseDiagnostics *sparse,
                            const Eigen::VectorXd &lammps_eigenvalues,
                            double lammps_asr_error,
                            const EnsembleSafetyDiagnostics *iteration_safety,
                            const TIHarmonicReference::EnsembleSamplingResult *validation_sampling,
                            const EnsembleSafetyDiagnostics *validation_safety)
{
    std::ofstream report(prefix + "_REPORT.dat");
    if (!report) throw std::runtime_error("Cannot open TI FC report output");
    report << std::setprecision(16);
    report << "# ti_fit_fc\nworkflow " << workflow
           << "\ninitial_stabilization " << initial_stabilization
           << "\ninitial_corrected_modes " << initial_diagnostics.corrected_modes
           << "\ninitial_minimum_frequency_before_thz "
           << TIHarmonicReference::signed_frequency_thz(
                  initial_diagnostics.eigenvalues_before.minCoeff())
           << "\ninitial_minimum_frequency_after_thz "
           << TIHarmonicReference::signed_frequency_thz(
                  initial_diagnostics.eigenvalues_after.minCoeff())
           << "\ninitial_relative_hessian_change "
           << initial_diagnostics.relative_hessian_change
           << "\ntemperature_kelvin " << temperature
           << "\ntrajectory_frames " << trajectory_frames << '\n';
    if (std::isfinite(training_rmse)) {
        report << "training_force_rmse " << training_rmse << '\n';
        report << "md_selection_training_force_rmse " << training_rmse << '\n';
    }
    if (std::isfinite(validation_force_rmse)) {
        report << "validation_force_rmse " << validation_force_rmse << '\n';
        report << "md_selection_validation_force_rmse " << validation_force_rmse << '\n';
    }
    if (std::isfinite(final_md_force_rmse))
        report << "final_md_force_rmse " << final_md_force_rmse << '\n';
    if (std::isfinite(harmonic_validation_force_rmse))
        report << "harmonic_validation_force_rmse "
               << harmonic_validation_force_rmse << '\n';
    if (sparse) {
        report << "lammps_sparse_source " << sparse_source
               << "\nlammps_sparse_total_offdiagonal_blocks "
               << sparse->total_offdiagonal_blocks
               << "\nlammps_sparse_retained_offdiagonal_blocks "
               << sparse->retained_offdiagonal_blocks
               << "\nlammps_sparse_retained_fraction " << sparse->retained_fraction
               << "\nlammps_sparse_relative_hessian_change "
               << sparse->relative_hessian_change
               << "\nlammps_sparse_asr_error " << sparse->acoustic_sum_rule_error
               << "\nlammps_sparse_minimum_frequency_before_thz "
               << sparse->minimum_frequency_before_thz
               << "\nlammps_sparse_minimum_frequency_after_thz "
               << sparse->minimum_frequency_after_thz
               << "\nlammps_sparse_isotropic_stabilizer "
               << sparse->isotropic_stabilizer << '\n';
    }
    report << "asr_error " << reference.diagnostics.acoustic_sum_rule_error_after
           << "\nspace_group_error " << reference.diagnostics.space_group_error_after
           << "\nminimum_frequency_thz "
           << TIHarmonicReference::signed_frequency_thz(
                  reference.diagnostics.eigenvalues_after.minCoeff())
           << "\ncondition_number_internal " << reference.diagnostics.condition_number_internal
           << "\nlog_determinant_internal " << reference.diagnostics.log_determinant_internal << '\n';
    if (lammps_eigenvalues.size() > 0) {
        report << "lammps_asr_error " << lammps_asr_error
               << "\nlammps_minimum_frequency_thz "
               << TIHarmonicReference::signed_frequency_thz(lammps_eigenvalues.minCoeff())
               << "\nlammps_condition_number_internal "
               << lammps_eigenvalues.maxCoeff() / lammps_eigenvalues.minCoeff()
               << "\nlammps_log_determinant_internal "
               << lammps_eigenvalues.array().log().sum() << '\n';
    }
    if (iteration) {
        int attempts = 0;
        int rejected = 0;
        for (const auto &record : iteration->iterations) {
            attempts += record.sampling_attempts;
            rejected += record.rejected_samples;
        }
        report << "self_consistent_converged " << (iteration->converged ? 1 : 0)
               << "\nself_consistent_iterations " << iteration->iterations.size()
               << "\nsampling_attempts " << attempts
               << "\nsampling_rejected_samples " << rejected
               << "\nsampling_acceptance_rate "
               << (attempts > 0 ? static_cast<double>(attempts - rejected) / attempts : 1.0)
               << '\n';
    }
    if (iteration_safety) {
        report << "sampling_displacement_rejected_samples "
               << iteration_safety->displacement_rejected_samples
               << "\nsampling_distance_rejected_samples "
               << iteration_safety->distance_rejected_samples
               << "\nsampling_maximum_attempted_displacement "
               << iteration_safety->maximum_attempted_displacement
               << "\nsampling_minimum_attempted_distance "
               << iteration_safety->minimum_attempted_distance << '\n';
    }
    if (validation_sampling && validation_safety) {
        report << "validation_sampling_attempts " << validation_sampling->attempted_samples
               << "\nvalidation_rejected_samples " << validation_sampling->rejected_samples
               << "\nvalidation_acceptance_rate "
               << (validation_sampling->attempted_samples > 0
                   ? static_cast<double>(validation_sampling->samples.size())
                     / validation_sampling->attempted_samples : 1.0)
               << "\nvalidation_displacement_rejected_samples "
               << validation_safety->displacement_rejected_samples
               << "\nvalidation_distance_rejected_samples "
               << validation_safety->distance_rejected_samples
               << "\nvalidation_maximum_attempted_displacement "
               << validation_safety->maximum_attempted_displacement
               << "\nvalidation_minimum_attempted_distance "
               << validation_safety->minimum_attempted_distance << '\n';
    }
    report << "# mode eigenvalue frequency_thz\n";
    for (Eigen::Index i = 0; i < reference.diagnostics.eigenvalues_after.size(); ++i)
        report << i << ' ' << reference.diagnostics.eigenvalues_after(i) << ' '
               << TIHarmonicReference::signed_frequency_thz(
                      reference.diagnostics.eigenvalues_after(i)) << '\n';

    report << "# lammps_mode eigenvalue frequency_thz\n";
    for (Eigen::Index i = 0; i < lammps_eigenvalues.size(); ++i)
        report << i << ' ' << lammps_eigenvalues(i) << ' '
               << TIHarmonicReference::signed_frequency_thz(lammps_eigenvalues(i)) << '\n';

    if (!iteration) return;
    std::ofstream csv(prefix + "_ITERATIONS.csv");
    if (!csv) throw std::runtime_error("Cannot open TI FC iteration output");
    csv << "iteration,relative_dynamical_change,maximum_frequency_change_thz,"
           "minimum_frequency_thz,force_rmse,mean_force_rms,floor_modes,coupled_blocks,"
           "sampling_attempts,rejected_samples,acceptance_rate\n";
    csv << std::setprecision(16);
    for (const auto &record : iteration->iterations)
        csv << record.iteration << ',' << record.relative_dynamical_change << ','
            << record.maximum_frequency_change_thz << ',' << record.minimum_frequency_thz << ','
            << record.force_rmse << ',' << record.mean_force_rms << ','
            << record.floor_modes << ',' << record.coupled_blocks << ','
            << record.sampling_attempts << ',' << record.rejected_samples << ','
            << record.sampling_acceptance_rate << '\n';
}

} // namespace

int ti_fit_fc(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    std::string workflow;
    std::string trajectory_file;
    std::string trajectory_format;
    std::string input_file;
    std::string input_format;
    std::string initial_fc_file;
    std::string potential_file = "potential.mtp";
    std::string potential_type = "mtp";
    std::string index_file = "index.map";
    std::string force_source = "potential";
    std::string root_name = "sqrt";
    std::string initial_stabilization = "spectral-floor";
    std::string lammps_fc_source = "fitted";
    std::string output_prefix = "TI_FC";
    std::string hessian_method = "analytic";
    int start = 1;
    int end = INT_MAX;
    int stride = 1;
    int ensemble_size = 128;
    int maximum_sampling_attempts = 0;
    int warmup_iterations = 5;
    int minimum_iterations = 3;
    int maximum_iterations = 20;
    int consecutive_converged = 2;
    int seed = 5489;
    double temperature = 300.0;
    double initial_floor_thz = 1.0;
    double pd_floor_thz = 1e-4;
    double ridge = 1e-8;
    double mixing = 0.25;
    double relative_tolerance = 1e-3;
    double frequency_tolerance_thz = 0.02;
    double soft_block_frequency_thz = 2.0;
    int soft_block_max_modes = 32;
    double degeneracy_gap_thz = 0.05;
    int degeneracy_block_max_modes = 8;
    double maximum_displacement = 1.0;
    double minimum_distance_ratio = 0.65;
    double symprec = 1e-5;
    double hessian_nudge = 0.001;
    double hessian_cutoff = 0.0;
    double lammps_fc_cutoff = 0.0;
    double lammps_fc_block_threshold = 0.0;
    double lammps_fc_minimum_frequency_thz = -1.0;
    bool write_validation_ensemble = true;
    bool lammps_fc_stabilize = true;

    cmdopts.addop(workflow, "workflow", "", "required: md, ensemble, md-ensemble, or export")
        .addop(trajectory_file, "traj", "", "MD trajectory")
        .addop(trajectory_format, "traj-format", "", "trajectory format")
        .addop(input_file, "input", "in", "optional explicit reference structure")
        .addop(input_format, "input-format", "if", "reference structure format")
        .addop(initial_fc_file, "initial-fc", "", "optional initial FORCE_CONSTANTS")
        .addop(potential_file, "potfile", "pot", "potential used to evaluate forces")
        .addop(potential_type, "potential-type", "pt", "potential type")
        .addop(index_file, "index", "id", "index map")
        .addop(force_source, "force-source", "", "potential or trajectory")
        .addop(root_name, "root-representation", "", "normal, sqrt, or root4")
        .addop(initial_stabilization, "initial-stabilization", "",
               "negative-floor, spectral-floor, or abs")
        .addop(lammps_fc_source, "lammps-fc-source", "", "fitted or initial FC for sparse LAMMPS export")
        .addop(output_prefix, "output-prefix", "", "output filename prefix")
        .addop(start, "start", "", "first trajectory frame, one-based")
        .addop(end, "end", "", "last trajectory frame, one-based and inclusive")
        .addop(stride, "stride", "", "trajectory frame stride")
        .addop(temperature, "temperature", "T", "ensemble temperature in K")
        .addop(ensemble_size, "ensemble-size", "", "samples per self-consistent iteration")
        .addop(maximum_sampling_attempts, "max-sampling-attempts", "",
               "maximum candidate samples; zero uses 20 times the requested ensemble size")
        .addop(seed, "seed", "", "base random seed")
        .addop(initial_floor_thz, "initial-floor-thz", "", "initial and iterative physical sampling floor")
        .addop(pd_floor_thz, "pd-floor-thz", "", "strict positive-definite floor")
        .addop(ridge, "ridge", "", "ridge toward the current reference")
        .addop(mixing, "mixing", "", "root-space update fraction")
        .addop(warmup_iterations, "warmup-iterations", "", "modal-only iterations")
        .addop(minimum_iterations, "min-iterations", "", "minimum self-consistent iterations")
        .addop(maximum_iterations, "max-iterations", "", "maximum self-consistent iterations")
        .addop(consecutive_converged, "consecutive-converged", "", "required converged iterations")
        .addop(relative_tolerance, "relative-tolerance", "", "relative dynamical-matrix tolerance")
        .addop(frequency_tolerance_thz, "frequency-tolerance-thz", "", "maximum frequency-change tolerance")
        .addop(soft_block_frequency_thz, "soft-block-frequency-thz", "", "soft coupled-block cutoff")
        .addop(soft_block_max_modes, "soft-block-max-modes", "", "maximum soft coupled-block size")
        .addop(degeneracy_gap_thz, "degeneracy-gap-thz", "", "near-degenerate block gap")
        .addop(degeneracy_block_max_modes, "degeneracy-block-max-modes", "", "maximum near-degenerate block size")
        .addop(maximum_displacement, "max-displacement", "", "hard ensemble displacement limit in Angstrom")
        .addop(minimum_distance_ratio, "min-distance-ratio", "", "hard pair-distance ratio")
        .addop(symprec, "symprec", "tol", "space-group tolerance")
        .addop(hessian_method, "hessian-method", "", "analytic, auto, force_fd, energy_fd, or hybrid")
        .addop(hessian_nudge, "hessian-nudge-pos", "", "finite-difference displacement")
        .addop(hessian_cutoff, "hessian-cutoff", "", "finite-difference cutoff")
        .addop(lammps_fc_cutoff, "lammps-fc-cutoff", "", "LAMMPS FC real-space cutoff in Angstrom; zero disables")
        .addop(lammps_fc_block_threshold, "lammps-fc-block-threshold", "", "LAMMPS FC off-diagonal block norm threshold")
        .addop(lammps_fc_minimum_frequency_thz, "lammps-fc-min-frequency-thz", "", "minimum sparse-reference frequency; negative uses pd-floor-thz")
        .addop(lammps_fc_stabilize, "lammps-fc-stabilize", "", "add a sparse isotropic graph stabilizer when needed")
        .addop(write_validation_ensemble, "write-validation-ensemble", "", "write final exact-force validation ensemble")
        .phrase_options(argc, argv);
    if (cmdopts.displayhelp) {
        cmdopts.showhelp(std::cout);
        return 0;
    }
    if (workflow != "md" && workflow != "ensemble" && workflow != "md-ensemble"
        && workflow != "export")
        throw std::invalid_argument("--workflow must be md, ensemble, md-ensemble, or export");
    const bool use_md = workflow == "md" || workflow == "md-ensemble";
    const bool use_iteration = workflow == "ensemble" || workflow == "md-ensemble";
    if (use_md && trajectory_file.empty())
        throw std::invalid_argument("MD workflows require --traj");
    if (!use_md && input_file.empty())
        throw std::invalid_argument("The ensemble/export workflow requires --input");
    if (force_source != "potential" && force_source != "trajectory")
        throw std::invalid_argument("--force-source must be potential or trajectory");
    if (lammps_fc_source != "fitted" && lammps_fc_source != "initial")
        throw std::invalid_argument("--lammps-fc-source must be fitted or initial");
    if (start <= 0 || end < start || stride <= 0 || temperature <= 0.0
        || ensemble_size <= 0 || maximum_sampling_attempts < 0
        || (maximum_sampling_attempts > 0 && maximum_sampling_attempts < ensemble_size)
        || maximum_displacement <= 0.0
        || minimum_distance_ratio <= 0.0 || minimum_distance_ratio >= 1.0
        || lammps_fc_cutoff < 0.0 || lammps_fc_block_threshold < 0.0)
        throw std::invalid_argument("Invalid trajectory, temperature, ensemble, or safety option");
    const auto root = parse_root(root_name);
    const auto stabilization = parse_stabilization(initial_stabilization);

    GPIO io;
    if (!io.readmap(index_file))
        throw std::runtime_error("Cannot read index map: " + index_file);
    std::vector<Atoms> trajectory;
    if (use_md) {
        std::vector<Atoms> all_frames;
        io.read(all_frames, trajectory_file, trajectory_format);
        for (int i = start - 1; i < static_cast<int>(all_frames.size()) && i < end; i += stride)
            trajectory.push_back(std::move(all_frames[i]));
        validate_fixed_cell_trajectory(trajectory);
    }

    Atoms reference;
    if (!input_file.empty()) {
        std::vector<Atoms> structures;
        io.read(structures, input_file, input_format);
        if (structures.empty()) throw std::runtime_error("No reference structure was read");
        reference = structures.front();
    } else {
        const auto provisional_masses = atomic_masses(trajectory.front(), io);
        reference = fixed_cell_centroid(trajectory, provisional_masses);
    }
    reference.wrap();
    const auto masses = atomic_masses(reference, io);
    if (use_md) {
        for (const auto &frame : trajectory) {
            if (frame.size() != reference.size() || frame.chemmap_ != reference.chemmap_)
                throw std::invalid_argument("Reference and trajectory atom ordering differ");
            if ((frame.cell_ - reference.cell_).norm()
                > 1e-7 * std::max(1.0, reference.cell_.norm()))
                throw std::invalid_argument("Reference and trajectory cells differ");
        }
    }

    const bool reference_energy_missing = !reference.has_energy() || !std::isfinite(reference.energy_);
    const bool needs_potential = (use_md && force_source == "potential") || use_iteration
        || initial_fc_file.empty() || write_validation_ensemble || reference_energy_missing;
    std::unique_ptr<Potentials::potential_wrapper> wrapper;
    Potentials::AnyPotential *potential = nullptr;
    if (needs_potential) {
        GPSettings potential_settings;
        potential_settings.Modify("typemap", index_file);
        wrapper = std::make_unique<Potentials::potential_wrapper>(
            potential_file, potential_type, potential_settings);
        potential = wrapper->pot;
        potential->omp_init();
        potential->getEFSOMP(reference);
    }
    if (!reference.has_energy() || !std::isfinite(reference.energy_))
        throw std::runtime_error("Reference structure does not have a finite potential energy");

    Eigen::MatrixXd initial_hessian;
    if (!initial_fc_file.empty()) {
        initial_hessian = HessianIO::read_force_constants(initial_fc_file, reference.size());
    } else {
        GPSettings hessian_settings;
        hessian_settings.Modify("method", hessian_method);
        hessian_settings.Modify("nudge_pos", std::to_string(hessian_nudge));
        hessian_settings.Modify("cutoff", std::to_string(hessian_cutoff));
        ComputeHessian calculator(hessian_settings);
        calculator.symprec = symprec;
        Potentials::HessianMatrix blocks;
        calculator.compute_fc_blocks(reference, blocks, potential);
        initial_hessian = HessianUtils::to_dense(blocks);
    }

    auto symmetry = TIReferenceApp::symmetry_operations(reference, symprec);
    TIHarmonicReference::Options initial_options;
    initial_options.stabilization = stabilization;
    initial_options.root_representation = root;
    initial_options.unstable_frequency_thz = initial_floor_thz;
    initial_options.positive_definite_floor_thz = pd_floor_thz;
    TIHarmonicReference::ReferenceResult current = TIHarmonicReference::build_reference(
        initial_hessian, masses, initial_options, symmetry);
    const TIHarmonicReference::Diagnostics initial_diagnostics = current.diagnostics;

    TIHarmonicReference::ForceMatchOptions fit_options;
    fit_options.root_representation = root;
    fit_options.positive_definite_floor_thz = pd_floor_thz;
    fit_options.ridge = ridge;
    fit_options.soft_block_frequency_thz = soft_block_frequency_thz;
    fit_options.soft_block_max_modes = soft_block_max_modes;
    fit_options.degeneracy_gap_thz = degeneracy_gap_thz;
    fit_options.degeneracy_block_max_modes = degeneracy_block_max_modes;
    double training_rmse = std::numeric_limits<double>::quiet_NaN();
    double validation_force_rmse = std::numeric_limits<double>::quiet_NaN();
    double final_md_force_rmse = std::numeric_limits<double>::quiet_NaN();
    double harmonic_validation_force_rmse = std::numeric_limits<double>::quiet_NaN();
    std::vector<TIHarmonicReference::ForceSample> md_samples;

    if (use_md) {
        md_samples = evaluate_frames(
            trajectory, reference, masses, potential, force_source == "trajectory");
        std::vector<TIHarmonicReference::ForceSample> training, validation;
        for (size_t i = 0; i < md_samples.size(); ++i)
            (i % 5 == 4 ? validation : training).push_back(md_samples[i]);
        if (training.empty()) training = md_samples;
        fit_options.coupled_blocks = false;
        auto modal = TIHarmonicReference::fit_force_samples(
            current, masses, fit_options, training, symmetry);
        fit_options.coupled_blocks = true;
        auto blocked = TIHarmonicReference::fit_force_samples(
            modal.reference, masses, fit_options, training, symmetry);
        training_rmse = blocked.diagnostics.force_rmse;
        validation_force_rmse = TIHarmonicReference::force_sample_rmse(blocked.reference, validation);
        auto final_fit = TIHarmonicReference::fit_force_samples(
            blocked.reference, masses, fit_options, md_samples, symmetry);
        current = std::move(final_fit.reference);
    }

    std::optional<TIHarmonicReference::SelfConsistentResult> iterative_result;
    EnsembleSafetyDiagnostics iteration_safety;
    const double reference_minimum_distance = minimum_distance(reference);
    const auto sampling_attempt_budget = [&](int requested) {
        if (maximum_sampling_attempts > 0)
            return std::max(maximum_sampling_attempts, requested);
        return requested > INT_MAX / 20 ? INT_MAX : 20 * requested;
    };
    if (use_iteration) {
        TIHarmonicReference::SelfConsistentOptions iteration_options;
        iteration_options.fit = fit_options;
        iteration_options.ensemble.temperature_kelvin = temperature;
        iteration_options.ensemble.sampling_floor_thz = initial_floor_thz;
        iteration_options.ensemble.size = ensemble_size;
        iteration_options.ensemble.seed = static_cast<std::uint32_t>(seed);
        iteration_options.ensemble.antithetic = true;
        iteration_options.warmup_iterations = warmup_iterations;
        iteration_options.minimum_iterations = minimum_iterations;
        iteration_options.maximum_iterations = maximum_iterations;
        iteration_options.consecutive_converged = consecutive_converged;
        iteration_options.mixing = mixing;
        iteration_options.relative_dynamical_tolerance = relative_tolerance;
        iteration_options.maximum_frequency_change_thz = frequency_tolerance_thz;
        iteration_options.maximum_sampling_attempts = sampling_attempt_budget(ensemble_size);
        auto filter = [&](std::span<const TIHarmonicReference::EnsembleSample> candidates) {
            return accept_ensemble_group(
                reference, candidates, maximum_displacement, minimum_distance_ratio,
                reference_minimum_distance, iteration_safety);
        };
        auto evaluator = [&](const TIHarmonicReference::ReferenceResult &,
                             const std::vector<TIHarmonicReference::EnsembleSample> &ensemble) {
            std::vector<TIHarmonicReference::ForceSample> samples;
            samples.reserve(ensemble.size());
            for (const auto &generated_sample : ensemble) {
                Atoms generated = evaluate_ensemble_structure(reference, generated_sample, potential);
                TIHarmonicReference::ForceSample sample;
                sample.displacement_cartesian = generated_sample.displacement_cartesian;
                sample.force_cartesian.resize(3 * generated.size());
                for (int i = 0; i < generated.size(); ++i)
                    sample.force_cartesian.segment<3>(3 * i) = generated.forces_[i];
                sample.energy = generated.energy_;
                samples.push_back(std::move(sample));
            }
            return samples;
        };
        try {
            iterative_result = TIHarmonicReference::iterate_reference(
                current, masses, symmetry, iteration_options, evaluator, filter);
        } catch (const std::runtime_error &error) {
            if (std::string(error.what()).starts_with(
                    "Unable to collect the requested safe ensemble"))
                throw std::runtime_error(sampling_failure_message(
                    error.what(), iteration_safety, maximum_displacement,
                    minimum_distance_ratio, reference_minimum_distance));
            throw;
        }
        current = iterative_result->reference;
    }
    if (use_md) final_md_force_rmse = TIHarmonicReference::force_sample_rmse(current, md_samples);

    TIReferenceIO::write_force_constants(output_prefix + "_FORCE_CONSTANTS", current.hessian);
    io.write_one(reference, output_prefix + "_REFERENCE.xyz", "extxyz");
    std::optional<TIReferenceIO::SparseResult> sparse_result;
    const Eigen::MatrixXd *lammps_hessian = &current.hessian;
    if (lammps_fc_cutoff > 0.0 || lammps_fc_block_threshold > 0.0) {
        TIReferenceIO::SparseOptions sparse_options;
        sparse_options.cutoff_angstrom = lammps_fc_cutoff;
        sparse_options.block_threshold = lammps_fc_block_threshold;
        sparse_options.minimum_frequency_thz = lammps_fc_minimum_frequency_thz < 0.0
            ? pd_floor_thz : lammps_fc_minimum_frequency_thz;
        sparse_options.stabilize = lammps_fc_stabilize;
        const Eigen::MatrixXd &sparse_source = lammps_fc_source == "initial"
            ? initial_hessian : current.hessian;
        sparse_result = TIReferenceIO::truncate_force_constants(
            reference, sparse_source, masses, sparse_options);
        lammps_hessian = &sparse_result->hessian;
    }
    const Eigen::VectorXd lammps_eigenvalues = TIReferenceIO::internal_mode_eigenvalues(
        *lammps_hessian, masses);
    if (lammps_eigenvalues.size() > 0 && lammps_eigenvalues.minCoeff() <= 0.0)
        throw std::runtime_error("LAMMPS harmonic reference is not positive definite");
    const Atoms lammps_reference = TIReferenceIO::write_lammps_reference(
        output_prefix + "_LAMMPS_REFERENCE", reference, *lammps_hessian, reference.energy_);
    Atoms lammps_output = lammps_reference;
    io.write_mass = true;
    io.write_one(lammps_output, output_prefix + "_REFERENCE.lmp", "lmp");
    std::optional<TIHarmonicReference::EnsembleSamplingResult> validation_sampling;
    EnsembleSafetyDiagnostics validation_safety;
    if (write_validation_ensemble) {
        TIHarmonicReference::EnsembleOptions validation_options;
        validation_options.temperature_kelvin = temperature;
        validation_options.size = 64;
        validation_options.seed = static_cast<std::uint32_t>(seed + 1000003);
        validation_options.antithetic = true;
        auto filter = [&](std::span<const TIHarmonicReference::EnsembleSample> candidates) {
            return accept_ensemble_group(
                reference, candidates, maximum_displacement, minimum_distance_ratio,
                reference_minimum_distance, validation_safety);
        };
        validation_sampling = TIHarmonicReference::sample_classical_ensemble_filtered(
            current, validation_options, sampling_attempt_budget(validation_options.size), filter);
        if (validation_sampling->samples.size() != static_cast<size_t>(validation_options.size))
            throw std::runtime_error(sampling_failure_message(
                fmt::format("Unable to collect the requested validation ensemble: accepted {} of {} "
                            "samples after {} attempts",
                            validation_sampling->samples.size(), validation_options.size,
                            validation_sampling->attempted_samples),
                validation_safety, maximum_displacement,
                minimum_distance_ratio, reference_minimum_distance));
        std::vector<Atoms> validation_structures;
        std::vector<TIHarmonicReference::ForceSample> validation_force_samples;
        validation_structures.reserve(validation_sampling->samples.size());
        validation_force_samples.reserve(validation_sampling->samples.size());
        for (const auto &sample : validation_sampling->samples) {
            Atoms generated = evaluate_ensemble_structure(reference, sample, potential);
            TIHarmonicReference::ForceSample force_sample;
            force_sample.displacement_cartesian = sample.displacement_cartesian;
            force_sample.force_cartesian.resize(3 * generated.size());
            for (int i = 0; i < generated.size(); ++i)
                force_sample.force_cartesian.segment<3>(3 * i) = generated.forces_[i];
            force_sample.energy = generated.energy_;
            validation_force_samples.push_back(std::move(force_sample));
            validation_structures.push_back(std::move(generated));
        }
        harmonic_validation_force_rmse = TIHarmonicReference::force_sample_rmse(
            current, validation_force_samples);
        auto output = open_ofile(output_prefix + "_VALIDATION.xyz");
        if (!output || !io.seq_write_init(*output, "extxyz", output_prefix + "_VALIDATION.xyz"))
            throw std::runtime_error("Cannot open validation ensemble output");
        for (size_t isample = 0; isample < validation_structures.size(); ++isample) {
            Atoms &generated = validation_structures[isample];
            generated.global_attribs_i_["ti_validation_sample"] = static_cast<int>(isample);
            generated.global_attribs_d_["ti_temperature_K"] = temperature;
            io.seq_write(*output, generated);
        }
        close_ofile(output);
    }

    write_iteration_report(output_prefix, workflow, initial_stabilization,
        initial_diagnostics, temperature, current,
        iterative_result ? &*iterative_result : nullptr, training_rmse,
        validation_force_rmse, final_md_force_rmse,
        harmonic_validation_force_rmse, trajectory.size(),
        lammps_fc_source,
        sparse_result ? &sparse_result->diagnostics : nullptr,
        lammps_eigenvalues, HessianUtils::acoustic_sum_rule_error(*lammps_hessian),
        use_iteration ? &iteration_safety : nullptr,
        validation_sampling ? &*validation_sampling : nullptr,
        validation_sampling ? &validation_safety : nullptr);

    fmt::print("ti_fit_fc: {} internal modes, minimum frequency {:.8g} THz, ASR {:.3e}\n",
               current.internal_dynamical_matrix.rows(),
               TIHarmonicReference::signed_frequency_thz(
                   current.diagnostics.eigenvalues_after.minCoeff()),
               current.diagnostics.acoustic_sum_rule_error_after);
    if (iterative_result)
        fmt::print("self-consistent iterations: {}, converged: {}\n",
                   iterative_result->iterations.size(), iterative_result->converged ? "yes" : "no");
    if (sparse_result) {
        const auto &diagnostics = sparse_result->diagnostics;
        fmt::print("LAMMPS sparse FC ({}): {}/{} off-diagonal blocks ({:.2f}%), "
                   "relative change {:.5g}, minimum frequency {:.8g} THz, ASR {:.3e}\n",
                   lammps_fc_source, diagnostics.retained_offdiagonal_blocks,
                   diagnostics.total_offdiagonal_blocks,
                   100.0 * diagnostics.retained_fraction,
                   diagnostics.relative_hessian_change,
                   diagnostics.minimum_frequency_after_thz,
                   diagnostics.acoustic_sum_rule_error);
    }
    return iterative_result && !iterative_result->converged ? 2 : 0;
}
