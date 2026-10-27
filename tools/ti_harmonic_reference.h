#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace TIHarmonicReference {

constexpr double frequency_conversion_thz = 15.633304237154925;

enum class StabilizationPolicy {
    NegativeFloor,
    SpectralFloor,
    AbsoluteValue,
};

enum class RootRepresentation {
    Normal,
    SquareRoot,
    FourthRoot,
};

enum class FitMethod {
    None,
    ModalForce,
    FullForce,
};

struct SymmetryOperation {
    std::vector<int> atom_mapping;
    Eigen::Matrix3d cartesian_rotation = Eigen::Matrix3d::Identity();
};

struct TrajectorySample {
    Eigen::VectorXd displacement_cartesian;
    Eigen::VectorXd delta_force_cartesian;
    double delta_energy = std::numeric_limits<double>::quiet_NaN();
};

struct Options {
    StabilizationPolicy stabilization = StabilizationPolicy::NegativeFloor;
    RootRepresentation root_representation = RootRepresentation::SquareRoot;
    FitMethod fit_method = FitMethod::None;
    double unstable_frequency_thz = 1.0;
    double positive_definite_floor_thz = 1e-4;
    double ridge = 1e-8;
    int max_iterations = 200;
    double gradient_tolerance = 1e-8;
};

struct Diagnostics {
    Eigen::VectorXd eigenvalues_before;
    Eigen::VectorXd eigenvalues_after;
    int corrected_modes = 0;
    int optimizer_iterations = 0;
    bool optimizer_converged = true;
    double transpose_error_before = 0.0;
    double acoustic_sum_rule_error_before = 0.0;
    double acoustic_sum_rule_error_after = 0.0;
    double space_group_error_before = 0.0;
    double space_group_error_after = 0.0;
    double relative_hessian_change = 0.0;
    double force_rmse = std::numeric_limits<double>::quiet_NaN();
    double delta_energy_mean = std::numeric_limits<double>::quiet_NaN();
    double delta_energy_stddev = std::numeric_limits<double>::quiet_NaN();
    double log_determinant_internal = std::numeric_limits<double>::quiet_NaN();
    double condition_number_internal = std::numeric_limits<double>::infinity();
};

struct ReferenceResult {
    Eigen::MatrixXd hessian;
    Eigen::MatrixXd dynamical_matrix;
    Eigen::MatrixXd internal_dynamical_matrix;
    Eigen::MatrixXd internal_basis;
    Eigen::MatrixXd root_matrix;
    Eigen::VectorXd sqrt_mass_coordinates;
    Diagnostics diagnostics;
};

struct EnsembleOptions {
    double temperature_kelvin = 300.0;
    double sampling_floor_thz = 0.0;
    int size = 0;
    std::uint32_t seed = 5489u;
    bool antithetic = false;
};

struct EnsembleSample {
    Eigen::VectorXd displacement_cartesian;
    Eigen::VectorXd reference_force_cartesian;
    double harmonic_energy = 0.0;
};

struct EnsembleSamplingResult {
    std::vector<EnsembleSample> samples;
    int attempted_samples = 0;
    int rejected_samples = 0;
};

using EnsembleSampleFilter = std::function<bool(std::span<const EnsembleSample>)>;

struct ForceSample {
    Eigen::VectorXd displacement_cartesian;
    Eigen::VectorXd force_cartesian;
    double energy = std::numeric_limits<double>::quiet_NaN();
    double weight = 1.0;
};

struct ForceMatchOptions {
    RootRepresentation root_representation = RootRepresentation::SquareRoot;
    double positive_definite_floor_thz = 1e-4;
    double ridge = 1e-8; // Penalty relative to the weighted mean squared force residual.
    int max_iterations = 200;
    double gradient_tolerance = 1e-8;
    bool coupled_blocks = false;
    double soft_block_frequency_thz = 2.0;
    int soft_block_max_modes = 32;
    double degeneracy_gap_thz = 0.05;
    int degeneracy_block_max_modes = 8;
};

struct ForceMatchDiagnostics {
    double force_rmse = std::numeric_limits<double>::quiet_NaN();
    double mean_force_rms = std::numeric_limits<double>::quiet_NaN();
    double relative_dynamical_change = std::numeric_limits<double>::quiet_NaN();
    int optimizer_iterations = 0;
    int coupled_blocks = 0;
    int floor_modes = 0;
};

struct ForceMatchResult {
    ReferenceResult reference;
    ForceMatchDiagnostics diagnostics;
};

struct SelfConsistentOptions {
    ForceMatchOptions fit;
    EnsembleOptions ensemble;
    int warmup_iterations = 5;
    int minimum_iterations = 3;
    int maximum_iterations = 20;
    int consecutive_converged = 2;
    double mixing = 0.25;
    double relative_dynamical_tolerance = 1e-3;
    double maximum_frequency_change_thz = 0.02;
    int maximum_sampling_attempts = 0;
};

struct IterationRecord {
    int iteration = 0;
    double relative_dynamical_change = 0.0;
    double maximum_frequency_change_thz = 0.0;
    double minimum_frequency_thz = 0.0;
    double force_rmse = std::numeric_limits<double>::quiet_NaN();
    double mean_force_rms = std::numeric_limits<double>::quiet_NaN();
    int floor_modes = 0;
    int coupled_blocks = 0;
    int sampling_attempts = 0;
    int rejected_samples = 0;
    double sampling_acceptance_rate = 1.0;
};

struct SelfConsistentResult {
    ReferenceResult reference;
    std::vector<IterationRecord> iterations;
    bool converged = false;
};

using EnsembleForceEvaluator = std::function<std::vector<ForceSample>(
    const ReferenceResult &, const std::vector<EnsembleSample> &)>;

ReferenceResult build_reference(
    const Eigen::MatrixXd &hessian,
    const std::vector<double> &masses,
    const Options &options,
    const std::vector<SymmetryOperation> &symmetry_operations = {},
    const std::vector<TrajectorySample> &samples = {});

std::vector<EnsembleSample> sample_classical_ensemble(
    const ReferenceResult &reference,
    const EnsembleOptions &options);

EnsembleSamplingResult sample_classical_ensemble_filtered(
    const ReferenceResult &reference,
    const EnsembleOptions &options,
    int maximum_attempts,
    const EnsembleSampleFilter &filter);

ForceMatchResult fit_force_samples(
    const ReferenceResult &initial,
    const std::vector<double> &masses,
    const ForceMatchOptions &options,
    const std::vector<ForceSample> &samples,
    const std::vector<SymmetryOperation> &symmetry_operations = {});

// Weighted RMSE of the actual harmonic forces in the mass-weighted internal
// coordinates. No sample centering or fitted force intercept is applied.
double force_sample_rmse(
    const ReferenceResult &reference,
    const std::vector<ForceSample> &samples);

SelfConsistentResult iterate_reference(
    const ReferenceResult &initial,
    const std::vector<double> &masses,
    const std::vector<SymmetryOperation> &symmetry_operations,
    const SelfConsistentOptions &options,
    const EnsembleForceEvaluator &evaluator,
    const EnsembleSampleFilter &filter = {});

Eigen::MatrixXd to_root(const Eigen::MatrixXd &matrix,
                        RootRepresentation representation,
                        double eigenvalue_offset = 0.0);
Eigen::MatrixXd from_root(const Eigen::MatrixXd &root,
                          RootRepresentation representation,
                          double eigenvalue_offset = 0.0);
Eigen::MatrixXd pullback_gradient(const Eigen::MatrixXd &root,
                                  const Eigen::MatrixXd &matrix_gradient,
                                  RootRepresentation representation);

double signed_frequency_thz(double eigenvalue);
std::string policy_name(StabilizationPolicy policy);
std::string root_name(RootRepresentation representation);

} // namespace TIHarmonicReference
