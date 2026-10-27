#include "tools/ti_harmonic_reference.h"

#include "deplibs/LBFGSpp/include/LBFGS.h"
#include "tools/constants.h"
#include "tools/hessian_utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace TIHarmonicReference {
namespace {

using MatrixList = std::vector<Eigen::MatrixXd>;

void validate_square(const Eigen::MatrixXd &matrix, const char *name)
{
    if (matrix.rows() == 0 || matrix.rows() != matrix.cols())
        throw std::invalid_argument(std::string(name) + " must be a non-empty square matrix");
    if (!matrix.allFinite())
        throw std::invalid_argument(std::string(name) + " contains non-finite values");
}

MatrixList internal_symmetry_operators(const std::vector<SymmetryOperation> &operations,
                                       const Eigen::MatrixXd &basis,
                                       Eigen::Index natoms)
{
    MatrixList result;
    for (const auto &operation : operations) {
        if (operation.atom_mapping.size() != static_cast<size_t>(natoms))
            throw std::invalid_argument("Symmetry atom mapping has the wrong size");
        if (!operation.cartesian_rotation.allFinite())
            throw std::invalid_argument("Symmetry rotation contains non-finite values");
        Eigen::MatrixXd full = Eigen::MatrixXd::Zero(3 * natoms, 3 * natoms);
        std::vector<bool> seen(natoms, false);
        for (Eigen::Index atom = 0; atom < natoms; ++atom) {
            const int mapped = operation.atom_mapping[atom];
            if (mapped < 0 || mapped >= natoms || seen[mapped])
                throw std::invalid_argument("Symmetry atom mapping is not a permutation");
            seen[mapped] = true;
            full.block<3, 3>(3 * mapped, 3 * atom) = operation.cartesian_rotation.transpose();
        }
        Eigen::MatrixXd reduced = basis.transpose() * full * basis;
        const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(reduced.rows(), reduced.cols());
        if ((reduced.transpose() * reduced - identity).norm() > 1e-7 * std::max(1.0, identity.norm()))
            throw std::invalid_argument("Symmetry operation does not preserve the internal subspace");
        result.push_back(std::move(reduced));
    }
    return result;
}

Eigen::MatrixXd transform_full_symmetry(const Eigen::MatrixXd &matrix,
                                        const SymmetryOperation &operation)
{
    const Eigen::Index natoms = matrix.rows() / 3;
    Eigen::MatrixXd transformed = Eigen::MatrixXd::Zero(matrix.rows(), matrix.cols());
    const Eigen::Matrix3d rotation_transpose = operation.cartesian_rotation.transpose();
    for (Eigen::Index i = 0; i < natoms; ++i) {
        const int mapped_i = operation.atom_mapping[i];
        for (Eigen::Index j = 0; j < natoms; ++j) {
            const int mapped_j = operation.atom_mapping[j];
            transformed.block<3, 3>(3 * mapped_i, 3 * mapped_j).noalias()
                = rotation_transpose * matrix.block<3, 3>(3 * i, 3 * j)
                * operation.cartesian_rotation;
        }
    }
    return transformed;
}

Eigen::MatrixXd project_full_symmetry(const Eigen::MatrixXd &matrix,
                                      const std::vector<SymmetryOperation> &operations)
{
    Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
    if (operations.empty()) return symmetric;
    Eigen::MatrixXd projected = Eigen::MatrixXd::Zero(matrix.rows(), matrix.cols());
    for (const auto &operation : operations)
        projected += transform_full_symmetry(symmetric, operation);
    projected /= static_cast<double>(operations.size());
    return 0.5 * (projected + projected.transpose());
}

double full_symmetry_error(const Eigen::MatrixXd &matrix,
                           const std::vector<SymmetryOperation> &operations)
{
    if (operations.empty()) return 0.0;
    const double scale = std::max(1.0, matrix.norm());
    double error = 0.0;
    for (const auto &operation : operations)
        error = std::max(error, (transform_full_symmetry(matrix, operation) - matrix).norm() / scale);
    return error;
}

Eigen::MatrixXd project_symmetry(const Eigen::MatrixXd &matrix, const MatrixList &operations)
{
    Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
    if (operations.empty()) return symmetric;
    Eigen::MatrixXd projected = Eigen::MatrixXd::Zero(matrix.rows(), matrix.cols());
    for (const auto &operation : operations)
        projected.noalias() += operation * symmetric * operation.transpose();
    projected /= static_cast<double>(operations.size());
    return 0.5 * (projected + projected.transpose());
}

Eigen::VectorXd pack_symmetric(const Eigen::MatrixXd &matrix)
{
    Eigen::VectorXd packed(matrix.rows() * (matrix.rows() + 1) / 2);
    Eigen::Index position = 0;
    for (Eigen::Index i = 0; i < matrix.rows(); ++i)
        for (Eigen::Index j = 0; j <= i; ++j)
            packed(position++) = matrix(i, j);
    return packed;
}

Eigen::MatrixXd unpack_symmetric(const Eigen::VectorXd &packed, Eigen::Index dim)
{
    if (packed.size() != dim * (dim + 1) / 2)
        throw std::invalid_argument("Packed symmetric matrix has the wrong size");
    Eigen::MatrixXd matrix(dim, dim);
    Eigen::Index position = 0;
    for (Eigen::Index i = 0; i < dim; ++i) {
        for (Eigen::Index j = 0; j <= i; ++j) {
            matrix(i, j) = packed(position++);
            matrix(j, i) = matrix(i, j);
        }
    }
    return matrix;
}

Eigen::VectorXd pack_symmetric_gradient(const Eigen::MatrixXd &gradient)
{
    Eigen::VectorXd packed(gradient.rows() * (gradient.rows() + 1) / 2);
    Eigen::Index position = 0;
    for (Eigen::Index i = 0; i < gradient.rows(); ++i)
        for (Eigen::Index j = 0; j <= i; ++j)
            packed(position++) = i == j ? gradient(i, j) : gradient(i, j) + gradient(j, i);
    return packed;
}

struct RootFitObjective {
    Eigen::MatrixXd x;
    Eigen::MatrixXd force;
    Eigen::MatrixXd initial;
    MatrixList symmetry;
    RootRepresentation representation;
    double offset;
    double ridge;
    double sample_weight_sum = 0.0;

    double operator()(const Eigen::VectorXd &packed, Eigen::VectorXd &gradient)
    {
        Eigen::MatrixXd root = project_symmetry(unpack_symmetric(packed, initial.rows()), symmetry);
        const Eigen::MatrixXd dynamical = from_root(root, representation, offset);
        const Eigen::MatrixXd residual = force + dynamical * x;
        const double count = sample_weight_sum > 0.0
            ? sample_weight_sum : static_cast<double>(x.cols());
        const Eigen::MatrixXd delta = dynamical - initial;
        const double objective = residual.squaredNorm() / count + ridge * delta.squaredNorm();
        Eigen::MatrixXd grad_d = 2.0 * residual * x.transpose() / count + 2.0 * ridge * delta;
        grad_d = project_symmetry(grad_d, symmetry);
        Eigen::MatrixXd grad_root = pullback_gradient(root, grad_d, representation);
        grad_root = project_symmetry(grad_root, symmetry);
        gradient = pack_symmetric_gradient(grad_root);
        return objective;
    }
};

void evaluate_samples(const Eigen::MatrixXd &dynamical,
                      const Eigen::MatrixXd &basis,
                      const Eigen::VectorXd &roots,
                      const std::vector<TrajectorySample> &samples,
                      Diagnostics &diagnostics)
{
    if (samples.empty()) return;
    double force_squared = 0.0;
    std::vector<double> energy_residuals;
    for (const auto &sample : samples) {
        const Eigen::VectorXd x = basis.transpose()
            * (roots.array() * sample.displacement_cartesian.array()).matrix();
        const Eigen::VectorXd f = basis.transpose()
            * (sample.delta_force_cartesian.array() / roots.array()).matrix();
        force_squared += (f + dynamical * x).squaredNorm();
        if (std::isfinite(sample.delta_energy))
            energy_residuals.push_back(sample.delta_energy - 0.5 * x.dot(dynamical * x));
    }
    diagnostics.force_rmse = std::sqrt(force_squared
        / (static_cast<double>(samples.size()) * static_cast<double>(dynamical.rows())));
    if (!energy_residuals.empty()) {
        diagnostics.delta_energy_mean = std::accumulate(
            energy_residuals.begin(), energy_residuals.end(), 0.0) / energy_residuals.size();
        double variance = 0.0;
        for (double value : energy_residuals)
            variance += std::pow(value - diagnostics.delta_energy_mean, 2);
        diagnostics.delta_energy_stddev = std::sqrt(variance / energy_residuals.size());
    }
}

} // namespace

Eigen::MatrixXd to_root(const Eigen::MatrixXd &matrix,
                        RootRepresentation representation,
                        double eigenvalue_offset)
{
    validate_square(matrix, "Matrix");
    if (!std::isfinite(eigenvalue_offset) || eigenvalue_offset < 0.0)
        throw std::invalid_argument("Root eigenvalue offset must be finite and non-negative");
    if (representation == RootRepresentation::Normal) return 0.5 * (matrix + matrix.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(0.5 * (matrix + matrix.transpose()));
    if (solver.info() != Eigen::Success) throw std::runtime_error("Failed to diagonalize matrix for root conversion");
    Eigen::VectorXd values = solver.eigenvalues().array() - eigenvalue_offset;
    const double tolerance = 1e-12 * std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
    if (values.minCoeff() < -tolerance)
        throw std::invalid_argument("Matrix is below the requested root eigenvalue offset");
    values = values.cwiseMax(0.0);
    const double power = representation == RootRepresentation::SquareRoot ? 0.5 : 0.25;
    values = values.array().pow(power);
    return solver.eigenvectors() * values.asDiagonal() * solver.eigenvectors().transpose();
}

Eigen::MatrixXd from_root(const Eigen::MatrixXd &root,
                          RootRepresentation representation,
                          double eigenvalue_offset)
{
    validate_square(root, "Root matrix");
    if (!std::isfinite(eigenvalue_offset) || eigenvalue_offset < 0.0)
        throw std::invalid_argument("Root eigenvalue offset must be finite and non-negative");
    Eigen::MatrixXd result = 0.5 * (root + root.transpose());
    if (representation == RootRepresentation::SquareRoot) result = result * result;
    else if (representation == RootRepresentation::FourthRoot) {
        result = result * result;
        result = result * result;
    }
    if (representation != RootRepresentation::Normal)
        result.diagonal().array() += eigenvalue_offset;
    return 0.5 * (result + result.transpose());
}

Eigen::MatrixXd pullback_gradient(const Eigen::MatrixXd &root,
                                  const Eigen::MatrixXd &matrix_gradient,
                                  RootRepresentation representation)
{
    validate_square(root, "Root matrix");
    if (matrix_gradient.rows() != root.rows() || matrix_gradient.cols() != root.cols())
        throw std::invalid_argument("Root and gradient dimensions differ");
    const Eigen::MatrixXd gradient = 0.5 * (matrix_gradient + matrix_gradient.transpose());
    if (representation == RootRepresentation::Normal) return gradient;
    if (representation == RootRepresentation::SquareRoot)
        return root * gradient + gradient * root;
    const Eigen::MatrixXd r2 = root * root;
    const Eigen::MatrixXd r3 = r2 * root;
    return gradient * r3 + root * gradient * r2 + r2 * gradient * root + r3 * gradient;
}

ReferenceResult build_reference(const Eigen::MatrixXd &hessian,
                                const std::vector<double> &masses,
                                const Options &options,
                                const std::vector<SymmetryOperation> &symmetry_operations,
                                const std::vector<TrajectorySample> &samples)
{
    validate_square(hessian, "Hessian");
    if (hessian.rows() % 3 != 0 || hessian.rows() / 3 != static_cast<Eigen::Index>(masses.size()))
        throw std::invalid_argument("Hessian and mass dimensions differ");
    if (!std::isfinite(options.unstable_frequency_thz) || options.unstable_frequency_thz <= 0.0
        || !std::isfinite(options.positive_definite_floor_thz) || options.positive_definite_floor_thz < 0.0
        || !std::isfinite(options.ridge) || options.ridge < 0.0
        || options.max_iterations <= 0 || !std::isfinite(options.gradient_tolerance)
        || options.gradient_tolerance <= 0.0)
        throw std::invalid_argument("Invalid TI harmonic reference options");

    ReferenceResult result;
    result.sqrt_mass_coordinates = HessianUtils::sqrt_mass_coordinates(masses);
    result.internal_basis = HessianUtils::mass_weighted_internal_basis(masses);
    result.diagnostics.transpose_error_before = (hessian - hessian.transpose()).norm()
        / std::max(1.0, hessian.norm());
    result.diagnostics.acoustic_sum_rule_error_before = HessianUtils::acoustic_sum_rule_error(hessian);
    const Eigen::MatrixXd input_symmetric = 0.5 * (hessian + hessian.transpose());
    const Eigen::MatrixXd input_dynamical = HessianUtils::to_dynamical_matrix(
        input_symmetric, masses);
    if (result.internal_basis.cols() == 0) {
        result.hessian = Eigen::MatrixXd::Zero(3, 3);
        result.dynamical_matrix = Eigen::MatrixXd::Zero(3, 3);
        result.internal_dynamical_matrix.resize(0, 0);
        result.root_matrix.resize(0, 0);
        result.diagnostics.acoustic_sum_rule_error_after = 0.0;
        return result;
    }

    for (const auto &operation : symmetry_operations) {
        if (operation.atom_mapping.size() != masses.size())
            throw std::invalid_argument("Symmetry atom mapping has the wrong size");
        if (!operation.cartesian_rotation.allFinite())
            throw std::invalid_argument("Symmetry rotation contains non-finite values");
        if ((operation.cartesian_rotation.transpose() * operation.cartesian_rotation
             - Eigen::Matrix3d::Identity()).norm() > 1e-8)
            throw std::invalid_argument("Symmetry Cartesian rotation is not orthogonal");
        std::vector<bool> seen(masses.size(), false);
        for (size_t atom = 0; atom < masses.size(); ++atom) {
            const int mapped = operation.atom_mapping[atom];
            if (mapped < 0 || mapped >= static_cast<int>(masses.size()) || seen[mapped])
                throw std::invalid_argument("Symmetry atom mapping is not a permutation");
            if (std::abs(masses[atom] - masses[mapped]) > 1e-12 * std::max(masses[atom], masses[mapped]))
                throw std::invalid_argument("Symmetry operation maps atoms with different masses");
            seen[mapped] = true;
        }
    }
    result.diagnostics.space_group_error_before = full_symmetry_error(
        input_dynamical, symmetry_operations);
    const Eigen::MatrixXd projected_input = project_full_symmetry(
        input_dynamical, symmetry_operations);
    Eigen::MatrixXd internal = result.internal_basis.transpose()
        * projected_input * result.internal_basis;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(internal);
    if (solver.info() != Eigen::Success) throw std::runtime_error("Failed to diagonalize internal dynamical matrix");
    result.diagnostics.eigenvalues_before = solver.eigenvalues();
    Eigen::VectorXd values = solver.eigenvalues();
    const double unstable_floor = std::pow(options.unstable_frequency_thz / frequency_conversion_thz, 2);
    const double positive_definite_floor = std::pow(
        options.positive_definite_floor_thz / frequency_conversion_thz, 2);
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        const double value = values(i);
        bool correct = value <= 0.0;
        if (options.stabilization == StabilizationPolicy::SpectralFloor)
            correct = value < unstable_floor;
        double stabilized = value;
        if (correct)
            stabilized = options.stabilization == StabilizationPolicy::AbsoluteValue
                ? std::max(std::abs(value), unstable_floor) : unstable_floor;
        stabilized = std::max(stabilized, positive_definite_floor);
        if (stabilized != value) {
            values(i) = stabilized;
            ++result.diagnostics.corrected_modes;
        }
    }
    internal = solver.eigenvectors() * values.asDiagonal() * solver.eigenvectors().transpose();
    const bool fit_full_force = !samples.empty() && options.fit_method == FitMethod::FullForce;
    MatrixList symmetry;
    if (fit_full_force)
        symmetry = internal_symmetry_operators(symmetry_operations, result.internal_basis, masses.size());
    else if (!symmetry_operations.empty()) {
        Eigen::MatrixXd full_internal = result.internal_basis * internal * result.internal_basis.transpose();
        full_internal = project_full_symmetry(full_internal, symmetry_operations);
        internal = result.internal_basis.transpose() * full_internal * result.internal_basis;
        internal = 0.5 * (internal + internal.transpose());
    }

    if (fit_full_force) {
        const Eigen::Index dim = internal.rows();
        Eigen::MatrixXd x(dim, samples.size()), force(dim, samples.size());
        for (Eigen::Index isample = 0; isample < static_cast<Eigen::Index>(samples.size()); ++isample) {
            if (samples[isample].displacement_cartesian.size() != hessian.rows()
                || samples[isample].delta_force_cartesian.size() != hessian.rows()
                || !samples[isample].displacement_cartesian.allFinite()
                || !samples[isample].delta_force_cartesian.allFinite())
                throw std::invalid_argument("Trajectory sample has invalid dimensions or values");
            x.col(isample) = result.internal_basis.transpose()
                * (result.sqrt_mass_coordinates.array()
                   * samples[isample].displacement_cartesian.array()).matrix();
            force.col(isample) = result.internal_basis.transpose()
                * (samples[isample].delta_force_cartesian.array()
                   / result.sqrt_mass_coordinates.array()).matrix();
        }
        const double offset = std::pow(
            options.positive_definite_floor_thz / frequency_conversion_thz, 2);
        RootFitObjective objective{x, force, internal, symmetry,
                                   options.root_representation, offset, options.ridge};
        Eigen::MatrixXd initial_root = to_root(internal, options.root_representation,
                                               options.root_representation == RootRepresentation::Normal ? 0.0 : offset);
        Eigen::VectorXd packed = pack_symmetric(initial_root);
        LBFGSpp::LBFGSParam<double> parameters;
        parameters.max_iterations = options.max_iterations;
        parameters.epsilon = options.gradient_tolerance;
        parameters.epsilon_rel = options.gradient_tolerance;
        parameters.epsgmax = options.gradient_tolerance;
        LBFGSpp::LBFGSSolver<double> optimizer(parameters);
        double objective_value = 0.0;
        try {
            result.diagnostics.optimizer_iterations = optimizer.minimize(objective, packed, objective_value);
        } catch (const std::exception &error) {
            throw std::runtime_error(std::string("Root force matching failed: ") + error.what());
        }
        Eigen::MatrixXd fitted_root = project_symmetry(unpack_symmetric(packed, dim), symmetry);
        internal = from_root(fitted_root, options.root_representation,
                             options.root_representation == RootRepresentation::Normal ? 0.0 : offset);
        if (options.root_representation == RootRepresentation::Normal) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> fitted_solver(internal);
            Eigen::VectorXd fitted_values = fitted_solver.eigenvalues().cwiseMax(offset);
            internal = fitted_solver.eigenvectors() * fitted_values.asDiagonal()
                     * fitted_solver.eigenvectors().transpose();
        }
        internal = project_symmetry(internal, symmetry);
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> final_solver(internal);
    if (final_solver.info() != Eigen::Success) throw std::runtime_error("Failed to diagonalize final internal matrix");
    result.diagnostics.eigenvalues_after = final_solver.eigenvalues();
    if (result.diagnostics.eigenvalues_after.minCoeff() <= 0.0)
        throw std::runtime_error("TI harmonic reference is not positive definite in the internal subspace");
    result.internal_dynamical_matrix = internal;
    result.root_matrix = to_root(internal, options.root_representation,
        options.root_representation == RootRepresentation::Normal ? 0.0
        : std::pow(options.positive_definite_floor_thz / frequency_conversion_thz, 2));
    result.dynamical_matrix = result.internal_basis * internal * result.internal_basis.transpose();
    result.dynamical_matrix = 0.5 * (result.dynamical_matrix + result.dynamical_matrix.transpose());
    result.hessian = HessianUtils::from_dynamical_matrix(result.dynamical_matrix, masses);
    result.diagnostics.acoustic_sum_rule_error_after = HessianUtils::acoustic_sum_rule_error(result.hessian);
    result.diagnostics.space_group_error_after = full_symmetry_error(
        result.dynamical_matrix, symmetry_operations);
    result.diagnostics.relative_hessian_change = (result.hessian - input_symmetric).norm()
        / std::max(1.0, input_symmetric.norm());
    result.diagnostics.log_determinant_internal = result.diagnostics.eigenvalues_after.array().log().sum();
    result.diagnostics.condition_number_internal = result.diagnostics.eigenvalues_after.maxCoeff()
        / result.diagnostics.eigenvalues_after.minCoeff();
    evaluate_samples(internal, result.internal_basis, result.sqrt_mass_coordinates,
                     samples, result.diagnostics);
    return result;
}

std::vector<EnsembleSample> sample_classical_ensemble(const ReferenceResult &reference,
                                                       const EnsembleOptions &options)
{
    return sample_classical_ensemble_filtered(reference, options, options.size, {}).samples;
}

EnsembleSamplingResult sample_classical_ensemble_filtered(
    const ReferenceResult &reference,
    const EnsembleOptions &options,
    int maximum_attempts,
    const EnsembleSampleFilter &filter)
{
    if (!std::isfinite(options.temperature_kelvin) || options.temperature_kelvin <= 0.0)
        throw std::invalid_argument("Ensemble temperature must be finite and positive");
    if (!std::isfinite(options.sampling_floor_thz) || options.sampling_floor_thz < 0.0
        || options.size < 0 || maximum_attempts < options.size)
        throw std::invalid_argument(
            "Ensemble size and sampling floor must be non-negative, and the attempt budget "
            "must cover the requested ensemble");
    EnsembleSamplingResult result;
    if (options.size == 0) return result;
    validate_square(reference.hessian, "Reference Hessian");
    const bool has_internal_modes = reference.internal_dynamical_matrix.rows() != 0;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
    Eigen::VectorXd sigma;
    if (has_internal_modes) {
        solver.compute(reference.internal_dynamical_matrix);
        if (solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() <= 0.0)
            throw std::runtime_error("Cannot sample a non-positive internal dynamical matrix");
        const double sampling_floor = std::pow(
            options.sampling_floor_thz / frequency_conversion_thz, 2);
        const Eigen::VectorXd sampling_values = solver.eigenvalues().cwiseMax(sampling_floor);
        sigma = (Boltzmann * options.temperature_kelvin / sampling_values.array()).sqrt();
    }
    std::mt19937 generator(options.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    result.samples.reserve(options.size);
    while (static_cast<int>(result.samples.size()) < options.size) {
        const int copies = options.antithetic
                && static_cast<int>(result.samples.size()) + 1 < options.size ? 2 : 1;
        if (result.attempted_samples + copies > maximum_attempts) break;
        std::vector<EnsembleSample> candidates;
        candidates.reserve(copies);
        Eigen::VectorXd gaussian(sigma.size());
        for (Eigen::Index i = 0; i < gaussian.size(); ++i) gaussian(i) = normal(generator);
        for (int copy = 0; copy < copies; ++copy) {
            const double sign = copy == 0 ? 1.0 : -1.0;
            EnsembleSample sample;
            if (has_internal_modes) {
                const Eigen::VectorXd q = sign * solver.eigenvectors()
                    * (sigma.array() * gaussian.array()).matrix();
                const Eigen::VectorXd mass_weighted = reference.internal_basis * q;
                sample.displacement_cartesian = (mass_weighted.array()
                    / reference.sqrt_mass_coordinates.array()).matrix();
            } else {
                sample.displacement_cartesian = Eigen::VectorXd::Zero(reference.hessian.rows());
            }
            sample.reference_force_cartesian = -reference.hessian * sample.displacement_cartesian;
            sample.harmonic_energy = 0.5 * sample.displacement_cartesian.dot(
                reference.hessian * sample.displacement_cartesian);
            candidates.push_back(std::move(sample));
        }
        result.attempted_samples += copies;
        if (filter && !filter(candidates)) {
            result.rejected_samples += copies;
            continue;
        }
        for (auto &candidate : candidates) result.samples.push_back(std::move(candidate));
    }
    return result;
}

ForceMatchResult fit_force_samples(
    const ReferenceResult &initial,
    const std::vector<double> &masses,
    const ForceMatchOptions &options,
    const std::vector<ForceSample> &samples,
    const std::vector<SymmetryOperation> &symmetry_operations)
{
    if (samples.empty()) throw std::invalid_argument("Force matching requires at least one sample");
    if (initial.hessian.rows() != 3 * static_cast<Eigen::Index>(masses.size())
        || initial.internal_dynamical_matrix.rows() != initial.internal_basis.cols())
        throw std::invalid_argument("Initial reference and mass dimensions differ");
    if (!std::isfinite(options.positive_definite_floor_thz)
        || options.positive_definite_floor_thz <= 0.0
        || !std::isfinite(options.ridge) || options.ridge < 0.0
        || options.max_iterations <= 0
        || !std::isfinite(options.gradient_tolerance) || options.gradient_tolerance <= 0.0
        || !std::isfinite(options.soft_block_frequency_thz)
        || options.soft_block_frequency_thz < 0.0 || options.soft_block_max_modes <= 0
        || !std::isfinite(options.degeneracy_gap_thz) || options.degeneracy_gap_thz < 0.0
        || options.degeneracy_block_max_modes <= 0)
        throw std::invalid_argument("Invalid force-matching options");

    ForceMatchResult result;
    const Eigen::Index full_dim = initial.hessian.rows();
    const Eigen::Index dim = initial.internal_dynamical_matrix.rows();
    if (dim == 0) {
        result.reference = initial;
        result.diagnostics.force_rmse = 0.0;
        result.diagnostics.mean_force_rms = 0.0;
        result.diagnostics.relative_dynamical_change = 0.0;
        return result;
    }

    Eigen::MatrixXd q(dim, samples.size()), force(dim, samples.size());
    Eigen::VectorXd weights(samples.size());
    double weight_sum = 0.0;
    for (Eigen::Index isample = 0; isample < static_cast<Eigen::Index>(samples.size()); ++isample) {
        const auto &sample = samples[isample];
        if (sample.displacement_cartesian.size() != full_dim
            || sample.force_cartesian.size() != full_dim
            || !sample.displacement_cartesian.allFinite() || !sample.force_cartesian.allFinite()
            || !std::isfinite(sample.weight) || sample.weight <= 0.0)
            throw std::invalid_argument("Force sample has invalid dimensions, values, or weight");
        q.col(isample) = initial.internal_basis.transpose()
            * (initial.sqrt_mass_coordinates.array()
               * sample.displacement_cartesian.array()).matrix();
        force.col(isample) = initial.internal_basis.transpose()
            * (sample.force_cartesian.array()
               / initial.sqrt_mass_coordinates.array()).matrix();
        weights(isample) = sample.weight;
        weight_sum += sample.weight;
    }
    const Eigen::VectorXd mean_q = q * weights / weight_sum;
    const Eigen::VectorXd mean_force = force * weights / weight_sum;
    q.colwise() -= mean_q;
    force.colwise() -= mean_force;
    for (Eigen::Index isample = 0; isample < q.cols(); ++isample) {
        const double scale = std::sqrt(weights(isample));
        q.col(isample) *= scale;
        force.col(isample) *= scale;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(initial.internal_dynamical_matrix);
    if (solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() <= 0.0)
        throw std::runtime_error("Force matching requires a positive initial reference");
    const Eigen::MatrixXd modes = solver.eigenvectors();
    const Eigen::VectorXd original = solver.eigenvalues();
    const Eigen::MatrixXd modal_q = modes.transpose() * q;
    const Eigen::MatrixXd modal_force = modes.transpose() * force;
    Eigen::MatrixXd fitted_modal = original.asDiagonal();
    const double floor = std::pow(
        options.positive_definite_floor_thz / frequency_conversion_thz, 2);

    const double weighted_ridge = options.ridge * weight_sum;
    for (Eigen::Index mode = 0; mode < dim; ++mode) {
        const double denominator = modal_q.row(mode).squaredNorm() + weighted_ridge;
        double fitted = original(mode);
        if (denominator > 0.0)
            fitted = (-modal_force.row(mode).dot(modal_q.row(mode))
                      + weighted_ridge * original(mode)) / denominator;
        fitted_modal(mode, mode) = std::max(fitted, floor);
    }

    std::vector<std::vector<int>> blocks;
    if (options.coupled_blocks) {
        std::vector<bool> assigned(dim, false);
        std::vector<int> soft;
        for (Eigen::Index mode = 0; mode < dim
             && static_cast<int>(soft.size()) < options.soft_block_max_modes; ++mode) {
            if (signed_frequency_thz(original(mode)) <= options.soft_block_frequency_thz) {
                soft.push_back(static_cast<int>(mode));
                assigned[mode] = true;
            }
        }
        if (soft.size() > 1) blocks.push_back(std::move(soft));
        for (Eigen::Index first = 0; first < dim;) {
            if (assigned[first]) {
                ++first;
                continue;
            }
            std::vector<int> block{static_cast<int>(first)};
            Eigen::Index next = first + 1;
            while (next < dim && !assigned[next]
                   && static_cast<int>(block.size()) < options.degeneracy_block_max_modes
                   && std::abs(signed_frequency_thz(original(next))
                               - signed_frequency_thz(original(next - 1)))
                        <= options.degeneracy_gap_thz) {
                block.push_back(static_cast<int>(next));
                ++next;
            }
            if (block.size() > 1) blocks.push_back(block);
            first = next;
        }
    }

    for (const auto &block : blocks) {
        const Eigen::Index block_dim = block.size();
        Eigen::MatrixXd x(block_dim, samples.size()), f(block_dim, samples.size());
        Eigen::MatrixXd block_initial(block_dim, block_dim);
        Eigen::MatrixXd block_start = Eigen::MatrixXd::Zero(block_dim, block_dim);
        for (Eigen::Index i = 0; i < block_dim; ++i) {
            x.row(i) = modal_q.row(block[i]);
            f.row(i) = modal_force.row(block[i]);
            for (Eigen::Index j = 0; j < block_dim; ++j)
                block_initial(i, j) = i == j ? original(block[i]) : 0.0;
            // Use the new modal fit, retaining the old FC only as the ridge target.
            // Stay inside the root domain: at the floor the root gradient vanishes.
            block_start(i, i) = std::max(fitted_modal(block[i], block[i]), 2.0 * floor);
        }
        RootFitObjective objective{x, f, block_initial, {}, options.root_representation,
                                   options.root_representation == RootRepresentation::Normal
                                       ? 0.0 : floor,
                                   options.ridge, weight_sum};
        Eigen::MatrixXd root = to_root(block_start, options.root_representation,
            options.root_representation == RootRepresentation::Normal ? 0.0 : floor);
        Eigen::VectorXd packed = pack_symmetric(root);
        LBFGSpp::LBFGSParam<double> parameters;
        parameters.max_iterations = options.max_iterations;
        parameters.epsilon = options.gradient_tolerance;
        parameters.epsilon_rel = options.gradient_tolerance;
        parameters.epsgmax = options.gradient_tolerance;
        LBFGSpp::LBFGSSolver<double> optimizer(parameters);
        double objective_value = 0.0;
        result.diagnostics.optimizer_iterations += optimizer.minimize(
            objective, packed, objective_value);
        root = unpack_symmetric(packed, block_dim);
        Eigen::MatrixXd block_fitted = from_root(root, options.root_representation,
            options.root_representation == RootRepresentation::Normal ? 0.0 : floor);
        if (options.root_representation == RootRepresentation::Normal) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> block_solver(block_fitted);
            const Eigen::VectorXd values = block_solver.eigenvalues().cwiseMax(floor);
            block_fitted = block_solver.eigenvectors() * values.asDiagonal()
                         * block_solver.eigenvectors().transpose();
        }
        for (Eigen::Index i = 0; i < block_dim; ++i)
            for (Eigen::Index j = 0; j < block_dim; ++j)
                fitted_modal(block[i], block[j]) = block_fitted(i, j);
        ++result.diagnostics.coupled_blocks;
    }

    Eigen::MatrixXd fitted_internal = modes * fitted_modal * modes.transpose();
    if (!symmetry_operations.empty()) {
        const MatrixList operators = internal_symmetry_operators(
            symmetry_operations, initial.internal_basis, masses.size());
        fitted_internal = project_symmetry(fitted_internal, operators);
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> fitted_solver(fitted_internal);
    if (fitted_solver.info() != Eigen::Success)
        throw std::runtime_error("Failed to diagonalize force-matched dynamical matrix");
    Eigen::VectorXd fitted_values = fitted_solver.eigenvalues().cwiseMax(floor);
    fitted_internal = fitted_solver.eigenvectors() * fitted_values.asDiagonal()
                    * fitted_solver.eigenvectors().transpose();

    Options reference_options;
    reference_options.stabilization = StabilizationPolicy::NegativeFloor;
    reference_options.root_representation = options.root_representation;
    reference_options.unstable_frequency_thz = options.positive_definite_floor_thz;
    reference_options.positive_definite_floor_thz = options.positive_definite_floor_thz;
    reference_options.ridge = options.ridge;
    reference_options.max_iterations = options.max_iterations;
    reference_options.gradient_tolerance = options.gradient_tolerance;
    const Eigen::MatrixXd dynamical = initial.internal_basis * fitted_internal
                                    * initial.internal_basis.transpose();
    result.reference = build_reference(
        HessianUtils::from_dynamical_matrix(dynamical, masses), masses,
        reference_options, symmetry_operations);

    const Eigen::MatrixXd residual = force
        + result.reference.internal_dynamical_matrix * q;
    result.diagnostics.force_rmse = std::sqrt(
        residual.squaredNorm() / (weight_sum * static_cast<double>(dim)));
    result.diagnostics.mean_force_rms = mean_force.norm() / std::sqrt(static_cast<double>(dim));
    result.diagnostics.relative_dynamical_change =
        (result.reference.internal_dynamical_matrix - initial.internal_dynamical_matrix).norm()
        / std::max(1.0, initial.internal_dynamical_matrix.norm());
    result.diagnostics.floor_modes = static_cast<int>((result.reference.diagnostics.eigenvalues_after.array()
        <= floor * (1.0 + 1e-8)).count());
    return result;
}

double force_sample_rmse(const ReferenceResult &reference,
                         const std::vector<ForceSample> &samples)
{
    if (samples.empty()) return std::numeric_limits<double>::quiet_NaN();
    double squared_error = 0.0;
    double weight_sum = 0.0;
    for (const auto &sample : samples) {
        if (sample.displacement_cartesian.size() != reference.hessian.rows()
            || sample.force_cartesian.size() != reference.hessian.rows()
            || !sample.displacement_cartesian.allFinite() || !sample.force_cartesian.allFinite()
            || !std::isfinite(sample.weight) || sample.weight <= 0.0)
            throw std::invalid_argument("Force sample has invalid dimensions, values, or weight");
        const Eigen::VectorXd residual = sample.force_cartesian
            + reference.hessian * sample.displacement_cartesian;
        const Eigen::VectorXd internal_residual = reference.internal_basis.transpose()
            * (residual.array() / reference.sqrt_mass_coordinates.array()).matrix();
        squared_error += sample.weight * internal_residual.squaredNorm();
        weight_sum += sample.weight;
    }
    const Eigen::Index dim = reference.internal_basis.cols();
    return dim == 0 ? 0.0 : std::sqrt(squared_error / (weight_sum * static_cast<double>(dim)));
}

SelfConsistentResult iterate_reference(
    const ReferenceResult &initial,
    const std::vector<double> &masses,
    const std::vector<SymmetryOperation> &symmetry_operations,
    const SelfConsistentOptions &options,
    const EnsembleForceEvaluator &evaluator,
    const EnsembleSampleFilter &filter)
{
    if (!evaluator) throw std::invalid_argument("Self-consistent iteration requires a force evaluator");
    if (options.ensemble.size <= 0 || options.warmup_iterations < 0
        || options.minimum_iterations <= 0 || options.maximum_iterations < options.minimum_iterations
        || options.consecutive_converged <= 0
        || !std::isfinite(options.mixing) || options.mixing <= 0.0 || options.mixing > 1.0
        || !std::isfinite(options.relative_dynamical_tolerance)
        || options.relative_dynamical_tolerance <= 0.0
        || !std::isfinite(options.maximum_frequency_change_thz)
        || options.maximum_frequency_change_thz <= 0.0
        || (options.maximum_sampling_attempts > 0
            && options.maximum_sampling_attempts < options.ensemble.size))
        throw std::invalid_argument("Invalid self-consistent iteration options");

    SelfConsistentResult result;
    result.reference = initial;
    int consecutive = 0;
    const double floor = std::pow(
        options.fit.positive_definite_floor_thz / frequency_conversion_thz, 2);
    for (int iteration = 0; iteration < options.maximum_iterations; ++iteration) {
        EnsembleOptions ensemble_options = options.ensemble;
        ensemble_options.seed += static_cast<std::uint32_t>(iteration);
        const int automatic_attempts = options.ensemble.size
                > std::numeric_limits<int>::max() / 20
            ? std::numeric_limits<int>::max() : 20 * options.ensemble.size;
        const int maximum_attempts = options.maximum_sampling_attempts > 0
            ? options.maximum_sampling_attempts : (filter ? automatic_attempts : options.ensemble.size);
        const auto sampling = sample_classical_ensemble_filtered(
            result.reference, ensemble_options, maximum_attempts, filter);
        if (sampling.samples.size() != static_cast<size_t>(ensemble_options.size))
            throw std::runtime_error("Unable to collect the requested safe ensemble: accepted "
                + std::to_string(sampling.samples.size()) + " of "
                + std::to_string(ensemble_options.size) + " samples after "
                + std::to_string(sampling.attempted_samples) + " attempts");
        const auto samples = evaluator(result.reference, sampling.samples);
        if (samples.size() != sampling.samples.size())
            throw std::runtime_error("Force evaluator returned the wrong number of samples");

        ForceMatchOptions fit_options = options.fit;
        fit_options.coupled_blocks = iteration >= options.warmup_iterations;
        const ForceMatchResult fitted = fit_force_samples(
            result.reference, masses, fit_options, samples, symmetry_operations);
        const Eigen::MatrixXd previous = result.reference.internal_dynamical_matrix;
        const Eigen::VectorXd previous_values = result.reference.diagnostics.eigenvalues_after;
        const double offset = options.fit.root_representation == RootRepresentation::Normal
            ? 0.0 : floor;
        const Eigen::MatrixXd previous_root = to_root(
            previous, options.fit.root_representation, offset);
        const Eigen::MatrixXd fitted_root = to_root(
            fitted.reference.internal_dynamical_matrix, options.fit.root_representation, offset);
        Eigen::MatrixXd mixed_internal = from_root(
            (1.0 - options.mixing) * previous_root + options.mixing * fitted_root,
            options.fit.root_representation, offset);

        Options reference_options;
        reference_options.stabilization = StabilizationPolicy::NegativeFloor;
        reference_options.root_representation = options.fit.root_representation;
        reference_options.unstable_frequency_thz = options.fit.positive_definite_floor_thz;
        reference_options.positive_definite_floor_thz = options.fit.positive_definite_floor_thz;
        const Eigen::MatrixXd full_dynamical = result.reference.internal_basis * mixed_internal
                                             * result.reference.internal_basis.transpose();
        result.reference = build_reference(
            HessianUtils::from_dynamical_matrix(full_dynamical, masses), masses,
            reference_options, symmetry_operations);

        IterationRecord record;
        record.iteration = iteration + 1;
        record.relative_dynamical_change =
            (result.reference.internal_dynamical_matrix - previous).norm()
            / std::max(1.0, previous.norm());
        record.force_rmse = fitted.diagnostics.force_rmse;
        record.mean_force_rms = fitted.diagnostics.mean_force_rms;
        record.coupled_blocks = fitted.diagnostics.coupled_blocks;
        record.sampling_attempts = sampling.attempted_samples;
        record.rejected_samples = sampling.rejected_samples;
        record.sampling_acceptance_rate = sampling.attempted_samples > 0
            ? static_cast<double>(sampling.samples.size()) / sampling.attempted_samples : 1.0;
        const Eigen::VectorXd current_values = result.reference.diagnostics.eigenvalues_after;
        record.floor_modes = static_cast<int>((current_values.array()
            <= floor * (1.0 + 1e-8)).count());
        for (Eigen::Index mode = 0; mode < current_values.size(); ++mode) {
            record.maximum_frequency_change_thz = std::max(
                record.maximum_frequency_change_thz,
                std::abs(signed_frequency_thz(current_values(mode))
                         - signed_frequency_thz(previous_values(mode))));
        }
        record.minimum_frequency_thz = signed_frequency_thz(current_values.minCoeff());
        result.iterations.push_back(record);

        // Modal stationarity does not imply stationarity once couplings are enabled.
        const bool converged = fit_options.coupled_blocks
            && iteration + 1 >= options.minimum_iterations
            && record.relative_dynamical_change <= options.relative_dynamical_tolerance
            && record.maximum_frequency_change_thz <= options.maximum_frequency_change_thz;
        consecutive = converged ? consecutive + 1 : 0;
        if (consecutive >= options.consecutive_converged) {
            result.converged = true;
            break;
        }
    }
    return result;
}

double signed_frequency_thz(double eigenvalue)
{
    if (eigenvalue == 0.0) return 0.0;
    return std::copysign(frequency_conversion_thz * std::sqrt(std::abs(eigenvalue)), eigenvalue);
}

std::string policy_name(StabilizationPolicy policy)
{
    if (policy == StabilizationPolicy::NegativeFloor) return "negative-floor";
    if (policy == StabilizationPolicy::SpectralFloor) return "spectral-floor";
    return "abs";
}

std::string root_name(RootRepresentation representation)
{
    if (representation == RootRepresentation::Normal) return "normal";
    if (representation == RootRepresentation::SquareRoot) return "sqrt";
    return "root4";
}

} // namespace TIHarmonicReference
