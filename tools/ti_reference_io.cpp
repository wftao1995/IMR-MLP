#include "tools/ti_reference_io.h"

#include "tools/hessian_utils.h"
#include "tools/ti_harmonic_reference.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace TIReferenceIO {
namespace {

Eigen::Matrix3d lower_cell(const Eigen::Matrix3d &cell)
{
    const double a = cell.row(0).norm();
    const double b = cell.row(1).norm();
    const double c = cell.row(2).norm();
    if (a <= 0.0 || b <= 0.0 || c <= 0.0)
        throw std::invalid_argument("Cannot export a degenerate reference cell");
    const double cos_beta = std::clamp(cell.row(2).dot(cell.row(0)) / (c * a), -1.0, 1.0);
    const double cos_gamma = std::clamp(cell.row(0).dot(cell.row(1)) / (a * b), -1.0, 1.0);
    const double cos_alpha = std::clamp(cell.row(1).dot(cell.row(2)) / (b * c), -1.0, 1.0);
    const double sin_gamma = std::sqrt(std::max(0.0, 1.0 - cos_gamma * cos_gamma));
    if (sin_gamma < 1e-12) throw std::invalid_argument("Cannot export a collinear reference cell");
    Eigen::Matrix3d lower = Eigen::Matrix3d::Zero();
    lower(0, 0) = a;
    lower(1, 0) = b * cos_gamma;
    lower(1, 1) = b * sin_gamma;
    lower(2, 0) = c * cos_beta;
    lower(2, 1) = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
    lower(2, 2) = std::sqrt(std::max(
        0.0, c * c - lower(2, 0) * lower(2, 0) - lower(2, 1) * lower(2, 1)));
    return lower;
}

void validate_hessian(const Eigen::MatrixXd &hessian, int natoms)
{
    if (hessian.rows() != 3 * natoms || hessian.cols() != 3 * natoms
        || !hessian.allFinite())
        throw std::invalid_argument("Hessian dimensions or values are invalid");
}

Eigen::MatrixXd internal_dynamical_matrix(const Eigen::MatrixXd &hessian,
                                          const std::vector<double> &masses)
{
    const int natoms = masses.size();
    Eigen::VectorXd inverse_sqrt_mass(3 * natoms);
    for (int i = 0; i < natoms; ++i) {
        if (!std::isfinite(masses[i]) || masses[i] <= 0.0)
            throw std::invalid_argument("Sparse FC export requires positive masses");
        inverse_sqrt_mass.segment<3>(3 * i).setConstant(1.0 / std::sqrt(masses[i]));
    }
    const Eigen::MatrixXd dynamical = inverse_sqrt_mass.asDiagonal()
        * hessian * inverse_sqrt_mass.asDiagonal();
    const Eigen::MatrixXd basis = HessianUtils::mass_weighted_internal_basis(masses);
    if (basis.cols() == 0) return Eigen::MatrixXd(0, 0);
    return basis.transpose() * dynamical * basis;
}

double minimum_internal_frequency(const Eigen::MatrixXd &hessian,
                                  const std::vector<double> &masses)
{
    const Eigen::VectorXd eigenvalues = internal_mode_eigenvalues(hessian, masses);
    if (eigenvalues.size() == 0) return std::numeric_limits<double>::infinity();
    return TIHarmonicReference::signed_frequency_thz(eigenvalues.minCoeff());
}

double minimum_image_distance(const Atoms &reference,
                              const Eigen::Matrix3d &inverse_cell,
                              int i,
                              int j)
{
    Eigen::RowVector3d fractional =
        (reference.pos_[i] - reference.pos_[j]) * inverse_cell;
    fractional.array() -= fractional.array().round();
    return (fractional * reference.cell_).norm();
}

} // namespace

Eigen::VectorXd internal_mode_eigenvalues(const Eigen::MatrixXd &hessian,
                                          const std::vector<double> &masses)
{
    validate_hessian(hessian, static_cast<int>(masses.size()));
    const Eigen::MatrixXd internal = internal_dynamical_matrix(hessian, masses);
    if (internal.rows() == 0) return Eigen::VectorXd{};
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        internal, Eigen::EigenvaluesOnly);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("TI FC eigenvalue analysis failed");
    return solver.eigenvalues();
}

SparseResult truncate_force_constants(const Atoms &reference,
                                      const Eigen::MatrixXd &hessian,
                                      const std::vector<double> &masses,
                                      const SparseOptions &options)
{
    const int natoms = reference.size();
    validate_hessian(hessian, natoms);
    if (static_cast<int>(masses.size()) != natoms
        || !std::isfinite(options.cutoff_angstrom) || options.cutoff_angstrom < 0.0
        || !std::isfinite(options.block_threshold) || options.block_threshold < 0.0
        || !std::isfinite(options.minimum_frequency_thz)
        || options.minimum_frequency_thz < 0.0)
        throw std::invalid_argument("Invalid sparse FC option or mass array");

    SparseResult result;
    result.hessian = Eigen::MatrixXd::Zero(3 * natoms, 3 * natoms);
    Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(3 * natoms, 3 * natoms);
    const Eigen::MatrixXd symmetric = 0.5 * (hessian + hessian.transpose());
    const Eigen::Matrix3d inverse_cell = reference.cell_.inverse();
    for (int i = 0; i < natoms; ++i) {
        for (int j = i + 1; j < natoms; ++j) {
            ++result.diagnostics.total_offdiagonal_blocks;
            const Eigen::Matrix3d block = symmetric.block<3, 3>(3 * i, 3 * j);
            const bool inside_cutoff = options.cutoff_angstrom <= 0.0
                || minimum_image_distance(reference, inverse_cell, i, j)
                    <= options.cutoff_angstrom;
            if (!inside_cutoff || block.norm() <= options.block_threshold) continue;
            ++result.diagnostics.retained_offdiagonal_blocks;
            result.hessian.block<3, 3>(3 * i, 3 * j) = block;
            result.hessian.block<3, 3>(3 * j, 3 * i) = block.transpose();
            for (int a = 0; a < 3; ++a) {
                laplacian(3 * i + a, 3 * i + a) += 1.0;
                laplacian(3 * j + a, 3 * j + a) += 1.0;
                laplacian(3 * i + a, 3 * j + a) -= 1.0;
                laplacian(3 * j + a, 3 * i + a) -= 1.0;
            }
        }
    }
    for (int i = 0; i < natoms; ++i) {
        Eigen::Matrix3d onsite = Eigen::Matrix3d::Zero();
        for (int j = 0; j < natoms; ++j)
            if (i != j) onsite -= result.hessian.block<3, 3>(3 * i, 3 * j);
        result.hessian.block<3, 3>(3 * i, 3 * i) = onsite;
    }

    result.diagnostics.retained_fraction = result.diagnostics.total_offdiagonal_blocks == 0
        ? 1.0 : static_cast<double>(result.diagnostics.retained_offdiagonal_blocks)
                    / result.diagnostics.total_offdiagonal_blocks;
    result.diagnostics.minimum_frequency_before_thz =
        minimum_internal_frequency(result.hessian, masses);
    if (result.diagnostics.minimum_frequency_before_thz
        + 1e-10 < options.minimum_frequency_thz) {
        if (!options.stabilize || result.diagnostics.retained_offdiagonal_blocks == 0)
            throw std::runtime_error("Truncated FC is not positive enough and stabilization is disabled");
        const Eigen::MatrixXd dynamical = internal_dynamical_matrix(result.hessian, masses);
        const Eigen::MatrixXd graph = internal_dynamical_matrix(laplacian, masses);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> graph_solver(
            graph, Eigen::EigenvaluesOnly);
        if (graph_solver.info() != Eigen::Success
            || graph_solver.eigenvalues().minCoeff() <= 1e-12)
            throw std::runtime_error("Sparse FC graph is disconnected or cannot be stabilized");
        const double target = std::pow(
            options.minimum_frequency_thz / TIHarmonicReference::frequency_conversion_thz, 2);
        const Eigen::MatrixXd deficit = target
            * Eigen::MatrixXd::Identity(dynamical.rows(), dynamical.cols()) - dynamical;
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
            deficit, graph, Eigen::EigenvaluesOnly);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Sparse FC stabilization eigenproblem failed");
        result.diagnostics.isotropic_stabilizer = std::max(0.0,
            solver.eigenvalues().maxCoeff() * (1.0 + 1e-10) + 1e-12);
        result.hessian += result.diagnostics.isotropic_stabilizer * laplacian;
    }
    result.diagnostics.minimum_frequency_after_thz =
        minimum_internal_frequency(result.hessian, masses);
    result.diagnostics.acoustic_sum_rule_error =
        HessianUtils::acoustic_sum_rule_error(result.hessian);
    result.diagnostics.relative_hessian_change =
        (result.hessian - symmetric).norm() / std::max(1e-30, symmetric.norm());
    return result;
}

void write_force_constants(const std::string &filename,
                           const Eigen::MatrixXd &hessian)
{
    if (hessian.rows() == 0 || hessian.rows() != hessian.cols() || hessian.rows() % 3 != 0
        || !hessian.allFinite())
        throw std::invalid_argument("Cannot write an invalid Hessian");
    const int natoms = hessian.rows() / 3;
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Cannot open FORCE_CONSTANTS output: " + filename);
    output << std::setprecision(15) << natoms << ' ' << natoms << '\n';
    for (int i = 0; i < natoms; ++i) {
        for (int j = 0; j < natoms; ++j) {
            output << i + 1 << ' ' << j + 1 << '\n';
            for (int a = 0; a < 3; ++a)
                output << std::setw(22) << hessian(3 * i + a, 3 * j) << ' '
                       << std::setw(22) << hessian(3 * i + a, 3 * j + 1) << ' '
                       << std::setw(22) << hessian(3 * i + a, 3 * j + 2) << '\n';
        }
    }
}

Atoms write_lammps_reference(const std::string &filename,
                             const Atoms &reference,
                             const Eigen::MatrixXd &hessian,
                             double energy0)
{
    validate_hessian(hessian, reference.size());
    if (!std::isfinite(energy0))
        throw std::invalid_argument("Invalid LAMMPS harmonic-reference output option");
    const Eigen::Matrix3d cell = lower_cell(reference.cell_);
    const Eigen::Matrix3d rotation = reference.cell_.inverse() * cell;
    if ((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-8)
        throw std::runtime_error("Failed to construct the LAMMPS reference-cell rotation");
    Atoms rotated = reference;
    rotated.cell_ = cell;
    for (auto &position : rotated.pos_) position = position * rotation;
    rotated.wrap();

    int block_count = 0;
    for (int i = 0; i < reference.size(); ++i)
        for (int j = i; j < reference.size(); ++j)
            if (hessian.block<3, 3>(3 * i, 3 * j).norm() > 0.0)
                ++block_count;

    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Cannot open LAMMPS reference output: " + filename);
    output << std::setprecision(17);
    output << "TI_HARMONIC_REFERENCE 1\nunits metal\n";
    output << "natoms " << reference.size() << "\nenergy0 " << energy0 << "\ncell\n";
    for (int a = 0; a < 3; ++a)
        output << cell(a, 0) << ' ' << cell(a, 1) << ' ' << cell(a, 2) << '\n';
    output << "atoms\n";
    for (int i = 0; i < rotated.size(); ++i)
        output << i + 1 << ' ' << rotated.chemmap_[i] + 1 << ' '
               << rotated.pos_[i](0) << ' ' << rotated.pos_[i](1) << ' '
               << rotated.pos_[i](2) << '\n';
    output << "blocks " << block_count << '\n';
    for (int i = 0; i < reference.size(); ++i) {
        for (int j = i; j < reference.size(); ++j) {
            const Eigen::Matrix3d block = rotation.transpose()
                * hessian.block<3, 3>(3 * i, 3 * j) * rotation;
            if (block.norm() == 0.0) continue;
            output << i + 1 << ' ' << j + 1 << '\n';
            for (int a = 0; a < 3; ++a)
                output << block(a, 0) << ' ' << block(a, 1) << ' ' << block(a, 2) << '\n';
        }
    }
    output << "end\n";
    return rotated;
}

} // namespace TIReferenceIO
