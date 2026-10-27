// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "trajectory_transform.h"
#include "compute_posavg.h"
#include "tools/tools.h"

#include <stdexcept>

// ── Center-of-mass utilities ────────────────────────────────────────────
namespace {
template<typename Accessor>
Eigen::RowVector3d get_com_impl(const Atoms &ref,
                                 const std::vector<double> &atom_masses,
                                 Accessor &&acc)
{
    Eigen::RowVector3d res = Eigen::RowVector3d::Zero();
    double total_mass = 0;
    for (int i = 0; i < ref.size(); ++i)
    {
        int type = ref.chemmap_[i];
        if (type < 0 || type >= (int)atom_masses.size()) continue;
        total_mass += atom_masses[type];
        res += atom_masses[type] * acc(ref, i);
    }
    if (total_mass > 0)
        res /= total_mass;
    return res;
}
} // anonymous namespace

Eigen::RowVector3d get_center_of_mass(const Atoms &ref,
                                       const std::vector<double> &atom_masses)
{
    return get_com_impl(ref, atom_masses,
                        [](const Atoms &a, int i) -> const auto& { return a.pos_[i]; });
}

Eigen::RowVector3d get_center_of_mass_v(const Atoms &ref,
                                         const std::vector<double> &atom_masses)
{
    return get_com_impl(ref, atom_masses,
                        [](const Atoms &a, int i) -> const auto& { return a.velocities_[i]; });
}

void validate_fixed_cell_trajectory(const std::vector<Atoms> &frames,
                                    double relative_tolerance)
{
    if (frames.empty()) throw std::invalid_argument("Trajectory contains no frames");
    if (!std::isfinite(relative_tolerance) || relative_tolerance < 0.0)
        throw std::invalid_argument("Cell tolerance must be finite and non-negative");
    const auto &first = frames.front();
    if (first.size() == 0 || std::abs(first.cell_.determinant()) < 1e-14)
        throw std::invalid_argument("Trajectory has an empty structure or singular cell");
    const double scale = std::max(1.0, first.cell_.norm());
    for (const auto &frame : frames) {
        if (frame.size() != first.size() || frame.chemmap_ != first.chemmap_)
            throw std::invalid_argument("Trajectory atom count or type ordering changes between frames");
        if ((frame.cell_ - first.cell_).norm() > relative_tolerance * scale)
            throw std::invalid_argument("Trajectory cell changes; this operation requires a fixed cell");
    }
}

Eigen::VectorXd minimum_image_displacement(const Atoms &reference,
                                           const Atoms &frame,
                                           const std::vector<double> &masses,
                                           bool remove_center_of_mass)
{
    if (frame.size() != reference.size() || frame.chemmap_ != reference.chemmap_)
        throw std::invalid_argument("Frame and reference atom ordering differ");
    if (masses.size() != static_cast<size_t>(reference.size()))
        throw std::invalid_argument("Mass count differs from atom count");
    Eigen::VectorXd displacement(3 * reference.size());
    if (reference.size() == 0) return displacement;
    // Choose a common periodic branch before subtracting the mass-weighted COM.
    // Wrapping each raw displacement first splits a drifting crystal at L/2.
    const Eigen::RowVector3d common_shift = remove_center_of_mass
        ? Eigen::RowVector3d(frame.pos_[0] - reference.pos_[0])
        : Eigen::RowVector3d::Zero();
    const Eigen::Matrix3d coordinate_map = trajectory_periodic_coordinate_map(reference);
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    double total_mass = 0.0;
    for (int iat = 0; iat < reference.size(); ++iat) {
        if (!std::isfinite(masses[iat]) || masses[iat] <= 0.0)
            throw std::invalid_argument("Masses must be finite and positive");
        const Eigen::RowVector3d delta = frame.pos_[iat] - reference.pos_[iat] - common_shift;
        Eigen::Vector3d cartesian = delta.transpose();
        if (!reference.is_nonperiodic()) {
            const Eigen::RowVector3d fractional = delta * coordinate_map;
            Eigen::RowVector3d image = Eigen::RowVector3d::Zero();
            for (int axis = 0; axis < 3; ++axis)
                if (reference.pbc_[axis]) image(axis) = std::round(fractional(axis));
            cartesian = (delta - image * reference.cell_).transpose();
        }
        displacement.segment<3>(3 * iat) = cartesian;
        center += masses[iat] * cartesian;
        total_mass += masses[iat];
    }
    if (remove_center_of_mass) {
        center /= total_mass;
        for (int iat = 0; iat < reference.size(); ++iat)
            displacement.segment<3>(3 * iat) -= center;
    }
    return displacement;
}

Atoms fixed_cell_centroid(std::vector<Atoms> frames,
                          const std::vector<double> &masses,
                          double relative_cell_tolerance,
                          bool wrap_result)
{
    validate_fixed_cell_trajectory(frames, relative_cell_tolerance);
    PositionAverageOptions options;
    options.coordinates = AverageCoordinates::Cartesian;
    options.remove_drift = true;
    options.geometry_only = true;
    options.wrap_result = wrap_result;
    options.relative_cell_tolerance = relative_cell_tolerance;
    options.masses = masses;
    ComputePosavg average(std::move(options));
    average.init(frames.front());
    for (const auto &frame : frames) average.accumulate(frame);
    average.compute();
    return average.atomacc;
}

Eigen::Matrix3d trajectory_periodic_coordinate_map(const Atoms &frame)
{
    if (frame.is_nonperiodic()) return Eigen::Matrix3d::Zero();
    if (frame.cell_.fullPivLu().isInvertible()) return frame.cell_.inverse();

    int count = 0;
    for (bool periodic : frame.pbc_) if (periodic) ++count;
    Eigen::MatrixXd basis(count, 3);
    int row = 0;
    for (int axis = 0; axis < 3; ++axis)
        if (frame.pbc_[axis]) basis.row(row++) = frame.cell_.row(axis);
    const Eigen::MatrixXd gram = basis * basis.transpose();
    const auto decomposition = gram.fullPivLu();
    if (!decomposition.isInvertible())
        throw std::invalid_argument("Trajectory unwrapping requires independent periodic cell vectors");
    const Eigen::MatrixXd dual = decomposition.solve(basis);
    Eigen::Matrix3d map = Eigen::Matrix3d::Zero();
    row = 0;
    for (int axis = 0; axis < 3; ++axis)
        if (frame.pbc_[axis]) map.col(axis) = dual.row(row++).transpose();
    return map;
}

void unwarp(std::vector<Atoms> &atoms)
{
    std::vector<Eigen::RowVector3d> shift(atoms[0].size(), Eigen::RowVector3d::Zero());
    Eigen::RowVector3d diff, fdiff;
    for (int i = 1; i < atoms.size(); ++i)
    {
        if (atoms[i].is_nonperiodic()) continue;
        Eigen::Matrix3d cell = atoms[i].cell_;
        Eigen::Matrix3d invcell = trajectory_periodic_coordinate_map(atoms[i]);
        for (int iat = 0; iat < atoms[i].size(); ++iat)
        {
            atoms[i].pos_[iat] += shift[iat] * cell;
            diff = atoms[i].pos_[iat] - atoms[i-1].pos_[iat];
            fdiff = diff * invcell;
            for (int a = 0; a < 3; ++a)
            {
                diff[a] = atoms[i].pbc_[a] ? -floor(fdiff[a] + 0.5) : 0.0;
            }
            atoms[i].pos_[iat] += diff * cell;
            shift[iat] += diff;
        }
    }
}

void unwarp(const Atoms &atfirst, std::vector<Atoms> &atoms)
{
    std::vector<Eigen::RowVector3d> shift(atoms[0].size(), Eigen::RowVector3d::Zero());
    Eigen::RowVector3d diff, fdiff;
    if (!atoms[0].is_nonperiodic())
    {
        Eigen::Matrix3d cell = atoms[0].cell_;
        Eigen::Matrix3d invcell = trajectory_periodic_coordinate_map(atoms[0]);
        for (int iat = 0; iat < atoms[0].size(); ++iat)
        {
            atoms[0].pos_[iat] += shift[iat] * cell;
            diff = atoms[0].pos_[iat] - atfirst.pos_[iat];
            fdiff = diff * invcell;
            for (int a = 0; a < 3; ++a)
            {
                diff[a] = atoms[0].pbc_[a] ? -floor(fdiff[a] + 0.5) : 0.0;
            }
            atoms[0].pos_[iat] += diff * cell;
            shift[iat] += diff;
        }
    }
    for (int i = 1; i < atoms.size(); ++i)
    {
        if (atoms[i].is_nonperiodic()) continue;
        Eigen::Matrix3d cell = atoms[i].cell_;
        Eigen::Matrix3d invcell = trajectory_periodic_coordinate_map(atoms[i]);
        for (int iat = 0; iat < atoms[i].size(); ++iat)
        {
            atoms[i].pos_[iat] += shift[iat] * cell;
            diff = atoms[i].pos_[iat] - atoms[i-1].pos_[iat];
            fdiff = diff * invcell;
            for (int a = 0; a < 3; ++a)
            {
                diff[a] = atoms[i].pbc_[a] ? -floor(fdiff[a] + 0.5) : 0.0;
            }
            atoms[i].pos_[iat] += diff * cell;
            shift[iat] += diff;
        }
    }
}

void stormervelet(std::vector<Atoms> &atoms, double ts)
{
    int natoms = atoms[0].size();
    double invts = 1.0 / ts;
    // first frame is less accurate
    // last frams is un-trustable
    atoms[0].velocities_.resize(natoms);
    atoms.back().velocities_.resize(natoms);
    for (int i = 0; i < natoms; ++i)
    {
        atoms[0].velocities_[i] = (atoms[1].pos_[i] - atoms[0].pos_[i]) * invts;
        atoms.back().velocities_[i] = (atoms.back().pos_[i] - atoms[atoms.size() - 2].pos_[i]) * invts;
    }

    for (int i = 1; i < atoms.size() - 1; ++i)
    {
        atoms[i].velocities_.resize(natoms);
        for (int j = 0; j < natoms; ++j)
            atoms[i].velocities_[j] = (atoms[i+1].pos_[j] - atoms[i-1].pos_[j]) * 0.5 * invts;
    }
}

void storveloskipfl(std::vector<Atoms> &atoms, double ts)
{
    int natoms = atoms[0].size();
    double invts = 1.0 / ts;
    // first frame is less accurate
    // last frams is un-trustable
    atoms[0].velocities_.resize(natoms);
    atoms.back().velocities_.resize(natoms);
    for (int i = 0; i < natoms; ++i)
    {
        atoms[0].velocities_[i] = (atoms[1].pos_[i] - atoms[0].pos_[i]) * invts;
        atoms.back().velocities_[i] = (atoms.back().pos_[i] - atoms[atoms.size() - 2].pos_[i]) * invts;
    }

    for (int i = 1; i < atoms.size() - 1; ++i)
    {
        atoms[i].velocities_.resize(natoms);
        for (int j = 0; j < natoms; ++j)
            atoms[i].velocities_[j] = (atoms[i+1].pos_[j] - atoms[i-1].pos_[j]) * 0.5 * invts;
    }
}
// assume unwarpped and all mass are same
void rmdriftv1(std::vector<Atoms> &atoms) {
    Eigen::RowVector3d cm;
    cm << 0,0,0;
    for (int i = 0; i < atoms.size(); ++i) {
        for (int j = 0; j < atoms[i].size(); ++j) {
            cm += atoms[i].pos_[j];
        }
        cm /= atoms[i].size();
        for (int j = 0; j < atoms[i].size(); ++j) {
            atoms[i].pos_[j] -= cm;
        }
    }
}
