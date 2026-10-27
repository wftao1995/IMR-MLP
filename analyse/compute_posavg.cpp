// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_posavg.h"
#include "trajectory_transform.h"

#include <cmath>
#include <stdexcept>

namespace {
Atoms geometry(const Atoms &atom) {
    Atoms result;
    result.cell_ = atom.cell_;
    result.pos_ = atom.pos_;
    result.chemmap_ = atom.chemmap_;
    result.pbc_ = atom.pbc_;
    result.cartesian = atom.cartesian;
    return result;
}
}

void ComputePosavg::init(const Atoms &atom) {
    if (atom.size() == 0 || atom.chemmap_.size() != atom.pos_.size()) throw std::invalid_argument("Cannot average an empty structure");
    if (!std::isfinite(options_.relative_cell_tolerance) || options_.relative_cell_tolerance < 0.0)
        throw std::invalid_argument("Cell tolerance must be finite and non-negative");
    if (options_.remove_drift) {
        if (options_.masses.size() != static_cast<size_t>(atom.size()))
            throw std::invalid_argument("Mass count differs from atom count");
        for (double mass : options_.masses)
            if (!std::isfinite(mass) || mass <= 0.0)
                throw std::invalid_argument("Masses must be finite and positive");
    }
    nframes = 0;
    cell_sum_.setZero();
    position_sum_.assign(atom.size(), Eigen::RowVector3d::Zero());
    site_energy_sum_.assign(!options_.geometry_only && atom.site_energies_.size() == static_cast<size_t>(atom.size()) ? atom.size() : 0, 0.0);
    coordinates_ = options_.coordinates == AverageCoordinates::Input
        ? (atom.cartesian ? AverageCoordinates::Cartesian : AverageCoordinates::Fractional)
        : options_.coordinates;
    reference_ = geometry(atom);
    previous_ = reference_;
    previous_fractional_.resize(atom.size());
    images_.assign(atom.size(), Eigen::RowVector3d::Zero());
    unwrapped_.clear();
    atomacc = reference_;
    if (!site_energy_sum_.empty()) atomacc.site_energies_ = site_energy_sum_;
    atomacc.cell_.setZero();
    for (auto &position : atomacc.pos_) position.setZero();
    for (auto &energy : atomacc.site_energies_) energy = 0.0;
}

void ComputePosavg::accumulate(const Atoms &atom) {
    if (atom.size() != reference_.size() || atom.chemmap_ != reference_.chemmap_ ||
        atom.pbc_ != reference_.pbc_)
        throw std::invalid_argument("Trajectory atom count, type ordering or periodic axes change between frames");
    if (atomacc.has_site_energies() && atom.site_energies_.size() != atomacc.site_energies_.size())
    {
        site_energy_sum_.clear();
        atomacc.site_energies_.clear();
    }
    if (options_.remove_drift && (atom.cell_ - reference_.cell_).norm() >
        options_.relative_cell_tolerance * std::max(1.0, reference_.cell_.norm()))
        throw std::invalid_argument("Drift-corrected averaging requires a fixed cell");
    const bool fractional = coordinates_ == AverageCoordinates::Fractional;
    const bool needs_cell = !atom.is_nonperiodic() || !atom.cartesian || fractional;
    const bool needs_full_cell = !atom.cartesian || fractional;
    if ((needs_cell && !atom.cell_.allFinite()) ||
        (needs_full_cell && std::abs(atom.cell_.determinant()) < 1e-14))
        throw std::invalid_argument("Trajectory averaging requires a nonsingular cell");
    Eigen::Matrix3d invcell = Eigen::Matrix3d::Identity();
    if (needs_cell) {
        if (needs_full_cell) invcell = atom.cell_.inverse();
        else invcell = trajectory_periodic_coordinate_map(atom);
    }
    if (options_.remove_drift) {
        Atoms current = geometry(atom);
        if (!current.cartesian) current.to_cartesian();
        if (nframes == 0) unwrapped_ = current.pos_;
        else {
            const Eigen::VectorXd step = minimum_image_displacement(previous_, current, options_.masses, true);
            for (int i = 0; i < current.size(); ++i)
                unwrapped_[i] += step.segment<3>(3 * i).transpose();
        }
        previous_ = std::move(current);
    }
    cell_sum_ += atom.cell_;
    for (int i = 0; i < atom.size(); ++i) {
        if (!atom.pos_[i].allFinite())
            throw std::invalid_argument("Trajectory position is not finite");
        Eigen::RowVector3d position;
        if (options_.remove_drift) {
            position = unwrapped_[i];
            if (fractional) position = position * invcell;
        } else {
            position = atom.pos_[i];
            if (needs_cell) {
                Eigen::RowVector3d fraction = atom.cartesian ? Eigen::RowVector3d(position * invcell) : position;
                if (nframes > 0)
                    for (int axis = 0; axis < 3; ++axis)
                        if (atom.pbc_[axis])
                            images_[i][axis] -= std::round(fraction[axis] - previous_fractional_[i][axis]);
                previous_fractional_[i] = fraction;
                if (atom.cartesian && !fractional) {
                    position += images_[i] * atom.cell_;
                } else {
                    position = fraction + images_[i];
                    if (!fractional) position = position * atom.cell_;
                }
            }
        }
        position_sum_[i] += position;
        if (!site_energy_sum_.empty()) site_energy_sum_[i] += atom.site_energies_[i];
    }
    ++nframes;
}

void ComputePosavg::compute() {
    if (nframes == 0) throw std::invalid_argument("Average positions require at least one accumulated frame");
    atomacc.cell_ = cell_sum_ / nframes;
    for (int i = 0; i < atomacc.size(); ++i) atomacc.pos_[i] = position_sum_[i] / nframes;
    for (size_t i = 0; i < site_energy_sum_.size(); ++i) atomacc.site_energies_[i] = site_energy_sum_[i] / nframes;
    atomacc.cartesian = coordinates_ == AverageCoordinates::Cartesian;
    if (reference_.cartesian != atomacc.cartesian) {
        if (reference_.cartesian) atomacc.to_cartesian();
        else atomacc.to_fractional();
    }
    if (options_.wrap_result) atomacc.wrap();
}
