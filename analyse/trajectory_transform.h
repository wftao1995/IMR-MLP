// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_TRAJECTORY_TRANSFORM_H
#define IMRMLP_TRAJECTORY_TRANSFORM_H
#include "base/structures.h"

// Row-vector Cartesian displacements times this map give wrapping coordinates.
// Uses the full inverse for invertible cells, otherwise the independent periodic
// rows' dual with zero columns for open axes; NPBC returns zero.
Eigen::Matrix3d trajectory_periodic_coordinate_map(const Atoms &frame);

void unwarp(std::vector<Atoms> &atoms);
void unwarp(const Atoms &atfirst, std::vector<Atoms> &atoms);
void stormervelet(std::vector<Atoms> &atoms, double ts);
void storveloskipfl(std::vector<Atoms> &atoms, double ts);
void rmdriftv1(std::vector<Atoms> &atoms);

// Fixed-cell crystalline trajectory validation and drift-corrected averaging.
void validate_fixed_cell_trajectory(const std::vector<Atoms> &frames,
                                    double relative_tolerance = 1e-7);
Atoms fixed_cell_centroid(std::vector<Atoms> frames,
                          const std::vector<double> &masses,
                          double relative_cell_tolerance = 1e-7,
                          bool wrap_result = true);
// With COM removal, choose images relative to a common translation. Crystalline
// relative displacements must stay within half a period of one another.
Eigen::VectorXd minimum_image_displacement(const Atoms &reference,
                                           const Atoms &frame,
                                           const std::vector<double> &masses,
                                           bool remove_center_of_mass = true);

// ── Center-of-mass utilities (shared by MSD, VACF, NMSD) ────────────
// Returns the mass-weighted center of mass of a configuration.
// Uses Chemicals::atomic_masses; each atom must have a valid chemmap_ entry.
Eigen::RowVector3d get_center_of_mass(const Atoms &ref,
                                       const std::vector<double> &atom_masses);

// Returns the mass-weighted center-of-mass velocity.
Eigen::RowVector3d get_center_of_mass_v(const Atoms &ref,
                                         const std::vector<double> &atom_masses);

#endif
