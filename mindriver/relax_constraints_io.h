#pragma once

#include "mindriver/constraints.h"
#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>

class RelaxationDrvV2;

namespace Constraints {

enum class ConstraintKind { Volume, Length, Angle, Ratio, Distance, AtomicAngle, Dihedral, Position, Wall, Formula, Symmetry };
using LatticeConstraintKind = ConstraintKind;

struct ConstraintDeclaration {
    ConstraintKind kind = ConstraintKind::Volume;
    int version = 1;
    std::optional<double> lower, upper;
    std::string expression;
    int component = -1;
    Eigen::RowVector3d normal = Eigen::RowVector3d::Zero();
    int axis = -1;
    int second_axis = -1;
    // Physical units: Angstrom^3, Angstrom, degrees, or a positive length ratio.
    std::optional<double> target;
    // Residual units: det-V, length^2-L^2, cos(angle)-cos(target), or ratio^2-r^2.
    double residual_tolerance = 0.0;
    std::vector<int> operations, sites;
    std::vector<int> atoms; // Zero-based indices; the middle index is the angle vertex.
};

using ConstraintDeclarations = std::vector<ConstraintDeclaration>;
using LatticeConstraintDeclaration = ConstraintDeclaration;
using LatticeConstraintDeclarations = ConstraintDeclarations;
inline constexpr std::size_t lattice_constraints_max_bytes = 64 * 1024;
inline constexpr std::size_t lattice_constraints_max_count = 32;

// Version 1 supports lattice declarations and atomic distance/angle. Parsing
// checks schema; binding selects each frame's initial targets and atomic images.
void configure_symmetry(const ConstraintDeclarations &, RelaxationDrvV2 &);
ConstraintDeclarations parse_constraints(std::istream &input);
ConstraintDeclarations read_constraints(const std::string &path);
std::vector<regulator> bind_constraints(const ConstraintDeclarations &declarations,
                                      const Atoms &initial_frame);

// Strict version-1 schema; malformed input throws invalid_argument, IO errors
// throw runtime_error. Parsing does not bind omitted targets to a structure.
LatticeConstraintDeclarations parse_lattice_constraints(std::istream &input);
LatticeConstraintDeclarations read_lattice_constraints(const std::string &path);

// Bind fresh AUL equality regulators to this frame's initial right-handed cell.
// The caller must separately enforce the raw9 driver/layout restrictions.
std::vector<regulator> bind_lattice_constraints(
    const LatticeConstraintDeclarations &declarations,
    const Eigen::Matrix3d &initial_cell);

} // namespace Constraints
