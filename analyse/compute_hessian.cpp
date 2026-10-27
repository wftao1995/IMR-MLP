// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_hessian.h"

#include "base/neighborhood.h"
#include "external/spglib_driver.h"
#include "tools/hessian_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVec3 = Eigen::RowVector3d;

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

Atoms cartesian_copy(const Atoms &atom)
{
    Atoms result = atom;
    if (!result.cartesian)
        result.to_cartesian();
    result.cartesian = true;
    return result;
}

void validate_common(const Atoms &atom, const Potentials::AnyPotential *pot,
                     double nudgepos, double nudgecell, double nudgefrac,
                     bool request_cell, bool fractional_pos)
{
    if (pot == nullptr)
        GPERROR("compute_hessian requires a non-null potential");
    if (!std::isfinite(nudgepos) || nudgepos <= 0.0)
        GPERROR("nudge_pos must be finite and positive");
    if (!std::isfinite(nudgecell) || nudgecell <= 0.0)
        GPERROR("nudge_cell must be finite and positive");
    if (!std::isfinite(nudgefrac) || nudgefrac <= 0.0)
        GPERROR("nudge_frac must be finite and positive");

    if (request_cell || fractional_pos || !atom.cartesian) {
        const double volume = atom.cell_.determinant();
        if (!atom.cell_.allFinite() || !std::isfinite(volume) ||
            std::abs(volume) <= 1.0e-14)
            GPERROR("cell Hessian and fractional coordinates require a nonsingular cell");
    }
}

double effective_cutoff(const ComputeHessian &settings,
                        const Potentials::AnyPotential &pot)
{
    // A positive user cutoff overrides the potential cutoff.  Very small
    // values are treated as the historical "no pair filter" sentinel because
    // Neighborhoods uses the cutoff as a spatial-bin size in some builds.
    return settings.get_cutoff() > 1.0e-5 ? settings.get_cutoff()
                                         : pot.cutoff;
}

double structure_volume(const Atoms &atom)
{
    return std::abs(atom.cell_.determinant());
}

double evaluate_energy(const Atoms &candidate,
                       Potentials::AnyPotential *pot,
                       double cutoff,
                       ComputeHessian::HessianObjective objective,
                       double pressure)
{
    if (candidate.size() == 0) {
        // Empty structures are useful as a cheap edge case for the atomic
        // interface.  There is no atom energy to differentiate.
        return objective == ComputeHessian::HessianObjective::Enthalpy
                   ? pressure * structure_volume(candidate)
                   : 0.0;
    }

    Atoms work = candidate;
    Neighborhoods nbhs(work, cutoff);
    pot->getEOMP(work, nbhs);
    double value = work.energy_;
    if (objective == ComputeHessian::HessianObjective::Enthalpy)
        value += pressure * structure_volume(work);
    return value;
}

Matrix evaluate_forces(const Atoms &candidate,
                       Potentials::AnyPotential *pot,
                       double cutoff)
{
    Atoms work = candidate;
    Neighborhoods nbhs(work, cutoff);
    pot->getEFSOMP(work, nbhs);
    if (static_cast<int>(work.forces_.size()) != work.size())
        GPERROR("potential returned an invalid force array while computing a Hessian");

    Matrix forces(work.size(), 3);
    for (int i = 0; i < work.size(); ++i)
        forces.row(i) = work.forces_[i].transpose();
    return forces;
}

Eigen::Matrix3d strain_from_voigt(const Vector &q, bool engineering_shear)
{
    Eigen::Matrix3d strain = Eigen::Matrix3d::Zero();
    strain(0, 0) = q(0);
    strain(1, 1) = q(1);
    strain(2, 2) = q(2);
    const double shear_scale = engineering_shear ? 0.5 : 1.0;
    strain(1, 2) = strain(2, 1) = shear_scale * q(3);
    strain(0, 2) = strain(2, 0) = shear_scale * q(4);
    strain(0, 1) = strain(1, 0) = shear_scale * q(5);
    return strain;
}

Eigen::Matrix<double, 9, 6> raw_cell_jacobian(const Eigen::Matrix3d &cell,
                                               bool engineering_shear)
{
    // The rows are row-major cell components C_ab.  We use
    // C' = C (I + epsilon), which remains correct for a general triclinic
    // cell.  The matrix below is d vec(C') / d voigt(epsilon).
    Eigen::Matrix<double, 9, 6> jac = Eigen::Matrix<double, 9, 6>::Zero();
    const double shear_scale = engineering_shear ? 0.5 : 1.0;
    for (int a = 0; a < 3; ++a) {
        jac(3 * a + 0, 0) = cell(a, 0);
        jac(3 * a + 1, 1) = cell(a, 1);
        jac(3 * a + 2, 2) = cell(a, 2);
        jac(3 * a + 2, 3) = shear_scale * cell(a, 1); // epsilon_yz
        jac(3 * a + 1, 3) = shear_scale * cell(a, 2);
        jac(3 * a + 2, 4) = shear_scale * cell(a, 0); // epsilon_xz
        jac(3 * a + 0, 4) = shear_scale * cell(a, 2);
        jac(3 * a + 1, 5) = shear_scale * cell(a, 0); // epsilon_xy
        jac(3 * a + 0, 5) = shear_scale * cell(a, 1);
    }
    return jac;
}

Vector coordinate_steps(int cell_dof, int atom_dof,
                        double nudgecell, double nudgepos,
                        double nudgefrac)
{
    Vector steps(cell_dof + atom_dof);
    steps.setConstant(nudgepos);
    for (int i = 0; i < cell_dof; ++i)
        steps(i) = nudgecell;
    // The caller changes the atomic part to nudge_frac when fractional
    // coordinates are in use.
    if (nudgefrac != nudgepos)
        for (int i = cell_dof; i < steps.size(); ++i)
            steps(i) = nudgefrac;
    return steps;
}

Atoms perturbed_structure(const Atoms &reference, const Vector &q,
                          int cell_dof, bool request_fc, bool cell_strain,
                          bool fractional_pos, bool engineering_shear)
{
    Atoms candidate = reference;
    const int atom_dof = request_fc ? 3 * reference.size() : 0;
    const int expected = cell_dof + atom_dof;
    if (q.size() != expected)
        GPERROR("internal Hessian coordinate dimension mismatch");

    if (cell_dof > 0) {
        if (cell_strain) {
            candidate.cell_ = reference.cell_ *
                              (Eigen::Matrix3d::Identity() +
                               strain_from_voigt(q.head<6>(), engineering_shear));
        } else {
            candidate.cell_ = reference.cell_;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    candidate.cell_(a, b) += q(3 * a + b);
        }
    }

    if (fractional_pos) {
        const Eigen::Matrix3d invcell = reference.cell_.inverse();
        for (int i = 0; i < reference.size(); ++i) {
            RowVec3 frac = reference.pos_[i] * invcell;
            if (request_fc)
                frac += q.segment<3>(cell_dof + 3 * i).transpose();
            candidate.pos_[i] = frac * candidate.cell_;
        }
    } else {
        candidate.pos_ = reference.pos_;
        if (request_fc)
            for (int i = 0; i < reference.size(); ++i)
                candidate.pos_[i] += q.segment<3>(cell_dof + 3 * i).transpose();
    }
    candidate.cartesian = true;
    return candidate;
}

Matrix energy_fd(const Atoms &reference, Potentials::AnyPotential *pot,
                 int cell_dof, bool request_fc, bool cell_strain, bool fractional_pos,
                 bool engineering_shear, double nudgecell, double nudgepos,
                 double nudgefrac, double cutoff,
                 ComputeHessian::HessianObjective objective, double pressure)
{
    const int atom_dof = request_fc ? 3 * reference.size() : 0;
    const int ndim = cell_dof + atom_dof;
    Matrix hessian(ndim, ndim);
    hessian.setZero();
    if (ndim == 0)
        return hessian;

    const Vector steps = coordinate_steps(cell_dof, atom_dof, nudgecell,
                                          nudgepos,
                                          fractional_pos ? nudgefrac : nudgepos);
    Vector zero = Vector::Zero(ndim);
    const double e0 = evaluate_energy(perturbed_structure(
                                          reference, zero, cell_dof,
                                          request_fc, cell_strain, fractional_pos,
                                          engineering_shear),
                                      pot, cutoff, objective, pressure);

    for (int i = 0; i < ndim; ++i) {
        Vector qp = zero;
        Vector qm = zero;
        qp(i) = steps(i);
        qm(i) = -steps(i);
        const double ep = evaluate_energy(
            perturbed_structure(reference, qp, cell_dof, request_fc, cell_strain,
                                fractional_pos, engineering_shear),
            pot, cutoff, objective, pressure);
        const double em = evaluate_energy(
            perturbed_structure(reference, qm, cell_dof, request_fc, cell_strain,
                                fractional_pos, engineering_shear),
            pot, cutoff, objective, pressure);
        hessian(i, i) = (ep + em - 2.0 * e0) / (steps(i) * steps(i));
    }

    for (int i = 0; i < ndim; ++i) {
        for (int j = i + 1; j < ndim; ++j) {
            Vector qpp = zero;
            Vector qpm = zero;
            Vector qmp = zero;
            Vector qmm = zero;
            qpp(i) = steps(i);
            qpp(j) = steps(j);
            qpm(i) = steps(i);
            qpm(j) = -steps(j);
            qmp(i) = -steps(i);
            qmp(j) = steps(j);
            qmm(i) = -steps(i);
            qmm(j) = -steps(j);

            const double epp = evaluate_energy(
                perturbed_structure(reference, qpp, cell_dof, request_fc, cell_strain,
                                    fractional_pos, engineering_shear),
                pot, cutoff, objective, pressure);
            const double epm = evaluate_energy(
                perturbed_structure(reference, qpm, cell_dof, request_fc, cell_strain,
                                    fractional_pos, engineering_shear),
                pot, cutoff, objective, pressure);
            const double emp = evaluate_energy(
                perturbed_structure(reference, qmp, cell_dof, request_fc, cell_strain,
                                    fractional_pos, engineering_shear),
                pot, cutoff, objective, pressure);
            const double emm = evaluate_energy(
                perturbed_structure(reference, qmm, cell_dof, request_fc, cell_strain,
                                    fractional_pos, engineering_shear),
                pot, cutoff, objective, pressure);
            hessian(i, j) = (epp + emm - epm - emp) /
                            (4.0 * steps(i) * steps(j));
            hessian(j, i) = hessian(i, j);
        }
    }
    return 0.5 * (hessian + hessian.transpose());
}

Matrix force_fd(const Atoms &reference, Potentials::AnyPotential *pot,
                double nudgepos, double cutoff)
{
    const int ndim = 3 * reference.size();
    Matrix hessian(ndim, ndim);
    hessian.setZero();
    if (ndim == 0)
        return hessian;

    Vector zero = Vector::Zero(ndim);
    const double inv2d = 1.0 / (2.0 * nudgepos);
    for (int column = 0; column < ndim; ++column) {
        Vector qp = zero;
        Vector qm = zero;
        qp(column) = nudgepos;
        qm(column) = -nudgepos;
        const Matrix fp = evaluate_forces(
            perturbed_structure(reference, qp, 0, true, false, false, false),
            pot, cutoff);
        const Matrix fm = evaluate_forces(
            perturbed_structure(reference, qm, 0, true, false, false, false),
            pot, cutoff);
        for (int row = 0; row < ndim; ++row) {
            const int atom = row / 3;
            const int component = row % 3;
            hessian(row, column) = -(fp(atom, component) -
                                     fm(atom, component)) * inv2d;
        }
    }
    return 0.5 * (hessian + hessian.transpose());
}

bool valid_cell_for_symmetry(const Atoms &atom)
{
    const double det = atom.cell_.determinant();
    // These operations describe a 3D periodic crystal, not a molecule or slab.
    return atom.is_fully_periodic() && atom.cell_.allFinite() && std::isfinite(det) &&
           std::abs(det) > 1.0e-14 && atom.size() > 0 &&
           atom.chemmap_.size() == static_cast<size_t>(atom.size());
}

bool map_symmetry_operation(const Atoms &atom,
                            const Symmetry3D::symopt_type &operation,
                            std::vector<int> &mapping,
                            double symprec)
{
    std::vector<RowVec3> fractional;
    atom.get_dir_pos(fractional);
    const Eigen::Matrix3d &rotation = operation.first;
    const RowVec3 &translation = operation.second;
    mapping.assign(atom.size(), -1);

    for (int i = 0; i < atom.size(); ++i) {
        RowVec3 transformed = fractional[i] * rotation + translation;
        if (atom.is_fully_periodic()) {
            for (int d = 0; d < 3; ++d)
                transformed(d) -= std::floor(transformed(d));
        }

        double best = std::numeric_limits<double>::infinity();
        int best_atom = -1;
        for (int j = 0; j < atom.size(); ++j) {
            if (atom.chemmap_[j] != atom.chemmap_[i])
                continue;
            RowVec3 delta = transformed - fractional[j];
            if (atom.is_fully_periodic())
                for (int d = 0; d < 3; ++d)
                    delta(d) -= std::round(delta(d));
            const double distance = (delta * atom.cell_).norm();
            if (distance < best) {
                best = distance;
                best_atom = j;
            }
        }
        if (best_atom < 0 || best > symprec)
            return false;
        mapping[i] = best_atom;
    }
    return true;
}

Eigen::Matrix3d cartesian_rotation(const Atoms &atom,
                                   const Symmetry3D::symopt_type &operation)
{
    return atom.cell_.inverse() * operation.first * atom.cell_;
}

Eigen::Matrix<double, 6, 1> voigt_from_strain(const Eigen::Matrix3d &strain,
                                               bool engineering_shear)
{
    Eigen::Matrix<double, 6, 1> result;
    result(0) = strain(0, 0);
    result(1) = strain(1, 1);
    result(2) = strain(2, 2);
    const double shear_scale = engineering_shear ? 2.0 : 1.0;
    result(3) = shear_scale * strain(1, 2);
    result(4) = shear_scale * strain(0, 2);
    result(5) = shear_scale * strain(0, 1);
    return result;
}

Eigen::Matrix<double, 6, 6> strain_representation(const Eigen::Matrix3d &rotation,
                                                    bool engineering_shear)
{
    Eigen::Matrix<double, 6, 6> result;
    result.setZero();
    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> basis =
            Eigen::Matrix<double, 6, 1>::Zero();
        basis(i) = 1.0;
        const Eigen::Matrix3d strain = strain_from_voigt(basis, engineering_shear);
        result.col(i) = voigt_from_strain(rotation.transpose() * strain * rotation,
                                          engineering_shear);
    }
    return result;
}

void apply_symmetry_projection(const Atoms &atom, Matrix &hessian,
                               int cell_dof, bool request_fc,
                               bool cell_strain, bool symm_strain,
                               bool use_symmetry, bool engineering_shear,
                               double symprec)
{
    if (!use_symmetry || !valid_cell_for_symmetry(atom) || hessian.rows() == 0)
        return;
    if (!request_fc && !(cell_strain && symm_strain))
        return;

    std::vector<Symmetry3D::symopt_type> operations;
    if (Symmetry3D::get_symmetry_operations(atom, operations, 192, symprec) <= 0)
        return;

    const int atom_dof = request_fc ? 3 * atom.size() : 0;
    const int ndim = cell_dof + atom_dof;
    Matrix projected = Matrix::Zero(ndim, ndim);
    int valid_operations = 0;

    for (const auto &operation : operations) {
        std::vector<int> mapping;
        if (!map_symmetry_operation(atom, operation, mapping, symprec))
            continue;

        Matrix transform = Matrix::Zero(ndim, ndim);
        if (cell_dof == 0) {
            // no cell coordinates
        } else if (cell_strain && symm_strain && cell_dof == 6) {
            transform.block<6, 6>(0, 0) = strain_representation(
                cartesian_rotation(atom, operation), engineering_shear);
        } else {
            // Raw cell components do not form a point-group tensor in a
            // fixed basis.  Leave this block untouched and only project the
            // atomic sector below.
            transform.block(cell_dof, cell_dof, atom_dof, atom_dof).setIdentity();
        }

        if (request_fc) {
            const Eigen::Matrix3d rotation = cartesian_rotation(atom, operation);
            for (int i = 0; i < atom.size(); ++i)
                transform.block<3, 3>(cell_dof + 3 * mapping[i],
                                      cell_dof + 3 * i) = rotation.transpose();
        }

        // For raw-cell output, retain the cell coordinates exactly.  For
        // strain output, transform both the strain and atomic coordinates.
        if (cell_dof > 0 && !(cell_strain && symm_strain && cell_dof == 6))
            transform.block(0, 0, cell_dof, cell_dof).setIdentity();

        projected += transform * hessian * transform.transpose();
        ++valid_operations;
    }

    if (valid_operations > 0)
        hessian = projected / static_cast<double>(valid_operations);
}

} // namespace

ComputeHessian::HessianMethod ComputeHessian::parse_method() const
{
    const std::string name = lower_copy(method_name);
    if (name.empty() || name == "auto")
        return HessianMethod::Auto;
    if (name == "analytic")
        return HessianMethod::Analytic;
    if (name == "force_fd")
        return HessianMethod::ForceFD;
    if (name == "energy_fd")
        return HessianMethod::EnergyFD;
    if (name == "hybrid")
        return HessianMethod::Hybrid;
    GPERROR("method must be auto, analytic, force_fd, energy_fd, or hybrid");
}

ComputeHessian::HessianObjective ComputeHessian::parse_objective() const
{
    const std::string name = lower_copy(objective_name);
    if (name.empty() || name == "energy")
        return HessianObjective::Energy;
    if (name == "enthalpy")
        return HessianObjective::Enthalpy;
    GPERROR("objective must be energy or enthalpy");
}

ComputeHessian::HessianMethod ComputeHessian::resolve_atomic_method(
    const Potentials::AnyPotential *pot, const Atoms &atom) const
{
    HessianMethod requested = method;
    if (requested == HessianMethod::Auto && !fractional_pos &&
        pot->supports_analytical_hessian_for(atom))
        return HessianMethod::Analytic;
    if (requested == HessianMethod::Auto)
        return fractional_pos ? HessianMethod::EnergyFD
                              : HessianMethod::ForceFD;
    return requested;
}

void ComputeHessian::compute_hvp(const Atoms &atom,
                                 const Eigen::VectorXd &direction,
                                 Eigen::VectorXd &product,
                                 Potentials::AnyPotential *pot)
{
    validate_common(atom, pot, nudgepos, nudgecell, nudgefrac, false, false);
    if (fractional_pos)
        GPERROR("Hessian-vector products use Cartesian directions");
    if (method != HessianMethod::Auto && method != HessianMethod::Analytic &&
        method != HessianMethod::ForceFD)
        GPERROR("Hessian-vector method must be auto, analytic, or force_fd");
    if (direction.size() != 3 * atom.size() || !direction.allFinite())
        GPERROR("Hessian-vector direction must be finite with size 3*N");
    const bool analytical = method == HessianMethod::Analytic ||
        (method == HessianMethod::Auto && pot->supports_analytical_hessian_vector_product_for(atom));
    if (analytical && !pot->supports_analytical_hessian_vector_product_for(atom))
        GPERROR("analytic Hessian-vector product is unavailable for this potential");
    double maximum_norm = 0.0;
    for (int i = 0; i < atom.size(); ++i)
        maximum_norm = std::max(maximum_norm, direction.segment<3>(3 * i).stableNorm());
    if (maximum_norm == 0.0) {
        product = Eigen::VectorXd::Zero(direction.size());
        return;
    }
    Atoms reference = cartesian_copy(atom);
    reference.resetEFS();
    Eigen::VectorXd result;
    if (analytical) {
        Neighborhoods nbhs(reference, effective_cutoff(*this, *pot));
        pot->applyHessianVectorProduct(reference, nbhs, direction, result);
    } else {
        const double delta = nudgepos / maximum_norm;
        if (!std::isfinite(delta) || delta <= 0.0)
            GPERROR("Invalid Hessian-vector displacement scale");
        Atoms plus = reference, minus = reference;
        for (int i = 0; i < atom.size(); ++i) {
            const Eigen::RowVector3d displacement = delta * direction.segment<3>(3 * i).transpose();
            plus.pos_[i] += displacement;
            minus.pos_[i] -= displacement;
        }
        const double cutoff_this = effective_cutoff(*this, *pot);
        const Matrix fp = evaluate_forces(plus, pot, cutoff_this);
        const Matrix fm = evaluate_forces(minus, pot, cutoff_this);
        result.resize(direction.size());
        for (int i = 0; i < atom.size(); ++i)
            result.segment<3>(3 * i) = -(fp.row(i) - fm.row(i)).transpose() / (2.0 * delta);
    }
    product = std::move(result);
}



void ComputeHessian::compute_fc(Atoms &atom, Eigen::MatrixXd &hessian,
                                Potentials::AnyPotential *pot)
{
    validate_common(atom, pot, nudgepos, nudgecell, nudgefrac, false,
                    fractional_pos);
    const Atoms reference = cartesian_copy(atom);
    const int atom_dof = 3 * reference.size();
    if (atom_dof == 0) {
        hessian.resize(0, 0);
        return;
    }

    const HessianMethod requested = resolve_atomic_method(pot, atom);

    const double cutoff_this = effective_cutoff(*this, *pot);
    if (requested == HessianMethod::Analytic) {
        if (fractional_pos || !pot->supports_analytical_hessian_for(atom))
            GPERROR("analytic atomic Hessian is unavailable for the requested coordinates");
        Atoms work = reference;
        Potentials::HessianMatrix blocks;
        if (cutoff_this != pot->cutoff || !reference.is_fully_periodic()) {
            Neighborhoods nbhs(work, cutoff_this);
            pot->getEFSHessian(work, nbhs, blocks);
        } else {
            pot->getEFSHessianSymmOMP(work, blocks, symprec);
        }
        HessianUtils::symmetrize_blocks(blocks);
        hessian = HessianUtils::to_dense(blocks);
        return;
    }

    if (requested == HessianMethod::ForceFD) {
        if (fractional_pos)
            GPERROR("force_fd is defined for Cartesian atomic coordinates; use energy_fd for fractional coordinates");
        hessian = force_fd(reference, pot, nudgepos, cutoff_this);
        return;
    }

    const Matrix energy_hessian = energy_fd(
        reference, pot, 0, true, false, fractional_pos, engineering_shear, nudgecell,
        nudgepos, nudgefrac, cutoff_this, objective, pressure);
    if (requested == HessianMethod::Hybrid && !fractional_pos) {
        const Matrix force_hessian = force_fd(reference, pot, nudgepos,
                                              cutoff_this);
        hessian = energy_hessian;
        for (int i = 0; i < atom_dof; ++i)
            hessian(i, i) = force_hessian(i, i);
        hessian = 0.5 * (hessian + hessian.transpose());
    } else {
        hessian = energy_hessian;
    }
}

void ComputeHessian::compute_fc_blocks(Atoms &atom,
                                       Potentials::HessianMatrix &hessian,
                                       Potentials::AnyPotential *pot)
{
    validate_common(atom, pot, nudgepos, nudgecell, nudgefrac, false,
                    fractional_pos);
    const Atoms reference = cartesian_copy(atom);
    if (reference.size() == 0) {
        hessian.resize(0);
        return;
    }

    const HessianMethod requested = resolve_atomic_method(pot, atom);

    if (requested == HessianMethod::Analytic) {
        if (fractional_pos || !pot->supports_analytical_hessian_for(atom))
            GPERROR("analytic atomic Hessian is unavailable for the requested coordinates");
        Atoms work = reference;
        const double cutoff_this = effective_cutoff(*this, *pot);
        if (cutoff_this != pot->cutoff || !reference.is_fully_periodic()) {
            Neighborhoods nbhs(work, cutoff_this);
            pot->getEFSHessian(work, nbhs, hessian);
        } else {
            pot->getEFSHessianSymmOMP(work, hessian, symprec);
        }
        HessianUtils::symmetrize_blocks(hessian);
        return;
    }

    Matrix dense;
    compute_fc(atom, dense, pot);
    HessianUtils::from_dense(dense, hessian);
}

void ComputeHessian::compute(Atoms &atom, Eigen::MatrixXd &hessian,
                             Potentials::AnyPotential *pot)
{
    validate_common(atom, pot, nudgepos, nudgecell, nudgefrac,
                    request_cell, fractional_pos);

    const Atoms reference = cartesian_copy(atom);
    const bool use_strain = request_cell && cell_strain;
    const int cell_dof = request_cell ? (use_strain ? 6 : 9) : 0;
    const int atom_dof = request_fc ? 3 * reference.size() : 0;
    const int ndim = cell_dof + atom_dof;
    hessian.resize(ndim, ndim);
    hessian.setZero();
    if (ndim == 0)
        return;

    HessianMethod requested = method;

    // The atomic-only path can use the same method selection as compute_fc.
    // The generalized cell path below is energy-based because its coordinate
    // set may contain cell/position response terms.
    if (!request_cell) {
        if (request_fc)
            compute_fc(atom, hessian, pot);
        apply_symmetry_projection(reference, hessian, 0, request_fc, false,
                                  false, use_symmetry, engineering_shear,
                                  symprec);
        hessian = 0.5 * (hessian + hessian.transpose());
        return;
    }

    const bool can_analytic_cell =
        request_cell && !fractional_pos && objective == HessianObjective::Energy &&
        pot->supports_analytical_cell_hessian_for(atom);

    if (request_cell && requested == HessianMethod::Analytic &&
        !can_analytic_cell)
        GPERROR("analytic cell Hessian is unavailable for the requested coordinates or objective");
    if (request_cell && requested == HessianMethod::ForceFD)
        GPERROR("force_fd supports atomic Hessians only; use auto, energy_fd, or hybrid for cell coordinates");

    if (can_analytic_cell &&
        (requested == HessianMethod::Auto || requested == HessianMethod::Analytic)) {
        Atoms work = reference;
        Potentials::HessianMatrix blocks;
        Eigen::MatrixXd raw_cell;
        Eigen::MatrixXd raw_cross;
        const double cutoff_this = effective_cutoff(*this, *pot);
        if (cutoff_this != pot->cutoff || !reference.is_fully_periodic()) {
            Neighborhoods nbhs(work, cutoff_this);
            pot->getEFSHessianCellOMP(work, nbhs, blocks, raw_cell, raw_cross);
        } else {
            pot->getEFSHessianCellSymmOMP(work, blocks, raw_cell, raw_cross,
                                        symprec);
        }

        Matrix atomic;
        HessianUtils::symmetrize_blocks(blocks);
        atomic = HessianUtils::to_dense(blocks);
        const int cartesian_dof = 3 * reference.size();
        if (atomic.rows() != cartesian_dof || atomic.cols() != cartesian_dof ||
            raw_cell.rows() != 9 || raw_cell.cols() != 9 ||
            raw_cross.rows() != 9 || raw_cross.cols() != cartesian_dof)
            GPERROR("analytic cell Hessian returned invalid atomic, cell, or mixed blocks");

        // Native cell derivatives use affine cell motion plus Cartesian
        // residuals: dr = A dC + du, A(3*i+b,3*a+b) = fractional_i[a].
        // Our cell DOFs hold Cartesian positions fixed, so du = dr - A dC.
        // Apply J^T H J before the optional strain mapping; A is dimensionless,
        // all three raw Hessian blocks have units of energy / length^2.
        Matrix affine = Matrix::Zero(cartesian_dof, 9);
        const Eigen::Matrix3d invcell = reference.cell_.inverse();
        for (int i = 0; i < reference.size(); ++i) {
            const RowVec3 fractional = reference.pos_[i] * invcell;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    affine(3 * i + b, 3 * a + b) = fractional(a);
        }
        const Matrix mixed_affine = raw_cross * affine;
        const Matrix atomic_affine = atomic * affine;
        raw_cell = (raw_cell - mixed_affine - mixed_affine.transpose()
                    + affine.transpose() * atomic_affine).eval();
        raw_cross = (raw_cross - atomic_affine.transpose()).eval();
        if (request_fc)
            hessian.block(cell_dof, cell_dof, atom_dof, atom_dof) = atomic;

        if (!use_strain) {
            hessian.block(0, 0, 9, 9) = raw_cell;
            if (request_fc && request_response_pos_cell) {
                hessian.block(0, cell_dof, 9, atom_dof) = raw_cross;
                hessian.block(cell_dof, 0, atom_dof, 9) = raw_cross.transpose();
            }
        } else {
            const Eigen::Matrix<double, 9, 6> jac = raw_cell_jacobian(
                reference.cell_, engineering_shear);
            hessian.block(0, 0, 6, 6) = jac.transpose() * raw_cell * jac;
            if (request_fc && request_response_pos_cell) {
                const Matrix cross = jac.transpose() * raw_cross;
                hessian.block(0, cell_dof, 6, atom_dof) = cross;
                hessian.block(cell_dof, 0, atom_dof, 6) = cross.transpose();
            }
        }
    } else {
        const double cutoff_this = effective_cutoff(*this, *pot);
        Matrix numerical = energy_fd(
            reference, pot, cell_dof, request_fc, use_strain, fractional_pos,
            engineering_shear, nudgecell, nudgepos, nudgefrac, cutoff_this,
            objective, pressure);

        if (requested == HessianMethod::Hybrid && request_fc &&
            !fractional_pos) {
            const Matrix force_hessian = force_fd(reference, pot, nudgepos,
                                                  cutoff_this);
            for (int i = 0; i < atom_dof; ++i)
                numerical(cell_dof + i, cell_dof + i) = force_hessian(i, i);
            numerical = 0.5 * (numerical + numerical.transpose());
        }
        hessian = numerical;
    }

    if (!request_response_pos_cell && cell_dof > 0 && atom_dof > 0) {
        hessian.block(0, cell_dof, cell_dof, atom_dof).setZero();
        hessian.block(cell_dof, 0, atom_dof, cell_dof).setZero();
    }

    apply_symmetry_projection(reference, hessian, cell_dof, request_fc,
                              use_strain, symm_strain, use_symmetry,
                              engineering_shear, symprec);
    hessian = 0.5 * (hessian + hessian.transpose());
}
