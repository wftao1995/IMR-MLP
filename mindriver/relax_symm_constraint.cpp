// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "relax_new.h"
#include "relax_symmetry_selection.h"
#include <charconv>
#include <cctype>
#include <deque>
#include <numeric>
#include <set>
#include <Eigen/SVD>
#include <map>
#include <stdexcept>

namespace {
Eigen::MatrixXd kernel(const Eigen::MatrixXd &c, double tol = 1e-8) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(c, Eigen::ComputeFullV);
    const double scale = c.size() ? std::max(1.0, c.cwiseAbs().maxCoeff()) : 1.0;
    int rank = 0;
    for (double value : svd.singularValues()) if (value > tol * scale) ++rank;
    return svd.matrixV().rightCols(c.cols() - rank);
}
Eigen::Matrix3d voigt_matrix(int component) {
    Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
    int current = 0;
    for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b, ++current)
            if (current == component) result(a, b) = result(b, a) = 1.0;
    return result;
}
}

namespace RelaxSymmetry {
namespace {
std::vector<int> indices(const std::string &text, int count, const char *name) {
    if (text.find_first_not_of(" \t\r\n") == std::string::npos) return {};
    std::set<int> result;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const size_t end = comma == std::string::npos ? text.size() : comma;
        size_t first = begin, last = end;
        while (first < last && std::isspace(static_cast<unsigned char>(text[first]))) ++first;
        while (last > first && std::isspace(static_cast<unsigned char>(text[last-1]))) --last;
        int value = -1;
        const auto parsed = std::from_chars(text.data()+first, text.data()+last, value);
        if (first == last || parsed.ec != std::errc{} || parsed.ptr != text.data()+last || value < 0 || value >= count)
            throw std::invalid_argument(std::string(name)+" requires comma-separated in-range nonnegative integer indices");
        result.insert(value);
        if (comma == std::string::npos) break;
        begin = comma+1;
    }
    return {result.begin(), result.end()};
}

int identity(const Symmetry3D::symdataset &data) {
    if (data.n_operations <= 0 || data.n_atoms < 0 || data.symm_ops.size() != data.n_operations ||
        data.symop_mapping.rows() != data.n_operations || data.symop_mapping.cols() != data.n_atoms)
        throw std::invalid_argument("Initial symmetry dataset has invalid operation/mapping dimensions");
    int result = -1;
    for (int op = 0; op < data.n_operations; ++op) {
        const auto &[rotation, translation] = data.symm_ops[op];
        if (!rotation.allFinite() || !translation.allFinite())
            throw std::invalid_argument("Initial symmetry dataset has non-finite operations");
        std::vector<bool> used(data.n_atoms, false);
        bool maps_identity = true;
        for (int atom = 0; atom < data.n_atoms; ++atom) {
            const int target = data.symop_mapping(op, atom);
            if (target < 0 || target >= data.n_atoms || used[target])
                throw std::invalid_argument("Initial symmetry operation mapping is not a permutation");
            used[target] = true;
            maps_identity = maps_identity && target == atom;
        }
        const auto shift = (translation.array()-translation.array().round()).matrix();
        if (rotation.isApprox(Eigen::Matrix3d::Identity(), 1e-10) && shift.norm() < 1e-8 && maps_identity)
            result = op;
    }
    if (result < 0) throw std::invalid_argument("Initial symmetry dataset has no identity operation");
    return result;
}

int product(const Symmetry3D::symdataset &data, int a, int b) {
    const Eigen::Matrix3d rotation = data.symm_ops[a].first*data.symm_ops[b].first;
    const Eigen::RowVector3d translation = data.symm_ops[a].second*data.symm_ops[b].first+data.symm_ops[b].second;
    for (int op = 0; op < data.n_operations; ++op) {
        if (!rotation.isApprox(data.symm_ops[op].first, 1e-10)) continue;
        const Eigen::RowVector3d shift = translation-data.symm_ops[op].second;
        if ((shift.array()-shift.array().round()).matrix().norm() > 1e-8) continue;
        bool matches = true;
        for (int atom = 0; atom < data.n_atoms; ++atom)
            if (data.symop_mapping(op, atom) != data.symop_mapping(b, data.symop_mapping(a, atom))) { matches = false; break; }
        if (matches) return op;
    }
    throw std::invalid_argument("Requested operations do not close within the initial mapped symmetry dataset");
}

std::vector<int> closure(const Symmetry3D::symdataset &data, std::vector<int> generators, int unit) {
    if (generators.size() == data.n_operations) return generators;
    std::set<int> known{unit};
    std::deque<int> pending{unit};
    while (!pending.empty()) {
        const int current = pending.front(); pending.pop_front();
        for (int generator : generators) {
            const int next = product(data, current, generator);
            if (known.insert(next).second) pending.push_back(next);
        }
    }
    return {known.begin(), known.end()};
}

Symmetry3D::symdataset subset(const Symmetry3D::symdataset &data,
                            const std::vector<int> &operations, bool independent_sites) {
    // This is a bound subgroup for the coordinate filter, not a fresh spglib
    // classification; do not carry the larger group's Wyckoff/SG labels.
    Symmetry3D::symdataset result{};
    result.n_atoms = data.n_atoms;
    result.n_operations = operations.size();
    result.symop_mapping.resize(result.n_operations, data.n_atoms);
    result.equivalent_atoms.resize(data.n_atoms);
    std::iota(result.equivalent_atoms.begin(), result.equivalent_atoms.end(), 0);
    const auto root = [&](int atom) {
        while (result.equivalent_atoms[atom] != atom) atom = result.equivalent_atoms[atom];
        return atom;
    };
    for (int row = 0; row < operations.size(); ++row) {
        const int op = operations[row];
        result.symm_ops.push_back(data.symm_ops[op]);
        result.symop_mapping.row(row) = data.symop_mapping.row(op);
        if (!independent_sites)
            for (int atom = 0; atom < data.n_atoms; ++atom) {
                const int a = root(atom), b = root(data.symop_mapping(op, atom));
                result.equivalent_atoms[std::max(a,b)] = std::min(a,b);
            }
    }
    for (int atom = 0; atom < data.n_atoms; ++atom) result.equivalent_atoms[atom] = root(atom);
    return result;
}
}

InitialSelection select_initial(const Symmetry3D::symdataset &data,
                                const std::string &operation_text,
                                const std::string &site_text) {
    const int unit = identity(data);
    InitialSelection result;
    auto requested = indices(operation_text, data.n_operations, "symmetry_operations");
    if (requested.empty()) {
        requested.resize(data.n_operations);
        std::iota(requested.begin(), requested.end(), 0);
    }
    result.operations = closure(data, requested, unit);
    result.sites = indices(site_text, data.n_atoms, "symmetry_sites");
    result.site_only = !result.sites.empty();
    result.positions = subset(data, result.operations, result.site_only);
    if (!result.site_only) result.cell_operations = result.operations;
    else {
        std::vector<int> stabilizers;
        for (int op : result.operations)
            for (int atom : result.sites)
                if (data.symop_mapping(op, atom) == atom) { stabilizers.push_back(op); break; }
        result.cell_operations = closure(data, stabilizers, unit);
    }
    result.cell = subset(data, result.cell_operations, false);
    return result;
}
}

void FilterSymmCoord::setup_cell_constraints(
    const Symmetry3D::symdataset &dataset, const Eigen::Matrix3d &reference_cell) {
    Eigen::MatrixXd constraints(9 * dataset.n_operations, 6);
    for (int component = 0; component < 6; ++component) {
        const Eigen::Matrix3d dG = reference_cell * voigt_matrix(component) *
            reference_cell.transpose();
        for (int op = 0; op < dataset.n_operations; ++op) {
            const Eigen::Matrix3d residual = dataset.symm_ops[op].first * dG *
                dataset.symm_ops[op].first.transpose() - dG;
            constraints.block(9 * op, component, 9, 1) =
                Eigen::Map<const Eigen::VectorXd>(residual.data(), 9);
        }
    }
    cell_basis = kernel(constraints);
    n_red_cell = static_cast<int>(cell_basis.cols());
}

void FilterSymmCoord::setup_position_orbits(
    RelaxationDrvV2 *pdrv, const Symmetry3D::symdataset &dataset) {
    std::map<int, std::vector<int>> groups;
    for (int atom = 0; atom < dataset.n_atoms; ++atom) {
        const int rep = dataset.equivalent_atoms[atom];
        if (rep < 0 || rep >= dataset.n_atoms)
            throw std::runtime_error("Symmetry dataset contains an invalid atom mapping");
        if (pdrv->atom.chemmap_[rep] != pdrv->atom.chemmap_[atom])
            throw std::runtime_error("Symmetry mapping joins atoms of different elements");
        groups[rep].push_back(atom);
    }
    const bool recast = pdrv->flags.atomic_force_recast;
    const auto &cell = pdrv->refcell;
    const auto &inverse = pdrv->ivrefcell;
    std::vector<Eigen::RowVector3d> fractional;
    pdrv->atom.get_dir_pos(fractional);
    orbits.clear();
    for (const auto &[unused, atoms] : groups) {
        Orbit orbit;
        orbit.iat_rep = atoms.front();
        orbit.iat_orbit = atoms;
        if (site_only_ && !std::binary_search(selected_sites_.begin(), selected_sites_.end(), orbit.iat_rep)) {
            orbit.basis = Eigen::Matrix3d::Identity();
            orbit.maps.push_back(Eigen::Matrix3d::Identity());
            orbit.origins.push_back((recast ? fractional[orbit.iat_rep]*cell : fractional[orbit.iat_rep]).transpose());
            orbit.n_free = 3;
            orbits.push_back(std::move(orbit));
            continue;
        }
        Eigen::MatrixXd site = Eigen::MatrixXd::Zero(3 * dataset.n_operations, 3);
        for (int op = 0; op < dataset.n_operations; ++op) {
            if (dataset.symop_mapping(op, orbit.iat_rep) != orbit.iat_rep) continue;
            Eigen::Matrix3d map;
            if (recast) map = (inverse * dataset.symm_ops[op].first * cell).transpose();
            else map = dataset.symm_ops[op].first.transpose();
            site.block<3, 3>(3 * op, 0) = map - Eigen::Matrix3d::Identity();
        }
        orbit.basis = kernel(site);
        orbit.n_free = static_cast<int>(orbit.basis.cols());
        // Project the representative onto the affine fixed set of its site
        // group.  The displacement kernel above alone cannot remove a small
        // symmetry error already present in the input.
        Eigen::MatrixXd affine(3 * dataset.n_operations, 3);
        Eigen::VectorXd rhs(3 * dataset.n_operations);
        affine.setZero(); rhs.setZero();
        const Eigen::Vector3d rep0 = fractional[orbit.iat_rep].transpose();
        for (int op = 0; op < dataset.n_operations; ++op) {
            if (dataset.symop_mapping(op, orbit.iat_rep) != orbit.iat_rep) continue;
            const auto &W = dataset.symm_ops[op].first;
            const auto &t = dataset.symm_ops[op].second;
            affine.block<3, 3>(3 * op, 0) = W.transpose() - Eigen::Matrix3d::Identity();
            const Eigen::Vector3d image_shift = operation_images_[op].row(orbit.iat_rep).transpose();
            rhs.segment<3>(3 * op) = image_shift - t.transpose();
        }
        const Eigen::Vector3d exact_rep = rep0 + affine.completeOrthogonalDecomposition().solve(rhs - affine * rep0);
        if (((exact_rep - rep0).transpose() * cell).norm() > pdrv->tol.symprec)
            throw std::runtime_error("Exact site-symmetry correction exceeds symprec");
        for (int target : atoms) {
            int found = -1;
            for (int op = 0; op < dataset.n_operations; ++op)
                if (dataset.symop_mapping(op, orbit.iat_rep) == target) { found = op; break; }
            if (found < 0) throw std::runtime_error("Symmetry operation for atom orbit is missing");
            const auto &W = dataset.symm_ops[found].first;
            Eigen::Matrix3d map;
            if (recast) map = (inverse * W * cell).transpose();
            else map = W.transpose();
            orbit.maps.push_back(map * orbit.basis);
            Eigen::RowVector3d exact = exact_rep.transpose()*W+dataset.symm_ops[found].second
                -operation_images_[found].row(orbit.iat_rep);
            const double correction = ((exact - fractional[target]) * cell).norm();
            if (!std::isfinite(correction) || correction > pdrv->tol.symprec)
                throw std::runtime_error("Exact symmetry correction exceeds symprec");
            pdrv->symmetry_atom_correction = std::max(pdrv->symmetry_atom_correction, correction);
            if (!pdrv->flags.relax_atom && correction > 1e-10)
                throw std::runtime_error("Exact symmetry requires moving frozen atoms");
            orbit.origins.push_back((recast ? exact * cell : exact).transpose());
        }
        orbits.push_back(std::move(orbit));
    }
}

void FilterSymmCoord::init(RelaxationDrvV2 *pdrv, void *ptr) {
    const double symprec = ptr ? *reinterpret_cast<double *>(ptr) : pdrv->tol.symprec;
    if (!selection_bound_) {
        const auto initial = Symmetry3D::get_sym_dataset(pdrv->atom, symprec, true);
        const auto selected = RelaxSymmetry::select_initial(initial, pdrv->control.symmetry_operations, pdrv->control.symmetry_sites);
        position_group_ = selected.positions;
        cell_group_ = selected.cell;
        selected_sites_ = selected.sites;
        site_only_ = selected.site_only;
        bound_operations_ = pdrv->control.symmetry_operations;
        bound_sites_ = pdrv->control.symmetry_sites;
        bound_species_ = pdrv->atom.chemmap_;
        bound_recast_ = pdrv->flags.atomic_force_recast;
        pdrv->atom.get_dir_pos(initial_fractional_);
        operation_images_.resize(position_group_.n_operations);
        for (int op = 0; op < position_group_.n_operations; ++op) {
            operation_images_[op].resize(position_group_.n_atoms, 3);
            for (int atom = 0; atom < position_group_.n_atoms; ++atom) {
                const int target = position_group_.symop_mapping(op, atom);
                if (bound_species_[target] != bound_species_[atom])
                    throw std::invalid_argument("Selected initial symmetry joins atoms of different species");
                operation_images_[op].row(atom) = (initial_fractional_[atom]*position_group_.symm_ops[op].first
                    +position_group_.symm_ops[op].second-initial_fractional_[target]).array().round();
            }
        }
    } else if (pdrv->control.symmetry_operations != bound_operations_ || pdrv->control.symmetry_sites != bound_sites_ ||
               pdrv->atom.chemmap_ != bound_species_ || pdrv->flags.atomic_force_recast != bound_recast_) {
        throw std::invalid_argument("Initial symmetry selection/species/coordinate mode cannot change on filter restart; initialize a new driver");
    }
    const auto &dataset = position_group_;
    pdrv->symmetry_atom_correction = pdrv->symmetry_cell_correction = 0;
    const Eigen::Matrix3d old_cell = pdrv->atom.cell_;
    std::vector<Eigen::RowVector3d> fractional;
    pdrv->atom.get_dir_pos(fractional);
    if (selection_bound_) {
        auto reference = initial_fractional_;
        // EFS wraps atom positions, while the accepted reduced checkpoint keeps
        // the unwrapped branch. Reconstruct that branch using the previous
        // basis before rebasing to the current accepted cell.
        if (pdrv->checkpoint_x.size() == n_red_cell+n_red_pos) {
            int offset = n_red_cell;
            for (const auto &orbit : orbits) {
                const Eigen::VectorXd q = pdrv->checkpoint_x.segment(offset, orbit.n_free);
                for (size_t slot = 0; slot < orbit.iat_orbit.size(); ++slot) {
                    Eigen::RowVector3d position = (orbit.origins[slot]+orbit.maps[slot]*q).transpose();
                    if (bound_recast_) position *= coordinate_reference_cell_.inverse();
                    reference[orbit.iat_orbit[slot]] = position;
                }
                offset += orbit.n_free;
            }
        }
        for (int atom = 0; atom < fractional.size(); ++atom)
            fractional[atom] += (reference[atom]-fractional[atom]).array().round().matrix();
    }
    const Eigen::Matrix3d metric = old_cell * old_cell.transpose();
    Eigen::Matrix3d projected = Eigen::Matrix3d::Zero();
    for (const auto &op : cell_group_.symm_ops)
        projected += op.first*metric*op.first.transpose();
    projected /= cell_group_.n_operations;
    Eigen::LLT<Eigen::Matrix3d> chol(projected);
    if (chol.info() != Eigen::Success) throw std::runtime_error("Projected symmetry metric is not positive definite");
    Eigen::Matrix3d root = chol.matrixL();
    Eigen::JacobiSVD<Eigen::Matrix3d> align(root.transpose() * old_cell, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d exact_cell = root * align.matrixU() * align.matrixV().transpose();
    for (int d = 0; d < 3; ++d)
        pdrv->symmetry_cell_correction = std::max(pdrv->symmetry_cell_correction, (exact_cell.row(d) - old_cell.row(d)).norm());
    if (pdrv->symmetry_cell_correction > symprec)
        throw std::runtime_error("Exact lattice correction exceeds symprec");
    if (!pdrv->flags.relax_cell && pdrv->symmetry_cell_correction > 1e-10)
        throw std::runtime_error("Exact symmetry requires changing frozen cell");
    if (pdrv->flags.relax_cell) pdrv->atom.cell_ = exact_cell;
    for (int i = 0; i < pdrv->atom.size(); ++i) pdrv->atom.pos_[i] = fractional[i] * pdrv->atom.cell_;
    pdrv->refcell = pdrv->atom.cell_;
    pdrv->ivrefcell = pdrv->refcell.inverse();
    pdrv->vol0 = pdrv->atom.get_volume();
    pdrv->LatPos2X();
    n_full_cell = pdrv->xposoffset;
    n_full_pos = pdrv->flags.relax_atom ? 3 * pdrv->atom.size() : 0;
    xposoffset_full = pdrv->xposoffset;
    if (pdrv->flags.relax_cell) setup_cell_constraints(cell_group_, pdrv->refcell);
    else { cell_basis.resize(n_full_cell, 0); n_red_cell = 0; }
    setup_position_orbits(pdrv, dataset);
    if (!pdrv->flags.relax_atom) orbits.clear();
    n_red_pos = 0;
    for (const auto &orbit : orbits) {
        n_red_pos += orbit.n_free;
        for (size_t slot = 0; slot < orbit.iat_orbit.size(); ++slot)
            pdrv->x.segment<3>(xposoffset_full + 3 * orbit.iat_orbit[slot]) = orbit.origins[slot];
    }
    pdrv->X2LatPos();
    coordinate_reference_cell_ = pdrv->refcell;
    selection_bound_ = true;
}

void FilterSymmCoord::forward(RelaxationDrvV2 *pdrv) {
    x_save = pdrv->x;
    const Eigen::VectorXd full_gradient = pdrv->g;
    Eigen::VectorXd rx = Eigen::VectorXd::Zero(n_red_cell + n_red_pos);
    Eigen::VectorXd rg = Eigen::VectorXd::Zero(n_red_cell + n_red_pos);
    if (n_red_cell) {
        rx.head(n_red_cell) = cell_basis.transpose() * x_save.head(n_full_cell);
        rg.head(n_red_cell) = cell_basis.transpose() * full_gradient.head(n_full_cell);
    }
    int offset = n_red_cell;
    for (const auto &orbit : orbits) {
        const int rep_base = xposoffset_full + 3 * orbit.iat_rep;
        rx.segment(offset, orbit.n_free) = orbit.basis.transpose() *
            (x_save.segment<3>(rep_base) - orbit.origins.front());
        for (size_t slot = 0; slot < orbit.iat_orbit.size(); ++slot) {
            const int base = xposoffset_full + 3 * orbit.iat_orbit[slot];
            rg.segment(offset, orbit.n_free) += orbit.maps[slot].transpose() *
                full_gradient.segment<3>(base);
        }
        offset += orbit.n_free;
    }
    pdrv->x = std::move(rx); pdrv->g = std::move(rg);
}

void FilterSymmCoord::reverse(RelaxationDrvV2 *pdrv) {
    const Eigen::VectorXd reduced = pdrv->x;
    Eigen::VectorXd full = Eigen::VectorXd::Zero(xposoffset_full + n_full_pos);
    if (n_red_cell) full.head(n_full_cell) = cell_basis * reduced.head(n_red_cell);
    int offset = n_red_cell;
    for (const auto &orbit : orbits) {
        const Eigen::VectorXd q = reduced.segment(offset, orbit.n_free);
        for (size_t slot = 0; slot < orbit.iat_orbit.size(); ++slot) {
            const int base = xposoffset_full + 3 * orbit.iat_orbit[slot];
            full.segment<3>(base) = orbit.origins[slot] + orbit.maps[slot] * q;
        }
        offset += orbit.n_free;
    }
    pdrv->x = std::move(full); pdrv->g.resize(pdrv->x.size());
}

void FilterSymmCoord::forward_scaling(RelaxationDrvV2 *pdrv) {
    pdrv->prec_scaling = Eigen::VectorXd::Ones(n_red_cell + n_red_pos);
}
