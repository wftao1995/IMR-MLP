// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "structures.h"
#include "neighborhoodKD.h"
#include "tools/tools.h"
#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <iterator>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>
#include <string>
#define ROUND_PREC 1e-6

namespace {

// The dual bounds every translation that can improve the current distance.
// Only periodic rows participate, so free cell directions need not span space.
double periodic_minimum_squared(const Eigen::RowVector3d &delta,
                                const Eigen::MatrixXd &basis,
                                const Eigen::MatrixXd &dual,
                                double best_sq, bool self,
                                Eigen::RowVector3d *best_vector = nullptr)
{
    const int rank = basis.rows();
    const Eigen::VectorXd projected = dual * delta.transpose();
    if (!self) {
        const Eigen::VectorXd nearest = projected.array().round().matrix();
        const Eigen::RowVector3d candidate = delta - nearest.transpose() * basis;
        const double candidate_sq = candidate.squaredNorm();
        if (candidate_sq < best_sq) {
            best_sq = candidate_sq;
            if (best_vector) *best_vector = candidate;
        }
        if (best_sq == 0) return 0;
    }
    const double radius = std::sqrt(best_sq);
    std::array<std::int64_t, 3> lower{}, upper{};
    for (int axis = 0; axis < rank; ++axis) {
        const double extent = radius * dual.row(axis).norm();
        const double padding = 32 * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(projected[axis]), extent});
        const double lo = std::ceil(projected[axis] - extent - padding);
        const double hi = std::floor(projected[axis] + extent + padding);
        // 2^63 is exactly representable; the largest int64_t is not.
        const double integer_limit = std::ldexp(1.0, 63);
        if (!std::isfinite(lo) || !std::isfinite(hi)
            || lo <= -integer_limit || hi >= integer_limit)
            throw std::runtime_error("Minimum-distance translation range is not representable");
        if (lo > hi) return best_sq;
        lower[axis] = static_cast<std::int64_t>(lo);
        upper[axis] = static_cast<std::int64_t>(hi);
    }
    for (auto i = lower[0];; ++i) {
        for (auto j = lower[1];; ++j) {
            for (auto k = lower[2];; ++k) {
                if (!self || i != 0 || j != 0 || k != 0) {
                    Eigen::RowVector3d candidate = delta;
                    const std::array<std::int64_t, 3> shift{i, j, k};
                    for (int axis = 0; axis < rank; ++axis)
                        candidate -= static_cast<double>(shift[axis]) * basis.row(axis);
                    const double candidate_sq = candidate.squaredNorm();
                    if (candidate_sq < best_sq) {
                        best_sq = candidate_sq;
                        if (best_vector) *best_vector = candidate;
                    }
                    if (best_sq == 0) return 0;
                }
                if (k == upper[2]) break;
            }
            if (j == upper[1]) break;
        }
        if (i == upper[0]) break;
    }
    return best_sq;
}

// A missing open cell direction does not prevent wrapping along independent
// periodic rows. Full cells keep their existing fractional-coordinate convention.
bool wrap_singular_cell(Atoms &atom, Eigen::RowVector3d *point = nullptr)
{
    if (atom.cell_.fullPivLu().isInvertible()) return false;
    const int rank = static_cast<int>(atom.pbc_[0]) + static_cast<int>(atom.pbc_[1])
        + static_cast<int>(atom.pbc_[2]);
    Eigen::MatrixXd basis(rank, 3);
    int row = 0;
    for (int axis = 0; axis < 3; ++axis)
        if (atom.pbc_[axis]) basis.row(row++) = atom.cell_.row(axis);
    if (!basis.allFinite())
        throw std::runtime_error("Wrapping requires finite periodic cell vectors");
    const Eigen::MatrixXd gram = basis * basis.transpose();
    if (!gram.allFinite())
        throw std::runtime_error("Wrapping periodic metric is not finite");
    const auto decomposition = gram.fullPivLu();
    if (!decomposition.isInvertible())
        throw std::runtime_error("Wrapping requires independent periodic cell vectors");
    const Eigen::MatrixXd dual = decomposition.solve(basis);
    if (!dual.allFinite())
        throw std::runtime_error("Wrapping periodic dual is not finite");
    auto wrap_position = [&](Eigen::RowVector3d &position) {
        const Eigen::VectorXd coefficients = dual * position.transpose();
        for (int axis = 0; axis < rank; ++axis)
            position -= std::floor(coefficients[axis]) * basis.row(axis);
    };
    if (point) wrap_position(*point);
    else for (auto &position : atom.pos_) wrap_position(position);
    return true;
}

struct DuplicateAtomEdge {
    int to = -1;
    Eigen::RowVector3d disp = Eigen::RowVector3d::Zero();
};

struct DuplicateAtomComponent {
    std::vector<int> nodes;
    Eigen::RowVector3d mean = Eigen::RowVector3d::Zero();
    int representative = -1;
};

template<typename T>
void compact_atom_vector(std::vector<T> &values, const std::vector<int> &keep)
{
    if (values.empty()) return;
    std::vector<T> old = values;
    values.clear();
    values.reserve(keep.size());
    for (int id : keep) {
        if (id < static_cast<int>(old.size())) {
            values.push_back(old[id]);
        }
    }
}

void compact_local_atom_attribs(Atoms &atom, const std::vector<int> &keep, int old_size)
{
    for (auto &it : atom.local_attribs_d_) {
        if (it.second.size() == 0) continue;
        if (it.second.size() != old_size) {
            it.second.resize(0);
            continue;
        }
        Eigen::VectorXd old = it.second;
        it.second.resize(keep.size());
        for (int i = 0; i < static_cast<int>(keep.size()); ++i) {
            it.second(i) = old(keep[i]);
        }
    }
    for (auto &it : atom.local_attribs_i_) {
        if (it.second.size() == 0) continue;
        if (it.second.size() != old_size) {
            it.second.resize(0);
            continue;
        }
        Eigen::VectorXi old = it.second;
        it.second.resize(keep.size());
        for (int i = 0; i < static_cast<int>(keep.size()); ++i) {
            it.second(i) = old(keep[i]);
        }
    }
    for (auto &it : atom.local_attribs_v_) {
        if (it.second.cols() == 0) continue;
        if (it.second.cols() != old_size) {
            it.second.resize(it.second.rows(), 0);
            continue;
        }
        Eigen::MatrixXd old = it.second;
        it.second.resize(old.rows(), keep.size());
        for (int i = 0; i < static_cast<int>(keep.size()); ++i) {
            it.second.col(i) = old.col(keep[i]);
        }
    }
    for (auto &it : atom.local_attribs_vi_) {
        if (it.second.cols() == 0) continue;
        if (it.second.cols() != old_size) {
            it.second.resize(it.second.rows(), 0);
            continue;
        }
        Eigen::MatrixXi old = it.second;
        it.second.resize(old.rows(), keep.size());
        for (int i = 0; i < static_cast<int>(keep.size()); ++i) {
            it.second.col(i) = old.col(keep[i]);
        }
    }
}

void invalidate_duplicate_atom_indexed_data(Atoms &atom, int old_size)
{
    atom.clear_hvp();
    atom.bonds_.clear();
    atom.angles_.clear();
    atom.molid_.clear();
    atom.molsize_.clear();
    atom.mol_types_.clear();
    atom.molnum_ = 0;

    if (atom.hessian_.rows() == 3 * old_size && atom.hessian_.cols() == 3 * old_size) {
        atom.hessian_.resize(0, 0);
        atom.have_hessian = false;
    }

    atom.have_ghost = false;
    atom.nlocal_ = atom.size();
    atom.nghost_ = 0;
    atom.map_ghost_.clear();
}

void repeat_supercell_properties(Atoms &atom, int old_size, int copies)
{
    if (copies == 1) return;
    auto repeat = [old_size, copies](auto &values) {
        if (values.empty()) return;
        if (values.size() != static_cast<size_t>(old_size))
            throw std::invalid_argument("Supercell property size must match the original atom count");
        const auto old = values;
        values.resize(old_size * copies);
        for (int c = 0; c < copies; ++c)
            std::copy(old.begin(), old.end(), values.begin() + c * old_size);
    };
    repeat(atom.forces_);
    repeat(atom.velocities_);
    repeat(atom.site_energies_);
    repeat(atom.charges_);
    repeat(atom.magmom_scalar_);
    repeat(atom.magmoms_);
    repeat(atom.born_eff_chgs_);
    repeat(atom.atomic_fitness);
    repeat(atom.masks);
    repeat(atom.sel_dynpos_);
    auto repeat_scalars = [old_size, copies](auto &attributes) {
        for (auto &[name, value] : attributes) {
            if (value.size() == 0) continue;
            if (value.size() != old_size)
                throw std::invalid_argument("Supercell local attribute size must match the atom count");
            value = value.replicate(copies, 1).eval();
        }
    };
    auto repeat_vectors = [old_size, copies](auto &attributes) {
        for (auto &[name, value] : attributes) {
            if (value.cols() == 0) continue;
            if (value.cols() != old_size)
                throw std::invalid_argument("Supercell local attribute size must match the atom count");
            value = value.replicate(1, copies).eval();
        }
    };
    repeat_scalars(atom.local_attribs_d_);
    repeat_scalars(atom.local_attribs_i_);
    repeat_vectors(atom.local_attribs_v_);
    repeat_vectors(atom.local_attribs_vi_);
    if (atom.fitting_factor.size() > 1) {
        const auto old = atom.fitting_factor;
        if (old.size() != old_size + 1 && old.size() != old_size + 2)
            throw std::invalid_argument("Invalid supercell fitting-factor size");
        atom.fitting_factor.resize(old.size() + old_size * (copies - 1));
        for (int c = 0; c < copies; ++c)
            std::copy_n(old.begin() + 1, old_size, atom.fitting_factor.begin() + 1 + c * old_size);
        if (old.size() == old_size + 2) atom.fitting_factor.back() = old.back();
    }
    if (atom.have_energy) atom.energy_ *= copies;
    if (atom.have_fenergy) atom.fenergy_ *= copies;
    if (atom.have_stress) atom.stress_ *= copies; // stored as an extensive virial (eV)
    auto total_charge = atom.global_attribs_d_.find("total_charge");
    if (total_charge != atom.global_attribs_d_.end()) total_charge->second *= copies;
    // Atom identifiers, topology and inter-image second derivatives need rebuilding.
    atom.ids_.clear();
    invalidate_duplicate_atom_indexed_data(atom, old_size);
}

void rotate_atom_properties(Atoms &atom, const Eigen::Matrix3d &rotation)
{
    for (auto &f : atom.forces_) f = rotation * f;
    for (auto &v : atom.velocities_) v = rotation * v;
    for (auto &m : atom.magmoms_) m = rotation * m;
    const auto initial_moments = atom.local_attribs_v_.find("initial_magmoms");
    if (initial_moments != atom.local_attribs_v_.end() &&
        initial_moments->second.rows() == 3 && initial_moments->second.cols() == atom.size())
        initial_moments->second = (rotation * initial_moments->second).eval();
    for (auto &z : atom.born_eff_chgs_) z = (rotation * z * rotation.transpose()).eval();
    if (atom.have_stress)
        atom.stress_ = (rotation * atom.stress_ * rotation.transpose()).eval();
    if (atom.has_hessian()) {
        for (int i = 0; i < atom.size(); ++i)
            for (int j = 0; j < atom.size(); ++j) {
                auto block = atom.hessian_.block<3, 3>(3 * i, 3 * j);
                block = (rotation * block * rotation.transpose()).eval();
            }
    }
    for (auto *values : {&atom.hvp_directions_, &atom.hvp_targets_})
        if (values->rows() == 3 * atom.size())
            for (int i = 0; i < atom.size(); ++i)
                values->middleRows(3 * i, 3) = (rotation * values->middleRows(3 * i, 3)).eval();
}

void compact_duplicate_atoms(Atoms &atom, const std::vector<int> &keep, int old_size)
{
    compact_atom_vector(atom.pos_, keep);
    compact_atom_vector(atom.chemmap_, keep);
    compact_atom_vector(atom.ids_, keep);
    compact_atom_vector(atom.forces_, keep);
    compact_atom_vector(atom.velocities_, keep);
    compact_atom_vector(atom.site_energies_, keep);
    compact_atom_vector(atom.charges_, keep);
    compact_atom_vector(atom.magmom_scalar_, keep);
    compact_atom_vector(atom.magmoms_, keep);
    compact_atom_vector(atom.born_eff_chgs_, keep);
    compact_atom_vector(atom.atomic_fitness, keep);
    compact_atom_vector(atom.masks, keep);
    compact_atom_vector(atom.sel_dynpos_, keep);
    compact_local_atom_attribs(atom, keep, old_size);
    invalidate_duplicate_atom_indexed_data(atom, old_size);
}

void wrap_duplicate_atom_position(Atoms &atom, Eigen::RowVector3d &pos)
{
    if (std::abs(atom.cell_.determinant()) > 1.0e-14) {
        atom.wrap(pos);
    }
}

std::vector<std::vector<DuplicateAtomEdge>> build_duplicate_atom_graph(Atoms &atom, double rtol)
{
    std::vector<std::vector<DuplicateAtomEdge>> graph(atom.size());
    std::map<std::pair<int, int>, std::pair<double, Eigen::RowVector3d>> shortest_edges;
    NeighborhoodKD nbhskd(&atom, rtol);

    for (int i = 0; i < atom.size(); ++i) {
        auto nbh = nbhskd.query(atom.pos_[i].data(), rtol, atom.chemmap_[i], true);
        for (int k = 0; k < nbh.nbhcount_; ++k) {
            int j = nbh.indices_[k];
            if (j == i || j < 0 || j >= atom.size()) continue;
            if (atom.chemmap_[i] != atom.chemmap_[j]) continue;
            if (nbh.distances_[k] > rtol) continue;

            int a = std::min(i, j);
            int b = std::max(i, j);
            Eigen::RowVector3d disp = (a == i) ? nbh.pos_[k] : -nbh.pos_[k];
            auto key = std::make_pair(a, b);
            auto it = shortest_edges.find(key);
            if (it == shortest_edges.end() || nbh.distances_[k] < it->second.first) {
                shortest_edges[key] = {nbh.distances_[k], disp};
            }
        }
    }

    for (const auto &it : shortest_edges) {
        int a = it.first.first;
        int b = it.first.second;
        const auto &disp = it.second.second;
        graph[a].push_back({b, disp});
        graph[b].push_back({a, -disp});
    }
    return graph;
}

DuplicateAtomComponent make_duplicate_atom_component(
    const Atoms &atom,
    const std::vector<std::vector<DuplicateAtomEdge>> &graph,
    int start,
    std::vector<char> &visited)
{
    DuplicateAtomComponent component;
    std::queue<int> pending;
    std::map<int, Eigen::RowVector3d> unwrapped_by_id;

    visited[start] = 1;
    pending.push(start);
    unwrapped_by_id[start] = atom.pos_[start];

    while (!pending.empty()) {
        int current = pending.front();
        pending.pop();
        component.nodes.push_back(current);

        for (const auto &edge : graph[current]) {
            if (visited[edge.to]) continue;
            visited[edge.to] = 1;
            unwrapped_by_id[edge.to] = unwrapped_by_id[current] + edge.disp;
            pending.push(edge.to);
        }
    }

    std::sort(component.nodes.begin(), component.nodes.end());
    for (int id : component.nodes) {
        component.mean += unwrapped_by_id[id];
    }
    component.mean /= static_cast<double>(component.nodes.size());

    double best_dist2 = std::numeric_limits<double>::max();
    for (int id : component.nodes) {
        double dist2 = (unwrapped_by_id[id] - component.mean).squaredNorm();
        if (dist2 < best_dist2 - 1.0e-18 ||
            (std::abs(dist2 - best_dist2) <= 1.0e-18 && id < component.representative)) {
            best_dist2 = dist2;
            component.representative = id;
        }
    }
    return component;
}

} // namespace

// Assume mat is in HNF format, and all elements are positive
void Atoms::mksuperL_hnf(const Eigen::Matrix3i &mat)
{
    Eigen::Matrix3d rmatd = mat.cast<double>().inverse();
    Eigen::Matrix3d newcell = mat.cast<double>() * cell_;
    int csize = size();
    int nexpand = mat(0, 0) * mat(1, 1) * mat(2, 2);
    int newsize = nexpand * size();

    /*Find lattice points in supercell*/
    std::vector<Eigen::RowVector3d> latpoints;
    latpoints.resize(nexpand);
    // Correct cell
    // for (int ii = 0; ii < 3; ii++)
    // {
    //     for (int jj = 0; jj < 3; jj++)
    //     {
    //         if (fabs(newcell(ii, jj)) < ROUND_PREC)
    //         {
    //             newcell(ii, jj) = 0;
    //         }
    //     }
    // }
    int ct = 0;
    Eigen::RowVector3d po;
    for (int i = 0; i < mat(0, 0); ++i)
    {
        for (int j = 0; j < mat(1, 1); ++j)
        {
            for (int k = 0; k < mat(2, 2); ++k)
            {
                // if ((int)round(mat(2, 1) * k - mat(2, 2) * j) > 0)
                //     continue;
                // if ((int)round(mat(1, 1) * mat(2, 2) * i - mat(1, 0) * mat(2, 2) * j + (mat(1, 0) * mat(2, 1) - mat(2, 0) * mat(1, 1)) * k) < 0)
                //     continue;
                po << i, j, k;
                po = po * rmatd;
                //po = po * newcell.inverse();
                if (!(po(0) <= 1 - 1e-4 && po(1) <= 1 - 1e-4 && po(2) <= 1 - 1e-4))
                    continue;
                latpoints[ct] = po * newcell;
                ++ct;
            }
        }
    }
    /*Build positions*/
    std::vector<Eigen::RowVector3d> posold = pos_;
    if (!cartesian)
        for (auto &position : posold) position = dir2car(position);
    std::vector<int> chemmapold = chemmap_;
    pos_.clear();
    chemmap_.clear();
    pos_.reserve(newsize);
    chemmap_.reserve(newsize);
    /*Put as this
    Original atoms
    Original atoms with off set 1,2,3..., same as in enum.x
    */
    for (int j = 0; j < nexpand; ++j)
    {
        for (int i = 0; i < csize; ++i)
        {
            pos_.emplace_back(posold[i] + latpoints[j]);
            chemmap_.emplace_back(chemmapold[i]);
        }
    }
    repeat_supercell_properties(*this, csize, nexpand);
    cell_ = newcell;
    if (!cartesian) {
        const Eigen::Matrix3d inverse = cell_.inverse();
        for (auto &position : pos_) position *= inverse;
    }
}

/**
 * @brief Construct a supercell from an integer lattice transformation.
 *
 * Given a 3x3 integer transformation matrix mat (row/column convention
 * consistent with the class), this function constructs the corresponding
 * supercell:
 *  - computes the new cell vectors as newcell = mat * cell_
 *  - enumerates the lattice translation points that map the original cell
 *    into the fundamental domain of the new cell (fractional coordinates in [0,1))
 *  - replicates all original atomic positions by adding each lattice translation
 *    (in Cartesian coordinates) to produce the supercell atomic positions
 *  - updates the internal cell_ to the new supercell cell
 *
 * The algorithm is robust to floating-point rounding by:
 *  - computing integer enumeration bounds from the transformed cube corners
 *    (to avoid missing translations when mat contains nontrivial entries)
 *  - expanding bounds slightly by an eps tolerance before converting to integers
 *  - testing membership in [0,1) with a ROUND_PREC tolerance
 *
 * Parameters:
 *  - mat: 3x3 integer transformation matrix describing the supercell. The
 *         expected volume expansion is abs(det(mat)). mat is assumed to be
 *         the mapping from original lattice indices to supercell lattice indices.
 *
 * Preconditions / assumptions:
 *  - det(mat) != 0 (the matrix must be invertible). A singular matrix will
 *    cause the inverse computation to fail or produce invalid results.
 *  - cell_, pos_, and chemmap_ are initialized and consistent: pos_.size()
 *    equals chemmap_.size() (the original number of atoms).
 *  - ROUND_PREC is available in the translation unit and encodes the
 *    fractional tolerance for classifying points as inside [0,1).
 *
 * Side effects (state changes):
 *  - pos_ and chemmap_ are replaced with the supercell-expanded lists.
 *  - cell_ is replaced with the transformed cell (newcell).
 *  - intermediate copies of the original positions and chemistry mapping
 *    are created (posold, chemmapold) to perform replication.
 *
 * Error handling and diagnostics:
 *  - If the number of lattice points found does not equal the expected
 *    expansion factor (abs(det(mat))), a warning is emitted to std::cerr
 *    but the routine continues to build the supercell with the points found.
 *  - After building, if the final atom count does not match expected
 *    nexpand * original_count, a warning is emitted to std::cerr.
 *
 * Numerical details:
 *  - Uses a small eps (1e-8) to enlarge floating-point bounds before converting
 *    to integer enumeration ranges to counteract tiny numeric errors.
 *  - Tests fractional coordinates with ROUND_PREC to decide inclusion in [0,1).
 *  - The function uses Eigen::Matrix3d and Eigen::RowVector3d arithmetic; it
 *    computes inv_transformation = transformation.inverse() and therefore
 *    relies on a numerically stable invertible transformation.
 *
 * Complexity:
 *  - Time complexity is O(nexpand * csize) where csize is original atom count
 *    and nexpand = abs(det(mat)) is the supercell multiplicity (dominant work
 *    is copying every original atom for every lattice translation).
 *  - Additional overhead for enumerating candidate integer vectors bounded by
 *    the transformed corner extents, which is independent of csize but related
 *    to the magnitude and structure of mat.
 *
 * Exceptions:
 *  - May throw from Eigen (e.g., if inverse() is called on a singular matrix)
 *    or from standard library allocations (bad_alloc).
 *
 * Notes:
 *  - The function treats negative determinants (orientation-reversing
 *    transformations) by using the absolute determinant for the expected
 *    number of translations.
 *  - If stricter handling of singular or otherwise invalid transformations is
 *    required, callers should validate mat (e.g., check det != 0) before calling.
 *
 * @return void
 */
void Atoms::mksuperL(const Eigen::Matrix3i &mat)
{
    // TODO: use this algo:
    // 1. find lattice points in the 六面体内
    // int nexpand = static_cast<int>(std::abs(mat.determinant()));
    // int csize = size();
    // int newsize = nexpand * csize;



    // Transformation matrix in double
    Eigen::Matrix3d transformation = mat.cast<double>();
    Eigen::Matrix3d newcell = transformation * cell_;
    Eigen::Matrix3d inv_transformation = transformation.inverse();

    int nexpand = static_cast<int>(std::abs(mat.determinant()));
    int csize = size();
    int newsize = nexpand * csize;

    // backup
    std::vector<Eigen::RowVector3d> posold = pos_;
    if (!cartesian)
        for (auto &position : posold) position = dir2car(position);
    std::vector<int> chemmapold = chemmap_;

    // prepare new containers
    pos_.clear();
    chemmap_.clear();
    pos_.reserve(newsize);
    chemmap_.reserve(newsize);

    // --- Determine robust integer ranges for p (the integer vectors we will test)
    // For all fractional corners x in {0,1}^3 compute p_corner = x * transformation.
    // Then use min/max of those p_corner components to bound integer p to enumerate.
    Eigen::RowVector3d corners[8] = {
        Eigen::RowVector3d(0,0,0), Eigen::RowVector3d(1,0,0),
        Eigen::RowVector3d(0,1,0), Eigen::RowVector3d(0,0,1),
        Eigen::RowVector3d(1,1,0), Eigen::RowVector3d(1,0,1),
        Eigen::RowVector3d(0,1,1), Eigen::RowVector3d(1,1,1)
    };

    Eigen::Vector3d pmin = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector3d pmax = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
    for (int ci = 0; ci < 8; ++ci) {
        Eigen::RowVector3d p_corner = corners[ci] * transformation; // row * mat -> row
        for (int d = 0; d < 3; ++d) {
            pmin(d) = std::min(pmin(d), p_corner(d));
            pmax(d) = std::max(pmax(d), p_corner(d));
        }
    }
    // expand a bit to account for numerical tolerance, then convert to integer bounds
    const double eps = 1e-8;
    Eigen::Vector3i imin, imax;
    for (int d = 0; d < 3; ++d) {
        imin(d) = static_cast<int>(std::floor(pmin(d) - eps));
        imax(d) = static_cast<int>(std::ceil (pmax(d) + eps));
    }

    // Collect lattice translation displacements (cartesian) corresponding to fractional coords in [0,1)
    std::vector<Eigen::RowVector3d> latpoints;
    latpoints.reserve(nexpand);

    bool stop = false;
    for (int i = imin(0); i <= imax(0) && !stop; ++i) {
        for (int j = imin(1); j <= imax(1) && !stop; ++j) {
            for (int k = imin(2); k <= imax(2) && !stop; ++k) {
                Eigen::RowVector3d p(i, j, k); // integer vector
                // fractional coordinate in new-cell basis
                Eigen::RowVector3d fractional = p * inv_transformation; // row * inv_transformation

                // Check if within the [0,1) range (accounting for numerical precision)
                if (fractional(0) >= -ROUND_PREC && fractional(0) < 1.0 - ROUND_PREC &&
                    fractional(1) >= -ROUND_PREC && fractional(1) < 1.0 - ROUND_PREC &&
                    fractional(2) >= -ROUND_PREC && fractional(2) < 1.0 - ROUND_PREC) {

                    Eigen::RowVector3d cartesian_displacement = fractional * newcell; // row * newcell -> cartesian
                    latpoints.push_back(cartesian_displacement);

                    if (static_cast<int>(latpoints.size()) >= nexpand) {
                        stop = true;
                        break;
                    }
                }
            }
        }
    }

    // Safety: if we didn't find enough lattice points something is wrong — but continue to build anyway
    if (static_cast<int>(latpoints.size()) != nexpand) {
        std::cerr << "Warning: expected " << nexpand << " lattice points but found "
                  << latpoints.size() << ".\n";
    }

    // build supercell atoms
    for (const auto& displacement : latpoints) {
        for (int ii = 0; ii < csize; ++ii) {
            pos_.emplace_back(posold[ii] + displacement);
            chemmap_.emplace_back(chemmapold[ii]);
        }
    }

    // update cell
    repeat_supercell_properties(*this, csize, nexpand);
    cell_ = newcell;
    if (!cartesian) {
        const Eigen::Matrix3d inverse = cell_.inverse();
        for (auto &position : pos_) position *= inverse;
    }

    // final check (optional)
    if (static_cast<int>(pos_.size()) != newsize) {
        std::cerr << "Warning: Supercell creation may have issues. Expected "
                  << newsize << " atoms, got " << pos_.size() << std::endl;
    }
}

void Atoms::mksuper(const Eigen::Vector3i &vec)
{
    int mul = vec.prod();
    int osize = size();
    std::vector<Eigen::RowVector3d> posold = pos_;
    if (!cartesian)
        for (auto &position : posold) position = dir2car(position);
    pos_.resize(osize * mul);
    chemmap_.resize(osize * mul);
    int ct = 0;
    for (int i = 0; i < vec(0); ++i)
    {
        for (int j = 0; j < vec(1); ++j)
        {
            for (int k = 0; k < vec(2); ++k)
            {
                for (int c = 0; c < osize; c++)
                {
                    pos_[ct] = posold[c] + i * cell_.row(0) + j * cell_.row(1) + k * cell_.row(2);
                    chemmap_[ct] = chemmap_[c];
                    ++ct;
                }
            }
        }
    }
    repeat_supercell_properties(*this, osize, mul);
    cell_.row(0) = cell_.row(0) * vec(0);
    cell_.row(1) = cell_.row(1) * vec(1);
    cell_.row(2) = cell_.row(2) * vec(2);
    if (!cartesian) {
        const Eigen::Matrix3d inverse = cell_.inverse();
        for (auto &position : pos_) position *= inverse;
    }
}
// mat must all elements positive
void Atoms::createSuperL(const Eigen::Matrix3d &mat) {
    int nexpand = mat.determinant();
    int n_orig_atoms = size();
    int n_expand_atoms = n_orig_atoms * nexpand;

    Eigen::Matrix3d newcell = mat * cell_;
    Eigen::Matrix3d invmat = mat.inverse();
    Eigen::Matrix3d invcell = cell_.inverse();
    Eigen::Matrix<double, 8, 3> boundvecs;
    boundvecs << 0, 0, 0,
        0, 0, 1,
        0, 1, 0,
        0, 1, 1,
        1, 0, 0,
        1, 0, 1,
        1, 1, 0,
        1, 1, 1;
    int mina = INT_MAX, maxa = -INT_MAX, minb = INT_MAX, maxb = -INT_MAX, minc = INT_MAX, maxc = -INT_MAX;
    for (int i = 0; i < 8; ++i)
    {
        boundvecs.row(i) = boundvecs.row(i) * mat;
        mina = std::min(mina, (int)floor(boundvecs.row(i)(0)));
        minb = std::min(minb, (int)floor(boundvecs.row(i)(1)));
        minc = std::min(minc, (int)floor(boundvecs.row(i)(2)));
        maxa = std::max(maxa, (int)ceil(boundvecs.row(i)(0)));
        maxb = std::max(maxb, (int)ceil(boundvecs.row(i)(1)));
        maxc = std::max(maxc, (int)ceil(boundvecs.row(i)(2)));
        // primitive cell must contained insied superell
        boundvecs.row(i) = boundvecs.row(i) * invmat;
        for (int j = 0; j < 3; ++j) {
            if (boundvecs.row(i)(j) > 1 + ROUND_PREC || boundvecs.row(i)(j) < -ROUND_PREC) {
                throw std::runtime_error("Primitive basis changed, currently not supported");
            }
        }
    }
    // search extensively
    std::vector<Eigen::RowVector3d> latpoints(nexpand);
    latpoints.clear();
    Eigen::RowVector3d po;
    for (int ia = mina; ia <= maxa; ++ia) {
        for (int ib = minb; ib <= maxb; ++ib) {
            for (int ic = minc; ic <= maxc; ++ic) {
                po << ia, ib, ic;
                po = po * invmat;
                if (!(po(0) <= 1 - ROUND_PREC && po(1) <= 1 - ROUND_PREC && po(2) <= 1 - ROUND_PREC))
                    continue;
                if (!(po(0) >= -ROUND_PREC && po(1) >= -ROUND_PREC && po(2) >= -ROUND_PREC))
                    continue;
                latpoints.push_back(po * newcell);
            }
        }
    }
    if (latpoints.size() != nexpand) {
        throw std::runtime_error("Supercell creation failed: unexpected number of lattice points");
    }
    /*Build positions*/
    std::vector<Eigen::RowVector3d> posold = pos_;
    if (!cartesian)
        for (auto &position : posold) position = dir2car(position);
    std::vector<int> chemmapold = chemmap_;
    pos_.clear();
    chemmap_.clear();
    pos_.reserve(n_expand_atoms);
    chemmap_.reserve(n_expand_atoms);

    for (int j = 0; j < nexpand; ++j)
    {
        for (int i = 0; i < n_orig_atoms; ++i)
        {
            pos_.emplace_back(posold[i] + latpoints[j]);
            chemmap_.emplace_back(chemmapold[i]);
        }
    }
    repeat_supercell_properties(*this, n_orig_atoms, nexpand);
    cell_ = newcell;
    if (!cartesian) {
        const Eigen::Matrix3d inverse = cell_.inverse();
        for (auto &position : pos_) position *= inverse;
    }
}

void Atoms::deformR(const Eigen::Matrix3d &mat)
{
    std::vector<Eigen::RowVector3d> vec;
    get_dir_pos(vec);
    cell_ = cell_ * mat;
    if (cartesian)
        for (int i = 0; i < size(); ++i)
            pos_[i] = vec[i] * cell_;
}

/**
 * @brief Apply a linear deformation to the simulation cell and update atomic positions.
 *
 * Left-multiplies the internal 3x3 cell matrix by the given deformation matrix and then
 * preserves direct (fractional) coordinates obtained via get_dir_pos(). Cartesian
 * inputs are rebuilt in the new cell; fractional inputs retain their representation.
 *
 * @param mat 3x3 deformation matrix to apply to the cell (cell_ <- mat * cell_).
 *
 * Effects:
 * - Calls get_dir_pos(vec) to obtain fractional (direct) coordinates for all atoms.
 * - Updates cell_ by left-multiplication with mat.
 * - For Cartesian input, updates pos_[i] = vec[i] * cell_.
 *   Fractional input retains its direct-coordinate values and flag.
 * - Atom ordering and counts are not changed.
 *
 * Preconditions:
 * - get_dir_pos(vec) must fill vec with exactly size() row-vectors (vec.size() == size()).
 * - pos_ must have capacity/size at least size() for assignment.
 * - mat and cell_ are 3x3 matrices.
 *
 * Complexity: O(N) in the number of atoms (dominated by the loop that updates positions).
 *
 * Exceptions: May propagate exceptions from get_dir_pos or from out-of-range/indexing operations;
 * no explicit validation of mat (e.g., invertibility) is performed.
 */
void Atoms::deformL(const Eigen::Matrix3d &mat)
{
    std::vector<Eigen::RowVector3d> vec;
    get_dir_pos(vec);
    cell_ = mat * cell_;
    if (cartesian)
        for (int i = 0; i < size(); ++i)
            pos_[i] = vec[i] * cell_;
}

void Atoms::scale_by_factor(double factor)
{
    cell_ *= factor;
    if (cartesian)
        for (auto &it : pos_) it *= factor;
};

void Atoms::scale_to_volume(const double volume) {
    double s = std::pow(volume / get_volume(), 1.0 / 3.0);
    scale_by_factor(s);
}

void Atoms::resize(int newsize)
{
    clear_hvp();
    pos_.resize(newsize);
    chemmap_.resize(newsize);
}

// TODO: optimize
void Atoms::build_bonds_mol(const double &rc, int t1, int t2)
{
    int pid = 0;
    double rcsq = rc * rc;
    for (int i = 0; i < size(); ++i)
    {
        if (chemmap_[i] != t1)
            continue;
        for (int j = 0; j < size(); ++j)
        {
            if (molid_[i] == molid_[j] && chemmap_[j] == t2)
            {
                if ((pos_[i] - pos_[j]).squaredNorm() < rcsq)
                {
                    bonds_.push_back(std::array<int, 2>{i, j});
                }
                else
                {
                    auto dc = car2dir(pos_[i] - pos_[j]);
                    for (int kk = 0; kk < 3; ++kk)
                    {
                        if (dc[kk] > 0.5)
                            dc(kk) -= 1.0;
                        else if (dc[kk] < -0.5)
                            dc(kk) += 1.0;
                    }
                    auto dc2 = dir2car(dc);
                    if (dc2.squaredNorm() < rcsq)
                    {
                        bonds_.push_back(std::array<int, 2>{i, j});
                    }
                }
            }
        }
    }
}
// TODO: heavy optimize
// BUG: ONLY TRICK HERE
void Atoms::build_angles_mol(const double &rc, int t1, int t2, int t3)
{
    int pid = 0;
    double rcsq = rc * rc;
    for (int i = 0; i < size(); ++i)
    {
        if (chemmap_[i] != t2)
            continue;
        for (int j = 0; j < size(); ++j)
        {
            if (chemmap_[j] != t1)
                continue;
            for (int k = j + 1; k < size(); ++k)
            {
                if (molid_[i] == molid_[j] && molid_[j] == molid_[k] && chemmap_[k] == t3)
                {
                    if ((pos_[i] - pos_[j]).squaredNorm() < rcsq && (pos_[i] - pos_[k]).squaredNorm() < rcsq)
                    {
                        angles_.push_back(std::array<int, 3>{j, i, k});
                    }
                    else
                    {
                        auto dc1 = car2dir(pos_[i] - pos_[j]);
                        auto dc2 = car2dir(pos_[i] - pos_[k]);

                        for (int kk = 0; kk < 3; ++kk)
                        {
                            if (dc1[kk] > 0.5)
                                dc1(kk) -= 1.0;
                            else if (dc1[kk] < -0.5)
                                dc1(kk) += 1.0;
                            if (dc2[kk] > 0.5)
                                dc2(kk) -= 1.0;
                            else if (dc2[kk] < -0.5)
                                dc2(kk) += 1.0;
                        }

                        auto c1 = dir2car(dc1);
                        auto c2 = dir2car(dc2);
                        if (c1.squaredNorm() < rcsq && c2.squaredNorm() < rcsq)
                        {
                            angles_.push_back(std::array<int, 3>{j, i, k});
                        }
                    }
                }
            }
        }
    }
}

void Atoms::reserve(int newsize)
{
    pos_.reserve(newsize);
    chemmap_.reserve(newsize);
}

bool Atoms::isorth() const
{
    return (abs(cell_.row(0).dot(cell_.row(1))) < 1e-5 &&
            abs(cell_.row(0).dot(cell_.row(2))) < 1e-5 &&
            abs(cell_.row(2).dot(cell_.row(1))) < 1e-5);
};

double Atoms::report_maxforce(void)
{
    double max_force = 0;
    double f;
    for (int i = 0; i < forces_.size(); ++i)
    {
        f = forces_[i].norm();
        max_force = max_force < f ? f : max_force;
    }
    return max_force;
}

void Atoms::clear(void)
{
    pos_.clear();
    forces_.clear();
    site_energies_.clear();
    energy_ = 0;
    fenergy_ = 0;
    have_energy = false;
    have_fenergy = false;
    chemmap_.clear();
    molid_.clear();
    molsize_.clear();
    molnum_ = 0;
    mol_types_.clear();
    bonds_.clear();
    angles_.clear();
    charges_.clear();
    cell_ = Eigen::Matrix3d::Zero();
    pbc_ = {{true, true, true}};
    stress_ = Eigen::Matrix3d::Zero();
    have_stress = false;
    hessian_.resize(0, 0);
    have_hessian = false;
    clear_hvp();
    velocities_.clear();
    born_eff_chgs_.clear();
    have_eff_chg = false;
    magmom_scalar_.clear();
    magmoms_.clear();
    collinear_ = true;
    ids_.clear();
    fitting_factor.clear();
    fitness = 1.0;
    atomic_fitness.clear();
    masks.clear();
    group_count = 0;
    sel_dyncell_ = Eigen::Matrix3d::Zero();
    sel_dynpos_.clear();
    features_.clear();
    global_attribs_d_.clear();
    global_attribs_v_.clear();
    global_attribs_i_.clear();
    global_attribs_vi_.clear();
    local_attribs_d_.clear();
    local_attribs_v_.clear();
    local_attribs_i_.clear();
    local_attribs_vi_.clear();
}
void Atoms::find_mindist(void)
{
    double best_sq = std::numeric_limits<double>::infinity();
    if (pos_.empty()) {
        mindist_ = 1e10;
        return;
    }
    std::vector<Eigen::RowVector3d> positions = pos_;
    if (!cartesian)
        for (auto &position : positions) position = dir2car(position);

    const int rank = static_cast<int>(pbc_[0]) + static_cast<int>(pbc_[1])
        + static_cast<int>(pbc_[2]);
    Eigen::MatrixXd basis(rank, 3), dual(rank, 3);
    if (rank > 0) {
        int row = 0;
        for (int axis = 0; axis < 3; ++axis) {
            if (!pbc_[axis]) continue;
            basis.row(row++) = cell_.row(axis);
            best_sq = std::min(best_sq, cell_.row(axis).squaredNorm());
        }
        if (!basis.allFinite())
            throw std::runtime_error("Minimum distance requires finite periodic cell vectors");
        const Eigen::MatrixXd gram = basis * basis.transpose();
        if (!gram.allFinite())
            throw std::runtime_error("Minimum-distance periodic metric is not finite");
        const auto decomposition = gram.fullPivLu();
        if (!decomposition.isInvertible())
            throw std::runtime_error("Minimum distance requires independent finite periodic cell vectors");
        dual = decomposition.solve(basis);
        if (!dual.allFinite() || !std::isfinite(best_sq))
            throw std::runtime_error("Minimum-distance periodic dual is not finite");
        best_sq = periodic_minimum_squared(Eigen::RowVector3d::Zero(), basis, dual,
                                          best_sq, true);
    }
    for (size_t i = 0; i < positions.size(); ++i) {
        for (size_t j = i + 1; j < positions.size(); ++j) {
            const Eigen::RowVector3d delta = positions[i] - positions[j];
            if (rank == 0) best_sq = std::min(best_sq, delta.squaredNorm());
            else best_sq = periodic_minimum_squared(delta, basis, dual, best_sq, false);
            if (best_sq == 0) {
                mindist_ = 0;
                return;
            }
        }
    }
    mindist_ = std::isinf(best_sq) ? 1e10 : std::sqrt(best_sq);
}

double Atoms::report_mindist(void)
{
    find_mindist();
    return mindist_;
}

void Atoms::find_maxdist(void)
{
    //maybe that "stupid" O(N^2) is more effecient?
    //This only works effeciently for VERY LARGE cells!
    maxdist_ = 0;
    for (int i = 0; i < pos_.size(); ++i)
    {
        Eigen::RowVector3d &posdi = pos_[i];
        for (int j = i + 1; j < pos_.size(); ++j)
        {
            Eigen::RowVector3d &posdj = pos_[j];
            Eigen::RowVector3d posdiff = posdi - posdj;
            posdiff = car2dir(posdiff);
            for (int k = 0; k < 3; ++k)
            {
                if (posdiff[k] > 0.5)
                    posdiff[k] -= 1;
                else if (posdiff[k] < -0.5)
                    posdiff[k] += 1;
            }
            posdiff = dir2car(posdiff);
            double dist = posdiff.norm();
            maxdist_ = maxdist_ > dist ? maxdist_ : dist;
        }
    }

    // for small cells, this may be faster

    // mindist_ = 0;
    // for (int ind2 = 0; ind2 < size(); ind2++)
    //     for (int ind = ind2; ind < size(); ind++)
    //         for (int i = -1; i <= 1; ++i)
    //             for (int j = -1; j <= 1; ++j)
    //                 for (int k = -1; k <= 1; ++k)
    //                     if ((ind2 != ind) || (i != 0) || (j != 0) || (k != 0)){
    //                         Eigen::RowVector3d candidate = pos_[ind];
    //                         candidate += i * cell_.row(0) + j * cell_.row(1) + k * cell_.row(2);
    //                         maxdist_ = maxdist_ > candidate.norm() ? maxdist_ : candidate.norm();
    //                     }
}

// std::map<int, int> Atoms::mapindex; // atom number to 0,1,2
// std::map<int, int> Atoms::revmapindex;

std::map<int, int> Atoms::types_count_map(void) const
{
    std::map<int, int> ret;
    for (auto &it : chemmap_)
    {
        ret[it]++;
    }
    return ret;
}

/**
 * @brief Retrieve the unique atom type identifiers as a vector.
 *
 * Collects the set of unique types provided by get_types_set() and returns
 * them as a std::vector<int>. Each type appears exactly once in the result,
 * in the order imposed by the underlying std::set (typically ascending).
 *
 * @return std::vector<int> A copy of the unique atom type IDs.
 *
 * @note Modifying the returned vector does not affect the Atoms instance.
 * @see get_types_set()
 */
std::vector<int> Atoms::get_types(void) const
{
    std::set<int> types = get_types_set();
    std::vector<int> ret;
    for (auto it = types.begin(); it != types.end(); ++it)
    {
        ret.push_back(*it);
    }
    return ret;
}

std::set<int> Atoms::get_types_set(void) const
{
    std::set<int> types;
    for (int i = 0; i < size(); ++i)
    {
        types.insert(chemmap_[i]);
    }
    return types;
}

Eigen::Matrix3d Atoms::get_norm_cell(void) const
{
    Eigen::Matrix3d ret;
    for (int i = 0; i < 3; ++i)
        ret.row(i) = cell_.row(i) / cell_.row(i).norm();
    return ret;
}
// Wrap periodic fractional components into [0, 1), including exact cell boundaries.
void Atoms::wrap()
{
    if (is_nonperiodic()) return;
    if (!cartesian) {
        for (auto &pos : pos_)
            for (int j = 0; j < 3; ++j)
                if (pbc_[j]) pos[j] -= floor(pos[j]);
        return;
    }
    if (wrap_singular_cell(*this)) return;
    std::vector<Eigen::RowVector3d> npos;
    get_dir_pos(npos);
    for (int i = 0; i < npos.size(); ++i)
    {
        for (int j = 0; j < 3; ++j)
            if (pbc_[j]) npos[i][j] -= floor(npos[i][j]);
        pos_[i] = dir2car(npos[i]);
    }
}

void Atoms::wrap(Eigen::RowVector3d &pos) {
    if (is_nonperiodic()) return;
    if (wrap_singular_cell(*this, &pos)) return;
    auto t = car2dir(pos);
    for (int j = 0; j < 3; ++j)
        if (pbc_[j]) t[j] -= floor(t[j]);
    pos = dir2car(t);
}

void Atoms::hwrap()
{
    if (is_nonperiodic()) return;
    if (!cartesian) { wrap(); return; }
    if (wrap_singular_cell(*this)) return;
    std::vector<Eigen::RowVector3d> npos;
    get_dir_pos(npos);
    for (int i = 0; i < npos.size(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            // npos[i][j] -= floor(npos[i][j]);
            if (pbc_[j]) pos_[i] -= floor(npos[i][j]) * cell_.row(j);
        }
    }
}

// refuse to merge if cell_ is not equal
Atoms merge_atoms(const Atoms &a1, const Atoms &a2)
{
if ((a1.cell_ - a2.cell_).norm() > 1e-4)
    {
        std::cerr << "ERROR: Merging wrong size of cells" << std::endl;
    }
    Atoms ret;
    ret.cell_ = a1.cell_;
    ret.pos_.reserve(a1.size() + a2.size());
    ret.chemmap_.reserve(a1.size() + a2.size());
    ret.pos_.insert(ret.pos_.end(), a1.pos_.begin(), a1.pos_.end());
    ret.chemmap_.insert(ret.chemmap_.end(), a1.chemmap_.begin(), a1.chemmap_.end());
    ret.pos_.insert(ret.pos_.end(), a2.pos_.begin(), a2.pos_.end());
    ret.chemmap_.insert(ret.chemmap_.end(), a2.chemmap_.begin(), a2.chemmap_.end());
    return ret;
    // ret.features_ = a1.features_;
    // ret.features_.insert(a2.features_.begin(), a2.features_.end());
}

void Atoms::resetEFS() {
    energy_ = 0;
    stress_ = Eigen::Matrix3d::Zero();
    forces_.assign(size(), Eigen::RowVector3d::Zero());
    site_energies_.assign(size(), 0);
    have_energy = true;
    have_stress = true;
}

void Atoms::resetE()
{
    energy_ = 0;
    site_energies_.assign(size(), 0);
    have_energy = true;
    have_stress = false;
}

void Atoms::clearEFS() {
    forces_.clear();
    site_energies_.clear();
    have_energy = false;
    have_stress = false;
}

void Atoms::merge_atoms(const Atoms &a1)
{
    clear_hvp();
    if ((cell_ - a1.cell_).norm() > 1e-4)
    {
        std::cerr << "ERROR: Merging wrong size of cells" << std::endl;
    }
    pos_.reserve(size() + a1.size());
    chemmap_.reserve(size() + a1.size());
    pos_.insert(pos_.end(), a1.pos_.begin(), a1.pos_.end());
    chemmap_.insert(chemmap_.end(), a1.chemmap_.begin(), a1.chemmap_.end());
    // others
}
// Use another function with info extracted
Atoms Atoms::extract_substructures_from_mask(const GPMask &mask) const
{
    Atoms ret;
    int nsize = __min(mask.size(), size());
    ret.reserve(0.7 * nsize);
    for (int i = 0; i < nsize; ++i)
    {
        if (mask[i] == 1)
        {
            ret.pos_.push_back(pos_[i]);
            ret.chemmap_.push_back(chemmap_[i]);
        }
    }
    return ret;
}

Atoms extract_substructures_from_mask(const Atoms &a1, const GPMask &mask)
{
    return a1.extract_substructures_from_mask(mask);
}

void extract_substructures_from_mask(const Atoms &a1, const GPMask &mask, Atoms &masked, Atoms &nonmasked)
{
    int nsize = __min(mask.size(), a1.size());
    masked.reserve(0.6 * nsize);
    nonmasked.reserve(0.6 * nsize);
    for (int i = 0; i < nsize; ++i)
    {
        if (mask[i] == 1)
        {
            masked.pos_.push_back(a1.pos_[i]);
            masked.chemmap_.push_back(a1.chemmap_[i]);
        }
        else
        {
            nonmasked.pos_.push_back(a1.pos_[i]);
            nonmasked.chemmap_.push_back(a1.chemmap_[i]);
        }
    }
}

std::ostream &operator<<(std::ostream &os, const Atoms &atom)
{
    os.setf(std::ios_base::fixed, std::ios_base::floatfield);
    os.width(10);
    os << "Cell: " << std::endl;
    os << atom.cell_ << std::endl;
    os << "Positions Chem (SiteEn): " << std::endl;
    for (int i = 0; i < atom.size(); ++i)
    {
        os << atom.pos_[i] << "   " << atom.chemmap_[i];
        if (atom.has_site_energies()) 
            os << " " << atom.site_energies_[i];
        os << std::endl;
    }
    if (atom.has_forces())
    {
        os << "Forces: " << std::endl;
        for (int i = 0; i < atom.size(); ++i)
        {
            os << atom.forces_[i].transpose() << std::endl;
        }
    }
    if (atom.has_magmom())
    {
        os << "Magmoms: " << std::endl;
        for (int i = 0; i < atom.size(); ++i)
        {
            os << atom.magmoms_[i].transpose() << std::endl;
        }
    }
    if (atom.has_energy())
    {
        os << "Energy: " << atom.energy_ << std::endl;
    }
    if (atom.features_.size() > 0)
    {
        for (auto &it : atom.features_)
        {
            os << "Features  " << it.first << "  " << it.second << '\n';
        }
    }
    os << std::endl;
    return os;
}

Eigen::Matrix3d Atoms::get_lower_cell(void) const
{
    auto para = get_cell_parameters();
    Eigen::Matrix3d cell;
    double a = para[0];
    double b = para[1];
    double c = para[2];
    double alpha = para[3];
    double beta = para[4];
    double gamma = para[5];
    cell(0, 0) = a;
    cell(0, 1) = 0;
    cell(0, 2) = 0;
    cell(1, 0) = b * std::cos(gamma);
    cell(1, 1) = b * std::sin(gamma);
    cell(1, 2) = 0;
    cell(2, 0) = c * std::cos(beta);
    cell(2, 1) = c * (std::cos(alpha) - std::cos(beta) * std::cos(gamma)) / std::sin(gamma);
    cell(2, 2) = std::sqrt(c * c - cell(2, 0) * cell(2, 0) - cell(2, 1) * cell(2, 1));
    return cell;
}
// TODO: verfiy this, not checked if phicsically meaningful
Eigen::Matrix3d Atoms::get_strain(Atoms &ref) const
{
    // auto refcell = ref.get_lower_cell();
    // auto mycell = get_lower_cell();
    // std::array<double, 6> strain;
    // auto diffcell = mycell - refcell;
    // strain[0] = diffcell(0, 0) / refcell(0, 0);
    // strain[1] = diffcell(1, 1) / refcell(1, 1);
    // strain[2] = diffcell(2, 2) / refcell(2, 2);
    // // shear, TODO: double check this
    // strain[3] = diffcell(1, 0) / refcell(1, 1);
    // strain[4] = diffcell(2, 0) / refcell(2, 2);
    // strain[5] = diffcell(2, 1) / refcell(2, 2);
    // return strain;
    return (cell_ * ref.cell_.inverse()) - Eigen::Matrix3d::Identity();
}

std::vector<double> Atoms::get_cell_parameters(void) const
{
    std::vector<double> paras(6);
    paras[0] = cell_.row(0).norm();
    paras[1] = cell_.row(1).norm();
    paras[2] = cell_.row(2).norm();
    paras[3] = std::acos(cell_.row(1).dot(cell_.row(2)) / paras[1] / paras[2]);
    paras[4] = std::acos(cell_.row(2).dot(cell_.row(0)) / paras[2] / paras[0]);
    paras[5] = std::acos(cell_.row(0).dot(cell_.row(1)) / paras[0] / paras[1]);
    return paras;
}
// Assume right hand
void Atoms::set_cell(double a, double b, double c, double alpha, double beta, double gamma)
{
    Eigen::Matrix3d cell;
    cell(0, 0) = a;
    cell(0, 1) = 0;
    cell(0, 2) = 0;
    cell(1, 0) = b * std::cos(gamma);
    cell(1, 1) = b * std::sin(gamma);
    cell(1, 2) = 0;
    cell(2, 0) = c * std::cos(beta);
    cell(2, 1) = c * (std::cos(alpha) - std::cos(beta) * std::cos(gamma)) / std::sin(gamma);
    cell(2, 2) = std::sqrt(c * c - cell(2, 0) * cell(2, 0) - cell(2, 1) * cell(2, 1));
    set_cell(cell);
}

void Atoms::set_cell(double a, double b, double c, double alpha, double beta, double gamma, bool sc_pos)
{
    if (sc_pos) {
        set_cell(a, b, c, alpha, beta, gamma);
        return;
    }
    Eigen::Matrix3d cell;
    cell(0, 0) = a;
    cell(0, 1) = 0;
    cell(0, 2) = 0;
    cell(1, 0) = b * std::cos(gamma);
    cell(1, 1) = b * std::sin(gamma);
    cell(1, 2) = 0;
    cell(2, 0) = c * std::cos(beta);
    cell(2, 1) = c * (std::cos(alpha) - std::cos(beta) * std::cos(gamma)) / std::sin(gamma);
    cell(2, 2) = std::sqrt(c * c - cell(2, 0) * cell(2, 0) - cell(2, 1) * cell(2, 1));
    if (!cartesian) {
        const Eigen::Matrix3d change = cell_ * cell.inverse();
        for (auto &position : pos_) position *= change;
    }
    cell_ = cell;
}
void Atoms::reorder(int *map)
{
    if (!map && !pos_.empty())
        throw std::invalid_argument("Atoms::reorder requires a nonnull permutation");
    reorder(std::span<const int>(map, pos_.size()));
}

void Atoms::reorder(std::span<const int> order)
{
    const size_t n = pos_.size();
    if (order.size() != n)
        throw std::invalid_argument("Atoms::reorder permutation length must equal the atom count");
    std::vector<bool> seen(n, false);
    for (int index : order) {
        if (index < 0 || size_t(index) >= n || seen[index])
            throw std::invalid_argument("Atoms::reorder requires a permutation of [0, atom count)");
        seen[index] = true;
    }
    if (!molid_.empty() || !molsize_.empty() || !mol_types_.empty() ||
        !bonds_.empty() || !angles_.empty() || have_ghost || !map_ghost_.empty())
        throw std::invalid_argument("Atoms::reorder does not support molecular topology or ghost layouts");
    if (chemmap_.size() != n)
        throw std::invalid_argument("Atoms::reorder chemistry size must equal the atom count");
    const auto check_vector = [n](const auto &values, const char *name) {
        if (!values.empty() && values.size() != n)
            throw std::invalid_argument(std::string("Atoms::reorder incomplete atom field: ") + name);
    };
    check_vector(ids_, "ids");
    check_vector(forces_, "forces");
    check_vector(velocities_, "velocities");
    check_vector(site_energies_, "site_energies");
    check_vector(charges_, "charges");
    check_vector(magmom_scalar_, "magmom_scalar");
    check_vector(magmoms_, "magmoms");
    check_vector(born_eff_chgs_, "born_eff_chgs");
    check_vector(atomic_fitness, "atomic_fitness");
    check_vector(masks, "masks");
    check_vector(sel_dynpos_, "sel_dynpos");
    const auto check_scalars = [n](const auto &attributes) {
        for (const auto &[name, values] : attributes)
            if (values.size() != 0 && size_t(values.size()) != n)
                throw std::invalid_argument("Atoms::reorder incomplete local scalar attribute: " + name);
    };
    const auto check_columns = [n](const auto &attributes) {
        for (const auto &[name, values] : attributes)
            if (values.cols() != 0 && size_t(values.cols()) != n)
                throw std::invalid_argument("Atoms::reorder incomplete local vector attribute: " + name);
    };
    check_scalars(local_attribs_d_);
    check_scalars(local_attribs_i_);
    check_columns(local_attribs_v_);
    check_columns(local_attribs_vi_);
    if (!fitting_factor.empty() && fitting_factor.size() != 1 &&
        fitting_factor.size() != n + 1 && fitting_factor.size() != n + 2)
        throw std::invalid_argument("Atoms::reorder invalid fitting-factor size");
    const Eigen::Index coordinates = 3 * Eigen::Index(n);
    if ((have_hessian || hessian_.rows() != 0 || hessian_.cols() != 0) &&
        (hessian_.rows() != coordinates || hessian_.cols() != coordinates))
        throw std::invalid_argument("Atoms::reorder Hessian must have shape 3N x 3N");
    validate_hvp();

    // Prepare every changed field before committing any of them. In particular,
    // order may alias ids_, and unrelated Atoms scalars need not be initialized.
    const auto permute_vector = [&order](const auto &values) {
        auto result = values;
        if (!values.empty())
            for (size_t i = 0; i < order.size(); ++i) result[i] = values[order[i]];
        return result;
    };
    const auto permute_scalars = [&order](const auto &attributes) {
        auto result = attributes;
        for (auto &[name, values] : result) {
            if (values.size() == 0) continue;
            const auto &old = attributes.at(name);
            for (size_t i = 0; i < order.size(); ++i) values(i) = old(order[i]);
        }
        return result;
    };
    const auto permute_columns = [&order](const auto &attributes) {
        auto result = attributes;
        for (auto &[name, values] : result) {
            if (values.cols() == 0) continue;
            const auto &old = attributes.at(name);
            for (size_t i = 0; i < order.size(); ++i) values.col(i) = old.col(order[i]);
        }
        return result;
    };
    auto positions = permute_vector(pos_);
    auto chemistry = permute_vector(chemmap_);
    auto ids = permute_vector(ids_);
    auto forces = permute_vector(forces_);
    auto velocities = permute_vector(velocities_);
    auto site_energies = permute_vector(site_energies_);
    auto charges = permute_vector(charges_);
    auto scalar_moments = permute_vector(magmom_scalar_);
    auto moments = permute_vector(magmoms_);
    auto born_charges = permute_vector(born_eff_chgs_);
    auto atom_weights = permute_vector(atomic_fitness);
    auto atom_masks = permute_vector(masks);
    auto selective_positions = permute_vector(sel_dynpos_);
    auto local_scalars = permute_scalars(local_attribs_d_);
    auto local_integers = permute_scalars(local_attribs_i_);
    auto local_vectors = permute_columns(local_attribs_v_);
    auto local_integer_vectors = permute_columns(local_attribs_vi_);
    auto fitting = fitting_factor;
    if (fitting.size() > 1)
        for (size_t i = 0; i < n; ++i) fitting[i + 1] = fitting_factor[order[i] + 1];
    Eigen::MatrixXd hessian = hessian_;
    if (hessian.size() != 0)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                hessian.block<3, 3>(3 * Eigen::Index(i), 3 * Eigen::Index(j)) =
                    hessian_.block<3, 3>(3 * Eigen::Index(order[i]), 3 * Eigen::Index(order[j]));
    Eigen::MatrixXd directions = hvp_directions_, targets = hvp_targets_;
    if (has_hvp())
        for (size_t i = 0; i < n; ++i) {
            directions.middleRows(3 * Eigen::Index(i), 3) = hvp_directions_.middleRows(3 * Eigen::Index(order[i]), 3);
            targets.middleRows(3 * Eigen::Index(i), 3) = hvp_targets_.middleRows(3 * Eigen::Index(order[i]), 3);
        }

    pos_.swap(positions);
    chemmap_.swap(chemistry);
    ids_.swap(ids);
    forces_.swap(forces);
    velocities_.swap(velocities);
    site_energies_.swap(site_energies);
    charges_.swap(charges);
    magmom_scalar_.swap(scalar_moments);
    magmoms_.swap(moments);
    born_eff_chgs_.swap(born_charges);
    atomic_fitness.swap(atom_weights);
    masks.swap(atom_masks);
    sel_dynpos_.swap(selective_positions);
    local_attribs_d_.swap(local_scalars);
    local_attribs_i_.swap(local_integers);
    local_attribs_v_.swap(local_vectors);
    local_attribs_vi_.swap(local_integer_vectors);
    fitting_factor.swap(fitting);
    hessian_.swap(hessian);
    hvp_directions_.swap(directions);
    hvp_targets_.swap(targets);
}

void Atoms::set_cell(Eigen::Matrix3d &cell)
{
    const bool was_cartesian = cartesian;
    if (was_cartesian) to_fractional();
    double d0 = cell_.determinant();
    double d1 = cell.determinant();
    if (d0 * d1 < 0) {
        Eigen::Matrix3d p = Eigen::Matrix3d::Identity();
        p(2, 2) = -1.0;
        cell_ = p * cell_;
        for (auto &t : pos_) {
            t *= p;
        }
    }
    cell_ = cell;
    if (was_cartesian) to_cartesian();
}

void Atoms::mkLower()
{
    to_lower_standard();
}

void Atoms::translate(Eigen::Vector3d vec)
{
    Eigen::RowVector3d displacement = vec.transpose();
    if (!cartesian) displacement *= cell_.inverse();
    for (auto &p : pos_) p += displacement;
}

void Atoms::to_fractional()
{
    auto invcell = cell_.inverse();
    for (int i = 0; i < size(); ++i)
        pos_[i] *= invcell;
    cartesian = false;
}

void Atoms::to_cartesian() 
{
    for (int i = 0; i < size(); ++i) 
        pos_[i] *= cell_;
    cartesian = true;
}

void Atoms::set_dir_pos(std::vector<Eigen::RowVector3d> pos)
{
    pos_.resize(pos.size());
    for (int i = 0; i < pos.size(); ++i)
    {
        pos_[i] = pos[i] * cell_;
    }
    cartesian = true;
}

void Atoms::to_lower_standard()
{
    if (!cell_.allFinite() || std::abs(cell_.determinant()) <= 1e-14)
        throw std::invalid_argument("Cell standardization requires a finite, non-singular cell");
    if (!cartesian) to_cartesian();
    to_righthand();
    const Eigen::Matrix3d old_cell = cell_;
    std::vector<double> lat = get_cell_parameters();
    set_cell(lat[0], lat[1], lat[2], lat[3], lat[4], lat[5]);
    const Eigen::Matrix3d rotation = (old_cell.inverse() * cell_).transpose();
    rotate_atom_properties(*this, rotation);
}

void Atoms::to_righthand() {
    if (cell_.determinant() < 0) {
        cell_.row(2) *= -1.0;
        if (!cartesian)
            for (auto &pos : pos_) pos[2] *= -1.0;
    }
}

Eigen::Matrix3d Atoms::standard_form(bool upper)
{
    int sign = cell_.determinant() > 0 ? 1 : -1;
    Eigen::Matrix3d Q, L;
    if (upper) {
        Eigen::Matrix3d crev = cell_.transpose().colwise().reverse().rowwise().reverse();
        Eigen::HouseholderQR<Eigen::Matrix3d> qr(crev);
        Eigen::Matrix3d Qh = qr.householderQ();
        Eigen::Matrix3d Lh = qr.matrixQR().triangularView<Eigen::Upper>();
        Q = Qh.transpose().colwise().reverse().rowwise().reverse();
        L = Lh.transpose().colwise().reverse().rowwise().reverse();
    } else [[likely]] {
        Eigen::HouseholderQR<Eigen::Matrix3d> qr(cell_.transpose());
        Eigen::Matrix3d Qh = qr.householderQ();
        Eigen::Matrix3d Lh = qr.matrixQR().triangularView<Eigen::Upper>();
        Q = Qh.transpose();
        L = Lh.transpose();
    }
    Eigen::Vector3d diag = L.diagonal();
    Eigen::Vector3d sgn;

    for (int i = 0; i < 3; ++i)
    {
        if (std::abs(diag(i)) < 1e-14)
            sgn(i) = 1.0;
        else
            sgn(i) = (diag(i) > 0 ? 1.0 : -1.0);
    }

    // enforce signs = handedness
    for (int i = 0; i < 3; ++i)
    {
        if (sgn(i) != sign) {
            L.col(i) *= -1;
            Q.row(i) *= -1;
        }
    }
    set_cell(L);
    rotate_atom_properties(*this, Q);
    return Q;
}

void Atoms::delete_atom(int &id) {
    clear_hvp();
    if (id < size()) {
        pos_.erase(pos_.begin() + id);
        chemmap_.erase(chemmap_.begin() + id);
        // if (has_forces()) {
        //     forces_.erase(forces_.begin() + id);
        // }
        // if (has_velo()) {
        //     velocities_.erase(velocities_.begin() + id);
        // }
        // if (has_site_energies()) {
        //     site_energies_.erase(site_energies_.begin() + id);
        // }
        // if (has_charges()) {
        //     charges_.erase(charges_.begin() + id);
        // }
        // if (has_magmom()) {
        //     magmom_scalar_.erase(magmom_scalar_.begin() + id);
        //     magmoms_.erase(magmoms_.begin() + id);
        // }
        // if (has_local_attribs_d()) {
        //     local_attribs_d_.erase(local_attribs_d_.begin() + id);
        // }
        // if (has_local_attribs_v()) {
        //     local_attribs_v_.erase(local_attribs_v_.begin() + id);
        // }
        // if (has_local_attribs_i()) {
        //     local_attribs_i_.erase(local_attribs_i_.begin() + id);
        // }
        // if (has_local_attribs_vi()) {
        //     local_attribs_vi_.erase(local_attribs_vi_.begin() + id);
        // }
    }
}

void Atoms::delete_atoms(std::vector<int> &inds) {
    clear_hvp();
    auto indt = inds;
    std::sort(indt.begin(), indt.end());
    auto it = indt.rbegin();
    for (; it != indt.rend(); ++it) {
        delete_atom(*it);
    }
}

Atoms::DuplicateAtomStats Atoms::remove_duplicate_atoms(double rtol, DuplicateAtomMode mode)
{
    if (rtol <= 0.0) {
        throw std::invalid_argument("Atoms::remove_duplicate_atoms rtol must be positive");
    }

    DuplicateAtomStats stats;
    stats.initial_atoms = size();
    stats.final_atoms = size();
    if (size() <= 1) return stats;

    auto graph = build_duplicate_atom_graph(*this, rtol);
    std::vector<char> visited(size(), 0);
    std::vector<char> remove(size(), 0);

    for (int i = 0; i < size(); ++i) {
        if (visited[i]) continue;
        auto component = make_duplicate_atom_component(*this, graph, i, visited);
        if (component.nodes.size() <= 1) continue;

        ++stats.duplicate_groups;
        if (mode == DuplicateAtomMode::Merge) {
            Eigen::RowVector3d merged_pos = component.mean;
            wrap_duplicate_atom_position(*this, merged_pos);
            pos_[component.representative] = merged_pos;
        }
        for (int id : component.nodes) {
            if (id != component.representative) {
                remove[id] = 1;
            }
        }
    }

    std::vector<int> keep;
    keep.reserve(size());
    for (int i = 0; i < size(); ++i) {
        if (!remove[i]) keep.push_back(i);
    }

    stats.removed_atoms = size() - static_cast<int>(keep.size());
    if (stats.removed_atoms > 0) {
        compact_duplicate_atoms(*this, keep, stats.initial_atoms);
    }
    stats.final_atoms = size();
    return stats;
}

Eigen::RowVector3d Atoms::get_cm(void) const {
    Eigen::RowVector3d cm;
    cm << 0,0,0;
    for (int i = 0; i < size(); ++i) {
        cm += pos_[i];
    }
    cm /= size();
    return cm;
}

std::istream& Atoms::read_from_bin(std::istream& is)
{
    // 读取基本的标量值
    is.read(reinterpret_cast<char*>(&cartesian), sizeof(cartesian));
    is.read(reinterpret_cast<char*>(&have_ghost), sizeof(have_ghost));
    is.read(reinterpret_cast<char*>(&nlocal_), sizeof(nlocal_));
    is.read(reinterpret_cast<char*>(&nghost_), sizeof(nghost_));
    is.read(reinterpret_cast<char*>(&molnum_), sizeof(molnum_));
    is.read(reinterpret_cast<char*>(&collinear_), sizeof(collinear_));
    is.read(reinterpret_cast<char*>(&have_energy), sizeof(have_energy));
    is.read(reinterpret_cast<char*>(&have_fenergy), sizeof(have_fenergy));
    is.read(reinterpret_cast<char*>(&have_stress), sizeof(have_stress));
    is.read(reinterpret_cast<char*>(&have_eff_chg), sizeof(have_eff_chg));
    is.read(reinterpret_cast<char*>(&group_count), sizeof(group_count));
    is.read(reinterpret_cast<char*>(&energy_), sizeof(energy_));
    is.read(reinterpret_cast<char*>(&fenergy_), sizeof(fenergy_));
    is.read(reinterpret_cast<char*>(&fitness), sizeof(fitness));
    is.read(reinterpret_cast<char*>(&mvgrade_), sizeof(mvgrade_));
    is.read(reinterpret_cast<char*>(&mindist_), sizeof(mindist_));
    is.read(reinterpret_cast<char*>(&maxdist_), sizeof(maxdist_));
    is.read(reinterpret_cast<char*>(&id), sizeof(id));

    // 读取Eigen矩阵
    is.read(reinterpret_cast<char*>(cell_.data()), sizeof(double) * 9);
    is.read(reinterpret_cast<char*>(stress_.data()), sizeof(double) * 9);
    is.read(reinterpret_cast<char*>(sel_dyncell_.data()), sizeof(double) * 9);

    // 读取向量
    size_t size;
    
    // pos_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    pos_.resize(size);
    for (auto& p : pos_)
        is.read(reinterpret_cast<char*>(p.data()), sizeof(double) * 3);
    
    // chemmap_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    chemmap_.resize(size);
    is.read(reinterpret_cast<char*>(chemmap_.data()), sizeof(int) * size);
    
    // map_ghost_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    map_ghost_.resize(size);
    is.read(reinterpret_cast<char*>(map_ghost_.data()), sizeof(int) * size);
    
    // ids_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    ids_.resize(size);
    is.read(reinterpret_cast<char*>(ids_.data()), sizeof(int) * size);
    
    // molid_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    molid_.resize(size);
    is.read(reinterpret_cast<char*>(molid_.data()), sizeof(int) * size);
    
    // molsize_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    molsize_.resize(size);
    is.read(reinterpret_cast<char*>(molsize_.data()), sizeof(int) * size);
    
    // mol_types_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    mol_types_.resize(size);
    is.read(reinterpret_cast<char*>(mol_types_.data()), sizeof(int) * size);
    
    // forces_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    forces_.resize(size);
    for (auto& f : forces_)
        is.read(reinterpret_cast<char*>(f.data()), sizeof(double) * 3);
    
    // velocities_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    velocities_.resize(size);
    for (auto& v : velocities_)
        is.read(reinterpret_cast<char*>(v.data()), sizeof(double) * 3);
    
    // site_energies_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    site_energies_.resize(size);
    is.read(reinterpret_cast<char*>(site_energies_.data()), sizeof(double) * size);
    
    // charges_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    charges_.resize(size);
    is.read(reinterpret_cast<char*>(charges_.data()), sizeof(double) * size);
    
    // magmom_scalar_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    magmom_scalar_.resize(size);
    is.read(reinterpret_cast<char*>(magmom_scalar_.data()), sizeof(double) * size);
    
    // magmoms_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    magmoms_.resize(size);
    for (auto& m : magmoms_)
        is.read(reinterpret_cast<char*>(m.data()), sizeof(double) * 3);
    
    // born_eff_chgs_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    born_eff_chgs_.resize(size);
    for (auto& b : born_eff_chgs_)
        is.read(reinterpret_cast<char*>(b.data()), sizeof(double) * 9);
    
    // fitting_factor
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    fitting_factor.resize(size);
    is.read(reinterpret_cast<char*>(fitting_factor.data()), sizeof(double) * size);
    
    // atomic_fitness
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    atomic_fitness.resize(size);
    is.read(reinterpret_cast<char*>(atomic_fitness.data()), sizeof(double) * size);
    
    // masks
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    masks.resize(size);
    is.read(reinterpret_cast<char*>(masks.data()), sizeof(int) * size);
    
    // sel_dynpos_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    sel_dynpos_.resize(size);
    for (auto& s : sel_dynpos_)
        is.read(reinterpret_cast<char*>(s.data()), sizeof(double) * 3);
    
    // 读取map容器
    
    // global_attribs_d_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    global_attribs_d_.clear();
    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        is.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        is.read(&key[0], key_size);
        double value;
        is.read(reinterpret_cast<char*>(&value), sizeof(value));
        global_attribs_d_[key] = value;
    }
    
    // local_attribs_d_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    local_attribs_d_.clear();
    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        is.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        is.read(&key[0], key_size);
        size_t vec_size;
        is.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
        Eigen::VectorXd value(vec_size);
        is.read(reinterpret_cast<char*>(value.data()), sizeof(double) * vec_size);
        local_attribs_d_[key] = value;
    }
    
    // global_attribs_v_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    global_attribs_v_.clear();
    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        is.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        is.read(&key[0], key_size);
        size_t vec_size;
        is.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
        Eigen::VectorXd value(vec_size);
        is.read(reinterpret_cast<char*>(value.data()), sizeof(double) * vec_size);
        global_attribs_v_[key] = value;
    }
    
    // local_attribs_v_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    local_attribs_v_.clear();
    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        is.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        is.read(&key[0], key_size);
        int rows, cols;
        is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        Eigen::MatrixXd value(rows, cols);
        is.read(reinterpret_cast<char*>(value.data()), sizeof(double) * rows * cols);
        local_attribs_v_[key] = value;
    }
    
    // global_attribs_i_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    global_attribs_i_.clear();
    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        is.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        is.read(&key[0], key_size);
        int value;
        is.read(reinterpret_cast<char*>(&value), sizeof(value));
        global_attribs_i_[key] = value;
    }
    
    // global_attribs_vi_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    global_attribs_vi_.clear();
    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        is.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        is.read(&key[0], key_size);
        size_t vec_size;
        is.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
        Eigen::VectorXi value(vec_size);
        is.read(reinterpret_cast<char*>(value.data()), sizeof(int) * vec_size);
        global_attribs_vi_[key] = value;
    }
    
    // local_attribs_i_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    local_attribs_i_.clear();
    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        is.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        is.read(&key[0], key_size);
        size_t vec_size;
        is.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
        Eigen::VectorXi value(vec_size);
        is.read(reinterpret_cast<char*>(value.data()), sizeof(int) * vec_size);
        local_attribs_i_[key] = value;
    }
    
    // local_attribs_vi_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    local_attribs_vi_.clear();
    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        is.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        is.read(&key[0], key_size);
        int rows, cols;
        is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        Eigen::MatrixXi value(rows, cols);
        is.read(reinterpret_cast<char*>(value.data()), sizeof(int) * rows * cols);
        local_attribs_vi_[key] = value;
    }
    
    // features_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    features_.clear();
    for (size_t i = 0; i < size; ++i)
    {
        size_t key_size;
        is.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        std::string key(key_size, ' ');
        is.read(&key[0], key_size);
        size_t value_size;
        is.read(reinterpret_cast<char*>(&value_size), sizeof(value_size));
        std::string value(value_size, ' ');
        is.read(&value[0], value_size);
        features_[key] = value;
    }
    
    // 读取bonds_和angles_
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    bonds_.resize(size);
    is.read(reinterpret_cast<char*>(bonds_.data()), sizeof(std::array<int, 2>) * size);
    
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    angles_.resize(size);
    is.read(reinterpret_cast<char*>(angles_.data()), sizeof(std::array<int, 3>) * size);
    
    return is;
}

std::ostream& Atoms::write_to_bin(std::ostream& os) const
{
    // 写入基本的标量值
    os.write(reinterpret_cast<const char*>(&cartesian), sizeof(cartesian));
    os.write(reinterpret_cast<const char*>(&have_ghost), sizeof(have_ghost));
    os.write(reinterpret_cast<const char*>(&nlocal_), sizeof(nlocal_));
    os.write(reinterpret_cast<const char*>(&nghost_), sizeof(nghost_));
    os.write(reinterpret_cast<const char*>(&molnum_), sizeof(molnum_));
    os.write(reinterpret_cast<const char*>(&collinear_), sizeof(collinear_));
    os.write(reinterpret_cast<const char*>(&have_energy), sizeof(have_energy));
    os.write(reinterpret_cast<const char*>(&have_fenergy), sizeof(have_fenergy));
    os.write(reinterpret_cast<const char*>(&have_stress), sizeof(have_stress));
    os.write(reinterpret_cast<const char*>(&have_eff_chg), sizeof(have_eff_chg));
    os.write(reinterpret_cast<const char*>(&group_count), sizeof(group_count));
    os.write(reinterpret_cast<const char*>(&energy_), sizeof(energy_));
    os.write(reinterpret_cast<const char*>(&fenergy_), sizeof(fenergy_));
    os.write(reinterpret_cast<const char*>(&fitness), sizeof(fitness));
    os.write(reinterpret_cast<const char*>(&mvgrade_), sizeof(mvgrade_));
    os.write(reinterpret_cast<const char*>(&mindist_), sizeof(mindist_));
    os.write(reinterpret_cast<const char*>(&maxdist_), sizeof(maxdist_));
    os.write(reinterpret_cast<const char*>(&id), sizeof(id));

    // 写入Eigen矩阵
    os.write(reinterpret_cast<const char*>(cell_.data()), sizeof(double) * 9);
    os.write(reinterpret_cast<const char*>(stress_.data()), sizeof(double) * 9);
    os.write(reinterpret_cast<const char*>(sel_dyncell_.data()), sizeof(double) * 9);

    // 写入向量
    size_t size;
    
    // pos_
    size = pos_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& p : pos_)
        os.write(reinterpret_cast<const char*>(p.data()), sizeof(double) * 3);
    
    // chemmap_
    size = chemmap_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(chemmap_.data()), sizeof(int) * size);
    
    // map_ghost_
    size = map_ghost_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(map_ghost_.data()), sizeof(int) * size);
    
    // ids_
    size = ids_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(ids_.data()), sizeof(int) * size);
    
    // molid_
    size = molid_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(molid_.data()), sizeof(int) * size);
    
    // molsize_
    size = molsize_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(molsize_.data()), sizeof(int) * size);
    
    // mol_types_
    size = mol_types_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(mol_types_.data()), sizeof(int) * size);
    
    // forces_
    size = forces_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& f : forces_)
        os.write(reinterpret_cast<const char*>(f.data()), sizeof(double) * 3);
    
    // velocities_
    size = velocities_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& v : velocities_)
        os.write(reinterpret_cast<const char*>(v.data()), sizeof(double) * 3);
    
    // site_energies_
    size = site_energies_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(site_energies_.data()), sizeof(double) * size);
    
    // charges_
    size = charges_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(charges_.data()), sizeof(double) * size);
    
    // magmom_scalar_
    size = magmom_scalar_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(magmom_scalar_.data()), sizeof(double) * size);
    
    // magmoms_
    size = magmoms_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& m : magmoms_)
        os.write(reinterpret_cast<const char*>(m.data()), sizeof(double) * 3);
    
    // born_eff_chgs_
    size = born_eff_chgs_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& b : born_eff_chgs_)
        os.write(reinterpret_cast<const char*>(b.data()), sizeof(double) * 9);
    
    // fitting_factor
    size = fitting_factor.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(fitting_factor.data()), sizeof(double) * size);
    
    // atomic_fitness
    size = atomic_fitness.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(atomic_fitness.data()), sizeof(double) * size);
    
    // masks
    size = masks.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(masks.data()), sizeof(int) * size);
    
    // sel_dynpos_
    size = sel_dynpos_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& s : sel_dynpos_)
        os.write(reinterpret_cast<const char*>(s.data()), sizeof(double) * 3);
    
    // 写入map容器
    
    // global_attribs_d_
    size = global_attribs_d_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : global_attribs_d_)
    {
        size_t key_size = pair.first.size();
        os.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        os.write(pair.first.c_str(), key_size);
        os.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }
    
    // local_attribs_d_
    size = local_attribs_d_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : local_attribs_d_)
    {
        size_t key_size = pair.first.size();
        os.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        os.write(pair.first.c_str(), key_size);
        size_t vec_size = pair.second.size();
        os.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
        os.write(reinterpret_cast<const char*>(pair.second.data()), sizeof(double) * vec_size);
    }
    
    // global_attribs_v_
    size = global_attribs_v_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : global_attribs_v_)
    {
        size_t key_size = pair.first.size();
        os.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        os.write(pair.first.c_str(), key_size);
        size_t vec_size = pair.second.size();
        os.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
        os.write(reinterpret_cast<const char*>(pair.second.data()), sizeof(double) * vec_size);
    }
    
    // local_attribs_v_
    size = local_attribs_v_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : local_attribs_v_)
    {
        size_t key_size = pair.first.size();
        os.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        os.write(pair.first.c_str(), key_size);
        int rows = pair.second.rows();
        int cols = pair.second.cols();
        os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        os.write(reinterpret_cast<const char*>(pair.second.data()), sizeof(double) * rows * cols);
    }
    
    // global_attribs_i_
    size = global_attribs_i_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : global_attribs_i_)
    {
        size_t key_size = pair.first.size();
        os.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        os.write(pair.first.c_str(), key_size);
        os.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }
    
    // global_attribs_vi_
    size = global_attribs_vi_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : global_attribs_vi_)
    {
        size_t key_size = pair.first.size();
        os.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        os.write(pair.first.c_str(), key_size);
        size_t vec_size = pair.second.size();
        os.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
        os.write(reinterpret_cast<const char*>(pair.second.data()), sizeof(int) * vec_size);
    }
    
    // local_attribs_i_
    size = local_attribs_i_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : local_attribs_i_)
    {
        size_t key_size = pair.first.size();
        os.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        os.write(pair.first.c_str(), key_size);
        size_t vec_size = pair.second.size();
        os.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
        os.write(reinterpret_cast<const char*>(pair.second.data()), sizeof(int) * vec_size);
    }
    
    // local_attribs_vi_
    size = local_attribs_vi_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : local_attribs_vi_)
    {
        size_t key_size = pair.first.size();
        os.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        os.write(pair.first.c_str(), key_size);
        int rows = pair.second.rows();
        int cols = pair.second.cols();
        os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        os.write(reinterpret_cast<const char*>(pair.second.data()), sizeof(int) * rows * cols);
    }
    
    // features_
    size = features_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const auto& pair : features_)
    {
        size_t key_size = pair.first.size();
        os.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        os.write(pair.first.c_str(), key_size);
        size_t value_size = pair.second.size();
        os.write(reinterpret_cast<const char*>(&value_size), sizeof(value_size));
        os.write(pair.second.c_str(), value_size);
    }
    
    // 写入bonds_和angles_
    size = bonds_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(bonds_.data()), sizeof(std::array<int, 2>) * size);
    
    size = angles_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    os.write(reinterpret_cast<const char*>(angles_.data()), sizeof(std::array<int, 3>) * size);
    
    return os;
}

double Atoms::get_distance(int a, int b) {
    return get_rab(a, b).norm();
}

Eigen::RowVector3d Atoms::get_rab(int a, int b) {
    Eigen::RowVector3d delta = pos_[b] - pos_[a];
    if (!cartesian) delta = (delta * cell_).eval();
    const int rank = static_cast<int>(pbc_[0]) + static_cast<int>(pbc_[1])
        + static_cast<int>(pbc_[2]);
    if (rank == 0 || a == b) return delta;

    Eigen::MatrixXd basis(rank, 3);
    int row = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (pbc_[axis]) basis.row(row++) = cell_.row(axis);
    }
    if (!basis.allFinite())
        throw std::runtime_error("Minimum distance requires finite periodic cell vectors");
    const Eigen::MatrixXd gram = basis * basis.transpose();
    if (!gram.allFinite())
        throw std::runtime_error("Minimum-distance periodic metric is not finite");
    const auto decomposition = gram.fullPivLu();
    if (!decomposition.isInvertible())
        throw std::runtime_error("Minimum distance requires independent finite periodic cell vectors");
    const Eigen::MatrixXd dual = decomposition.solve(basis);
    if (!dual.allFinite() || !std::isfinite(delta.squaredNorm()))
        throw std::runtime_error("Minimum-distance periodic dual is not finite");
    Eigen::RowVector3d result = delta;
    periodic_minimum_squared(delta, basis, dual, delta.squaredNorm(), false, &result);
    return result;
}
// currently hosted by Neighborhood class.
// this is a slow version
void Atoms::build_ghost(double r) {
    double rsq = r * r;
    Eigen::Vector3d max_pos = pos_[0];
    Eigen::Vector3d min_pos = pos_[0];
    for (int i = 0; i < size(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (min_pos[j] > pos_[i][j])
                min_pos[j] = pos_[i][j];
            if (max_pos[j] < pos_[i][j])
                max_pos[j] = pos_[i][j];
        }
    }

    bool add_new = true;
    int ghostid = 0;
    for (int l = 1; add_new; ++l)
    {
        add_new = false;

        for (int i = -l; i <= l; ++i)
        {
            for (int j = -l; j <= l; ++j)
            {
                for (int k = -l; k <= l; ++k)
                {
                    if (__max(__max(abs(i), abs(j)), abs(k)) == l)
                    {
                        for (int ind = 0; ind < size(); ind++)
                        {
                            Eigen::RowVector3d newvec = pos_[ind];
                            newvec += i * cell_.row(0) +
                                      j * cell_.row(1) +
                                      k * cell_.row(2);

                            if ((newvec[0] < min_pos[0] - r) ||
                                (newvec[1] < min_pos[1] - r) ||
                                (newvec[2] < min_pos[2] - r) ||
                                (newvec[0] > max_pos[0] + r) ||
                                (newvec[1] > max_pos[1] + r) ||
                                (newvec[2] > max_pos[2] + r))
                                continue;
                            //@note:by IF NONBOX, REDUCE CONSTRUCTING TIME
                            int jnd;
                            for (jnd = 0;
                                 (jnd < size()) && (pos_[jnd] - newvec).squaredNorm() > rsq;
                                 jnd++)
                                ;
                            if (jnd < size())
                            {
                                pos_.emplace_back(newvec);
                                // indices_orig.emplace_back(ind);
                                map_ghost_.push_back(ind);
                                ghostid++;
                                add_new = true;
                            }
                        }
                    }
                }
            }
        }
    }
}

Atoms Atoms::create_supercell(const Eigen::Matrix3i &mat, std::map<int, int> &s2p) const {
    const int determinant = mat.determinant();
    if (determinant == 0)
        throw std::invalid_argument("Supercell matrix must be nonsingular");
    Atoms supercell;

    Eigen::Matrix3d rmatd = mat.cast<double>().inverse();
    Eigen::Matrix3d newcell = mat.cast<double>() * cell_;
    int nexpand = std::abs(determinant);
    int newsize = nexpand * size();
    int csize = size();
    supercell.pos_.reserve(newsize);
    supercell.chemmap_.reserve(newsize);

    /*Find lattice points in supercell*/
    std::vector<Eigen::RowVector3d> latpoints;
    Eigen::Matrix<double, 8, 3> boundvecs;
    boundvecs << 0, 0, 0,
        0, 0, 1,
        0, 1, 0,
        0, 1, 1,
        1, 0, 0,
        1, 0, 1,
        1, 1, 0,
        1, 1, 1;
    // for (int i = 0; i < 8; ++i)
    // {
    //     boundvecs.row(i) *= mat;
    // }
    boundvecs = boundvecs * mat.cast<double>();
    Eigen::RowVector3i maxi, mini;
    maxi = boundvecs.row(0).cast<int>();
    mini = maxi;
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (maxi(j) < boundvecs(i, j))
            {
                maxi(j) = boundvecs(i, j);
            }
            if (mini(j) > boundvecs(i, j))
            {
                mini(j) = boundvecs(i, j);
            }
        }
    }
    // _DEBUG_OUTPUT_(maxi);
    // _DEBUG_OUTPUT_(mini);
    // for (int j = 0; j < 3; ++j)
    // {
    //     maxi(j) += 1;
    // }
    latpoints.resize(nexpand);
    for (int ii = 0; ii < 3; ii++)
    {
        for (int jj = 0; jj < 3; jj++)
        {
            if (fabs(newcell(ii, jj)) < ROUND_PREC)
            {
                newcell(ii, jj) = 0;
            }
        }
    }
    int ct = 0;
    Eigen::RowVector3d po;
    for (int i = mini(0); i < 1 + maxi(0); ++i)
    {
        for (int j = mini(1); j < 1 + maxi(1); ++j)
        {
            for (int k = mini(2); k < 1 + maxi(2); ++k)
            {
                // if ((int)round(mat(2, 1) * k - mat(2, 2) * j) > 0)
                //     continue;
                // if ((int)round(mat(1, 1) * mat(2, 2) * i - mat(1, 0) * mat(2, 2) * j + (mat(1, 0) * mat(2, 1) - mat(2, 0) * mat(1, 1)) * k) < 0)
                //     continue;
                po << i, j, k;
                po = po * rmatd;
                //po = po * newcell.inverse();
                if (!(po(0) <= 1 - ROUND_PREC && po(1) <= 1 - ROUND_PREC && po(2) <= 1 - ROUND_PREC))
                    continue;
                if (!(po(0) >= -ROUND_PREC && po(1) >= -ROUND_PREC && po(2) >= -ROUND_PREC))
                    continue;
                latpoints[ct] = po * newcell;
                ++ct;
            }
        }
    }
    /*Build positions*/
    std::vector<Eigen::RowVector3d> posold = pos_;
    if (!cartesian)
        for (auto &position : posold) position = dir2car(position);
    std::vector<int> chemmapold = chemmap_;
    supercell.pos_.clear();
    s2p.clear();
    /*Put as this
    Original atoms
    Original atoms with off set 1,2,3..., same as in enum.x
    */
    int iat = 0;
    for (int j = 0; j < nexpand; ++j)
    {
        for (int i = 0; i < csize; ++i)
        {
            supercell.pos_.emplace_back(posold[i] + latpoints[j]);
            supercell.chemmap_.emplace_back(chemmapold[i]);
            s2p[iat] = i;
            ++iat;
        }
    }
    supercell.cell_ = newcell;
    supercell.cartesian = true;
    supercell.pbc_ = pbc_;

    return supercell;
}

Eigen::VectorXd Atoms::get_voigt_stress() const{
    Eigen::VectorXd voigt(6);
    voigt(0) = stress_(0,0);
    voigt(1) = stress_(1,1);
    voigt(2) = stress_(2,2);
    voigt(3) = stress_(1,2);
    voigt(4) = stress_(0,2);
    voigt(5) = stress_(0,1);
    return voigt;
}

void Atoms::swap_axis(int mode, bool keephand) {
    Eigen::Matrix3i mat = Eigen::Matrix3i::Zero();
    // negative value to keep handedness
    switch (mode) {
    case 123:
        return;
    case 132:
        mat(0, 0) = 1;
        mat(1, 2) = 1;
        mat(2, 1) = keephand ? -1 : 1;
        break;
    case 213:
        mat(0, 1) = 1;
        mat(1, 0) = keephand ? -1 : 1;
        mat(2, 2) = 1;
        break;
    case 231:
        mat(0, 1) = 1;
        mat(1, 2) = 1;
        mat(2, 0) = 1;
        break;
    case 312:
        mat(0, 2) = 1;
        mat(1, 0) = 1;
        mat(2, 1) = 1;
        break;
    case 321:
        mat(0, 2) = 1;
        mat(1, 1) = 1;
        mat(2, 0) = keephand ? -1 : 1;
        break;
    default:
        return;
    }
    mksuperL(mat);
    const auto old_pbc = pbc_;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (mat(i, j) != 0) pbc_[i] = old_pbc[j];
}
#ifdef EXTERN_EXPCLIT_TEMPLATE
template class std::vector<Atoms>;
#endif
