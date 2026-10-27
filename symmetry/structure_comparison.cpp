// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "structure_comparison.h"

#include "base/neighborhoodKD.h"
#include "external/spglib_driver.h"
#include "potentials/quick_mtp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Symmetry3D {

StructureComparator::StructureComparator(StructureComparisonOptions options) : options_(options)
{
    for (double value : {options.symmetry_tolerance, options.position_tolerance,
                         options.volume_tolerance, options.descriptor_tolerance,
                         options.anchor_descriptor_tolerance})
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument("Structure comparison tolerances must be finite and positive");
    if (options.use_descriptors) descriptor_potential_.reset(Potentials::quick_mtp_3b_s14_2f0ee25e());
}

void StructureComparator::describe(ComparableStructure &structure) const
{
    if (!options_.use_descriptors) return;
    descriptor_potential_->CalcLocalDescriptors(structure.atoms, structure.descriptors);
    structure.fingerprint = structure.descriptors.rowwise().sum();
}

ComparableStructure StructureComparator::prepare(Atoms atoms)
{
    if (atoms.size() == 0 || atoms.chemmap_.size() != atoms.pos_.size() ||
        !atoms.cell_.allFinite() || std::abs(atoms.cell_.determinant()) < 1e-14)
        throw std::invalid_argument("Structure comparison requires nonempty structures, matching species, and nonsingular cells");
    for (const auto &position : atoms.pos_)
        if (!position.allFinite())
            throw std::invalid_argument("Structure comparison requires finite positions");
    if (!atoms.cartesian) atoms.to_cartesian();
    // Reduction changes the atom count; per-atom results from the input must not
    // survive on the comparison geometry. Keep original structures at the caller.
    Atoms geometry;
    geometry.cell_ = atoms.cell_;
    geometry.pos_ = std::move(atoms.pos_);
    geometry.chemmap_ = std::move(atoms.chemmap_);
    geometry.pbc_ = atoms.pbc_;
    if (geometry.is_fully_periodic()) {
        if (!standardize_cell(geometry, 1, 1, options_.symmetry_tolerance) ||
            !niggli_reduce(geometry, options_.symmetry_tolerance))
            throw std::runtime_error("Cannot reduce structure for comparison");
    }
    geometry.wrap();
    ComparableStructure result;
    result.atoms = std::move(geometry);
    std::vector<symopt_type> operations;
    if (!get_box_symmetry_operations(result.atoms.cell_, operations, options_.symmetry_tolerance))
        throw std::runtime_error("Cannot identify comparison cell symmetry");
    for (const auto &operation : operations) {
        // Periodic translations may only map into the periodic subspace.
        bool preserves_pbc = true;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (result.atoms.pbc_[i] && !result.atoms.pbc_[j] && operation.first(i, j) != 0)
                    preserves_pbc = false;
        if (preserves_pbc) result.lattice_operations.push_back(operation.first);
    }
    result.composition = result.atoms.types_count_map();
    describe(result);
    return result;
}

bool StructureComparator::equivalent(const ComparableStructure &first,
                                      const ComparableStructure &second) const
{
    if (first.atoms.size() == 0 || first.composition != second.composition ||
        first.atoms.size() != second.atoms.size() ||
        std::count(first.atoms.pbc_.begin(), first.atoms.pbc_.end(), true) !=
        std::count(second.atoms.pbc_.begin(), second.atoms.pbc_.end(), true)) return false;
    ComparableStructure scaled;
    const ComparableStructure *other = &second;
    if (options_.scale_volume) {
        scaled = second;
        scaled.atoms.scale_to_volume(std::abs(first.atoms.get_volume()));
        // Use a local potential so comparisons remain safe to run in parallel.
        if (options_.use_descriptors) {
            std::unique_ptr<Potentials::PotMTP> pot(Potentials::quick_mtp_3b_s14_2f0ee25e());
            pot->CalcLocalDescriptors(scaled.atoms, scaled.descriptors);
            scaled.fingerprint = scaled.descriptors.rowwise().sum();
        }
        other = &scaled;
    } else if (std::abs(std::abs(first.atoms.get_volume()) - std::abs(second.atoms.get_volume())) >
               options_.volume_tolerance) return false;
    if (options_.use_descriptors &&
        (first.fingerprint - other->fingerprint).norm() > options_.descriptor_tolerance) return false;

    const int anchor_type = std::min_element(first.composition.begin(), first.composition.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; })->first;
    Atoms reference = first.atoms;
    int anchor = 0;
    while (reference.chemmap_[anchor] != anchor_type) ++anchor;
    const Eigen::RowVector3d origin = reference.pos_[anchor];
    for (auto &position : reference.pos_) position -= origin;
    reference.wrap();
    NeighborhoodKD tree(&reference, options_.position_tolerance);
    const Eigen::Matrix3d metric = reference.cell_ * reference.cell_.transpose();
    for (int axes : {123, 132, 213, 231, 312, 321}) {
        Atoms candidate = other->atoms;
        candidate.swap_axis(axes);
        if (candidate.pbc_ != reference.pbc_) continue;
        const double scale = std::max(1.0, reference.cell_.norm() + candidate.cell_.norm());
        if ((metric - candidate.cell_ * candidate.cell_.transpose()).norm() >
            3 * options_.symmetry_tolerance * scale) continue;
        std::vector<Eigen::RowVector3d> positions;
        candidate.get_dir_pos(positions);
        for (const auto &operation : first.lattice_operations) {
            const double determinant = candidate.cell_.determinant() *
                                       operation.determinant() * reference.cell_.determinant();
            if (!options_.allow_reflections && determinant < 0) continue;
            for (int jat = 0; jat < candidate.size(); ++jat) {
                if (candidate.chemmap_[jat] != anchor_type) continue;
                if (options_.use_descriptors && !first.descriptors.col(anchor).isApprox(
                    other->descriptors.col(jat), options_.anchor_descriptor_tolerance)) continue;
                std::vector<std::vector<int>> matches(candidate.size());
                bool possible = true;
                for (int i = 0; i < candidate.size(); ++i) {
                    Eigen::RowVector3d position =
                        ((positions[i] - positions[jat]) * operation) * reference.cell_;
                    reference.wrap(position);
                    const auto neighbors = tree.query(position.data(), options_.position_tolerance,
                                                      candidate.chemmap_[i], true);
                    for (int n = 0; n < neighbors.nbhcount_; ++n) {
                        const int id = neighbors.indices_[n];
                        if (reference.chemmap_[id] == candidate.chemmap_[i]) matches[i].push_back(id);
                    }
                    if (matches[i].empty()) { possible = false; break; }
                }
                if (!possible) continue;
                // A bipartite match enforces a bijection even when tolerance spheres overlap.
                std::vector<int> owner(reference.size(), -1);
                const auto assign = [&](auto &&self, int i, std::vector<bool> &seen) -> bool {
                    for (int id : matches[i]) {
                        if (seen[id]) continue;
                        seen[id] = true;
                        if (owner[id] < 0 || self(self, owner[id], seen)) {
                            owner[id] = i;
                            return true;
                        }
                    }
                    return false;
                };
                bool same = true;
                for (int i = 0; i < candidate.size() && same; ++i) {
                    std::vector<bool> seen(reference.size(), false);
                    same = assign(assign, i, seen);
                }
                if (same) return true;
            }
        }
    }
    return false;
}

} // namespace Symmetry3D
