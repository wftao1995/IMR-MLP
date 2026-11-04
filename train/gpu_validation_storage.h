// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base/neighborhood.h"
#include "base/structures.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace Training
{

inline size_t gpu_validation_checked_add(size_t a, size_t b)
{
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::length_error("GPU validation owned storage size overflow");
    return a + b;
}

inline size_t gpu_validation_checked_mul(size_t a, size_t b)
{
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::length_error("GPU validation owned storage size overflow");
    return a * b;
}

inline size_t gpu_validation_matrix_bytes(size_t rows, size_t columns, size_t scalar_bytes)
{
    return gpu_validation_checked_mul(gpu_validation_checked_mul(rows, columns), scalar_bytes);
}

namespace GpuValidationStorageDetail
{

template <class T, class Allocator>
size_t vector_bytes(const std::vector<T, Allocator> &value, bool copied)
{
    return gpu_validation_checked_mul(copied ? value.size() : value.capacity(), sizeof(T));
}

inline size_t string_bytes(const std::string &value, bool copied)
{
    // Inline short-string characters are already in sizeof(string). A fresh
    // copy reserves size+1 conservatively, even when that copy can use SSO.
    if (!copied)
    {
        const auto object = reinterpret_cast<std::uintptr_t>(&value);
        const auto data = reinterpret_cast<std::uintptr_t>(value.data());
        if (data >= object && data - object < sizeof(value))
            return 0;
    }
    return gpu_validation_checked_add(copied ? value.size() : value.capacity(), 1);
}

template <class T> size_t value_bytes(const T &value, bool copied)
{
    if constexpr (std::is_arithmetic_v<T>)
    {
        return 0; // The node owns the scalar inline; do not read its value.
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
        return string_bytes(value, copied);
    }
    else
    {
        if (value.rows() < 0 || value.cols() < 0)
            throw std::length_error("GPU validation matrix has negative dimensions");
        return gpu_validation_matrix_bytes(size_t(value.rows()), size_t(value.cols()),
                                           sizeof(typename T::Scalar));
    }
}

template <class Map> size_t map_bytes(const Map &value, bool copied)
{
    // Conservative tree-node bookkeeping, separate from the inline value.
    // This accounts owned container storage, not malloc/runtime/RSS overhead.
    // Keep this policy alongside the complete Atoms member inventory below.
    using Entry = typename Map::value_type;
    constexpr size_t overhead = 8 * sizeof(void *) + 2 * sizeof(size_t) + alignof(Entry) - 1;
    size_t bytes = gpu_validation_checked_mul(value.size(),
                                              gpu_validation_checked_add(sizeof(Entry), overhead));
    for (const auto &[key, item] : value)
    {
        bytes = gpu_validation_checked_add(bytes, string_bytes(key, copied));
        bytes = gpu_validation_checked_add(bytes, value_bytes(item, copied));
    }
    return bytes;
}

inline size_t atom_dynamic_bytes(const Atoms &a, bool copied)
{
    size_t bytes = 0;
    const auto vector = [&](const auto &v)
    {
        bytes = gpu_validation_checked_add(bytes, vector_bytes(v, copied));
    };
    vector(a.pos_);
    vector(a.chemmap_);
    vector(a.map_ghost_);
    vector(a.ids_);
    vector(a.molid_);
    vector(a.molsize_);
    vector(a.mol_types_);
    vector(a.bonds_);
    vector(a.angles_);
    vector(a.forces_);
    vector(a.velocities_);
    vector(a.site_energies_);
    vector(a.charges_);
    vector(a.magmom_scalar_);
    vector(a.magmoms_);
    vector(a.born_eff_chgs_);
    vector(a.fitting_factor);
    vector(a.atomic_fitness);
    vector(a.masks);
    vector(a.sel_dynpos_);
    for (const auto *matrix : {&a.hessian_, &a.hvp_directions_, &a.hvp_targets_})
        bytes = gpu_validation_checked_add(bytes, value_bytes(*matrix, copied));
    const auto map = [&](const auto &v)
    {
        bytes = gpu_validation_checked_add(bytes, map_bytes(v, copied));
    };
    map(a.global_attribs_d_);
    map(a.local_attribs_d_);
    map(a.global_attribs_v_);
    map(a.local_attribs_v_);
    map(a.global_attribs_i_);
    map(a.global_attribs_vi_);
    map(a.local_attribs_i_);
    map(a.local_attribs_vi_);
    map(a.features_);
    return bytes;
}

} // namespace GpuValidationStorageDetail

// Complete dynamic payload of one borrowed Atoms, without constructing a
// temporary vector or copying labels. Add its inline storage separately.
inline size_t gpu_validation_atom_dynamic_bytes(const Atoms &atom, bool copied = false)
{
    return GpuValidationStorageDetail::atom_dynamic_bytes(atom, copied);
}

// Include the outer vector backing and every dynamic Atoms member. copied=true
// preflights a fresh, direct vector copy (no growth or in-place assignment).
// It does not validate labels or read numerical payloads. The caller adds old
// and new owners and execution workspace before allocating the staged copy.
inline size_t gpu_validation_atoms_bytes(const std::vector<Atoms> &atoms, bool copied = false)
{
    size_t bytes = GpuValidationStorageDetail::vector_bytes(atoms, copied);
    for (const auto &a : atoms)
        bytes = gpu_validation_checked_add(bytes, gpu_validation_atom_dynamic_bytes(a, copied));
    return bytes;
}

inline size_t gpu_validation_neighbors_bytes(const std::vector<Neighborhoods> &neighbors,
                                             bool copied = false)
{
    size_t bytes = GpuValidationStorageDetail::vector_bytes(neighbors, copied);
    for (const auto &n : neighbors)
        bytes =
            gpu_validation_checked_add(bytes, n.owned_memory_bytes(copied) - sizeof(Neighborhoods));
    return bytes;
}

} // namespace Training
