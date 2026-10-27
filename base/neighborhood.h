// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file neighborhood.h
 * @brief Default concrete neighborhood implementation using CRTP.
 */

#pragma once

#include "anyneighborhoods.h"
#include "neighborhoods_base.h"

/**
 * @brief Concrete neighborhood builder using CRTP with NeighborhoodsBase.
 * @ingroup core
 *
 * Serves as the default neighborhood implementation. The CRTP pattern
 * (NeighborhoodsBase<Neighborhoods, AnyNeighborhoods>) provides the full
 * extent/reset/init pipeline with AnyNeighborhoods as the type-erased base.
 *
 * Multiple neighborhood implementations exist (KD-tree, Voronoi, cluster,
 * shell, ghost) — this is the standard one used throughout the codebase.
 *
 * @see NeighborhoodsBase, AnyNeighborhoods
 */
class Neighborhoods : public NeighborhoodsBase<Neighborhoods, AnyNeighborhoods>
{
public:
    /// @brief Build neighbor lists for given structure and cutoff radius
    void construct(const Atoms &atom, const double cutoff);

    Neighborhoods() = default;
    ~Neighborhoods() = default;

    /// @brief Construct and initialize neighbor lists in one call
    Neighborhoods(const Atoms &atom, const double cutoff);

    /// @brief Reset and rebuild neighbor lists
    void init(const Atoms &atom, const double cutoff);
};
