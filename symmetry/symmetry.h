// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file symmetry.h
 * @brief Space group analysis and symmetry operations via spglib.
 * @ingroup symmetry
 */
#pragma once
#include "base/structures.h"
#include "external/spglib_driver.h"
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <cstdint>

/// @brief 3D symmetry analysis: space groups, Wyckoff positions, cell standardization.
namespace Symmetry3D {
enum CellType // also CrystalFamily
{
    TRIC, // triclinc, C_i
    MONO, // monoclinic, C_2h
    ORTH, // orthorhombic, D_2h
    TETR, // tetragonal, D_4h
    HEXA, // hexgonal, D_6h
    RHOM, // rhombonhedral, D_3d
    CUBI, // cubic, O_h
};
enum Centering {
    PRIM, // primitive
    BODY, // body-centered
    FACE, // face-centered
    AFAC,
    BFAC,
    CFAC, // A/B/C face centered
    BASE, // Base-centered, (C)
    RCEN, // R-centering
    CERR, // unknown
};
enum BravsLatt {
    aP, // TRIC + Primitive
    mP,
    mS, // Monoclinic + Permitive|Base-centered
    oP,
    oS,
    oI,
    oF, // Orthorhombic + Permitive|Base-centered|Body-centered|Face-centered
    tP, // Tetragonal + Primitive
    tI,
    hR, // Rhombonhedral + Primitive
    hP, // Hexgonal + Primitive
    cP,
    cI,
    cF, // Cubic + Primitive|Body-centered|Face-centered
};
enum Holohedry {
    HOLO_NONE,
    HOLO_TRIC,
    HOLO_MONO,
    HOLO_ORTH,
    HOLO_TETR,
    HOLO_TRIG, //
    HOLO_HEXA,
    HOLO_CUBI,
};
/***
 * There are 34 point gropus, where there is no translation subgroups
 * Put then in order
 * One can use mertric tensor of a permitive cell to determind which Bravis lattice belongs to
 * ***/
// array<>;
/****
 * Operations:
 * 1. identity operation
 * 2. translation
 * 3. rotation
 * 4. screw rotation
 * 5. N-fold rotoinversion
 * 6. invsrsion
 * 7. reflection (2-flod rotoinversion)
 * 8. glide relfection
 * All symmtery optrations can be divided into a linear part and translation part
 * (W2,w2)(W1,w1) = (W2W1, W2w1+w2)
 * ****/

// Opt-in candidate ordering and exact necessary-condition pruning. The
// default/legacy overloads remain unchanged. Counters accumulate into the
// caller-owned stats; caches are bounded and discarded after every reduction.
struct ReductionSearchOptions {
    bool enabled=false;
    bool rare_anchor=true;
    bool order_candidates=true;
    bool prefilter=true;
    bool cache_candidates=true;
    size_t cache_capacity=4096;
};
struct ReductionSearchStats {
    std::uint64_t candidates=0,necessary_rejections=0,full_checks=0,cache_hits=0,geometry_stages=0;
};
int trans_reduce(Atoms&,double,const ReductionSearchOptions&,ReductionSearchStats* stats=nullptr);
int trans_reduce_1d(Atoms&,int,double,const ReductionSearchOptions&,ReductionSearchStats* stats=nullptr);
int find_primitive(Atoms&,double,const ReductionSearchOptions&,ReductionSearchStats* stats=nullptr);
int all_reduce(Atoms&,double,const ReductionSearchOptions&,ReductionSearchStats* stats=nullptr);

// trans_reduce, trans_reduce_1d, find_primitive and all_reduce require a nonempty
// structure, matching position/species counts, finite positions, a finite
// nonsingular cell with a
// representable inverse, and positive finite tol (including 5*tol). Invalid
// inputs throw std::invalid_argument before changing the structure. Left-handed
// cells are accepted; a valid structure with no reduction returns 1.
// Cartesian and fractional positions are supported; the input representation
// is retained, with fractional output expressed in the final cell.
// Reduce translational multiplicity only along periodic lattice directions,
// preserving the original PBC flags and nonperiodic cell directions.
int trans_reduce(Atoms &atoms, double tol = 1e-5);
// Only search one direction; cdir must be 0, 1 or 2. A nonperiodic direction
// returns 1 without changing the structure, after the input validation above.
int trans_reduce_1d(Atoms &atoms, int cdir, double tol = 1e-5);
// Search primitive cell, no rotation, one can combined with correct_cell to avoid strange cell parameters
// Currently must do trans_reduce first to find real primitive cell
// Requires full periodicity; use trans_reduce for mixed/nonperiodic structures.
int find_primitive(Atoms &atoms, double tol = 1e-5);
// Whole-structure species-preserving 2D translation group in the selected
// periodic rows. The remaining row and all layer heights are unchanged. Supports
// singular Cartesian slabs; fractional storage requires an invertible full cell.
// Reduction retains the same extensive-label/metadata policy as 3D reduction.
int find_primitive_2d(Atoms &atom, int cdir1, int cdir2, double tol = 1e-5);
// Try to make the cell more orthogonal without changing multiplicity or rotating.
// Preserves Cartesian/fractional representation and Cartesian geometry modulo
// periodic translations; does not require a nonempty structure.

int correct_cell(Atoms &atoms, double tol = 1e-5);
// first trans_reduce, then find_primitive, (finally correct cell, no rotation)
// Requires full periodicity, checked before any stage changes the structure.
int all_reduce(Atoms &atoms, double tol = 1e-5);

int aligned_orientation(Atoms &atom, double tol = 1e-4);

BravsLatt get_bravs_type(Atoms &atom, double tol = 1e-4);

}; // namespace Symmetry3D
