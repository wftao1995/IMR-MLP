// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "base/structures.h"
extern "C"
{
#include <spglib/spglib.h>
}
// Note that rotation operation is transpose of spglib one
// Here, x' = xW + w
namespace Symmetry3D
{
    typedef Eigen::Matrix3d RotType;
    typedef Eigen::RowVector3d TraType;
    typedef std::pair<RotType, TraType> symopt_type; // Use matrix3d instead of matrix3i is just to ommit cast process
    int standardize_cell(Atoms &atom, const int to_pertimitive, const int no_idealization, const double symprec);
    // TODO: may have bug, what if two axis swapped?
    int niggli_reduce(Atoms &atom, const double symprec);
    int delaunay_reduce(Atoms &atom, const double symprec);
    int get_symmetry_operations(const Atoms &atom, std::vector<symopt_type> &symopts, const int max_size, const double symprec);
    int get_box_symmetry_operations(const Eigen::Matrix3d &cell, std::vector<symopt_type> &symopts, double symprec);
    int schoenfiles_spgnumber(Atoms &atom, const double symprec);
    int spacegroup_number(Atoms &atom, const double symprec);
    struct symdataset {
        int space_group_number;
        int hall_number;
        std::string international_symbol;
        std::string pointgroup_symbol;
        std::string hall_symbol;
        // 
        char choice[6];
        Eigen::Matrix3d transformation_matrix;
        Eigen::RowVector3d origin_shift;
        // ops
        int n_operations;
        std::vector<symopt_type> symm_ops;
        Eigen::MatrixXi symop_mapping; // mapping of atomic ids after apply symmetry operation, each row for one operation
        // site
        int n_atoms;
        std::vector<int> wyckoffs;
        std::vector<std::string> site_symmetry_symbols;
        std::vector<int> equivalent_atoms;
        std::vector<int> crystallographic_orbits;
        Eigen::Matrix3d primitive_lattice;
        std::vector<int> mapping_to_primitive;
        int nprimitive;
    };

    symdataset get_sym_dataset(Atoms &atom, double symprec = 1e-5, bool getmap = false);

} // namespace Symmetry3D
