// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <memory>
#include <stdexcept>
#include "spglib_driver.h"
#include "tools/tools.h"
#include "base/neighborhoodKD.h"
#include <ranges>
#define PREALLOC 10

/***
 * TODO: read xtalcomp and refine codes
 * ***/
#include "io/gpio.h"
namespace Symmetry3D
{
    /**
     * spglib's spg_standardize_cell function wapper
     * Now warpped as 
     *  Atoms atom
     *  const int to_primitive
     *  const int no_idealize
     *  const double symprec
     * **/
    int standardize_cell(Atoms &atom, const int to_primitive = 1, const int no_idealize = 0, const double symprec = 1e-5)
    {
        //in spglib lattice is stored in colume wise
        double lattice[3][3] = {
            {atom.cell_(0, 0), atom.cell_(1, 0), atom.cell_(2, 0)},
            {atom.cell_(0, 1), atom.cell_(1, 1), atom.cell_(2, 1)},
            {atom.cell_(0, 2), atom.cell_(1, 2), atom.cell_(2, 2)}};
        int num_atom = atom.size();
        //int *types = new int[num_atom];
        //auto position = new double[num_atom][3];
        /*PREALLOC here because positions may grow when finding conventional cell*/
        int *types = (int *)malloc(PREALLOC * num_atom * sizeof(int));
        double(*position)[3] = (double(*)[3])malloc(sizeof(double[3]) * num_atom * PREALLOC);
        std::vector<Eigen::RowVector3d> dirpos;
        /*spglib receives and returns direct positions*/
        atom.get_dir_pos(dirpos);
        for (int i = 0; i < num_atom; ++i)
        {
            types[i] = atom.chemmap_[i];
            for (int j = 0; j < 3; ++j)
            {
                position[i][j] = dirpos[i](j);
            }
        }
        int state;
        state = spg_standardize_cell(lattice,
                                     position,
                                     types,
                                     num_atom,
                                     to_primitive,
                                     no_idealize,
                                     symprec);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                atom.cell_(i, j) = lattice[j][i];
        atom.pos_.resize(state);
        atom.chemmap_.resize(state);
        Eigen::RowVector3d tmpvec;
        for (int i = 0; i < state; ++i)
        {
            atom.chemmap_[i] = types[i];
            for (int j = 0; j < 3; ++j)
            {
                tmpvec(j) = position[i][j];
            }
            atom.pos_[i] = tmpvec * atom.cell_;
        }
        //delete types;
        //detele[] position;
        free(types);
        free(position);
        return state;
    }
    // TODO : avoid this
    int schoenfiles_spgnumber(Atoms &atom, const double symprec)
    {
        //in spglib lattice is stored in colume wise
        double lattice[3][3] = {
            {atom.cell_(0, 0), atom.cell_(1, 0), atom.cell_(2, 0)},
            {atom.cell_(0, 1), atom.cell_(1, 1), atom.cell_(2, 1)},
            {atom.cell_(0, 2), atom.cell_(1, 2), atom.cell_(2, 2)}};
        int num_atom = atom.size();
        int *types = (int *)malloc(PREALLOC * num_atom * sizeof(int));
        double(*position)[3] = (double(*)[3])malloc(sizeof(double[3]) * num_atom * PREALLOC);
        std::vector<Eigen::RowVector3d> dirpos;
        /*spglib receives and returns direct positions*/
        atom.get_dir_pos(dirpos);
        for (int i = 0; i < num_atom; ++i)
        {
            types[i] = atom.chemmap_[i];
            for (int j = 0; j < 3; ++j)
            {
                position[i][j] = dirpos[i](j);
            }
        }
        int state;
        char symbol[7];
        state = spg_get_schoenflies(symbol,
                                    lattice,
                                    position,
                                    types,
                                    num_atom,
                                    symprec);
        free(types);
        free(position);
        return state;
    }

    int spacegroup_number(Atoms& atom, const double symprec)
    {
        double lattice[3][3] = {
            {atom.cell_(0, 0), atom.cell_(1, 0), atom.cell_(2, 0)},
            {atom.cell_(0, 1), atom.cell_(1, 1), atom.cell_(2, 1)},
            {atom.cell_(0, 2), atom.cell_(1, 2), atom.cell_(2, 2)}};
        int num_atom = atom.size();
        int *types = new int[num_atom];
        auto position = new double[num_atom][3];
        std::vector<Eigen::RowVector3d> dirpos;
        /*spglib receives and returns direct positions*/
        atom.get_dir_pos(dirpos);
        for (int i = 0; i < num_atom; ++i)
        {
            types[i] = atom.chemmap_[i];
            for (int j = 0; j < 3; ++j)
            {
                position[i][j] = dirpos[i](j);
            }
        }
        SpglibDataset *pdataset = spg_get_dataset(
                                 lattice,
                                 position,
                                 types,
                                 num_atom,
                                 symprec);
        int spgnumber = pdataset->spacegroup_number;
        spg_free_dataset(pdataset);
        delete[] types;
        delete[] position;
        return spgnumber;
    }

    int niggli_reduce(Atoms &atom, const double symprec)
    {
        double lattice[3][3] = {
            {atom.cell_(0, 0), atom.cell_(1, 0), atom.cell_(2, 0)},
            {atom.cell_(0, 1), atom.cell_(1, 1), atom.cell_(2, 1)},
            {atom.cell_(0, 2), atom.cell_(1, 2), atom.cell_(2, 2)}};
        int state;
        state = spg_niggli_reduce(lattice, symprec);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                atom.cell_(i, j) = lattice[j][i];
        atom.wrap();
        return state;
    }

    int delaunay_reduce(Atoms &atom, const double symprec)
    {
        double lattice[3][3] = {
            {atom.cell_(0, 0), atom.cell_(1, 0), atom.cell_(2, 0)},
            {atom.cell_(0, 1), atom.cell_(1, 1), atom.cell_(2, 1)},
            {atom.cell_(0, 2), atom.cell_(1, 2), atom.cell_(2, 2)}};
        int state;
        state = spg_delaunay_reduce(lattice, symprec);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                atom.cell_(i, j) = lattice[j][i];
        atom.hwrap();
        return state;
    }

    int get_box_symmetry_operations(const Eigen::Matrix3d &cell, std::vector<symopt_type> &symopts, double symprec)
    {
        double lattice[3][3] = {
            {cell(0, 0), cell(1, 0), cell(2, 0)},
            {cell(0, 1), cell(1, 1), cell(2, 1)},
            {cell(0, 2), cell(1, 2), cell(2, 2)}};
        int state;
        int num_atom = 1;
        int types[1];
        types[0] = 1;
        int max_size = 48;
        auto position = new double[num_atom][3];
        auto rotation = new int[max_size][3][3];
        auto translation = new double[max_size][3];
        std::vector<Eigen::RowVector3d> dirpos;
        /*spglib receives and returns direct positions*/
        for (int j = 0; j < 3; ++j)
        {
            position[0][j] = 0;
        }

        state = spg_get_symmetry(rotation,
                                 translation,
                                 max_size,
                                 lattice,
                                 position,
                                 types,
                                 num_atom,
                                 symprec);

        symopts.resize(state);
        for (int ii = 0; ii < state; ii++)
        {
            symopt_type tmpopt;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                    tmpopt.first(i, j) = rotation[ii][j][i]; // transpose
                tmpopt.second(i) = translation[ii][i];
            }
            symopts[ii] = tmpopt;
        }

        delete[] position;
        delete[] rotation;
        delete[] translation;
        return state;
    }

    int get_symmetry_operations(const Atoms &atom, std::vector<symopt_type> &symopts, const int max_size, const double symprec)
    {
        double lattice[3][3] = {
            {atom.cell_(0, 0), atom.cell_(1, 0), atom.cell_(2, 0)},
            {atom.cell_(0, 1), atom.cell_(1, 1), atom.cell_(2, 1)},
            {atom.cell_(0, 2), atom.cell_(1, 2), atom.cell_(2, 2)}};
        int state;
        int num_atom = atom.size();
        int *types = new int[num_atom];
        auto position = new double[num_atom][3];
        auto rotation = new int[max_size][3][3];
        auto translation = new double[max_size][3];
        std::vector<Eigen::RowVector3d> dirpos;
        /*spglib receives and returns direct positions*/
        atom.get_dir_pos(dirpos);
        for (int i = 0; i < num_atom; ++i)
        {
            types[i] = atom.chemmap_[i];
            for (int j = 0; j < 3; ++j)
            {
                position[i][j] = dirpos[i](j);
            }
        }
        state = spg_get_symmetry(rotation,
                                 translation,
                                 max_size,
                                 lattice,
                                 position,
                                 types,
                                 num_atom,
                                 symprec);

        symopts.resize(state);
        for (int ii = 0; ii < state; ii++)
        {
            symopt_type tmpopt;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                    tmpopt.first(i, j) = rotation[ii][j][i]; // transpose
                tmpopt.second(i) = translation[ii][i];
            }
            symopts[ii] = tmpopt;
        }

        delete[] types;
        delete[] position;
        delete[] rotation;
        delete[] translation;
        return state;
    }

    symdataset get_sym_dataset(Atoms &atom, double symprec, bool getmap) {
        const double volume = atom.get_volume();
        if (!std::isfinite(symprec) || symprec <= 0 || atom.size() == 0 ||
            !atom.is_fully_periodic() || atom.chemmap_.size() != atom.pos_.size() ||
            !std::isfinite(volume) || std::abs(volume) < 1e-14)
            throw std::invalid_argument("Invalid symmetry input geometry or species");
        double lattice[3][3] = {
            {atom.cell_(0, 0), atom.cell_(1, 0), atom.cell_(2, 0)},
            {atom.cell_(0, 1), atom.cell_(1, 1), atom.cell_(2, 1)},
            {atom.cell_(0, 2), atom.cell_(1, 2), atom.cell_(2, 2)}};
        int num_atom = atom.size();
        std::vector<int> types(num_atom);
        std::unique_ptr<double[][3]> position(new double[num_atom][3]);
        std::vector<Eigen::RowVector3d> dirpos;
        /*spglib receives and returns direct positions*/
        atom.get_dir_pos(dirpos);
        for (int i = 0; i < num_atom; ++i)
        {
            if (!dirpos[i].allFinite()) throw std::invalid_argument("Non-finite fractional symmetry coordinates");
            types[i] = atom.chemmap_[i];
            for (int j = 0; j < 3; ++j)
            {
                position[i][j] = dirpos[i](j);
            }
        }
        std::unique_ptr<SpglibDataset, decltype(&spg_free_dataset)> pdataset(spg_get_dataset(
                                 lattice,
                                 position.get(),
                                 types.data(),
                                 num_atom,
                                 symprec), &spg_free_dataset);
        if (!pdataset || pdataset->n_operations <= 0 || pdataset->n_atoms != num_atom)
            throw std::runtime_error("spglib failed to identify a valid symmetry dataset");
        symdataset dataset;
        
        dataset.space_group_number = pdataset->spacegroup_number;
        strcpy(dataset.choice, pdataset->choice);
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                dataset.transformation_matrix(a, b) = pdataset->transformation_matrix[a][b];
                dataset.primitive_lattice(a, b) = pdataset->primitive_lattice[a][b];
            }
            dataset.origin_shift[a] = pdataset->origin_shift[a];
        }
        dataset.n_operations = pdataset->n_operations;
        dataset.symm_ops.resize(pdataset->n_operations);
        for (int ii = 0; ii < dataset.n_operations; ii++)
        {
            symopt_type tmpopt;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                    tmpopt.first(i, j) = pdataset->rotations[ii][j][i]; // transpose
                tmpopt.second(i) = pdataset->translations[ii][i];
            }
            dataset.symm_ops[ii] = tmpopt;
        }
        if (getmap) {
            Eigen::MatrixXi symop_mapping(dataset.n_operations, num_atom);
            for (int op = 0; op < dataset.n_operations; ++op) {
                std::vector<bool> used(num_atom, false);
                for (int i = 0; i < num_atom; ++i) {
                    const Eigen::RowVector3d image = dirpos[i] * dataset.symm_ops[op].first + dataset.symm_ops[op].second;
                    int match = -1;
                    for (int j = 0; j < num_atom; ++j) {
                        if (types[i] != types[j]) continue;
                        Eigen::RowVector3d delta = image - dirpos[j];
                        delta -= delta.array().round().matrix();
                        if ((delta * atom.cell_).norm() <= symprec) {
                            if (match != -1) throw std::runtime_error("Ambiguous symmetry atom mapping");
                            match = j;
                        }
                    }
                    if (match < 0 || used[match]) throw std::runtime_error("Incomplete or non-bijective symmetry mapping");
                    used[match] = true;
                    symop_mapping(op, i) = match;
                }
            }
            dataset.symop_mapping = symop_mapping;
        }
        // site
        dataset.n_atoms = pdataset->n_atoms;
        dataset.wyckoffs.resize(dataset.n_atoms);
        memcpy(dataset.wyckoffs.data(), pdataset->wyckoffs, dataset.n_atoms * sizeof(int));
        dataset.site_symmetry_symbols.resize(dataset.n_atoms);
        for (int i = 0; i < dataset.n_atoms; ++i) {
            dataset.site_symmetry_symbols[i] = pdataset->site_symmetry_symbols[i];
        }
        std::set<int> ieqatoms;
        dataset.equivalent_atoms.resize(dataset.n_atoms);
        dataset.mapping_to_primitive.resize(dataset.n_atoms);
        for (int i = 0; i < dataset.n_atoms; ++i) {
            dataset.equivalent_atoms[i] = pdataset->equivalent_atoms[i];
            dataset.mapping_to_primitive[i] = pdataset->mapping_to_primitive[i];
            ieqatoms.insert(dataset.mapping_to_primitive[i]);
        }
        dataset.nprimitive = dataset.mapping_to_primitive.size();
        dataset.crystallographic_orbits.resize(dataset.n_atoms);
        // ???
        for (int i = 0; i < dataset.n_atoms; ++i) {
            dataset.crystallographic_orbits[i] = pdataset->crystallographic_orbits[i];
        }
        dataset.hall_number = pdataset->hall_number;
        dataset.hall_symbol = pdataset->hall_symbol;
        dataset.international_symbol = std::string(pdataset->international_symbol);
        dataset.pointgroup_symbol = std::string(pdataset->pointgroup_symbol);
        return dataset;
    }


} // namespace Symmetry3D
