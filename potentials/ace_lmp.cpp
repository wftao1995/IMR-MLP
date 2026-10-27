// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifdef WITH_ACELMP

#include "base/neighborhood.h"
#include "extheaders/ace_c_basis.h"
#include "extheaders/ace_evaluator.h"
#include "extheaders/ace_recursive.h"
#include "extheaders/ace_version.h"
#include "tools/tools.h"
#include "utils/gputils.h"

#include "ace_lmp.h"
#include <numeric>

namespace Potentials {
    struct ACEImpl {
        ACEImpl() : basis_set(nullptr), ace(nullptr) {}
        ~ACEImpl()
        {
            delete basis_set;
            delete ace;
        }
        ACECTildeBasisSet *basis_set;
        ACERecursiveEvaluator *ace;
    };


    PotentialACELmp::PotentialACELmp(const std::string &potname, std::vector<std::string> &_elenames, bool recursive_) {
        recursive = recursive_;
        elenames = _elenames;
        Load(potname);
        cutlim = aceimpl->basis_set->cutoffmax;
        cutoff = cutlim; cutoffsq = cutoff * cutoff; species_count = elenames.size();
        buffer_x_internal = new double[buffer_size * 3];
        buffer_x = new double*[buffer_size];
        buffer_type = new int[buffer_size];
        buffer_jlist = new int[buffer_size];
        for (int i = 0; i < buffer_size; ++i) {
            buffer_x[i] = &buffer_x_internal[3*i];
            buffer_jlist[i] = i+1;
        }
        buffer_x[0][0] = 0;
        buffer_x[0][1] = 0;
        buffer_x[0][2] = 0;
    };
    void PotentialACELmp::Load(const std::string &potname) { 
        auto model_file = open_ifile(potname);
        if (!model_file) throw std::runtime_error("Cannot read ACE resource: " + potname);
        std::ostringstream model_bytes; model_bytes << model_file->rdbuf();
        if (model_file->bad()) throw std::runtime_error("Cannot read ACE resource: " + potname);
        serialized_basis_ = model_bytes.str();
        aceimpl = new ACEImpl;
        aceimpl->basis_set = new ACECTildeBasisSet(potname);
        int nelements = aceimpl->basis_set->nelements;
        aceimpl->ace = new ACERecursiveEvaluator();
        aceimpl->ace->set_recursive(recursive);
        aceimpl->ace->element_type_mapping.init(elenames.size()+1);
        for (int i = 1; i <= elenames.size(); ++i) {
            auto &name = elenames[i-1];
            int mu = aceimpl->basis_set->get_species_index_by_name(name);
            std::cout << "Mapping " << name << "into " << mu << std::endl;
            aceimpl->ace->element_type_mapping(i) = mu;
        }
        aceimpl->ace->set_basis(*(aceimpl->basis_set), 1);
        Message("Loaded");
    }

    void PotentialACELmp::getEFS(Atoms &atom) {
        atom.have_energy = true;
        atom.forces_.assign(atom.size(), Eigen::Vector3d::Zero());
        atom.have_stress = true;
        atom.stress_ = Eigen::Matrix3d::Zero();
        atom.site_energies_.assign(atom.size(), 0);
        atom.energy_ = 0;
        double fij[3];
        // double **x = new double*[atom.size()];
        // double *p = atom.pos_[0].data();
        // for (int i = 0; i < atom.size(); ++i) {
        //     x[i] = p + 3*i;
        // }
        // int *type = atom.chemmap_.data();
        // for (int i = 0; i < atom.size(); ++i) {
        //     type[i] += 1;
        // }
        Neighborhoods nbhs(atom, cutlim);
        int maxjnum = 0;
        for (int i = 0; i < nbhs.size(); ++i) {
            maxjnum = max(maxjnum, nbhs[i].nbhcount_);
        }
        aceimpl->ace->resize_neighbours_cache(maxjnum);
        if (maxjnum > buffer_size) {
            buffer_size = maxjnum * allocate_ampifer + 1;
            delete [] buffer_x_internal;
            delete [] buffer_x;
            delete [] buffer_jlist;
            delete [] buffer_type;
            buffer_x_internal = new double[buffer_size * 3];
            buffer_x = new double*[buffer_size];
            buffer_type = new int[buffer_size];
            buffer_jlist = new int[buffer_size];
            for (int i = 0; i < buffer_size; ++i) {
                buffer_x[i] = &buffer_x_internal[3*i];
                buffer_jlist[i] = i+1;
            }
            buffer_x[0][0] = 0;
            buffer_x[0][1] = 0;
            buffer_x[0][2] = 0;
        }
        // loop over atom
        for (int ii = 0; ii < atom.size(); ++ii) {
            int i = ii;
            int jnum = nbhs[ii].nbhcount_;
            // buffer_x contains self and neighbors
            buffer_type[0] = atom.chemmap_[i] + 1;
            for (int tt = 1; tt <= jnum; ++tt) {
                for (int a = 0; a < 3; ++a) {
                    buffer_x[tt][a] = nbhs[ii].pos_[tt-1](a);
                }
                buffer_type[tt] = nbhs[ii].chemmap_[tt-1] + 1;
            }
            aceimpl->ace->compute_atom(0, buffer_x, buffer_type, jnum, buffer_jlist);
            for (int jj = 0; jj < jnum; ++jj) {
                int j = nbhs[ii].indices_[jj];
                fij[0] = aceimpl->ace->neighbours_forces(jj, 0);
                fij[1] = aceimpl->ace->neighbours_forces(jj, 1);
                fij[2] = aceimpl->ace->neighbours_forces(jj, 2);
                atom.forces_[i][0] += fij[0];
                atom.forces_[i][1] += fij[1];
                atom.forces_[i][2] += fij[2];
                atom.forces_[j][0] -= fij[0];
                atom.forces_[j][1] -= fij[1];
                atom.forces_[j][2] -= fij[2];
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        atom.stress_(a, b) -= fij[a] * nbhs[ii].pos_[jj](b);
                    }
                }   
            }
            atom.site_energies_[ii] = aceimpl->ace->e_atom;
            atom.energy_ += aceimpl->ace->e_atom;
        }
        // for (int i = 0; i < atom.size(); ++i) {
        //     type[i] -= 1;
        // }
    }

    PotentialACELmp::~PotentialACELmp() {
        if (aceimpl)
            delete aceimpl;
        if (buffer_size > 0) {
            delete [] buffer_jlist;
            delete [] buffer_x;
            delete [] buffer_x_internal;
            delete [] buffer_type;
        }
    }
}


#endif
