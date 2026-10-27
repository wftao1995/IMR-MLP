// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "symbasis.h"
#include "tools/tools.h"
#include "base/neighborhoodKD.h"

namespace Symmetry3D {
    // each row is a basis
    // given symmetry operations
    // no cell parameter
    void get_symm_basis(std::vector<symopt_type> &symops, Eigen::MatrixXd &result, int natoms) {
        result.resize(3*natoms, 3*natoms);
        result.setIdentity();
        int nop = symops.size();
        for (int iop = 0; iop < nop; ++iop) {
            auto &op = symops[iop];
            ;
        }
    }
    ;
    /**
     * @brief 
     * For a symmetric cell (we assume that create supercell will break point group symmetry):
     * Numerically, L*W = L*P = P*L where P is a diagonal matrix which only change sign. det(P) = +-1
     * This is because L is rotationally invarient under symop
     * Thus
     * r*W*L = r*L^-1*L*W*L = r*L^-1*L*P*L = r*L*P = R*P
     * R*W = R*L^-1*L*W = R*L^-1*L*P = R*P
     * so r*W*L = R*W
     * apply symop in car or frac is both OK
     * We take average on car mode.
     * 
     * TODO: basis at any q-point
     * TODO: how to deal cell?
     * 
     * @param atom 
     * @param symops 
     * @param prec  should not be too small. 0.1 is OK, this is actually a radius. 
     */
    bool symmetrize(Atoms &atom, std::vector<symopt_type> &symops, double prec)
    {
        static double nugval = std::min({prec * 0.01, 0.005});
        int nops = symops.size();
        atom.wrap();
        std::vector<int> nfold(atom.size(), 0);
        // symmetrize cell
        // Eigen::Matrix3d ncell = Eigen::Matrix3d::Zero();
        // for (int i = 0; i < nops; ++i) {
        //     auto tcell = atom.cell_ * symops[i].first;
        //     // correct orientation
        //     for (int a = 0; a < 3; ++a) {
        //         double t = tcell.row(a).dot(atom.cell_.row(a));
        //         ;
        //     }
        // }
        // ncell /= nops;
        // atom.set_cell(ncell);
        // symmetrize positions
        std::vector<Eigen::RowVector3d> npos(atom.size(), Eigen::RowVector3d::Zero());
        NeighborhoodKD nbhs(&atom, prec);
        for (int iop = 0; iop < nops; ++iop) {
            auto &op = symops[iop];
            for (int i = 0; i < atom.size(); ++i) {
                int nlev = 0;
                Eigen::RowVector3d tpos = atom.pos_[i] * op.first + op.second * atom.cell_;
                atom.wrap(tpos);
                auto nbh = nbhs.query(tpos.data(), 1, 0, true);
                if (nbh.nbhcount_ != 1) return false; // failed
                int oid = nbh.indices_[0];
                int nid = i;
                // wrap back pos if cross boundary
                if ((atom.pos_[oid] - tpos).norm() > 10*prec) {
                    tpos = atom.pos_[oid] - nbh.pos_[0];
                }
                tpos(0) -= nugval;
                if (nlev) {
                    tpos(3) -= nugval;
                }
                // _DEBUG_OUTPUT_(atom.pos_[oid]);
                // _DEBUG_OUTPUT_(tpos0);
                nfold[oid] += 1;
                npos[oid] += tpos;
            }
        }
        for (int iat = 0; iat < atom.size(); ++iat) {
            npos[iat] /= nfold[iat];
        }
        atom.pos_ = npos;
        return true;
    }
    /**
     * @brief 
     * @param op 
     * @param atom 
     */
    // void apply_symop(symopt_type &op, Atoms &atom)
    // {
    //     atom.to_fractional();
    //     // cell
    //     auto ncell = atom.cell_ * op.first;
    //     // convert ncell to correct orientation
    //     ;
    //     atom.cell_ = ncell;
    //     // position
    //     for (int i = 0; i < atom.size(); ++i) {
    //         atom.pos_[i] = atom.pos_[i] * op.first + op.second;
    //     }
    //     atom.to_cartesian();
    //     atom.wrap();
    // }
};