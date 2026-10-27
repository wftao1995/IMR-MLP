// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base/neighborhood.h"
#include "base/structures.h"
#include "potential.h"
namespace Potentials
{
    // Classical LJ 6-12 two body potential
    class LJ612 : public AnyPotential
    {
    friend struct GPMCodecAccess;
    protected:
        int ntypes = 2;
        int ncoeffs;
        int nljcoeffs;
        Eigen::MatrixXd eps, sig; // original value
        Eigen::VectorXd elemoffset; // element energy, somewhat linear coeff
        // transformed lj value
        // E = r6inv * (lj3 * r6inv - lj4) 
        // F = r6inv * (lj1 * r6inv - lj2)
        Eigen::MatrixXd lj1, lj2, lj3, lj4, dlj1deps, dlj2deps, dlj3deps, dlj4deps, dlj1dsig, dlj2dsig, dlj3dsig, dlj4dsig; 

        Eigen::Matrix3d pair_curvature(int itype, int jtype, double distance,
                                       const Eigen::RowVector3d &displacement) const;

    public:
        LJ612(GPSettings settings);
        ~LJ612(){};
        using AnyPotential::Load;
        using AnyPotential::Save;
        // convert to original value
        void convert_eps2lj();
        // convert to special value
        void convert_lj2eps();

        void InitSettings() {
            Connect(ntypes, "ntypes");
        }

        virtual void getEFS(Atoms &atom, AnyNeighborhoods &nbhs);

        void getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs,
                           HessianMatrix &hessian) override;
        void getEFSHessian(Atoms &atom, HessianMatrix &hessian) override;
        void getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian,
                               double symprec = 1e-4) override;
        bool supports_analytical_hessian() const override { return true; }
        bool supports_analytical_hessian_vector_product() const override { return true; }
        bool supports_analytical_cell_hessian() const override { return true; }
        void getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs,
            HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
        void getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
        void getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs,
            HessianMatrix &hessian) override;
        void getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs,
            HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
        void getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) override;
        void getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) override;
        using AnyPotential::applyHessianVectorProduct;
        void applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
                                       const Eigen::VectorXd &direction,
                                       Eigen::VectorXd &product) override;

        virtual void random_init();

        // NL fitting
        virtual int get_NL_coeffs_count() const;
        virtual void nl_get_coeffs(double *) const;
        virtual void nl_set_coeffs(const double *);
        virtual void AccumulateFPGard(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,   
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight);
        void Load(std::istream &is) override;
        void Save(std::ostream &os) override;
        void Save(const std::string &fnm) override;
        std::map<int, std::pair<int, int>> mapgid2oid;
        std::map<std::pair<int, int>, int> mapoid2gid;
        virtual void train_init();
    };
};
