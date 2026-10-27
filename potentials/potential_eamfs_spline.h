// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

// see LAMMPS eam/fs pair style
#include "potential.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace Potentials
{
    class EAMFsSpline : public AnyPotential
    {
    friend struct GPMCodecAccess;
    protected:
        int nelements;
        int Nrho, Nr;
        double drho, dr;
        double rdr, rdrho;
        double rmax, rhomin = 0.0, rhomax;
        bool is_he_format = false;
        std::vector<Eigen::MatrixXd> frho_spline;
        std::vector<std::vector<Eigen::MatrixXd>> rhor_spline;
        std::vector<Eigen::MatrixXd> z2r_spline;

        void Load(const std::string &filename, std::map<std::string, int> &typemap);
        void assemble_hessian(Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &hessian,
            Eigen::MatrixXd *cellhess,Eigen::MatrixXd *lrhess);


    public:
        EAMFsSpline() = default;
        EAMFsSpline(std::string &filename, std::map<std::string, int> &typemap) {
            Load(filename, typemap);
        }

        using AnyPotential::applyHessianVectorProduct;
        bool supports_analytical_hessian_vector_product() const override { return true; }
        void applyHessianVectorProduct(Atoms &atom,AnyNeighborhoods &nbhs,
            const Eigen::VectorXd &direction,Eigen::VectorXd &product) override;
        bool supports_analytical_hessian() const override { return true; }
        bool supports_analytical_cell_hessian() const override { return true; }
        void getEFSHessian(Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &hessian) override;
        void getEFSHessian(Atoms &atom,HessianMatrix &hessian) override;
        void getEFSHessianSymm(Atoms &atom,HessianMatrix &hessian,double symprec=1e-4) override;
        void getEFSHessianCellOMP(Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess) override;
        void getEFSHessianCellSymmOMP(Atoms &atom,HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess,double symprec=1e-4) override;
        void getEFSHessianCellSymm(Atoms &atom,HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess,double symprec=1e-4) override;
        void getEFS(Atoms &atom);
        void getEFS(Atoms &atom, AnyNeighborhoods &nbhs);

        void getEFSOMP(Atoms &atom);
        void getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs);
    };
};
