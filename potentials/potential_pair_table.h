// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential.h"

namespace Potentials {

class PairTable : public AnyPotential {
    friend struct GPMCodecAccess;
public:
    PairTable() = default;
    PairTable(const std::string &filename,
              const std::map<std::string, int> &typemap);
    ~PairTable() = default;

    void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override;
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
    void Load(std::istream &is) override;
    void Save(std::ostream &os) override;

private:
    int nelements = 0;
    int npoints = 0;
    double rmin = 0, rmax = 0, dr = 0, inv_dr = 0;
    std::vector<std::string> element_names;
    // pair_splines indexed by pairid = j*(j+1)/2 + i (i <= j)
    std::vector<Eigen::MatrixXd> pair_splines;

    Eigen::Matrix3d pair_curvature(int itype, int jtype, double distance,
                                   const Eigen::RowVector3d &displacement) const;

    int get_pairid(int i, int j) const {
        if (i > j) std::swap(i, j);
        return j * (j + 1) / 2 + i;
    }

    void load_file(const std::string &file,
                   const std::map<std::string, int> &typemap);
    void load_stream(std::istream &is, const std::map<std::string, int> &typemap,
                     const std::string &source);
};

} // namespace Potentials
