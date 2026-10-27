// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential.h"

namespace Potentials {

// Equal-bond table T(r, theta) on a uniform grid, with bilinear interpolation.
// Unequal bonds use V(rij, rik, theta) = (T(rij, theta) + T(rik, theta)) / 2.
class ThreebodyTable : public AnyPotential {
    friend struct GPMCodecAccess;
public:
    ThreebodyTable() = default;
    ThreebodyTable(const std::string &filename,
                   const std::map<std::string, int> &typemap);
    ~ThreebodyTable() = default;

    void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override;
    bool supports_analytical_hessian() const override { return true; }
    void getEFSHessian(Atoms &atom, HessianMatrix &hessian) override;
    void getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) override;
    void getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double symprec = 1e-4) override;
    bool supports_analytical_cell_hessian() const override { return true; }
    void getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs,
        HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
    void getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
        Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
    void getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs,
        HessianMatrix &hessian) override;
    void getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
        Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
    void getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
        Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) override;
    void getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
        Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) override;
    using AnyPotential::applyHessianVectorProduct;
    bool supports_analytical_hessian_vector_product() const override { return true; }
    void applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
        const Eigen::VectorXd &direction, Eigen::VectorXd &product) override;
    void Load(std::istream &is) override;
    void Save(std::ostream &os) override;

private:
    void assemble_hessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
        Eigen::MatrixXd *cellhess, Eigen::MatrixXd *lrhess);
    Eigen::Matrix<double,6,6> triplet_curvature(const Neighborhood &nbh, int j, int k) const;
    int nelements = 0;
    int nr_points = 0, ntheta_points = 0;
    double rmin = 0, rmax = 0, dr = 0, inv_dr = 0;
    double theta_min = 0, theta_max = 0, dtheta = 0, inv_dtheta = 0;
    std::vector<std::string> element_names;
    // 2D grid per type triplet. Stored row-major: [ir * ntheta + itheta]
    std::vector<Eigen::MatrixXd> tables;

    int get_triplet_id(int i, int j, int k) const {
        // sort i >= j >= k
        if (i < j) std::swap(i, j);
        if (j < k) std::swap(j, k);
        if (i < j) std::swap(i, j);
        return i * (i + 1) * (i + 2) / 6 + j * (j + 1) / 2 + k;
    }

    void load_file(const std::string &file,
                   const std::map<std::string, int> &typemap);
    void load_stream(std::istream &is, const std::map<std::string, int> &typemap,
                     const std::string &source);

    // Bilinear evaluation: V(rij, rik, theta) + derivatives
    void eval_3body(int tid, double rij, double rik, double theta,
                    double &val,
                    double &dV_drij, double &dV_drik, double &dV_dtheta,
                    double *dV_drij_dtheta = nullptr, double *dV_drik_dtheta = nullptr) const;
};

} // namespace Potentials
