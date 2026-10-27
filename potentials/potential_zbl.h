// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential.h"

namespace Potentials {

class PotentialZBL : public AnyPotential {
public:
    double r0 = 1.0, r1 = 2.0;
    std::vector<double> z;
    // ZBL universal screening constants
    static constexpr double a0 = 0.46850;
    static constexpr double zp = 0.23;
    static constexpr double cc_ev = 14.3997; // eV·Å / e²
    // c_i and p_i in ZBL screening function: sum c_i * exp(p_i * r/a)
    // p_i are negative (matching exp(-d_i * r/a) convention)
    static constexpr double c1 = 0.18175, c2 = 0.50986, c3 = 0.28022, c4 = 0.02817;
    static constexpr double p1 = -3.19980, p2 = -0.94229, p3 = -0.40290, p4 = -0.20162;

    PotentialZBL() = default;
    PotentialZBL(GPSettings settings);
    ~PotentialZBL() = default;

    bool supports_site_efs() const override { return !z.empty(); }
    void CalcSiteEFS(Neighborhood &, double &, SiteCoordinateMatrix &) override;
    void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void getE(Atoms &atom, AnyNeighborhoods &nbhs) override;
    bool supports_analytical_hessian() const override { return true; }
    void getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) override;
    void getEFSHessian(Atoms &atom, HessianMatrix &hessian) override;
    void getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double symprec = 1e-4) override;
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
    bool supports_analytical_hessian_vector_product() const override { return true; }
    void applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
        const Eigen::VectorXd &direction, Eigen::VectorXd &product) override;
    void Load(std::istream &is) override;
    void Save(std::ostream &os) override;

private:
    void initialize(GPSettings settings);
    double zbl_energy(double r, int zi, int zj) const;
    double zbl_energy_deriv(double r, int zi, int zj) const;
    double zbl_energy_second_deriv(double r, int zi, int zj) const;
    Eigen::Matrix3d pair_curvature(double r, const Eigen::RowVector3d &delta, int zi, int zj) const;
    inline double switch_fn(double r) const;
    inline double switch_fn_deriv(double r) const;
};

} // namespace Potentials
