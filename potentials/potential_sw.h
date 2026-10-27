// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential.h"

namespace Potentials {

class SW : public AnyPotential {
    friend struct GPMCodecAccess;
public:
    SW() = default;
    SW(const std::string &filename,
        const std::map<std::string, int> &typemap);
    ~SW() = default;

    void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override;
    bool supports_analytical_hessian() const override { return true; }
    bool supports_analytical_cell_hessian() const override { return true; }
    void getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) override;
    void getEFSHessian(Atoms &atom, HessianMatrix &hessian) override;
    void getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) override;
    void getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double symprec = 1e-4) override;
    // Native cell chart: r_i = f_i*C + u_i, row-major cell components.
    void getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
        Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
    void getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
        Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
    void getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
        Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
    void getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
        Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) override;
    void getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
        Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) override;
    using AnyPotential::applyHessianVectorProduct;
    bool supports_analytical_hessian_vector_product() const override { return true; }
    void applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
                                  const Eigen::VectorXd &direction,
                                  Eigen::VectorXd &product) override;
    void Load(std::istream &is) override;
    void Save(std::ostream &os) override;
    void Save(const std::string &fnm) override;

    // ── Nonlinear fitting interface ─────────────────────────────────────
    int get_NL_coeffs_count() const override;
    void nl_get_coeffs(double *x) const override;
    void nl_set_coeffs(const double *x) override;
    void train_init() override;
    void random_init() override;
    void AccumulateEFSCombinationGard(Atoms &atom,
                                      AnyNeighborhoods &nbhs,
                                      double dLdE,
                                      Eigen::Matrix<double, -1, 3> &dLdF,
                                      Eigen::Matrix3d &dLdS,
                                      Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &wgt_buf,
                                      double *out_gard_accumulator) override;
    void AccumulateEFSCombinationGard(Atoms &atom,
                                      AnyNeighborhoods &nbhs,
                                      double dLdE,
                                      double *out_gard_accumulator) override;

private:
    int nelements = 0;
    std::vector<std::string> element_names_;
    void load_stream(std::istream &is, const std::map<std::string, int> &typemap);

    // 2-body params per pair (i,j): epsilon, sigma, A, B, p, q, a, cutoff
    struct PairParam {
        double epsilon = 0, sigma = 0, A = 0, B = 0;
        double p = 0, q = 0, a = 0, cutoff = 0;
    };

    // 3-body params per triplet (i,j,k): epsilon, sigma, a, lambda, gamma, cos_theta0, cutoff
    struct TripletParam {
        double epsilon = 0, sigma = 0, a = 0;
        double lambda = 0, gamma = 0, cos_theta0 = 0;
        double cutoff = 0;
    };

    std::vector<std::vector<PairParam>> pair_params;
    std::vector<std::vector<std::vector<TripletParam>>> trip_params;

    Eigen::Matrix3d pair_curvature(const Neighborhood &nbh, int j) const;
    Eigen::Matrix<double,6,6> triplet_curvature(const Neighborhood &nbh, int j, int k) const;
    void assemble_hessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
        Eigen::MatrixXd *cellhess, Eigen::MatrixXd *lrhess);

    // ── Training internals ──────────────────────────────────────────────

    // Parameter map entry: maps a flat coefficient index to a specific field
    // in pair_params or trip_params.
    enum class Field {
        // Pair fields (8)
        PAIR_EPSILON, PAIR_SIGMA, PAIR_A, PAIR_B, PAIR_P, PAIR_Q, PAIR_A_PARAM, PAIR_CUTOFF,
        // Triplet fields (7)
        TRIP_EPSILON, TRIP_SIGMA, TRIP_A, TRIP_LAMBDA, TRIP_GAMMA, TRIP_COS_THETA0, TRIP_CUTOFF
    };

    struct ParamMapEntry {
        Field field;
        int i, j, k = 0;      // indices into pair_params or trip_params
        bool is_triplet = false;
    };

    std::vector<ParamMapEntry> param_map_;
    std::vector<double> coeffs_;   // flat coefficient array

    void build_param_map();
    void unpack_coeffs();
    void set_single_param(int idx, double val);
    double get_single_param(int idx) const;
};

} // namespace Potentials
