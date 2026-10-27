// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential.h"

namespace Potentials {

/// \brief Tersoff bond-order potential with optional Brenner variant.
///
/// Two modes are supported:
/// - Mode::Tersoff: standard Tersoff (1988) form
///   V_R = A * exp(-lambda1 * r)
///   V_A = B * exp(-lambda2 * r)
///   b_ij = (1 + beta^n * zeta^n)^(-1/(2n))
///   zeta includes exp(lambda3^3 * (r_ij - r_ik)^3)
///
/// - Mode::Brenner: Brenner REBO-style form (e.g. Fe-H systems)
///   V_R = D0/(S-1) * exp(-sqrt(2*S) * beta * (r - r0))
///   V_A = D0*S/(S-1) * exp(-sqrt(2/S) * beta * (r - r0))
///   chi_ij = (1 + zeta)^(-1/2)
///   zeta has NO distance-difference exponential term
///   Angular function includes a gamma pre-factor
class Tersoff : public AnyPotential {
    friend struct GPMCodecAccess;
public:
    /// \brief Potential mode selector.
    enum class Mode { Tersoff, Brenner };

    Tersoff() = default;
    Tersoff(const std::string &filename,
            const std::map<std::string, int> &typemap);
    ~Tersoff() = default;

    void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override;
    bool supports_analytical_hessian() const override;
    bool supports_analytical_hessian_vector_product() const override;
    bool supports_analytical_cell_hessian() const override;
    bool supports_site_hessian() const override;
    bool supports_site_hvp() const override;
    void evaluate_site_hessian(Neighborhood &nbh, double &energy,
        SiteCoordinateMatrix &gradient, HessianMatrix &hessian) override;
    void apply_site_hvp(Neighborhood &nbh, const SiteCoordinateMatrix &direction,
        double &energy, SiteCoordinateMatrix &gradient, SiteCoordinateMatrix &product) override;
    using AnyPotential::applyHessianVectorProduct;
    void applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
        const Eigen::VectorXd &direction, Eigen::VectorXd &product) override;
    void getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) override;
    void getEFSHessian(Atoms &atom, HessianMatrix &hessian) override;
    void getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) override;
    void getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double symprec = 1e-4) override;
    // Native cell chart r_i=f_i*C+u_i. Cutoff joins retain the existing
    // piecewise branch; second derivatives are queried away from the joins.
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
    void Load(std::istream &is) override;
    void Save(std::ostream &os) override;
    void Save(const std::string &fnm) override;

    /// Set the potential mode (default Tersoff).
    void set_mode(Mode m) { mode_ = m; }
    Mode get_mode() const { return mode_; }

    /// \brief Per-pair parameters. Some fields are mode-specific.
    struct Param {
        // ── Common ──
        double R = 0, D = 0;           // cutoff: on at R-D, off at R+D
        double c = 0, d = 0, h = 0;    // angular function g(theta)

        // ── Tersoff mode ──
        double A = 0, B = 0;           // repulsive / attractive prefactors
        double lambda1 = 0, lambda2 = 0; // repulsive / attractive exponents
        double lambda3 = 0, m = 3.0;   // zeta distance-difference exponent
        double n = 0;                  // bond-order power

        // ── Common (formula differs per mode) ──
        double beta = 0;               // exponent / bond-order weight

        // ── Brenner mode ──
        double D0 = 0;                 // dimer energy
        double S = 0;                  // repulsive-to-attractive ratio
        double r0 = 0;                 // equilibrium distance
        double gamma = 0;              // angular pre-factor
    };

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

    // Expose parameters for pot2table
    const std::vector<std::vector<Param>> &get_params() const { return params; }
    int get_nelements() const { return nelements; }

private:
    void site_directional(Neighborhood &nbh, const SiteCoordinateMatrix &direction,
        double &energy, SiteCoordinateMatrix &gradient, SiteCoordinateMatrix &product);
    void assemble_hessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
        Eigen::MatrixXd *cellhess, Eigen::MatrixXd *lrhess);
    Mode mode_ = Mode::Tersoff;
    int nelements = 0;
    std::vector<std::string> element_names_;
    void load_stream(std::istream &is, const std::map<std::string, int> &typemap);
    std::vector<std::vector<Param>> params;
    double cut_global = 0;

    // ── Training internals ──────────────────────────────────────────────

    // Fields that can be trained
    enum class Field {
        R, D, c, d, h,              // common (5)
        A, B, lambda1, lambda2, lambda3, m, n, beta,  // Tersoff (8)
        D0, S, r0, gamma            // Brenner (4)
    };

    struct ParamMapEntry {
        Field field;
        int i, j;                   // indices into params[i][j]
    };

    std::vector<ParamMapEntry> param_map_;
    std::vector<double> coeffs_;    // flat coefficient array

    void build_param_map();
    void unpack_coeffs();
    void set_single_param(int idx, double val);
    double get_single_param(int idx) const;

    // Which fields are active (mode-dependent)
    bool is_field_active(Field f) const;
};

} // namespace Potentials
