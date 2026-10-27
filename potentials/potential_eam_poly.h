// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once 
/**
 * @brief EAM potential, using chebyshev polymonials 
 * For EAM/FS(HE) type:
 * Assume i has type a, j loops over neighbor of i and has type b
 * U_i = E0_a + 1/2 \sum_{j!=i} V_{ab}(r_ij) + F_a(\sum_j \phi_{ab}(r_ij))
 *       constant       pair-wise                embedding
 * Q_k is Chebyshev polynomial of order k
 * V(r) = (1 - r/rc)^2 * (\sum_k c_k Q_k(r')) = \sum_k c_k Q'_k(r')
 * \phi(r) = (1 - r/rd)^2 * (\sum_k d_k Q_k(r')) = \sum_k d_k Q'_K(r')
 * c_k, d_k are fitting coeffs; rc, rd are cutoff radius
 * r' = 2 * (r - rmin) / (rmax - rmin) - 1
 * for F_a, we can simply choose sqrt, or any other functional form, i.e polynominal. Here we use
 * F(p) =  \sum_k l_k * p^k
 * 
 * V is symmetric, i.e V_{ab} = V_{ba}, but \phi is not symmetric, i.e \phi_{ab} != \phi_{ba}
 * 
 * For EAM/ALLOY type:
 * U_i = E0_a + 1/2 \sum_{j!=i} V_{ab}(r_ij) + F(\sum_j \phi_{a}(r_ij))
 * 
 * Since only pair-wise interactions are considered directly, rotational invariance is kept.
 * 
 * TODO: try constraint \sum_j \rho(r_ij) between 0 and rho_max, or at least positive
 */
#include "potential_eam_any.h"
#include <Eigen/Dense>
#include "basis/basis.h"

namespace Potentials {
    // currently onle HE varient, TODO: add a switch parameter is_he, 
    class PotentialEAMPoly : public PotEAMAny {
    private:
        void assemble_hessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
            Eigen::MatrixXd *cellhess, Eigen::MatrixXd *lrhess);
    public: 
        /**
         * layout:
         * density_coeffs: eam/alloy: nelems * npoly_rho ; eam/fs: nelems * nelems * npoly_rho
         * embeeding_coeffs: nelems * npoly_f
         * pair_coeffs: (nelems +1) * nelems / 2 * npoly_pair
         * species_coeffs: nelems
         */
        std::vector<double> regression_coeffs; 
        int n_density_coeffs = 0, n_embeeding_coeffs = 0, n_pair_coeffs = 0, n_coeffs = 0;
        double *density_coeffs = nullptr, *embedding_coeffs = nullptr,
               *pair_coeffs = nullptr, *species_coeffs = nullptr; // segments of regression_coeffs
        std::vector<AnyBasis*> basis_rho, basis_pair; // Use ChebyshevPoly{2,3,4,5} basis
        Eigen::MatrixXd rmax_rho, rmax_pair;

        // This should be estimated from the data
        // double rho_min = 9e99, rho_max = -9e99;

        // buffers
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buff_site_energy_ders;
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buff_drhodx;
    public:
        int get_pairid(int i, int j) const {
            if (i > j) std::swap(i, j);
            // only i < j, get id form 0 
            return (j+1)*j/2 + i;
        }
        int get_rhoid(int i, int j) const {
            return i * nelems + j;
        }
        // npoly_rho, npoly_pair: order of chebyshev polynomials; npoly_f: power 1 to npoly_f, constant term is included in species_coeffs.
        int npoly_rho = 8, npoly_f = 8, npoly_pair = 8;
        int nelems = 2;
        int basis_poly = 3;
        double b = -4./3.; // for ChebyshevPoly3, 4, 5

        PotentialEAMPoly(int _npoly_rho, int _npoly_f, int _npoly_pair, int _nelems,
            Eigen::MatrixXd _rmax_pair, Eigen::MatrixXd _rmax_rho,
            int basis_poly = 3, double b = -4./3.);

        // Construct from GPSettings (template mode for potential_wrapper)
        PotentialEAMPoly(GPSettings settings,
                         const std::map<std::string, int> &mappingtype2id = {});

        // Construct by loading a tag-value file
        PotentialEAMPoly(const std::string &filename,
                         const std::map<std::string, int> &mappingtype2id = {});

        ~PotentialEAMPoly();

        using AnyPotential::applyHessianVectorProduct;
        bool supports_analytical_hessian() const override { return true; }
        void getEFSHessian(Atoms &atoms, AnyNeighborhoods &nbhs, HessianMatrix &hessian) override;
        void getEFSHessian(Atoms &atoms, HessianMatrix &hessian) override;
        void getEFSHessianSymm(Atoms &atoms, HessianMatrix &hessian, double symprec = 1e-4) override;
        bool supports_analytical_cell_hessian() const override { return true; }
        void getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
        void getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) override;
        void getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) override;
        bool supports_analytical_hessian_vector_product() const override { return true; }
        void applyHessianVectorProduct(Atoms &atoms, AnyNeighborhoods &nbhs,
            const Eigen::VectorXd &direction, Eigen::VectorXd &product) override;
        void getEFS(Atoms &atoms, AnyNeighborhoods &nbhs);
        bool supports_site_parameter_gradients() const override { return true; }
        void CalcSiteEFS(Neighborhood &nbh, double &site_en, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &site_energy_ders) override;
        void CalcSiteE(Neighborhood &nbh, double &site_en) override;
        void CalcSiteEFS(Neighborhood &nbh, double &site_en, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &site_energy_ders, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &drhodx);
        
        void setup();
        // Tag-value format, with row-major cutoff matrices and optional coeffs.
        // Missing/partial coefficient lists remain zero-filled training templates;
        // malformed scalar values or incomplete cutoff matrices are rejected.
        // Streams and files use the same format, with full double precision.
        void Load(std::istream &is) override;
        void Save(std::ostream &os) override;
        void Load(const std::string &filename) override;
        void Save(const std::string &potfile) override;
        void random_init();
        int get_NL_coeffs_count() const {return n_coeffs;};
        virtual void nl_get_coeffs(double *) const;
        virtual void nl_set_coeffs(const double *);
        virtual void train_init();
        virtual void AddPenaltyGard(double weight, double &loss, double *loss_gard);
        // Direct evaluation helpers for pair/density/embedding at arbitrary r/rho
        double eval_pair(int itype, int jtype, double r);
        double eval_density(int itype, int jtype, double r);
        double eval_embed(int itype, double rho);
        virtual void OutputLAMMPSset(std::string outfile, int nr, int nrho, double rhomin = -1, double rhomax = -1);
        // compute dU_i/dx and d^2U_i/dR_{ija}dx for single i
        virtual void AccumulateFPGard(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,       // 2*wgt_eng*(E_ref - E_efs) for MSE loss
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight); 
    private:
        void load_stream(std::istream &is, const std::string &source);
    };
};
