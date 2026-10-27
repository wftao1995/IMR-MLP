// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "potential_mtp.h"
#include "NNBase/nn1hidden.h"
#include "NNBase/nn2hidden.h"

namespace Potentials {
    class PotMTPNN1 : public AnyPotential {
    friend struct GPMCodecAccess;
    public:
        bool supports_site_efs() const override {
            return p_mtp && p_mtp->supports_site_efs() &&
                (nnlay > 1 ? p_nn2 != nullptr : p_nn != nullptr);
        }
        void CalcSiteEFS(Neighborhood &nbh, double &energy, SiteCoordinateMatrix &gradient) override;
        bool supports_site_parameter_gradients() const override { return true; }
        void set_element_map(const std::map<int, std::string> &mapping) override {
            if (p_mtp) p_mtp->set_element_map(mapping);
        }
        IMRMLPNN::NN1lay *p_nn = nullptr;
        IMRMLPNN::NN2lay *p_nn2 = nullptr;
        PotMTP *p_mtp = nullptr;
        std::vector<double> species_coeffs; // last term in grad_accu
        int nnlay = 1;

        PotMTPNN1(std::string potfile, GPSettings settings = GPSettings()) {
            InitSettings();
            ApplySettings(settings);
            Load(potfile);
        }

        ~PotMTPNN1() {
            if (p_mtp) delete p_mtp;
            if (p_nn) delete p_nn;
            if (p_nn2) delete p_nn2;
        }

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
        // Native cell chart r_i=f_i*C+u_i, with row-major cell components.
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

        int nn_coeff_size = 0;

        PotMTPNN1(GPSettings settings);

        // THESE ARE TEMPLATE VARIABLES
        // DO NOT USE THIS ELSEWHERE
        std::string mtpfile = "potential.mtp";
        std::string nnfile = "potential.nn";
        std::string spcfile = "potential.sc";
        int nn_nhidden = 10;
        int nn_nhidden2 = 0;
        void InitSettings() {
            Connect(mtpfile, "mtpcoeff");
            Connect(nnfile, "nncoeff");
            Connect(spcfile, "spcoeff");
            Connect(nn_nhidden, "nnhidden");
            Connect(nn_nhidden2, "nnhidden2");
            Connect(nnlay, "nnlay");
            Connect(train_nn_only, "train_nn_only");
            Connect(random_init_mtp, "mtp_randominit");
        }

        void getEFS(Atoms &atom, AnyNeighborhoods &nbhs);
        void getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs);
        void getE(Atoms &atom, AnyNeighborhoods &nbhs);
        void getEOMP(Atoms &atom, AnyNeighborhoods &nbhs);
        void omp_init();

        // For training
        bool train_nn_only = false;
        bool random_init_mtp = true;
        virtual void random_init();
        virtual void train_init();
        virtual void set_scale(double s);
        virtual int get_NL_coeffs_count() const;
        std::vector<ConditionalLinearBlock> conditional_linear_blocks(double) const override;
        virtual void nl_get_coeffs(double *x) const;
        virtual void nl_set_coeffs(const double *x);
        virtual void AccumulateFPGard(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,       // 2*wgt_eng*(E_ref - E_efs) for MSE loss
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight); 
        virtual void Save(const std::string &potfile);
        virtual void Load(const std::string &potfile);
        virtual void AddPenaltyGard(double weight, double &loss, double *loss_gard);
    private:
        void evaluate_site_kernel(Neighborhood &nbh, double &energy,
            SiteCoordinateMatrix &gradient, HessianMatrix &hessian,
            const SiteCoordinateMatrix *direction = nullptr, SiteCoordinateMatrix *product = nullptr);
        void assemble_hessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
            Eigen::MatrixXd *cellhess, Eigen::MatrixXd *lrhess);
        PotMTPNN1() = default;
    };
};
