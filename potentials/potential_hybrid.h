// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential.h"
#include <algorithm>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>

// for each sub-stype
#include "potential_mtp.h"
#include "potential_zbl.h"

/**
 * @brief Hybrid mtp with specific variable
 * 
 * Currently supported variables: r_{min}^{typei, typej}
 * 
 */

// namespace SwitchFuncs {
//     double switchfunc1(double, double, double);
// };

namespace Potentials {
    /**
     * @brief Hybrid potential
     * strightforward scheme:
     * @f$E = \sum_i s_i E_i@f$
     * s_i is constant scaling factor for each type. 
     * e.g. MTP + ZBL
     * 
     * Supports MTP and ZBL sub-potentials initialized from GPSettings.
     * Training accumulates MTP parameter gradients; ZBL parameters remain fixed.
     * Each sub-potential contributes with add_efs enabled.
	 * 
	 * NOTE: DO NOT USE TRAINING METHOD IN POTMTP
     * 
     * For MTP type potential, regression coeffs will go into regression_coeffs.
     */
    class PotentialHybrid : public AnyPotential {
    friend struct GPMCodecAccess;
    public:
        void set_element_map(const std::map<int, std::string> &mapping) override {
            for (auto &pot : ppots) pot->set_element_map(mapping);
        }
        int npot = 0;  
        int ntrains = 0;
        std::vector<int> train_ids;
        std::vector<std::string> potnames;
        double scaling = 1.0;
        // layout
        // radial_coeffs_p1, radial_coeffs_p2, .., species_coeffs, moment_coeffs_p1, moment_coeffs_p2, ...
        double *regression_coeffs = nullptr; // regression coeffs, put all data here for all MTP type potentials
        double *radial_coeffs = nullptr;
        double *linear_coeffs = nullptr;
        double *species_coeffs = nullptr;
        double *moment_coeffs = nullptr;

        std::vector<const AnyPotential *> children() const {
            std::vector<const AnyPotential *> result;
            for (const auto &child : ppots) result.push_back(child.get());
            return result;
        }
        bool supports_site_efs() const override;
        void CalcSiteEFS(Neighborhood &, double &, SiteCoordinateMatrix &) override;

        int nreg_coeffs = 0;
        int nrad_coeffs = 0;
        int nlin_coeffs = 0; // = nmom_coeffs + species_count
        int nmom_coeffs = 0;
        std::vector<std::unique_ptr<Potentials::AnyPotential>> ppots;
        std::vector<double> cutoffs;
        // bool has_scale = false;
        // std::vector<double> scale;
        double random_upper = 0.1, random_lower = -0.1;

    public:
        PotentialHybrid(GPSettings settings, std::vector<std::unique_ptr<AnyPotential>> children = {});

        ~PotentialHybrid() override = default;
        PotentialHybrid(const PotentialHybrid &) = delete;
        PotentialHybrid &operator=(const PotentialHybrid &) = delete;

        void Save(const std::string &fnm) override {
            if (save_gpm_if_requested(*this, fnm)) return;
            for (auto &i : train_ids) {
                dynamic_cast<PotMTP*>(ppots[i].get())->Save("HMTP_" + std::to_string(i+1) + "_" + fnm);
            }
        }

        // CAUSION!!!!
        double *get_inter_pointer_coeffs() { return regression_coeffs; }; 
        double *get_inter_pointer_linear_coeffs() { return linear_coeffs; };
        double *get_inter_pointer_radial_coeffs() { return radial_coeffs; };

        void set_inited_flag() {
            for (auto &it : train_ids) {
                dynamic_cast<PotMTP*>(ppots[it].get())->empty = false;
            }
        }

        double get_scale() {
            return scaling;
        }

        int get_lin_coeff_size() {
            return nlin_coeffs; 
        }

        double get_linear_coeff(int id) {
            return linear_coeffs[id];
        }

        void set_scale(double s) {
            scaling = s;
            for (int i = 0; i < ntrains; ++i) {
                ppots[train_ids[i]]->set_scale(s);
            }
        }

        void train_init();

        void ae_get_coeffs(alglib::real_1d_array &_x) const {
            for (int i = 0; i < nreg_coeffs; ++i) {
                _x[i] = regression_coeffs[i];
            }
        }

        void ae_set_coeffs(const alglib::real_1d_array &_x) {
            for (int i = 0; i < nreg_coeffs; ++i)
                regression_coeffs[i] = _x[i];
        }

        void ae_get_coeffs(alglib::real_1d_array &_x, bool fix_radial, bool fix_linear) const {
            if (fix_radial) {
                for (int i = 0; i < nlin_coeffs; ++i) {
                    _x[i] = linear_coeffs[i];
                }
            } else if (fix_linear) {
                for (int i = 0; i < nrad_coeffs; ++i) {
                    _x[i] = radial_coeffs[i];
                }
            } else {
                for (int i = 0; i < nreg_coeffs; ++i)
                    _x[i] = regression_coeffs[i];
            }
        }

        void ae_set_coeffs(const alglib::real_1d_array &_x, bool fix_radial, bool fix_linear) {
            if (fix_radial) {
                for (int i = 0; i < nlin_coeffs; ++i) {
                    linear_coeffs[i] = _x[i];
                }
            } else if (fix_linear) {
                for (int i = 0; i < nrad_coeffs; ++i) {
                    radial_coeffs[i] = _x[i];
                }
            } else {
                for (int i = 0; i < nreg_coeffs; ++i) {
                    regression_coeffs[i] = _x[i];
                }
            }
        }

        bool supports_analytical_cell_hessian() const override;
        void getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs,
            HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
        void getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
        void getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs,
            HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) override;
        void getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) override;
        void getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
            Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) override;
        bool supports_analytical_hessian() const override;
        void getEFSHessian(Atoms &atom, HessianMatrix &hessian) override;
        void getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs,
            HessianMatrix &hessian) override;
        void getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs,
            HessianMatrix &hessian) override;
        void getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian,
            double symprec = 1e-4) override;

        using AnyPotential::applyHessianVectorProduct;
        bool supports_analytical_hessian_vector_product() const override;
        void applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
            const Eigen::VectorXd &direction, Eigen::VectorXd &product) override;

        void getEFS(Atoms &atom) override;
        void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override;
        void getEFSOMP(Atoms &atom) override;
        void getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs) override;
        void getEFSOMPthread(Atoms &atom) override;
        void getEFSOMPthread(Atoms &atom, AnyNeighborhoods &nbhs) override;
        void getE(Atoms &atom) override;
        void getE(Atoms &atom, AnyNeighborhoods &nbhs) override;

        int get_NL_coeffs_count() const override { return nreg_coeffs; }
        void nl_get_coeffs(double *coeffs) const override {
            std::copy_n(regression_coeffs, nreg_coeffs, coeffs);
        }
        void nl_set_coeffs(const double *coeffs) override {
            std::copy_n(coeffs, nreg_coeffs, regression_coeffs);
        }
        std::vector<NLParameterGroup> nl_parameter_groups() const override {
            if (train_ids.empty()) return {};
            return {
                {"radial", 0, nrad_coeffs},
                {"species", nrad_coeffs, nrad_coeffs + species_count},
                {"moments", nrad_coeffs + species_count, nreg_coeffs},
            };
        }
        void random_init() override {
            random_init_radial();
            random_init_linear();
            set_inited();
        }
        void init_coeffs() override { set_inited(); }
        void set_inited() override {
            nl_inited = true;
            set_inited_flag();
        }

        // For training
        void CalcEFSCmpt(Atoms &atom, AnyNeighborhoods &nbhs, 
                            std::vector<double> &ecmpt, 
                            Eigen::Tensor<double, 3> &fcmpt,
                            Eigen::Tensor<double, 3> &scmpt);
        
        void CalcECmpt(Atoms &atom, AnyNeighborhoods &nbhs, 
                            std::vector<double> &ecmpt);

        void Orthogonalize();

        void set_moment_coeff(double *);
        void set_linear_coeff(double *);

        // if not inited, then init
        void train_coeff_init(bool random = false);

        void random_init_linear();
        void random_init_radial();

        void set_radial_regtype(int mode, double wgt_rad_w0, double wgt_rad_w1, double wgt_rad_w2, double randlow, double randupw, int quad_point);

        void InitSettings() {
            // example
            /**
             * npot  : number of potentials
             * pot1:pottype  type of pot1    
             * pot1:xxx    settings of pot1
             * pot2:pottype  type of pot2
             * pot2:xxx    settings of pot2
             */
            // Connect(has_scale, "scale", "enable constant scaling factor for each substyle or not");
            Connect(npot, "npot", "number of potentials");
            Connect(species_count, "spcount", "number of species");
        };

        void AddPenaltyGard(const double weight, double &loss, double *loss_gard, bool fix_radial, bool fix_linear, bool fix_species);
        void AddPenaltyGard(double weight, double &loss, double *loss_gard) override {
            AddPenaltyGard(weight, loss, loss_gard, false, false, false);
        }

        void AccumulateEFSCombinationGard(Atoms &atom, AnyNeighborhoods &nbhs,
                                           double dLdE, Eigen::Matrix<double, -1, 3> &dLdF,
                                           Eigen::Matrix3d &dLdS, double *out_gard_accumulator);
        void AccumulateEFSCombinationGard(
            Atoms &atom, AnyNeighborhoods &nbhs, double dLdE,
            Eigen::Matrix<double, -1, 3> &dLdF, Eigen::Matrix3d &dLdS,
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &gradient_buffer,
            double *out_gard_accumulator) override;
        void AccumulateEFSCombinationGard(
            Atoms &atom, AnyNeighborhoods &nbhs, double dLdE,
            double *out_gard_accumulator) override;
        
        void AccumulateEFSCombinationGardRonly(Atoms &atom, AnyNeighborhoods &nbhs,
                                           double dLdE, Eigen::Matrix<double, -1, 3> &dLdF,
                                           Eigen::Matrix3d &dLdS, double *out_gard_accumulator);

        void AccumulateEFSCombinationGardLonly(Atoms &atom, AnyNeighborhoods &nbhs,
                                           double dLdE, Eigen::Matrix<double, -1, 3> &dLdF,
                                           Eigen::Matrix3d &dLdS, double *out_gard_accumulator);

    private:
        std::unique_ptr<double[]> regression_storage_;
        std::vector<std::vector<double>> secondary_species_coeffs_;
    };
    /**
     * @brief 
     * Each style weight is calculated by a weight function.
     * One need to define order parameters
     */
    // class PotentialHybridScaleFunc : public AnyPotential {
    //     ;
    // };

};
