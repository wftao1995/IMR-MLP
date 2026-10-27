// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once 

#include "potential.h"
#include "potential_mtp.h"
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * @brief 
 * 
 * Hybrid two or three MTP potentials.
 * @todo Support additional submodels and ZBL in this HMTP variant.
 * 
 * Sub-potentials may use different moment and radial bases.
 * 
 * @todo: How to set scaling factors???? may be grid search
 * 
 */

namespace Potentials {

    // three mtp potential, each has its own interval
    class PotMTPHyb3 : public AnyPotential {
    friend struct GPMCodecAccess;
    private:
        int pcount = 2; // assume two potentials.
        // AnyScalarFunction *pscalar = nullptr;
        // AnySwitchFunction *pswitch = nullptr;
        PotMTP *pot1 = nullptr; // inner
        PotMTP *pot2 = nullptr; // inter
        PotMTP *pot3 = nullptr; // outer

        double global_scaling = 1.0;

        bool empty = true;
        bool needfree = false;

        // pot1 in [0, r1], pot2 in [0, r2], pot3 in [r2, r3]
        double r1 = 0, r2 = 0, r3 = 0, rc;

        // init when read potential
        int lin_coeff_size; // pcount*species_count and moment coeffs count
        // int nl_coeff_size; // non-linear coeff size
        int reg_coeff_size; // regression coeff size of all potentials

        // int l_offset1, l_offset2, l_offset3; // linear coeff offset
        int c_offset1, c_offset2, c_offset3; // coeffs offset

        int l_size1, l_size2, l_size3;

    public:
        std::vector<const PotMTP *> children() const {
            std::vector<const PotMTP *> result;
            for (const auto *child : {pot1, pot2, pot3})
                if (child) result.push_back(child);
            return result;
        }
        std::vector<double> reg_L2_vector; // of linear_coeff_size, 
        std::vector<double> reg_coeffs; // collection of regression coeffs

        PotMTPHyb3(std::string pfile, int plev) {
            Load(pfile, plev);
            setup();
        }

        void setup() {
            if (!pot1 || !pot2 || pot1->species_count != pot2->species_count ||
                (pot3 && pot3->species_count != pot1->species_count))
                GPERROR("HMTP requires two or three MTPs with matching species counts");
            if (pot3) pcount = 3;
            else pcount = 2;
            c_offset1 = 0;
            c_offset2 = pot1->get_coeff_size();
            c_offset3 = c_offset2 + pot2->get_coeff_size();
            // l_offset1 = c_offset1;
            // l_offset2 = c_offset2;
            // l_offset3 = c_offset3;
            r1 = pot1->get_cutoff();
            r2 = pot2->get_cutoff();
            if (pot3) r3 = pot3->get_cutoff();
            l_size1 = pot1->get_lin_coeff_size();
            l_size2 = pot2->get_lin_coeff_size();
            if (pot3) l_size3 = pot3->get_lin_coeff_size();

            reg_coeff_size = pot1->get_coeff_size() + pot2->get_coeff_size();
            if (pot3) reg_coeff_size += pot3->get_coeff_size();
            lin_coeff_size = pot1->get_lin_coeff_size() + pot2->get_lin_coeff_size();
            if (pot3) lin_coeff_size += pot3->get_lin_coeff_size();

            rc = std::max({r1, r2, r3});
            cutoff = rc;
            cutoffsq = rc * rc;
            species_count = pot1->species_count;
            cutoffs = Eigen::MatrixXd::Constant(species_count, species_count, rc);
            cutoffsqs = cutoffs.array().square().matrix();
            r_mins = Eigen::MatrixXd::Zero(species_count, species_count);
            empty = !(pot1->inited() && pot2->inited() && (!pot3 || pot3->inited()));

            global_scaling = pot1->get_scale();
        }

        PotMTPHyb3(PotMTP *_pot1, PotMTP *_pot2, PotMTP *_pot3) {
            pot1 = _pot1;
            pot2 = _pot2;
            pot3 = _pot3;
            setup();
        }

        ~PotMTPHyb3() {
            if (needfree) {
                if (pot1) delete pot1;
                if (pot2) delete pot2;
                if (pot3) delete pot3;
            }
        }

        bool supports_site_efs() const override {
            return pot1 && pot2 && pot1->supports_site_efs() &&
                pot2->supports_site_efs() && (!pot3 || pot3->supports_site_efs());
        }
        void CalcSiteEFS(Neighborhood &nbh, double &energy, SiteCoordinateMatrix &gradient) override;
        double get_scale() const { return global_scaling; };
        void set_scale(double scale) {
            global_scaling = scale; 
            pot1->set_scale(global_scaling);
            pot2->set_scale(global_scaling);
            if (pot3) pot3->set_scale(global_scaling);
        };

        int get_coeff_size() { return reg_coeff_size; };

        void collect_coeffs() {
            reg_coeffs.resize(reg_coeff_size);
            if (pot1) {
                memcpy(reg_coeffs.data(), pot1->get_inter_pointer_coeffs(), pot1->get_coeff_size() * sizeof(double));
            }
            if (pot2) {
                memcpy(reg_coeffs.data() + c_offset2, pot2->get_inter_pointer_coeffs(), pot2->get_coeff_size() * sizeof(double));
            }
            if (pot3) {
                memcpy(reg_coeffs.data() + c_offset3, pot3->get_inter_pointer_coeffs(), pot3->get_coeff_size() * sizeof(double));
            }
        }

        void ae_get_coeffs(alglib::real_1d_array &_x) {
            collect_coeffs();
            _x.setcontent(reg_coeff_size, reg_coeffs.data()); 
        }

        void ae_set_coeffs(const alglib::real_1d_array &_x) {
            memcpy(reg_coeffs.data(), _x.getcontent(), reg_coeff_size * sizeof(double));
            update_coeffs();
        }

        void random_init() {
            pot1->random_init();
            if (pot2) pot2->random_init();
            if (pot3) pot3->random_init();
            empty = false;
        }

        void init_coeffs() override {
            if (!pot1->inited()) pot1->init_coeffs();
            if (!pot2->inited()) pot2->init_coeffs();
            if (pot3 && !pot3->inited()) pot3->init_coeffs();
            empty = false;
        }

        int get_NL_coeffs_count() const override { return reg_coeff_size; }
        void nl_get_coeffs(double *x) const override {
            pot1->nl_get_coeffs(x);
            pot2->nl_get_coeffs(x + c_offset2);
            if (pot3) pot3->nl_get_coeffs(x + c_offset3);
        }
        void nl_set_coeffs(const double *x) override {
            pot1->nl_set_coeffs(x);
            pot2->nl_set_coeffs(x + c_offset2);
            if (pot3) pot3->nl_set_coeffs(x + c_offset3);
        }

        void train_init() {
            pot1->train_init();
            pot2->train_init();
            if (pot3) pot3->train_init();
        }

        void update_coeffs() {
            set_coeffs(reg_coeffs.data());
        }

        void get_cutoffs(double &r1, double &r2, double &r3) {
            r1 = pot1->get_cutoff();
            r2 = pot2->get_cutoff();
            r3 = pot3->get_cutoff();
        }

        void get_cutoffs(double &r1, double &r2) {
            r1 = pot1->get_cutoff();
            r2 = pot2->get_cutoff();
        }

        void get_coeffs(Eigen::VectorXd &_x) {
            collect_coeffs();
            _x.resize(reg_coeff_size);
            memcpy(_x.data(), reg_coeffs.data(), reg_coeff_size * sizeof(double));
        }

        double get_linear_coeff(int i) {
            if (i < l_size1) {
                return pot1->get_linear_coeff(i);
            } else if (i < l_size1 + l_size2) {
                return pot2->get_linear_coeff(i - l_size1);
            } else {
                return pot3->get_linear_coeff(i - l_size1 - l_size2);
            }
        }
        // reg_vector acts on linear coeffs
        void dist_reg_vec() {
            pot1->reg_L2_vector.resize(pot1->get_lin_coeff_size());
            pot2->reg_L2_vector.resize(pot2->get_lin_coeff_size());
            if (pot3) pot3->reg_L2_vector.resize(pot3->get_lin_coeff_size());
            memcpy(pot1->reg_L2_vector.data(), reg_L2_vector.data(), l_size1 * sizeof(double));
            memcpy(pot2->reg_L2_vector.data(), reg_L2_vector.data() + l_size1, l_size2 * sizeof(double));
            if (pot3) 
                memcpy(pot3->reg_L2_vector.data(), reg_L2_vector.data() + l_size1 + l_size2, l_size3 * sizeof(double));
        }

        void init_reg_vec(double reg_param) {
            reg_L2_vector.assign(lin_coeff_size, reg_param);
            dist_reg_vec();
        }
        
        void set_linear_coeff(double *l) {
            pot1->set_linear_coeff(l);
            if (pot2) pot2->set_linear_coeff(l + l_size1);
            if (pot3) pot3->set_linear_coeff(l + l_size1 + l_size2);
        }

        void set_coeffs(const double *x) {
            pot1->set_coeffs(x);
            if (pot2) pot2->set_coeffs(x + c_offset2);
            if (pot3) pot3->set_coeffs(x + c_offset3);
        }

        void Save(const std::string &potname);

        void Load(const std::string potname, int pcount);

        // getEFSH
        void getEFS(Atoms &atom);
        void getEFS(Atoms &atom, AnyNeighborhoods &nbhs);
        // void getEFS(Atoms &atom, AnyNeighborhoods &nbhs1, AnyNeighborhoods &nbhs2);
        void getEFS(Atoms &atom, AnyNeighborhoods &nbhs1, AnyNeighborhoods &nbhs2, AnyNeighborhoods &nbhs3);
        // omp version
        void getEFSOMP(Atoms &atom);
        void getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs);
        void getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs1, AnyNeighborhoods &nbhs2, AnyNeighborhoods &nbhs3);
        // void 
        // for train
        // void CalcEFSCmpt(Atoms &atom, 
        //                  AnyNeighborhoods &nbhs1, 
        //                  AnyNeighborhoods &nbhs2, 
        //                  std::vector<double> &energy_cmpts,
        //                  Eigen::Tensor<double, 3> &force_cmpts,
        //                  Eigen::Tensor<double, 3> &stress_cmpts);

        void CalcEFSCmpt(Atoms &atom, 
                         AnyNeighborhoods &nbhs1, 
                         AnyNeighborhoods &nbhs2, 
                         AnyNeighborhoods &nbhs3,
                         std::vector<double> &energy_cmpts,
                         Eigen::Tensor<double, 3> &force_cmpts,
                         Eigen::Tensor<double, 3> &stress_cmpts);

        // void CalcECmpt(Atoms &atom, 
        //                  AnyNeighborhoods &nbhs1, 
        //                  AnyNeighborhoods &nbhs2, 
        //                  std::vector<double> &energy_cmpts);

        void CalcECmpt(Atoms &atom, 
                         AnyNeighborhoods &nbhs1, 
                         AnyNeighborhoods &nbhs2, 
                         AnyNeighborhoods &nbhs3,
                         std::vector<double> &energy_cmpts);

        void Orthogonalize() {
            if (pot1) pot1->Orthogonalize();
            if (pot2) pot2->Orthogonalize();
            if (pot3) pot3->Orthogonalize();
        }

        int get_lin_coeff_size() {
            int ret = 0;
            if (pot1) ret += pot1->get_lin_coeff_size();
            if (pot2) ret += pot2->get_lin_coeff_size();
            if (pot3) ret += pot3->get_lin_coeff_size();
            return ret;
        }

        bool inited() {return !empty;};
        void set_empty_flag() {empty = !empty;};
        void AddPenaltyGard(const double weight_penatly, double &ploss, double *pgard);
        void AccumulateEFSCombinationGard(Atoms &atom, AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &buffer,
                                          double *out_gard_accumulator) override;
        void AccumulateEFSCombinationGard(Atoms &atom, AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          double *out_gard_accumulator) override;
        void AccumulateEFSCombinationGard(Atoms &atom,
                                          AnyNeighborhoods &nbhs1,
                                          AnyNeighborhoods &nbhs2,
                                          AnyNeighborhoods &nbhs3,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          double *out_gard_accumulator);
    };

}
