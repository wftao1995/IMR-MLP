// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base/neighborhood.h"
#include "potential.h"
#include "basis_cost.h"
#include "mtp_radial_local_cache.h"
#include "tools/tools.h"
#include "tools/mtp_metadata.h"
#include "basis/basis.h"
#include "libalglib/ap.h"
#include <Eigen/Sparse>
#include <unordered_map>
#include <utility>
#include <unsupported/Eigen/CXX11/Tensor>
#include <type_traits>
#include <memory>
#include <typeinfo>

/****
 * MTP potential
 *
 * scaling:
 * species_count: number of species
 * min_dist, max_dist: orthogonal interval of radical functions
 * radical_basis_size: number of radical size, i.e.
 *
 * The algo as follows:
 * First, a scalar/non-scalar valued B_\alpha can decomposed into contractions of B_\beta and B_\eta.
 * Each element b_j in B_\alpha can be expressed by b_j = \sum c_{jkl} * b_l * b_k, where c_j is a multiplication factor caused by symmetry,
 *      b_l, b_k are elements of B_\beta and B_\gamma, respectively.
 * Doing this recursively, we get a decomposition tree which lowest nodes are moment tensors.
 * The element of a moment tensor can be expressed as：
 *      bi = f_{\mu,\nu}(|r|)*x^a*y^b*z^c/r^\nu, where a + b + c = \nu, the r^(-\nu) term is a normalization term to avoid too large values.
 * Using Chebyshev f_{\mu}(r),
 *      f_{\mu}(r) = \sum_k c_{\mu,z_i,z_j}^{k} Q^{k}(r)
 *
 * When calcualte forces:
 *      We have  b_j = \sum c_{jkl} * b_l * b_k, thus
 *      d b_j / d b_k = \sum c_{jkl} * b_l
 *      d E / d b_k = \sum c_{jkl} d E / d b_j * d b_j / d b_k
 *                  = \sum c_{jkl} d E / d b_j * b_l
 * Starting from d E / d b_{scalar} = liner_coeffs, we can get d E / d b_{basic} using backpropagation.
 * It's clear that E and be written as functions of only BASIC b_i s, thus
 *      \partial E / \partial x_{j,a} = \sum_i \partial E / \partial b_j * \partial b_j / \partial x_{j,a}
 * So does E wrt coeffs.
 * For E wrt coeffs, we first compute \partial b_i / \partial c for basic terms, then,
 *      \partial b_j / \partial c = c_{ijk} * (\partail b_l / \partial c * b_k + \partail b_k / \partial c * b_l)
 * and we get all d moments / d coeffs
 * For F wrt coeffs, we first compute \partial^2 b_i / (\partial x \partial c) for basic terms, then,
 *      \partial^2 b_j / (\partial x \partial c) = c_{ijk} * (
 *                                                 \partial b_l / \partial c * \partial b_k / \partial x +
 *                                                 \partial b_l / \partial x * \partail b_k / \partail c +
 *                                                 \partial^2 b_l / (\partail x \partial c) * b_k +
 *                                                 \partial^2 b_k / (\partail x \partial c) * b_l
 *                                                  )
 * for non-basic terms.
 * For linear part, just scalar moment values
 * For E wrt coeffs, we write:
 *      \partial F / \partial c = \partial/\partial c (\partial E / \partail x)
 *                              = \sum_i \partial/\partial c (\partial E / \partial b_i * \partial b_i / \partial x)
 *                              = \sum_i (\partail^2 E / (\partail b_i \partial c) * \partial b_i / \partial x + \partial E / \partial b_i * \parital^2 b_i / (\partial x \partial c) )
 * OR  we have
 *      \partial F / \partial c = \partial/\partial x (\partial E / \partail c)
 *                              = \sum_i \partial/\partial x (\partial E / \partial b_i * \partial b_i / \partial c)
 *                              = \sum_i (\partail^2 E / (\partail b_i \partial x) * \partial b_i / \partial c + \partial E / \partial b_i * \parital^2 b_i / (\partial x \partial c) )
 * We choose the later one since it can be merged with dE/dc (with same part db_i/dc)
 * For linear part, just dB/dx
 *
 * @todo: add repelsive potential for inner part
 * @todo: train inner repulsize potential seperatelly
 * Species pairs can use separate cutoffs and, with radial_split, radial bases.
 * ****/
namespace Potentials {
    class LES;
    /**
     * @brief Moment Tensor Potential (MTP) implementation (Shapeev 2016 formalism).
     * @ingroup potentials
     *
     * The potential energy of an atom is expressed as a linear combination of moment tensor
     * invariants B_alpha built recursively from elementary moment tensors via complete Cartesian
     * index contractions. Supports OpenMP parallelism, Hessian computation, and external NN models
     * through template-based CalcSiteEFS<Model> methods.
     *
     * Coefficient layout: radial_coeffs + species_coeffs + moment_coeffs.
     *
     * @see AnyPotential, Trainer_MTP
     */
    class PotMTP : public AnyPotential {
        friend class Trainer_MTP;

    public:
        void set_element_map(const std::map<int, std::string> &mapping) override {
            if (mapping.empty()) return;
            std::vector<std::string> names;
            for (const auto &[id, name] : mapping) {
                MTPMetadata::require(id == static_cast<int>(names.size()), "element map must use consecutive species IDs from zero");
                names.push_back(name);
            }
            potential_tag.set_elements(names, species_count);
        }

        bool coeffs_handled_external = false;
        bool read_flag_immediate_return = false;
        bool read_flag_early_exit = false;
        bool radial_basis_split = false; // use different radial basis for different pair types
        bool radial_local_cache = false;
        bool full_symmetry_reuse = false;
        std::size_t last_symmetry_site_evaluations = 0;


        MTPMetadata::Tag potential_tag;
        virtual std::string metadata_model_type() const { return "MTP"; }
        virtual MTPMetadata::Object model_metadata() const {
            return {{"type",MTPMetadata::Value("MTP")},{"model",MTPMetadata::Value("linear")}};
        }
        std::vector<std::string> element_symbols() const { return potential_tag.elements(); }
        void set_element_symbols(const std::vector<std::string> &names) { potential_tag.set_elements(names,species_count); }
        int reg_coeff_size = 0;    // Total coeff size
        int species_coeff_size = 0; // currently only for mtp_m
        int moment_coeff_size = 0; // currently only for mtp_m
        int lin_coeff_size = 0;    // linear coeff size, = moment_coeffs_size + species_coeffs_size, this won't use in mtp_m
        int radial_coeff_size = 0; // radial coeff size

        int alpha_moments_count = 0;     // number of moments, basic + non-basic, length of moment_vals
        int alpha_index_basic_count = 0; // number of elementary b_i
        int alpha_index_times_count = 0; // length of alpha_index_times, number of rules for non-elementary b_i
        int alpha_scalar_moments = 0;    // number of scalar moments, also length of alpha_moment_mapping
        int *alpha_moment_mapping = nullptr; // mapping of b_i to scalar B_\alpha

        Eigen::Matrix<int, -1, -1, Eigen::RowMajor> alpha_index_basic; // {{mu, beta1, beta2, beta3},...}, elementary b_i
        Eigen::Matrix<int, -1, -1, Eigen::RowMajor> alpha_index_times; // tuples for calculating non-elementary b_i, {id1, id2, times, bid}

        // radical basis
        int rb_size = 0;
        int radial_func_count = 0;
        BasisType btype;
        int radial_mode = 0; // 0 for single radial, 1 for multiple radial for each species pair
        AnyBasis *p_radical_basis = nullptr; // Shared radial basis when radial_basis_split is false.
        std::vector<AnyBasis*> p_radical_basis_pair;

        // regression coeffs
        double *regression_coeffs = nullptr; // all coeffs, layout: radial coeffs c_{z_i,z_j,\mu,k} + species_coeffs + moment_coeffs
        // just pointers, points to segments of regression_coeffs
        double *radial_coeffs = nullptr;     // newly added, radical
        double *linear_coeffs = nullptr;     // spice_coeffs + moment_coeffs
        double *species_coeffs = nullptr;    // newly added, constant term
        double *moment_coeffs = nullptr;     // newly added, moment term
        int *nu_arr_list = nullptr;
        int max_alpha_index_basic = 0; // max power + 1, nv max
        int max_mu = 0;

        bool empty = false;
        bool radial_empty = true;
        bool linear_empty = true;

        bool hc_correction = false;
        bool rR_correction = false;

        // level specific
        int max_level = 0;
        std::vector<int> lev_scalar_nodes;
        std::vector<int> lev_scalar_count;
        std::vector<int> lev_times_nodes;
        std::vector<int> lev_times_count;

        // Buffers
        struct ompbuffers {
            int thread_id;
            MTPadialLocalCache radial_cache;

            double energy;
            std::vector<Eigen::Vector3d> force;
            std::vector<double> siteenergy;
            Eigen::Matrix3d stress;
            HessianMatrix hessian;
            std::vector<double> hessian_diag;

            std::vector<std::vector<double>> buff_moment_hessian_diag, buff_basis_derders_diag;
            std::vector<double> buff_site_hessian_diag;

            double buff_site_energy;
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buff_site_energy_ders;
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buff_site_hvp;
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buff_site_hvp_direction;
            std::vector<double> basis_vals;
            Eigen::Tensor<double, 3, Eigen::RowMajor> basis_ders;
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> coords_powers;
            std::vector<double> moment_vals;
            Eigen::Tensor<double, 3> moment_ders;
            std::vector<double> dist_powers; // rij^k
            std::vector<double> rb_vals;
            std::vector<double> rb_ders;
            std::vector<double> rb_derders;
            std::vector<double> val_mu;                                 // basis function value
            std::vector<double> der_mu;                                 // basis function der wrt r
            std::vector<double> derder_mu;
            Eigen::Tensor<double, 3, Eigen::RowMajor> moment_jacobian;           // dbi/dx_{ja}, basic only
            Eigen::Tensor<double, 3, Eigen::RowMajor> Q_jacobian;           // dQ/dx_{ja}
            // FC2-only compact storage: basic moments are sums over neighbours,
            // so their coordinate Hessians contain diagonal 3x3 blocks only.
            std::vector<Eigen::Matrix3d> basic_moment_hessian_diag;
            std::vector<double> moment_adjoint;
            std::vector<HessianMatrix> moment_hessian, basis_derders, Q_hessian;  // dbi/dx_{ja}dx_{kb}, (i,j,a,k,b)
            std::vector<HessianMatrix> moment_hessian_sparse, basis_derders_sparse;
            HessianMatrix buff_site_hessian;
            double d2fdrab[3][3];
            double dfdra[3];
            double dmdra[3];
            double dpdra[3];
            double d2mdrab[3][3];
            double d2pdrab[3][3];

            Eigen::Tensor<double, 3, Eigen::RowMajor> moment_jacobian_coeffs;    // dbi/dradial_coeffs, basic only
            Eigen::Tensor<double, 3, Eigen::RowMajor> moment_jacobian_coeff_coord; // \partial^2 bi / (\partial x_{ja} \ partial coeffs)
            std::vector<double> site_energy_ders_wrt_moments_;          // dE/dbi, all, but only basic is used, however basic must be calculated via BP process
            std::vector<double> dloss_dmom;                             // = se_weight * buff_site_energy_ders_wrt_moments for basic inds when dE/dc requested only, and = se_weight * (d^2E/dbidx + dE/dbi) * dbi/dc when force der is also requested
            std::vector<double> dloss_dsenders;                         // = dB/dx
            Eigen::Tensor<double, 3> arraybuffer3d;             // any array buffer
            Eigen::Matrix<double, -1, 3> matrixx3dbuffer;          // any array buffer

            std::vector<double> dSEdsi; // site-energy der_wrt parameters
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> tmp_se_ders_wgts;

            Eigen::MatrixXd cellhess, lrhess;
            Eigen::VectorXd ext_dUdB, ext_dUdQ;
            Eigen::MatrixXd ext_d2UdBdB;
            Eigen::VectorXd ext_d2udqdt;
            Eigen::VectorXd Qs;

            ompbuffers(int max_alpha_index_basic, int alpha_moments_count, int rb_size, int max_mu);
            ompbuffers(){};
        };
        std::vector<ompbuffers> omp_buffers_list;


        double buff_site_energy;
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buff_site_energy_ders;
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buff_site_hvp;
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buff_site_hvp_direction;
        std::vector<double> basis_vals;
        Eigen::Tensor<double, 3, Eigen::RowMajor> basis_ders;
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> coords_powers;
        std::vector<double> moment_vals;
        Eigen::Tensor<double, 3> moment_ders;
        std::vector<double> dist_powers; // rij^k
        std::vector<double> rb_vals;
        MTPadialLocalCache radial_cache;
        std::vector<double> rb_ders;
        std::vector<double> rb_derders;
        std::vector<double> val_mu;                                 // basis function value
        std::vector<double> der_mu;                                 // basis function der wrt r
        std::vector<double> derder_mu;
        Eigen::Tensor<double, 3, Eigen::RowMajor> moment_jacobian;           // dbi/dx_{ja}, usually basic only
        Eigen::Tensor<double, 3, Eigen::RowMajor> Q_jacobian;           // dQ/dx_{ja}
        std::vector<Eigen::Matrix3d> basic_moment_hessian_diag;
        std::vector<double> moment_adjoint;
        std::vector<HessianMatrix> moment_hessian, basis_derders, Q_hessian;  // dbi/dx_{ja}dx_{kb}, (i,j,a,k,b)
        std::vector<HessianMatrix> moment_hessian_sparse, basis_derders_sparse;
        HessianMatrix buff_site_hessian;
        double d2fdrab[3][3];
        double dfdra[3];
        double dmdra[3];
        double dpdra[3];
        double d2mdrab[3][3];
        double d2pdrab[3][3];

        Eigen::Tensor<double, 3, Eigen::RowMajor> moment_jacobian_coeffs;    // dbi/dradial_coeffs, basic only
        Eigen::Tensor<double, 3, Eigen::RowMajor> moment_jacobian_coeff_coord; // \partial^2 bi / (\partial x_{ja} \ partial coeffs)
        std::vector<double> site_energy_ders_wrt_moments_;          // dE/dbi, all, but only basic is used, however basic must be calculated via BP process
        std::vector<double> dloss_dmom;                             // = se_weight * buff_site_energy_ders_wrt_moments for basic inds when dE/dc requested only, and = se_weight * (d^2E/dbidx + dE/dbi) * dbi/dc when force der is also requested
        std::vector<double> dloss_dsenders;                         // = dB/dx
        Eigen::Tensor<double, 3> arraybuffer3d;             // any array buffer
        Eigen::Matrix<double, -1, 3> matrixx3dbuffer;          // any array buffer

        std::vector<double> dSEdsi; // site-energy der_wrt parameters
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> tmp_se_ders_wgts;

        // for external model only
        Eigen::VectorXd ext_dUdB, ext_dUdQ;
        Eigen::VectorXd ext_dUdt;
        Eigen::MatrixXd ext_d2UdBdt, ext_d2UdQdt;
        Eigen::VectorXd ext_d2udqdt;
        Eigen::MatrixXd ext_d2UdBdB, ext_d2UdQdQ;
        Eigen::VectorXd Qs;

        inline void rb_calcderder(const int &itype, const int &jtype, const double &r) {
            if (radial_basis_split) {
                p_radical_basis_pair[itype * species_count + jtype]->calc_derder(r, rb_vals, rb_ders, rb_derders);
            } else {
                p_radical_basis->calc_derder(r, rb_vals, rb_ders, rb_derders);
            }
        }
        // request rb_vals, rb_ders
        inline void rb_calcder(const int &itype, const int &jtype, const double &r) {
            if (radial_basis_split) {
                p_radical_basis_pair[itype * species_count + jtype]->calc_der(r, rb_vals, rb_ders);
            } else {
                p_radical_basis->calc_der(r, rb_vals, rb_ders);
            }
        }

        inline void rb_calcval(const int &itype, const int &jtype, const double &r) {
            if (radial_basis_split) {
                p_radical_basis_pair[itype * species_count + jtype]->calc_val(r, rb_vals);
            } else {
                p_radical_basis->calc_val(r, rb_vals);
            }
        }

        inline void rb_calcderder_omp(const int &itype, const int &jtype, const double &r, ompbuffers &buffer) {
            if (radial_basis_split) {
                p_radical_basis_pair[itype * species_count + jtype]->calc_derder(r, buffer.rb_vals, buffer.rb_ders, buffer.rb_derders);
            } else {
                p_radical_basis->calc_derder(r, buffer.rb_vals, buffer.rb_ders, buffer.rb_derders);
            }
        }
        // request rb_vals, rb_ders
        inline void rb_calcder_omp(const int &itype, const int &jtype, const double &r, ompbuffers &buffer) {
            if (radial_basis_split) {
                p_radical_basis_pair[itype * species_count + jtype]->calc_der(r, buffer.rb_vals, buffer.rb_ders);
            } else {
                p_radical_basis->calc_der(r, buffer.rb_vals, buffer.rb_ders);
            }
        }

        inline void rb_calcval_omp(const int &itype, const int &jtype, const double &r, ompbuffers &buffer) {
            if (radial_basis_split) {
                p_radical_basis_pair[itype * species_count + jtype]->calc_val(r, buffer.rb_vals);
            } else {
                p_radical_basis->calc_val(r, buffer.rb_vals);
            }
        }

        inline void rb_calcder_safe(const int &itype, const int &jtype, const double &r) {
            if (r < r_mins(itype, jtype) || r > cutoffs(itype, jtype)) {
                std::fill(rb_vals.begin(), rb_vals.end(), 0);
                std::fill(rb_ders.begin(), rb_ders.end(), 0);
                return;
            }
            if (radial_basis_split) {
                p_radical_basis_pair[itype * species_count + jtype]->calc_der(r, rb_vals, rb_ders);
            } else {
                p_radical_basis->calc_der(r, rb_vals, rb_ders);
            }
        }

        inline void rb_calcderder_safe(const int &itype, const int &jtype, const double &r) {
            if (r < r_mins(itype, jtype) || r > cutoffs(itype, jtype)) {
                std::fill(rb_vals.begin(), rb_vals.end(), 0);
                std::fill(rb_ders.begin(), rb_ders.end(), 0);
                std::fill(rb_derders.begin(), rb_derders.end(), 0);
                return;
            }
            if (radial_basis_split) {
                p_radical_basis_pair[itype * species_count + jtype]->calc_derder(r, rb_vals, rb_ders, rb_derders);
            } else {
                p_radical_basis->calc_derder(r, rb_vals, rb_ders, rb_derders);
            }
        }

        inline void rb_calcval_safe(const int &itype, const int &jtype, const double &r) {
            if (r < r_mins(itype, jtype) || r > cutoffs(itype, jtype)) {
                std::fill(rb_vals.begin(), rb_vals.end(), 0);
                return;
            }
            if (radial_basis_split) {
                p_radical_basis_pair[itype * species_count + jtype]->calc_val(r, rb_vals);
            } else {
                p_radical_basis->calc_val(r, rb_vals);
            }
        }


    protected:
        // During an early-exit load, radial_coeffs is not yet an alias into
        // regression_coeffs. Keep that allocation owned across derived parsing.
        std::unique_ptr<double[]> loading_radial_coeffs_;
        void clear_loaded_model() noexcept;
        virtual void clear_failed_load() noexcept { clear_loaded_model(); }
        void allocate_loaded_coefficients();
        void require_loaded_model() const {
            if (species_count <= 0 || !radial_coeffs ||
                (alpha_scalar_moments > 0 && !alpha_moment_mapping) ||
                (alpha_index_basic_count > 0 && !nu_arr_list))
                GPERROR("MTP potential has no loaded model");
        }

    public:
        PotMTP(const std::string &potfile) {
            try
            {
                Load(potfile);
            }
            catch(const GPException& e)
            {
                std::cerr << e.What() << '\n';
                // A failed load leaves the potential half-initialized (zero
                // basis counts); continuing would crash later in omp_init()
                // with out-of-range matrix access. Fail loudly instead.
                throw;
            }
        }

        PotMTP(PotMTP &opt);
        PotMTP &operator=(const PotMTP &) = delete;

        void InitSettings() {
            Connect(radial_basis_split, "radial_split");
            Connect(radial_local_cache, "radial_local_cache");
            Connect(full_symmetry_reuse, "full_symmetry_reuse");
        }

        void InitBySettings(GPSettings &settings) {
            InitSettings();
            ApplySettings(settings);
            Load(settings["potfile"]);
        }

        PotMTP(GPSettings settings) {
            InitBySettings(settings);
        }
        // use this with causion!
        PotMTP() {};

        virtual ~PotMTP();

        double weight_rad_w0 = 1e-6, weight_rad_w1 = 1e-6, weight_rad_w2 = 0;
        int quad_point = 100;
        double random_upper =  0.1;
        double random_lower = -0.1;

        void init_basic();

        // Basic exception guarantee: a failed load clears owned model data;
        // the object may be destroyed or loaded again. Borrowed coefficient
        // buffers are detached, never freed; a new load owns its coefficients.
        void Load(std::istream &is);
        void Load(const std::string &potfile);

        void Save(std::ostream &os);
        void Save(const std::string &potf);
        void SaveSpline(std::string filename, double rmin, double dr);

        void write_header(std::ostream &os, bool include_radial_split = true);
        void write_radial_info(std::ostream &os);
        void write_radial_coeff(std::ostream &os);
        void write_rules(std::ostream &os);
        void write_linear_coeff(std::ostream &os);

        void adjust_species(const std::vector<int> &sps);
        bool supports_species_append() const override {
            return typeid(*this)==typeid(PotMTP) && !coeffs_handled_external;
        }
        SpeciesAppendResult append_species(const std::vector<int>& clone_sources,
            const std::vector<std::string>& new_elements = {}) override;


        void ReorderByLevel();
        // same as reorder by level, but sort alpha_index_times in each level by moment id and group them together, like {a+b->c, a+b->d, a+b->e}
        void ReorderByLevelGroup();
        // alpha_index_times -> NN like structure
        void NNMap();

        void SaveOrdered(std::string filename);

        void random_init();
        void random_init_linear();
        void random_init_radial();
        void random_init_moments();


        void train_init();

        bool set_empty_flag() { empty = !empty; return empty; }

        bool inited() const { return !empty; };

        void Orthogonalize();

        void get_alpha_moment_mapping(std::vector<int> &ret) {
            ret.resize(alpha_scalar_moments);
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                ret[i] = alpha_moment_mapping[i];
            }
        }

        inline double get_scale() const { return p_radical_basis->scaling; };

        inline void set_scale(double scl) { p_radical_basis->scaling = scl; };

        void set_linear_coeff(double *val);
        void set_moment_coeff(double *val);

        int get_lin_coeff_size() const { return lin_coeff_size; };

        int get_coeff_size() const { return reg_coeff_size; };

        int get_NL_coeffs_count() const { return reg_coeff_size; };

        std::vector<NLParameterGroup> nl_parameter_groups() const override {
            // Keep the historical all selector; specific groups follow it so
            // group_auto scaling uses the scale of each physical block.
            return {{"all", 0, reg_coeff_size},
                    {"radial", 0, radial_coeff_size},
                    {"species", radial_coeff_size, radial_coeff_size + species_count},
                    {"moment", radial_coeff_size + species_count, reg_coeff_size}};
        }

        std::vector<ConditionalLinearBlock> conditional_linear_blocks(double) const override {
            if (typeid(*this) != typeid(PotMTP)) return {};
            ConditionalLinearBlock block; block.name = "sr";
            block.composition_count = species_count;
            for (int i = 0; i < lin_coeff_size; ++i)
                block.append(radial_coeff_size + i,
                    i >= species_count && i < int(reg_L1_vector.size()) ? reg_L1_vector[i] : 0.,
                    i >= species_count && i < int(reg_L2_vector.size()) ? reg_L2_vector[i] : 0.);
            return {std::move(block)};
        }

        void conditional_linear_set_coeffs(const std::vector<int>& indices, const Eigen::VectorXd& values) override {
            if (values.size()!=static_cast<Eigen::Index>(indices.size()) || !values.allFinite())
                GPERROR("Invalid conditional linear MTP update");
            for (int index : indices)
                if (index<radial_coeff_size || index>=reg_coeff_size)
                    GPERROR("Invalid conditional linear MTP index");
            for (int j=0;j<values.size();++j) regression_coeffs[indices[j]]=values[j];
        }

        int get_alpha_scalar_moments() const { return alpha_scalar_moments; };

        int get_species_count() const { return species_count; };

        double get_linear_coeff(const int &id) const { return linear_coeffs[id]; };

        double get_cutoff() const { return cutoff; };

        void ae_get_coeffs(alglib::real_1d_array &_x) const;

        void ae_set_coeffs(const alglib::real_1d_array &_x);

        void ae_get_coeffs(alglib::real_1d_array &_x, bool fix_radial, bool fix_linear) const;

        void ae_set_coeffs(const alglib::real_1d_array &_x, bool fix_radial, bool fix_linear);

        void get_coeffs(Eigen::VectorXd &_x) const;

        void get_coeffs(double *) const;

        void nl_get_coeffs(double *_x) const {get_coeffs(_x);};

        void model_get_coeffs(double *_x) const;

        void get_coeffs(Eigen::VectorXd &_x, bool fix_radial, bool fix_linear) const;

        void set_coeffs(const Eigen::VectorXd &_x);

        void set_coeffs(const Eigen::VectorXd &_x, bool fix_radial, bool fix_linear);

        void set_coeffs(const double *x);

        void nl_set_coeffs(const double *_x) {set_coeffs(_x);};

        void model_set_coeffs(const double *_x);

        void set_coeffs(const double *x, bool fix_radial, bool fix_linear);

        void omp_init();

        // CAUSION!!!!
        double *get_inter_pointer_coeffs() { return regression_coeffs; };
        double *get_inter_pointer_linear_coeffs() { return linear_coeffs; };
        double *get_inter_pointer_radial_coeffs() { return radial_coeffs; };

        void getEFS(Atoms &atom);

        void getEFS(Atoms &atom, AnyNeighborhoods &nbhs);

        void getEFSFluxStress(Atoms &atom);

        void getEFSFluxStress(Atoms &atom, AnyNeighborhoods &nbhs);

        void getEFSmpi(Atoms &atom, AnyNeighborhoods &nbhs);

        void getEFSmpi(Atoms &atom);

        void getEFSadd(Atoms &atom, AnyNeighborhoods &nbhs);

        void getEFSadd(Atoms &atom, AnyNeighborhoods &nbhs, double sf);

        void getEFSOMPadd(Atoms &atom, AnyNeighborhoods &nbhs);

        void getEFSOMPadd(Atoms &atom, AnyNeighborhoods &nbhs, double sf);


        void getEFSOMP(Atoms &atom);

        void getEFSOMPthread(Atoms &atom);


        void getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs);

        void getEFSOMPthread(Atoms &atom, AnyNeighborhoods &nbhs);


        void getEOMP(Atoms &atom);

        void getEOMP(Atoms &atom, AnyNeighborhoods &nbhs);

        void getE(Atoms &atom);

        void getE(Atoms &atom, AnyNeighborhoods &nbhs);

        void update_layout() {
            radial_coeffs = regression_coeffs;
            species_coeffs = regression_coeffs + radial_coeff_size;
            linear_coeffs = species_coeffs;
            moment_coeffs = linear_coeffs + species_count;
        }

        // Full space-group reuse is opt-in; the default remains translational.
        virtual void getEFSSymmOMP(Atoms &atom, double symprec = 1e-2);
        bool try_full_symmetry_efs(Atoms &atom, AnyNeighborhoods &nbhs,
                                   double symprec = 1e-5);

        void getEFSSymmOMP(Atoms &atom, std::vector<int> &mapping_to_privitive);


        bool supports_site_efs() const override { return true; }
        bool supports_site_hvp() const override { return typeid(*this) == typeid(PotMTP); }
        bool supports_site_hessian() const override { return typeid(*this) == typeid(PotMTP); }

        void apply_site_hvp(Neighborhood &nbh, const SiteCoordinateMatrix &direction,
                            double &energy, SiteCoordinateMatrix &gradient,
                            SiteCoordinateMatrix &product) override;
        void evaluate_site_hessian(Neighborhood &nbh, double &energy,
                                   SiteCoordinateMatrix &gradient, HessianMatrix &hessian) override;



        bool supports_site_parameter_gradients() const override { return true; }

        void CalcSiteEFS(Neighborhood &nbh, double &site_energy,
                                 Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &site_energy_ders) override;

        // model should report dU/dB
        // SFLAG 0: pass [species_coeffs, scalars], default
        // SFLAG 1: pass [1, scalars]
        // SFLAG 2: pass [scalars]
        // REQ_F: request force and stress
        template<class T, int SFLAG = 2, int REQ_F = 1>
        void CalcSiteEFS(Neighborhood &nbh, double &site_energy, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &site_energy_ders, T &model, int ithread = 0) {
            constexpr int soffset = [] {
                if constexpr (SFLAG == 0 || SFLAG == 1) return 1;
                else if constexpr (SFLAG == 2) return 0;
                else {
                    static_assert(SFLAG >= 0 && SFLAG <= 2, "SFLAG must be 0, 1, or 2!");
                    return -1;
                }
            }();

            // Delegate descriptor computation to shared implementation
            if constexpr (REQ_F) {
                CalcSiteDescriptors(nbh);
                site_energy_ders.resize(nbh.nbhcount_, 3);
                site_energy_ders.setZero();
            } else {
                CalcSiteBasis(nbh);
            }

            // Build basis_vals for external model
            basis_vals.resize(alpha_scalar_moments + soffset, 0);
            if constexpr (SFLAG == 0) {
                basis_vals[0] = species_coeffs[nbh.centertype];
            } else if (SFLAG == 1) {
                basis_vals[0] = 1.;
            }
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                basis_vals[i + soffset] = moment_vals[alpha_moment_mapping[i]];
            }
            Eigen::Map<Eigen::VectorXd> bvals(basis_vals.data(), basis_vals.size());
            if constexpr (REQ_F) {
                ext_dUdB.resize(basis_vals.size());
                model.propagate(bvals, site_energy, ext_dUdB, nbh.centertype, ithread);

                // Backpropagation step 1: dE/dB from model gradient
                std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
                for (int i = 0; i < alpha_scalar_moments; ++i) {
                    site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = ext_dUdB[i + soffset];
                }

                // Backpropagation step 2: reverse traversal through product rules
                backpropagate_moments();

                // Backpropagation step 3: contract with Jacobian → site_energy_ders
                for (int i = 0; i < alpha_index_basic_count; ++i)
                    for (int j = 0; j < nbh.nbhcount_; ++j)
                        for (int a = 0; a < 3; a++) {
                            site_energy_ders(j, a) += site_energy_ders_wrt_moments_[i] * moment_jacobian(i, j, a);
                        }
            } else {
                model.propagate(bvals, site_energy, nbh.centertype, ithread);
            }
        }

        // E_i = T(q_i) ; q_i = \sum_\alpha \xi_\alpha * B_\alpha
        // REQ_F: request force and stress
        template<class T, int REQ_F = 1>
        void CalcSiteEFS_n1(Neighborhood &nbh, double &site_energy, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &site_energy_ders, T &model, int ithread = 0) {
            // Delegate descriptor computation to shared implementation
            if constexpr (REQ_F) {
                CalcSiteDescriptors(nbh);
                site_energy_ders.resize(nbh.nbhcount_, 3);
                site_energy_ders.setZero();
            } else {
                CalcSiteBasis(nbh);
            }

            // Build scalar Q = species_coeff + Σ moment_coeffs[a] * B_a
            double q = species_coeffs[nbh.centertype];
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                q += moment_vals[alpha_moment_mapping[i]] * moment_coeffs[i];
            }

            if constexpr (REQ_F) {
                double ext_dudq = 0;
                model.propagate(q, site_energy, ext_dudq, nbh.centertype, ithread);

                // Backpropagation step 1: dE/dB = moment_coeffs * dE/dq
                std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
                for (int i = 0; i < alpha_scalar_moments; ++i) {
                    site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = moment_coeffs[i] * ext_dudq;
                }

                // Backpropagation step 2: reverse traversal through product rules
                backpropagate_moments();

                // Backpropagation step 3: contract with Jacobian → site_energy_ders
                for (int i = 0; i < alpha_index_basic_count; ++i)
                    for (int j = 0; j < nbh.nbhcount_; ++j)
                        for (int a = 0; a < 3; a++) {
                            site_energy_ders(j, a) += site_energy_ders_wrt_moments_[i] * moment_jacobian(i, j, a);
                        }
            } else {
                model.propagate(q, site_energy, nbh.centertype, ithread);
            }
        }

        void CalcSiteEFSOMP(Neighborhood &nbh, ompbuffers &buffer);

        template<class T, int SFLAG = 2, int REQ_F = 1>
        void CalcSiteEFSOMP(Neighborhood &nbh, ompbuffers &buffer, T &model, int ithread = 0) {
            constexpr int soffset = [] {
                if constexpr (SFLAG == 0 || SFLAG == 1) return 1;
                else if constexpr (SFLAG == 2) return 0;
                else {
                    static_assert(SFLAG >= 0 && SFLAG <= 2, "SFLAG must be 0, 1, or 2!");
                    return -1;
                }
            }();

            // Delegate descriptor computation to shared OMP implementation
            if constexpr (REQ_F) {
                CalcSiteDescriptorsOMP(nbh, buffer);
                buffer.buff_site_energy_ders.resize(nbh.nbhcount_, 3);
                buffer.buff_site_energy_ders.setZero();
            } else {
                CalcSiteBasisOMP(nbh, buffer);
            }

            // Build basis_vals for external model
            buffer.basis_vals.resize(alpha_scalar_moments + soffset, 0);
            if constexpr (SFLAG == 0) {
                buffer.basis_vals[0] = species_coeffs[nbh.centertype];
            } else if (SFLAG == 1) {
                buffer.basis_vals[0] = 1.;
            }
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                buffer.basis_vals[i + soffset] = buffer.moment_vals[alpha_moment_mapping[i]];
            }
            Eigen::Map<Eigen::VectorXd> bvals(buffer.basis_vals.data(), buffer.basis_vals.size());
            if constexpr (REQ_F) {
                buffer.ext_dUdB.resize(buffer.basis_vals.size());
                model.propagate(bvals, buffer.buff_site_energy, buffer.ext_dUdB, nbh.centertype, ithread);

                // Backpropagation step 1: dE/dB from model gradient
                std::fill(buffer.site_energy_ders_wrt_moments_.begin(), buffer.site_energy_ders_wrt_moments_.end(), 0);
                for (int i = 0; i < alpha_scalar_moments; ++i) {
                    buffer.site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = buffer.ext_dUdB[i + soffset];
                }

                // Backpropagation step 2: reverse traversal through product rules
                for (int i = alpha_index_times_count - 1; i >= 0; i--) {
                    double val0 = buffer.moment_vals[alpha_index_times(i, 0)];
                    double val1 = buffer.moment_vals[alpha_index_times(i, 1)];
                    int val2 = alpha_index_times(i, 2);

                    buffer.site_energy_ders_wrt_moments_[alpha_index_times(i, 1)] +=
                            buffer.site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val0;
                    buffer.site_energy_ders_wrt_moments_[alpha_index_times(i, 0)] +=
                            buffer.site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val1;
                }

                // Backpropagation step 3: contract with Jacobian → site_energy_ders
                for (int i = 0; i < alpha_index_basic_count; ++i)
                    for (int j = 0; j < nbh.nbhcount_; ++j)
                        for (int a = 0; a < 3; a++) {
                            buffer.buff_site_energy_ders(j, a) += buffer.site_energy_ders_wrt_moments_[i] * buffer.moment_jacobian(i, j, a);
                        }
            } else {
                model.propagate(bvals, buffer.buff_site_energy, nbh.centertype, ithread);
            }
        }


        void CalcSiteEOMP(Neighborhood &nbh, ompbuffers &buffer);

        /// OMP-thread-local lightweight descriptor + basic Jacobian.
        /// \see CalcSiteDescriptors
        void CalcSiteDescriptorsOMP(Neighborhood &nbh, ompbuffers &buffer);

        /// OMP-thread-local energy-only descriptor.
        /// \see CalcSiteBasis
        void CalcSiteBasisOMP(Neighborhood &nbh, ompbuffers &buffer);

        void CalcSiteE(Neighborhood &nbh, double &site_energy) override;


        void CalcSiteHess(Neighborhood &nbh);

        // ── Unified Hessian computation (serial / OMP) ────────────────────
        template<bool IsOMP, class T, int SFLAG = 2>
        void CalcSiteHessImpl(Neighborhood &nbh, ompbuffers &buf,
                              double &site_energy,
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &site_energy_ders,
                              HessianMatrix& site_hessian, T &model, int ithread = 0,
                              const Eigen::Matrix<double, -1, 3, Eigen::RowMajor>
                                  *hvp_direction = nullptr,
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor>
                                  *hvp_product = nullptr) {
            constexpr int soffset = [] {
                if constexpr (SFLAG == 0 || SFLAG == 1) return 1;
                else if constexpr (SFLAG == 2) return 0;
                else {
                    static_assert(SFLAG >= 0 && SFLAG <= 2, "SFLAG must be 0, 1, or 2!");
                    return -1;
                }
            }();

            // Ternary references → compile-time resolution, zero overhead
            auto& mv  = IsOMP ? buf.moment_vals               : moment_vals;
            auto& rv  = IsOMP ? buf.rb_vals                   : rb_vals;
            auto& rd  = IsOMP ? buf.rb_ders                   : rb_ders;
            auto& rdd = IsOMP ? buf.rb_derders                : rb_derders;
            auto& vm  = IsOMP ? buf.val_mu                    : val_mu;
            auto& dm  = IsOMP ? buf.der_mu                    : der_mu;
            auto& ddm = IsOMP ? buf.derder_mu                 : derder_mu;
            auto& mj  = IsOMP ? buf.moment_jacobian           : moment_jacobian;
            auto& bhd = IsOMP ? buf.basic_moment_hessian_diag : basic_moment_hessian_diag;
            auto& madj = IsOMP ? buf.moment_adjoint            : moment_adjoint;
            auto& cp  = IsOMP ? buf.coords_powers             : coords_powers;
            auto& dp  = IsOMP ? buf.dist_powers               : dist_powers;
            auto& bv  = IsOMP ? buf.basis_vals                : basis_vals;
            // Raw pointers to auxiliary arrays (array decay in ternary → pointer)
            double *df  = IsOMP ? buf.dfdra       : dfdra;
            double *dpd = IsOMP ? buf.dpdra       : dpdra;
            double *dmd = IsOMP ? buf.dmdra       : dmdra;
            double *d2f = IsOMP ? &buf.d2fdrab[0][0] : &d2fdrab[0][0];
            double *d2p = IsOMP ? &buf.d2pdrab[0][0] : &d2pdrab[0][0];
            double *d2m = IsOMP ? &buf.d2mdrab[0][0] : &d2mdrab[0][0];
            // Access: df[a], d2f[a*3+b], etc.
            auto& edudb  = IsOMP ? buf.ext_dUdB    : ext_dUdB;
            auto& ed2udb = IsOMP ? buf.ext_d2UdBdB : ext_d2UdBdB;

            site_energy_ders.resize(nbh.nbhcount_, 3);
            site_energy_ders.setZero();
            const bool compute_hvp = hvp_direction != nullptr;
            if (compute_hvp) {
                if (hvp_product == nullptr ||
                    hvp_direction->rows() != nbh.nbhcount_)
                    GPERROR("invalid site Hessian-vector product buffers");
                hvp_product->resize(nbh.nbhcount_, 3);
                hvp_product->setZero();
            } else {
                site_hessian.resize(nbh.nbhcount_);
                site_hessian.setzero();
            }
            bv.resize(alpha_scalar_moments + 1);
            mj.resize(alpha_moments_count, nbh.nbhcount_, 3);
            mj.setZero();
            bhd.assign(static_cast<size_t>(alpha_index_basic_count) *
                           nbh.nbhcount_,
                       Eigen::Matrix3d::Zero());
            madj.assign(alpha_moments_count, 0.0);

            std::fill(IsOMP ? buf.site_energy_ders_wrt_moments_.begin() : site_energy_ders_wrt_moments_.begin(),
                      IsOMP ? buf.site_energy_ders_wrt_moments_.end()   : site_energy_ders_wrt_moments_.end(), 0);
            std::fill(mv.begin(), mv.end(), 0);
            std::fill(rv.begin(), rv.end(), 0);
            std::fill(rd.begin(), rd.end(), 0);
            std::fill(rdd.begin(), rdd.end(), 0);
            std::fill(vm.begin(), vm.end(), 0);
            std::fill(dm.begin(), dm.end(), 0);
            const int& itype = nbh.centertype;
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                const double &rij = nbh.distances_[jj];
                const int& jtype = nbh.chemmap_[jj];
                if (rij > cutoffs(itype, jtype)) continue;
                const double rijinv = 1.0 / rij;
                if constexpr (IsOMP) {
                    rb_calcderder_omp(itype, jtype, rij, buf);
                } else {
                    rb_calcderder(itype, jtype, rij);
                }

                dp[0] = 1;
                for (int k = 1; k < max_alpha_index_basic + 4; ++k) {
                    dp[k] = dp[k - 1] * rijinv;
                }
                for (int k = 1; k < max_alpha_index_basic; ++k) {
                    for (int a = 0; a < 3; a++) {
                        cp(k + 1, a) = cp(k, a) * nbh.pos_[jj][a];
                    }
                }

                for (int mu = 0; mu < max_mu; ++mu) {
                    vm[mu] = 0;
                    dm[mu] = 0;
                    ddm[mu] = 0;
                    int idxoffset =
                            (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
                    for (int xi = 0; xi < rb_size; xi++) {
                        vm[mu]  += radial_coeffs[idxoffset + xi] * rv[xi];
                        dm[mu]  += radial_coeffs[idxoffset + xi] * rd[xi];
                        ddm[mu] += radial_coeffs[idxoffset + xi] * rdd[xi];
                    }
                }

                for (int i = 0; i < alpha_index_basic_count; ++i) {
                    int mu = alpha_index_basic(i, 0);
                    double val = vm[mu];
                    double der = dm[mu];
                    double derder = ddm[mu];
                    const auto &r = nbh.pos_[jj];
                    int k = nu_arr_list[i];
                    double &powk = dp[k];
                    double pow0 = cp(alpha_index_basic(i, 1) + 1, 0);
                    double pow1 = cp(alpha_index_basic(i, 2) + 1, 1);
                    double pow2 = cp(alpha_index_basic(i, 3) + 1, 2);
                    double mult0 = pow0 * pow1 * pow2;
                    double multa1[3] = {cp(alpha_index_basic(i, 1), 0) * pow1 * pow2,
                                        pow0 * cp(alpha_index_basic(i, 2), 1) * pow2,
                                        pow0 * pow1 * cp(alpha_index_basic(i, 3), 2)};
                    double multa2[3][3];

                    if (alpha_index_basic(i, 1) < 2) multa2[0][0] = 0;
                    else multa2[0][0] = cp(alpha_index_basic(i, 1) - 1, 0) * pow1 * pow2;
                    if (alpha_index_basic(i, 2) < 2) multa2[1][1] = 0;
                    else multa2[1][1] = pow0 * cp(alpha_index_basic(i, 2) - 1, 1) * pow2;
                    if (alpha_index_basic(i, 3) < 2) multa2[2][2] = 0;
                    else multa2[2][2] = pow0 * pow1 * cp(alpha_index_basic(i, 3) - 1, 2);
                    if (alpha_index_basic(i, 1) < 1 || alpha_index_basic(i, 2) < 1) multa2[0][1] = 0;
                    else multa2[0][1] = cp(alpha_index_basic(i, 1), 0) * cp(alpha_index_basic(i, 2), 1) * pow2;
                    if (alpha_index_basic(i, 1) < 1 || alpha_index_basic(i, 3) < 1) multa2[0][2] = 0;
                    else multa2[0][2] = cp(alpha_index_basic(i, 1), 0) * pow1 * cp(alpha_index_basic(i, 3), 2);
                    if (alpha_index_basic(i, 2) < 1 || alpha_index_basic(i, 3) < 1) multa2[1][2] = 0;
                    else multa2[1][2] = pow0 * cp(alpha_index_basic(i, 2), 1) * cp(alpha_index_basic(i, 3), 2);
                    multa2[1][0] = multa2[0][1];
                    multa2[2][0] = multa2[0][2];
                    multa2[2][1] = multa2[1][2];

                    double dptmp1 = -k * dp[k+2];
                    double dptmp2 = k * (k+2) * dp[k+4];
                    double dertmp = der * dp[1];
                    double der2tmp = (derder - dp[1] * der) * dp[2];
                    double ftmpr = der * dp[1];
                    for (int a = 0; a < 3; ++a) {
                        df[a] = dertmp * r[a];
                        dpd[a] = dptmp1 * r[a];
                        dmd[a] = alpha_index_basic(i, a + 1) * multa1[a];
                        d2f[a*3+a] = r[a] * r[a] * der2tmp + ftmpr;
                        d2m[a*3+a] = (alpha_index_basic(i, a + 1) - 1) * alpha_index_basic(i, a + 1) * multa2[a][a];
                        d2p[a*3+a] = dptmp2 * r[a] * r[a] + dptmp1;
                        for (int b = a + 1; b < 3; ++b) {
                            d2f[a*3+b] = r[a] * r[b] * der2tmp;
                            d2f[b*3+a] = d2f[a*3+b];
                            d2p[a*3+b] = dptmp2 * r[a] * r[b];
                            d2p[b*3+a] = d2p[a*3+b];
                            d2m[a*3+b] = alpha_index_basic(i, b + 1) * alpha_index_basic(i, a + 1) * multa2[a][b];
                            d2m[b*3+a] = d2m[a*3+b];
                        }
                    }
                    mv[i] += val * mult0 * powk;
                    for (int a = 0; a < 3; ++a) {
                        mj(i, jj, a) += df[a] * mult0 * powk + val * dmd[a] * powk + val * mult0 * dpd[a];
                        for (int b = a; b < 3; ++b) {
                            bhd[static_cast<size_t>(i) * nbh.nbhcount_ + jj](a, b) +=
                                d2f[a*3+b] * mult0 * powk +
                                val * d2m[a*3+b] * powk +
                                val * mult0 * d2p[a*3+b] +
                                df[a] * (dmd[b] * powk + mult0 * dpd[b]) +
                                df[b] * (dmd[a] * powk + mult0 * dpd[a]) +
                                val * (dmd[a] * dpd[b] + dmd[b] * dpd[a]);
                        }
                    }
                }
            }
            // Fill the lower triangle of the compact basic-moment blocks.
            for (int i = 0; i < alpha_index_basic_count; ++i) {
                for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < a; ++b) {
                            auto &block = bhd[static_cast<size_t>(i) *
                                              nbh.nbhcount_ + jj];
                            block(a, b) = block(b, a);
                        }
                    }
                }
            }

            // Forward product rules: moment_vals + moment_jacobian
            for (int i = 0; i < alpha_index_times_count; ++i) {
                const int &bi = alpha_index_times(i, 0);
                const int &bj = alpha_index_times(i, 1);
                const int &bk = alpha_index_times(i, 3);
                double vbi = mv[bi];
                double vbj = mv[bj];
                double times = alpha_index_times(i, 2);
                mv[bk] += times * vbi * vbj;
                for (int j = 0; j < nbh.nbhcount_; ++j) {
                    for (int a = 0; a < 3; ++a) {
                        mj(bk, j, a) +=
                            times * (mj(bi, j, a) * vbj + vbi * mj(bj, j, a));
                    }
                }
            }
            // Call external model
            edudb.resize(bv.size());
            if constexpr (SFLAG == 0) {
                bv[0] = species_coeffs[nbh.centertype];
            } else if (SFLAG == 1) {
                bv[0] = 1.;
            }
            for (int i = 1; i <= alpha_scalar_moments; ++i) {
                bv[i] = mv[alpha_moment_mapping[i - soffset]];
            }
            model.propagate(bv, site_energy, edudb, ed2udb, nbh.centertype, ithread);
            // Force contribution
            for (int a = 0; a < 3; ++a)
                for (int j = 0; j < nbh.nbhcount_; ++j)
                    for (int i = 0; i < alpha_scalar_moments; ++i)
                        site_energy_ders(j, a) += edudb(i + soffset) * mj(alpha_moment_mapping[i], j, a);
            // Hessian first term, sum_i dU/dB_i * d2B_i/dxdy.  Reverse the
            // product DAG and emit each node's local curvature directly into
            // the one site Hessian, instead of retaining one dense neighbour
            // Hessian for every moment.
            for (int i = 0; i < alpha_scalar_moments; ++i)
                madj[alpha_moment_mapping[i]] += edudb(i + soffset);
            std::vector<double> moment_directional;
            if (compute_hvp) {
                moment_directional.assign(alpha_moments_count, 0.0);
                for (int i = 0; i < alpha_moments_count; ++i)
                    for (int j = 0; j < nbh.nbhcount_; ++j)
                        for (int a = 0; a < 3; ++a)
                            moment_directional[i] +=
                                mj(i, j, a) * (*hvp_direction)(j, a);
            }
            for (int irule = alpha_index_times_count - 1; irule >= 0; --irule) {
                const int bi = alpha_index_times(irule, 0);
                const int bj = alpha_index_times(irule, 1);
                const int bk = alpha_index_times(irule, 3);
                const double times = alpha_index_times(irule, 2);
                const double weight = madj[bk] * times;
                if (weight != 0.0) {
                    if (compute_hvp) {
                        for (int ai = 0; ai < nbh.nbhcount_; ++ai)
                            for (int a = 0; a < 3; ++a)
                                (*hvp_product)(ai, a) += weight * (
                                    mj(bi, ai, a) * moment_directional[bj] +
                                    mj(bj, ai, a) * moment_directional[bi]);
                    } else {
                        for (int ai = 0; ai < nbh.nbhcount_; ++ai)
                            for (int aj = 0; aj < nbh.nbhcount_; ++aj)
                                for (int a = 0; a < 3; ++a)
                                    for (int b = 0; b < 3; ++b)
                                        site_hessian(ai, a, aj, b) += weight * (
                                            mj(bi, ai, a) * mj(bj, aj, b) +
                                            mj(bj, ai, a) * mj(bi, aj, b));
                    }
                }
                madj[bi] += madj[bk] * times * mv[bj];
                madj[bj] += madj[bk] * times * mv[bi];
            }
            for (int i = 0; i < alpha_index_basic_count; ++i) {
                const double weight = madj[i];
                if (weight == 0.0) continue;
                for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                    const auto &block = bhd[static_cast<size_t>(i) *
                                            nbh.nbhcount_ + jj];
                    if (compute_hvp)
                        hvp_product->row(jj) += weight *
                            (block * hvp_direction->row(jj).transpose()).transpose();
                    else
                        site_hessian(jj, jj) += weight * block;
                }
            }
            // Hessian: second term  Σ d²U/dB_i dB_j · dB_i/dx · dB_j/dy.
            // Linear models signal an identically-zero output Hessian with a
            // zero-sized matrix, avoiding an unnecessary O(B^2) allocation.
            if (compute_hvp) {
                if constexpr (requires {
                    model.add_coordinate_hessian_vector(
                        mj, alpha_moment_mapping, alpha_scalar_moments,
                        nbh.nbhcount_, *hvp_direction, *hvp_product);
                }) {
                    model.add_coordinate_hessian_vector(
                        mj, alpha_moment_mapping, alpha_scalar_moments,
                        nbh.nbhcount_, *hvp_direction, *hvp_product);
                } else if (ed2udb.size() != 0) {
                    Eigen::VectorXd basis_directional =
                        Eigen::VectorXd::Zero(alpha_scalar_moments);
                    for (int i = 0; i < alpha_scalar_moments; ++i)
                        basis_directional(i) =
                            moment_directional[alpha_moment_mapping[i]];
                    const Eigen::VectorXd weighted =
                        ed2udb.block(soffset, soffset, alpha_scalar_moments,
                                    alpha_scalar_moments) * basis_directional;
                    for (int i = 0; i < alpha_scalar_moments; ++i)
                        for (int ai = 0; ai < nbh.nbhcount_; ++ai)
                            for (int a = 0; a < 3; ++a)
                                (*hvp_product)(ai, a) += weighted(i) *
                                    mj(alpha_moment_mapping[i], ai, a);
                }
            } else if constexpr (requires {
                model.add_coordinate_hessian(
                    mj, alpha_moment_mapping, alpha_scalar_moments,
                    nbh.nbhcount_, site_hessian);
            }) {
                model.add_coordinate_hessian(
                    mj, alpha_moment_mapping, alpha_scalar_moments,
                    nbh.nbhcount_, site_hessian);
            } else if (ed2udb.size() != 0)
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        for (int i = 0; i < alpha_scalar_moments; ++i)
                            for (int j = 0; j < alpha_scalar_moments; ++j)
                                for (int ai = 0; ai < nbh.nbhcount_; ++ai)
                                    for (int aj = 0; aj < nbh.nbhcount_; ++aj)
                                        site_hessian(ai, a, aj, b) +=
                                            ed2udb(i + soffset, j + soffset) * mj(alpha_moment_mapping[i], ai, a) * mj(alpha_moment_mapping[j], aj, b);
        }

        struct LinearFC2Model {
            const double *coeffs = nullptr;
            int count = 0;

            void propagate(const std::vector<double> &basis, double &energy,
                           Eigen::VectorXd &gradient, Eigen::MatrixXd &hessian,
                           int &, int &) const {
                energy = basis[0];
                gradient.setZero(basis.size());
                gradient[0] = 1.0;
                for (int i = 0; i < count; ++i) {
                    energy += coeffs[i] * basis[i + 1];
                    gradient[i + 1] = coeffs[i];
                }
                hessian.resize(0, 0);
            }
        };

        void CalcSiteHessLowMemory(Neighborhood &nbh) {
            ompbuffers unused;
            LinearFC2Model model{moment_coeffs, alpha_scalar_moments};
            CalcSiteHessImpl<false, LinearFC2Model, 0>(
                nbh, unused, buff_site_energy, buff_site_energy_ders,
                buff_site_hessian, model, 0);
        }

        void CalcSiteHessLowMemoryOMP(Neighborhood &nbh, ompbuffers &buffer,
                                      int ithread) {
            LinearFC2Model model{moment_coeffs, alpha_scalar_moments};
            CalcSiteHessImpl<true, LinearFC2Model, 0>(
                nbh, buffer, buffer.buff_site_energy,
                buffer.buff_site_energy_ders, buffer.buff_site_hessian,
                model, ithread);
        }

        void CalcSiteHessianVectorProduct(
            Neighborhood &nbh,
            const Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &direction,
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &product) {
            ompbuffers unused;
            LinearFC2Model model{moment_coeffs, alpha_scalar_moments};
            CalcSiteHessImpl<false, LinearFC2Model, 0>(
                nbh, unused, buff_site_energy, buff_site_energy_ders,
                buff_site_hessian, model, 0, &direction, &product);
        }

        void CalcSiteHessianVectorProductOMP(
            Neighborhood &nbh, ompbuffers &buffer, int ithread,
            const Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &direction,
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &product) {
            LinearFC2Model model{moment_coeffs, alpha_scalar_moments};
            CalcSiteHessImpl<true, LinearFC2Model, 0>(
                nbh, buffer, buffer.buff_site_energy,
                buffer.buff_site_energy_ders, buffer.buff_site_hessian,
                model, ithread, &direction, &product);
        }

        template<class T, int SFLAG = 2>
        void CalcSiteHess(Neighborhood &nbh, double &site_energy, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &site_energy_ders, HessianMatrix& site_hessian, T &model, int ithread = 0) {
            ompbuffers unused;
            CalcSiteHessImpl<false, T, SFLAG>(nbh, unused, site_energy, site_energy_ders, site_hessian, model, ithread);
        }

        void CalcSiteHessOMP(Neighborhood &nbh, ompbuffers &buffer);

        template<class T, int SFLAG = 2>
        void CalcSiteHessOMP(Neighborhood &nbh, ompbuffers &buffer, T &model, int ithread = 0) {
            CalcSiteHessImpl<true, T, SFLAG>(nbh, buffer, buffer.buff_site_energy, buffer.buff_site_energy_ders, buffer.buff_site_hessian, model, ithread);
        }


        void getEFSHessian(Atoms &atom, HessianMatrix &hessian);

        void getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian);

        void getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess);

        void getEFSHessianCell(Atoms &atom, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess);

        void getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double symprec = 1e-4);

        void getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4);

        void getEFSHessianSymmOMP(Atoms &atom, HessianMatrix &hessian, double symprec = 1e-4);

        bool supports_analytical_hessian() const override { return true; }

        bool supports_analytical_hessian_vector_product() const override {
            return true;
        }

        using AnyPotential::applyHessianVectorProduct;
        /// Columns are the Cartesian H_j * direction for each linear basis j.
        /// Does not construct any dense coordinate Hessian.
        void CalcHessianVectorCmpt(Atoms &atom, AnyNeighborhoods &nbhs,
            const Eigen::VectorXd &direction, Eigen::MatrixXd &components);
        void applyHessianVectorProduct(
            Atoms &atom, AnyNeighborhoods &nbhs,
            const Eigen::VectorXd &direction,
            Eigen::VectorXd &product) override;

        void getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4);

        bool supports_analytical_cell_hessian() const override { return true; }


        void getHessian(Atoms &atom, AnyNeighborhoods &nbhs, std::vector<std::pair<int, int>> inds, HessianSparse &hessian);

        void getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian);

        void getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess);

        void CalcSiteBasisDers(Neighborhood &nbh);

        void CalcSiteBasis(Neighborhood &nbh);

        /// Unified template: IsOMP=false → member buffers, IsOMP=true → ompbuffers.
        /// Ternary references are resolved at compile time → zero runtime overhead.
        template<bool IsOMP>
        void CalcSiteDescriptorsImpl(Neighborhood &nbh, ompbuffers &buf);

        template<bool IsOMP>
        void CalcSiteBasisImpl(Neighborhood &nbh, ompbuffers &buf);

        /// Lightweight descriptor + basic Jacobian for CalcSiteEFS backprop.
        /// Fills moment_vals (all moments, forward only) and moment_jacobian
        /// (basic-moment rows only, no forward propagation through product rules).
        /// Does NOT touch site_energy_ders_wrt_moments_, basis_vals, or basis_ders.
        void CalcSiteDescriptors(Neighborhood &nbh);

        /// Backpropagate site_energy_ders_wrt_moments_ through product rules
        /// (alpha_index_times).  Caller must fill the scalar-moment entries
        /// (step 1) before calling; this performs step 2 (reverse traversal).
        void backpropagate_moments();

        void CalcEFSCmpt(Atoms &atom, AnyNeighborhoods &nbhs,
                         std::vector<double> &energy_cmpts,
                         Eigen::Tensor<double, 3> &force_cmpts,
                         Eigen::Tensor<double, 3> &stress_cmpts);
        // fcmpt: (iat, dir, bid)
        void CalcEFSCmpt_R(Atoms &atom, AnyNeighborhoods &nbhs,
                         std::vector<double> &energy_cmpts,
                         Eigen::Tensor<double, 3, Eigen::RowMajor> &force_cmpts,
                         Eigen::Tensor<double, 3, Eigen::RowMajor> &stress_cmpts);

        // for train_mtp fast linear version
        void CalcEFSCmpt(Atoms &atom, AnyNeighborhoods &nbhs,
                            Eigen::VectorXd &weights,
                            Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &lin_matrix,
                            Eigen::VectorXd &lin_vector,
                            double weng, double wfce, double wstr,
                            int poffset, bool fix_species_coeffs);

        void CalcEFSCmpt(Atoms &atom, AnyNeighborhoods &nbhs,
                         std::vector<double> &energy_cmpts,
                         Eigen::Tensor<double, 3> &force_cmpts,
                         Eigen::Tensor<double, 3> &stress_cmpts,
                         int offset);

        void CalcEFSCmptNSp(Atoms &atom, AnyNeighborhoods &nbhs,
                         std::vector<double> &energy_cmpts,
                         Eigen::Tensor<double, 3> &force_cmpts,
                         Eigen::Tensor<double, 3> &stress_cmpts,
                         int offset);

        void CalcECmpt(Atoms &atom, AnyNeighborhoods &nbhs,
                         std::vector<double> &energy_cmpts);

        void CalcECmpt(Atoms &atom, AnyNeighborhoods &nbhs,
                         std::vector<double> &energy_cmpts, int offset);

        void CalcECmptNSp(Atoms &atom, AnyNeighborhoods &nbhs,
                         std::vector<double> &energy_cmpts, int offset);

        void AddPenaltyGard(double weight, double &loss, double *loss_gard) override {
            AddPenaltyGard(weight, loss, loss_gard, false, false, false);
        }
        void AddPenaltyGard(double weight, double &loss, double *loss_gard, bool fix_radial, bool fix_linear = false, bool fix_species = false);

        // only for external model, assume loss_gard is correctly offseted
        void AddPenaltyGardModel(double weight, double &loss, double *loss_gard);

        // Shared descriptor response for MTP+LES and MTP-M+LES.
        void add_latent_charge_forces(int center, Neighborhood &, double dEdq,
            const double *charge_moments, Atoms &);
        void latent_charge_force_vjp(Atoms &, AnyNeighborhoods &, LES &,
            const Eigen::VectorXd &charges, const double *charge_moments,
            const Eigen::Matrix<double, -1, 3> &dLdF, const Eigen::Matrix3d &dLdS,
            Eigen::VectorXd &dEdq, Eigen::VectorXd &charge_vjp,
            Eigen::VectorXd &moment_vjp, Eigen::VectorXd *response = nullptr);
        void accumulate_latent_charge_radial_gradient(Atoms &, AnyNeighborhoods &,
            const double *charge_moments, const Eigen::VectorXd &dLdq,
            const Eigen::VectorXd &dEdq,
            const Eigen::Matrix<double, -1, 3> &dLdF, const Eigen::Matrix3d &dLdS,
            double *radial_gradient);

        void AccumulateEFSCombinationGard(Atoms &atom,
                                          AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          double *out_gard_accumulator);
        // will skip species coeffs
        void AccumulateEFSCombinationGard(Atoms &atom,
                                          AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          double *out_gard_accumulator,
                                          int offset_r,
                                          int offset_l);

        // E-only overload — skips force/stress weight computation entirely
        void AccumulateEFSCombinationGardLonly(Atoms &atom,
                                          AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          double *out_gard_accumulator);

        void AccumulateEFSCombinationGardLonly(Atoms &atom,
                                          AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          double *out_gard_accumulator);

        void AccumulateEFSCombinationGardLonly(Atoms &atom,
                                          AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          double *out_gard_accumulator, int coffset);

        void AccumulateEFSCombinationGardRonly(Atoms &atom,
                                          AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          double *out_gard_accumulator);

        void AccumulateEFSCombinationGardRonly(Atoms &atom,
                                          AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          double *out_gard_accumulator, int coffset);

        void AccumulateFPGard(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,                              // usually 2*wgt_eng*(E_ref - E_efs)
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight); // dE/dpara dF/dpara dS/dpara
        // SFLAG: as above
        // REQ_RADICAL: request calculate MTP radial coeffs
        // REQ_MODEL: request calculate external model coeffs
        // REQ_F: request force related terms
        // if SFLAG != 0, REQ_SPECIES should always be 0. SFLAG should be 2 in most cases
        template<class T, int SFLAG = 2, int REQ_RADICAL = 1, int REQ_SPECIES = 1, int REQ_MODEL = 1, int REQ_F = 1>
        void AccumulateFPGard(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight,
                              T &model,
                              int ithread = 0) {
            constexpr int soffset = [] {
                if constexpr (SFLAG == 0 || SFLAG == 1) return 1;
                else if constexpr (SFLAG == 2) return 0;
                else {
                    static_assert(SFLAG >= 0 && SFLAG <= 2, "SFLAG must be 0, 1, or 2!");
                    return -1;
                }
            }();
            static_assert(!(SFLAG > 0 && REQ_SPECIES > 0), "When SFLAG > 0, REQ_SPECIES should always be zero!");
            if constexpr (REQ_F) {
                moment_jacobian.resize(alpha_index_basic_count, nbh.nbhcount_, 3);
                moment_jacobian.setZero();
                moment_jacobian_coeff_coord.setZero();
                std::fill(rb_ders.begin(), rb_ders.end(), 0);
                std::fill(der_mu.begin(), der_mu.end(), 0);
            }
            moment_jacobian_coeffs.setZero();
            basis_vals.resize(alpha_scalar_moments + soffset, 0);

            std::fill(moment_vals.begin(), moment_vals.end(), 0);
            std::fill(rb_vals.begin(), rb_vals.end(), 0);
            std::fill(val_mu.begin(), val_mu.end(), 0);
            std::fill(dloss_dmom.begin(), dloss_dmom.end(), 0);
            std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0);
            std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
            const int& itype = nbh.centertype;
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                const double &rij = nbh.distances_[jj];
                const int &jtype = nbh.chemmap_[jj];
                if (rij > cutoffs(itype, jtype)) continue;
                const double rijinv = 1.0 / rij;
                if constexpr (REQ_F)
                    rb_calcder(itype, jtype, rij);
                else
                    rb_calcval(itype, jtype, rij);

                dist_powers[0] = 1;
                for (int k = 1; k < max_alpha_index_basic; ++k) {
                    dist_powers[k] = dist_powers[k - 1] * rijinv;
                    for (int a = 0; a < 3; a++) {
                        coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a]; // off set by 1 to aviod if later
                    }
                }

                for (int mu = 0; mu < max_mu; ++mu) {
                    val_mu[mu] = 0;
                    if constexpr (REQ_F) der_mu[mu] = 0;
                    int idxoffset =
                            (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
                    for (int xi = 0; xi < rb_size; ++xi) {
                        val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                        if constexpr (REQ_F) der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
                    }
                }

                for (int i = 0; i < alpha_index_basic_count; ++i) {
                    // @OPTIMIZE: init mu-array while loading potentials, do not need offset search
                    int mu = alpha_index_basic(i, 0);
                    double val = val_mu[mu];
                    double der;
                    if constexpr (REQ_F) der = der_mu[mu];
                    int k = nu_arr_list[i];
                    double &powk = dist_powers[k];
                    // TODO: test weather move this out can accelearte computation by defining val_mu_nv[:] and der_mu_nv[:]
                    // since some (mu, nv) pair may not exist.
                    val *= powk; // normalize, all tenser element is devided by rij^\nv to avoid extremely large values
                    if constexpr (REQ_F) der = der * powk - k * val / rij;
                    // offset by 1
                    double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0); // x
                    double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1); // y
                    double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2); // z

                    double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)

                    moment_vals[i] += val * mult0;
                    for (int xi = 0; xi < rb_size; xi++) {
                        // dbm / dc
                        moment_jacobian_coeffs(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                                rb_vals[xi] * powk * mult0;
                        // force related
                        if constexpr (REQ_F)
                        if (f_weight != nullptr) {
                            // here, derx = d2bm / dxdc
                            double tail = (rb_ders[xi] * powk * mult0 - rb_vals[xi] * k * powk * mult0 / nbh.distances_[jj]);
                            double derx = (nbh.pos_[jj][0] / nbh.distances_[jj]) * tail;
                            double dery = (nbh.pos_[jj][1] / nbh.distances_[jj]) * tail;
                            double derz = (nbh.pos_[jj][2] / nbh.distances_[jj]) * tail;
                            derx += rb_vals[xi] * powk *
                                    alpha_index_basic(i, 1) *
                                    coords_powers(alpha_index_basic(i, 1), 0) *
                                    pow1 * pow2;
                            dery += rb_vals[xi] * powk *
                                    alpha_index_basic(i, 2) *
                                    coords_powers(alpha_index_basic(i, 2), 1) *
                                    pow0 * pow2;
                            derz += rb_vals[xi] * powk *
                                    alpha_index_basic(i, 3) *
                                    coords_powers(alpha_index_basic(i, 3), 2) *
                                    pow0 * pow1;
                            // this is w_i \cdot d2bm/drdc
                            moment_jacobian_coeff_coord(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                                    (*f_weight)(jj, 0) * derx + (*f_weight)(jj, 1) * dery + (*f_weight)(jj, 2) * derz;
                        }
                    }
                    if constexpr (REQ_F) {
                        mult0 *= der / rij;
                        moment_jacobian(i, jj, 0) += mult0 * nbh.pos_[jj][0];
                        moment_jacobian(i, jj, 1) += mult0 * nbh.pos_[jj][1];
                        moment_jacobian(i, jj, 2) += mult0 * nbh.pos_[jj][2];
                        // coords_powers is shifted by 3 to avoid if statement
                        // note that 0 * Inf = NaN
                        moment_jacobian(i, jj, 0) +=
                                alpha_index_basic(i, 1) * val *
                                coords_powers(alpha_index_basic(i, 1), 0) *
                                pow1 * pow2;
                        moment_jacobian(i, jj, 1) +=
                                alpha_index_basic(i, 2) * val *
                                coords_powers(alpha_index_basic(i, 2), 1) *
                                pow0 * pow2;
                        moment_jacobian(i, jj, 2) +=
                                alpha_index_basic(i, 3) * val *
                                coords_powers(alpha_index_basic(i, 3), 2) *
                                pow0 * pow1;
                    }
                }
            }

            // non-elementary b_i
            for (int i = 0; i < alpha_index_times_count; ++i) {
                double val0 = moment_vals[alpha_index_times(i, 0)];
                double val1 = moment_vals[alpha_index_times(i, 1)];
                double val2 = alpha_index_times(i, 2);
                moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
            }

            // Call external model
            if constexpr (SFLAG == 0) {
                basis_vals[0] = species_coeffs[nbh.centertype];
            } else if (SFLAG == 1) {
                basis_vals[0] = 1.;
            }
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                basis_vals[i + soffset] = moment_vals[alpha_moment_mapping[i]];
            }
            Eigen::Map<Eigen::VectorXd> bvals(basis_vals.data(), basis_vals.size());
            ext_dUdB.resize(basis_vals.size());
            // bvals = [species_coeff, scalars]
            double site_energy = 0;

            // dudt: req_model
            // d2udbdt: req_model req_force
            // d2udbdb: req_force
            /**
             * Request Force True:
             *                 dUdB     dUdt     d2UdBdt     d2UdBdB
             * REQ_RADICAL      T         F         F            T
             * REQ_SPECIES      T         F         F            T
             * REQ_MODEL        F         T         T            F
             * Request Froce False:
             *                 dUdB     dUdt     d2UdBdt     d2UdBdB
             * REQ_RADICAL      T         F         F            F
             * REQ_SPECIES      T         F         F            F
             * REQ_MODEL        F         T         F            F
             */
            if constexpr (REQ_F) {
                if constexpr ((REQ_RADICAL == 1 || REQ_SPECIES == 1) && REQ_MODEL == 1) {
                    model.backpropgate(bvals, site_energy, ext_dUdB, ext_dUdt, ext_d2UdBdt, ext_d2UdBdB, nbh.centertype, ithread);
                } else if (REQ_RADICAL ==1 || REQ_SPECIES == 1) {
                    model.propagate(bvals, site_energy, ext_dUdB, ext_d2UdBdB, nbh.centertype, ithread);
                } else if (REQ_MODEL == 1) {
                    model.backpropgate(bvals, site_energy, ext_dUdt, ext_d2UdBdt, nbh.centertype, ithread);
                } else {
                    static_assert(std::is_constant_evaluated(), "Invalid combination of requirements");
                }
            } else {
                if constexpr ((REQ_RADICAL == 1 || REQ_SPECIES == 1) && REQ_MODEL == 1) {
                    model.backpropgate(bvals, site_energy, ext_dUdB, ext_dUdt, nbh.centertype, ithread);
                } else if (REQ_RADICAL == 1 || REQ_SPECIES == 1) {
                    model.propagate(bvals, site_energy, ext_dUdB, nbh.centertype, ithread);
                } else if (REQ_MODEL == 1) {
                    model.backpropgate(bvals, site_energy, ext_dUdt, nbh.centertype, ithread);
                } else {
                    static_assert(std::is_constant_evaluated(), "Invalid combination of requirements");
                }
            }

            // Backpropagation step 1: site energy derivative is the corresponding linear combination
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                // for non-linear model, just modify here to model value, then EVERY THING IS JUST FINE!
                site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = ext_dUdB[i + soffset]; // +1 because we have a constant center species term leading
            }

            // Backpropagation step 2: expressing through basic moments:
            for (int i = alpha_index_times_count - 1; i >= 0; --i) {
                double val0 = moment_vals[alpha_index_times(i, 0)];
                double val1 = moment_vals[alpha_index_times(i, 1)];
                double val2 = alpha_index_times(i, 2);

                site_energy_ders_wrt_moments_[alpha_index_times(i, 1)] +=
                        site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val0;
                site_energy_ders_wrt_moments_[alpha_index_times(i, 0)] +=
                        site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val1;
            }

            for (int i = 0; i < alpha_index_basic_count; ++i)
                dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i]; // currently this is weight_eng * dUi/dbm, these basic term won't be inkoved during BP process later if force is requesed
            if constexpr (REQ_F) {
                if (f_weight != nullptr) {
                    // Backpropagation step 3: multiply by the Jacobian:
                    for (int i = 0; i < alpha_index_basic_count; ++i)
                        for (int j = 0; j < nbh.nbhcount_; ++j)
                            for (int a = 0; a < 3; a++)
                                dloss_dsenders[i] += (*f_weight)(j, a) * moment_jacobian(i, j, a); // this is wgt_fceij \cdot dbm/drij
                    // f_weight * db_j/dr
                    for (int i = 0; i < alpha_index_times_count; ++i) {
                        double val2 = alpha_index_times(i, 2);
                        double val0 = moment_vals[alpha_index_times(i, 0)];
                        double val1 = moment_vals[alpha_index_times(i, 1)];
                        dloss_dsenders[alpha_index_times(i, 3)] +=
                                dloss_dsenders[alpha_index_times(i, 0)] * val2 * val1 +
                                dloss_dsenders[alpha_index_times(i, 1)] * val2 * val0;
                    }

                    if constexpr (REQ_RADICAL) {
                        // fill d/dr(dU/dB)
                        for (int i = 0; i < alpha_scalar_moments; ++i) {
                            for (int j = 0; j < alpha_scalar_moments; ++j) {
                                dloss_dmom[alpha_moment_mapping[i]] += ext_d2UdBdB(i+soffset, j+soffset) * dloss_dsenders[alpha_moment_mapping[j]];
                            }
                        }

                        // d^2E/db_jdr,
                        /**
                         * Note1. Backpropagation only ADD things to basic moments index terms, thus dloss_dmom still valid for energy der components.
                         * Note2. This loop
                         */
                        // for local energy U_i, we have dU/dbj += c * bi * dU/dbk over rule {i, j, c, k}
                        // then we have d/dr(dU/dbj) = c dbi/dr dU/dbk + c bi d/dr(dU/dbk), this loop add the first term
                        for (int i = 0; i < alpha_index_times_count; ++i) {
                            double val2 = alpha_index_times(i, 2);

                            dloss_dmom[alpha_index_times(i, 1)] += dloss_dsenders[alpha_index_times(i, 0)] *
                                                                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
                            dloss_dmom[alpha_index_times(i, 0)] += dloss_dsenders[alpha_index_times(i, 1)] *
                                                                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
                        }

                        // Backpropagation, calculate contribution of d/dr(dU/dbm), bm is basic
                        for (int i = alpha_index_times_count - 1; i >= 0; --i) {
                            double val0 = moment_vals[alpha_index_times(i, 0)];
                            double val1 = moment_vals[alpha_index_times(i, 1)];
                            double val2 = alpha_index_times(i, 2);

                            dloss_dmom[alpha_index_times(i, 0)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val1;
                            dloss_dmom[alpha_index_times(i, 1)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val0;
                        }
                    }
                }
            }
            if constexpr (REQ_RADICAL) {
                // radial coeffs
                for (int i = 0; i < alpha_index_basic_count; ++i) {
                    for (int j = 0; j < species_count; ++j) {
                        for (int k = 0; k < radial_func_count * rb_size; ++k) {
                            int idx = (nbh.centertype * species_count + j) * rb_size * radial_func_count + k;
                            out_gard_accumulator[idx] += dloss_dmom[i] * moment_jacobian_coeffs(i, j, k);
                            if constexpr (REQ_F) {
                                if (f_weight != nullptr) {
                                    // dE/db_i*d^2b/(dxdc) term, the d^2E/(db_mdrij)*db_m/dc term is included in the dloss_dmom array
                                    out_gard_accumulator[idx] +=
                                            site_energy_ders_wrt_moments_[i] * moment_jacobian_coeff_coord(i, j, k);
                                }
                            }
                        }
                    }
                }
            }
            // if SFLAG = 1, dUdB[species] === 0
            if constexpr (REQ_SPECIES && SFLAG == 0) {
                // species coeffs
                out_gard_accumulator[radial_coeff_size + nbh.centertype] += se_weight * ext_dUdB[0];
                if (f_weight != nullptr) {
                    for (int a = 0; a < alpha_scalar_moments; ++a) {
                        out_gard_accumulator[radial_coeff_size + nbh.centertype] += ext_d2UdBdB(0, a + soffset) * dloss_dsenders[alpha_moment_mapping[a]];
                    }
                }
            }
            // model coeffs. Energy part is simple, just replace B to dU/dt, force part also simple
            if constexpr (REQ_MODEL) {
                int model_offset = 0;
                if constexpr (REQ_RADICAL) model_offset += radial_coeff_size;
                if constexpr (SFLAG == 0 && REQ_SPECIES) model_offset += species_count;
                for (int i = 0; i < ext_dUdt.size(); ++i) {
                    out_gard_accumulator[model_offset + i] += se_weight * ext_dUdt[i];
                    if constexpr (REQ_F) {
                        if (f_weight != nullptr) {
                            // ignore species coeffs since dB0/drij === 0
                            for (int a = 0; a < alpha_scalar_moments; ++a) {
                                // a+1 here because we need skip first species term
                                out_gard_accumulator[model_offset + i] += ext_d2UdBdt(a + soffset, i) * dloss_dsenders[alpha_moment_mapping[a]];
                            }
                        }
                    }
                }
            }
        }

        // REQ_RADICAL: request calculate MTP radial coeffs
        // REQ_MODEL: request calculate external model coeffs
        // REQ_F: request force related terms
        // if SFLAG != 0, REQ_SPECIES should always be 0. SFLAG should be 2 in most cases
        // incorrect, need calculate q
        template<class T, int REQ_RADICAL = 1, int REQ_SPECIES = 1, int REQ_MODEL = 1, int REQ_F = 1>
        void AccumulateFPGard_n1(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight,
                              T &model,
                              int ithread = 0) {
            if constexpr (REQ_F) {
                moment_jacobian.resize(alpha_index_basic_count, nbh.nbhcount_, 3);
                moment_jacobian.setZero();
                moment_jacobian_coeff_coord.setZero();
                std::fill(rb_ders.begin(), rb_ders.end(), 0);
                std::fill(der_mu.begin(), der_mu.end(), 0);
            }
            moment_jacobian_coeffs.setZero();

            std::fill(moment_vals.begin(), moment_vals.end(), 0);
            std::fill(rb_vals.begin(), rb_vals.end(), 0);
            std::fill(val_mu.begin(), val_mu.end(), 0);
            std::fill(dloss_dmom.begin(), dloss_dmom.end(), 0);
            std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0);
            std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
            const int& itype = nbh.centertype;
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                const double &rij = nbh.distances_[jj];
                const int &jtype = nbh.chemmap_[jj];
                if (rij > cutoffs(itype, jtype)) continue;
                const double rijinv = 1.0 / rij;
                if constexpr (REQ_F)
                    rb_calcder(itype, jtype, rij);
                else
                    rb_calcval(itype, jtype, rij);

                dist_powers[0] = 1;
                for (int k = 1; k < max_alpha_index_basic; ++k) {
                    dist_powers[k] = dist_powers[k - 1] * rijinv;
                    for (int a = 0; a < 3; a++) {
                        coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a]; // off set by 1 to aviod if later
                    }
                }

                for (int mu = 0; mu < max_mu; ++mu) {
                    val_mu[mu] = 0;
                    if constexpr (REQ_F) der_mu[mu] = 0;
                    int idxoffset =
                            (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
                    for (int xi = 0; xi < rb_size; ++xi) {
                        val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                        if constexpr (REQ_F) der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
                    }
                }

                for (int i = 0; i < alpha_index_basic_count; ++i) {
                    // @OPTIMIZE: init mu-array while loading potentials, do not need offset search
                    int mu = alpha_index_basic(i, 0);
                    double val = val_mu[mu];
                    double der;
                    if constexpr (REQ_F) der = der_mu[mu];
                    int k = nu_arr_list[i];
                    double &powk = dist_powers[k];
                    // TODO: test weather move this out can accelearte computation by defining val_mu_nv[:] and der_mu_nv[:]
                    // since some (mu, nv) pair may not exist.
                    val *= powk; // normalize, all tenser element is devided by rij^\nv to avoid extremely large values
                    if constexpr (REQ_F) der = der * powk - k * val / rij;
                    // offset by 1
                    double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0); // x
                    double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1); // y
                    double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2); // z

                    double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)

                    moment_vals[i] += val * mult0;
                    for (int xi = 0; xi < rb_size; xi++) {
                        // dbm / dc
                        moment_jacobian_coeffs(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                                rb_vals[xi] * powk * mult0;
                        // force related
                        if constexpr (REQ_F)
                        if (f_weight != nullptr) {
                            // here, derx = d2bm / dxdc
                            double tail = (rb_ders[xi] * powk * mult0 - rb_vals[xi] * k * powk * mult0 / nbh.distances_[jj]);
                            double derx = (nbh.pos_[jj][0] / nbh.distances_[jj]) * tail;
                            double dery = (nbh.pos_[jj][1] / nbh.distances_[jj]) * tail;
                            double derz = (nbh.pos_[jj][2] / nbh.distances_[jj]) * tail;
                            derx += rb_vals[xi] * powk *
                                    alpha_index_basic(i, 1) *
                                    coords_powers(alpha_index_basic(i, 1), 0) *
                                    pow1 * pow2;
                            dery += rb_vals[xi] * powk *
                                    alpha_index_basic(i, 2) *
                                    coords_powers(alpha_index_basic(i, 2), 1) *
                                    pow0 * pow2;
                            derz += rb_vals[xi] * powk *
                                    alpha_index_basic(i, 3) *
                                    coords_powers(alpha_index_basic(i, 3), 2) *
                                    pow0 * pow1;
                            // this is w_i \cdot d2bm/drdc
                            moment_jacobian_coeff_coord(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                                    (*f_weight)(jj, 0) * derx + (*f_weight)(jj, 1) * dery + (*f_weight)(jj, 2) * derz;
                        }
                    }
                    if constexpr (REQ_F) {
                        mult0 *= der / rij;
                        moment_jacobian(i, jj, 0) += mult0 * nbh.pos_[jj][0];
                        moment_jacobian(i, jj, 1) += mult0 * nbh.pos_[jj][1];
                        moment_jacobian(i, jj, 2) += mult0 * nbh.pos_[jj][2];
                        // coords_powers is shifted by 3 to avoid if statement
                        // note that 0 * Inf = NaN
                        moment_jacobian(i, jj, 0) +=
                                alpha_index_basic(i, 1) * val *
                                coords_powers(alpha_index_basic(i, 1), 0) *
                                pow1 * pow2;
                        moment_jacobian(i, jj, 1) +=
                                alpha_index_basic(i, 2) * val *
                                coords_powers(alpha_index_basic(i, 2), 1) *
                                pow0 * pow2;
                        moment_jacobian(i, jj, 2) +=
                                alpha_index_basic(i, 3) * val *
                                coords_powers(alpha_index_basic(i, 3), 2) *
                                pow0 * pow1;
                    }
                }
            }

            // non-elementary b_i
            for (int i = 0; i < alpha_index_times_count; ++i) {
                double val0 = moment_vals[alpha_index_times(i, 0)];
                double val1 = moment_vals[alpha_index_times(i, 1)];
                double val2 = alpha_index_times(i, 2);
                moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
            }

            // Call external model
            double q = species_coeffs[nbh.centertype];
            double ext_dudq = 0;
            double ext_d2udq2 = 0;
            double site_energy = 0;

            // dudt: req_model
            // d2udbdt: req_model req_force
            // d2udbdb: req_force
            /**
             * Request Force True:
             *                 dUdB     dUdt     d2UdBdt     d2UdBdB
             * REQ_RADICAL      T         F         F            T
             * REQ_SPECIES      T         F         F            T
             * REQ_MODEL        F         T         T            F
             * Request Froce False:
             *                 dUdB     dUdt     d2UdBdt     d2UdBdB
             * REQ_RADICAL      T         F         F            F
             * REQ_SPECIES      T         F         F            F
             * REQ_MODEL        F         T         F            F
             */
            if constexpr (REQ_F) {
                if constexpr ((REQ_RADICAL == 1 || REQ_SPECIES == 1) && REQ_MODEL == 1) {
                    model.backpropgate(q, site_energy, ext_dudq, ext_dUdt, ext_d2udqdt, ext_d2udq2, nbh.centertype, ithread);
                } else if (REQ_RADICAL ==1 || REQ_SPECIES == 1) {
                    model.propagate(q, site_energy, ext_dudq, ext_d2udq2, nbh.centertype, ithread);
                } else if (REQ_MODEL == 1) {
                    model.backpropgate(q, site_energy, ext_dUdt, ext_d2udqdt, nbh.centertype, ithread);
                } else {
                    static_assert(std::is_constant_evaluated(), "Invalid combination of requirements");
                }
            } else {
                if constexpr ((REQ_RADICAL == 1 || REQ_SPECIES == 1) && REQ_MODEL == 1) {
                    model.backpropgate(q, site_energy, ext_dudq, ext_dUdt, nbh.centertype, ithread);
                } else if (REQ_RADICAL == 1 || REQ_SPECIES == 1) {
                    model.propagate(q, site_energy, ext_dudq, nbh.centertype, ithread);
                } else if (REQ_MODEL == 1) {
                    model.backpropgate(q, site_energy, ext_dUdt, nbh.centertype, ithread);
                } else {
                    static_assert(std::is_constant_evaluated(), "Invalid combination of requirements");
                }
            }

            // Backpropagation step 1: site energy derivative is the corresponding linear combination
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                // for non-linear model, just modify here to model value, then EVERY THING IS JUST FINE!
                site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = moment_coeffs[i] * ext_dudq;
            }

            // Backpropagation step 2: expressing through basic moments:
            for (int i = alpha_index_times_count - 1; i >= 0; --i) {
                double val0 = moment_vals[alpha_index_times(i, 0)];
                double val1 = moment_vals[alpha_index_times(i, 1)];
                double val2 = alpha_index_times(i, 2);

                site_energy_ders_wrt_moments_[alpha_index_times(i, 1)] +=
                        site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val0;
                site_energy_ders_wrt_moments_[alpha_index_times(i, 0)] +=
                        site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val1;
            }

            for (int i = 0; i < alpha_index_basic_count; ++i)
                dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i]; // currently this is weight_eng * dUi/dbm, these basic term won't be inkoved during BP process later if force is requesed
            if constexpr (REQ_F) {
                if (f_weight != nullptr) {
                    // Backpropagation step 3: multiply by the Jacobian:
                    for (int i = 0; i < alpha_index_basic_count; ++i)
                        for (int j = 0; j < nbh.nbhcount_; ++j)
                            for (int a = 0; a < 3; a++)
                                dloss_dsenders[i] += (*f_weight)(j, a) * moment_jacobian(i, j, a); // this is wgt_fceij \cdot dbm/drij
                    // f_weight * db_j/dr
                    for (int i = 0; i < alpha_index_times_count; ++i) {
                        double val2 = alpha_index_times(i, 2);
                        double val0 = moment_vals[alpha_index_times(i, 0)];
                        double val1 = moment_vals[alpha_index_times(i, 1)];
                        dloss_dsenders[alpha_index_times(i, 3)] +=
                                dloss_dsenders[alpha_index_times(i, 0)] * val2 * val1 +
                                dloss_dsenders[alpha_index_times(i, 1)] * val2 * val0;
                    }

                    if constexpr (REQ_RADICAL) {
                        // fill d/dr(dU/dB)
                        double sumdBdr = 0;
                        for (int i = 0; i < alpha_scalar_moments; ++i) {
                            sumdBdr += dloss_dsenders[alpha_moment_mapping[i]];
                        }
                        for (int i = 0; i < alpha_scalar_moments; ++i) {
                            dloss_dmom[alpha_moment_mapping[i]] += moment_coeffs[i] * ext_d2udq2 * sumdBdr;
                        }

                        // d^2E/db_jdr,
                        /**
                         * Note1. Backpropagation only ADD things to basic moments index terms, thus dloss_dmom still valid for energy der components.
                         * Note2. This loop
                         */
                        // for local energy U_i, we have dU/dbj += c * bi * dU/dbk over rule {i, j, c, k}
                        // then we have d/dr(dU/dbj) = c dbi/dr dU/dbk + c bi d/dr(dU/dbk), this loop add the first term
                        for (int i = 0; i < alpha_index_times_count; ++i) {
                            double val2 = alpha_index_times(i, 2);

                            dloss_dmom[alpha_index_times(i, 1)] += dloss_dsenders[alpha_index_times(i, 0)] *
                                                                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
                            dloss_dmom[alpha_index_times(i, 0)] += dloss_dsenders[alpha_index_times(i, 1)] *
                                                                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
                        }

                        // Backpropagation, calculate contribution of d/dr(dU/dbm), bm is basic
                        for (int i = alpha_index_times_count - 1; i >= 0; --i) {
                            double val0 = moment_vals[alpha_index_times(i, 0)];
                            double val1 = moment_vals[alpha_index_times(i, 1)];
                            double val2 = alpha_index_times(i, 2);

                            dloss_dmom[alpha_index_times(i, 0)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val1;
                            dloss_dmom[alpha_index_times(i, 1)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val0;
                        }
                    }
                }
            }
            if constexpr (REQ_RADICAL) {
                // radial coeffs
                for (int i = 0; i < alpha_index_basic_count; ++i) {
                    for (int j = 0; j < species_count; ++j) {
                        for (int k = 0; k < radial_func_count * rb_size; ++k) {
                            int idx = (nbh.centertype * species_count + j) * rb_size * radial_func_count + k;
                            out_gard_accumulator[idx] += dloss_dmom[i] * moment_jacobian_coeffs(i, j, k);
                            if constexpr (REQ_F) {
                                if (f_weight != nullptr) {
                                    // dE/db_i*d^2b/(dxdc) term, the d^2E/(db_mdrij)*db_m/dc term is included in the dloss_dmom array
                                    out_gard_accumulator[idx] +=
                                            site_energy_ders_wrt_moments_[i] * moment_jacobian_coeff_coord(i, j, k);
                                }
                            }
                        }
                    }
                }
            }
            // if SFLAG = 1, dUdB[species] === 0
            if constexpr (REQ_SPECIES) {
                // species coeffs
                out_gard_accumulator[radial_coeff_size + nbh.centertype] += se_weight * ext_dudq;// ext_dUdB[0];
                if (f_weight != nullptr) {
                    // TODO next: move this out to avoid multiple idential calculation
                    double moment_sum = 0;
                    for (int a = 0; a < alpha_scalar_moments; ++a) {
                        moment_sum += moment_coeffs[a];
                    }
                    out_gard_accumulator[radial_coeff_size + nbh.centertype] += moment_sum * ext_d2udq2;
                }
            }
            // model coeffs. Energy part is simple, just replace B to dU/dt, force part also simple
            if constexpr (REQ_MODEL) {
                int model_offset = 0;
                if constexpr (REQ_RADICAL) model_offset += radial_coeff_size;
                if constexpr (REQ_SPECIES) model_offset += species_count;
                double sumcbr = 0;
                for (int a = 0; a < alpha_scalar_moments; ++a) {
                    sumcbr += moment_coeffs[a] * dloss_dsenders[alpha_moment_mapping[a]];
                }
                for (int i = 0; i < ext_dUdt.size(); ++i) {
                    out_gard_accumulator[model_offset + i] += se_weight * ext_dUdt[i];
                    if constexpr (REQ_F) {
                        if (f_weight != nullptr) {
                            // ignore species coeffs since dB0/drij === 0
                            out_gard_accumulator[model_offset + i] += sumcbr * ext_d2udqdt[i];
                        }
                    }
                }
            }
        }

        void AccumulateFPGard(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,                              // usually 2*wgt_eng*(E_ref - E_efs)
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight,
                              int roffset, int loffset);

        // only linear coeffs
        void AccumulateFPGardLonly(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,                              // usually 2*wgt_eng*(E_ref - E_efs)
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight);

        void AccumulateFPGardLonly(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,                              // usually 2*wgt_eng*(E_ref - E_efs)
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight, int coffset);

        void AccumulateFPGardRonly(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,                              // usually 2*wgt_eng*(E_ref - E_efs)
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight);

        void AccumulateFPGardRonly(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,                              // usually 2*wgt_eng*(E_ref - E_efs)
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight, int coffset);

        void compute_descriptors(Neighborhood &nbh,
                                 std::vector<double> &disp); // scalar descriptors, CalcSiteBasis without der

        void CalcLocalDescriptors(Atoms &atom, Eigen::MatrixXd &desc);

        void compute_local_ecmpt(Neighborhood &nbh, std::vector<double> &disp, bool fuse);

        void compute_sensitive_matrix_with_center(Neighborhood &nbh, Eigen::MatrixXd &senmatrix, Eigen::VectorXd &vec);

        void compute_sensitive_matrix(Neighborhood &nbh, Eigen::MatrixXd &senmatrix, Eigen::VectorXd &vec);

        void corr_matrix(std::vector<Atoms> &atoms,
                        Eigen::MatrixXd &corrcoeffs,
                        Eigen::VectorXd &m,
                        Eigen::VectorXd &s,
                        int max_smaple_size = INT_MAX);

        // give scalar moment id
        void remove_scalar_by_moment(std::vector<int> &scalars);
        // position in scalar_moment_mapping
        void remove_scalar_by_id(std::set<int> &ids);

        BasisSubsetAnalysis analyze_basis(const std::vector<int> &scalar_ids) const;
        void remove_orphans();
        /**
         * @brief delete items in removed ids that are scalar on the decomp path of left scalar tensors.
         * @param ids ids to be removed
         */
        void add_back_path_scalars_to_remove_id(std::set<int> &ids);


        // Hackings for MTP_M potential
        virtual void get_n_props(int &nprops) { GPERROR("This MTP does not expose an MTP_M model head"); };
        virtual void get_n_coeffs(int &n_radical_coeffs, int &n_species_coeffs, int &n_moment_coeffs, int &n_model_coeffs) { GPERROR("This MTP does not expose an MTP_M model head"); };
        virtual void get_pointers(double* &radical_coeffs_ptr, double* &species_coeffs, double* &moment_coeffs, double* &model_coeffs) { GPERROR("This MTP does not expose an MTP_M model head"); };
    };
    /**
     * @brief
     * Hybrid MTP with a 2-body core repulsive term
     */



};
