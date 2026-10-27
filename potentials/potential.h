// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include <functional>
#include "base/structures.h"
#include "base/neighborhood.h"
#include "utils/gputils.h"
#include "utils/compressed_stream.h"
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * @brief Basic class for potentials
 *
 * potential_wrapper constructs supported potential types from GPSettings.
 *
 */
namespace GPM { class Reader; }

namespace Potentials
{
    struct NLParameterGroup {
        std::string name;
        int begin = 0;
        int end = 0;
    };

    // A model explicitly opts in only when its complete E/F/stress prediction is
    // affine in these packed optimizer parameters with every other parameter fixed.
    // The first composition_count columns are additive species constants. Penalty
    // entries are the actual coefficients of L1*abs(x) + L2*x*x in AddPenaltyGard.
    struct ConditionalLinearBlock {
        std::string name;
        std::vector<int> indices;
        int composition_count = 0;
        std::vector<double> l1, l2;
        void append(int index, double weight_l1 = 0., double weight_l2 = 0.) {
            indices.push_back(index); l1.push_back(weight_l1); l2.push_back(weight_l2);
        }
    };

    using SiteCoordinateMatrix = Eigen::Matrix<double, -1, 3, Eigen::RowMajor>;

    /// @brief Dense Hessian matrix storage as vector of 3x3 blocks.
    /// @details Provides 4D indexing: operator()(i, j)(a, b) where (i,j) are atom indices and (a,b) Cartesian components.
    struct HessianMatrix {
        int size;
        std::vector<Eigen::Matrix3d> hessian;

        void resize(int _size) {
            size = _size;
            // hessian.assign(size * (size + 1) / 2, Eigen::Matrix3d::Zero());
            hessian.assign(size * size, Eigen::Matrix3d::Zero());
        }

        void setzero() {
            hessian.assign(size * size, Eigen::Matrix3d::Zero());
        }

        int idx(int i, int j) const {
            // if (i < j) std::swap(i, j);
            // return (i+i)*i/2+j;
            return i * size + j;
        }
        
        Eigen::Matrix3d &operator()(int i, int j) {
            return hessian[idx(i,j)];
        }

        Eigen::Matrix3d operator()(int i, int j) const {
            return hessian[idx(i,j)];
        }

        double &operator()(int i, int a, int j, int b) {
            return hessian[idx(i,j)](a,b);
        }

        double operator()(int i, int a, int j, int b) const {
            return hessian[idx(i,j)](a,b);
        }
    };

    /// @brief Sparse Hessian matrix using map of atom-pair -> 3x3 block (i <= j).
    struct HessianSparse {
        int size;
        std::map<std::pair<int, int>, Eigen::Matrix3d> hessian;
        
        // Note: make sure that i <= j
        Eigen::Matrix3d &operator()(int i, int j) {
            return hessian[std::make_pair(i, j)];
        }

        Eigen::Matrix3d operator()(int i, int j) const {
            if (i > j) {
                std::swap(i, j);
                return hessian.at(std::make_pair(i, j)).transpose();
            } else {
                return hessian.at(std::make_pair(i, j));
            }
        }

        double &operator()(int i, int a, int j, int b) {
            if (i > j) {
                std::swap(i, j);
                std::swap(a, b);
            }
            return hessian[std::make_pair(i, j)](a,b);
        }

        double operator()(int i, int a, int j, int b) const {
            if (i > j) {
                std::swap(i, j);
                std::swap(a, b);
            }
            return hessian.at(std::make_pair(i, j))(a,b);
        }
    };

/**
 * @brief Abstract base class for all interatomic potentials.
 * @ingroup potentials
 *
 * Defines the virtual interface for energy, force, stress (EFS) calculations.
 * Key virtual methods to override:
 * - getEFS(): single-threaded EFS computation
 * - getEFSOMP(): OpenMP-parallel EFS
 * - getEFSOMPthread(): thread-safe per-thread EFS
 * - getEFSHessian() / getEFSHessianCell(): Hessian matrix computation
 *
 * Training interface (for nonlinear fitting):
 * - AccumulateEFSCombinationGard(): accumulate loss gradient w.r.t. coefficients
 * - nl_get_coeffs() / nl_set_coeffs(): get/set trainable parameters
 *
 * Derived classes must implement Load/Save for serialization.
 *
 * @see PotMTP, ConnectSettings
 */
class AnyPotential;
bool load_gpm_if_requested(AnyPotential &potential,const std::string &path);
bool save_gpm_if_requested(AnyPotential &, const std::string &);

class AnyPotential : public ConnectSettings
    {
    private:

    protected:
        std::string pot_type_;
        enum properties
        {
            ENERGY,
            STRESS,
            FORCE,
            MAGMOMS,
            MAGMOM
        };
        std::vector<AnyPotential::properties> properties_avaliable_;

    public:
        bool omp_inited = false;
        bool add_efs = false; // do not clean state
        double cutoff = 5.0;
        double cutoffsq = 25.0;
        int species_count = 0;
        // Optional chemical identities in the same order as internal species IDs.
        std::map<int, std::string> gpm_elements;
        std::shared_ptr<GPM::Reader> gpm_source;
        std::shared_ptr<void> gpm_resources;
        virtual void set_element_map(const std::map<int, std::string> &mapping) { if (!mapping.empty()) gpm_elements = mapping; }
        struct SpeciesAppendResult {
            // Mapping for every parameter present before this operation.
            std::vector<int> old_to_new_parameters;
        };
        virtual bool supports_species_append() const { return false; }
        // Append only. Each source is an existing internal species ID; callers
        // must reinitialize trainers after a nonempty append.
        virtual SpeciesAppendResult append_species(const std::vector<int>&,
            const std::vector<std::string>& = {}) {
            GPERROR("This potential does not support appending species");
        }

        std::vector<double> reg_L2_vector, reg_L1_vector;
        Eigen::MatrixXd cutoffs, cutoffsqs, r_mins; // cutoffs: rmax, r_mins: rmin
        AnyPotential(/* args */){};
        virtual ~AnyPotential(){};

        virtual void getEFS(Atoms &atom) {
            Neighborhoods nbhs(atom, cutoff);
            getEFS(atom, nbhs);
        };
        virtual void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) = 0;
        
        virtual void omp_init() {};

        virtual void getEFSOMP(Atoms &atom) {
            Neighborhoods nbhs(atom, cutoff);
            getEFSOMP(atom, nbhs);
        }
        virtual void getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs) {getEFS(atom, nbhs);};

        // this version must be thread-safe, for openmp
        virtual void getEFSOMPthread(Atoms &atom) {
            Neighborhoods nbhs(atom, cutoff);
            getEFSOMPthread(atom, nbhs);
        };
        virtual void getEFSOMPthread(Atoms &atom, AnyNeighborhoods &nbhs) {
            getEFS(atom, nbhs);
        };
        virtual void Load(std::istream &is) {
            GPERROR("Load(istream) is not supported by this potential");
        };
        virtual void Load(const std::string &potfile) {
            auto ifs = open_ifile(potfile);
            if (!ifs) {
                throw std::runtime_error("Failed to open file: " + potfile);
            }
            Load(*ifs);
        };
        // TODO: make them abstract
        virtual void getE(Atoms &atom) { 
            Neighborhoods nbhs(atom, cutoff);
            getE(atom, nbhs); 
        };
        virtual void getE(Atoms &atom, AnyNeighborhoods &nbhs) {getEFS(atom, nbhs);};
        virtual void getEOMP(Atoms &atom) { 
            Neighborhoods nbhs(atom, cutoff);
            getEOMP(atom, nbhs); 
        };
        virtual void getEOMP(Atoms &atom, AnyNeighborhoods &nbhs) {getE(atom, nbhs);};

        virtual bool supports_site_efs() const { return false; }
        // Derivatives of one site's energy with respect to independent relative
        // neighbor coordinates r_neighbor-r_center, in neighborhood row order.
        // The gradient is dE/dr, not a force. Repeated image IDs remain separate
        // rows. Calls overwrite outputs and use the same serial workspace as
        // CalcSiteEFS; callers must serialize access to one potential instance.
        virtual bool supports_site_hvp() const { return false; }
        virtual bool supports_site_hessian() const { return false; }

        virtual void apply_site_hvp(Neighborhood &nbh, const SiteCoordinateMatrix &direction,
                                    double &energy, SiteCoordinateMatrix &gradient,
                                    SiteCoordinateMatrix &product);
        virtual void evaluate_site_hessian(Neighborhood &nbh, double &energy,
                                           SiteCoordinateMatrix &gradient, HessianMatrix &hessian);
        // H(i,a,j,b) = d2E/dr_i,a dr_j,b;
        // FC3(i,j,k,a,b,c) = d3E/dr_i,a dr_j,b dr_k,c (full ordered storage).

        // Opt-in contract: the full energy is exactly the sum of CalcSiteE,
        // each depending only on types/geometry within cutoff. No global terms.

        // Independent, energy-only worker for concurrent local MC evaluations.
        // Must preserve the current parameters and own all mutable workspaces.
        // No atom/neighbor data is copied. Unsupported adapters return nullptr.

        /// Whether AccumulateFPGard provides analytical parameter gradients for
        /// the same site-energy decomposition exposed by CalcSiteEFS.
        virtual bool supports_site_parameter_gradients() const { return false; }
        /// Optional electrical-response interface used by phonon/NAC tools.
        virtual bool supports_born_effective_charges() const { return false; }
        virtual void compute_born_effective_charges(
            Atoms &, const std::vector<int> & = {}) {
            GPERROR("This potential does not support Born effective charges");
        }
        virtual void CalcSiteEFS(Neighborhood &nbh, double &site_energy,
                                 Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &site_energy_ders) {
            GPERROR("This potential does not support site EFS evaluation");
        }
        virtual void CalcSiteE(Neighborhood &nbh, double &site_energy) {
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> site_energy_ders;
            CalcSiteEFS(nbh, site_energy, site_energy_ders);
        }
        // Compatibility fallback for site adapters without a direct view path.
        // Short-range MTP implementations override this to avoid copies.


        virtual void getEFSHessian(Atoms &atom, HessianMatrix &hessian) {
            GPERROR("getEFSHessian(Atoms&, HessianMatrix&) is not supported by this potential");
        };

        virtual void getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
            GPERROR("getEFSHessian(Atoms&, AnyNeighborhoods&, HessianMatrix&) is not supported by this potential");
        };



        virtual void getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) { GPERROR("This potential does not support this analytical Hessian evaluation"); };

        virtual void getEFSHessianCell(Atoms &atom, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) { GPERROR("This potential does not support this analytical Hessian evaluation"); };

        virtual void getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double symprec = 1e-4) { GPERROR("This potential does not support this analytical Hessian evaluation"); };

        virtual void getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) {
            GPERROR("getEFSHessianCellSymm is not supported by this potential");
        };

        virtual void getEFSHessianSymmOMP(Atoms &atom, HessianMatrix &hessian, double symprec = 1e-4) {
            getEFSHessianSymm(atom, hessian, symprec);
        };

        /// Whether this potential provides an analytical atomic Hessian.
        virtual bool supports_analytical_hessian() const { return false; }
        // Some native kernels are valid only on a particular geometry/branch.
        // Auto queries consult these hooks; older models keep their capability.
        virtual bool supports_analytical_hessian_for(const Atoms &) const {
            return supports_analytical_hessian();
        }

        /// Whether this potential can apply its analytical Cartesian Hessian
        /// without materialising the full matrix.
        virtual bool supports_analytical_hessian_vector_product() const {
            return false;
        }
        virtual bool supports_analytical_hessian_vector_product_for(const Atoms &) const {
            return supports_analytical_hessian_vector_product();
        }

        /// Apply the Cartesian atomic Hessian to an atom-major xyz vector.
        /// Implementations must overwrite @p product and must not retain
        /// references to either vector after returning.
        virtual void applyHessianVectorProduct(
            Atoms &, AnyNeighborhoods &, const Eigen::VectorXd &,
            Eigen::VectorXd &) {
            GPERROR("This potential does not support analytical Hessian-vector products");
        }

        virtual void applyHessianVectorProduct(
            Atoms &atom, const Eigen::VectorXd &direction,
            Eigen::VectorXd &product) {
            Neighborhoods nbhs(atom, cutoff);
            applyHessianVectorProduct(atom, nbhs, direction, product);
        }

        virtual void getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec = 1e-4) {
            GPERROR("getEFSHessianCellSymmOMP is not supported by this potential");
        };

        /// Whether this potential supports analytical cell Hessian computation (e.g. getEFSHessianCellSymmOMP).
        /// Override in derived classes that implement it.
        virtual bool supports_analytical_cell_hessian() const { return false; }
        virtual bool supports_analytical_cell_hessian_for(const Atoms &) const {
            return supports_analytical_cell_hessian();
        }

        virtual void getEFSHessianSymmDiaOMP(Atoms &atom, std::vector<double> &hessian, double symprec = 1e-4) {
            GPERROR("getEFSHessianSymmDiaOMP is not supported by this potential");
        };

        virtual void getHessian(Atoms &atom, AnyNeighborhoods &nbhs, std::vector<std::pair<int, int>> inds, HessianSparse &hessian) {
            GPERROR("getHessian (sparse Hessian) is not supported by this potential");
        };

        virtual void getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
            GPERROR("getEFSHessianOMP is not supported by this potential");
        };

        virtual void getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
            GPERROR("getEFSHessianCellOMP is not supported by this potential");
        };

        /* For NL_TRANINING */
        // One may overload these definitions
        bool nl_inited = false;
        virtual void set_inited() {nl_inited = true;};
        virtual void set_empty() {nl_inited = false;};
        virtual bool inited() {return nl_inited;};
        // dLdX: 2 * weight * (Xprd - Xref) for MSE loss
        // Usually, one does not need to override this unless the loss function can not be regarded as a linear operator.
        virtual void AccumulateEFSCombinationGard(Atoms &atom,
                                          AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &wgt_d2UdRdx, // buffer
                                          double *out_gard_accumulator);
        // For energy-only case
        virtual void AccumulateEFSCombinationGard(Atoms &atom,
                                          AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          double *out_gard_accumulator);
        // penalty term, incliding L1/L2. This should be user supplied since some coeffs may no need regulation
        virtual void AddPenaltyGard(double weight, double &loss, double *loss_gard) {};
        
        // One should give these definitions
        virtual void random_init() {};
        virtual void init_coeffs() {};
        virtual void train_init() {}; // mainly for allocate memory
        virtual void set_scale(double) {}; // Optional scaling hook; models with scale parameters override it.

        // One MUST give these definitions
        virtual int get_NL_coeffs_count() const {return 0;};
        virtual void nl_get_coeffs(double *) const {};
        virtual void nl_set_coeffs(const double *){};
        virtual std::vector<NLParameterGroup> nl_parameter_groups() const {
            return {{"all", 0, get_NL_coeffs_count()}};
        }
        virtual std::vector<ConditionalLinearBlock> conditional_linear_blocks(double penalty_weight) const {
            return {};
        }
        // Update only advertised affine coordinates. Models with transformed
        // optimizer coordinates override this to leave frozen physical state exact.
        virtual void conditional_linear_set_coeffs(const std::vector<int>& indices,
                                                    const Eigen::VectorXd& values);
        // Optional latent-charge training interface. getEFS() must be called
        // first so implementations can reuse prediction caches.
        virtual bool supports_total_charge_training() const { return false; }
        virtual double predicted_total_charge(const Atoms &) const {
            GPERROR("This potential does not support total-charge training");
        }
        virtual void AccumulateTotalChargeGard(Atoms &, AnyNeighborhoods &,
                                               double, double *) {
            GPERROR("This potential does not support total-charge training");
        }
        // compute dU_i/dx and d^2U_i/dR_{ija}dx for single i
        virtual void AccumulateFPGard(Neighborhood &nbh,
                              double *out_gard_accumulator,
                              double se_weight,       // 2*wgt_eng*(E_ref - E_efs) for MSE loss
                              Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight) {
            GPERROR("This potential does not support analytical site parameter gradients");
        };

        // ── Numerical gradient helper ────────────────────────────────────
        /// Central-difference numerical gradient for potentials without
        /// analytical derivatives. Used by SW and Tersoff fitting.
        /// \param set_param  callback: set_param(idx, value) sets coeff idx to value
        /// \param coeffs     current coefficient values (updated in-place as side effect of set_param)
        /// \param n_params   number of trainable parameters
        void numerical_gradient_efs(
            Atoms &atom, AnyNeighborhoods &nbhs,
            double dLdE,
            Eigen::Matrix<double, -1, 3> &dLdF,
            Eigen::Matrix3d &dLdS,
            double *out_gard_accumulator,
            int n_params,
            const std::function<void(int, double)> &set_param);

        void numerical_gradient_eonly(
            Atoms &atom, AnyNeighborhoods &nbhs,
            double dLdE,
            double *out_gard_accumulator,
            int n_params,
            const std::function<void(int, double)> &set_param);

        // ── Linear fitting interface ────────────────────────────────────────
        /// Whether this potential supports direct linear fitting (assembly of
        /// the design matrix A and solving A^T A w = A^T y).
        virtual bool supports_linear_fit() const { return false; }
        /// Compute the descriptor vector for one atom's neighbourhood.
        /// \p desc_out length = get_NL_coeffs_count().
        virtual void linear_compute_descriptor(Neighborhood &nbh, double *desc_out) const { GPERROR("This potential does not support linear fitting descriptors"); }
        /// Compute descriptor vector + position gradients (for force/stress fitting).
        /// \p desc_out length = feature_size.
        /// \p grad_out flat row-major (feature_size, n_neighbors, 3): grad[f*nnb*3 + j*3 + a].
        virtual void linear_compute_descriptor_grad(Neighborhood &nbh, double *desc_out, double *grad_out) const { GPERROR("This potential does not support linear fitting descriptors"); }

        virtual void Save(std::ostream &os) { GPERROR("This potential does not support saving to a stream"); };
        virtual void Save(const std::string &potfile) {
            if (save_gpm_if_requested(*this, potfile)) return;
            auto ofs = open_ofile(potfile);
            if (!ofs) {
                throw std::runtime_error("Failed to open file: " + potfile);
            }
            Save(*ofs);
            if (!close_ofile(ofs))
                throw std::runtime_error("Failed to write potential file: " + potfile);
        };
    };
}; // Potentials
