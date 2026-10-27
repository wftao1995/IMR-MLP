// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential_mtp.h"
#include "les.h"
#include "utils/compressed_stream.h"

namespace Potentials {

/**
 * @brief PotMTP with Latent Ewald Summation (LES) for long-range electrostatics.
 *
 * Three-stage evaluation:
 *   1. Compute latent charge q_i = les_species[type_i] + Σ les_moment[a] * B_i[a]
 *   2. Ewald sum → LR energy, direct forces, dE/dq_i
 *   3. Backpropagate dE/dq_i through descriptor Jacobian → indirect forces
 *
 * Coefficients are stored separately from PotMTP's regression_coeffs:
 *   - les_species_coeffs[species_count] : species bias for latent charge
 *   - les_moment_coeffs[alpha_scalar_moments] : linear weights for MTP descriptors
 */
class PotMTP_LES : public PotMTP {
protected:
    void clear_failed_load() noexcept override;
public:
    bool supports_site_parameter_gradients() const override { return false; }
    bool lr_les = false;
    int les_version = LES::format_version;
    std::string les_kernel = "coulomb";
    int sog_terms = 8; // Per channel.
    int sog_channels = 1;
    int channels() const {return uses_sog() ? sog_channels : 1;}
    int latent_species_count() const {return channels()*species_count;}
    int latent_moment_count() const {return channels()*alpha_scalar_moments;}
    int kernel_count() const {return channels()*sog_terms;}
    double sog_log_bandwidth_min = -4.0;
    double sog_log_bandwidth_max = 1.0;
    bool uses_sog() const { return les_kernel == "sog"; }
    double lr_sigma = 0.5;
    double dl = 2.0;                // Real-space grid resolution (Å)
    double les_taper_fraction = 0.1;
    std::string les_cutoff_mode = "smooth";
    std::string les_charge_treatment = "uniform_background";
    double les_init_range = 0.0;    // LES init range: <0=conservative(1e-3), 0=auto(random_upper), >0=explicit
    bool les_balance_initial_charges = false; // explicit new-model initialization only
    double les_hessian_fd_nudge = 1e-3;
    bool remove_self_interaction = true; // Remove reciprocal Gaussian self energy, as in reference LES
    int les_response_parity = 1; // Sign convention for charge-like observables (+1 or -1)

    double *les_species_coeffs = nullptr;  // [species_count]
    double *les_moment_coeffs  = nullptr;  // [alpha_scalar_moments]

    Eigen::MatrixXd LES_all_latents; // N x C, populated by prediction.
    Eigen::VectorXd LES_all_qs;    // buffer: one latent charge per atom
    LES model_les;                 // Ewald summation engine

    // ── Construction / destruction ─────────────────────────────────────
    PotMTP_LES() = default;
    explicit PotMTP_LES(const std::string &potfile) : PotMTP() {
        auto ifs = open_ifile(potfile);
        if (!ifs)
            GPERROR("Cannot open potential file: " + potfile);
        Load(*ifs);
    }
    /// Construct from template file + settings (like PotMTP_M).
    /// Accepts valid MTP graph templates without fitted coefficients.
    /// NOTE: for LES templates, use trainnlpot with fit:random_init true instead.
    PotMTP_LES(const std::string &potfile, GPSettings &settings);
    ~PotMTP_LES() override {
        delete[] les_species_coeffs;
        delete[] les_moment_coeffs;
    }
    // Prevent shallow copy of owning raw pointers
    PotMTP_LES(const PotMTP_LES &) = delete;
    PotMTP_LES &operator=(const PotMTP_LES &) = delete;

    // ── Serialisation ──────────────────────────────────────────────────
    void Load(std::istream &is) override;
    void Save(std::ostream &os) override;

    std::string metadata_model_type() const override {return "MTP_LES";}
    MTPMetadata::Object model_metadata() const override {
        using MTPMetadata::Value;
        auto fields = PotMTP::model_metadata();
        fields["type"] = Value("MTP_LES");
        if (lr_les) fields["long_range"] = Value(MTPMetadata::Object{
            {"kernel", Value(les_kernel)}, {"channels", Value(double(channels()))}});
        return fields;
    }

    // ── Core evaluation ────────────────────────────────────────────────
    void getEFS(Atoms &atom) override;
    void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void getEFSOMP(Atoms &atom) override;
    void getEFSSymmOMP(Atoms &atom, double symprec = 1e-2) override;
    void getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void getEFSOMPthread(Atoms &atom) override;
    void getEFSOMPthread(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void getE(Atoms &atom) override;
    void getE(Atoms &atom, AnyNeighborhoods &nbhs) override;
    void getEOMP(Atoms &atom) override;
    void getEOMP(Atoms &atom, AnyNeighborhoods &nbhs) override;

    bool supports_site_efs() const override { return false; }
    void CalcSiteEFS(Neighborhood &, double &,
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &) override;
    void CalcSiteE(Neighborhood &, double &) override;
    void getEFSHessian(Atoms &, HessianMatrix &) override;
    void getEFSHessian(Atoms &, AnyNeighborhoods &, HessianMatrix &) override;
    void getEFSHessianSymm(Atoms &, HessianMatrix &, double = 1e-4) override;
    void getEFSHessianSymmOMP(Atoms &, HessianMatrix &, double = 1e-4) override;
    void getEFSHessianOMP(Atoms &, AnyNeighborhoods &, HessianMatrix &) override;
    void getHessian(Atoms &, AnyNeighborhoods &, std::vector<std::pair<int, int>>, HessianSparse &) override;
    void getEFSHessianSymmDiaOMP(Atoms &, std::vector<double> &, double = 1e-4) override;
    bool supports_analytical_hessian() const override { return supports_analytical_hessian_vector_product(); }
    bool supports_analytical_hessian_vector_product() const override {
        return typeid(*this) == typeid(PotMTP_LES) && !hc_correction && !rR_correction
            && (!lr_les || (les_species_coeffs && les_moment_coeffs));
    }
    bool supports_analytical_cell_hessian() const override {
        return supports_analytical_hessian_vector_product() && (!lr_les || les_cutoff_mode == "smooth");
    }
    using PotMTP::applyHessianVectorProduct;
    void applyHessianVectorProduct(Atoms &, AnyNeighborhoods &,
        const Eigen::VectorXd &, Eigen::VectorXd &) override;
    void getEFSHessianCell(Atoms &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &) override;
    void getEFSHessianCell(Atoms &, AnyNeighborhoods &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &) override;
    void getEFSHessianCellSymm(Atoms &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &, double = 1e-4) override;
    void getEFSHessianCellSymmOMP(Atoms &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &, double = 1e-4) override;
    void getEFSHessianCellOMP(Atoms &, AnyNeighborhoods &, HessianMatrix &, Eigen::MatrixXd &, Eigen::MatrixXd &) override;

    // ── Born Effective Charges ─────────────────────────────────────────
    /// Compute the finite-wavevector BEC with analytic local charge derivatives.
    /// Recomputes charges independently; no preceding energy evaluation needed.
    /// This preserves the dephased finite-k definition, not a conservative
    /// uniform-field energy or a cell-representation-independent k=0 limit.
    /// @param remove_mean    Center charges (subtract mean) before computing BEC
    /// @param epsilon_factor High-frequency dielectric constant scaling (√ε_∞)
    void compute_bec(Atoms& atom,
                     bool remove_mean = true,
                     double epsilon_factor = 1.0);
    bool supports_born_effective_charges() const override { return lr_les && !uses_sog(); }
    void compute_born_effective_charges(
        Atoms& atom, const std::vector<int>& = {}) override {
        compute_bec(atom);
    }

    // ── Charge-balanced initialization ──────────────────────────────────
    /// Adjust les_species_coeffs so each structure in the training set has
    /// approximately zero total charge. Requires already-initialized
    /// les_moment_coeffs and training structures with computed descriptors.
    /// Called only when les_balance_initial_charges is explicitly enabled.
    /// @param atoms Training structures with descriptor info accessible
    void balance_les_charges(const std::vector<Atoms>& atoms,
                             const std::vector<double>& targets = {});
    bool initialize_zero_les_coeffs(const std::vector<Atoms>& atoms,
        const std::vector<double>& targets, double range, int seed);

    // ── Training overrides ─────────────────────────────────────────────
    int  get_NL_coeffs_count() const override;
    void nl_get_coeffs(double *x) const override;
    void nl_set_coeffs(const double *x) override;
    std::vector<NLParameterGroup> nl_parameter_groups() const override;
    std::vector<ConditionalLinearBlock> conditional_linear_blocks(double penalty_weight) const override;
    void conditional_linear_set_coeffs(const std::vector<int>& indices, const Eigen::VectorXd& values) override;
    bool supports_total_charge_training() const override;
    double predicted_total_charge(const Atoms &atom) const override;
    void AccumulateTotalChargeGard(Atoms &atom, AnyNeighborhoods &nbhs,
        double dLdQ, double *out_gard_accumulator) override;
    void train_init() override;
    Eigen::MatrixXd linear_sog_components(Atoms&,AnyNeighborhoods&);
    void set_sog_linear_weights(const Eigen::VectorXd& weights);
    void random_init() override;
    void AccumulateEFSCombinationGard(Atoms &atom,
        AnyNeighborhoods &nbhs,
        double dLdE,
        Eigen::Matrix<double, -1, 3> &dLdF,
        Eigen::Matrix3d &dLdS,
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &wgt_d2UdRdx,
        double *out_gard_accumulator) override;
    void AccumulateEFSCombinationGard(Atoms &atom,
        AnyNeighborhoods &nbhs,
        double dLdE,
        double *out_gard_accumulator) override;
    void AddPenaltyGard(double weight, double &loss, double *loss_gard) override;

private:
    static thread_local bool allow_internal_site_evaluation_;
    void InitSettings();
    void validate_response_parity() const;
    void sync_les_engine();
    int allocated_channels_=1;
    void accumulate_multichannel(Atoms&,AnyNeighborhoods&,double,
        const Eigen::Matrix<double,-1,3>&,const Eigen::Matrix3d&,double*);
    void accumulate_sog_gradient(Atoms&, double,
        const Eigen::Matrix<double,-1,3>&, const Eigen::Matrix3d&,
        const Eigen::VectorXd&, double*);
    Eigen::VectorXd sog_response_;
    // Helper: compute latent charges for all atoms
    void compute_les_charges(Atoms &atom, AnyNeighborhoods &nbhs, Eigen::VectorXd &qs, int channel=0);
    // Add LES energy and, when requested, direct and indirect forces.
    void append_les(Atoms &atom, AnyNeighborhoods &nbhs, bool need_forces);
    // ── Training helpers ───────────────────────────────────────────────
    // Compute energy-only LES gradients into out_gard (at les offset)
    void accumulate_les_energy_gradient(Atoms &atom, AnyNeighborhoods &nbhs,
        double dLdE, double *out_gard);

    // Buffer: per-atom descriptor values B_i[a] for gradient computation
    Eigen::MatrixXd grad_moment_vals_cache_; // [n_atoms x alpha_scalar_moments]
    // Cache for per-atom type during gradient computation
    std::vector<int> grad_atom_types_;
    // Accumulate LES contributions to SR radial coefficient gradients
    void accumulate_les_sr_gradient(Atoms &atom, AnyNeighborhoods &nbhs,
        double dLdE, const Eigen::Matrix<double, -1, 3> &dLdF,
        const Eigen::Matrix3d &dLdS, double *out_gard,
        const Eigen::VectorXd *extra_dLdq = nullptr);

    // Compose the force/stress VJP through F_indirect=-dq/dr^T dE/dq.
    void compute_indirect_force_vjp(Atoms& atom, AnyNeighborhoods& nbhs,
        const Eigen::Matrix<double, -1, 3>& dLdF,
        const Eigen::Matrix3d& dLdS);
    Eigen::VectorXd force_charge_vjp_cache_;
    Eigen::VectorXd force_moment_explicit_vjp_cache_;

    // Cache dEdq from getEFS to avoid recomputing Ewald in gradient
    Eigen::VectorXd cached_dEdq_;
    bool dEdq_valid_ = false;
};

} // namespace Potentials
