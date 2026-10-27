// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potential.h"
#include "utils/gputils.h"
#include "reciprocal_phases.h"

namespace Potentials {
class PotMTP;

/** Periodic Gaussian latent-Ewald kernel.
 *
 * The lattice convention is the one used by Structures: real and reciprocal
 * lattice vectors are rows and r = s H.  The returned stress is the virial
 * W = -dE/d(epsilon), i.e. the quantity stored in Atoms::stress_.
 */
class LES : public ConnectSettings {
public:
    using KVectorList = std::vector<Eigen::Vector3d,
        Eigen::aligned_allocator<Eigen::Vector3d>>;

    enum class CutoffMode { Smooth, ReferenceHard };
    enum class Kernel { Coulomb, SOG };

    struct Workspace {
        bool grid_valid_=false, kernel_valid_=false;
        Eigen::Matrix3d cached_cell_=Eigen::Matrix3d::Zero();
        double cached_dl_=0., cached_taper_=0., cached_sigma_=0.;
        CutoffMode cached_cutoff_=CutoffMode::Smooth;
        Kernel cached_kernel_=Kernel::Coulomb;
        bool cached_zero_=false;
        int cached_channel_=-1, cached_channels_=0;
        std::vector<double> cached_weights_, cached_widths_, inverse_widths_;
        KVectorList cached_grid_;
        std::vector<std::array<int,3>> k_indices_;
        std::array<int,3> phase_bounds_{};
        std::vector<double> grid_factors_, grid_taper_, grid_taper_derivative_;
        std::vector<double> cached_values_, cached_derivatives_;
        KVectorList kvecs;
        std::vector<double> kvec_factors,k_kernel,k_kernel_derivative;
        ReciprocalPhases phases;
        std::vector<double> cosine,sine,cosine_directional,sine_directional,weighted_k;
    };

    Kernel kernel = Kernel::Coulomb;
    // SOG Fourier multiplier: sum w_m exp(-k^2/s_m^2), s_m=exp(u_m)/A.
    // For dimensionless latent variables, weights have units eV A^3.
    std::vector<double> sog_weights;
    std::vector<double> sog_log_bandwidths;
    bool sog_include_zero = true;
    int sog_channels = 1;
    int sog_channel = 0; // Legacy API selection; explicit workspaces take a channel argument.
    int channel_terms() const {return sog_channels>0 ? sog_weights.size()/sog_channels : 0;}
    struct ChannelScope {
        LES& model; int previous;
        ChannelScope(LES& m,int c):model(m),previous(m.sog_channel){m.sog_channel=c;}
        ~ChannelScope(){model.sog_channel=previous;}
        ChannelScope(const ChannelScope&)=delete;
    };
    // Execution settings, deliberately not part of the physical model file.
    bool use_recurrence = true;
    bool use_cache = true;

    struct Evaluation {
        double energy = 0.0;
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> forces;
        Eigen::Matrix3d virial = Eigen::Matrix3d::Zero();
        // Atom-major symmetric xx, yy, zz, xy, xz, yz; sums to virial.
        Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> per_atom_virial;
        Eigen::VectorXd dEdq;
        Eigen::MatrixXd charge_hessian;
        Eigen::VectorXd per_atom_energy;
    };

    static constexpr int format_version = 2;
    struct ChannelEvaluation {
        double energy=0.;
        Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor> forces;
        Eigen::Matrix3d virial=Eigen::Matrix3d::Zero();
        Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> per_atom_virial;
        Eigen::MatrixXd dEdq; // N x C; no cross-channel coupling.
        Eigen::VectorXd per_atom_energy;
    };
    ChannelEvaluation evaluate_channels(const Atoms& atom,const Eigen::MatrixXd& q);
    struct KernelComponents {
        // Rows: total E, atom-major Cartesian F, row-major 3x3 virial.
        Eigen::MatrixXd efs, dEdq; // columns: channel-major Gaussian terms
    };
    KernelComponents kernel_components(const Atoms&,const Eigen::MatrixXd& q);
    Eigen::MatrixXd apply_charge_hessian_channels(const Atoms& atom,const Eigen::MatrixXd& v);
    static constexpr double coulomb_constant = 14.3996454784255; // eV A / e^2
    double self_coefficient(int channel=0) const;

    double lr_sigma = 0.5;       // Gaussian width, A
    double dl = 2.0;             // k_max = 2*pi/dl, A
    double taper_fraction = 0.1; // final fraction of k sphere, C2 taper
    CutoffMode cutoff_mode = CutoffMode::Smooth;
    bool remove_self_interaction = true;

    // Last generated grid, exposed for diagnostics and legacy callers.
    KVectorList kvecs;
    std::vector<double> kvec_factors;
    std::vector<double> k_kernel;
    std::vector<double> k_kernel_derivative;

    static const char* cutoff_mode_name(CutoffMode mode);
    static CutoffMode parse_cutoff_mode(const std::string& value);

    void load_coulomb_definition(std::istream& in);
    void save_coulomb_definition(std::ostream& out) const;
    static void reject_obsolete_parameter(const std::string& name);
    static void reject_obsolete_settings(const GPSettings& settings);
    void validate() const;
    static void validate_periodic_geometry(const Atoms& atom);
    void generate_k_vectors(const Atoms& atom, KVectorList& out, bool update_kernel = true);
    // Valid after generate_k_vectors; preserves the CPU cutoff conventions
    // when preparing a backend-neutral GPU grid.
    const std::vector<double>& reciprocal_taper() const { return default_workspace_.grid_taper_; }
    const std::vector<double>& reciprocal_taper_derivative() const { return default_workspace_.grid_taper_derivative_; }
    // Publish diagnostics from a serial owner wrapper without copying its cache.
    // Explicit workspace evaluation itself never calls this mutating helper.
    void publish_legacy_diagnostics(const Workspace&);
    Evaluation evaluate(const Atoms& atom, const Eigen::VectorXd& q,
                        bool need_charge_hessian = false);
    struct CartesianResponse {
        SiteCoordinateMatrix force_directional;
        Eigen::VectorXd dEdq, dEdq_directional;
    };
    // Fixed-cell Cartesian differential, allowing simultaneous latent changes.
    CartesianResponse evaluate_cartesian_directional(const Atoms& atom,
        const Eigen::VectorXd& q, const Eigen::VectorXd& q_directional,
        const SiteCoordinateMatrix& velocity);
    struct NativeResponse {
        SiteCoordinateMatrix forces, force_directional;
        Eigen::Matrix3d virial = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d virial_directional = Eigen::Matrix3d::Zero();
        Eigen::VectorXd dEdq, dEdq_directional;
    };
    NativeResponse evaluate_native_directional(const Atoms& atom,
        const Eigen::VectorXd& q, const Eigen::VectorXd& q_directional,
        const SiteCoordinateMatrix& velocity, const Eigen::Matrix3d& cell_direction);
    Eigen::VectorXd apply_charge_hessian(const Atoms& atom, const Eigen::VectorXd& v);
    void add_latent_ewald_efs(Atoms& atom, AnyNeighborhoods& nbhs,
                             Eigen::VectorXd& q_latent,
                             Eigen::VectorXd* dEdq_out = nullptr);

    // VJP of the fixed-q reciprocal force and analytic virial with respect
    // to q.  Indirect descriptor-force terms are composed by PotMTP_LES.
    void compute_les_dLdq_from_forces(const Atoms& atom,
        const Eigen::VectorXd& q_latent,
        const Eigen::Matrix<double, -1, 3>& dLdF,
        const Eigen::Matrix3d& dLdS,
        Eigen::VectorXd& dLdq_out);

    // Parameters are weights followed by log bandwidths. The last input
    // contracts the descriptor response: -L_F:dq/dr - L_W:dq/depsilon.
    void accumulate_kernel_vjp(const Atoms& atom, const Eigen::VectorXd& q,
        double dLdE, const Eigen::Matrix<double, -1, 3>& dLdF,
        const Eigen::Matrix3d& dLdW, const Eigen::VectorXd& response,
        double* gradient);

    // Compatibility wrapper used by PotMTP_M.
    void compute_les_force_stress_grad(const Atoms& atom,
        const Eigen::VectorXd& q_latent,
        const Eigen::MatrixXd& descriptors,
        const std::vector<int>& atom_types,
        int species_count, int alpha_scalar_moments,
        const Eigen::Matrix<double, -1, 3>& dLdF,
        const Eigen::Matrix3d& dLdS,
        double* out_gard);

    /// Complex generalized dipole in fractional lattice components.  The
    /// phase must be retained until after differentiation/dephasing when this
    /// quantity is used to construct periodic Born effective charges.
    static Eigen::Vector3cd compute_periodic_dipole_fractional(
        const Atoms& atom, const Eigen::VectorXd& q);

    void compute_bec(const Atoms& atom, const Eigen::VectorXd& q_latent,
                     std::vector<Eigen::Matrix3d>& bec_out,
                     bool remove_mean = true,
                     double epsilon_factor = 1.0) const;


    // Explicit workspaces are caller-owned. These paths never update owner
    // diagnostics, caches or the legacy selected channel.
    void generate_k_vectors(const Atoms&,KVectorList&,bool,Workspace&,int channel=0) const;
    Evaluation evaluate(const Atoms&,const Eigen::VectorXd&,Workspace&,int channel=0) const;
    Evaluation evaluate(const Atoms&,const Eigen::VectorXd&,bool,Workspace&,int channel=0) const;
    ChannelEvaluation evaluate_channels(const Atoms&,const Eigen::MatrixXd&,Workspace&) const;
    KernelComponents kernel_components(const Atoms&,const Eigen::MatrixXd&,Workspace&) const;
    Eigen::VectorXd apply_charge_hessian(const Atoms&,const Eigen::VectorXd&,Workspace&,int channel=0) const;
    Eigen::MatrixXd apply_charge_hessian_channels(const Atoms&,const Eigen::MatrixXd&,Workspace&) const;
    CartesianResponse evaluate_cartesian_directional(const Atoms&,const Eigen::VectorXd&,
        const Eigen::VectorXd&,const SiteCoordinateMatrix&,Workspace&,int channel=0) const;
    NativeResponse evaluate_native_directional(const Atoms&,const Eigen::VectorXd&,
        const Eigen::VectorXd&,const SiteCoordinateMatrix&,const Eigen::Matrix3d&,Workspace&,int channel=0) const;
    void add_latent_ewald_efs(Atoms&,AnyNeighborhoods&,Eigen::VectorXd&,Eigen::VectorXd*,Workspace&,int channel=0) const;
    void compute_les_dLdq_from_forces(const Atoms&,const Eigen::VectorXd&,
        const Eigen::Matrix<double,-1,3>&,const Eigen::Matrix3d&,Eigen::VectorXd&,Workspace&,int channel=0) const;
    void accumulate_kernel_vjp(const Atoms&,const Eigen::VectorXd&,double,
        const Eigen::Matrix<double,-1,3>&,const Eigen::Matrix3d&,const Eigen::VectorXd&,
        double*,Workspace&,int channel=0) const;
    void compute_les_force_stress_grad(const Atoms&,const Eigen::VectorXd&,const Eigen::MatrixXd&,
        const std::vector<int>&,int,int,const Eigen::Matrix<double,-1,3>&,
        const Eigen::Matrix3d&,double,double*,Workspace&,int channel=0) const;

private:
    Workspace default_workspace_;
    void publish_workspace();
    void validate(int channel) const;
    const ReciprocalPhases& make_phases(const Atoms& atom,Workspace&) const;
    double cutoff_weight(double k, double& derivative_wrt_k2) const;
    void kernel_value(double k2,double& value,double& derivative,const Workspace&,int channel) const;
};

// Full latent response, including mixed position/charge terms and charge-head
// curvature. The moment graph is shared by ordinary MTP and MTP-M heads.
Eigen::VectorXd latent_cartesian_hvp(PotMTP& basis, LES& engine,
    const Atoms& atom, AnyNeighborhoods& neighborhoods,
    const Eigen::VectorXd& direction, const double* species,
    const double* moments, int channels);

using LocalCartesianHVP = std::function<void(Neighborhood&,const SiteCoordinateMatrix&,SiteCoordinateMatrix&)>;
// The assembly formulas are independent of how local moment/head scratch is
// owned. Explicit callers provide their kernels and the final EFS/HVP actions.
struct LatentCurvatureCallbacks {
    std::function<void(Neighborhood&,int,const SiteCoordinateMatrix&,double&,
        SiteCoordinateMatrix&,SiteCoordinateMatrix&)> charge_hvp;
    std::function<void(Neighborhood&,int,const SiteCoordinateMatrix&,const SiteCoordinateMatrix&,
        double&,SiteCoordinateMatrix&,SiteCoordinateMatrix&,SiteCoordinateMatrix&)> charge_second;
    std::function<void(const Eigen::VectorXd&,Eigen::VectorXd&)> whole_hvp;
    std::function<void()> finish_efs;
};
void assemble_latent_native_hessian(PotMTP& basis, AnyPotential& potential, LES& engine,
    Atoms& atom, AnyNeighborhoods& neighborhoods, const double* species,
    const double* moments, int channels, bool long_range, const LocalCartesianHVP& short_range,
    HessianMatrix& hessian, Eigen::MatrixXd& cell, Eigen::MatrixXd& mixed);
void assemble_cartesian_hessian(AnyPotential& potential, Atoms& atom,
    AnyNeighborhoods& neighborhoods, HessianMatrix& hessian);
void assemble_cartesian_hessian(AnyPotential&,Atoms&,AnyNeighborhoods&,HessianMatrix&,
    const LatentCurvatureCallbacks&);
void assemble_latent_native_hessian(PotMTP&,AnyPotential&,LES&,Atoms&,AnyNeighborhoods&,
    const double*,const double*,int,bool,const LocalCartesianHVP&,HessianMatrix&,
    Eigen::MatrixXd&,Eigen::MatrixXd&,const LatentCurvatureCallbacks&,LES::Workspace&);
void assemble_requested_hessian(AnyPotential& potential, Atoms& atom,
    AnyNeighborhoods& neighborhoods, const std::vector<std::pair<int,int>>& indices,
    HessianSparse& hessian);
void assemble_cartesian_diagonal(AnyPotential& potential, Atoms& atom,
    AnyNeighborhoods& neighborhoods, std::vector<double>& diagonal);


} // namespace Potentials
