// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
namespace AL { struct GPMALAccess; }

#include "base/structures.h"
#include "descriptors/descriptors.h"
#include "utils/gputils.h"
#include <Eigen/Core>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace Potentials { class AnyPotential; }
class GPSettings;

enum class RowScalingMode {
    Uniform = 0,
    MlipCompatible = 1,
};

enum class BRowKind {
    Energy = 0,
    Force = 1,
    Stress = 2,
    Site = 3,
    Descriptor = 4,
    Unknown = 5,
};

struct BRowRef {
    int cfg_index = -1;
    int local_row = -1;
    int global_row = -1;
    BRowKind kind = BRowKind::Unknown;
    int atom_index = -1;
    int component = -1;
};

const char *to_string(RowScalingMode mode);
const char *to_string(BRowKind kind);
RowScalingMode row_scaling_mode_from_string(const std::string &value);

/// Shared settings for creating B-matrix builders.
///
/// In settings files these keys are normally used under the "builder:"
/// namespace, e.g. "builder:dtype pot_efs".
class BMatrixBuilderSettings : public ConnectSettings
{
public:
    std::string dtype = "mtm";
    std::string potfile = "potential.mtp";
    std::string pottype = "mtp";
    std::string mode = "nbh";
    bool all_coeffs = false;
    std::string row_scaling = "uniform";
    double w_energy = 1.0;
    double w_site = 0.0;
    double w_force = 0.0;
    double w_stress = 0.0;

    BMatrixBuilderSettings();
    void RegisterConsole(CMDOption::CommandLineOptions &cmdopts);
};

/// @brief Abstract interface for building B-matrix rows from atomic structures.
/// @ingroup maxvol
///
/// Each concrete builder produces row vectors of length coeff_count().
/// A single structure may contribute one row (structure-level, e.g. energy
/// component sum) or multiple rows (per-atom, e.g. scalar basis / site
/// descriptors).  The caller manages the accumulation of rows into the full
/// B matrix.
class BMatrixBuilder
{
public:
    using RowMatrix = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

    virtual ~BMatrixBuilder() = default;
    BMatrixBuilder() = default;
    BMatrixBuilder(const BMatrixBuilder &) = delete;
    BMatrixBuilder &operator=(const BMatrixBuilder &) = delete;

    /// Number of columns in each B-matrix row.
    virtual int coeff_count() const = 0;

    /// Cutoff radius for neighbor-list construction.
    virtual double cutoff() const = 0;

    /// Build B-matrix rows for one structure and append them to B.
    /// @param cfg         Input atomic configuration.
    /// @param B           B matrix (resized to accommodate new rows).
    /// @param weight      Row weight multiplier.
    /// @param pow_scale   natoms scaling exponent: row *= weight / natoms^pow.
    /// @return            Number of rows appended.
    virtual int build_rows(Atoms &cfg,
                           Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
                           double weight, double pow_scale) = 0;

    /// Metadata-producing overload. The default preserves compatibility with
    /// third-party builders and marks appended rows as Unknown.
    virtual int build_rows(Atoms &cfg,
                           Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
                           double weight, double pow_scale,
                           std::vector<BRowRef> *refs);

    /// Return the exact number of rows produced for a configuration without
    /// evaluating descriptors.  Third-party builders may keep the default -1,
    /// in which case callers fall back to the append-style API.
    virtual int row_count(const Atoms &cfg) const { return -1; }

    /// Fill a preallocated configuration-local row block.  The default adapter
    /// preserves compatibility with existing builders by building into a
    /// temporary matrix and copying once.
    virtual int build_rows_into(
        Atoms &cfg, Eigen::Ref<RowMatrix> out,
        double weight, double pow_scale,
        std::vector<BRowRef> *refs = nullptr);

    RowScalingMode row_scaling = RowScalingMode::Uniform;

    /// Create an independent copy for thread-safe parallel use.
    virtual std::unique_ptr<BMatrixBuilder> clone() const = 0;

    /// Save reconstruction parameters to stream (for embedding in state files).
    virtual void save_config(std::ostream &os) const {}

    /// Factory: reconstruct a builder from saved config + pot_file override.
    /// @param is   Input stream positioned after the builder-config tag.
    /// @param pf   Required pot file override; empty means loading fails.
    static std::unique_ptr<BMatrixBuilder>
    load_config(std::istream &is, const std::string &pf = "");
};

// ── Generic gradient-based builder (AccumulateEFSCombinationGard) ──────────

/// Builds B-matrix rows from property gradients w.r.t. ALL potential
/// coefficients (energy dE/dc, optionally forces dF/dc and stress dS/dc).
/// Works with any AnyPotential subclass that implements
/// AccumulateEFSCombinationGard / AccumulateFPGard.
class PotEFSBuilder : public BMatrixBuilder
{
public:
    /// @param pot  Any potential. Must outlive the builder.
    explicit PotEFSBuilder(Potentials::AnyPotential &pot);
    explicit PotEFSBuilder(std::unique_ptr<Potentials::AnyPotential> pot);
    ~PotEFSBuilder() override;

    int    coeff_count() const override;   // = get_NL_coeffs_count()
    double cutoff() const override;
    int    build_rows(Atoms &cfg,
                      Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
                      double weight, double pow_scale) override;
    int    build_rows(Atoms &cfg,
                      Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
                      double weight, double pow_scale,
                      std::vector<BRowRef> *refs) override;
    int    row_count(const Atoms &cfg) const override;
    std::unique_ptr<BMatrixBuilder> clone() const override;
    void save_config(std::ostream &os) const override;

    // Per-component weights (mirrors MLIP-3 PrepareMatrix):
    double w_energy = 1.0;   // weight for energy equation row (dE/dc)
    double w_force  = 0.0;   // weight for force equation rows (dF/dc)
    double w_stress = 0.0;   // weight for stress equation rows (dS/dc)
    double w_site   = 0.0;   // weight for per-atom site-energy rows (dE_i/dc)

    /// For plain PotMTP: use only linear coefficients (species + moments)
    /// rather than the full coefficient set (including radial basis params).
    /// Default true — linear coefficients are faster to fit and sufficient
    /// for active learning since they capture the same scalar basis diversity.
    /// Derived/composite models always use their full parameter gradients.
    bool linear_only = true;

    Potentials::AnyPotential &potential() const { return *pot_; }
    void set_clone_config(const std::string &pot_file,
                          const std::string &pottype,
                          const GPSettings &settings);

private:
    void init_from_potential();

    Potentials::AnyPotential *pot_;
    std::unique_ptr<Potentials::AnyPotential> owned_pot_;
    int ncols_;               // get_NL_coeffs_count() — full coefficient dimension
    std::string clone_pot_file_;
    std::string clone_pottype_;
    GPSettings clone_settings_;
    bool can_clone_ = false;

    // Internal buffers
    std::vector<double> grad_buf_;
    Eigen::Matrix<double, -1, 3> dLdF_buf_;
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> wgt_buf_;
};

// ── Generic descriptor builder (any Descriptors subclass) ──────────────────

class DescriptorBuilder : public BMatrixBuilder
{
    friend struct AL::GPMALAccess;
public:
    /// Row grouping mode (mirrors MLIP-3 equation types).
    enum Mode {
        Nbh = 0,  // per-atom rows: one row per neighborhood (default)
        Cfg = 1,  // one structure-level row: sum over all atoms
    };

    /// @param desc  A fully configured descriptor instance (must outlive).
    DescriptorBuilder(Descriptors &desc);
    /// Construct a builder that owns its descriptor and optional dependency.
    /// The optional potential keeps descriptor implementations such as MTM
    /// valid when they reference a potential loaded specifically for this builder.
    explicit DescriptorBuilder(std::unique_ptr<Descriptors> desc,
                               std::unique_ptr<Potentials::AnyPotential> pot = nullptr);
    ~DescriptorBuilder() override;
    int    coeff_count() const override;
    double cutoff() const override;
    int    build_rows(Atoms &cfg,
                      Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
                      double weight, double pow_scale) override;
    int    build_rows(Atoms &cfg,
                      Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
                      double weight, double pow_scale,
                      std::vector<BRowRef> *refs) override;
    int    row_count(const Atoms &cfg) const override;
    int    build_rows_into(
        Atoms &cfg, Eigen::Ref<RowMatrix> out,
        double weight, double pow_scale,
        std::vector<BRowRef> *refs = nullptr) override;
    std::unique_ptr<BMatrixBuilder> clone() const override;
    void save_config(std::ostream &os) const override;
    void set_config_metadata(const std::string &dtype, const std::string &mode,
                             const GPSettings &settings = GPSettings{});

    Mode mode = Nbh;

private:
    Descriptors *desc_;
    std::unique_ptr<Potentials::AnyPotential> owned_pot_;
    // Must be declared after owned_pot_: members are destroyed in reverse order,
    // and MTM descriptors may reference the owned potential.
    std::unique_ptr<Descriptors> owned_desc_;  // owns cloned descriptor for thread safety
    int ncols_;
    std::string dtype_ = "mtm";
    GPSettings settings_;
};

// ── Factory function ──────────────────────────────────────────────────────────

/// @brief Factory: create a BMatrixBuilder based on descriptor type.
/// @ingroup maxvol
///
/// Supported dtypes: mtm, pot_efs/mtp_energy, behler, bondorder, biso4.
/// @param dtype       Descriptor type string.
/// @param pot_file    Path to potential file (for mtm and pot_efs).
/// @param pottype     Potential type name (e.g. "mtp").
/// @param settings    INI settings (used for descriptor-specific subsections).
/// @param mode_str    Row mode: "nbh" (per-atom) or "cfg" (per-structure).
/// @param w_energy, w_site, w_force, w_stress  Weights for PotEFSBuilder.
/// @param all_coeffs  Use all potential coefficients (default: linear only for MTP).
/// @return            Builder instance, or nullptr on error.
std::unique_ptr<BMatrixBuilder> make_builder(
    const std::string &dtype, const std::string &pot_file, const std::string &pottype,
    GPSettings &settings, const std::string &mode_str,
    double w_energy, double w_site, double w_force, double w_stress,
    bool all_coeffs = false);

std::unique_ptr<BMatrixBuilder> make_builder(
    const BMatrixBuilderSettings &builder_settings, GPSettings &settings);
