// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "maxvol/b_matrix_builder.h"
#include "potentials/potential_mtp.h"
#include "base/neighborhood.h"
#include <cmath>
#include "descriptors/descriptor_MTM.h"
#include "descriptors/descriptor_Behler.h"
#include "descriptors/descriptor_BondOrder.h"
#include "descriptors/descriptor_BispectrumSO4.h"
#include "potentials/potential_mtp.h"
#include "potentials/potential_wrapper.h"
#include "utils/gputils.h"
#include <fmt/format.h>
#include <stdexcept>

const char *to_string(RowScalingMode mode)
{
    return mode == RowScalingMode::MlipCompatible ? "mlip" : "uniform";
}

const char *to_string(BRowKind kind)
{
    switch (kind) {
    case BRowKind::Energy: return "energy";
    case BRowKind::Force: return "force";
    case BRowKind::Stress: return "stress";
    case BRowKind::Site: return "site";
    case BRowKind::Descriptor: return "descriptor";
    default: return "unknown";
    }
}

RowScalingMode row_scaling_mode_from_string(const std::string &value)
{
    if (value == "mlip" || value == "mlip-compatible" || value == "mlip_compatible")
        return RowScalingMode::MlipCompatible;
    return RowScalingMode::Uniform;
}

BMatrixBuilderSettings::BMatrixBuilderSettings()
{
    Connect(dtype, "dtype", "descriptor type: mtm, pot_efs, behler, bondorder, biso4");
    Connect(potfile, "potfile", "potential file");
    Connect(pottype, "pottype", "potential type");
    Connect(mode, "mode", "row mode: nbh (per-atom) | cfg (per-structure)");
    Connect(all_coeffs, "all-coeffs", "use all potential coefficients");
    Connect(row_scaling, "row-scaling", "row scaling: uniform | mlip");
    Connect(w_energy, "w-energy", "energy row weight for pot_efs");
    Connect(w_site, "w-site", "site-energy row weight for pot_efs");
    Connect(w_force, "w-force", "force row weight for pot_efs");
    Connect(w_stress, "w-stress", "stress row weight for pot_efs");
}

void BMatrixBuilderSettings::RegisterConsole(CMDOption::CommandLineOptions &cmdopts)
{
    for (const auto &name : {"potfile", "pottype", "dtype", "mode", "all-coeffs",
                             "row-scaling", "w-energy", "w-site", "w-force", "w-stress"})
        BindConsole(cmdopts, name);
}

int BMatrixBuilder::build_rows(
    Atoms &cfg, Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
    double weight, double pow_scale, std::vector<BRowRef> *refs)
{
    const int old_rows = static_cast<int>(B.rows());
    const int count = build_rows(cfg, B, weight, pow_scale);
    if (refs) {
        refs->reserve(refs->size() + static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            BRowRef ref;
            ref.local_row = i;
            ref.global_row = old_rows + i;
            refs->push_back(ref);
        }
    }
    return count;
}

int BMatrixBuilder::build_rows_into(
    Atoms &cfg, Eigen::Ref<RowMatrix> out,
    double weight, double pow_scale, std::vector<BRowRef> *refs)
{
    RowMatrix tmp;
    std::vector<BRowRef> tmp_refs;
    const int count = build_rows(
        cfg, tmp, weight, pow_scale, refs ? &tmp_refs : nullptr);
    if (count < 0 || out.cols() != coeff_count() || out.rows() < count)
        throw std::invalid_argument("BMatrixBuilder::build_rows_into block size mismatch");
    if (count > 0)
        out.topRows(count) = tmp.topRows(count);
    if (refs)
        refs->insert(refs->end(), tmp_refs.begin(), tmp_refs.end());
    return count;
}


// ═══════════════════════════════════════════════════════════════════════════
// PotEFSBuilder — full coefficient gradients via AnyPotential base API
// ═══════════════════════════════════════════════════════════════════════════

PotEFSBuilder::PotEFSBuilder(Potentials::AnyPotential &pot)
    : pot_(&pot)
{
    init_from_potential();
}

PotEFSBuilder::PotEFSBuilder(std::unique_ptr<Potentials::AnyPotential> pot)
    : pot_(pot.get()), owned_pot_(std::move(pot))
{
    init_from_potential();
}

PotEFSBuilder::~PotEFSBuilder() = default;

void PotEFSBuilder::init_from_potential()
{
    if (pot_) pot_->train_init();
    ncols_ = pot_ ? pot_->get_NL_coeffs_count() : 0;
    grad_buf_.resize(ncols_);
}

void PotEFSBuilder::set_clone_config(const std::string &pot_file,
                                     const std::string &pottype,
                                     const GPSettings &settings)
{
    clone_pot_file_ = pot_file;
    clone_pottype_ = pottype;
    clone_settings_ = settings;
    can_clone_ = !clone_pottype_.empty() ||
                 !clone_settings_["pottype"].empty();
}

int PotEFSBuilder::coeff_count() const
{
    if (linear_only && pot_ && typeid(*pot_) == typeid(Potentials::PotMTP)) {
        auto *mtp = dynamic_cast<const Potentials::PotMTP *>(pot_);
        if (mtp) return mtp->get_lin_coeff_size();
    }
    return ncols_;
}

double PotEFSBuilder::cutoff() const { return pot_->cutoff; }

int PotEFSBuilder::row_count(const Atoms &cfg) const
{
    const int natoms = cfg.size();
    int count = 0;
    if (w_energy != 0.0) count += 1;
    if (w_force != 0.0) count += 3 * natoms;
    if (w_stress != 0.0) count += 9;
    if (w_site != 0.0) count += natoms;
    return count;
}

int PotEFSBuilder::build_rows(
    Atoms &cfg, Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
    double weight, double pow_scale)
{
    return build_rows(cfg, B, weight, pow_scale, nullptr);
}

int PotEFSBuilder::build_rows(
    Atoms &cfg, Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
    double weight, double pow_scale, std::vector<BRowRef> *refs)
{
    if (w_site != 0.0 && !pot_->supports_site_parameter_gradients())
        GPERROR("MaxVol site rows are unavailable for this composite/long-range potential; use w-site 0 and E/F/S rows");
    Neighborhoods nbhs(cfg, cutoff());
    int natoms = cfg.size();
    double scale_n = weight / std::pow(static_cast<double>(natoms), pow_scale);
    const double local_scale = row_scaling == RowScalingMode::MlipCompatible
                             ? weight : scale_n;
    int total_rows = 0;
    int cur_rows = static_cast<int>(B.rows());
    int ncols = coeff_count();
    auto add_ref = [&](BRowKind kind, int atom, int component) {
        if (!refs) return;
        BRowRef ref;
        ref.local_row = total_rows;
        ref.global_row = cur_rows + total_rows;
        ref.kind = kind;
        ref.atom_index = atom;
        ref.component = component;
        refs->push_back(ref);
    };

    // Base-MTP linear routines are not virtual. Derived models must use their
    // complete parameter Jacobian, including nonlinear and long-range terms.
    auto *mtp = linear_only && typeid(*pot_) == typeid(Potentials::PotMTP)
        ? dynamic_cast<Potentials::PotMTP *>(pot_)
        : nullptr;

    // Count rows needed
    int max_rows = 0;
    if (w_energy != 0.0)                         max_rows += 1;
    if (w_force  != 0.0)                         max_rows += 3 * natoms;
    if (w_stress != 0.0)                         max_rows += 9;
    if (w_site   != 0.0)                         max_rows += natoms;
    if (max_rows == 0) return 0;
    B.conservativeResize(cur_rows + max_rows, ncols);

    auto accumulate_efs = [&](double dLdE,
                              Eigen::Matrix<double, -1, 3> &dLdF,
                              Eigen::Matrix3d &dLdS) {
        std::fill(grad_buf_.begin(), grad_buf_.end(), 0.0);
        if (mtp) {
            mtp->AccumulateEFSCombinationGardLonly(cfg, nbhs, dLdE,
                                                     dLdF, dLdS,
                                                     grad_buf_.data());
        } else {
            wgt_buf_.resize(natoms, 3);
            pot_->AccumulateEFSCombinationGard(cfg, nbhs, dLdE, dLdF,
                                               dLdS, wgt_buf_,
                                               grad_buf_.data());
        }
    };

    // ── Energy equation (structure-level) ─────────────────────────────────
    if (mtp && (w_force != 0.0 || w_stress != 0.0)) {
        // The complete linear E/F/S Jacobian needs one neighborhood pass.
        // Repeating a full reverse pass for each force component costs O(N²).
        std::vector<double> energy;
        Eigen::Tensor<double, 3, Eigen::RowMajor> forces, stress;
        mtp->CalcEFSCmpt_R(cfg, nbhs, energy, forces, stress);
        if (w_energy != 0.0) {
            B.row(cur_rows + total_rows) = w_energy * scale_n *
                Eigen::Map<const Eigen::RowVectorXd>(energy.data(), ncols);
            add_ref(BRowKind::Energy, -1, -1);
            ++total_rows;
        }
        if (w_force != 0.0) {
            for (int i = 0; i < natoms; ++i)
                for (int a = 0; a < 3; ++a) {
                    B.row(cur_rows + total_rows) = w_force * local_scale *
                        Eigen::Map<const Eigen::RowVectorXd>(&forces(i, a, 0), ncols);
                    add_ref(BRowKind::Force, i, a);
                    ++total_rows;
                }
        }
        if (w_stress != 0.0) {
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    B.row(cur_rows + total_rows) = w_stress * scale_n *
                        Eigen::Map<const Eigen::RowVectorXd>(&stress(a, b, 0), ncols);
                    add_ref(BRowKind::Stress, -1, 3 * a + b);
                    ++total_rows;
                }
        }
    }
    else if (w_energy != 0.0 && w_force == 0.0 && w_stress == 0.0)
    {
        std::fill(grad_buf_.begin(), grad_buf_.end(), 0.0);
        if (mtp) {
            // Linear MTP energy gradients are exactly the summed energy
            // components dE/dc_l; use the forward-only path to avoid derivative
            // scratch state in the L-only accumulator.
            std::vector<double> energy_cmpts;
            mtp->CalcECmpt(cfg, nbhs, energy_cmpts);
            for (int k = 0; k < ncols; ++k)
                grad_buf_[k] = w_energy * energy_cmpts[k];
        } else {
            pot_->AccumulateEFSCombinationGard(cfg, nbhs, w_energy, grad_buf_.data());
        }
        for (int k = 0; k < ncols; ++k)
            B(cur_rows, k) = scale_n * grad_buf_[k];
        add_ref(BRowKind::Energy, -1, -1);
        total_rows = 1;
    }
    else if (w_energy != 0.0 || w_force != 0.0 || w_stress != 0.0)
    {
        dLdF_buf_.resize(natoms, 3);
        dLdF_buf_.setZero();
        Eigen::Matrix3d dLdS = Eigen::Matrix3d::Zero();
        if (w_energy != 0.0) {
            accumulate_efs(1.0, dLdF_buf_, dLdS);
            for (int k = 0; k < ncols; ++k)
                B(cur_rows, k) = w_energy * scale_n * grad_buf_[k];
            add_ref(BRowKind::Energy, -1, -1);
            total_rows += 1;
        }
        if (w_force != 0.0) {
            for (int i = 0; i < natoms; ++i) {
                for (int a = 0; a < 3; ++a) {
                    dLdF_buf_.setZero();
                    dLdS.setZero();
                    dLdF_buf_(i, a) = 1.0;
                    accumulate_efs(0.0, dLdF_buf_, dLdS);
                    for (int k = 0; k < ncols; ++k)
                        B(cur_rows + total_rows, k) = w_force * local_scale * grad_buf_[k];
                    add_ref(BRowKind::Force, i, a);
                    ++total_rows;
                }
            }
        }
        if (w_stress != 0.0) {
            dLdF_buf_.setZero();
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    dLdS.setZero();
                    dLdS(a, b) = 1.0;
                    accumulate_efs(0.0, dLdF_buf_, dLdS);
                    for (int k = 0; k < ncols; ++k)
                        B(cur_rows + total_rows, k) = w_stress * scale_n * grad_buf_[k];
                    add_ref(BRowKind::Stress, -1, 3 * a + b);
                    ++total_rows;
                }
            }
        }
    }

    // ── Site-energy equations (per-atom) ──────────────────────────────────
    if (w_site != 0.0)
    {
        for (int i = 0; i < nbhs.size(); ++i) {
            std::fill(grad_buf_.begin(), grad_buf_.end(), 0.0);
            if (mtp)
                mtp->AccumulateFPGardLonly(nbhs[i], grad_buf_.data(), w_site, nullptr);
            else
                pot_->AccumulateFPGard(nbhs[i], grad_buf_.data(), w_site, nullptr);
            for (int k = 0; k < ncols; ++k)
                B(cur_rows + total_rows, k) = local_scale * grad_buf_[k];
            add_ref(BRowKind::Site, i, -1);
            ++total_rows;
        }
    }

    if (total_rows < max_rows)
        B.conservativeResize(cur_rows + total_rows, ncols);

    return total_rows;
}

// ═══════════════════════════════════════════════════════════════════════════
// DescriptorBuilder — generic, any Descriptors subclass
// ═══════════════════════════════════════════════════════════════════════════

DescriptorBuilder::DescriptorBuilder(Descriptors &desc)
    : desc_(&desc), ncols_(desc.feature_size()) {}

DescriptorBuilder::DescriptorBuilder(std::unique_ptr<Descriptors> desc,
                                     std::unique_ptr<Potentials::AnyPotential> pot)
    : desc_(desc.get()),
      owned_desc_(std::move(desc)),
      owned_pot_(std::move(pot)),
      ncols_(desc_->feature_size()) {}

DescriptorBuilder::~DescriptorBuilder() = default;

int DescriptorBuilder::coeff_count() const { return ncols_; }
double DescriptorBuilder::cutoff() const { return desc_->cutoff_; }

int DescriptorBuilder::row_count(const Atoms &cfg) const
{
    return mode == Cfg ? 1 : cfg.size();
}

int DescriptorBuilder::build_rows(
    Atoms &cfg, Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
    double weight, double pow_scale)
{
    return build_rows(cfg, B, weight, pow_scale, nullptr);
}

int DescriptorBuilder::build_rows(
    Atoms &cfg, Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
    double weight, double pow_scale, std::vector<BRowRef> *refs)
{
    double rc = cutoff();
    Neighborhoods nbhs(cfg, rc);
    int natoms = cfg.size();
    double scale = weight / std::pow(static_cast<double>(natoms), pow_scale);

    int cur_rows = static_cast<int>(B.rows());

    if (mode == Cfg) {
        // Structure-level: sum all per-atom descriptors → 1 row
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_new(cur_rows + 1, ncols_);
        if (cur_rows > 0)
            B_new.topRows(cur_rows) = B;
        B.swap(B_new);
        for (int j = 0; j < ncols_; ++j)
            B(cur_rows, j) = 0.0;

        for (int i = 0; i < natoms; ++i) {
            Eigen::VectorXd out;
            desc_->compute(nbhs[i], out);
            for (int j = 0; j < ncols_; ++j)
                B(cur_rows, j) += scale * out[j];
        }
        if (refs) {
            BRowRef ref;
            ref.local_row = 0;
            ref.global_row = cur_rows;
            ref.kind = BRowKind::Descriptor;
            refs->push_back(ref);
        }
        return 1;
    }

    // Nbh mode (default): one row per atom
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_new(cur_rows + natoms, ncols_);
    if (cur_rows > 0)
        B_new.topRows(cur_rows) = B;
    B.swap(B_new);

    for (int i = 0; i < natoms; ++i) {
        Eigen::VectorXd out;
        desc_->compute(nbhs[i], out);
        for (int j = 0; j < ncols_; ++j)
            B(cur_rows + i, j) = scale * out[j];
        if (refs) {
            BRowRef ref;
            ref.local_row = i;
            ref.global_row = cur_rows + i;
            ref.kind = BRowKind::Descriptor;
            ref.atom_index = i;
            refs->push_back(ref);
        }
    }

    return natoms;
}

int DescriptorBuilder::build_rows_into(
    Atoms &cfg, Eigen::Ref<RowMatrix> out,
    double weight, double pow_scale, std::vector<BRowRef> *refs)
{
    const int expected = row_count(cfg);
    if (out.rows() < expected || out.cols() != ncols_)
        throw std::invalid_argument("DescriptorBuilder::build_rows_into block size mismatch");

    const int natoms = cfg.size();
    const double scale =
        weight / std::pow(static_cast<double>(natoms), pow_scale);
    Neighborhoods nbhs(cfg, cutoff());

    if (mode == Cfg) {
        out.row(0).setZero();
        for (int i = 0; i < natoms; ++i) {
            Eigen::VectorXd values;
            desc_->compute(nbhs[i], values);
            out.row(0).noalias() += scale * values.transpose();
        }
        if (refs) {
            BRowRef ref;
            ref.local_row = 0;
            ref.global_row = 0;
            ref.kind = BRowKind::Descriptor;
            refs->push_back(ref);
        }
        return 1;
    }

    for (int i = 0; i < natoms; ++i) {
        Eigen::VectorXd values;
        desc_->compute(nbhs[i], values);
        out.row(i) = scale * values.transpose();
        if (refs) {
            BRowRef ref;
            ref.local_row = i;
            ref.global_row = i;
            ref.kind = BRowKind::Descriptor;
            ref.atom_index = i;
            refs->push_back(ref);
        }
    }
    return natoms;
}

// ═══════════════════════════════════════════════════════════════════════════
// clone() — thread-safe copy for OpenMP parallel build_rows
// ═══════════════════════════════════════════════════════════════════════════

static std::unique_ptr<BMatrixBuilder>
make_descriptor_builder_from_settings(const std::string &dtype,
                                      const std::string &mode_str,
                                      GPSettings settings,
                                      const std::string &pf_override);

std::unique_ptr<BMatrixBuilder> PotEFSBuilder::clone() const
{
    if (!can_clone_) return nullptr;

    try {
        GPSettings settings = clone_settings_;
        if (!clone_pot_file_.empty())
            settings.Modify("potfile", clone_pot_file_);
        if (!clone_pottype_.empty())
            settings.Modify("pottype", clone_pottype_);

        Potentials::potential_wrapper wrapper(settings);
        if (!wrapper.pot) return nullptr;

        std::unique_ptr<Potentials::AnyPotential> owned(wrapper.pot);
        wrapper.pot = nullptr;
        auto cp = std::make_unique<PotEFSBuilder>(std::move(owned));
        cp->w_energy = w_energy;
        cp->w_force = w_force;
        cp->w_stress = w_stress;
        cp->w_site = w_site;
        cp->linear_only = linear_only;
        cp->row_scaling = row_scaling;
        cp->set_clone_config(clone_pot_file_, clone_pottype_, clone_settings_);
        return cp;
    } catch (...) {
        return nullptr;
    }
}

std::unique_ptr<BMatrixBuilder> DescriptorBuilder::clone() const
{
    if (dtype_ == "behler" || dtype_ == "bondorder" || dtype_ == "biso4") {
        auto cp = make_descriptor_builder_from_settings(
            dtype_, (mode == Cfg) ? "cfg" : "nbh", settings_, "");
        if (cp) cp->row_scaling = row_scaling;
        return cp;
    }

    auto cloned_desc = desc_->clone();
    if (!cloned_desc)
        return nullptr;

    auto *cp = new DescriptorBuilder(std::move(cloned_desc));
    cp->mode = mode;
    cp->dtype_ = dtype_;
    cp->settings_ = settings_;
    cp->row_scaling = row_scaling;
    return std::unique_ptr<BMatrixBuilder>(cp);
}

void DescriptorBuilder::set_config_metadata(const std::string &dtype,
                                            const std::string &mode_str,
                                            const GPSettings &settings)
{
    dtype_ = dtype;
    mode = (mode_str == "cfg") ? Cfg : Nbh;
    settings_ = settings;
}


// ═══════════════════════════════════════════════════════════════════════════
// Builder serialization (for state-file embedded reconstruction)
// ═══════════════════════════════════════════════════════════════════════════

static void bm_wr(std::ostream &os, const std::string &s) {
    int n = static_cast<int>(s.size());
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    if (n > 0) os.write(s.data(), n);
}
static std::string bm_rd(std::istream &is) {
    int n = 0; is.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n < 0 || n > 1 << 20) return {};
    std::string s(n, '\0'); is.read(s.data(), n);
    return is ? s : std::string{};
}
static void bm_wr_settings(std::ostream &os, const GPSettings &settings) {
    int n = static_cast<int>(settings.settings.size());
    os.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto &kv : settings.settings) {
        bm_wr(os, kv.first);
        bm_wr(os, kv.second);
    }
}
static GPSettings bm_rd_settings(std::istream &is) {
    int n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!is || n < 0 || n > 100000) {
        is.setstate(std::ios::failbit);
        return {};
    }
    GPSettings settings;
    for (int i = 0; i < n; ++i) {
        std::string key = bm_rd(is);
        std::string val = bm_rd(is);
        if (!is || key.empty()) {
            is.setstate(std::ios::failbit);
            return {};
        }
        settings.settings.emplace(std::move(key), std::move(val));
    }
    return settings;
}

static constexpr const char *kRowScalingSetting =
    "__bmatrix_row_scaling";

static RowScalingMode take_row_scaling_setting(GPSettings &settings)
{
    auto it = settings.settings.find(kRowScalingSetting);
    if (it == settings.settings.end())
        return RowScalingMode::Uniform;
    const RowScalingMode mode = row_scaling_mode_from_string(it->second);
    settings.settings.erase(it);
    return mode;
}

static std::unique_ptr<BMatrixBuilder>
make_descriptor_builder_from_settings(const std::string &dtype,
                                      const std::string &mode_str,
                                      GPSettings settings,
                                      const std::string &pf_override)
{
    const RowScalingMode row_scaling = take_row_scaling_setting(settings);
    std::unique_ptr<Descriptors> desc;
    std::unique_ptr<Potentials::AnyPotential> owned_pot;

    if (dtype == "mtm" || dtype == "mtp_scalar" || dtype == "mtp") {
        if (pf_override.empty()) return nullptr;
        auto mtp = std::make_unique<Potentials::PotMTP>(pf_override);
        desc = std::make_unique<DescriptorMTM>(*mtp);
        owned_pot = std::move(mtp);
    } else if (dtype == "behler") {
        desc = std::make_unique<DescriptorBehler>(
            settings.ExtractSubSettings("behler"));
    } else if (dtype == "bondorder") {
        desc = std::make_unique<DescriptorBondOrder>(
            settings.ExtractSubSettings("bondorder"));
    } else if (dtype == "biso4") {
        desc = std::make_unique<Descriptor_BispectrumSO4>(
            settings.ExtractSubSettings("biso4"));
    } else {
        return nullptr;
    }

    auto *db = new DescriptorBuilder(std::move(desc), std::move(owned_pot));
    db->set_config_metadata(dtype, mode_str, settings);
    db->row_scaling = row_scaling;
    return std::unique_ptr<BMatrixBuilder>(db);
}

void DescriptorBuilder::save_config(std::ostream &os) const {
    bm_wr(os, dtype_);                            // dtype
    bm_wr(os, (mode == Cfg) ? "cfg" : "nbh");     // mode
    GPSettings settings = settings_;
    settings.settings.erase(kRowScalingSetting);
    settings.settings.emplace(kRowScalingSetting, to_string(row_scaling));
    bm_wr_settings(os, settings);
}

void PotEFSBuilder::save_config(std::ostream &os) const {
    bm_wr(os, "pot_efs");  bm_wr(os, "cfg");
    os.write(reinterpret_cast<const char*>(&w_energy), sizeof(w_energy));
    os.write(reinterpret_cast<const char*>(&w_site),   sizeof(w_site));
    os.write(reinterpret_cast<const char*>(&w_force),  sizeof(w_force));
    os.write(reinterpret_cast<const char*>(&w_stress), sizeof(w_stress));
    char flags = linear_only ? 1 : 0;
    if (row_scaling == RowScalingMode::MlipCompatible)
        flags |= 2;
    os.write(&flags, 1);
}

std::unique_ptr<BMatrixBuilder>
BMatrixBuilder::load_config(std::istream &is, const std::string &pf_override)
{
    std::string dtype = bm_rd(is), mode_str = bm_rd(is);
    if (!is || dtype.empty() || mode_str.empty()) return nullptr;
    if (dtype == "mtm" || dtype == "mtp_scalar" || dtype == "mtp") {
        GPSettings settings = bm_rd_settings(is);
        if (!is) return nullptr;
        return make_descriptor_builder_from_settings(dtype, mode_str,
                                                     std::move(settings),
                                                     pf_override);
    }
    if (dtype == "behler" || dtype == "bondorder" || dtype == "biso4") {
        GPSettings settings = bm_rd_settings(is);
        if (!is) return nullptr;
        return make_descriptor_builder_from_settings(dtype, mode_str,
                                                     std::move(settings), "");
    }
    if (dtype == "pot_efs" || dtype == "mtp_energy") {
        double we = 0.0, ws = 0.0, wf = 0.0, wv = 0.0;
        char flags = 0;
        is.read(reinterpret_cast<char*>(&we), sizeof(we));
        is.read(reinterpret_cast<char*>(&ws), sizeof(ws));
        is.read(reinterpret_cast<char*>(&wf), sizeof(wf));
        is.read(reinterpret_cast<char*>(&wv), sizeof(wv));
        is.read(&flags, 1);
        if (!is) return nullptr;
        std::string pf = pf_override;
        std::string pt = "mtp";
        if (pf.empty()) return nullptr;

        GPSettings settings;
        if (!pf.empty()) settings.Modify("potfile", pf);
        if (!pt.empty()) settings.Modify("pottype", pt);
        Potentials::potential_wrapper wrapper(settings);
        if (!wrapper.pot) return nullptr;
        std::unique_ptr<Potentials::AnyPotential> owned(wrapper.pot);
        wrapper.pot = nullptr;
        auto *pb = new PotEFSBuilder(std::move(owned));
        pb->w_energy = we; pb->w_site = ws;
        pb->w_force = wf;  pb->w_stress = wv;
        pb->linear_only = (flags & 1) != 0;
        pb->row_scaling = (flags & 2) != 0
                        ? RowScalingMode::MlipCompatible
                        : RowScalingMode::Uniform;
        pb->set_clone_config(pf, pt, settings);
        return std::unique_ptr<BMatrixBuilder>(pb);
    }
    return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared builder factory — used by maxvol_select and fps_select commands
// ═══════════════════════════════════════════════════════════════════════════

std::unique_ptr<BMatrixBuilder>
make_builder(const std::string &dtype, const std::string &pot_file,
             const std::string &pottype, GPSettings &settings,
             const std::string &mode_str,
             double w_energy, double w_site, double w_force, double w_stress,
             bool all_coeffs)
{
    BMatrixBuilderSettings builder_settings;
    builder_settings.dtype = dtype;
    builder_settings.potfile = pot_file;
    builder_settings.pottype = pottype;
    builder_settings.mode = mode_str;
    builder_settings.w_energy = w_energy;
    builder_settings.w_site = w_site;
    builder_settings.w_force = w_force;
    builder_settings.w_stress = w_stress;
    builder_settings.all_coeffs = all_coeffs;
    return make_builder(builder_settings, settings);
}

std::unique_ptr<BMatrixBuilder>
make_builder(const BMatrixBuilderSettings &builder_settings, GPSettings &settings)
{
    std::unique_ptr<BMatrixBuilder> builder;
    const std::string &dtype = builder_settings.dtype;
    const std::string &pot_file = builder_settings.potfile;
    const std::string &pottype = builder_settings.pottype;
    const std::string &mode_str = builder_settings.mode;
    double w_energy = builder_settings.w_energy;
    double w_site = builder_settings.w_site;
    double w_force = builder_settings.w_force;
    double w_stress = builder_settings.w_stress;
    bool all_coeffs = builder_settings.all_coeffs;

    auto make_desc_builder = [&](std::unique_ptr<Descriptors> desc,
                                 std::unique_ptr<Potentials::AnyPotential> pot = nullptr) {
        auto *db = new DescriptorBuilder(std::move(desc), std::move(pot));
        db->set_config_metadata(dtype, mode_str, settings);
        return db;
    };

    auto make_efs_builder = [&](std::unique_ptr<Potentials::AnyPotential> pot,
                                const GPSettings &pot_settings) {
        auto *pb = new PotEFSBuilder(std::move(pot));
        pb->w_energy = w_energy;
        pb->w_site   = w_site;
        pb->w_force  = w_force;
        pb->w_stress = w_stress;
        pb->linear_only = !all_coeffs;
        pb->set_clone_config(pot_file, pottype, pot_settings);
        return pb;
    };

    if (dtype == "mtm" || dtype == "mtp_scalar" || dtype == "mtp") {
        auto pot_settings = settings.ExtractSubSettings("pot");
        pot_settings.Modify("potfile", pot_file);
        pot_settings.Modify("pottype", pottype);
        Potentials::potential_wrapper wrapper(pot_settings);
        std::unique_ptr<Potentials::AnyPotential> owned_pot(wrapper.pot);
        wrapper.pot = nullptr;
        auto *mtp = dynamic_cast<Potentials::PotMTP *>(owned_pot.get());
        if (!mtp)
            GPERROR("MaxVol mtm requires a PotMTP-derived model; use dtype pot_efs for composite potentials");
        auto desc = std::make_unique<DescriptorMTM>(*mtp);
        builder.reset(make_desc_builder(std::move(desc), std::move(owned_pot)));
    }
    else if (dtype == "pot_efs" || dtype == "mtp_energy") {
        auto pot_settings = settings.ExtractSubSettings("pot");
        if (!pot_file.empty()) pot_settings.Modify("potfile", pot_file);
        if (!pottype.empty())  pot_settings.Modify("pottype", pottype);
        Potentials::potential_wrapper wrapper(pot_settings);
        if (!wrapper.pot) {
            Message(fmt::format("ERROR: failed to load potential type '{}'", pottype));
            return nullptr;
        }
        std::unique_ptr<Potentials::AnyPotential> owned(wrapper.pot);
        wrapper.pot = nullptr;
        builder.reset(make_efs_builder(std::move(owned), pot_settings));
    }
    else if (dtype == "behler") {
        auto desc = std::make_unique<DescriptorBehler>(
            settings.ExtractSubSettings("behler"));
        builder.reset(make_desc_builder(std::move(desc)));
    }
    else if (dtype == "bondorder") {
        auto desc = std::make_unique<DescriptorBondOrder>(
            settings.ExtractSubSettings("bondorder"));
        builder.reset(make_desc_builder(std::move(desc)));
    }
    else if (dtype == "biso4") {
        auto desc = std::make_unique<Descriptor_BispectrumSO4>(
            settings.ExtractSubSettings("biso4"));
        builder.reset(make_desc_builder(std::move(desc)));
    }
    else {
        Message(fmt::format("ERROR: unknown dtype '{}'", dtype));
        return nullptr;
    }
    if (builder)
        builder->row_scaling = row_scaling_mode_from_string(builder_settings.row_scaling);
    return builder;
}
