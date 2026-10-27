// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_hybrid_scale.h"

#include "potential_eam_poly.h"
#include "potential_mtp.h"
#include "utils/compressed_stream.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace Potentials {
namespace {

bool is_child_potfile_key(const std::string &key) {
    return key.starts_with("pot") && key.ends_with(":potfile");
}

} // namespace

void PotentialHybridScale::InitSettings() {
    Connect(npot, "npot", "number of sub-potentials");
    Connect(feature, "feature", "local feature used by scale function");
    Connect(transition, "transition", "scale transition function");
    Connect(feature_r0, "feature_r0", "inner radius for smooth_coordination");
    Connect(feature_r1, "feature_r1", "outer radius for smooth_coordination");
    Connect(species_count, "spcount", "number of species");
}

GPSettings PotentialHybridScale::load_manifest_settings(const GPSettings &settings) {
    const std::string manifest_name = settings["potfile"];
    if (manifest_name.empty()) {
        return settings;
    }

    GPSettings merged(manifest_name);
    const std::string manifest_type = merged["pottype"];
    if (!manifest_type.empty() && manifest_type != "hybrid_scale") {
        GPERROR("hybrid_scale: manifest pottype must be hybrid_scale");
    }

    const std::filesystem::path base =
        std::filesystem::absolute(std::filesystem::path(manifest_name)).parent_path();
    for (auto &[key, value] : merged.settings) {
        if (is_child_potfile_key(key) && !value.empty()) {
            std::filesystem::path child(value);
            if (child.is_relative()) {
                value = (base / child).lexically_normal().string();
            }
        }
    }

    for (const auto &[key, value] : settings.settings) {
        if (key != "potfile") {
            merged.Modify(key, value);
        }
    }
    return merged;
}

PotentialHybridScale::PotentialHybridScale(GPSettings settings, std::vector<std::unique_ptr<AnyPotential>> children) {
    if (children.empty()) settings = load_manifest_settings(settings);
    InitSettings();
    ApplySettings(settings);

    if (feature == "min_distance") {
        feature_kind_ = FeatureKind::MinDistance;
    } else if (feature == "smooth_coordination") {
        feature_kind_ = FeatureKind::SmoothCoordination;
    } else {
        GPERROR("hybrid_scale: unsupported feature: " + feature);
    }

    if (transition == "smoothstep") {
        transition_kind_ = TransitionKind::Smoothstep;
    } else {
        GPERROR("hybrid_scale: unsupported transition: " + transition);
    }

    load_potentials(settings, std::move(children));
    load_boundaries(settings);

    if (feature_r1 == -1.0) {
        feature_r1 = cutoff;
    }
    if (!std::isfinite(feature_r0) || !std::isfinite(feature_r1)
        || feature_r0 < 0.0 || feature_r1 <= feature_r0) {
        GPERROR("hybrid_scale: feature_r0/feature_r1 must be finite and satisfy 0 <= feature_r0 < feature_r1");
    }
    // Mixing weights can depend on neighbors outside every child potential.
    if (feature_kind_ == FeatureKind::SmoothCoordination)
        cutoff = std::max(cutoff, feature_r1);
    else if (!boundaries_.empty())
        cutoff = std::max(cutoff, boundaries_.back().right);
    cutoffsq = cutoff * cutoff;
    cutoffs = Eigen::MatrixXd::Constant(species_count, species_count, cutoff);
    cutoffsqs = Eigen::MatrixXd::Constant(species_count, species_count, cutoffsq);
    nl_inited = true;
}

void PotentialHybridScale::load_potentials(const GPSettings &settings, std::vector<std::unique_ptr<AnyPotential>> children) {
    if (npot <= 0) {
        GPERROR("hybrid_scale: npot must be positive");
    }

    const int configured_species_count = species_count;
    species_count = 0;
    ppots_.reserve(npot);
    pot_cutoffs_.reserve(npot);
    pot_types_.reserve(npot);
    parameter_offsets_.reserve(npot + 1);
    parameter_offsets_.push_back(0);
    cutoff = 0.0;

    for (int i = 0; i < npot; ++i) {
        const std::string prefix = "pot" + std::to_string(i + 1);
        GPSettings pset = settings.ExtractSubSettings(prefix);
        const std::string ptype = pset["pottype"];
        if (ptype.empty()) {
            GPERROR("hybrid_scale: missing " + prefix + ":pottype");
        }
        if (!children.empty() && children.size() != size_t(npot)) GPERROR("hybrid_scale: child count mismatch");
        auto pot = children.empty() ? make_subpotential(pset) : std::move(children[i]);
        if (!pot->supports_site_efs()) {
            GPERROR("hybrid_scale: sub-potential " + std::to_string(i + 1)
                    + " does not support site EFS");
        }
        if (!pot->supports_site_parameter_gradients()) {
            GPERROR("hybrid_scale: sub-potential " + std::to_string(i + 1)
                    + " does not support analytical site parameter gradients");
        }
        if (!std::isfinite(pot->cutoff) || pot->cutoff <= 0.0) {
            GPERROR("hybrid_scale: sub-potential " + std::to_string(i + 1)
                    + " has an invalid cutoff");
        }
        if (pot->species_count <= 0) {
            GPERROR("hybrid_scale: sub-potential " + std::to_string(i + 1)
                    + " has an invalid species count");
        }
        if (species_count == 0) {
            species_count = pot->species_count;
        } else if (species_count != pot->species_count) {
            GPERROR("hybrid_scale: all sub-potentials must have the same species count");
        }
        if (configured_species_count > 0 && configured_species_count != pot->species_count) {
            GPERROR("hybrid_scale: spcount does not match sub-potential species count");
        }

        pot->add_efs = true;
        pot->omp_init();
        cutoff = std::max(cutoff, pot->cutoff);
        pot_cutoffs_.push_back(pot->cutoff);
        pot_types_.push_back(ptype == "eam_poly" ? "eampoly" : ptype);
        parameter_offsets_.push_back(parameter_offsets_.back()
                                     + pot->get_NL_coeffs_count());
        ppots_.push_back(std::move(pot));
    }
}

std::unique_ptr<AnyPotential>
PotentialHybridScale::make_subpotential(const GPSettings &settings) const {
    const std::string ptype = settings["pottype"];
    const std::string potfile = settings["potfile"];
    if (ptype == "mtp") {
        if (potfile.empty()) {
            return std::make_unique<PotMTP>(settings);
        }
        return std::make_unique<PotMTP>(potfile);
    }
    if (ptype == "eampoly" || ptype == "eam_poly") {
        if (potfile.empty()) {
            return std::make_unique<PotentialEAMPoly>(settings);
        }
        return std::make_unique<PotentialEAMPoly>(potfile);
    }
    GPERROR("hybrid_scale: unsupported site sub-potential type: " + ptype);
}

void PotentialHybridScale::load_boundaries(const GPSettings &settings) {
    if (npot == 1) {
        return;
    }

    boundaries_.reserve(npot - 1);
    for (int i = 0; i < npot - 1; ++i) {
        const std::string key = "boundary" + std::to_string(i + 1);
        const std::string value = settings[key];
        if (value.empty()) {
            GPERROR("hybrid_scale: missing " + key);
        }
        Boundary boundary = parse_boundary(value, key);
        if (!boundaries_.empty()) {
            const Boundary &previous = boundaries_.back();
            if (boundary.left < previous.left || boundary.right < previous.right) {
                GPERROR("hybrid_scale: boundaries must be ordered by both endpoints");
            }
        }
        boundaries_.push_back(boundary);
    }
}

PotentialHybridScale::Boundary
PotentialHybridScale::parse_boundary(const std::string &text,
                                     const std::string &key) {
    Boundary boundary;
    std::stringstream ss(text);
    std::string trailing;
    if (!(ss >> boundary.left >> boundary.right) || (ss >> trailing)
        || !std::isfinite(boundary.left) || !std::isfinite(boundary.right)
        || boundary.right <= boundary.left) {
        GPERROR("hybrid_scale: " + key
                + " must contain exactly two finite increasing numbers");
    }
    return boundary;
}

void PotentialHybridScale::getEFS(Atoms &atom) {
    Neighborhoods nbhs(atom, cutoff);
    getEFS(atom, nbhs);
}

bool PotentialHybridScale::supports_site_efs() const {
    return !ppots_.empty() && std::all_of(ppots_.begin(), ppots_.end(),
        [](const auto &child) { return child && child->supports_site_efs(); });
}

void PotentialHybridScale::CalcSiteEFS(
    Neighborhood &nbh, double &energy, SiteCoordinateMatrix &gradient) {
    if (!supports_site_efs()) GPERROR("hybrid_scale requires local site EFS sub-potentials");
    std::vector<double> site_energies(ppots_.size(), 0.0);
    std::vector<double> weights(ppots_.size(), 0.0);
    std::vector<double> weight_ders(ppots_.size(), 0.0);
    std::vector<Eigen::Matrix<double, -1, 3, Eigen::RowMajor>>
        site_ders(ppots_.size());
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> q_ders;

    double q = 0.0;
    calc_feature(nbh, q, q_ders);
    calc_weights(q, weights, weight_ders);
    energy = 0.0;
    gradient.setZero(nbh.nbhcount_, 3);
    Neighborhood sub_nbh;
    std::vector<int> source_rows;
    for (int ipot = 0; ipot < npot; ++ipot) {
        make_site_neighborhood(nbh, pot_cutoffs_[ipot], sub_nbh, source_rows);
        ppots_[ipot]->CalcSiteEFS(sub_nbh, site_energies[ipot], site_ders[ipot]);
        energy += weights[ipot] * site_energies[ipot];
        for (int j = 0; j < static_cast<int>(source_rows.size()); ++j) {
            gradient.row(source_rows[j]) += weights[ipot] * site_ders[ipot].row(j);
        }
        gradient += (weight_ders[ipot] * site_energies[ipot]) * q_ders;
    }
}

void PotentialHybridScale::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    SiteCoordinateMatrix mixed_ders;
    for (int ii = 0; ii < nbhs.size(); ++ii) {
        Neighborhood &nbh = nbhs[ii];
        double mixed_site_energy = 0.0;
        CalcSiteEFS(nbh, mixed_site_energy, mixed_ders);

        atom.energy_ += mixed_site_energy;
        atom.site_energies_[ii] += mixed_site_energy;

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            atom.forces_[ii] += mixed_ders.row(j);
            atom.forces_[nbh.indices_[j]] -= mixed_ders.row(j);
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    atom.stress_(a, b) -= mixed_ders(j, a) * nbh.pos_[j](b);
                }
            }
        }
    }
}

void PotentialHybridScale::getEFSOMP(Atoms &atom) {
    getEFS(atom);
}

void PotentialHybridScale::getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs) {
    getEFS(atom, nbhs);
}

void PotentialHybridScale::getEFSOMPthread(Atoms &atom) {
    getEFS(atom);
}

void PotentialHybridScale::getEFSOMPthread(Atoms &atom,
                                           AnyNeighborhoods &nbhs) {
    getEFS(atom, nbhs);
}

void PotentialHybridScale::getE(Atoms &atom) {
    Neighborhoods nbhs(atom, cutoff);
    getE(atom, nbhs);
}

void PotentialHybridScale::getE(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) {
        atom.resetE();
    } else {
        atom.setEflag();
    }

    std::vector<double> weights(ppots_.size(), 0.0);
    std::vector<double> weight_ders(ppots_.size(), 0.0);
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> q_ders;
    Neighborhood sub_nbh;
    std::vector<int> source_rows;

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        Neighborhood &nbh = nbhs[ii];
        double q = 0.0;
        calc_feature(nbh, q, q_ders);
        calc_weights(q, weights, weight_ders);

        double mixed_site_energy = 0.0;
        for (int ipot = 0; ipot < npot; ++ipot) {
            double site_energy = 0.0;
            make_site_neighborhood(nbh, pot_cutoffs_[ipot], sub_nbh, source_rows);
            ppots_[ipot]->CalcSiteE(sub_nbh, site_energy);
            mixed_site_energy += weights[ipot] * site_energy;
        }
        atom.energy_ += mixed_site_energy;
        atom.site_energies_[ii] += mixed_site_energy;
    }
}

int PotentialHybridScale::get_NL_coeffs_count() const {
    return parameter_offsets_.empty() ? 0 : parameter_offsets_.back();
}

void PotentialHybridScale::nl_get_coeffs(double *x) const {
    for (int i = 0; i < npot; ++i) {
        ppots_[i]->nl_get_coeffs(x + parameter_offsets_[i]);
    }
}

void PotentialHybridScale::nl_set_coeffs(const double *x) {
    for (int i = 0; i < npot; ++i) {
        ppots_[i]->nl_set_coeffs(x + parameter_offsets_[i]);
    }
}

std::vector<NLParameterGroup> PotentialHybridScale::nl_parameter_groups() const {
    std::vector<NLParameterGroup> result;
    for (int i = 0; i < npot; ++i) {
        for (NLParameterGroup group : ppots_[i]->nl_parameter_groups()) {
            group.name = "pot" + std::to_string(i + 1) + "/" + group.name;
            group.begin += parameter_offsets_[i];
            group.end += parameter_offsets_[i];
            result.push_back(std::move(group));
        }
    }
    return result;
}

std::vector<ConditionalLinearBlock> PotentialHybridScale::conditional_linear_blocks(double penalty_weight) const {
    if (typeid(*this) != typeid(PotentialHybridScale)) return {};
    std::vector<ConditionalLinearBlock> blocks;
    for (int i=0;i<npot;++i) for (auto block : ppots_[i]->conditional_linear_blocks(penalty_weight)) {
        block.name="pot"+std::to_string(i+1)+"/"+block.name;
        for (int &index : block.indices) index+=parameter_offsets_[i];
        blocks.push_back(std::move(block));
    }
    return blocks;
}

void PotentialHybridScale::conditional_linear_set_coeffs(const std::vector<int>& indices, const Eigen::VectorXd& values) {
    if (values.size()!=static_cast<Eigen::Index>(indices.size()) || !values.allFinite())
        GPERROR("Invalid conditional linear HybridScale update");
    std::vector<bool> allowed(get_NL_coeffs_count(),false);
    for (const auto &block : conditional_linear_blocks(0.))
        for (int index : block.indices) allowed[index]=true;
    for (int index : indices)
        if (index<0 || index>=int(allowed.size()) || !allowed[index])
            GPERROR("Invalid conditional linear HybridScale index");
    for (int i=0;i<npot;++i) {
        std::vector<int> child_indices;std::vector<double> child_values;
        for (int j=0;j<values.size();++j) {
            const int index=indices[j];
            if (index>=parameter_offsets_[i] && index<parameter_offsets_[i+1]) {
                child_indices.push_back(index-parameter_offsets_[i]);child_values.push_back(values[j]);
            }
        }
        if (!child_indices.empty()) ppots_[i]->conditional_linear_set_coeffs(child_indices,
            Eigen::Map<const Eigen::VectorXd>(child_values.data(),child_values.size()));
    }
}

void PotentialHybridScale::random_init() {
    for (auto &pot : ppots_) {
        pot->random_init();
    }
    set_inited();
}

void PotentialHybridScale::init_coeffs() {
    for (auto &pot : ppots_) {
        pot->init_coeffs();
    }
    set_inited();
}

void PotentialHybridScale::train_init() {
    for (auto &pot : ppots_) {
        pot->train_init();
    }
}

void PotentialHybridScale::set_inited() {
    nl_inited = true;
    for (auto &pot : ppots_) {
        pot->set_inited();
    }
}

void PotentialHybridScale::set_empty() {
    nl_inited = false;
    for (auto &pot : ppots_) {
        pot->set_empty();
    }
}

bool PotentialHybridScale::inited() {
    return nl_inited;
}

void PotentialHybridScale::AddPenaltyGard(double weight, double &loss,
                                          double *loss_gard) {
    for (int i = 0; i < npot; ++i) {
        ppots_[i]->AddPenaltyGard(weight, loss,
                                 loss_gard + parameter_offsets_[i]);
    }
}

void PotentialHybridScale::AccumulateEFSCombinationGard(
    Atoms &, AnyNeighborhoods &nbhs, double dLdE,
    Eigen::Matrix<double, -1, 3> &dLdF, Eigen::Matrix3d &dLdS,
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &wgt_d2UdRdx,
    double *out_gard_accumulator) {
    std::vector<double> weights(ppots_.size(), 0.0);
    std::vector<double> weight_ders(ppots_.size(), 0.0);
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> q_ders;
    Neighborhood sub_nbh;
    std::vector<int> source_rows;

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        Neighborhood &nbh = nbhs[ii];
        double q = 0.0;
        calc_feature(nbh, q, q_ders);
        calc_weights(q, weights, weight_ders);

        wgt_d2UdRdx.resize(nbh.nbhcount_, 3);
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a) {
                wgt_d2UdRdx(j, a) =
                    dLdF(ii, a) - dLdF(nbh.indices_[j], a);
                for (int b = 0; b < 3; ++b) {
                    wgt_d2UdRdx(j, a) -= dLdS(a, b) * nbh.pos_[j][b];
                }
            }
        }
        const double feature_adjoint = (wgt_d2UdRdx.array()
                                        * q_ders.array()).sum();

        for (int ipot = 0; ipot < npot; ++ipot) {
            make_site_neighborhood(nbh, pot_cutoffs_[ipot], sub_nbh, source_rows);
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor>
                child_force_weights(source_rows.size(), 3);
            for (int j = 0; j < static_cast<int>(source_rows.size()); ++j) {
                child_force_weights.row(j) =
                    weights[ipot] * wgt_d2UdRdx.row(source_rows[j]);
            }
            const double site_weight = dLdE * weights[ipot]
                                     + weight_ders[ipot] * feature_adjoint;
            ppots_[ipot]->AccumulateFPGard(
                sub_nbh, out_gard_accumulator + parameter_offsets_[ipot],
                site_weight, &child_force_weights);
        }
    }
}

void PotentialHybridScale::AccumulateEFSCombinationGard(
    Atoms &, AnyNeighborhoods &nbhs, double dLdE,
    double *out_gard_accumulator) {
    std::vector<double> weights(ppots_.size(), 0.0);
    std::vector<double> weight_ders(ppots_.size(), 0.0);
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> q_ders;
    Neighborhood sub_nbh;
    std::vector<int> source_rows;

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        Neighborhood &nbh = nbhs[ii];
        double q = 0.0;
        calc_feature(nbh, q, q_ders);
        calc_weights(q, weights, weight_ders);
        for (int ipot = 0; ipot < npot; ++ipot) {
            make_site_neighborhood(nbh, pot_cutoffs_[ipot], sub_nbh, source_rows);
            ppots_[ipot]->AccumulateFPGard(
                sub_nbh, out_gard_accumulator + parameter_offsets_[ipot],
                dLdE * weights[ipot], nullptr);
        }
    }
}

void PotentialHybridScale::Save(const std::string &path) {
    if (save_gpm_if_requested(*this, path)) return;
    const std::filesystem::path manifest(path);
    const std::filesystem::path directory = manifest.has_parent_path()
        ? manifest.parent_path() : std::filesystem::path(".");
    const std::string stem = manifest.filename().string();

    std::vector<std::string> child_names;
    child_names.reserve(npot);
    for (int i = 0; i < npot; ++i) {
        const std::string name = stem + ".pot" + std::to_string(i + 1);
        ppots_[i]->Save((directory / name).string());
        child_names.push_back(name);
    }

    auto os = open_ofile(manifest);
    if (!os) {
        throw std::runtime_error("Failed to open hybrid manifest: " + path);
    }
    *os << std::setprecision(17);
    *os << "pottype hybrid_scale\n";
    *os << "npot " << npot << '\n';
    *os << "spcount " << species_count << '\n';
    *os << "feature " << feature << '\n';
    *os << "transition " << transition << '\n';
    *os << "feature_r0 " << feature_r0 << '\n';
    *os << "feature_r1 " << feature_r1 << '\n';
    for (int i = 0; i < static_cast<int>(boundaries_.size()); ++i) {
        *os << "boundary" << i + 1 << ' ' << boundaries_[i].left
           << ' ' << boundaries_[i].right << '\n';
    }
    for (int i = 0; i < npot; ++i) {
        *os << "pot" << i + 1 << ":pottype " << pot_types_[i] << '\n';
        *os << "pot" << i + 1 << ":potfile " << child_names[i] << '\n';
    }
    if (!close_ofile(os)) throw std::runtime_error("Failed to write hybrid manifest: " + path);
}

void PotentialHybridScale::make_site_neighborhood(
    const Neighborhood &nbh, double rcut, Neighborhood &sub_nbh,
    std::vector<int> &source_rows) {
    sub_nbh.nbhcount_ = 0;
    sub_nbh.indices_.clear();
    sub_nbh.chemmap_.clear();
    sub_nbh.pos_.clear();
    sub_nbh.distances_.clear();
    sub_nbh.centertype = nbh.centertype;
    source_rows.clear();

    for (int j = 0; j < nbh.nbhcount_; ++j) {
        if (nbh.distances_[j] > rcut) {
            continue;
        }
        sub_nbh.indices_.push_back(nbh.indices_[j]);
        sub_nbh.chemmap_.push_back(nbh.chemmap_[j]);
        sub_nbh.pos_.push_back(nbh.pos_[j]);
        sub_nbh.distances_.push_back(nbh.distances_[j]);
        source_rows.push_back(j);
    }
    sub_nbh.nbhcount_ = static_cast<int>(sub_nbh.indices_.size());
}

void PotentialHybridScale::calc_feature(
    const Neighborhood &nbh, double &q,
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &q_ders) const {
    q_ders.resize(nbh.nbhcount_, 3);
    q_ders.setZero();

    if (feature_kind_ == FeatureKind::MinDistance) {
        if (nbh.nbhcount_ == 0) {
            q = std::numeric_limits<double>::infinity();
            return;
        }
        int min_id = 0;
        q = nbh.distances_[0];
        for (int j = 1; j < nbh.nbhcount_; ++j) {
            if (nbh.distances_[j] < q) {
                q = nbh.distances_[j];
                min_id = j;
            }
        }
        if (q > 0.0) {
            q_ders.row(min_id) = nbh.pos_[min_id] / q;
        }
        return;
    }

    q = 0.0;
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        const double r = nbh.distances_[j];
        if (r <= 0.0) {
            continue;
        }
        double s = 0.0;
        double dsdr = 0.0;
        smoothstep(feature_r0, feature_r1, r, s, dsdr);
        q += 1.0 - s;
        q_ders.row(j) = -dsdr * nbh.pos_[j] / r;
    }
}

void PotentialHybridScale::calc_weights(
    double q, std::vector<double> &weights,
    std::vector<double> &weight_ders) const {
    weights.assign(ppots_.size(), 0.0);
    weight_ders.assign(ppots_.size(), 0.0);

    if (npot == 1) {
        weights[0] = 1.0;
        return;
    }

    double prefix = 1.0;
    double prefix_derivative = 0.0;
    for (int i = 0; i < npot - 1; ++i) {
        double value = 0.0;
        double derivative = 0.0;
        smoothstep(boundaries_[i].left, boundaries_[i].right, q,
                   value, derivative);
        weights[i] = prefix * (1.0 - value);
        weight_ders[i] = prefix_derivative * (1.0 - value)
                       - prefix * derivative;
        prefix_derivative = prefix_derivative * value + prefix * derivative;
        prefix *= value;
    }
    weights.back() = prefix;
    weight_ders.back() = prefix_derivative;
}

void PotentialHybridScale::smoothstep(double x0, double x1, double x,
                                      double &value, double &derivative) {
    if (x <= x0) {
        value = 0.0;
        derivative = 0.0;
        return;
    }
    if (x >= x1) {
        value = 1.0;
        derivative = 0.0;
        return;
    }
    const double t = (x - x0) / (x1 - x0);
    value = t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
    derivative = (30.0 * t * t * (1.0 - t) * (1.0 - t)) / (x1 - x0);
}

} // namespace Potentials
