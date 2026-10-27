// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_hybrid.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace {
GPSettings hybrid_settings(GPSettings settings) {
    if (settings["potfile"].empty()) return settings;
    // Legacy commands also pass a top-level native-model filename alongside
    // a complete explicit Hybrid definition. Those child settings take priority.
    if (settings.count("npot")) {
        const int count = Parameters::parse_value<int>(settings["npot"]);
        bool explicit_children = count > 0;
        for (int i = 1; i <= count && explicit_children; ++i) {
            const auto key = "pot" + std::to_string(i) + ":pottype";
            explicit_children = settings.count(key) && !settings[key].empty();
        }
        if (explicit_children) return settings;
    }
    const auto path = std::filesystem::absolute(settings["potfile"]);
    GPSettings merged(path.string());
    if (merged["pottype"] != "hybrid") GPERROR("hybrid: expected a hybrid manifest");
    for (auto &[key, value] : merged.settings)
        if (key.starts_with("pot") && key.ends_with(":potfile") && !value.empty()
            && std::filesystem::path(value).is_relative())
            value = (path.parent_path() / value).lexically_normal().string();
    for (const auto &[key, value] : settings.settings)
        if (key != "potfile") merged.Modify(key, value);
    return merged;
}
}

Potentials::PotentialHybrid::PotentialHybrid(GPSettings settings, std::vector<std::unique_ptr<AnyPotential>> children)
{
    if (children.empty()) settings = hybrid_settings(std::move(settings));
    else if (settings["npot"] != std::to_string(children.size())) GPERROR("hybrid: child count mismatch");
    InitSettings();
    ApplySettings(settings);
    if (npot <= 0) {
        GPERROR("hybrid: npot must be positive");
    }

    const int configured_species_count = species_count;
    species_count = 0;
    ppots.resize(npot);
    potnames.resize(npot);
    cutoffs.resize(npot);
    cutoff = 0.0;
    Message("Loading " + std::to_string(npot) + " potentials");
    for (int i = 0; i < npot; ++i) {
        GPSettings pset = settings.ExtractSubSettings("pot" + std::to_string(i+1));
        const std::string ptype = pset["pottype"];
        if (ptype.empty()) {
            GPERROR("hybrid: missing pot" + std::to_string(i + 1) + ":pottype");
        }
        Message("Potential number " + std::to_string(i+1) + " type is " + ptype);
        if (ptype == "zbl") {
            ppots[i] = children.empty() ? std::make_unique<PotentialZBL>(pset) : std::move(children[i]);
            if (!dynamic_cast<PotentialZBL*>(ppots[i].get())) GPERROR("hybrid: child type mismatch");
        } else if (ptype == "mtp") {
            std::unique_ptr<PotMTP> pmtp_owner;
            if (children.empty()) pmtp_owner = std::make_unique<PotMTP>(pset);
            else {
                if (!children[i] || typeid(*children[i]) != typeid(PotMTP)) GPERROR("hybrid: child type mismatch");
                pmtp_owner.reset(static_cast<PotMTP*>(children[i].release()));
            }
            PotMTP *pmtp = pmtp_owner.get();
            int species_count_this = pmtp->species_count;
            if (species_count == 0) {
                species_count = species_count_this;
            } else if (species_count_this != species_count) {
                GPERROR("hybrid: all MTP sub-potentials must have the same species count");
            }
            ppots[i] = std::move(pmtp_owner);
            if (!pset.count("train") || Parameters::parse_value<bool>(pset["train"])) {
                if (train_ids.empty()) nreg_coeffs += species_count_this;
                train_ids.push_back(i);
                nreg_coeffs += pmtp->get_coeff_size() - species_count;
                nrad_coeffs += pmtp->radial_coeff_size;
                nmom_coeffs += pmtp->alpha_scalar_moments;
            }
            potnames[i] = pset["potfile"];
        } else {
            GPERROR("hybrid: unsupported sub-potential type: " + ptype);
        }
        if (!std::isfinite(ppots[i]->cutoff) || ppots[i]->cutoff <= 0.0) {
            GPERROR("hybrid: sub-potential " + std::to_string(i + 1)
                    + " has an invalid cutoff");
        }
        if (ppots[i]->species_count > 0) {
            if (species_count == 0) {
                species_count = ppots[i]->species_count;
            } else if (species_count != ppots[i]->species_count) {
                GPERROR("hybrid: all sub-potentials must have the same species count");
            }
        }
        ppots[i]->add_efs = true;
        ppots[i]->omp_init();
        cutoffs[i] = ppots[i]->cutoff;
        cutoff = std::max(cutoff, cutoffs[i]);
    }
    if (configured_species_count > 0 && configured_species_count != species_count) {
        GPERROR("hybrid: spcount does not match sub-potential species count");
    }
    ntrains = static_cast<int>(train_ids.size());
    nlin_coeffs = ntrains ? nmom_coeffs + species_count : 0;
    reg_L1_vector.assign(nreg_coeffs, 0.0);
    reg_L2_vector.assign(nreg_coeffs, 0.0);
    // Combine radial, shared species, and moment coefficients.
    if (nreg_coeffs == 0) {
        nl_inited = true;
        return;
    }
    regression_storage_ = std::make_unique<double[]>(nreg_coeffs);
    regression_coeffs = regression_storage_.get();
    std::fill_n(regression_coeffs, nreg_coeffs, 1e-3);
    linear_coeffs = regression_coeffs + nrad_coeffs;
    species_coeffs = linear_coeffs;
    moment_coeffs = regression_coeffs + nrad_coeffs + species_count;
    radial_coeffs = regression_coeffs;
    int roffset = 0; // offset for current radial coeffs
    int loffset = nrad_coeffs + species_count; // as above
    // _DEBUG_OUTPUT_(nrad_coeffs);
    // _DEBUG_OUTPUT_(nreg_coeffs);
    // _DEBUG_OUTPUT_(nlin_coeffs);
    // _DEBUG_OUTPUT_(nmom_coeffs);
    // _DEBUG_OUTPUT_(species_count);
    secondary_species_coeffs_.resize(train_ids.size());
    for (int i = 0; i < static_cast<int>(train_ids.size()); ++i) {
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[train_ids[i]].get());
        // copy radial coeffs
        memcpy(regression_coeffs + roffset, p_mtp->radial_coeffs, sizeof(double) * p_mtp->radial_coeff_size);
        // copy species coeffs, only use value of first potential
        if (i == 0) {
            for (int is = 0; is < species_count; ++is) {
                regression_coeffs[nrad_coeffs + is] = p_mtp->species_coeffs[is];
            }
        }
        // copy moment coeffs
        memcpy(regression_coeffs + loffset, p_mtp->moment_coeffs, sizeof(double) * p_mtp->alpha_scalar_moments);
        // update pointer
        delete [] p_mtp->regression_coeffs;
        p_mtp->regression_coeffs = nullptr;
        // p_mtp->species_coeffs = regression_coeffs + nrad_coeffs;
        if (i == 0) {
            p_mtp->species_coeffs = regression_coeffs + nrad_coeffs;
            p_mtp->linear_coeffs = p_mtp->species_coeffs;
        } else {
            secondary_species_coeffs_[i].assign(species_count, 0.0);
            p_mtp->species_coeffs = secondary_species_coeffs_[i].data();
            p_mtp->linear_coeffs = nullptr; // Only the first sub-potential contributes the shared species term.
        }
        p_mtp->radial_coeffs = regression_coeffs + roffset;
        p_mtp->moment_coeffs = regression_coeffs + loffset;
        // Keep the original buffer owned by the sub-potential until the
        // handoff succeeds, so earlier constructor failures release it.
        p_mtp->coeffs_handled_external = true;
        // update offset
        roffset += p_mtp->radial_coeff_size;
        loffset += p_mtp->alpha_scalar_moments;
    }
    nl_inited = true;
}

void Potentials::PotentialHybrid::train_init() {
    for (auto &i: train_ids) {
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[i].get());
        p_mtp->train_init();
    }
}

void Potentials::PotentialHybrid::getEFS(Atoms &atom)
{
    Neighborhoods nbhs(atom, cutoff);
    getEFS(atom, nbhs);
}

bool Potentials::PotentialHybrid::supports_site_efs() const {
    return !ppots.empty() && std::all_of(ppots.begin(), ppots.end(),
        [](const auto &child) { return child && child->supports_site_efs(); });
}

void Potentials::PotentialHybrid::CalcSiteEFS(
    Neighborhood &nbh, double &energy, SiteCoordinateMatrix &gradient) {
    if (!supports_site_efs()) GPERROR("hybrid requires local site EFS sub-potentials");
    energy = 0.0;
    gradient.setZero(nbh.nbhcount_, 3);
    Neighborhood child_site;
    std::vector<int> source_rows;
    SiteCoordinateMatrix child_gradient;
    for (auto &child : ppots) {
        child_site.centertype = nbh.centertype;
        child_site.indices_.clear(); child_site.chemmap_.clear();
        child_site.pos_.clear(); child_site.distances_.clear();
        source_rows.clear();
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            if (nbh.distances_[j] >= child->cutoff) continue;
            child_site.indices_.push_back(nbh.indices_[j]);
            child_site.chemmap_.push_back(nbh.chemmap_[j]);
            child_site.pos_.push_back(nbh.pos_[j]);
            child_site.distances_.push_back(nbh.distances_[j]);
            source_rows.push_back(j);
        }
        child_site.nbhcount_ = source_rows.size();
        double child_energy = 0.0;
        child->CalcSiteEFS(child_site, child_energy, child_gradient);
        energy += child_energy;
        for (int j = 0; j < child_site.nbhcount_; ++j)
            gradient.row(source_rows[j]) += child_gradient.row(j);
    }
}

bool Potentials::PotentialHybrid::supports_analytical_hessian() const {
    return !ppots.empty() && std::all_of(ppots.begin(), ppots.end(), [](const auto &pot) {
        return pot && pot->supports_analytical_hessian();
    });
}

void Potentials::PotentialHybrid::getEFSHessian(
    Atoms &atom, HessianMatrix &hessian) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessian(atom, nbhs, hessian);
}

void Potentials::PotentialHybrid::getEFSHessian(
    Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    if (!supports_analytical_hessian())
        GPERROR("hybrid: every sub-potential must support analytical atomic Hessians");
    if (nbhs.size() != atom.size())
        GPERROR("Hessian neighborhoods have the wrong size");
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    HessianMatrix result, child_hessian;
    result.resize(atom.size());
    for (auto &pot : ppots) {
        auto child_nbhs = nbhs.sub_neighborhoods(pot->cutoff);
        pot->getEFSHessian(atom, child_nbhs, child_hessian);
        for (int i = 0; i < atom.size(); ++i)
            for (int j = 0; j < atom.size(); ++j)
                result(i, j) += child_hessian(i, j);
    }
    hessian = std::move(result);
}

void Potentials::PotentialHybrid::getEFSHessianOMP(
    Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    // Every legal child has a dense serial implementation; some lack an
    // explicit OMP entry point. Keep the same complete sum for both APIs.
    getEFSHessian(atom, nbhs, hessian);
}

void Potentials::PotentialHybrid::getEFSHessianSymm(
    Atoms &atom, HessianMatrix &hessian, double) {
    getEFSHessian(atom, hessian);
}

bool Potentials::PotentialHybrid::supports_analytical_hessian_vector_product() const {
    return !ppots.empty() && std::all_of(ppots.begin(), ppots.end(), [](const auto &pot) {
        return pot && pot->supports_analytical_hessian_vector_product();
    });
}

void Potentials::PotentialHybrid::applyHessianVectorProduct(
    Atoms &atom, AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction, Eigen::VectorXd &product) {
    if (!supports_analytical_hessian_vector_product())
        GPERROR("hybrid: every sub-potential must support analytical Hessian-vector products");
    if (direction.size() != 3 * atom.size())
        GPERROR("Hessian-vector direction has the wrong size");
    if (nbhs.size() != atom.size())
        GPERROR("Hessian-vector neighborhoods have the wrong size");
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());
    Eigen::VectorXd child_product;
    for (auto &pot : ppots) {
        auto child_nbhs = nbhs.sub_neighborhoods(pot->cutoff);
        pot->applyHessianVectorProduct(atom, child_nbhs, direction, child_product);
        result += child_product;
    }
    product = std::move(result);
}

void Potentials::PotentialHybrid::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    for (int i = 0; i < static_cast<int>(ppots.size()); ++i) {
        auto snbhs = nbhs.sub_neighborhoods(ppots[i]->cutoff);
        ppots[i]->getEFS(atom, snbhs);
    }
}

void Potentials::PotentialHybrid::getEFSOMP(Atoms &atom) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSOMP(atom, nbhs);
}

void Potentials::PotentialHybrid::getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    for (int i = 0; i < static_cast<int>(ppots.size()); ++i) {
        auto snbhs = nbhs.sub_neighborhoods(ppots[i]->cutoff);
        ppots[i]->getEFSOMP(atom, snbhs);
    }
}

void Potentials::PotentialHybrid::getEFSOMPthread(Atoms &atom) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSOMPthread(atom, nbhs);
}

void Potentials::PotentialHybrid::getEFSOMPthread(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    for (int i = 0; i < static_cast<int>(ppots.size()); ++i) {
        auto snbhs = nbhs.sub_neighborhoods(cutoffs[i]);
        ppots[i]->getEFSOMPthread(atom, snbhs);
    }
}

void Potentials::PotentialHybrid::getE(Atoms &atom) {
    Neighborhoods nbhs(atom, cutoff);
    getE(atom, nbhs);
}

void Potentials::PotentialHybrid::getE(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetE(); else atom.setEflag();
    for (int i = 0; i < static_cast<int>(ppots.size()); ++i) {
        auto snbhs = nbhs.sub_neighborhoods(ppots[i]->cutoff);
        ppots[i]->getE(atom, snbhs);
    }
}

void Potentials::PotentialHybrid::CalcEFSCmpt(Atoms &atom, 
                        AnyNeighborhoods &nbhs, 
                        std::vector<double> &energy_cmpts, 
                        Eigen::Tensor<double, 3> &force_cmpts, 
                        Eigen::Tensor<double, 3> &stress_cmpts) {
    energy_cmpts.assign(nlin_coeffs, 0);
    force_cmpts.resize(atom.size(), nlin_coeffs, 3);
    force_cmpts.setZero();
    stress_cmpts.resize(nlin_coeffs, 3, 3);
    stress_cmpts.setZero();
    if (train_ids.empty()) return;
    for (int iat = 0; iat < atom.size(); ++iat) {
        energy_cmpts[atom.chemmap_[iat]] += 1;
    }
    int offset = species_count;
    for (int i = 0; i < train_ids.size(); ++i) {
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[train_ids[i]].get());
        auto nbhs_this = nbhs.sub_neighborhoods(p_mtp->cutoff);
        p_mtp->CalcEFSCmpt(atom, nbhs_this, energy_cmpts, force_cmpts, stress_cmpts, offset);
        offset += p_mtp->alpha_scalar_moments;
    }
}

void Potentials::PotentialHybrid::CalcECmpt(Atoms &atom, AnyNeighborhoods &nbhs, std::vector<double> &energy_cmpts) {
    energy_cmpts.assign(nlin_coeffs, 0);
    if (train_ids.empty()) return;
    for (int iat = 0; iat < atom.size(); ++iat) {
        energy_cmpts[atom.chemmap_[iat]] += 1;
    }
    int offset = species_count;
    for (int i = 0; i < train_ids.size(); ++i) {
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[train_ids[i]].get());
        auto nbhs_this = nbhs.sub_neighborhoods(p_mtp->cutoff);
        p_mtp->CalcECmpt(atom, nbhs_this, energy_cmpts, offset);
        offset += p_mtp->alpha_scalar_moments;
    }
}

void Potentials::PotentialHybrid::Orthogonalize() {
    for (int i = 0; i < train_ids.size(); ++i) {
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[train_ids[i]].get());
        p_mtp->Orthogonalize();
    }
}

void Potentials::PotentialHybrid::set_moment_coeff(double *_x) {
    memcpy(regression_coeffs + nrad_coeffs + species_count, _x, sizeof(double) * nmom_coeffs);
}

void Potentials::PotentialHybrid::set_linear_coeff(double *_x) {
    memcpy(regression_coeffs + nrad_coeffs, _x, sizeof(double) * nlin_coeffs);
}

void Potentials::PotentialHybrid::train_coeff_init(bool random) {
    if (train_ids.size() > 0) {
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[train_ids[0]].get());
        if (!p_mtp->inited()) {
            if (random) {
                p_mtp->random_init_radial();
                p_mtp->random_init_linear();
                p_mtp->empty = true;
            }
        }
    }
    for (auto &i : train_ids) {
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[i].get());
        if (!p_mtp->inited()) {
            if (random) {
                p_mtp->random_init_radial();
                p_mtp->random_init_moments();
                p_mtp->empty = false;
            }
        }
    }
}

void Potentials::PotentialHybrid::random_init_linear() {
    std::uniform_real_distribution<double> uniform(random_lower, random_upper);
    for (int i = 0; i < nlin_coeffs; ++i) {
        linear_coeffs[i] = uniform(RNG.mtd);
    }
    #ifdef IMRMLP_MPI
    MPI_Bcast(linear_coeffs, nlin_coeffs, MPI_DOUBLE, 0, mpi.comm);
    #endif
    for (auto &i : train_ids) {
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[i].get());
        p_mtp->empty = false;
    }
}

void Potentials::PotentialHybrid::random_init_radial() {
    std::uniform_real_distribution<double> uniform(random_lower, random_upper);
    for (int i = 0; i < nrad_coeffs; ++i) {
        radial_coeffs[i] = uniform(RNG.mtd);
    }
    #ifdef IMRMLP_MPI
    MPI_Bcast(radial_coeffs, nrad_coeffs, MPI_DOUBLE, 0, mpi.comm);
    #endif
    for (auto &i : train_ids) {
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[i].get());
        p_mtp->empty = false;
    }
}

void Potentials::PotentialHybrid::set_radial_regtype(int mode, double wgt_rad_w0, double wgt_rad_w1, double wgt_rad_w2, double randlow, double randupw, int quad_point)
{
    random_lower = randlow;
    random_upper = randupw;
    for (auto &i : train_ids) {
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[i].get());
        p_mtp->radial_mode = mode;
        p_mtp->weight_rad_w0 = wgt_rad_w0;
        p_mtp->weight_rad_w1 = wgt_rad_w1;
        p_mtp->weight_rad_w2 = wgt_rad_w2;
        p_mtp->random_lower = randlow;
        p_mtp->random_upper = randupw;
        p_mtp->quad_point = quad_point;
    }
}

void Potentials::PotentialHybrid::AddPenaltyGard(const double weight, double &loss, double *loss_gard, bool fix_radial, bool fix_linear, bool fix_species) {
    int coffset = 0;
    if (!fix_radial) {
        for (int it = 0; it < train_ids.size(); ++it) {
            auto &i = train_ids[it];
            // we must fix linear coeffs since species_coeffs are missed for some potentials
            PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[i].get());
            p_mtp->AddPenaltyGard(weight, loss, loss_gard + coffset, fix_radial, true, true);
            coffset += p_mtp->radial_coeff_size;
        }
    }
    // add species and moment part
    if (!fix_linear) {
        for (int i = 0; i < nlin_coeffs; ++i) {
            loss += fabs(linear_coeffs[i]) * reg_L1_vector[i];
            loss += linear_coeffs[i] * linear_coeffs[i] * reg_L2_vector[i];
            loss_gard[coffset + i] += linear_coeffs[i] > 0 ? reg_L1_vector[i] : -reg_L1_vector[i];
            loss_gard[coffset + i] += 2 * linear_coeffs[i] * reg_L2_vector[i];
        }
    }
}

void Potentials::PotentialHybrid::AccumulateEFSCombinationGard(Atoms &atom, AnyNeighborhoods &nbhs, double dLdE, Eigen::Matrix<double, -1, 3> &dLdF, Eigen::Matrix3d &dLdS, double *out_gard_accumulator) {
    if (train_ids.empty()) return;
    int roffset = 0;
    int loffset = nrad_coeffs + species_count;
    for (int it = 0; it < train_ids.size(); ++it) {
        auto &i = train_ids[it];
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[i].get());
        auto snbhs = nbhs.sub_neighborhoods(ppots[i]->cutoff);
        // species coeffs is handled on our own
        p_mtp->AccumulateEFSCombinationGard(atom, snbhs, dLdE, dLdF, dLdS, out_gard_accumulator, roffset, loffset);
        roffset += p_mtp->radial_coeff_size;
        loffset += p_mtp->alpha_scalar_moments;
    }
    // The shared species coefficient contributes once per atom of that type.
    for (int type : atom.chemmap_) {
        if (type < 0 || type >= species_count) {
            GPERROR("hybrid: atom species index is out of range");
        }
        out_gard_accumulator[nrad_coeffs + type] += dLdE;
    }
}

void Potentials::PotentialHybrid::AccumulateEFSCombinationGard(
    Atoms &atom, AnyNeighborhoods &nbhs, double dLdE,
    Eigen::Matrix<double, -1, 3> &dLdF, Eigen::Matrix3d &dLdS,
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &,
    double *out_gard_accumulator)
{
    AccumulateEFSCombinationGard(atom, nbhs, dLdE, dLdF, dLdS,
                                 out_gard_accumulator);
}

void Potentials::PotentialHybrid::AccumulateEFSCombinationGard(
    Atoms &atom, AnyNeighborhoods &nbhs, double dLdE,
    double *out_gard_accumulator)
{
    Eigen::Matrix<double, -1, 3> zero_force(atom.size(), 3);
    zero_force.setZero();
    Eigen::Matrix3d zero_stress = Eigen::Matrix3d::Zero();
    AccumulateEFSCombinationGard(atom, nbhs, dLdE, zero_force, zero_stress,
                                 out_gard_accumulator);
}

void Potentials::PotentialHybrid::AccumulateEFSCombinationGardRonly(Atoms &atom, AnyNeighborhoods &nbhs, double dLdE, Eigen::Matrix<double, -1, 3> &dLdF, Eigen::Matrix3d &dLdS, double *out_gard_accumulator)
{
    int roffset = 0;
    for (int it = 0; it < train_ids.size(); ++it) {
        auto &i = train_ids[it];
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[i].get());
        auto snbhs = nbhs.sub_neighborhoods(ppots[i]->cutoff);
        // species coeffs is handled on our own
        p_mtp->AccumulateEFSCombinationGardRonly(atom, snbhs, dLdE, dLdF, dLdS, out_gard_accumulator, roffset);
        roffset += p_mtp->radial_coeff_size;
    }
}

void Potentials::PotentialHybrid::AccumulateEFSCombinationGardLonly(Atoms &atom, AnyNeighborhoods &nbhs, double dLdE, Eigen::Matrix<double, -1, 3> &dLdF, Eigen::Matrix3d &dLdS, double *out_gard_accumulator)
{
    if (train_ids.empty()) return;
    int loffset = species_count;
    for (int it = 0; it < train_ids.size(); ++it) {
        auto &i = train_ids[it];
        PotMTP *p_mtp = dynamic_cast<PotMTP*>(ppots[i].get());
        auto snbhs = nbhs.sub_neighborhoods(ppots[i]->cutoff);
        // species coeffs is handled on our own
        p_mtp->AccumulateEFSCombinationGardLonly(atom, snbhs, dLdE, dLdF, dLdS, out_gard_accumulator, loffset);
        loffset += p_mtp->alpha_scalar_moments;
    }
    for (int type : atom.chemmap_) {
        if (type < 0 || type >= species_count) {
            GPERROR("hybrid: atom species index is out of range");
        }
        out_gard_accumulator[type] += dLdE;
    }
}

bool Potentials::PotentialHybrid::supports_analytical_cell_hessian() const {
    return !ppots.empty() && std::all_of(ppots.begin(), ppots.end(), [](const auto &pot) {
        if (!pot || !pot->supports_analytical_cell_hessian()) return false;
        const auto *mtp = dynamic_cast<const PotMTP *>(pot.get());
        // These legacy corrections change the native affine/residual blocks.
        // A hybrid sum exposes the uncorrected provider contract only.
        return !mtp || (!mtp->hc_correction && !mtp->rR_correction);
    });
}

void Potentials::PotentialHybrid::getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    if (!supports_analytical_cell_hessian())
        GPERROR("hybrid: every sub-potential must support uncorrected analytical cell Hessians");
    if (nbhs.size() != atom.size()) GPERROR("Hessian neighborhoods have the wrong size");
    if (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible())
        GPERROR("Analytical hybrid cell Hessian requires a finite invertible cell");
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    HessianMatrix result, child_hessian;
    result.resize(atom.size());
    Eigen::MatrixXd cell = Eigen::MatrixXd::Zero(9, 9);
    Eigen::MatrixXd cross = Eigen::MatrixXd::Zero(9, 3 * atom.size());
    for (auto &pot : ppots) {
        auto child_nbhs = nbhs.sub_neighborhoods(pot->cutoff);
        Eigen::MatrixXd child_cell, child_cross;
        pot->getEFSHessianCellOMP(atom, child_nbhs, child_hessian, child_cell, child_cross);
        if (child_hessian.size != atom.size() || child_cell.rows() != 9 || child_cell.cols() != 9
            || child_cross.rows() != 9 || child_cross.cols() != 3 * atom.size())
            GPERROR("hybrid: invalid analytical cell Hessian child dimensions");
        for (int i = 0; i < atom.size(); ++i)
            for (int j = 0; j < atom.size(); ++j)
                result(i, j) += child_hessian(i, j);
        cell += child_cell;
        cross += child_cross;
    }
    hessian = std::move(result);
    cellhess = std::move(cell);
    lrhess = std::move(cross);
}

void Potentials::PotentialHybrid::getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}
void Potentials::PotentialHybrid::getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec) {
    getEFSHessianCellSymmOMP(atom, hessian, cellhess, lrhess, symprec);
}

void Potentials::PotentialHybrid::getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}

void Potentials::PotentialHybrid::getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}
