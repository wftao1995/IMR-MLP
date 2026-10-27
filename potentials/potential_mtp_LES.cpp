// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_mtp_LES.h"
#include "utils/compressed_stream.h"

#include <memory>
#include <set>

namespace Potentials {

thread_local bool PotMTP_LES::allow_internal_site_evaluation_ = false;

void PotMTP_LES::clear_failed_load() noexcept {
    delete[] les_species_coeffs;
    delete[] les_moment_coeffs;
    les_species_coeffs = les_moment_coeffs = nullptr;
    lr_les = dEdq_valid_ = false;
    les_kernel.clear();
    sog_channels = allocated_channels_ = 1;
    model_les.kernel = LES::Kernel::Coulomb;
    model_les.sog_weights.clear();
    model_les.sog_log_bandwidths.clear();
    LES_all_qs.resize(0);
    LES_all_latents.resize(0, 0);
    cached_dEdq_.resize(0);
    clear_loaded_model();
}



// ═══════════════════════════════════════════════════════════════════════
//  Load / Save
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
//  Settings-based constructor (handles header-only templates)
// ═══════════════════════════════════════════════════════════════════════

PotMTP_LES::PotMTP_LES(const std::string &potfile, GPSettings &settings) : PotMTP() {
    InitSettings();
    LES::reject_obsolete_settings(settings);
    // Preserve LES scalar parameters from header/template files even when the
    // full PotMTP load later throws on missing coefficients.
    {
        auto ifs = open_ifile(potfile);
        std::string tmpstr;
        while (ifs && (*ifs >> tmpstr)) {
            LES::reject_obsolete_parameter(tmpstr);
            if (tmpstr == "les_kernel") { ifs->ignore(1000, '='); *ifs >> les_kernel; }
            else if (tmpstr == "sog_channels") { ifs->ignore(1000, '='); *ifs >> sog_channels; }
            else if (tmpstr == "sog_terms") { ifs->ignore(1000, '='); *ifs >> sog_terms; }
            else if (tmpstr == "sog_log_bandwidth_min") { ifs->ignore(1000, '='); *ifs >> sog_log_bandwidth_min; }
            else if (tmpstr == "sog_log_bandwidth_max") { ifs->ignore(1000, '='); *ifs >> sog_log_bandwidth_max; }
            else if (tmpstr == "les_version") { ifs->ignore(1000, '='); *ifs >> les_version; }
            else if (tmpstr == "lr_sigma" || tmpstr == "les_sigma") { ifs->ignore(1000, '='); *ifs >> lr_sigma; }
            else if (tmpstr == "les_dl") { ifs->ignore(1000, '='); *ifs >> dl; }
            else if (tmpstr == "les_taper_fraction") { ifs->ignore(1000, '='); *ifs >> les_taper_fraction; }
            else if (tmpstr == "les_cutoff_mode") { ifs->ignore(1000, '='); *ifs >> les_cutoff_mode; }
            else if (tmpstr == "les_charge_treatment") { ifs->ignore(1000, '='); *ifs >> les_charge_treatment; }
            else if (tmpstr == "les_init_range") { ifs->ignore(1000, '='); *ifs >> les_init_range; }
            else if (tmpstr == "les_balance_initial_charges") { ifs->ignore(1000, '='); *ifs >> les_balance_initial_charges; }
            else if (tmpstr == "remove_self_interaction") { ifs->ignore(1000, '='); *ifs >> remove_self_interaction; }
            else if (tmpstr == "les_response_parity") { ifs->ignore(1000, '='); *ifs >> les_response_parity; }
        }
    }
    // Complete graph templates may omit fitted coefficients. A true parse
    // failure is not a usable template and propagates after Load's cleanup.
    auto input = open_ifile(potfile);
    if (!input) GPERROR("Cannot open potential file: " + potfile);
    Load(*input);
    try {
        ApplySettings(settings);
        sync_les_engine();
    } catch (...) {
        // The derived destructor is not called if construction fails.
        delete[] les_species_coeffs;
        delete[] les_moment_coeffs;
        les_species_coeffs = les_moment_coeffs = nullptr;
        throw;
    }
}

void PotMTP_LES::InitSettings() {
    PotMTP::InitSettings();
    Connect(les_kernel, "les_kernel");
    Connect(model_les.use_recurrence, "les_recurrence");
    Connect(model_les.use_cache, "les_cache");
    Connect(sog_terms, "sog_terms");
    Connect(sog_channels, "sog_channels");
    Connect(sog_log_bandwidth_min, "sog_log_bandwidth_min");
    Connect(sog_log_bandwidth_max, "sog_log_bandwidth_max");
    Connect(lr_sigma, "lr_sigma");
    Connect(dl, "les_dl");
    Connect(les_taper_fraction, "les_taper_fraction");
    Connect(les_cutoff_mode, "les_cutoff_mode");
    Connect(les_init_range, "les_init_range");
    Connect(les_balance_initial_charges, "les_balance_initial_charges");
    Connect(les_hessian_fd_nudge, "les_hessian_fd_nudge");
    Connect(remove_self_interaction, "remove_self_interaction");
    Connect(les_response_parity, "les_response_parity");
}

void PotMTP_LES::validate_response_parity() const {
    if (les_response_parity != 1 && les_response_parity != -1)
        GPERROR("LES les_response_parity must be +1 or -1");
}

void PotMTP_LES::sync_les_engine() {
    if (les_kernel != "coulomb" && les_kernel != "sog")
        GPERROR("Unknown LES kernel: " + les_kernel);
    if(sog_channels<1 || sog_channels>16 || (!uses_sog() && sog_channels!=1))
        GPERROR("SOG channels must be in [1,16]; Coulomb LES requires one channel");
    if(allocated_channels_!=channels() && les_species_coeffs && les_moment_coeffs){
        auto species=std::make_unique<double[]>(latent_species_count());
        auto moments=std::make_unique<double[]>(latent_moment_count());
        std::copy_n(les_species_coeffs,std::min(allocated_channels_,channels())*species_count,species.get());
        std::copy_n(les_moment_coeffs,std::min(allocated_channels_,channels())*alpha_scalar_moments,moments.get());
        delete[] les_species_coeffs;delete[] les_moment_coeffs;
        les_species_coeffs=species.release();les_moment_coeffs=moments.release();
        allocated_channels_=channels();
    }
    if(uses_sog() && model_les.sog_channels==1 && channels()>1
        && model_les.sog_weights.size()==size_t(sog_terms)){
        const auto w=model_les.sog_weights,u=model_les.sog_log_bandwidths;
        for(int c=1;c<channels();++c){model_les.sog_weights.insert(model_les.sog_weights.end(),w.begin(),w.end());model_les.sog_log_bandwidths.insert(model_les.sog_log_bandwidths.end(),u.begin(),u.end());}
    }
    model_les.sog_channels=channels();model_les.sog_channel=0;
    model_les.kernel=uses_sog() ? LES::Kernel::SOG : LES::Kernel::Coulomb;
    if (uses_sog()) {
        if (les_balance_initial_charges) GPERROR("Charge balancing is not defined for generic SOG latent variables");
        if (sog_terms<1 || sog_terms>256 || !std::isfinite(sog_log_bandwidth_min)
            || !std::isfinite(sog_log_bandwidth_max)
            || sog_log_bandwidth_min < -30.0 || sog_log_bandwidth_max > 30.0
            || sog_log_bandwidth_min >= sog_log_bandwidth_max)
            GPERROR("Invalid SOG term count or log bandwidth bounds");
        if (model_les.sog_weights.empty() && model_les.sog_log_bandwidths.empty()) {
            model_les.sog_weights.assign(kernel_count(),1.0/sog_terms);
            for (int m=0;m<kernel_count();++m)
                model_les.sog_log_bandwidths.push_back(sog_log_bandwidth_min
                    +(sog_log_bandwidth_max-sog_log_bandwidth_min)*((m%sog_terms)+1.0)/(sog_terms+1.0));
        }
        if (model_les.sog_weights.size()!=static_cast<size_t>(kernel_count())
            || model_les.sog_log_bandwidths.size()!=static_cast<size_t>(kernel_count()))
            GPERROR("SOG parameter count differs from sog_terms");
        for (double u:model_les.sog_log_bandwidths)
            if (!(u>sog_log_bandwidth_min && u<sog_log_bandwidth_max))
                GPERROR("SOG log bandwidth must be strictly inside its saved bounds");
    }
    validate_response_parity();
    model_les.lr_sigma = lr_sigma;
    model_les.dl = dl;
    model_les.taper_fraction = les_taper_fraction;
    model_les.cutoff_mode = LES::parse_cutoff_mode(les_cutoff_mode);
    model_les.remove_self_interaction = remove_self_interaction;
    model_les.validate();
}

void PotMTP_LES::Load(std::istream &is) try {
    dEdq_valid_=false;
    lr_les=false;
    LES_all_qs.resize(0);
    cached_dEdq_.resize(0);
    delete[] les_species_coeffs;
    delete[] les_moment_coeffs;
    les_species_coeffs=nullptr;
    les_moment_coeffs=nullptr;
    // Load the short-range graph first, then reconstruct the long-range head
    // from defaults and the file. Parse failures clear both parts below.
    PotMTP::Load(is);
    const PotMTP_LES defaults;
    les_version=defaults.les_version;
    les_kernel=defaults.les_kernel;
    sog_terms=defaults.sog_terms;
    sog_channels=1;allocated_channels_=1;
    sog_log_bandwidth_min=defaults.sog_log_bandwidth_min;
    sog_log_bandwidth_max=defaults.sog_log_bandwidth_max;
    model_les.kernel=LES::Kernel::Coulomb;
    model_les.sog_weights.clear();
    model_les.sog_log_bandwidths.clear();
    model_les.sog_include_zero=true;
    lr_sigma=defaults.lr_sigma;
    dl=defaults.dl;
    les_taper_fraction=defaults.les_taper_fraction;
    les_cutoff_mode=defaults.les_cutoff_mode;
    les_charge_treatment=defaults.les_charge_treatment;
    les_init_range=defaults.les_init_range;
    les_balance_initial_charges=defaults.les_balance_initial_charges;
    remove_self_interaction=defaults.remove_self_interaction;
    les_response_parity=defaults.les_response_parity;

    // 2. Reset stream to beginning and scan for LES sections
    is.clear();
    is.seekg(0, std::ios::beg);

    std::unique_ptr<double[]> species, moments;
    const auto read_coefficients = [&](int count, const char *field) {
        if (count <= 0) GPERROR(std::string("Invalid coefficient count for ") + field);
        auto values = std::make_unique<double[]>(count);
        const auto expect = [&](char expected) {
            char actual;
            if (!(is >> actual) || actual != expected)
                GPERROR(std::string("Malformed LES coefficient block: ") + field);
        };
        expect('=');
        expect('{');
        for (int i=0; i<count; ++i) {
            if (i) expect(',');
            if (!(is >> values[i]) || !std::isfinite(values[i]))
                GPERROR(std::string("Invalid or missing LES coefficient: ") + field);
        }
        is >> std::ws;
        if (is.peek() == ',') is.get(); // Both trailing-comma conventions are valid.
        expect('}');
        return values;
    };
    std::string tmpstr;
    bool found_species = false, found_moment = false, found_version = false;

    std::set<std::string> sog_fields;
    bool found_legacy_self=false, found_charge_treatment=false;
    while (is >> tmpstr) {
        LES::reject_obsolete_parameter(tmpstr);
        if (tmpstr == "les_kernel" || tmpstr.starts_with("sog_")) {
            if (!sog_fields.insert(tmpstr).second) GPERROR("Duplicate field: "+tmpstr);
            if (tmpstr=="sog_weights" || tmpstr=="sog_log_bandwidths") {
                if (!sog_fields.contains("sog_terms") || sog_terms<1 || sog_terms>256)
                    GPERROR("SOG arrays require preceding sog_terms in [1,256]");
                auto values=read_coefficients(kernel_count(),tmpstr.c_str());
                auto& target=tmpstr=="sog_weights" ? model_les.sog_weights : model_les.sog_log_bandwidths;
                target.assign(values.get(),values.get()+kernel_count());
            } else {
                char eq; if (!(is>>eq) || eq!='=') GPERROR("Malformed SOG field: "+tmpstr);
                if (tmpstr=="les_kernel") is>>les_kernel;
                else if (tmpstr=="sog_terms") is>>sog_terms;
                else if (tmpstr=="sog_log_bandwidth_min") is>>sog_log_bandwidth_min;
                else if (tmpstr=="sog_log_bandwidth_max") is>>sog_log_bandwidth_max;
                else if (tmpstr=="sog_channels") {
                    if (!(is>>sog_channels) || sog_channels<1 || sog_channels>16
                        || (sog_channels>1 && (!uses_sog() || sog_fields.contains("sog_weights")
                            || sog_fields.contains("sog_log_bandwidths") || found_species || found_moment)))
                        GPERROR("Invalid SOG channel count/order");
                } else if (tmpstr=="sog_zero_mode") {
                    std::string v; is>>v;
                    if (v!="include" && v!="omit") GPERROR("Unknown SOG zero mode");
                    model_les.sog_include_zero=v=="include";
                } else if (tmpstr=="sog_self_mode") {
                    std::string v; is>>v;
                    if (v!="onsite" && v!="none") GPERROR("Unknown SOG self mode");
                    remove_self_interaction=v=="onsite";
                } else GPERROR("Unknown SOG field: "+tmpstr);
            }
        } else if (tmpstr == "les_version") {
            if (found_version) GPERROR("Duplicate les_version field");
            is.ignore(1000, '=');
            is >> les_version;
            found_version = true;
        } else if (tmpstr == "lr_sigma" || tmpstr == "les_sigma") {
            is.ignore(1000, '=');
            is >> lr_sigma;
        } else if (tmpstr == "les_dl") {
            is.ignore(1000, '=');
            is >> dl;
        } else if (tmpstr == "les_taper_fraction") {
            is.ignore(1000, '=');
            is >> les_taper_fraction;
        } else if (tmpstr == "les_cutoff_mode") {
            is.ignore(1000, '=');
            is >> les_cutoff_mode;
        } else if (tmpstr == "les_charge_treatment") {
            found_charge_treatment=true;
            is.ignore(1000, '=');
            is >> les_charge_treatment;
        } else if (tmpstr == "les_init_range") {
            is.ignore(1000, '=');
            is >> les_init_range;
        } else if (tmpstr == "les_balance_initial_charges") {
            is.ignore(1000, '=');
            is >> les_balance_initial_charges;
        } else if (tmpstr == "remove_self_interaction") {
            found_legacy_self=true;
            is.ignore(1000, '=');
            is >> remove_self_interaction;
        } else if (tmpstr == "les_response_parity") {
            is.ignore(1000, '=');
            is >> les_response_parity;
        } else if (tmpstr == "les_species_coeffs") {
            if (found_species) GPERROR("Duplicate les_species_coeffs block");
            species=read_coefficients(latent_species_count(), "les_species_coeffs");
            found_species=true;
        } else if (tmpstr == "les_moment_coeffs") {
            if (found_moment) GPERROR("Duplicate les_moment_coeffs block");
            moments=read_coefficients(latent_moment_count(), "les_moment_coeffs");
            found_moment=true;
        }
        if (is.fail()) GPERROR("Invalid or missing LES field value: " + tmpstr);
    }
    const bool has_les = found_species || found_moment;

    if (has_les && (!found_species || !found_moment)) {
        // Partial LES parameters → error
        GPERROR("Incomplete LES parameters in potential file");
    }
    if (uses_sog()) {
        if (found_legacy_self || found_charge_treatment)
            GPERROR("SOG requires sog_self_mode and sog_zero_mode, not legacy LES self/charge fields");
        for (const auto* field:{"les_kernel","sog_terms","sog_weights","sog_log_bandwidths",
                "sog_log_bandwidth_min","sog_log_bandwidth_max","sog_zero_mode","sog_self_mode"})
            if (!sog_fields.contains(field)) GPERROR(std::string("Missing SOG field: ")+field);
        if (!has_les) GPERROR("SOG model requires latent coefficient blocks");
    } else if (!sog_fields.empty() && !(sog_fields.size()==1 && sog_fields.contains("les_kernel")))
        GPERROR("SOG fields require les_kernel=sog");
    if (has_les && (!found_version || les_version != (uses_sog() ? (channels()==1 ? 3 : 4) : LES::format_version)))
        GPERROR("Unversioned or unsupported MTP+LES potential");
    if (has_les && !uses_sog() && les_charge_treatment != "uniform_background")
        GPERROR("LES v2 supports les_charge_treatment=uniform_background only");
    if (!has_les) {
        species=std::make_unique<double[]>(species_count);
        moments=std::make_unique<double[]>(alpha_scalar_moments);
    }
    sync_les_engine();
    les_species_coeffs=species.release();
    les_moment_coeffs=moments.release();
    allocated_channels_=channels();
    lr_les=has_les;
    if(potential_tag.text("type")=="MTP_LES") {
        const auto expected=model_metadata();
        const bool tagged=potential_tag.fields.count("long_range");
        MTPMetadata::require(tagged==lr_les,"long_range conflicts with LES body");
        if(tagged) for(const auto &key:{"kernel","channels"})
            MTPMetadata::require(potential_tag.fields.at("long_range").get(key)==expected.at("long_range").get(key),std::string("long_range.")+key+" conflicts with LES body");
    }
}

catch (...) {
    clear_failed_load();
    throw;
}

void PotMTP_LES::Save(std::ostream &os) {
    validate_response_parity();
    PotMTP::Save(os);

    if (!lr_les) return;
    sync_les_engine();
    os << "les_version = " << (uses_sog() ? (channels()==1 ? 3 : 4) : LES::format_version) << "\n";
    if (uses_sog()) {
        os << "les_kernel = sog\nsog_channels = " << channels() << "\nsog_terms = " << sog_terms << "\n";
        os << "sog_log_bandwidth_min = " << sog_log_bandwidth_min << "\n";
        os << "sog_log_bandwidth_max = " << sog_log_bandwidth_max << "\n";
        os << "sog_zero_mode = " << (model_les.sog_include_zero ? "include" : "omit") << "\n";
        os << "sog_self_mode = " << (remove_self_interaction ? "onsite" : "none") << "\n";
        os << "sog_weights = { ";
        for (double w:model_les.sog_weights) os << w << ", ";
        os << "}\nsog_log_bandwidths = { ";
        for (double u:model_les.sog_log_bandwidths) os << u << ", ";
        os << "}\n";
    }
    if (!uses_sog()) os << "les_sigma = " << lr_sigma << "\n";
    os << "les_dl = " << dl << "\n";
    os << "les_cutoff_mode = " << les_cutoff_mode << "\n";
    os << "les_taper_fraction = " << les_taper_fraction << "\n";
    if (!uses_sog()) os << "les_charge_treatment = uniform_background\n";
    os << "les_init_range = " << les_init_range << "\n";
    if (!uses_sog()) os << "remove_self_interaction = " << remove_self_interaction << "\n";
    os << "les_response_parity = " << les_response_parity << "\n";

    os << "les_species_coeffs = { ";
    for (int i = 0; i < latent_species_count(); ++i)
        os << les_species_coeffs[i] << ", ";
    os << "}\n";

    os << "les_moment_coeffs = { ";
    for (int i = 0; i < latent_moment_count(); ++i)
        os << les_moment_coeffs[i] << ", ";
    os << "}\n";
}

// ═══════════════════════════════════════════════════════════════════════
//  Stage 1 — compute latent charges q_i for all atoms
// ═══════════════════════════════════════════════════════════════════════

void PotMTP_LES::compute_les_charges(Atoms &atom, AnyNeighborhoods &nbhs,
                                       Eigen::VectorXd &qs, int channel) {
    const int n_atoms = atom.size();
    qs.resize(n_atoms);

    for (int i = 0; i < n_atoms; ++i) {
        Neighborhood &nbh = nbhs[i];
        CalcSiteBasis(nbh);

        double q = les_species_coeffs[channel*species_count+nbh.centertype];
        for (int a = 0; a < alpha_scalar_moments; ++a) {
            q += les_moment_coeffs[channel*alpha_scalar_moments+a] * moment_vals[alpha_moment_mapping[a]];
        }
        qs(i) = q;
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Stage 3 — indirect forces via descriptor backpropagation (single atom)
// ═══════════════════════════════════════════════════════════════════════

void PotMTP::add_latent_charge_forces(int center_index, Neighborhood &nbh, double dEdq_i,
    const double *charge_moments, Atoms &atom) {
    // Recompute descriptors + Jacobian for this atom
    CalcSiteBasisDers(nbh);

    // Step 1 — initialise dE/d(moment) from dE/dq_i × les_moment
    // Internal charge: raw_q_i = les_species[type_i] + Σ les_moment[a] * B_a
    // dEdq_i = ∂E_lr / ∂(raw_q_i)   (computed by add_latent_ewald_efs using raw charges)
    // d(raw_q_i)/dB_a = les_moment[a]
    // → dE_lr/dB_a = dEdq_i * les_moment[a]
    std::ranges::fill(site_energy_ders_wrt_moments_, 0);
    for (int a = 0; a < alpha_scalar_moments; ++a) {
        int idx = alpha_moment_mapping[a];
        site_energy_ders_wrt_moments_[idx] = dEdq_i * charge_moments[a];
    }

    // Step 2 — backpropagate through product rules (shared helper)
    backpropagate_moments();

    // Step 3 — contract with moment_jacobian → forces + stress
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        Eigen::Vector3d f_contrib = Eigen::Vector3d::Zero();
        for (int k = 0; k < alpha_index_basic_count; ++k) {
            double dEdM = site_energy_ders_wrt_moments_[k];
            f_contrib[0] += dEdM * moment_jacobian(k, j, 0);
            f_contrib[1] += dEdM * moment_jacobian(k, j, 1);
            f_contrib[2] += dEdM * moment_jacobian(k, j, 2);
        }
        atom.forces_[center_index] += f_contrib;
        atom.forces_[nbh.indices_[j]] -= f_contrib;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                atom.stress_(a, b) -= f_contrib[a] * nbh.pos_[j](b);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  getEFS — serial path
// ═══════════════════════════════════════════════════════════════════════

void PotMTP_LES::append_les(Atoms &atom, AnyNeighborhoods &nbhs, bool need_forces) {
    if (!lr_les) return;
    validate_response_parity();
    LES::validate_periodic_geometry(atom);
    dEdq_valid_=false;
    atom.charges_.clear();LES_all_latents.resize(atom.size(),channels());
    for(int channel=0;channel<channels();++channel){
        LES::ChannelScope selected(model_les,channel);
        compute_les_charges(atom,nbhs,LES_all_qs,channel);
        LES_all_latents.col(channel)=LES_all_qs;
        auto result=model_les.evaluate(atom,LES_all_qs);
        atom.energy_+=result.energy;
        if(atom.site_energies_.size()!=size_t(atom.size()))atom.site_energies_.resize(atom.size(),0.);
        for(int i=0;i<atom.size();++i)atom.site_energies_[i]+=result.per_atom_energy(i);
        if(channel==0)cached_dEdq_=result.dEdq;
        if(need_forces){
            atom.stress_+=result.virial;
            for(int i=0;i<atom.size();++i){
                atom.forces_[i]+=result.forces.row(i).transpose();
                add_latent_charge_forces(i,nbhs[i],result.dEdq(i),les_moment_coeffs+channel*alpha_scalar_moments,atom);
            }
        }
    }
    LES_all_qs=LES_all_latents.col(0);dEdq_valid_=channels()==1;
    if(!uses_sog())for(int i=0;i<atom.size();++i)atom.charges_.push_back(les_response_parity*LES_all_qs(i));

}

void PotMTP_LES::getEFS(Atoms &atom) { require_loaded_model();
    if (lr_les) LES::validate_periodic_geometry(atom);
    Neighborhoods nbhs(atom, cutoff);
    getEFS(atom, nbhs);
}

void PotMTP_LES::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (lr_les) LES::validate_periodic_geometry(atom);
    allow_internal_site_evaluation_ = true;
    try { PotMTP::getEFS(atom, nbhs); }
    catch (...) { allow_internal_site_evaluation_ = false; throw; }
    allow_internal_site_evaluation_ = false;
    append_les(atom, nbhs, true);
}

// ═══════════════════════════════════════════════════════════════════════
//  getEFSOMP — OpenMP path
// ═══════════════════════════════════════════════════════════════════════

void PotMTP_LES::getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (lr_les) LES::validate_periodic_geometry(atom);
    PotMTP::getEFSOMP(atom, nbhs);
    append_les(atom, nbhs, true);
}

void PotMTP_LES::getEFSOMP(Atoms &atom) { require_loaded_model(); if (lr_les) LES::validate_periodic_geometry(atom); Neighborhoods nbhs(atom, cutoff); getEFSOMP(atom, nbhs); }
void PotMTP_LES::getEFSSymmOMP(Atoms &atom, double symprec) {
    if (lr_les) getEFSOMP(atom);
    else PotMTP::getEFSSymmOMP(atom, symprec);
}
void PotMTP_LES::getEFSOMPthread(Atoms &atom) { require_loaded_model(); if (lr_les) LES::validate_periodic_geometry(atom); Neighborhoods nbhs(atom, cutoff); getEFSOMPthread(atom, nbhs); }
void PotMTP_LES::getEFSOMPthread(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (lr_les) LES::validate_periodic_geometry(atom);
    PotMTP::getEFSOMPthread(atom, nbhs);
#pragma omp critical(mtp_les_shared_evaluation)
    { append_les(atom, nbhs, true); }
}

// ═══════════════════════════════════════════════════════════════════════
//  getE — energy-only evaluation (must include LES for numerical diff)
// ═══════════════════════════════════════════════════════════════════════

void PotMTP_LES::getE(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (lr_les) LES::validate_periodic_geometry(atom);
    allow_internal_site_evaluation_ = true;
    try { PotMTP::getE(atom, nbhs); }
    catch (...) { allow_internal_site_evaluation_ = false; throw; }
    allow_internal_site_evaluation_ = false;
    append_les(atom, nbhs, false);
}

void PotMTP_LES::getE(Atoms &atom) { require_loaded_model(); if (lr_les) LES::validate_periodic_geometry(atom); Neighborhoods nbhs(atom, cutoff); getE(atom, nbhs); }
void PotMTP_LES::getEOMP(Atoms &atom) { require_loaded_model(); if (lr_les) LES::validate_periodic_geometry(atom); Neighborhoods nbhs(atom, cutoff); getEOMP(atom, nbhs); }
void PotMTP_LES::getEOMP(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (lr_les) LES::validate_periodic_geometry(atom);
    PotMTP::getEOMP(atom, nbhs);
    append_les(atom, nbhs, false);
}

void PotMTP_LES::CalcSiteEFS(Neighborhood &nbh, double &energy,
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &ders) {
    if (allow_internal_site_evaluation_) {
        PotMTP::CalcSiteEFS(nbh, energy, ders);
        return;
    }
    GPERROR("MTP+LES site EFS is undefined because Ewald energy is global");
}
void PotMTP_LES::CalcSiteE(Neighborhood &nbh, double &energy) {
    if (allow_internal_site_evaluation_) {
        PotMTP::CalcSiteE(nbh, energy);
        return;
    }
    GPERROR("MTP+LES site energy is undefined for an isolated neighborhood");
}

void PotMTP_LES::getEFSHessian(Atoms &atom, HessianMatrix &hessian) {
    if (lr_les) LES::validate_periodic_geometry(atom);
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessian(atom, nbhs, hessian);
}
void PotMTP_LES::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    require_loaded_model();
    assemble_cartesian_hessian(*this,atom,nbhs,hessian);
}
void PotMTP_LES::getEFSHessianSymm(Atoms &a, HessianMatrix &h, double) { getEFSHessian(a, h); }
void PotMTP_LES::getEFSHessianSymmOMP(Atoms &a, HessianMatrix &h, double) { getEFSHessian(a, h); }
void PotMTP_LES::getEFSHessianOMP(Atoms &a, AnyNeighborhoods &n, HessianMatrix &h) { getEFSHessian(a, n, h); }
void PotMTP_LES::getHessian(Atoms &a, AnyNeighborhoods &n,
    std::vector<std::pair<int,int>> inds, HessianSparse &out) {
    assemble_requested_hessian(*this,a,n,inds,out);
}
void PotMTP_LES::getEFSHessianSymmDiaOMP(Atoms &a,std::vector<double> &out,double) {
    Neighborhoods nbhs(a,cutoff);assemble_cartesian_diagonal(*this,a,nbhs,out);
}

void PotMTP_LES::applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction, Eigen::VectorXd &product) {
    require_loaded_model();
    if (!supports_analytical_hessian_vector_product()) GPERROR("Unsupported MTP+LES HVP");
    if (direction.size() != 3*atom.size() || !direction.allFinite() || nbhs.size() != atom.size())
        GPERROR("Invalid MTP+LES Hessian-vector direction or neighborhoods");
    if (lr_les) LES::validate_periodic_geometry(atom);
    Eigen::VectorXd result;
    PotMTP::applyHessianVectorProduct(atom, nbhs, direction, result);
    if (lr_les) {
        sync_les_engine();
        result += latent_cartesian_hvp(*this,model_les,atom,nbhs,direction,
            les_species_coeffs,les_moment_coeffs,channels());
    }
    product = std::move(result);
}

void PotMTP_LES::getEFSHessianCell(Atoms &atom,HessianMatrix &h,Eigen::MatrixXd &cell,Eigen::MatrixXd &cross) {
    Neighborhoods nbhs(atom,cutoff);getEFSHessianCell(atom,nbhs,h,cell,cross);
}
void PotMTP_LES::getEFSHessianCell(Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &h,
    Eigen::MatrixXd &cell,Eigen::MatrixXd &cross) {
    require_loaded_model();
    if(!supports_analytical_cell_hessian()) GPERROR("MTP+LES native cell Hessian requires smooth reciprocal cutoff");
    if(lr_les) sync_les_engine();
    const LocalCartesianHVP short_range=[this](Neighborhood& n,const SiteCoordinateMatrix& v,SiteCoordinateMatrix& out) {
        CalcSiteHessianVectorProduct(n,v,out);
    };
    assemble_latent_native_hessian(*this,*this,model_les,atom,nbhs,les_species_coeffs,les_moment_coeffs,
        channels(),lr_les,short_range,h,cell,cross);
}
void PotMTP_LES::getEFSHessianCellSymm(Atoms &a,HessianMatrix &h,Eigen::MatrixXd &c,Eigen::MatrixXd &x,double) {
    getEFSHessianCell(a,h,c,x);
}
void PotMTP_LES::getEFSHessianCellSymmOMP(Atoms &a,HessianMatrix &h,Eigen::MatrixXd &c,Eigen::MatrixXd &x,double) {
    getEFSHessianCell(a,h,c,x);
}
void PotMTP_LES::getEFSHessianCellOMP(Atoms &a,AnyNeighborhoods &n,HessianMatrix &h,Eigen::MatrixXd &c,Eigen::MatrixXd &x) {
    getEFSHessianCell(a,n,h,c,x);
}


// ═══════════════════════════════════════════════════════════════════════
//  Training overrides
// ═══════════════════════════════════════════════════════════════════════

int PotMTP_LES::get_NL_coeffs_count() const {
    return reg_coeff_size + latent_species_count() + latent_moment_count() + (uses_sog() ? 2*kernel_count() : 0);
}

void PotMTP_LES::nl_get_coeffs(double *x) const {
    // 1. SR part (radial + species + moment)
    PotMTP::nl_get_coeffs(x);
    // 2. LES species coefficients
    int off = reg_coeff_size;
    for (int i = 0; i < latent_species_count(); ++i)
        x[off + i] = les_species_coeffs[i];
    off += latent_species_count();
    // 3. LES moment coefficients
    for (int i = 0; i < latent_moment_count(); ++i)
        x[off + i] = les_moment_coeffs[i];
    off += latent_moment_count();
    if (uses_sog()) {
        const double mid=.5*(sog_log_bandwidth_max+sog_log_bandwidth_min);
        const double span=sog_log_bandwidth_max-sog_log_bandwidth_min;
        for (int m=0;m<kernel_count();++m) {
            x[off+m]=model_les.sog_weights[m];
            x[off+kernel_count()+m]=std::tan(M_PI*(model_les.sog_log_bandwidths[m]-mid)/span);
        }
    }
}

void PotMTP_LES::nl_set_coeffs(const double *x) {
    dEdq_valid_=false;
    if (uses_sog()) {
        for (int i=0;i<get_NL_coeffs_count();++i)
            if (!std::isfinite(x[i])) GPERROR("Non-finite SOG optimizer coefficient");
    }
    // 1. SR part
    PotMTP::nl_set_coeffs(x);
    // 2. LES species coefficients
    int off = reg_coeff_size;
    for (int i = 0; i < latent_species_count(); ++i)
        les_species_coeffs[i] = x[off + i];
    off += latent_species_count();
    // 3. LES moment coefficients
    for (int i = 0; i < latent_moment_count(); ++i)
        les_moment_coeffs[i] = x[off + i];
    off += latent_moment_count();
    if (uses_sog()) {
        const double mid=.5*(sog_log_bandwidth_max+sog_log_bandwidth_min);
        const double span=sog_log_bandwidth_max-sog_log_bandwidth_min;
        for (int m=0;m<kernel_count();++m) {
            model_les.sog_weights[m]=x[off+m];
            model_les.sog_log_bandwidths[m]=mid+span/M_PI*std::atan(x[off+kernel_count()+m]);
        }
        sync_les_engine();
    }
}

std::vector<NLParameterGroup> PotMTP_LES::nl_parameter_groups() const {
    std::vector<NLParameterGroup> groups={
        {"sr_radial", 0, radial_coeff_size},
        {"sr_linear", radial_coeff_size, reg_coeff_size},
        {"les_species", reg_coeff_size, reg_coeff_size + latent_species_count()},
        {"les_moment", reg_coeff_size + latent_species_count(),
                       reg_coeff_size + latent_species_count() + latent_moment_count()}
    };
    if (uses_sog()) {
        int off=reg_coeff_size+latent_species_count()+latent_moment_count();
        groups.push_back({"sog_weights",off,off+kernel_count()});
        groups.push_back({"sog_log_bandwidths",off+kernel_count(),off+2*kernel_count()});
    }
    return groups;
}

std::vector<ConditionalLinearBlock> PotMTP_LES::conditional_linear_blocks(double penalty_weight) const {
    if (typeid(*this) != typeid(PotMTP_LES)) return {};
    ConditionalLinearBlock sr; sr.name="sr"; sr.composition_count=species_count;
    for (int i=0; i<lin_coeff_size; ++i)
        sr.append(radial_coeff_size+i,
            i>=species_count && i<int(reg_L1_vector.size()) ? reg_L1_vector[i] : 0.,
            i>=species_count && i<int(reg_L2_vector.size()) ? reg_L2_vector[i] : 0.);
    std::vector<ConditionalLinearBlock> blocks{std::move(sr)};
    if (uses_sog()) {
        ConditionalLinearBlock sog; sog.name="sog";
        const int offset=reg_coeff_size+latent_species_count()+latent_moment_count();
        for (int i=0;i<kernel_count();++i) sog.append(offset+i,0.,penalty_weight);
        blocks.push_back(std::move(sog));
    }
    return blocks;
}

void PotMTP_LES::conditional_linear_set_coeffs(const std::vector<int>& indices, const Eigen::VectorXd& values) {
    if (values.size()!=static_cast<Eigen::Index>(indices.size()) || !values.allFinite())
        GPERROR("Invalid conditional linear LES update");
    const int sog_offset=reg_coeff_size+latent_species_count()+latent_moment_count();
    for (int index : indices)
        if (!(index>=radial_coeff_size && index<reg_coeff_size) &&
            !(uses_sog() && index>=sog_offset && index<sog_offset+kernel_count()))
            GPERROR("Invalid conditional linear LES index");
    for (int j=0;j<values.size();++j) {
        const int index=indices[j];
        if (index<reg_coeff_size) regression_coeffs[index]=values[j];
        else model_les.sog_weights[index-sog_offset]=values[j];
    }
    dEdq_valid_=false;
}

bool PotMTP_LES::supports_total_charge_training() const {
    return lr_les && !uses_sog() && les_species_coeffs && les_moment_coeffs;
}

double PotMTP_LES::predicted_total_charge(const Atoms &atom) const {
    if (uses_sog()) GPERROR("Generic SOG latent variables are not physical charges");
    if (atom.charges_.size() != static_cast<size_t>(atom.size()))
        GPERROR("MTP+LES total charge requested before getEFS");
    double total = 0.0;
    for (double q : atom.charges_)
        total += q;
    return total;
}

void PotMTP_LES::AccumulateTotalChargeGard(Atoms &atom,
    AnyNeighborhoods &nbhs, double dLdQ, double *out_gard) {
    if (!supports_total_charge_training())
        GPERROR("Total-charge training requires an active MTP+LES model");
    validate_response_parity();
    compute_les_charges(atom,nbhs,LES_all_qs);
    cached_dEdq_=model_les.apply_charge_hessian(atom,LES_all_qs);
    dEdq_valid_=true;

    const int n_atoms = atom.size();
    const double physical_dLdQ=dLdQ*les_response_parity;
    const int les_species_offset = reg_coeff_size;
    const int les_moment_offset = reg_coeff_size + species_count;
    for (int i = 0; i < n_atoms; ++i) {
        Neighborhood &nbh = nbhs[i];
        CalcSiteBasis(nbh);
        out_gard[les_species_offset + nbh.centertype] += physical_dLdQ;
        for (int a = 0; a < alpha_scalar_moments; ++a)
            out_gard[les_moment_offset + a] +=
                physical_dLdQ * moment_vals[alpha_moment_mapping[a]];
    }

    Eigen::Matrix<double, -1, 3> zero_force;
    Eigen::Matrix3d zero_stress = Eigen::Matrix3d::Zero();
    Eigen::VectorXd direct_dLdq = Eigen::VectorXd::Constant(n_atoms, physical_dLdQ);
    accumulate_les_sr_gradient(atom, nbhs, 0.0, zero_force, zero_stress,
                               out_gard, &direct_dLdq);
}

void PotMTP_LES::train_init() {
    dEdq_valid_=false;
    sync_les_engine();
    PotMTP::train_init();

    if (!les_species_coeffs || !les_moment_coeffs) return;
    // train_init must never overwrite loaded/checkpoint coefficients.
    // Explicit fit:random_init invokes random_init() through the trainer.
    lr_les = true;
    sync_les_engine();
}

void PotMTP_LES::set_sog_linear_weights(const Eigen::VectorXd& weights) {
    if(!uses_sog() || weights.size()!=kernel_count() || !weights.allFinite())
        GPERROR("Invalid linear SOG weight update");
    std::copy_n(weights.data(),weights.size(),model_les.sog_weights.data());
    dEdq_valid_=false;
    // LES compares physical weights when validating its kernel cache.
}

Eigen::MatrixXd PotMTP_LES::linear_sog_components(Atoms& atom,AnyNeighborhoods& nbhs) {
    if(!lr_les || !uses_sog()) GPERROR("SOG components require an active SOG model");
    validate_response_parity();
    const int n=atom.size(), count=channels();
    Eigen::MatrixXd q(n,count);
    std::vector<Eigen::MatrixXd> dq(n);
    // Cache dq_i / d r_ij for every channel in one descriptor traversal.
    for(int i=0;i<n;++i) {
        auto& nbh=nbhs[i];CalcSiteBasisDers(nbh);
        dq[i]=Eigen::MatrixXd::Zero(3*nbh.nbhcount_,count);
        for(int c=0;c<count;++c) {
            q(i,c)=les_species_coeffs[c*species_count+nbh.centertype];
            for(int a=0;a<alpha_scalar_moments;++a) {
                const double coefficient=les_moment_coeffs[c*alpha_scalar_moments+a];
                q(i,c)+=coefficient*basis_vals[a+1];
                for(int j=0;j<nbh.nbhcount_;++j) for(int d=0;d<3;++d)
                    dq[i](3*j+d,c)+=coefficient*basis_ders(a+1,j,d);
            }
        }
    }
    auto out=model_les.kernel_components(atom,q);
    for(int i=0;i<n;++i) {
        auto& nbh=nbhs[i];
        for(int c=0;c<count;++c) for(int m=c*sog_terms;m<(c+1)*sog_terms;++m)
            for(int j=0;j<nbh.nbhcount_;++j) for(int d=0;d<3;++d) {
                const double contribution=out.dEdq(i,m)*dq[i](3*j+d,c);
                out.efs(1+3*i+d,m)+=contribution;
                out.efs(1+3*nbh.indices_[j]+d,m)-=contribution;
                for(int e=0;e<3;++e) out.efs(1+3*n+3*d+e,m)-=contribution*nbh.pos_[j][e];
            }
    }
    dEdq_valid_=false;
    return out.efs;
}

void PotMTP_LES::random_init() {
    dEdq_valid_=false;
    sync_les_engine();
    PotMTP::random_init();

    if (!les_species_coeffs || !les_moment_coeffs) return;

    double init_range = (les_init_range > 0) ? les_init_range : random_upper;
    for (int i = 0; i < latent_species_count(); ++i)
        les_species_coeffs[i] = init_range * (2.0 * rand() / RAND_MAX - 1.0);
    for (int i = 0; i < latent_moment_count(); ++i)
        les_moment_coeffs[i] = init_range * (2.0 * rand() / RAND_MAX - 1.0);

    if (uses_sog()) {
        model_les.sog_weights.clear();
        model_les.sog_log_bandwidths.clear();
    }
    // Activate LES path now that coefficients are non-zero
    lr_les = true;
    sync_les_engine();

#ifdef IMRMLP_MPI
    MPI_Bcast(les_species_coeffs, latent_species_count(), MPI_DOUBLE, 0, mpi.comm);
    MPI_Bcast(les_moment_coeffs, latent_moment_count(), MPI_DOUBLE, 0, mpi.comm);
#endif
}

// ═══════════════════════════════════════════════════════════════════════
//  Gradient accumulation helpers
// ═══════════════════════════════════════════════════════════════════════

void PotMTP_LES::accumulate_les_energy_gradient(Atoms &atom,
    AnyNeighborhoods &nbhs, double dLdE, double *out_gard) {
    if (!lr_les) return;
    if (!dEdq_valid_ || cached_dEdq_.size()!=atom.size())
        GPERROR("LES energy gradient requested without a current dE/dq cache");

    const int n_atoms = atom.size();
    const int les_species_offset = reg_coeff_size;
    const int les_moment_offset  = reg_coeff_size + species_count;

    // Recompute descriptors (cached_dEdq_ already valid from getEFS)
    grad_moment_vals_cache_.resize(n_atoms, alpha_scalar_moments);
    grad_atom_types_.resize(n_atoms);

    for (int i = 0; i < n_atoms; ++i) {
        Neighborhood &nbh = nbhs[i];
        CalcSiteBasis(nbh);  // fills moment_vals
        grad_atom_types_[i] = nbh.centertype;
        for (int a = 0; a < alpha_scalar_moments; ++a)
            grad_moment_vals_cache_(i, a) = moment_vals[alpha_moment_mapping[a]];
    }

    // Energy gradient: raw_q_i = les_species[type_i] + Σ les_moment[a]·B_i[a]
    // dEdq = ∂E_lr/∂(raw_q_i)  (computed by add_latent_ewald_efs using raw charges)
    // dL/d(les_species[t]) = dLdE * Σ_{i:type_i=t} dEdq[i]
    // dL/d(les_moment[a])  = dLdE * Σ_i dEdq[i] * B_i[a]
    const auto& dEdq = cached_dEdq_;
    for (int i = 0; i < n_atoms; ++i)
        out_gard[les_species_offset + grad_atom_types_[i]] += dLdE * dEdq(i);

    for (int a = 0; a < alpha_scalar_moments; ++a) {
        double grad = 0.0;
        for (int i = 0; i < n_atoms; ++i)
            grad += dEdq(i) * grad_moment_vals_cache_(i, a);
        out_gard[les_moment_offset + a] += dLdE * grad;
    }
}

void PotMTP_LES::compute_indirect_force_vjp(Atoms& atom,
    AnyNeighborhoods& nbhs,
    const Eigen::Matrix<double, -1, 3>& dLdF,
    const Eigen::Matrix3d& dLdS)
{
    latent_charge_force_vjp(atom, nbhs, model_les, LES_all_qs, les_moment_coeffs,
        dLdF, dLdS, cached_dEdq_, force_charge_vjp_cache_,
        force_moment_explicit_vjp_cache_, &sog_response_);
    dEdq_valid_ = true;
}

void PotMTP::latent_charge_force_vjp(Atoms& atom, AnyNeighborhoods& nbhs,
    LES& model_les, const Eigen::VectorXd& charges,
    const double *charge_moments,
    const Eigen::Matrix<double, -1, 3>& dLdF, const Eigen::Matrix3d& dLdS,
    Eigen::VectorXd& dEdq, Eigen::VectorXd& charge_vjp,
    Eigen::VectorXd& moment_vjp, Eigen::VectorXd* response)
{
    const int n = atom.size();
    charge_vjp.setZero(n);
    if (response) response->setZero(n);
    moment_vjp.setZero(alpha_scalar_moments);
    // The Ewald functional is quadratic in q.  H*A accounts for the
    // variation of dE/dq in the indirect descriptor force.
    // E(q)=q^T H q/2: the same matrix-free operator returns dE/dq.
    dEdq = model_les.apply_charge_hessian(atom, charges);
    if (dLdF.squaredNorm() <= 1e-30 && dLdS.squaredNorm() <= 1e-30)
        return;

    Eigen::VectorXd direct_vjp;
    model_les.compute_les_dLdq_from_forces(atom, charges, dLdF, dLdS,
                                            direct_vjp);
    Eigen::VectorXd contracted_dqdr = Eigen::VectorXd::Zero(n);

    for (int i = 0; i < n; ++i) {
        Neighborhood& nbh = nbhs[i];
        CalcSiteBasisDers(nbh);
        tmp_se_ders_wgts.setZero(nbh.nbhcount_, 3);
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a) {
                tmp_se_ders_wgts(j, a) = dLdF(i, a)
                    - dLdF(nbh.indices_[j], a);
                for (int b = 0; b < 3; ++b)
                    tmp_se_ders_wgts(j, a) -= dLdS(a, b) * nbh.pos_[j](b);
            }
        }

        std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0.0);
        for (int m = 0; m < alpha_index_basic_count; ++m)
            for (int j = 0; j < nbh.nbhcount_; ++j)
                for (int a = 0; a < 3; ++a)
                    dloss_dsenders[m] += tmp_se_ders_wgts(j, a)
                        * moment_jacobian(m, j, a);
        for (int p = 0; p < alpha_index_times_count; ++p) {
            const int left = alpha_index_times(p, 0);
            const int right = alpha_index_times(p, 1);
            const int result = alpha_index_times(p, 3);
            const double factor = alpha_index_times(p, 2);
            dloss_dsenders[result] += factor
                * (dloss_dsenders[left] * moment_vals[right]
                   + dloss_dsenders[right] * moment_vals[left]);
        }
        for (int a = 0; a < alpha_scalar_moments; ++a) {
            const double v = dloss_dsenders[alpha_moment_mapping[a]];
            contracted_dqdr(i) += charge_moments[a] * v;
            moment_vjp(a) += dEdq(i) * v;
        }
    }

    if (response) *response=contracted_dqdr;
    charge_vjp = direct_vjp
        + model_les.apply_charge_hessian(atom, contracted_dqdr);
}

// ═══════════════════════════════════════════════════════════════════════
//  AccumulateEFSCombinationGard — energy-only
// ═══════════════════════════════════════════════════════════════════════

void PotMTP_LES::AccumulateEFSCombinationGard(Atoms &atom,
    AnyNeighborhoods &nbhs, double dLdE, double *out_gard) {
    // Accumulate the full SR layout (radial, species, moments).
    for (int i = 0; i < nbhs.size(); ++i)
        PotMTP::AccumulateFPGard(nbhs[i], out_gard, dLdE, nullptr);
    if(lr_les && channels()>1){
        const Eigen::Matrix<double,-1,3> zero=Eigen::Matrix<double,-1,3>::Zero(atom.size(),3);
        accumulate_multichannel(atom,nbhs,dLdE,zero,Eigen::Matrix3d::Zero(),out_gard);return;
    }
    if (lr_les) {
        compute_les_charges(atom,nbhs,LES_all_qs);
        cached_dEdq_=model_les.apply_charge_hessian(atom,LES_all_qs);
        dEdq_valid_=true;
    }
    // LES energy gradients (also adds SR radial gradient from LES)
    accumulate_les_energy_gradient(atom, nbhs, dLdE, out_gard);
    // LES → SR radial coefficient gradient
    Eigen::Matrix<double, -1, 3> dLdF_zero;
    Eigen::Matrix3d dLdS_zero = Eigen::Matrix3d::Zero();
    accumulate_les_sr_gradient(atom, nbhs, dLdE, dLdF_zero, dLdS_zero, out_gard);
    if (lr_les && uses_sog()) {
        dLdF_zero.setZero(atom.size(),3);
        accumulate_sog_gradient(atom,dLdE,dLdF_zero,dLdS_zero,
            Eigen::VectorXd::Zero(atom.size()),out_gard);
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  AccumulateEFSCombinationGard — force + stress (full)
// ═══════════════════════════════════════════════════════════════════════

void PotMTP_LES::AccumulateEFSCombinationGard(Atoms &atom,
    AnyNeighborhoods &nbhs,
    double dLdE,
    Eigen::Matrix<double, -1, 3> &dLdF,
    Eigen::Matrix3d &dLdS,
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &wgt_d2UdRdx,
    double *out_gard) {
    // SR gradients: use PotMTP's own force+stress overload (not AnyPotential base)
    PotMTP::AccumulateEFSCombinationGard(atom, nbhs, dLdE, dLdF, dLdS, out_gard);
    if (!lr_les) return;
    if(channels()>1){accumulate_multichannel(atom,nbhs,dLdE,dLdF,dLdS,out_gard);return;}
    compute_les_charges(atom, nbhs, LES_all_qs);
    compute_indirect_force_vjp(atom, nbhs, dLdF, dLdS);
    accumulate_sog_gradient(atom,dLdE,dLdF,dLdS,sog_response_,out_gard);
    // LES energy gradients
    accumulate_les_energy_gradient(atom, nbhs, dLdE, out_gard);

    // LES → SR radial coefficient gradient (energy part)
    accumulate_les_sr_gradient(atom, nbhs, dLdE, dLdF, dLdS, out_gard);

    // LES force + stress gradients, including charge response and the
    // explicit derivative of the descriptor Jacobian.
    if (lr_les) {
        bool has_force = dLdF.squaredNorm() > 1e-30;
        bool has_stress = dLdS.squaredNorm() > 1e-30;
        if (has_force || has_stress) {
            const int n_atoms = atom.size();
            const int les_species_offset = reg_coeff_size;
            const int les_moment_offset  = reg_coeff_size + species_count;

            // Recompute descriptors for all atoms
            grad_moment_vals_cache_.resize(n_atoms, alpha_scalar_moments);
            grad_atom_types_.resize(n_atoms);
            for (int i = 0; i < n_atoms; ++i) {
                Neighborhood &nbh = nbhs[i];
                CalcSiteBasis(nbh);
                grad_atom_types_[i] = nbh.centertype;
                for (int a = 0; a < alpha_scalar_moments; ++a)
                    grad_moment_vals_cache_(i, a) = moment_vals[alpha_moment_mapping[a]];
            }

            // Compute force+stress gradient contributions into LES section of out_gard
            {
                for (int i = 0; i < n_atoms; ++i)
                    out_gard[les_species_offset + grad_atom_types_[i]]
                        += force_charge_vjp_cache_(i);
                for (int a = 0; a < alpha_scalar_moments; ++a) {
                    double grad = force_moment_explicit_vjp_cache_(a);
                    for (int i = 0; i < n_atoms; ++i)
                        grad += force_charge_vjp_cache_(i)
                            * grad_moment_vals_cache_(i, a);
                    out_gard[les_moment_offset + a] += grad;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Accumulate LES contributions to SR radial coefficient gradients
// ═══════════════════════════════════════════════════════════════════════

void PotMTP_LES::accumulate_les_sr_gradient(Atoms &atom, AnyNeighborhoods &nbhs,
    double dLdE, const Eigen::Matrix<double, -1, 3> &dLdF,
    const Eigen::Matrix3d &dLdS, double *out_gard,
    const Eigen::VectorXd *extra_dLdq) {
    if (!lr_les) return;
    if (!dEdq_valid_ || cached_dEdq_.size()!=atom.size())
        GPERROR("LES radial gradient requested without a current dE/dq cache");

    Eigen::VectorXd dLdq = dLdE * cached_dEdq_;
    if (dLdF.squaredNorm() > 1e-30 || dLdS.squaredNorm() > 1e-30)
        dLdq += force_charge_vjp_cache_;
    if (extra_dLdq) dLdq += *extra_dLdq;
    accumulate_latent_charge_radial_gradient(atom, nbhs, les_moment_coeffs,
        dLdq, cached_dEdq_, dLdF, dLdS, out_gard);
}

void PotMTP::accumulate_latent_charge_radial_gradient(Atoms &atom,
    AnyNeighborhoods &nbhs, const double *charge_moments,
    const Eigen::VectorXd &dLdq, const Eigen::VectorXd &dEdq,
    const Eigen::Matrix<double, -1, 3> &dLdF, const Eigen::Matrix3d &dLdS,
    double *out_gard) {
    const int n_atoms = atom.size();
    const bool has_force = dLdF.squaredNorm() > 1e-30;
    const bool has_stress = dLdS.squaredNorm() > 1e-30;
    for (int i = 0; i < n_atoms; ++i) {
        Neighborhood &nbh = nbhs[i];
        CalcSiteBasisDers(nbh);

        const double dLdq_i = dLdq(i);

        // q_der is dq/d(moment node), propagated through the product graph.
        std::vector<double> q_der(alpha_moments_count, 0.0);
        for (int a = 0; a < alpha_scalar_moments; ++a) {
            int k = alpha_moment_mapping[a];
            q_der[k] += charge_moments[a];
        }
        for (int p = alpha_index_times_count - 1; p >= 0; --p) {
            double val0 = moment_vals[alpha_index_times(p, 0)];
            double val1 = moment_vals[alpha_index_times(p, 1)];
            int    val2 = alpha_index_times(p, 2);
            int    id3  = alpha_index_times(p, 3);
            q_der[alpha_index_times(p, 0)] += q_der[id3] * val2 * val1;
            q_der[alpha_index_times(p, 1)] += q_der[id3] * val2 * val0;
        }

        std::fill(dloss_dmom.begin(), dloss_dmom.end(), 0.0);
        for (int m=0; m<alpha_index_basic_count; ++m)
            dloss_dmom[m] = dLdq_i*q_der[m];

        // The explicit indirect-force term is
        // g_i w·d/dc(dq_i/dr), with g_i=dE/dq_i.  It contains both the
        // product-graph mixed derivative and the basic radial mixed
        // derivative accumulated in moment_jacobian_coeff_coord below.
        std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0.0);
        if (has_force || has_stress) {
            tmp_se_ders_wgts.setZero(nbh.nbhcount_, 3);
            for (int j=0; j<nbh.nbhcount_; ++j)
                for (int a=0; a<3; ++a) {
                    tmp_se_ders_wgts(j,a)=dLdF(i,a)-dLdF(nbh.indices_[j],a);
                    for (int b=0; b<3; ++b)
                        tmp_se_ders_wgts(j,a)-=dLdS(a,b)*nbh.pos_[j](b);
                }
            for (int m=0; m<alpha_index_basic_count; ++m)
                for (int j=0; j<nbh.nbhcount_; ++j)
                    for (int a=0; a<3; ++a)
                        dloss_dsenders[m] += tmp_se_ders_wgts(j,a)
                            * moment_jacobian(m,j,a);
            for (int p=0; p<alpha_index_times_count; ++p) {
                const int left=alpha_index_times(p,0), right=alpha_index_times(p,1);
                const int result=alpha_index_times(p,3);
                const double factor=alpha_index_times(p,2);
                dloss_dsenders[result] += factor
                    *(dloss_dsenders[left]*moment_vals[right]
                      +dloss_dsenders[right]*moment_vals[left]);
            }

            std::vector<double> mixed(alpha_moments_count,0.0);
            for (int p=0; p<alpha_index_times_count; ++p) {
                const int left=alpha_index_times(p,0), right=alpha_index_times(p,1);
                const int result=alpha_index_times(p,3);
                const double factor=alpha_index_times(p,2)*dEdq(i)*q_der[result];
                mixed[left] += factor*dloss_dsenders[right];
                mixed[right] += factor*dloss_dsenders[left];
            }
            for (int p=alpha_index_times_count-1; p>=0; --p) {
                const int left=alpha_index_times(p,0), right=alpha_index_times(p,1);
                const int result=alpha_index_times(p,3);
                const double factor=alpha_index_times(p,2);
                mixed[left] += mixed[result]*factor*moment_vals[right];
                mixed[right] += mixed[result]*factor*moment_vals[left];
            }
            for (int m=0; m<alpha_index_basic_count; ++m)
                dloss_dmom[m] += mixed[m];
        }

        // Zero the Jacobian buffers
        moment_jacobian_coeffs.setZero();
        moment_jacobian_coeff_coord.setZero();
        std::ranges::fill(rb_vals, 0);
        std::ranges::fill(rb_ders, 0);
        std::ranges::fill(val_mu, 0);
        std::ranges::fill(der_mu, 0);
        std::ranges::fill(moment_vals, 0);

        // Compute Jacobian for each neighbor
        const int itype = nbh.centertype;
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            const double rij = nbh.distances_[jj];
            const int jtype = nbh.chemmap_[jj];
            if (rij > cutoffs(itype, jtype)) continue;
            const double rijinv = 1.0 / rij;
            rb_calcder(itype, jtype, rij);

            dist_powers[0] = 1;
            for (int k = 1; k < max_alpha_index_basic; ++k) {
                dist_powers[k] = dist_powers[k-1] * rijinv;
                for (int a = 0; a < 3; a++)
                    coords_powers(k+1, a) = coords_powers(k, a) * nbh.pos_[jj][a];
            }

            for (int mu = 0; mu < max_mu; ++mu) {
                val_mu[mu] = 0; der_mu[mu] = 0;
                int off = (itype * species_count + jtype) * radial_func_count * rb_size + mu * rb_size;
                for (int xi = 0; xi < rb_size; ++xi) {
                    val_mu[mu] += radial_coeffs[off + xi] * rb_vals[xi];
                    der_mu[mu] += radial_coeffs[off + xi] * rb_ders[xi];
                }
            }

            for (int ii = 0; ii < alpha_index_basic_count; ++ii) {
                int mu = alpha_index_basic(ii, 0);
                double val = val_mu[mu], der = der_mu[mu];
                int k = nu_arr_list[ii];
                double powk = dist_powers[k];
                val *= powk;
                double pow0 = coords_powers(alpha_index_basic(ii, 1)+1, 0);
                double pow1 = coords_powers(alpha_index_basic(ii, 2)+1, 1);
                double pow2 = coords_powers(alpha_index_basic(ii, 3)+1, 2);
                double mult0 = pow0 * pow1 * pow2;
                moment_vals[ii] += val * mult0;

                for (int xi = 0; xi < rb_size; xi++) {
                    moment_jacobian_coeffs(ii, jtype, mu * rb_size + xi) +=
                        rb_vals[xi] * powk * mult0;
                    if (has_force || has_stress) {
                        const double tail=(rb_ders[xi]*powk*mult0
                            -rb_vals[xi]*k*powk*mult0/rij);
                        double derx=nbh.pos_[jj][0]/rij*tail;
                        double dery=nbh.pos_[jj][1]/rij*tail;
                        double derz=nbh.pos_[jj][2]/rij*tail;
                        derx += rb_vals[xi]*powk*alpha_index_basic(ii,1)
                            *coords_powers(alpha_index_basic(ii,1),0)*pow1*pow2;
                        dery += rb_vals[xi]*powk*alpha_index_basic(ii,2)
                            *coords_powers(alpha_index_basic(ii,2),1)*pow0*pow2;
                        derz += rb_vals[xi]*powk*alpha_index_basic(ii,3)
                            *coords_powers(alpha_index_basic(ii,3),2)*pow0*pow1;
                        moment_jacobian_coeff_coord(ii,jtype,mu*rb_size+xi) +=
                            tmp_se_ders_wgts(jj,0)*derx
                            +tmp_se_ders_wgts(jj,1)*dery
                            +tmp_se_ders_wgts(jj,2)*derz;
                    }
                }
            }
        }

        // Multiply by dloss_dmom and accumulate radial gradient
        // dloss_dmom already contains the full dL/d(moment), no extra factor needed
        for (int ii = 0; ii < alpha_index_basic_count; ++ii) {
            for (int j = 0; j < species_count; ++j) {
                for (int k = 0; k < radial_func_count * rb_size; ++k) {
                    int idx = (itype * species_count + j) * radial_func_count * rb_size + k;
                    out_gard[idx] += dloss_dmom[ii] * moment_jacobian_coeffs(ii, j, k);
                    if (has_force || has_stress)
                        out_gard[idx] += dEdq(i)*q_der[ii]
                            *moment_jacobian_coeff_coord(ii,j,k);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  AddPenaltyGard — LES regularization
// ═══════════════════════════════════════════════════════════════════════

void PotMTP_LES::accumulate_sog_gradient(Atoms& atom, double dLdE,
    const Eigen::Matrix<double,-1,3>& dLdF, const Eigen::Matrix3d& dLdW,
    const Eigen::VectorXd& response, double* out)
{
    if (!uses_sog()) return;
    std::vector<double> gradient(2*kernel_count(),0.0);
    model_les.accumulate_kernel_vjp(atom,LES_all_qs,dLdE,dLdF,dLdW,response,gradient.data());
    const int off=reg_coeff_size+latent_species_count()+latent_moment_count();
    const double span=sog_log_bandwidth_max-sog_log_bandwidth_min;
    const double mid=.5*(sog_log_bandwidth_max+sog_log_bandwidth_min);
    for (int m=0;m<kernel_count();++m) {
        const double c=std::cos(M_PI*(model_les.sog_log_bandwidths[m]-mid)/span);
        out[off+m]+=gradient[m];
        out[off+kernel_count()+m]+=gradient[kernel_count()+m]*span/M_PI*c*c;
    }
}

void PotMTP_LES::accumulate_multichannel(Atoms& atom,AnyNeighborhoods& nbhs,
    double dLdE,const Eigen::Matrix<double,-1,3>& dLdF,const Eigen::Matrix3d& dLdW,double* out)
{
    for(int channel=0;channel<channels();++channel){
        LES::ChannelScope selected(model_les,channel);
        const double* moments=les_moment_coeffs+channel*alpha_scalar_moments;
        compute_les_charges(atom,nbhs,LES_all_qs,channel);
        Eigen::VectorXd dEdq,qv,mv,response;
        latent_charge_force_vjp(atom,nbhs,model_les,LES_all_qs,moments,dLdF,dLdW,dEdq,qv,mv,&response);
        const Eigen::VectorXd dLdq=dLdE*dEdq+qv;
        const int species_offset=reg_coeff_size+channel*species_count;
        const int moment_offset=reg_coeff_size+latent_species_count()+channel*alpha_scalar_moments;
        for(int i=0;i<atom.size();++i){
            CalcSiteBasis(nbhs[i]);out[species_offset+nbhs[i].centertype]+=dLdq(i);
            for(int a=0;a<alpha_scalar_moments;++a)
                out[moment_offset+a]+=dLdq(i)*moment_vals[alpha_moment_mapping[a]];
        }
        for(int a=0;a<alpha_scalar_moments;++a)out[moment_offset+a]+=mv(a);
        accumulate_latent_charge_radial_gradient(atom,nbhs,moments,dLdq,dEdq,dLdF,dLdW,out);
        accumulate_sog_gradient(atom,dLdE,dLdF,dLdW,response,out);
    }
    dEdq_valid_=false;
}

void PotMTP_LES::AddPenaltyGard(double weight, double &loss,
                                  double *loss_gard) {
    // SR regularization
    PotMTP::AddPenaltyGard(weight, loss, loss_gard);

    if (!lr_les) return;

    const int les_species_offset = reg_coeff_size;
    const int les_moment_offset  = reg_coeff_size + latent_species_count();

    if (uses_sog()) {
        const int off=reg_coeff_size+latent_species_count()+latent_moment_count();
        for (int m=0;m<kernel_count();++m) {
            const double w=model_les.sog_weights[m];
            loss+=weight*w*w;
            loss_gard[off+m]+=2.0*weight*w;
        }
    }
    // L2 on les_species_coeffs
    for (int i = 0; i < latent_species_count(); ++i) {
        double c = les_species_coeffs[i];
        loss += weight * c * c;
        loss_gard[les_species_offset + i] += 2.0 * weight * c;
    }
    // L2 on les_moment_coeffs
    for (int i = 0; i < latent_moment_count(); ++i) {
        double c = les_moment_coeffs[i];
        loss += weight * c * c;
        loss_gard[les_moment_offset + i] += 2.0 * weight * c;
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Charge-balanced initialization
// ═══════════════════════════════════════════════════════════════════════

bool PotMTP_LES::initialize_zero_les_coeffs(
    const std::vector<Atoms>& atoms, const std::vector<double>& targets,
    double range, int seed) {
    dEdq_valid_=false;
    if(uses_sog()){
        if(!(range>0.) || !std::isfinite(range) || seed<0)GPERROR("Invalid SOG initialization range or seed");
        std::mt19937 rng(static_cast<unsigned int>(seed));
        std::uniform_real_distribution<double> dist(-range,range);
        bool changed=false;
        for(int c=0;c<channels();++c){
            bool zero=true;
            for(int i=0;i<species_count;++i)zero=zero && std::abs(les_species_coeffs[c*species_count+i])<=1e-14;
            for(int i=0;i<alpha_scalar_moments;++i)zero=zero && std::abs(les_moment_coeffs[c*alpha_scalar_moments+i])<=1e-14;
            if(!zero)continue;
            changed=true;
            for(int i=0;i<alpha_scalar_moments;++i)les_moment_coeffs[c*alpha_scalar_moments+i]=dist(rng);
            for(int i=0;i<species_count;++i)les_species_coeffs[c*species_count+i]=dist(rng);
        }
#ifdef IMRMLP_MPI
        MPI_Bcast(les_species_coeffs,latent_species_count(),MPI_DOUBLE,0,mpi.comm);
        MPI_Bcast(les_moment_coeffs,latent_moment_count(),MPI_DOUBLE,0,mpi.comm);
#endif
        return changed;
    }
    bool all_zero = true;
    for (int i = 0; i < species_count; ++i)
        all_zero = all_zero && std::abs(les_species_coeffs[i]) <= 1e-14;
    for (int i = 0; i < alpha_scalar_moments; ++i)
        all_zero = all_zero && std::abs(les_moment_coeffs[i]) <= 1e-14;
    if (!all_zero) return false;
    if (!(range > 0.0) || !std::isfinite(range))
        GPERROR("fit:les_zero_init_range must be finite and positive");
    if (seed < 0) GPERROR("fit:les_zero_init_seed must be non-negative");
    if (mpi.rank == 0) {
        std::mt19937 rng(static_cast<unsigned int>(seed));
        std::uniform_real_distribution<double> dist(-range, range);
        for (int i = 0; i < alpha_scalar_moments; ++i)
            les_moment_coeffs[i] = dist(rng);
        if (uses_sog())
            for (int i=0;i<species_count;++i) les_species_coeffs[i]=dist(rng);
    }
#ifdef IMRMLP_MPI
    MPI_Bcast(les_moment_coeffs, alpha_scalar_moments, MPI_DOUBLE, 0, mpi.comm);
#endif
#ifdef IMRMLP_MPI
    if (uses_sog()) MPI_Bcast(les_species_coeffs,species_count,MPI_DOUBLE,0,mpi.comm);
#endif
    if (!uses_sog()) balance_les_charges(atoms, targets);
    const double moment_norm = Eigen::Map<Eigen::VectorXd>(
        les_moment_coeffs, alpha_scalar_moments).norm();
    const double bias_norm = Eigen::Map<Eigen::VectorXd>(
        les_species_coeffs, species_count).norm();
    if (!std::isfinite(moment_norm) || !std::isfinite(bias_norm))
        GPERROR("Non-finite automatic LES initialization");
    if (mpi.rank == 0)
        Message("MTP+LES zero initialization broken: moment norm = "
            + std::to_string(moment_norm) + ", species bias norm = "
            + std::to_string(bias_norm));
    return true;
}

void PotMTP_LES::balance_les_charges(const std::vector<Atoms>& atoms,
                                      const std::vector<double>& targets) {
    dEdq_valid_=false;
    // Empty local partitions must still participate in MPI collectives.
    if (!lr_les || !les_moment_coeffs) return;
    validate_response_parity();
    const int n_cfgs = atoms.size();
    const double out_scale = les_response_parity;
    int n_scalars = alpha_scalar_moments;

    // Compute per-config moment charge contribution: Σ_i Σ_a les_moment[a] * B_i[a]
    std::vector<double> local_disp(n_scalars + 1);
    Eigen::VectorXd moment_q(n_cfgs);
    Eigen::VectorXd weights(n_cfgs);
    for (int ic = 0; ic < n_cfgs; ++ic) {
        Neighborhoods nbhs(atoms[ic], cutoff);
        double sum = 0.0;
        for (int j = 0; j < atoms[ic].size(); ++j) {
            compute_descriptors(nbhs[j], local_disp);
            for (int a = 0; a < n_scalars; ++a)
                sum += local_disp[a + 1] * les_moment_coeffs[a];
        }
        moment_q(ic) = sum;
        weights(ic) = 1.0 / std::max(1, atoms[ic].size());
    }

    // Build linear system: A * les_species = -moment_q
    // A[cfg, t] = n_{t, cfg} (counts of species t in config cfg)
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_cfgs, species_count);
    if (!targets.empty() && targets.size() != atoms.size())
        GPERROR("LES charge target count does not match configuration count");
    Eigen::VectorXd b(n_cfgs);
    for (int ic = 0; ic < n_cfgs; ++ic) {
        Eigen::VectorXd counts = Eigen::VectorXd::Zero(species_count);
        for (int j = 0; j < atoms[ic].size(); ++j)
            counts(atoms[ic].chemmap_[j]) += 1.0;
        A.row(ic) = weights(ic) * counts;
        const double physical_target = targets.empty() ? 0.0 : targets[ic];
        if (!std::isfinite(physical_target)) GPERROR("Non-finite LES total-charge target");
        const double raw_target=les_response_parity*physical_target;
        b(ic) = weights(ic) * (raw_target - moment_q(ic));
    }

    // Solve one global normal equation.  Training structures are already
    // partitioned by rank when this function is called.
    Eigen::MatrixXd ATA = A.transpose() * A;
    Eigen::VectorXd ATb = A.transpose() * b;
#ifdef IMRMLP_MPI
    Eigen::MatrixXd global_ATA = Eigen::MatrixXd::Zero(species_count, species_count);
    Eigen::VectorXd global_ATb = Eigen::VectorXd::Zero(species_count);
    MPI_Allreduce(ATA.data(), global_ATA.data(), species_count * species_count,
                  MPI_DOUBLE, MPI_SUM, mpi.comm);
    MPI_Allreduce(ATb.data(), global_ATb.data(), species_count,
                  MPI_DOUBLE, MPI_SUM, mpi.comm);
    ATA = std::move(global_ATA);
    ATb = std::move(global_ATb);
#endif
    double reg = std::max(1.0e-12,
        1.0e-12 * ATA.trace() / std::max(1,species_count));
    for (int t = 0; t < species_count; ++t) ATA(t, t) += reg;

    if (mpi.rank == 0) {
        Eigen::VectorXd sol = ATA.ldlt().solve(ATb);
        if (!sol.allFinite()) GPERROR("Non-finite LES charge-balancing solution");
        for (int t = 0; t < species_count; ++t)
            les_species_coeffs[t] = sol(t);
    }
#ifdef IMRMLP_MPI
    MPI_Bcast(les_species_coeffs, species_count, MPI_DOUBLE, 0, mpi.comm);
#endif

    // Report
    double max_q = 0, rms_q = 0;
    for (int ic = 0; ic < n_cfgs; ++ic) {
        Neighborhoods nbhs(atoms[ic], cutoff);
        double cfg_q = 0.0;
        for (int j = 0; j < atoms[ic].size(); ++j) {
            double qi = les_species_coeffs[atoms[ic].chemmap_[j]];
            compute_descriptors(nbhs[j], local_disp);
            for (int a = 0; a < n_scalars; ++a)
                qi += local_disp[a + 1] * les_moment_coeffs[a];
            cfg_q += qi * out_scale;
        }
        const double residual = cfg_q - (targets.empty() ? 0.0 : targets[ic]);
        max_q = std::max(max_q, std::fabs(residual));
        rms_q += residual * residual;
    }
    double global_max_q = max_q;
    double global_sum_q2 = rms_q;
    long long global_n_cfgs = n_cfgs;
#ifdef IMRMLP_MPI
    MPI_Allreduce(&max_q, &global_max_q, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
    MPI_Allreduce(&rms_q, &global_sum_q2, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    long long local_n_cfgs = n_cfgs;
    MPI_Allreduce(&local_n_cfgs, &global_n_cfgs, 1, MPI_LONG_LONG, MPI_SUM, mpi.comm);
#endif
    const double global_rms_q = std::sqrt(global_sum_q2
        / std::max<long long>(1, global_n_cfgs));
    if (mpi.rank == 0)
        std::cout << "LES charge initialization: max|Q_cfg| = " << global_max_q
                  << " e, RMS Q_cfg = " << global_rms_q << " e\n";
}

// ═══════════════════════════════════════════════════════════════════════
//  Born Effective Charges — wrapper
// ═══════════════════════════════════════════════════════════════════════

void PotMTP_LES::compute_bec(Atoms& atom, bool remove_mean, double epsilon_factor) {
    atom.born_eff_chgs_.clear();
    atom.have_eff_chg=false;
    dEdq_valid_=false;
    if (!lr_les || uses_sog()) GPERROR("BEC requires a Coulomb LES model with charge-like latent variables");
    validate_response_parity();
    LES::validate_periodic_geometry(atom);
    if (!(epsilon_factor>0.0) || !std::isfinite(epsilon_factor))
        GPERROR("LES BEC epsilon_factor must be finite and positive");
    const int n=atom.size();
    const double scale=les_response_parity*std::sqrt(epsilon_factor);
    const Eigen::Matrix3d inv_cell=atom.cell_.inverse();
    std::vector<Eigen::Vector3cd> phases(n);
    Eigen::Vector3cd mean_phase=Eigen::Vector3cd::Zero();
    for (int i=0;i<n;++i) {
        const Eigen::RowVector3d fractional=atom.pos_[i]*inv_cell;
        for (int alpha=0;alpha<3;++alpha)
            phases[i](alpha)=std::exp(std::complex<double>(
                0.0,2.0*M_PI*fractional(alpha)));
        mean_phase+=phases[i];
    }
    if (remove_mean && n>0) mean_phase/=n;
    else mean_phase.setZero();

    // D_alpha = sum_j (q_j - mean(q)) phase_j / (i 2 pi).
    // Its charge derivative has weight (phase_j - mean(phase))/(i 2 pi).
    // Accumulate three complex responses using one local charge Jacobian per
    // site, rather than constructing the dense N by 3N charge Jacobian or
    // rebuilding all neighborhoods for 6N displaced configurations.
    std::vector<Eigen::Matrix3cd> response(n,Eigen::Matrix3cd::Zero());
    Neighborhoods nbhs(atom,cutoff);
    LES_all_qs.resize(n);
    for (int i=0;i<n;++i) {
        auto& nbh=nbhs[i];
        CalcSiteDescriptors(nbh);
        double q=les_species_coeffs[nbh.centertype];
        std::fill(site_energy_ders_wrt_moments_.begin(),
                  site_energy_ders_wrt_moments_.end(),0.0);
        for (int a=0;a<alpha_scalar_moments;++a) {
            const int k=alpha_moment_mapping[a];
            q+=les_moment_coeffs[a]*moment_vals[k];
            site_energy_ders_wrt_moments_[k]+=les_moment_coeffs[a];
        }
        LES_all_qs(i)=q;
        backpropagate_moments();
        const Eigen::Vector3cd weight=(phases[i]-mean_phase)
            /std::complex<double>(0.0,2.0*M_PI);
        for (int j=0;j<nbh.nbhcount_;++j) {
            const int neighbor=nbh.indices_[j];
            // An image of this same atom moves together with the center.
            if (neighbor==i) continue;
            Eigen::RowVector3d dq=Eigen::RowVector3d::Zero();
            for (int k=0;k<alpha_index_basic_count;++k)
                for (int gamma=0;gamma<3;++gamma)
                    dq(gamma)+=site_energy_ders_wrt_moments_[k]
                        *moment_jacobian(k,j,gamma);
            const Eigen::Matrix3cd contribution=weight*dq.cast<std::complex<double>>();
            response[neighbor]+=contribution;
            response[i]-=contribution;
        }
    }
    if (!LES_all_qs.allFinite()) GPERROR("LES BEC charges contain non-finite values");
    const double mean_q=remove_mean && n>0 ? LES_all_qs.mean() : 0.0;
    std::vector<Eigen::Matrix3d> bec(n,Eigen::Matrix3d::Zero());
    for (int i=0;i<n;++i) {
        Eigen::Matrix3d dephased;
        for (int alpha=0;alpha<3;++alpha)
            for (int gamma=0;gamma<3;++gamma)
                dephased(alpha,gamma)=
                    (std::conj(phases[i](alpha))*response[i](alpha,gamma)).real();
        // The derivative of the phase itself gives centered q_i times I
        // after conversion from fractional to Cartesian response axes.
        bec[i]=scale*(atom.cell_.transpose()*dephased
            +(LES_all_qs(i)-mean_q)*Eigen::Matrix3d::Identity());
        if (!bec[i].allFinite()) GPERROR("LES BEC derivative contains non-finite values");
    }
    atom.born_eff_chgs_=std::move(bec);
    atom.have_eff_chg = true;
}

} // namespace Potentials
