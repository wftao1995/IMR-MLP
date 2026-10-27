#include "native_inference.h"
#include "model_bundle.h"
#include "native_profile.h"
#include "native_long_range.h"
#include "potential_wrapper.h"
#include "potential_mtp_LES.h"
#include "potential_mtp_m.h"
#include "potential_mtpnn1.h"
#include "potential_hybmtp.h"
#include "potential_hybrid.h"
#include "potential_hybrid_scale.h"
#include "mtp_gpu_adapter.h"
#include "native_gpu_inference.h"
#include "base/neighborhood.h"

#include <filesystem>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace Potentials {
namespace {
std::string canonical_type(std::string type) {
    if (type == "mtp") return "mtp";
    if (type == "mtp_m") return "mtp_m";
    if (type == "mtpnn1") return "mtpnn1";
    if (type == "hmtp") return "hmtp";
    if (type == "mtp_les") return "mtp_les";
    return type;
}

void merge_setting(GPSettings &settings, const std::string &key, const std::string &value) {
    if (value.empty()) return;
    auto normalized=[&](std::string input) {
        if(key=="activation_function")for(char &c:input)c=std::toupper(static_cast<unsigned char>(c));
        if(key=="extmodel")input=MTPMetadata::canonical_model(input);
        if(key=="lr_les") {
            for(char &c:input)c=std::tolower(static_cast<unsigned char>(c));
            if(input=="1")input="true";if(input=="0")input="false";
        }
        return input;
    };
    if (settings.count(key) && normalized(settings[key]) != normalized(value))
        throw std::runtime_error("mlip/native: " + key + " conflicts with model metadata");
    settings.Modify(key, value);
}

GPSettings load_settings(const std::string &file, const NativeInferenceOptions &options) {
    GPSettings settings;
    if (!options.settings_file.empty()) {
        GPSettings all(options.settings_file);
        settings = all.ExtractSubSettings("pot");
        if (settings.settings.empty()) settings = all;
        const auto directory = std::filesystem::absolute(options.settings_file).parent_path();
        for (auto &[key, value] : settings.settings)
            if ((key == "typemap" || key.ends_with(":potfile")) && !value.empty()) {
                const std::filesystem::path path(value);
                if (path.is_relative()) value = (directory / path).lexically_normal().string();
            }
    }
    const std::string type = canonical_type(options.pottype);
    if (settings.count("pottype") && canonical_type(settings["pottype"]) != type)
        throw std::runtime_error("mlip/native: pottype conflicts with settings");
    if (!GPM::is_file(file) && (type == "mtp" || type == "mtp_m" || type == "mtp_les")) {
        auto stream = open_ifile(file);
        if (!stream) throw std::runtime_error("mlip/native: cannot open " + file);
        const auto header = MTPMetadata::read_header(*stream);
        if (header.tag.structured()) {
            const auto expected = type == "mtp" ? "MTP" : type == "mtp_m" ? "MTP_M" : "MTP_LES";
            if (header.tag.text("type") != expected)
                throw std::runtime_error("mlip/native: pottype conflicts with model type");
            const auto long_range=header.tag.fields.find("long_range");
            const bool has_long_range=long_range!=header.tag.fields.end();
            merge_setting(settings,"lr_les",has_long_range?"true":"false");
            if(has_long_range) {
                const auto &definition=long_range->second;
                const auto kernel=MTPMetadata::string(definition.get("kernel"),"long_range.kernel");
                merge_setting(settings,"les_kernel",kernel);
                if(kernel=="sog")merge_setting(settings,"sog_channels",std::to_string(
                    MTPMetadata::integer(definition.get("channels"),"long_range.channels",1)));
            }
            if (type == "mtp_m") {
                const auto fields = MTPMetadata::legacy_model_fields(header.tag);
                for (const auto &[source, target] : {
                    std::pair{"mtp_model", "extmodel"}, {"mtp_descriptor_count", "n_input"},
                    {"mtp_hidden1", "n_neuron"}, {"mtp_hidden2", "n_neuron2"},
                    {"mtp_activation", "activation_function"}}) {
                    const auto it = fields.find(source);
                    // Analytic heads have a fixed descriptor count; their
                    // trainer ignores the NN-only n_input setting.
                    if (std::string(source)=="mtp_descriptor_count" &&
                        !header.tag.fields.count("nn")) continue;
                    if (it != fields.end() && it->second != "0") merge_setting(settings, target, it->second);
                }
            }
        }
    }
    settings.Modify("pottype", type);
    settings.Modify("potfile", std::filesystem::absolute(file).lexically_normal().string());
    // An incidental index.map in the LAMMPS working directory must never
    // override the trained model's species order.
    if (!settings.count("typemap")) settings.Modify("typemap", "");
    return settings;
}

NativeInferenceResult empty_result(int count, const NativeInferenceRequest &request = {}) {
    NativeInferenceResult out;
    if (request.forces) out.forces.setZero(count, 3);
    if (request.site_energy) out.site_energy.setZero(count);
    if (request.site_virial) out.site_virial.setZero(count, 6);
    return out;
}

void add_edges(NativeInferenceResult &out, int center, const Neighborhood &nbh,
               const SiteCoordinateMatrix &gradient, const NativeInferenceRequest &request = {}) {
    for (int row = 0; row < nbh.nbhcount_; ++row) {
        const int neighbor = nbh.indices_[row];
        const Eigen::Vector3d g = gradient.row(row).transpose();
        if (request.forces) {
            out.forces.row(center) += g.transpose();
            out.forces.row(neighbor) -= g.transpose();
        }
        if (!request.virial && !request.site_virial) continue;
        const Eigen::Matrix3d tensor = -g * nbh.pos_[row];
        if (request.virial) out.virial += tensor;
        if (!request.site_virial) continue;
        Eigen::Matrix<double, 1, 6> w;
        w << tensor(0,0), tensor(1,1), tensor(2,2),
             .5*(tensor(0,1)+tensor(1,0)), .5*(tensor(0,2)+tensor(2,0)),
             .5*(tensor(1,2)+tensor(2,1));
        out.site_virial.row(center) += .5*w;
        out.site_virial.row(neighbor) += .5*w;
    }
}

void latent_site(PotMTP &pot, Neighborhood &nbh, const double *species,
                 const double *coefficients, double &value, SiteCoordinateMatrix &gradient) {
    // Only basic-moment Jacobians are needed by reverse-mode contraction.
    // CalcSiteBasisDers additionally propagates every scalar Jacobian.
    pot.CalcSiteDescriptors(nbh);
    value = species[nbh.centertype];
    std::fill(pot.site_energy_ders_wrt_moments_.begin(), pot.site_energy_ders_wrt_moments_.end(), 0.);
    for (int a = 0; a < pot.alpha_scalar_moments; ++a) {
        const int index = pot.alpha_moment_mapping[a];
        value += coefficients[a] * pot.moment_vals[index];
        pot.site_energy_ders_wrt_moments_[index] += coefficients[a];
    }
    pot.backpropagate_moments();
    gradient.setZero(nbh.nbhcount_, 3);
    for (int m = 0; m < pot.alpha_index_basic_count; ++m)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int axis = 0; axis < 3; ++axis)
                gradient(j,axis) += pot.site_energy_ders_wrt_moments_[m] * pot.moment_jacobian(m,j,axis);
}

void latent_values(PotMTP &pot, Neighborhood &nbh, const double *species,
                   const double *coefficients, int species_count, int channels,
                   double *values) {
    pot.CalcSiteBasis(nbh);
    for (int c = 0; c < channels; ++c) {
        double value = species[c*species_count+nbh.centertype];
        for (int a = 0; a < pot.alpha_scalar_moments; ++a)
            value += coefficients[c*pot.alpha_scalar_moments+a] * pot.moment_vals[pot.alpha_moment_mapping[a]];
        values[c] = value;
    }
}

void latent_response(PotMTP &pot, Neighborhood &nbh, const double *coefficients,
                     const Eigen::Ref<const Eigen::RowVectorXd> &weights,
                     SiteCoordinateMatrix &gradient) {
    pot.CalcSiteDescriptors(nbh);
    std::fill(pot.site_energy_ders_wrt_moments_.begin(), pot.site_energy_ders_wrt_moments_.end(), 0.);
    for (int c = 0; c < weights.size(); ++c)
        for (int a = 0; a < pot.alpha_scalar_moments; ++a)
            pot.site_energy_ders_wrt_moments_[pot.alpha_moment_mapping[a]] +=
                weights[c] * coefficients[c*pot.alpha_scalar_moments+a];
    pot.backpropagate_moments();
    gradient.setZero(nbh.nbhcount_, 3);
    for (int m = 0; m < pot.alpha_index_basic_count; ++m)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int axis = 0; axis < 3; ++axis)
                gradient(j,axis) += pot.site_energy_ders_wrt_moments_[m] * pot.moment_jacobian(m,j,axis);
}

void honor_request(NativeInferenceResult &out, const NativeInferenceRequest &request) {
    if (!request.forces) out.forces.resize(0,3);
    if (!request.energy) out.energy = 0.;
    if (!request.site_energy) out.site_energy.resize(0);
    if (!request.virial) out.virial.setZero();
    if (!request.site_virial) out.site_virial.resize(0,6);
    if (!request.charges) out.charges.resize(0);
    if (!request.latents) out.latents.resize(0,0);
}

void describe_radial(const AnyPotential &model,std::set<std::string> &descriptions) {
    if(const auto *p=dynamic_cast<const PotMTP *>(&model)) {
        auto describe=[&](const AnyBasis &basis){
            std::ostringstream text;text<<basis.name<<"[n="<<basis.basis_size<<",r="<<basis.min_val<<":"<<basis.max_val
                <<",scale="<<basis.scaling<<",mode="<<p->radial_mode;
            if(const auto *config=dynamic_cast<const RBasis_ChebyshevConfigurable *>(&basis))
                text<<","<<basis_function_type_name(config->basis_function_type())<<"/"
                    <<Cutoff::cutoff_type_name(config->cutoff_function().type())<<"/"
                    <<XTransForm::shift_type_name(config->shift_function().type());
            if(p->radial_basis_split)text<<",ordered-pair";
            text<<"]";descriptions.insert(text.str());
        };
        if(p->radial_basis_split){for(const auto *basis:p->p_radical_basis_pair)if(basis)describe(*basis);}
        else if(p->p_radical_basis)describe(*p->p_radical_basis);
    } else if(const auto *p=dynamic_cast<const PotMTPNN1 *>(&model))describe_radial(*p->p_mtp,descriptions);
    else if(const auto *p=dynamic_cast<const PotMTPHyb3 *>(&model))for(const auto *child:p->children())describe_radial(*child,descriptions);
    else if(const auto *p=dynamic_cast<const PotentialHybrid *>(&model))for(const auto *child:p->children())describe_radial(*child,descriptions);
    else if(const auto *p=dynamic_cast<const PotentialHybridScale *>(&model))for(const auto *child:p->children())describe_radial(*child,descriptions);
}

std::vector<std::string> model_elements(const AnyPotential &model) {
    if (const auto *p = dynamic_cast<const PotMTP *>(&model)) return p->potential_tag.elements();
    if (const auto *p = dynamic_cast<const PotMTPNN1 *>(&model)) return p->p_mtp->potential_tag.elements();
    if (const auto *p = dynamic_cast<const PotMTPHyb3 *>(&model)) return p->children().front()->potential_tag.elements();
    if (const auto *p = dynamic_cast<const PotentialHybrid *>(&model))
        for (const auto &child : p->ppots) { auto names = model_elements(*child); if (!names.empty()) return names; }
    if (const auto *p = dynamic_cast<const PotentialHybridScale *>(&model))
        for (const auto *child : p->children()) { auto names = model_elements(*child); if (!names.empty()) return names; }
    return {};
}
} // namespace

namespace {
// This storage is never returned to a caller. In particular, communication
// callbacks cannot mutate borrowed caller coordinates after validation.
struct CandidateWorkspace {
    Atoms atoms;
    std::vector<int> centers, offsets, neighbors;
    std::vector<double> positions, distances;
    const Atoms *provider = nullptr;
    std::uint64_t revision = 0;
    bool valid = false;

    void prepare(const Atoms &source, const NativeCandidateNeighborhoodView &c,
                 int species_count, double cutoff) {
        IMRMLP_NATIVE_PROFILE_SCOPE("candidate.prepare");
        if(c.owned_count<0 || c.total_count<0 || c.owned_count>c.total_count ||
           static_cast<std::size_t>(c.total_count)!=source.pos_.size() ||
           source.chemmap_.size()!=source.pos_.size() ||
           c.total_count>std::numeric_limits<int>::max()/3 || c.candidate_count<0 ||
           !c.offsets || (c.candidate_count && !c.neighbors) ||
           c.offsets[0]!=0 || c.offsets[c.owned_count]!=c.candidate_count ||
           !(cutoff>0.) || !std::isfinite(cutoff) || !source.cartesian)
            throw std::runtime_error("mlip/native: invalid candidate neighborhood dimensions");
        for(int i=0;i<c.owned_count;++i)
            if(c.offsets[i]<0 || c.offsets[i+1]<c.offsets[i] || c.offsets[i+1]>c.candidate_count)
                throw std::runtime_error("mlip/native: invalid candidate neighborhood offsets");
        bool changed=!valid || provider!=&source || atoms.size()!=c.total_count ||
            centers.size()!=static_cast<std::size_t>(c.owned_count);
        {
        IMRMLP_NATIVE_PROFILE_SCOPE("candidate.snapshot");
        atoms.cell_=source.cell_;atoms.pbc_=source.pbc_;atoms.cartesian=true;
        atoms.nlocal_=c.owned_count;atoms.nghost_=c.total_count-c.owned_count;
        atoms.have_ghost=atoms.nghost_!=0;
        atoms.global_attribs_d_.clear();atoms.features_.clear();
        atoms.pos_.resize(c.total_count);atoms.chemmap_.resize(c.total_count);
        positions.resize(static_cast<std::size_t>(c.total_count)*3);
        for(int i=0;i<c.total_count;++i) {
            const int type=source.chemmap_[i];
            if(type<0 || type>=species_count)
                throw std::runtime_error("mlip/native: candidate atom type outside the model species range");
            if(atoms.chemmap_[i]!=type)changed=true;
            atoms.chemmap_[i]=type;
            for(int d=0;d<3;++d) {
                const double value=source.pos_[i][d];
                if(!std::isfinite(value))
                    throw std::runtime_error("mlip/native: invalid candidate atom position");
                atoms.pos_[i][d]=value;positions[3*i+d]=value;
            }
        }
        }
        {
        IMRMLP_NATIVE_PROFILE_SCOPE("candidate.filter_pack");
        if(centers.size()!=static_cast<std::size_t>(c.owned_count)) {
            centers.resize(c.owned_count);std::iota(centers.begin(),centers.end(),0);
        }
        offsets.resize(c.owned_count+1);offsets[0]=0;
        const std::size_t old_edges=neighbors.size();
        neighbors.resize(c.candidate_count);distances.resize(c.candidate_count);
        std::size_t edge=0;
        for(int i=0;i<c.owned_count;++i) {
            for(int k=c.offsets[i];k<c.offsets[i+1];++k) {
                const int j=c.neighbors[k];
                if(j<0 || j>=c.total_count)
                    throw std::runtime_error("mlip/native: invalid candidate neighbor index");
                // Retain the Pair's Eigen norm and strict sqrt cutoff, not a
                // squared-cutoff approximation at a floating-point boundary.
                const double distance=(atoms.pos_[j]-atoms.pos_[i]).norm();
                if(!(distance<cutoff))continue;
                double squared=0.;
                for(int d=0;d<3;++d) {
                    const double delta=positions[3*j+d]-positions[3*i+d];
                    squared+=delta*delta;
                }
                // Preserve the public GPU packed gate too: its scalar
                // accumulation can differ from Eigen for extreme values.
                if(!(distance>0.) || !std::isfinite(distance) ||
                   !(squared>0.) || !std::isfinite(squared))
                    throw std::runtime_error("mlip/native: invalid candidate neighbor geometry");
                if(edge>=static_cast<std::size_t>(std::numeric_limits<int>::max()/3))
                    throw std::runtime_error("mlip/native: candidate active neighbors exceed the supported index range");
                if(edge>=old_edges || neighbors[edge]!=j)changed=true;
                neighbors[edge]=j;distances[edge]=distance;++edge;
            }
            if(offsets[i+1]!=static_cast<int>(edge))changed=true;
            offsets[i+1]=static_cast<int>(edge);
        }
        if(old_edges!=edge)changed=true;
        neighbors.resize(edge);
        if(changed && ++revision==0)
            throw std::runtime_error("mlip/native: candidate topology revision overflow");
        }
        provider=&source;valid=true;
    }

    NativePackedNeighborhoodView view() const noexcept {
        return {static_cast<int>(centers.size()),atoms.size(),static_cast<int>(neighbors.size()),
            centers.data(),offsets.data(),neighbors.data(),atoms.chemmap_.data(),
            positions.data(),nullptr,distances.data(),reinterpret_cast<std::uintptr_t>(this),revision};
    }

    void materialize(AnyNeighborhoods &nbhs) const {
        nbhs.resize(centers.size());
        for(int i=0;i<nbhs.size();++i) {
            auto &n=nbhs[i];n.centertype=atoms.chemmap_[i];n.nbhcount_=offsets[i+1]-offsets[i];
            n.indices_.resize(n.nbhcount_);n.chemmap_.resize(n.nbhcount_);
            n.pos_.resize(n.nbhcount_);n.distances_.resize(n.nbhcount_);
            for(int j=0;j<n.nbhcount_;++j) {
                const int k=offsets[i]+j,neighbor=neighbors[k];
                n.indices_[j]=neighbor;n.chemmap_[j]=atoms.chemmap_[neighbor];
                n.pos_[j]=atoms.pos_[neighbor]-atoms.pos_[i];n.distances_[j]=distances[k];
            }
        }
    }
};
}

struct NativeInference::Impl {
    std::unique_ptr<AnyPotential> model;
    NativeInferenceOptions options;
    NativeInferenceStatistics statistics;
    PotMTP *latent_model = nullptr;
    LES *reciprocal = nullptr;
    const double *latent_species = nullptr, *latent_moments = nullptr;
    int channels = 0, parity = 1;
    std::function<void(Neighborhood &, double &, SiteCoordinateMatrix &, bool)> short_range;
    CandidateWorkspace candidates;
#ifdef IMRMLP_GPU
    std::unique_ptr<NativeGpuInference> gpu;
#endif

    template<class Head> bool select_mtp_m() {
        auto *p = dynamic_cast<PotMTP_M<Head> *>(model.get());
        if (!p) return false;
        short_range = [p](Neighborhood &nbh, double &energy, SiteCoordinateMatrix &gradient, bool derivatives) {
            energy = 0.;
            if (derivatives) p->template CalcSiteEFS<1>(nbh,energy,gradient);
            else p->template CalcSiteEFS<0>(nbh,energy,gradient);
        };
        if (p->lr_les) {
            latent_model = p; reciprocal = &p->model_les;
            latent_species = p->les_species; latent_moments = p->les_moment;
            channels = p->les_channels();
        }
        return true;
    }

    NativeInferenceResult cpu_local(Atoms &atoms, AnyNeighborhoods &nbhs,
                                    const NativeInferenceRequest &request = {}) {
        IMRMLP_NATIVE_PROFILE_SCOPE("cpu.short_range");
        auto out = empty_result(atoms.size(),request);
        const bool derivatives = request.forces || request.virial || request.site_virial;
        if (!derivatives && !request.energy && !request.site_energy) return out;
        SiteCoordinateMatrix gradient;
        for (int i = 0; i < nbhs.size(); ++i) {
            double value = 0.; short_range(nbhs[i],value,gradient,derivatives);
            if (request.energy) out.energy += value;
            if (request.site_energy) out.site_energy[i] += value;
            if (derivatives) add_edges(out,i,nbhs[i],gradient,request);
        }
        return out;
    }

    NativeInferenceResult cpu_frame(Atoms &atoms, AnyNeighborhoods &nbhs,
                                    const NativeInferenceRequest &request) {
        auto out = cpu_local(atoms,nbhs,reciprocal ? NativeInferenceRequest{} : request);
        if (reciprocal) {
            LES::validate_periodic_geometry(atoms);
            out.latents.resize(atoms.size(),channels);
            std::vector<SiteCoordinateMatrix> gradients(atoms.size());
            for (int c = 0; c < channels; ++c) {
                for (int i = 0; i < atoms.size(); ++i) {
                    double q = 0.;
                    latent_site(*latent_model,nbhs[i],latent_species+c*model->species_count,
                        latent_moments+c*latent_model->alpha_scalar_moments,q,gradients[i]);
                    out.latents(i,c)=q;
                }
                LES::ChannelScope selected(*reciprocal,c);
                const auto lr = reciprocal->evaluate(atoms,out.latents.col(c));
                out.energy += lr.energy; out.site_energy += lr.per_atom_energy;
                out.virial += lr.virial; out.site_virial += lr.per_atom_virial;
                out.forces += lr.forces;
                for (int i = 0; i < atoms.size(); ++i) {
                    gradients[i] *= lr.dEdq[i]; add_edges(out,i,nbhs[i],gradients[i]);
                }
            }
            if (reciprocal->kernel == LES::Kernel::Coulomb) out.charges = parity*out.latents.col(0);
        }
        return out;
    }

    NativeInferenceResult cpu_distributed(Atoms &atoms, AnyNeighborhoods &nbhs,
                                         const NativeCollectives &calls,
                                         const NativeInferenceRequest &request) {
        IMRMLP_NATIVE_PROFILE_SCOPE("cpu.distributed");
        const int n=nbhs.size();
        NativeInferenceResult out;
        calls.stage([&] {
            IMRMLP_NATIVE_PROFILE_SCOPE("cpu.initialization_and_short_range");
            if(n>atoms.size()) throw std::runtime_error("mlip/native: invalid owned atom count");
            for(int i=0;i<atoms.size();++i)
                if(atoms.chemmap_[i]<0 || atoms.chemmap_[i]>=model->species_count)
                    throw std::runtime_error("mlip/native: atom type outside the model species range");
            out=cpu_local(atoms,nbhs,reciprocal ? NativeInferenceRequest{} : request);
        });
        if(reciprocal) {
            Eigen::MatrixXd owned_latents;
            calls.stage([&] {
                IMRMLP_NATIVE_PROFILE_SCOPE("cpu.les.latent_forward");
                out.latents.setZero(atoms.size(),channels);
                for(int i=0;i<n;++i)
                    latent_values(*latent_model,nbhs[i],latent_species,latent_moments,
                        model->species_count,channels,out.latents.row(i).data());
                owned_latents=out.latents.topRows(n);
            });
            const auto lr=native_les_distributed(*reciprocal,atoms,owned_latents,n,calls);
            calls.stage([&] {
                IMRMLP_NATIVE_PROFILE_SCOPE("cpu.les.response");
                out.energy+=lr.energy;out.virial+=lr.virial;
                out.site_energy.head(n)+=lr.per_atom_energy;
                out.site_virial.topRows(n)+=lr.per_atom_virial;
                out.forces.topRows(n)+=lr.forces;
                SiteCoordinateMatrix gradient;
                for(int i=0;i<n;++i) {
                    latent_response(*latent_model,nbhs[i],latent_moments,lr.dEdq.row(i),gradient);
                    add_edges(out,i,nbhs[i],gradient);
                }
                if(reciprocal->kernel==LES::Kernel::Coulomb) out.charges=parity*out.latents.col(0);
            });
        }
        return out;
    }


};

NativeInference::NativeInference(const std::string &file, const NativeInferenceOptions &options)
    : impl_(std::make_unique<Impl>()) {
    impl_->options=options; impl_->options.pottype=canonical_type(options.pottype);
    if(GPM::is_file(file)) {
        auto manifest=ModelBundle::inspect(file);
        auto actual=GPM::string(GPM::at(GPM::object(GPM::at(GPM::object(GPM::at(manifest,"models")),GPM::string(GPM::at(manifest,"root_model")))),"type"));
        GPM::require(impl_->options.pottype.empty()||impl_->options.pottype=="auto"||impl_->options.pottype==actual,"native potential type conflicts with GPM");
        impl_->options.pottype=actual;
    }
    if (impl_->options.device=="dcu") impl_->options.device="hip";
    const std::vector<std::string> types={"mtp","mtp_m","mtpnn1","hmtp","mtp_les","hybrid","hybrid_scale"};
    if (std::find(types.begin(),types.end(),impl_->options.pottype)==types.end())
        throw std::runtime_error("mlip/native: explicit supported MTP pottype is required");
    if (impl_->options.device!="cpu" && impl_->options.device!="cuda" && impl_->options.device!="hip")
        throw std::runtime_error("mlip/native: device must be cpu, cuda, hip, or dcu");
    auto settings=load_settings(file,impl_->options);
    potential_wrapper wrapper(settings);
    impl_->model.reset(wrapper.pot); wrapper.pot=nullptr;
    auto *model=impl_->model.get();
    if (const auto *p=dynamic_cast<const PotentialHybridScale *>(model))
        for (const auto *child:p->children())
            if (typeid(*child)!=typeid(PotMTP))
                throw std::runtime_error("mlip/native: hybrid_scale deployment currently requires MTP children");
    impl_->short_range=[model](Neighborhood &nbh,double &energy,SiteCoordinateMatrix &gradient,bool derivatives) {
        energy=0.;
        if (derivatives) model->CalcSiteEFS(nbh,energy,gradient);
        else model->CalcSiteE(nbh,energy);
    };
    if (auto *p=dynamic_cast<PotMTP_LES *>(model)) {
        impl_->short_range=[p](Neighborhood &nbh,double &energy,SiteCoordinateMatrix &gradient,bool derivatives) {
            energy=0.;
            if (derivatives) p->PotMTP::CalcSiteEFS(nbh,energy,gradient);
            else p->PotMTP::CalcSiteE(nbh,energy);
        };
        if(p->lr_les) {
            impl_->latent_model=p;impl_->reciprocal=&p->model_les;
            impl_->latent_species=p->les_species_coeffs;impl_->latent_moments=p->les_moment_coeffs;
            impl_->channels=p->channels();impl_->parity=p->les_response_parity;
        }
    }
    impl_->select_mtp_m<m_model_mtpli>(); impl_->select_mtp_m<m_model_mtpq2>();
    impl_->select_mtp_m<m_model_mtpfs>(); impl_->select_mtp_m<m_model_nn1>(); impl_->select_mtp_m<m_model_nn2>();
    if (impl_->options.device!="cpu") {
#ifdef IMRMLP_GPU
        if(impl_->options.device!=MLIPCore::GpuDirect::MTPGpuDirect::backend_name())
            throw std::runtime_error("mlip/native: requested GPU backend differs from the IMR-MLP build");
        impl_->gpu=std::make_unique<NativeGpuInference>(*model,options.device_id,impl_->latent_model,
            impl_->reciprocal,impl_->latent_species,impl_->latent_moments,impl_->channels,impl_->parity);
#else
        throw std::runtime_error("mlip/native: this IMR-MLP library has no GPU backend; CPU fallback is disabled");
#endif
    }
}
NativeInference::~NativeInference()=default;
int NativeInference::species_count() const {return impl_->model->species_count;}
double NativeInference::cutoff() const {return impl_->model->cutoff;}
std::vector<std::string> NativeInference::elements() const {return model_elements(*impl_->model);}
bool NativeInference::supports_local() const {return !impl_->reciprocal;}
bool NativeInference::has_charges() const {return impl_->reciprocal && impl_->reciprocal->kernel==LES::Kernel::Coulomb;}
int NativeInference::latent_channels() const {return impl_->reciprocal && impl_->reciprocal->kernel==LES::Kernel::SOG ? impl_->channels : 0;}
std::string NativeInference::description() const {
    std::set<std::string> radials;describe_radial(*impl_->model,radials);
    std::string radial;for(const auto &value:radials){if(!radial.empty())radial+=";";radial+=value;}
    return "mlip/native "+impl_->options.pottype+" radial="+radial+" device="+impl_->options.device+
        " precision=fp64 owned-center"+(supports_local()?std::string{}:" distributed-long-range");
}
NativeInferenceStatistics NativeInference::statistics() const {
#ifdef IMRMLP_GPU
    if(impl_->gpu) {
        auto result=impl_->gpu->statistics();
        result.candidate_topology_revision=impl_->statistics.candidate_topology_revision;
        return result;
    }
#endif
    return impl_->statistics;
}
AnyPotential &NativeInference::potential() {return *impl_->model;}
NativeInferenceResult NativeInference::evaluate(Atoms &atoms,const NativeInferenceRequest &request) {
    impl_->statistics.candidate_topology_revision=0;
    Neighborhoods nbhs(atoms,cutoff());
    NativeInferenceResult out;
    if(impl_->options.device=="cpu")out=impl_->cpu_frame(atoms,nbhs,request);
#ifdef IMRMLP_GPU
    else out=impl_->gpu->evaluate(atoms,nbhs,NativeCollectives{},request);
#endif
    honor_request(out,request);return out;
}
NativeInferenceResult NativeInference::evaluate_local(Atoms &atoms,AnyNeighborhoods &nbhs,
    const NativeInferenceRequest &request) {
    impl_->statistics.candidate_topology_revision=0;
    if(!supports_local())throw std::runtime_error("mlip/native: long-range model requires distributed reciprocal coordination");
    if(nbhs.size()>atoms.size())throw std::runtime_error("mlip/native: invalid owned atom count");
    NativeInferenceResult out;
    if(impl_->options.device=="cpu")out=impl_->cpu_local(atoms,nbhs,request);
#ifdef IMRMLP_GPU
    else out=impl_->gpu->evaluate(atoms,nbhs,NativeCollectives{},request);
#endif
    honor_request(out,request);return out;
}
NativeInferenceResult NativeInference::evaluate_distributed(Atoms &atoms,AnyNeighborhoods &nbhs,
    const NativeCollectives &calls,const NativeInferenceRequest &request) {
    impl_->statistics.candidate_topology_revision=0;
    NativeInferenceResult out;
    if(impl_->options.device=="cpu")out=impl_->cpu_distributed(atoms,nbhs,calls,request);
#ifdef IMRMLP_GPU
    else out=impl_->gpu->evaluate(atoms,nbhs,calls,request);
#endif
    impl_->statistics.owned_atoms=nbhs.size();impl_->statistics.ghost_atoms=atoms.size()-nbhs.size();
    impl_->statistics.neighbor_edges=0;for(int i=0;i<nbhs.size();++i)impl_->statistics.neighbor_edges+=nbhs[i].nbhcount_;
    honor_request(out,request);return out;
}

NativeInferenceResult NativeInference::evaluate_distributed_packed(Atoms &atoms,
    const NativePackedNeighborhoodView &view,const NativeCollectives &calls,
    const NativeInferenceRequest &request) {
    impl_->statistics.candidate_topology_revision=0;
    NativeInferenceResult out;
#ifdef IMRMLP_GPU
    if (impl_->gpu) out=impl_->gpu->evaluate_packed(atoms,view,calls,request);
    else
#endif
    {
        // CPU callers normally already have Neighborhoods. Keep a correct
        // fallback for the public packed contract without requiring a GPU.
        AnyNeighborhoods nbhs;
        calls.stage([&] {
            if (view.owned_count<0 || view.owned_count>view.total_count ||
                view.total_count!=atoms.size() || view.edge_count<0 || !view.offsets ||
                (view.owned_count && !view.centers) ||
                (view.total_count && (!view.types || !view.positions)) ||
                (view.edge_count && !view.neighbors) ||
                view.offsets[0]!=0 || view.offsets[view.owned_count]!=view.edge_count)
                throw std::runtime_error("mlip/native: invalid packed neighborhood dimensions");
            for (int i=0;i<view.total_count;++i)
                if (view.types[i]<0 || view.types[i]>=impl_->model->species_count)
                    throw std::runtime_error("mlip/native: packed atom type outside the model species range");
            nbhs.resize(view.owned_count);
            for (int i=0;i<view.owned_count;++i) {
                if (view.centers[i]!=i || view.offsets[i]<0 || view.offsets[i+1]<view.offsets[i] ||
                    view.offsets[i+1]>view.edge_count)
                    throw std::runtime_error("mlip/native: invalid packed neighborhood offsets/centers");
                auto &nbh=nbhs[i];nbh.centertype=view.types[i];
                for (int edge=view.offsets[i];edge<view.offsets[i+1];++edge) {
                    const int j=view.neighbors[edge];
                    if (j<0 || j>=view.total_count)
                        throw std::runtime_error("mlip/native: invalid packed neighbor index");
                    Eigen::RowVector3d delta;
                    for (int axis=0;axis<3;++axis)
                        delta[axis]=view.vectors ? view.vectors[3*edge+axis] :
                            view.positions[3*j+axis]-view.positions[3*i+axis];
                    const double squared_distance=delta.squaredNorm();
                    const double distance=view.distances ? view.distances[edge] : std::sqrt(squared_distance);
                    if (!(distance>0.) || !std::isfinite(distance) ||
                        !(squared_distance>0.) || !std::isfinite(squared_distance))
                        throw std::runtime_error("mlip/native: invalid packed neighbor geometry");
                    nbh.indices_.push_back(j);nbh.chemmap_.push_back(view.types[j]);
                    nbh.pos_.push_back(delta);nbh.distances_.push_back(distance);
                }
                nbh.nbhcount_=nbh.indices_.size();
            }
        });
        out=impl_->cpu_distributed(atoms,nbhs,calls,request);
    }
    impl_->statistics.owned_atoms=view.owned_count;
    impl_->statistics.ghost_atoms=view.total_count-view.owned_count;
    impl_->statistics.neighbor_edges=view.edge_count;
    honor_request(out,request);return out;
}

NativeInferenceResult NativeInference::evaluate_distributed_candidates(const Atoms &atoms,
    const NativeCandidateNeighborhoodView &candidates,const NativeCollectives &calls,
    const NativeInferenceRequest &request) {
    auto &prepared=impl_->candidates;
    NativeInferenceResult out;
    AnyNeighborhoods cpu_neighborhoods;
    try {
        // Replaces the public packed-validation stage, rather than adding a
        // new success-path reduction. No caller buffers survive this stage.
        calls.stage([&] {
            prepared.prepare(atoms,candidates,species_count(),cutoff());
            if(impl_->options.device=="cpu")prepared.materialize(cpu_neighborhoods);
        });
#ifdef IMRMLP_GPU
        if(impl_->gpu)
            out=impl_->gpu->evaluate_prepared(prepared.atoms,prepared.view(),calls,request);
        else
#endif
            out=impl_->cpu_distributed(prepared.atoms,cpu_neighborhoods,calls,request);
        impl_->statistics.owned_atoms=prepared.centers.size();
        impl_->statistics.ghost_atoms=prepared.atoms.size()-prepared.centers.size();
        impl_->statistics.neighbor_edges=prepared.neighbors.size();
        impl_->statistics.candidate_topology_revision=prepared.revision;
        honor_request(out,request);
    } catch(...) {
        // A peer failure can occur after local preparation succeeded. Never
        // reuse its generation on the next attempted inference.
        prepared.valid=false;
        impl_->statistics.candidate_topology_revision=0;
        throw;
    }
    return out;
}
} // namespace Potentials
