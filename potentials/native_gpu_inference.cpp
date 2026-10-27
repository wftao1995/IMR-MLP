#include "native_gpu_inference.h"
#include "mtp_gpu_adapter.h"
#ifdef IMRMLP_GPU
#include "native_inference.h"
#include "native_profile.h"
#include "native_gpu_ops.h"
#include "native_long_range_gpu.h"
#include "potential_hybrid.h"
#include "potential_hybrid_scale.h"
#include "potential_hybmtp.h"
#include "potential_mtpnn1.h"
#include "potential_zbl.h"
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace Potentials {
namespace {
NativeInferenceResult empty(int total) {
    NativeInferenceResult out;out.forces.setZero(total,3);
    out.site_energy.setZero(total);out.site_virial.setZero(total,6);return out;
}
NativeGpu::Output arrays(NativeInferenceResult &out) {
    return {out.forces.data(),out.site_energy.data(),out.site_virial.data()};
}
void summarize(NativeInferenceResult &out) {
    out.energy=out.site_energy.sum();
    const Eigen::Matrix<double,1,6> w=out.site_virial.colwise().sum();
    out.virial<<w[0],w[3],w[4],w[3],w[1],w[5],w[4],w[5],w[2];
}
void set_virial(NativeInferenceResult &out,const double *w) {
    out.virial<<w[0],w[3],w[4],w[3],w[1],w[5],w[4],w[5],w[2];
}
void honor_request(NativeInferenceResult &out,const NativeInferenceRequest &r) {
    if(!r.forces)out.forces.resize(0,3);
    if(!r.energy)out.energy=0.;
    if(!r.site_energy)out.site_energy.resize(0);
    if(!r.virial)out.virial.setZero();
    if(!r.site_virial)out.site_virial.resize(0,6);
    if(!r.charges)out.charges.resize(0);
    if(!r.latents)out.latents.resize(0,0);
}
struct PreparedOwned {};
struct Packed {
    std::vector<int> centers,offsets,neighbors;
    std::vector<double> vectors,positions,distances;
    NativePackedNeighborhoodView data;
    Atoms &atoms;
    explicit Packed(Atoms &a,AnyNeighborhoods &nbhs,int species_count):atoms(a) {
        IMRMLP_NATIVE_PROFILE_SCOPE("gpu.pack_neighborhoods");
        if(nbhs.size()>a.size())throw std::runtime_error("mlip/native: invalid owned atom count");
        centers.resize(nbhs.size());std::iota(centers.begin(),centers.end(),0);
        offsets.resize(nbhs.size()+1);positions.resize(3*a.size());
        for(int i=0;i<a.size();++i)for(int d=0;d<3;++d)positions[3*i+d]=a.pos_[i][d];
        for(int i=0;i<nbhs.size();++i) {
            auto &nbh=nbhs[i];
            for(int j=0;j<nbh.nbhcount_;++j){
                if(nbh.indices_[j]<0 || nbh.indices_[j]>=a.size() || !(nbh.distances_[j]>0.) ||
                   !nbh.pos_[j].allFinite())throw std::runtime_error("mlip/native: invalid GPU neighborhood");
                neighbors.push_back(nbh.indices_[j]);
                distances.push_back(nbh.distances_[j]);
                for(int d=0;d<3;++d)vectors.push_back(nbh.pos_[j][d]);
            }
            offsets[i+1]=neighbors.size();
        }
        data.owned_count=nbhs.size();data.total_count=a.size();data.edge_count=neighbors.size();
        data.centers=centers.data();data.offsets=offsets.data();data.neighbors=neighbors.data();
        data.types=a.chemmap_.data();data.positions=positions.data();
        data.vectors=vectors.data();data.distances=distances.data();
        validate_types(species_count);
    }
    Packed(Atoms &a,const NativePackedNeighborhoodView &v,int species_count):data(v),atoms(a) {
        IMRMLP_NATIVE_PROFILE_SCOPE("gpu.packed_validate");
        if(v.owned_count<0 || v.total_count!=a.size() || v.owned_count>v.total_count ||
           v.edge_count<0 || v.edge_count>std::numeric_limits<int>::max()/3 ||
           !v.offsets || (v.owned_count && !v.centers) ||
           (v.total_count && (!v.types || !v.positions)) || (v.edge_count && !v.neighbors) ||
           v.offsets[0]!=0 || v.offsets[v.owned_count]!=v.edge_count)
            throw std::runtime_error("mlip/native: invalid packed GPU neighborhood");
        for(int i=0;i<v.owned_count;++i)
            if(v.centers[i]!=i || v.offsets[i+1]<v.offsets[i])
                throw std::runtime_error("mlip/native: invalid packed GPU center or offsets");
        // Geometry is mutable under a stable topology token. Validate without
        // flattening or allocating, including callers that omit distances.
        {
        IMRMLP_NATIVE_PROFILE_SCOPE("gpu.positions_validate");
        for(int i=0;i<v.total_count;++i)for(int d=0;d<3;++d)
            if(!std::isfinite(v.positions[3*i+d]))
                throw std::runtime_error("mlip/native: invalid packed GPU position");
        }
        {
        IMRMLP_NATIVE_PROFILE_SCOPE("gpu.edges_validate");
        for(int i=0;i<v.owned_count;++i)for(int k=v.offsets[i];k<v.offsets[i+1];++k) {
            const int j=v.neighbors[k];
            if(j<0 || j>=v.total_count)throw std::runtime_error("mlip/native: invalid packed GPU neighbor");
            double squared=0.;
            for(int d=0;d<3;++d) {
                const double delta=v.vectors?v.vectors[3*k+d]:v.positions[3*j+d]-v.positions[3*i+d];
                squared+=delta*delta;
            }
            if(!(squared>0.) || !std::isfinite(squared) ||
               (v.distances && (!(v.distances[k]>0.) || !std::isfinite(v.distances[k]))))
                throw std::runtime_error("mlip/native: invalid packed GPU neighbor geometry");
        }
        }
        validate_types(species_count);
    }
    // Reachable only through NativeGpuInference's private friend entry. The
    // library builder owns and has validated every referenced input array.
    Packed(Atoms &a,const NativePackedNeighborhoodView &v,PreparedOwned) noexcept:data(v),atoms(a) {}
    void validate_types(int species_count) const {
        IMRMLP_NATIVE_PROFILE_SCOPE("gpu.types_validate");
        for(int i=0;i<data.total_count;++i)
            if(data.types[i]<0 || data.types[i]>=species_count)
                throw std::runtime_error("mlip/native: packed atom type is outside the species range");
    }
    // Composite kernels need explicit vectors.
    // Ordinary MTP heads directly subtract the supplied ghost coordinates.
    void geometry() {
        IMRMLP_NATIVE_PROFILE_SCOPE("gpu.materialize_geometry");
        if(!data.vectors && data.edge_count) {
            vectors.resize(3*data.edge_count);
            for(int i=0;i<data.owned_count;++i)for(int k=data.offsets[i];k<data.offsets[i+1];++k) {
                const int j=data.neighbors[k];
                if(j<0 || j>=data.total_count)throw std::runtime_error("mlip/native: invalid packed GPU neighbor");
                for(int d=0;d<3;++d)vectors[3*k+d]=data.positions[3*j+d]-data.positions[3*i+d];
            }
            data.vectors=vectors.data();
        }
        if(!data.distances && data.edge_count) {
            distances.resize(data.edge_count);
            for(int k=0;k<data.edge_count;++k) {
                double rr=0.;for(int d=0;d<3;++d)rr+=data.vectors[3*k+d]*data.vectors[3*k+d];
                distances[k]=std::sqrt(rr);
            }
            data.distances=distances.data();
        }
    }
    NativeGpu::Neighborhood view() {
        geometry();
        return {data.owned_count,data.total_count,data.edge_count,data.offsets,data.neighbors,
            data.types,data.vectors,data.distances};
    }
    NativeInferenceResult compute(MTPGpuEvaluator &model,const double *weights=nullptr,bool derivatives=true,
                                  const NativeInferenceRequest *requested=nullptr) const {
        IMRMLP_NATIVE_PROFILE_SCOPE("gpu.head_compute");
        const NativeInferenceRequest q=requested?*requested:NativeInferenceRequest{};
        const bool derivative=derivatives && (q.forces || q.virial || q.site_virial);
        NativeInferenceResult out;
        {
        IMRMLP_NATIVE_PROFILE_SCOPE("gpu.output_allocate");
        if(q.forces)out.forces.setZero(data.total_count,3);
        if(q.site_energy)out.site_energy.setZero(data.total_count);
        if(q.site_virial)out.site_virial.setZero(data.total_count,6);
        }
        if(data.owned_count && (derivative || q.energy || q.site_energy)) {
            MLIPCore::GpuDirect::MTPGpuBatchRequest r;
            r.centers=data.centers;r.offsets=data.offsets;r.neighbors=data.neighbors;
            r.neighbor_vectors=data.vectors;r.center_count=data.owned_count;r.nlocal=data.owned_count;
            r.ntotal=data.total_count;r.type=data.types;r.x=data.positions;
            r.topology_token={data.topology_owner,data.topology_revision};
            r.reuse_topology=r.topology_token.valid();r.topology_includes_types=true;
            // Geometry is mutable even when CSR indices are unchanged.
            r.center_energy_weights=weights;r.force=derivative&&q.forces?out.forces.data():nullptr;
            r.copy_forces_to_host=q.forces;
            r.site_energy=q.site_energy?out.site_energy.data():nullptr;
            r.site_virial=derivative&&q.site_virial?out.site_virial.data():nullptr;
            double w[6]{};r.global_virial=derivative&&q.virial&&!q.site_virial?w:nullptr;
            r.total_energy=&out.energy;r.accumulate_force=false;r.compute_forces=derivative;
            r.compute_global_energy=q.energy&&!q.site_energy;r.compute_site_energy=q.site_energy;
            {
            IMRMLP_NATIVE_PROFILE_SCOPE("gpu.compute_batch");
            std::string error;if(!model.context().compute_batch(r,&error))throw std::runtime_error("mlip/native: "+error);
            }
            if(q.energy && q.site_energy)out.energy=out.site_energy.sum();
            if(derivative && q.virial) {
                if(q.site_virial){const Eigen::Matrix<double,1,6> sum=out.site_virial.colwise().sum();set_virial(out,sum.data());}
                else set_virial(out,w);
            }
        }
        if(!q.forces)out.forces.resize(0,3);
        return out;
    }
};
struct Node {
    enum Kind {Head,Sum,Scale,ZBL} kind=Head;
    int device;
    std::unique_ptr<MTPGpuEvaluator> head;
    const PotentialHybridScale *scale=nullptr;
    const PotentialZBL *zbl=nullptr;
    std::vector<std::unique_ptr<Node>> children;
    std::uint64_t evaluations=0;
    Node(const AnyPotential &model,int id):device(id) {
        std::vector<const AnyPotential *> parts;
        if(const auto *p=dynamic_cast<const PotMTPHyb3 *>(&model)) {
            kind=Sum;for(auto *child:p->children())parts.push_back(child);
        } else if(const auto *p=dynamic_cast<const PotentialHybrid *>(&model)) {
            kind=Sum;parts=p->children();
        } else if(const auto *p=dynamic_cast<const PotentialHybridScale *>(&model)) {
            kind=Scale;scale=p;parts=p->children();
        } else if(const auto *p=dynamic_cast<const PotentialZBL *>(&model)) {
            kind=ZBL;zbl=p;
        } else {
            head=std::make_unique<MTPGpuEvaluator>();std::string error;bool success;
            if(const auto *p=dynamic_cast<const PotMTPNN1 *>(&model))success=head->initialize_legacy_nn(*p,id,&error);
            else success=head->initialize_short_range(model,id,&error);
            if(!success)throw std::runtime_error("mlip/native: "+error);
        }
        for(auto *child:parts)children.push_back(std::make_unique<Node>(*child,id));
    }
    NativeInferenceResult compute(Packed &p,const double *weights=nullptr,bool derivatives=true,
                                  const NativeInferenceRequest *request=nullptr) {
        if(kind==Head){auto result=p.compute(*head,weights,derivatives,request);
            if(p.data.owned_count && (!request || request->forces || request->energy || request->site_energy ||
                request->virial || request->site_virial))++evaluations;
            return result;}
        auto out=empty(p.atoms.size());
        if(kind==ZBL)NativeGpu::zbl(device,p.view(),zbl->z,zbl->r0,zbl->r1,weights,arrays(out));
        else if(kind==Sum)for(auto &child:children){auto value=child->compute(p,weights,derivatives);
            NativeGpu::add(device,p.atoms.size(),arrays(out),arrays(value));}
        else {
            if(weights)throw std::runtime_error("mlip/native: nested weighted hybrid_scale is unsupported by trainer");
            std::vector<double> bounds;
            for(const auto &b:scale->boundaries()){bounds.push_back(b.left);bounds.push_back(b.right);}
            const auto state=NativeGpu::scale(device,p.view(),scale->feature_kind()==PotentialHybridScale::FeatureKind::MinDistance,
                scale->feature_r0,scale->feature_r1,bounds);
            std::vector<double> energies(children.size()*p.atoms.size());
            for(int c=0;c<static_cast<int>(children.size());++c) {
                if(derivatives){const auto raw=children[c]->compute(p,nullptr,false);
                    std::copy_n(raw.site_energy.data(),p.atoms.size(),energies.data()+c*p.atoms.size());}
                auto weighted=children[c]->compute(p,state.weights.data()+c*p.atoms.size(),derivatives);
                NativeGpu::add(device,p.atoms.size(),arrays(out),arrays(weighted));
            }
            if(derivatives)NativeGpu::scale_response(device,p.view(),children.size(),state,energies,arrays(out));
        }
        summarize(out);return out;
    }
    void statistics(NativeInferenceStatistics &out) const {
        if(head){const auto s=head->context().runtime_stats();
            out.model_evaluations+=evaluations;
            out.force_kernel_launches+=s.basic_tile_force_launches+s.center_neighbors_force_launches;
            out.model_device_bytes+=head->context().memory_stats().tracked_device_bytes;}
        for(const auto &child:children)child->statistics(out);
    }
};
template<class LongRange> void add_field(int device,NativeInferenceResult &out,const LongRange &lr) {
    auto field=empty(out.site_energy.size());const int n=lr.per_atom_energy.size();
    field.site_energy.head(n)=lr.per_atom_energy;field.forces.topRows(n)=lr.forces;
    field.site_virial.topRows(n)=lr.per_atom_virial;
    NativeGpu::add(device,out.site_energy.size(),arrays(out),arrays(field));
}
}
struct NativeGpuInference::Impl {
    int device,channels,parity,species_count;
    const LES *reciprocal;
    std::unique_ptr<Node> short_range;
    std::vector<std::unique_ptr<MTPGpuEvaluator>> latents;
    NativeInferenceStatistics statistics;
    NativeInferenceResult evaluate(Packed &,const NativeCollectives &,const NativeInferenceRequest &);
};
NativeGpuInference::NativeGpuInference(const AnyPotential &model,int device,const PotMTP *latent_model,
    const LES *reciprocal,const double *species,const double *moments,int channels,int parity)
    :impl_(std::make_unique<Impl>()) {
    // Device -1 has already been resolved by the LAMMPS MPI placement helper.
    impl_->device=device<0?0:device;impl_->channels=channels;impl_->parity=parity;impl_->species_count=model.species_count;
    impl_->reciprocal=reciprocal;
    impl_->short_range=std::make_unique<Node>(model,impl_->device);
    std::string error;
    if(reciprocal)for(int c=0;c<channels;++c) {
        auto latent=std::make_unique<MTPGpuEvaluator>();
        if(!latent->initialize_linear_head(*latent_model,species+c*model.species_count,
            moments+c*latent_model->alpha_scalar_moments,impl_->device,&error))
            throw std::runtime_error("mlip/native: "+error);
        impl_->latents.push_back(std::move(latent));
    }
}
NativeGpuInference::~NativeGpuInference()=default;
NativeInferenceStatistics NativeGpuInference::statistics() const {
    auto out=impl_->statistics;impl_->short_range->statistics(out);
    auto add=[&](const MTPGpuEvaluator &model){const auto s=model.context().runtime_stats();
        out.force_kernel_launches+=s.basic_tile_force_launches+s.center_neighbors_force_launches;
        out.model_device_bytes+=model.context().memory_stats().tracked_device_bytes;};
    for(const auto &latent:impl_->latents)add(*latent);
    return out;
}
NativeInferenceResult NativeGpuInference::evaluate(Atoms &atoms,AnyNeighborhoods &nbhs,const NativeCollectives &calls,
    const NativeInferenceRequest &request) {
    IMRMLP_NATIVE_PROFILE_SCOPE("gpu.evaluate_neighborhoods");
    std::unique_ptr<Packed> packed;
    calls.stage([&]{packed=std::make_unique<Packed>(atoms,nbhs,impl_->species_count);});
    return impl_->evaluate(*packed,calls,request);
}
NativeInferenceResult NativeGpuInference::evaluate_packed(Atoms &atoms,const NativePackedNeighborhoodView &view,
    const NativeCollectives &calls,const NativeInferenceRequest &request) {
    IMRMLP_NATIVE_PROFILE_SCOPE("gpu.evaluate_packed");
    std::unique_ptr<Packed> packed;
    calls.stage([&]{packed=std::make_unique<Packed>(atoms,view,impl_->species_count);});
    return impl_->evaluate(*packed,calls,request);
}
NativeInferenceResult NativeGpuInference::evaluate_prepared(Atoms &atoms,const NativePackedNeighborhoodView &view,
    const NativeCollectives &calls,const NativeInferenceRequest &request) {
    IMRMLP_NATIVE_PROFILE_SCOPE("gpu.evaluate_prepared");
    Packed packed(atoms,view,PreparedOwned{});
    return impl_->evaluate(packed,calls,request);
}
NativeInferenceResult NativeGpuInference::Impl::evaluate(Packed &storage,
    const NativeCollectives &calls,const NativeInferenceRequest &request) {
    Packed *packed=&storage;Atoms &atoms=storage.atoms;
    const int n=packed->data.owned_count,total=atoms.size();
    NativeInferenceResult out;
    const bool simple=!reciprocal && short_range->kind==Node::Head;
    calls.stage([&]{
        out=short_range->compute(*packed,nullptr,true,simple?&request:nullptr);
    });
    if(reciprocal) {
        Eigen::MatrixXd owned_latents;
        calls.stage([&]{out.latents.setZero(total,channels);
            for(int c=0;c<channels;++c)out.latents.col(c)=packed->compute(*latents[c],nullptr,false).site_energy;
            owned_latents=out.latents.topRows(n);});
        const auto lr=native_les_distributed_gpu(*reciprocal,atoms,owned_latents,n,calls,device);
        calls.stage([&]{add_field(device,out,lr);Eigen::VectorXd weights=Eigen::VectorXd::Zero(total);
            for(int c=0;c<channels;++c){weights.head(n)=lr.dEdq.col(c);
                auto response=packed->compute(*latents[c],weights.data());
                NativeGpu::add(device,total,arrays(out),arrays(response),false);}
            if(reciprocal->kernel==LES::Kernel::Coulomb)out.charges=parity*out.latents.col(0);
        });
    }
    if(!simple)summarize(out);
    honor_request(out,request);statistics.owned_atoms=n;statistics.ghost_atoms=total-n;
    statistics.neighbor_edges=packed->data.edge_count;return out;
}
}
#endif
