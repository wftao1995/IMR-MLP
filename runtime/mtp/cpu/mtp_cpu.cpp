#include "mtp_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace MLIPCore::Cpu {
namespace {
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max()/b)
        throw std::overflow_error("MTP CPU workspace size overflow");
    return a*b;
}
double activate(double x, mlip::activation_type activation, double &derivative) {
    switch (activation) {
    case mlip::ACTIVATION_TANH: {
        const double value = std::tanh(x); derivative = 1.0-value*value; return value;
    }
    case mlip::ACTIVATION_RELU: derivative = x > 0.0 ? 1.0 : 0.0; return std::max(0.0,x);
    case mlip::ACTIVATION_SIGMOID: {
        const double value = 1.0/(1.0+std::exp(-x)); derivative = value*(1.0-value); return value;
    }
    case mlip::ACTIVATION_LINEAR: derivative = 1.0; return x;
    }
    throw std::invalid_argument("Unknown MTP CPU activation");
}
}

DerivativeMode derivative_mode(const char *name) {
    if (!std::strcmp(name,"auto")) return DerivativeMode::Auto;
    if (!std::strcmp(name,"cached")) return DerivativeMode::Cached;
    if (!std::strcmp(name,"recompute")) return DerivativeMode::Recompute;
    throw std::invalid_argument("derivative_mode must be auto, cached, or recompute");
}

bool has_model_coefficients(const char *filename) {
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Cannot open MTP model");
    std::string line;
    while (std::getline(input,line)) {
        const auto first=line.find_first_not_of(" \t\r");
        if(first==std::string::npos) continue;
        if(line.compare(first,9,"#MVS_v1.1")==0) break;
        if(line.compare(first,12,"model_coeffs")==0) return true;
    }
    return false;
}

Workspace::Workspace(const mlip &model) : buffer_(create_mlip_buffer(&model)) {
    q_.resize(std::max(1,model.descriptor_count));
    dq_.resize(q_.size());
    hidden1_.resize(model.model_hidden1);
    derivative1_.resize(model.model_hidden1);
    hidden2_.resize(model.model_hidden2);
    fixed_bytes_ = (2*static_cast<std::size_t>(model.alpha_moments_count)+
        2*static_cast<std::size_t>(model.max_mu)+2*static_cast<std::size_t>(model.rb_size)+
        model.max_alpha_index_basic)*sizeof(double)+
        (static_cast<std::size_t>(model.max_alpha_index_basic)+1)*sizeof(coordpow_t);
}

void Workspace::reserve(const mlip &model, std::size_t neighbors, DerivativeMode mode) {
    if (neighbors > neighbors_.size()) {
        const auto old = neighbors_.size();
        const auto capacity = std::max(neighbors,old ? old+old/2 : std::size_t(64));
        neighbors_.resize(capacity);
        displacements_.resize(capacity);
        gradients_.resize(capacity);
    }
    const auto jacobian_values = product(product(neighbors_.size(),3),model.alpha_index_basic_count);
    // Base Auto on the monotonically growing capacity, avoiding repeated
    // cached/recompute allocation oscillations at neighbor-list rebuilds.
    mode_ = mode == DerivativeMode::Auto
        ? (product(jacobian_values,sizeof(double)) <= 256*1024
            ? DerivativeMode::Cached : DerivativeMode::Recompute) : mode;
    if (mode_ == DerivativeMode::Cached) {
        const auto needed = product(product(neighbors_.size(),3),model.alpha_index_basic_count);
        if (jacobian_.size() < needed) jacobian_.resize(needed);
    } else if (!jacobian_.empty()) {
        // A switch to recomputation must actually release the large Jacobian.
        std::vector<double>().swap(jacobian_);
    }
}

std::size_t Workspace::allocated_bytes() const {
    return fixed_bytes_ + neighbors_.capacity()*sizeof(int) +
        (displacements_.capacity()+gradients_.capacity())*sizeof(std::array<double,3>) +
        (jacobian_.capacity()+q_.capacity()+dq_.capacity()+hidden1_.capacity()+
         hidden2_.capacity()+derivative1_.capacity())*sizeof(double);
}

Executor::Executor(const mlip &model) : model_(model) {
    if (model.alpha_moments_count < model.alpha_index_basic_count ||
        model.alpha_scalar_moments < 0 || model.species_count <= 0)
        throw std::invalid_argument("Invalid MTP CPU model dimensions");
    linear_ = model.site_model == mlip::MODEL_LINEAR || model.site_model == mlip::MODEL_MTPM_LI;
    const int descriptors = linear_ ? 1 : model.descriptor_count;
    if (descriptors <= 0) throw std::invalid_argument("Invalid MTP CPU descriptor count");
    std::vector<int> compact(model.alpha_moments_count,-1);
    for (int a=0; a<model.alpha_scalar_moments; ++a) {
        const int id=model.alpha_moment_mapping[a];
        if (id<0 || id>=model.alpha_moments_count) throw std::invalid_argument("Invalid scalar moment index");
        if (compact[id]<0) { compact[id]=static_cast<int>(scalar_ids_.size()); scalar_ids_.push_back(id); }
    }
    projection_.assign(product(descriptors,scalar_ids_.size()),0.0);
    for (int p=0; p<descriptors; ++p)
        for (int a=0; a<model.alpha_scalar_moments; ++a)
            projection_[p*scalar_ids_.size()+compact[model.alpha_moment_mapping[a]]] +=
                model.moment_coeffs[p*model.alpha_scalar_moments+a];
    if (linear_) {
        linear_species_.assign(model.species_coeffs,model.species_coeffs+model.species_count);
        if (model.site_model==mlip::MODEL_MTPM_LI)
            for (int s=0;s<model.species_count;++s) linear_species_[s]+=model.descriptor_species_coeffs[s];
    }
    std::vector<char> initialized(model.alpha_moments_count,0);
    std::fill_n(initialized.begin(),model.alpha_index_basic_count,1);
    for (int begin=0;begin<model.alpha_index_times_count;) {
        const int out=model.alpha_index_times[begin].id;
        if (out<model.alpha_index_basic_count || out>=model.alpha_moments_count)
            throw std::invalid_argument("Invalid MTP CPU product output");
        int end=begin+1;
        while(end<model.alpha_index_times_count && model.alpha_index_times[end].id==out) ++end;
        for(int k=begin;k<end;++k) {
            const auto &term=model.alpha_index_times[k];
            if(term.l1<0 || term.l1>=model.alpha_moments_count || term.l2<0 ||
               term.l2>=model.alpha_moments_count || !initialized[term.l1] || !initialized[term.l2] ||
               term.l1==out || term.l2==out)
                throw std::invalid_argument("MTP CPU product graph is not topologically ordered");
        }
        groups_.push_back({begin,end,out,!initialized[out]});
        initialized[out]=1; begin=end;
    }
    for(int id:scalar_ids_) if(!initialized[id]) throw std::invalid_argument("Undefined scalar moment");
}

bool Executor::accepts(int pair, double r2) const {
    if (!(r2>0.0)) return false;
    const auto *basis=model_.pair_radial_basis_count ? model_.pair_radial_bases+pair : nullptr;
    const double hi=basis ? basis->max_dist : model_.cutoff;
    if (!(r2<hi*hi)) return false;
    const auto rb=basis ? basis->rb : model_.rb;
    const double lo=basis ? basis->min_dist : model_.min_dist;
    return !((rb==mlip::RB_CHEBYPOLY4 || rb==mlip::RB_CHEBYPOLY5 || rb==mlip::RB_CHEBYCONFIGURABLE)
             && lo>0.0 && r2<lo*lo);
}

double Executor::radial(int pair, const double *r, mlip_buffers &buffer) const {
    const double distance=std::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    if (model_.pair_radial_basis_count)
        RB_CalcDerPair(&model_,model_.pair_radial_bases+pair,&buffer,distance);
    else model_.rbcalc(&model_,&buffer,distance);
    const double inverse=1.0/distance;
    for(int k=1;k<model_.max_alpha_index_basic;++k) {
        buffer.dist_powers[k]=buffer.dist_powers[k-1]*inverse;
        buffer.coords_powers[k+1]={buffer.coords_powers[k].x*r[0],
            buffer.coords_powers[k].y*r[1],buffer.coords_powers[k].z*r[2]};
    }
    for(int mu=0;mu<model_.max_mu;++mu) {
        double value=0.0,derivative=0.0;
        const auto offset=(static_cast<std::size_t>(pair)*model_.radial_func_count+mu)*model_.rb_size;
        const double *coefficients=model_.radial_coeffs+offset;
        #pragma omp simd reduction(+:value,derivative)
        for(int k=0;k<model_.rb_size;++k) {
            value+=coefficients[k]*buffer.rb_vals[k];
            derivative+=coefficients[k]*buffer.rb_ders[k];
        }
        buffer.val_mu[mu]=value; buffer.der_mu[mu]=derivative;
    }
    return inverse;
}

void Executor::basic(const double *r, double inverse, mlip_buffers &buffer, double *jacobian,
                     double *gradient, bool accumulate) const {
    const int count=model_.alpha_index_basic_count;
    double g0=0.0,g1=0.0,g2=0.0;
    #pragma omp simd reduction(+:g0,g1,g2)
    for(int b=0;b<count;++b) {
        const auto &index=model_.alpha_index_basic[b];
        const int degree=index.b1+index.b2+index.b3;
        const double radial=buffer.val_mu[index.mu]*buffer.dist_powers[degree];
        const double derivative=buffer.der_mu[index.mu]*buffer.dist_powers[degree]-degree*radial*inverse;
        const double x=buffer.coords_powers[index.b1+1].x;
        const double y=buffer.coords_powers[index.b2+1].y;
        const double z=buffer.coords_powers[index.b3+1].z;
        if(accumulate) buffer.moment_vals[b]+=radial*x*y*z;
        if(jacobian || gradient) {
            const double common=derivative*inverse*x*y*z;
            const double dx=common*r[0]+index.b1*radial*buffer.coords_powers[index.b1].x*y*z;
            const double dy=common*r[1]+index.b2*radial*buffer.coords_powers[index.b2].y*x*z;
            const double dz=common*r[2]+index.b3*radial*buffer.coords_powers[index.b3].z*x*y;
            if(jacobian) { jacobian[b]=dx; jacobian[count+b]=dy; jacobian[2*count+b]=dz; }
            if(gradient) {
                const double adjoint=buffer.site_energy_ders_wrt_moments_[b];
                g0+=adjoint*dx; g1+=adjoint*dy; g2+=adjoint*dz;
            }
        }
    }
    if(gradient) { gradient[0]=g0; gradient[1]=g1; gradient[2]=g2; }
}

double Executor::head(Workspace &w, bool energy, bool derivatives) const {
    const auto &m=model_; const double *c=m.model_coeffs;
    const int q=m.descriptor_count,h1=m.model_hidden1,h2=m.model_hidden2;
    double result=0.0;
    if(m.site_model==mlip::MODEL_MTPM_Q2) {
        const double x=w.q_[0],y=w.q_[1];
        if(derivatives) { w.dq_[0]=c[0]+2.0*c[2]*x+c[4]*y; w.dq_[1]=c[1]+2.0*c[3]*y+c[4]*x; }
        return energy ? c[0]*x+c[1]*y+c[2]*x*x+c[3]*y*y+c[4]*x*y : 0.0;
    }
    if(m.site_model==mlip::MODEL_MTPM_FS) {
        if(derivatives) { w.dq_[0]=1.0; w.dq_[1]=c[0]/std::sqrt(1.0+w.q_[1]*w.q_[1]); }
        return energy ? w.q_[0]+c[0]*std::asinh(w.q_[1]) : 0.0;
    }
    const bool nn2=m.site_model==mlip::MODEL_MTPM_NN2;
    if(!nn2 && m.site_model!=mlip::MODEL_MTPM_NN1)
        throw std::invalid_argument("Unsupported MTP CPU head");
    const double *b1=c+q*h1+(nn2 ? 0 : h1);
    const double *weights2=nn2 ? b1+h1 : nullptr;
    const double *b2=nn2 ? weights2+h1*h2 : nullptr;
    const double *output=nn2 ? b2+h2 : c+q*h1;
    if(derivatives) std::fill(w.dq_.begin(),w.dq_.end(),0.0);
    for(int h=0;h<h1;++h) {
        double value=b1[h];
        #pragma omp simd reduction(+:value)
        for(int p=0;p<q;++p) value+=c[h*q+p]*w.q_[p];
        w.hidden1_[h]=activate(value,m.model_activation,w.derivative1_[h]);
        if(!nn2 && energy) result+=output[h]*w.hidden1_[h];
    }
    if(nn2) {
        // hidden2_ stores the head adjoint of each second-layer preactivation.
        for(int h=0;h<h2;++h) {
            double value=b2[h],derivative=0.0;
            #pragma omp simd reduction(+:value)
            for(int k=0;k<h1;++k) value+=weights2[h*h1+k]*w.hidden1_[k];
            const double activated=activate(value,m.model_activation,derivative);
            if(energy) result+=output[h]*activated;
            w.hidden2_[h]=output[h]*derivative;
        }
    }
    if(derivatives) for(int h=0;h<h1;++h) {
        double response=nn2 ? 0.0 : output[h];
        if(nn2) for(int k=0;k<h2;++k) response+=weights2[k*h1+h]*w.hidden2_[k];
        response*=w.derivative1_[h];
        #pragma omp simd
        for(int p=0;p<q;++p) w.dq_[p]+=c[h*q+p]*response;
    }
    return result;
}

double Executor::evaluate_site(int center, int neighbor_count, const int *neighbors,
    const int *types, const double *const *positions, const int *mapping,
    bool energy, bool derivatives, Workspace &w) const {
    if(neighbor_count<0 || static_cast<std::size_t>(neighbor_count)>w.capacity())
        throw std::length_error("MTP CPU workspace must be reserved before evaluation");
    const int species=mapping[types[center]];
    if(species<0 || species>=model_.species_count) throw std::invalid_argument("Invalid center species");
    auto &buffer=*w.buffer_; const int basic_count=model_.alpha_index_basic_count;
    std::fill_n(buffer.moment_vals,basic_count,0.0);
    w.edge_count_=0;
    const bool cached=derivatives && w.mode_==DerivativeMode::Cached;
    for(int jj=0;jj<neighbor_count;++jj) {
        const int j=neighbors[jj]&mlip_neighbor_index_mask;
        const int neighbor_species=mapping[types[j]];
        if(neighbor_species<0 || neighbor_species>=model_.species_count)
            throw std::invalid_argument("Invalid neighbor species");
        const int pair=species*model_.species_count+neighbor_species;
        const double r[3]={positions[j][0]-positions[center][0],positions[j][1]-positions[center][1],positions[j][2]-positions[center][2]};
        if(!accepts(pair,r[0]*r[0]+r[1]*r[1]+r[2]*r[2])) continue;
        const int edge=w.edge_count_++;
        w.neighbors_[edge]=j; w.displacements_[edge]={r[0],r[1],r[2]};
        const double inverse=radial(pair,r,buffer);
        basic(r,inverse,buffer,cached ? w.jacobian_.data()+static_cast<std::size_t>(edge)*3*basic_count : nullptr,nullptr,true);
    }
    for(const auto &group:groups_) {
        double value=group.assign ? 0.0 : buffer.moment_vals[group.output];
        for(int k=group.begin;k<group.end;++k) {
            const auto &term=model_.alpha_index_times[k];
            value+=term.c*buffer.moment_vals[term.l1]*buffer.moment_vals[term.l2];
        }
        buffer.moment_vals[group.output]=value;
    }
    double result=energy ? (linear_ ? linear_species_[species] : model_.species_coeffs[species]) : 0.0;
    const int scalars=static_cast<int>(scalar_ids_.size());
    if(linear_) {
        if(energy) for(int a=0;a<scalars;++a) result+=projection_[a]*buffer.moment_vals[scalar_ids_[a]];
    } else {
        for(int p=0;p<model_.descriptor_count;++p) {
            double value=model_.descriptor_species_coeffs[p*model_.species_count+species];
            const double *weights=projection_.data()+static_cast<std::size_t>(p)*scalars;
            #pragma omp simd reduction(+:value)
            for(int a=0;a<scalars;++a) value+=weights[a]*buffer.moment_vals[scalar_ids_[a]];
            w.q_[p]=value;
        }
        result+=head(w,energy,derivatives);
    }
    if(!derivatives) return result;
    auto *adjoint=buffer.site_energy_ders_wrt_moments_;
    std::fill_n(adjoint,model_.alpha_moments_count,0.0);
    for(int a=0;a<scalars;++a) {
        double value=linear_ ? projection_[a] : 0.0;
        if(!linear_) for(int p=0;p<model_.descriptor_count;++p) value+=projection_[static_cast<std::size_t>(p)*scalars+a]*w.dq_[p];
        adjoint[scalar_ids_[a]]=value;
    }
    for(int k=model_.alpha_index_times_count-1;k>=0;--k) {
        const auto &term=model_.alpha_index_times[k]; const double response=term.c*adjoint[term.id];
        adjoint[term.l1]+=response*buffer.moment_vals[term.l2];
        adjoint[term.l2]+=response*buffer.moment_vals[term.l1];
    }
    for(int edge=0;edge<w.edge_count_;++edge) {
        auto *gradient=w.gradients_[edge].data();
        if(cached) {
            const double *jac=w.jacobian_.data()+static_cast<std::size_t>(edge)*3*basic_count;
            double g0=0.0,g1=0.0,g2=0.0;
            #pragma omp simd reduction(+:g0,g1,g2)
            for(int b=0;b<basic_count;++b) {
                g0+=adjoint[b]*jac[b]; g1+=adjoint[b]*jac[basic_count+b]; g2+=adjoint[b]*jac[2*basic_count+b];
            }
            gradient[0]=g0; gradient[1]=g1; gradient[2]=g2;
        } else {
            const int pair=species*model_.species_count+mapping[types[w.neighbors_[edge]]];
            const double inverse=radial(pair,w.displacements_[edge].data(),buffer);
            basic(w.displacements_[edge].data(),inverse,buffer,nullptr,gradient,false);
        }
    }
    return result;
}
} // namespace MLIPCore::Cpu
