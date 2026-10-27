// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "sog_gpu.h"
#include "gpu_runtime.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace MLIPCore::GpuDirect {
namespace {
using namespace Runtime;
constexpr int threads = 128;
constexpr double pi = 3.141592653589793238462643383279502884;

bool fail(std::string *error, const std::string &message) {
    if (error) *error = message;
    return false;
}
bool check(gpuError_t result, std::string *error) {
    return result == gpuSuccess || fail(error, std::string("SOG GPU: ") + gpuGetErrorString(result));
}
template<class T> struct Buffer {
    T *data = nullptr;
    size_t capacity = 0;
    ~Buffer() { if (data) gpuFree(data); }
    bool resize(size_t count, std::string *error) {
        if (count <= capacity) return true;
        if (count > std::numeric_limits<size_t>::max()/sizeof(T))
            return fail(error, "SOG GPU allocation size overflow");
        T *next = nullptr;
        if (!check(gpuMalloc(reinterpret_cast<void **>(&next), count*sizeof(T)), error)) return false;
        if (data) gpuFree(data);
        data = next; capacity = count;
        return true;
    }
    bool upload(const T *source, size_t count, std::string *error) {
        return resize(count,error) && (!count || check(gpuMemcpy(data,source,count*sizeof(T),gpuMemcpyHostToDevice),error));
    }
    size_t bytes() const { return capacity*sizeof(T); }
};

struct View {
    int n, c, m, b, nk;
    const int *ao, *ko, *as, *ks;
    const double *volume, *k, *mult, *taper, *tp, *w, *u;
    double *q, *r, *phi, *gq, *gtheta, *sf, *dsf, *h, *hp, *self;
};

// Fixed tree reductions keep repeats deterministic without relying on a
// hardware warp size. Every thread, including zero-valued tails, participates.
template<int Count>
__device__ void reduce(double (&values)[Count], double *shared) {
    const int t = threadIdx.x;
    for (int j=0;j<Count;++j) shared[j*threads+t]=values[j];
    __syncthreads();
    for (int stride=threads/2;stride;stride/=2) {
        if (t<stride) for(int j=0;j<Count;++j)
            shared[j*threads+t]+=shared[j*threads+t+stride];
        __syncthreads();
    }
    for (int j=0;j<Count;++j) values[j]=shared[j*threads];
    __syncthreads();
}

__global__ void kernel_values(View v, bool onsite) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index<v.c*v.nk) {
        const int channel=index/v.nk, k=index%v.nk;
        const double x=v.k[3*k]*v.k[3*k]+v.k[3*k+1]*v.k[3*k+1]+v.k[3*k+2]*v.k[3*k+2];
        double g=0.,gp=0.;
        for(int m=channel*v.m;m<(channel+1)*v.m;++m) {
            const double inv=exp(-2*v.u[m]);
            const double term=v.w[m]*exp(-x*inv);
            g+=term; gp-=inv*term;
        }
        v.h[index]=v.mult[k]*v.taper[k]*g;
        v.hp[index]=v.mult[k]*(v.taper[k]*gp+v.tp[k]*g);
    }
    if (index<v.c) {
        double s=0.;
        if(onsite) for(int m=index*v.m;m<(index+1)*v.m;++m)
            s+=v.w[m]*exp(3*v.u[m])/(8*pow(pi,1.5));
        v.self[index]=s;
    }
}

template<bool Backward>
__global__ void structure_factors(View v, const double *positions, const double *force_adjoint) {
    __shared__ double shared[(Backward ? 4 : 2)*threads];
    const int k=blockIdx.x, channel=blockIdx.y, b=v.ks[k];
    const double kx=v.k[3*k],ky=v.k[3*k+1],kz=v.k[3*k+2];
    double sums[Backward ? 4 : 2]={};
    for(int i=v.ao[b]+threadIdx.x;i<v.ao[b+1];i+=threads) {
        double s,c;
        sincos(kx*positions[3*i]+ky*positions[3*i+1]+kz*positions[3*i+2],&s,&c);
        const size_t at=static_cast<size_t>(channel)*v.n+i;
        if constexpr(Backward) {
            const double wk=kx*force_adjoint[3*i]+ky*force_adjoint[3*i+1]+kz*force_adjoint[3*i+2];
            sums[0]+=v.r[at]*c; sums[1]+=v.r[at]*s;
            sums[2]+=v.q[at]*wk*c; sums[3]+=v.q[at]*wk*s;
        } else { sums[0]+=v.q[at]*c; sums[1]+=v.q[at]*s; }
    }
    reduce(sums,shared);
    if(threadIdx.x==0) {
        const size_t at=static_cast<size_t>(channel)*v.nk+k;
        if constexpr(Backward) for(int d=0;d<4;++d) v.dsf[4*at+d]=sums[d];
        else { v.sf[2*at]=sums[0];v.sf[2*at+1]=sums[1]; }
    }
}

template<bool Forces, bool Virial>
__global__ void gather_forward(View v,const double *positions,double *energy,double *force,double *virial) {
    __shared__ double shared[10*threads];
    const int i=blockIdx.x,b=v.as[i];
    const double iv=1./v.volume[b];
    // One block owns an atom, including all its channels: no output atomics.
    for(int channel=0;channel<v.c;++channel) {
        double sum[10]={};
        const size_t at=static_cast<size_t>(channel)*v.n+i;
        const double q=v.q[at];
        for(int k=v.ko[b]+threadIdx.x;k<v.ko[b+1];k+=threads) {
            const size_t ck=static_cast<size_t>(channel)*v.nk+k;
            const double kx=v.k[3*k],ky=v.k[3*k+1],kz=v.k[3*k+2];
            double s,c;sincos(kx*positions[3*i]+ky*positions[3*i+1]+kz*positions[3*i+2],&s,&c);
            const double re=c*v.sf[2*ck]+s*v.sf[2*ck+1];
            sum[0]+=iv*v.h[ck]*re;
            if constexpr(Forces) {
                const double f=iv*v.h[ck]*q*(s*v.sf[2*ck]-c*v.sf[2*ck+1]);
                sum[1]+=f*kx;sum[2]+=f*ky;sum[3]+=f*kz;
            }
            if constexpr(Virial) {
                const double e=.5*iv*v.h[ck]*q*re, w=iv*v.hp[ck]*q*re;
                sum[4]+=e+w*kx*kx;sum[5]+=e+w*ky*ky;sum[6]+=e+w*kz*kz;
                sum[7]+=w*kx*ky;sum[8]+=w*kx*kz;sum[9]+=w*ky*kz;
            }
        }
        reduce(sum,shared);
        if(threadIdx.x==0) {
            v.phi[at]=sum[0]-v.self[channel]*q;
            if(energy) energy[i]+=.5*q*v.phi[at];
            if constexpr(Forces) for(int d=0;d<3;++d) force[3*i+d]+=sum[1+d];
            if constexpr(Virial) for(int d=0;d<6;++d) virial[6*i+d]+=sum[4+d];
        }
    }
}

__device__ double strain_contraction(const double *w,const double *k) {
    double sum=0.;for(int a=0;a<3;++a)for(int b=0;b<3;++b)sum+=k[a]*w[3*a+b]*k[b];
    return sum;
}
__global__ void gather_backward(View v,const double *positions,const double *force,const double *virial) {
    __shared__ double shared[threads];
    const int i=blockIdx.x,b=v.as[i];
    const double *w=virial+9*b;
    const double trace=w[0]+w[4]+w[8],iv=1./v.volume[b];
    for(int channel=0;channel<v.c;++channel) {
        double sum[1]={};
        for(int k=v.ko[b]+threadIdx.x;k<v.ko[b+1];k+=threads) {
            const size_t ck=static_cast<size_t>(channel)*v.nk+k;
            const double *kv=v.k+3*k;
            double s,c;sincos(kv[0]*positions[3*i]+kv[1]*positions[3*i+1]+kv[2]*positions[3*i+2],&s,&c);
            const double re=c*v.sf[2*ck]+s*v.sf[2*ck+1],im=s*v.sf[2*ck]-c*v.sf[2*ck+1];
            const double wk=kv[0]*force[3*i]+kv[1]*force[3*i+1]+kv[2]*force[3*i+2];
            sum[0]+=iv*(v.h[ck]*(wk*im+c*v.dsf[4*ck+3]-s*v.dsf[4*ck+2]
                         +c*v.dsf[4*ck]+s*v.dsf[4*ck+1])
                     +(v.h[ck]*trace+2*v.hp[ck]*strain_contraction(w,kv))*re);
        }
        reduce(sum,shared);
        const size_t at=static_cast<size_t>(channel)*v.n+i;
        if(threadIdx.x==0) v.gq[at]=sum[0]-v.self[channel]*v.r[at];
    }
}

__global__ void kernel_gradient_kernel(View v,const double *energy,const double *virial,bool onsite) {
    __shared__ double shared[2*threads];
    const int m=blockIdx.x,channel=m/v.m;
    const double inv=exp(-2*v.u[m]);
    double sums[2]={};
    for(int k=threadIdx.x;k<v.nk;k+=threads) {
        const int b=v.ks[k];const double *kv=v.k+3*k,*w=virial+9*b;
        const size_t ck=static_cast<size_t>(channel)*v.nk+k;
        const double sr=v.sf[2*ck],si=v.sf[2*ck+1],sk2=sr*sr+si*si;
        const double x=kv[0]*kv[0]+kv[1]*kv[1]+kv[2]*kv[2];
        const double value=(.5*(energy[b]+w[0]+w[4]+w[8])*sk2
            +sr*v.dsf[4*ck+3]-si*v.dsf[4*ck+2]
            +sr*v.dsf[4*ck]+si*v.dsf[4*ck+1])*v.mult[k]/v.volume[b];
        const double derivative=sk2*strain_contraction(w,kv)*v.mult[k]/v.volume[b];
        const double a=exp(-x*inv),t=v.taper[k],tp=v.tp[k];
        sums[0]+=a*(value*t+derivative*(tp-t*inv));
        sums[1]+=2*v.w[m]*a*inv*(value*x*t+derivative*(x*tp+(1-x*inv)*t));
    }
    if(onsite) for(int i=threadIdx.x;i<v.n;i+=threads) {
        const size_t at=static_cast<size_t>(channel)*v.n+i;
        const double q=v.q[at],c=exp(3*v.u[m])/(8*pow(pi,1.5));
        const double value=(-.5*energy[v.as[i]]*q*q-v.r[at]*q)*c;
        sums[0]+=value;sums[1]+=3*v.w[m]*value;
    }
    reduce(sums,shared);
    if(threadIdx.x==0){v.gtheta[m]=sums[0];v.gtheta[v.c*v.m+m]=sums[1];}
}
} // namespace

struct SOGGpu::State {
    int device=-1;
    View v{};
    bool ready=false,forward_valid=false,onsite=true;
    GpuCacheToken grid_token;
    std::vector<double> weights, widths;
    std::uint64_t upload_bytes=0, reuses=0;
    Buffer<int> ao,ko,as,ks;
    Buffer<double> volume,k,mult,taper,tp,w,u,q,r,phi,gq,gtheta,sf,dsf,h,hp,self;
    std::vector<int> atom_structure,wave_structure;
};
SOGGpu::SOGGpu():state_(new State){}
SOGGpu::~SOGGpu(){if(state_->device>=0)gpuSetDevice(state_->device);delete state_;}

bool SOGGpu::prepare(const SOGGpuModelView &m,const SOGGpuGridView &g,std::string *error) {
    auto &s=*state_;
    const bool was_ready=s.ready;
    const bool reuse_grid=s.ready && g.cache_token.valid() && g.cache_token==s.grid_token
        && s.v.n==g.atoms && s.v.b==g.structures && s.v.nk==g.waves;
    s.ready=s.forward_valid=false;
    s.grid_token={};
    int device=-1;if(!check(gpuGetDevice(&device),error))return false;
    if(s.device>=0 && s.device!=device)return fail(error,"SOG GPU context belongs to a different device");
    s.device=device;
    if(m.channels<1||m.channels>16||m.terms<1||m.terms>256||!m.weights||!m.log_bandwidths
       ||g.atoms<1||g.atoms>std::numeric_limits<int>::max()/3||g.structures<1||g.waves<0
       ||g.waves>std::numeric_limits<int>::max()/std::max(3,m.channels)
       ||!g.atom_offsets||!g.wave_offsets||!g.volumes
       ||(g.waves && (!g.wave_vectors||!g.multiplicities||!g.taper||!g.taper_derivative)))
        return fail(error,"Invalid SOG GPU model/grid dimensions or pointers");
    if(g.atom_offsets[0]!=0||g.atom_offsets[g.structures]!=g.atoms||g.wave_offsets[0]!=0||g.wave_offsets[g.structures]!=g.waves)
        return fail(error,"Invalid SOG GPU structure offsets");
    if (!reuse_grid) {
    s.atom_structure.resize(g.atoms);s.wave_structure.resize(g.waves);
    for(int b=0;b<g.structures;++b) {
        if(g.atom_offsets[b]<0||g.atom_offsets[b+1]<g.atom_offsets[b]||g.atom_offsets[b+1]>g.atoms
           ||g.wave_offsets[b]<0||g.wave_offsets[b+1]<g.wave_offsets[b]||g.wave_offsets[b+1]>g.waves
           ||!(g.volumes[b]>1e-12)||!std::isfinite(g.volumes[b]))return fail(error,"Invalid SOG GPU cell/offsets");
        std::fill(s.atom_structure.begin()+g.atom_offsets[b],s.atom_structure.begin()+g.atom_offsets[b+1],b);
        std::fill(s.wave_structure.begin()+g.wave_offsets[b],s.wave_structure.begin()+g.wave_offsets[b+1],b);
    }
    }
    for(int i=0;i<m.channels*m.terms;++i)if(!std::isfinite(m.weights[i])||!std::isfinite(m.log_bandwidths[i])
        ||std::abs(m.log_bandwidths[i])>30)return fail(error,"Invalid SOG GPU Gaussian coefficient");
    if (!reuse_grid) for(int i=0;i<g.waves;++i)if(!std::isfinite(g.multiplicities[i])||!std::isfinite(g.taper[i])||!std::isfinite(g.taper_derivative[i])
        ||!std::isfinite(g.wave_vectors[3*size_t(i)])||!std::isfinite(g.wave_vectors[3*size_t(i)+1])||!std::isfinite(g.wave_vectors[3*size_t(i)+2]))
        return fail(error,"Non-finite SOG GPU grid");
#define UP(field,src,count) if(!s.field.upload(src,count,error))return false
#define RES(field,count) if(!s.field.resize(count,error))return false
    if (!reuse_grid) {
    UP(ao,g.atom_offsets,size_t(g.structures)+1);UP(ko,g.wave_offsets,size_t(g.structures)+1);
    UP(as,s.atom_structure.data(),g.atoms);UP(ks,s.wave_structure.data(),g.waves);
    UP(volume,g.volumes,g.structures);UP(k,g.wave_vectors,3*size_t(g.waves));
    UP(mult,g.multiplicities,g.waves);UP(taper,g.taper,g.waves);UP(tp,g.taper_derivative,g.waves);
    s.upload_bytes += (2*(static_cast<std::uint64_t>(g.structures)+1)+g.atoms+g.waves)*sizeof(int)
        +(static_cast<std::uint64_t>(g.structures)+6*static_cast<std::uint64_t>(g.waves))*sizeof(double);
    } else ++s.reuses;
    const size_t parameters=size_t(m.channels)*m.terms;
    const bool model_changed=!was_ready || s.v.c!=m.channels || s.v.m!=m.terms || s.onsite!=m.onsite
        || s.weights.size()!=parameters || s.widths.size()!=parameters
        || !std::equal(s.weights.begin(),s.weights.end(),m.weights)
        || !std::equal(s.widths.begin(),s.widths.end(),m.log_bandwidths);
    if (model_changed) {
        UP(w,m.weights,parameters);UP(u,m.log_bandwidths,parameters);
    }
    const size_t nc=size_t(g.atoms)*m.channels,kc=size_t(g.waves)*m.channels;
    RES(q,nc);RES(r,nc);RES(phi,nc);RES(gq,nc);RES(gtheta,2*m.channels*m.terms);
    RES(sf,2*kc);RES(dsf,4*kc);RES(h,kc);RES(hp,kc);RES(self,m.channels);
#undef UP
#undef RES
    s.v={g.atoms,m.channels,m.terms,g.structures,g.waves,s.ao.data,s.ko.data,s.as.data,s.ks.data,
        s.volume.data,s.k.data,s.mult.data,s.taper.data,s.tp.data,s.w.data,s.u.data,
        s.q.data,s.r.data,s.phi.data,s.gq.data,s.gtheta.data,s.sf.data,s.dsf.data,s.h.data,s.hp.data,s.self.data};
    s.onsite=m.onsite;
    if (!reuse_grid || model_changed) kernel_values<<<(std::max(kc,size_t(m.channels))+threads-1)/threads,threads>>>(s.v,m.onsite);
    s.ready=check(gpuGetLastError(),error);
    if (s.ready) {
        s.grid_token=g.cache_token;
        s.weights.assign(m.weights,m.weights+parameters);
        s.widths.assign(m.log_bandwidths,m.log_bandwidths+parameters);
    }
    return s.ready;
}
double *SOGGpu::latents(){return state_->q.data;}
double *SOGGpu::response(){return state_->r.data;}
const double *SOGGpu::phi() const{return state_->phi.data;}
const double *SOGGpu::charge_vjp() const{return state_->gq.data;}
const double *SOGGpu::kernel_gradient() const{return state_->gtheta.data;}

bool SOGGpu::forward(const double *positions,double *energy,double *force,double *virial,std::string *error) {
    auto &s=*state_;s.forward_valid=false;
    if(!s.ready||!positions)return fail(error,"SOG GPU forward requires prepared geometry and device positions");
    if(!check(gpuSetDevice(s.device),error))return false;
    if(s.v.nk)structure_factors<false><<<dim3(s.v.nk,s.v.c),threads>>>(s.v,positions,nullptr);
    if(force && virial)gather_forward<true,true><<<s.v.n,threads>>>(s.v,positions,energy,force,virial);
    else if(force)gather_forward<true,false><<<s.v.n,threads>>>(s.v,positions,energy,force,virial);
    else if(virial)gather_forward<false,true><<<s.v.n,threads>>>(s.v,positions,energy,force,virial);
    else gather_forward<false,false><<<s.v.n,threads>>>(s.v,positions,energy,force,virial);
    s.forward_valid=check(gpuGetLastError(),error);return s.forward_valid;
}
bool SOGGpu::backward(const double *positions,const double *energy,const double *force,const double *virial,std::string *error) {
    auto &s=*state_;
    if(!s.forward_valid||!positions||!energy||!force||!virial)return fail(error,"SOG GPU backward requires forward and device adjoints");
    if(!check(gpuSetDevice(s.device),error))return false;
    if(s.v.nk)structure_factors<true><<<dim3(s.v.nk,s.v.c),threads>>>(s.v,positions,force);
    gather_backward<<<s.v.n,threads>>>(s.v,positions,force,virial);
    kernel_gradient_kernel<<<s.v.c*s.v.m,threads>>>(s.v,energy,virial,s.onsite);
    return check(gpuGetLastError(),error);
}
std::uint64_t SOGGpu::grid_upload_bytes() const { return state_->upload_bytes; }
std::uint64_t SOGGpu::grid_reuses() const { return state_->reuses; }
size_t SOGGpu::memory_bytes() const {
    const auto &s=*state_;
    return s.ao.bytes()+s.ko.bytes()+s.as.bytes()+s.ks.bytes()+s.volume.bytes()+s.k.bytes()+s.mult.bytes()
        +s.taper.bytes()+s.tp.bytes()+s.w.bytes()+s.u.bytes()+s.q.bytes()+s.r.bytes()+s.phi.bytes()+s.gq.bytes()
        +s.gtheta.bytes()+s.sf.bytes()+s.dsf.bytes()+s.h.bytes()+s.hp.bytes()+s.self.bytes();
}
} // namespace MLIPCore::GpuDirect
