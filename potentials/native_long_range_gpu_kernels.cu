#include "native_long_range_gpu_kernels.h"
#include "runtime/mtp/gpu-direct/gpu_runtime.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace Potentials {
namespace {
using namespace MLIPCore::GpuDirect::Runtime;
constexpr int threads=128;

void check(gpuError_t error) {
    if(error!=gpuSuccess)
        throw std::runtime_error(std::string("Native long-range GPU: ")+gpuGetErrorString(error));
}
void completed() { check(gpuGetLastError());check(gpuStreamSynchronize(nullptr)); }

template<class T> struct Buffer {
    T *data=nullptr;
    std::size_t capacity=0;
    ~Buffer() { if(data) gpuFree(data); }
    void resize(std::size_t count) {
        if(count<=capacity) return;
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T))
            throw std::runtime_error("Native long-range GPU allocation size overflow");
        T *next=nullptr;check(gpuMalloc(reinterpret_cast<void **>(&next),count*sizeof(T)));
        if(data) gpuFree(data);
        data=next;capacity=count;
    }
    void upload(const T *source,std::size_t count) {
        resize(count);
        if(count) check(gpuMemcpy(data,source,count*sizeof(T),gpuMemcpyHostToDevice));
    }
    void download(T *target,std::size_t count) const {
        if(count) check(gpuMemcpy(target,data,count*sizeof(T),gpuMemcpyDeviceToHost));
    }
    void zero(std::size_t count) {
        resize(count);
        if(count) check(gpuMemset(data,0,count*sizeof(T)));
    }
};

struct View {
    int n,nk;
    const double *x,*q,*wave,*sf;
    double *e,*f,*w,*phi;
};

template<int Count> __device__ void reduce(double (&value)[Count],double *shared) {
    const int t=threadIdx.x;
    for(int d=0;d<Count;++d) shared[d*threads+t]=value[d];
    __syncthreads();
    for(int stride=threads/2;stride;stride/=2) {
        if(t<stride) for(int d=0;d<Count;++d)
            shared[d*threads+t]+=shared[d*threads+t+stride];
        __syncthreads();
    }
    for(int d=0;d<Count;++d) value[d]=shared[d*threads];
    __syncthreads();
}

// e, Fx,Fy,Fz, Wxx,Wyy,Wzz,Wxy,Wxz,Wyz, phi.
__device__ void save(View v,int i,const double *sum) {
    v.e[i]+=sum[0];
    for(int d=0;d<3;++d) v.f[3*i+d]+=sum[d+1];
    for(int d=0;d<6;++d) v.w[6*i+d]+=sum[d+4];
    v.phi[i]+=sum[10];
}

__global__ void structure_kernel(View v,double *output) {
    __shared__ double shared[2*threads];
    const int k=blockIdx.x;const double *wave=v.wave+6*k;
    double sum[2]={};
    for(int i=threadIdx.x;i<v.n;i+=threads) {
        double s,c;sincos(wave[0]*v.x[3*i]+wave[1]*v.x[3*i+1]+wave[2]*v.x[3*i+2],&s,&c);
        sum[0]+=v.q[i]*c;sum[1]+=v.q[i]*s;
    }
    reduce(sum,shared);
    if(threadIdx.x==0) { output[2*k]=sum[0];output[2*k+1]=sum[1]; }
}

__global__ void reciprocal_kernel(View v,double prefactor) {
    __shared__ double shared[11*threads];
    const int i=blockIdx.x;double sum[11]={};
    const double q=v.q[i];
    for(int k=threadIdx.x;k<v.nk;k+=threads) {
        const double *wave=v.wave+6*k;
        const double kx=wave[0],ky=wave[1],kz=wave[2];
        double s,c;sincos(kx*v.x[3*i]+ky*v.x[3*i+1]+kz*v.x[3*i+2],&s,&c);
        const double re=c*v.sf[2*k]+s*v.sf[2*k+1],im=s*v.sf[2*k]-c*v.sf[2*k+1];
        const double energy=prefactor*wave[3]*q*re;
        const double force=2.*prefactor*wave[3]*q*im;
        const double virial=2.*prefactor*wave[4]*q*re;
        sum[0]+=energy;sum[1]+=force*kx;sum[2]+=force*ky;sum[3]+=force*kz;
        sum[4]+=energy+virial*kx*kx;sum[5]+=energy+virial*ky*ky;sum[6]+=energy+virial*kz*kz;
        sum[7]+=virial*kx*ky;sum[8]+=virial*kx*kz;sum[9]+=virial*ky*kz;
        sum[10]+=2.*prefactor*wave[3]*re;
    }
    reduce(sum,shared);
    if(threadIdx.x==0) save(v,i,sum);
}

__global__ void les_self_kernel(View v,double coefficient) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=v.n) return;
    v.e[i]-=.5*coefficient*v.q[i]*v.q[i];v.phi[i]-=coefficient*v.q[i];
}

} // namespace

struct NativeLongRangeGpu::State {
    int device,n=0,total=0,nk=0;
    Buffer<double> x,q,wave,sf,e,f,w,phi;
    explicit State(int selected):device(selected) { check(gpuSetDevice(device)); }
    ~State() { gpuSetDevice(device); }
    void activate() { check(gpuSetDevice(device)); }
    View view() { return {n,nk,x.data,q.data,wave.data,sf.data,e.data,f.data,w.data,phi.data}; }
};

NativeLongRangeGpu::NativeLongRangeGpu(int device):state_(std::make_unique<State>(device)) {}
NativeLongRangeGpu::~NativeLongRangeGpu()=default;

void NativeLongRangeGpu::atoms(int owned,int total,const double *positions,const double *charges) {
    auto &s=*state_;s.activate();s.n=owned;s.total=total;
    s.x.upload(positions,std::size_t(owned)*3);s.q.upload(charges,total);
    s.e.zero(owned);s.f.zero(std::size_t(owned)*3);s.w.zero(std::size_t(owned)*6);
    s.phi.zero(owned);
}
void NativeLongRangeGpu::charges(int total,const double *charges) {
    auto &s=*state_;s.activate();s.total=total;s.q.upload(charges,total);s.phi.zero(s.n);
}
void NativeLongRangeGpu::waves(int count,const double *waves) {
    auto &s=*state_;s.activate();s.nk=count;s.wave.upload(waves,std::size_t(count)*6);s.sf.resize(std::size_t(count)*2);
}
void NativeLongRangeGpu::structure(double *output) {
    auto &s=*state_;s.activate();
    if(s.nk) { structure_kernel<<<s.nk,threads>>>(s.view(),s.sf.data);completed();s.sf.download(output,std::size_t(s.nk)*2); }
}
void NativeLongRangeGpu::reciprocal(const double *global,double prefactor) {
    auto &s=*state_;s.activate();s.sf.upload(global,std::size_t(s.nk)*2);
    if(s.n && s.nk) { reciprocal_kernel<<<s.n,threads>>>(s.view(),prefactor);completed(); }
}
void NativeLongRangeGpu::les_self(double coefficient) {
    auto &s=*state_;s.activate();
    if(s.n) { les_self_kernel<<<(s.n+threads-1)/threads,threads>>>(s.view(),coefficient);completed(); }
}
void NativeLongRangeGpu::response(double *phi) {
    auto &s=*state_;s.activate();s.phi.download(phi,s.n);
}
void NativeLongRangeGpu::outputs(double *energy,double *force,double *virial) {
    auto &s=*state_;s.activate();s.e.download(energy,s.n);s.f.download(force,std::size_t(s.n)*3);s.w.download(virial,std::size_t(s.n)*6);
}
} // namespace Potentials
