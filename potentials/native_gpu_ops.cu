#include "native_gpu_ops.h"
#include "runtime/mtp/gpu-direct/gpu_runtime.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace Potentials::NativeGpu {
using namespace MLIPCore::GpuDirect::Runtime;
namespace {
void check(gpuError_t status) {
    if(status!=gpuSuccess) throw std::runtime_error(std::string("mlip/native GPU: ")+gpuGetErrorString(status));
}
template<class T> struct Buffer {
    T *p=nullptr;std::size_t n;
    explicit Buffer(std::size_t size,const T *host=nullptr):n(size) {
        if(n) {check(gpuMalloc(reinterpret_cast<void **>(&p),n*sizeof(T)));
            try {if(host)check(gpuMemcpy(p,host,n*sizeof(T),gpuMemcpyHostToDevice));
                 else check(gpuMemset(p,0,n*sizeof(T)));}
            catch(...) {gpuFree(p);throw;}}
    }
    ~Buffer(){if(p)gpuFree(p);}
    void download(T *host){if(n)check(gpuMemcpy(host,p,n*sizeof(T),gpuMemcpyDeviceToHost));}
};
struct Geometry {
    Buffer<int> offsets,neighbors,types;
    Buffer<double> vectors,distances;
    Neighborhood view;
    explicit Geometry(Neighborhood input):offsets(input.nlocal+1,input.offsets),
        neighbors(input.edges,input.neighbors),types(input.ntotal,input.types),
        vectors(3*input.edges,input.vectors),distances(input.edges,input.distances),view(input) {
        view.offsets=offsets.p;view.neighbors=neighbors.p;view.types=types.p;view.vectors=vectors.p;view.distances=distances.p;
    }
};
struct Result {
    Buffer<double> forces,energy,virial;
    explicit Result(int n,Output host={}):forces(3*n,host.forces),energy(n,host.energy),virial(6*n,host.virial){}
    Output view(){return {forces.p,energy.p,virial.p};}
    void download(Output host){forces.download(host.forces);energy.download(host.energy);virial.download(host.virial);}
};
__device__ void edge(Output out,int i,int j,const double *r,const double *g) {
    for(int d=0;d<3;++d){atomicAdd(out.forces+3*i+d,g[d]);atomicAdd(out.forces+3*j+d,-g[d]);}
    const double w[6]={-g[0]*r[0],-g[1]*r[1],-g[2]*r[2],
        -.5*(g[0]*r[1]+g[1]*r[0]),-.5*(g[0]*r[2]+g[2]*r[0]),-.5*(g[1]*r[2]+g[2]*r[1])};
    for(int d=0;d<6;++d){atomicAdd(out.virial+6*i+d,.5*w[d]);atomicAdd(out.virial+6*j+d,.5*w[d]);}
}
__device__ void smooth(double a,double b,double x,double &s,double &d) {
    if(x<=a){s=0.;d=0.;return;}if(x>=b){s=1.;d=0.;return;}
    const double t=(x-a)/(b-a);s=t*t*t*(10.+t*(-15.+6.*t));d=30.*t*t*(1.-t)*(1.-t)/(b-a);
}
__global__ void addition(int n,Output out,Output source,bool energies) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(energies)out.energy[i]+=source.energy[i];
    for(int d=0;d<3;++d)out.forces[3*i+d]+=source.forces[3*i+d];
    for(int d=0;d<6;++d)out.virial[6*i+d]+=source.virial[6*i+d];
}
__global__ void zbl_kernel(Neighborhood n,const double *z,double r0,double r1,
                           const double *weights,Output out) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n.nlocal)return;
    const int zi=static_cast<int>(z[n.types[i]]);double energy=0.;
    for(int e=n.offsets[i];e<n.offsets[i+1];++e) {
        const double *v=n.vectors+3*e;const double r=n.distances[e];
        if(r>=r1)continue;
        const int j=n.neighbors[e],zj=static_cast<int>(z[n.types[j]]);
        const double a=.46850/(pow(double(zi),.23)+pow(double(zj),.23));
        const double c[4]={.18175,.50986,.28022,.02817},p[4]={-3.19980,-.94229,-.40290,-.20162};
        double sum=0.,dsum=0.;for(int k=0;k<4;++k){const double t=c[k]*exp(p[k]*r/a);sum+=t;dsum+=t*p[k];}
        const double factor=14.3997*zi*zj/r,value=factor*sum,derivative=factor*(dsum/a-sum/r);
        double s=1.,ds=0.;if(r>r0){const double t=(r-r0)/(r1-r0);s=1.-t*t*(3.-2.*t);ds=-6.*t*(1.-t)/(r1-r0);}
        const double w=weights?weights[i]:1.;energy+=.5*w*s*value;
        const double f=.5*w*(s*derivative+ds*value)/r;
        const double g[3]={f*v[0],f*v[1],f*v[2]};edge(out,i,j,v,g);
    }
    out.energy[i]=energy;
}
__global__ void scale_kernel(Neighborhood n,bool nearest,double r0,double r1,int children,
    const double *bounds,double *weights,double *derivatives,double *gradient) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n.nlocal)return;
    double q=nearest?INFINITY:0.;int selected=-1;
    for(int e=n.offsets[i];e<n.offsets[i+1];++e){
        // Reuse the supplied geometric distance to select the same branch
        // as CPU at nearest-neighbor ties, independent of device FMA rules.
        const double *v=n.vectors+3*e;const double r=n.distances[e];
        if(nearest){if(r<q){q=r;selected=e;}}
        else if(r>0.){double s,d;smooth(r0,r1,r,s,d);q+=1.-s;for(int a=0;a<3;++a)gradient[3*e+a]=-d*v[a]/r;}
    }
    if(nearest && selected>=0 && q>0.)for(int a=0;a<3;++a)gradient[3*selected+a]=n.vectors[3*selected+a]/q;
    double prefix=1.,dprefix=0.;
    for(int c=0;c<children-1;++c){double s,d;smooth(bounds[2*c],bounds[2*c+1],q,s,d);
        weights[c*n.ntotal+i]=prefix*(1.-s);derivatives[c*n.ntotal+i]=dprefix*(1.-s)-prefix*d;
        dprefix=dprefix*s+prefix*d;prefix*=s;}
    weights[(children-1)*n.ntotal+i]=prefix;derivatives[(children-1)*n.ntotal+i]=dprefix;
}
__global__ void scale_response_kernel(Neighborhood n,int children,const double *energies,
    const double *derivatives,const double *gradient,Output out) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n.nlocal)return;
    double factor=0.;for(int c=0;c<children;++c)factor+=energies[c*n.ntotal+i]*derivatives[c*n.ntotal+i];
    for(int e=n.offsets[i];e<n.offsets[i+1];++e){double g[3];for(int a=0;a<3;++a)g[a]=factor*gradient[3*e+a];
        edge(out,i,n.neighbors[e],n.vectors+3*e,g);}
}
}
void add(int device,int n,Output out,Output source,bool energies) {
    check(gpuSetDevice(device));if(!n)return;Result a(n,out),b(n,source);
    addition<<<(n+127)/128,128>>>(n,a.view(),b.view(),energies);check(gpuGetLastError());a.download(out);
}
void zbl(int device,Neighborhood n,const std::vector<double> &z,double r0,double r1,const double *w,Output out) {
    check(gpuSetDevice(device));if(!n.nlocal)return;Geometry geometry(n);Result result(n.ntotal);
    Buffer<double> numbers(z.size(),z.data()),weights(w?n.ntotal:0,w);
    zbl_kernel<<<(n.nlocal+127)/128,128>>>(geometry.view,numbers.p,r0,r1,weights.p,result.view());
    check(gpuGetLastError());result.download(out);
}
Scale scale(int device,Neighborhood n,bool nearest,double r0,double r1,const std::vector<double> &bounds) {
    check(gpuSetDevice(device));const int children=1+bounds.size()/2;Scale result;
    result.weights.resize(children*n.ntotal);result.derivatives.resize(children*n.ntotal);result.feature_gradient.resize(3*n.edges);
    if(!n.nlocal)return result;Geometry geometry(n);Buffer<double> b(bounds.size(),bounds.data());
    Buffer<double> w(result.weights.size()),d(result.derivatives.size()),g(result.feature_gradient.size());
    scale_kernel<<<(n.nlocal+127)/128,128>>>(geometry.view,nearest,r0,r1,children,b.p,w.p,d.p,g.p);
    check(gpuGetLastError());w.download(result.weights.data());d.download(result.derivatives.data());g.download(result.feature_gradient.data());return result;
}
void scale_response(int device,Neighborhood n,int children,const Scale &s,const std::vector<double> &energies,Output out) {
    check(gpuSetDevice(device));if(!n.nlocal)return;Geometry geometry(n);Result result(n.ntotal,out);
    Buffer<double> e(energies.size(),energies.data()),d(s.derivatives.size(),s.derivatives.data()),g(s.feature_gradient.size(),s.feature_gradient.data());
    scale_response_kernel<<<(n.nlocal+127)/128,128>>>(geometry.view,children,e.p,d.p,g.p,result.view());
    check(gpuGetLastError());result.download(out);
}
}
