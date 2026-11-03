// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

// Typed inference kernels. Included inside mtp_gpu_direct.cu after model validation.
// Training and the established FP64 kernels do not instantiate this path.
namespace PrecisionInference {

template<class C> struct NumericPotential : DevicePotential {
    C *radial_basis_parameters=nullptr,*radial_coeffs=nullptr,*species_coeffs=nullptr;
    C *moment_coeffs=nullptr,*descriptor_species_coeffs=nullptr,*descriptor_moment_coeffs=nullptr;
    C *input_mean=nullptr,*input_inverse_std=nullptr,*input_species=nullptr,*model_coeffs=nullptr;
    C min_dist=0,max_dist=0,cutoff=0,cutoffsq=0,scaling=1,p0=1,pp=1;
};
inline NumericPotential<double> numeric(const DevicePotential &p) {
    NumericPotential<double> n;static_cast<DevicePotential&>(n)=p;
#define COPY(f) n.f=p.f
    COPY(radial_basis_parameters);COPY(radial_coeffs);COPY(species_coeffs);COPY(moment_coeffs);
    COPY(descriptor_species_coeffs);COPY(descriptor_moment_coeffs);COPY(model_coeffs);
    COPY(input_mean);COPY(input_inverse_std);COPY(input_species);
    COPY(min_dist);COPY(max_dist);COPY(cutoff);COPY(cutoffsq);COPY(scaling);COPY(p0);COPY(pp);
#undef COPY
    return n;
}
inline void checked(gpuError_t e) {if(e!=gpuSuccess)throw std::runtime_error(gpuGetErrorString(e));}
struct Storage {
    void *data=nullptr;std::size_t bytes=0;
    Storage()=default;Storage(const Storage&)=delete;Storage&operator=(const Storage&)=delete;
    ~Storage(){if(data)gpuFree(data);}
    void clear(){if(data)checked(gpuFree(data));data=nullptr;bytes=0;}
    template<class T> T *get()const{return static_cast<T*>(data);}
    template<class T> void reserve(std::size_t n) {
        if(n>std::numeric_limits<std::size_t>::max()/sizeof(T))throw std::runtime_error("precision workspace size overflow");
        if(n*sizeof(T)<=bytes)return;
        clear();checked(gpuMalloc(&data,n*sizeof(T)));bytes=n*sizeof(T);
    }
    void zero(){if(bytes)checked(gpuMemset(data,0,bytes));}
};
template<class T> __global__ void convert(std::size_t n,const T *from,double *to,int *error) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n){to[i]=double(from[i]);if(!isfinite(to[i]))atomicExch(error,1);}
}
__global__ void shadow(std::size_t n,const double *from,float *to,int *error) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n){to[i]=float(from[i]);if(!isfinite(to[i]))atomicExch(error,1);}
}
template<class T> __device__ T ipow(T x,int n){T v=T(1);for(int i=0;i<n;++i)v*=x;return v;}
template<class T> __device__ void add(T *p,T v){atomicAdd(p,v);}
template<> __device__ void add<double>(double *p,double v){atomic_add_double(p,v);}
template<class A> __device__ A block_sum(A value,A *partial) {
    const int lane=threadIdx.x%32,width=blockDim.x<32?blockDim.x:32;
    for(int offset=16;offset;offset/=2) {
#if defined(IMRMLP_HIP)
        const A other=__shfl_down(value,offset,32);
#else
        const A other=__shfl_down_sync(__activemask(),value,offset,32);
#endif
        if(lane+offset<width)value+=other;
    }
    if(lane==0)partial[threadIdx.x/32]=value;
    __syncthreads();
    A sum=A(0);
    if(threadIdx.x==0)for(int warp=0;warp<(blockDim.x+31)/32;++warp)sum+=partial[warp];
    __syncthreads();
    return sum;
}
// Reduce center outputs hierarchically in A. In FP32, this avoids a single
// float atomic accumulating thousands of center energies/virials; no FP64
// intermediate is introduced into the short-range reduction.
template<class A,int Components>
__global__ void reduce_output(int centers,const A *values,double *output,int *error) {
    __shared__ A partial[32];
    const int c=blockIdx.x;A value=A(0);
    for(int i=threadIdx.x;i<centers;i+=blockDim.x)value+=values[std::size_t(i)*Components+c];
    value=block_sum(value,partial);
    if(threadIdx.x==0){output[c]=double(value);if(!isfinite(value))atomicExch(error,1);}
}
template<class C>
__device__ void radial(const NumericPotential<C> &pot, int pair, C r,
                        C *rb_vals, C *rb_ders, bool need_derivatives, int stride)
{
    const bool split = pot.pair_radial_basis_count != 0;
    const DeviceRadialBasis *basis = split ? pot.pair_radial_bases + pair : nullptr;
    const auto type = static_cast<RadialBasis>(split ? basis->rb_type : pot.rb_type);
    const C lo = split ? basis->min_dist : pot.min_dist;
    const C hi = split ? basis->max_dist : pot.max_dist;
    const C scaling = split ? basis->scaling : pot.scaling;
    if (type == RadialBasis::PolyB || type == RadialBasis::PolyBCos) {
        const C *matrix = pot.radial_basis_parameters + basis->polynomial_offset;
        const C *normalization = matrix + size_t(pot.rb_size) * pot.rb_size;
        const C delta = hi - r;
        if constexpr(std::is_same_v<C,float>) {
            // The uploaded polynomial is expressed in Chebyshev coordinates.
            // Monomial cancellation in PolyB at order 8 already overwhelms
            // FP32. Value and derivative share this stable Clenshaw recurrence.
            const C t=C(2)*delta/(hi-lo)-C(1),dt=-C(2)/(hi-lo);
            const int power=type==RadialBasis::PolyB?3:2;
            const C envelope=ipow(delta,power),denvelope=-C(power)*ipow(delta,power-1);
            C fc=C(1),dfc=C(0);
            if(type==RadialBasis::PolyBCos)RadialMath::envelope<C>(RadialCutoff::CosineLeft,r,lo,hi,0,0,0,fc,dfc);
            for(int i=0;i<pot.rb_size;++i){C b1=C(0),b2=C(0),d1=C(0),d2=C(0);
                for(int k=pot.rb_size-1;k>0;--k){
                    const C b=C(2)*t*b1-b2+matrix[std::size_t(k)*pot.rb_size+i];
                    const C d=C(2)*b1+C(2)*t*d1-d2;b2=b1;b1=b;d2=d1;d1=d;}
                const C q=t*b1-b2+matrix[i],dq=(b1+t*d1-d2)*dt;
                const C value=scaling*envelope*q;
                rb_vals[i*stride]=value*fc;
                if(need_derivatives)rb_ders[i*stride]=scaling*(denvelope*q+envelope*dq)*fc+value*dfc;
            }
            return;
        }
        C fc = C(1.0), dfc = C(0.0);
        const int first_power = type == RadialBasis::PolyB ? 3 : 2;
        if (type == RadialBasis::PolyBCos)
            RadialMath::envelope<C>(RadialCutoff::CosineLeft, r, lo, hi, 0, 0, 0, fc, dfc);
        for (int i=0;i<pot.rb_size;++i) {
            C value = C(0.0), derivative = C(0.0);
            C previous = scaling * delta;
            if (first_power == 3) previous *= delta;
            for (int j=0;j<pot.rb_size;++j) {
                const C power = previous * delta;
                const C coefficient = matrix[size_t(j) * pot.rb_size + i];
                value += coefficient * (normalization[j] * power);
                if (need_derivatives) derivative += coefficient * (-(normalization[j] * (j + first_power)) * previous);
                previous = power;
            }
            rb_vals[i*stride] = value * fc;
            if (need_derivatives) rb_ders[i*stride] = derivative * fc + value * dfc;
        }
        return;
    }
    C coordinate = (C(2.0) * r - lo - hi) / (hi - lo);
    C dx = C(2.0) / (hi - lo), fc = C(0.0), dfc = C(0.0);
    bool flipped = false;
    if (type == RadialBasis::ChebyshevConfigurable) {
        RadialMath::coordinate<C>(static_cast<RadialTransform>(basis->transform_type),
            r, lo, hi, basis->transform_parameter, basis->transform_scale_parameter, coordinate, dx);
        RadialMath::envelope<C>(static_cast<RadialCutoff>(basis->cutoff_type),
            r, lo, hi, basis->cutoff_parameter, basis->cutoff_r1, basis->cutoff_r2, fc, dfc);
        flipped = basis->one_minus_chebyshev;
    } else if (type == RadialBasis::ChebyCos) {
        RadialMath::envelope<C>(RadialCutoff::CosineLeft, r, lo, hi, 0, 0, 0, fc, dfc);
    } else if (type == RadialBasis::ChebyshevPoly4 || (type == RadialBasis::ChebyshevPoly5 && split)) {
        RadialMath::envelope<C>(type == RadialBasis::ChebyshevPoly4 ? RadialCutoff::Poly4 : RadialCutoff::Poly5,
            r, lo, hi, basis->cutoff_parameter, 0, 0, fc, dfc);
    } else if (type == RadialBasis::ChebyshevPoly5) {
        // Preserve the historical bare-view Poly5 convention. Native models
        // always supply their explicit parameterized pair descriptors.
        const C q = (C(1.0)-r/pot.cutoff)/pot.p0;
        fc = q*q*q*(C(10.0)+q*(-C(15.0)+C(6.0)*q));
        dfc = -C(30.0)*q*q*(C(1.0)-q)*(C(1.0)-q)*pot.pp;
    } else {
        RadialMath::envelope<C>(RadialCutoff::Quadratic, r, lo, hi, 0, 0, 0, fc, dfc);
    }
    C previous = C(1.0), current = coordinate;
    C dprevious = C(0.0), dcurrent = dx;
    for (int i=0;i<pot.rb_size;++i) {
        const C polynomial = flipped ? C(1.0)-current : previous;
        const C derivative = flipped ? -dcurrent : dprevious;
        rb_vals[i*stride] = scaling * polynomial * fc;
        if (need_derivatives) rb_ders[i*stride] = scaling * (derivative * fc + polynomial * dfc);
        const C next = C(2.0) * coordinate * current - previous;
        const C dnext = C(2.0) * (dx * current + coordinate * dcurrent) - dprevious;
        previous = current; current = next; dprevious = dcurrent; dcurrent = dnext;
    }
}


// Candidate topology remains an integer LAMMPS contract. Only FP32 evaluates
// geometry after casting coordinates, rather than casting an FP64 difference.
template<class C,bool FloatGeometry>
__device__ bool geometry(const NumericPotential<C> &p,const MTPGpuDeviceInput &in,
    int i,int j,int edge,C *v,C &r,int &pair,int *error) {
    using G=std::conditional_t<FloatGeometry,float,double>;
    const int ti=p.spmap[in.type[i]],tj=p.spmap[in.type[j]];
    if(ti<0||tj<0)return false;pair=ti*p.species_count+tj;
    G r2=G(0);
    for(int d=0;d<3;++d){G x;
        if constexpr(FloatGeometry) x=in.explicit_neighbor_vectors?G(in.neighbor_vectors[3*edge+d]):G(in.positions[3*j+d])-G(in.positions[3*i+d]);
        else x=in.neighbor_vectors?in.neighbor_vectors[3*edge+d]:in.positions[3*j+d]-in.positions[3*i+d];
        v[d]=C(x);r2+=x*x;}
    if(!isfinite(r2)){atomicExch(error,1);return false;}
    const auto *b=p.pair_radial_basis_count?p.pair_radial_bases+pair:nullptr;
    // Mixed-compute tests the original double cutoffs, not their float shadows.
    const G hi=G(b?b->max_dist:static_cast<const DevicePotential&>(p).cutoff);
    const G hi2=FloatGeometry?hi*hi:G(b?b->cutoffsq:static_cast<const DevicePotential&>(p).cutoffsq);
    if(r2<=G(0)){if constexpr(FloatGeometry){
        if(in.positions[3*i]!=in.positions[3*j]||in.positions[3*i+1]!=in.positions[3*j+1]||in.positions[3*i+2]!=in.positions[3*j+2])atomicExch(error,2);}
        return false;}
    if(r2>=hi2)return false;
    if(b&&b->min_dist>0&&(b->rb_type==int(RadialBasis::ChebyshevPoly4)||b->rb_type==int(RadialBasis::ChebyshevPoly5)||b->rb_type==int(RadialBasis::ChebyshevConfigurable))) {
        const G lo=G(b->min_dist);if(r2<(FloatGeometry?lo*lo:G(b->min_distsq)))return false;}
    r=C(sqrt(r2));return true;
}

template<class C,class A,bool FloatGeometry>
__global__ void basic(NumericPotential<C> p,MTPGpuDeviceInput in,int first,float *mom,int *error) {
    extern __shared__ unsigned char scratch[];
    C *rb=reinterpret_cast<C*>(scratch)+threadIdx.x;
    const int ci=first+blockIdx.x,i=in.centers[ci];
    float *m=mom+std::size_t(blockIdx.x)*p.alpha_moments_count;
    for(int bi=threadIdx.x;bi<p.alpha_moments_count;bi+=blockDim.x) {
        A sum=A(0);
        if(bi<p.alpha_index_basic_count)for(int e=in.offsets[ci];e<in.offsets[ci+1];++e){
            C v[3],r;int pair;if(!geometry<C,FloatGeometry>(p,in,i,in.neighbors[e],e,v,r,pair,error))continue;
            radial(p,pair,r,rb,static_cast<C*>(nullptr),false,blockDim.x);
            A radial_sum=A(0);const int off=(pair*p.radial_func_count+p.basic_mu[bi])*p.rb_size;
            for(int k=0;k<p.rb_size;++k)radial_sum+=A(C(p.radial_coeffs[off+k]*rb[k*blockDim.x]));
            const C value=C(radial_sum)*ipow(C(1)/r,p.basic_nu[bi])*ipow(v[0],p.basic_b1[bi])*ipow(v[1],p.basic_b2[bi])*ipow(v[2],p.basic_b3[bi]);
            sum+=A(value);
        }
        m[bi]=float(sum);if(!isfinite(m[bi]))atomicExch(error,1);
    }
}

// As in the FP64 path, x lanes own neighbors and y lanes share the
// radial channels and basic moments. All arithmetic in these caches uses C;
// only reductions use A, including the neighbor partials in mixed-compute.
template<class C, bool Derivatives> struct PairCache {
    C *rb, *der, *value, *slope, *distance, *coordinate, *vector, *inverse, *valid;
    __device__ PairCache(C *data, const NumericPotential<C> &p, int lanes) {
        rb=data; data+=lanes*p.rb_size;
        der=Derivatives?data:nullptr; if constexpr(Derivatives)data+=lanes*p.rb_size;
        value=data; data+=lanes*p.max_mu;
        slope=Derivatives?data:nullptr; if constexpr(Derivatives)data+=lanes*p.max_mu;
        distance=data; data+=lanes*p.max_alpha_index_basic;
        coordinate=data; data+=lanes*p.coord_power_count*3;
        vector=data; data+=lanes*3; inverse=data; data+=lanes; valid=data;
    }
};
template<class C, bool Derivatives> std::size_t pair_cache_bytes(const DevicePotential &p,int lanes) {
    return std::size_t(lanes)*((Derivatives?2:1)*(p.rb_size+p.max_mu)+
        p.max_alpha_index_basic+3*p.coord_power_count+5)*sizeof(C);
}
template<class C,class A,bool FloatGeometry,bool Derivatives>
__device__ void fill_cache(const NumericPotential<C> &p,const MTPGpuDeviceInput &in,
    int ci,int edge,PairCache<C,Derivatives> q,int *error) {
    const int x=threadIdx.x,y=threadIdx.y,i=in.centers[ci];
    if(y==0){
        q.valid[x]=C(0);
        if(edge<in.offsets[ci+1]){
            C v[3],r;int pair;
            if(geometry<C,FloatGeometry>(p,in,i,in.neighbors[edge],edge,v,r,pair,error)){
                q.valid[x]=C(1);q.inverse[x]=C(1)/r;
                radial(p,pair,r,q.rb+x*p.rb_size,Derivatives?q.der+x*p.rb_size:nullptr,Derivatives,1);
                C *dp=q.distance+x*p.max_alpha_index_basic,*cp=q.coordinate+x*p.coord_power_count*3;
                dp[0]=C(1);
                for(int d=0;d<3;++d){q.vector[3*x+d]=v[d];cp[d]=C(0);cp[3+d]=C(1);}
                for(int k=1;k<p.max_alpha_index_basic;++k){
                    dp[k]=dp[k-1]*q.inverse[x];
                    for(int d=0;d<3;++d)cp[3*(k+1)+d]=cp[3*k+d]*v[d];
                }
            }
        }
    }
    __syncthreads();
    if(q.valid[x]!=C(0)){
        const int pair=p.spmap[in.type[i]]*p.species_count+p.spmap[in.type[in.neighbors[edge]]];
        for(int mu=y;mu<p.max_mu;mu+=blockDim.y){
            A v=A(0),d=A(0);const int off=(pair*p.radial_func_count+mu)*p.rb_size;
            for(int k=0;k<p.rb_size;++k){
                v+=A(C(p.radial_coeffs[off+k]*q.rb[x*p.rb_size+k]));
                if constexpr(Derivatives)d+=A(C(p.radial_coeffs[off+k]*q.der[x*p.rb_size+k]));
            }
            q.value[x*p.max_mu+mu]=C(v);
            if constexpr(Derivatives)q.slope[x*p.max_mu+mu]=C(d);
        }
    }
    __syncthreads();
}
template<class C,class A,bool FloatGeometry>
__global__ void basic_cached(NumericPotential<C> p,MTPGpuDeviceInput in,int first,
    float *mom,int tile,int *error) {
    extern __shared__ unsigned char storage[];
    const int x=threadIdx.x,y=threadIdx.y,nx=blockDim.x,tid=y*nx+x;
    const int begin=blockIdx.y*tile,remaining=p.alpha_index_basic_count-begin;
    const int count=remaining<tile?remaining:tile,ci=first+blockIdx.x,stride=tile+1;
    A *partial=reinterpret_cast<A*>(storage);
    PairCache<C,false> q(reinterpret_cast<C*>(partial+nx*stride),p,nx);
    for(int k=tid;k<nx*stride;k+=nx*blockDim.y)partial[k]=A(0);
    __syncthreads();
    for(int e=in.offsets[ci];e<in.offsets[ci+1];e+=nx){
        fill_cache<C,A,FloatGeometry,false>(p,in,ci,e+x,q,error);
        if(q.valid[x]!=C(0)){
            const C *cp=q.coordinate+x*p.coord_power_count*3,*dp=q.distance+x*p.max_alpha_index_basic;
            for(int b=y;b<count;b+=blockDim.y){const int bi=begin+b;
                const C value=q.value[x*p.max_mu+p.basic_mu[bi]]*dp[p.basic_nu[bi]]*
                    cp[3*(p.basic_b1[bi]+1)]*cp[3*(p.basic_b2[bi]+1)+1]*cp[3*(p.basic_b3[bi]+1)+2];
                partial[x*stride+b]+=A(value);
            }
        }
        __syncthreads();
    }
    for(int b=tid;b<count;b+=nx*blockDim.y){A sum=A(0);
        for(int lane=0;lane<nx;++lane)sum+=partial[lane*stride+b];
        const float value=float(sum);mom[std::size_t(blockIdx.x)*p.alpha_moments_count+begin+b]=value;
        if(!isfinite(value))atomicExch(error,1);
    }
}
template<class C,class A,class D,bool FloatGeometry,bool Virial>
__global__ void forces_cached(NumericPotential<C> p,MTPGpuDeviceInput in,int first,
    const D *adj,A *force,A *site,A *global,int tile,int *error) {
    extern __shared__ unsigned char storage[];
    const int x=threadIdx.x,y=threadIdx.y,nx=blockDim.x,ny=blockDim.y;
    const int ci=first+blockIdx.x,i=in.centers[ci],begin=blockIdx.y*tile;
    const int remaining=p.alpha_index_basic_count-begin,count=remaining<tile?remaining:tile;
    A *partial=reinterpret_cast<A*>(storage);
    PairCache<C,true> q(reinterpret_cast<C*>(partial+nx*ny*3),p,nx);
    const D *a=adj+std::size_t(blockIdx.x)*p.alpha_moments_count;
    A center[3]={A(0),A(0),A(0)},virial[6]={A(0),A(0),A(0),A(0),A(0),A(0)};
    for(int e=in.offsets[ci];e<in.offsets[ci+1];e+=nx){
        fill_cache<C,A,FloatGeometry,true>(p,in,ci,e+x,q,error);
        A f[3]={A(0),A(0),A(0)};
        if(q.valid[x]!=C(0)){
            const C *cp=q.coordinate+x*p.coord_power_count*3,*dp=q.distance+x*p.max_alpha_index_basic;
            for(int b=y;b<count;b+=ny){const int bi=begin+b,mu=p.basic_mu[bi];
                const int b0=p.basic_b1[bi],b1=p.basic_b2[bi],b2=p.basic_b3[bi],nu=p.basic_nu[bi];
                const C val=q.value[x*p.max_mu+mu]*dp[nu];
                const C der=q.slope[x*p.max_mu+mu]*dp[nu]-C(nu)*val*q.inverse[x];
                const C pow0=cp[3*(b0+1)],pow1=cp[3*(b1+1)+1],pow2=cp[3*(b2+1)+2];
                const C common=pow0*pow1*pow2*der*q.inverse[x];
                const C gx=common*q.vector[3*x]+C(b0)*val*cp[3*b0]*pow1*pow2;
                const C gy=common*q.vector[3*x+1]+C(b1)*val*cp[3*b1+1]*pow0*pow2;
                const C gz=common*q.vector[3*x+2]+C(b2)*val*cp[3*b2+2]*pow0*pow1;
                f[0]+=A(C(C(a[bi])*gx));f[1]+=A(C(C(a[bi])*gy));f[2]+=A(C(C(a[bi])*gz));
            }
        }
        for(int d=0;d<3;++d)partial[3*(y*nx+x)+d]=f[d];
        __syncthreads();
        if(y==0&&q.valid[x]!=C(0)){
            for(int d=0;d<3;++d){f[d]=A(0);for(int lane=0;lane<ny;++lane)f[d]+=partial[3*(lane*nx+x)+d];
                if(!isfinite(f[d]))atomicExch(error,1);
                center[d]+=f[d];if(force)add(force+3*in.neighbors[e+x]+d,-f[d]);}
            if constexpr(Virial){
                const A rx=A(q.vector[3*x]),ry=A(q.vector[3*x+1]),rz=A(q.vector[3*x+2]);
                const A w[6]={-f[0]*rx,-f[1]*ry,-f[2]*rz,-A(.5)*(f[0]*ry+f[1]*rx),
                    -A(.5)*(f[0]*rz+f[2]*rx),-A(.5)*(f[1]*rz+f[2]*ry)};
                for(int d=0;d<6;++d){virial[d]+=w[d];if(site)add(site+6*in.neighbors[e+x]+d,A(.5)*w[d]);}
            }
        }
        __syncthreads();
    }
    for(int d=0;d<3;++d){
        if(y==0)partial[x]=center[d];__syncthreads();
        if(y==0&&x==0&&force){A sum=A(0);for(int lane=0;lane<nx;++lane)sum+=partial[lane];add(force+3*i+d,sum);}
        __syncthreads();
    }
    if constexpr(Virial)for(int d=0;d<6;++d){
        if(y==0)partial[x]=virial[d];__syncthreads();
        if(y==0&&x==0){A sum=A(0);for(int lane=0;lane<nx;++lane)sum+=partial[lane];
            if(site)add(site+6*i+d,A(.5)*sum);if(global)add(global+6*ci+d,sum);}
        __syncthreads();
    }
}
template<class A> __device__ A warp_sum_t(A value) {
    for(int offset=16;offset;offset/=2){
#if defined(IMRMLP_HIP)
        value+=__shfl_down(value,offset,32);
#else
        value+=__shfl_down_sync(__activemask(),value,offset,32);
#endif
    }
    return value;
}

template<class C> __device__ void activation(int kind,C z,C &v,C &d) {
    if(kind==int(MTPGpuActivation::Tanh)){
        v=tanh(z);
        if constexpr(std::is_same_v<C,float>){
            // Avoid cancellation of 1-tanh(z)^2 in saturated FP32 heads.
            const C e=exp(-C(2)*abs(z)),den=C(1)+e;d=C(4)*e/(den*den);
        } else d=C(1)-v*v;
    }
    else if(kind==int(MTPGpuActivation::ReLU)){v=z>C(0)?z:C(0);d=z>C(0)?C(1):C(0);}
    else if(kind==int(MTPGpuActivation::Sigmoid)){
        const C e=exp(z>=C(0)?-z:z),den=C(1)+e;v=z>=C(0)?C(1)/den:e/den;
        if constexpr(std::is_same_v<C,float>)d=e/(den*den);else d=v*(C(1)-v);
    }
    else {v=z;d=C(1);}
}
__host__ __device__ inline int cache_stride(const DevicePotential &p){return 2*p.descriptor_count+3*p.model_hidden1+2*p.model_hidden2+2;}

template<class C,class A> struct ActivationSum {
    A main=A(0),correction=A(0);
    __device__ void add(int kind,C upstream,C weight,C value,C derivative) {
        if constexpr(std::is_same_v<C,float>) {
            if(kind==int(MTPGpuActivation::Tanh)&&abs(value)<C(.5)) {
                // Keep the small h^2 correction when 1-h^2 rounds close to 1.
                // Products are still C; each sum uses the selected A policy.
                // This also preserves cancellation between output channels.
                const C product=upstream*weight;
                main+=A(product);correction+=A(C(product*C(value*value)));return;
            }
        }
        main+=A(C(upstream*derivative*weight));
    }
    __device__ C value()const{return C(main-correction);}
};

template<class C,class A,class D,bool LinearOnly>
__device__ A head(const NumericPotential<C> &p,int atom,int type,const float *m,D *adj,C *cache) {
    const int tid=threadIdx.x;
    const C weight=p.center_energy_weights?C(p.center_energy_weights[atom]):C(1);
    A energy=A(p.species_coeffs[type]);
    if(LinearOnly||p.model_type==int(MTPGpuModel::Linear)) {
        __shared__ A sums[32];
        A local=A(0);
        for(int a=tid;a<p.alpha_scalar_moments;a+=blockDim.x){const int k=p.alpha_moment_mapping[a];
            local+=A(C(p.moment_coeffs[a]*C(m[k])));
            if(adj&&p.unique_scalar_mapping)adj[k]=D(C(weight*p.moment_coeffs[a]));}
        // Non-unique mappings retain ordered seeding; the energy reduction is parallel.
        if(adj&&!p.unique_scalar_mapping&&tid==0)
            for(int a=0;a<p.alpha_scalar_moments;++a)adj[p.alpha_moment_mapping[a]]+=D(C(weight*p.moment_coeffs[a]));
        energy+=block_sum(local,sums);
    } else {
        C *q=cache,*g=q+p.descriptor_count,*h1=g+p.descriptor_count,*d1=h1+p.model_hidden1;
        C *up=d1+p.model_hidden1,*h2=up+p.model_hidden1,*d2=h2+p.model_hidden2;
        for(int d=tid;d<p.descriptor_count;d+=blockDim.x){A value=A(0);
            if(p.scalar_gather){const C raw=d?C(m[p.alpha_moment_mapping[d-1]]):p.input_species[type];q[d]=(raw-p.input_mean[d])*p.input_inverse_std[d];}
            else {value=A(p.descriptor_species_coeffs[d*p.species_count+type]);
                for(int a=0;a<p.alpha_scalar_moments;++a)value+=A(C(p.descriptor_moment_coeffs[d*p.alpha_scalar_moments+a]*C(m[p.alpha_moment_mapping[a]])));q[d]=C(value);}
            g[d]=C(0);
        }
        __syncthreads();
        const bool nn1=p.model_type==int(MTPGpuModel::MTPMNN1),nn2=p.model_type==int(MTPGpuModel::MTPMNN2);
        if(nn1||nn2){const int end=p.descriptor_count*p.model_hidden1,b1=end+(nn1?p.model_hidden1:0);
            for(int h=tid;h<p.model_hidden1;h+=blockDim.x){A z=A(p.model_coeffs[b1+h]);
                for(int d=0;d<p.descriptor_count;++d)z+=A(C(p.model_coeffs[h*p.descriptor_count+d]*q[d]));activation(p.model_activation,C(z),h1[h],d1[h]);}
            __syncthreads();
            if(nn1){
                for(int h=tid;h<p.model_hidden1;h+=blockDim.x)up[h]=p.model_coeffs[end+h];
                if(tid==0)for(int h=0;h<p.model_hidden1;++h)energy+=A(C(p.model_coeffs[end+h]*h1[h]));
            }
            else {const int w2=end+p.model_hidden1,b2=w2+p.model_hidden1*p.model_hidden2,out=b2+p.model_hidden2;
                for(int h=tid;h<p.model_hidden2;h+=blockDim.x){A z=A(p.model_coeffs[b2+h]);for(int j=0;j<p.model_hidden1;++j)z+=A(C(p.model_coeffs[w2+h*p.model_hidden1+j]*h1[j]));
                    activation(p.model_activation,C(z),h2[h],d2[h]);}
                __syncthreads();
                if(tid==0)for(int h=0;h<p.model_hidden2;++h)energy+=A(C(p.model_coeffs[out+h]*h2[h]));
                for(int j=tid;j<p.model_hidden1;j+=blockDim.x){ActivationSum<C,A> u;for(int h=0;h<p.model_hidden2;++h)
                    u.add(p.model_activation,p.model_coeffs[out+h],p.model_coeffs[w2+h*p.model_hidden1+j],h2[h],d2[h]);up[j]=u.value();}
            }
            __syncthreads();
            for(int d=tid;d<p.descriptor_count;d+=blockDim.x){ActivationSum<C,A> v;for(int h=0;h<p.model_hidden1;++h)
                v.add(p.model_activation,up[h],p.model_coeffs[h*p.descriptor_count+d],h1[h],d1[h]);g[d]=v.value();}
        } else if(tid==0) {
        if(p.model_type==int(MTPGpuModel::MTPMLinear)){energy+=A(q[0]);g[0]=C(1);}
        else if(p.model_type==int(MTPGpuModel::MTPMQuadratic)){const C *c=p.model_coeffs;
            energy+=A(C(c[0]*q[0]+c[1]*q[1]+c[2]*q[0]*q[0]+c[3]*q[1]*q[1]+c[4]*q[0]*q[1]));
            g[0]=c[0]+C(2)*c[2]*q[0]+c[4]*q[1];g[1]=c[1]+C(2)*c[3]*q[1]+c[4]*q[0];
        } else {energy+=A(C(q[0]+p.model_coeffs[0]*asinh(q[1])));g[0]=C(1);g[1]=p.model_coeffs[0]/sqrt(C(1)+q[1]*q[1]);}
        }
        __syncthreads();
        if(adj&&(p.unique_scalar_mapping||tid==0))for(int a=p.unique_scalar_mapping?tid:0;a<p.alpha_scalar_moments;a+=p.unique_scalar_mapping?blockDim.x:1){A seed=A(0);
            if(p.scalar_gather)seed=A(C(g[a+1]*p.input_inverse_std[a+1]));
            else for(int d=0;d<p.descriptor_count;++d)seed+=A(C(g[d]*p.descriptor_moment_coeffs[d*p.alpha_scalar_moments+a]));
            adj[p.alpha_moment_mapping[a]]+=D(C(weight*C(seed)));}
    }
    __syncthreads();
    if(adj&&p.response&&(p.unique_scalar_mapping||tid==0))for(int a=p.unique_scalar_mapping?tid:0;a<p.alpha_scalar_moments;a+=p.unique_scalar_mapping?blockDim.x:1){A seed=A(0);
        for(int c=0;c<p.response_channels;++c)seed+=A(C(C(p.response[atom*p.response_channels+c])*C(p.response_coefficients[c*p.alpha_scalar_moments+a])));
        adj[p.alpha_moment_mapping[a]]+=D(seed);}
    return energy;
}

template<class C,class A,class D,bool LinearOnly=false>
__global__ void propagate(NumericPotential<C> p,MTPGpuDeviceInput in,int first,float *mom,D *adj,C *cache,
    A *site,A *energy,double *unweighted,int channels,const double *species,const double *coeff,double *projection,bool reuse,int *error) {
    const int ci=first+blockIdx.x,i=in.centers[ci],t=p.spmap[in.type[i]];
    float *m=mom+std::size_t(blockIdx.x)*p.alpha_moments_count;
    D *d=adj?adj+std::size_t(blockIdx.x)*p.alpha_moments_count:nullptr;
    if(d)for(int k=threadIdx.x;k<p.alpha_moments_count;k+=blockDim.x)d[k]=D(0);
    __shared__ A partials[32];
    const int tid=threadIdx.x;
    if(!reuse) for (int level = 1; level <= p.max_level; ++level) {
        const int begin = p.forward_level_offsets[level];
        const int end = p.forward_level_offsets[level + 1];
        const bool classified = blockDim.x >= propagation_warp_size;
        const int short_begin = classified ? p.forward_short_level_offsets[level] : begin;
        const int short_end = classified ? p.forward_short_level_offsets[level + 1] : end;
        for (int index = short_begin + tid; index < short_end;
             index += blockDim.x) {
            const int group = classified ? p.forward_short_groups[index] : index;
            const int term_begin = p.forward_term_offsets[group];
            const int term_end = p.forward_term_offsets[group + 1];
            A sum = A(0);
            for (int pos = term_begin; pos < term_end; ++pos) {
                const ForwardTerm term = p.forward_terms[pos];
                sum += A(C(C(term.c)*C(m[term.l1])*C(m[term.l2])));
            }
            m[p.forward_ids[group]] += sum;
        }
        if (blockDim.x >= propagation_warp_size) {
            const int lane = tid & (propagation_warp_size - 1);
            const int warp = tid / propagation_warp_size;
            const int warps = blockDim.x / propagation_warp_size;
            const int warp_begin = p.forward_warp_level_offsets[level];
            const int warp_end = p.forward_warp_level_offsets[level + 1];
            for (int index = warp_begin + warp; index < warp_end; index += warps) {
                const int group = p.forward_warp_groups[index];
                const int term_begin = p.forward_term_offsets[group];
                const int term_end = p.forward_term_offsets[group + 1];
                A sum = A(0);
                for (int pos = term_begin + lane; pos < term_end;
                     pos += propagation_warp_size) {
                    const ForwardTerm term = p.forward_terms[pos];
                    sum += A(C(C(term.c)*C(m[term.l1])*C(m[term.l2])));
                }
                sum = warp_sum_t(sum);
                if (lane == 0) m[p.forward_ids[group]] += sum;
            }
            const int block_begin =
                p.forward_block_level_offsets[level];
            const int block_end =
                p.forward_block_level_offsets[level + 1];
            const int group_warps = warps < propagation_multiwarp_group_warps
                ? warps : propagation_multiwarp_group_warps;
            const int group_threads = group_warps * propagation_warp_size;
            const int groups_per_batch = warps / group_warps;
            const int group_slot = tid / group_threads;
            const int group_lane = tid % group_threads;
            for (int batch = block_begin; batch < block_end;
                 batch += groups_per_batch) {
                const int block_group = batch + group_slot;
                const bool active = block_group < block_end;
                int group = 0;
                int term_begin = 0;
                int term_end = 0;
                if (active) {
                    group = p.forward_block_groups[block_group];
                    term_begin = p.forward_term_offsets[group];
                    term_end = p.forward_term_offsets[group + 1];
                }
                A sum = A(0);
                for (int pos = term_begin + group_lane; pos < term_end;
                     pos += group_threads) {
                    const ForwardTerm term = p.forward_terms[pos];
                    sum += A(C(C(term.c)*C(m[term.l1])*C(m[term.l2])));
                }
                sum = warp_sum_t(sum);
                if (lane == 0) partials[warp] = sum;
                __syncthreads();
                if (active && group_lane == 0) {
                    A group_sum = A(0);
                    const int first_warp = group_slot * group_warps;
                    for (int index = 0; index < group_warps; ++index) {
                        group_sum += partials[first_warp + index];
                    }
                    m[p.forward_ids[group]] += group_sum;
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }
    __syncthreads();
    const A raw=head<C,A,D,LinearOnly>(p,i,t,m,d,cache+std::size_t(blockIdx.x)*cache_stride(p));
    if(threadIdx.x==0){
        const C weight=p.center_energy_weights?C(p.center_energy_weights[i]):C(1);const A e=A(C(weight)*raw);
        if(site)site[i]=e;if(energy)energy[ci]=e;if(unweighted)unweighted[i]=double(raw);
        if(!isfinite(e))atomicExch(error,1);
        if(projection)for(int c=0;c<channels;++c){A v=A(C(species[c*p.species_count+t]));
            for(int a=0;a<p.alpha_scalar_moments;++a)v+=A(C(C(coeff[c*p.alpha_scalar_moments+a])*C(m[p.alpha_moment_mapping[a]])));
            projection[i*channels+c]=double(v);if(!isfinite(v))atomicExch(error,1);}
    }
    __syncthreads();
    if(d)for (int level = p.max_level; level >= 1; --level) {
        const int begin = p.reverse_level_offsets[level];
        const int end = p.reverse_level_offsets[level + 1];
        const bool classified = blockDim.x >= propagation_warp_size;
        const int short_begin = classified ? p.reverse_short_level_offsets[level] : begin;
        const int short_end = classified ? p.reverse_short_level_offsets[level + 1] : end;
        for (int index = short_begin + tid; index < short_end;
             index += blockDim.x) {
            const int group = classified ? p.reverse_short_groups[index] : index;
            const int entry_begin = p.reverse_entry_offsets[group];
            const int entry_end = p.reverse_entry_offsets[group + 1];
            A sum = A(0);
            for (int pos = entry_begin; pos < entry_end; ++pos) {
                const ReverseTerm term = p.reverse_entries[pos];
                sum += A(C(C(term.c)*C(d[term.source])*C(m[term.other])));
            }
            d[p.reverse_ids[group]] += sum;
        }
        if (blockDim.x >= propagation_warp_size) {
            const int lane = tid & (propagation_warp_size - 1);
            const int warp = tid / propagation_warp_size;
            const int warps = blockDim.x / propagation_warp_size;
            const int warp_begin = p.reverse_warp_level_offsets[level];
            const int warp_end = p.reverse_warp_level_offsets[level + 1];
            for (int index = warp_begin + warp; index < warp_end; index += warps) {
                const int group = p.reverse_warp_groups[index];
                const int entry_begin = p.reverse_entry_offsets[group];
                const int entry_end = p.reverse_entry_offsets[group + 1];
                A sum = A(0);
                for (int pos = entry_begin + lane; pos < entry_end;
                     pos += propagation_warp_size) {
                    const ReverseTerm term = p.reverse_entries[pos];
                    sum += A(C(C(term.c)*C(d[term.source])*C(m[term.other])));
                }
                sum = warp_sum_t(sum);
                if (lane == 0) d[p.reverse_ids[group]] += sum;
            }
            const int block_begin =
                p.reverse_block_level_offsets[level];
            const int block_end =
                p.reverse_block_level_offsets[level + 1];
            const int group_warps = warps < propagation_multiwarp_group_warps
                ? warps : propagation_multiwarp_group_warps;
            const int group_threads = group_warps * propagation_warp_size;
            const int groups_per_batch = warps / group_warps;
            const int group_slot = tid / group_threads;
            const int group_lane = tid % group_threads;
            for (int batch = block_begin; batch < block_end;
                 batch += groups_per_batch) {
                const int block_group = batch + group_slot;
                const bool active = block_group < block_end;
                int group = 0;
                int entry_begin = 0;
                int entry_end = 0;
                if (active) {
                    group = p.reverse_block_groups[block_group];
                    entry_begin = p.reverse_entry_offsets[group];
                    entry_end = p.reverse_entry_offsets[group + 1];
                }
                A sum = A(0);
                for (int pos = entry_begin + group_lane; pos < entry_end;
                     pos += group_threads) {
                    const ReverseTerm term = p.reverse_entries[pos];
                    sum += A(C(C(term.c)*C(d[term.source])*C(m[term.other])));
                }
                sum = warp_sum_t(sum);
                if (lane == 0) partials[warp] = sum;
                __syncthreads();
                if (active && group_lane == 0) {
                    A group_sum = A(0);
                    const int first_warp = group_slot * group_warps;
                    for (int index = 0; index < group_warps; ++index) {
                        group_sum += partials[first_warp + index];
                    }
                    d[p.reverse_ids[group]] += group_sum;
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }
}

template<class C,class A,class D,bool FloatGeometry>
__global__ void forces(NumericPotential<C> p,MTPGpuDeviceInput in,int first,const D *adj,A *force,A *site,A *global,int *error) {
    extern __shared__ unsigned char scratch[];
    C *rb=reinterpret_cast<C*>(scratch)+threadIdx.x,*der=rb+blockDim.x*p.rb_size;
    const int ci=first+blockIdx.x,i=in.centers[ci];const D *a=adj+std::size_t(blockIdx.x)*p.alpha_moments_count;
    A center_force[3]={A(0),A(0),A(0)},center_virial[6]={A(0),A(0),A(0),A(0),A(0),A(0)};
    for(int e=in.offsets[ci]+threadIdx.x;e<in.offsets[ci+1];e+=blockDim.x){const int j=in.neighbors[e];C v[3],r;int pair;
        if(!geometry<C,FloatGeometry>(p,in,i,j,e,v,r,pair,error))continue;
        radial(p,pair,r,rb,der,true,blockDim.x);A gradient[3]={A(0),A(0),A(0)};const C inv=C(1)/r;
        for(int mu=0;mu<p.max_mu;++mu){A vs=A(0),ds=A(0);const int off=(pair*p.radial_func_count+mu)*p.rb_size;
            for(int k=0;k<p.rb_size;++k){vs+=A(C(p.radial_coeffs[off+k]*rb[k*blockDim.x]));ds+=A(C(p.radial_coeffs[off+k]*der[k*blockDim.x]));}
            for(int k=p.basic_mu_offsets[mu];k<p.basic_mu_offsets[mu+1];++k){const int bi=p.basic_mu_indices[k],nu=p.basic_nu[bi];
                const int b[3]={p.basic_b1[bi],p.basic_b2[bi],p.basic_b3[bi]};const C pow[3]={ipow(v[0],b[0]),ipow(v[1],b[1]),ipow(v[2],b[2])};
                const C rp=ipow(inv,nu),val=C(vs)*rp,dval=C(ds)*rp-C(nu)*val*inv;
                const C common=pow[0]*pow[1]*pow[2]*dval*inv;
                for(int d=0;d<3;++d){C g=common*v[d];if(b[d])g+=C(b[d])*val*ipow(v[d],b[d]-1)*pow[(d+1)%3]*pow[(d+2)%3];gradient[d]+=A(C(C(a[bi])*g));}
            }
        }
        for(int d=0;d<3;++d){if(!isfinite(gradient[d]))atomicExch(error,1);if(force){center_force[d]+=gradient[d];add(force+3*j+d,-gradient[d]);}}
        if(site||global){const A w[6]={-gradient[0]*A(v[0]),-gradient[1]*A(v[1]),-gradient[2]*A(v[2]),
            -A(.5)*(gradient[0]*A(v[1])+gradient[1]*A(v[0])),-A(.5)*(gradient[0]*A(v[2])+gradient[2]*A(v[0])),-A(.5)*(gradient[1]*A(v[2])+gradient[2]*A(v[1]))};
            for(int d=0;d<6;++d){center_virial[d]+=w[d];if(site)add(site+6*j+d,A(.5)*w[d]);}}
    }
    // Avoid all neighbor lanes contending on six global virial scalars. Reuse
    // radial scratch after the last neighbor; even rb_size=1 has enough space
    // for one accumulator per logical warp (also for deterministic block=1).
    __syncthreads();
    A *partial=reinterpret_cast<A*>(scratch);
    if(force)for(int d=0;d<3;++d){const A sum=block_sum(center_force[d],partial);if(threadIdx.x==0)add(force+3*i+d,sum);}
    if(site||global)for(int d=0;d<6;++d){const A sum=block_sum(center_virial[d],partial);if(threadIdx.x==0){if(site)add(site+6*i+d,A(.5)*sum);if(global)add(global+d,sum);}}
}

struct Workspace {
    NumericPotential<float> fp;
    std::array<Storage,11> parameters;
    Storage moments,adjoints,cache,force,energy,site,virial,global,status;
    int last_chunk=0,multiprocessors=0;bool complete=false;
    std::size_t shadow_bytes()const{std::size_t n=0;for(const auto &p:parameters)n+=p.bytes;return n;}
    std::size_t center_bytes()const{return moments.bytes+adjoints.bytes+cache.bytes;}
    std::size_t conversion_bytes()const{return force.bytes+energy.bytes+site.bytes+virial.bytes+global.bytes+status.bytes;}
    void clear_centers(){moments.clear();adjoints.clear();cache.clear();complete=false;}
    void clear_work(){clear_centers();force.clear();energy.clear();site.clear();virial.clear();global.clear();}
    void verify(){int failure=0;checked(gpuMemcpy(&failure,status.data,sizeof(int),gpuMemcpyDeviceToHost));
        if(failure)throw std::runtime_error(failure==2?"FP32 geometry collapsed distinct coordinates; use mixed-compute or fp64":"nonfinite reduced-precision parameter, geometry or inference result");}
    void initialize(const DevicePotential &p,std::size_t polynomial_count) {
        int device=0;gpuDeviceProp props{};checked(gpuGetDevice(&device));checked(gpuGetDeviceProperties(&props,device));
        multiprocessors=props.multiProcessorCount;
        static_cast<DevicePotential&>(fp)=p;
#define COPY(f) fp.f=float(p.f);if(!std::isfinite(fp.f))throw std::runtime_error("nonfinite FP32 scalar parameter: " #f)
        COPY(min_dist);COPY(max_dist);COPY(cutoff);COPY(cutoffsq);COPY(scaling);COPY(p0);COPY(pp);
#undef COPY
        status.reserve<int>(1);status.zero();int index=0;
        auto upload=[&](const double *src,std::size_t n,float *&dst){auto &b=parameters[index++];if(!n){dst=nullptr;return;}b.reserve<float>(n);dst=b.get<float>();shadow<<<(n+127)/128,128>>>(n,src,dst,status.get<int>());checked(gpuGetLastError());};
        upload(p.radial_basis_parameters,polynomial_count,fp.radial_basis_parameters);
        if(polynomial_count){
            std::vector<double> original(polynomial_count);
            std::vector<float> stable(polynomial_count);
            std::vector<DeviceRadialBasis> bases(p.pair_radial_basis_count);
            checked(gpuMemcpy(original.data(),p.radial_basis_parameters,polynomial_count*sizeof(double),gpuMemcpyDeviceToHost));
            checked(gpuMemcpy(bases.data(),p.pair_radial_bases,bases.size()*sizeof(DeviceRadialBasis),gpuMemcpyDeviceToHost));
            std::transform(original.begin(),original.end(),stable.begin(),[](double v){return float(v);});
            const int n=p.rb_size;const long double pi=std::acos(-1.L);
            for(const auto &basis:bases){
                if(basis.rb_type!=int(RadialBasis::PolyB)&&basis.rb_type!=int(RadialBasis::PolyBCos))continue;
                const auto off=basis.polynomial_offset;const double *matrix=original.data()+off,*norm=matrix+std::size_t(n)*n;
                for(int i=0;i<n;++i){std::vector<long double> samples(n);
                    for(int j=0;j<n;++j){const long double t=std::cos(pi*(j+.5L)/n);
                        const long double delta=(basis.max_dist-basis.min_dist)*(t+1.L)/2.L;
                        long double value=0;for(int k=n-1;k>=0;--k)value=value*delta+static_cast<long double>(matrix[std::size_t(k)*n+i])*norm[k];
                        samples[j]=value;}
                    for(int k=0;k<n;++k){long double sum=0;
                        for(int j=0;j<n;++j)sum+=samples[j]*std::cos(pi*k*(j+.5L)/n);
                        const float value=float(sum*(k?2.L:1.L)/n);
                        if(!std::isfinite(value))throw std::runtime_error("nonfinite FP32 polynomial representation");
                        stable[off+std::size_t(k)*n+i]=value;}
                }
            }
            checked(gpuMemcpy(fp.radial_basis_parameters,stable.data(),polynomial_count*sizeof(float),gpuMemcpyHostToDevice));
        }
        upload(p.radial_coeffs,std::size_t(p.species_count)*p.species_count*p.radial_func_count*p.rb_size,fp.radial_coeffs);
        upload(p.species_coeffs,p.species_count,fp.species_coeffs);
        upload(p.moment_coeffs,p.model_type==int(MTPGpuModel::Linear)?p.alpha_scalar_moments:0,fp.moment_coeffs);
        upload(p.descriptor_species_coeffs,p.scalar_gather?0:p.descriptor_count*p.species_count,fp.descriptor_species_coeffs);
        upload(p.descriptor_moment_coeffs,p.scalar_gather?0:p.descriptor_count*p.alpha_scalar_moments,fp.descriptor_moment_coeffs);
        upload(p.model_coeffs,p.model_coeffs_count,fp.model_coeffs);
        upload(p.input_mean,p.scalar_gather?p.descriptor_count:0,fp.input_mean);
        upload(p.input_inverse_std,p.scalar_gather?p.descriptor_count:0,fp.input_inverse_std);
        upload(p.input_species,p.scalar_gather?p.species_count:0,fp.input_species);
        verify();
    }
    template<class C,class A,class D,bool FloatGeometry>
    int execute(NumericPotential<C> p,const MTPGpuComputeRequest &r,MTPGpuDeviceInput in,
        MTPGpuDeviceOutput out,std::size_t available,std::size_t shared_limit,MTPGpuRuntimeStats &stats) {
        const bool own_moments=r.device_moments.values_fp32 && r.device_moments.values_fp32==moments.get<float>();
        bool reuse=r.device_moments.values_fp32 && (!own_moments || complete);
        complete=false;
        const bool deriv=r.compute_forces||r.site_virial||r.global_virial;
        const std::size_t nf=r.compute_forces?3ull*r.ntotal:0,ns=r.compute_site_energy?r.nlocal:0,nw=r.site_virial?6ull*r.ntotal:0;
        const std::size_t fixed=(nf+ns+nw+(r.compute_global_energy?r.inum:0)+(r.global_virial?6ull*r.inum:0))*sizeof(A)+sizeof(int);
        const std::size_t each=p.alpha_moments_count*(sizeof(float)+(deriv?sizeof(D):0))+std::size_t(cache_stride(p))*sizeof(C);
        int chunk=r.centers_per_chunk?std::min(r.centers_per_chunk,r.inum):r.inum;
        if(available){if(fixed>available||(r.inum&&each>available-fixed))throw std::runtime_error("precision memory budget cannot fit fixed outputs and one center");
            if(each)chunk=int(std::min<std::size_t>(chunk,(available-fixed)/each));
            const auto growth=[](std::size_t n,std::size_t old){return n>old?n-old:0;};
            const std::size_t extra=growth(nf*sizeof(A),force.bytes)+growth(ns*sizeof(A),site.bytes)+growth(nw*sizeof(A),virial.bytes)+growth((r.compute_global_energy?r.inum:0)*sizeof(A),energy.bytes)+growth((r.global_virial?6ull*r.inum:0)*sizeof(A),global.bytes)+growth(sizeof(int),status.bytes)+growth(std::size_t(chunk)*p.alpha_moments_count*sizeof(float),moments.bytes)+growth(deriv?std::size_t(chunk)*p.alpha_moments_count*sizeof(D):0,adjoints.bytes)+growth(std::size_t(chunk)*cache_stride(p)*sizeof(C),cache.bytes);
            if(center_bytes()+conversion_bytes()+extra>available){if(own_moments)reuse=false;clear_work();}
        }
        // A borrowed view into this evaluator cannot survive a resize or an
        // in-place chunked forward/reverse pass. Recompute from the input.
        if(own_moments && (chunk!=r.inum || std::size_t(chunk)*p.alpha_moments_count*sizeof(float)>moments.bytes))reuse=false;
        const std::size_t before=center_bytes()+conversion_bytes();
        moments.reserve<float>(std::size_t(chunk)*p.alpha_moments_count);
        if(deriv)adjoints.reserve<D>(std::size_t(chunk)*p.alpha_moments_count);
        cache.reserve<C>(std::size_t(chunk)*cache_stride(p));
        force.reserve<A>(nf);site.reserve<A>(ns);virial.reserve<A>(nw);energy.reserve<A>(r.compute_global_energy?r.inum:0);global.reserve<A>(r.global_virial?6ull*r.inum:0);status.reserve<int>(1);
        if(before!=center_bytes()+conversion_bytes())++stats.buffer_reallocations;
        force.zero();site.zero();virial.zero();energy.zero();global.zero();status.zero();
        const int nx=r.neighbor_lanes,ny=r.basic_lanes;
        int tile=r.basic_tile;
        const auto basic_bytes=[&](){return std::size_t(nx)*(tile+1)*sizeof(A)+pair_cache_bytes<C,false>(p,nx);};
        const std::size_t force_bytes=std::size_t(nx)*ny*3*sizeof(A)+pair_cache_bytes<C,true>(p,nx);
        // Stay within the portable dynamic shared-memory allocation, without
        // modifying the FP64 kernel's opt-in attributes.
        const std::size_t limit=std::min<std::size_t>(shared_limit,48*1024);
        while(tile>1&&basic_bytes()>limit)tile=std::max(1,tile/2);
        if(basic_bytes()>limit||(deriv&&force_bytes>limit))throw std::runtime_error("cached precision kernel exceeds shared memory capacity");
        const int tiles=(p.alpha_index_basic_count+tile-1)/tile;
        const dim3 block(nx,ny);
        for(int first=0;first<r.inum;first+=chunk){const int count=std::min(chunk,r.inum-first);
            if(reuse){const float *src=r.device_moments.values_fp32+std::size_t(first)*p.alpha_moments_count;
                if(src!=moments.get<float>())checked(gpuMemcpy(moments.data,src,std::size_t(count)*p.alpha_moments_count*sizeof(float),gpuMemcpyDeviceToDevice));}
            else {NvtxRange range("precision/basic");
                checked(gpuMemset(moments.data,0,std::size_t(count)*p.alpha_moments_count*sizeof(float)));
                if(r.deterministic_reduction)basic<C,A,FloatGeometry><<<count,1,p.rb_size*sizeof(C)>>>(p,in,first,moments.get<float>(),status.get<int>());
                else basic_cached<C,A,FloatGeometry><<<dim3(count,tiles),block,basic_bytes()>>>(p,in,first,moments.get<float>(),tile,status.get<int>());
            }
            {NvtxRange range("precision/propagate");
                if(!r.deterministic_reduction&&p.model_type==int(MTPGpuModel::Linear))propagate<C,A,D,true><<<count,r.propagate_threads>>>(p,in,first,moments.get<float>(),deriv?adjoints.get<D>():nullptr,cache.get<C>(),ns?site.get<A>():nullptr,r.compute_global_energy?energy.get<A>():nullptr,r.device_unweighted_site_energy,r.projection_channels,r.device_projection_species,r.device_projection_coefficients,r.device_projection,reuse,status.get<int>());
                else if(!r.deterministic_reduction)propagate<C,A,D><<<count,r.propagate_threads>>>(p,in,first,moments.get<float>(),deriv?adjoints.get<D>():nullptr,cache.get<C>(),ns?site.get<A>():nullptr,r.compute_global_energy?energy.get<A>():nullptr,r.device_unweighted_site_energy,r.projection_channels,r.device_projection_species,r.device_projection_coefficients,r.device_projection,reuse,status.get<int>());
                else for(int c=0;c<count;++c)propagate<C,A,D><<<1,1>>>(p,in,first+c,moments.get<float>()+std::size_t(c)*p.alpha_moments_count,deriv?adjoints.get<D>()+std::size_t(c)*p.alpha_moments_count:nullptr,cache.get<C>()+std::size_t(c)*cache_stride(p),ns?site.get<A>():nullptr,r.compute_global_energy?energy.get<A>():nullptr,r.device_unweighted_site_energy,r.projection_channels,r.device_projection_species,r.device_projection_coefficients,r.device_projection,reuse,status.get<int>());
            }
            if(deriv){NvtxRange range("precision/force");
                const bool center_neighbors=r.force_kernel==MTPGpuForceKernel::CenterNeighbors||
                    (r.force_kernel==MTPGpuForceKernel::Auto&&count>=4*multiprocessors&&p.alpha_index_basic_count>tile);
                const int force_tile=center_neighbors?p.alpha_index_basic_count:tile;
                const int force_tiles=(p.alpha_index_basic_count+force_tile-1)/force_tile;
                if(!r.deterministic_reduction){
                    if(nw||r.global_virial)forces_cached<C,A,D,FloatGeometry,true><<<dim3(count,force_tiles),block,force_bytes>>>(p,in,first,adjoints.get<D>(),nf?force.get<A>():nullptr,nw?virial.get<A>():nullptr,r.global_virial?global.get<A>():nullptr,force_tile,status.get<int>());
                    else forces_cached<C,A,D,FloatGeometry,false><<<dim3(count,force_tiles),block,force_bytes>>>(p,in,first,adjoints.get<D>(),nf?force.get<A>():nullptr,nullptr,nullptr,force_tile,status.get<int>());
                }
                else for(int c=0;c<count;++c)forces<C,A,D,FloatGeometry><<<1,1,2*p.rb_size*sizeof(C)>>>(p,in,first+c,adjoints.get<D>()+std::size_t(c)*p.alpha_moments_count,nf?force.get<A>():nullptr,nw?virial.get<A>():nullptr,r.global_virial?global.get<A>()+6*(first+c):nullptr,status.get<int>());
                if(center_neighbors)++stats.center_neighbors_force_launches;else ++stats.basic_tile_force_launches;}
            checked(gpuGetLastError());++stats.chunks_executed;
        }
        {NvtxRange range("precision/output_conversion");
            auto copy=[&](std::size_t n,Storage &buffer,double *target){if(n)convert<A><<<(n+127)/128,128>>>(n,buffer.get<A>(),target,status.get<int>());};
            copy(nf,force,out.force);copy(ns,site,out.site_energy);copy(nw,virial,out.site_virial);
            if(r.compute_global_energy)reduce_output<A,1><<<1,256>>>(r.inum,energy.get<A>(),out.energy,status.get<int>());
            if(r.global_virial)reduce_output<A,6><<<6,256>>>(r.inum,global.get<A>(),out.virial,status.get<int>());
            checked(gpuGetLastError());}
        verify();
        complete=chunk==r.inum&&r.inum>0;last_chunk=chunk;
        stats.last_centers_per_chunk=chunk;stats.last_chunk_count=chunk?(r.inum-1)/chunk+1:0;if(reuse)++stats.moment_reuses;
        return chunk;
    }
    int run(const DevicePotential &p,MTPGpuPrecision precision,const MTPGpuComputeRequest &r,MTPGpuDeviceInput in,MTPGpuDeviceOutput out,std::size_t available,std::size_t shared_limit,MTPGpuRuntimeStats &stats) {
        if(precision==MTPGpuPrecision::Mixed)return execute<double,double,double,false>(numeric(p),r,in,out,available,shared_limit,stats);
        static_cast<DevicePotential&>(fp)=p;
        if(precision==MTPGpuPrecision::FP32)return execute<float,float,float,true>(fp,r,in,out,available,shared_limit,stats);
        return execute<float,double,float,false>(fp,r,in,out,available,shared_limit,stats);
    }
};
} // namespace PrecisionInference
