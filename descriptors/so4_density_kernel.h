#pragma once

#include "solid_harmonic_curvature.h"
#include "so4_first_order.h"
#include <span>
#include <complex>
#include <string>
#include <type_traits>
#include <limits>

namespace SO4Density {
using DescriptorCurvature::Jet6;
inline double scalar_value(double x) { return x; }
inline double scalar_value(const Jet6 &x) { return x.value; }
inline double sine(double x) { return std::sin(x); }
inline double cosine(double x) { return std::cos(x); }
inline double arctangent(double x) { return std::atan(x); }
inline Jet6 sine(const Jet6 &x) {
    return DescriptorCurvature::compose(x,std::sin(x.value),std::cos(x.value),-std::sin(x.value));
}
inline Jet6 cosine(const Jet6 &x) { return DescriptorCurvature::cos(x); }
inline Jet6 arctangent(const Jet6 &x) {
    const double d=1.+x.value*x.value;
    return DescriptorCurvature::compose(x,std::atan(x.value),1./d,-2.*x.value/(d*d));
}

// Runtime conventions reproduce the old compile-time embeddings, including
// the sign branch of r/tan(theta) in Q1/Q2. Differentiation uses this same map.
template<class T>
void parameters(const T &x,const T &y,const T &z,const T &r,double radius,
                int convention,bool atan_map,T &ar,T &ai,T &br,T &bi,bool reject_branch=true) {
    const T angle=atan_map ? arctangent(r/T(radius)) : T(M_PI/radius)*r;
    T s=sine(angle),c=cosine(angle);
    if (convention<=2) {
        if constexpr(std::is_same_v<T,Jet6> || std::is_same_v<T,FirstOrder>) {
            if(reject_branch && !atan_map && scalar_value(angle)>M_PI/2. &&
               std::abs(scalar_value(s))<=64.*std::numeric_limits<double>::epsilon())
                throw std::invalid_argument("SO4 Q1/Q2 derivative is undefined at a radial sign-branch change");
        }
        const double sign=scalar_value(s)<0. ? -1. : 1.;
        s=T(sign)*s; c=T(sign)*c;
    }
    const T sr=s/r;
    ar=c;
    if (convention==1) { ai=sr*z; br=sr*x; bi=sr*y; }
    else if (convention==2 || convention==3) { ai=sr*z; br=sr*y; bi=sr*x; }
    else { ai=sr*x; br=sr*z; bi=sr*y; }
}

// Flattened matrices retain the existing neighbor/site coefficient ordering.
inline std::vector<int> offsets(int maximum) {
    std::vector<int> result(maximum+2);
    for (int j=0;j<=maximum;++j) result[j+1]=result[j]+(j+1)*(j+1);
    return result;
}
inline int index(const std::vector<int> &offset,int j,int ma,int mb) {
    return offset[j]+ma*(j+1)+mb;
}
inline std::vector<std::complex<double>> wigner_values(std::complex<double> a,std::complex<double> b,
    int maximum,const std::vector<int> &offset) {
    std::vector<std::complex<double>> u(offset.back());u[0]=1.;
    const auto ix=[&](int j,int ma,int mb){return index(offset,j,ma,mb);};
    for(int j=1;j<=maximum;++j) {
        for(int mb=0;mb<=j/2;++mb) {
            const double root=std::sqrt(double(j)/(j-mb));
            u[ix(j,0,mb)]=root*a*u[ix(j-1,0,mb)];
            u[ix(j,j,mb)]=-root*b*u[ix(j-1,j-1,mb)];
            for(int ma=1;ma<j;++ma)
                u[ix(j,ma,mb)]=std::sqrt(double(j-ma)/(j-mb))*a*u[ix(j-1,ma,mb)]
                    -std::sqrt(double(ma)/(j-mb))*b*u[ix(j-1,ma-1,mb)];
        }
        for(int mb=j/2+1;mb<=j;++mb)for(int ma=0;ma<=j;++ma)
            u[ix(j,ma,mb)]=((ma+mb)%2?-1.:1.)*std::conj(u[ix(j,j-ma,j-mb)]);
    }
    return u;
}
// The caller owns the coefficient buffer and may reuse it across edges. All
// entries are assigned in recurrence order; no per-edge allocation or Hessian.
inline void wigner_first_order(const ComplexFirstOrder &a,const ComplexFirstOrder &b,
    int maximum,const std::vector<int> &offset,std::span<ComplexFirstOrder> u) {
    if(u.size()!=static_cast<std::size_t>(offset.back()))
        throw std::invalid_argument("SO4 first-order coefficient buffer has incompatible size");
    u[0]=ComplexFirstOrder(FirstOrder(1.));
    const auto ix=[&](int j,int ma,int mb){return index(offset,j,ma,mb);};
    for(int j=1;j<=maximum;++j) {
        for(int mb=0;mb<=j/2;++mb) {
            const double root=std::sqrt(double(j)/(j-mb));
            u[ix(j,0,mb)]=FirstOrder(root)*(a*u[ix(j-1,0,mb)]);
            u[ix(j,j,mb)]=FirstOrder(-root)*(b*u[ix(j-1,j-1,mb)]);
            for(int ma=1;ma<j;++ma)
                u[ix(j,ma,mb)]=FirstOrder(std::sqrt(double(j-ma)/(j-mb)))*(a*u[ix(j-1,ma,mb)])
                    -FirstOrder(std::sqrt(double(ma)/(j-mb)))*(b*u[ix(j-1,ma-1,mb)]);
        }
        for(int mb=j/2+1;mb<=j;++mb)for(int ma=0;ma<=j;++ma)
            u[ix(j,ma,mb)]=FirstOrder((ma+mb)%2?-1.:1.)*conjugate(u[ix(j,j-ma,j-mb)]);
    }
}
inline std::vector<DescriptorCurvature::ComplexJet> wigner(
    const DescriptorCurvature::ComplexJet &a,const DescriptorCurvature::ComplexJet &b,
    int maximum,const std::vector<int> &offset) {
    using namespace DescriptorCurvature;
    std::vector<ComplexJet> u(offset.back());
    u[0]=ComplexJet(Jet6(1.));
    const auto ix=[&](int j,int ma,int mb){return index(offset,j,ma,mb);};
    for (int j=1;j<=maximum;++j) {
        for (int mb=0;mb<=j/2;++mb) {
            const double root=std::sqrt(double(j)/(j-mb));
            u[ix(j,0,mb)]=Jet6(root)*(a*u[ix(j-1,0,mb)]);
            u[ix(j,j,mb)]=Jet6(-root)*(b*u[ix(j-1,j-1,mb)]);
            for (int ma=1;ma<j;++ma)
                u[ix(j,ma,mb)]=Jet6(std::sqrt(double(j-ma)/(j-mb)))*(a*u[ix(j-1,ma,mb)])
                    -Jet6(std::sqrt(double(ma)/(j-mb)))*(b*u[ix(j-1,ma-1,mb)]);
        }
        for (int mb=j/2+1;mb<=j;++mb)
            for (int ma=0;ma<=j;++ma)
                u[ix(j,ma,mb)]=Jet6((ma+mb)%2?-1.:1.)*conjugate(u[ix(j,j-ma,j-mb)]);
    }
    return u;
}
} // namespace SO4Density
