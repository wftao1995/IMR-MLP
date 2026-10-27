#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace AngularFuncs::detail {
// Orthonormal real solid harmonics r^l Y_lm, Condon--Shortley convention,
// index l*l+l+m. Scalar needs construction from double and +,-,* (e.g. Jet6).
// Passing unit Cartesian coordinates gives angular harmonics; their caller
// supplies derivatives of the normalization. No division or pole branch here.
template<class Scalar>
std::vector<Scalar> real_solid_harmonics(int maximum_l,int maximum_m,
    const Scalar& x,const Scalar& y,const Scalar& z) {
    if(maximum_l<0 || maximum_m<0 || maximum_m>maximum_l ||
        (static_cast<long long>(maximum_l)+1)*(static_cast<long long>(maximum_l)+2)>std::numeric_limits<int>::max())
        throw std::invalid_argument("Invalid real spherical harmonic dimensions");
    struct Pair {Scalar re{0.},im{0.};};
    const auto triangular=[](int l,int m){return size_t(l)*(l+1)/2+m;};
    const auto index=[](int l,int m){return size_t(l)*(l+1)+m;};
    std::vector<Pair> values(size_t(maximum_l+1)*(maximum_l+2)/2);
    std::vector<Scalar> result(size_t(maximum_l+1)*(maximum_l+1),Scalar(0.));
    const Scalar r2=x*x+y*y+z*z;
    values[0].re=Scalar(1/std::sqrt(4*std::acos(-1.)));
    for(int m=0;m<=maximum_m;++m) {
        auto& diagonal=values[triangular(m,m)];
        if(m) {
            const auto& previous=values[triangular(m-1,m-1)];
            const Scalar factor(-std::sqrt((2.*m+1)/(2.*m)));
            diagonal.re=factor*(x*previous.re-y*previous.im);
            diagonal.im=factor*(x*previous.im+y*previous.re);
        }
        for(int l=m+1;l<=maximum_l;++l) {
            auto& current=values[triangular(l,m)];const auto& previous=values[triangular(l-1,m)];
            const Scalar a(l==m+1 ? std::sqrt(2.*m+3) : std::sqrt((4.*l*l-1)/(double(l)*l-double(m)*m)));
            if(l==m+1) {current.re=a*z*previous.re;current.im=a*z*previous.im;}
            else {
                const auto& before=values[triangular(l-2,m)];
                const Scalar b(-std::sqrt((double(l-1)*(l-1)-double(m)*m)/(4.*(l-1)*(l-1)-1)));
                current.re=a*(z*previous.re+b*r2*before.re);
                current.im=a*(z*previous.im+b*r2*before.im);
            }
        }
        const Scalar scale(m ? std::sqrt(2.) : 1.);
        for(int l=m;l<=maximum_l;++l) {
            const auto& current=values[triangular(l,m)];
            result[index(l,m)]=scale*current.re;
            if(m) result[index(l,-m)]=scale*current.im;
        }
    }
    return result;
}
}
