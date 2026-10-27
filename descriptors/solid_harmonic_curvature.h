#pragma once

#include "local_curvature_jet.h"
#include <vector>

namespace DescriptorCurvature {
struct ComplexJet {
    Jet6 real, imag;
    ComplexJet() = default;
    ComplexJet(Jet6 re, Jet6 im = Jet6(0.)) : real(std::move(re)), imag(std::move(im)) {}
};
inline ComplexJet operator+(const ComplexJet &a, const ComplexJet &b) {
    return {a.real+b.real, a.imag+b.imag};
}
inline ComplexJet operator-(const ComplexJet &a, const ComplexJet &b) {
    return {a.real-b.real, a.imag-b.imag};
}
inline ComplexJet operator*(const ComplexJet &a, const ComplexJet &b) {
    return {a.real*b.real-a.imag*b.imag, a.real*b.imag+a.imag*b.real};
}
inline ComplexJet operator*(const Jet6 &a, const ComplexJet &b) { return {a*b.real, a*b.imag}; }
inline ComplexJet conjugate(const ComplexJet &a) { return {a.real, (-1.)*a.imag}; }

// r^l Y_lm in the existing complex Condon--Shortley convention. Cartesian
// recurrences have finite derivatives on both poles and at the origin.
inline std::vector<ComplexJet> solid_harmonics(const Eigen::RowVector3d &position, int maximum_l,
                                              bool angular_only = false) {
    const Jet6 x = Jet6::variable(position[0], 0), y = Jet6::variable(position[1], 1),
               z = Jet6::variable(position[2], 2);
    const Jet6 r2 = x*x+y*y+z*z;
    const int size = (maximum_l+1)*(maximum_l+1);
    std::vector<ComplexJet> raw(size), result(size);
    const auto index = [](int l, int m) { return l*(l+1)+m; };
    raw[0] = ComplexJet(Jet6(1.));
    for (int m = 0; m <= maximum_l; ++m) {
        if (m > 0) raw[index(m,m)] = Jet6(-(2.*m-1.)) *
            (ComplexJet(x,y)*raw[index(m-1,m-1)]);
        if (m < maximum_l) raw[index(m+1,m)] = ((2.*m+1.)*z)*raw[index(m,m)];
        for (int l = m+2; l <= maximum_l; ++l)
            raw[index(l,m)] = Jet6(1./(l-m))*
                (((2.*l-1.)*z)*raw[index(l-1,m)] - ((l+m-1.)*r2)*raw[index(l-2,m)]);
        for (int l = m; l <= maximum_l; ++l) {
            const double normalization = std::sqrt((2.*l+1.)/(4.*M_PI)
                *std::exp(std::lgamma(l-m+1.)-std::lgamma(l+m+1.)));
            const Jet6 scale = angular_only && l > 0
                ? normalization*DescriptorCurvature::pow(r2, -.5*l) : Jet6(normalization);
            result[index(l,m)] = scale*raw[index(l,m)];
            if (m > 0) result[index(l,-m)] = Jet6(m%2 ? -1. : 1.)*conjugate(result[index(l,m)]);
        }
    }
    return result;
}
} // namespace DescriptorCurvature
