#pragma once

#include <Eigen/Core>
#include <cmath>
#include <utility>

// Private Cartesian first-order scalar for SO4's Jacobian-only consumers.
// No Hessian fields, dynamic allocation, owner state, or cross-call cache.
namespace SO4Density {
struct FirstOrder {
    double value = 0.;
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
    FirstOrder() = default;
    FirstOrder(double v) : value(v) {}
    static FirstOrder variable(double value,int coordinate) {
        FirstOrder result(value);result.gradient[coordinate]=1.;return result;
    }
};
inline FirstOrder operator+(const FirstOrder &a,const FirstOrder &b) {
    FirstOrder result(a.value+b.value);result.gradient=a.gradient+b.gradient;return result;
}
inline FirstOrder operator-(const FirstOrder &a,const FirstOrder &b) {
    FirstOrder result(a.value-b.value);result.gradient=a.gradient-b.gradient;return result;
}
inline FirstOrder operator*(const FirstOrder &a,const FirstOrder &b) {
    FirstOrder result(a.value*b.value);result.gradient=a.gradient*b.value+b.gradient*a.value;return result;
}
inline FirstOrder compose(const FirstOrder &x,double value,double first) {
    FirstOrder result(value);result.gradient=first*x.gradient;return result;
}
inline FirstOrder operator/(const FirstOrder &a,const FirstOrder &b) {
    const double inverse=1./b.value;
    return a*compose(b,inverse,-inverse*inverse);
}
inline FirstOrder sqrt(const FirstOrder &x) {
    const double value=std::sqrt(x.value);return compose(x,value,.5/value);
}
inline FirstOrder sine(const FirstOrder &x) {return compose(x,std::sin(x.value),std::cos(x.value));}
inline FirstOrder cosine(const FirstOrder &x) {return compose(x,std::cos(x.value),-std::sin(x.value));}
inline FirstOrder arctangent(const FirstOrder &x) {return compose(x,std::atan(x.value),1./(1.+x.value*x.value));}
inline double scalar_value(const FirstOrder &x) {return x.value;}
struct ComplexFirstOrder {
    FirstOrder real,imag;
    ComplexFirstOrder() = default;
    ComplexFirstOrder(FirstOrder re,FirstOrder im=FirstOrder(0.)) : real(std::move(re)),imag(std::move(im)) {}
};
inline ComplexFirstOrder operator+(const ComplexFirstOrder &a,const ComplexFirstOrder &b) {
    return {a.real+b.real,a.imag+b.imag};
}
inline ComplexFirstOrder operator-(const ComplexFirstOrder &a,const ComplexFirstOrder &b) {
    return {a.real-b.real,a.imag-b.imag};
}
inline ComplexFirstOrder operator*(const ComplexFirstOrder &a,const ComplexFirstOrder &b) {
    return {a.real*b.real-a.imag*b.imag,a.real*b.imag+a.imag*b.real};
}
inline ComplexFirstOrder operator*(const FirstOrder &a,const ComplexFirstOrder &b) {
    return {a*b.real,a*b.imag};
}
inline ComplexFirstOrder conjugate(const ComplexFirstOrder &a) {return {a.real,FirstOrder(-1.)*a.imag};}
}
