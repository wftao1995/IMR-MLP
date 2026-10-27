#pragma once

#include <Eigen/Core>
#include <cmath>

namespace DescriptorCurvature {
// Constant-size local second-order differentiation. A Behler contribution
// involves at most two neighbors, regardless of total environment size.
struct Jet6 {
    double value = 0.;
    Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Jet6() = default;
    Jet6(double v) : value(v) {}
    static Jet6 variable(double value, int coordinate) {
        Jet6 result(value);
        result.gradient[coordinate] = 1.;
        return result;
    }
};
inline Jet6 operator+(const Jet6 &a, const Jet6 &b) {
    Jet6 result(a.value + b.value);
    result.gradient = a.gradient + b.gradient;
    result.hessian = a.hessian + b.hessian;
    return result;
}
inline Jet6 operator-(const Jet6 &a, const Jet6 &b) {
    Jet6 result(a.value - b.value);
    result.gradient = a.gradient - b.gradient;
    result.hessian = a.hessian - b.hessian;
    return result;
}
inline Jet6 operator*(const Jet6 &a, const Jet6 &b) {
    Jet6 result(a.value * b.value);
    result.gradient = a.gradient * b.value + b.gradient * a.value;
    result.hessian = a.hessian * b.value + b.hessian * a.value
        + a.gradient * b.gradient.transpose() + b.gradient * a.gradient.transpose();
    return result;
}
inline Jet6 compose(const Jet6 &x, double value, double first, double second) {
    Jet6 result(value);
    result.gradient = first * x.gradient;
    result.hessian = first * x.hessian + second * x.gradient * x.gradient.transpose();
    return result;
}
inline Jet6 inverse(const Jet6 &x) {
    const double v = 1. / x.value;
    return compose(x, v, -v*v, 2.*v*v*v);
}
inline Jet6 operator/(const Jet6 &a, const Jet6 &b) { return a * inverse(b); }
inline Jet6 sqrt(const Jet6 &x) {
    const double v = std::sqrt(x.value);
    return compose(x, v, .5/v, -.25/(x.value*v));
}
inline Jet6 exp(const Jet6 &x) {
    const double v = std::exp(x.value);
    return compose(x, v, v, v);
}
inline Jet6 cos(const Jet6 &x) {
    return compose(x, std::cos(x.value), -std::sin(x.value), -std::cos(x.value));
}
inline Jet6 pow(const Jet6 &x, double exponent) {
    if (exponent == 0.) return Jet6(1.);
    if (exponent == 1.) return x;
    if (exponent == 2.) return x*x;
    return compose(x, std::pow(x.value, exponent), exponent*std::pow(x.value, exponent-1.),
        exponent*(exponent-1.)*std::pow(x.value, exponent-2.));
}
} // namespace DescriptorCurvature
