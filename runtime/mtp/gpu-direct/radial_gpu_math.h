// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mtp_gpu_direct.h"
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define IMRMLP_RADIAL_HD __host__ __device__
#else
#define IMRMLP_RADIAL_HD
#endif

namespace MLIPCore::GpuDirect::RadialMath {

template<class T> struct Scalar {using type=T;};
template<class T> using ScalarType=typename Scalar<T>::type;

// Kept independent of device storage so the complete formula family can be
// checked against the native polymorphic bases in CPU-only test builds too.
template<class Real>
IMRMLP_RADIAL_HD inline void coordinate(RadialTransform type, Real r,
    ScalarType<Real> lo, ScalarType<Real> hi, ScalarType<Real> lambda, ScalarType<Real> scale_lambda,
    Real &value, Real &derivative)
{
    const Real width = hi - lo;
    if (type == RadialTransform::I2) {
        const Real q = (r - hi) / width;
        value = Real(2.0) * q * q - Real(1.0);
        derivative = Real(4.0) * q / width;
        return;
    }
    const bool left = type == RadialTransform::ExponentialLeft;
    const Real range = left ? hi : width;
    const Real t = left ? r / hi : (r - lo) / width;
    if (type == RadialTransform::Linear || lambda == Real(0.0)) {
        value = Real(2.0) * t - Real(1.0);
        derivative = Real(2.0) / range;
    } else {
        const Real denominator = std::expm1(lambda);
        const Real exponent = lambda * (Real(1.0) - t);
        const Real numerator = type == RadialTransform::Exponential
            ? std::expm1(exponent) : std::exp(exponent) - Real(1.0);
        value = Real(1.0) - Real(2.0) * numerator / denominator;
        derivative = Real(2.0) * lambda / range * std::exp(exponent) / denominator;
    }
    if (type == RadialTransform::ExponentialTanh && scale_lambda != Real(0.0)) {
        const Real scaled = std::tanh(scale_lambda * value);
        const Real denominator = std::tanh(scale_lambda);
        value = scaled / denominator;
        derivative *= scale_lambda * (Real(1.0) - scaled * scaled) / denominator;
    }
}

template<class Real>
IMRMLP_RADIAL_HD inline void envelope(RadialCutoff type, Real r,
    ScalarType<Real> lo, ScalarType<Real> hi, ScalarType<Real> b, ScalarType<Real> r1, ScalarType<Real> r2,
    Real &value, Real &derivative)
{
    constexpr Real pi = Real(3.141592653589793238462643383279502884);
    const Real width = hi - lo;
    const Real t = (r - lo) / width;
    const Real q = Real(1.0) - t;
    switch (type) {
    case RadialCutoff::Quadratic:
        value = (r - hi) * (r - hi); derivative = Real(2.0) * (r - hi); return;
    case RadialCutoff::NormalizedQuadratic:
        value = q * q; derivative = -Real(2.0) * q / width; return;
    case RadialCutoff::Cosine:
    case RadialCutoff::CosineLeft:
    case RadialCutoff::CosineUnscaled:
    case RadialCutoff::CosineSquared:
    case RadialCutoff::SineSquared: {
        const bool left = type == RadialCutoff::CosineLeft;
        const Real frequency = pi / (left ? hi : width);
        const Real phase = left ? r * frequency : pi * t;
        const Real s = std::sin(phase), c = std::cos(phase);
        if (type == RadialCutoff::SineSquared) {
            value = s * s; derivative = Real(2.0) * s * c * frequency;
        } else if (type == RadialCutoff::CosineSquared) {
            value = (Real(1.0) + c) * (Real(1.0) + c);
            derivative = -Real(2.0) * (Real(1.0) + c) * s * frequency;
        } else {
            const Real factor = type == RadialCutoff::CosineUnscaled ? Real(1.0) : Real(0.5);
            value = factor * (Real(1.0) + c); derivative = -factor * frequency * s;
        }
        return;
    }
    case RadialCutoff::Poly3:
    case RadialCutoff::Poly4:
    case RadialCutoff::Poly5: {
        const int degree = type == RadialCutoff::Poly3 ? 2 :
            (type == RadialCutoff::Poly4 ? 3 : 4);
        const Real previous = degree == 2 ? q : (degree == 3 ? q * q : q * q * q);
        const Real power = previous * q;
        value = power * (Real(1.0) + (b + degree) * t);
        derivative = (-degree * previous * (Real(1.0) + (b + degree) * t)
                      + power * (b + degree)) / width;
        return;
    }
    case RadialCutoff::Smoothstep5:
        value = q * q * q * (Real(10.0) + q * (-Real(15.0) + Real(6.0) * q));
        derivative = -Real(30.0) * q * q * (Real(1.0) - q) * (Real(1.0) - q) / width;
        return;
    case RadialCutoff::Smoothstep7: {
        if (r > r1 && r < r2) { value = Real(1.0); derivative = Real(0.0); return; }
        const bool rising = r <= r1;
        const Real interval = rising ? r1 - lo : hi - r2;
        const Real p = rising ? (r - lo) / interval : (r - r2) / interval;
        const Real p2 = p * p, p3 = p2 * p, p4 = p2 * p2;
        const Real f = p4 * (Real(35.0) + p * (-Real(84.0) + p * (Real(70.0) - Real(20.0) * p)));
        const Real df = Real(140.0) * p3 * (Real(1.0) - p) * (Real(1.0) - p) * (Real(1.0) - p) / interval;
        value = rising ? f : Real(1.0) - f; derivative = rising ? df : -df;
        return;
    }
    }
    value = derivative = Real(0.0);
}

// Analytic radial curvature. Keep the established value/first-derivative
// arithmetic untouched for force callers and make the second formulas usable
// by CPU-only coverage tests as well as the device HVP kernel.
template<class Real>
IMRMLP_RADIAL_HD inline void coordinate_second(RadialTransform type, Real r,
    ScalarType<Real> lo, ScalarType<Real> hi, ScalarType<Real> lambda,
    ScalarType<Real> scale_lambda, Real &value, Real &derivative, Real &second)
{
    coordinate(type, r, lo, hi, lambda, scale_lambda, value, derivative);
    const Real width = hi - lo;
    if (type == RadialTransform::I2) {
        second = Real(4.0) / (width * width); return;
    }
    const Real range = type == RadialTransform::ExponentialLeft ? Real(hi) : width;
    Real base = value, first = derivative;
    if (type == RadialTransform::ExponentialTanh && scale_lambda != Real(0.0))
        coordinate(RadialTransform::ExponentialTanh, r, lo, hi, lambda,
                   ScalarType<Real>(0.0), base, first);
    second = type == RadialTransform::Linear || lambda == Real(0.0)
        ? Real(0.0) : -lambda * first / range;
    if (type == RadialTransform::ExponentialTanh && scale_lambda != Real(0.0)) {
        const Real y = std::tanh(scale_lambda * base);
        const Real factor = scale_lambda * (Real(1.0) - y*y) / std::tanh(scale_lambda);
        second = factor * (second - Real(2.0)*scale_lambda*y*first*first);
    }
}

template<class Real>
IMRMLP_RADIAL_HD inline void envelope_second(RadialCutoff type, Real r,
    ScalarType<Real> lo, ScalarType<Real> hi, ScalarType<Real> b,
    ScalarType<Real> r1, ScalarType<Real> r2,
    Real &value, Real &derivative, Real &second)
{
    envelope(type, r, lo, hi, b, r1, r2, value, derivative);
    constexpr Real pi = Real(3.141592653589793238462643383279502884);
    const Real width = hi - lo, t = (r-lo)/width, q = Real(1.0)-t;
    switch (type) {
    case RadialCutoff::Quadratic: second = Real(2.0); return;
    case RadialCutoff::NormalizedQuadratic: second = Real(2.0)/(width*width); return;
    case RadialCutoff::Cosine:
    case RadialCutoff::CosineLeft:
    case RadialCutoff::CosineUnscaled:
    case RadialCutoff::CosineSquared:
    case RadialCutoff::SineSquared: {
        const bool left = type == RadialCutoff::CosineLeft;
        const Real frequency = pi / (left ? Real(hi) : width);
        const Real phase = left ? r*frequency : pi*t;
        const Real s = std::sin(phase), c = std::cos(phase);
        if (type == RadialCutoff::SineSquared)
            second = Real(2.0)*frequency*frequency*(c*c-s*s);
        else if (type == RadialCutoff::CosineSquared)
            second = Real(2.0)*frequency*frequency*(s*s-(Real(1.0)+c)*c);
        else
            second = -(type == RadialCutoff::CosineUnscaled ? Real(1.0) : Real(0.5))*frequency*frequency*c;
        return;
    }
    case RadialCutoff::Poly3:
    case RadialCutoff::Poly4:
    case RadialCutoff::Poly5: {
        const int n = type == RadialCutoff::Poly3 ? 2 : (type == RadialCutoff::Poly4 ? 3 : 4);
        const Real power = n == 2 ? Real(1.0) : (n == 3 ? q : q*q);
        second = (n*(n-1)*power*(Real(1.0)+(b+n)*t)-Real(2.0)*n*(b+n)*power*q)/(width*width);
        return;
    }
    case RadialCutoff::Smoothstep5:
        second = Real(60.0)*q*(Real(1.0)-q)*(Real(1.0)-Real(2.0)*q)/(width*width); return;
    case RadialCutoff::Smoothstep7: {
        if (r > r1 && r < r2) { second = Real(0.0); return; }
        const bool rising = r <= r1;
        const Real interval = rising ? r1-lo : hi-r2;
        const Real p = rising ? (r-lo)/interval : (r-r2)/interval;
        const Real z = Real(1.0)-p;
        second = (rising ? Real(420.0) : -Real(420.0))*p*p*z*z*(Real(1.0)-Real(2.0)*p)/(interval*interval);
        return;
    }
    }
    second = Real(0.0);
}

} // namespace MLIPCore::GpuDirect::RadialMath

#undef IMRMLP_RADIAL_HD
