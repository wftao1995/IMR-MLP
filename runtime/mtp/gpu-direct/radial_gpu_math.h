#pragma once

#include "mtp_gpu_direct.h"
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define IMRMLP_RADIAL_HD __host__ __device__
#else
#define IMRMLP_RADIAL_HD
#endif

namespace MLIPCore::GpuDirect::RadialMath {

// Kept independent of device storage so the complete formula family can be
// checked against the native polymorphic bases in CPU-only test builds too.
IMRMLP_RADIAL_HD inline void coordinate(RadialTransform type, double r,
    double lo, double hi, double lambda, double scale_lambda,
    double &value, double &derivative)
{
    const double width = hi - lo;
    if (type == RadialTransform::I2) {
        const double q = (r - hi) / width;
        value = 2.0 * q * q - 1.0;
        derivative = 4.0 * q / width;
        return;
    }
    const bool left = type == RadialTransform::ExponentialLeft;
    const double range = left ? hi : width;
    const double t = left ? r / hi : (r - lo) / width;
    if (type == RadialTransform::Linear || lambda == 0.0) {
        value = 2.0 * t - 1.0;
        derivative = 2.0 / range;
    } else {
        const double denominator = ::expm1(lambda);
        const double exponent = lambda * (1.0 - t);
        const double numerator = type == RadialTransform::Exponential
            ? ::expm1(exponent) : ::exp(exponent) - 1.0;
        value = 1.0 - 2.0 * numerator / denominator;
        derivative = 2.0 * lambda / range * ::exp(exponent) / denominator;
    }
    if (type == RadialTransform::ExponentialTanh && scale_lambda != 0.0) {
        const double scaled = ::tanh(scale_lambda * value);
        const double denominator = ::tanh(scale_lambda);
        value = scaled / denominator;
        derivative *= scale_lambda * (1.0 - scaled * scaled) / denominator;
    }
}

IMRMLP_RADIAL_HD inline void envelope(RadialCutoff type, double r,
    double lo, double hi, double b, double r1, double r2,
    double &value, double &derivative)
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double width = hi - lo;
    const double t = (r - lo) / width;
    const double q = 1.0 - t;
    switch (type) {
    case RadialCutoff::Quadratic:
        value = (r - hi) * (r - hi); derivative = 2.0 * (r - hi); return;
    case RadialCutoff::NormalizedQuadratic:
        value = q * q; derivative = -2.0 * q / width; return;
    case RadialCutoff::Cosine:
    case RadialCutoff::CosineLeft:
    case RadialCutoff::CosineUnscaled:
    case RadialCutoff::CosineSquared:
    case RadialCutoff::SineSquared: {
        const bool left = type == RadialCutoff::CosineLeft;
        const double frequency = pi / (left ? hi : width);
        const double phase = left ? r * frequency : pi * t;
        const double s = ::sin(phase), c = ::cos(phase);
        if (type == RadialCutoff::SineSquared) {
            value = s * s; derivative = 2.0 * s * c * frequency;
        } else if (type == RadialCutoff::CosineSquared) {
            value = (1.0 + c) * (1.0 + c);
            derivative = -2.0 * (1.0 + c) * s * frequency;
        } else {
            const double factor = type == RadialCutoff::CosineUnscaled ? 1.0 : 0.5;
            value = factor * (1.0 + c); derivative = -factor * frequency * s;
        }
        return;
    }
    case RadialCutoff::Poly3:
    case RadialCutoff::Poly4:
    case RadialCutoff::Poly5: {
        const int degree = type == RadialCutoff::Poly3 ? 2 :
            (type == RadialCutoff::Poly4 ? 3 : 4);
        const double previous = degree == 2 ? q : (degree == 3 ? q * q : q * q * q);
        const double power = previous * q;
        value = power * (1.0 + (b + degree) * t);
        derivative = (-degree * previous * (1.0 + (b + degree) * t)
                      + power * (b + degree)) / width;
        return;
    }
    case RadialCutoff::Smoothstep5:
        value = q * q * q * (10.0 + q * (-15.0 + 6.0 * q));
        derivative = -30.0 * q * q * (1.0 - q) * (1.0 - q) / width;
        return;
    case RadialCutoff::Smoothstep7: {
        if (r > r1 && r < r2) { value = 1.0; derivative = 0.0; return; }
        const bool rising = r <= r1;
        const double interval = rising ? r1 - lo : hi - r2;
        const double p = rising ? (r - lo) / interval : (r - r2) / interval;
        const double p2 = p * p, p3 = p2 * p, p4 = p2 * p2;
        const double f = p4 * (35.0 + p * (-84.0 + p * (70.0 - 20.0 * p)));
        const double df = 140.0 * p3 * (1.0 - p) * (1.0 - p) * (1.0 - p) / interval;
        value = rising ? f : 1.0 - f; derivative = rising ? df : -df;
        return;
    }
    }
    value = derivative = 0.0;
}

} // namespace MLIPCore::GpuDirect::RadialMath

#undef IMRMLP_RADIAL_HD
