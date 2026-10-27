// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

// Portable radial-basis implementation shared by the standalone MTP KIM
// core and its tests.  Keep this file free of IMRMLP, Eigen, and KIM API
// dependencies so an installed model driver can be built from its sources.

#include <algorithm>
#include <cmath>
#include <istream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mtp_kim {

enum class BasisType {
    ChebyshevConfigurable,
    Chebyshev,
    ChebyshevI2,
    ChebyshevPoly5BSide,
    ChebyshevPoly7BSideC,
    ChebyshevPoly2,
    ChebyshevPoly3,
    ChebyshevPoly4,
    ChebyshevPoly5,
    ChebyshevFlip,
    ChebyshevShiftedFlip,
    PolyB,
    PolyBCos,
    ChebyCos,
    ChebyDoubleCosShiftedScaled,
    ChebyDoubleSinDoubleScaledShiftedMiddle,
    ChebyDoubleSinMiddle,
    ChebyDoubleCosLeft,
    ChebyDoubleCosShiftedScaledLeft,
    ChebyCosShiftedScaled,
    Constant,
};

namespace radial_detail {

struct Dual {
    double val = 0.0;
    double der = 0.0;
};

inline Dual operator+(Dual a, Dual b) { return {a.val + b.val, a.der + b.der}; }
inline Dual operator-(Dual a, Dual b) { return {a.val - b.val, a.der - b.der}; }
inline Dual operator-(Dual a) { return {-a.val, -a.der}; }
inline Dual operator*(Dual a, Dual b)
{
    return {a.val * b.val, a.der * b.val + a.val * b.der};
}
inline Dual operator/(Dual a, Dual b)
{
    const double inv = 1.0 / b.val;
    return {a.val * inv, (a.der * b.val - a.val * b.der) * inv * inv};
}
inline Dual operator+(Dual a, double b) { return a + Dual{b, 0.0}; }
inline Dual operator+(double a, Dual b) { return Dual{a, 0.0} + b; }
inline Dual operator-(Dual a, double b) { return a - Dual{b, 0.0}; }
inline Dual operator-(double a, Dual b) { return Dual{a, 0.0} - b; }
inline Dual operator*(Dual a, double b) { return {a.val * b, a.der * b}; }
inline Dual operator*(double a, Dual b) { return b * a; }
inline Dual operator/(Dual a, double b) { return {a.val / b, a.der / b}; }

inline Dual sin(Dual x) { return {std::sin(x.val), std::cos(x.val) * x.der}; }
inline Dual cos(Dual x) { return {std::cos(x.val), -std::sin(x.val) * x.der}; }
inline Dual exp(Dual x)
{
    const double value = std::exp(x.val);
    return {value, value * x.der};
}
inline Dual tanh(Dual x)
{
    const double value = std::tanh(x.val);
    return {value, (1.0 - value * value) * x.der};
}
inline Dual pow(Dual x, int exponent)
{
    Dual result{1.0, 0.0};
    while (exponent > 0) {
        if (exponent & 1) result = result * x;
        x = x * x;
        exponent >>= 1;
    }
    return result;
}

inline std::string read_value(std::istream &is, const char *key)
{
    std::string actual;
    std::string equals;
    std::string value;
    if (!(is >> actual >> equals >> value) || actual != key || equals != "=")
        throw std::runtime_error(std::string("expected ") + key + " = <value>");
    return value;
}

inline double read_double(std::istream &is, const char *key)
{
    const std::string text = read_value(is, key);
    std::size_t parsed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid ") + key);
    }
    if (parsed != text.size() || !std::isfinite(value))
        throw std::runtime_error(std::string("invalid ") + key);
    return value;
}

inline int read_int(std::istream &is, const char *key)
{
    const std::string text = read_value(is, key);
    std::size_t parsed = 0;
    long value = 0;
    try {
        value = std::stol(text, &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid ") + key);
    }
    if (parsed != text.size() || value <= 0 ||
        value > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string("invalid ") + key);
    return static_cast<int>(value);
}

inline BasisType basis_type_from_name(const std::string &name)
{
    if (name == "RBChebyshevConfigurable") return BasisType::ChebyshevConfigurable;
    if (name == "RBChebyshev") return BasisType::Chebyshev;
    if (name == "RBChebyshevI2") return BasisType::ChebyshevI2;
    if (name == "RBChebyshevPoly5BSide") return BasisType::ChebyshevPoly5BSide;
    if (name == "RBChebyshevPoly7BSideC") return BasisType::ChebyshevPoly7BSideC;
    if (name == "RBChebyshevPoly2") return BasisType::ChebyshevPoly2;
    if (name == "RBChebyshevPoly3") return BasisType::ChebyshevPoly3;
    if (name == "RBChebyshevPoly4") return BasisType::ChebyshevPoly4;
    if (name == "RBChebyshevPoly5") return BasisType::ChebyshevPoly5;
    if (name == "RBChebyshevFlip") return BasisType::ChebyshevFlip;
    if (name == "RBChebyshevShiftedFlip") return BasisType::ChebyshevShiftedFlip;
    if (name == "RBPolyB") return BasisType::PolyB;
    if (name == "RBPolyBCos") return BasisType::PolyBCos;
    if (name == "RBChebyCos") return BasisType::ChebyCos;
    if (name == "RBChebyDoubleCosShiftedScaled")
        return BasisType::ChebyDoubleCosShiftedScaled;
    if (name == "RBChebyDoubleSinDoubleScaledShiftedMiddle")
        return BasisType::ChebyDoubleSinDoubleScaledShiftedMiddle;
    if (name == "RBChebyDoubleSinMiddle") return BasisType::ChebyDoubleSinMiddle;
    if (name == "RBChebyDoubleCosLeft") return BasisType::ChebyDoubleCosLeft;
    if (name == "RBChebyDoubleCosShiftedScaledLeft")
        return BasisType::ChebyDoubleCosShiftedScaledLeft;
    if (name == "RBChebyCosShiftedScaled") return BasisType::ChebyCosShiftedScaled;
    if (name == "RBConstant") return BasisType::Constant;
    throw std::runtime_error("unsupported basis type " + name);
}

// Compute the symmetric inverse square root used by RBPolyB/RBPolyBCos.
// A cyclic Jacobi eigensolver keeps the KIM core independent of Eigen.
inline std::vector<double> symmetric_inverse_sqrt(std::vector<double> matrix,
                                                  int n)
{
    // The monomial overlap matrix becomes ill-conditioned rather quickly.
    // Accumulate the eigendecomposition in long double so that the portable
    // solver remains useful for the basis sizes seen in trained models.
    std::vector<long double> work(matrix.begin(), matrix.end());
    std::vector<long double> vectors(static_cast<std::size_t>(n) * n, 0.0L);
    for (int i = 0; i < n; ++i) vectors[i * n + i] = 1.0L;

    const int max_sweeps = std::max(64, 32 * n);
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        long double off = 0.0L;
        long double diagonal = 0.0L;
        for (int p = 0; p < n; ++p) {
            diagonal = std::max(diagonal, std::abs(work[p * n + p]));
            for (int q = p + 1; q < n; ++q)
                off = std::max(off, std::abs(work[p * n + q]));
        }
        if (off <= 32.0L * std::numeric_limits<long double>::epsilon() *
                       std::max(1.0L, diagonal))
            break;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const long double apq = work[p * n + q];
                if (std::abs(apq) <= std::numeric_limits<long double>::epsilon() *
                                     std::max(1.0L, diagonal))
                    continue;
                const long double tau = (work[q * n + q] - work[p * n + p]) /
                                        (2.0L * apq);
                const long double t = std::copysign(
                    1.0L / (std::abs(tau) + std::sqrt(1.0L + tau * tau)), tau);
                const long double c = 1.0L / std::sqrt(1.0L + t * t);
                const long double s = t * c;

                const long double app = work[p * n + p];
                const long double aqq = work[q * n + q];
                work[p * n + p] = app - t * apq;
                work[q * n + q] = aqq + t * apq;
                work[p * n + q] = work[q * n + p] = 0.0L;
                for (int k = 0; k < n; ++k) {
                    if (k == p || k == q) continue;
                    const long double akp = work[k * n + p];
                    const long double akq = work[k * n + q];
                    work[k * n + p] = work[p * n + k] = c * akp - s * akq;
                    work[k * n + q] = work[q * n + k] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const long double vkp = vectors[k * n + p];
                    const long double vkq = vectors[k * n + q];
                    vectors[k * n + p] = c * vkp - s * vkq;
                    vectors[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    std::vector<long double> result_work(static_cast<std::size_t>(n) * n, 0.0L);
    for (int k = 0; k < n; ++k) {
        const long double eigenvalue = work[k * n + k];
        if (!(eigenvalue > 0.0L) || !std::isfinite(eigenvalue))
            throw std::runtime_error("RBPolyB overlap matrix is not positive definite");
        const long double weight = 1.0L / std::sqrt(eigenvalue);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                result_work[i * n + j] +=
                    vectors[i * n + k] * weight * vectors[j * n + k];
    }
    std::vector<double> result(result_work.size());
    std::transform(result_work.begin(), result_work.end(), result.begin(),
                   [](long double value) { return static_cast<double>(value); });
    return result;
}

} // namespace radial_detail

struct RadialBasis {
    BasisType type = BasisType::Chebyshev;
    std::string name;
    double min_dist = 1.0;
    double max_dist = 6.0;
    int basis_size = 0;
    double scaling = 1.0;

    // Optional legacy/configurable parameters.
    double parameter1 = 0.0;
    double parameter2 = 0.0;
    std::string basis_function_type = "chebyshev";
    std::string cutoff_type;
    std::string shift_type;
    double cutoff_b = -1.0 / 3.0;
    double cutoff_r1 = 0.0;
    double cutoff_r2 = 0.0;
    double shift_lambda = 0.0;
    double shift_scale_lambda = 0.0;
    std::vector<double> poly_coefficients;
    std::vector<double> poly_normalization;

    static std::unique_ptr<RadialBasis> load(const std::string &basis_name,
                                             std::istream &is,
                                             double model_scaling)
    {
        using namespace radial_detail;
        auto basis = std::make_unique<RadialBasis>();
        basis->name = basis_name;
        basis->type = basis_type_from_name(basis_name);
        basis->min_dist = read_double(is, "min_dist");
        basis->max_dist = read_double(is, "max_dist");
        basis->basis_size = read_int(is, "radial_basis_size");
        basis->scaling = model_scaling;
        if (!(basis->min_dist < basis->max_dist))
            throw std::runtime_error("radial basis requires min_dist < max_dist");

        switch (basis->type) {
        case BasisType::ChebyshevPoly3:
        case BasisType::ChebyshevPoly4:
        case BasisType::ChebyshevPoly5:
            basis->parameter1 = read_double(is, "rb_b");
            break;
        case BasisType::ChebyshevPoly7BSideC:
            basis->parameter1 = read_double(is, "r1");
            basis->parameter2 = read_double(is, "r2");
            if (!(basis->min_dist < basis->parameter1 &&
                  basis->parameter1 <= basis->parameter2 &&
                  basis->parameter2 < basis->max_dist))
                throw std::runtime_error(
                    "RBChebyshevPoly7BSideC requires min_dist < r1 <= r2 < max_dist");
            break;
        case BasisType::ChebyshevShiftedFlip:
        case BasisType::ChebyDoubleCosShiftedScaled:
        case BasisType::ChebyDoubleCosShiftedScaledLeft:
        case BasisType::ChebyCosShiftedScaled:
            basis->parameter1 = read_double(is, "lambda");
            break;
        case BasisType::ChebyDoubleSinDoubleScaledShiftedMiddle:
            basis->parameter1 = read_double(is, "rb_lambda_shift");
            basis->parameter2 = read_double(is, "rb_lambda_scaled");
            break;
        case BasisType::ChebyshevConfigurable: {
            is >> std::ws;
            if (is.peek() == 'b')
                basis->basis_function_type =
                    read_value(is, "basis_function_type");
            if (basis->basis_function_type != "chebyshev" &&
                basis->basis_function_type != "one_minus_chebyshev")
                throw std::runtime_error(
                    "unknown basis_function_type " +
                    basis->basis_function_type);

            basis->cutoff_type = read_value(is, "cutoff_type");
            if (basis->cutoff_type == "poly3" ||
                basis->cutoff_type == "poly4" ||
                basis->cutoff_type == "poly5")
                basis->cutoff_b = read_double(is, "cutoff_b");
            else if (basis->cutoff_type == "smoothstep7") {
                basis->cutoff_r1 = read_double(is, "cutoff_r1");
                basis->cutoff_r2 = read_double(is, "cutoff_r2");
                if (!(basis->min_dist < basis->cutoff_r1 &&
                      basis->cutoff_r1 <= basis->cutoff_r2 &&
                      basis->cutoff_r2 < basis->max_dist))
                    throw std::runtime_error(
                        "smoothstep7 requires min_dist < cutoff_r1 <= "
                        "cutoff_r2 < max_dist");
            } else if (basis->cutoff_type != "quadratic" &&
                       basis->cutoff_type != "normalized_quadratic" &&
                       basis->cutoff_type != "cosine" &&
                       basis->cutoff_type != "cosine_left" &&
                       basis->cutoff_type != "cosine_unscaled" &&
                       basis->cutoff_type != "cosine_squared" &&
                       basis->cutoff_type != "sine_squared" &&
                       basis->cutoff_type != "smoothstep5")
                throw std::runtime_error("unknown cutoff_type " + basis->cutoff_type);
            if (basis->cutoff_type == "cosine_left" && basis->max_dist == 0.0)
                throw std::runtime_error(
                    "cosine_left requires nonzero max_dist");

            basis->shift_type = read_value(is, "shift_type");
            if (basis->shift_type == "exponential" ||
                basis->shift_type == "exponential_left" ||
                basis->shift_type == "exponential_tanh")
                basis->shift_lambda = read_double(is, "shift_lambda");
            if (basis->shift_type == "exponential_tanh")
                basis->shift_scale_lambda =
                    read_double(is, "shift_scale_lambda");
            else if (basis->shift_type != "linear" &&
                     basis->shift_type != "exponential" &&
                     basis->shift_type != "exponential_left" &&
                     basis->shift_type != "i2")
                throw std::runtime_error("unknown shift_type " + basis->shift_type);
            if (basis->shift_type == "exponential_left" &&
                basis->max_dist == 0.0)
                throw std::runtime_error(
                    "exponential_left requires nonzero max_dist");
            break;
        }
        default:
            break;
        }

        if (basis->type == BasisType::Constant) basis->basis_size = 1;
        if (basis->type == BasisType::PolyB || basis->type == BasisType::PolyBCos)
            basis->initialize_poly_basis();
        return basis;
    }

    void evaluate(double r, double *vals, double *ders) const
    {
        using namespace radial_detail;
        std::fill(vals, vals + basis_size, 0.0);
        std::fill(ders, ders + basis_size, 0.0);
        if (r < min_dist || r > max_dist) return;

        const Dual radius{r, 1.0};
        const double range = max_dist - min_dist;
        const Dual unit = (radius - min_dist) / range;
        const Dual linear_shift = 2.0 * unit - 1.0;
        Dual shift = linear_shift;
        Dual cutoff{1.0, 0.0};

        switch (type) {
        case BasisType::Chebyshev:
            cutoff = pow(radius - max_dist, 2);
            break;
        case BasisType::ChebyshevI2: {
            const Dual k0 = (radius - max_dist) / range;
            cutoff = k0 * k0;
            shift = 2.0 * cutoff - 1.0;
            break;
        }
        case BasisType::ChebyshevPoly2:
            cutoff = pow(1.0 - unit, 2);
            break;
        case BasisType::ChebyshevPoly3:
            cutoff = pow(1.0 - unit, 2) * (1.0 + (2.0 + parameter1) * unit);
            break;
        case BasisType::ChebyshevPoly4:
            cutoff = pow(1.0 - unit, 3) * (1.0 + (3.0 + parameter1) * unit);
            break;
        case BasisType::ChebyshevPoly5:
            cutoff = pow(1.0 - unit, 4) * (1.0 + (4.0 + parameter1) * unit);
            break;
        case BasisType::ChebyshevPoly5BSide: {
            const Dual q = (max_dist - radius) / range;
            cutoff = 6.0 * pow(q, 5) - 15.0 * pow(q, 4) + 10.0 * pow(q, 3);
            break;
        }
        case BasisType::ChebyshevPoly7BSideC: {
            if (r <= parameter1) {
                const Dual p = (radius - min_dist) / (parameter1 - min_dist);
                cutoff = -(20.0 * pow(p, 7) - 70.0 * pow(p, 6) +
                           84.0 * pow(p, 5) - 35.0 * pow(p, 4));
            } else if (r >= parameter2) {
                const Dual p = (radius - parameter2) / (max_dist - parameter2);
                cutoff = 20.0 * pow(p, 7) - 70.0 * pow(p, 6) +
                         84.0 * pow(p, 5) - 35.0 * pow(p, 4) + 1.0;
            } else {
                cutoff = {1.0, 0.0};
            }
            break;
        }
        case BasisType::ChebyCos:
            cutoff = 0.5 * (1.0 + cos(radius * (std::acos(-1.0) / max_dist)));
            break;
        case BasisType::ChebyDoubleSinMiddle:
            cutoff = pow(sin((radius - min_dist) * (std::acos(-1.0) / range)), 2);
            break;
        case BasisType::ChebyDoubleCosLeft:
            cutoff = pow(cos(radius * (std::acos(-1.0) / (2.0 * max_dist))), 2);
            break;
        case BasisType::ChebyDoubleCosShiftedScaled:
            shift = exponential_shift(unit, parameter1);
            cutoff = pow(1.0 + cos((radius - min_dist) *
                                   (std::acos(-1.0) / range)), 2);
            break;
        case BasisType::ChebyCosShiftedScaled:
            shift = exponential_shift(unit, parameter1);
            cutoff = 1.0 + cos((radius - min_dist) *
                               (std::acos(-1.0) / range));
            break;
        case BasisType::ChebyDoubleCosShiftedScaledLeft:
            shift = exponential_shift(radius / max_dist, parameter1);
            cutoff = pow(cos(radius * (std::acos(-1.0) / (2.0 * max_dist))), 2);
            break;
        case BasisType::ChebyDoubleSinDoubleScaledShiftedMiddle: {
            const Dual shifted = exponential_shift(unit, parameter1);
            const double lambda = parameter2 / 4.0;
            shift = lambda == 0.0 ? shifted : tanh(lambda * shifted) / std::tanh(lambda);
            cutoff = pow(sin(unit * std::acos(-1.0)), 2);
            break;
        }
        case BasisType::ChebyshevConfigurable:
            shift = configurable_shift(radius, unit);
            cutoff = configurable_cutoff(radius, unit);
            if (basis_function_type == "one_minus_chebyshev") {
                fill_one_minus_chebyshev(shift, cutoff, vals, ders);
                return;
            }
            break;
        case BasisType::ChebyshevFlip:
            fill_one_minus_chebyshev(
                linear_shift, pow(radius - max_dist, 2), vals, ders);
            return;
        case BasisType::ChebyshevShiftedFlip:
            fill_one_minus_chebyshev(
                shifted_flip_shift(radius), pow(radius - max_dist, 2),
                vals, ders);
            return;
        case BasisType::PolyB:
        case BasisType::PolyBCos:
            evaluate_poly(radius, vals, ders);
            return;
        case BasisType::Constant: {
            const Dual value = pow(radius - max_dist, 2);
            vals[0] = value.val;
            ders[0] = value.der;
            return;
        }
        }
        fill_chebyshev(shift, cutoff, vals, ders);
    }

private:
    static radial_detail::Dual exponential_shift(radial_detail::Dual unit,
                                                  double lambda)
    {
        using namespace radial_detail;
        if (lambda == 0.0) return 2.0 * unit - 1.0;
        return 1.0 - 2.0 * (exp(lambda * (1.0 - unit)) - 1.0) /
                         std::expm1(lambda);
    }

    radial_detail::Dual configurable_shift(radial_detail::Dual radius,
                                            radial_detail::Dual unit) const
    {
        using namespace radial_detail;
        if (shift_type == "linear") return 2.0 * unit - 1.0;
        if (shift_type == "exponential")
            return exponential_shift(unit, shift_lambda);
        if (shift_type == "exponential_left")
            return exponential_shift(radius / max_dist, shift_lambda);
        if (shift_type == "i2") {
            const Dual k =
                (radius - max_dist) / (max_dist - min_dist);
            return 2.0 * k * k - 1.0;
        }
        const Dual shifted = exponential_shift(unit, shift_lambda);
        if (shift_scale_lambda == 0.0) return shifted;
        return tanh(shift_scale_lambda * shifted) /
               std::tanh(shift_scale_lambda);
    }

    radial_detail::Dual configurable_cutoff(radial_detail::Dual radius,
                                             radial_detail::Dual unit) const
    {
        using namespace radial_detail;
        if (cutoff_type == "quadratic") return pow(radius - max_dist, 2);
        const Dual remaining = 1.0 - unit;
        if (cutoff_type == "normalized_quadratic")
            return pow(remaining, 2);
        if (cutoff_type == "cosine")
            return 0.5 * (1.0 + cos(unit * std::acos(-1.0)));
        if (cutoff_type == "cosine_left")
            return 0.5 *
                   (1.0 + cos(radius * (std::acos(-1.0) / max_dist)));
        const Dual cosine = cos(unit * std::acos(-1.0));
        if (cutoff_type == "cosine_unscaled") return 1.0 + cosine;
        if (cutoff_type == "cosine_squared")
            return pow(1.0 + cosine, 2);
        if (cutoff_type == "sine_squared")
            return pow(sin(unit * std::acos(-1.0)), 2);
        if (cutoff_type == "poly3")
            return pow(remaining, 2) *
                   (1.0 + (2.0 + cutoff_b) * unit);
        if (cutoff_type == "poly4")
            return pow(remaining, 3) *
                   (1.0 + (3.0 + cutoff_b) * unit);
        if (cutoff_type == "poly5")
            return pow(remaining, 4) *
                   (1.0 + (4.0 + cutoff_b) * unit);
        if (cutoff_type == "smoothstep5")
            return 6.0 * pow(remaining, 5) -
                   15.0 * pow(remaining, 4) +
                   10.0 * pow(remaining, 3);
        if (radius.val <= cutoff_r1) {
            const Dual p =
                (radius - min_dist) / (cutoff_r1 - min_dist);
            return 35.0 * pow(p, 4) - 84.0 * pow(p, 5) +
                   70.0 * pow(p, 6) - 20.0 * pow(p, 7);
        }
        if (radius.val >= cutoff_r2) {
            const Dual p =
                (radius - cutoff_r2) / (max_dist - cutoff_r2);
            return 1.0 - 35.0 * pow(p, 4) + 84.0 * pow(p, 5) -
                   70.0 * pow(p, 6) + 20.0 * pow(p, 7);
        }
        return {1.0, 0.0};
    }

    radial_detail::Dual shifted_flip_shift(radial_detail::Dual radius) const
    {
        using namespace radial_detail;
        if (parameter1 == 0.0)
            return 2.0 * (radius - min_dist) / (max_dist - min_dist) - 1.0;
        const double beta = 2.0 * std::expm1(parameter1) /
            std::expm1(-parameter1 * (min_dist / max_dist - 1.0));
        const double inv = beta / std::expm1(parameter1);
        return 1.0 + inv - inv * exp((-parameter1 / max_dist) * radius + parameter1);
    }

    void fill_chebyshev(radial_detail::Dual shift,
                        radial_detail::Dual cutoff,
                        double *vals, double *ders) const
    {
        using namespace radial_detail;
        Dual previous = scaling * cutoff;
        vals[0] = previous.val;
        ders[0] = previous.der;
        if (basis_size == 1) return;
        Dual current = shift * previous;
        vals[1] = current.val;
        ders[1] = current.der;
        for (int i = 2; i < basis_size; ++i) {
            const Dual next = 2.0 * shift * current - previous;
            vals[i] = next.val;
            ders[i] = next.der;
            previous = current;
            current = next;
        }
    }

    void fill_one_minus_chebyshev(radial_detail::Dual shift,
                                  radial_detail::Dual cutoff,
                                  double *vals, double *ders) const
    {
        using namespace radial_detail;
        Dual previous{1.0, 0.0};
        Dual current = shift;
        cutoff = scaling * cutoff;
        for (int i = 0; i < basis_size; ++i) {
            const Dual result = cutoff * (1.0 - current);
            vals[i] = result.val;
            ders[i] = result.der;
            const Dual next = 2.0 * shift * current - previous;
            previous = current;
            current = next;
        }
    }

    void initialize_poly_basis()
    {
        const int n = basis_size;
        const double range = max_dist - min_dist;
        poly_normalization.resize(n);
        std::vector<double> overlap(static_cast<std::size_t>(n) * n);
        for (int i = 0; i < n; ++i) {
            const int ipower = i + 1;
            poly_normalization[i] = std::sqrt(
                (2.0 * ipower + 5.0) /
                std::pow(range, 2 * ipower + 5));
            for (int j = 0; j < n; ++j) {
                const int jpower = j + 1;
                overlap[i * n + j] =
                    std::sqrt((2.0 * ipower + 5.0) *
                              (2.0 * jpower + 5.0)) /
                    (ipower + jpower + 5.0);
            }
        }
        poly_coefficients =
            radial_detail::symmetric_inverse_sqrt(std::move(overlap), n);
    }

    void evaluate_poly(radial_detail::Dual radius,
                       double *vals, double *ders) const
    {
        using namespace radial_detail;
        const Dual remaining = max_dist - radius;
        const Dual cosine = type == BasisType::PolyBCos
            ? 0.5 * (1.0 + cos(radius * (std::acos(-1.0) / max_dist)))
            : Dual{1.0, 0.0};
        std::vector<Dual> primitive(basis_size);
        for (int j = 0; j < basis_size; ++j) {
            const int exponent = type == BasisType::PolyB ? j + 3 : j + 2;
            primitive[j] = scaling * poly_normalization[j] *
                           pow(remaining, exponent) * cosine;
        }
        for (int i = 0; i < basis_size; ++i) {
            Dual value{};
            for (int j = 0; j < basis_size; ++j)
                value = value + poly_coefficients[i * basis_size + j] * primitive[j];
            vals[i] = value.val;
            ders[i] = value.der;
        }
    }
};

} // namespace mtp_kim
