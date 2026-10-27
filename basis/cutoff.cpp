// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "cutoff.h"

#include <cmath>
#include <istream>
#include <numbers>
#include <ostream>
#include <stdexcept>

namespace Cutoff {
namespace {

struct Jet2 {
    double val;
    double der;
    double derder;
};

Jet2 operator+(const Jet2 &a, const Jet2 &b)
{
    return {a.val + b.val, a.der + b.der, a.derder + b.derder};
}

Jet2 operator-(const Jet2 &a, const Jet2 &b)
{
    return {a.val - b.val, a.der - b.der, a.derder - b.derder};
}

Jet2 operator*(const Jet2 &a, const Jet2 &b)
{
    return {a.val * b.val,
            a.der * b.val + a.val * b.der,
            a.derder * b.val + 2.0 * a.der * b.der + a.val * b.derder};
}

Jet2 operator*(double scalar, const Jet2 &a)
{
    return {scalar * a.val, scalar * a.der, scalar * a.derder};
}

Jet2 operator/(const Jet2 &a, double scalar)
{
    return (1.0 / scalar) * a;
}

Jet2 sin(const Jet2 &a)
{
    const double s = std::sin(a.val);
    const double c = std::cos(a.val);
    return {s, c * a.der, c * a.derder - s * a.der * a.der};
}

Jet2 cos(const Jet2 &a)
{
    const double s = std::sin(a.val);
    const double c = std::cos(a.val);
    return {c, -s * a.der, -s * a.derder - c * a.der * a.der};
}

Jet2 pow2(const Jet2 &a) { return a * a; }
Jet2 pow3(const Jet2 &a) { return a * a * a; }
Jet2 pow4(const Jet2 &a) { return pow2(pow2(a)); }
Jet2 pow5(const Jet2 &a) { return pow4(a) * a; }
Jet2 pow6(const Jet2 &a) { return pow3(a) * pow3(a); }
Jet2 pow7(const Jet2 &a) { return pow6(a) * a; }

void validate_interval(double min_val, double max_val)
{
    if (!std::isfinite(min_val) || !std::isfinite(max_val) ||
        !(min_val < max_val)) {
        throw std::invalid_argument(
            "cutoff interval requires finite min_dist < max_dist");
    }
}

std::string read_value(std::istream &is, const char *expected_key)
{
    std::string key;
    std::string equals;
    std::string value;
    if (!(is >> key) || key != expected_key || !(is >> equals) ||
        equals != "=" || !(is >> value)) {
        throw std::runtime_error(
            std::string("Error loading configurable radial basis: expected ") +
            expected_key + " = <value>");
    }
    return value;
}

double read_double(std::istream &is, const char *expected_key)
{
    const std::string text = read_value(is, expected_key);
    std::size_t parsed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &parsed);
    } catch (const std::exception &) {
        throw std::runtime_error(
            std::string("Error loading configurable radial basis: invalid ") +
            expected_key);
    }
    if (parsed != text.size() || !std::isfinite(value)) {
        throw std::runtime_error(
            std::string("Error loading configurable radial basis: invalid ") +
            expected_key);
    }
    return value;
}

class FormulaCutoff final : public AnyCutoff {
    CutoffType type_;
    double min_val_;
    double max_val_;
    double inv_range_;
    Parameters parameters_;

    Jet2 evaluate(double r) const
    {
        const Jet2 radius{r, 1.0, 0.0};
        const Jet2 one{1.0, 0.0, 0.0};
        const Jet2 t = (radius - Jet2{min_val_, 0.0, 0.0}) *
                       Jet2{inv_range_, 0.0, 0.0};
        const Jet2 q = one - t;
        const double pi = std::numbers::pi_v<double>;

        switch (type_) {
        case CutoffType::NormalizedQuadratic:
            return pow2(q);
        case CutoffType::CosineLeft:
            return 0.5 * (one + cos((pi / max_val_) * radius));
        case CutoffType::CosineUnscaled:
            return one + cos(pi * t);
        case CutoffType::CosineSquared:
            return pow2(one + cos(pi * t));
        case CutoffType::SineSquared:
            return pow2(sin(pi * t));
        case CutoffType::Poly3:
            return pow2(q) *
                   (one + (parameters_.b + 2.0) * t);
        case CutoffType::Poly4:
            return pow3(q) *
                   (one + (parameters_.b + 3.0) * t);
        case CutoffType::Smoothstep5:
            return 6.0 * pow5(q) - 15.0 * pow4(q) + 10.0 * pow3(q);
        case CutoffType::Smoothstep7: {
            if (r <= parameters_.r1) {
                const Jet2 p =
                    (radius - Jet2{min_val_, 0.0, 0.0}) /
                    (parameters_.r1 - min_val_);
                return 35.0 * pow4(p) - 84.0 * pow5(p) +
                       70.0 * pow6(p) - 20.0 * pow7(p);
            }
            if (r >= parameters_.r2) {
                const Jet2 p =
                    (radius - Jet2{parameters_.r2, 0.0, 0.0}) /
                    (max_val_ - parameters_.r2);
                return one - 35.0 * pow4(p) + 84.0 * pow5(p) -
                       70.0 * pow6(p) + 20.0 * pow7(p);
            }
            return one;
        }
        default:
            throw std::logic_error("invalid formula cutoff type");
        }
    }

public:
    FormulaCutoff(CutoffType type, double min_val, double max_val,
                  const Parameters &parameters)
        : type_(type), min_val_(min_val), max_val_(max_val),
          inv_range_(1.0 / (max_val - min_val)), parameters_(parameters)
    {
        validate_interval(min_val_, max_val_);
        if (type_ == CutoffType::CosineLeft && max_val_ == 0.0)
            throw std::invalid_argument(
                "cosine_left cutoff requires nonzero max_dist");
        if ((type_ == CutoffType::Poly3 || type_ == CutoffType::Poly4) &&
            !std::isfinite(parameters_.b))
            throw std::invalid_argument("polynomial cutoff requires finite cutoff_b");
        if (type_ == CutoffType::Smoothstep7 &&
            !(min_val_ < parameters_.r1 &&
              parameters_.r1 <= parameters_.r2 &&
              parameters_.r2 < max_val_))
            throw std::invalid_argument(
                "smoothstep7 cutoff requires min_dist < cutoff_r1 <= "
                "cutoff_r2 < max_dist");
    }

    CutoffType type() const noexcept override { return type_; }
    double min_value() const noexcept override { return min_val_; }
    double max_value() const noexcept override { return max_val_; }

    double compute(double r) const override { return evaluate(r).val; }

    void compute_der(double r, double &val, double &der) const override
    {
        const Jet2 result = evaluate(r);
        val = result.val;
        der = result.der;
    }

    void compute_derder(double r, double &val, double &der,
                        double &derder) const override
    {
        const Jet2 result = evaluate(r);
        val = result.val;
        der = result.der;
        derder = result.derder;
    }

    void SaveToStream(std::ostream &os) const override
    {
        os << "\tcutoff_type = " << cutoff_type_name(type_) << '\n';
        if (type_ == CutoffType::Poly3 || type_ == CutoffType::Poly4)
            os << "\tcutoff_b = " << parameters_.b << '\n';
        if (type_ == CutoffType::Smoothstep7) {
            os << "\tcutoff_r1 = " << parameters_.r1 << '\n';
            os << "\tcutoff_r2 = " << parameters_.r2 << '\n';
        }
    }
};

} // namespace

const char *cutoff_type_name(CutoffType type) noexcept
{
    switch (type) {
    case CutoffType::Quadratic: return "quadratic";
    case CutoffType::NormalizedQuadratic: return "normalized_quadratic";
    case CutoffType::Cosine: return "cosine";
    case CutoffType::CosineLeft: return "cosine_left";
    case CutoffType::CosineUnscaled: return "cosine_unscaled";
    case CutoffType::CosineSquared: return "cosine_squared";
    case CutoffType::SineSquared: return "sine_squared";
    case CutoffType::Poly3: return "poly3";
    case CutoffType::Poly4: return "poly4";
    case CutoffType::Poly5: return "poly5";
    case CutoffType::Smoothstep5: return "smoothstep5";
    case CutoffType::Smoothstep7: return "smoothstep7";
    }
    return "unknown";
}

CutoffType cutoff_type_from_name(const std::string &name)
{
    if (name == "quadratic") return CutoffType::Quadratic;
    if (name == "normalized_quadratic")
        return CutoffType::NormalizedQuadratic;
    if (name == "cosine") return CutoffType::Cosine;
    if (name == "cosine_left") return CutoffType::CosineLeft;
    if (name == "cosine_unscaled") return CutoffType::CosineUnscaled;
    if (name == "cosine_squared") return CutoffType::CosineSquared;
    if (name == "sine_squared") return CutoffType::SineSquared;
    if (name == "poly3") return CutoffType::Poly3;
    if (name == "poly4") return CutoffType::Poly4;
    if (name == "poly5") return CutoffType::Poly5;
    if (name == "smoothstep5") return CutoffType::Smoothstep5;
    if (name == "smoothstep7") return CutoffType::Smoothstep7;
    throw std::runtime_error("Unknown configurable radial cutoff type: " + name);
}

QuadraticCutoff::QuadraticCutoff(double min_val, double max_val)
    : min_val_(min_val), max_val_(max_val)
{
    validate_interval(min_val_, max_val_);
}

double QuadraticCutoff::compute(double r) const
{
    const double delta = r - max_val_;
    return delta * delta;
}

void QuadraticCutoff::compute_der(double r, double &val, double &der) const
{
    const double delta = r - max_val_;
    val = delta * delta;
    der = 2.0 * delta;
}

void QuadraticCutoff::compute_derder(double r, double &val, double &der,
                                     double &derder) const
{
    compute_der(r, val, der);
    derder = 2.0;
}

void QuadraticCutoff::SaveToStream(std::ostream &os) const
{
    os << "\tcutoff_type = quadratic\n";
}

CosineCutoff::CosineCutoff(double min_val, double max_val)
    : min_val_(min_val), max_val_(max_val), frequency_(0.0)
{
    validate_interval(min_val_, max_val_);
    frequency_ = std::numbers::pi_v<double> / (max_val_ - min_val_);
}

double CosineCutoff::compute(double r) const
{
    return 0.5 * (1.0 + std::cos((r - min_val_) * frequency_));
}

void CosineCutoff::compute_der(double r, double &val, double &der) const
{
    const double phase = (r - min_val_) * frequency_;
    val = 0.5 * (1.0 + std::cos(phase));
    der = -0.5 * frequency_ * std::sin(phase);
}

void CosineCutoff::compute_derder(double r, double &val, double &der,
                                  double &derder) const
{
    const double phase = (r - min_val_) * frequency_;
    val = 0.5 * (1.0 + std::cos(phase));
    der = -0.5 * frequency_ * std::sin(phase);
    derder = -0.5 * frequency_ * frequency_ * std::cos(phase);
}

void CosineCutoff::SaveToStream(std::ostream &os) const
{
    os << "\tcutoff_type = cosine\n";
}

Poly5Cutoff::Poly5Cutoff(double min_val, double max_val, double b)
    : min_val_(min_val), max_val_(max_val), inv_range_(0.0), b_(b)
{
    validate_interval(min_val_, max_val_);
    if (!std::isfinite(b_))
        throw std::invalid_argument("poly5 cutoff requires finite cutoff_b");
    inv_range_ = 1.0 / (max_val_ - min_val_);
}

double Poly5Cutoff::compute(double r) const
{
    const double x = (r - min_val_) * inv_range_;
    const double one_minus_x = 1.0 - x;
    const double square = one_minus_x * one_minus_x;
    return square * square * (1.0 + (4.0 + b_) * x);
}

void Poly5Cutoff::compute_der(double r, double &val, double &der) const
{
    const double x = (r - min_val_) * inv_range_;
    const double one_minus_x = 1.0 - x;
    const double square = one_minus_x * one_minus_x;
    const double cube = square * one_minus_x;
    val = square * square * (1.0 + (4.0 + b_) * x);
    der = -cube * (20.0 * x + b_ * (5.0 * x - 1.0)) * inv_range_;
}

void Poly5Cutoff::compute_derder(double r, double &val, double &der,
                                 double &derder) const
{
    const double x = (r - min_val_) * inv_range_;
    const double one_minus_x = 1.0 - x;
    const double square = one_minus_x * one_minus_x;
    const double cube = square * one_minus_x;
    const double inv_range_sq = inv_range_ * inv_range_;
    val = square * square * (1.0 + (4.0 + b_) * x);
    der = -cube * (20.0 * x + b_ * (5.0 * x - 1.0)) * inv_range_;
    derder = 4.0 * square *
             (-5.0 + 20.0 * x + b_ * (5.0 * x - 2.0)) * inv_range_sq;
}

void Poly5Cutoff::SaveToStream(std::ostream &os) const
{
    os << "\tcutoff_type = poly5\n";
    os << "\tcutoff_b = " << b_ << '\n';
}

std::unique_ptr<AnyCutoff> make_cutoff(CutoffType type, double min_val,
                                      double max_val,
                                      const Parameters &parameters)
{
    switch (type) {
    case CutoffType::Quadratic:
        return std::make_unique<QuadraticCutoff>(min_val, max_val);
    case CutoffType::Cosine:
        return std::make_unique<CosineCutoff>(min_val, max_val);
    case CutoffType::Poly5:
        return std::make_unique<Poly5Cutoff>(
            min_val, max_val, parameters.b);
    default:
        return std::make_unique<FormulaCutoff>(
            type, min_val, max_val, parameters);
    }
}

std::unique_ptr<AnyCutoff> make_cutoff(CutoffType type, double min_val,
                                      double max_val, double b)
{
    Parameters parameters;
    parameters.b = b;
    return make_cutoff(type, min_val, max_val, parameters);
}

std::unique_ptr<AnyCutoff> load_cutoff(std::istream &is, double min_val,
                                      double max_val)
{
    const CutoffType type =
        cutoff_type_from_name(read_value(is, "cutoff_type"));
    Parameters parameters;
    if (type == CutoffType::Poly3 || type == CutoffType::Poly4 ||
        type == CutoffType::Poly5)
        parameters.b = read_double(is, "cutoff_b");
    if (type == CutoffType::Smoothstep7) {
        parameters.r1 = read_double(is, "cutoff_r1");
        parameters.r2 = read_double(is, "cutoff_r2");
    }
    return make_cutoff(type, min_val, max_val, parameters);
}

} // namespace Cutoff
