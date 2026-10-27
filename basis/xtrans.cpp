// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "xtrans.h"

#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace XTransForm {
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

Jet2 exp(const Jet2 &a)
{
    const double value = std::exp(a.val);
    return {value, value * a.der,
            value * (a.derder + a.der * a.der)};
}

Jet2 tanh(const Jet2 &a)
{
    const double value = std::tanh(a.val);
    const double sech_sq = 1.0 - value * value;
    return {value, sech_sq * a.der,
            sech_sq * (a.derder - 2.0 * value * a.der * a.der)};
}

void validate_interval(double min_val, double max_val)
{
    if (!std::isfinite(min_val) || !std::isfinite(max_val) ||
        !(min_val < max_val)) {
        throw std::invalid_argument(
            "shift interval requires finite min_dist < max_dist");
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

class FormulaShift final : public AnyTrans {
    ShiftType type_;
    double min_val_;
    double max_val_;
    double inv_range_;
    Parameters parameters_;

    Jet2 evaluate(double r) const
    {
        const Jet2 radius{r, 1.0, 0.0};
        const Jet2 one{1.0, 0.0, 0.0};
        const Jet2 t =
            (radius - Jet2{min_val_, 0.0, 0.0}) *
            Jet2{inv_range_, 0.0, 0.0};

        switch (type_) {
        case ShiftType::I2: {
            const Jet2 k =
                (radius - Jet2{max_val_, 0.0, 0.0}) *
                Jet2{inv_range_, 0.0, 0.0};
            return 2.0 * (k * k) - one;
        }
        case ShiftType::ExponentialLeft: {
            if (parameters_.lambda == 0.0)
                return 2.0 * radius / max_val_ - one;
            const double denominator = std::expm1(parameters_.lambda);
            const Jet2 exponent =
                parameters_.lambda *
                (one - radius / max_val_);
            return one -
                   (2.0 / denominator) * (exp(exponent) - one);
        }
        case ShiftType::ExponentialTanh: {
            Jet2 shifted;
            if (parameters_.lambda == 0.0) {
                shifted = 2.0 * t - one;
            } else {
                const double denominator =
                    std::expm1(parameters_.lambda);
                shifted =
                    one - (2.0 / denominator) *
                              (exp(parameters_.lambda * (one - t)) - one);
            }
            if (parameters_.scale_lambda == 0.0)
                return shifted;
            return tanh(parameters_.scale_lambda * shifted) /
                   std::tanh(parameters_.scale_lambda);
        }
        default:
            throw std::logic_error("invalid formula shift type");
        }
    }

public:
    FormulaShift(ShiftType type, double min_val, double max_val,
                 const Parameters &parameters)
        : type_(type), min_val_(min_val), max_val_(max_val),
          inv_range_(1.0 / (max_val - min_val)), parameters_(parameters)
    {
        validate_interval(min_val_, max_val_);
        if (!std::isfinite(parameters_.lambda) ||
            !std::isfinite(parameters_.scale_lambda))
            throw std::invalid_argument(
                "shift parameters must be finite");
        if (type_ == ShiftType::ExponentialLeft && max_val_ == 0.0)
            throw std::invalid_argument(
                "exponential_left shift requires nonzero max_dist");
        if ((type_ == ShiftType::ExponentialLeft ||
             type_ == ShiftType::ExponentialTanh) &&
            parameters_.lambda != 0.0 &&
            !std::isfinite(std::expm1(parameters_.lambda)))
            throw std::invalid_argument(
                "shift_lambda produces a non-finite exponential normalization");
        if (type_ == ShiftType::ExponentialTanh &&
            parameters_.scale_lambda != 0.0 &&
            std::tanh(parameters_.scale_lambda) == 0.0)
            throw std::invalid_argument(
                "shift_scale_lambda produces an invalid normalization");
    }

    ShiftType type() const noexcept override { return type_; }
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
        os << "\tshift_type = " << shift_type_name(type_) << '\n';
        if (type_ == ShiftType::ExponentialLeft ||
            type_ == ShiftType::ExponentialTanh)
            os << "\tshift_lambda = " << parameters_.lambda << '\n';
        if (type_ == ShiftType::ExponentialTanh)
            os << "\tshift_scale_lambda = "
               << parameters_.scale_lambda << '\n';
    }
};

} // namespace

const char *shift_type_name(ShiftType type) noexcept
{
    switch (type) {
    case ShiftType::Linear: return "linear";
    case ShiftType::Exponential: return "exponential";
    case ShiftType::ExponentialLeft: return "exponential_left";
    case ShiftType::I2: return "i2";
    case ShiftType::ExponentialTanh: return "exponential_tanh";
    }
    return "unknown";
}

ShiftType shift_type_from_name(const std::string &name)
{
    if (name == "linear") return ShiftType::Linear;
    if (name == "exponential") return ShiftType::Exponential;
    if (name == "exponential_left") return ShiftType::ExponentialLeft;
    if (name == "i2") return ShiftType::I2;
    if (name == "exponential_tanh") return ShiftType::ExponentialTanh;
    throw std::runtime_error("Unknown configurable radial shift type: " + name);
}

LinearShift::LinearShift(double min_val, double max_val)
    : min_val_(min_val), max_val_(max_val), slope_(0.0)
{
    validate_interval(min_val_, max_val_);
    slope_ = 2.0 / (max_val_ - min_val_);
}

double LinearShift::compute(double r) const
{
    return -1.0 + slope_ * (r - min_val_);
}

void LinearShift::compute_der(double r, double &val, double &der) const
{
    val = compute(r);
    der = slope_;
}

void LinearShift::compute_derder(double r, double &val, double &der,
                                 double &derder) const
{
    compute_der(r, val, der);
    derder = 0.0;
}

void LinearShift::SaveToStream(std::ostream &os) const
{
    os << "\tshift_type = linear\n";
}

ExponentialShift::ExponentialShift(double min_val, double max_val,
                                   double lambda)
    : min_val_(min_val), max_val_(max_val), inv_range_(0.0),
      lambda_(lambda), denominator_(0.0)
{
    validate_interval(min_val_, max_val_);
    if (!std::isfinite(lambda_))
        throw std::invalid_argument(
            "exponential shift requires finite shift_lambda");
    inv_range_ = 1.0 / (max_val_ - min_val_);
    if (lambda_ != 0.0) {
        denominator_ = std::expm1(lambda_);
        if (!std::isfinite(denominator_) || denominator_ == 0.0)
            throw std::invalid_argument(
                "shift_lambda produces a non-finite exponential normalization");
        const double rate = lambda_ * inv_range_;
        const double first_scale = 2.0 * rate / denominator_;
        const double second_scale = -rate * first_scale;
        const double third_scale = -rate * second_scale;
        if (!std::isfinite(first_scale) || !std::isfinite(second_scale) ||
            !std::isfinite(third_scale))
            throw std::invalid_argument(
                "shift_lambda produces non-finite derivatives");
    }
}

double ExponentialShift::compute(double r) const
{
    if (lambda_ == 0.0)
        return -1.0 + 2.0 * (r - min_val_) * inv_range_;
    const double one_minus_t = (max_val_ - r) * inv_range_;
    return 1.0 -
           2.0 * std::expm1(lambda_ * one_minus_t) / denominator_;
}

void ExponentialShift::compute_der(double r, double &val, double &der) const
{
    if (lambda_ == 0.0) {
        val = -1.0 + 2.0 * (r - min_val_) * inv_range_;
        der = 2.0 * inv_range_;
        return;
    }
    const double exponent = lambda_ * (max_val_ - r) * inv_range_;
    const double exponential = std::exp(exponent);
    val = 1.0 - 2.0 * std::expm1(exponent) / denominator_;
    der = 2.0 * lambda_ * inv_range_ * exponential / denominator_;
}

void ExponentialShift::compute_derder(double r, double &val, double &der,
                                      double &derder) const
{
    compute_der(r, val, der);
    derder = lambda_ == 0.0 ? 0.0 : -lambda_ * inv_range_ * der;
}

void ExponentialShift::SaveToStream(std::ostream &os) const
{
    os << "\tshift_type = exponential\n";
    os << "\tshift_lambda = " << lambda_ << '\n';
}

std::unique_ptr<AnyTrans> make_shift(
    ShiftType type, double min_val, double max_val,
    const Parameters &parameters)
{
    switch (type) {
    case ShiftType::Linear:
        return std::make_unique<LinearShift>(min_val, max_val);
    case ShiftType::Exponential:
        return std::make_unique<ExponentialShift>(
            min_val, max_val, parameters.lambda);
    default:
        return std::make_unique<FormulaShift>(
            type, min_val, max_val, parameters);
    }
}

std::unique_ptr<AnyTrans> make_shift(ShiftType type, double min_val,
                                    double max_val, double lambda)
{
    Parameters parameters;
    parameters.lambda = lambda;
    return make_shift(type, min_val, max_val, parameters);
}

std::unique_ptr<AnyTrans> load_shift(std::istream &is, double min_val,
                                    double max_val)
{
    const ShiftType type =
        shift_type_from_name(read_value(is, "shift_type"));
    Parameters parameters;
    if (type == ShiftType::Exponential ||
        type == ShiftType::ExponentialLeft ||
        type == ShiftType::ExponentialTanh)
        parameters.lambda = read_double(is, "shift_lambda");
    if (type == ShiftType::ExponentialTanh)
        parameters.scale_lambda =
            read_double(is, "shift_scale_lambda");
    return make_shift(type, min_val, max_val, parameters);
}

} // namespace XTransForm
