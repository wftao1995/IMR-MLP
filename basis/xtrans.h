// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "utils/gputils.h"

/** Distance-to-polynomial-coordinate transforms for radial bases. */
namespace XTransForm {

enum class ShiftType {
    Linear,
    Exponential,
    ExponentialLeft,
    I2,
    ExponentialTanh,
};

const char *shift_type_name(ShiftType type) noexcept;
ShiftType shift_type_from_name(const std::string &name);

class AnyTrans : public ConnectSettings {
public:
    virtual ~AnyTrans() = default;

    virtual ShiftType type() const noexcept = 0;
    virtual double min_value() const noexcept = 0;
    virtual double max_value() const noexcept = 0;

    virtual double compute(double r) const = 0;
    virtual void compute_der(double r, double &val, double &der) const = 0;
    virtual void compute_derder(double r, double &val, double &der,
                                double &derder) const = 0;

    virtual void SaveToStream(std::ostream &os) const = 0;
};

class LinearShift final : public AnyTrans {
    double min_val_;
    double max_val_;
    double slope_;

public:
    LinearShift(double min_val, double max_val);

    ShiftType type() const noexcept override { return ShiftType::Linear; }
    double min_value() const noexcept override { return min_val_; }
    double max_value() const noexcept override { return max_val_; }
    double compute(double r) const override;
    void compute_der(double r, double &val, double &der) const override;
    void compute_derder(double r, double &val, double &der,
                        double &derder) const override;
    void SaveToStream(std::ostream &os) const override;
};

class ExponentialShift final : public AnyTrans {
    double min_val_;
    double max_val_;
    double inv_range_;
    double lambda_;
    double denominator_;

public:
    ExponentialShift(double min_val, double max_val, double lambda);

    ShiftType type() const noexcept override { return ShiftType::Exponential; }
    double min_value() const noexcept override { return min_val_; }
    double max_value() const noexcept override { return max_val_; }
    double lambda() const noexcept { return lambda_; }
    double compute(double r) const override;
    void compute_der(double r, double &val, double &der) const override;
    void compute_derder(double r, double &val, double &der,
                        double &derder) const override;
    void SaveToStream(std::ostream &os) const override;
};

struct Parameters {
    double lambda = 0.0;
    double scale_lambda = 0.0;
};

std::unique_ptr<AnyTrans> make_shift(
    ShiftType type, double min_val, double max_val,
    const Parameters &parameters = {});

// Compatibility overload for existing exponential-shift callers.
std::unique_ptr<AnyTrans> make_shift(ShiftType type, double min_val,
                                    double max_val, double lambda);

std::unique_ptr<AnyTrans> load_shift(std::istream &is, double min_val,
                                    double max_val);

} // namespace XTransForm
