// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "utils/gputils.h"

/** Independently configurable cutoff envelopes for radial bases. */
namespace Cutoff {

enum class CutoffType {
    Quadratic,
    NormalizedQuadratic,
    Cosine,
    CosineLeft,
    CosineUnscaled,
    CosineSquared,
    SineSquared,
    Poly3,
    Poly4,
    Poly5,
    Smoothstep5,
    Smoothstep7,
};

const char *cutoff_type_name(CutoffType type) noexcept;
CutoffType cutoff_type_from_name(const std::string &name);

class AnyCutoff : public ConnectSettings {
public:
    virtual ~AnyCutoff() = default;

    virtual CutoffType type() const noexcept = 0;
    virtual double min_value() const noexcept = 0;
    virtual double max_value() const noexcept = 0;

    virtual double compute(double r) const = 0;
    virtual void compute_der(double r, double &val, double &der) const = 0;
    virtual void compute_derder(double r, double &val, double &der,
                                double &derder) const = 0;

    virtual void SaveToStream(std::ostream &os) const = 0;
};

class QuadraticCutoff final : public AnyCutoff {
    double min_val_;
    double max_val_;

public:
    QuadraticCutoff(double min_val, double max_val);

    CutoffType type() const noexcept override { return CutoffType::Quadratic; }
    double min_value() const noexcept override { return min_val_; }
    double max_value() const noexcept override { return max_val_; }
    double compute(double r) const override;
    void compute_der(double r, double &val, double &der) const override;
    void compute_derder(double r, double &val, double &der,
                        double &derder) const override;
    void SaveToStream(std::ostream &os) const override;
};

class CosineCutoff final : public AnyCutoff {
    double min_val_;
    double max_val_;
    double frequency_;

public:
    CosineCutoff(double min_val, double max_val);

    CutoffType type() const noexcept override { return CutoffType::Cosine; }
    double min_value() const noexcept override { return min_val_; }
    double max_value() const noexcept override { return max_val_; }
    double compute(double r) const override;
    void compute_der(double r, double &val, double &der) const override;
    void compute_derder(double r, double &val, double &der,
                        double &derder) const override;
    void SaveToStream(std::ostream &os) const override;
};

class Poly5Cutoff final : public AnyCutoff {
    double min_val_;
    double max_val_;
    double inv_range_;
    double b_;

public:
    Poly5Cutoff(double min_val, double max_val, double b);

    CutoffType type() const noexcept override { return CutoffType::Poly5; }
    double min_value() const noexcept override { return min_val_; }
    double max_value() const noexcept override { return max_val_; }
    double b() const noexcept { return b_; }
    double compute(double r) const override;
    void compute_der(double r, double &val, double &der) const override;
    void compute_derder(double r, double &val, double &der,
                        double &derder) const override;
    void SaveToStream(std::ostream &os) const override;
};

/** Parameters used only by cutoff types that need them. */
struct Parameters {
    double b = -1.0 / 3.0;
    double r1 = 0.0;
    double r2 = 0.0;
};

std::unique_ptr<AnyCutoff> make_cutoff(CutoffType type, double min_val,
                                      double max_val,
                                      const Parameters &parameters = {});

// Compatibility overload for existing Poly5 callers.
std::unique_ptr<AnyCutoff> make_cutoff(CutoffType type, double min_val,
                                      double max_val, double b);

std::unique_ptr<AnyCutoff> load_cutoff(std::istream &is, double min_val,
                                      double max_val);

} // namespace Cutoff
