// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <cmath>
#include <stdexcept>

namespace Training {

class SGDScheduler {
public:
    virtual ~SGDScheduler() = default;
    virtual void on_step(int step) = 0;
    virtual double get_lr() const = 0;

    static std::unique_ptr<SGDScheduler> create(const std::string &type,
                                                  double initial_lr,
                                                  double final_lr,
                                                  int total_steps,
                                                  double decay_rate,
                                                  int decay_steps);

    // Like create(), but wraps the result in a WarmupWrapper if warmup_steps > 0.
    static std::unique_ptr<SGDScheduler> create_with_warmup(
        const std::string &type,
        double initial_lr, double final_lr,
        int total_steps, double decay_rate, int decay_steps,
        int warmup_steps, double warmup_initial_lr);
};

class StepDecayScheduler : public SGDScheduler {
    double initial_lr, decay_rate;
    int decay_steps;
    double current_lr;
public:
    StepDecayScheduler(double lr, double rate, int steps)
        : initial_lr(lr), decay_rate(rate), decay_steps(steps), current_lr(lr) {
        if (!std::isfinite(lr) || lr < 0 || !std::isfinite(rate) || rate < 0 || steps <= 0)
            throw std::invalid_argument("Step SGD scheduler requires finite nonnegative rates and positive decay steps");
    }
    void on_step(int step) override {
        int n = step / decay_steps;
        current_lr = initial_lr * std::pow(decay_rate, n);
    }
    double get_lr() const override { return current_lr; }
};

class CosineDecayScheduler : public SGDScheduler {
    double initial_lr, final_lr;
    int total_steps;
    double current_lr = 0;
public:
    CosineDecayScheduler(double lr, double final, int total)
        : initial_lr(lr), final_lr(final), total_steps(total) {
        if (!std::isfinite(lr) || lr < 0 || !std::isfinite(final) || final < 0 || total <= 0)
            throw std::invalid_argument("Cosine SGD scheduler requires finite nonnegative rates and positive total steps");
    }
    void on_step(int step) override {
        double progress = static_cast<double>(step) / total_steps;
        if (progress > 1.0) progress = 1.0;
        current_lr = final_lr + 0.5 * (initial_lr - final_lr)
                     * (1.0 + std::cos(M_PI * progress));
    }
    double get_lr() const override { return current_lr; }
};

class ExponentialDecayScheduler : public SGDScheduler {
    double initial_lr, decay_rate;
    double current_lr;
public:
    ExponentialDecayScheduler(double lr, double rate)
        : initial_lr(lr), decay_rate(rate), current_lr(lr) {
        if (!std::isfinite(lr) || lr < 0 || !std::isfinite(rate) || rate < 0)
            throw std::invalid_argument("Exponential SGD scheduler requires finite nonnegative rates");
    }
    void on_step(int step) override {
        current_lr = initial_lr * std::exp(-decay_rate * step);
    }
    double get_lr() const override { return current_lr; }
};

// Wraps any scheduler with a linear warmup phase.
// During warmup: lr = warmup_start + (target_lr - warmup_start) * step / warmup_steps
// After warmup: delegates to the inner scheduler (step offset by warmup_steps).
class WarmupWrapper : public SGDScheduler {
    std::unique_ptr<SGDScheduler> inner;
    int warmup_steps;
    double warmup_start_lr, target_lr;
    double current_lr_ = 0;
public:
    WarmupWrapper(std::unique_ptr<SGDScheduler> inner_scheduler,
                  int warmup_steps,
                  double warmup_start_lr,
                  double target_lr)
        : inner(std::move(inner_scheduler)),
          warmup_steps(warmup_steps),
          warmup_start_lr(warmup_start_lr),
          target_lr(target_lr) {
        if (!inner || warmup_steps <= 0 || !std::isfinite(warmup_start_lr)
            || warmup_start_lr < 0 || !std::isfinite(target_lr) || target_lr < 0)
            throw std::invalid_argument("SGD warmup requires a scheduler, positive steps and finite nonnegative rates");
    }

    void on_step(int step) override {
        if (step < warmup_steps) {
            double progress = static_cast<double>(step) / warmup_steps;
            current_lr_ = warmup_start_lr + (target_lr - warmup_start_lr) * progress;
        } else {
            inner->on_step(step - warmup_steps);
            current_lr_ = inner->get_lr();
        }
    }
    double get_lr() const override { return current_lr_; }
};

} // namespace Training
