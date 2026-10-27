// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "sgd_scheduler.h"

namespace Training {

std::unique_ptr<SGDScheduler> SGDScheduler::create(
        const std::string &type, double initial_lr, double final_lr,
        int total_steps, double decay_rate, int decay_steps) {
    if (type == "step") {
        return std::make_unique<StepDecayScheduler>(
            initial_lr, decay_rate, decay_steps);
    } else if (type == "cosine") {
        return std::make_unique<CosineDecayScheduler>(
            initial_lr, final_lr, total_steps);
    } else if (type == "exponential") {
        return std::make_unique<ExponentialDecayScheduler>(
            initial_lr, decay_rate);
    }
    if (type == "none") return nullptr;
    throw std::invalid_argument("Unknown SGD learning rate scheduler: " + type);
}

std::unique_ptr<SGDScheduler> SGDScheduler::create_with_warmup(
        const std::string &type, double initial_lr, double final_lr,
        int total_steps, double decay_rate, int decay_steps,
        int warmup_steps, double warmup_initial_lr) {
    auto inner = create(type, initial_lr, final_lr, total_steps,
                        decay_rate, decay_steps);
    if (inner && warmup_steps < 0)
        throw std::invalid_argument("SGD warmup steps must be nonnegative");
    if (warmup_steps > 0 && inner) {
        return std::make_unique<WarmupWrapper>(
            std::move(inner), warmup_steps, warmup_initial_lr, initial_lr);
    }
    return inner;
}

} // namespace Training
